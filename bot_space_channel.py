# bot_space_channel.py
import asyncio
import os
import hashlib
import sqlite3
import random
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from zoneinfo import ZoneInfo

from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.methods import SendPhoto
from aiogram.types import BufferedInputFile

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from yandex_cloud_ml_sdk import YCloudML


# ---------- CONFIG ----------
@dataclass
class Config:
    tg_token: str
    tg_channel: str  # например: "@pro_kosmos_knl"
    yc_folder_id: str
    yc_api_key: str
    tz: str = "Europe/Moscow"


def env_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Env var {name} is not set")
    return val


# ---------- DEDUP STORAGE (SQLite) ----------
DB_PATH = "posted.db"


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        key TEXT PRIMARY KEY,
        created_at TEXT NOT NULL
    )
    """)

    # Храним нормализованные факты отдельной таблицей для near-duplicate проверки
    cur.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_norm TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_id ON facts(id)")

    con.commit()
    con.close()


def was_posted(key: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT 1 FROM posts WHERE key = ? LIMIT 1", (key,))
    row = cur.fetchone()
    con.close()
    return row is not None


def mark_posted(key: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO posts(key, created_at) VALUES(?, ?) "
        "ON CONFLICT(key) DO NOTHING",
        (key, datetime.now(UTC).isoformat(timespec="seconds"))
    )
    con.commit()
    con.close()


def save_fact_norm(text_norm: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO facts(text_norm, created_at) VALUES(?, ?)",
        (text_norm, datetime.now(UTC).isoformat(timespec="seconds"))
    )
    con.commit()
    con.close()


def get_recent_fact_norms(limit: int = 400) -> list[str]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT text_norm FROM facts ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


# ---------- NEAR-DUPLICATE (TEXT) ----------
def word_ngrams(text: str, n: int = 3) -> set[str]:
    words = text.split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def is_near_duplicate_fact(new_fact_norm: str, lookback: int = 400, threshold: float = 0.72) -> bool:
    new_set = word_ngrams(new_fact_norm, n=3)
    for old in get_recent_fact_norms(limit=lookback):
        if jaccard(new_set, word_ngrams(old, n=3)) >= threshold:
            return True
    return False


# ---------- PROMPTS ----------
TOPIC_HINTS = [
    "стыковки: как корабли сближаются на орбите",
    "теплозащита и вход в атмосферу",
    "парашютные системы спускаемых аппаратов",
    "солнечные батареи и энергобаланс на орбите",
    "разгерметизация: что происходит с давлением и почему это критично",
    "орбитальный мусор и риск для кораблей",
    "баллистический спуск vs управляемый",
    "аварии на орбите: вращение станции и борьба за ориентацию",
    "человеческий фактор и процедуры безопасности",
    "как комиссии расследуют космические инциденты",
]

SCENE_HINTS = [
    "вход в атмосферу: плазма вокруг корабля, след как комета",
    "ночной океан и поисковые корабли в зоне приводнения",
    "центр управления: тишина после потери телеметрии",
    "закопчённый теплозащитный экран крупным планом",
    "станция на орбите, медленно вращающаяся на фоне Земли",
    "обломки на фоне рассвета, очень далеко и без «экшена»",
]

SPACE_ANCHORS = [
    "photoreal space photo, deep space, dense starfield, faint nebula, natural colors",
    "photoreal space photo, low Earth orbit, Earth curvature, atmosphere glow, spacecraft exterior",
    "photoreal space photo, space station in orbit, solar arrays, Earth background, natural shadows",
    "photoreal space photo, planetary surface, black sky, visible stars, harsh sunlight",
]

PHOTO_TECH_HINTS = [
    "DSLR RAW, full-frame, 35mm, f/2.8, ISO 800, long exposure, HDR, ultra sharp",
    "space astrophotography, realistic optics, subtle sensor noise, fine film grain, natural contrast",
    "documentary photo style, slight chromatic aberration, natural vignetting, high detail",
]


# ---------- PROMPT LIMIT HELPERS ----------
MAX_POSITIVE_PROMPT_CHARS = 500


def _total_text_len(messages: list[dict]) -> int:
    return sum(len(m.get("text", "") or "") for m in messages)


def fit_messages_to_limit(messages: list[dict], limit: int = MAX_POSITIVE_PROMPT_CHARS) -> list[dict]:
    msgs = [{"text": (m.get("text", "") or "").strip(), "weight": m.get("weight", 1)} for m in messages]
    if _total_text_len(msgs) <= limit:
        return msgs

    # что резать раньше: prompt_text -> photo_tech -> anchor -> restrictions
    cut_order = [2, 1, 0, 3]
    min_keep = {0: 40, 1: 40, 2: 60, 3: 35}

    def cut_from(i: int, n: int) -> int:
        t = msgs[i]["text"]
        if not t:
            return 0
        keep = min_keep.get(i, 30)
        can_cut = max(0, len(t) - keep)
        cut = min(n, can_cut)
        if cut > 0:
            msgs[i]["text"] = t[:-cut].rstrip()
        return cut

    while _total_text_len(msgs) > limit:
        excess = _total_text_len(msgs) - limit
        progress = 0
        for i in cut_order:
            if excess <= 0:
                break
            cut = cut_from(i, excess)
            if cut:
                excess -= cut
                progress += cut
        if progress == 0:
            msgs[2]["text"] = msgs[2]["text"][:-1].rstrip()

    return msgs


# ---------- YANDEX ML ----------
def build_sdk(cfg: Config) -> YCloudML:
    return YCloudML(folder_id=cfg.yc_folder_id, auth=cfg.yc_api_key)


def generate_image_and_fact_sync(cfg: Config) -> tuple[bytes, str, str]:
    sdk = build_sdk(cfg)

    topic = random.choice(TOPIC_HINTS)
    scene = random.choice(SCENE_HINTS)

    # TEXT (YandexGPT)
    gpt = sdk.models.completions("yandexgpt")

    prompt_res = gpt.run([
        "Сгенерируй короткий, но детальный prompt для модели генерации изображений. "
        "Требования: фотореализм, как реальная космическая фотография/астрофото; "
        "без людей, без текста/надписей/логотипов; без иллюстрации и без 3D-рендера. "
        f"Тема: {topic}. Сцена: {scene}. "
        "Выведи ТОЛЬКО prompt, без пояснений, без кавычек."
    ])
    prompt_text = prompt_res.text.strip().replace("\n", " ")

    # Чтобы факты реже повторялись: просим «не использовать шаблонные формулировки»
    fact_res = gpt.run([
        "Напиши 1 короткий и точный научный факт о космосе (1–2 предложения). "
        "Не используй клише и шаблонные обороты, избегай одинаковых начал фраз. "
        "Без префиксов типа 'Факт дня', без списков, без метаданных. "
        f"Тема: {topic}. Свяжи факт со сценой: {scene}."
    ])
    fact_text = fact_res.text.strip().replace("\n", " ")

    # IMAGE (YandexART)
    img_model = sdk.models.image_generation("yandex-art")

    space_anchor = random.choice(SPACE_ANCHORS)
    photo_tech = random.choice(PHOTO_TECH_HINTS)
    restrictions = "no people, no text, no logos, no watermark, no CGI, no 3d render"

    messages = [
        {"text": space_anchor, "weight": 10},
        {"text": photo_tech, "weight": 6},
        {"text": prompt_text, "weight": 4},
        {"text": restrictions, "weight": 3},
    ]
    messages = fit_messages_to_limit(messages, MAX_POSITIVE_PROMPT_CHARS)

    operation = img_model.run_deferred(messages)
    result = operation.wait()
    image_bytes = result.image_bytes

    return image_bytes, prompt_text, fact_text


# ---------- TELEGRAM ----------
async def send_photo_with_retries(bot: Bot, channel: str, photo_bytes: bytes, caption: str):
    photo = BufferedInputFile(photo_bytes, filename="space.jpg")
    method = SendPhoto(chat_id=channel, photo=photo, caption=caption)

    for attempt in range(1, 4):
        try:
            await bot(method, request_timeout=180)
            return
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except TelegramNetworkError:
            if attempt >= 3:
                raise
            await asyncio.sleep(5 * attempt)


# ---------- JOB ----------
async def post_job(cfg: Config, bot: Bot):
    print("JOB: start", flush=True)

    for attempt in range(1, 10):  # чуть больше попыток, т.к. теперь строгая проверка уникальности
        image_bytes, prompt_text, fact_text = await asyncio.to_thread(
            generate_image_and_fact_sync, cfg
        )
        print(f"JOB: generated {len(image_bytes)} bytes", flush=True)

        fact_norm = normalize_text(fact_text)
        fact_key = "fact_norm:" + sha256_hex(fact_norm.encode("utf-8"))
        img_key = "img:" + sha256_hex(image_bytes)

        # 1) точный дубль факта
        if was_posted(fact_key):
            print(f"JOB: duplicate fact (normalized), retry {attempt}/9", flush=True)
            continue

        # 2) близкий дубль факта (перефраз)
        if is_near_duplicate_fact(fact_norm, lookback=400, threshold=0.72):
            print(f"JOB: near-duplicate fact (jaccard), retry {attempt}/9", flush=True)
            continue

        # 3) дубль картинки
        if was_posted(img_key):
            print(f"JOB: duplicate image, retry {attempt}/9", flush=True)
            continue

        caption = f"#ФактДня: {fact_text}\n\nПодписывайся! @pro_kosmos_knl"
        await send_photo_with_retries(bot, cfg.tg_channel, image_bytes, caption)

        mark_posted(fact_key)
        mark_posted(img_key)
        save_fact_norm(fact_norm)

        print("JOB: sent", flush=True)
        return

    print("JOB: failed to generate unique post after retries", flush=True)


# ---------- MAIN ----------
async def main():
    cfg = Config(
        tg_token=env_required("TG_TOKEN"),
        tg_channel=env_required("TG_CHANNEL"),
        yc_folder_id=env_required("YC_FOLDER_ID"),
        yc_api_key=env_required("YC_API_KEY"),
    )

    init_db()

    bot = Bot(token=cfg.tg_token)
    async with bot.context() as bot_ctx:
        scheduler = AsyncIOScheduler(timezone=ZoneInfo(cfg.tz))

        scheduler.add_job(
            post_job,
            trigger="cron",
            hour="8-22/2",
            minute=0,
            args=[cfg, bot_ctx],
            id="space_post_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        scheduler.start()
        print("Scheduler started", flush=True)

        if os.getenv("STARTUP_POST", "0") == "1":
            try:
                await post_job(cfg, bot_ctx)
            except Exception as e:
                print("First post failed:", repr(e), flush=True)

        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
