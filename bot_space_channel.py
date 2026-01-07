# bot_space_channel.py
import asyncio
import os
import hashlib
import sqlite3
import random
import re
from dataclasses import dataclass
from datetime import datetime
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
    tg_channel: str          # например: "@pro_kosmos_knl"
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
        (key, datetime.utcnow().isoformat(timespec="seconds"))
    )
    con.commit()
    con.close()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(s: str) -> str:
    # убираем пунктуацию/спецсимволы, приводим к нижнему регистру, схлопываем пробелы
    s = s.lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


# ---------- PROMPTS (VARIETY) ----------
TOPIC_HINTS = [
    "чёрные дыры и горизонты событий",
    "экзопланеты и поиски жизни",
    "звёздные эволюции: от протозвезды до сверхновой",
    "нейтронные звёзды и пульсары",
    "галактики и структура Млечного Пути",
    "тёмная материя и тёмная энергия",
    "планеты Солнечной системы и их атмосферы",
    "кометы, астероиды и метеорные потоки",
    "космические телескопы и методы наблюдений",
    "гравитационные волны и слияния компактных объектов",
]

SCENE_HINTS = [
    "туманность с нитями газа",
    "спиральная галактика крупным планом",
    "кольца газового гиганта в контровом свете",
    "ледяной спутник с трещинами и свечением",
    "звёздное скопление на фоне космической пыли",
    "протопланетный диск вокруг молодой звезды",
    "сияние в верхних слоях атмосферы планеты",
]


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
        "Требования: без людей, без текста/надписей/логотипов, реалистично, научная визуализация, 4k. "
        f"Тема: {topic}. Сцена: {scene}. "
        "Выведи ТОЛЬКО prompt, без пояснений, без кавычек."
    ])
    prompt_text = prompt_res.text.strip()

    fact_res = gpt.run([
        "Напиши 1 короткий и точный научный факт о космосе (1–2 предложения). "
        "Без префиксов типа 'Факт дня', без списков, без метаданных. "
        f"Тема: {topic}. Свяжи факт со сценой: {scene}."
    ])
    fact_text = fact_res.text.strip().replace("\n", " ")

    # IMAGE (YandexART)
    img_model = sdk.models.image_generation("yandex-art")
    operation = img_model.run_deferred(prompt_text)
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

    for attempt in range(1, 8):  # до 7 перегенераций, чтобы уйти от дублей
        image_bytes, prompt_text, fact_text = await asyncio.to_thread(
            generate_image_and_fact_sync, cfg
        )
        print(f"JOB: generated {len(image_bytes)} bytes", flush=True)

        fact_norm = normalize_text(fact_text)
        fact_key = "fact_norm:" + sha256_hex(fact_norm.encode("utf-8"))
        img_key = "img:" + sha256_hex(image_bytes)

        # “умнее”: проверяем именно нормализованный текст факта (почти одинаковые тоже считаем дублем)
        if was_posted(fact_key):
            print(f"JOB: duplicate fact (normalized), retry {attempt}/7", flush=True)
            continue

        caption = f"#ФактДня: {fact_text}\n\nПодписывайся! @pro_kosmos_knl"
        await send_photo_with_retries(bot, cfg.tg_channel, image_bytes, caption)

        mark_posted(fact_key)
        mark_posted(img_key)

        print("JOB: sent", flush=True)
        return

    print("JOB: failed to generate unique post after 7 tries", flush=True)


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
    async with bot.context() as bot:
        scheduler = AsyncIOScheduler(timezone=ZoneInfo(cfg.tz))

        scheduler.add_job(
            post_job,
            trigger="cron",
            hour="8-22/2",
            minute=0,
            args=[cfg, bot],
            id="space_post_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        scheduler.start()
        print("Scheduler started", flush=True)

        if os.getenv("STARTUP_POST", "0") == "1":
            try:
                await post_job(cfg, bot)
            except Exception as e:
                print("First post failed:", repr(e), flush=True)

        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
