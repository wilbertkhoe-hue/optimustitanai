from flask import Flask, render_template, request, jsonify
import requests
import yfinance as yf
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================= CONFIG =================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "titan-brain:latest"
DB_PATH = "titan_brain.db"

TIMEOUT_NORMAL = 35
TIMEOUT_FORCE = 45
MIN_OUTPUT_LENGTH = 100

# ================= DB =================
@contextmanager
def get_db():
    """Context manager untuk koneksi DB — otomatis commit dan close."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        c.execute("PRAGMA table_info(chat_history)")
        cols = {col[1] for col in c.fetchall()}

        for col, col_type in [("timestamp", "TEXT"), ("role", "TEXT"), ("content", "TEXT")]:
            if col not in cols:
                c.execute(f"ALTER TABLE chat_history ADD COLUMN {col} {col_type}")

    logger.info("DB migration selesai.")


def save_chat(role: str, content: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, ts),
        )


# ================= MARKET =================
def fetch_btc_price() -> float | None:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            timeout=5,
        )
        r.raise_for_status()
        return float(r.json()["bitcoin"]["usd"])
    except requests.RequestException as e:
        logger.warning("Gagal ambil harga BTC: %s", e)
        return None
    except (KeyError, ValueError) as e:
        logger.warning("Format respons BTC tidak valid: %s", e)
        return None


def fetch_gold_price() -> float | None:
    try:
        g = yf.Ticker("GC=F").history(period="1d")
        if not g.empty:
            return float(g["Close"].iloc[-1])
        logger.warning("Data gold kosong dari yfinance.")
        return None
    except Exception as e:
        logger.warning("Gagal ambil harga gold: %s", e)
        return None


def get_market_data() -> tuple[float | None, float | None]:
    return fetch_btc_price(), fetch_gold_price()


# ================= AI =================
def build_prompt(user_msg: str, btc: float | None, gold: float | None) -> str:
    btc_str = f"${btc:,.2f}" if btc else "Tidak tersedia"
    gold_str = f"${gold:,.2f}" if gold else "Tidak tersedia"

    return f"""
Kamu adalah analis hedge fund senior.

WAJIB:
- Jawaban harus DALAM dan LOGIS
- Tidak boleh template kosong
- Gunakan reasoning makro: suku bunga, inflasi, likuiditas
- Gunakan harga sebagai anchor
- Fokus hanya 1 aset

DATA PASAR:
BTC: {btc_str}
GOLD: {gold_str}

OUTPUT FORMAT:

📊 OUTLOOK

🎯 Target akhir tahun: $XXXXX
📈 Range: $XXXXX — $XXXXX

🟢 Bull Case (XX%)
(analisis nyata, bukan umum)

🟡 Base Case (XX%)
(analisis)

🔴 Bear Case (XX%)
(analisis)

🧠 Insight:
(1 kalimat tajam, bukan klise)

⚡ Confidence: XX%

USER:
{user_msg}
""".strip()


def call_ollama(prompt: str, temperature: float, num_predict: int, timeout: int) -> str:
    """
    Panggil Ollama API. Raise exception kalau gagal atau output tidak valid.
    """
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 2048,
                "num_predict": num_predict,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()

    output = resp.json().get("response", "").strip()
    if not output:
        raise ValueError("Ollama mengembalikan respons kosong.")

    return output


def is_valid_output(output: str) -> bool:
    """Cek apakah output cukup panjang dan mengandung konten yang diharapkan."""
    return len(output) >= MIN_OUTPUT_LENGTH and "Target" in output


def run_llm(user_msg: str) -> tuple[str, int, list]:
    btc, gold = get_market_data()
    prompt = build_prompt(user_msg, btc, gold)

    # --- TRY 1: Normal mode ---
    try:
        output = call_ollama(prompt, temperature=0.5, num_predict=500, timeout=TIMEOUT_NORMAL)
        if is_valid_output(output):
            logger.info("LLM berhasil (normal mode).")
            return output, 50, []
        logger.warning("Output tidak valid (normal mode), coba force mode.")
    except requests.Timeout:
        logger.warning("Timeout di normal mode (>%ds).", TIMEOUT_NORMAL)
    except requests.RequestException as e:
        logger.error("Request error di normal mode: %s", e)
    except ValueError as e:
        logger.warning("Output tidak valid: %s", e)

    # --- TRY 2: Force mode (lebih panjang, temperature lebih rendah) ---
    force_prompt = "Jawab lebih dalam, konkret, dan berbasis data. " + prompt
    try:
        output = call_ollama(force_prompt, temperature=0.3, num_predict=600, timeout=TIMEOUT_FORCE)
        if output:
            logger.info("LLM berhasil (force mode).")
            return output, 50, []
        logger.warning("Output kosong di force mode.")
    except requests.Timeout:
        logger.warning("Timeout di force mode (>%ds). Ollama mungkin overload.", TIMEOUT_FORCE)
    except requests.RequestException as e:
        logger.error("Request error di force mode: %s", e)
    except ValueError as e:
        logger.warning("Output tidak valid di force mode: %s", e)

    # --- FALLBACK: Snapshot statis ---
    logger.error("Semua upaya LLM gagal. Mengembalikan fallback.")
    btc_str = f"${btc:,.2f}" if btc else "N/A"
    gold_str = f"${gold:,.2f}" if gold else "N/A"

    fallback = (
        f"📊 MARKET SNAPSHOT\n\n"
        f"BTC: {btc_str}\n"
        f"GOLD: {gold_str}\n\n"
        f"Titan sedang overload atau tidak dapat dijangkau.\n"
        f"Pastikan Ollama berjalan dengan: `ollama serve`\n"
        f"Dan model tersedia: `ollama pull {MODEL_NAME}`\n\n"
        f"Coba ulangi dalam beberapa saat."
    )
    return fallback, 50, []


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/proses", methods=["POST"])
def proses():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()

    if not msg:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    save_chat("user", msg)
    reply, sentiment, news = run_llm(msg)
    save_chat("ai", reply)

    return jsonify({"reply": reply, "sentiment": sentiment})


# ================= RUN =================
if __name__ == "__main__":
    print("🚀 TITAN V8 ACTIVE (DEEP MODE)")
    migrate_db()
    app.run(debug=True)