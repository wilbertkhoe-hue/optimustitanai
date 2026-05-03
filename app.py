"""
TITAN V7 - Backend Flask
Menyatukan: Ollama Chat + XGBoost AI Trading + Macro Data
"""

from flask import Flask, render_template, request, jsonify
import requests
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
import json
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================= CONFIG =================
OLLAMA_URL  = "http://127.0.0.1:11434/api/generate"
MODEL_NAME  = "titan-brain:latest"
DB_PATH     = "titan_brain.db"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")   # opsional, untuk data makro FRED

# ================= DB =================
@contextmanager
def get_db():
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
        for col, col_type in [("timestamp","TEXT"), ("role","TEXT"), ("content","TEXT")]:
            if col not in cols:
                c.execute(f"ALTER TABLE chat_history ADD COLUMN {col} {col_type}")
    logger.info("DB migration selesai.")

def save_chat(role: str, content: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, ts)
        )

# ================= MARKET DATA (server-side, no CORS) =================
def fetch_btc_price() -> float | None:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            timeout=5
        )
        r.raise_for_status()
        return float(r.json()["bitcoin"]["usd"])
    except Exception as e:
        logger.warning("Gagal ambil BTC: %s", e)
        return None

def fetch_gold_price() -> float | None:
    try:
        import yfinance as yf
        g = yf.Ticker("GC=F").history(period="1d")
        if not g.empty:
            return float(g["Close"].iloc[-1])
    except Exception as e:
        logger.warning("Gagal ambil Gold: %s", e)
    return None

def fetch_yahoo_quote(symbol: str) -> float | None:
    """Ambil harga dari Yahoo Finance — server-side, tidak kena CORS."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        result = r.json()["chart"]["result"][0]
        return float(result["meta"]["regularMarketPrice"])
    except Exception as e:
        logger.warning("Gagal ambil Yahoo %s: %s", symbol, e)
        return None

def fetch_macro_data() -> dict:
    """Kumpulkan data makro di server — hasilnya dikirim ke frontend."""
    vix   = fetch_yahoo_quote("%5EVIX")
    sp500 = fetch_yahoo_quote("%5EGSPC")
    dxy   = fetch_yahoo_quote("DX-Y.NYB")

    # Fear & Greed
    fng_value, fng_label = None, None
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        d = r.json()["data"][0]
        fng_value = int(d["value"])
        fng_label = d["value_classification"]
    except Exception as e:
        logger.warning("Gagal ambil F&G: %s", e)

    # Suku bunga & inflasi via FRED (opsional)
    rate, inflation = None, None
    if FRED_API_KEY:
        try:
            base = "https://api.stlouisfed.org/fred/series/observations"
            params_rate = {"series_id": "FEDFUNDS", "api_key": FRED_API_KEY,
                           "file_type": "json", "limit": 1, "sort_order": "desc"}
            r = requests.get(base, params=params_rate, timeout=8)
            rate = float(r.json()["observations"][0]["value"])
        except Exception as e:
            logger.warning("FRED rate gagal: %s", e)
        try:
            params_inf = {"series_id": "CPIAUCSL", "api_key": FRED_API_KEY,
                          "file_type": "json", "limit": 1, "sort_order": "desc"}
            r = requests.get(base, params=params_inf, timeout=8)
            inflation = r.json()["observations"][0]["value"]
        except Exception as e:
            logger.warning("FRED inflation gagal: %s", e)

    return {
        "vix":       vix,
        "sp500":     sp500,
        "dxy":       dxy,
        "fng_value": fng_value,
        "fng_label": fng_label,
        "rate":      rate,
        "inflation": inflation,
    }

# ================= AI TRADING (XGBoost) =================
def run_ai_trading(ticker: str) -> dict:
    """
    Jalankan analisis XGBoost dari ai_trading.py.
    Mengembalikan dict {output, signal, sentiment}.
    """
    try:
        # Import fungsi dari file trading terpisah
        from ai_trading import jalankan_ai_trading_v6, output_web
        result_text = jalankan_ai_trading_v6(ticker)

        # Deteksi signal dari output teks
        signal = "HOLD ⚖️"
        if "MARKET KUAT" in result_text or "NAIK 🔼" in result_text:
            signal = "BUY 🚀"
        elif "MARKET BAHAYA" in result_text or "TURUN 🔽" in result_text:
            signal = "SELL ⚠️"

        # Deteksi sentimen kasar dari teks
        sentiment = 50
        if "BULLISH" in result_text:
            sentiment = 75
        elif "BEARISH" in result_text:
            sentiment = 25

        return {"output": result_text, "signal": signal, "sentiment": sentiment}

    except ImportError:
        logger.error("ai_trading.py tidak ditemukan! Pastikan file ada di folder yang sama.")
        return {
            "output": "⚠️ File ai_trading.py tidak ditemukan.\nPastikan file ada di folder yang sama dengan app.py.",
            "signal": "ERROR",
            "sentiment": 50
        }
    except Exception as e:
        logger.error("Error saat run AI trading: %s", e)
        return {
            "output": f"⚠️ Error saat analisis: {str(e)}",
            "signal": "ERROR",
            "sentiment": 50
        }

# ================= OLLAMA CHAT =================
def build_chat_prompt(user_msg: str, btc: float | None, gold: float | None) -> str:
    btc_str  = f"${btc:,.2f}"  if btc  else "Tidak tersedia"
    gold_str = f"${gold:,.2f}" if gold else "Tidak tersedia"
    return f"""Kamu adalah analis hedge fund senior.

WAJIB:
- Jawaban harus DALAM dan LOGIS
- Gunakan reasoning makro: suku bunga, inflasi, likuiditas
- Fokus hanya 1 aset

DATA PASAR:
BTC: {btc_str}
GOLD: {gold_str}

OUTPUT FORMAT:
📊 OUTLOOK
🎯 Target akhir tahun: $XXXXX
📈 Range: $XXXXX — $XXXXX
🟢 Bull Case (XX%) - analisis nyata
🟡 Base Case (XX%) - analisis
🔴 Bear Case (XX%) - analisis
🧠 Insight: (1 kalimat tajam)
⚡ Confidence: XX%

USER: {user_msg}""".strip()

def call_ollama(prompt: str, temperature: float, num_predict: int, timeout: int) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 2048, "num_predict": num_predict}
        },
        timeout=timeout
    )
    resp.raise_for_status()
    output = resp.json().get("response", "").strip()
    if not output:
        raise ValueError("Ollama mengembalikan respons kosong.")
    return output

def run_chat(user_msg: str) -> dict:
    btc  = fetch_btc_price()
    gold = fetch_gold_price()
    prompt = build_chat_prompt(user_msg, btc, gold)

    # TRY 1
    try:
        output = call_ollama(prompt, temperature=0.5, num_predict=500, timeout=35)
        if len(output) >= 100 and "Target" in output:
            return {"reply": output, "sentiment": 50, "news": []}
        logger.warning("Output tidak valid, coba force mode.")
    except requests.Timeout:
        logger.warning("Timeout normal mode.")
    except Exception as e:
        logger.error("Error normal mode: %s", e)

    # TRY 2
    try:
        output = call_ollama(
            "Jawab lebih dalam, konkret, dan berbasis data. " + prompt,
            temperature=0.3, num_predict=600, timeout=45
        )
        if output:
            return {"reply": output, "sentiment": 50, "news": []}
    except requests.Timeout:
        logger.warning("Timeout force mode.")
    except Exception as e:
        logger.error("Error force mode: %s", e)

    # FALLBACK
    btc_str  = f"${btc:,.2f}"  if btc  else "N/A"
    gold_str = f"${gold:,.2f}" if gold else "N/A"
    return {
        "reply": (
            f"📊 MARKET SNAPSHOT\n\nBTC: {btc_str}\nGOLD: {gold_str}\n\n"
            f"Titan sedang overload. Pastikan Ollama berjalan:\n"
            f"  `ollama serve`\n"
            f"  `ollama pull {MODEL_NAME}`"
        ),
        "sentiment": 50,
        "news": []
    }

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/proses", methods=["POST"])
def proses():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "CHAT")

    # ── MODE: ANALISIS (XGBoost AI Trading) ──
    if mode == "ANALISIS":
        ticker = data.get("asset", "BTC-USD")
        logger.info("Analisis XGBoost untuk: %s", ticker)
        result = run_ai_trading(ticker)
        return jsonify({
            "reply":     result["output"],
            "signal":    result["signal"],
            "sentiment": result["sentiment"],
            "news":      []
        })

    # ── MODE: CHAT (Ollama LLM) ──
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    save_chat("user", msg)
    result = run_chat(msg)
    save_chat("ai", result["reply"])
    return jsonify(result)

@app.route("/macro")
def macro():
    """Endpoint khusus untuk data makro — menghindari CORS di frontend."""
    data = fetch_macro_data()
    return jsonify(data)

@app.route("/market")
def market():
    """Harga BTC & Gold real-time dari server."""
    return jsonify({
        "btc":  fetch_btc_price(),
        "gold": fetch_gold_price()
    })

# ================= RUN =================
if __name__ == "__main__":
    print("🚀 TITAN V7 ACTIVE")
    migrate_db()
    app.run(debug=True, port=5000)
