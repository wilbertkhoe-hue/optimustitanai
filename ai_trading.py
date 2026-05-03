"""
ai_trading.py — TITAN AI Trading Engine V6 (XGBoost Macro Edition)
Import oleh app.py via: from ai_trading import jalankan_ai_trading_v6
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import csv
import json
import requests
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from textblob import TextBlob

RESET_BRAIN = False

output_web: list[str] = []

def lapor(teks: str):
    print(teks)
    output_web.append(str(teks))

def hitung_skor_sentimen(daftar_berita):
    if not daftar_berita:
        return 0.0
    skor_total = 0.0
    for b in daftar_berita:
        konten = b.get('content', b)
        judul  = konten.get('title', b.get('title', ''))
        skor_total += TextBlob(judul).sentiment.polarity
    return skor_total / len(daftar_berita)

def kelola_log_evaluasi(harga_live, pred_1, pred_3, pred_14, pred_90, ticker,
                         tol_3=5.0, tol_14=5.0, tol_90=5.0):
    ticker_bersih = ticker.replace("=", "_")
    draft_file = f"ai_draft_v6_{ticker_bersih}.json"
    log_file   = f"ai_evaluasi_log_{ticker_bersih}.csv"

    sekarang        = datetime.now()
    tanggal_sekarang = sekarang.strftime("%Y-%m-%d")
    tanggal_kemarin  = (sekarang - timedelta(days=1)).strftime("%Y-%m-%d")
    tgl_3_hari      = (sekarang - timedelta(days=3)).strftime("%Y-%m-%d")
    tgl_14_hari     = (sekarang - timedelta(days=14)).strftime("%Y-%m-%d")
    tgl_90_hari     = (sekarang - timedelta(days=90)).strftime("%Y-%m-%d")

    draft = {
        "history_prediksi": {},
        "winrate_history": {
            "3D":  {"total": 0, "correct": 0},
            "14D": {"total": 0, "correct": 0},
            "90D": {"total": 0, "correct": 0}
        }
    }
    if os.path.exists(draft_file):
        try:
            with open(draft_file) as f:
                draft.update(json.load(f))
        except json.JSONDecodeError:
            pass

    history = draft["history_prediksi"]
    winrate = draft["winrate_history"]
    pesan   = ""

    def cek_akurasi(tgl_target, nama_target, key_prediksi, tol):
        if tgl_target in history:
            data_hist = history[tgl_target]
            pred_lama = (data_hist.get(key_prediksi, data_hist)
                         if isinstance(data_hist, dict) else data_hist)
            error_pct = abs(harga_live - pred_lama) / harga_live * 100
            status = ("🎯 TEPAT!"  if error_pct <= tol
                      else "⚠️ LUMAYAN" if error_pct <= tol + 2.0
                      else "❌ MELESET")
            return (f"▶ {nama_target:<14} (Tgl {tgl_target}) | "
                    f"Target: ${pred_lama:,.2f} | Error: {error_pct:.2f}% ({status})\n")
        return f"⏳ Belum ada data — evaluasi akan dilakukan pada {tgl_target}.\n"

    pesan += cek_akurasi(tanggal_kemarin, "Harian (1D)",   "pred_1",  2.5)
    pesan += cek_akurasi(tgl_3_hari,     "Menengah (3D)",  "pred_3",  tol_3)
    pesan += cek_akurasi(tgl_14_hari,    "Panjang (14D)",  "pred_14", tol_14)
    pesan += cek_akurasi(tgl_90_hari,    "Makro (90D)",    "pred_90", tol_90)

    # Catat 1D ke CSV
    if tanggal_kemarin in history:
        data_hist = history[tanggal_kemarin]
        pred_kemarin = (data_hist.get("pred_1") if isinstance(data_hist, dict) else data_hist)
        if pred_kemarin is not None:
            error_pct = abs(harga_live - pred_kemarin) / harga_live * 100
            file_exists = os.path.exists(log_file)
            if file_exists:
                df_log = pd.read_csv(log_file)
                mask = df_log["Tanggal Evaluasi"] == tanggal_sekarang
                if mask.any():
                    idx = df_log.index[mask][0]
                    if error_pct < df_log.at[idx, "Error Terbaik (%)"]:
                        df_log.at[idx, "Error Terbaik (%)"]    = round(error_pct, 2)
                        df_log.at[idx, "Harga Aktual Terbaik"] = round(harga_live, 2)
                        df_log.to_csv(log_file, index=False)
                else:
                    with open(log_file, "a", newline="") as f:
                        csv.writer(f).writerow([tanggal_sekarang,
                                                round(pred_kemarin, 2),
                                                round(harga_live, 2),
                                                round(error_pct, 2)])
            else:
                with open(log_file, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Tanggal Evaluasi", "Prediksi Kemarin",
                                "Harga Aktual Terbaik", "Error Terbaik (%)"])
                    w.writerow([tanggal_sekarang, round(pred_kemarin, 2),
                                round(harga_live, 2), round(error_pct, 2)])

    # Win rate multi-timeframe
    for tgl_target, key, tol, label in [
        (tgl_3_hari,  "pred_3",  tol_3,  "3D"),
        (tgl_14_hari, "pred_14", tol_14, "14D"),
        (tgl_90_hari, "pred_90", tol_90, "90D"),
    ]:
        if tgl_target in history:
            data_hist = history[tgl_target]
            pred_lama = (data_hist.get(key, data_hist) if isinstance(data_hist, dict) else data_hist)
            error = abs(harga_live - pred_lama) / harga_live * 100
            winrate[label]["total"] += 1
            if error <= tol:
                winrate[label]["correct"] += 1

    for tf, label in [("3D", "Menengah (3D)"), ("14D", "Panjang (14D)"), ("90D", "Makro (90D)")]:
        total   = winrate[tf]["total"]
        correct = winrate[tf]["correct"]
        if total > 0:
            wr = (correct / total) * 100
            pesan += f"\n📊 WIN RATE {label} : {wr:.1f}% ({correct}/{total} Tepat)"

    history[tanggal_sekarang] = {
        "pred_1":  float(pred_1),
        "pred_3":  float(pred_3),
        "pred_14": float(pred_14),
        "pred_90": float(pred_90)
    }

    # Jaga hanya 100 entri terakhir
    for k in sorted(history.keys())[:-100]:
        del history[k]

    with open(draft_file, "w") as f:
        json.dump(draft, f, indent=4)

    pesan += f"\n💾 Prediksi hari ini (1D, 3D, 14D, 90D) disimpan."

    if os.path.exists(log_file):
        try:
            df_log   = pd.read_csv(log_file)
            total    = len(df_log)
            akurat   = len(df_log[df_log["Error Terbaik (%)"] <= 2.5])
            win_rate = (akurat / total * 100) if total else 0
            pesan += f"\n\n📊 WIN RATE HISTORIS AI (1D) : {win_rate:.1f}% ({akurat}/{total} Hari)"
        except Exception:
            pesan += "\n\n📊 WIN RATE HISTORIS AI (1D) : N/A"
    else:
        pesan += "\n\n📊 WIN RATE HISTORIS AI (1D) : 0.0% (Belum ada data)"

    return pesan

def get_safe_close(ticker):
    d = yf.download(ticker, period="4y", interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        return d['Close'][ticker]
    return d['Close']

def jalankan_ai_trading_v6(ticker: str) -> str:
    global output_web
    output_web = []

    nama_aset = "BITCOIN" if "BTC" in ticker else "EMAS"
    lapor("\n" + "="*75)
    lapor(f"🚀 Memuat Data Teknikal & Makro untuk {nama_aset} ({ticker})...")
    lapor("="*75)

    ticker_bersih = ticker.replace("=", "_")
    if RESET_BRAIN:
        for f in [f"ai_draft_v6_{ticker_bersih}.json", f"ai_evaluasi_log_{ticker_bersih}.csv"]:
            if os.path.exists(f): os.remove(f)
        lapor(f"⚠️ RESET BRAIN: semua memory {ticker} dihapus!")

    close = get_safe_close(ticker)
    data_full = yf.download(ticker, period="4y", interval="1d", progress=False)
    if isinstance(data_full.columns, pd.MultiIndex):
        df = pd.DataFrame({
            col: data_full[col][ticker]
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']
        })
    else:
        df = data_full.copy()

    berita_aset = yf.Ticker(ticker).news
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype(float)
    close, high, low, volume, open_p = (df["Close"], df["High"], df["Low"],
                                        df["Volume"], df["Open"])

    lapor("🌍 Mengambil Data Makro Ekonomi...")
    df['DXY']  = get_safe_close("UUP")
    df['TNX']  = get_safe_close("^TNX")
    df['VIX']  = get_safe_close("VIXY")
    df['GSPC'] = get_safe_close("SPY")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get("https://api.alternative.me/fng/?limit=1500",
                         headers=headers, timeout=30)
        r.raise_for_status()
        fng_df = pd.DataFrame(r.json()['data'])
        fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'].astype(int), unit='s')
        fng_df.set_index('timestamp', inplace=True)
        fng_df.index = fng_df.index.normalize()
        df.index = df.index.tz_localize(None).normalize()
        df = df.join(fng_df['value'].astype(float).rename('FNG'), how='left')
    except Exception as e:
        df['FNG'] = 50.0
        lapor(f"⚠️ Gagal ambil Fear & Greed: {e}. Pakai nilai netral (50).")

    df[['DXY','TNX','VIX','GSPC','FNG']] = df[['DXY','TNX','VIX','GSPC','FNG']].ffill().bfill()

    # ── 24+ FITUR TEKNIKAL ──
    df["MA20"]   = SMAIndicator(close, 20).sma_indicator()
    df["MA50"]   = SMAIndicator(close, 50).sma_indicator()
    df["MA200"]  = SMAIndicator(close, 200).sma_indicator()
    df["EMA20"]  = EMAIndicator(close, 20).ema_indicator()
    df["EMA50"]  = EMAIndicator(close, 50).ema_indicator()
    df["EMA200"] = EMAIndicator(close, 200).ema_indicator()
    df["RSI"]    = RSIIndicator(close).rsi()
    macd_ind     = MACD(close)
    df["MACD"], df["MACD_SIGNAL"] = macd_ind.macd(), macd_ind.macd_signal()
    bb           = BollingerBands(close)
    df["BB_LOW"], df["BB_HIGH"], df["BB_MID"] = (bb.bollinger_lband(),
                                                   bb.bollinger_hband(),
                                                   bb.bollinger_mavg())
    df["MOMENTUM"]   = close - close.shift(10)
    df["ROC"]        = ROCIndicator(close).roc()
    df["ATR"]        = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["OBV"]        = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["MFI"]        = MFIIndicator(high, low, close, volume).money_flow_index()
    df["ADX"]        = ADXIndicator(high, low, close).adx()
    df["STOCH"]      = StochasticOscillator(high, low, close).stoch()
    df["VOL_MA"]     = volume.rolling(20).mean()
    df["FIBO"]       = low + (high - low) * 0.618
    df['Body_Size']    = abs(close - open_p)
    df['Upper_Shadow'] = high - df[['Open','Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open','Close']].min(axis=1) - low
    df['Return_1D']    = close.pct_change(1)
    df['Day_Of_Week']  = df.index.dayofweek

    fitur_teknikal = ["MA20","MA50","MA200","EMA20","EMA50","EMA200",
                      "RSI","MACD","MACD_SIGNAL","BB_LOW","BB_HIGH","BB_MID",
                      "MOMENTUM","ROC","ATR","OBV","MFI","ADX","STOCH","VOL_MA",
                      "FIBO","Body_Size","Upper_Shadow","Lower_Shadow","Return_1D","Day_Of_Week"]

    skor_sentimen = hitung_skor_sentimen(berita_aset)
    df['SENTIMEN_NEWS'] = skor_sentimen
    fitur_makro = fitur_teknikal + ['DXY','TNX','VIX','GSPC','FNG','SENTIMEN_NEWS']

    # Target
    df['Target_1D']     = close.shift(-1)
    df['Target_3D_Base']  = close.shift(-3)
    df['Target_3D_Max']   = high.rolling(3).max().shift(-3)
    df['Target_3D_Min']   = low.rolling(3).min().shift(-3)
    df['Target_14D_Base'] = close.shift(-14)
    df['Target_14D_Max']  = high.rolling(14).max().shift(-14)
    df['Target_14D_Min']  = low.rolling(14).min().shift(-14)
    df['Target_90D_Base'] = close.shift(-90)
    df['Target_90D_Max']  = high.rolling(90).max().shift(-90)
    df['Target_90D_Min']  = low.rolling(90).min().shift(-90)

    data_hari_ini_teknikal = df[fitur_teknikal].iloc[-1:]
    data_hari_ini_makro    = df[fitur_makro].iloc[-1:]
    harga_sekarang = float(close.iloc[-1])

    df_base        = df.dropna(subset=fitur_makro)
    df_clean_short = df_base.dropna(subset=['Target_1D'])
    df_clean_mid   = df_base.dropna(subset=['Target_3D_Base','Target_3D_Max','Target_3D_Min'])
    df_clean_long  = df_base.dropna(subset=['Target_14D_Base','Target_14D_Max','Target_14D_Min'])
    df_clean_macro = df_base.dropna(subset=['Target_90D_Base','Target_90D_Max','Target_90D_Min'])

    last    = df.iloc[-2]
    last_3  = df.iloc[-5]
    last_14 = df.iloc[-15]
    last_90 = df.iloc[-91]

    def hitung_opini(dt_awal, dt_banding, harga_banding):
        return {
            "MA20":     int(harga_banding > dt_banding["MA20"]),
            "MA50":     int(harga_banding > dt_banding["MA50"]),
            "MA200":    int(harga_banding > dt_banding["MA200"]),
            "EMA20":    int(harga_banding > dt_banding["EMA20"]),
            "EMA50":    int(harga_banding > dt_banding["EMA50"]),
            "EMA200":   int(harga_banding > dt_banding["EMA200"]),
            "RSI":      int(dt_awal["RSI"] < 70 and dt_awal["RSI"] > 40),
            "MACD":     int(dt_awal["MACD"] > dt_awal["MACD_SIGNAL"]),
            "BB_MID":   int(harga_banding > dt_awal["BB_MID"]),
            "MOMENTUM": int(dt_awal["MOMENTUM"] > 0),
            "ROC":      int(dt_awal["ROC"] > 0),
            "ATR":      int(dt_awal["ATR"] > 0),
            "OBV":      int(dt_awal["OBV"] > dt_banding["OBV"]),
            "MFI":      int(dt_awal["MFI"] < 80),
            "ADX":      int(dt_awal["ADX"] > 20),
            "STOCH":    int(dt_awal["STOCH"] < 80),
            "VOL_MA":   int(float(volume.iloc[-1]) > dt_awal["VOL_MA"]),
            "FIBO":     int(harga_banding > dt_banding["FIBO"]),
            "BB_LOW":   int(harga_banding > dt_awal["BB_LOW"]),
            "BB_HIGH":  int(harga_banding < dt_awal["BB_HIGH"]),
            "RSI_50":   int(dt_awal["RSI"] > 50),
            "MACD_POS": int(dt_awal["MACD"] > 0),
            "EMA_CROSS":int(dt_awal["EMA20"] > dt_awal["EMA50"]),
            "MA_CROSS": int(dt_awal["MA20"] > dt_awal["MA50"]),
        }

    total_short = sum(hitung_opini(last, last,    harga_sekarang).values())
    total_mid   = sum(hitung_opini(last, last_3,  harga_sekarang).values())
    total_long  = sum(hitung_opini(last, last_14, harga_sekarang).values())
    total_macro = sum(hitung_opini(last, last_90, harga_sekarang).values())

    opini_short = "NAIK 🔼" if total_short >= 12 else "TURUN 🔽"
    opini_mid   = "NAIK 🔼" if total_mid   >= 12 else "TURUN 🔽"
    opini_long  = "NAIK 📈" if total_long  >= 12 else "TURUN 📉"
    opini_macro = "BULL MARKET 🔥" if total_macro >= 12 else "BEAR MARKET ❄️"

    prob_atas_14d   = (total_long / 24.0) * 100
    prob_bawah_14d  = 100.0 - prob_atas_14d
    prob_macro_bull = min(100, (total_macro / 24.0) * 80 + (float(last['FNG']) / 100) * 20)
    prob_macro_bear = 100.0 - prob_macro_bull

    xgb_params = {'n_estimators':200, 'learning_rate':0.05, 'max_depth':5,
                  'subsample':0.8, 'random_state':42}

    lapor("🤖 Training model XGBoost...")
    model_1       = XGBRegressor(**xgb_params).fit(df_clean_short[fitur_teknikal], df_clean_short['Target_1D'])
    model_3_base  = XGBRegressor(**xgb_params).fit(df_clean_mid[fitur_teknikal],   df_clean_mid['Target_3D_Base'])
    model_3_max   = XGBRegressor(**xgb_params).fit(df_clean_mid[fitur_teknikal],   df_clean_mid['Target_3D_Max'])
    model_3_min   = XGBRegressor(**xgb_params).fit(df_clean_mid[fitur_teknikal],   df_clean_mid['Target_3D_Min'])
    model_14_base = XGBRegressor(**xgb_params).fit(df_clean_long[fitur_teknikal],  df_clean_long['Target_14D_Base'])
    model_14_max  = XGBRegressor(**xgb_params).fit(df_clean_long[fitur_teknikal],  df_clean_long['Target_14D_Max'])
    model_14_min  = XGBRegressor(**xgb_params).fit(df_clean_long[fitur_teknikal],  df_clean_long['Target_14D_Min'])
    model_90_base = XGBRegressor(**xgb_params).fit(df_clean_macro[fitur_makro],    df_clean_macro['Target_90D_Base'])
    model_90_max  = XGBRegressor(**xgb_params).fit(df_clean_macro[fitur_makro],    df_clean_macro['Target_90D_Max'])
    model_90_min  = XGBRegressor(**xgb_params).fit(df_clean_macro[fitur_makro],    df_clean_macro['Target_90D_Min'])

    pred_1       = float(model_1.predict(data_hari_ini_teknikal)[0])
    pred_3_base  = float(model_3_base.predict(data_hari_ini_teknikal)[0])
    pred_3_max   = float(model_3_max.predict(data_hari_ini_teknikal)[0])
    pred_3_min   = float(model_3_min.predict(data_hari_ini_teknikal)[0])
    pred_14_base = float(model_14_base.predict(data_hari_ini_teknikal)[0])
    pred_14_max  = float(model_14_max.predict(data_hari_ini_teknikal)[0])
    pred_14_min  = float(model_14_min.predict(data_hari_ini_teknikal)[0])
    pred_90_base = float(model_90_base.predict(data_hari_ini_makro)[0])
    pred_90_max  = float(model_90_max.predict(data_hari_ini_makro)[0])
    pred_90_min  = float(model_90_min.predict(data_hari_ini_makro)[0])

    arah_ml = "NAIK 🔼" if pred_1 > harga_sekarang else "TURUN 🔽"
    divergensi = ""
    if opini_short != arah_ml:
        divergensi = (f"\n⚠️ Divergensi: Teknikal = {opini_short}, "
                      f"XGBoost = {arah_ml}. Pasar harian labil!")

    skor_dxy  = 1 if float(last["DXY"])  < float(last_3["DXY"])  else 0
    skor_tnx  = 1 if float(last["TNX"])  < float(last_3["TNX"])  else 0
    skor_vix  = 1 if float(last["VIX"])  < 20.0                  else 0
    skor_gspc = 1 if float(last["GSPC"]) > float(last_3["GSPC"]) else 0
    skor_fng  = 1 if float(last["FNG"])  > 40.0                  else 0
    total_makro_skor = skor_dxy + skor_tnx + skor_vix + skor_gspc + skor_fng
    skor_ai   = 1 if pred_1 > harga_sekarang else 0
    total_30  = total_short + total_makro_skor + skor_ai

    if total_30 >= 20:
        status_market = "MARKET KUAT 🚀"
        rekomendasi   = f"{status_market}\n  Entry/Hold aman."
    elif total_30 >= 12:
        status_market = "MARKET SIDEWAYS ⚖️"
        rekomendasi   = f"{status_market}\n  Waspada false breakout."
    else:
        status_market = "MARKET BAHAYA ⚠️"
        rekomendasi   = f"{status_market}\n  Hindari entry, amankan modal."

    if divergensi:
        rekomendasi += divergensi

    # Conviction & toleransi dinamis
    conv_3_bull   = (total_mid  / 24.0) * 100
    conv_14_bull  = prob_atas_14d
    conv_90_bull  = prob_macro_bull

    eff_pred_3  = (pred_3_max  if conv_3_bull  >= 65 else
                   pred_3_min  if (100-conv_3_bull)  >= 65 else pred_3_base)
    eff_pred_14 = (pred_14_max if conv_14_bull >= 65 else
                   pred_14_min if (100-conv_14_bull) >= 65 else pred_14_base)
    eff_pred_90 = (pred_90_max if conv_90_bull >= 65 else
                   pred_90_min if (100-conv_90_bull) >= 65 else pred_90_base)

    tol_3  = 2.5 if max(conv_3_bull,  100-conv_3_bull)  >= 65 else 3.0
    tol_14 = 2.5 if max(conv_14_bull, 100-conv_14_bull) >= 65 else 3.0
    tol_90 = 2.5 if max(conv_90_bull, 100-conv_90_bull) >= 65 else 3.0

    hasil_logger = kelola_log_evaluasi(
        harga_sekarang, pred_1, eff_pred_3, eff_pred_14, eff_pred_90,
        ticker, tol_3, tol_14, tol_90
    )

    # ── OUTPUT ──
    lapor("\n" + "💸"*25)
    lapor(f" WALL STREET AI PRO MAX V6.0 (MACRO EDITION): {nama_aset} ({ticker}) ")
    lapor("💸"*25)
    lapor(hasil_logger)

    lapor("\n[ 1. OPINI AI BERDASARKAN INDIKATOR TEKNIKAL & MAKRO ]")
    lapor("-" * 75)
    lapor(f"Harga Real-Time Saat Ini : ${harga_sekarang:,.2f}")
    lapor(f"▶ Jangka Pendek  (Besok)   : {total_short}/24 -> {opini_short}")
    lapor(f"▶ Jangka Menengah (3 Hari) : {total_mid}/24 -> {opini_mid}")
    lapor(f"▶ Jangka Panjang (14 Hari) : {total_long}/24 -> {opini_long}")
    lapor(f"▶ Kuartal Makro  (3 Bulan) : {total_macro}/24 -> {opini_macro}")

    lapor("\n  [ STATUS 5 INDIKATOR MAKRO ]")
    lapor(f"  • Fear & Greed  : {last['FNG']}")
    lapor(f"  • S&P 500       : {last['GSPC']:,.2f}")
    lapor(f"  • DXY           : {last['DXY']:.2f}")
    lapor(f"  • US 10-Yr Yield: {last['TNX']:.2f}%")
    lapor(f"  • VIX           : {last['VIX']:.2f}")
    status_nlp = ("BULLISH 📈" if skor_sentimen > 0.05
                  else "BEARISH 📉" if skor_sentimen < -0.05
                  else "NETRAL 😐")
    lapor(f"  • Sentimen Berita: {skor_sentimen:.2f} ({status_nlp})")

    lapor("\n[ 2. TARGET HARGA AI (XGBOOST) ]")
    lapor("-" * 75)
    lapor(f"▶ Jangka Pendek  (Besok)    : ${pred_1:,.2f} ({arah_ml})")
    lapor(f"▶ Jangka Menengah (3 Hari):")
    lapor(f"  Tengah    : ${pred_3_base:,.2f}")
    lapor(f"  Batas Atas: ${pred_3_max:,.2f}")
    lapor(f"  Batas Bawah: ${pred_3_min:,.2f}")
    lapor(f"▶ Jangka Panjang (14 Hari):")
    lapor(f"  Tengah    : ${pred_14_base:,.2f}")
    lapor(f"  Batas Atas: ${pred_14_max:,.2f} 🔼 ({prob_atas_14d:.1f}%)")
    lapor(f"  Batas Bawah: ${pred_14_min:,.2f} 🔽 ({prob_bawah_14d:.1f}%)")
    lapor(f"▶ Kuartal Makro (90 Hari):")
    lapor(f"  Tengah    : ${pred_90_base:,.2f}")
    lapor(f"  ATH Makro : ${pred_90_max:,.2f} 📈 ({prob_macro_bull:.1f}% Bull)")
    lapor(f"  Bottom    : ${pred_90_min:,.2f} 📉 ({prob_macro_bear:.1f}% Bear)")

    lapor("\n[ 3. KESIMPULAN FINAL (SKOR SUPER 30) ]")
    lapor("-" * 75)
    lapor(f"📊 {total_short} (Teknikal) + {total_makro_skor} (Makro) + {skor_ai} (AI) = {total_30}/30")
    lapor(f"🎯 REKOMENDASI: {rekomendasi}")
    lapor("="*75 + "\n")

    return "\n".join(output_web)
