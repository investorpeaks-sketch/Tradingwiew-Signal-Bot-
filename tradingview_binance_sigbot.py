# tradingview_binance_sigbot.py
"""
TradingView Webhook → Binance Fiyat → Telegram Sinyal Bot
Mantık: HTF trend + LTF pullback + EMA/ATR/MACD/RSI + hacim + risk/ödül
Patron: HTF=4h, LTF=1h, watchlist Binance tokenları
"""

import os
import time
import logging
import sqlite3
import html
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests
from binance.client import Client
import ta
from flask import Flask, request, jsonify

# ---------------- CONFIG ----------------
WATCHLIST_TV = [
    "BINANCE:SOLUSDT",
    "BINANCE:TRXUSDT",
    "BINANCE:INJUSDT",
    "BINANCE:LDOUSDT",
    "BINANCE:AAVEUSDT",
    "BINANCE:AVAXUSDT",
    "BINANCE:SUIUSDT",
    "BINANCE:ADAUSDT",
    "BINANCE:DOTUSDT",
    "BINANCE:XLMUSDT"
]

HTF = "4h"
LTF = "1h"
HTF_EMA_FAST = 50
HTF_EMA_SLOW = 200
LTF_EMA_FAST = 20
LTF_EMA_SLOW = 50
ATR_PERIOD = 14
ATR_K = 1.8
PULLBACK_TOL = 0.007
VOLUME_MULTIPLIER = 1.2
DB_PATH = "signals.db"
LOG_LEVEL = logging.INFO
MAX_RETRIES = 3
KLIMIT_HTF = 500
KLIMIT_LTF = 200
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
# ----------------------------------------

# Env vars
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise SystemExit("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set.")

# Logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tv-sigbot")

# Binance client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
session = requests.Session()
session.headers.update({"User-Agent": "tv-sigbot/1.0"})

# Flask app for webhook
app = Flask(__name__)

# ----------------- DB -------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        side TEXT,
        entry REAL,
        stop REAL,
        tp REAL,
        risk_pct REAL,
        reason TEXT,
        created_at TEXT,
        sent_at TEXT,
        UNIQUE(symbol, side, entry, stop)
    )
    """)
    conn.commit()
    conn.close()

def save_signal_to_db(symbol, side, entry, stop, tp, risk_pct, reason) -> bool:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO signals (symbol, side, entry, stop, tp, risk_pct, reason, created_at, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, side, float(entry), float(stop), float(tp), float(risk_pct), reason, datetime.now(timezone.utc).isoformat(), None)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def mark_signal_sent(symbol, entry, stop):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute("UPDATE signals SET sent_at=? WHERE symbol=? AND entry=? AND stop=?",
                (datetime.now(timezone.utc).isoformat(), symbol, float(entry), float(stop)))
    conn.commit()
    conn.close()

# ----------------- Telegram -----------------
def send_telegram(text):
    if TEST_MODE:
        log.info("[TEST_MODE] Telegram message would be sent:\n%s", text)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    backoff = 1
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                return True
            elif r.status_code == 429:
                log.warning("Telegram rate limited. Retry after %s s", backoff)
                time.sleep(backoff)
                backoff *= 2
            else:
                log.error("Telegram send failed: %s %s", r.status_code, r.text)
                if 500 <= r.status_code < 600:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    return False
        except Exception as e:
            log.exception("Telegram send exception: %s", e)
            time.sleep(backoff)
            backoff *= 2
    return False

# ----------------- Binance Data -----------------
def fetch_klines_futures(symbol, interval, limit=500):
    backoff = 1
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume","close_time","quote_asset_volume",
                "num_trades","taker_buy_base","taker_buy_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
            df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            return df[["open_time","open","high","low","close","volume","close_time"]]
        except Exception as e:
            log.warning("Error fetching klines %s %s: %s", symbol, interval, e)
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"Failed to fetch klines for {symbol} {interval}")

# ----------------- Indicators -----------------
def add_indicators(df, ema_periods=(20,50,200), atr_period=14):
    df = df.copy()
    for p in ema_periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    df["atr"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=atr_period).average_true_range()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    macd_obj = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    return df

# ----------------- Signal Logic -----------------
def analyze_symbol(tv_symbol: str):
    try:
        symbol = tv_symbol.split(":")[1]  # BINANCE:SOLUSDT -> SOLUSDT

        # HTF trend
        htf = fetch_klines_futures(symbol, HTF, limit=KLIMIT_HTF)
        htf = add_indicators(htf, ema_periods=(HTF_EMA_FAST, HTF_EMA_SLOW), atr_period=ATR_PERIOD)
        ema50_htf = htf.get(f"ema_{HTF_EMA_FAST}").iloc[-1]
        ema200_htf = htf.get(f"ema_{HTF_EMA_SLOW}").iloc[-1]

        if pd.isna(ema50_htf) or pd.isna(ema200_htf):
            log.warning("Insufficient HTF EMA data for %s", symbol)
            return None

        trend = "neutral"
        if ema50_htf > ema200_htf: trend = "bull"
        elif ema50_htf < ema200_htf: trend = "bear"
        if trend == "neutral": return None

        # LTF pullback
        ltf = fetch_klines_futures(symbol, LTF, limit=KLIMIT_LTF)
        ltf = add_indicators(ltf, ema_periods=(LTF_EMA_FAST, LTF_EMA_SLOW), atr_period=ATR_PERIOD)
        last, prev = ltf.iloc[-1], ltf.iloc[-2]
        entry_price, atr = float(last["close"]), float(last["atr"])
        if np.isnan(atr) or atr == 0: return None
        vol_ok = not np.isnan(last["vol_sma20"]) and last["volume"] >= last["vol_sma20"] * VOLUME_MULTIPLIER

        reasons, side, stop_price = [], None, None

        if trend == "bull":
            ema20, ema50 = last["ema_20"], last["ema_50"]
            near20 = (not pd.isna(ema20)) and (abs(entry_price - ema20)/ema20 <= PULLBACK_TOL)
            near50 = (not pd.isna(ema50)) and (abs(entry_price - ema50)/ema50 <= PULLBACK_TOL)
            candle_bullish = last["close"] > last["open"]
            prev_touched_20 = (not pd.isna(ema20)) and (prev["low"] <= ema20 <= prev["high"])
            prev_touched_50 = (not pd.isna(ema50)) and (prev["low"] <= ema50 <= prev["high"])
            macd_ok = (not pd.isna(last["macd"])) and (not pd.isna(last["macd_signal"])) and (last["macd"] > last["macd_signal"])
            rsi_ok = (not pd.isna(last["rsi"])) and (last["rsi"] < 70)

            if (near20 or near50) and candle_bullish and vol_ok and (prev_touched_20 or prev_touched_50) and macd_ok and rsi_ok:
                side = "LONG"
                stop_price = entry_price - ATR_K * atr
                reasons.append("HTF bull + LTF pullback + bullish candle + vol + MACD + RSI")

        elif trend == "bear":
            ema20, ema50 = last["ema_20"], last["ema_50"]
            near20 = (not pd.isna(ema20)) and (abs(entry_price - ema20)/ema20 <= PULLBACK_TOL)
            near50 = (not pd.isna(ema50)) and (abs(entry_price - ema50)/ema50 <= PULLBACK_TOL)
            candle_bearish = last["close"] < last["open"]
            prev_touched_20 = (not pd.isna(ema20)) and (prev["high"] >= ema20 >= prev["low"])
            prev_touched_50 = (not pd.isna(ema50)) and (prev["high"] >= ema50 >= prev["low"])
            macd_ok = (not pd.isna(last["macd"])) and (not pd.isna(last["macd_signal"])) and (last["macd"] < last["macd_signal"])
            rsi_ok = (not pd.isna(last["rsi"])) and (last["rsi"] > 30)

            if (near20 or near50) and candle_bearish and vol_ok and (prev_touched_20 or prev_touched_50) and macd_ok and rsi_ok:
                side = "SHORT"
                stop_price = entry_price + ATR_K * atr
                reasons.append("HTF bear + LTF pullback + bearish candle + vol + MACD + RSI")

        if side is None: return None

        reason = "; ".join(reasons) if reasons else "Matched filters"
        suggested_risk_pct = 1.0
        rr = 2.0
        tp = entry_price + (entry_price - stop_price) * rr if side == "LONG" else entry_price - (stop_price - entry_price) * rr

        return {
            "symbol": symbol,
            "side": side,
            "entry": round(entry_price,6),
            "stop": round(stop_price,6),
            "tp": round(tp,6),
            "risk_pct": suggested_risk_pct,
            "reason": reason,
            "time": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        log.exception("Error analyzing %s: %s", tv_symbol, e)
        return None

# ----------------- Format Message -----------------
def format_signal_msg(sig):
    s = sig
    return (
        f"ALERT <b>SIGNAL</b>\n"
        f"<b>{html.escape(s['symbol'])}</b> — {html.escape(s['side'])}\n"
        f"Entry: <code>{html.escape(str(s['entry']))}</code>\n"
        f"Stop: <code>{html.escape(str(s['stop']))}</code>\n"
        f"TP (suggested): <code>{html.escape(str(s['tp']))}</code>\n"
        f"Risk pct: {html.escape(str(s['risk_pct']))}%\n"
        f"Reason: {html.escape(s['reason'])}\n"
        f"Time (UTC): {html.escape(s['time'])}\n"
        f"Note: HTF={HTF} EMA{HTF_EMA_FAST}/{HTF_EMA_SLOW} + MACD + RSI + LTF={LTF} pullback + volume."
    )

# ----------------- Webhook -----------------
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    tv_symbol = data.get("symbol")
    if not tv_symbol or tv_symbol not in WATCHLIST_TV:
        return jsonify({"status": "ignored"}), 200

    sig = analyze_symbol(tv_symbol)
    if sig:
        inserted = save_signal_to_db(sig["symbol"], sig["side"], sig["entry"], sig["stop"], sig["tp"], sig["risk_pct"], sig["reason"])
        if inserted:
            msg = format_signal_msg(sig)
            if send_telegram(msg):
                mark_signal_sent(sig["symbol"], sig["entry"], sig["stop"])
                return jsonify({"status": "signal_sent"}), 200
            else:
                return jsonify({"status": "telegram_failed"}), 500
        else:
            return jsonify({"status": "duplicate_signal"}), 200
    else:
        return jsonify({"status": "no_signal"}), 200

# ----------------- Start -----------------
if __name__ == "__main__":
    init_db()
    log.info("TradingView Binance Signal Bot started. Watchlist: %s", WATCHLIST_TV)
    app.run(host="0.0.0.0", port=5000)
