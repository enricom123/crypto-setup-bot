import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import schedule
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def get_klines(symbol, interval, limit=100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df
    except Exception as e:
        logger.error(f"Binance error {symbol} {interval}: {e}")
        return None

def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_daily_bias(symbol):
    df = get_klines(symbol, "1d", limit=60)
    if df is None or len(df) < 52:
        return None
    ema200 = get_ema(df["close"], 200)
    # Con 60 candele non abbiamo EMA200 stabile, usiamo EMA50 daily come proxy
    ema50 = get_ema(df["close"], 50)
    last_close = df["close"].iloc[-2]  # candela chiusa, non quella corrente
    if last_close > ema50.iloc[-2]:
        return "LONG"
    elif last_close < ema50.iloc[-2]:
        return "SHORT"
    return None

def get_weekly_levels(symbol):
    df = get_klines(symbol, "1w", limit=10)
    if df is None or len(df) < 2:
        return None, None
    prev_week = df.iloc[-2]
    return float(prev_week["high"]), float(prev_week["low"])

def get_daily_levels(symbol):
    df = get_klines(symbol, "1d", limit=10)
    if df is None or len(df) < 2:
        return None, None
    prev_day = df.iloc[-2]
    return float(prev_day["high"]), float(prev_day["low"])

def has_fvg(df, index, direction):
    if index < 2:
        return False
    c1 = df.iloc[index - 2]
    c3 = df.iloc[index]
    if direction == "LONG":
        return c3["low"] > c1["high"]
    elif direction == "SHORT":
        return c3["high"] < c1["low"]
    return False

def check_breakout(symbol, tf_high, bias, pwh, pwl, pdh, pdl):
    df = get_klines(symbol, tf_high, limit=50)
    if df is None or len(df) < 10:
        return None

    levels = {
        "PWH": pwh, "PWL": pwl,
        "PDH": pdh, "PDL": pdl
    }

    for i in range(len(df) - 4, len(df) - 1):
        candle = df.iloc[i]
        body = abs(candle["close"] - candle["open"])
        candle_range = candle["high"] - candle["low"]
        if candle_range == 0:
            continue
        body_ratio = body / candle_range

        if body_ratio < 0.6:
            continue

        broken_level = None
        broken_value = None

        if bias == "LONG" and candle["close"] > candle["open"]:
            for name, val in levels.items():
                if val and candle["close"] > val > candle["open"]:
                    broken_level = name
                    broken_value = val
                    break

        elif bias == "SHORT" and candle["close"] < candle["open"]:
            for name, val in levels.items():
                if val and candle["close"] < val < candle["open"]:
                    broken_level = name
                    broken_value = val
                    break

        if broken_level is None:
            continue

        fvg = has_fvg(df, i, bias)

        return {
            "symbol": symbol,
            "bias": bias,
            "tf": tf_high,
            "broken_level": broken_level,
            "broken_value": broken_value,
            "breakout_candle_idx": i,
            "fvg": fvg,
            "df": df
        }

    return None

def check_retest_and_rejection(setup):
    symbol = setup["symbol"]
    bias = setup["bias"]
    tf_high = setup["tf"]
    broken_value = setup["broken_value"]
    fvg = setup["fvg"]

    tf_low = "15m" if tf_high == "1h" else "1h"
    df_low = get_klines(symbol, tf_low, limit=50)
    if df_low is None or len(df_low) < 5:
        return None

    tolerance = broken_value * 0.002  # 0.2% tolerance

    for i in range(len(df_low) - 5, len(df_low) - 1):
        candle = df_low.iloc[i]
        body = abs(candle["close"] - candle["open"])
        candle_range = candle["high"] - candle["low"]
        if candle_range == 0:
            continue
        body_ratio = body / candle_range
        if body_ratio < 0.5:
            continue

        retested = False
        rejected = False

        if bias == "LONG":
            retested = candle["low"] <= broken_value + tolerance
            rejected = candle["close"] > broken_value

        elif bias == "SHORT":
            retested = candle["high"] >= broken_value - tolerance
            rejected = candle["close"] < broken_value

        if not (retested and rejected):
            continue

        entry = candle["close"]

        if bias == "LONG":
            sl = candle["low"] * 0.999
            tp = entry + (entry - sl) * 2
        else:
            sl = candle["high"] * 1.001
            tp = entry - (sl - entry) * 2

        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        if rr < 2.0:
            continue

        return {
            "symbol": symbol,
            "bias": bias,
            "tf_context": tf_high,
            "tf_entry": tf_low,
            "broken_level": setup["broken_level"],
            "broken_value": round(broken_value, 2),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": round(rr, 2),
            "fvg": fvg
        }

    return None

def format_alert(result):
    emoji = "🟢" if result["bias"] == "LONG" else "🔴"
    fvg_tag = "✅ FVG confluente" if result["fvg"] else "➖ No FVG"
    msg = (
        f"{emoji} <b>{result['symbol']} — {result['bias']}</b>\n"
        f"📊 Contesto: {result['tf_context']} | Entry: {result['tf_entry']}\n"
        f"🔑 Livello rotto: {result['broken_level']} @ {result['broken_value']}\n"
        f"─────────────────\n"
        f"📥 Entry:  <b>{result['entry']}</b>\n"
        f"🛑 SL:     <b>{result['sl']}</b>\n"
        f"🎯 TP:     <b>{result['tp']}</b>\n"
        f"📐 RR:     <b>1:{result['rr']}</b>\n"
        f"─────────────────\n"
        f"{fvg_tag}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
    )
    return msg

alerted = set()

def run_scan():
    logger.info(f"Scan avviato: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    for symbol in SYMBOLS:
        bias = get_daily_bias(symbol)
        if bias is None:
            continue

        pwh, pwl = get_weekly_levels(symbol)
        pdh, pdl = get_daily_levels(symbol)

        for tf in ["1h", "4h"]:
            setup = check_breakout(symbol, tf, bias, pwh, pwl, pdh, pdl)
            if setup is None:
                continue

            result = check_retest_and_rejection(setup)
            if result is None:
                continue

            alert_key = f"{symbol}_{tf}_{result['broken_level']}_{result['entry']}"
            if alert_key in alerted:
                continue

            alerted.add(alert_key)
            msg = format_alert(result)
            send_telegram(msg)
            logger.info(f"Alert inviato: {alert_key}")

def main():
    send_telegram("🤖 <b>Bot avviato</b> — scansione BTC/ETH ogni 15 minuti.")
    run_scan()
    schedule.every(15).minutes.do(run_scan)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
