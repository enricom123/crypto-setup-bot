import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import schedule
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

alerted = set()

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def get_klines(symbol, interval, limit=100):
    interval_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D", "1w": "W"
    }
    bybit_interval = interval_map.get(interval, interval)
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        logger.info(f"Bybit {symbol} {interval}: {r.text[:200]}")
        data = r.json()
        if data.get("retCode") != 0:
            logger.error(f"Bybit error: {data}")
            return None
        rows = data["result"]["list"]
        df = pd.DataFrame(rows, columns=[
            "open_time","open","high","low","close","volume","turnover"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Bybit error {symbol} {interval}: {e}")
        return None

def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_daily_bias(symbol):
    df = get_klines(symbol, "1d", limit=200)
    if df is None or len(df) < 10:
        logger.info(f"{symbol}: bias non determinabile")
        return None
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    last_close = df["close"].iloc[-2]
    if last_close > ema50.iloc[-2]:
        return "LONG"
    elif last_close < ema50.iloc[-2]:
        return "SHORT"
    return None

def get_weekly_levels(symbol):
    df = get_klines(symbol, "1w", limit=10)
    if df is None or len(df) < 2:
        return None, None
    prev = df.iloc[-2]
    return float(prev["high"]), float(prev["low"])

def get_daily_levels(symbol):
    df = get_klines(symbol, "1d", limit=10)
    if df is None or len(df) < 2:
        return None, None
    prev = df.iloc[-2]
    return float(prev["high"]), float(prev["low"])

def build_market_context(symbol, bias, pwh, pwl, pdh, pdl):
    df_1h = get_klines(symbol, "1h", limit=50)
    df_4h = get_klines(symbol, "4h", limit=50)
    df_15m = get_klines(symbol, "15m", limit=50)

    if df_1h is None or df_4h is None or df_15m is None:
        return None

    def candles_to_text(df, label, n=10):
        rows = []
        for _, row in df.tail(n).iterrows():
            body = abs(row["close"] - row["open"])
            rng = row["high"] - row["low"]
            body_pct = round(body / rng * 100, 1) if rng > 0 else 0
            direction = "BULL" if row["close"] > row["open"] else "BEAR"
            rows.append(
                f"{row['open_time'].strftime('%m-%d %H:%M')} | {direction} | "
                f"O:{row['open']:.1f} H:{row['high']:.1f} L:{row['low']:.1f} C:{row['close']:.1f} | "
                f"Body:{body_pct}%"
            )
        return f"\n[{label}]\n" + "\n".join(rows)

    current_price = df_15m["close"].iloc[-1]

    context = f"""SYMBOL: {symbol}
CURRENT PRICE: {current_price:.2f}
DAILY BIAS: {bias}
KEY LEVELS:
  PWH: {pwh:.2f} | PWL: {pwl:.2f}
  PDH: {pdh:.2f} | PDL: {pdl:.2f}

RECENT CANDLES:
{candles_to_text(df_4h, '4H', 8)}
{candles_to_text(df_1h, '1H', 10)}
{candles_to_text(df_15m, '15m', 12)}
"""
    return context, current_price

def ask_claude(symbol, context_text, current_price, bias):
    prompt = f"""You are an expert price action trader. Analyze the following market data and determine if there is an A+ setup RIGHT NOW.

STRATEGY RULES:
- Daily bias determines direction: {bias} only
- Look for a strong displacement candle on 4H or 1H that broke a key level (PWH/PWL/PDH/PDL) with body > 60% of range
- After the breakout, price must retest the broken level (now acting as support/resistance)
- On 15m timeframe: look for a rejection candle at the retest zone (candle that enters the level and closes back above/below it, body > 50%)
- Minimum RR: 2.0
- SL: below/above the rejection candle wick
- TP: next significant level

MARKET DATA:
{context_text}

RESPOND ONLY IN THIS EXACT JSON FORMAT, nothing else:
{{
  "setup_found": true/false,
  "grade": "A+" or "A" or "B" or "none",
  "direction": "LONG" or "SHORT" or "none",
  "entry": price or null,
  "sl": price or null,
  "tp": price or null,
  "rr": number or null,
  "broken_level": "PWH/PWL/PDH/PDL" or null,
  "fvg_confluence": true/false,
  "reasoning": "brief explanation in English, max 2 sentences"
}}

Only report setup_found: true if grade is A+. Be strict. If in doubt, grade is NOT A+."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        text = data["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None

def format_alert(symbol, result, current_price):
    emoji = "🟢" if result["direction"] == "LONG" else "🔴"
    fvg_tag = "✅ FVG confluente" if result.get("fvg_confluence") else "➖ No FVG"
    msg = (
        f"{emoji} <b>{symbol} — {result['direction']} [A+]</b>\n"
        f"💰 Prezzo attuale: {current_price:.2f}\n"
        f"🔑 Livello rotto: {result.get('broken_level', 'N/A')}\n"
        f"─────────────────\n"
        f"📥 Entry:  <b>{result['entry']}</b>\n"
        f"🛑 SL:     <b>{result['sl']}</b>\n"
        f"🎯 TP:     <b>{result['tp']}</b>\n"
        f"📐 RR:     <b>1:{result['rr']}</b>\n"
        f"─────────────────\n"
        f"{fvg_tag}\n"
        f"🧠 {result.get('reasoning', '')}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
    )
    return msg

def run_scan():
    logger.info(f"Scan avviato: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    for symbol in SYMBOLS:
        bias = get_daily_bias(symbol)
        if bias is None:
            logger.info(f"{symbol}: bias non determinabile")
            continue

        pwh, pwl = get_weekly_levels(symbol)
        pdh, pdl = get_daily_levels(symbol)

        result_ctx = build_market_context(symbol, bias, pwh, pwl, pdh, pdl)
        if result_ctx is None:
            continue

        context_text, current_price = result_ctx

        claude_result = ask_claude(symbol, context_text, current_price, bias)
        if claude_result is None:
            continue

        logger.info(f"{symbol}: grade={claude_result.get('grade')} setup={claude_result.get('setup_found')}")

        if not claude_result.get("setup_found"):
            continue

        if claude_result.get("rr") and float(claude_result["rr"]) < 2.0:
            continue

        alert_key = f"{symbol}_{claude_result.get('broken_level')}_{claude_result.get('entry')}"
        if alert_key in alerted:
            continue

        alerted.add(alert_key)
        msg = format_alert(symbol, claude_result, current_price)
        send_telegram(msg)
        logger.info(f"Alert A+ inviato: {alert_key}")

def main():
    send_telegram("🤖 <b>Bot AI avviato</b> — Claude analizza BTC/ETH ogni 15 minuti. Solo setup A+ con RR ≥ 2.")
    run_scan()
    schedule.every(15).minutes.do(run_scan)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
