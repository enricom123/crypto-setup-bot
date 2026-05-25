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
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")

SYMBOLS = [
    {"symbol": "BTCUSDT", "twelve": "BTC/USD", "label": "BTC"},
    {"symbol": "ETHUSDT", "twelve": "ETH/USD", "label": "ETH"},
]

alerted = set()

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def get_klines(twelve_symbol, interval, limit=50):
    interval_map = {
        "15m": "15min", "1h": "1h", "4h": "4h",
        "1d": "1day", "1w": "1week"
    }
    twelve_interval = interval_map.get(interval, interval)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": twelve_symbol,
        "interval": twelve_interval,
        "outputsize": limit,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            logger.error(f"Twelve Data error {twelve_symbol} {interval}: {data.get('message','unknown')}")
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "open_time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Twelve Data fetch error {twelve_symbol} {interval}: {e}")
        return None

def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_daily_bias(twelve_symbol):
    df = get_klines(twelve_symbol, "1d", limit=60)
    time.sleep(8)
    if df is None or len(df) < 10:
        return None
    ema50 = get_ema(df["close"], 50)
    last_close = df["close"].iloc[-2]
    if last_close > ema50.iloc[-2]:
        return "LONG"
    elif last_close < ema50.iloc[-2]:
        return "SHORT"
    return None

def get_weekly_levels(twelve_symbol):
    df = get_klines(twelve_symbol, "1w", limit=10)
    time.sleep(8)
    if df is None or len(df) < 2:
        return None, None
    prev = df.iloc[-2]
    return float(prev["high"]), float(prev["low"])

def get_daily_levels(twelve_symbol):
    df = get_klines(twelve_symbol, "1d", limit=10)
    time.sleep(8)
    if df is None or len(df) < 2:
        return None, None
    prev = df.iloc[-2]
    return float(prev["high"]), float(prev["low"])

def build_market_context(twelve_symbol, bias, pwh, pwl, pdh, pdl):
    df_4h = get_klines(twelve_symbol, "4h", limit=50)
    time.sleep(8)
    df_1h = get_klines(twelve_symbol, "1h", limit=50)
    time.sleep(8)
    df_15m = get_klines(twelve_symbol, "15m", limit=50)
    time.sleep(8)

    if df_4h is None or df_1h is None or df_15m is None:
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
                f"O:{row['open']:.2f} H:{row['high']:.2f} "
                f"L:{row['low']:.2f} C:{row['close']:.2f} | Body:{body_pct}%"
            )
        return f"\n[{label}]\n" + "\n".join(rows)

    current_price = df_15m["close"].iloc[-1]
    context = f"""SYMBOL: {twelve_symbol}
CURRENT PRICE: {current_price:.2f}
DAILY BIAS: {bias}
KEY LEVELS:
  PWH: {pwh:.2f} | PWL: {pwl:.2f}
  PDH: {pdh:.2f} | PDL: {pdl:.2f}
{candles_to_text(df_4h, '4H', 8)}
{candles_to_text(df_1h, '1H', 10)}
{candles_to_text(df_15m, '15m', 12)}"""
    return context, current_price

def ask_claude(symbol_label, context_text, bias):
    prompt = f"""You are an expert price action trader. Analyze this market data for {symbol_label} and determine if there is an A+ setup RIGHT NOW.

STRATEGY RULES:
- Bias: {bias} only — no trades against the trend
- Step 1: Find a displacement candle on 4H or 1H breaking PWH/PWL/PDH/PDL with body > 60% of range
- Step 2: Price must retest that broken level (now S/R flip)
- Step 3: On 15m, find a rejection candle at the retest (enters level, closes back on correct side, body > 50%)
- Minimum RR: 2.0 — if RR < 2.0, setup_found must be false
- SL: beyond rejection candle wick
- TP: next significant HTF level
- FVG confluence: bonus if a Fair Value Gap aligns with the retest zone

Be STRICT. Only A+ if ALL conditions are met perfectly. If any doubt, return setup_found: false.

MARKET DATA:
{context_text}

RESPOND ONLY WITH THIS JSON, no other text:
{{"setup_found": false, "grade": "none", "direction": "none", "entry": null, "sl": null, "tp": null, "rr": null, "broken_level": null, "fvg_confluence": false, "reasoning": "no setup"}}

OR if A+ found:
{{"setup_found": true, "grade": "A+", "direction": "LONG or SHORT", "entry": 00000.00, "sl": 00000.00, "tp": 00000.00, "rr": 0.0, "broken_level": "PWH or PWL or PDH or PDL", "fvg_confluence": true or false, "reasoning": "max 2 sentences"}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        if "content" not in data:
            logger.error(f"Claude error: {data}")
            return None
        text = data["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None

def format_alert(label, result, current_price):
    emoji = "🟢" if result["direction"] == "LONG" else "🔴"
    fvg_tag = "✅ FVG confluente" if result.get("fvg_confluence") else "➖ No FVG"
    return (
        f"{emoji} <b>{label} — {result['direction']} [A+]</b>\n"
        f"💰 Prezzo: {current_price:.2f}\n"
        f"🔑 Livello: {result.get('broken_level', 'N/A')}\n"
        f"─────────────────\n"
        f"📥 Entry: <b>{result['entry']}</b>\n"
        f"🛑 SL:    <b>{result['sl']}</b>\n"
        f"🎯 TP:    <b>{result['tp']}</b>\n"
        f"📐 RR:    <b>1:{result['rr']}</b>\n"
        f"─────────────────\n"
        f"{fvg_tag}\n"
        f"🧠 {result.get('reasoning', '')}\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
    )

def scan_symbol(sym):
    twelve_symbol = sym["twelve"]
    label = sym["label"]

    bias = get_daily_bias(twelve_symbol)
    if bias is None:
        logger.info(f"{label}: bias non determinabile")
        return

    pwh, pwl = get_weekly_levels(twelve_symbol)
    pdh, pdl = get_daily_levels(twelve_symbol)

    result_ctx = build_market_context(twelve_symbol, bias, pwh, pwl, pdh, pdl)
    if result_ctx is None:
        logger.info(f"{label}: dati insufficienti")
        return

    context_text, current_price = result_ctx
    claude_result = ask_claude(label, context_text, bias)
    if claude_result is None:
        return

    logger.info(f"{label}: grade={claude_result.get('grade')} setup={claude_result.get('setup_found')} rr={claude_result.get('rr')}")

    if not claude_result.get("setup_found"):
        return
    if not claude_result.get("rr") or float(claude_result["rr"]) < 2.0:
        return

    alert_key = f"{label}_{claude_result.get('broken_level')}_{claude_result.get('entry')}"
    if alert_key in alerted:
        return

    alerted.add(alert_key)
    msg = format_alert(label, claude_result, current_price)
    send_telegram(msg)
    logger.info(f"Alert A+ inviato: {alert_key}")

def run_scan():
    logger.info(f"Scan avviato: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    for sym in SYMBOLS:
        scan_symbol(sym)
        time.sleep(10)

def main():
    send_telegram("🤖 <b>Bot AI avviato</b> — Claude analizza BTC/ETH ogni 15 minuti. Solo setup A+ con RR ≥ 2.")
    run_scan()
    schedule.every(15).minutes.do(run_scan)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
