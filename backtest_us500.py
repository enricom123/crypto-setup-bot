import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

SYMBOL = "SPX"
LOOKBACK_DAYS = 180  # 6 mesi

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def get_klines(interval, limit=500):
    interval_map = {
        "15m": "15min", "1h": "1h", "4h": "4h",
        "1d": "1day", "1w": "1week"
    }
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval_map.get(interval, interval),
        "outputsize": limit,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            logger.error(f"Error {interval}: {data.get('message','unknown')}")
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "open_time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Fetch error {interval}: {e}")
        return None

def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_session(dt):
    hour_est = (dt.hour - 5) % 24  # UTC to EST approx
    if 9 <= hour_est < 11:
        return "Open"
    elif 11 <= hour_est < 14:
        return "Midday"
    elif 14 <= hour_est <= 16:
        return "Close"
    return "Other"

def get_day_of_week(dt):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[dt.weekday()]

def find_next_liquidity_level(entry, direction, pwh, pwl, pdh, pdl, current_price):
    levels = {
        "PWH": pwh, "PWL": pwl,
        "PDH": pdh, "PDL": pdl
    }
    candidates = []
    for name, val in levels.items():
        if val is None:
            continue
        if direction == "LONG" and val > entry:
            candidates.append((name, val))
        elif direction == "SHORT" and val < entry:
            candidates.append((name, val))

    if not candidates:
        return None, None

    if direction == "LONG":
        candidates.sort(key=lambda x: x[1])
    else:
        candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[0]

def simulate_trade(entry, sl, tp_fixed, tp_liq, direction, future_candles):
    sl_hit = False
    tp_fixed_hit = False
    tp_liq_hit = False
    max_adverse = 0
    candles_to_result = 0

    for i, row in future_candles.iterrows():
        candles_to_result += 1
        if direction == "LONG":
            adverse = entry - row["low"]
            if row["low"] <= sl:
                sl_hit = True
                break
            if tp_liq and row["high"] >= tp_liq:
                tp_liq_hit = True
                break
            if row["high"] >= tp_fixed:
                tp_fixed_hit = True
                break
        else:
            adverse = row["high"] - entry
            if row["high"] >= sl:
                sl_hit = True
                break
            if tp_liq and row["low"] <= tp_liq:
                tp_liq_hit = True
                break
            if row["low"] <= tp_fixed:
                tp_fixed_hit = True
                break
        max_adverse = max(max_adverse, adverse)

        if candles_to_result >= 96:  # max 4 giorni su 15m
            break

    if sl_hit:
        result = "LOSS"
    elif tp_liq_hit:
        result = "WIN_LIQ"
    elif tp_fixed_hit:
        result = "WIN_RR2"
    else:
        result = "OPEN"

    return result, round(max_adverse, 2), candles_to_result

def detect_setups(df_htf, df_15m, df_daily, df_weekly, timeframe_label):
    setups = []
    ema200 = get_ema(df_daily["close"], 200)
    ema50 = get_ema(df_daily["close"], 50)

    for i in range(5, len(df_htf) - 1):
        candle = df_htf.iloc[i]
        candle_time = candle["open_time"]
        body = abs(candle["close"] - candle["open"])
        rng = candle["high"] - candle["low"]
        if rng == 0:
            continue
        body_ratio = body / rng
        if body_ratio < 0.6:
            continue

        # Trova daily index più vicino
        daily_idx = df_daily["open_time"].searchsorted(candle_time) - 1
        if daily_idx < 10 or daily_idx >= len(df_daily):
            continue

        prev_daily = df_daily.iloc[daily_idx - 1]
        pdh = float(prev_daily["high"])
        pdl = float(prev_daily["low"])

        weekly_idx = df_weekly["open_time"].searchsorted(candle_time) - 1
        if weekly_idx < 1 or weekly_idx >= len(df_weekly):
            continue
        prev_weekly = df_weekly.iloc[weekly_idx - 1]
        pwh = float(prev_weekly["high"])
        pwl = float(prev_weekly["low"])

        current_close = candle["close"]
        ema200_val = ema200.iloc[daily_idx]
        ema50_val = ema50.iloc[daily_idx]
        price_vs_ema200 = "Above" if current_close > ema200_val else "Below"
        bias = "LONG" if df_daily["close"].iloc[daily_idx - 1] > ema50_val else "SHORT"

        direction = None
        broken_level = None
        broken_value = None

        levels = {"PWH": pwh, "PWL": pwl, "PDH": pdh, "PDL": pdl}

        if candle["close"] > candle["open"]:  # bullish
            for name, val in levels.items():
                if val and candle["close"] > val > candle["open"]:
                    direction = "LONG"
                    broken_level = name
                    broken_value = val
                    break
        else:  # bearish
            for name, val in levels.items():
                if val and candle["close"] < val < candle["open"]:
                    direction = "SHORT"
                    broken_level = name
                    broken_value = val
                    break

        if direction is None or direction != bias:
            continue

        # Cerca retest su 15m
        tolerance = broken_value * 0.002
        fifteen_after = df_15m[df_15m["open_time"] > candle_time].head(96)

        for j, row_15m in fifteen_after.iterrows():
            r_body = abs(row_15m["close"] - row_15m["open"])
            r_rng = row_15m["high"] - row_15m["low"]
            if r_rng == 0:
                continue
            r_body_ratio = r_body / r_rng
            if r_body_ratio < 0.5:
                continue

            retested = False
            rejected = False

            if direction == "LONG":
                retested = row_15m["low"] <= broken_value + tolerance
                rejected = row_15m["close"] > broken_value
            else:
                retested = row_15m["high"] >= broken_value - tolerance
                rejected = row_15m["close"] < broken_value

            if not (retested and rejected):
                continue

            entry = row_15m["close"]
            if direction == "LONG":
                sl = row_15m["low"] * 0.9995
                tp_fixed = entry + (entry - sl) * 2
            else:
                sl = row_15m["high"] * 1.0005
                tp_fixed = entry - (sl - entry) * 2

            risk = abs(entry - sl)
            if risk == 0:
                continue

            liq_name, liq_val = find_next_liquidity_level(
                entry, direction, pwh, pwl, pdh, pdl, entry
            )
            tp_liq = liq_val
            rr_liq = round(abs(liq_val - entry) / risk, 2) if liq_val else None

            if abs(tp_fixed - entry) / risk < 2.0:
                continue

            # FVG check
            fvg = False
            if j >= 2:
                c1 = df_15m.iloc[j - 2]
                c3 = row_15m
                if direction == "LONG":
                    fvg = c3["low"] > c1["high"]
                else:
                    fvg = c3["high"] < c1["low"]

            signal_time = row_15m["open_time"]
            session = get_session(signal_time)
            day_of_week = get_day_of_week(signal_time)

            # Simula esito
            future = df_15m[df_15m["open_time"] > signal_time].head(96)
            result, max_adverse, candles_count = simulate_trade(
                entry, sl, tp_fixed, tp_liq, direction, future
            )

            setups.append({
                "Date": signal_time.strftime("%Y-%m-%d"),
                "Time CET": (signal_time + timedelta(hours=2)).strftime("%H:%M"),
                "Day": day_of_week,
                "TF Context": timeframe_label,
                "Level Broken": broken_level,
                "Direction": direction,
                "FVG": "Yes" if fvg else "No",
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "TP RR2": round(tp_fixed, 2),
                "TP Liquidity": round(tp_liq, 2) if tp_liq else "N/A",
                "RR Fixed": 2.0,
                "RR Liquidity": rr_liq if rr_liq else "N/A",
                "Liq Level Target": f"{liq_name} @ {round(liq_val, 2)}" if liq_name else "N/A",
                "Price vs EMA200": price_vs_ema200,
                "Daily Bias": bias,
                "Session": session,
                "Result": result,
                "Max Adverse Excursion": max_adverse,
                "Candles to Result": candles_count,
            })
            break  # un solo setup per candela HTF

    return setups

def run_backtest():
    send_telegram("⏳ <b>Backtest US500 avviato</b> — elaborazione 6 mesi di dati...")
    logger.info("Scarico dati storici...")

    df_daily = get_klines("1d", limit=500)
    time.sleep(8)
    df_weekly = get_klines("1w", limit=100)
    time.sleep(8)
    df_1h = get_klines("1h", limit=500)
    time.sleep(8)
    df_4h = get_klines("4h", limit=500)
    time.sleep(8)
    df_15m = get_klines("15m", limit=500)
    time.sleep(8)

    if any(df is None for df in [df_daily, df_weekly, df_1h, df_4h, df_15m]):
        send_telegram("❌ Errore nel download dei dati storici.")
        return

    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    df_1h = df_1h[df_1h["open_time"] > cutoff].reset_index(drop=True)
    df_4h = df_4h[df_4h["open_time"] > cutoff].reset_index(drop=True)
    df_15m = df_15m[df_15m["open_time"] > cutoff].reset_index(drop=True)

    logger.info(f"Dati: 1H={len(df_1h)}, 4H={len(df_4h)}, 15m={len(df_15m)}")

    setups_1h = detect_setups(df_1h, df_15m, df_daily, df_weekly, "1H")
    setups_4h = detect_setups(df_4h, df_15m, df_daily, df_weekly, "4H")

    all_setups = setups_1h + setups_4h
    all_setups.sort(key=lambda x: x["Date"])

    if not all_setups:
        send_telegram("📊 <b>Backtest completato</b> — Nessun setup A+ trovato nel periodo.")
        return

    df_results = pd.DataFrame(all_setups)

    # Statistiche
    total = len(df_results)
    wins = len(df_results[df_results["Result"].isin(["WIN_RR2", "WIN_LIQ"])])
    losses = len(df_results[df_results["Result"] == "LOSS"])
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    wins_liq = len(df_results[df_results["Result"] == "WIN_LIQ"])
    wins_rr2 = len(df_results[df_results["Result"] == "WIN_RR2"])

    aligned = df_results[
        ((df_results["Direction"] == "LONG") & (df_results["Price vs EMA200"] == "Above")) |
        ((df_results["Direction"] == "SHORT") & (df_results["Price vs EMA200"] == "Below"))
    ]
    wr_aligned = round(len(aligned[aligned["Result"].isin(["WIN_RR2", "WIN_LIQ"])]) / len(aligned) * 100, 1) if len(aligned) > 0 else 0

    summary = (
        f"📊 <b>Backtest US500 — Ultimi 6 mesi</b>\n"
        f"─────────────────\n"
        f"Setup A+ totali: <b>{total}</b>\n"
        f"Win rate: <b>{win_rate}%</b>\n"
        f"WIN (RR2): {wins_rr2} | WIN (Liq): {wins_liq} | LOSS: {losses}\n"
        f"─────────────────\n"
        f"🎯 WR con EMA200 allineata: <b>{wr_aligned}%</b>\n"
        f"─────────────────\n"
        f"Per TF:\n"
    )

    for tf in ["1H", "4H"]:
        tf_df = df_results[df_results["TF Context"] == tf]
        if len(tf_df) > 0:
            tf_wr = round(len(tf_df[tf_df["Result"].isin(["WIN_RR2", "WIN_LIQ"])]) / len(tf_df) * 100, 1)
            summary += f"  {tf}: {len(tf_df)} setup, WR {tf_wr}%\n"

    summary += "\nPer giorno:\n"
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        day_df = df_results[df_results["Day"] == day]
        if len(day_df) > 0:
            day_wr = round(len(day_df[day_df["Result"].isin(["WIN_RR2", "WIN_LIQ"])]) / len(day_df) * 100, 1)
            summary += f"  {day[:3]}: {len(day_df)} setup, WR {day_wr}%\n"

    summary += "\nPer sessione:\n"
    for sess in ["Open", "Midday", "Close"]:
        s_df = df_results[df_results["Session"] == sess]
        if len(s_df) > 0:
            s_wr = round(len(s_df[s_df["Result"].isin(["WIN_RR2", "WIN_LIQ"])]) / len(s_df) * 100, 1)
            summary += f"  {sess}: {len(s_df)} setup, WR {s_wr}%\n"

    send_telegram(summary)

    # Salva CSV
    csv_path = "/tmp/backtest_us500.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"CSV salvato: {csv_path}")
    logger.info(f"Backtest completato: {total} setup, WR {win_rate}%")

if __name__ == "__main__":
    run_backtest()
