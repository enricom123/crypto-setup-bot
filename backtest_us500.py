import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")

SYMBOL = "SPX"
TARGET_SETUPS = 200

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def send_telegram_file(filepath, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(filepath, "rb") as f:
            requests.post(url, data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption
            }, files={"document": f}, timeout=30)
    except Exception as e:
        logger.error(f"Telegram file error: {e}")

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

def get_klines_paginated(interval, pages=3):
    interval_map = {
        "15m": "15min", "1h": "1h", "4h": "4h",
        "1d": "1day", "1w": "1week"
    }
    all_dfs = []
    end_date = None

    for page in range(pages):
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": SYMBOL,
            "interval": interval_map.get(interval, interval),
            "outputsize": 5000,
            "apikey": TWELVE_DATA_KEY,
            "format": "JSON",
            "order": "DESC"
        }
        if end_date:
            params["end_date"] = end_date

        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if "values" not in data:
                logger.error(f"Pagination error page {page}: {data.get('message','unknown')}")
                break
            df = pd.DataFrame(data["values"])
            df = df.rename(columns={"datetime": "open_time"})
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"])
            all_dfs.append(df)
            logger.info(f"Pagina {page+1}: {len(df)} candele, da {df['open_time'].min()} a {df['open_time'].max()}")

            # Imposta end_date per pagina successiva
            oldest = df["open_time"].min()
            end_date = (oldest - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
            time.sleep(3)

        except Exception as e:
            logger.error(f"Pagination fetch error page {page}: {e}")
            break

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"])
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined

def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_session(dt):
    hour_est = (dt.hour - 4) % 24
    if 9 <= hour_est < 11:
        return "Open"
    elif 11 <= hour_est < 14:
        return "Midday"
    elif 14 <= hour_est <= 16:
        return "Close"
    return "Other"

def get_day_of_week(dt):
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return days[dt.weekday()]

def find_next_liquidity_level(entry, direction, levels_dict):
    candidates = []
    for name, val in levels_dict.items():
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
    max_adverse = 0
    candles_count = 0
    for _, row in future_candles.iterrows():
        candles_count += 1
        if direction == "LONG":
            adverse = entry - row["low"]
            if row["low"] <= sl:
                return "LOSS", round(max_adverse, 2), candles_count
            if tp_liq and row["high"] >= tp_liq:
                return "WIN_LIQ", round(max_adverse, 2), candles_count
            if row["high"] >= tp_fixed:
                return "WIN_RR2", round(max_adverse, 2), candles_count
        else:
            adverse = row["high"] - entry
            if row["high"] >= sl:
                return "LOSS", round(max_adverse, 2), candles_count
            if tp_liq and row["low"] <= tp_liq:
                return "WIN_LIQ", round(max_adverse, 2), candles_count
            if row["low"] <= tp_fixed:
                return "WIN_RR2", round(max_adverse, 2), candles_count
        max_adverse = max(max_adverse, max(adverse, 0))
        if candles_count >= 192:
            break
    return "OPEN", round(max_adverse, 2), candles_count

def detect_setups(df_htf, df_15m, df_daily, df_weekly, tf_label, target=200):
    setups = []
    ema200 = get_ema(df_daily["close"], 200)
    ema50 = get_ema(df_daily["close"], 50)

    for i in range(5, len(df_htf) - 1):
        if len(setups) >= target:
            break

        candle = df_htf.iloc[i]
        candle_time = candle["open_time"]

        body = abs(candle["close"] - candle["open"])
        rng = candle["high"] - candle["low"]
        if rng == 0 or body / rng < 0.6:
            continue

        daily_mask = df_daily["open_time"] <= candle_time
        if daily_mask.sum() < 10:
            continue
        daily_idx = daily_mask.values.nonzero()[0][-1]
        prev_daily = df_daily.iloc[daily_idx - 1]
        pdh = float(prev_daily["high"])
        pdl = float(prev_daily["low"])

        weekly_mask = df_weekly["open_time"] <= candle_time
        if weekly_mask.sum() < 2:
            continue
        weekly_idx = weekly_mask.values.nonzero()[0][-1]
        prev_weekly = df_weekly.iloc[weekly_idx - 1]
        pwh = float(prev_weekly["high"])
        pwl = float(prev_weekly["low"])

        ema200_val = ema200.iloc[daily_idx]
        ema50_val = ema50.iloc[daily_idx]
        price_vs_ema200 = "Above" if candle["close"] > ema200_val else "Below"
        bias = "LONG" if df_daily["close"].iloc[daily_idx - 1] > ema50_val else "SHORT"

        levels = {"PWH": pwh, "PWL": pwl, "PDH": pdh, "PDL": pdl}
        direction = None
        broken_level = None
        broken_value = None

        if candle["close"] > candle["open"]:
            for name, val in levels.items():
                if val and candle["close"] > val > candle["open"]:
                    direction = "LONG"
                    broken_level = name
                    broken_value = val
                    break
        else:
            for name, val in levels.items():
                if val and candle["close"] < val < candle["open"]:
                    direction = "SHORT"
                    broken_level = name
                    broken_value = val
                    break

        if direction is None or direction != bias:
            continue

        tolerance = broken_value * 0.002
        fifteen_after = df_15m[df_15m["open_time"] > candle_time].head(192)

        for _, row_15m in fifteen_after.iterrows():
            r_body = abs(row_15m["close"] - row_15m["open"])
            r_rng = row_15m["high"] - row_15m["low"]
            if r_rng == 0 or r_body / r_rng < 0.5:
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
            if risk == 0 or abs(tp_fixed - entry) / risk < 2.0:
                continue

            liq_name, liq_val = find_next_liquidity_level(entry, direction, levels)
            tp_liq = liq_val
            rr_liq = round(abs(liq_val - entry) / risk, 2) if liq_val else None

            fvg = False
            idx_15m = df_15m.index[df_15m["open_time"] == row_15m["open_time"]]
            if len(idx_15m) > 0:
                idx = idx_15m[0]
                if idx >= 2:
                    c1 = df_15m.iloc[idx - 2]
                    if direction == "LONG":
                        fvg = row_15m["low"] > c1["high"]
                    else:
                        fvg = row_15m["high"] < c1["low"]

            signal_time = row_15m["open_time"]
            future = df_15m[df_15m["open_time"] > signal_time].head(192)
            result, max_adverse, candles_count = simulate_trade(
                entry, sl, tp_fixed, tp_liq, direction, future
            )

            setups.append({
                "Date": signal_time.strftime("%Y-%m-%d"),
                "Time CET": (signal_time + timedelta(hours=2)).strftime("%H:%M"),
                "Day": get_day_of_week(signal_time),
                "TF Context": tf_label,
                "Level Broken": broken_level,
                "Direction": direction,
                "FVG": "Yes" if fvg else "No",
                "Signal Candle Open Time": signal_time.strftime("%Y-%m-%d %H:%M"),
                "Signal Candle Open": round(row_15m["open"], 2),
                "Signal Candle High": round(row_15m["high"], 2),
                "Signal Candle Low": round(row_15m["low"], 2),
                "Signal Candle Close": round(row_15m["close"], 2),
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "TP RR2": round(tp_fixed, 2),
                "TP Liquidity": round(tp_liq, 2) if tp_liq else "N/A",
                "RR Fixed": 2.0,
                "RR Liquidity": rr_liq if rr_liq else "N/A",
                "Liq Level Target": f"{liq_name} @ {round(liq_val,2)}" if liq_name else "N/A",
                "Price vs EMA200": price_vs_ema200,
                "Daily Bias": bias,
                "Session": get_session(signal_time),
                "Result": result,
                "Max Adverse Excursion": max_adverse,
                "Candles to Result": candles_count,
            })
            break

    return setups

def run_backtest():
    send_telegram("⏳ <b>Backtest US500 avviato</b> — scarico dati Twelve Data...")
    logger.info("Download dati storici...")

    df_daily = get_klines("1d", limit=500)
    df_weekly = get_klines("1w", limit=200)
    df_1h = get_klines_paginated("1h", pages=2)
    df_15m = get_klines_paginated("15m", pages=3)

    if any(df is None for df in [df_daily, df_weekly, df_1h, df_15m]):
        send_telegram("❌ Errore nel download dei dati.")
        return

    logger.info(f"Dati: daily={len(df_daily)}, weekly={len(df_weekly)}, 1h={len(df_1h)}, 15m={len(df_15m)}")
    logger.info(f"1H: da {df_1h['open_time'].iloc[0]} a {df_1h['open_time'].iloc[-1]}")
    logger.info(f"15m: da {df_15m['open_time'].iloc[0]} a {df_15m['open_time'].iloc[-1]}")

    setups_1h = detect_setups(df_1h, df_15m, df_daily, df_weekly, "1H", target=150)
    logger.info(f"Setup 1H: {len(setups_1h)}")

    setups_4h = detect_setups(df_daily, df_15m, df_daily, df_weekly, "4H", target=50)
    logger.info(f"Setup 4H: {len(setups_4h)}")

    all_setups = setups_1h + setups_4h
    all_setups.sort(key=lambda x: x["Date"])

    if not all_setups:
        send_telegram("📊 Nessun setup trovato.")
        return

    df_results = pd.DataFrame(all_setups)
    total = len(df_results)
    wins = len(df_results[df_results["Result"].isin(["WIN_RR2","WIN_LIQ"])])
    losses = len(df_results[df_results["Result"] == "LOSS"])
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    wins_liq = len(df_results[df_results["Result"] == "WIN_LIQ"])
    wins_rr2 = len(df_results[df_results["Result"] == "WIN_RR2"])

    aligned = df_results[
        ((df_results["Direction"] == "LONG") & (df_results["Price vs EMA200"] == "Above")) |
        ((df_results["Direction"] == "SHORT") & (df_results["Price vs EMA200"] == "Below"))
    ]
    wr_aligned = round(len(aligned[aligned["Result"].isin(["WIN_RR2","WIN_LIQ"])]) / len(aligned) * 100, 1) if len(aligned) > 0 else 0

    date_from = df_results["Date"].min()
    date_to = df_results["Date"].max()

    summary = (
        f"📊 <b>Backtest US500 completato</b>\n"
        f"📅 {date_from} → {date_to}\n"
        f"─────────────────\n"
        f"Setup totali: <b>{total}</b>\n"
        f"Win rate: <b>{win_rate}%</b>\n"
        f"WIN RR2: {wins_rr2} | WIN Liq: {wins_liq} | LOSS: {losses}\n"
        f"─────────────────\n"
        f"🎯 WR EMA200 allineata: <b>{wr_aligned}%</b>\n"
        f"─────────────────\n"
        f"Per TF:\n"
    )
    for tf in ["1H","4H"]:
        tf_df = df_results[df_results["TF Context"] == tf]
        if len(tf_df) > 0:
            tf_wr = round(len(tf_df[tf_df["Result"].isin(["WIN_RR2","WIN_LIQ"])]) / len(tf_df) * 100, 1)
            summary += f"  {tf}: {len(tf_df)} setup, WR {tf_wr}%\n"

    summary += "\nPer giorno:\n"
    for day in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        day_df = df_results[df_results["Day"] == day]
        if len(day_df) > 0:
            day_wr = round(len(day_df[day_df["Result"].isin(["WIN_RR2","WIN_LIQ"])]) / len(day_df) * 100, 1)
            summary += f"  {day[:3]}: {len(day_df)} setup, WR {day_wr}%\n"

    summary += "\nPer sessione:\n"
    for sess in ["Open","Midday","Close"]:
        s_df = df_results[df_results["Session"] == sess]
        if len(s_df) > 0:
            s_wr = round(len(s_df[s_df["Result"].isin(["WIN_RR2","WIN_LIQ"])]) / len(s_df) * 100, 1)
            summary += f"  {sess}: {len(s_df)} setup, WR {s_wr}%\n"

    summary += "\nPer livello:\n"
    for lvl in ["PWH","PWL","PDH","PDL"]:
        l_df = df_results[df_results["Level Broken"] == lvl]
        if len(l_df) > 0:
            l_wr = round(len(l_df[l_df["Result"].isin(["WIN_RR2","WIN_LIQ"])]) / len(l_df) * 100, 1)
            summary += f"  {lvl}: {len(l_df)} setup, WR {l_wr}%\n"

    send_telegram(summary)

    csv_path = "/tmp/backtest_us500.csv"
    df_results.to_csv(csv_path, index=False)
    send_telegram_file(csv_path, f"📎 Backtest US500 — {total} setup")
    logger.info("Completato.")

if __name__ == "__main__":
    run_backtest()

