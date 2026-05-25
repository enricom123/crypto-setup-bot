import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import time
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SYMBOL = "SPY"
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

def download_data(period="5y"):
    logger.info("Download dati Yahoo Finance...")
    spy = yf.Ticker(SYMBOL)

    df_15m = spy.history(period="60d", interval="15m")
    df_1h = spy.history(period="2y", interval="1h")
    df_4h = spy.history(period="5y", interval="1d")  # useremo daily come proxy 4H
    df_daily = spy.history(period="5y", interval="1d")
    df_weekly = spy.history(period="10y", interval="1wk")

    for df in [df_15m, df_1h, df_4h, df_daily, df_weekly]:
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close"
        }, inplace=True)
        df["open_time"] = df.index

    return df_15m, df_1h, df_daily, df_weekly

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
        if candles_count >= 192:  # max 2 giorni su 15m
            break
    return "OPEN", round(max_adverse, 2), candles_count

def detect_setups(df_htf, df_15m, df_daily, df_weekly, tf_label, target=200):
    setups = []
    ema200 = get_ema(df_daily["close"], 200)
    ema50 = get_ema(df_daily["close"], 50)
    df_daily_reset = df_daily.reset_index(drop=True)
    df_weekly_reset = df_weekly.reset_index(drop=True)
    df_htf_reset = df_htf.reset_index(drop=True)
    df_15m_reset = df_15m.reset_index(drop=True)

    for i in range(5, len(df_htf_reset) - 1):
        if len(setups) >= target:
            break

        candle = df_htf_reset.iloc[i]
        candle_time = candle["open_time"]

        body = abs(candle["close"] - candle["open"])
        rng = candle["high"] - candle["low"]
        if rng == 0:
            continue
        if body / rng < 0.6:
            continue

        # Daily context
        daily_mask = df_daily_reset["open_time"] <= candle_time
        if daily_mask.sum() < 10:
            continue
        daily_idx = daily_mask.values.nonzero()[0][-1]
        prev_daily = df_daily_reset.iloc[daily_idx - 1]
        pdh = float(prev_daily["high"])
        pdl = float(prev_daily["low"])

        # Weekly context
        weekly_mask = df_weekly_reset["open_time"] <= candle_time
        if weekly_mask.sum() < 2:
            continue
        weekly_idx = weekly_mask.values.nonzero()[0][-1]
        prev_weekly = df_weekly_reset.iloc[weekly_idx - 1]
        pwh = float(prev_weekly["high"])
        pwl = float(prev_weekly["low"])

        ema200_val = ema200.iloc[daily_idx]
        ema50_val = ema50.iloc[daily_idx]
        price_vs_ema200 = "Above" if candle["close"] > ema200_val else "Below"
        bias = "LONG" if df_daily_reset["close"].iloc[daily_idx - 1] > ema50_val else "SHORT"

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
        fifteen_after = df_15m_reset[df_15m_reset["open_time"] > candle_time].head(192)

        for _, row_15m in fifteen_after.iterrows():
            r_body = abs(row_15m["close"] - row_15m["open"])
            r_rng = row_15m["high"] - row_15m["low"]
            if r_rng == 0:
                continue
            if r_body / r_rng < 0.5:
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

            # FVG check su 15m
            fvg = False
            fifteen_idx = df_15m_reset.index[df_15m_reset["open_time"] == row_15m["open_time"]]
            if len(fifteen_idx) > 0:
                idx = fifteen_idx[0]
                if idx >= 2:
                    c1 = df_15m_reset.iloc[idx - 2]
                    c3 = row_15m
                    if direction == "LONG":
                        fvg = c3["low"] > c1["high"]
                    else:
                        fvg = c3["high"] < c1["low"]

            signal_time = row_15m["open_time"]
            future = df_15m_reset[df_15m_reset["open_time"] > signal_time].head(192)
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
    send_telegram("⏳ <b>Backtest US500 avviato</b> — elaborazione fino a 200 setup su dati Yahoo Finance...")

    df_15m, df_1h, df_daily, df_weekly = download_data()
    
    logger.info(f"15m shape: {df_15m.shape if df_15m is not None else 'None'}")
    logger.info(f"1H shape: {df_1h.shape if df_1h is not None else 'None'}")
    logger.info(f"Daily shape: {df_daily.shape if df_daily is not None else 'None'}")
    logger.info(f"Weekly shape: {df_weekly.shape if df_weekly is not None else 'None'}")
    
    if df_15m is not None and len(df_15m) > 0:
        logger.info(f"15m primo: {df_15m['open_time'].iloc[0]} ultimo: {df_15m['open_time'].iloc[-1]}")
    if df_1h is not None and len(df_1h) > 0:
        logger.info(f"1H primo: {df_1h['open_time'].iloc[0]} ultimo: {df_1h['open_time'].iloc[-1]}")

    setups_1h = detect_setups(df_1h, df_15m, df_daily, df_weekly, "1H", target=150)
    logger.info(f"Setup 1H trovati: {len(setups_1h)}")

    setups_4h = detect_setups(df_daily, df_15m, df_daily, df_weekly, "4H", target=50)
    logger.info(f"Setup 4H trovati: {len(setups_4h)}")

    all_setups = setups_1h + setups_4h
    
    send_telegram(f"Debug: 15m={len(df_15m) if df_15m is not None else 0} righe, 1H={len(df_1h) if df_1h is not None else 0} righe, setup={len(all_setups)}")
    
    if not all_setups:
        send_telegram("📊 Nessun setup trovato. Controlla i parametri.")
        return

    df_results = pd.DataFrame(all_setups)
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
    wr_aligned = round(len(aligned[aligned["Result"].isin(["WIN_RR2","WIN_LIQ"])]) / len(aligned) * 100, 1) if len(aligned) > 0 else 0

    # Date range
    date_from = df_results["Date"].min()
    date_to = df_results["Date"].max()

    summary = (
        f"📊 <b>Backtest US500 completato</b>\n"
        f"📅 Periodo: {date_from} → {date_to}\n"
        f"─────────────────\n"
        f"Setup A+ totali: <b>{total}</b>\n"
        f"Win rate: <b>{win_rate}%</b>\n"
        f"WIN RR2: {wins_rr2} | WIN Liq: {wins_liq} | LOSS: {losses}\n"
        f"─────────────────\n"
        f"🎯 WR con EMA200 allineata: <b>{wr_aligned}%</b>\n"
        f"─────────────────\n"
        f"Per TF:\n"
    )

    for tf in ["1H", "4H"]:
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

    # Invia CSV su Telegram
    csv_path = "/tmp/backtest_us500.csv"
    df_results.to_csv(csv_path, index=False)
    send_telegram_file(csv_path, f"📎 Backtest US500 — {total} setup completi")
    logger.info("Backtest completato e CSV inviato.")

if __name__ == "__main__":
    run_backtest()
