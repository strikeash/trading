
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# INDIAN STOCK SWING TRADER - PAPER TRADING ONLY
# No broker connection. No live orders.
# =========================================================

CAPITAL = 1000.0
MAX_POSITIONS = 3
MAX_ALLOCATION = 0.25
RISK_PER_TRADE = 0.01

# Illustrative costs per side. Replace with actual costs
# before evaluating real-world viability.
FEE_RATE = 0.0015
SLIPPAGE_RATE = 0.0010

HISTORY = "5y"
MIN_HISTORY = 100

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS",
    "LT.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "NTPC.NS",
    "POWERGRID.NS", "SUNPHARMA.NS", "ONGC.NS", "BEL.NS",
    "TATASTEEL.NS", "HINDALCO.NS", "MARUTI.NS", "TATAMOTORS.NS",
]

BENCHMARK = "^NSEI"
OUT = Path("output")
OUT.mkdir(exist_ok=True)
STATE_FILE = Path("paper_state.json")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read {path}. Back up and inspect it before resetting."
        ) from exc


def download_history(symbol):
    df = yf.download(
        symbol,
        period=HISTORY,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError("No price data returned")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        raise ValueError("Missing OHLCV columns")

    df = df[required].copy().dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df[(df["Close"] > 0) & (df["Open"] > 0)]

    if df.empty:
        raise ValueError("No valid daily candles")

    # Indicators
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()
    df["PRIOR_HIGH20"] = df["High"].shift(1).rolling(20).max()

    # Uses only information available at the candle close.
    df["BUY_SIGNAL"] = (
        (df["EMA20"] > df["EMA50"])
        & (df["Close"] > df["PRIOR_HIGH20"])
        & df["RSI"].between(50, 70)
        & (df["ATR"] > 0)
        & (df["Volume"] > 0)
    )

    df["EXIT_SIGNAL"] = (
        (df["Close"] < df["EMA20"]) | (df["RSI"] < 45)
    )

    return df


def download_all():
    data = {}
    errors = []

    for symbol in WATCHLIST:
        try:
            df = download_history(symbol)
            if len(df) < MIN_HISTORY:
                raise ValueError(f"Insufficient history: {len(df)} rows")
            data[symbol] = df
            print(f"OK {symbol}: {len(df)} rows")
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
            print(f"DATA ERROR {symbol}: {exc}")

    try:
        benchmark = download_history(BENCHMARK)
    except Exception as exc:
        benchmark = pd.DataFrame()
        errors.append({"symbol": BENCHMARK, "error": str(exc)})

    if not data:
        raise RuntimeError("All stock downloads failed. No simulation run.")

    return data, benchmark, errors


def buy_fill(raw_price):
    return float(raw_price) * (1 + SLIPPAGE_RATE)


def sell_fill(raw_price):
    return float(raw_price) * (1 - SLIPPAGE_RATE)


def estimate_fee(amount):
    return float(amount) * FEE_RATE


def portfolio_value(state):
    return float(state["cash"]) + sum(
        int(pos["qty"]) * float(pos["last_close"])
        for pos in state["positions"].values()
    )


def open_position(state, symbol, raw_price, atr, date):
    if symbol in state["positions"]:
        return False

    if len(state["positions"]) >= MAX_POSITIONS:
        return False

    price = buy_fill(raw_price)
    if not np.isfinite(price) or price <= 0:
        return False
    if not np.isfinite(atr) or atr <= 0:
        return False

    stop = price - 2 * float(atr)
    target = price + 3 * float(atr)

    if stop <= 0:
        return False

    equity = portfolio_value(state)
    risk_cash = equity * RISK_PER_TRADE
    qty_risk = math.floor(risk_cash / (price - stop))
    qty_alloc = math.floor(equity * MAX_ALLOCATION / price)
    qty_cash = math.floor(state["cash"] / (price * (1 + FEE_RATE)))

    qty = min(qty_risk, qty_alloc, qty_cash)
    if qty < 1:
        return False

    principal = qty * price
    entry_fee = estimate_fee(principal)
    total_cost = principal + entry_fee

    if total_cost > state["cash"]:
        return False

    state["cash"] -= total_cost
    state["positions"][symbol] = {
        "qty": qty,
        "entry": price,
        "entry_date": str(pd.Timestamp(date).date()),
        "entry_fee": entry_fee,
        "entry_cost": total_cost,
        "stop": stop,
        "target": target,
        "last_close": price,
    }
    return True


def close_position(state, symbol, raw_price, date, reason, trades):
    pos = state["positions"].pop(symbol)
    qty = int(pos["qty"])
    price = sell_fill(raw_price)
    proceeds = qty * price
    exit_fee = estimate_fee(proceeds)
    net_proceeds = proceeds - exit_fee
    pnl = net_proceeds - float(pos["entry_cost"])

    state["cash"] += net_proceeds

    trades.append({
        "symbol": symbol,
        "entry_date": pos["entry_date"],
        "exit_date": str(pd.Timestamp(date).date()),
        "quantity": qty,
        "entry_price": round(float(pos["entry"]), 4),
        "exit_price": round(price, 4),
        "entry_fee": round(float(pos["entry_fee"]), 4),
        "exit_fee": round(exit_fee, 4),
        "pnl_after_estimated_costs": round(pnl, 2),
        "return_pct": round(100 * pnl / float(pos["entry_cost"]), 2),
        "reason": reason,
    })


def process_day(state, data, date, pending, trades):
    # Existing positions: first check gaps at the new session's open.
    for symbol in list(state["positions"]):
        df = data.get(symbol)
        if df is None or date not in df.index:
            continue

        i = df.index.get_loc(date)
        row = df.iloc[i]
        pos = state["positions"][symbol]

        # Exit signal was known at the previous close, so execute
        # at this session's open.
        if i > 0 and bool(df.iloc[i - 1]["EXIT_SIGNAL"]):
            close_position(
                state, symbol, float(row["Open"]), date,
                "PRIOR_CLOSE_EXIT", trades
            )
            continue

        if float(row["Open"]) <= float(pos["stop"]):
            close_position(
                state, symbol, float(row["Open"]), date, "GAP_STOP", trades
            )
        elif float(row["Open"]) >= float(pos["target"]):
            close_position(
                state, symbol, float(row["Open"]), date, "GAP_TARGET", trades
            )

    # Entries are based on signals from the previous completed candle.
    for symbol in pending:
        df = data.get(symbol)
        if df is None or date not in df.index:
            continue
        if symbol in state["positions"]:
            continue

        previous = df.loc[df.index < date]
        if previous.empty:
            continue

        atr = float(previous.iloc[-1]["ATR"])
        open_position(state, symbol, float(df.loc[date, "Open"]), atr, date)

    # Intraday stop/target handling. If both are touched, assume the stop
    # was hit first. This avoids optimistic same-day target assumptions.
    for symbol in list(state["positions"]):
        df = data.get(symbol)
        if df is None or date not in df.index:
            continue

        row = df.loc[date]
        pos = state["positions"][symbol]

        if float(row["Low"]) <= float(pos["stop"]):
            close_position(
                state, symbol, float(pos["stop"]), date, "STOP", trades
            )
        elif float(row["High"]) >= float(pos["target"]):
            close_position(
                state, symbol, float(pos["target"]), date, "TARGET", trades
            )

    # Mark remaining positions to today's close.
    for symbol, pos in state["positions"].items():
        df = data.get(symbol)
        if df is not None and date in df.index:
            pos["last_close"] = float(df.loc[date, "Close"])


def portfolio_backtest(data, benchmark):
    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))
    if len(all_dates) < MIN_HISTORY:
        raise RuntimeError("Insufficient aligned date history")

    state = {"cash": CAPITAL, "positions": {}}
    trades = []
    curve = []
    pending = []

    for idx, date in enumerate(all_dates):
        if idx < 60:
            continue

        # Fill yesterday's signals at today's open and process exits.
        process_day(state, data, date, pending, trades)

        equity = portfolio_value(state)
        curve.append({
            "date": str(pd.Timestamp(date).date()),
            "cash": round(float(state["cash"]), 2),
            "equity": round(equity, 2),
            "open_positions": len(state["positions"]),
        })

        # Generate signals after today's close for the next session.
        pending = []
        for symbol, df in data.items():
            if date in df.index and bool(df.loc[date, "BUY_SIGNAL"]):
                pending.append(symbol)

    # Liquidate positions at the last close for final realized equity.
    final_date = all_dates[-1]
    for symbol in list(state["positions"]):
        df = data.get(symbol)
        if df is not None and final_date in df.index:
            close_position(
                state, symbol, float(df.loc[final_date, "Close"]),
                final_date, "END_OF_TEST", trades
            )

    return pd.DataFrame(curve), pd.DataFrame(trades), float(state["cash"])


def max_drawdown(values):
    values = pd.Series(values, dtype=float)
    if values.empty:
        return 0.0
    peaks = values.cummax()
    drawdowns = values / peaks - 1
    return float(drawdowns.min() * 100)


def summarize(curve, trades, final_equity, benchmark):
    if curve.empty:
        return {"error": "No backtest observations"}

    equity = curve["equity"].astype(float)
    total_return = (final_equity / CAPITAL - 1) * 100

    start = pd.Timestamp(curve["date"].iloc[0])
    end = pd.Timestamp(curve["date"].iloc[-1])
    days = max(1, (end - start).days)
    cagr = (
        ((final_equity / CAPITAL) ** (365.25 / days) - 1) * 100
        if final_equity > 0 else -100
    )

    daily_returns = equity.pct_change().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = float(
            np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1)
        )

    pnl = (
        trades["pnl_after_estimated_costs"].astype(float)
        if not trades.empty else pd.Series(dtype=float)
    )

    # Align benchmark to the actual backtest window.
    benchmark_result = {"available": False}
    if not benchmark.empty:
        prices = benchmark["Close"].dropna()
        prices = prices[
            (prices.index >= start) & (prices.index <= end)
        ]
        if len(prices) >= 2:
            benchmark_result = {
                "available": True,
                "symbol": BENCHMARK,
                "start": str(prices.index[0].date()),
                "end": str(prices.index[-1].date()),
                "return_pct": round(
                    (float(prices.iloc[-1]) / float(prices.iloc[0]) - 1) * 100,
                    2
                ),
                "note": "Price index only; dividends excluded.",
            }

    return {
        "mode": "HISTORICAL_SIMULATION",
        "starting_capital": CAPITAL,
        "ending_equity_after_liquidation": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_drawdown(equity), 2),
        "daily_sharpe_approx": round(sharpe, 2),
        "closed_trades": int(len(trades)),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2)
            if len(pnl) else 0.0,
        "trade_pnl_after_estimated_costs": round(float(pnl.sum()), 2)
            if len(pnl) else 0.0,
        "average_trade_pnl": round(float(pnl.mean()), 2)
            if len(pnl) else 0.0,
        "fee_per_side_pct": FEE_RATE * 100,
        "slippage_per_side_pct": SLIPPAGE_RATE * 100,
        "benchmark": benchmark_result,
        "warnings": [
            "Historical simulation is not a prediction.",
            "Yahoo Finance data is unofficial.",
            "Costs are estimates, not actual broker charges.",
            "Daily candles cannot establish exact intraday execution order.",
            "Current constituents can introduce survivorship bias.",
            "Benchmark excludes dividends and investment costs.",
        ],
    }


def paper_account(data):
    state = load_json(STATE_FILE, {
        "mode": "PAPER_ONLY",
        "cash": CAPITAL,
        "positions": {},
        "closed_trades": [],
        "last_processed_date": None,
        "pending_entries": [],
    })

    if state.get("mode") != "PAPER_ONLY":
        raise RuntimeError("Invalid paper state mode. Refusing to continue.")

    dates = sorted(set().union(*(set(df.index) for df in data.values())))
    if not dates:
        raise RuntimeError("No dates for paper account")

    if state["last_processed_date"] is None:
        # Start with current signals only; do not backfill years of trades.
        latest = dates[-1]
        state["last_processed_date"] = str(pd.Timestamp(latest).date())
        state["pending_entries"] = [
            s for s, df in data.items()
            if latest in df.index and bool(df.loc[latest, "BUY_SIGNAL"])
        ]
        save_json(STATE_FILE, state)
    else:
        last = pd.Timestamp(state["last_processed_date"])
        new_dates = [d for d in dates if d > last]

        for date in new_dates:
            completed = []
            process_day(
                state, data, date,
                list(state.get("pending_entries", [])),
                completed
            )
            state.setdefault("closed_trades", []).extend(completed)
            state["pending_entries"] = [
                s for s, df in data.items()
                if date in df.index and bool(df.loc[date, "BUY_SIGNAL"])
            ]
            state["last_processed_date"] = str(pd.Timestamp(date).date())
            save_json(STATE_FILE, state)

    rows = []
    market_value = 0.0
    for symbol, pos in state["positions"].items():
        df = data.get(symbol)
        close = (
            float(df["Close"].iloc[-1])
            if df is not None else float(pos["last_close"])
        )
        pos["last_close"] = close
        value = int(pos["qty"]) * close
        market_value += value
        rows.append({
            "symbol": symbol,
            "quantity": int(pos["qty"]),
            "entry_date": pos["entry_date"],
            "entry_price": round(float(pos["entry"]), 2),
            "latest_close": round(close, 2),
            "stop": round(float(pos["stop"]), 2),
            "target": round(float(pos["target"]), 2),
            "market_value": round(value, 2),
        })

    equity = float(state["cash"]) + market_value
    report = {
        "mode": "PAPER_ONLY",
        "capital": CAPITAL,
        "cash": round(float(state["cash"]), 2),
        "market_value": round(market_value, 2),
        "equity": round(equity, 2),
        "pnl": round(equity - CAPITAL, 2),
        "open_positions": len(state["positions"]),
        "last_processed_date": state["last_processed_date"],
        "pending_entries": state.get("pending_entries", []),
        "closed_paper_trades": len(state.get("closed_trades", [])),
        "warning": "No real orders are sent.",
    }

    save_json(STATE_FILE, state)
    save_json(OUT / "paper_report.json", report)
    pd.DataFrame(rows).to_csv(OUT / "paper_positions.csv", index=False)
    pd.DataFrame(state.get("closed_trades", [])).to_csv(
        OUT / "paper_closed_trades.csv", index=False
    )
    return report


def main():
    print("INDIAN STOCK PAPER TRADER")
    print("Live trading: DISABLED")
    print("Run UTC:", datetime.now(timezone.utc).isoformat())

    data, benchmark, errors = download_all()

    signals = []
    for symbol, df in data.items():
        row = df.iloc[-1]
        signals.append({
            "symbol": symbol,
            "date": str(df.index[-1].date()),
            "close": round(float(row["Close"]), 2),
            "ema20": round(float(row["EMA20"]), 2),
            "ema50": round(float(row["EMA50"]), 2),
            "rsi": round(float(row["RSI"]), 1),
            "atr": round(float(row["ATR"]), 2)
                if np.isfinite(row["ATR"]) else None,
            "signal": "BUY_CANDIDATE" if bool(row["BUY_SIGNAL"])
                else "EXIT_REVIEW" if bool(row["EXIT_SIGNAL"]) else "WATCH",
        })

    pd.DataFrame(signals).to_csv(OUT / "signals.csv", index=False)
    pd.DataFrame(errors).to_csv(OUT / "data_errors.csv", index=False)

    curve, trades, final_equity = portfolio_backtest(data, benchmark)
    curve.to_csv(OUT / "backtest_equity_curve.csv", index=False)
    trades.to_csv(OUT / "backtest_trades.csv", index=False)

    summary = summarize(curve, trades, final_equity, benchmark)
    save_json(OUT / "backtest_summary.json", summary)

    paper = paper_account(data)

    print("\nBACKTEST SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nPAPER ACCOUNT")
    print(json.dumps(paper, indent=2))
    print("\nReports saved in output/")


if __name__ == "__main__":
    main()
