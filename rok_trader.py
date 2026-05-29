"""
ROK Auto Trader — Alpaca Paper Trading Engine
Runs via GitHub Actions every 30 minutes during US market hours.
Reads live signals, buys high-score setups, sells at stop/target.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone

import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- Alpaca config (keys loaded from environment / GitHub Secrets) ---
ALPACA_KEY    = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE   = "https://paper-api.alpaca.markets"   # switch to live URL when ready

# --- Trading rules ---
WATCHLIST = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN",
    "META", "GOOGL", "AMD", "PLTR", "SPY",
    "QQQ",  "COIN", "MSTR", "SOFI", "IBIT",
]
MAX_POSITIONS      = 8      # max open positions at once
MAX_POSITION_PCT   = 0.12   # max 12% of portfolio per trade
STOP_LOSS_PCT      = 0.07   # sell if down 7%
PROFIT_TARGET_PCT  = 0.20   # sell if up 20%
MIN_BUY_SCORE      = 60     # minimum signal score to enter a trade


# ── Alpaca helpers ──────────────────────────────────────────────────────────

def _headers():
    return {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type":        "application/json",
    }

def alpaca_get(path):
    r = requests.get(f"{ALPACA_BASE}{path}", headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()

def alpaca_post(path, data):
    r = requests.post(f"{ALPACA_BASE}{path}", headers=_headers(), json=data, timeout=15)
    r.raise_for_status()
    return r.json()

def alpaca_delete(path):
    r = requests.delete(f"{ALPACA_BASE}{path}", headers=_headers(), timeout=15)
    return r.status_code


# ── Market data ─────────────────────────────────────────────────────────────

def fetch_live_data(tickers):
    """Fetch live prices + volume ratios via yfinance."""
    result = {}
    for tk in tickers:
        try:
            t    = yf.Ticker(tk)
            hist = t.history(period="5d", interval="1d")
            if hist.empty:
                continue
            price     = float(hist["Close"].iloc[-1])
            prev      = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            chg_pct   = (price - prev) / prev * 100 if prev else 0
            vol       = float(hist["Volume"].iloc[-1]) if not hist.empty else 0
            avg_vol   = float(hist["Volume"].mean())   if not hist.empty else vol
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
            week_high = float(hist["High"].max())
            week_low  = float(hist["Low"].min())
            result[tk] = {
                "price":     round(price, 2),
                "change_pct": round(chg_pct, 2),
                "vol_ratio":  round(vol_ratio, 2),
                "week_high":  round(week_high, 2),
                "week_low":   round(week_low, 2),
            }
        except Exception as e:
            logger.warning(f"Data fetch failed for {tk}: {e}")
    return result


# ── Signal engine ────────────────────────────────────────────────────────────

def signal_score(tk, d):
    """
    Score 0-100. Same logic as the dashboard panels.
    Higher = stronger buy signal.
    """
    score   = 0
    chg     = d.get("change_pct", 0) or 0
    vr      = d.get("vol_ratio", 1)  or 1
    price   = d.get("price", 0)      or 0
    wh      = d.get("week_high", price) or price
    wl      = d.get("week_low",  price) or price

    # Momentum
    if chg > 3:   score += 30
    elif chg > 1: score += 18
    elif chg > 0: score += 8
    elif chg < -3: score -= 25
    elif chg < -1: score -= 12

    # Volume surge
    if vr > 2.5:  score += 30
    elif vr > 1.8: score += 20
    elif vr > 1.3: score += 10
    elif vr < 0.6: score -= 8

    # Position in 5-day range
    rng = wh - wl
    if rng > 0:
        pos = (price - wl) / rng * 100
        if 45 < pos < 80: score += 20
        elif pos >= 80:   score += 10
        elif pos < 20:    score -= 10

    return max(0, min(100, int(score)))


# ── Main run ─────────────────────────────────────────────────────────────────

def run():
    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.error("Alpaca keys missing. Add ALPACA_KEY_ID and ALPACA_SECRET_KEY as GitHub Secrets.")
        sys.exit(1)

    # Check market hours
    try:
        clock = alpaca_get("/v2/clock")
        if not clock.get("is_open"):
            logger.info("Market closed — nothing to do.")
            return
        logger.info(f"Market OPEN. Next close: {clock.get('next_close', 'unknown')}")
    except Exception as e:
        logger.error(f"Could not reach Alpaca clock: {e}")
        sys.exit(1)

    # Account snapshot
    account        = alpaca_get("/v2/account")
    portfolio_val  = float(account.get("portfolio_value", 0))
    buying_power   = float(account.get("buying_power", 0))
    logger.info(f"Portfolio: ${portfolio_val:,.2f} | Buying power: ${buying_power:,.2f}")

    # Current positions
    positions = alpaca_get("/v2/positions")
    held      = {p["symbol"]: p for p in positions}
    logger.info(f"Open positions ({len(held)}): {', '.join(held.keys()) or 'none'}")

    # Live data for watchlist + held tickers
    all_tickers = list(set(WATCHLIST + list(held.keys())))
    live        = fetch_live_data(all_tickers)
    logger.info(f"Live data fetched for {len(live)} tickers")

    # ── SELL: stop loss / profit target ──────────────────────────────────
    for symbol, pos in list(held.items()):
        try:
            cost    = float(pos.get("avg_entry_price", 0))
            qty     = float(pos.get("qty", 0))
            current = live.get(symbol, {}).get("price", cost)
            if cost <= 0 or qty <= 0:
                continue
            pnl_pct = (current - cost) / cost * 100

            if pnl_pct <= -(STOP_LOSS_PCT * 100):
                logger.info(f"SELL {symbol} — stop loss hit ({pnl_pct:+.1f}%)")
                alpaca_post("/v2/orders", {
                    "symbol": symbol, "qty": str(int(qty)),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
                del held[symbol]
            elif pnl_pct >= (PROFIT_TARGET_PCT * 100):
                logger.info(f"SELL {symbol} — profit target hit ({pnl_pct:+.1f}%)")
                alpaca_post("/v2/orders", {
                    "symbol": symbol, "qty": str(int(qty)),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
                del held[symbol]
            else:
                logger.info(f"HOLD {symbol} — {pnl_pct:+.1f}% | cost ${cost:.2f} | now ${current:.2f}")
        except Exception as e:
            logger.warning(f"Error processing position {symbol}: {e}")

    # ── BUY: find high-score setups ───────────────────────────────────────
    open_slots = MAX_POSITIONS - len(held)
    if open_slots <= 0:
        logger.info(f"At max positions ({MAX_POSITIONS}). No new buys this cycle.")
        _log_scores(live, held)
        return

    scores = {
        tk: signal_score(tk, live[tk])
        for tk in WATCHLIST
        if tk in live and tk not in held
    }
    _log_scores_dict(scores)

    top = sorted(
        [(tk, sc) for tk, sc in scores.items() if sc >= MIN_BUY_SCORE],
        key=lambda x: -x[1],
    )

    if not top:
        logger.info(f"No tickers scored >= {MIN_BUY_SCORE}. Holding cash this cycle.")
        return

    for tk, sc in top[:open_slots]:
        try:
            price = live[tk]["price"]
            if price <= 0:
                continue
            max_spend = min(portfolio_val * MAX_POSITION_PCT, buying_power * 0.95)
            qty       = int(max_spend / price)
            if qty < 1:
                logger.info(f"SKIP {tk} — not enough buying power for 1 share (${price:.2f})")
                continue
            cost_est = qty * price
            logger.info(f"BUY {tk} — {qty} shares @ ~${price:.2f} = ${cost_est:.0f} (score {sc})")
            alpaca_post("/v2/orders", {
                "symbol": tk, "qty": str(qty),
                "side": "buy", "type": "market", "time_in_force": "day",
            })
            buying_power -= cost_est
        except Exception as e:
            logger.warning(f"Order failed for {tk}: {e}")

    logger.info("Trading cycle complete.")


def _log_scores(live, held):
    scores = {tk: signal_score(tk, live[tk]) for tk in WATCHLIST if tk in live}
    _log_scores_dict(scores)

def _log_scores_dict(scores):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    logger.info("Scores: " + " | ".join(f"{tk}:{sc}" for tk, sc in ranked))


if __name__ == "__main__":
    run()
