"""
ROK Auto Trader — Alpaca Paper Trading Engine
Runs via GitHub Actions every 5 minutes during US market hours.
Reads live signals, buys high-score setups, sells at stop/target.
Writes trade log to docs/trades.json so the dashboard shows live activity.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
MAX_POSITIONS     = 8      # max open positions at once
MAX_POSITION_PCT  = 0.12   # max 12% of portfolio per trade
STOP_LOSS_PCT     = 0.07   # sell if down 7%
PROFIT_TARGET_PCT = 0.20   # sell if up 20%
MIN_BUY_SCORE     = 20     # paper trading: buy anything with slight momentum

TRADES_FILE = Path("docs/trades.json")


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


# ── Trade log helpers ──────────────────────────────────────────────────────

def load_trade_log():
    try:
        return json.loads(TRADES_FILE.read_text()) if TRADES_FILE.exists() else {"trades": [], "positions": [], "last_updated": ""}
    except Exception:
        return {"trades": [], "positions": [], "last_updated": ""}

def save_trade_log(data):
    TRADES_FILE.parent.mkdir(exist_ok=True)
    TRADES_FILE.write_text(json.dumps(data, indent=2))

def log_trade(trade_log, action, symbol, price, notional_or_qty, score=None, pnl_pct=None):
    entry = {
        "time":    datetime.now(timezone.utc).isoformat(),
        "action":  action,
        "ticker":  symbol,
        "price":   round(price, 2),
        "score":   score,
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
    }
    if action == "BUY":
        entry["notional"] = round(float(notional_or_qty), 2)
    else:
        entry["qty"] = float(notional_or_qty)
    trade_log["trades"].insert(0, entry)
    trade_log["trades"] = trade_log["trades"][:200]  # keep last 200


# ── Market data ─────────────────────────────────────────────────────────────

def fetch_live_data(tickers):
    """Fetch prices + intraday momentum + RSI via yfinance."""
    result = {}
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            daily  = t.history(period="5d",  interval="1d")
            hourly = t.history(period="2d",  interval="1h")

            if daily.empty:
                continue

            price   = float(daily["Close"].iloc[-1])
            prev    = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else price
            chg_pct = (price - prev) / prev * 100 if prev else 0

            vol     = float(daily["Volume"].iloc[-1])
            avg_vol = float(daily["Volume"].mean())
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0

            week_high = float(daily["High"].max())
            week_low  = float(daily["Low"].min())

            intraday_mom = 0.0
            if not hourly.empty and len(hourly) >= 5:
                recent    = float(hourly["Close"].iloc[-1])
                reference = float(hourly["Close"].iloc[-5])
                if reference > 0:
                    intraday_mom = (recent - reference) / reference * 100

            rsi = 50.0
            if not hourly.empty and len(hourly) >= 15:
                closes = list(hourly["Close"].iloc[-15:])
                gains  = [max(0.0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
                losses = [max(0.0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
                ag = sum(gains)  / len(gains)  if gains  else 0.0
                al = sum(losses) / len(losses) if losses else 0.0
                if al > 0:
                    rsi = 100 - (100 / (1 + ag / al))
                elif ag > 0:
                    rsi = 100.0

            result[tk] = {
                "price":        round(price, 2),
                "change_pct":   round(chg_pct, 2),
                "vol_ratio":    round(vol_ratio, 2),
                "week_high":    round(week_high, 2),
                "week_low":     round(week_low, 2),
                "intraday_mom": round(intraday_mom, 2),
                "rsi":          round(rsi, 1),
            }
        except Exception as e:
            logger.warning(f"Data fetch failed for {tk}: {e}")
    return result


# ── Signal engine ────────────────────────────────────────────────────────────

def signal_score(tk, d):
    """Score 0-100. Threshold 20 = paper trading mode."""
    score = 10  # base

    chg   = d.get("change_pct",   0)  or 0
    vr    = d.get("vol_ratio",    1)  or 1
    price = d.get("price",        0)  or 0
    wh    = d.get("week_high", price) or price
    wl    = d.get("week_low",  price) or price
    intra = d.get("intraday_mom", 0)  or 0
    rsi   = d.get("rsi",         50)  or 50

    if   chg >  3:  score += 25
    elif chg >  1:  score += 15
    elif chg >  0:  score +=  8
    elif chg < -3:  score -= 20
    elif chg < -1:  score -= 10

    if   intra >  1.0: score += 20
    elif intra >  0.5: score += 12
    elif intra >  0.0: score +=  5
    elif intra < -1.0: score -= 15
    elif intra < -0.5: score -=  8

    if   vr > 2.5:  score += 25
    elif vr > 1.8:  score += 18
    elif vr > 1.3:  score += 10
    elif vr < 0.4:  score -=  8

    if   45 < rsi < 75: score += 15
    elif rsi >= 75:     score +=  5
    elif rsi >  50:     score +=  8
    elif rsi <  25:     score -= 10

    rng = wh - wl
    if rng > 0:
        pos = (price - wl) / rng * 100
        if   30 < pos < 85: score += 15
        elif pos >= 85:     score +=  8
        elif pos <  15:     score -=  8

    return max(0, min(100, int(score)))


# ── Main run ─────────────────────────────────────────────────────────────────

def run():
    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.error("Alpaca keys missing. Add ALPACA_KEY_ID and ALPACA_SECRET_KEY as GitHub Secrets.")
        sys.exit(1)

    try:
        clock = alpaca_get("/v2/clock")
        if not clock.get("is_open"):
            logger.info("Market closed — nothing to do.")
            logger.info(f"Next open: {clock.get('next_open', 'unknown')}")
            return
        logger.info(f"Market OPEN. Next close: {clock.get('next_close', 'unknown')}")
    except Exception as e:
        logger.error(f"Could not reach Alpaca: {e}")
        sys.exit(1)

    account       = alpaca_get("/v2/account")
    portfolio_val = float(account.get("portfolio_value", 0))
    buying_power  = float(account.get("buying_power",   0))
    logger.info(f"Portfolio: ${portfolio_val:,.2f} | Buying power: ${buying_power:,.2f}")

    positions = alpaca_get("/v2/positions")
    held      = {p["symbol"]: p for p in positions}
    logger.info(f"Open positions ({len(held)}): {', '.join(held.keys()) or 'none'}")

    all_tickers = list(set(WATCHLIST + list(held.keys())))
    live        = fetch_live_data(all_tickers)
    logger.info(f"Live data fetched for {len(live)} tickers")

    trade_log = load_trade_log()
    made_trades = False

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
                logger.info(f"SELL {symbol} — stop loss ({pnl_pct:+.1f}%)")
                alpaca_post("/v2/orders", {
                    "symbol": symbol, "qty": str(qty),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
                log_trade(trade_log, "SELL", symbol, current, qty, pnl_pct=pnl_pct)
                made_trades = True
                del held[symbol]
            elif pnl_pct >= (PROFIT_TARGET_PCT * 100):
                logger.info(f"SELL {symbol} — profit target ({pnl_pct:+.1f}%)")
                alpaca_post("/v2/orders", {
                    "symbol": symbol, "qty": str(qty),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
                log_trade(trade_log, "SELL", symbol, current, qty, pnl_pct=pnl_pct)
                made_trades = True
                del held[symbol]
            else:
                logger.info(f"HOLD {symbol} — {pnl_pct:+.1f}% | cost ${cost:.2f} | now ${current:.2f}")
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    # ── BUY: find high-score setups ───────────────────────────────────────
    open_slots = MAX_POSITIONS - len(held)
    if open_slots <= 0:
        logger.info(f"At max positions ({MAX_POSITIONS}). No buys this cycle.")
        _log_scores(live, held)
    else:
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
        else:
            for tk, sc in top[:open_slots]:
                try:
                    price = live[tk]["price"]
                    if price <= 0:
                        continue
                    notional = round(min(portfolio_val * MAX_POSITION_PCT, buying_power * 0.95), 2)
                    if notional < 1:
                        logger.info(f"SKIP {tk} — buying power too low (${buying_power:.2f})")
                        continue
                    logger.info(f"BUY {tk} — ${notional:.0f} notional @ ~${price:.2f} (score {sc})")
                    alpaca_post("/v2/orders", {
                        "symbol":        tk,
                        "notional":      str(notional),
                        "side":          "buy",
                        "type":          "market",
                        "time_in_force": "day",
                    })
                    log_trade(trade_log, "BUY", tk, price, notional, score=sc)
                    made_trades = True
                    buying_power -= notional
                except Exception as e:
                    logger.warning(f"Order failed for {tk}: {e}")

    # Snapshot current positions for dashboard
    try:
        curr_positions = alpaca_get("/v2/positions")
        trade_log["positions"] = [
            {
                "ticker":    p.get("symbol"),
                "qty":       float(p.get("qty", 0)),
                "cost":      float(p.get("avg_entry_price", 0)),
                "price":     float(p.get("current_price", 0)),
                "pnl_pct":   float(p.get("unrealized_plpc", 0)) * 100,
                "pnl_usd":   float(p.get("unrealized_pl", 0)),
                "market_val": float(p.get("market_value", 0)),
            }
            for p in curr_positions
        ]
    except Exception as e:
        logger.warning(f"Could not snapshot positions: {e}")

    trade_log["last_updated"]    = datetime.now(timezone.utc).isoformat()
    trade_log["portfolio_value"] = portfolio_val
    trade_log["buying_power"]    = buying_power

    save_trade_log(trade_log)
    if made_trades:
        logger.info(f"Trade log updated — {len(trade_log['trades'])} total entries")
    else:
        logger.info("No trades this cycle — log updated with position snapshot")

    logger.info("Trading cycle complete.")


def _log_scores(live, held):
    scores = {tk: signal_score(tk, live[tk]) for tk in WATCHLIST if tk in live}
    _log_scores_dict(scores)

def _log_scores_dict(scores):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    logger.info("Scores: " + " | ".join(f"{tk}:{sc}" for tk, sc in ranked))


if __name__ == "__main__":
    run()
