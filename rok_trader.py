"""
ROK Auto Trader — Full Market Scanner + Alpaca Paper Trading Engine
Runs every 5 minutes during US market hours via GitHub Actions.
Dynamically scans the full market for movers, not just a fixed watchlist.
Writes trade log to docs/trades.json so the dashboard shows live activity.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- Alpaca config (keys from GitHub Secrets only — never hardcode) ---
ALPACA_KEY    = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE   = "https://paper-api.alpaca.markets"

# --- Trading rules ---
MAX_POSITIONS     = 10     # max open at once
MAX_POSITION_PCT  = 0.10   # max 10% per trade
STOP_LOSS_PCT     = 0.07   # sell if down 7%
PROFIT_TARGET_PCT = 0.20   # sell if up 20%
MIN_BUY_SCORE     = 20     # paper trading: broad entry threshold

TRADES_FILE = Path("docs/trades.json")

# --- Base universe: large, liquid, Alpaca-supported stocks ---
BASE_UNIVERSE = [
    # Mega cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    # Finance
    "JPM", "V", "MA", "BAC", "GS", "MS", "AXP", "SCHW", "BLK", "C",
    # Tech / software
    "ORCL", "CRM", "CSCO", "IBM", "INTU", "NOW", "PANW", "AMAT", "TXN", "MU", "ADI",
    # Healthcare
    "UNH", "JNJ", "ABT", "MRK", "TMO", "ISRG", "AMGN", "GILD", "PFE", "MDT", "SYK",
    # Consumer
    "HD", "MCD", "KO", "PEP", "COST", "PG", "TJX", "LOW", "NKE", "SBUX",
    # Energy
    "XOM", "CVX", "OXY", "SLB",
    # Industrial
    "CAT", "DE", "HON", "BA", "RTX", "GE", "UPS", "ETN", "LMT",
    # Other large caps
    "NFLX", "UBER", "BKNG", "ACN", "SPGI", "MMC", "PYPL", "WMT",
    # High-momentum / crypto-adjacent
    "PLTR", "COIN", "MSTR", "SOFI", "IBIT", "AMD",
    # ETFs
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
    # Popular growth / retail favorites
    "SHOP", "SQ", "RBLX", "HOOD", "DKNG", "ABNB", "DASH", "ROKU",
    "RIVN", "LCID", "JOBY", "ACHR", "SMCI", "ARM", "DELL",
]


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


# ── Trade log helpers ──────────────────────────────────────────────────────

def load_trade_log():
    try:
        return json.loads(TRADES_FILE.read_text()) if TRADES_FILE.exists() \
               else {"trades": [], "positions": [], "last_updated": ""}
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
        "price":   round(float(price), 2),
        "score":   score,
        "pnl_pct": round(float(pnl_pct), 2) if pnl_pct is not None else None,
    }
    if action == "BUY":
        entry["notional"] = round(float(notional_or_qty), 2)
    else:
        entry["qty"] = float(notional_or_qty)
    trade_log["trades"].insert(0, entry)
    trade_log["trades"] = trade_log["trades"][:500]


# ── Market screener ────────────────────────────────────────────────────────────

def get_market_movers():
    """Pull top daily gainers + most active from yfinance screener."""
    movers = []
    for screen_name in ("day_gainers", "most_actives", "day_losers"):
        try:
            result = yf.screen(screen_name)
            quotes = result.get("quotes") or []
            for q in quotes[:30]:
                sym = q.get("symbol", "")
                if sym and 1 < len(sym) <= 5 and sym.isalpha() and sym.isupper():
                    movers.append(sym)
        except Exception as e:
            logger.debug(f"Screener {screen_name} failed: {e}")
    return list(set(movers))


def get_tradeable_fractionable(candidates):
    """Filter to stocks Alpaca says are active + fractionable."""
    try:
        assets = alpaca_get("/v2/assets?status=active&asset_class=us_equity")
        fractionable = {
            a["symbol"]
            for a in assets
            if a.get("tradable") and a.get("fractionable")
        }
        filtered = [s for s in candidates if s in fractionable]
        logger.info(f"Fractionable filter: {len(candidates)} → {len(filtered)} stocks")
        return filtered
    except Exception as e:
        logger.warning(f"Could not fetch Alpaca assets: {e} — using all candidates")
        return candidates


# ── Batch market data ─────────────────────────────────────────────────────────────

def _extract_signals(daily, hourly):
    """Extract signal data from per-ticker daily/hourly DataFrames."""
    if daily is None or len(daily) < 2:
        return None
    daily = daily.dropna(subset=["Close"])
    if len(daily) < 2:
        return None

    price   = float(daily["Close"].iloc[-1])
    prev    = float(daily["Close"].iloc[-2])
    chg_pct = (price - prev) / prev * 100 if prev else 0
    vol     = float(daily["Volume"].iloc[-1]) if "Volume" in daily else 0
    avg_vol = float(daily["Volume"].mean())   if "Volume" in daily else vol
    vol_ratio   = vol / avg_vol if avg_vol > 0 else 1.0
    week_high   = float(daily["High"].max())
    week_low    = float(daily["Low"].min())

    intraday_mom = 0.0
    rsi = 50.0
    if hourly is not None and len(hourly) >= 5 and "Close" in hourly.columns:
        h = hourly.dropna(subset=["Close"])
        if len(h) >= 5:
            recent    = float(h["Close"].iloc[-1])
            reference = float(h["Close"].iloc[-5])
            if reference > 0:
                intraday_mom = (recent - reference) / reference * 100

        if len(h) >= 15:
            closes = list(h["Close"].iloc[-15:])
            gains  = [max(0.0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
            losses = [max(0.0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
            ag = sum(gains)  / len(gains)  if gains  else 0.0
            al = sum(losses) / len(losses) if losses else 0.0
            if al > 0:
                rsi = 100 - (100 / (1 + ag / al))
            elif ag > 0:
                rsi = 100.0

    return {
        "price":        round(price, 2),
        "change_pct":   round(chg_pct, 2),
        "vol_ratio":    round(vol_ratio, 2),
        "week_high":    round(week_high, 2),
        "week_low":     round(week_low, 2),
        "intraday_mom": round(intraday_mom, 2),
        "rsi":          round(rsi, 1),
    }


def fetch_live_data_batch(tickers):
    """Batch download all tickers at once — much faster than one-by-one."""
    if not tickers:
        return {}

    tickers = list(set(tickers))
    result  = {}
    CHUNK   = 60

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            raw_d = yf.download(
                " ".join(chunk), period="5d", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
            raw_h = yf.download(
                " ".join(chunk), period="2d", interval="1h",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )

            for tk in chunk:
                try:
                    if len(chunk) == 1:
                        d = raw_d
                        h = raw_h
                    else:
                        if tk not in raw_d.columns.get_level_values(0):
                            continue
                        d = raw_d[tk]
                        h = raw_h[tk] if tk in raw_h.columns.get_level_values(0) else None

                    sig = _extract_signals(d, h)
                    if sig and sig["price"] > 0:
                        result[tk] = sig
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Batch chunk failed: {e}")

    logger.info(f"Live data: {len(result)}/{len(tickers)} tickers fetched")
    return result


# ── Signal engine ────────────────────────────────────────────────────────────

def signal_score(tk, d):
    """Score 0-100. Base 10 so mild movers still pass threshold 20."""
    score = 10
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
        logger.error("Alpaca keys missing — set ALPACA_KEY_ID and ALPACA_SECRET_KEY as GitHub Secrets.")
        sys.exit(1)

    try:
        clock = alpaca_get("/v2/clock")
        if not clock.get("is_open"):
            logger.info(f"Market closed. Next open: {clock.get('next_open','?')}")
            return
        logger.info(f"Market OPEN — next close: {clock.get('next_close','?')}")
    except Exception as e:
        logger.error(f"Alpaca unreachable: {e}")
        sys.exit(1)

    account       = alpaca_get("/v2/account")
    portfolio_val = float(account.get("portfolio_value", 0))
    buying_power  = float(account.get("buying_power",   0))
    logger.info(f"Portfolio: ${portfolio_val:,.2f} | Buying power: ${buying_power:,.2f}")

    positions = alpaca_get("/v2/positions")
    held      = {p["symbol"]: p for p in positions}
    logger.info(f"Open positions ({len(held)}): {', '.join(held.keys()) or 'none'}")

    movers = get_market_movers()
    logger.info(f"Market movers found: {len(movers)} tickers from screener")

    candidates = list(set(BASE_UNIVERSE + movers + list(held.keys())))
    candidates = get_tradeable_fractionable(candidates)
    logger.info(f"Scanning {len(candidates)} Alpaca-fractionable tickers")

    live = fetch_live_data_batch(candidates)

    trade_log   = load_trade_log()
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

    # ── BUY: score everything, buy top movers ───────────────────────────────────────
    open_slots = MAX_POSITIONS - len(held)
    if open_slots <= 0:
        logger.info(f"At max positions ({MAX_POSITIONS}). No buys this cycle.")
    else:
        scores = {
            tk: signal_score(tk, live[tk])
            for tk in live
            if tk not in held
        }
        top = sorted(
            [(tk, sc) for tk, sc in scores.items() if sc >= MIN_BUY_SCORE],
            key=lambda x: -x[1],
        )

        ranked_str = " | ".join(f"{tk}:{sc}" for tk, sc in top[:15])
        logger.info(f"Top candidates: {ranked_str or 'none above threshold'}")

        if not top:
            logger.info(f"No tickers scored >= {MIN_BUY_SCORE}. Holding cash.")
        else:
            for tk, sc in top[:open_slots]:
                try:
                    price = live[tk]["price"]
                    if price <= 0:
                        continue
                    notional = round(min(portfolio_val * MAX_POSITION_PCT, buying_power * 0.95), 2)
                    if notional < 1:
                        logger.info(f"SKIP {tk} — buying power low (${buying_power:.2f})")
                        continue
                    logger.info(f"BUY {tk} — ${notional:.0f} @ ~${price:.2f} (score {sc})")
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

    # Snapshot positions for dashboard
    try:
        curr = alpaca_get("/v2/positions")
        trade_log["positions"] = [
            {
                "ticker":     p.get("symbol"),
                "qty":        float(p.get("qty", 0)),
                "cost":       float(p.get("avg_entry_price", 0)),
                "price":      float(p.get("current_price", 0)),
                "pnl_pct":    float(p.get("unrealized_plpc", 0)) * 100,
                "pnl_usd":    float(p.get("unrealized_pl", 0)),
                "market_val": float(p.get("market_value", 0)),
            }
            for p in curr
        ]
    except Exception as e:
        logger.warning(f"Could not snapshot positions: {e}")

    trade_log["last_updated"]    = datetime.now(timezone.utc).isoformat()
    trade_log["portfolio_value"] = portfolio_val
    trade_log["buying_power"]    = round(buying_power, 2)

    save_trade_log(trade_log)
    logger.info(f"Cycle complete. Trades this run: {'yes' if made_trades else 'none'}. "
                f"Log total: {len(trade_log['trades'])} entries.")


if __name__ == "__main__":
    run()
