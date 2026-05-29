"""
ROK Auto Trader v3 — Advanced Market Intelligence Engine
Fully automated: scans full market, uses AI sentiment + technical analysis,
sizes positions by volatility, manages trailing stops, logs everything.
Zero human intervention required.
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

# ── Credentials ───────────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_KEY_ID",     "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ALPACA_BASE   = "https://paper-api.alpaca.markets"

# ── Trading parameters ────────────────────────────────────────────────────────
MAX_POSITIONS      = 10
MAX_POSITION_PCT   = 0.10
RISK_PER_TRADE_PCT = 0.01
STOP_LOSS_PCT      = 0.07
PROFIT_TARGET_PCT  = 0.20
TRAILING_STOP_PCT  = 0.05
MIN_BUY_SCORE      = 22

TRADES_FILE = Path("docs/trades.json")
PEAK_FILE   = Path("docs/peaks.json")

# ── Base universe ──────────────────────────────────────────────────────────────
BASE_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","WMT",
    "JPM","V","MA","BAC","GS","MS","AXP","SCHW","BLK","C","PNC",
    "ORCL","CRM","CSCO","IBM","INTU","NOW","PANW","AMAT","TXN","MU","ADI",
    "UNH","JNJ","ABT","MRK","TMO","ISRG","AMGN","GILD","PFE","MDT","SYK","REGN",
    "HD","MCD","KO","PEP","COST","PG","TJX","LOW","NKE","SBUX",
    "XOM","CVX","OXY","SLB","COP",
    "CAT","DE","HON","BA","RTX","GE","UPS","ETN","LMT","NOC",
    "NFLX","UBER","BKNG","ACN","SPGI","MMC","PYPL",
    "PLTR","COIN","MSTR","SOFI","IBIT","AMD",
    "SPY","QQQ","IWM","XLK","XLF","XLE","XLV","XLRE",
    "SHOP","SQ","RBLX","HOOD","DKNG","ABNB","DASH","ROKU",
    "RIVN","SMCI","ARM","DELL","SNOW","DDOG","NET","CRWD","AXON",
    "ENPH","FSLR","CEG","VST","GEV",
]


# ── Alpaca API ───────────────────────────────────────────────────────────────
def _h():
    return {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type":        "application/json",
    }

def alpaca_get(path):
    r = requests.get(f"{ALPACA_BASE}{path}", headers=_h(), timeout=15)
    r.raise_for_status()
    return r.json()

def alpaca_post(path, data):
    r = requests.post(f"{ALPACA_BASE}{path}", headers=_h(), json=data, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Trade / peak log ───────────────────────────────────────────────────────────
def _load(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default

def _save(path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def log_trade(tlog, action, sym, price, notional_or_qty, score=None, pnl=None):
    e = {
        "time":    datetime.now(timezone.utc).isoformat(),
        "action":  action,
        "ticker":  sym,
        "price":   round(float(price), 2),
        "score":   score,
        "pnl_pct": round(float(pnl), 2) if pnl is not None else None,
    }
    if action == "BUY":
        e["notional"] = round(float(notional_or_qty), 2)
    else:
        e["qty"] = float(notional_or_qty)
    tlog["trades"].insert(0, e)
    tlog["trades"] = tlog["trades"][:500]


# ── Technical indicators ────────────────────────────────────────────────────────
def _ema(prices, period):
    if len(prices) < period:
        return None
    k, val = 2 / (period + 1), sum(prices[:period]) / period
    for p in prices[period:]:
        val = p * k + val * (1 - k)
    return val

def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    diffs  = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(0.0, d) for d in diffs[-period:]]
    losses = [max(0.0, -d) for d in diffs[-period:]]
    ag, al = sum(gains)/period, sum(losses)/period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag/al), 1)

def _atr(high, low, close, period=14):
    if len(high) < period + 1:
        return None
    trs = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
           for i in range(1, len(high))]
    return sum(trs[-period:]) / period

def _bollinger(closes, period=20, num_std=2):
    if len(closes) < period:
        return 50.0
    w   = closes[-period:]
    mid = sum(w) / period
    std = (sum((p - mid)**2 for p in w) / period) ** 0.5
    upper = mid + num_std * std
    lower = mid - num_std * std
    if upper == lower:
        return 50.0
    return round((closes[-1] - lower) / (upper - lower) * 100, 1)


# ── AI news sentiment ──────────────────────────────────────────────────────────
def ai_sentiment(ticker):
    if not ANTHROPIC_KEY:
        return 0
    try:
        headlines = [n.get("title", "") for n in yf.Ticker(ticker).news[:6] if n.get("title")]
        if not headlines:
            return 0
        text = "\n".join(headlines)
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 60,
                "messages": [{
                    "role":    "user",
                    "content": (
                        f"Rate stock trading sentiment of these {ticker} headlines "
                        f"from -10 (very bearish) to +10 (very bullish). "
                        f"Return ONLY JSON: {{\"s\":<number>}}\n\n{text}"
                    ),
                }],
            },
            timeout=8,
        )
        result = json.loads(r.json()["content"][0]["text"].strip())
        return max(-10, min(10, float(result.get("s", 0))))
    except Exception as e:
        logger.debug(f"Sentiment error {ticker}: {e}")
        return 0


# ── Market screener ────────────────────────────────────────────────────────────
def get_market_movers():
    movers = []
    for name in ("day_gainers", "most_actives"):
        try:
            res = yf.screen(name)
            for q in (res.get("quotes") or [])[:30]:
                s = q.get("symbol", "")
                if s and 1 < len(s) <= 5 and s.isalpha() and s.isupper():
                    movers.append(s)
        except Exception:
            pass
    return list(set(movers))


def get_tradeable(candidates):
    try:
        assets = alpaca_get("/v2/assets?status=active&asset_class=us_equity")
        ok = {a["symbol"] for a in assets if a.get("tradable") and a.get("fractionable")}
        filtered = [s for s in candidates if s in ok]
        logger.info(f"Fractionable filter: {len(candidates)} → {len(filtered)}")
        return filtered
    except Exception as e:
        logger.warning(f"Asset filter failed: {e} — using full list")
        return candidates


# ── Batch data fetch ─────────────────────────────────────────────────────────────
def _extract(daily, hourly):
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

    atr_val = None
    if len(daily) >= 15:
        highs  = list(daily["High"].iloc[-15:])
        lows   = list(daily["Low"].iloc[-15:])
        closes = list(daily["Close"].iloc[-15:])
        atr_val = _atr(highs, lows, closes)

    rsi_val = 50.0
    ema_cross = 0.0
    macd_val  = 0.0
    bb_pos    = 50.0
    intraday  = 0.0

    if hourly is not None and "Close" in hourly.columns:
        h  = hourly.dropna(subset=["Close"])
        hc = list(h["Close"])

        if len(hc) >= 5:
            intraday = (hc[-1] - hc[-5]) / hc[-5] * 100 if hc[-5] else 0
        if len(hc) >= 15:
            rsi_val = _rsi(hc)
        if len(hc) >= 26:
            e9  = _ema(hc, 9)  or 0
            e21 = _ema(hc, 21) or 0
            e12 = _ema(hc, 12) or 0
            e26 = _ema(hc, 26) or 0
            if e21: ema_cross = (e9  - e21) / e21 * 100
            if e26: macd_val  = (e12 - e26) / e26 * 100
        if len(hc) >= 20:
            bb_pos = _bollinger(hc)

    return {
        "price":      round(price, 2),
        "change_pct": round(chg_pct, 2),
        "vol_ratio":  round(vol_ratio, 2),
        "week_high":  round(week_high, 2),
        "week_low":   round(week_low, 2),
        "intraday":   round(intraday, 2),
        "rsi":        round(rsi_val, 1),
        "ema_cross":  round(ema_cross, 3),
        "macd":       round(macd_val, 3),
        "bb_pos":     round(bb_pos, 1),
        "atr":        round(atr_val, 3) if atr_val else None,
    }


def fetch_batch(tickers):
    if not tickers:
        return {}
    tickers = list(set(tickers))
    result  = {}
    CHUNK   = 60

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            kw    = dict(group_by="ticker", auto_adjust=True, progress=False, threads=True)
            raw_d = yf.download(" ".join(chunk), period="5d", interval="1d", **kw)
            raw_h = yf.download(" ".join(chunk), period="3d", interval="1h", **kw)
            for tk in chunk:
                try:
                    if len(chunk) == 1:
                        d, h = raw_d, raw_h
                    else:
                        lvl = raw_d.columns.get_level_values(0)
                        if tk not in lvl:
                            continue
                        d = raw_d[tk]
                        h = raw_h[tk] if tk in raw_h.columns.get_level_values(0) else None
                    sig = _extract(d, h)
                    if sig and sig["price"] > 0:
                        result[tk] = sig
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Chunk error: {e}")

    logger.info(f"Data: {len(result)}/{len(tickers)} tickers ready")
    return result


# ── Signal scoring ───────────────────────────────────────────────────────────────
def score(tk, d, sentiment=0):
    s     = 10
    chg   = d.get("change_pct",  0) or 0
    vr    = d.get("vol_ratio",   1) or 1
    price = d.get("price",       0) or 0
    wh    = d.get("week_high", price) or price
    wl    = d.get("week_low",  price) or price
    intra = d.get("intraday",    0) or 0
    rsi   = d.get("rsi",        50) or 50
    ema_c = d.get("ema_cross",   0) or 0
    macd  = d.get("macd",        0) or 0
    bb    = d.get("bb_pos",     50) or 50

    if   chg >  4:  s += 25
    elif chg >  2:  s += 18
    elif chg >  1:  s += 12
    elif chg >  0:  s +=  6
    elif chg < -4:  s -= 22
    elif chg < -2:  s -= 14
    elif chg < -1:  s -= 8

    if   intra >  1.5: s += 18
    elif intra >  0.8: s += 11
    elif intra >  0.2: s +=  5
    elif intra < -1.5: s -= 14
    elif intra < -0.8: s -=  8

    if   vr > 3.0:  s += 22
    elif vr > 2.0:  s += 16
    elif vr > 1.5:  s += 10
    elif vr > 1.2:  s +=  5
    elif vr < 0.4:  s -=  8

    if   50 < rsi < 70: s += 14
    elif rsi >= 70:     s +=  4
    elif rsi >  45:     s +=  7
    elif rsi <  25:     s -= 10

    if   ema_c > 0.5: s += 13
    elif ema_c > 0.1: s +=  7
    elif ema_c < -0.5: s -= 11
    elif ema_c < -0.1: s -= 5

    if   macd > 0.3:  s += 12
    elif macd > 0.08: s +=  7
    elif macd < -0.3: s -= 10
    elif macd < -0.08: s -= 5

    if   40 < bb < 75: s += 10
    elif bb >= 75:     s +=  4
    elif bb < 20:      s -= 8

    rng = wh - wl
    if rng > 0:
        pos = (price - wl) / rng * 100
        if   35 < pos < 82: s += 12
        elif pos >= 82:     s +=  5
        elif pos < 18:      s -= 8

    if   sentiment >= 5:  s += 14
    elif sentiment >= 2:  s +=  7
    elif sentiment <= -5: s -= 14
    elif sentiment <= -2: s -=  7

    return max(0, min(100, int(s)))


# ── Dynamic position sizing ────────────────────────────────────────────────────
def calc_notional(portfolio_val, buying_power, price, atr):
    if atr and atr > 0 and price > 0:
        stop_dist   = 2 * atr
        dollar_risk = portfolio_val * RISK_PER_TRADE_PCT
        notional    = (dollar_risk / stop_dist) * price
    else:
        notional = portfolio_val * MAX_POSITION_PCT
    cap = min(portfolio_val * MAX_POSITION_PCT, buying_power * 0.95)
    return round(min(notional, cap), 2)


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

    acct          = alpaca_get("/v2/account")
    portfolio_val = float(acct.get("portfolio_value", 0))
    buying_power  = float(acct.get("buying_power",   0))
    logger.info(f"Portfolio: ${portfolio_val:,.2f} | Cash: ${buying_power:,.2f}")

    positions = alpaca_get("/v2/positions")
    held      = {p["symbol"]: p for p in positions}
    peaks     = _load(PEAK_FILE, {})
    logger.info(f"Positions ({len(held)}): {', '.join(held.keys()) or 'none'}")

    movers     = get_market_movers()
    candidates = get_tradeable(list(set(BASE_UNIVERSE + movers + list(held.keys()))))
    logger.info(f"Scanning {len(candidates)} tickers")

    live = fetch_batch(candidates)
    tlog = _load(TRADES_FILE, {"trades": [], "positions": [], "last_updated": ""})
    made_trades = False

    # ── SELL ─────────────────────────────────────────────────────────────────────
    for sym, pos in list(held.items()):
        try:
            cost    = float(pos.get("avg_entry_price", 0))
            qty     = float(pos.get("qty", 0))
            current = live.get(sym, {}).get("price", cost)
            if cost <= 0 or qty <= 0:
                continue
            pnl_pct = (current - cost) / cost * 100

            prev_peak = peaks.get(sym, current)
            peak      = max(prev_peak, current)
            peaks[sym] = peak
            trail_drop = (current - peak) / peak * 100

            reason = None
            if pnl_pct <= -(STOP_LOSS_PCT * 100):
                reason = f"stop loss ({pnl_pct:+.1f}%)"
            elif pnl_pct >= (PROFIT_TARGET_PCT * 100):
                reason = f"profit target ({pnl_pct:+.1f}%)"
            elif trail_drop <= -(TRAILING_STOP_PCT * 100) and pnl_pct > 0:
                reason = f"trailing stop ({trail_drop:.1f}% from peak ${peak:.2f})"

            if reason:
                logger.info(f"SELL {sym} — {reason}")
                alpaca_post("/v2/orders", {
                    "symbol": sym, "qty": str(qty),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
                log_trade(tlog, "SELL", sym, current, qty, pnl_pct=pnl_pct)
                made_trades = True
                del held[sym]
                peaks.pop(sym, None)
            else:
                logger.info(f"HOLD {sym} — {pnl_pct:+.1f}% | peak ${peak:.2f} | trail {trail_drop:.1f}%")
        except Exception as e:
            logger.warning(f"Sell check error {sym}: {e}")

    # ── BUY ─────────────────────────────────────────────────────────────────────
    open_slots = MAX_POSITIONS - len(held)
    if open_slots <= 0:
        logger.info(f"Max positions ({MAX_POSITIONS}) reached. No buys.")
    else:
        tech_scores = {
            tk: score(tk, live[tk]) for tk in live if tk not in held
        }
        candidates_buy = sorted(
            [(tk, sc) for tk, sc in tech_scores.items() if sc >= MIN_BUY_SCORE - 5],
            key=lambda x: -x[1],
        )[:12]

        logger.info(f"Tech candidates: {' | '.join(f'{t}:{s}' for t,s in candidates_buy[:8])}")

        final_scores = []
        for tk, tech_sc in candidates_buy:
            sent     = ai_sentiment(tk)
            final_sc = score(tk, live[tk], sentiment=sent)
            if final_sc >= MIN_BUY_SCORE:
                final_scores.append((tk, final_sc, sent))
                logger.info(f"  {tk}: tech={tech_sc} sent={sent:+.1f} final={final_sc}")

        final_scores.sort(key=lambda x: -x[1])

        if not final_scores:
            logger.info("No tickers passed final threshold.")
        else:
            for tk, sc, sent in final_scores[:open_slots]:
                try:
                    d        = live[tk]
                    price    = d["price"]
                    atr      = d.get("atr")
                    notional = calc_notional(portfolio_val, buying_power, price, atr)
                    if notional < 1:
                        logger.info(f"SKIP {tk} — insufficient buying power")
                        continue
                    stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
                    logger.info(
                        f"BUY {tk} — ${notional:.0f} @ ~${price:.2f} "
                        f"| stop ${stop_price} | score {sc} | sent {sent:+.0f}"
                    )
                    alpaca_post("/v2/orders", {
                        "symbol":        tk,
                        "notional":      str(notional),
                        "side":          "buy",
                        "type":          "market",
                        "time_in_force": "day",
                    })
                    log_trade(tlog, "BUY", tk, price, notional, score=sc)
                    peaks[tk] = price
                    made_trades = True
                    buying_power -= notional
                except Exception as e:
                    logger.warning(f"Order failed {tk}: {e}")

    _save(PEAK_FILE, peaks)

    try:
        curr = alpaca_get("/v2/positions")
        tlog["positions"] = [
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
        logger.warning(f"Position snapshot failed: {e}")

    tlog["last_updated"]    = datetime.now(timezone.utc).isoformat()
    tlog["portfolio_value"] = portfolio_val
    tlog["buying_power"]    = round(buying_power, 2)

    _save(TRADES_FILE, tlog)
    logger.info(
        f"Cycle done. Trades: {'yes' if made_trades else 'none'}. "
        f"Log: {len(tlog['trades'])} entries."
    )


if __name__ == "__main__":
    run()
