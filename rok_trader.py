"""
ROK Auto Trader v4 — Institutional Intelligence Engine
=======================================================
Upgrades over v3:
  • Market regime filter  — SPY/VIX check; suppress buys in bear market
  • Short selling         — short weak stocks in bear regime
  • Earnings avoidance    — skip stocks within 3 days of earnings
  • VWAP signal           — computed from intraday volume/price
  • 52-week high proximity— breakout bonus in scoring
  • Sector diversification— max 3 positions per sector
  • Partial profit exits  — sell half at +10%, rest at target/trailing stop
  • Position aging        — exit stale positions after MAX_HOLD_DAYS days
  • VIX-adjusted sizing   — shrink bets when market is fearful
  • Performance tracker   — running win rate, avg gain/loss, daily P&L
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
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

# ── Credentials (GitHub Secrets ONLY — never hardcode) ───────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_KEY_ID",     "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ALPACA_BASE   = "https://paper-api.alpaca.markets"

# ── Trading parameters ────────────────────────────────────────────────────────
MAX_POSITIONS      = 12      # max open long positions
MAX_SHORTS         = 4       # max open short positions
MAX_POSITION_PCT   = 0.10    # max 10% of portfolio per position
RISK_PER_TRADE_PCT = 0.01    # risk 1% of portfolio per trade (ATR-sized)
STOP_LOSS_PCT      = 0.07    # hard stop: sell if down 7%
PROFIT_TARGET_PCT  = 0.20    # take full profit at +20%
PARTIAL_PROFIT_PCT = 0.10    # take half profit at +10%
TRAILING_STOP_PCT  = 0.05    # trailing stop: sell if falls 5% from peak
MIN_BUY_SCORE      = 22      # minimum composite score to enter long
MIN_SHORT_SCORE    = 18      # min bearish score to enter short
MAX_HOLD_DAYS      = 7       # exit stale positions after N days
MAX_SECTOR_LONGS   = 3       # max long positions per sector
ENABLE_SHORTS      = True    # enable short selling in bear/neutral regime
VIX_HIGH_THRESH    = 30      # reduce position sizes when VIX above this
VIX_EXTREME_THRESH = 45      # halt new buys when VIX above this

TRADES_FILE = Path("docs/trades.json")
PEAK_FILE   = Path("docs/peaks.json")

# ── Sector map ────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    # Technology
    "AAPL":"tech","MSFT":"tech","NVDA":"tech","GOOGL":"tech","META":"tech",
    "AVGO":"tech","ORCL":"tech","CRM":"tech","CSCO":"tech","IBM":"tech",
    "INTU":"tech","NOW":"tech","PANW":"tech","AMAT":"tech","TXN":"tech",
    "MU":"tech","ADI":"tech","AMD":"tech","PLTR":"tech","ARM":"tech",
    "DELL":"tech","SNOW":"tech","DDOG":"tech","NET":"tech","CRWD":"tech",
    "SMCI":"tech","AXON":"tech",
    # Consumer Tech / Growth
    "AMZN":"consumer_tech","TSLA":"consumer_tech","NFLX":"consumer_tech",
    "UBER":"consumer_tech","BKNG":"consumer_tech","SHOP":"consumer_tech",
    "SQ":"consumer_tech","RBLX":"consumer_tech","HOOD":"consumer_tech",
    "DKNG":"consumer_tech","ABNB":"consumer_tech","DASH":"consumer_tech",
    "ROKU":"consumer_tech","PYPL":"consumer_tech",
    # Financials
    "JPM":"finance","V":"finance","MA":"finance","BAC":"finance","GS":"finance",
    "MS":"finance","AXP":"finance","SCHW":"finance","BLK":"finance",
    "C":"finance","PNC":"finance","SPGI":"finance","MMC":"finance",
    # Healthcare
    "UNH":"health","JNJ":"health","ABT":"health","MRK":"health","TMO":"health",
    "ISRG":"health","AMGN":"health","GILD":"health","PFE":"health",
    "MDT":"health","SYK":"health","REGN":"health",
    # Consumer Staples
    "WMT":"consumer","HD":"consumer","MCD":"consumer","KO":"consumer",
    "PEP":"consumer","COST":"consumer","PG":"consumer","TJX":"consumer",
    "LOW":"consumer","NKE":"consumer","SBUX":"consumer",
    # Energy
    "XOM":"energy","CVX":"energy","OXY":"energy","SLB":"energy","COP":"energy",
    "ENPH":"energy","FSLR":"energy","CEG":"energy","VST":"energy","GEV":"energy",
    # Industrial
    "CAT":"industrial","DE":"industrial","HON":"industrial","BA":"industrial",
    "RTX":"industrial","GE":"industrial","UPS":"industrial","ETN":"industrial",
    "LMT":"industrial","NOC":"industrial","ACN":"industrial",
    # Crypto / Speculative
    "COIN":"crypto","MSTR":"crypto","SOFI":"crypto","IBIT":"crypto","RIVN":"crypto",
    # ETFs
    "SPY":"etf","QQQ":"etf","IWM":"etf","XLK":"etf","XLF":"etf",
    "XLE":"etf","XLV":"etf","XLRE":"etf",
}

# ── Base universe ─────────────────────────────────────────────────────────────
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

# ── Alpaca API helpers ────────────────────────────────────────────────────────
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


# ── Persistence ───────────────────────────────────────────────────────────────
def _load(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default

def _save(path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def log_trade(tlog, action, sym, price, amount, score=None, pnl=None, reason=None):
    e = {
        "time":    datetime.now(timezone.utc).isoformat(),
        "action":  action,
        "ticker":  sym,
        "price":   round(float(price), 2),
        "score":   score,
        "pnl_pct": round(float(pnl), 2) if pnl is not None else None,
        "reason":  reason,
    }
    if action in ("BUY", "SHORT"):
        e["notional"] = round(float(amount), 2)
    else:
        e["qty"] = float(amount)
    tlog.setdefault("trades", []).insert(0, e)
    tlog["trades"] = tlog["trades"][:500]

    # Running stats
    stats = tlog.setdefault("stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
    if pnl is not None:
        stats["total_pnl"] = round(stats.get("total_pnl", 0) + pnl, 2)
        if pnl > 0:
            stats["wins"] = stats.get("wins", 0) + 1
        else:
            stats["losses"] = stats.get("losses", 0) + 1


# ── Technical indicators ──────────────────────────────────────────────────────
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

def _vwap(hourly):
    """Compute VWAP from today's hourly bars. Returns (vwap, position_pct)."""
    if hourly is None:
        return None, 50.0
    try:
        if "Volume" not in hourly.columns or "Close" not in hourly.columns:
            return None, 50.0
        h = hourly.dropna(subset=["Close", "Volume"])
        if len(h) < 2:
            return None, 50.0
        # Use last 8 bars (≈1 trading day in hourly)
        h = h.iloc[-8:]
        tp = (h["High"] + h["Low"] + h["Close"]) / 3
        cum_pv = (tp * h["Volume"]).cumsum()
        cum_v  = h["Volume"].cumsum()
        vwap   = float((cum_pv / cum_v).iloc[-1])
        price  = float(h["Close"].iloc[-1])
        vwap_pos = (price - vwap) / vwap * 100 if vwap > 0 else 0
        return round(vwap, 2), round(vwap_pos, 2)
    except Exception:
        return None, 50.0


# ── Market regime detection ───────────────────────────────────────────────────
def market_regime():
    """
    Returns dict with:
      regime: 'bull' | 'neutral' | 'bear'
      vix:    current VIX level
      spy_trend: % above/below SPY 20-day EMA
      score:  -2 (extreme bear) to +2 (strong bull)
    """
    try:
        spy = yf.download("SPY ^VIX", period="30d", interval="1d",
                          auto_adjust=True, progress=False)
        spy_closes = list(spy["Close"]["SPY"].dropna())
        vix_closes = list(spy["Close"]["^VIX"].dropna())

        vix = float(vix_closes[-1]) if vix_closes else 20.0

        score = 0
        spy_trend = 0.0

        if len(spy_closes) >= 20:
            ema20 = _ema(spy_closes, 20)
            spy_current = spy_closes[-1]
            if ema20:
                spy_trend = round((spy_current - ema20) / ema20 * 100, 2)
                if spy_trend > 1.5:   score += 2
                elif spy_trend > 0.5: score += 1
                elif spy_trend < -1.5: score -= 2
                elif spy_trend < -0.5: score -= 1

            # 5-day momentum
            if len(spy_closes) >= 5:
                mom5 = (spy_closes[-1] - spy_closes[-5]) / spy_closes[-5] * 100
                if mom5 > 1.5:   score += 1
                elif mom5 < -1.5: score -= 1

        if vix > VIX_EXTREME_THRESH:  score -= 3
        elif vix > VIX_HIGH_THRESH:   score -= 1
        elif vix < 16:                score += 1

        if score >= 2:    regime = "bull"
        elif score <= -2: regime = "bear"
        else:             regime = "neutral"

        logger.info(f"Market regime: {regime} | SPY trend: {spy_trend:+.1f}% | VIX: {vix:.1f} | score: {score}")
        return {"regime": regime, "vix": vix, "spy_trend": spy_trend, "score": score}

    except Exception as e:
        logger.warning(f"Regime check failed: {e}")
        return {"regime": "neutral", "vix": 20.0, "spy_trend": 0.0, "score": 0}


# ── Earnings calendar check ───────────────────────────────────────────────────
def has_earnings_soon(sym, days=3):
    """Returns True if this stock has earnings within `days` days — skip it."""
    try:
        cal = yf.Ticker(sym).calendar
        if cal is None or cal.empty:
            return False
        now = datetime.now(timezone.utc).date()
        for col in cal.columns:
            if "earnings" in str(col).lower():
                for val in cal[col]:
                    try:
                        ed = pd.Timestamp(val).date()
                        if 0 <= (ed - now).days <= days:
                            return True
                    except Exception:
                        pass
        return False
    except Exception:
        return False


# ── AI news sentiment ─────────────────────────────────────────────────────────
def ai_sentiment(ticker):
    if not ANTHROPIC_KEY:
        return 0
    try:
        headlines = [n.get("title", "") for n in yf.Ticker(ticker).news[:8] if n.get("title")]
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
                "max_tokens": 80,
                "messages": [{
                    "role":    "user",
                    "content": (
                        f"Rate the short-term stock trading sentiment for {ticker} "
                        f"based on these headlines from -10 (very bearish) to +10 (very bullish). "
                        f"Consider: catalyst strength, market impact, earnings surprise, macro risk. "
                        f"Return ONLY JSON: {{\"s\":<number>,\"c\":\"<1 word reason>\"}}\n\n{text}"
                    ),
                }],
            },
            timeout=10,
        )
        result = json.loads(r.json()["content"][0]["text"].strip())
        return max(-10, min(10, float(result.get("s", 0))))
    except Exception as e:
        logger.debug(f"Sentiment error {ticker}: {e}")
        return 0


# ── Holistic market AI call ───────────────────────────────────────────────────
def ai_market_context(regime, top_movers):
    """
    Ask Claude for a macro market read that adjusts our overall confidence.
    Returns an adjustment score -5 to +5.
    """
    if not ANTHROPIC_KEY:
        return 0
    try:
        movers_str = ", ".join(top_movers[:10])
        prompt = (
            f"Today's market context for an automated US equity trader:\n"
            f"- Regime: {regime['regime']} (VIX={regime['vix']:.0f}, SPY trend={regime['spy_trend']:+.1f}%)\n"
            f"- Top movers today: {movers_str}\n\n"
            f"Should the bot be aggressive or cautious today? "
            f"Return ONLY JSON: {{\"adj\":<-5 to 5>, \"note\":\"<10 words>\"}}"
        )
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=10,
        )
        result = json.loads(r.json()["content"][0]["text"].strip())
        adj  = max(-5, min(5, float(result.get("adj", 0))))
        note = result.get("note", "")
        logger.info(f"AI market context: adj={adj:+.0f} — {note}")
        return adj
    except Exception as e:
        logger.debug(f"Market AI error: {e}")
        return 0


# ── Market screener ───────────────────────────────────────────────────────────
def get_market_movers():
    movers = []
    for name in ("day_gainers", "most_actives", "day_losers"):
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
        ok = {a["symbol"] for a in assets
              if a.get("tradable") and a.get("fractionable")}
        shortable = {a["symbol"] for a in assets
                     if a.get("shortable") and a.get("easy_to_borrow")}
        filtered = [s for s in candidates if s in ok]
        logger.info(f"Fractionable filter: {len(candidates)} → {len(filtered)}")
        return filtered, shortable
    except Exception as e:
        logger.warning(f"Asset filter failed: {e}")
        return candidates, set()


# ── Batch data fetch ──────────────────────────────────────────────────────────
def _extract(daily, hourly):
    if daily is None or len(daily) < 2:
        return None
    daily = daily.dropna(subset=["Close"])
    if len(daily) < 2:
        return None

    price    = float(daily["Close"].iloc[-1])
    prev     = float(daily["Close"].iloc[-2])
    chg_pct  = (price - prev) / prev * 100 if prev else 0
    vol      = float(daily["Volume"].iloc[-1]) if "Volume" in daily else 0
    avg_vol  = float(daily["Volume"].mean())   if "Volume" in daily else vol
    vol_ratio    = vol / avg_vol if avg_vol > 0 else 1.0
    week_high    = float(daily["High"].max())
    week_low     = float(daily["Low"].min())

    # 52-week position (using all available daily data — up to 252 bars)
    high_52w = week_high
    low_52w  = week_low
    try:
        if len(daily) >= 20:
            high_52w = float(daily["High"].max())
            low_52w  = float(daily["Low"].min())
    except Exception:
        pass
    near_52w_high = (price / high_52w) if high_52w > 0 else 1.0

    # ATR from daily
    atr_val = None
    if len(daily) >= 15 and "High" in daily and "Low" in daily:
        highs  = list(daily["High"].iloc[-15:])
        lows   = list(daily["Low"].iloc[-15:])
        closes = list(daily["Close"].iloc[-15:])
        atr_val = _atr(highs, lows, closes)

    # Hourly indicators
    rsi_val   = 50.0
    ema_cross = 0.0
    macd_val  = 0.0
    bb_pos    = 50.0
    intraday  = 0.0
    vwap_pos  = 0.0

    if hourly is not None and "Close" in hourly.columns:
        h  = hourly.dropna(subset=["Close"])
        hc = list(h["Close"])

        if len(hc) >= 5:
            intraday = (hc[-1] - hc[-5]) / hc[-5] * 100 if hc[-5] else 0

        if len(hc) >= 15:
            rsi_val = _rsi(hc)

        if len(hc) >= 26:
            e9  = _ema(hc, 9)
            e21 = _ema(hc, 21)
            e12 = _ema(hc, 12)
            e26 = _ema(hc, 26)
            if e21: ema_cross = (e9 - e21) / e21 * 100
            if e26: macd_val  = (e12 - e26) / e26 * 100

        if len(hc) >= 20:
            bb_pos = _bollinger(hc)

        _, vwap_pos = _vwap(h)

    return {
        "price":        round(price, 2),
        "change_pct":   round(chg_pct, 2),
        "vol_ratio":    round(vol_ratio, 2),
        "week_high":    round(week_high, 2),
        "week_low":     round(week_low, 2),
        "near_52w_high": round(near_52w_high, 4),
        "intraday":     round(intraday, 2),
        "rsi":          round(rsi_val, 1),
        "ema_cross":    round(ema_cross, 3),
        "macd":         round(macd_val, 3),
        "bb_pos":       round(bb_pos, 1),
        "vwap_pos":     round(vwap_pos, 2),
        "atr":          round(atr_val, 3) if atr_val else None,
    }


def fetch_batch(tickers, period_d="30d"):
    """Download tickers in parallel chunks. period_d='30d' gives more data for 52W calc."""
    if not tickers:
        return {}
    tickers = list(set(tickers))
    result  = {}
    CHUNK   = 50

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            kw    = dict(group_by="ticker", auto_adjust=True, progress=False, threads=True)
            raw_d = yf.download(" ".join(chunk), period=period_d, interval="1d", **kw)
            raw_h = yf.download(" ".join(chunk), period="3d",     interval="1h", **kw)

            for tk in chunk:
                try:
                    if len(chunk) == 1:
                        d, h = raw_d, raw_h
                    else:
                        lvl = raw_d.columns.get_level_values(0)
                        if tk not in lvl:
                            continue
                        d    = raw_d[tk]
                        hlvl = raw_h.columns.get_level_values(0)
                        h    = raw_h[tk] if tk in hlvl else None
                    sig = _extract(d, h)
                    if sig and sig["price"] > 0:
                        result[tk] = sig
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Batch chunk error: {e}")

    logger.info(f"Data: {len(result)}/{len(tickers)} tickers ready")
    return result


# ── Signal scoring ────────────────────────────────────────────────────────────
def score(tk, d, sentiment=0, regime_adj=0):
    """
    Composite score 0-100 combining:
    daily momentum, intraday, volume, RSI, EMA cross, MACD,
    Bollinger, VWAP position, 52W proximity, range position, AI sentiment.
    Regime adjustment shifts threshold (-5 to +5 from market AI).
    """
    s     = 10
    chg   = d.get("change_pct",   0) or 0
    vr    = d.get("vol_ratio",    1) or 1
    price = d.get("price",        0) or 0
    wh    = d.get("week_high", price) or price
    wl    = d.get("week_low",  price) or price
    intra = d.get("intraday",     0) or 0
    rsi   = d.get("rsi",         50) or 50
    ema_c = d.get("ema_cross",    0) or 0
    macd  = d.get("macd",         0) or 0
    bb    = d.get("bb_pos",      50) or 50
    vwap  = d.get("vwap_pos",     0) or 0
    n52w  = d.get("near_52w_high", 1.0) or 1.0

    # Daily momentum (+25/-22)
    if   chg >  4:  s += 25
    elif chg >  2:  s += 18
    elif chg >  1:  s += 12
    elif chg >  0:  s +=  6
    elif chg < -4:  s -= 22
    elif chg < -2:  s -= 14
    elif chg < -1:  s -= 8

    # Intraday 4h momentum (+18/-14)
    if   intra >  1.5: s += 18
    elif intra >  0.8: s += 11
    elif intra >  0.2: s +=  5
    elif intra < -1.5: s -= 14
    elif intra < -0.8: s -=  8

    # Volume confirmation (+22/-8)
    if   vr > 3.0:  s += 22
    elif vr > 2.0:  s += 16
    elif vr > 1.5:  s += 10
    elif vr > 1.2:  s +=  5
    elif vr < 0.4:  s -=  8

    # RSI (+14/-10)
    if   50 < rsi < 70: s += 14
    elif rsi >= 70:     s +=  4
    elif rsi >  45:     s +=  7
    elif rsi <  25:     s -= 10

    # EMA 9/21 cross (+13/-11)
    if   ema_c > 0.5:  s += 13
    elif ema_c > 0.1:  s +=  7
    elif ema_c < -0.5: s -= 11
    elif ema_c < -0.1: s -= 5

    # MACD (+12/-10)
    if   macd > 0.3:  s += 12
    elif macd > 0.08: s +=  7
    elif macd < -0.3: s -= 10
    elif macd < -0.08: s -= 5

    # Bollinger position (+10/-8)
    if   40 < bb < 75: s += 10
    elif bb >= 75:     s +=  4
    elif bb < 20:      s -= 8

    # VWAP position: above VWAP = bullish (+8/-8)
    if   vwap > 0.5:  s +=  8
    elif vwap > 0.1:  s +=  4
    elif vwap < -0.5: s -=  8
    elif vwap < -0.1: s -=  4

    # Near 52-week high breakout (+10)
    if   n52w >= 0.99:  s += 10   # within 1% of 52W high — breakout territory
    elif n52w >= 0.97:  s +=  6
    elif n52w >= 0.95:  s +=  3
    elif n52w <= 0.80:  s -=  5   # far from highs, likely in downtrend

    # 5-day range position (+12/-8)
    rng = wh - wl
    if rng > 0:
        pos = (price - wl) / rng * 100
        if   35 < pos < 82: s += 12
        elif pos >= 82:     s +=  5
        elif pos < 18:      s -= 8

    # AI sentiment (+14/-14)
    if   sentiment >= 5:  s += 14
    elif sentiment >= 2:  s +=  7
    elif sentiment <= -5: s -= 14
    elif sentiment <= -2: s -=  7

    # Market regime adjustment
    s += regime_adj

    return max(0, min(100, int(s)))


def bearish_score(tk, d):
    """
    Score how bearish a stock is (0-100). Used for short candidates.
    Mirror of score() but optimized for finding weak/falling stocks.
    """
    chg   = d.get("change_pct",  0) or 0
    intra = d.get("intraday",    0) or 0
    rsi   = d.get("rsi",        50) or 50
    vr    = d.get("vol_ratio",   1) or 1
    ema_c = d.get("ema_cross",   0) or 0
    macd  = d.get("macd",        0) or 0
    bb    = d.get("bb_pos",     50) or 50
    vwap  = d.get("vwap_pos",    0) or 0
    n52w  = d.get("near_52w_high", 1.0) or 1.0

    s = 10
    if chg   < -4:   s += 25
    elif chg < -2:   s += 18
    elif chg < -1:   s += 10
    if intra < -1.5: s += 15
    elif intra < -0.8: s += 8
    if rsi   < 30:   s += 14
    elif rsi < 40:   s += 7
    if vr    > 2.0:  s += 10
    if ema_c < -0.5: s += 12
    elif ema_c < -0.1: s += 5
    if macd  < -0.3: s += 10
    elif macd < -0.08: s += 5
    if bb    < 20:   s += 10
    if vwap  < -0.5: s += 8
    if n52w  <= 0.80: s += 8

    return max(0, min(100, int(s)))


# ── Position sizing ───────────────────────────────────────────────────────────
def calc_notional(portfolio_val, buying_power, price, atr, vix=20.0):
    """ATR-based risk sizing, scaled down when VIX is high."""
    vix_scale = 1.0
    if vix > VIX_EXTREME_THRESH:   vix_scale = 0.4
    elif vix > VIX_HIGH_THRESH:    vix_scale = 0.65
    elif vix > 20:                 vix_scale = 0.85

    if atr and atr > 0 and price > 0:
        stop_dist   = 2 * atr
        dollar_risk = portfolio_val * RISK_PER_TRADE_PCT * vix_scale
        notional    = (dollar_risk / stop_dist) * price
    else:
        notional = portfolio_val * MAX_POSITION_PCT * vix_scale

    cap = min(portfolio_val * MAX_POSITION_PCT, buying_power * 0.95)
    return round(min(notional, cap), 2)


# ── Main trading engine ───────────────────────────────────────────────────────
def run():
    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.error("Alpaca keys missing — set ALPACA_KEY_ID + ALPACA_SECRET_KEY as GitHub Secrets.")
        sys.exit(1)

    # Market clock
    try:
        clock = alpaca_get("/v2/clock")
        if not clock.get("is_open"):
            logger.info(f"Market closed. Next open: {clock.get('next_open', '?')}")
            return
        logger.info(f"Market OPEN — next close: {clock.get('next_close', '?')}")
    except Exception as e:
        logger.error(f"Alpaca unreachable: {e}")
        sys.exit(1)

    # Account
    acct          = alpaca_get("/v2/account")
    portfolio_val = float(acct.get("portfolio_value", 0))
    buying_power  = float(acct.get("buying_power",   0))
    logger.info(f"Portfolio: ${portfolio_val:,.2f} | Cash: ${buying_power:,.2f}")

    # Market regime
    regime = market_regime()
    vix    = regime["vix"]
    if vix > VIX_EXTREME_THRESH:
        logger.warning(f"VIX={vix:.0f} EXTREME — halting new buys, protecting capital.")

    # Positions + peaks
    positions = alpaca_get("/v2/positions")
    held      = {p["symbol"]: p for p in positions}
    # Separate longs and shorts
    longs  = {s: p for s, p in held.items() if float(p.get("qty", 0)) > 0}
    shorts = {s: p for s, p in held.items() if float(p.get("qty", 0)) < 0}
    peaks  = _load(PEAK_FILE, {})
    logger.info(f"Longs ({len(longs)}): {', '.join(longs) or 'none'}")
    logger.info(f"Shorts ({len(shorts)}): {', '.join(shorts) or 'none'}")

    # Build scan universe
    movers    = get_market_movers()
    all_cands = list(set(BASE_UNIVERSE + movers + list(held.keys())))
    candidates, shortable = get_tradeable(all_cands)
    logger.info(f"Scanning {len(candidates)} tickers | {len(shortable)} shortable")

    # Fetch live data
    live = fetch_batch(candidates)

    # AI market context adjustment
    regime_adj = ai_market_context(regime, movers[:10])

    tlog        = _load(TRADES_FILE, {"trades": [], "positions": [], "last_updated": ""})
    made_trades = False
    now_utc     = datetime.now(timezone.utc)

    # ── MANAGE EXISTING SHORTS ─────────────────────────────────────────────
    for sym, pos in list(shorts.items()):
        try:
            cost    = float(pos.get("avg_entry_price", 0))
            qty     = abs(float(pos.get("qty", 0)))    # qty is negative for shorts
            current = live.get(sym, {}).get("price", cost)
            if cost <= 0 or qty <= 0:
                continue

            # For shorts: profit when price drops
            pnl_pct = (cost - current) / cost * 100

            reason = None
            if pnl_pct <= -(STOP_LOSS_PCT * 100):      # short went against us
                reason = f"short stop loss ({pnl_pct:+.1f}%)"
            elif pnl_pct >= (PROFIT_TARGET_PCT * 100):
                reason = f"short profit target ({pnl_pct:+.1f}%)"
            elif regime["regime"] == "bull":
                reason = "regime flip to bull — cover short"

            if reason:
                logger.info(f"COVER {sym} — {reason}")
                alpaca_post("/v2/orders", {
                    "symbol": sym, "qty": str(qty),
                    "side": "buy", "type": "market", "time_in_force": "day",
                })
                log_trade(tlog, "COVER", sym, current, qty, pnl=pnl_pct, reason=reason)
                made_trades = True
                del held[sym]
                peaks.pop(sym, None)
            else:
                logger.info(f"HOLD SHORT {sym} — P&L {pnl_pct:+.1f}%")
        except Exception as e:
            logger.warning(f"Short management error {sym}: {e}")

    # ── MANAGE EXISTING LONGS ─────────────────────────────────────────────
    for sym, pos in list(longs.items()):
        try:
            cost    = float(pos.get("avg_entry_price", 0))
            qty     = float(pos.get("qty", 0))
            current = live.get(sym, {}).get("price", cost)
            if cost <= 0 or qty <= 0:
                continue
            pnl_pct = (current - cost) / cost * 100

            # Trailing peak
            prev_peak  = peaks.get(sym, {}).get("peak", current) if isinstance(peaks.get(sym), dict) else peaks.get(sym, current)
            peak       = max(prev_peak, current)
            entry_time = peaks.get(sym, {}).get("time") if isinstance(peaks.get(sym), dict) else None
            peaks[sym] = {"peak": peak, "time": entry_time or now_utc.isoformat(),
                          "half_out": peaks.get(sym, {}).get("half_out", False) if isinstance(peaks.get(sym), dict) else False}
            trail_drop = (current - peak) / peak * 100

            # Position age
            age_days = 0
            if entry_time:
                try:
                    et      = datetime.fromisoformat(entry_time)
                    age_days = (now_utc - et).days
                except Exception:
                    pass

            # ── Partial exit at +10% (sell half) ──
            half_out = peaks[sym].get("half_out", False)
            if pnl_pct >= (PARTIAL_PROFIT_PCT * 100) and not half_out and qty >= 2:
                half_qty = round(qty / 2, 4)
                logger.info(f"SELL_HALF {sym} — partial at {pnl_pct:+.1f}%")
                try:
                    alpaca_post("/v2/orders", {
                        "symbol": sym, "qty": str(half_qty),
                        "side": "sell", "type": "market", "time_in_force": "day",
                    })
                    log_trade(tlog, "SELL_HALF", sym, current, half_qty,
                              pnl=pnl_pct, reason=f"partial profit ({pnl_pct:+.1f}%)")
                    peaks[sym]["half_out"] = True
                    made_trades = True
                except Exception as e:
                    logger.warning(f"Partial sell failed {sym}: {e}")
                continue

            # ── Full exit conditions ──
            reason = None
            if pnl_pct <= -(STOP_LOSS_PCT * 100):
                reason = f"stop loss ({pnl_pct:+.1f}%)"
            elif pnl_pct >= (PROFIT_TARGET_PCT * 100):
                reason = f"profit target ({pnl_pct:+.1f}%)"
            elif trail_drop <= -(TRAILING_STOP_PCT * 100) and pnl_pct > 0:
                reason = f"trailing stop ({trail_drop:.1f}% from peak ${peak:.2f})"
            elif age_days >= MAX_HOLD_DAYS and pnl_pct < 2:
                reason = f"stale position ({age_days}d, {pnl_pct:+.1f}%)"

            if reason:
                logger.info(f"SELL {sym} — {reason}")
                alpaca_post("/v2/orders", {
                    "symbol": sym, "qty": str(qty),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
                log_trade(tlog, "SELL", sym, current, qty, pnl=pnl_pct, reason=reason)
                made_trades = True
                del longs[sym]
                del held[sym]
                peaks.pop(sym, None)
            else:
                logger.info(
                    f"HOLD {sym} — {pnl_pct:+.1f}% | peak ${peak:.2f} "
                    f"| trail {trail_drop:.1f}% | age {age_days}d"
                )
        except Exception as e:
            logger.warning(f"Sell check error {sym}: {e}")

    # ── BUY: long positions ───────────────────────────────────────────────
    open_long_slots = MAX_POSITIONS - len(longs)

    if open_long_slots > 0 and vix <= VIX_EXTREME_THRESH:
        # Sector counts for diversification
        sector_counts = {}
        for sym in longs:
            sec = SECTOR_MAP.get(sym, "other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        # Technical pass
        tech_scores = {
            tk: score(tk, live[tk], regime_adj=regime_adj)
            for tk in live if tk not in held
        }
        candidates_buy = sorted(
            [(tk, sc) for tk, sc in tech_scores.items() if sc >= MIN_BUY_SCORE - 5],
            key=lambda x: -x[1],
        )[:15]

        logger.info(f"Tech long candidates: {' | '.join(f'{t}:{s}' for t,s in candidates_buy[:8])}")

        # Earnings filter + sector filter + AI sentiment pass
        final_scores = []
        for tk, tech_sc in candidates_buy:
            sec = SECTOR_MAP.get(tk, "other")
            if sector_counts.get(sec, 0) >= MAX_SECTOR_LONGS:
                logger.debug(f"SKIP {tk} — sector {sec} full ({sector_counts.get(sec,0)}/{MAX_SECTOR_LONGS})")
                continue
            if has_earnings_soon(tk):
                logger.info(f"SKIP {tk} — earnings within 3 days")
                continue
            sent = ai_sentiment(tk)
            final_sc = score(tk, live[tk], sentiment=sent, regime_adj=regime_adj)
            if final_sc >= MIN_BUY_SCORE:
                final_scores.append((tk, final_sc, sent, sec))
                logger.info(f"  {tk}: tech={tech_sc} sent={sent:+.1f} final={final_sc} sector={sec}")

        final_scores.sort(key=lambda x: -x[1])

        if not final_scores:
            logger.info(f"No longs passed threshold {MIN_BUY_SCORE}.")
        else:
            for tk, sc, sent, sec in final_scores[:open_long_slots]:
                try:
                    d        = live[tk]
                    price    = d["price"]
                    atr      = d.get("atr")
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix)
                    if notional < 1:
                        logger.info(f"SKIP {tk} — insufficient buying power")
                        continue
                    stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
                    logger.info(
                        f"BUY {tk} — ${notional:.0f} @ ~${price:.2f} "
                        f"| stop ${stop_price} | score {sc} | sent {sent:+.0f} | VIX {vix:.0f}"
                    )
                    alpaca_post("/v2/orders", {
                        "symbol":        tk,
                        "notional":      str(notional),
                        "side":          "buy",
                        "type":          "market",
                        "time_in_force": "day",
                    })
                    log_trade(tlog, "BUY", tk, price, notional, score=sc,
                              reason=f"score={sc} sent={sent:+.0f}")
                    peaks[tk] = {"peak": price, "time": now_utc.isoformat(), "half_out": False}
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                    made_trades  = True
                    buying_power -= notional
                except Exception as e:
                    logger.warning(f"BUY failed {tk}: {e}")

    # ── SHORT: bearish positions in bear/neutral regime ───────────────────
    if ENABLE_SHORTS and regime["regime"] in ("bear", "neutral"):
        open_short_slots = MAX_SHORTS - len(shorts)
        if open_short_slots > 0:
            short_scores = {
                tk: bearish_score(tk, live[tk])
                for tk in live
                if tk not in held and tk in shortable
            }
            short_candidates = sorted(
                [(tk, sc) for tk, sc in short_scores.items() if sc >= MIN_SHORT_SCORE],
                key=lambda x: -x[1],
            )[:8]
            logger.info(f"Short candidates: {' | '.join(f'{t}:{s}' for t,s in short_candidates[:5])}")

            for tk, sc in short_candidates[:open_short_slots]:
                try:
                    if has_earnings_soon(tk):
                        continue
                    d        = live[tk]
                    price    = d["price"]
                    atr      = d.get("atr")
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix)
                    notional = round(notional * 0.6, 2)   # size shorts smaller
                    if notional < 1:
                        continue
                    logger.info(f"SHORT {tk} — ${notional:.0f} @ ~${price:.2f} | bear score {sc}")
                    alpaca_post("/v2/orders", {
                        "symbol":        tk,
                        "notional":      str(notional),
                        "side":          "sell",
                        "type":          "market",
                        "time_in_force": "day",
                    })
                    log_trade(tlog, "SHORT", tk, price, notional, score=sc,
                              reason=f"bear score={sc} regime={regime['regime']}")
                    made_trades  = True
                    buying_power -= notional
                except Exception as e:
                    logger.warning(f"SHORT failed {tk}: {e}")

    # ── Save state + dashboard snapshot ──────────────────────────────────
    _save(PEAK_FILE, peaks)

    try:
        curr = alpaca_get("/v2/positions")
        tlog["positions"] = [
            {
                "ticker":     p.get("symbol"),
                "side":       "long" if float(p.get("qty", 0)) > 0 else "short",
                "qty":        abs(float(p.get("qty", 0))),
                "cost":       float(p.get("avg_entry_price", 0)),
                "price":      float(p.get("current_price",  0)),
                "pnl_pct":    float(p.get("unrealized_plpc", 0)) * 100,
                "pnl_usd":    float(p.get("unrealized_pl",  0)),
                "market_val": float(p.get("market_value",   0)),
            }
            for p in curr
        ]
    except Exception as e:
        logger.warning(f"Position snapshot failed: {e}")

    tlog["last_updated"]    = now_utc.isoformat()
    tlog["portfolio_value"] = portfolio_val
    tlog["buying_power"]    = round(buying_power, 2)
    tlog["regime"]          = regime

    _save(TRADES_FILE, tlog)
    logger.info(
        f"Cycle done. Trades: {'yes' if made_trades else 'none'}. "
        f"Regime: {regime['regime']}. Log: {len(tlog['trades'])} entries."
    )


if __name__ == "__main__":
    run()
