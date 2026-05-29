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
MIN_BUY_SCORE      = 18      # minimum composite score to enter long
MIN_SHORT_SCORE    = 18      # min bearish score to enter short
MAX_HOLD_DAYS      = 5       # exit stale positions after N days
MAX_SECTOR_LONGS   = 3       # max long positions per sector
ENABLE_SHORTS      = True    # enable short selling in bear/neutral regime
VIX_HIGH_THRESH    = 30      # reduce position sizes when VIX above this
VIX_EXTREME_THRESH = 45      # halt new buys when VIX above this

TRADES_FILE = Path("docs/trades.json")
PEAK_FILE   = Path("docs/peaks.json")

# ── Crypto config ─────────────────────────────────────────────────────────────
ENABLE_CRYPTO    = True
MAX_CRYPTO_POS   = 3        # max concurrent crypto positions
CRYPTO_MAX_PCT   = 0.07     # max 7% portfolio per crypto position
# Alpaca crypto symbols → yfinance equivalents
CRYPTO_UNIVERSE  = {
    "BTC/USD":  "BTC-USD",
    "ETH/USD":  "ETH-USD",
    "SOL/USD":  "SOL-USD",
    "AVAX/USD": "AVAX-USD",
    "DOGE/USD": "DOGE-USD",
    "XRP/USD":  "XRP-USD",
}
CRYPTO_STOP_PCT  = 0.10     # crypto is volatile: 10% stop
CRYPTO_TARGET_PCT= 0.25     # 25% profit target for crypto

# ── Runtime caches (live only — not persisted) ────────────────────────────────
_EARNINGS_CACHE: dict  = {}   # sym -> bool
_SPY_PERF_CACHE: dict  = {}   # cached SPY returns for relative strength calc

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


_ORDER_ENTRY_CACHE: dict = {}

def get_position_entry_times() -> dict:
    """
    Fetch recent filled BUY orders from Alpaca to determine when each position was opened.
    Returns {symbol: filled_at_isostring}. Cached per run.
    """
    global _ORDER_ENTRY_CACHE
    if _ORDER_ENTRY_CACHE:
        return _ORDER_ENTRY_CACHE
    try:
        orders = alpaca_get("/v2/orders?status=closed&limit=200&direction=desc")
        times: dict = {}
        for o in (orders or []):
            sym       = o.get("symbol", "")
            side      = o.get("side", "")
            filled_at = o.get("filled_at") or ""
            if sym and side == "buy" and filled_at and sym not in times:
                times[sym] = filled_at
        _ORDER_ENTRY_CACHE = times
        logger.debug(f"Order entry times loaded for {len(times)} symbols")
    except Exception as e:
        logger.debug(f"Order history fetch failed: {e}")
    return _ORDER_ENTRY_CACHE


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

def _stoch_rsi(closes, rsi_period=14, stoch_period=14):
    """Stochastic RSI — measures RSI's position within its recent range (0-100).
    Returns (k, d) where k < 20 = oversold bounce, k > 80 = overbought."""
    if len(closes) < rsi_period + stoch_period + 1:
        return 50.0, 50.0
    # Compute RSI series
    rsi_vals = []
    for i in range(stoch_period + 1):
        subset = closes[:-(stoch_period - i) if (stoch_period - i) > 0 else None]
        rsi_vals.append(_rsi(subset, rsi_period))
    if not rsi_vals:
        return 50.0, 50.0
    lo, hi = min(rsi_vals), max(rsi_vals)
    if hi == lo:
        return 50.0, 50.0
    k = (rsi_vals[-1] - lo) / (hi - lo) * 100
    d = sum(rsi_vals[-3:]) / min(3, len(rsi_vals)) if len(rsi_vals) >= 3 else k
    d = (d - lo) / (hi - lo) * 100
    return round(k, 1), round(d, 1)

def _roc(closes, period=5):
    """Rate of change: % move over `period` bars."""
    if len(closes) <= period:
        return 0.0
    prev = closes[-period - 1]
    if prev <= 0:
        return 0.0
    return round((closes[-1] - prev) / prev * 100, 2)

def _vwap(hourly):
    """Compute VWAP with ±2σ bands from today's hourly bars.
    Returns (vwap, position_pct, vwap_zscore, vwap_reclaim).
    vwap_zscore: price's z-score from VWAP (>2 = overbought band, <-2 = oversold band).
    vwap_reclaim: True if price dipped below VWAP intraday and has since reclaimed it."""
    if hourly is None:
        return None, 50.0, 0.0, False
    try:
        if "Volume" not in hourly.columns or "Close" not in hourly.columns:
            return None, 50.0, 0.0, False
        h = hourly.dropna(subset=["Close", "Volume"])
        if len(h) < 2:
            return None, 50.0, 0.0, False
        # Use last 8 bars (≈1 trading day in hourly)
        h = h.iloc[-8:]
        tp = (h["High"] + h["Low"] + h["Close"]) / 3
        cum_pv  = (tp * h["Volume"]).cumsum()
        cum_v   = h["Volume"].cumsum()
        vwap_v  = cum_pv / cum_v
        vwap    = float(vwap_v.iloc[-1])
        price   = float(h["Close"].iloc[-1])
        vwap_pos = (price - vwap) / vwap * 100 if vwap > 0 else 0

        # VWAP σ: volume-weighted std of typical price around VWAP
        cum_vol  = float(cum_v.iloc[-1])
        cum_var  = ((tp - vwap_v)**2 * h["Volume"]).cumsum()
        vwap_std = float((cum_var / cum_v).iloc[-1]) ** 0.5 if cum_vol > 0 else 0
        vwap_z   = (price - vwap) / vwap_std if vwap_std > 0 else 0

        # VWAP reclaim: price dipped below VWAP intraday then closed back above it
        vwap_reclaim = False
        if len(h) >= 4 and vwap > 0:
            tp_list   = list(tp)
            vwap_list = list(vwap_v)
            # Last bar must be above VWAP
            if tp_list[-1] > vwap_list[-1] * 1.001:
                # Any of the previous 1-4 bars must have been below VWAP
                for i in range(max(0, len(tp_list) - 5), len(tp_list) - 1):
                    if tp_list[i] < vwap_list[i] * 0.999:
                        vwap_reclaim = True
                        break

        return round(vwap, 2), round(vwap_pos, 2), round(vwap_z, 2), vwap_reclaim
    except Exception:
        return None, 50.0, 0.0, False


def _williams_r(closes, highs, lows, period=10):
    """Williams %R: -100 to 0; -80 to -100 = oversold (buy), 0 to -20 = overbought (sell)."""
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return -50.0
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    if hh == ll:
        return -50.0
    return round(-100 * (hh - closes[-1]) / (hh - ll), 1)


def _macd_slope(closes):
    """MACD histogram slope: positive = momentum building, negative = fading."""
    if len(closes) < 27:
        return 0.0
    e12_now  = _ema(closes,      12)
    e26_now  = _ema(closes,      26)
    e12_prev = _ema(closes[:-1], 12)
    e26_prev = _ema(closes[:-1], 26)
    if not all([e12_now, e26_now, e12_prev, e26_prev]):
        return 0.0
    return round((e12_now - e26_now) - (e12_prev - e26_prev), 4)


def _ttm_squeeze(closes, highs, lows, period=20):
    """TTM Squeeze: returns (in_squeeze, just_fired).
    just_fired = was squeezed (BB inside KC) last bar, now breaking out = high-conviction breakout."""
    if len(closes) < period + 2 or len(highs) < period + 1 or len(lows) < period + 1:
        return False, False
    try:
        def _squeeze_at(c, h, l):
            sma = sum(c[-period:]) / period
            std = (sum((x - sma)**2 for x in c[-period:]) / period) ** 0.5
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            atr = _atr(list(h[-(period+1):]), list(l[-(period+1):]), list(c[-(period+1):]), period)
            if not atr:
                return False
            kc_mid = _ema(list(c), period)
            if not kc_mid:
                return False
            kc_upper = kc_mid + 1.5 * atr
            kc_lower = kc_mid - 1.5 * atr
            return bb_upper < kc_upper and bb_lower > kc_lower

        sq_now  = _squeeze_at(closes,        highs,        lows)
        sq_prev = _squeeze_at(closes[:-1],   highs[:-1],   lows[:-1])
        return sq_now, (sq_prev and not sq_now)
    except Exception:
        return False, False


# ── Market regime detection ───────────────────────────────────────────────────
def market_regime():
    """
    Returns dict with:
      regime:    'bull' | 'neutral' | 'bear'
      vix:       current VIX level
      spy_trend: % above/below SPY 20-day EMA
      above_200: True if SPY is above its 200-day EMA
      score:     -3 (extreme bear) to +3 (strong bull)
    """
    try:
        # Fetch 250 days so we can compute the 200-day EMA
        spy = yf.download("SPY ^VIX", period="250d", interval="1d",
                          auto_adjust=True, progress=False)
        spy_closes = list(spy["Close"]["SPY"].dropna())
        vix_closes = list(spy["Close"]["^VIX"].dropna())

        vix = float(vix_closes[-1]) if vix_closes else 20.0

        score = 0
        spy_trend = 0.0
        above_200 = True

        if len(spy_closes) >= 20:
            ema20 = _ema(spy_closes, 20)
            spy_current = spy_closes[-1]
            if ema20:
                spy_trend = round((spy_current - ema20) / ema20 * 100, 2)
                if spy_trend > 2.0:   score += 2
                elif spy_trend > 0.5: score += 1
                elif spy_trend < -2.0: score -= 2
                elif spy_trend < -0.5: score -= 1

            # 5-day and 10-day momentum
            if len(spy_closes) >= 5:
                mom5 = (spy_closes[-1] - spy_closes[-5]) / spy_closes[-5] * 100
                if mom5 > 2.0:   score += 1
                elif mom5 < -2.0: score -= 1
            if len(spy_closes) >= 10:
                mom10 = (spy_closes[-1] - spy_closes[-10]) / spy_closes[-10] * 100
                if mom10 > 3.0:  score += 1
                elif mom10 < -3.0: score -= 1

            # 200-day EMA — the gold standard bull/bear dividing line
            if len(spy_closes) >= 200:
                ema200 = _ema(spy_closes, 200)
                if ema200:
                    above_200 = spy_current >= ema200
                    pos200 = (spy_current - ema200) / ema200 * 100
                    if pos200 > 5:    score += 1   # well above 200d = healthy bull
                    elif pos200 < -5: score -= 1   # below 200d = bear territory

        if vix > VIX_EXTREME_THRESH:  score -= 3   # extreme panic
        elif vix > VIX_HIGH_THRESH:   score -= 2
        elif vix > 22:                score -= 1
        elif vix < 16:                score += 1   # complacent = bull

        if score >= 2:    regime = "bull"
        elif score <= -2: regime = "bear"
        else:             regime = "neutral"

        logger.info(
            f"Market regime: {regime} | SPY trend: {spy_trend:+.1f}% | "
            f"VIX: {vix:.1f} | Above 200d: {above_200} | score: {score}"
        )
        return {"regime": regime, "vix": vix, "spy_trend": spy_trend,
                "score": score, "above_200": above_200}

    except Exception as e:
        logger.warning(f"Regime check failed: {e}")
        return {"regime": "neutral", "vix": 20.0, "spy_trend": 0.0,
                "score": 0, "above_200": True}


# ── Sector rotation engine ────────────────────────────────────────────────────
SECTOR_ETFS = {
    "tech":          "XLK",
    "finance":       "XLF",
    "health":        "XLV",
    "energy":        "XLE",
    "consumer":      "XLP",
    "consumer_tech": "XLY",
    "industrial":    "XLI",
    "crypto":        "IBIT",
}

def sector_rotation() -> dict:
    """
    Rank sectors by 1-day and 5-day ETF performance.
    Returns {sector: adj_score} where adj_score is -8 to +8.
    Hot sectors get a bonus; cold sectors get a penalty.
    """
    etfs = list(SECTOR_ETFS.values())
    try:
        kw  = dict(group_by="ticker", auto_adjust=True, progress=False)
        raw = yf.download(" ".join(etfs), period="10d", interval="1d", **kw)
        adj = {}
        for sec, etf in SECTOR_ETFS.items():
            try:
                if len(etfs) == 1:
                    closes = list(raw["Close"].dropna())
                else:
                    closes = list(raw["Close"][etf].dropna())
                if len(closes) < 2:
                    adj[sec] = 0
                    continue
                chg1d = (closes[-1] - closes[-2]) / closes[-2] * 100
                chg5d = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
                score = 0
                if   chg1d > 2:   score += 4
                elif chg1d > 0.5: score += 2
                elif chg1d < -2:  score -= 4
                elif chg1d < -0.5: score -= 2
                if   chg5d > 4:   score += 4
                elif chg5d > 1:   score += 2
                elif chg5d < -4:  score -= 4
                elif chg5d < -1:  score -= 2
                adj[sec] = max(-8, min(8, score))
            except Exception:
                adj[sec] = 0
        hot = sorted(adj.items(), key=lambda x: -x[1])[:3]
        logger.info(f"Sector rotation: {' | '.join(f'{s}:{v:+d}' for s,v in hot)}")
        return adj
    except Exception as e:
        logger.debug(f"Sector rotation error: {e}")
        return {}


# ── Pre-market gap scanner ────────────────────────────────────────────────────
def get_premarket_gaps(fractionable_set: set) -> list:
    """
    Detect stocks gapping up >3% or down >3% from prior close.
    Called once at market open. Returns [(sym, gap_pct, direction), ...].
    Uses 2-day 1h data to find pre-market vs prior close.
    """
    gaps = []
    # Check top movers from screeners for gaps
    screener_syms = get_market_movers()
    check_syms = list(set(screener_syms) & fractionable_set)[:50]
    if not check_syms:
        return gaps
    try:
        kw  = dict(group_by="ticker", auto_adjust=True, progress=False)
        raw = yf.download(" ".join(check_syms), period="2d", interval="1h", **kw)
        if raw.empty:
            return gaps
        for sym in check_syms:
            try:
                if len(check_syms) == 1:
                    closes = list(raw["Close"].dropna())
                else:
                    lvl = raw.columns.get_level_values(0)
                    if "Close" not in lvl or sym not in raw["Close"]:
                        continue
                    closes = list(raw["Close"][sym].dropna())
                if len(closes) < 8:
                    continue
                prev_close   = float(closes[-9]) if len(closes) >= 9 else float(closes[0])
                current      = float(closes[-1])
                gap_pct      = (current - prev_close) / prev_close * 100
                if abs(gap_pct) >= 3.0:
                    # Volume confirmation: today's volume should be elevated
                    try:
                        if len(check_syms) == 1:
                            vols = list(raw["Volume"].dropna())
                        else:
                            vols = list(raw["Volume"][sym].dropna()) if sym in raw["Volume"] else []
                        vol_ok = True
                        if len(vols) >= 4:
                            avg_prev = sum(vols[:-1]) / max(1, len(vols)-1)
                            vol_ok = vols[-1] >= avg_prev * 0.8  # at least 80% of prior avg
                    except Exception:
                        vol_ok = True
                    if vol_ok:
                        direction = "up" if gap_pct > 0 else "down"
                        gaps.append((sym, round(gap_pct, 2), direction))
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Gap scanner error: {e}")
    gaps.sort(key=lambda x: -abs(x[1]))
    if gaps:
        logger.info(f"Gap scan: {' | '.join(f'{s}:{g:+.1f}%' for s,g,_ in gaps[:5])}")
    return gaps


# ── Macro event calendar (FOMC, CPI, NFP) ────────────────────────────────────
# Dates when market moves wildly — reduce size, skip new buys same day
# FOMC decision dates 2025-2026 (announced months in advance by the Fed)
_MACRO_EVENTS = {
    "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}

def near_macro_event(days_before: int = 1) -> bool:
    """True if today or tomorrow is a major macro event (FOMC, etc.)."""
    today = datetime.now(timezone.utc).date()
    for d in range(days_before + 1):
        check = (today + timedelta(days=d)).isoformat()
        if check in _MACRO_EVENTS:
            logger.info(f"Macro event in {d}d ({check}) — cautious mode")
            return True
    return False


# ── Earnings calendar check ───────────────────────────────────────────────────
def has_earnings_soon(sym, days=3):
    """Returns True if this stock has earnings within `days` days — skip it. Cached."""
    if sym in _EARNINGS_CACHE:
        return _EARNINGS_CACHE[sym]
    result = False
    try:
        cal = yf.Ticker(sym).calendar
        if cal is not None and not cal.empty:
            now = datetime.now(timezone.utc).date()
            for col in cal.columns:
                if "earnings" in str(col).lower():
                    for val in cal[col]:
                        try:
                            ed = pd.Timestamp(val).date()
                            if 0 <= (ed - now).days <= days:
                                result = True
                        except Exception:
                            pass
    except Exception:
        pass
    _EARNINGS_CACHE[sym] = result
    return result


# ── Catalyst keyword detector (fast, no API) ─────────────────────────────────
_BULL_CATALYSTS = [
    # Earnings
    "earnings beat", "beats estimates", "record revenue", "raised guidance",
    "exceeds expectations", "blowout quarter", "record profit", "record earnings",
    "strong demand", "robust growth", "accelerating growth",
    # FDA / biotech
    "fda approval", "fda approved", "fda clears", "breakthrough therapy",
    "positive phase", "phase 3 success", "regulatory approval", "ema approval",
    # M&A
    "merger", "acquisition", "buyout", "takeover", "deal", "going private",
    "strategic acquisition", "bolt-on acquisition",
    # Partnerships / contracts
    "partnership", "contract win", "awarded contract", "major contract",
    "landmark deal", "government contract", "multi-year deal",
    # Capital returns
    "share buyback", "repurchase", "dividend increase", "special dividend",
    "stock split", "reverse split elimination",
    # Analyst actions
    "upgrade", "outperform", "buy rating", "price target raised",
    "strong buy", "overweight", "initiates coverage",
    # Technical
    "short squeeze", "all-time high", "52-week high", "breakout",
    "massive volume", "institutional buying", "insider buying",
    # AI / tech
    "ai partnership", "generative ai", "data center", "nvidia partnership",
    # General positive
    "record", "milestone", "launch", "expansion",
]
_BEAR_CATALYSTS = [
    # Earnings
    "misses estimates", "earnings miss", "revenue miss", "guidance cut",
    "lowers guidance", "below expectations", "disappoints", "weak results",
    # FDA / biotech
    "fda rejects", "clinical failure", "complete response letter",
    "fda hold", "trial failure", "negative data", "safety concern",
    # Legal / regulatory
    "lawsuit", "sec investigation", "fraud", "accounting irregularities",
    "criminal charges", "class action", "subpoena", "antitrust",
    # Analyst
    "downgrade", "underperform", "sell rating", "price target cut",
    "market perform", "reduces target",
    # Corporate
    "bankruptcy", "chapter 11", "layoffs", "restructuring", "ceo resigns",
    "coo departs", "delisted", "going concern", "liquidity concern",
    # Revenue
    "customer loss", "lost contract", "competition pressure",
]

def get_earnings_beat_candidates(candidates: list) -> set:
    """
    Find stocks that recently beat earnings estimates and are gapping up.
    These are high-probability continuation setups (earnings momentum).
    Returns set of ticker symbols.
    """
    beats = set()
    try:
        # Use yfinance screener for recent earnings beats
        res = yf.screen("earnings_beat")
        for q in (res.get("quotes") or [])[:20]:
            sym   = q.get("symbol", "")
            chg   = q.get("regularMarketChangePercent", 0) or 0
            # Only take stocks up 2%+ on earnings day (positive reaction)
            if sym and sym in candidates and chg >= 2.0:
                beats.add(sym)
    except Exception:
        pass
    if beats:
        logger.info(f"Earnings beat candidates: {', '.join(sorted(beats))}")
    return beats


def detect_catalyst(headlines: list) -> tuple[float, str]:
    """
    Fast keyword scan of headlines. Returns (boost, catalyst_label).
    boost is -15 to +15 additive score points.
    """
    text = " ".join(headlines).lower()
    bull_hits = [c for c in _BULL_CATALYSTS if c in text]
    bear_hits = [c for c in _BEAR_CATALYSTS if c in text]
    boost = min(15, len(bull_hits) * 6) - min(15, len(bear_hits) * 6)
    label = (bull_hits[0] if bull_hits else (bear_hits[0] if bear_hits else ""))
    return float(boost), label


# ── AI news sentiment ─────────────────────────────────────────────────────────
def ai_sentiment(ticker, use_sonnet=False, signals: dict = None):
    """
    Score news + technical sentiment -10 to +10 using Claude AI.
    use_sonnet=True for high-conviction candidates (better reasoning).
    signals: optional dict of technical indicators to include in prompt.
    Returns (score, catalyst_label).
    """
    if not ANTHROPIC_KEY:
        return 0, ""
    try:
        news_items = yf.Ticker(ticker).news[:10]
        headlines  = [n.get("title", "") for n in news_items if n.get("title")]

        # Fast keyword catalyst scan
        boost, catalyst = detect_catalyst(headlines) if headlines else (0, "")

        text  = "\n".join(headlines[:8]) if headlines else "(no recent news)"
        model = "claude-sonnet-4-6" if use_sonnet else "claude-haiku-4-5-20251001"

        # Build technical context string if signals provided
        tech_context = ""
        if signals:
            rsi       = signals.get("rsi", 50)
            roc5      = signals.get("roc5", 0)
            stoch_k   = signals.get("stoch_k", 50)
            ema50     = signals.get("price_vs_ema50", 0)
            vr        = signals.get("vol_ratio", 1)
            chg       = signals.get("change_pct", 0)
            w_r       = signals.get("williams_r", -50)
            rs5       = signals.get("rs5", 0)
            at_brk      = signals.get("at_breakout", False)
            consec      = signals.get("consec_green", 0)
            near_s      = signals.get("near_support", False)
            vol_dry     = signals.get("vol_dry_up", False)
            vwap_rcl    = signals.get("vwap_reclaim", False)
            nr7         = signals.get("nr7_signal", False)
            ttm         = signals.get("ttm_squeeze_fired", False)
            extras = []
            if at_brk:    extras.append("BREAKOUT above resistance")
            if vwap_rcl:  extras.append("VWAP reclaim (intraday dip bought)")
            if ttm:       extras.append("TTM squeeze breakout")
            if near_s:    extras.append("at support level")
            if vol_dry:   extras.append("volume dry-up (selling exhaustion)")
            if nr7:       extras.append("NR7 coiling — volatility expansion imminent")
            if consec >= 3: extras.append(f"{consec} consecutive green days")
            tech_context = (
                f"\nTechnical: RSI={rsi:.0f}, StochRSI_K={stoch_k:.0f}, "
                f"5d_ROC={roc5:+.1f}%, EMA50_pos={ema50:+.1f}%, "
                f"Vol_ratio={vr:.1f}x, Day_chg={chg:+.1f}%, W%R={w_r:.0f}, "
                f"RS5_vs_SPY={rs5:+.1f}%"
                + (f"\nSetup flags: {', '.join(extras)}" if extras else "")
            )

        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      model,
                "max_tokens": 140,
                "messages": [{
                    "role":    "user",
                    "content": (
                        f"You are an expert quantitative stock trader analyzing {ticker} for a 1-5 day swing trade.\n"
                        f"Rate the short-term outlook from -10 (very bearish) to +10 (very bullish).\n"
                        f"Focus on: catalysts (earnings/FDA/M&A), institutional interest, "
                        f"sector momentum, and technical confirmation.\n"
                        f"Headlines:{tech_context}\n{text}\n\n"
                        f"Return ONLY JSON: {{\"s\":<-10 to 10>,\"c\":\"<catalyst in 3 words>\"}}"
                    ),
                }],
            },
            timeout=12,
        )
        result   = json.loads(r.json()["content"][0]["text"].strip())
        ai_score = max(-10, min(10, float(result.get("s", 0))))
        cat_out  = result.get("c", catalyst) or catalyst
        logger.debug(f"AI {ticker}: score={ai_score:+.1f} catalyst='{cat_out}' model={'sonnet' if use_sonnet else 'haiku'}")
        return ai_score, cat_out
    except Exception as e:
        logger.debug(f"Sentiment error {ticker}: {e}")
        return 0, ""


# ── Holistic market AI call ───────────────────────────────────────────────────
_BREADTH_CACHE: dict = {}

def get_market_breadth() -> dict:
    """
    Estimate market breadth using a proxy basket of ETFs (advance/decline proxy).
    Returns {adv_pct: float, note: str}.
    """
    global _BREADTH_CACHE
    if _BREADTH_CACHE:
        return _BREADTH_CACHE
    try:
        # Use sector ETFs as breadth proxy: count how many are up today
        probe_syms = ["XLK","XLF","XLV","XLE","XLY","XLI","XLP","XLC","XLU","XLRE","XLB","XME","IWM","MDY"]
        raw = yf.download(" ".join(probe_syms), period="2d", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)
        adv = 0; total = 0
        for sym in probe_syms:
            try:
                closes = list(raw["Close"][sym].dropna()) if sym in raw["Close"] else []
                if len(closes) >= 2:
                    total += 1
                    if closes[-1] > closes[-2]: adv += 1
            except Exception:
                pass
        adv_pct = round(adv / max(1, total) * 100, 1)
        note = "broad advance" if adv_pct > 70 else "broad decline" if adv_pct < 30 else "mixed"
        _BREADTH_CACHE = {"adv_pct": adv_pct, "note": note, "adv": adv, "total": total}
        logger.info(f"Market breadth: {adv}/{total} sectors up ({adv_pct}%) — {note}")
        return _BREADTH_CACHE
    except Exception:
        return {"adv_pct": 50.0, "note": "unknown", "adv": 0, "total": 0}


def ai_market_context(regime, top_movers, sector_adjs: dict = None):
    """
    Ask Claude for a macro market read that adjusts our overall confidence.
    Returns an adjustment score -5 to +5.
    """
    if not ANTHROPIC_KEY:
        return 0
    try:
        movers_str = ", ".join(top_movers[:10])
        on_macro   = near_macro_event(1)
        sec_str    = ""
        if sector_adjs:
            hot  = sorted(sector_adjs.items(), key=lambda x: -x[1])[:3]
            cold = sorted(sector_adjs.items(), key=lambda x:  x[1])[:2]
            sec_str = (f"\n- Hot sectors: {', '.join(f'{s}({v:+d})' for s,v in hot)}"
                       f"\n- Cold sectors: {', '.join(f'{s}({v:+d})' for s,v in cold)}")
        breadth = get_market_breadth()
        prompt = (
            f"Automated US equity trader decision for today:\n"
            f"- Regime: {regime['regime']} (VIX={regime['vix']:.0f}, SPY 5d={regime['spy_trend']:+.1f}%)\n"
            f"- Market breadth: {breadth['adv_pct']}% sectors advancing ({breadth['note']})\n"
            f"- FOMC/macro event {'TODAY — be defensive' if on_macro else 'not imminent'}\n"
            f"- Top movers: {movers_str}{sec_str}\n\n"
            f"Should the bot be aggressive (+3 to +5 = higher scores unlock more buys) "
            f"or cautious (-3 to -5 = raise the bar, fewer buys) today?\n"
            f"Consider: VIX level, breadth, macro risk, sector leadership.\n"
            f"Return ONLY JSON: {{\"adj\":<-5 to 5>, \"note\":\"<8 words max>\"}}"
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
    """Pull live market movers from multiple yfinance screeners."""
    movers = []
    for name in ("day_gainers", "most_actives", "day_losers",
                 "growth_technology_stocks", "undervalued_growth_stocks"):
        try:
            res = yf.screen(name)
            for q in (res.get("quotes") or [])[:30]:
                s = q.get("symbol", "")
                if s and 1 < len(s) <= 5 and s.isalpha() and s.isupper():
                    movers.append(s)
        except Exception:
            pass
    return list(set(movers))


# ── Short squeeze + momentum surge detector ─────────────────────────────────
def get_squeeze_candidates(fractionable_set: set) -> set:
    """
    Find stocks with high short interest that are also gaining momentum.
    Short squeeze setups: price rising + high short float → explosive upside.
    Returns set of ticker symbols with squeeze potential.
    """
    candidates = set()
    try:
        # yfinance screener for high short interest / active
        for screen in ("most_actives", "day_gainers"):
            res = yf.screen(screen)
            for q in (res.get("quotes") or [])[:20]:
                sym = q.get("symbol", "")
                if not sym or sym not in fractionable_set:
                    continue
                short_float = q.get("shortPercentOfFloat", 0) or 0
                chg_pct     = q.get("regularMarketChangePercent", 0) or 0
                vol_ratio   = 1.0
                if q.get("averageDailyVolume3Month") and q.get("regularMarketVolume"):
                    vol_ratio = q.get("regularMarketVolume") / max(1, q.get("averageDailyVolume3Month"))
                # Squeeze criteria: >15% short float + rising + volume surge
                if short_float > 0.15 and chg_pct > 1.0 and vol_ratio > 1.5:
                    candidates.add(sym)
    except Exception as e:
        logger.debug(f"Squeeze scanner error: {e}")
    if candidates:
        logger.info(f"Squeeze candidates: {', '.join(sorted(candidates))}")
    return candidates


# ── Unusual options flow detector ────────────────────────────────────────────
def get_options_flow_candidates(tickers: list, max_check: int = 25) -> dict:
    """
    Detect unusual options activity via yfinance options chain.
    Returns {symbol: {'call_vol_ratio': float, 'put_call': float, 'bullish': bool}}

    Bullish flow: call volume >> put volume + high OI on near-term calls
    """
    result = {}
    checked = 0
    for sym in tickers[:max_check]:
        if checked >= max_check:
            break
        try:
            tk   = yf.Ticker(sym)
            exps = tk.options
            if not exps:
                continue
            # Use nearest expiry with enough time (7-30 days out)
            today   = datetime.now(timezone.utc).date()
            target  = None
            for exp in exps[:6]:
                try:
                    from datetime import date as _dt_date
                    exp_date = _dt_date.fromisoformat(exp)
                    days_out = (exp_date - today).days
                    if 5 <= days_out <= 35:
                        target = exp
                        break
                except Exception:
                    pass
            if not target:
                continue

            chain = tk.option_chain(target)
            calls = chain.calls
            puts  = chain.puts
            if calls.empty or puts.empty:
                continue

            call_vol = int(calls["volume"].fillna(0).sum())
            put_vol  = int(puts["volume"].fillna(0).sum())
            call_oi  = int(calls["openInterest"].fillna(0).sum())
            put_oi   = int(puts["openInterest"].fillna(0).sum())

            if call_vol + put_vol < 100:  # too little activity
                continue

            put_call = put_vol / max(1, call_vol)
            # Bullish: put/call < 0.5 AND calls >> puts (institutional accumulation)
            # Also bullish if OI heavily skewed to calls (dark pool positioning)
            oi_bull_skew = (call_oi > put_oi * 2 and call_oi > 1000)
            bullish = (put_call < 0.5 and call_vol > put_vol * 2) or oi_bull_skew
            bearish = (put_call > 2.0 and put_vol > call_vol * 2)

            if bullish or bearish:
                result[sym] = {
                    "call_vol":       call_vol,
                    "put_vol":        put_vol,
                    "call_oi":        call_oi,
                    "put_oi":         put_oi,
                    "put_call":       round(put_call, 2),
                    "bullish":        bullish,
                    "bearish":        bearish,
                }
            checked += 1
        except Exception:
            pass

    bullish_syms = [s for s, d in result.items() if d["bullish"]]
    bearish_syms = [s for s, d in result.items() if d["bearish"]]
    if bullish_syms:
        logger.info(f"Unusual call flow (bullish): {', '.join(bullish_syms)}")
    if bearish_syms:
        logger.info(f"Unusual put flow (bearish): {', '.join(bearish_syms)}")
    return result


# ── Crypto trading ────────────────────────────────────────────────────────────
_BTC_DOMINANCE_CACHE: dict = {}

def get_btc_dominance() -> float:
    """
    Estimate BTC dominance by comparing BTC market cap proxy vs total crypto.
    Uses BTC price * circulating supply heuristic. Returns 0-100 (%).
    """
    global _BTC_DOMINANCE_CACHE
    if _BTC_DOMINANCE_CACHE:
        return _BTC_DOMINANCE_CACHE.get("dom", 55.0)
    try:
        btc  = yf.download("BTC-USD",  period="1d", interval="1d", progress=False)
        eth  = yf.download("ETH-USD",  period="1d", interval="1d", progress=False)
        sol  = yf.download("SOL-USD",  period="1d", interval="1d", progress=False)
        btc_p = float(btc["Close"].iloc[-1]) if not btc.empty else 0
        eth_p = float(eth["Close"].iloc[-1]) if not eth.empty else 0
        sol_p = float(sol["Close"].iloc[-1]) if not sol.empty else 0
        # Rough market cap weights using circulating supply approximations
        BTC_SUPPLY = 19_700_000
        ETH_SUPPLY = 120_000_000
        SOL_SUPPLY = 460_000_000
        btc_mc  = btc_p * BTC_SUPPLY
        eth_mc  = eth_p * ETH_SUPPLY
        sol_mc  = sol_p * SOL_SUPPLY
        total   = btc_mc + eth_mc + sol_mc
        dom     = (btc_mc / total * 100) if total > 0 else 55.0
        _BTC_DOMINANCE_CACHE["dom"] = round(dom, 1)
        logger.info(f"BTC dominance estimate: {dom:.1f}%")
        return _BTC_DOMINANCE_CACHE["dom"]
    except Exception:
        return 55.0


def fetch_crypto_data() -> dict:
    """
    Fetch price + indicators for all CRYPTO_UNIVERSE coins via yfinance.
    Returns {alpaca_symbol: signal_dict}.
    """
    result = {}
    for alpaca_sym, yf_sym in CRYPTO_UNIVERSE.items():
        try:
            daily  = yf.download(yf_sym, period="15d", interval="1d",
                                 auto_adjust=True, progress=False)
            hourly = yf.download(yf_sym, period="3d",  interval="1h",
                                 auto_adjust=True, progress=False)
            sig = _extract(daily, hourly)
            if sig and sig["price"] > 0:
                result[alpaca_sym] = sig
        except Exception as e:
            logger.debug(f"Crypto data error {yf_sym}: {e}")
    logger.info(f"Crypto data: {list(result.keys())}")
    return result


def crypto_score(sig: dict) -> int:
    """
    Score a crypto asset 0-100. More weight on momentum + volume since
    crypto is purely sentiment/flow driven.
    """
    s      = 8
    chg    = sig.get("change_pct",  0) or 0
    intra  = sig.get("intraday",    0) or 0
    vr     = sig.get("vol_ratio",   1) or 1
    rsi    = sig.get("rsi",        50) or 50
    ema_c  = sig.get("ema_cross",   0) or 0
    macd   = sig.get("macd",        0) or 0
    bb     = sig.get("bb_pos",     50) or 50
    vwap   = sig.get("vwap_pos",    0) or 0
    roc5   = sig.get("roc5",        0) or 0
    stoch_k = sig.get("stoch_k",   50) or 50
    ema50  = sig.get("price_vs_ema50", 0) or 0
    w_r    = sig.get("williams_r", -50) or -50
    m_slp  = sig.get("macd_slope",   0) or 0
    ttm    = sig.get("ttm_squeeze_fired", False)

    if   chg > 5:   s += 28
    elif chg > 2:   s += 18
    elif chg > 0.5: s +=  8
    elif chg < -5:  s -= 22
    elif chg < -2:  s -= 12
    if   intra > 2:   s += 18
    elif intra > 0.8: s += 10
    elif intra < -2:  s -= 14
    if   vr > 2.5:  s += 18
    elif vr > 1.5:  s += 10
    if   50 < rsi < 75: s += 12
    elif rsi >= 75:     s +=  3
    elif rsi < 25:      s -= 10
    if   ema_c > 0.3:  s += 12
    elif ema_c < -0.3: s -= 10
    if   macd > 0.2:  s += 10
    elif macd < -0.2: s -= 8
    if   40 < bb < 80: s += 8
    if   vwap > 0.3:  s += 8
    elif vwap < -0.3: s -= 6
    # New momentum indicators
    if   roc5 >  8:  s += 14
    elif roc5 >  3:  s +=  8
    elif roc5 < -8:  s -= 12
    elif roc5 < -3:  s -=  6
    if   stoch_k < 20: s += 12  # oversold crypto = high-conviction bounce
    elif stoch_k > 85: s -=  8
    if   ema50 >  5:  s +=  8   # above 50-day EMA = uptrend
    elif ema50 < -5:  s -= 10
    if   w_r < -80:  s += 10    # oversold Williams %R
    elif w_r > -20:  s -=  6
    if   m_slp > 0:  s +=  8    # MACD slope rising
    elif m_slp < 0:  s -=  6
    if ttm:          s += 16    # TTM squeeze fired = breakout

    return max(0, min(100, int(s)))


def ai_crypto_sentiment(coin: str = "Bitcoin", signals: dict = None) -> float:
    """Ask Claude Haiku for crypto market sentiment (-10 to +10).
    Passes live technical signals if available for more accurate assessment."""
    if not ANTHROPIC_KEY:
        return 0
    try:
        # Fetch recent crypto headlines from yfinance
        yf_sym = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
                  "Solana": "SOL-USD", "Avalanche": "AVAX-USD"}.get(coin, f"{coin}-USD")
        news_items = yf.Ticker(yf_sym).news[:6]
        headlines  = " | ".join([n.get("title", "") for n in news_items if n.get("title")][:5])

        tech_ctx = ""
        if signals:
            roc5    = signals.get("roc5", 0)
            stoch_k = signals.get("stoch_k", 50)
            ema50   = signals.get("price_vs_ema50", 0)
            chg     = signals.get("change_pct", 0)
            vr      = signals.get("vol_ratio", 1)
            tech_ctx = (f"\nTech: 5d_ROC={roc5:+.1f}%, StochRSI={stoch_k:.0f}, "
                        f"EMA50_pos={ema50:+.1f}%, vol={vr:.1f}x, day={chg:+.1f}%")

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
                        f"Rate 24-48h trading outlook for {coin} from -10 to +10.{tech_ctx}\n"
                        f"News: {headlines or 'none'}\n"
                        f"Return ONLY JSON: {{\"s\":<number>}}"
                    ),
                }],
            },
            timeout=8,
        )
        result = json.loads(r.json()["content"][0]["text"].strip())
        return max(-10, min(10, float(result.get("s", 0))))
    except Exception:
        return 0


def run_crypto_trades(tlog: dict, peaks: dict, portfolio_val: float,
                      buying_power: float, made_trades_ref: list) -> float:
    """
    Manage all crypto positions — buys and sells.
    Operates 24/7 regardless of equity market hours.
    Returns updated buying_power.
    """
    if not ENABLE_CRYPTO:
        return buying_power

    now_utc = datetime.now(timezone.utc)
    crypto_data = fetch_crypto_data()
    if not crypto_data:
        return buying_power

    # Get current crypto positions from Alpaca
    try:
        all_pos   = alpaca_get("/v2/positions")
        held_crypto = {
            p["symbol"]: p for p in all_pos
            if "/" in p.get("symbol", "")
        }
    except Exception as e:
        logger.warning(f"Crypto positions fetch failed: {e}")
        return buying_power

    # ── Sell / manage open crypto positions ──────────────────────────────
    for sym, pos in list(held_crypto.items()):
        try:
            cost     = float(pos.get("avg_entry_price", 0))
            qty      = abs(float(pos.get("qty", 0)))
            sig      = crypto_data.get(sym, {})
            current  = sig.get("price") or float(pos.get("current_price", cost))
            if cost <= 0 or qty <= 0:
                continue
            pnl_pct = (current - cost) / cost * 100

            prev_peak  = peaks.get(sym, {}).get("peak", current) if isinstance(peaks.get(sym), dict) else current
            _half_out  = peaks.get(sym, {}).get("half_out", False) if isinstance(peaks.get(sym), dict) else False
            peak       = max(prev_peak, current)
            peaks[sym] = {
                "peak":     peak,
                "time":     peaks.get(sym, {}).get("time") if isinstance(peaks.get(sym), dict) else now_utc.isoformat(),
                "half_out": _half_out,
            }
            trail_drop = (current - peak) / peak * 100

            # Dynamic trailing stop: tightens as profit grows (crypto is more volatile)
            if   pnl_pct >= 20:  c_trail = 5.0
            elif pnl_pct >= 12:  c_trail = 8.0
            elif pnl_pct >=  5:  c_trail = 10.0
            else:                c_trail = 12.0   # give room early on

            # Partial profit at +15% (sell half) before full exit
            if pnl_pct >= 15 and not _half_out and qty > 0.001:
                half_qty = round(qty / 2, 8)
                logger.info(f"SELL_HALF {sym} — crypto partial at {pnl_pct:+.1f}%")
                try:
                    alpaca_post("/v2/orders", {
                        "symbol": sym, "qty": str(half_qty),
                        "side": "sell", "type": "market", "time_in_force": "gtc",
                    })
                    log_trade(tlog, "SELL_HALF", sym, current, half_qty,
                              pnl=pnl_pct, reason=f"crypto partial profit ({pnl_pct:+.1f}%)")
                    peaks[sym]["half_out"] = True
                    made_trades_ref.append(True)
                except Exception as e:
                    logger.warning(f"Crypto partial sell failed {sym}: {e}")
                continue

            reason = None
            if pnl_pct <= -(CRYPTO_STOP_PCT * 100):
                reason = f"crypto stop loss ({pnl_pct:+.1f}%)"
            elif pnl_pct >= (CRYPTO_TARGET_PCT * 100):
                # Check if momentum is still strong for extension to 35%
                c_macd = sig.get("macd_slope", 0) or 0
                c_roc5 = sig.get("roc5", 0) or 0
                if c_macd > 0 and c_roc5 > 3 and pnl_pct < 35:
                    logger.info(f"HOLD {sym} — crypto extending to 35% (macd={c_macd:.3f}, roc5={c_roc5:.1f}%)")
                else:
                    reason = f"crypto profit target ({pnl_pct:+.1f}%)"
            elif trail_drop <= -c_trail and pnl_pct > 0:
                reason = f"crypto trailing stop ({trail_drop:.1f}% / thr={c_trail:.0f}% from peak)"

            if reason:
                logger.info(f"SELL {sym} — {reason}")
                alpaca_post("/v2/orders", {
                    "symbol": sym, "qty": str(qty),
                    "side": "sell", "type": "market", "time_in_force": "gtc",
                })
                log_trade(tlog, "SELL", sym, current, qty, pnl=pnl_pct, reason=reason)
                made_trades_ref.append(True)
                peaks.pop(sym, None)
            else:
                logger.info(f"HOLD {sym} — {pnl_pct:+.1f}% | peak ${peak:,.0f} | trail {trail_drop:.1f}% | thr {c_trail:.0f}%")
        except Exception as e:
            logger.warning(f"Crypto sell error {sym}: {e}")

    # ── Buy crypto ──────────────────────────────────────────────────────
    open_slots = MAX_CRYPTO_POS - len(held_crypto)
    if open_slots <= 0:
        return buying_power

    # BTC dominance: high dom (>60%) = risk-off in crypto, prefer BTC; low dom (<50%) = altcoin season
    btc_dom = get_btc_dominance()
    logger.info(f"BTC dominance: {btc_dom:.1f}% — {'risk-off: prefer BTC' if btc_dom > 60 else 'altcoin season' if btc_dom < 50 else 'neutral'}")

    scored = []
    for alpaca_sym, sig in crypto_data.items():
        if alpaca_sym in held_crypto:
            continue
        sc = crypto_score(sig)
        # BTC dominance adjustments: boost BTC when dom > 60, boost alts when dom < 50
        is_btc = "BTC" in alpaca_sym
        if btc_dom > 60 and not is_btc:   sc = max(0, sc - 8)   # penalize alts in risk-off
        elif btc_dom < 50 and not is_btc: sc = min(100, sc + 5) # bonus for alts in alt season
        elif btc_dom > 60 and is_btc:     sc = min(100, sc + 5) # boost BTC in risk-off
        if sc >= 22:
            scored.append((alpaca_sym, sc, sig))
    scored.sort(key=lambda x: -x[1])

    for alpaca_sym, sc, sig in scored[:open_slots]:
        try:
            coin_name = alpaca_sym.split("/")[0]
            ai_score  = ai_crypto_sentiment(coin_name, signals=sig)
            if sc + ai_score < 20:
                logger.info(f"SKIP {alpaca_sym} — combined score too low (tech={sc} ai={ai_score:+.0f})")
                continue
            price    = sig["price"]
            notional = round(min(portfolio_val * CRYPTO_MAX_PCT, buying_power * 0.3), 2)
            if notional < 10:
                continue
            logger.info(f"BUY {alpaca_sym} — ${notional:.0f} @ ~${price:,.2f} | score {sc} | ai {ai_score:+.0f}")
            alpaca_post("/v2/orders", {
                "symbol":        alpaca_sym,
                "notional":      str(notional),
                "side":          "buy",
                "type":          "market",
                "time_in_force": "gtc",   # crypto uses GTC not DAY
            })
            log_trade(tlog, "BUY", alpaca_sym, price, notional, score=sc,
                      reason=f"crypto score={sc} ai={ai_score:+.0f}")
            peaks[alpaca_sym] = {"peak": price, "time": now_utc.isoformat(), "half_out": False}
            made_trades_ref.append(True)
            buying_power -= notional
        except Exception as e:
            logger.warning(f"Crypto buy failed {alpaca_sym}: {e}")

    return buying_power


def get_full_universe(held_symbols: set) -> tuple[list, set]:
    """
    Build a full-market scan universe from ALL Alpaca fractionable US equities,
    then intelligently filter to the top ~200 most promising candidates.

    Selection priority:
      1. All currently held positions (always included)
      2. Live market movers from yfinance screeners
      3. BASE_UNIVERSE anchor stocks (proven liquid names)
      4. High-volume active symbols from Alpaca's full equity list
         filtered by: price >= $5, avgVolume proxy via exchange rank

    Returns (candidates_list, shortable_set)
    """
    shortable = set()

    try:
        # Fetch ALL active fractionable US equities from Alpaca
        all_assets = alpaca_get("/v2/assets?status=active&asset_class=us_equity")
    except Exception as e:
        logger.warning(f"Alpaca asset fetch failed: {e} — using BASE_UNIVERSE")
        return list(set(BASE_UNIVERSE) | held_symbols), set()

    fractionable = {
        a["symbol"]: a for a in all_assets
        if a.get("tradable") and a.get("fractionable")
        and 1 < len(a.get("symbol", "")) <= 5
        and a.get("symbol", "").isalpha()
        and a.get("symbol", "").isupper()
    }
    for sym, a in fractionable.items():
        if a.get("shortable") and a.get("easy_to_borrow"):
            shortable.add(sym)

    logger.info(f"Alpaca fractionable universe: {len(fractionable)} stocks")

    # Layer 1: always include held positions
    universe = set(held_symbols)

    # Layer 2: live screener movers (highest priority — currently moving)
    movers = get_market_movers()
    universe.update(s for s in movers if s in fractionable)

    # Layer 3: BASE_UNIVERSE anchor stocks
    universe.update(s for s in BASE_UNIVERSE if s in fractionable)

    # Layer 4: fill up to MAX_SCAN_TICKERS with "exchange-quality" stocks
    # Use a simple proxy: prefer NYSE/NASDAQ-listed, symbol length <= 4
    # (shorter symbols tend to be more established companies)
    MAX_SCAN_TICKERS = 220
    if len(universe) < MAX_SCAN_TICKERS:
        # Prefer primary exchanges, shorter symbols
        extras = sorted(
            [s for s in fractionable if s not in universe],
            key=lambda s: (len(s), s)   # shorter = more established
        )
        for sym in extras:
            if len(universe) >= MAX_SCAN_TICKERS:
                break
            a = fractionable[sym]
            # Basic quality filter: prefer NYSE/NASDAQ listed
            if a.get("exchange") in ("NYSE", "NASDAQ", "CBOE", "ARCA"):
                universe.add(sym)

    candidates = list(universe)
    logger.info(f"Full scan universe: {len(candidates)} tickers (movers: {len(movers)}, held: {len(held_symbols)})")
    return candidates, shortable


# ── Batch data fetch ──────────────────────────────────────────────────────────
def _fetch_spy_perf() -> dict:
    """
    Fetch SPY's 1-day and 5-day return once per run.
    Stored in _SPY_PERF_CACHE so individual stocks can compute relative strength.
    """
    global _SPY_PERF_CACHE
    if _SPY_PERF_CACHE:
        return _SPY_PERF_CACHE
    try:
        spy = yf.download("SPY", period="15d", interval="1d",
                          auto_adjust=True, progress=False)
        closes = list(spy["Close"].dropna())
        if len(closes) >= 2:
            _SPY_PERF_CACHE["d1"]  = (closes[-1] - closes[-2]) / closes[-2] * 100
            _SPY_PERF_CACHE["d5"]  = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
            _SPY_PERF_CACHE["d10"] = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
    except Exception:
        _SPY_PERF_CACHE = {"d1": 0.0, "d5": 0.0, "d10": 0.0}
    return _SPY_PERF_CACHE


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
    rsi_val           = 50.0
    ema_cross         = 0.0
    macd_val          = 0.0
    bb_pos            = 50.0
    intraday          = 0.0
    vwap_pos          = 0.0
    vwap_z            = 0.0
    vwap_reclaim      = False
    williams_r        = -50.0
    macd_slope_val    = 0.0
    ttm_squeeze_fired = False

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

        if len(hc) >= 27:
            macd_slope_val = _macd_slope(hc)

        if len(hc) >= 20:
            bb_pos = _bollinger(hc)

        _, vwap_pos, vwap_z, vwap_reclaim = _vwap(h)

        if "High" in h.columns and "Low" in h.columns:
            hh = list(h["High"])
            hl = list(h["Low"])
            if len(hc) >= 10:
                williams_r = _williams_r(hc, hh, hl, period=10)
            if len(hc) >= 22:
                _, ttm_squeeze_fired = _ttm_squeeze(hc, hh, hl, period=20)

    # Daily trend alignment (weekly proxy using 15-day daily data)
    daily_trend = 0.0
    daily_rsi   = 50.0
    stoch_k     = 50.0
    stoch_d     = 50.0
    roc5        = 0.0
    price_vs_ema50 = 0.0
    try:
        dc = list(daily["Close"])
        if len(dc) >= 15:
            daily_rsi = _rsi(dc, period=min(14, len(dc)-1))
        if len(dc) >= 5:
            e5  = _ema(dc, 5)
            e10 = _ema(dc, min(10, len(dc)))
            if e10 and e10 > 0:
                daily_trend = (e5 - e10) / e10 * 100 if e5 else 0.0
            roc5 = _roc(dc, 5)
        if len(dc) >= 30:
            stoch_k, stoch_d = _stoch_rsi(dc, rsi_period=14, stoch_period=14)
        if len(dc) >= 50:
            e50 = _ema(dc, 50)
            if e50 and e50 > 0:
                price_vs_ema50 = (dc[-1] - e50) / e50 * 100
    except Exception:
        pass

    # Inside bar / NR7 narrow range volatility contraction
    # NR7: today's range is the narrowest in 7 days = coiling before a breakout
    # Inside bar: today's high < yesterday's high AND today's low > yesterday's low = consolidation
    nr7_signal  = False
    inside_bar  = False
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 7:
            highs_d = list(daily["High"])
            lows_d  = list(daily["Low"])
            ranges  = [highs_d[i] - lows_d[i] for i in range(len(highs_d))]
            today_range = ranges[-1]
            # NR7: today's range is the smallest in the last 7 days
            if len(ranges) >= 7 and today_range == min(ranges[-7:]) and today_range < (ranges[-8] if len(ranges) >= 8 else float('inf')):
                nr7_signal = True
            # Inside bar: today's entire range is within yesterday's range
            if (len(highs_d) >= 2 and len(lows_d) >= 2
                    and highs_d[-1] < highs_d[-2]
                    and lows_d[-1]  > lows_d[-2]):
                inside_bar = True
    except Exception:
        pass

    # Pivot-based support/resistance levels
    # If price is breaking above the most recent pivot high = resistance breakout (+bonus)
    # If price is below the most recent pivot low = in downtrend (score penalty applied via change_pct)
    at_breakout = False
    near_support = False
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 10:
            highs = list(daily["High"])
            lows  = list(daily["Low"])
            closes_pivot = list(daily["Close"])
            # Find most recent pivot high (local max in last 10 bars, excluding last 2)
            pivot_high = None
            for i in range(len(highs) - 3, max(0, len(highs) - 12), -1):
                if highs[i] >= highs[i-1] and highs[i] >= highs[i+1]:
                    pivot_high = highs[i]
                    break
            pivot_low = None
            for i in range(len(lows) - 3, max(0, len(lows) - 12), -1):
                if lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                    pivot_low = lows[i]
                    break
            if pivot_high and price > pivot_high * 1.003:  # broke above with 0.3% buffer
                at_breakout = True
            if pivot_low and price < pivot_low * 1.005:    # within 0.5% of support
                near_support = True
    except Exception:
        pass

    # RSI divergence signals
    rsi_divergence       = False   # bearish: price up, RSI down
    rsi_bull_divergence  = False   # bullish: price down, RSI up (hidden strength)
    try:
        dc = list(daily["Close"])
        if len(dc) >= 10:
            rsi_now  = _rsi(dc, 14)
            rsi_prev = _rsi(dc[:-3], 14)   # RSI 3 bars ago
            # Bearish divergence: price up 3+ bars but RSI down
            if dc[-1] > dc[-4] and rsi_now < rsi_prev - 3:
                rsi_divergence = True
            # Bullish divergence: price down 3+ bars but RSI up (hidden demand)
            elif dc[-1] < dc[-4] and rsi_now > rsi_prev + 3 and rsi_now < 45:
                rsi_bull_divergence = True
    except Exception:
        pass

    # Consecutive green candle count — institutional accumulation pattern
    consec_green = 0
    try:
        closes = list(daily["Close"])
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                consec_green += 1
            else:
                break
    except Exception:
        pass

    # Volume dry-up on pullback — bullish accumulation signal
    # When stock dips on lower-than-average volume: sellers exhausted, big money holding
    vol_dry_up = False
    try:
        if "Volume" in daily.columns and "Close" in daily.columns and len(daily) >= 5:
            vols   = list(daily["Volume"])
            closes2 = list(daily["Close"])
            avg_v  = sum(vols[-10:]) / max(1, len(vols[-10:]))
            # Last 2 days: price down OR flat, volume below 70% of average
            if (len(closes2) >= 3 and len(vols) >= 3
                    and closes2[-1] <= closes2[-2]   # price not rising today
                    and vols[-1] < avg_v * 0.70      # today volume < 70% avg
                    and vols[-2] < avg_v * 0.80):    # yesterday also low volume
                vol_dry_up = True
    except Exception:
        pass

    # Average volume (14-day) for minimum liquidity filter
    avg_vol_14 = 0
    try:
        if "Volume" in daily.columns and len(daily) >= 5:
            avg_vol_14 = int(daily["Volume"].tail(14).mean())
    except Exception:
        pass

    # Relative strength vs SPY (1-day and 5-day)
    spy  = _fetch_spy_perf()
    rs1  = round(chg_pct - spy.get("d1", 0), 2)   # outperformance vs SPY today
    rs5  = 0.0
    try:
        dc = list(daily["Close"])
        if len(dc) >= 5:
            ret5 = (dc[-1] - dc[-5]) / dc[-5] * 100
            rs5  = round(ret5 - spy.get("d5", 0), 2)
    except Exception:
        pass

    return {
        "price":           round(price, 2),
        "change_pct":      round(chg_pct, 2),
        "vol_ratio":       round(vol_ratio, 2),
        "avg_vol_14":      avg_vol_14,
        "week_high":       round(week_high, 2),
        "week_low":        round(week_low, 2),
        "near_52w_high":   round(near_52w_high, 4),
        "intraday":        round(intraday, 2),
        "rsi":             round(rsi_val, 1),
        "daily_rsi":       round(daily_rsi, 1),
        "daily_trend":     round(daily_trend, 3),
        "ema_cross":       round(ema_cross, 3),
        "macd":            round(macd_val, 3),
        "bb_pos":          round(bb_pos, 1),
        "vwap_pos":        round(vwap_pos, 2),
        "rs1":             rs1,
        "rs5":             rs5,
        "atr":             round(atr_val, 3) if atr_val else None,
        "stoch_k":            round(stoch_k, 1),
        "stoch_d":            round(stoch_d, 1),
        "roc5":               round(roc5, 2),
        "price_vs_ema50":     round(price_vs_ema50, 2),
        "williams_r":         round(williams_r, 1),
        "macd_slope":         round(macd_slope_val, 4),
        "ttm_squeeze_fired":  ttm_squeeze_fired,
        "vwap_z":             round(vwap_z, 2),
        "rsi_divergence":      rsi_divergence,
        "rsi_bull_divergence": rsi_bull_divergence,
        "consec_green":        consec_green,
        "vol_dry_up":          vol_dry_up,
        "at_breakout":         at_breakout,
        "near_support":        near_support,
        "nr7_signal":          nr7_signal,
        "inside_bar":          inside_bar,
        "vwap_reclaim":        vwap_reclaim,
    }


def _dl(tickers, period, interval):
    """Single yfinance batch download, returns per-ticker DataFrames."""
    result = {}
    if not tickers:
        return result
    kw  = dict(group_by="ticker", auto_adjust=True, progress=False, threads=True)
    raw = yf.download(" ".join(tickers), period=period, interval=interval, **kw)
    if raw is None or raw.empty:
        return result
    for tk in tickers:
        try:
            if len(tickers) == 1:
                result[tk] = raw
            else:
                lvl = raw.columns.get_level_values(0)
                if tk in lvl:
                    result[tk] = raw[tk]
        except Exception:
            pass
    return result


def _min_volume_ok(daily, min_avg_vol: int) -> bool:
    """True if stock's average daily volume meets minimum threshold."""
    if daily is None or "Volume" not in getattr(daily, "columns", []):
        return True  # can't check = allow through
    try:
        avg = float(daily["Volume"].mean())
        return avg >= min_avg_vol
    except Exception:
        return True


def _quick_score(daily_5d):
    """Fast momentum score from 5-day daily data. Used in Phase 1 pre-screen.
    Includes 1-day change, 5-day ROC, and volume ratio."""
    if daily_5d is None or len(daily_5d) < 2:
        return 0
    d = daily_5d.dropna(subset=["Close"])
    if len(d) < 2:
        return 0
    price = float(d["Close"].iloc[-1])
    prev  = float(d["Close"].iloc[-2])
    if price < 2 or prev <= 0:     # skip penny stocks
        return 0
    chg1d    = (price - prev) / prev * 100
    roc5     = (price - float(d["Close"].iloc[0])) / float(d["Close"].iloc[0]) * 100 if len(d) >= 5 else 0
    vol      = float(d["Volume"].iloc[-1]) if "Volume" in d else 0
    avg_vol  = float(d["Volume"].mean())   if "Volume" in d else vol
    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    s = 5
    # 1-day change signal
    if   chg1d >  3:  s += 20
    elif chg1d >  1:  s += 10
    elif chg1d >  0:  s +=  4
    elif chg1d < -3:  s -= 15
    # 5-day momentum (ROC5) — key pre-screen filter
    if   roc5 >  8:  s += 15
    elif roc5 >  3:  s +=  8
    elif roc5 >  0:  s +=  3
    elif roc5 < -8:  s -= 12
    elif roc5 < -3:  s -=  6
    # Volume confirmation
    if   vol_ratio > 2.5: s += 15
    elif vol_ratio > 1.5: s +=  8
    elif vol_ratio < 0.5: s -=  5
    return s


def fetch_batch(tickers, held_symbols=None, period_d="15d"):
    """
    Two-phase scan:
      Phase 1 — quick 1-day download for ALL tickers → rank by momentum × volume
      Phase 2 — full 15d daily + 3d hourly for top candidates only

    Always includes `held_symbols` in Phase 2 regardless of quick score.
    """
    if not tickers:
        return {}
    tickers = list(set(tickers))
    held    = set(held_symbols or [])
    result  = {}
    CHUNK    = 80     # larger chunks for speed
    FULL_CAP = 100   # max tickers for full technical analysis (increased from 80)
    MIN_AVG_VOL = 200_000   # minimum avg daily volume to avoid illiquid names

    # ── Phase 1: quick momentum pre-screen ─────────────────────────────
    quick_daily = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            quick_daily.update(_dl(chunk, "5d", "1d"))   # 5d gives us 5-day ROC
        except Exception as e:
            logger.warning(f"Quick scan chunk error: {e}")

    # Rank by quick score; filter out low-volume names
    quick_ranked = sorted(
        [(tk, _quick_score(quick_daily.get(tk))) for tk in tickers
         if _min_volume_ok(quick_daily.get(tk), MIN_AVG_VOL) or tk in held],
        key=lambda x: -x[1],
    )
    logger.info(
        f"Phase 1 pre-screen: {len(quick_daily)}/{len(tickers)} tickers. "
        f"Top 5: {' | '.join(f'{t}:{s}' for t,s in quick_ranked[:5])}"
    )

    # Phase 2 candidates: top FULL_CAP by quick score + all held positions
    phase2 = [tk for tk, _ in quick_ranked[:FULL_CAP]]
    for sym in held:
        if sym not in phase2:
            phase2.append(sym)

    # ── Phase 2: full technical analysis ─────────────────────────────
    CHUNK2 = 50
    for i in range(0, len(phase2), CHUNK2):
        chunk = phase2[i : i + CHUNK2]
        try:
            daily = _dl(chunk, period_d, "1d")
            hourly = _dl(chunk, "3d", "1h")
            for tk in chunk:
                try:
                    sig = _extract(daily.get(tk), hourly.get(tk))
                    if sig and sig["price"] > 0:
                        result[tk] = sig
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Full scan chunk error: {e}")

    logger.info(f"Data ready: {len(result)}/{len(phase2)} tickers (from {len(tickers)} universe)")
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
    macd       = d.get("macd",         0) or 0
    bb         = d.get("bb_pos",      50) or 50
    vwap       = d.get("vwap_pos",     0) or 0
    n52w       = d.get("near_52w_high", 1.0) or 1.0
    daily_rsi  = d.get("daily_rsi",   50) or 50
    daily_tr   = d.get("daily_trend",  0) or 0
    rs1        = d.get("rs1",          0) or 0   # relative strength vs SPY (1-day)
    rs5        = d.get("rs5",          0) or 0   # relative strength vs SPY (5-day)

    # Relative strength vs SPY (+14/-12)
    # Strong relative strength = institutional buying even when SPY flat
    if   rs5 > 5:   s += 14
    elif rs5 > 2:   s += 8
    elif rs5 > 0.5: s += 4
    elif rs5 < -5:  s -= 12
    elif rs5 < -2:  s -= 7

    if   rs1 > 3:   s += 8
    elif rs1 > 1:   s += 4
    elif rs1 < -3:  s -= 8
    elif rs1 < -1:  s -= 4

    # Multi-timeframe trend filter: daily EMA5/10 alignment (+8/-10)
    if   daily_tr > 0.5:  s +=  8
    elif daily_tr > 0.1:  s +=  4
    elif daily_tr < -0.5: s -= 10
    elif daily_tr < -0.1: s -=  5

    # Daily RSI confirmation (+8/-8)
    if   50 < daily_rsi < 70: s += 8
    elif daily_rsi >= 70:     s += 2
    elif daily_rsi < 30:      s -= 8
    elif daily_rsi < 40:      s -= 3

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

    # ── New high-conviction signals (research-backed) ─────────────────────────
    stoch_k   = d.get("stoch_k",           50) or 50
    stoch_d   = d.get("stoch_d",           50) or 50
    roc5      = d.get("roc5",               0) or 0
    ema50_pos = d.get("price_vs_ema50",     0) or 0
    w_r       = d.get("williams_r",       -50) or -50
    m_slope   = d.get("macd_slope",         0) or 0
    ttm_fired    = d.get("ttm_squeeze_fired", False)
    vwap_z_v     = d.get("vwap_z",             0) or 0
    rsi_div      = d.get("rsi_divergence",  False)

    # Stochastic RSI: oversold bounce (+14) or overbought (-8)
    if   stoch_k < 20 and stoch_d < 20: s += 14
    elif stoch_k > 80 and stoch_d > 80: s -=  8
    elif 30 < stoch_k < 70:             s +=  5

    # 5-day rate of change (+14/-12) — proven 66%+ win rate signal
    if   roc5 >  5:  s += 14
    elif roc5 >  2:  s +=  8
    elif roc5 >  0:  s +=  3
    elif roc5 < -5:  s -= 12
    elif roc5 < -2:  s -=  6

    # Price vs 50-day EMA — trend confirmation (+8/-10)
    if   ema50_pos >  3:  s +=  8
    elif ema50_pos >  1:  s +=  4
    elif ema50_pos < -3:  s -= 10
    elif ema50_pos < -1:  s -=  5

    # Williams %R: oversold = bounce setup (+12/-8), backtested >70% win rate
    if   w_r < -80:  s += 12
    elif w_r < -60:  s +=  5
    elif w_r > -20:  s -=  8

    # MACD histogram slope: rising = momentum building (+8/-6)
    if   m_slope > 0:  s +=  8
    elif m_slope < 0:  s -=  6

    # TTM Squeeze fired: BB just broke out of Keltner — high-conviction breakout (+16)
    if ttm_fired:  s += 16

    # VWAP z-score: price vs VWAP standard deviation bands
    if   vwap_z_v < -2.0:  s += 10  # deep oversold vs VWAP = bounce setup
    elif vwap_z_v < -1.0:  s +=  5
    elif vwap_z_v > 2.5:   s -=  8  # VWAP exhaustion zone
    elif vwap_z_v > 1.5:   s -=  3

    # RSI divergence: bearish when price up but RSI declining (-12 penalty)
    if rsi_div:  s -= 12

    # Bullish RSI divergence: hidden demand — price down but RSI rising (+10)
    rsi_bull = d.get("rsi_bull_divergence", False)
    if rsi_bull: s += 10

    # Consecutive green candles: institutional accumulation signature (+12/-8)
    # 3+ green days in a row = sustained buying pressure, not just a one-day pop
    consec = d.get("consec_green", 0) or 0
    if   consec >= 4: s += 12
    elif consec >= 3: s +=  8
    elif consec >= 2: s +=  4

    # Volume dry-up on pullback: selling exhaustion signal (+9)
    # Classic Wyckoff "test" pattern — big money not selling on the dip
    if d.get("vol_dry_up", False): s += 9

    # Pivot breakout: price just cleared a recent resistance level (+12)
    # High-probability setup — resistance becomes support
    if d.get("at_breakout", False): s += 12

    # Near support: buying at proven demand zone (+6)
    if d.get("near_support", False): s += 6

    # NR7 / inside bar: volatility contraction before expansion (+8/+6)
    # These patterns precede large directional moves — buy the squeeze
    if d.get("nr7_signal", False): s += 8
    if d.get("inside_bar", False): s += 6

    # VWAP reclaim: price dipped below VWAP intraday then reclaimed — institutions stepped in (+14)
    # One of the highest-conviction intraday signals; stops triggered below VWAP then buyers return
    if d.get("vwap_reclaim", False): s += 14

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

    stoch_k  = d.get("stoch_k",          50) or 50
    roc5     = d.get("roc5",              0) or 0
    ema50    = d.get("price_vs_ema50",    0) or 0
    w_r      = d.get("williams_r",      -50) or -50
    m_slope  = d.get("macd_slope",        0) or 0

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
    # New: bearish versions of new signals
    if   stoch_k > 80:  s += 12  # overbought = short candidate
    elif stoch_k < 20:  s -=  8  # oversold = don't short
    if   roc5 < -5:     s += 14  # downward momentum
    elif roc5 < -2:     s +=  8
    elif roc5 >  5:     s -= 12  # strong upward = don't short
    if   ema50 < -3:    s += 10  # below 50 EMA = bearish trend
    elif ema50 < -1:    s +=  5
    elif ema50 >  3:    s -= 10
    if   w_r > -20:     s += 10  # overbought = short signal
    elif w_r < -80:     s -=  8  # oversold = don't short
    if   m_slope < 0:   s +=  8  # MACD falling = bearish
    elif m_slope > 0:   s -=  6

    return max(0, min(100, int(s)))


# ── Position sizing ───────────────────────────────────────────────────────────
def calc_notional(portfolio_val, buying_power, price, atr, vix=20.0, macro_day=False,
                  score_val=0, win_rate=0.5, drawdown_pct=0.0):
    """
    ATR-based risk sizing with Kelly criterion scaling for high-conviction trades.
    Shrinks position when portfolio is in drawdown.
    """
    vix_scale = 1.0
    if vix > VIX_EXTREME_THRESH:   vix_scale = 0.4
    elif vix > VIX_HIGH_THRESH:    vix_scale = 0.65
    elif vix > 20:                 vix_scale = 0.85

    # Halve position size on FOMC/CPI day — expect extreme volatility
    if macro_day:
        vix_scale *= 0.5

    # Drawdown guard: shrink risk when portfolio is in significant drawdown
    if drawdown_pct > 10:    vix_scale *= 0.4   # -10%+ drawdown: very conservative
    elif drawdown_pct > 5:   vix_scale *= 0.65  # -5%+ drawdown: defensive
    elif drawdown_pct > 2:   vix_scale *= 0.85  # -2%+ drawdown: slightly cautious

    if atr and atr > 0 and price > 0:
        stop_dist   = 2 * atr
        dollar_risk = portfolio_val * RISK_PER_TRADE_PCT * vix_scale
        notional    = (dollar_risk / stop_dist) * price
    else:
        notional = portfolio_val * MAX_POSITION_PCT * vix_scale

    # Kelly criterion bonus for very high-conviction signals (score >= 75)
    if score_val >= 85 and win_rate > 0.55:
        kelly = win_rate - (1 - win_rate)   # simplified Kelly fraction
        kelly_scale = min(1.5, max(1.0, 1 + kelly * 0.5))
        notional = min(notional * kelly_scale, portfolio_val * MAX_POSITION_PCT * 1.5)
    elif score_val >= 75 and win_rate > 0.52:
        notional = min(notional * 1.25, portfolio_val * MAX_POSITION_PCT * 1.25)

    cap = min(portfolio_val * MAX_POSITION_PCT, buying_power * 0.95)
    return round(min(notional, cap), 2)


# ── Main trading engine ───────────────────────────────────────────────────────
def run():
    run_start = datetime.now(timezone.utc)

    def _elapsed():
        return (datetime.now(timezone.utc) - run_start).total_seconds()

    def _time_ok(budget_secs=380):
        return _elapsed() < budget_secs

    # Heartbeat: always write a minimal status to trades.json so the dashboard
    # shows the bot is alive even if an error occurs later in the run
    try:
        existing = _load(TRADES_FILE, {"trades": [], "positions": [], "last_updated": "",
                                        "portfolio_value": 0, "buying_power": 0})
        existing["last_updated"] = run_start.isoformat()
        existing.setdefault("status", "starting")
        _save(TRADES_FILE, existing)
    except Exception as _e:
        logger.warning(f"Heartbeat write failed: {_e}")

    if not ALPACA_KEY or not ALPACA_SECRET:
        logger.error("Alpaca keys missing — set ALPACA_KEY_ID + ALPACA_SECRET_KEY as GitHub Secrets.")
        # Write diagnostic to trades.json so dashboard shows the error
        diag = _load(TRADES_FILE, {})
        diag["last_updated"] = run_start.isoformat()
        diag["error"] = "ALPACA_KEY_ID or ALPACA_SECRET_KEY secrets not set in GitHub repository"
        _save(TRADES_FILE, diag)
        return  # let git commit step run

    # Market clock
    market_open = False
    next_close_str = ""
    try:
        clock = alpaca_get("/v2/clock")
        market_open    = bool(clock.get("is_open"))
        next_close_str = clock.get("next_close", "")
        if market_open:
            logger.info(f"Market OPEN — next close: {next_close_str}")
        else:
            logger.info(f"Market closed. Next open: {clock.get('next_open', '?')}")
    except Exception as e:
        logger.error(f"Alpaca unreachable: {e}")
        diag = _load(TRADES_FILE, {})
        diag["last_updated"] = run_start.isoformat()
        diag["error"] = f"Cannot reach Alpaca API: {e}"
        _save(TRADES_FILE, diag)
        return  # let git commit step run

    # Time-of-day flags (ET) — derived from next_close timestamp
    _now_et   = datetime.now(timezone.utc).astimezone()
    _et_hour  = _now_et.hour  # approximate; GitHub Actions uses UTC but clock gives wall time context
    try:
        from datetime import timezone as _tz
        import zoneinfo as _zi
        _et = _now_et.astimezone(_zi.ZoneInfo("America/New_York"))
        _et_hour, _et_min = _et.hour, _et.minute
    except Exception:
        _et_hour, _et_min = _now_et.hour - 4, _now_et.minute  # rough UTC-4
    _minutes_since_open = (_et_hour - 9) * 60 + (_et_min - 30) if market_open else 999
    _minutes_to_close   = (16 * 60) - (_et_hour * 60 + _et_min) if market_open else 999
    # Avoid new buys in first 10 min (wild open volatility) or last 20 min (end-of-day moves)
    _open_guard  = market_open and _minutes_since_open < 10
    _close_guard = market_open and _minutes_to_close < 20
    # Market close cleanup: last 8 min before close, liquidate losing positions > -3%
    _close_cleanup = market_open and _minutes_to_close < 8
    if _open_guard:
        logger.info(f"OPEN GUARD: {_minutes_since_open:.0f} min since open — skipping new buys")
    if _close_guard:
        logger.info(f"CLOSE GUARD: {_minutes_to_close:.0f} min to close — skipping new buys")
    if _close_cleanup:
        logger.info(f"CLOSE CLEANUP: {_minutes_to_close:.0f} min to close — will liquidate losers")

    # Account
    acct          = alpaca_get("/v2/account")
    portfolio_val = float(acct.get("portfolio_value", 0))
    buying_power  = float(acct.get("buying_power",   0))
    logger.info(f"Portfolio: ${portfolio_val:,.2f} | Cash: ${buying_power:,.2f}")

    # If market closed, only run crypto — skip equity pipeline
    if not market_open:
        if ENABLE_CRYPTO:
            tlog = _load(TRADES_FILE, {"trades": [], "positions": [], "last_updated": ""})
            peaks = _load(PEAK_FILE, {})
            made_ref = []
            buying_power = run_crypto_trades(
                tlog, peaks, portfolio_val, buying_power, made_ref
            )
            _save(PEAK_FILE, peaks)
            tlog["last_updated"]    = datetime.now(timezone.utc).isoformat()
            tlog["portfolio_value"] = portfolio_val
            tlog["buying_power"]    = round(buying_power, 2)
            _save(TRADES_FILE, tlog)
            logger.info(f"Off-hours crypto-only run complete.")
        return

    # Pre-cache SPY performance for relative strength calculations
    _fetch_spy_perf()

    # Market regime
    regime    = market_regime()
    vix       = regime["vix"]
    macro_day = near_macro_event(days_before=1)
    if vix > VIX_EXTREME_THRESH:
        logger.warning(f"VIX={vix:.0f} EXTREME — halting new buys, protecting capital.")

    # Portfolio drawdown guard — compute current drawdown from historical peak
    _prior_tlog  = _load(TRADES_FILE, {})
    _perf_hist   = _prior_tlog.get("perf_history", [])
    _hist_values = [h["v"] for h in _perf_hist if isinstance(h.get("v"), (int, float)) and h["v"] > 0]
    _peak_port   = max(_hist_values) if _hist_values else portfolio_val
    drawdown_pct = max(0.0, (_peak_port - portfolio_val) / _peak_port * 100) if _peak_port > 0 else 0.0
    if drawdown_pct > 2:
        logger.info(f"Portfolio drawdown: -{drawdown_pct:.1f}% from peak ${_peak_port:,.0f} — risk reduced")

    # Win rate from trade history for Kelly sizing
    _trade_stats = _prior_tlog.get("stats", {})
    _wins  = _trade_stats.get("wins",   0)
    _losses= _trade_stats.get("losses", 0)
    win_rate = _wins / max(1, _wins + _losses)

    # Positions + peaks
    positions = alpaca_get("/v2/positions")
    held      = {p["symbol"]: p for p in positions}
    # Separate longs and shorts
    longs  = {s: p for s, p in held.items() if float(p.get("qty", 0)) > 0}
    shorts = {s: p for s, p in held.items() if float(p.get("qty", 0)) < 0}
    peaks  = _load(PEAK_FILE, {})
    logger.info(f"Longs ({len(longs)}): {', '.join(longs) or 'none'}")
    logger.info(f"Shorts ({len(shorts)}): {', '.join(shorts) or 'none'}")

    # Build full-market scan universe (entire market, not just preset list)
    candidates, shortable = get_full_universe(set(held.keys()))
    logger.info(f"Scanning {len(candidates)} tickers | {len(shortable)} shortable")

    # Fetch live data (two-phase: quick pre-screen all, full analysis on top 80)
    live = fetch_batch(candidates, held_symbols=set(held.keys()))

    # Sector rotation (computed before AI context so it can be included in prompt)
    sector_adjs  = sector_rotation()   # {sector: -8..+8}

    # Market breadth (computed before AI context for richer prompt)
    breadth = get_market_breadth()

    # AI market context adjustment (use top movers from screeners)
    top_movers_for_ai = [s for s in candidates if s not in BASE_UNIVERSE][:12]
    regime_adj   = ai_market_context(regime, top_movers_for_ai, sector_adjs=sector_adjs)

    # Pre-market gap scan (bonus score for strong gap-up stocks)
    gap_ups = set()
    if _time_ok(260):
        gaps = get_premarket_gaps(set(candidates))
        gap_ups = {sym for sym, pct, direction in gaps if direction == "up" and pct >= 3}
        if gap_ups:
            logger.info(f"Gap-up candidates: {', '.join(sorted(gap_ups))}")

    # Mean reversion setups — deeply oversold stocks bouncing from support
    # High RSI divergence + Stoch RSI < 15 + EMA50 support = strong bounce setup
    mean_rev_cands = set()
    if _time_ok(255):
        for tk, sig in live.items():
            if tk in held:
                continue
            price_t  = sig.get("price", 0) or 0
            stoch_k_t = sig.get("stoch_k", 50) or 50
            rsi_t    = sig.get("daily_rsi", 50) or 50
            ema50_t  = sig.get("price_vs_ema50", 0) or 0
            rsi_bull_t = sig.get("rsi_bull_divergence", False)
            vol_dry_t  = sig.get("vol_dry_up", False)
            roc5_t   = sig.get("roc5", 0) or 0
            # Classic mean reversion: deeply oversold, bouncing on divergence, near EMA50 support
            if (stoch_k_t < 15 and rsi_t < 35 and ema50_t > -8
                    and (rsi_bull_t or vol_dry_t) and price_t >= 5
                    and roc5_t > -15):   # not in a full collapse
                mean_rev_cands.add(tk)
        if mean_rev_cands:
            logger.info(f"Mean-reversion bounce setups: {', '.join(sorted(mean_rev_cands))}")

    # Short squeeze detection — high short float + rising + volume surge → explosive upside
    squeeze_cands = get_squeeze_candidates(set(candidates)) if _time_ok(250) else set()

    # Unusual options flow — detect institutional call buying (bullish signal)
    options_flow: dict = {}
    if _time_ok(240):
        # Check top movers + any stocks with already high scores
        flow_check = list(gap_ups | squeeze_cands)[:15]
        flow_check += [s for s in candidates if s in BASE_UNIVERSE][:15]
        options_flow = get_options_flow_candidates(list(set(flow_check)), max_check=20)
    bullish_options = {s for s, d in options_flow.items() if d.get("bullish")}

    # Earnings beat plays — stocks that just beat estimates and are reacting positively
    earnings_beats = get_earnings_beat_candidates(set(candidates)) if _time_ok(230) else set()

    tlog        = _load(TRADES_FILE, {"trades": [], "positions": [], "last_updated": ""})
    made_trades = False
    now_utc     = datetime.now(timezone.utc)

    # ── MANAGE EXISTING SHORTS ─────────────────────────────────────────────
    for sym, pos in list(shorts.items()):
        try:
            cost    = float(pos.get("avg_entry_price", 0))
            qty     = abs(float(pos.get("qty", 0)))    # qty is negative for shorts
            current = (live.get(sym, {}).get("price")
                       or float(pos.get("current_price") or cost)
                       or cost)
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
    order_entry_times = get_position_entry_times()
    for sym, pos in list(longs.items()):
        try:
            cost    = float(pos.get("avg_entry_price", 0))
            qty     = float(pos.get("qty", 0))
            # Use live scan price first, then Alpaca's own current_price as backup
            current = (live.get(sym, {}).get("price")
                       or float(pos.get("current_price") or cost)
                       or cost)
            if cost <= 0 or qty <= 0:
                continue
            pnl_pct = (current - cost) / cost * 100

            # Trailing peak
            prev_peak  = peaks.get(sym, {}).get("peak", current) if isinstance(peaks.get(sym), dict) else peaks.get(sym, current)
            peak       = max(prev_peak, current)
            # Entry time: peaks.json → order history → now (for positions first seen this run)
            order_entry  = order_entry_times.get(sym, "")
            entry_time   = (peaks.get(sym, {}).get("time") if isinstance(peaks.get(sym), dict) else None) or order_entry or now_utc.isoformat()
            _ever_hit = (peaks.get(sym, {}).get("ever_hit_5pct", False) if isinstance(peaks.get(sym), dict) else False) or (pnl_pct >= 5)
            peaks[sym]   = {
                "peak":           peak,
                "time":           entry_time,
                "half_out":       peaks.get(sym, {}).get("half_out", False) if isinstance(peaks.get(sym), dict) else False,
                "ever_hit_5pct":  _ever_hit,
            }
            trail_drop = (current - peak) / peak * 100

            # Position age
            age_days = 0
            try:
                et       = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                age_days = (now_utc - et).total_seconds() / 86400
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

            # ── Profit lock at +8%: if momentum reverses with solid gain, sell 75% ──
            # This locks in most of the profit while keeping a runner going
            if (pnl_pct >= 8 and not half_out and qty >= 4):
                live_sig_lock = live.get(sym, {})
                macd_s = live_sig_lock.get("macd_slope", 0) or 0
                stoch_k_lock = live_sig_lock.get("stoch_k", 50) or 50
                rsi_lock = live_sig_lock.get("rsi", 50) or 50
                # MACD turning negative while overbought = momentum reversal
                if macd_s < -0.03 and stoch_k_lock > 75 and rsi_lock > 65:
                    lock_qty = round(qty * 0.75, 4)
                    logger.info(f"PROFIT_LOCK {sym} — selling 75% at {pnl_pct:+.1f}% (MACD slope={macd_s:.3f}, stoch={stoch_k_lock:.0f})")
                    try:
                        alpaca_post("/v2/orders", {
                            "symbol": sym, "qty": str(lock_qty),
                            "side": "sell", "type": "market", "time_in_force": "day",
                        })
                        log_trade(tlog, "SELL_HALF", sym, current, lock_qty,
                                  pnl=pnl_pct, reason=f"profit lock 75% — momentum reversal ({pnl_pct:+.1f}%)")
                        peaks[sym]["half_out"] = True
                        made_trades = True
                    except Exception as e:
                        logger.warning(f"Profit lock failed {sym}: {e}")
                    continue

            # ── Dynamic trailing stop: tightens as profit grows ──────────────
            # At +15%: trail tightens to 2%
            # At +10%: trail tightens to 3%
            # At +5%: use 5% trail
            # Below +5%: ATR-adaptive baseline (2.5× ATR, min 4%, max 9%)
            if   pnl_pct >= 15:  dyn_trail = 2.0
            elif pnl_pct >= 10:  dyn_trail = 3.0
            elif pnl_pct >=  5:  dyn_trail = TRAILING_STOP_PCT * 100
            else:
                # ATR-adaptive stop: gives volatile stocks more room, tighter on calm names
                sig_for_atr = live.get(sym, {})
                atr_v = sig_for_atr.get("atr") if sig_for_atr else None
                if atr_v and current > 0:
                    atr_pct = atr_v / current * 100
                    dyn_trail = max(4.0, min(9.0, atr_pct * 2.5))
                else:
                    dyn_trail = TRAILING_STOP_PCT * 100

            # ── Full exit conditions ──
            reason = None
            if pnl_pct <= -(STOP_LOSS_PCT * 100):
                reason = f"stop loss ({pnl_pct:+.1f}%)"
            elif pnl_pct >= (PROFIT_TARGET_PCT * 100):
                # Let strong momentum runners extend to 30% before selling
                live_sig_ext = live.get(sym, {})
                still_strong = (live_sig_ext.get("macd_slope", 0) or 0) > 0 and (live_sig_ext.get("roc5", 0) or 0) > 3
                if still_strong and pnl_pct < 30:
                    logger.info(f"HOLD {sym} — extending target to 30% (momentum strong, {pnl_pct:+.1f}%)")
                else:
                    reason = f"profit target ({pnl_pct:+.1f}%)"
            elif trail_drop <= -dyn_trail and pnl_pct > 0:
                reason = f"trailing stop ({trail_drop:.1f}% / thr={dyn_trail:.1f}% from peak ${peak:.2f})"
            elif pnl_pct < 2:
                # Adaptive hold time: strong momentum stocks get more time to work
                live_sig_age = live.get(sym, {})
                m_slope_age = live_sig_age.get("macd_slope", 0) or 0
                roc5_age    = live_sig_age.get("roc5", 0) or 0
                # Strong uptrend: extend hold to 8 days
                adaptive_max = MAX_HOLD_DAYS
                if m_slope_age > 0 and roc5_age > 2:
                    adaptive_max = 8
                elif m_slope_age < 0 and roc5_age < 0:
                    adaptive_max = 3  # weak momentum: exit sooner
                if age_days >= adaptive_max:
                    reason = f"stale position ({age_days:.0f}d, {pnl_pct:+.1f}%)"
            elif peaks.get(sym, {}).get("ever_hit_5pct") and pnl_pct <= 0.5:
                reason = f"breakeven lock ({pnl_pct:+.1f}%)"
            else:
                # Signal emergency exit: check live technical signal on held stock
                live_sig = live.get(sym, {})
                if live_sig:
                    live_sc  = score(sym, live_sig, regime_adj=regime_adj)
                    m_slope  = live_sig.get("macd_slope", 0) or 0
                    stoch_k  = live_sig.get("stoch_k",   50) or 50
                    w_r_val  = live_sig.get("williams_r", -50) or -50
                    vwap_z_live = live_sig.get("vwap_z", 0) or 0
                    rsi_div_live = live_sig.get("rsi_divergence", False)
                    if live_sc <= 8 and pnl_pct < 0:
                        reason = f"signal deteriorated (score={live_sc}, {pnl_pct:+.1f}%)"
                    elif live_sc <= 14 and pnl_pct < -3:
                        reason = f"weak signal + losing (score={live_sc}, {pnl_pct:+.1f}%)"
                    elif m_slope < -0.03 and stoch_k > 78 and w_r_val > -22 and pnl_pct > 3:
                        reason = f"momentum exhaustion (stoch={stoch_k:.0f}, W%R={w_r_val:.0f}, {pnl_pct:+.1f}%)"
                    elif vwap_z_live > 2.5 and pnl_pct > 2:
                        reason = f"VWAP exhaustion band (z={vwap_z_live:.1f}, {pnl_pct:+.1f}%)"
                    elif rsi_div_live and pnl_pct > 4:
                        # Bearish RSI divergence while in profit = early exit to lock gains
                        reason = f"RSI bearish divergence ({pnl_pct:+.1f}%)"

            # Market close cleanup: liquidate losing positions in last 8 min to avoid overnight risk
            if not reason and _close_cleanup and pnl_pct < -3:
                reason = f"close cleanup — avoid overnight loss ({pnl_pct:+.1f}%)"

            # Track ever-hit-5pct milestone for breakeven lock
            if pnl_pct >= 5 and sym in peaks:
                peaks[sym]["ever_hit_5pct"] = True

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

    # ── DCA: add to strong held positions on pullbacks ───────────────────
    if not _open_guard and not _close_guard and vix <= VIX_EXTREME_THRESH:
        for sym, pos in list(longs.items()):
            try:
                cost    = float(pos.get("avg_entry_price", 0))
                qty     = float(pos.get("qty", 0))
                if cost <= 0 or qty <= 0:
                    continue
                current = live.get(sym, {}).get("price", cost)
                pnl_pct = (current - cost) / cost * 100
                # Add to position if: small loss (-5% to -1.5%), strong signal, and position not at max
                mkt_val = current * qty
                if -5.0 <= pnl_pct <= -1.5 and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.8:
                    live_sig = live.get(sym, {})
                    if not live_sig:
                        continue
                    # Skip DCA if stock is in a clear downtrend (EMA50 falling + negative ROC)
                    ema50_pos = live_sig.get("price_vs_ema50", 0) or 0
                    roc5_val  = live_sig.get("roc5", 0) or 0
                    if ema50_pos < -3 and roc5_val < -5:
                        logger.debug(f"DCA SKIP {sym} — downtrend (EMA50={ema50_pos:.1f}%, ROC5={roc5_val:.1f}%)")
                        continue
                    dca_sc = score(sym, live_sig, regime_adj=regime_adj)
                    if dca_sc >= 28:  # high conviction only
                        dca_notional = min(
                            mkt_val * 0.5,                        # add up to 50% of current size
                            portfolio_val * MAX_POSITION_PCT - mkt_val,
                            buying_power * 0.15,
                        )
                        if dca_notional >= 50:
                            logger.info(f"DCA {sym} — adding ${dca_notional:.0f} (pnl={pnl_pct:+.1f}%, score={dca_sc})")
                            r = alpaca_post("/v2/orders", {
                                "symbol": sym, "notional": str(round(dca_notional, 2)),
                                "side": "buy", "type": "market", "time_in_force": "day",
                            })
                            if r:
                                buying_power -= dca_notional
                                log_trade(tlog, "DCA", sym, current, dca_notional, score=dca_sc, reason=f"dca pullback {pnl_pct:+.1f}%")
                                made_trades = True
            except Exception as e:
                logger.debug(f"DCA error {sym}: {e}")

    # ── BUY: long positions ───────────────────────────────────────────────
    open_long_slots = MAX_POSITIONS - len(longs)

    # Portfolio heat: compute total unrealized P&L across all open positions
    # Use this to subtly adjust position sizing (house money = can be slightly bolder)
    _total_unrealized_pnl = 0.0
    try:
        for pos in positions:
            cost_p  = float(pos.get("avg_entry_price", 0))
            qty_p   = abs(float(pos.get("qty", 0)))
            cur_p   = float(pos.get("current_price") or cost_p)
            if cost_p > 0 and qty_p > 0:
                _total_unrealized_pnl += (cur_p - cost_p) * qty_p
    except Exception:
        pass
    _portfolio_heat = _total_unrealized_pnl / portfolio_val * 100 if portfolio_val > 0 else 0
    if abs(_portfolio_heat) > 1:
        logger.info(f"Portfolio heat: {_portfolio_heat:+.1f}% (${_total_unrealized_pnl:+,.0f} unrealized)")

    # Consecutive loss guard: if last 3 closed trades are all losses, skip new buys this cycle
    _recent_trades = [t for t in tlog.get("trades", []) if t.get("action") in ("SELL", "COVER") and t.get("pnl_pct") is not None]
    _recent_trades.sort(key=lambda t: t.get("time", ""), reverse=True)
    _last3_pnl = [t["pnl_pct"] for t in _recent_trades[:3]]
    _consecutive_losses = len(_last3_pnl) >= 3 and all(p < 0 for p in _last3_pnl)
    if _consecutive_losses:
        logger.info(f"Consecutive loss guard: last 3 trades lost ({[round(p,1) for p in _last3_pnl]}) — skipping new buys this cycle")

    if open_long_slots > 0 and vix <= VIX_EXTREME_THRESH and not _open_guard and not _close_guard and not _consecutive_losses:
        # Sector counts for diversification
        sector_counts = {}
        for sym in longs:
            sec = SECTOR_MAP.get(sym, "other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        # Momentum re-entry: stocks recently sold for profit get a 3-day re-entry window
        # They get a +8 bonus score to reflect high-conviction setup
        recent_sells = set()
        cutoff = now_utc - timedelta(days=3)
        for t in tlog.get("trades", []):
            if t.get("action") in ("SELL", "SELL_HALF") and (t.get("pnl_pct") or 0) > 3:
                try:
                    if datetime.fromisoformat(t["time"].replace("Z", "+00:00")) > cutoff:
                        recent_sells.add(t.get("ticker", ""))
                except Exception:
                    pass

        # Technical pass — include sector rotation + gap + squeeze + earnings + mean-rev bonuses
        tech_scores = {
            tk: score(tk, live[tk],
                      regime_adj=regime_adj + sector_adjs.get(SECTOR_MAP.get(tk, "other"), 0)
                                + (10 if tk in gap_ups else 0)
                                + (12 if tk in squeeze_cands else 0)
                                + (8  if tk in recent_sells else 0)
                                + (14 if tk in bullish_options else 0)
                                + (18 if tk in earnings_beats else 0)
                                + (10 if tk in mean_rev_cands else 0))  # mean reversion bounce
            for tk in live if tk not in held
        }
        candidates_buy = sorted(
            [(tk, sc) for tk, sc in tech_scores.items() if sc >= MIN_BUY_SCORE - 5],
            key=lambda x: -x[1],
        )[:15]

        logger.info(
            f"Tech long candidates ({len(candidates_buy)}): "
            f"{' | '.join(f'{t}:{s}' for t,s in candidates_buy[:8])}"
        )
        logger.info(
            f"Buy state — slots:{open_long_slots} | drawdown:{drawdown_pct:.1f}% | "
            f"wr:{win_rate:.0%} | cons_losses:{_consecutive_losses} | "
            f"regime_adj:{regime_adj}"
        )

        # Earnings filter + sector filter + AI sentiment pass
        final_scores = []
        for tk, tech_sc in candidates_buy:
            sec = SECTOR_MAP.get(tk, "other")
            if sector_counts.get(sec, 0) >= MAX_SECTOR_LONGS:
                logger.info(f"SKIP {tk} — sector {sec} full ({sector_counts.get(sec,0)}/{MAX_SECTOR_LONGS})")
                continue
            if has_earnings_soon(tk):
                logger.info(f"SKIP {tk} — earnings within 3 days")
                continue
            # Minimum price filter: skip penny stocks (wide spreads, unreliable fills)
            _d_pre = live.get(tk, {})
            _price_pre = _d_pre.get("price", 0) or 0
            if _price_pre < 2.0:
                logger.debug(f"SKIP {tk} — price ${_price_pre:.2f} < $2 minimum")
                continue
            # Minimum average volume filter: 100k shares/day to ensure liquidity
            _avg_vol_pre = _d_pre.get("avg_vol_14", 0) or 0
            if 0 < _avg_vol_pre < 100_000:
                logger.debug(f"SKIP {tk} — avg volume {_avg_vol_pre:,} < 100k minimum")
                continue
            # Use Sonnet for top 3 candidates (better reasoning), Haiku for rest
            rank = len(final_scores)
            use_sonnet = (rank < 3) and _time_ok(200)
            if _time_ok(280):
                sent, catalyst = ai_sentiment(tk, use_sonnet=use_sonnet, signals=live.get(tk))
            else:
                sent, catalyst = 0, ""
            sec_adj       = sector_adjs.get(sec, 0)
            gap_adj       = 10 if tk in gap_ups else 0
            squeeze_adj   = 12 if tk in squeeze_cands else 0
            options_adj   = 14 if tk in bullish_options else 0
            reentry_adj   = 8  if tk in recent_sells else 0
            earnings_adj  = 18 if tk in earnings_beats else 0
            mean_rev_adj  = 10 if tk in mean_rev_cands else 0
            final_sc      = score(tk, live[tk], sentiment=sent,
                                  regime_adj=regime_adj + sec_adj + gap_adj + squeeze_adj
                                            + options_adj + reentry_adj + earnings_adj + mean_rev_adj)
            if final_sc >= MIN_BUY_SCORE:
                final_scores.append((tk, final_sc, sent, sec, catalyst))
                extras = []
                if gap_adj:       extras.append("gap")
                if squeeze_adj:   extras.append("squeeze")
                if options_adj:   extras.append("call-flow")
                if reentry_adj:   extras.append("re-entry")
                if earnings_adj:  extras.append("earnings-beat")
                if mean_rev_adj:  extras.append("mean-rev")
                logger.info(f"  {tk}: tech={tech_sc} sent={sent:+.1f} final={final_sc} sec={sec} cat='{catalyst}' [{','.join(extras) or 'base'}]")

        final_scores.sort(key=lambda x: -x[1])

        # Write diagnostics so dashboard can show why no trades happened
        tlog["last_scan_top"] = [
            {
                "ticker":      tk,
                "score":       sc,
                "sent":        round(sent, 1),
                "sector":      sec,
                "catalyst":    cat,
                "price":       round(live.get(tk, {}).get("price", 0), 2),
                "vol_ratio":   round(live.get(tk, {}).get("vol_ratio", 1), 2),
                "rsi":         round(live.get(tk, {}).get("daily_rsi", 50), 1),
                "chg_pct":     round(live.get(tk, {}).get("change_pct", 0), 2),
                "rs5":         round(live.get(tk, {}).get("rs5", 0), 2),
                "stop_price":  round(live.get(tk, {}).get("price", 0) * (1 - STOP_LOSS_PCT), 2),
                "at_breakout":  live.get(tk, {}).get("at_breakout", False),
                "vol_dry_up":   live.get(tk, {}).get("vol_dry_up", False),
                "consec_green": live.get(tk, {}).get("consec_green", 0),
                "ttm_squeeze":    live.get(tk, {}).get("ttm_squeeze_fired", False),
                "nr7":            live.get(tk, {}).get("nr7_signal", False),
                "inside_bar":     live.get(tk, {}).get("inside_bar", False),
                "vwap_reclaim":   live.get(tk, {}).get("vwap_reclaim", False),
            }
            for tk, sc, sent, sec, cat in (final_scores or [])[:8]
        ]
        tlog["last_scan_rejected"] = [
            {"ticker": tk, "score": sc}
            for tk, sc in candidates_buy[:5]
            if not any(tk == f[0] for f in final_scores)
        ]

        if not final_scores:
            logger.info(f"No longs passed threshold {MIN_BUY_SCORE}.")
            if candidates_buy:
                logger.info(f"  Top rejected: {' | '.join(f'{t}:{s}' for t,s in candidates_buy[:5])}")
        else:
            for tk, sc, sent, sec, catalyst in final_scores[:open_long_slots]:
                try:
                    d        = live[tk]
                    price    = d["price"]
                    atr      = d.get("atr")
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix,
                                             macro_day=macro_day, score_val=sc,
                                             win_rate=win_rate, drawdown_pct=drawdown_pct)
                    # Portfolio heat adjustment: if sitting on big unrealized gains ("house money"),
                    # allow slightly larger positions; if deeply underwater, shrink further
                    if _portfolio_heat > 5:
                        notional = min(notional * 1.1, portfolio_val * MAX_POSITION_PCT * 1.2)
                    elif _portfolio_heat < -5:
                        notional = notional * 0.8
                    # Size up further for strong catalysts or squeeze setups (on top of Kelly)
                    if catalyst and sent >= 5:
                        notional = min(notional * 1.4, portfolio_val * MAX_POSITION_PCT, buying_power * 0.4)
                    elif tk in squeeze_cands:
                        notional = min(notional * 1.2, portfolio_val * MAX_POSITION_PCT, buying_power * 0.35)
                    if notional < 1:
                        logger.info(f"SKIP {tk} — insufficient buying power")
                        continue
                    stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
                    logger.info(
                        f"BUY {tk} — ${notional:.0f} @ ~${price:.2f} "
                        f"| stop ${stop_price} | score {sc} | sent {sent:+.0f}"
                        + (f" | catalyst: {catalyst}" if catalyst else "")
                    )
                    alpaca_post("/v2/orders", {
                        "symbol":        tk,
                        "notional":      str(notional),
                        "side":          "buy",
                        "type":          "market",
                        "time_in_force": "day",
                    })
                    reason = f"score={sc} sent={sent:+.0f}"
                    if catalyst:
                        reason += f" [{catalyst}]"
                    log_trade(tlog, "BUY", tk, price, notional, score=sc, reason=reason)
                    peaks[tk] = {"peak": price, "time": now_utc.isoformat(), "half_out": False}
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                    made_trades  = True
                    buying_power -= notional
                except Exception as e:
                    logger.warning(f"BUY failed {tk}: {e}")

    # ── SHORT: bearish positions in bear/neutral regime ───────────────────
    if ENABLE_SHORTS and regime["regime"] in ("bear", "neutral") and not _open_guard and not _close_guard:
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
                    if price <= 0:
                        continue
                    atr      = d.get("atr")
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix,
                                             macro_day=macro_day, score_val=sc,
                                             win_rate=win_rate, drawdown_pct=drawdown_pct)
                    notional = round(notional * 0.6, 2)   # size shorts smaller
                    if notional < 1:
                        continue
                    # Short sells require qty (not notional)
                    qty = max(1, int(notional / price))
                    actual_notional = round(qty * price, 2)
                    logger.info(f"SHORT {tk} — {qty} shares @ ~${price:.2f} (${actual_notional:.0f}) | bear score {sc}")
                    alpaca_post("/v2/orders", {
                        "symbol":        tk,
                        "qty":           str(qty),
                        "side":          "sell",
                        "type":          "market",
                        "time_in_force": "day",
                    })
                    log_trade(tlog, "SHORT", tk, price, actual_notional, score=sc,
                              reason=f"bear score={sc} regime={regime['regime']}")
                    made_trades  = True
                    buying_power -= actual_notional
                except Exception as e:
                    logger.warning(f"SHORT failed {tk}: {e}")

    # ── Crypto trading (runs regardless of equity market hours) ──────────
    if ENABLE_CRYPTO and _time_ok(380):
        made_trades_ref = []
        buying_power = run_crypto_trades(
            tlog, peaks, portfolio_val, buying_power, made_trades_ref
        )
        if made_trades_ref:
            made_trades = True

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
                "stop_price": round(float(p.get("avg_entry_price", 0)) * (1 - STOP_LOSS_PCT), 2),
                "target_price": round(float(p.get("avg_entry_price", 0)) * (1 + PROFIT_TARGET_PCT), 2),
                "peak_price": peaks.get(p.get("symbol", ""), {}).get("peak", 0) if isinstance(peaks.get(p.get("symbol", "")), dict) else 0,
            }
            for p in curr
        ]
    except Exception as e:
        logger.warning(f"Position snapshot failed: {e}")

    # Compute profit factor, Sharpe-like ratio, and max drawdown from trade history
    _closed = [t for t in tlog.get("trades", []) if t.get("action") in ("SELL", "COVER") and t.get("pnl_pct") is not None]
    _gross_wins  = sum(t["pnl_pct"] for t in _closed if t["pnl_pct"] > 0) or 0
    _gross_losses= abs(sum(t["pnl_pct"] for t in _closed if t["pnl_pct"] < 0)) or 1
    _profit_factor = round(_gross_wins / _gross_losses, 2) if _closed else None
    _avg_win  = round(_gross_wins  / max(1, sum(1 for t in _closed if t["pnl_pct"] > 0)), 2) if _closed else None
    _avg_loss = round(_gross_losses/ max(1, sum(1 for t in _closed if t["pnl_pct"] < 0)), 2) if _closed else None

    # Sharpe-like ratio: avg trade return / std dev (measures consistency)
    _sharpe_ratio = None
    try:
        if len(_closed) >= 5:
            import statistics
            _pnls = [t["pnl_pct"] for t in _closed]
            _avg_pnl = statistics.mean(_pnls)
            _std_pnl = statistics.stdev(_pnls)
            if _std_pnl > 0:
                _sharpe_ratio = round(_avg_pnl / _std_pnl, 2)
    except Exception:
        pass

    # Max drawdown from portfolio history
    _max_dd = 0.0
    try:
        _hist_v = [h["v"] for h in tlog.get("perf_history", []) if isinstance(h.get("v"), (int, float)) and h["v"] > 0]
        if len(_hist_v) >= 2:
            _peak_running = _hist_v[0]
            for v in _hist_v:
                _peak_running = max(_peak_running, v)
                _dd = (_peak_running - v) / _peak_running * 100
                _max_dd = max(_max_dd, _dd)
    except Exception:
        pass

    tlog["last_updated"]    = now_utc.isoformat()
    tlog["portfolio_value"] = portfolio_val
    tlog["buying_power"]    = round(buying_power, 2)
    tlog["regime"]          = regime
    tlog["status"]          = "ok"
    tlog["macro_day"]       = macro_day
    tlog["open_positions"]  = len(tlog.get("positions", []))
    tlog["scan_universe"]   = len(candidates)
    tlog["drawdown_pct"]    = round(drawdown_pct, 2)
    tlog["win_rate"]        = round(win_rate, 3)
    tlog["portfolio_peak"]  = round(_peak_port, 2)
    tlog["market_breadth"]  = breadth
    tlog["profit_factor"]   = _profit_factor
    tlog["avg_win_pct"]     = _avg_win
    tlog["avg_loss_pct"]    = _avg_loss
    tlog["portfolio_heat"]  = round(_portfolio_heat, 2)
    tlog["sector_rotation"] = sector_adjs   # {sector: adj_score} for dashboard heatmap
    tlog["sharpe_ratio"]    = _sharpe_ratio
    tlog["max_drawdown"]    = round(_max_dd, 2)

    # Append to portfolio performance history (last 500 snapshots)
    snap = {
        "t": now_utc.isoformat(),
        "v": round(portfolio_val, 2),
        "c": round(buying_power, 2),
        "p": len(tlog.get("positions", [])),
    }
    history = tlog.setdefault("perf_history", [])
    history.append(snap)
    tlog["perf_history"] = history[-500:]

    _save(TRADES_FILE, tlog)
    logger.info(
        f"Cycle done. Trades: {'yes' if made_trades else 'none'}. "
        f"Regime: {regime['regime']}. Portfolio: ${portfolio_val:,.0f}. "
        f"Positions: {len(tlog.get('positions', []))}. "
        f"Log: {len(tlog['trades'])} entries."
    )


if __name__ == "__main__":
    try:
        run()
    except SystemExit:
        raise
    except Exception as _fatal:
        logger.exception(f"FATAL ERROR in run(): {_fatal}")
        try:
            diag = _load(TRADES_FILE, {})
            diag["last_updated"] = datetime.now(timezone.utc).isoformat()
            diag["error"] = f"Bot crashed: {_fatal}"
            _save(TRADES_FILE, diag)
        except Exception:
            pass
        sys.exit(1)
