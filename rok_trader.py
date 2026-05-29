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
ALPACA_BASE      = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE = "https://data.alpaca.markets"

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
_SIGNAL_WIN_RATES: dict = {}  # {signal_name: {win_rate: float, total: int}} loaded from tlog each cycle

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
    # Utilities
    "NEE":"utilities","DUK":"utilities","SO":"utilities","D":"utilities",
    "AEP":"utilities","EXC":"utilities","SRE":"utilities","PCG":"utilities",
    "ED":"utilities","WEC":"utilities","XEL":"utilities","ES":"utilities",
    # Materials
    "LIN":"materials","APD":"materials","SHW":"materials","FCX":"materials",
    "NEM":"materials","NUE":"materials","VMC":"materials","MLM":"materials",
    "ECL":"materials","ALB":"materials","CTVA":"materials","CF":"materials",
    # Real Estate
    "PLD":"real_estate","AMT":"real_estate","EQIX":"real_estate","CCI":"real_estate",
    "SPG":"real_estate","O":"real_estate","VICI":"real_estate","WY":"real_estate",
    "PSA":"real_estate","EQR":"real_estate","WELL":"real_estate","DLR":"real_estate",
    # Communication Services
    "GOOG":"comm_services","DIS":"comm_services","CMCSA":"comm_services",
    "T":"comm_services","VZ":"comm_services","TMUS":"comm_services",
    "CHTR":"comm_services","WBD":"comm_services","PARA":"comm_services",
    "EA":"comm_services","TTWO":"comm_services","OMC":"comm_services",
    # ETFs
    "SPY":"etf","QQQ":"etf","IWM":"etf","XLK":"etf","XLF":"etf",
    "XLE":"etf","XLV":"etf","XLRE":"etf","XLU":"etf","XLB":"etf",
    "XLC":"etf","XLY":"etf","XLI":"etf",
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
    "SPY","QQQ","IWM","XLK","XLF","XLE","XLV","XLRE","XLU","XLB","XLC",
    "SHOP","SQ","RBLX","HOOD","DKNG","ABNB","DASH","ROKU",
    "RIVN","SMCI","ARM","DELL","SNOW","DDOG","NET","CRWD","AXON",
    "ENPH","FSLR","CEG","VST","GEV",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","SRE","XEL",
    # Materials
    "LIN","APD","SHW","FCX","NEM","NUE","ECL","ALB",
    # Real Estate
    "PLD","AMT","EQIX","CCI","SPG","O","VICI","PSA","WELL","DLR",
    # Communication Services
    "GOOG","DIS","CMCSA","T","VZ","TMUS","CHTR","EA","TTWO",
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


def alpaca_snapshots(symbols: list) -> dict:
    """Fetch real-time Alpaca snapshots for a list of symbols.
    Returns dict of {symbol: {chg_pct, volume, prev_close, price, vol_ratio_est}}.
    Uses Alpaca Data API — much faster than yfinance for pre-screening.
    """
    if not symbols:
        return {}
    result = {}
    CHUNK = 500
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        try:
            sym_str = ",".join(chunk)
            r = requests.get(
                f"{ALPACA_DATA_BASE}/v2/stocks/snapshots",
                headers=_h(),
                params={"symbols": sym_str, "feed": "iex"},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            for sym, snap in data.items():
                try:
                    daily = snap.get("dailyBar", {}) or {}
                    prev  = snap.get("prevDailyBar", {}) or {}
                    trade = snap.get("latestTrade", {}) or {}
                    price = float(trade.get("p", 0) or daily.get("c", 0) or 0)
                    prev_c = float(prev.get("c", 0) or 0)
                    vol    = float(daily.get("v", 0) or 0)
                    prev_v = float(prev.get("v", 0) or 0)
                    if price <= 0 or prev_c <= 0:
                        continue
                    chg_pct = (price - prev_c) / prev_c * 100
                    # Estimate volume ratio: today's volume vs prev day (same time adjustment)
                    # Not perfect, but good enough for Phase 0 filter
                    vol_ratio_est = (vol / prev_v) if prev_v > 0 else 1.0
                    result[sym] = {
                        "price":         price,
                        "prev_close":    prev_c,
                        "chg_pct":       round(chg_pct, 3),
                        "volume":        vol,
                        "vol_ratio_est": round(vol_ratio_est, 2),
                    }
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Alpaca snapshot chunk failed: {e}")
    return result


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

def log_trade(tlog, action, sym, price, amount, score=None, pnl=None, reason=None, signals=None):
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

    # Store active signals at entry for performance tracking
    if action == "BUY" and signals:
        _SIGNAL_KEYS = [
            "cup_handle", "at_demand_zone", "mom_accel", "vcp", "obv_rising",
            "kc_breakout", "higher_lows", "double_bottom", "poc_breakout",
            "ema_stacked_bull", "trend_reversal", "bull_flag", "mtf_aligned",
            "ttm_squeeze_fired", "at_breakout", "vwap_reclaim", "gap_and_hold",
            "nr7_signal", "fib_support", "macd_bull_div", "rsi_bull_divergence",
            "ichimoku_above", "orb_breakout", "above_poc",
            "rvol_surge", "mfi_oversold", "mfi_bull_div", "supertrend_bull",
            "force_index_div", "force_index_rising", "ha_bull", "donchian_up",
        ]
        e["entry_signals"] = [k for k in _SIGNAL_KEYS if signals.get(k)]

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

    # Update signal performance stats when SELL closes a position
    if action in ("SELL", "SELL_HALF") and pnl is not None:
        # Find matching BUY for this ticker in recent trades
        for t in tlog.get("trades", []):
            if t.get("action") == "BUY" and t.get("ticker") == sym and t.get("entry_signals"):
                perf = tlog.setdefault("signal_performance", {})
                for sig in t["entry_signals"]:
                    sp = perf.setdefault(sig, {"wins": 0, "total": 0, "total_pnl": 0.0})
                    sp["total"] += 1
                    sp["total_pnl"] = round(sp.get("total_pnl", 0) + pnl, 2)
                    if pnl > 0:
                        sp["wins"] += 1
                break


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

def _vcp_pattern(highs, lows, closes, volumes=None, lookback: int = 30) -> bool:
    """
    Volatility Contraction Pattern (VCP) — Mark Minervini's signature setup.
    Stock forming tighter and tighter base with declining volume = spring loading.
    Detect 3 contracting price segments, each with lower range AND lower volume.
    Returns True if VCP structure detected (breakout imminent).
    """
    n = min(lookback, len(closes))
    if n < 12:
        return False
    try:
        c = closes[-n:]
        h = highs[-n:]
        l = lows[-n:]
        v = list(volumes[-n:]) if volumes and len(volumes) >= n else None

        seg = n // 3
        ranges = []
        avg_vols = []
        for i in range(3):
            start = i * seg
            end   = (i + 1) * seg
            r = max(h[start:end]) - min(l[start:end])
            ranges.append(r)
            if v:
                avg_vols.append(sum(v[start:end]) / max(1, end - start))

        # Check contracting ranges: each segment must be tighter than the last
        if not (ranges[0] > ranges[1] > ranges[2]):
            return False
        # Contraction threshold: last segment must be ≤60% of first segment
        if ranges[2] > ranges[0] * 0.6:
            return False
        # Volume contraction (if available): volume should be declining
        if avg_vols and len(avg_vols) == 3:
            if not (avg_vols[0] > avg_vols[1] and avg_vols[2] <= avg_vols[1] * 1.1):
                return False
        # Price near top of range (poised for breakout, not at the low)
        recent_range = max(h[-seg:]) - min(l[-seg:])
        if recent_range > 0:
            price_pos = (c[-1] - min(l[-seg:])) / recent_range
            if price_pos < 0.5:   # price at lower half of range = not ready
                return False
        return True
    except Exception:
        return False


def _obv_trend(closes, volumes, lookback: int = 20) -> dict:
    """
    On-Balance Volume (OBV): cumulative volume with direction.
    Rising OBV while price is flat/rising = accumulation (smart money buying).
    Returns {obv_rising: bool, obv_slope_pct: float}
    obv_slope_pct = % change in OBV over lookback period
    """
    n = len(closes)
    if n < 5 or not volumes or len(volumes) < 5:
        return {"obv_rising": False, "obv_slope_pct": 0.0}
    try:
        obv = 0.0
        obv_series = []
        for i in range(1, min(n, lookback + 5)):
            if closes[-(i)] > closes[-(i+1)]:
                obv += volumes[-(i)]
            elif closes[-(i)] < closes[-(i+1)]:
                obv -= volumes[-(i)]
            obv_series.append(obv)
        obv_series.reverse()
        if len(obv_series) < 2:
            return {"obv_rising": False, "obv_slope_pct": 0.0}
        early_obv = obv_series[0]
        recent_obv = obv_series[-1]
        slope_pct = 0.0
        if abs(early_obv) > 0:
            slope_pct = (recent_obv - early_obv) / abs(early_obv) * 100
        obv_rising = recent_obv > early_obv * 1.02  # OBV up ≥2% = accumulation
        return {"obv_rising": obv_rising, "obv_slope_pct": round(slope_pct, 1)}
    except Exception:
        return {"obv_rising": False, "obv_slope_pct": 0.0}


def _keltner_channel(highs, lows, closes, ema_period=20, atr_period=14, mult=2.0):
    """
    Keltner Channel: EMA ± mult*ATR.
    Returns (position_pct, above_upper, below_lower) where position_pct is 0-100
    (0 = at lower band, 50 = at midline, 100 = at upper band, >100 = above upper).
    Breakout above upper KC = strong momentum; below lower KC = oversold.
    """
    n = len(closes)
    if n < max(ema_period, atr_period) + 2:
        return 50.0, False, False
    try:
        midline = _ema(closes, ema_period)
        atr_val = _atr(highs, lows, closes, period=atr_period)
        if not midline or not atr_val or atr_val <= 0:
            return 50.0, False, False
        upper = midline + mult * atr_val
        lower = midline - mult * atr_val
        cur = closes[-1]
        if upper == lower:
            return 50.0, False, False
        pos = round((cur - lower) / (upper - lower) * 100, 1)
        return pos, cur > upper, cur < lower
    except Exception:
        return 50.0, False, False


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


def _force_index(closes, volumes, period=13):
    """Elder's Force Index: price-change × volume, smoothed with EMA.
    Positive = buying force (bulls in control). Negative = selling force.
    Divergence between FI and price = hidden institutional activity.
    Returns (fi_value, fi_rising, fi_bull_div) where fi_bull_div = price down but FI rising.
    """
    if len(closes) < period + 2 or len(volumes) < period + 2:
        return 0.0, False, False
    try:
        raw = [(closes[i] - closes[i-1]) * volumes[i] for i in range(1, len(closes))]
        fi_ema = _ema(raw, period)
        fi_prev = _ema(raw[:-3], period) if len(raw) > period + 3 else fi_ema
        fi_rising  = fi_ema > fi_prev
        price_down = closes[-1] < closes[-10] if len(closes) >= 10 else False
        fi_bull_div = price_down and fi_rising and fi_ema > 0
        return round(fi_ema, 0), fi_rising, fi_bull_div
    except Exception:
        return 0.0, False, False


def _beta_regression(stock_closes, spy_closes, period=63):
    """True beta via linear regression: covariance(r_stock, r_spy) / variance(r_spy).
    period=63 = 3-month beta (institutional standard).
    Returns (beta, alpha_annualized) — alpha > 0 means stock outperforms on risk-adjusted basis.
    """
    if len(stock_closes) < period + 2 or len(spy_closes) < period + 2:
        return 1.0, 0.0
    try:
        import numpy as np
        n = min(len(stock_closes), len(spy_closes), period + 1)
        sc = np.array(stock_closes[-n:], dtype=float)
        sp = np.array(spy_closes[-n:],  dtype=float)
        r_stock = np.diff(sc) / sc[:-1]
        r_spy   = np.diff(sp) / sp[:-1]
        cov = np.cov(r_stock, r_spy)
        if cov[1, 1] < 1e-10:
            return 1.0, 0.0
        beta = float(cov[0, 1] / cov[1, 1])
        beta = round(max(0.1, min(3.5, beta)), 2)
        # Jensen's alpha: avg_stock_return - beta * avg_spy_return, annualized
        alpha = (r_stock.mean() - beta * r_spy.mean()) * 252
        alpha = round(alpha * 100, 2)  # as percent
        return beta, alpha
    except Exception:
        return 1.0, 0.0


def _heikin_ashi_trend(opens, highs, lows, closes, lookback=5):
    """Heikin-Ashi candles remove noise and show pure trend.
    HA-Close = (O+H+L+C)/4. HA-Open = (prev_HA_O + prev_HA_C) / 2.
    Returns: (bullish_trend, bearish_trend, consecutive_bull, consecutive_bear)
    - consecutive_bull >= 3 with no lower shadows = strong trend
    """
    if len(closes) < lookback + 2:
        return False, False, 0, 0
    try:
        ha_open   = [(opens[0] + closes[0]) / 2]
        ha_close  = [(opens[0] + highs[0] + lows[0] + closes[0]) / 4]
        for i in range(1, len(closes)):
            ha_c = (opens[i] + highs[i] + lows[i] + closes[i]) / 4
            ha_o = (ha_open[-1] + ha_close[-1]) / 2
            ha_open.append(ha_o)
            ha_close.append(ha_c)
        # Count consecutive HA bull/bear candles
        c_bull = c_bear = 0
        for i in range(-1, -lookback-1, -1):
            if ha_close[i] > ha_open[i]:
                if c_bear > 0: break
                c_bull += 1
            elif ha_close[i] < ha_open[i]:
                if c_bull > 0: break
                c_bear += 1
            else:
                break
        bullish = c_bull >= 3
        bearish = c_bear >= 3
        return bullish, bearish, c_bull, c_bear
    except Exception:
        return False, False, 0, 0


def _donchian_breakout(closes, highs, lows, period=20):
    """Donchian Channel: N-period high/low breakout — used in Turtle Trading.
    Price above 20-day high = bullish breakout. Below 20-day low = bearish.
    Returns (breakout_up, breakout_down, channel_pct_position)
    channel_pct_position: 0=at low, 100=at high, 50=middle of channel.
    """
    if len(closes) < period + 2:
        return False, False, 50.0
    try:
        ch_high = max(highs[-period-1:-1])  # exclude today to check if today breaks it
        ch_low  = min(lows[-period-1:-1])
        price   = closes[-1]
        ch_range = ch_high - ch_low
        pos = (price - ch_low) / ch_range * 100 if ch_range > 0 else 50.0
        breakout_up   = price > ch_high * 1.001   # 0.1% buffer to avoid false breakouts
        breakout_down = price < ch_low  * 0.999
        return breakout_up, breakout_down, round(pos, 1)
    except Exception:
        return False, False, 50.0


def _candlestick_patterns(opens, highs, lows, closes, lookback=3):
    """Detect key institutional candlestick patterns.
    Returns dict of booleans: hammer, bullish_engulfing, morning_star, shooting_star,
    bearish_engulfing, doji, three_white_soldiers, three_black_crows.
    """
    result = {
        "hammer": False, "bullish_engulfing": False, "morning_star": False,
        "shooting_star": False, "bearish_engulfing": False, "doji": False,
        "three_white_soldiers": False, "three_black_crows": False,
    }
    if len(closes) < 3:
        return result
    try:
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        po, ph, pl, pc = opens[-2], highs[-2], lows[-2], closes[-2]
        ppo = opens[-3] if len(opens) >= 3 else po
        ppc = closes[-3] if len(closes) >= 3 else pc
        body  = abs(c - o)
        range_ = h - l if h != l else 0.001
        pbody = abs(pc - po)
        is_bull   = c > o
        is_bear   = c < o
        is_pbull  = pc > po
        is_pbear  = pc < po
        # Hammer: small body at top, long lower wick (≥2×body), tiny upper wick
        lower_wick = o - l if is_bull else c - l
        upper_wick = h - c if is_bull else h - o
        if body > 0 and lower_wick >= 2 * body and upper_wick <= 0.3 * body:
            result["hammer"] = True
        # Shooting Star: small body at bottom, long upper wick — bearish reversal
        if body > 0 and upper_wick >= 2 * body and lower_wick <= 0.3 * body and is_bear:
            result["shooting_star"] = True
        # Doji: open ≈ close (within 0.1% of range)
        if range_ > 0 and body / range_ < 0.1:
            result["doji"] = True
        # Bullish Engulfing: current bull candle engulfs prior bear candle
        if is_bull and is_pbear and o < pc and c > po and body > pbody * 0.9:
            result["bullish_engulfing"] = True
        # Bearish Engulfing: current bear engulfs prior bull
        if is_bear and is_pbull and o > pc and c < po and body > pbody * 0.9:
            result["bearish_engulfing"] = True
        # Morning Star: bearish→doji/small→bullish 3-candle reversal
        pdoji = abs(pc - po) < abs(ph - pl) * 0.15
        if is_pbear and pdoji and is_bull and c > (po + pc) / 2:
            result["morning_star"] = True
        # Three White Soldiers: 3 consecutive bull candles, each closing higher
        if (len(closes) >= 3 and closes[-3] < closes[-2] < closes[-1]
                and opens[-3] < closes[-3] and opens[-2] < closes[-2]):
            result["three_white_soldiers"] = True
        # Three Black Crows: 3 consecutive bear candles
        if (len(closes) >= 3 and closes[-3] > closes[-2] > closes[-1]
                and opens[-3] > closes[-3] and opens[-2] > closes[-2]):
            result["three_black_crows"] = True
    except Exception:
        pass
    return result


def _pivot_points(high, low, close):
    """Standard pivot points for support/resistance levels.
    Returns dict: pivot, r1, r2, s1, s2 (price levels).
    These are used by institutional traders as key intraday targets.
    """
    try:
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        return {
            "pivot": round(pp, 2),
            "r1": round(r1, 2), "r2": round(r2, 2),
            "s1": round(s1, 2), "s2": round(s2, 2),
        }
    except Exception:
        return {}


def _mfi(closes, highs, lows, volumes, period=14):
    """Money Flow Index: volume-weighted RSI (0-100).
    MFI < 20 = oversold (institutional accumulation). MFI > 80 = overbought.
    Divergence: price rising but MFI falling = distribution (smart money exiting).
    """
    if len(closes) < period + 2:
        return 50.0
    try:
        pos_flow = neg_flow = 0.0
        for i in range(-period, 0):
            tp     = (highs[i] + lows[i] + closes[i]) / 3
            tp_prv = (highs[i-1] + lows[i-1] + closes[i-1]) / 3
            mf     = tp * volumes[i]
            if tp > tp_prv:
                pos_flow += mf
            elif tp < tp_prv:
                neg_flow += mf
        if neg_flow == 0:
            return 100.0
        mfr = pos_flow / neg_flow
        return round(100 - 100 / (1 + mfr), 1)
    except Exception:
        return 50.0


def _supertrend(closes, highs, lows, period=10, multiplier=3.0):
    """Supertrend: ATR-based trailing stop. Returns (direction, stop_level).
    direction=1 means bullish (price above supertrend), -1 bearish.
    Widely used by institutional traders as a dynamic trend filter.
    """
    if len(closes) < period + 2:
        return 1, 0.0
    try:
        import numpy as np
        h = np.array(highs[-period*3:], dtype=float)
        l = np.array(lows[-period*3:],  dtype=float)
        c = np.array(closes[-period*3:],dtype=float)
        n = len(c)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = np.zeros(len(tr))
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        # Upper/lower basic bands
        hl2   = (h[1:] + l[1:]) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        # Final supertrend (follow-through logic)
        sup = np.zeros(len(c)-1)
        dir_ = np.ones(len(c)-1, dtype=int)
        sup[0]  = lower[0]
        dir_[0] = 1
        for i in range(1, len(c)-1):
            if c[i] > sup[i-1]:
                sup[i]  = max(lower[i], sup[i-1]) if dir_[i-1] == 1 else lower[i]
                dir_[i] = 1
            else:
                sup[i]  = min(upper[i], sup[i-1]) if dir_[i-1] == -1 else upper[i]
                dir_[i] = -1
        return int(dir_[-1]), round(float(sup[-1]), 4)
    except Exception:
        return 1, 0.0


def _adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength (0-100). >25 = trending."""
    if len(close) < period + 2:
        return 0.0
    try:
        import numpy as np
        h = np.array(high, dtype=float)
        l = np.array(low,  dtype=float)
        c = np.array(close,dtype=float)
        # True range
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        # Directional movement
        up   = h[1:] - h[:-1]
        dn   = l[:-1] - l[1:]
        pdm  = np.where((up > dn) & (up > 0), up, 0.0)
        ndm  = np.where((dn > up) & (dn > 0), dn, 0.0)
        # Smoothed (Wilder's)
        def _wilder(arr, p):
            out = np.zeros(len(arr))
            out[p-1] = arr[:p].sum()
            for i in range(p, len(arr)):
                out[i] = out[i-1] - out[i-1]/p + arr[i]
            return out
        atr14 = _wilder(tr,  period)
        pdm14 = _wilder(pdm, period)
        ndm14 = _wilder(ndm, period)
        pdi = np.where(atr14 > 0, 100 * pdm14 / atr14, 0.0)
        ndi = np.where(atr14 > 0, 100 * ndm14 / atr14, 0.0)
        dx  = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)
        adx = _wilder(dx, period)
        return round(float(adx[-1]), 1)
    except Exception:
        return 0.0


def _ichimoku(high, low, close):
    """
    Ichimoku Cloud analysis.
    Returns dict: above_cloud, cloud_bullish, tk_ks_bullish, chikou_bullish
    - above_cloud: price is above Senkou Span A and B (strong bull)
    - cloud_bullish: Senkou A > B (bullish cloud)
    - tk_ks_bullish: Tenkan-Sen (9) > Kijun-Sen (26) — TK cross
    - chikou_bullish: Chikou Span (26d lagged close) above past price
    """
    if len(close) < 52:
        return {"above_cloud": False, "cloud_bullish": False, "tk_ks_bullish": False, "chikou_bullish": False}
    try:
        h = list(high)
        l = list(low)
        c = list(close)
        def midpoint(h_s, l_s, n):
            return (max(h_s[-n:]) + min(l_s[-n:])) / 2

        tenkan  = midpoint(h, l, 9)    # Conversion line
        kijun   = midpoint(h, l, 26)   # Base line
        ssa     = (tenkan + kijun) / 2              # Senkou A (26d ahead)
        ssb     = midpoint(h, l, 52) if len(h) >= 52 else (tenkan + kijun) / 2  # Senkou B (52d)
        price   = c[-1]
        # Cloud boundary for current price comparison
        cloud_top = max(ssa, ssb)
        cloud_bot = min(ssa, ssb)
        above_cloud  = price > cloud_top
        below_cloud  = price < cloud_bot
        cloud_bullish = ssa > ssb   # Green cloud = bullish
        tk_ks_bullish = tenkan > kijun
        # Chikou span: today's close vs close 26 days ago
        chikou_bullish = c[-1] > c[-26] if len(c) >= 26 else False
        return {
            "above_cloud":    above_cloud,
            "below_cloud":    below_cloud,
            "cloud_bullish":  cloud_bullish,
            "tk_ks_bullish":  tk_ks_bullish,
            "chikou_bullish": chikou_bullish,
        }
    except Exception:
        return {"above_cloud": False, "cloud_bullish": False, "tk_ks_bullish": False, "chikou_bullish": False}


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


def _macd_divergence(closes):
    """Detect MACD bullish/bearish divergence.
    Bullish: price makes lower low, MACD histogram makes higher low (hidden buying pressure).
    Bearish: price makes higher high, MACD histogram makes lower high.
    Returns dict: {bullish_div, bearish_div, hist_now, hist_prev}
    """
    result = {"bullish_div": False, "bearish_div": False, "hist_now": 0.0, "hist_prev": 0.0}
    if len(closes) < 40:
        return result
    try:
        def _hist_at(c):
            e12 = _ema(c, 12)
            e26 = _ema(c, 26)
            if not e12 or not e26:
                return 0.0
            macd_line = e12 - e26
            signal = _ema([e12 - _ema(c[:i+1], 26) or 0 for i in range(26, len(c))], 9)
            return macd_line - (signal or 0)

        hist_now   = _hist_at(closes)
        hist_5ago  = _hist_at(closes[:-5])
        hist_10ago = _hist_at(closes[:-10])

        result["hist_now"]  = round(hist_now, 4)
        result["hist_prev"] = round(hist_5ago, 4)

        price_now  = closes[-1]
        price_5ago = closes[-6]

        # Bullish divergence: price lower but histogram higher (momentum not confirming weakness)
        if price_now < price_5ago and hist_now > hist_5ago and hist_now < 0:
            result["bullish_div"] = True
        # Bearish divergence: price higher but histogram lower (momentum not confirming strength)
        if price_now > price_5ago and hist_now < hist_5ago and hist_now > 0:
            result["bearish_div"] = True
    except Exception:
        pass
    return result


def _chandelier_exit(highs, closes, period=22, multiplier=3.0):
    """Chandelier Exit: highest close in N bars - multiplier × ATR.
    Best-in-class trailing stop — used by institutional traders.
    Returns the stop level as a price (0 if not computable)."""
    if len(highs) < period + 1 or len(closes) < period + 1:
        return 0.0
    try:
        highest_close = max(closes[-period:])
        atr_val = _atr(list(highs[-(period+1):]),
                       [min(highs[i], closes[i]) for i in range(-(period+1), 0)],
                       list(closes[-(period+1):]), period)
        if not atr_val:
            return 0.0
        return round(highest_close - multiplier * atr_val, 4)
    except Exception:
        return 0.0


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


def _cup_and_handle(highs, lows, closes, volumes=None) -> dict:
    """
    Cup & Handle breakout pattern (O'Neil CAN SLIM).
    Cup: 10-40% rounded U-base; handle: 3-15% pullback; pivot: right lip.
    breakout_ready=True when price is within 3% of pivot with contracted volume.
    """
    n = len(closes)
    if n < 45:
        return {"detected": False, "breakout_ready": False, "pivot_price": 0.0}
    try:
        w  = min(n, 65)
        c  = closes[-w:]
        h  = highs[-w:]
        l  = lows[-w:]
        v  = volumes[-w:] if volumes and len(volumes) >= w else None
        nw = len(c)

        # 3-segment: left shoulder (25%), base (50%), right shoulder (25%)
        s1 = max(1, int(nw * 0.25))
        s2 = max(s1 + 1, int(nw * 0.75))

        left_high  = max(h[:s1])
        base_low   = min(l[s1:s2])
        right_high = max(h[s2:])

        cup_depth  = (left_high - base_low) / left_high * 100
        symmetry   = abs(right_high - left_high) / left_high * 100

        # Cup valid: 10-40% depth, symmetric within 6%, right side near left high
        cup_valid = (
            10 <= cup_depth <= 40
            and symmetry < 6.0
            and right_high >= left_high * 0.94
        )
        if not cup_valid:
            return {"detected": False, "breakout_ready": False, "pivot_price": 0.0}

        # Handle: most recent 8-15 days, 3-15% internal range
        hw = min(15, max(5, nw // 5))
        hc = c[-hw:]
        h_high, h_low = max(hc), min(hc)
        handle_depth  = (h_high - h_low) / h_high * 100 if h_high > 0 else 0
        handle_valid  = 3 <= handle_depth <= 15

        # Breakout ready: current price within 3% below pivot, not >3% above
        pivot     = right_high
        price_now = c[-1]
        near_pivot = price_now >= pivot * 0.97 and price_now <= pivot * 1.03

        # Volume should contract during handle (< 75% of cup average)
        vol_ok = True
        if v and len(v) >= hw + 5:
            base_vol   = sum(v[:s2]) / max(1, s2)
            handle_vol = sum(v[-hw:]) / max(1, hw)
            vol_ok     = handle_vol < base_vol * 0.75 if base_vol > 0 else True

        return {
            "detected":       cup_valid,
            "breakout_ready": cup_valid and handle_valid and near_pivot and vol_ok,
            "pivot_price":    round(pivot, 2),
            "cup_depth_pct":  round(cup_depth, 1),
            "handle_pct":     round(handle_depth, 1) if handle_valid else 0.0,
        }
    except Exception:
        return {"detected": False, "breakout_ready": False, "pivot_price": 0.0}


def _supply_demand_zones(highs, lows, closes, volumes, lookback: int = 30) -> dict:
    """
    Identify institutional supply / demand zones from high-volume price clusters.
    Demand zone: heavy-volume up bars (institutions accumulating).
    Supply zone: heavy-volume down bars (institutions distributing).
    Returns at_demand / at_supply when current price is within 2.5% of a zone.
    """
    if len(closes) < 10 or not volumes or len(volumes) < 10:
        return {"at_demand": False, "at_supply": False}
    try:
        n  = min(lookback, len(closes))
        c  = closes[-n:]
        h  = highs[-n:]
        l  = lows[-n:]
        v  = volumes[-n:]

        avg_vol = sum(v) / len(v)
        if avg_vol <= 0:
            return {"at_demand": False, "at_supply": False}

        demand_lvls, supply_lvls = [], []
        for i in range(1, len(c)):
            if v[i] > avg_vol * 1.8:          # significant-volume bar
                mid = (h[i] + l[i]) / 2
                if c[i] > c[i - 1]:           # rising bar = demand
                    demand_lvls.append(mid)
                elif c[i] < c[i - 1]:         # falling bar = supply
                    supply_lvls.append(mid)

        cur = closes[-1]
        tol = 0.025   # within 2.5% of the zone

        return {
            "at_demand": any(abs(cur - p) / p < tol for p in demand_lvls),
            "at_supply": any(abs(cur - p) / p < tol for p in supply_lvls),
        }
    except Exception:
        return {"at_demand": False, "at_supply": False}


def _volume_profile_poc(highs, lows, closes, volumes, lookback: int = 30, bins: int = 20) -> dict:
    """
    Volume Profile: find the Point of Control (POC) — price level with the most traded volume.
    Uses a simplified VPSV (Volume Profile Session Volume) approach: distribute each bar's
    volume across its high-low range into price bins, find the highest-volume bin.
    Returns {poc_price, at_poc, above_poc, poc_breakout}.
    """
    if len(closes) < 10 or not volumes or len(volumes) < 10:
        return {"poc_price": 0.0, "at_poc": False, "above_poc": False, "poc_breakout": False}
    try:
        n  = min(lookback, len(closes))
        c  = closes[-n:]
        h  = highs[-n:]
        l  = lows[-n:]
        v  = volumes[-n:]

        lo_range = min(l)
        hi_range = max(h)
        if hi_range <= lo_range:
            return {"poc_price": 0.0, "at_poc": False, "above_poc": False, "poc_breakout": False}

        bucket_size = (hi_range - lo_range) / bins
        vol_bins = [0.0] * bins

        for i in range(len(c)):
            bar_lo = l[i]
            bar_hi = h[i]
            bar_vol = v[i] or 0
            bar_range = bar_hi - bar_lo
            if bar_range <= 0 or bar_vol <= 0:
                continue
            for b in range(bins):
                bin_lo = lo_range + b * bucket_size
                bin_hi = bin_lo + bucket_size
                overlap = max(0.0, min(bar_hi, bin_hi) - max(bar_lo, bin_lo))
                if overlap > 0:
                    vol_bins[b] += bar_vol * (overlap / bar_range)

        poc_bin = vol_bins.index(max(vol_bins))
        poc_price = round(lo_range + (poc_bin + 0.5) * bucket_size, 2)

        cur = c[-1]
        tol = 0.015   # within 1.5% = at POC
        at_poc = abs(cur - poc_price) / poc_price < tol
        above_poc = cur > poc_price * (1 + tol)

        # POC breakout: price was below POC last bar, now above = volume support reclaimed
        poc_breakout = False
        if len(c) >= 2:
            prev = c[-2]
            poc_breakout = (prev < poc_price * 0.985) and (cur >= poc_price * 0.985)

        return {
            "poc_price":    poc_price,
            "at_poc":       at_poc,
            "above_poc":    above_poc,
            "poc_breakout": poc_breakout,
        }
    except Exception:
        return {"poc_price": 0.0, "at_poc": False, "above_poc": False, "poc_breakout": False}


def _double_bottom(highs, lows, closes) -> dict:
    """
    Detect Double Bottom (W-pattern) — powerful bullish reversal pattern.
    Two lows at approximately the same level, with a bounce between them.
    Confirmed when price breaks above the intermediate high (neckline).
    Returns {detected: bool, neckline: float, first_low: float, second_low: float}
    """
    n = len(closes)
    if n < 30:
        return {"detected": False, "neckline": 0.0}
    try:
        c = closes[-min(n, 60):]
        l = lows[-min(n, 60):]
        h = highs[-min(n, 60):]
        nw = len(c)

        # Find two local lows in the window
        local_lows = []
        for i in range(2, nw - 2):
            if l[i] <= l[i-1] and l[i] <= l[i+1] and l[i] <= l[i-2]:
                local_lows.append((i, l[i]))

        if len(local_lows) < 2:
            return {"detected": False, "neckline": 0.0}

        # Take the two most recent local lows
        low1_idx, low1_val = local_lows[-2]
        low2_idx, low2_val = local_lows[-1]

        # Must have at least 5 bars between lows and low2 must be recent (within last 8 bars)
        if low2_idx - low1_idx < 5 or (nw - 1 - low2_idx) > 8:
            return {"detected": False, "neckline": 0.0}

        # Lows must be within 4% of each other (symmetry)
        if abs(low1_val - low2_val) / max(low1_val, low2_val) > 0.04:
            return {"detected": False, "neckline": 0.0}

        # Find the intermediate high (neckline) between the two lows
        neckline = max(h[low1_idx:low2_idx]) if low2_idx > low1_idx else 0
        if neckline <= 0:
            return {"detected": False, "neckline": 0.0}

        # Pattern confirmed: current price should be above or near neckline
        price_now = c[-1]
        confirmed = price_now >= neckline * 0.98   # at or above neckline

        return {
            "detected":   confirmed,
            "neckline":   round(neckline, 2),
            "first_low":  round(low1_val, 2),
            "second_low": round(low2_val, 2),
        }
    except Exception:
        return {"detected": False, "neckline": 0.0}


def _double_top(highs, lows, closes) -> dict:
    """
    Detect Double Top (M-pattern) — bearish reversal pattern for short setups.
    Two peaks at similar levels, price breaking below the intermediate low (neckline).
    Returns {detected: bool, neckline: float}
    """
    n = len(closes)
    if n < 30:
        return {"detected": False, "neckline": 0.0}
    try:
        c = closes[-min(n, 60):]
        l = lows[-min(n, 60):]
        h = highs[-min(n, 60):]
        nw = len(c)

        local_highs = []
        for i in range(2, nw - 2):
            if h[i] >= h[i-1] and h[i] >= h[i+1] and h[i] >= h[i-2]:
                local_highs.append((i, h[i]))

        if len(local_highs) < 2:
            return {"detected": False, "neckline": 0.0}

        high1_idx, high1_val = local_highs[-2]
        high2_idx, high2_val = local_highs[-1]

        if high2_idx - high1_idx < 5 or (nw - 1 - high2_idx) > 8:
            return {"detected": False, "neckline": 0.0}

        if abs(high1_val - high2_val) / max(high1_val, high2_val) > 0.04:
            return {"detected": False, "neckline": 0.0}

        neckline = min(l[high1_idx:high2_idx]) if high2_idx > high1_idx else 0
        if neckline <= 0:
            return {"detected": False, "neckline": 0.0}

        price_now = c[-1]
        confirmed = price_now <= neckline * 1.02  # at or below neckline

        return {"detected": confirmed, "neckline": round(neckline, 2)}
    except Exception:
        return {"detected": False, "neckline": 0.0}


def _higher_lows_trend(lows, closes, lookback: int = 20, min_pivots: int = 2) -> bool:
    """
    Detect higher lows pattern — the foundation of an uptrend.
    Finds local pivot lows in the last 'lookback' bars and checks if each
    successive low is higher than the previous (ascending floor of support).
    Requires at least min_pivots consecutive higher lows.
    """
    n = min(lookback, len(lows))
    if n < 8:
        return False
    try:
        l = lows[-n:]
        pivot_lows = []
        for i in range(1, n - 1):
            if l[i] <= l[i - 1] and l[i] <= l[i + 1]:
                pivot_lows.append(l[i])
        if len(pivot_lows) < min_pivots:
            return False
        # Check that consecutive pivots are strictly higher
        for i in range(1, len(pivot_lows)):
            if pivot_lows[i] <= pivot_lows[i - 1] * 1.001:  # tolerance 0.1%
                return False
        # Also confirm current close is above the last pivot low
        return closes[-1] > pivot_lows[-1]
    except Exception:
        return False


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
        # Fetch 250 days for SPY, VIX, QQQ (risk-on), IWM (small cap), HYG (credit risk)
        raw = yf.download("SPY QQQ IWM HYG ^VIX", period="250d", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)

        def _get_closes(sym):
            try:
                return list(raw["Close"][sym].dropna())
            except Exception:
                return []

        spy_closes = _get_closes("SPY")
        vix_closes = _get_closes("^VIX")
        qqq_closes = _get_closes("QQQ")
        iwm_closes = _get_closes("IWM")
        hyg_closes = _get_closes("HYG")

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

        # Risk-on indicator: QQQ outperforming SPY = tech/growth leading (bull signal)
        if len(qqq_closes) >= 5 and len(spy_closes) >= 5:
            qqq_5d = (qqq_closes[-1] - qqq_closes[-5]) / qqq_closes[-5] * 100
            spy_5d = (spy_closes[-1] - spy_closes[-5]) / spy_closes[-5] * 100
            qqq_rel = qqq_5d - spy_5d
            if   qqq_rel > 2.0:  score += 1  # QQQ leading = risk-on
            elif qqq_rel < -2.0: score -= 1  # QQQ lagging = risk-off, rotate defensive

        # Small cap signal: IWM vs SPY — small caps lead bull markets
        if len(iwm_closes) >= 5 and len(spy_closes) >= 5:
            iwm_5d = (iwm_closes[-1] - iwm_closes[-5]) / iwm_closes[-5] * 100
            iwm_rel = iwm_5d - spy_5d if len(spy_closes) >= 5 else 0
            if   iwm_rel > 2.0:  score += 1  # small caps leading = broad participation
            elif iwm_rel < -2.0: score -= 1  # small caps lagging = narrow/defensive market

        # Credit market signal: HYG (high yield bonds) relative to recent average
        # HYG rising = credit stress reducing = risk-on; HYG falling = stress rising = risk-off
        if len(hyg_closes) >= 20:
            hyg_ema20 = _ema(hyg_closes, 20)
            hyg_pos = (hyg_closes[-1] - hyg_ema20) / hyg_ema20 * 100 if hyg_ema20 else 0
            if   hyg_pos > 0.5:  score += 1  # credit market healthy
            elif hyg_pos < -0.5: score -= 1  # credit stress = be careful

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
    "utilities":     "XLU",
    "materials":     "XLB",
    "real_estate":   "XLRE",
    "comm_services": "XLC",
    "crypto":        "IBIT",
}

def sector_rotation() -> dict:
    """
    Rank sectors by 1-day, 5-day, and 20-day ETF performance.
    Returns {sector: adj_score} where adj_score is -12 to +12.
    3-timeframe momentum: recent (1d) + short-term (5d) + medium-term (20d).
    Hot sectors get a bonus; cold sectors get a penalty.
    """
    etfs = list(SECTOR_ETFS.values())
    try:
        kw  = dict(group_by="ticker", auto_adjust=True, progress=False)
        raw = yf.download(" ".join(etfs), period="30d", interval="1d", **kw)
        adj = {}
        detail = {}
        for sec, etf in SECTOR_ETFS.items():
            try:
                if len(etfs) == 1:
                    closes = list(raw["Close"].dropna())
                else:
                    col = raw["Close"]
                    closes = list(col[etf].dropna()) if etf in col.columns else []
                if len(closes) < 2:
                    adj[sec] = 0
                    continue
                chg1d  = (closes[-1] - closes[-2]) / closes[-2] * 100
                chg5d  = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5  else 0
                chg20d = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0
                sc = 0
                # 1-day momentum (weight: ±4)
                if   chg1d > 2.0:  sc += 4
                elif chg1d > 0.5:  sc += 2
                elif chg1d > 0.1:  sc += 1
                elif chg1d < -2.0: sc -= 4
                elif chg1d < -0.5: sc -= 2
                elif chg1d < -0.1: sc -= 1
                # 5-day momentum (weight: ±4)
                if   chg5d > 5.0:  sc += 4
                elif chg5d > 2.0:  sc += 2
                elif chg5d > 0.5:  sc += 1
                elif chg5d < -5.0: sc -= 4
                elif chg5d < -2.0: sc -= 2
                elif chg5d < -0.5: sc -= 1
                # 20-day trend (weight: ±4)
                if   chg20d > 8.0:  sc += 4
                elif chg20d > 3.0:  sc += 2
                elif chg20d > 1.0:  sc += 1
                elif chg20d < -8.0: sc -= 4
                elif chg20d < -3.0: sc -= 2
                elif chg20d < -1.0: sc -= 1
                adj[sec]    = max(-12, min(12, sc))
                detail[sec] = {"1d": round(chg1d,2), "5d": round(chg5d,2), "20d": round(chg20d,2)}
            except Exception:
                adj[sec] = 0
        hot  = sorted(adj.items(), key=lambda x: -x[1])[:3]
        cold = sorted(adj.items(), key=lambda x:  x[1])[:2]
        logger.info(f"Sector rotation hot:  {' | '.join(f'{s}:{v:+d}' for s,v in hot)}")
        logger.info(f"Sector rotation cold: {' | '.join(f'{s}:{v:+d}' for s,v in cold)}")

        # Sector momentum acceleration: 1d >> 5d average day = money flooding in NOW
        accel_sectors = []
        for sec, d_info in detail.items():
            avg_daily_5d = d_info["5d"] / 5 if d_info["5d"] else 0
            accel = d_info["1d"] - avg_daily_5d
            if accel > 1.5:   # today outpacing the recent average by >1.5% per day
                accel_sectors.append(sec)
                adj[sec] = min(12, adj.get(sec, 0) + 2)  # extra boost for accelerating sectors
        if accel_sectors:
            logger.info(f"Sector acceleration (money flowing in NOW): {', '.join(accel_sectors)}")

        return adj
    except Exception as e:
        logger.debug(f"Sector rotation error: {e}")
        return {}

_CORR_CACHE: dict = {}   # {frozenset({a,b}): corr_coef}

def is_correlated_with_held(candidate: str, held_syms: list, threshold: float = 0.85) -> bool:
    """
    Returns True if candidate has >threshold return correlation with any held position
    over the last 30 trading days. Prevents doubling down on the same market exposure.
    """
    if not held_syms:
        return False
    try:
        all_syms = [candidate] + held_syms
        raw = yf.download(" ".join(all_syms), period="40d", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)
        if raw.empty:
            return False
        def _returns(sym):
            try:
                col = raw["Close"]
                s = col[sym].dropna() if sym in col.columns else col.dropna()
                return s.pct_change().dropna()
            except Exception:
                return None
        cand_ret = _returns(candidate)
        if cand_ret is None or len(cand_ret) < 20:
            return False
        for h in held_syms:
            key = frozenset([candidate, h])
            if key in _CORR_CACHE:
                corr = _CORR_CACHE[key]
            else:
                held_ret = _returns(h)
                if held_ret is None or len(held_ret) < 20:
                    continue
                combined = cand_ret.align(held_ret, join="inner")[0]
                held_aln = cand_ret.align(held_ret, join="inner")[1]
                if len(combined) < 15:
                    continue
                try:
                    corr = float(combined.corr(held_aln))
                except Exception:
                    corr = 0.0
                _CORR_CACHE[key] = corr
            if corr >= threshold:
                logger.debug(f"Corr guard: {candidate} ↔ {h} corr={corr:.2f} ≥ {threshold} — skip")
                return True
        return False
    except Exception as e:
        logger.debug(f"Corr guard error for {candidate}: {e}")
        return False


_SECTOR_TREND_CACHE: dict = {}

def get_sector_etf_trend() -> dict:
    """
    Compute trend status for each sector ETF.
    Returns {sector: {'above_ema20': bool, 'chg1d': float, 'chg5d': float, 'bullish': bool}}
    Called once per run and cached. Provides sector-level confirmation filter.
    """
    global _SECTOR_TREND_CACHE
    if _SECTOR_TREND_CACHE:
        return _SECTOR_TREND_CACHE
    try:
        etfs = list(SECTOR_ETFS.values())
        raw  = yf.download(" ".join(etfs), period="30d", interval="1d",
                           group_by="ticker", auto_adjust=True, progress=False)
        result = {}
        for sec, etf in SECTOR_ETFS.items():
            try:
                closes = list(raw["Close"][etf].dropna()) if etf in raw["Close"] else []
                if len(closes) < 5:
                    result[sec] = {"above_ema20": True, "chg1d": 0, "chg5d": 0, "bullish": True}
                    continue
                chg1d  = (closes[-1] - closes[-2]) / closes[-2] * 100
                chg5d  = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
                above_ema20 = True
                if len(closes) >= 20:
                    ema20 = _ema(closes, 20)
                    above_ema20 = closes[-1] >= ema20 * 0.99 if ema20 else True
                # Bullish: above EMA20 AND at least flat on the day (or recovering)
                bullish = above_ema20 and chg5d > -5.0
                result[sec] = {
                    "above_ema20": above_ema20,
                    "chg1d":       round(chg1d, 2),
                    "chg5d":       round(chg5d, 2),
                    "bullish":     bullish,
                }
            except Exception:
                result[sec] = {"above_ema20": True, "chg1d": 0, "chg5d": 0, "bullish": True}
        _SECTOR_TREND_CACHE = result
        bearish_secs = [s for s, d in result.items() if not d["bullish"]]
        if bearish_secs:
            logger.info(f"Sector ETF bearish (filter active): {', '.join(bearish_secs)}")
        return result
    except Exception as e:
        logger.debug(f"Sector ETF trend error: {e}")
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
# FOMC decision dates 2025-2026
_FOMC_DATES = {
    "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}
# CPI release dates 2025-2026 (BLS schedule, typically 2nd/3rd Tuesday of month 8:30AM ET)
_CPI_DATES = {
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-15", "2025-11-13", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-15",
    "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
    "2026-09-09", "2026-10-14", "2026-11-12", "2026-12-09",
}
# Non-Farm Payrolls 2025-2026 (BLS, first Friday of each month 8:30AM ET)
_NFP_DATES = {
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
}
_MACRO_EVENTS = _FOMC_DATES | _CPI_DATES | _NFP_DATES

def near_macro_event(days_before: int = 1) -> bool:
    """True if today or tomorrow is a major macro event (FOMC, CPI, NFP)."""
    today = datetime.now(timezone.utc).date()
    for d in range(days_before + 1):
        check = (today + timedelta(days=d)).isoformat()
        if check in _MACRO_EVENTS:
            event_type = "FOMC" if check in _FOMC_DATES else "CPI" if check in _CPI_DATES else "NFP"
            logger.info(f"{event_type} macro event in {d}d ({check}) — cautious mode")
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


_PRE_EARN_CACHE:    dict = {}
_EARN_GUARD_CACHE:  dict = {}
_EARN_DAYS_CACHE:   dict = {}

def get_earnings_days(sym) -> int | None:
    """Return number of calendar days to next earnings, or None if unknown.
    Positive = future earnings, 0 = today, negative = already passed.
    Cached per run to avoid redundant yfinance calls.
    """
    if sym in _EARN_DAYS_CACHE:
        return _EARN_DAYS_CACHE[sym]
    result = None
    try:
        cal = yf.Ticker(sym).calendar
        if cal is not None and not cal.empty:
            now = datetime.now(timezone.utc).date()
            best = None
            for col in cal.columns:
                if "earnings" in str(col).lower():
                    for val in cal[col]:
                        try:
                            ed = pd.Timestamp(val).date()
                            days = (ed - now).days
                            if days >= -1:  # allow 1 day past (AM reporters)
                                if best is None or days < best:
                                    best = days
                        except Exception:
                            pass
            result = best
    except Exception:
        pass
    _EARN_DAYS_CACHE[sym] = result
    return result


def earnings_too_close(sym, guard_days: int = 2) -> bool:
    """
    Returns True if earnings are within guard_days days — too close to buy safely.
    Avoids gap-down risk from earnings surprises. Cached per run.
    """
    if sym in _EARN_GUARD_CACHE:
        return _EARN_GUARD_CACHE[sym]
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
                            days_out = (ed - now).days
                            if 0 <= days_out <= guard_days:
                                result = True
                                break
                        except Exception:
                            pass
                if result:
                    break
    except Exception:
        pass
    _EARN_GUARD_CACHE[sym] = result
    return result


def has_pre_earnings_setup(sym, min_days=4, max_days=20):
    """
    Returns True if this stock has earnings 4-20 days out AND shows momentum.
    Pre-earnings drift: stocks tend to move up 5-15 days before earnings as
    long funds position ahead of the catalyst. Cached per run.
    """
    key = f"{sym}_{min_days}_{max_days}"
    if key in _PRE_EARN_CACHE:
        return _PRE_EARN_CACHE[key]
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
                            days_out = (ed - now).days
                            if min_days <= days_out <= max_days:
                                result = True
                                break
                        except Exception:
                            pass
                if result:
                    break
    except Exception:
        pass
    _PRE_EARN_CACHE[key] = result
    return result


def get_pre_earnings_candidates(candidates: list, live: dict) -> set:
    """
    Find stocks with earnings 4-20 days out that are showing pre-earnings momentum.
    These are high-probability setups — sell BEFORE earnings day to avoid risk.
    """
    pre_earn = set()
    try:
        for sym in candidates[:40]:  # limit API calls
            if has_pre_earnings_setup(sym):
                # Require positive momentum: stock must be rising going into earnings
                sig = live.get(sym, {})
                chg = sig.get("change_pct", 0) or 0
                roc5 = sig.get("roc5", 0) or 0
                rsi = sig.get("daily_rsi", 50) or 50
                if chg > 0 and roc5 > 1 and 40 < rsi < 75:
                    pre_earn.add(sym)
    except Exception as e:
        logger.debug(f"Pre-earnings scan error: {e}")
    if pre_earn:
        logger.info(f"Pre-earnings momentum candidates: {', '.join(sorted(pre_earn))}")
    return pre_earn


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
    Find stocks that recently beat earnings estimates and are reacting positively.
    Catches both same-day beats AND post-earnings drift (PEAD) for up to 3 days.
    Returns set of ticker symbols.
    """
    beats = set()
    cand_set = set(candidates) if not isinstance(candidates, set) else candidates
    try:
        # Primary: use yfinance screener for recent earnings beats
        res = yf.screen("earnings_beat")
        for q in (res.get("quotes") or [])[:25]:
            sym   = q.get("symbol", "")
            chg   = q.get("regularMarketChangePercent", 0) or 0
            eps_surprise = q.get("epsActual", 0) or 0
            eps_estimate = q.get("epsEstimated", 0) or 0
            # Take stocks up 2%+ on earnings OR with large EPS beat (>20% surprise)
            if sym and sym in cand_set:
                eps_beat_pct = 0.0
                if eps_estimate and eps_estimate != 0:
                    eps_beat_pct = (eps_actual - eps_estimate) / abs(eps_estimate) * 100 if (eps_actual := eps_surprise) else 0
                if chg >= 2.0 or eps_beat_pct >= 20:
                    beats.add(sym)
    except Exception:
        pass

    # Secondary: look at day_gainers + earnings beat screener combo
    try:
        res2 = yf.screen("day_gainers")
        for q in (res2.get("quotes") or [])[:20]:
            sym   = q.get("symbol", "")
            chg   = q.get("regularMarketChangePercent", 0) or 0
            # Check earnings date — if earnings was within last 3 days AND stock is up 4%+
            if sym and sym in cand_set and chg >= 4.0:
                try:
                    cal = yf.Ticker(sym).calendar
                    if cal is not None and not cal.empty:
                        dates = cal.get("Earnings Date", [])
                        if dates:
                            from datetime import date as _date
                            today = _date.today()
                            for ed in (dates if hasattr(dates, '__iter__') else [dates]):
                                try:
                                    ed_date = ed.date() if hasattr(ed, 'date') else ed
                                    if 0 <= (today - ed_date).days <= 3:
                                        beats.add(sym)
                                        break
                                except Exception:
                                    pass
                except Exception:
                    pass
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


def get_52w_breakout_candidates(fractionable_set: set) -> set:
    """
    Find stocks breaking out to new 52-week highs with volume confirmation.
    William O'Neil's #1 institutional buying signal — strong stocks get stronger.
    Returns set of tickers at/near 52-week highs with ≥1.5x average volume.
    """
    breakouts = set()
    try:
        for screen in ("day_gainers", "most_actives"):
            res = yf.screen(screen)
            for q in (res.get("quotes") or [])[:30]:
                sym = q.get("symbol", "")
                if not sym or sym not in fractionable_set:
                    continue
                price   = q.get("regularMarketPrice", 0) or 0
                high52w = q.get("fiftyTwoWeekHigh", 0) or 0
                avg_vol = q.get("averageDailyVolume3Month", 0) or 0
                cur_vol = q.get("regularMarketVolume", 0) or 0
                chg_pct = q.get("regularMarketChangePercent", 0) or 0
                if (high52w > 0 and price > 0
                        and price >= high52w * 0.99   # within 1% of 52w high
                        and chg_pct >= 1.0             # up at least 1% today
                        and avg_vol > 0 and cur_vol >= avg_vol * 1.5  # volume confirmation
                        and price >= 5.0):             # not a penny stock
                    breakouts.add(sym)
    except Exception as e:
        logger.debug(f"52W breakout screener error: {e}")
    if breakouts:
        logger.info(f"52-week high breakouts: {', '.join(sorted(breakouts))}")
    return breakouts


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

        # News velocity: more recent positive news = higher conviction
        now_ts = datetime.now(timezone.utc).timestamp()
        recent_24h = [n for n in news_items if (now_ts - n.get("providerPublishTime", 0)) < 86400]
        older_48h  = [n for n in news_items if 86400 <= (now_ts - n.get("providerPublishTime", 0)) < 172800]
        _, recent_boost = detect_catalyst([n.get("title","") for n in recent_24h]) if recent_24h else (0, "")
        _, older_boost  = detect_catalyst([n.get("title","") for n in older_48h])  if older_48h  else (0, "")
        # Velocity bonus: accelerating positive news = up to +2 extra
        news_vel = (recent_boost - older_boost) * 0.1
        boost   += max(-2, min(2, news_vel))

        text  = "\n".join(headlines[:8]) if headlines else "(no recent news)"
        model = "claude-sonnet-4-6" if use_sonnet else "claude-haiku-4-5-20251001"

        # Build technical context string if signals provided
        tech_context = ""
        if signals:
            rsi       = signals.get("rsi", 50)
            roc5      = signals.get("roc5", 0)
            roc20     = signals.get("roc20", 0)
            stoch_k   = signals.get("stoch_k", 50)
            ema50     = signals.get("price_vs_ema50", 0)
            ema200    = signals.get("price_vs_ema200", 0)
            vr        = signals.get("vol_ratio", 1)
            chg       = signals.get("change_pct", 0)
            w_r       = signals.get("williams_r", -50)
            rs5       = signals.get("rs5", 0)
            rs63      = signals.get("rs63", 0)
            at_brk      = signals.get("at_breakout", False)
            consec      = signals.get("consec_green", 0)
            near_s      = signals.get("near_support", False)
            vol_dry     = signals.get("vol_dry_up", False)
            vwap_rcl    = signals.get("vwap_reclaim", False)
            nr7         = signals.get("nr7_signal", False)
            ttm         = signals.get("ttm_squeeze_fired", False)
            orb         = signals.get("orb_breakout", False)
            gap_hold    = signals.get("gap_and_hold", False)
            adx_v       = signals.get("adx", 0)
            ichi_cnt    = sum([signals.get("ichimoku_above", False),
                               signals.get("ichimoku_bull_cloud", False),
                               signals.get("ichimoku_tk_bull", False),
                               signals.get("ichimoku_chikou", False)])
            fib_sup     = signals.get("fib_support", False)
            macd_bdiv   = signals.get("macd_bull_div", False)
            daily_rsi   = signals.get("daily_rsi", 50)
            bb_pos      = signals.get("bb_pos", 50)
            extras = []
            if orb:            extras.append("ORB breakout (cleared first-hour high)")
            if vwap_rcl:       extras.append("VWAP reclaim (institutional buying on dip)")
            if gap_hold:       extras.append("gap-and-hold (opened higher, holding gains)")
            if at_brk:         extras.append("BREAKOUT above key resistance level")
            if ttm:            extras.append("TTM Squeeze breakout (coil released)")
            if near_s:         extras.append("at proven support level")
            if vol_dry:        extras.append("volume dry-up (selling exhaustion)")
            if nr7:            extras.append("NR7 coiling — volatility expansion imminent")
            if fib_sup:        extras.append("Fibonacci 38/50/62% support bounce")
            if macd_bdiv:      extras.append("MACD bullish divergence (hidden strength)")
            if ichi_cnt >= 3:  extras.append(f"Ichimoku {ichi_cnt}/4 signals bullish")
            if consec >= 3:    extras.append(f"{consec} consecutive green days")
            if signals.get("cup_handle"):
                piv = signals.get("cup_handle_pivot", 0)
                extras.append(f"Cup & Handle breakout (O'Neil pivot ${piv:.2f})" if piv else "Cup & Handle breakout")
            if signals.get("at_demand_zone"):
                extras.append("at institutional demand zone (high-vol accumulation)")
            if signals.get("mom_accel"):
                extras.append("momentum accelerating (ROC rising faster than prior week)")
            if signals.get("double_bottom"):
                nk = signals.get("double_bottom_neckline", 0)
                extras.append(f"Double Bottom W-pattern confirmed (neckline ${nk:.2f})" if nk else "Double Bottom W-pattern confirmed")
            if signals.get("double_top"):
                extras.append("Double Top M-pattern (bearish reversal — supply overhead)")
            if signals.get("poc_breakout"):
                poc = signals.get("poc_price", 0)
                extras.append(f"Volume Profile POC breakout (reclaimed ${poc:.2f} — highest-volume node)" if poc else "Volume POC breakout")
            elif signals.get("above_poc"):
                poc = signals.get("poc_price", 0)
                extras.append(f"Price above Volume POC ${poc:.2f}" if poc else "Above Volume POC")
            if signals.get("ema_stacked_bull"):
                extras.append("EMA5>EMA10>EMA20>EMA50 — full bull alignment (all timeframes agree)")
            if signals.get("ema_stacked_bear"):
                extras.append("EMA stack fully bearish (all EMAs declining)")
            if signals.get("vcp"):
                extras.append("VCP: Volatility Contraction Pattern — 3 tightening contractions, volume drying up (Minervini spring-loaded base)")
            if signals.get("obv_rising"):
                extras.append("OBV rising (On-Balance Volume up ≥2% — institutional accumulation signal)")
            if signals.get("rvol_surge"):
                rvol_val = signals.get("rvol", 1)
                extras.append(f"RVOL surge {rvol_val:.1f}× avg volume — institutional buying confirmed")
            if signals.get("mfi_bull_div"):
                extras.append(f"MFI bull divergence (MFI={signals.get('mfi',50):.0f}) — money flowing IN while price fell; institutional accumulation")
            elif signals.get("mfi_oversold"):
                extras.append(f"MFI oversold ({signals.get('mfi',50):.0f}) — volume-weighted RSI at accumulation zone")
            if signals.get("supertrend_bull"):
                extras.append(f"Supertrend bullish (stop ${signals.get('supertrend_stop',0):.2f}) — ATR-based trend confirmed; institutional algos are long")
            if signals.get("force_index_div"):
                extras.append(f"Force Index bull divergence — institutional volume flooding IN while price fell (smart money accumulation)")
            elif signals.get("force_index_rising"):
                extras.append(f"Force Index rising — sustained institutional buying force behind price move")
            true_alpha_v = signals.get("true_alpha", 0) or 0
            true_beta_v  = signals.get("true_beta",  1) or 1
            if true_alpha_v > 8:
                extras.append(f"Jensen's alpha {true_alpha_v:+.1f}% (beta={true_beta_v:.2f}) — outperforms SPY risk-adjusted; high-quality stock")
            if signals.get("ha_bull"):
                ha_n = signals.get("ha_consec_bull", 3)
                extras.append(f"Heikin-Ashi {ha_n} consecutive bull candles — clean institutional uptrend, no noise")
            if signals.get("donchian_up"):
                extras.append("Donchian 20-day high breakout — Turtle Trading momentum signal; institutional trend confirmation")
            if signals.get("kc_breakout"):
                extras.append("Keltner Channel breakout — price above EMA+2×ATR (strong momentum)")
            if signals.get("kc_oversold"):
                extras.append("Keltner Channel oversold — price below EMA-2×ATR (mean-reversion setup)")
            if signals.get("higher_lows"):
                extras.append("Higher Lows pattern — ascending support floor confirmed")
            if signals.get("trend_reversal"):
                extras.append("20-EMA trend reversal with volume + RSI recovery")
            if signals.get("bull_flag"):
                extras.append("bull flag pattern (tight consolidation after flagpole)")
            if signals.get("mtf_aligned"):
                extras.append("multi-timeframe confirmed (daily + hourly aligned)")
            tech_context = (
                f"\nTechnical: RSI={rsi:.0f}, DailyRSI={daily_rsi:.0f}, StochRSI_K={stoch_k:.0f}, "
                f"BB%={bb_pos:.0f}, W%R={w_r:.0f}, ADX={adx_v:.0f}, "
                f"5d_ROC={roc5:+.1f}%, 20d_ROC={roc20:+.1f}%, "
                f"EMA50_pos={ema50:+.1f}%, EMA200_pos={ema200:+.1f}%, "
                f"Vol_ratio={vr:.1f}x, Day_chg={chg:+.1f}%, "
                f"RS5_vs_SPY={rs5:+.1f}%, RS63_vs_SPY={rs63:+.1f}%"
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
                "max_tokens": 150,
                "messages": [{
                    "role":    "user",
                    "content": (
                        f"You are an elite quantitative swing trader analyzing {ticker} for a 2-5 day trade.\n"
                        f"Rate the short-term outlook from -10 (very bearish) to +10 (very bullish).\n"
                        f"Consider: hard catalysts (earnings beats/FDA/M&A/upgrades), "
                        f"institutional accumulation signals, sector momentum, "
                        f"technical setup quality, and risk/reward.\n"
                        f"Be aggressive on HIGH-CONVICTION setups (score 7-10) and decisive on bearish (score -7 to -10).\n"
                        f"Headlines:{tech_context}\n{text}\n\n"
                        f"Return ONLY JSON: {{\"s\":<-10 to 10>,\"c\":\"<catalyst 3 words>\"}}"
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
    # Priority screeners: high-volume movers first (most actionable for swing trades)
    screeners = [
        "day_gainers",               # stocks up the most today
        "most_actives",              # highest dollar volume — institutional interest
        "growth_technology_stocks",  # high-growth tech = momentum regime
        "undervalued_growth_stocks", # value + growth combo — mean reversion candidates
        "aggressive_small_caps",     # small caps with big moves — highest volatility
        "small_cap_gainers",         # small cap momentum
        "portfolio_anchors",         # dividend growers — defensive add when bear regime
    ]
    for name in screeners:
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


# ── Volume surge detector ─────────────────────────────────────────────────────
def get_volume_surge_candidates(fractionable_set: set) -> set:
    """
    Find stocks with 5x+ average daily volume that are also up on the day.
    Unusual volume = someone big is buying. Combine with price strength for high conviction.
    """
    candidates = set()
    try:
        for screen in ("most_actives", "day_gainers"):
            res = yf.screen(screen)
            for q in (res.get("quotes") or [])[:40]:
                sym = q.get("symbol", "")
                if not sym or sym not in fractionable_set:
                    continue
                avg_vol = q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day") or 0
                cur_vol = q.get("regularMarketVolume") or 0
                chg_pct = q.get("regularMarketChangePercent") or 0
                price   = q.get("regularMarketPrice") or 0
                if avg_vol > 0:
                    vol_ratio = cur_vol / avg_vol
                    # Volume surge: 4x+ average volume AND price up on the day AND liquid stock
                    if vol_ratio >= 4.0 and chg_pct > 0.5 and price >= 3.0:
                        candidates.add(sym)
    except Exception as e:
        logger.debug(f"Volume surge scanner error: {e}")
    if candidates:
        logger.info(f"Volume surge candidates: {', '.join(sorted(candidates))}")
    return candidates


# ── Unusual options flow detector ─────────────────────────────────────────────
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

            # Sweep detection: look for calls trading at/near ask with large premium
            sweep_bullish = False
            try:
                cur_price = yf.Ticker(sym).fast_info.get("lastPrice", 0) or 0
                if cur_price > 0 and not calls.empty:
                    # Near-the-money calls (within 5% of current price) with volume > OI (fresh)
                    ntm = calls[(calls["strike"] >= cur_price * 0.95) & (calls["strike"] <= cur_price * 1.10)]
                    if not ntm.empty:
                        ntm_vol = int(ntm["volume"].fillna(0).sum())
                        ntm_oi  = int(ntm["openInterest"].fillna(0).sum())
                        ntm_premium = float((ntm["lastPrice"].fillna(0) * ntm["volume"].fillna(0)).sum()) * 100
                        if ntm_vol > ntm_oi * 0.5 and ntm_premium > 50_000:  # fresh buying, $50k+ premium
                            sweep_bullish = True
            except Exception:
                pass
            bullish = bullish or sweep_bullish

            if bullish or bearish:
                result[sym] = {
                    "call_vol":       call_vol,
                    "put_vol":        put_vol,
                    "call_oi":        call_oi,
                    "put_oi":         put_oi,
                    "put_call":       round(put_call, 2),
                    "bullish":        bullish,
                    "bearish":        bearish,
                    "sweep":          sweep_bullish,
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
    Uses 90d daily data (same as equity) for full technical analysis.
    Returns {alpaca_symbol: signal_dict}.
    """
    result = {}
    for alpaca_sym, yf_sym in CRYPTO_UNIVERSE.items():
        try:
            daily  = yf.download(yf_sym, period="90d", interval="1d",
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
    Score a crypto asset 0-100. Crypto is 24/7 and sentiment-driven,
    so we weight momentum and volume heavily, but also apply institutional
    pattern signals now that we have full 90d data.
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
    ema200 = sig.get("price_vs_ema200", 0) or 0
    rs63   = sig.get("rs63", 0) or 0
    adx    = sig.get("adx", 0) or 0
    mtf    = sig.get("mtf_aligned", False)
    fib_s  = sig.get("fib_support", False)
    macd_bd= sig.get("macd_bull_div", False)
    cup_h  = sig.get("cup_handle", False)
    mom_a  = sig.get("mom_accel", False)
    demand = sig.get("at_demand_zone", False)
    trend_r= sig.get("trend_reversal", False)

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
    if   roc5 >  8:  s += 14
    elif roc5 >  3:  s +=  8
    elif roc5 < -8:  s -= 12
    elif roc5 < -3:  s -=  6
    if   stoch_k < 20: s += 12  # oversold = high-conviction bounce
    elif stoch_k > 85: s -=  8
    if   ema50 >  5:  s +=  8   # above 50-day EMA = uptrend
    elif ema50 < -5:  s -= 10
    if   w_r < -80:  s += 10    # oversold Williams %R
    elif w_r > -20:  s -=  6
    if   m_slp > 0:  s +=  8    # MACD slope rising
    elif m_slp < 0:  s -=  6
    if ttm:          s += 16    # TTM squeeze fired = breakout
    # Deep indicator signals (from 90d data)
    if   ema200 > 10: s += 8    # above 200d EMA = macro bull trend
    elif ema200 < -10: s -= 10
    if   rs63 > 15:   s += 8    # top quarterly performer
    elif rs63 > 5:    s += 4
    if   adx >= 30:   s += 6    # strong trend = ride it
    if mtf:           s += 10   # multi-timeframe aligned
    if fib_s:         s += 8    # at Fibonacci support
    if macd_bd:       s += 10   # MACD bullish divergence
    if cup_h:         s += 14   # Cup & Handle on crypto = very powerful
    if mom_a:         s += 9    # momentum accelerating
    if demand:        s += 7    # at demand zone
    if trend_r:       s += 12   # trend reversal detected

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

    # Layer 4: fill up to MAX_SCAN_TICKERS using Alpaca snapshot activity scoring
    # Pre-filter to quality exchanges, then rank by volume*|change| to pick the most active names
    MAX_SCAN_TICKERS = 220
    if len(universe) < MAX_SCAN_TICKERS:
        quality_candidates = [
            s for s in fractionable
            if s not in universe
            and fractionable[s].get("exchange") in ("NYSE", "NASDAQ", "CBOE", "ARCA")
            and len(s) <= 5
        ]
        # Try to rank by live snapshot activity (volume × |chg|) for better quality
        try:
            _snap_pool = quality_candidates[:800]  # sample from quality pool
            _snaps = alpaca_snapshots(_snap_pool) if _snap_pool else {}
            # Score each: vol * |chg| * price_threshold (avoid penny stocks)
            _ranked = []
            for s in quality_candidates:
                if s in _snaps:
                    snap = _snaps[s]
                    if snap.get("price", 0) < 5.0:
                        continue  # skip penny stocks
                    activity = snap.get("volume", 0) * abs(snap.get("chg_pct", 0))
                    _ranked.append((s, activity))
                else:
                    _ranked.append((s, 0))  # include unknowns at zero priority
            _ranked.sort(key=lambda x: -x[1])
            extras = [s for s, _ in _ranked]
        except Exception:
            extras = sorted(quality_candidates, key=lambda s: (len(s), s))
        for sym in extras:
            if len(universe) >= MAX_SCAN_TICKERS:
                break
            universe.add(sym)

    candidates = list(universe)
    logger.info(f"Full scan universe: {len(candidates)} tickers (movers: {len(movers)}, held: {len(held_symbols)})")
    return candidates, shortable


# ── Batch data fetch ──────────────────────────────────────────────────────────
def _fetch_spy_perf() -> dict:
    """
    Fetch SPY's 1-day and 5-day return once per run.
    Stored in _SPY_PERF_CACHE so individual stocks can compute relative strength and beta.
    """
    global _SPY_PERF_CACHE
    if _SPY_PERF_CACHE:
        return _SPY_PERF_CACHE
    try:
        spy = yf.download("SPY", period="90d", interval="1d",
                          auto_adjust=True, progress=False)
        closes = list(spy["Close"].dropna())
        if len(closes) >= 2:
            _SPY_PERF_CACHE["d1"]     = (closes[-1] - closes[-2]) / closes[-2] * 100
            _SPY_PERF_CACHE["d5"]     = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
            _SPY_PERF_CACHE["d10"]    = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
            _SPY_PERF_CACHE["d63"]    = (closes[-1] - closes[-63]) / closes[-63] * 100 if len(closes) >= 63 else 0
            _SPY_PERF_CACHE["closes"] = closes  # full close history for beta regression
    except Exception:
        _SPY_PERF_CACHE = {"d1": 0.0, "d5": 0.0, "d10": 0.0, "d63": 0.0, "closes": []}
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

    # Fibonacci retracement level detection — institutional support zones
    fib_support    = False   # price near 38.2% / 50% / 61.8% retracement and holding
    fib_resistance = False   # price near 61.8% / 78.6% when in downtrend
    try:
        if len(daily) >= 20 and "High" in daily.columns and "Low" in daily.columns:
            _fib_high = float(daily["High"].iloc[-20:].max())
            _fib_low  = float(daily["Low"].iloc[-20:].min())
            _range    = _fib_high - _fib_low
            if _range > 0:
                fib_382 = _fib_high - 0.382 * _range
                fib_500 = _fib_high - 0.500 * _range
                fib_618 = _fib_high - 0.618 * _range
                fib_786 = _fib_high - 0.786 * _range
                # Within 1% of Fibonacci level = at the zone
                for fib_lvl in [fib_382, fib_500, fib_618]:
                    if abs(price - fib_lvl) / fib_lvl < 0.012:
                        # Bouncing off support: price was below this level recently
                        dc_fib = list(daily["Close"].iloc[-5:])
                        if dc_fib and min(dc_fib[:-1]) < fib_lvl and price >= fib_lvl * 0.998:
                            fib_support = True
                        break
                if abs(price - fib_786) / fib_786 < 0.012:
                    fib_resistance = True  # near 78.6% = often strong resistance
    except Exception:
        pass

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
    daily_trend    = 0.0
    daily_rsi      = 50.0
    stoch_k        = 50.0
    stoch_d        = 50.0
    roc5           = 0.0
    roc20          = 0.0
    price_vs_ema50  = 0.0
    price_vs_ema200 = 0.0
    ema_stacked_bull = False   # EMA5 > EMA10 > EMA20 > EMA50 = "weekly aligned"
    ema_stacked_bear = False   # EMA5 < EMA10 < EMA20 < EMA50 = "weekly bear"
    e5 = e10 = e20 = e50 = 0.0
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
        if len(dc) >= 20:
            roc20 = _roc(dc, 20)
            e20 = _ema(dc, 20)
        if len(dc) >= 30:
            stoch_k, stoch_d = _stoch_rsi(dc, rsi_period=14, stoch_period=14)
        if len(dc) >= 50:
            e50 = _ema(dc, 50)
            if e50 and e50 > 0:
                price_vs_ema50 = (dc[-1] - e50) / e50 * 100
            # EMA stack: price > EMA5 > EMA10 > EMA20 > EMA50 = full bull alignment
            try:
                if all(x and x > 0 for x in [e5, e10, e20, e50]):
                    ema_stacked_bull = dc[-1] > e5 > e10 > e20 > e50
                    ema_stacked_bear = dc[-1] < e5 < e10 < e20 < e50
            except Exception:
                pass
        # 200-day EMA: above = institutional uptrend, below = bear territory
        if len(dc) >= 200:
            e200 = _ema(dc, 200)
            if e200 and e200 > 0:
                price_vs_ema200 = (dc[-1] - e200) / e200 * 100
        elif len(dc) >= 63:
            # Use 63-day as proxy if 200 not available (93-day period = 1 quarter)
            e200 = _ema(dc, len(dc))
            price_vs_ema200 = (dc[-1] - e200) / e200 * 100 if e200 else 0.0
    except Exception:
        pass

    # ADX — trend strength (0-100): >25 = strong trend, <20 = choppy/ranging
    adx_val = 0.0
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 30:
            adx_val = _adx(list(daily["High"]), list(daily["Low"]), list(daily["Close"]), period=14)
    except Exception:
        pass

    # MACD divergence on daily closes
    macd_div = {"bullish_div": False, "bearish_div": False, "hist_now": 0.0, "hist_prev": 0.0}
    try:
        if len(daily) >= 40:
            macd_div = _macd_divergence(list(daily["Close"]))
    except Exception:
        pass

    # Chandelier Exit — institutional trailing stop price
    chandelier_stop = 0.0
    try:
        if "High" in daily.columns and len(daily) >= 23:
            chandelier_stop = _chandelier_exit(
                list(daily["High"]), list(daily["Close"]), period=22, multiplier=3.0
            )
    except Exception:
        pass

    # Volatility Contraction Pattern (VCP): Mark Minervini's signature base setup
    vcp = False
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 12:
            _vols = list(daily["Volume"]) if "Volume" in daily.columns else None
            vcp = _vcp_pattern(list(daily["High"]), list(daily["Low"]), list(daily["Close"]),
                               volumes=_vols, lookback=30)
    except Exception:
        pass

    # On-Balance Volume (OBV): rising OBV = accumulation (smart money buying quietly)
    obv_rising = False
    obv_slope_pct = 0.0
    try:
        if "Volume" in daily.columns and len(daily) >= 5:
            _obv = _obv_trend(list(daily["Close"]), list(daily["Volume"]), lookback=20)
            obv_rising = _obv.get("obv_rising", False)
            obv_slope_pct = _obv.get("obv_slope_pct", 0.0)
    except Exception:
        pass

    # Keltner Channel: price vs EMA±ATR envelope
    kc_pos      = 50.0   # position 0-100+ (>100 = above upper band)
    kc_breakout = False  # price above upper Keltner = strong momentum breakout
    kc_oversold = False  # price below lower Keltner = mean-reversion candidate
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 22:
            kc_pos, kc_breakout, kc_oversold = _keltner_channel(
                list(daily["High"]), list(daily["Low"]), list(daily["Close"]),
                ema_period=20, atr_period=14, mult=2.0
            )
    except Exception:
        pass

    # Ichimoku Cloud — comprehensive trend/support/resistance analysis
    ichimoku = {"above_cloud": False, "cloud_bullish": False, "tk_ks_bullish": False, "chikou_bullish": False}
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 52:
            ichimoku = _ichimoku(list(daily["High"]), list(daily["Low"]), list(daily["Close"]))
    except Exception:
        pass

    # Money Flow Index (MFI): volume-weighted RSI (0-100)
    # MFI < 20 = oversold accumulation zone. MFI > 80 = overbought distribution.
    # More reliable than RSI alone because it incorporates volume (real buying pressure).
    mfi_val     = 50.0
    mfi_oversold  = False   # MFI < 25 with price stabilizing = smart money accumulation
    mfi_overbought = False  # MFI > 80 = distribution risk
    mfi_bull_div   = False  # price falling but MFI rising = hidden institutional buying
    try:
        if all(col in daily.columns for col in ["High", "Low", "Close", "Volume"]) and len(daily) >= 16:
            _dc = list(daily["Close"])
            _dh = list(daily["High"])
            _dl = list(daily["Low"])
            _dv = list(daily["Volume"].fillna(0))
            mfi_val        = _mfi(_dc, _dh, _dl, _dv, period=14)
            mfi_oversold   = mfi_val < 25 and chg_pct > -1.0   # cheap + stabilizing
            mfi_overbought = mfi_val > 80
            # Divergence: price down last 10 bars but MFI rising = smart money accumulating
            if len(_dc) >= 12:
                price_down   = _dc[-1] < _dc[-10]
                mfi_10ago    = _mfi(_dc[:-10], _dh[:-10], _dl[:-10], _dv[:-10], period=14)
                mfi_bull_div = price_down and mfi_val > mfi_10ago + 5
    except Exception:
        pass

    # Supertrend: ATR-based dynamic trailing stop. direction=1 = bullish (above supertrend).
    # Universally used by institutional and algorithmic traders as trend confirmation.
    # When price crosses below supertrend = trend reversal signal (high conviction sell).
    supertrend_dir   = 1    # 1=bullish, -1=bearish
    supertrend_stop  = 0.0  # the actual stop price level
    supertrend_bull  = False
    try:
        if all(col in daily.columns for col in ["High", "Low", "Close"]) and len(daily) >= 35:
            supertrend_dir, supertrend_stop = _supertrend(
                list(daily["Close"]), list(daily["High"]), list(daily["Low"]),
                period=10, multiplier=3.0
            )
            supertrend_bull = (supertrend_dir == 1)
    except Exception:
        pass

    # Elder's Force Index: (price_change × volume) smoothed — detects institutional power
    # Positive FI with price = strong hands buying. FI bull divergence = hidden accumulation.
    fi_val     = 0.0
    fi_rising  = False
    fi_bull_div = False
    try:
        if "Volume" in daily.columns and len(daily) >= 18:
            fi_val, fi_rising, fi_bull_div = _force_index(
                list(daily["Close"]), list(daily["Volume"].fillna(0)), period=13
            )
    except Exception:
        pass

    # True Beta (linear regression vs SPY over 63 days): institutional standard
    # Beta < 0.8 = defensive, Beta > 1.5 = high-volatility; Jensen's alpha > 0 = outperformer
    true_beta  = 1.0
    true_alpha = 0.0
    try:
        spy_cls = _fetch_spy_perf().get("closes", [])
        if spy_cls and len(daily) >= 65:
            true_beta, true_alpha = _beta_regression(
                list(daily["Close"]), spy_cls, period=63
            )
    except Exception:
        pass

    # Heikin-Ashi trend: noise-filtered candles — 3+ consecutive bull HA = strong trend
    # Eliminates head-fakes; institutional traders use HA for trend confirmation
    ha_bull = ha_bear = False
    ha_consec_bull = ha_consec_bear = 0
    try:
        if all(col in daily.columns for col in ["Open", "High", "Low", "Close"]) and len(daily) >= 10:
            ha_bull, ha_bear, ha_consec_bull, ha_consec_bear = _heikin_ashi_trend(
                list(daily["Open"]), list(daily["High"]), list(daily["Low"]), list(daily["Close"]),
                lookback=7
            )
    except Exception:
        pass

    # Donchian Channel breakout: N-day high breakout = turtle-trading momentum signal
    # Price above 20-day high = new sustained uptrend (high institutional conviction)
    donchian_up = donchian_down = False
    donchian_pct = 50.0
    try:
        if all(col in daily.columns for col in ["High", "Low", "Close"]) and len(daily) >= 22:
            donchian_up, donchian_down, donchian_pct = _donchian_breakout(
                list(daily["Close"]), list(daily["High"]), list(daily["Low"]), period=20
            )
    except Exception:
        pass

    # Candlestick patterns: key institutional reversal/continuation signals
    candle_patterns = {
        "hammer": False, "bullish_engulfing": False, "morning_star": False,
        "shooting_star": False, "bearish_engulfing": False,
        "three_white_soldiers": False, "three_black_crows": False,
    }
    try:
        if all(col in daily.columns for col in ["Open", "High", "Low", "Close"]) and len(daily) >= 3:
            candle_patterns = _candlestick_patterns(
                list(daily["Open"]), list(daily["High"]),
                list(daily["Low"]), list(daily["Close"]), lookback=3
            )
    except Exception:
        pass

    # Pivot points: yesterday's H/L/C → today's support/resistance levels
    pivot_levels = {}
    try:
        if all(col in daily.columns for col in ["High", "Low", "Close"]) and len(daily) >= 2:
            pivot_levels = _pivot_points(
                float(daily["High"].iloc[-2]),
                float(daily["Low"].iloc[-2]),
                float(daily["Close"].iloc[-2]),
            )
    except Exception:
        pass

    # Bull flag: flagpole (strong surge) followed by tight consolidation → breakout
    # One of the highest-probability momentum continuation patterns
    bull_flag = False
    try:
        if len(daily) >= 10 and "High" in daily.columns and "Low" in daily.columns:
            dc_f = list(daily["Close"])
            dh_f = list(daily["High"])
            dl_f = list(daily["Low"])
            dv_f = list(daily["Volume"]) if "Volume" in daily.columns else []
            if len(dc_f) >= 10:
                # Flagpole: look for 5%+ move in 1-3 days (the initial surge)
                flagpole_start = None
                for lookback in range(3, 8):
                    if len(dc_f) > lookback:
                        pole_ret = (dc_f[-lookback] - dc_f[-lookback-1]) / dc_f[-lookback-1] * 100
                        if pole_ret >= 5.0:   # strong flagpole
                            flagpole_start = lookback
                            break

                if flagpole_start:
                    # Flag: consolidation after pole — tight range, ideally declining volume
                    flag_highs  = dh_f[-flagpole_start+1:]
                    flag_lows   = dl_f[-flagpole_start+1:]
                    flag_high   = max(flag_highs) if flag_highs else price
                    flag_low    = min(flag_lows) if flag_lows else price
                    flag_range  = (flag_high - flag_low) / flag_high if flag_high > 0 else 1
                    flag_retr   = (dc_f[-flagpole_start] - price) / dc_f[-flagpole_start] * 100  # retracement from pole top
                    # Good flag: tight (<8% range), limited pullback (<50% of pole), near the top
                    if flag_range < 0.08 and -flag_retr < 50 and price > flag_low * 0.97:
                        # Volume check: volume should be declining during flag (sellers exhausted)
                        if dv_f and len(dv_f) >= flagpole_start:
                            pole_vol = dv_f[-flagpole_start-1]
                            flag_avg_vol = sum(dv_f[-flagpole_start+1:]) / max(1, flagpole_start - 1)
                            if flag_avg_vol < pole_vol * 0.7:  # flag volume < 70% of pole volume
                                bull_flag = True
    except Exception:
        pass

    # Trend reversal detection: stock transitioning from downtrend to uptrend
    # Highest expected-value moment — catching new uptrends early
    trend_reversal = False
    try:
        if len(daily) >= 25 and "Close" in daily.columns:
            dc_r = list(daily["Close"])
            e20_now  = _ema(dc_r, 20)
            e20_prev = _ema(dc_r[:-1], 20)
            if e20_now and e20_prev and e20_now > 0:
                # Price was below 20-EMA last bar but is now above it (MA crossover)
                price_above_now  = dc_r[-1] > e20_now
                price_below_prev = dc_r[-2] < e20_prev if e20_prev else False
                # RSI recovering: was below 40, now above 45
                rsi_now_r  = _rsi(dc_r, 14)
                rsi_prev_r = _rsi(dc_r[:-1], 14) if len(dc_r) > 14 else rsi_now_r
                rsi_recovery = rsi_prev_r < 42 and rsi_now_r > 45
                # Volume expansion on recovery day
                vol_expand = vol_ratio > 1.3 if vol_ratio else False
                # Confirm not still in deep downtrend (price within 15% of 20-day high)
                high_20d = max(dc_r[-20:]) if len(dc_r) >= 20 else dc_r[-1]
                near_high = dc_r[-1] > high_20d * 0.85
                if price_above_now and price_below_prev and rsi_recovery and vol_expand and near_high:
                    trend_reversal = True
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

    # Relative Volume (RVOL): today's volume vs 20-day average
    # RVOL > 2x = institutional participation confirmed; > 3x = strong institutional surge
    rvol = 1.0
    rvol_surge = False   # RVOL > 2.5 with positive price action
    try:
        if "Volume" in daily.columns and len(daily) >= 10:
            avg_vol_20 = float(daily["Volume"].tail(21).iloc[:-1].mean())  # exclude today
            today_vol = float(daily["Volume"].iloc[-1])
            if avg_vol_20 > 0:
                rvol = round(today_vol / avg_vol_20, 2)
                rvol_surge = rvol >= 2.5 and chg_pct > 0
    except Exception:
        pass

    # Opening Range Breakout (ORB): price clears the first-hour high on good volume
    # One of the most reliable institutional intraday signals — algos and funds chase ORBs
    orb_breakout = False
    try:
        if hourly is not None and "High" in hourly.columns and "Close" in hourly.columns:
            h_orb = hourly.dropna(subset=["Close"])
            if len(h_orb) >= 3 and "Volume" in h_orb.columns:
                # Opening range = first 1-2 bars of the session (today's early hours)
                # Use the first bar's high as the breakout level
                orb_high = float(h_orb["High"].iloc[0])
                cur_price_h = float(h_orb["Close"].iloc[-1])
                cur_vol_h   = float(h_orb["Volume"].iloc[-1])
                avg_vol_h   = float(h_orb["Volume"].mean())
                # ORB confirmed: price cleared first-hour high AND current volume is healthy
                if (cur_price_h > orb_high * 1.002       # at least 0.2% above ORB high
                        and cur_vol_h >= avg_vol_h * 0.6  # not just a thin-volume drift
                        and len(h_orb) >= 3):              # session is more than 1-2 bars old
                    orb_breakout = True
    except Exception:
        pass

    # Gap and hold: price opens with a gap up ≥1.5% vs yesterday's close and holds above it
    # Institutional confirmation — no one is selling into the gap
    gap_and_hold = False
    try:
        dc = list(daily["Close"])
        dc_open = list(daily["Open"]) if "Open" in daily.columns else []
        if len(dc) >= 2 and len(dc_open) >= 1:
            yesterday_close = dc[-2]
            today_open      = dc_open[-1]
            today_price     = dc[-1]
            gap_pct = (today_open - yesterday_close) / yesterday_close * 100
            # Gap up ≥ 1.5% and current price is still above 90% of the gap level
            gap_fill_level  = yesterday_close + (today_open - yesterday_close) * 0.90
            if gap_pct >= 1.5 and today_price >= gap_fill_level:
                gap_and_hold = True
    except Exception:
        pass

    # Intraday trend quality: higher intraday highs = price trending up vs fading
    intraday_trend_quality = 0.0  # positive = trending up all day, negative = fading
    try:
        if hourly is not None and "High" in hourly.columns and "Low" in hourly.columns:
            h_iq = hourly.dropna(subset=["Close"])
            if len(h_iq) >= 4:
                highs_iq = list(h_iq["High"])
                lows_iq  = list(h_iq["Low"])
                closes_iq = list(h_iq["Close"])
                # Count higher highs vs lower highs in the last 4 bars
                hh_count = sum(1 for i in range(1, min(4, len(highs_iq))) if highs_iq[-i] > highs_iq[-i-1])
                lh_count = sum(1 for i in range(1, min(4, len(highs_iq))) if highs_iq[-i] < highs_iq[-i-1])
                intraday_trend_quality = (hh_count - lh_count) / 3.0  # -1 to +1
    except Exception:
        pass

    # Multi-timeframe confluence: daily and hourly signals aligned?
    # True when BOTH daily and hourly confirm the same direction
    mtf_aligned = False   # daily uptrend + hourly uptrend = high confidence
    mtf_conflict = False  # daily downtrend + hourly uptrend = false signal risk
    try:
        _daily_up   = daily_trend > 0.2 and daily_rsi > 45    # daily uptrend
        _hourly_up  = ema_cross > 0.1 and rsi_val > 45        # hourly uptrend
        _daily_down = daily_trend < -0.2 and daily_rsi < 55
        mtf_aligned  = _daily_up and _hourly_up
        mtf_conflict = _daily_down and _hourly_up   # trying to buy against daily trend
    except Exception:
        pass

    # Relative strength vs SPY (1-day, 5-day, 63-day quarterly)
    spy  = _fetch_spy_perf()
    rs1  = round(chg_pct - spy.get("d1", 0), 2)   # outperformance vs SPY today
    rs5  = 0.0
    rs63 = 0.0
    try:
        dc = list(daily["Close"])
        if len(dc) >= 5:
            ret5 = (dc[-1] - dc[-5]) / dc[-5] * 100
            rs5  = round(ret5 - spy.get("d5", 0), 2)
        if len(dc) >= 63:
            ret63 = (dc[-1] - dc[-63]) / dc[-63] * 100
            rs63  = round(ret63 - spy.get("d63", 0), 2)
    except Exception:
        pass

    # Cup & Handle breakout pattern (O'Neil CAN SLIM)
    cup_handle = {"detected": False, "breakout_ready": False, "pivot_price": 0.0}
    try:
        if "High" in daily.columns and "Low" in daily.columns and "Volume" in daily.columns and len(daily) >= 45:
            cup_handle = _cup_and_handle(
                list(daily["High"]), list(daily["Low"]), list(daily["Close"]),
                volumes=list(daily["Volume"])
            )
    except Exception:
        pass

    # Supply / Demand zone detection
    sd_zones = {"at_demand": False, "at_supply": False}
    try:
        if "High" in daily.columns and "Low" in daily.columns and "Volume" in daily.columns and len(daily) >= 10:
            sd_zones = _supply_demand_zones(
                list(daily["High"]), list(daily["Low"]), list(daily["Close"]),
                list(daily["Volume"]), lookback=30
            )
    except Exception:
        pass

    # Volume Profile: Point of Control (POC) — highest-volume price level over last 30 bars
    vp_poc = {"poc_price": 0.0, "at_poc": False, "above_poc": False, "poc_breakout": False}
    try:
        if "High" in daily.columns and "Low" in daily.columns and "Volume" in daily.columns and len(daily) >= 10:
            vp_poc = _volume_profile_poc(
                list(daily["High"]), list(daily["Low"]), list(daily["Close"]),
                list(daily["Volume"]), lookback=30, bins=20
            )
    except Exception:
        pass

    # Momentum acceleration: ROC5 rising faster than prior ROC5 = early-stage breakout
    mom_accel = False
    try:
        dc_ma = list(daily["Close"])
        if len(dc_ma) >= 12:
            roc5_now  = _roc(dc_ma, 5)
            roc5_prev = _roc(dc_ma[:-5], 5)   # ROC5 as of 5 days ago
            # Accelerating upward by ≥2% and currently positive = trend gathering steam
            if roc5_now > roc5_prev + 2.0 and roc5_now > 1.5:
                mom_accel = True
    except Exception:
        pass

    # Higher Lows: ascending floor of support = established uptrend
    higher_lows = False
    try:
        if "Low" in daily.columns and len(daily) >= 10:
            higher_lows = _higher_lows_trend(list(daily["Low"]), list(daily["Close"]), lookback=20, min_pivots=2)
    except Exception:
        pass

    # Double Bottom: W-pattern bullish reversal
    double_bottom = False
    double_bottom_neckline = 0.0
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 30:
            _db = _double_bottom(list(daily["High"]), list(daily["Low"]), list(daily["Close"]))
            double_bottom = _db.get("detected", False)
            double_bottom_neckline = _db.get("neckline", 0.0)
    except Exception:
        pass

    # Double Top: M-pattern bearish reversal
    double_top = False
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 30:
            _dt = _double_top(list(daily["High"]), list(daily["Low"]), list(daily["Close"]))
            double_top = _dt.get("detected", False)
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
        "rs63":            rs63,
        "mtf_aligned":       mtf_aligned,
        "mtf_conflict":      mtf_conflict,
        "price_vs_ema200":   round(price_vs_ema200, 2),
        "trend_reversal":    trend_reversal,
        "bull_flag":         bull_flag,
        "atr":             round(atr_val, 3) if atr_val else None,
        "stoch_k":            round(stoch_k, 1),
        "stoch_d":            round(stoch_d, 1),
        "roc5":               round(roc5, 2),
        "roc20":              round(roc20, 2),
        "price_vs_ema50":     round(price_vs_ema50, 2),
        "ema_stacked_bull":   ema_stacked_bull,
        "ema_stacked_bear":   ema_stacked_bear,
        "williams_r":         round(williams_r, 1),
        "macd_slope":         round(macd_slope_val, 4),
        "ttm_squeeze_fired":  ttm_squeeze_fired,
        "vwap_z":             round(vwap_z, 2),
        "rsi_divergence":      rsi_divergence,
        "rsi_bull_divergence": rsi_bull_divergence,
        "consec_green":        consec_green,
        "vol_dry_up":          vol_dry_up,
        "rvol":                rvol,
        "rvol_surge":          rvol_surge,
        "at_breakout":         at_breakout,
        "near_support":        near_support,
        "nr7_signal":          nr7_signal,
        "inside_bar":          inside_bar,
        "vwap_reclaim":        vwap_reclaim,
        "orb_breakout":        orb_breakout,
        "gap_and_hold":        gap_and_hold,
        "adx":                 round(adx_val, 1),
        "intraday_tq":         round(intraday_trend_quality, 2),
        "ichimoku_above":      ichimoku.get("above_cloud", False),
        "ichimoku_bull_cloud": ichimoku.get("cloud_bullish", False),
        "ichimoku_tk_bull":    ichimoku.get("tk_ks_bullish", False),
        "ichimoku_chikou":     ichimoku.get("chikou_bullish", False),
        "mfi":                 round(mfi_val, 1),
        "mfi_oversold":        mfi_oversold,
        "mfi_overbought":      mfi_overbought,
        "mfi_bull_div":        mfi_bull_div,
        "supertrend_bull":     supertrend_bull,
        "supertrend_stop":     supertrend_stop,
        "supertrend_dir":      supertrend_dir,
        "force_index":         fi_val,
        "force_index_rising":  fi_rising,
        "force_index_div":     fi_bull_div,
        "true_beta":           true_beta,
        "true_alpha":          true_alpha,
        "ha_bull":             ha_bull,
        "ha_bear":             ha_bear,
        "ha_consec_bull":      ha_consec_bull,
        "donchian_up":         donchian_up,
        "donchian_down":       donchian_down,
        "donchian_pct":        donchian_pct,
        "hammer":              candle_patterns.get("hammer", False),
        "bullish_engulfing":   candle_patterns.get("bullish_engulfing", False),
        "morning_star":        candle_patterns.get("morning_star", False),
        "shooting_star":       candle_patterns.get("shooting_star", False),
        "bearish_engulfing":   candle_patterns.get("bearish_engulfing", False),
        "three_white_soldiers":candle_patterns.get("three_white_soldiers", False),
        "three_black_crows":   candle_patterns.get("three_black_crows", False),
        "pivot":               pivot_levels.get("pivot", 0),
        "pivot_r1":            pivot_levels.get("r1", 0),
        "pivot_r2":            pivot_levels.get("r2", 0),
        "pivot_s1":            pivot_levels.get("s1", 0),
        "pivot_s2":            pivot_levels.get("s2", 0),
        "fib_support":         fib_support,
        "fib_resistance":      fib_resistance,
        "macd_bull_div":       macd_div.get("bullish_div", False),
        "macd_bear_div":       macd_div.get("bearish_div", False),
        "chandelier_stop":     chandelier_stop,
        "kc_pos":              round(kc_pos, 1),
        "kc_breakout":         kc_breakout,
        "kc_oversold":         kc_oversold,
        "obv_rising":          obv_rising,
        "obv_slope_pct":       round(obv_slope_pct, 1),
        "vcp":                 vcp,
        "cup_handle":          cup_handle.get("breakout_ready", False),
        "cup_handle_pivot":    cup_handle.get("pivot_price", 0.0),
        "cup_depth_pct":       cup_handle.get("cup_depth_pct", 0.0),
        "at_demand_zone":      sd_zones.get("at_demand", False),
        "at_supply_zone":      sd_zones.get("at_supply", False),
        "poc_price":           vp_poc.get("poc_price", 0.0),
        "at_poc":              vp_poc.get("at_poc", False),
        "above_poc":           vp_poc.get("above_poc", False),
        "poc_breakout":        vp_poc.get("poc_breakout", False),
        "mom_accel":           mom_accel,
        "higher_lows":            higher_lows,
        "double_bottom":          double_bottom,
        "double_bottom_neckline": double_bottom_neckline,
        "double_top":             double_top,
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


def fetch_batch(tickers, held_symbols=None, period_d="90d"):
    """
    Three-phase scan:
      Phase 0 — Alpaca Snapshot API real-time filter: drop stocks with tiny move + low volume
      Phase 1 — quick 5d yfinance download for shortlisted tickers → rank by momentum × volume
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

    # ── Phase 0: Alpaca real-time snapshot pre-filter ─────────────────────
    # Use Alpaca's live data to pre-filter the universe before slow yfinance calls.
    # This trims low-volume/flat stocks without downloading historical data.
    p0_snaps = {}
    try:
        p0_snaps = alpaca_snapshots(tickers)
        if p0_snaps:
            # Keep stocks with notable price action OR held positions
            # Threshold: any move >0.3% OR vol_ratio > 1.2 OR held
            p0_pass = [tk for tk in tickers if
                       tk in held
                       or tk not in p0_snaps   # unknown = keep for safety
                       or abs(p0_snaps[tk]["chg_pct"]) > 0.3
                       or p0_snaps[tk]["vol_ratio_est"] > 1.2
                       or p0_snaps[tk]["volume"] > MIN_AVG_VOL * 0.3]
            logger.info(
                f"Phase 0 (Alpaca snapshot): {len(p0_snaps)} quotes, "
                f"{len(p0_pass)}/{len(tickers)} pass activity filter"
            )
            tickers = p0_pass
    except Exception as e:
        logger.debug(f"Phase 0 snapshot skipped: {e}")

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


# ── Momentum Grade: letter grade for signal quality ──────────────────────────
def momentum_grade(d, final_score=0):
    """
    Assign A+/A/B/C/D/F grade based on signal quality criteria.
    Based on how many institutional-quality signals are present.
    """
    criteria = 0
    # Trend quality
    if d.get("price_vs_ema200", 0) > 3:       criteria += 1  # above 200 EMA
    if d.get("price_vs_ema50", 0) > 0:        criteria += 1  # above 50 EMA
    if d.get("mtf_aligned", False):            criteria += 2  # daily + hourly aligned (double weight)
    # Momentum
    if (d.get("daily_rsi", 50) or 50) > 50:   criteria += 1  # daily RSI bullish
    if (d.get("adx", 0) or 0) >= 25:          criteria += 1  # strong trend
    if (d.get("rs5", 0) or 0) > 2:            criteria += 1  # outperforming SPY
    if (d.get("rs63", 0) or 0) > 5:           criteria += 1  # quarterly leader
    # Setup quality
    if d.get("ichimoku_above", False):         criteria += 1  # Ichimoku bullish
    if d.get("ttm_squeeze_fired", False):      criteria += 1  # squeeze breakout
    if d.get("fib_support", False):            criteria += 1  # Fibonacci support
    if d.get("macd_bull_div", False):          criteria += 1  # MACD divergence
    if d.get("at_breakout", False):            criteria += 1  # technical breakout
    if d.get("vwap_reclaim", False):           criteria += 1  # VWAP reclaim
    if d.get("trend_reversal", False):         criteria += 1  # 20-EMA crossover with volume
    if d.get("cup_handle", False):             criteria += 2  # cup & handle (double weight — elite pattern)
    if d.get("at_demand_zone", False):         criteria += 1  # at institutional demand zone
    if d.get("mom_accel", False):              criteria += 1  # momentum accelerating
    if d.get("higher_lows", False):           criteria += 1  # ascending support structure
    if d.get("double_bottom", False):         criteria += 1  # W-pattern bullish reversal
    if d.get("ema_stacked_bull", False):      criteria += 2  # EMA stack aligned (high quality trend)
    if d.get("vcp", False):                   criteria += 2  # VCP: spring-loaded contraction base

    # Score contribution
    if   final_score >= 80: criteria += 2
    elif final_score >= 65: criteria += 1

    if   criteria >= 12: return "A+"
    elif criteria >= 10: return "A"
    elif criteria >= 8:  return "B+"
    elif criteria >= 6:  return "B"
    elif criteria >= 4:  return "C"
    elif criteria >= 2:  return "D"
    else:                return "F"


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
    rs63       = d.get("rs63",         0) or 0   # 63-day RS vs SPY (quarterly — O'Neil style)

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

    # 63-day (quarterly) RS: O'Neil IBD-style — strongest stocks sustain leadership (+10/-6)
    if   rs63 > 15:  s += 10   # top-tier quarterly leader
    elif rs63 > 8:   s +=  6
    elif rs63 > 3:   s +=  3
    elif rs63 < -15: s -=  6   # persistent underperformer — avoid
    elif rs63 < -8:  s -=  3

    # Multi-timeframe trend filter: daily EMA5/10 alignment (+8/-10)
    if   daily_tr > 0.5:  s +=  8
    elif daily_tr > 0.1:  s +=  4
    elif daily_tr < -0.5: s -= 10
    elif daily_tr < -0.1: s -=  5

    # 200-day EMA position: institutional trend filter (+8/-12)
    # Above 200 EMA = institutional money in buy mode; below = distribution/bear phase
    ema200_pos = d.get("price_vs_ema200", 0) or 0
    if   ema200_pos > 10:   s +=  8   # clearly in bull territory
    elif ema200_pos > 3:    s +=  4
    elif ema200_pos < -10:  s -= 12   # bear territory — very cautious
    elif ema200_pos < -3:   s -=  6

    # Multi-timeframe confluence: bonus when daily+hourly both confirm direction (+12/-10)
    # MTF alignment is the single strongest quality filter — reduces false signals significantly
    if d.get("mtf_aligned", False):   s += 12   # daily uptrend + hourly uptrend = highest conviction
    if d.get("mtf_conflict", False):  s -= 10   # fighting the daily trend = high failure rate

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
    roc20     = d.get("roc20",              0) or 0
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

    # 20-day rate of change: medium-term trend confirmation (+6/-6)
    if   roc20 > 10:  s +=  6
    elif roc20 >  5:  s +=  3
    elif roc20 < -10: s -=  6
    elif roc20 <  -5: s -=  3

    # EMA stack: price > EMA5 > EMA10 > EMA20 > EMA50 = full bullish alignment (+8)
    # Mirror of institutional "trend qualification" — all timeframes agree
    if d.get("ema_stacked_bull", False): s += 8
    if d.get("ema_stacked_bear", False): s -= 8

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

    # Multi-timeframe confirmation: both hourly AND daily RSI aligned (+12/-10)
    # When both timeframes agree, signal reliability increases dramatically
    if rsi > 50 and daily_rsi > 50 and ema_c > 0 and daily_tr > 0:
        s += 12   # all four timeframe signals bullish = very high confidence
    elif rsi > 50 and daily_rsi > 50:
        s +=  6   # both RSIs bullish
    elif rsi < 40 and daily_rsi < 40 and ema_c < 0 and daily_tr < 0:
        s -= 10   # both timeframes bearish = strong avoid signal

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

    # Fibonacci retracement support: bouncing off 38.2/50/61.8% level = institutional buy zone (+9)
    if d.get("fib_support", False):    s += 9
    if d.get("fib_resistance", False): s -= 5

    # MACD divergence: price and momentum disagree — leading reversal signal (+11/-8)
    if d.get("macd_bull_div", False): s += 11   # price lower, MACD higher = hidden bullish strength
    if d.get("macd_bear_div", False): s -= 8    # price higher, MACD lower = momentum weakening

    # Trend reversal: price just crossed above 20-EMA with RSI recovery + volume expansion (+13)
    # Catches new uptrends at their earliest stage — highest expected value entry point
    if d.get("trend_reversal", False): s += 13

    # Bull flag: flagpole (5%+ surge) + tight consolidation (declining volume) = continuation (+12)
    # Institutions let weak hands shake out, then breakout resumes the original trend
    if d.get("bull_flag", False): s += 12

    # NR7 / inside bar: volatility contraction before expansion (+8/+6)
    # These patterns precede large directional moves — buy the squeeze
    if d.get("nr7_signal", False): s += 8
    if d.get("inside_bar", False): s += 6

    # VWAP reclaim: price dipped below VWAP intraday then reclaimed — institutions stepped in (+14)
    # One of the highest-conviction intraday signals; stops triggered below VWAP then buyers return
    if d.get("vwap_reclaim", False): s += 14

    # Opening Range Breakout: cleared the first-hour high on volume — algo algos and funds chase (+13)
    if d.get("orb_breakout", False): s += 13

    # Gap and hold: gapped up 1.5%+ and is holding above the gap — buyers own the tape (+10)
    if d.get("gap_and_hold", False): s += 10

    # ADX trend strength: high ADX confirms momentum signals, low ADX = choppy market
    adx = d.get("adx", 0) or 0
    if adx >= 35:  s += 8    # very strong trend — momentum strategies highly reliable
    elif adx >= 25: s += 4   # trending — signals more reliable
    elif adx < 15: s -= 5    # choppy market — signals less reliable

    # Intraday trend quality: stock making higher highs all day = strong conviction (+8/-5)
    itq = d.get("intraday_tq", 0) or 0
    if   itq >= 0.66:  s += 8   # 2+ higher highs — clear uptrend all day
    elif itq >= 0.33:  s += 4   # 1 higher high — mild uptrend
    elif itq <= -0.66: s -= 5   # 2+ lower highs — fading/distribution pattern

    # Ichimoku Cloud — comprehensive trend confirmation (up to +14 for full alignment)
    ichi_above = d.get("ichimoku_above", False)
    ichi_bull  = d.get("ichimoku_bull_cloud", False)
    ichi_tk    = d.get("ichimoku_tk_bull", False)
    ichi_chi   = d.get("ichimoku_chikou", False)
    ichi_signals = sum([ichi_above, ichi_bull, ichi_tk, ichi_chi])
    if   ichi_signals == 4: s += 14   # full Ichimoku alignment = highest conviction
    elif ichi_signals == 3: s +=  9
    elif ichi_signals == 2: s +=  4
    elif ichi_signals == 0 and d.get("ichimoku_above") is not None:
        # Explicitly computed and all signals bearish
        s -= 5

    # Cup & Handle breakout: highest-probability institutional continuation pattern (+15)
    # O'Neil's #1 pattern — found in virtually every winning stock before a big move
    if d.get("cup_handle", False): s += 15

    # Supply/Demand zone: price at institutional accumulation zone (+8) or distribution (-7)
    if d.get("at_demand_zone", False): s += 8
    if d.get("at_supply_zone", False): s -= 7

    # Volume Profile POC: price above POC = institutions are net-long (+6)
    # POC breakout = reclaimed highest-volume node, highly bullish (+10)
    if d.get("poc_breakout", False): s += 10
    elif d.get("above_poc", False):  s +=  6
    elif d.get("at_poc", False):     s +=  4

    # Momentum acceleration: ROC5 rising faster than prior ROC5 (+10)
    # Catches stocks early in a new breakout — before everyone piles in
    if d.get("mom_accel", False): s += 10

    # Volatility Contraction Pattern (VCP): Minervini's spring-loaded base setup (+12)
    # 3 contracting price segments + volume dry-up = stock ready to explode higher
    if d.get("vcp", False): s += 12

    # Relative Volume surge (RVOL): today's volume ≥2.5× 20-day avg + positive price action
    # = institutional participation confirmed — big money is chasing this move (+8)
    if d.get("rvol_surge", False): s += 8
    elif (d.get("rvol", 1) or 1) >= 1.8: s += 4  # moderate relative volume boost

    # On-Balance Volume: rising OBV = institutional accumulation (smart money buying) (+7)
    if d.get("obv_rising", False): s += 7

    # Money Flow Index (MFI): volume-weighted RSI
    # MFI < 20 = oversold accumulation (smart money buying quietly): +8
    # MFI bull divergence = price falling but money flowing IN: +7
    # MFI > 80 = overbought distribution: -4 (reduce conviction)
    mfi = d.get("mfi", 50) or 50
    if d.get("mfi_bull_div", False):   s += 7
    elif d.get("mfi_oversold", False): s += 8
    elif d.get("mfi_overbought", False): s -= 4

    # Supertrend: ATR-based dynamic trend confirmation
    # Price above Supertrend line = bullish trend confirmed (+7)
    # Price below Supertrend = bearish trend active (-5)
    if d.get("supertrend_bull", False):         s += 7
    elif d.get("supertrend_dir", 1) == -1:      s -= 5

    # Elder's Force Index: institutional buying pressure detector
    # FI bull divergence = price falling but institutional volume flowing IN: +8
    # FI rising = sustained buying force behind move: +4
    if d.get("force_index_div", False):         s += 8
    elif d.get("force_index_rising", False):    s += 4

    # Beta-quality scoring: high alpha stocks (outperforming on risk-adjusted basis)
    # Jensen's alpha > 10% annualized = stock consistently beats market after beta adjustment
    true_alpha_v = d.get("true_alpha", 0) or 0
    if true_alpha_v > 15:   s += 6   # exceptional alpha generator
    elif true_alpha_v > 8:  s += 3   # solid alpha

    # Heikin-Ashi trend: 3+ consecutive HA bull candles = clean institutional uptrend (+7)
    # 5+ consecutive = extremely strong trend (+10); bearish HA = -4
    ha_c = d.get("ha_consec_bull", 0) or 0
    if d.get("ha_bull", False):
        if ha_c >= 5:   s += 10
        elif ha_c >= 3: s += 7
    elif d.get("ha_bear", False): s -= 4

    # Donchian Channel breakout: 20-day high breakout = turtle-trading momentum signal (+8)
    # High position in channel (>80%): approaching resistance but still strong (+3)
    if d.get("donchian_up", False):                               s += 8
    elif (d.get("donchian_pct", 50) or 50) >= 80:                s += 3
    elif d.get("donchian_down", False):                           s -= 6

    # Keltner Channel breakout: price above upper band = strong institutional momentum (+9)
    # Oversold: price below lower band = mean-reversion setup (+5 for oversold bounces)
    if d.get("kc_breakout", False): s += 9
    elif d.get("kc_oversold", False): s += 5   # mean-reversion candidate

    # Candlestick patterns: institutional reversal and continuation signals
    # Bullish patterns (entry confirmation)
    if d.get("three_white_soldiers", False): s += 8   # 3 bull candles = sustained buying
    if d.get("morning_star", False):          s += 7   # 3-candle bullish reversal
    if d.get("bullish_engulfing", False):     s += 6   # strong single-candle reversal
    if d.get("hammer", False):                s += 5   # hammer = buying at lows
    # Bearish patterns (reduce conviction for longs)
    if d.get("three_black_crows", False):     s -= 7
    if d.get("bearish_engulfing", False):     s -= 5
    if d.get("shooting_star", False):         s -= 4

    # Higher Lows: ascending support floor = confirmed uptrend structure (+6)
    if d.get("higher_lows", False): s += 6

    # Double Bottom: W-pattern bullish reversal (+10) — confirmed break above neckline
    if d.get("double_bottom", False): s += 10

    # Double Top: M-pattern bearish reversal at resistance (-8) — reduces buy conviction
    if d.get("double_top", False): s -= 8

    # Adaptive scoring: boost/penalize signals based on historical win rates
    # Uses accumulated signal_performance data to learn which signals actually work
    if _SIGNAL_WIN_RATES:
        _adaptive_adj = 0
        for sig_key in ["cup_handle", "vcp", "at_demand_zone", "mom_accel",
                        "obv_rising", "kc_breakout", "higher_lows", "double_bottom",
                        "poc_breakout", "ema_stacked_bull", "trend_reversal", "bull_flag"]:
            if d.get(sig_key) and sig_key in _SIGNAL_WIN_RATES:
                wr = _SIGNAL_WIN_RATES[sig_key].get("win_rate", 50)
                n  = _SIGNAL_WIN_RATES[sig_key].get("total", 0)
                if n >= 5:  # only adapt with enough data
                    # Scale: +60% wr→+3, 70%→+4, 80%→+5; -30% wr→-3, -20%→-2
                    adj = (wr - 50) / 10  # -5 to +5 range
                    adj = max(-5, min(5, adj))
                    _adaptive_adj += adj
        s += max(-8, min(8, round(_adaptive_adj)))  # cap total adaptive boost at ±8

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

    # Bearish versions of new signals
    rsi_div = d.get("rsi_divergence", False)
    daily_rsi = d.get("daily_rsi", 50) or 50
    vwap_z_v = d.get("vwap_z", 0) or 0
    consec = d.get("consec_green", 0) or 0

    # RSI bearish divergence on the short: price up but RSI falling = confirmation
    if rsi_div: s += 12
    # VWAP exhaustion zone: price far above VWAP = short entry
    if vwap_z_v > 2.5: s += 10
    elif vwap_z_v > 1.5: s += 5
    # Many consecutive green candles = overbought, likely to mean-revert
    if   consec >= 5: s += 8   # extended run = eventual reversal
    elif consec >= 7: s += 12  # very extended = higher short conviction
    # Daily RSI overbought = bearish
    if daily_rsi > 75: s += 8
    elif daily_rsi > 65: s += 4
    elif daily_rsi < 35: s -= 8  # oversold daily = don't short here

    # Double Top: M-pattern confirmed = bearish reversal signal (+10 for shorts)
    if d.get("double_top", False): s += 10
    # Double Bottom: W-pattern = bullish reversal, reduces short conviction
    if d.get("double_bottom", False): s -= 8

    # POC: below POC = institutions net-short, bearish (+8 for shorts)
    if not d.get("above_poc", True) and not d.get("at_poc", False): s += 8
    # POC breakout above = don't short into strength
    if d.get("poc_breakout", False): s -= 10

    # EMA stack: fully stacked bear = reliable short setup (+8), bull = don't short (-8)
    if d.get("ema_stacked_bear", False): s += 8
    if d.get("ema_stacked_bull", False): s -= 8

    return max(0, min(100, int(s)))


# ── Position sizing ───────────────────────────────────────────────────────────
def calc_notional(portfolio_val, buying_power, price, atr, vix=20.0, macro_day=False,
                  score_val=0, win_rate=0.5, drawdown_pct=0.0, payoff_ratio=1.5,
                  true_beta=1.0):
    """
    ATR-based risk sizing with full Kelly criterion and beta-adjusted position sizing.
    Full Kelly: f* = (W*B - L) / B  where W=win%, L=loss%, B=avg_win/avg_loss (payoff)
    Beta adjustment: high-beta stocks get smaller positions (equal-risk sizing).
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

    # Beta-adjusted position sizing: equal-risk allocation across different beta stocks
    # A 2.0-beta stock moves 2× the market — size it at 1/2 of a 1.0-beta position
    # for equal market-risk exposure. Cap beta effect between 0.5× and 1.25×.
    beta_adj = 1.0
    if true_beta and true_beta > 0:
        beta_adj = max(0.5, min(1.25, 1.0 / true_beta))

    if atr and atr > 0 and price > 0:
        stop_dist   = 2 * atr
        dollar_risk = portfolio_val * RISK_PER_TRADE_PCT * vix_scale * beta_adj
        notional    = (dollar_risk / stop_dist) * price
    else:
        notional = portfolio_val * MAX_POSITION_PCT * vix_scale * beta_adj

    # Full Kelly criterion for high-conviction signals
    # f* = (W*B - (1-W)) / B  where B = payoff ratio (avg_win / avg_loss)
    if score_val >= 75 and win_rate > 0.50 and payoff_ratio > 0:
        W = win_rate
        B = max(0.5, min(5.0, payoff_ratio))   # cap payoff ratio 0.5-5x
        kelly_f = (W * B - (1 - W)) / B        # full Kelly fraction
        kelly_f = max(0, min(0.25, kelly_f))   # cap at 25% (quarter-Kelly for safety)
        if kelly_f > 0:
            # Kelly bonus: scale notional up by kelly fraction proportional to score
            score_boost = (score_val - 75) / 25   # 0→1 as score goes 75→100
            kelly_scale = 1 + kelly_f * score_boost
            notional = min(notional * kelly_scale, portfolio_val * MAX_POSITION_PCT * 1.5)

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

    # Cancel stale open orders: limit orders from prior cycles that didn't fill,
    # and orphaned bracket stop-loss orders from positions that were already sold
    try:
        open_orders = alpaca_get("/v2/orders?status=open&limit=100")
        held_syms_quick = {p["symbol"] for p in alpaca_get("/v2/positions")}
        _cancelled_orders = 0
        for o in (open_orders or []):
            sym   = o.get("symbol", "")
            otype = o.get("type", "")
            oclass= o.get("order_class", "")
            # Cancel: (1) day-limit orders older than 10 min that haven't filled
            # (2) orphaned child stop orders for positions we no longer hold
            created = o.get("created_at", "")
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - created_dt).total_seconds() / 60
            except Exception:
                age_min = 0
            is_orphan_stop = (oclass == "bracket" or otype in ("stop", "stop_limit")) and sym not in held_syms_quick
            is_stale_limit = otype == "limit" and age_min > 12
            if is_orphan_stop or is_stale_limit:
                try:
                    requests.delete(f"{ALPACA_BASE}/v2/orders/{o['id']}", headers=_h(), timeout=10)
                    _cancelled_orders += 1
                except Exception:
                    pass
        if _cancelled_orders:
            logger.info(f"Cancelled {_cancelled_orders} stale open orders")
    except Exception as _oe:
        logger.debug(f"Open order cleanup: {_oe}")

    # Pre-cache SPY performance for relative strength calculations
    _fetch_spy_perf()

    # Market regime
    regime    = market_regime()
    vix       = regime["vix"]
    macro_day = near_macro_event(days_before=1)
    if vix > VIX_EXTREME_THRESH:
        logger.warning(f"VIX={vix:.0f} EXTREME — halting new buys, protecting capital.")

    # VIX spike guard: if VIX jumped >25% from its 5-day average, the market is panicking
    # Skip new buys this cycle even if VIX hasn't hit the extreme threshold yet
    _vix_spike = False
    try:
        _vix_hist = yf.download("^VIX", period="10d", interval="1d",
                                auto_adjust=True, progress=False)
        if not _vix_hist.empty and len(_vix_hist) >= 5:
            _vix_vals  = list(_vix_hist["Close"].dropna())
            _vix_avg5  = sum(_vix_vals[-6:-1]) / 5 if len(_vix_vals) >= 6 else _vix_vals[-1]
            _vix_spike = vix > _vix_avg5 * 1.25 and vix > 20
            if _vix_spike:
                logger.warning(f"VIX spike guard: VIX={vix:.1f} vs 5d-avg={_vix_avg5:.1f} (+{(vix/_vix_avg5-1)*100:.0f}%) — buying cautiously")
    except Exception:
        pass

    # Portfolio drawdown guard — compute current drawdown from historical peak
    _prior_tlog  = _load(TRADES_FILE, {})
    _perf_hist   = _prior_tlog.get("perf_history", [])
    _hist_values = [h["v"] for h in _perf_hist if isinstance(h.get("v"), (int, float)) and h["v"] > 0]
    _peak_port   = max(_hist_values) if _hist_values else portfolio_val
    drawdown_pct = max(0.0, (_peak_port - portfolio_val) / _peak_port * 100) if _peak_port > 0 else 0.0
    if drawdown_pct > 2:
        logger.info(f"Portfolio drawdown: -{drawdown_pct:.1f}% from peak ${_peak_port:,.0f} — risk reduced")

    # Win rate + payoff ratio from trade history for full Kelly criterion
    _trade_stats = _prior_tlog.get("stats", {})
    _wins  = _trade_stats.get("wins",   0)
    _losses= _trade_stats.get("losses", 0)
    win_rate = _wins / max(1, _wins + _losses)
    # Compute actual payoff ratio from recent closed trades
    _avg_win_pct  = _prior_tlog.get("avg_win_pct",  0) or 0
    _avg_loss_pct = _prior_tlog.get("avg_loss_pct", 0) or 0
    _payoff_ratio = abs(_avg_win_pct / _avg_loss_pct) if _avg_loss_pct != 0 else 1.5

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

    # Internal scan breadth: how many of our scanned stocks are trending up?
    # This is a proprietary advance/decline ratio for our universe
    _scan_up   = sum(1 for sig in live.values() if (sig.get("change_pct", 0) or 0) > 0.3)
    _scan_down = sum(1 for sig in live.values() if (sig.get("change_pct", 0) or 0) < -0.3)
    _scan_total = max(1, _scan_up + _scan_down)
    _scan_adv_pct = round(_scan_up / _scan_total * 100, 1)
    logger.info(f"Internal scan breadth: {_scan_up}/{_scan_total} ({_scan_adv_pct}%) advancing")
    # If very few stocks are advancing in our universe, be more cautious with new buys
    _scan_breadth_poor = _scan_adv_pct < 30 and _scan_total > 20

    # Sector rotation (computed before AI context so it can be included in prompt)
    sector_adjs  = sector_rotation()   # {sector: -8..+8}

    # Sector ETF trend confirmation: identify bearish sectors to filter out buys
    sector_etf_trends = get_sector_etf_trend()

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
            adx_t = sig.get("adx", 0) or 0
            # Classic mean reversion: deeply oversold, bouncing on divergence, near EMA50 support
            # Low ADX (<20) = choppy, perfect for mean reversion; avoid trending stocks for MR
            if (stoch_k_t < 15 and rsi_t < 35 and ema50_t > -8
                    and (rsi_bull_t or vol_dry_t) and price_t >= 5
                    and roc5_t > -15 and adx_t < 30):   # not in a full collapse, not in strong downtrend
                mean_rev_cands.add(tk)
        if mean_rev_cands:
            logger.info(f"Mean-reversion bounce setups: {', '.join(sorted(mean_rev_cands))}")

    # Short squeeze detection — high short float + rising + volume surge → explosive upside
    squeeze_cands = get_squeeze_candidates(set(candidates)) if _time_ok(250) else set()

    # Volume surge candidates — 4x+ volume with price strength (institutional accumulation)
    vol_surge_cands = get_volume_surge_candidates(set(candidates)) if _time_ok(248) else set()
    if vol_surge_cands:
        logger.info(f"Volume surge candidates: {', '.join(sorted(vol_surge_cands))}")

    # Unusual options flow — detect institutional call buying (bullish signal)
    options_flow: dict = {}
    if _time_ok(240):
        # Check top movers + any stocks with already high scores
        flow_check = list(gap_ups | squeeze_cands | vol_surge_cands)[:15]
        flow_check += [s for s in candidates if s in BASE_UNIVERSE][:15]
        options_flow = get_options_flow_candidates(list(set(flow_check)), max_check=20)
    bullish_options = {s for s, d in options_flow.items() if d.get("bullish")}

    # Earnings beat plays — stocks that just beat estimates and are reacting positively
    earnings_beats   = get_earnings_beat_candidates(set(candidates)) if _time_ok(230) else set()
    pre_earn_cands   = get_pre_earnings_candidates(candidates[:50], live) if _time_ok(225) else set()

    # 52-week breakout screener — stocks at new annual highs with volume (O'Neil CAN SLIM)
    breakout_52w_cands = get_52w_breakout_candidates(set(candidates)) if _time_ok(220) else set()
    if breakout_52w_cands:
        logger.info(f"52W breakout candidates: {', '.join(sorted(breakout_52w_cands))}")

    tlog        = _load(TRADES_FILE, {"trades": [], "positions": [], "last_updated": ""})
    made_trades = False
    now_utc     = datetime.now(timezone.utc)

    # Load accumulated signal win rates for adaptive scoring
    global _SIGNAL_WIN_RATES
    _SIGNAL_WIN_RATES = tlog.get("signal_win_rates", {})

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

            # ── Position rebalancing: trim oversized positions ──────────────
            # If a position has grown to >1.5× MAX_POSITION_PCT of portfolio (due to appreciation),
            # sell 25% to bring it back into range. This protects against concentration risk.
            _mkt_val_check = current * qty
            _pos_pct_check = _mkt_val_check / portfolio_val * 100 if portfolio_val > 0 else 0
            if (_pos_pct_check > MAX_POSITION_PCT * 100 * 1.5
                    and pnl_pct > 10  # only trim winners, not losers
                    and qty >= 4       # enough shares to trim
                    and not peaks.get(sym, {}).get("half_out", False)):
                _trim_qty = round(qty * 0.25, 4)
                logger.info(f"TRIM {sym} — oversized ({_pos_pct_check:.1f}% of portfolio > {MAX_POSITION_PCT*150:.0f}%), selling 25%")
                try:
                    alpaca_post("/v2/orders", {
                        "symbol": sym, "qty": str(_trim_qty),
                        "side": "sell", "type": "market", "time_in_force": "day",
                    })
                    log_trade(tlog, "SELL_HALF", sym, current, _trim_qty,
                              pnl=pnl_pct, reason=f"rebalance trim ({_pos_pct_check:.1f}%>{MAX_POSITION_PCT*150:.0f}% of portfolio)")
                    made_trades = True
                except Exception as e:
                    logger.warning(f"Rebalance trim failed {sym}: {e}")

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
            _half_value = current * (qty / 2)  # value of half position
            if pnl_pct >= (PARTIAL_PROFIT_PCT * 100) and not half_out and _half_value >= 50:
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
            _lock_value = current * (qty * 0.75)  # value of 75% position
            if (pnl_pct >= 8 and not half_out and _lock_value >= 100):
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
            # At +25%: trail = 1.5% (lock in almost everything)
            # At +20%: trail = 1.8%
            # At +15%: trail = 2.0%
            # At +10%: trail = 3.0%
            # At +5%:  use default 5% trail
            # Below +5%: ATR-adaptive baseline (2.5× ATR, min 4%, max 9%)
            if   pnl_pct >= 25:  dyn_trail = 1.5
            elif pnl_pct >= 20:  dyn_trail = 1.8
            elif pnl_pct >= 15:  dyn_trail = 2.0
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
            # ATR-adaptive stop loss: 2.5× ATR from entry, capped at STOP_LOSS_PCT
            _atr_sig = live.get(sym, {})
            _atr_val = _atr_sig.get("atr") if _atr_sig else None
            if _atr_val and cost > 0:
                _atr_pct = _atr_val / cost * 100
                _atr_stop_pct = min(STOP_LOSS_PCT * 100, max(3.0, _atr_pct * 2.5))
            else:
                _atr_stop_pct = STOP_LOSS_PCT * 100
            # Chandelier Exit: if price closes below highest_close - 3×ATR, trend has reversed
            _chan_stop = _atr_sig.get("chandelier_stop", 0) or 0
            _use_chandelier = (_chan_stop > 0 and cost > 0 and pnl_pct > 2
                               and price < _chan_stop and age_days >= 3)
            # Supertrend exit: price crossed below Supertrend stop = trend reversal (high conviction)
            _st_stop  = _atr_sig.get("supertrend_stop", 0) or 0
            _st_bull  = _atr_sig.get("supertrend_bull", True)
            _use_supertrend = (not _st_bull and _st_stop > 0 and pnl_pct > 2
                               and age_days >= 2 and price < _st_stop * 1.01)
            # Pre-earnings sell: exit ANY position within 2 days of earnings to avoid binary risk
            # We rode the pre-earnings drift — now protect profits before the volatile event
            if has_earnings_soon(sym, days=2):
                reason = f"pre-earnings exit (earnings in <2d, {pnl_pct:+.1f}%)"
            elif _use_supertrend and not _use_chandelier:
                reason = f"supertrend reversal (price ${price:.2f} < stop ${_st_stop:.2f}, {pnl_pct:+.1f}%)"
            elif _use_chandelier:
                reason = f"chandelier exit (price ${price:.2f} < stop ${_chan_stop:.2f}, {pnl_pct:+.1f}%)"
            elif pnl_pct <= -_atr_stop_pct:
                reason = f"stop loss ({pnl_pct:+.1f}% ≤ -{_atr_stop_pct:.1f}%)"
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
            elif peaks.get(sym, {}).get("ever_hit_5pct") and pnl_pct <= -1.5 and age_days >= 1:
                reason = f"winner-turned-loser exit ({pnl_pct:+.1f}%, was up 5%+ earlier)"
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
                    elif (live_sig.get("vwap_pos", 0) or 0) < -1.0 and pnl_pct > 1 and age_days > 0.5:
                        # VWAP breakdown while in profit: institutional distribution signal
                        # Price dropped >1% below VWAP after being profitable = exit
                        vwap_p = live_sig.get("vwap_pos", 0) or 0
                        reason = f"VWAP breakdown ({vwap_p:.1f}% below, {pnl_pct:+.1f}%)"
                    elif (live_sig.get("adx", 0) or 0) < 15 and pnl_pct > 5 and (live_sig.get("daily_rsi", 50) or 50) > 70:
                        # ADX collapsed + overbought RSI = trend is exhausted, lock gains
                        adx_live = live_sig.get("adx", 0) or 0
                        reason = f"trend exhaustion (ADX={adx_live:.0f}, RSI overbought, {pnl_pct:+.1f}%)"
                    elif live_sig.get("ema_stacked_bear", False) and pnl_pct > 0 and age_days >= 1:
                        # All EMAs flipped bearish (EMA5<EMA10<EMA20<EMA50) — exit profitable position
                        reason = f"EMA stack turned bearish — exit profit ({pnl_pct:+.1f}%)"
                    elif live_sig.get("ema_stacked_bear", False) and pnl_pct <= -2 and age_days >= 0.5:
                        # EMA stack bearish while in a loss — cut early before it gets worse
                        reason = f"EMA stack bearish on losing position ({pnl_pct:+.1f}%)"
                    else:
                        # Proactive overbought exit: multiple overbought signals converging
                        d_rsi = live_sig.get("daily_rsi", 50) or 50
                        bb_pos_live = live_sig.get("bb_pos", 50) or 50
                        kc_break_live = live_sig.get("kc_breakout", False)
                        mfi_ob_live = live_sig.get("mfi_overbought", False)
                        st_bull     = live_sig.get("supertrend_bull", True)
                        ob_signals = sum([
                            d_rsi > 80,
                            stoch_k > 85,
                            bb_pos_live > 92,
                            kc_break_live and d_rsi > 75,
                            w_r_val > -10,
                            mfi_ob_live,              # MFI > 80 = distribution
                        ])
                        if ob_signals >= 4 and pnl_pct > 3 and not half_out:
                            reason = (
                                f"extreme overbought ({ob_signals}/6 signals) — "
                                f"RSI={d_rsi:.0f}, stoch={stoch_k:.0f}, BB={bb_pos_live:.0f}% "
                                f"MFI={live_sig.get('mfi',50):.0f} ({pnl_pct:+.1f}%)"
                            )
                        elif ob_signals >= 5 and pnl_pct > 1.5:
                            reason = (
                                f"overbought convergence ({ob_signals}/6) — "
                                f"proactive exit ({pnl_pct:+.1f}%)"
                            )

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

    # ── DCA: add to strong held positions on pullbacks OR VWAP reclaim ───────
    if not _open_guard and not _close_guard and vix <= VIX_EXTREME_THRESH:
        for sym, pos in list(longs.items()):
            try:
                cost    = float(pos.get("avg_entry_price", 0))
                qty     = float(pos.get("qty", 0))
                if cost <= 0 or qty <= 0:
                    continue
                current  = live.get(sym, {}).get("price", cost)
                pnl_pct  = (current - cost) / cost * 100
                mkt_val  = current * qty
                live_sig = live.get(sym, {})
                if not live_sig:
                    continue

                # Scenario A: Small loss pullback DCA (-5% to -1.5%)
                is_pullback_dca = -5.0 <= pnl_pct <= -1.5 and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.8

                # Scenario B: Winner pyramid — VWAP reclaim on a profitable position (+1% to +8%)
                # When a winner dips below VWAP then reclaims it, that's institutions adding on the dip
                is_winner_pyramid = (
                    1.0 <= pnl_pct <= 8.0
                    and live_sig.get("vwap_reclaim", False)
                    and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.9
                    and not peaks.get(sym, {}).get("half_out", False)   # don't pyramid after partial sell
                )

                if is_pullback_dca or is_winner_pyramid:
                    # Skip if stock is in a clear downtrend (EMA50 falling + negative ROC)
                    ema50_pos = live_sig.get("price_vs_ema50", 0) or 0
                    roc5_val  = live_sig.get("roc5", 0) or 0
                    if ema50_pos < -3 and roc5_val < -5:
                        logger.debug(f"DCA SKIP {sym} — downtrend (EMA50={ema50_pos:.1f}%, ROC5={roc5_val:.1f}%)")
                        continue
                    # Skip if full EMA stack is bearish — all timeframes declining
                    if live_sig.get("ema_stacked_bear", False):
                        logger.debug(f"DCA SKIP {sym} — EMA stack bearish (all EMAs declining)")
                        continue
                    # DCA only at equal or better price for pullback scenario
                    if is_pullback_dca and current >= cost * 1.01:
                        logger.debug(f"DCA SKIP {sym} — pullback DCA but price above cost (current={current:.2f}, cost={cost:.2f})")
                        continue
                    dca_sc = score(sym, live_sig, regime_adj=regime_adj)
                    min_score = 25 if is_winner_pyramid else 28
                    if dca_sc >= min_score:
                        # Pyramid adds are smaller (25% of current size vs 50%)
                        size_pct = 0.25 if is_winner_pyramid else 0.5
                        dca_notional = min(
                            mkt_val * size_pct,
                            portfolio_val * MAX_POSITION_PCT - mkt_val,
                            buying_power * 0.12,
                        )
                        if dca_notional >= 50:
                            dca_type = "pyramid (VWAP reclaim)" if is_winner_pyramid else "pullback"
                            logger.info(f"DCA {sym} [{dca_type}] — adding ${dca_notional:.0f} (pnl={pnl_pct:+.1f}%, score={dca_sc})")
                            r = alpaca_post("/v2/orders", {
                                "symbol": sym, "notional": str(round(dca_notional, 2)),
                                "side": "buy", "type": "market", "time_in_force": "day",
                            })
                            if r:
                                buying_power -= dca_notional
                                log_trade(tlog, "DCA", sym, current, dca_notional, score=dca_sc,
                                          reason=f"dca {dca_type} {pnl_pct:+.1f}%")
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

    # Portfolio beta estimation: estimate aggregate market exposure of open positions
    # Uses 63-day RS vs SPY as a beta proxy (positively correlated with actual beta)
    # If portfolio beta > 1.5 we're overexposed to market risk — raise buying threshold
    _port_beta_est = 0.0
    try:
        if longs:
            _beta_sum  = 0.0
            _beta_n    = 0
            _spy_perf  = _fetch_spy_perf()
            for _psym, _ppos in longs.items():
                _psig = live.get(_psym, {})
                _rs63 = _psig.get("rs63", 0) or 0
                _roc5 = _psig.get("roc5", 0) or 0
                _adx  = _psig.get("adx",  0) or 0
                # Rough beta proxy: high RS63 + high ROC + high ADX = high beta stock
                # Calibrated to approximate: beta = 1 + RS63/30 (a simple linear model)
                _beta_est = 1.0 + (_rs63 / 30.0) + (_roc5 / 50.0)
                _beta_est = max(0.3, min(2.5, _beta_est))
                _beta_sum += _beta_est
                _beta_n   += 1
            _port_beta_est = round(_beta_sum / max(1, _beta_n), 2)
            if _port_beta_est > 1.5:
                logger.info(f"Portfolio beta estimate: {_port_beta_est:.2f} (high-beta portfolio — risk elevated)")
    except Exception:
        pass

    # Dynamic score threshold: raise bar when market is fearful or below 200MA
    _vix_now_buy = vix or 20.0
    _above_200   = regime.get("above_200", True)
    if _vix_now_buy >= 30 or not _above_200:
        _eff_min_score = MIN_BUY_SCORE + 15   # need much stronger signal in bear market
        logger.info(f"Regime guard: raising MIN_BUY_SCORE to {_eff_min_score} (VIX={_vix_now_buy:.0f}, above200={_above_200})")
    elif _vix_now_buy >= 22:
        _eff_min_score = MIN_BUY_SCORE + 8    # elevated uncertainty
    else:
        _eff_min_score = MIN_BUY_SCORE

    # Internal scan breadth guard: if <30% of our universe is advancing, add +6 to threshold
    if _scan_breadth_poor:
        _eff_min_score += 6
        logger.info(f"Scan breadth guard: only {_scan_adv_pct}% advancing — raising threshold to {_eff_min_score}")

    # VIX spike guard: if VIX is spiking rapidly, add extra +8 to threshold
    if _vix_spike:
        _eff_min_score += 8
        logger.info(f"VIX spike guard active — raising threshold to {_eff_min_score}")

    # Portfolio beta guard: high-beta portfolio + new high-beta buy = double the risk
    # Add +5 to threshold when portfolio is already high-beta (>1.5 estimate)
    if _port_beta_est > 1.5:
        _eff_min_score += 5
        logger.info(f"Portfolio beta guard: beta≈{_port_beta_est:.2f} — raising threshold to {_eff_min_score}")

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

        # Signal persistence: track consecutive scans where a stock appears in top candidates
        # 1 run = +9 bonus, 2 runs = +13 bonus, 3+ runs = +17 bonus (sustained accumulation)
        prev_top = {entry.get("ticker"): entry.get("score", 0)
                    for entry in tlog.get("last_scan_top", [])
                    if entry.get("ticker")}
        _prev_persist = tlog.get("last_scan_persistence", {})   # {ticker: consecutive_count}
        _curr_persist = {}
        for tk in prev_top:
            _curr_persist[tk] = _prev_persist.get(tk, 0) + 1
        tlog["last_scan_persistence"] = _curr_persist
        def _persist_bonus(tk):
            cnt = _curr_persist.get(tk, 0)
            if   cnt >= 3: return 17
            elif cnt >= 2: return 13
            elif cnt >= 1: return 9
            return 0
        _persistent_cands = {tk for tk in prev_top if prev_top[tk] >= MIN_BUY_SCORE}
        if _persistent_cands:
            logger.info(f"Persistent signal candidates: {' | '.join(f'{tk}({_curr_persist.get(tk,0)}runs)' for tk in sorted(_persistent_cands))}")

        # Technical pass — include sector rotation + gap + squeeze + earnings + mean-rev bonuses
        tech_scores = {
            tk: score(tk, live[tk],
                      regime_adj=regime_adj + sector_adjs.get(SECTOR_MAP.get(tk, "other"), 0)
                                + (10 if tk in gap_ups else 0)
                                + (12 if tk in squeeze_cands else 0)
                                + (11 if tk in vol_surge_cands else 0)
                                + (8  if tk in recent_sells else 0)
                                + _persist_bonus(tk)                       # 0/9/13/17 by run count
                                + (14 if tk in bullish_options else 0)
                                + (18 if tk in earnings_beats else 0)
                                + (12 if tk in pre_earn_cands else 0)      # pre-earnings drift
                                + (10 if tk in mean_rev_cands else 0)      # mean reversion bounce
                                + (15 if tk in breakout_52w_cands else 0)) # 52-week high breakout
            for tk in live if tk not in held
        }
        candidates_buy = sorted(
            [(tk, sc) for tk, sc in tech_scores.items() if sc >= _eff_min_score - 5],
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
        final_scores  = []
        _rejected_log = []   # track rejections with reasons for dashboard
        for tk, tech_sc in candidates_buy:
            sec = SECTOR_MAP.get(tk, "other")
            if sector_counts.get(sec, 0) >= MAX_SECTOR_LONGS:
                logger.info(f"SKIP {tk} — sector {sec} full ({sector_counts.get(sec,0)}/{MAX_SECTOR_LONGS})")
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": f"sector {sec} full"})
                continue
            # Sector ETF confirmation: skip individual stock buy if the whole sector is bearish
            # Exception: mean-reversion setups in bearish sectors are fine (buying the dip)
            _sec_etf = sector_etf_trends.get(sec, {})
            if _sec_etf and not _sec_etf.get("bullish", True) and tk not in mean_rev_cands:
                _etf_5d = _sec_etf.get("chg5d", 0)
                # Only block if sector is truly falling (not just underperforming)
                if _etf_5d < -3.0:
                    logger.info(f"SKIP {tk} — sector {sec} ETF falling ({_etf_5d:+.1f}%5d)")
                    _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": f"sector ETF selling off ({_etf_5d:+.1f}%5d)"})
                    continue
            if has_earnings_soon(tk):
                logger.info(f"SKIP {tk} — earnings within 3 days")
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": "earnings in <3d"})
                continue
            # Minimum price filter: skip penny stocks (wide spreads, unreliable fills)
            _d_pre = live.get(tk, {})
            _price_pre = _d_pre.get("price", 0) or 0
            if _price_pre < 2.0:
                logger.debug(f"SKIP {tk} — price ${_price_pre:.2f} < $2 minimum")
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": f"price ${_price_pre:.2f} too low"})
                continue
            # Minimum average volume filter: 100k shares/day to ensure liquidity
            _avg_vol_pre = _d_pre.get("avg_vol_14", 0) or 0
            if 0 < _avg_vol_pre < 100_000:
                logger.debug(f"SKIP {tk} — avg volume {_avg_vol_pre:,} < 100k minimum")
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": f"low liquidity ({_avg_vol_pre//1000}k avg vol)"})
                continue
            # Correlation guard: skip if >0.85 correlated with a held position
            _held_syms = [s for s in longs if s != tk]
            if is_correlated_with_held(tk, _held_syms, threshold=0.85):
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": "corr≥0.85 w/ held position"})
                continue
            # Use Sonnet for top 3 candidates (better reasoning), Haiku for rest
            rank = len(final_scores)
            use_sonnet = (rank < 3) and _time_ok(200)
            if _time_ok(280):
                sent, catalyst = ai_sentiment(tk, use_sonnet=use_sonnet, signals=live.get(tk))
            else:
                sent, catalyst = 0, ""
            sec_adj        = sector_adjs.get(sec, 0)
            gap_adj        = 10 if tk in gap_ups else 0
            squeeze_adj    = 12 if tk in squeeze_cands else 0
            vol_surge_adj  = 11 if tk in vol_surge_cands else 0
            options_adj    = 14 if tk in bullish_options else 0
            reentry_adj    = 8  if tk in recent_sells else 0
            persist_adj    = _persist_bonus(tk)
            earnings_adj   = 18 if tk in earnings_beats else 0
            pre_earn_adj   = 12 if tk in pre_earn_cands else 0
            mean_rev_adj   = 10 if tk in mean_rev_cands else 0
            breakout_adj   = 15 if tk in breakout_52w_cands else 0
            final_sc       = score(tk, live[tk], sentiment=sent,
                                   regime_adj=regime_adj + sec_adj + gap_adj + squeeze_adj
                                             + vol_surge_adj + options_adj + reentry_adj
                                             + persist_adj + earnings_adj + pre_earn_adj + mean_rev_adj
                                             + breakout_adj)
            # Grade-based threshold: A+ setups get -5 to threshold (elite quality)
            _grade_now = momentum_grade(live.get(tk, {}), final_sc)
            _grade_thresh = _eff_min_score - 5 if _grade_now == "A+" else _eff_min_score

            if final_sc >= _grade_thresh:
                final_scores.append((tk, final_sc, sent, sec, catalyst))
                extras = []
                if gap_adj:        extras.append("gap")
                if squeeze_adj:    extras.append("squeeze")
                if vol_surge_adj:  extras.append("vol-surge")
                if options_adj:    extras.append("call-flow")
                if reentry_adj:    extras.append("re-entry")
                if persist_adj:    extras.append(f"persistent×{_curr_persist.get(tk,0)}")
                if earnings_adj:   extras.append("earnings-beat")
                if pre_earn_adj:   extras.append("pre-earnings")
                if mean_rev_adj:   extras.append("mean-rev")
                if breakout_adj:   extras.append("52W-breakout")
                if live.get(tk, {}).get("cup_handle"):    extras.append("C&H")
                if live.get(tk, {}).get("mom_accel"):     extras.append("accel")
                if live.get(tk, {}).get("higher_lows"):    extras.append("HL↑")
                if live.get(tk, {}).get("double_bottom"):  extras.append("2-BTM")
                if live.get(tk, {}).get("double_top"):     extras.append("2-TOP")
                if live.get(tk, {}).get("poc_breakout"):    extras.append("POC-BRK")
                elif live.get(tk, {}).get("above_poc"):    extras.append("abv-POC")
                if live.get(tk, {}).get("ema_stacked_bull"): extras.append("EMA-stack")
                if live.get(tk, {}).get("vcp"):               extras.append("VCP")
                if live.get(tk, {}).get("obv_rising"):        extras.append("OBV↑")
                if live.get(tk, {}).get("rvol_surge"):        extras.append(f"RVOL{live.get(tk,{}).get('rvol',1):.1f}x")
                if live.get(tk, {}).get("mfi_bull_div"):      extras.append(f"MFI-div{live.get(tk,{}).get('mfi',50):.0f}")
                elif live.get(tk, {}).get("mfi_oversold"):    extras.append(f"MFI-OS{live.get(tk,{}).get('mfi',50):.0f}")
                if live.get(tk, {}).get("supertrend_bull"):   extras.append("ST-BULL")
                if live.get(tk, {}).get("force_index_div"):     extras.append("FI-DIV")
                elif live.get(tk, {}).get("force_index_rising"): extras.append("FI↑")
                if live.get(tk, {}).get("ha_bull"):              extras.append(f"HA×{live.get(tk,{}).get('ha_consec_bull',0)}")
                if live.get(tk, {}).get("donchian_up"):          extras.append("DON-BRK")
                if live.get(tk, {}).get("kc_breakout"):      extras.append("KC-BRK")
                elif live.get(tk, {}).get("kc_oversold"):    extras.append("KC-OVS")
                if _grade_now == "A+":              extras.append("GRADE:A+")
                logger.info(f"  {tk}: tech={tech_sc} sent={sent:+.1f} final={final_sc} grade={_grade_now} sec={sec} cat='{catalyst}' [{','.join(extras) or 'base'}]")
            else:
                _rejected_log.append({"ticker": tk, "score": final_sc,
                                       "reason": f"AI sent={sent:+.0f} → final {final_sc} < {_eff_min_score}"})

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
                "orb_breakout":   live.get(tk, {}).get("orb_breakout", False),
                "gap_and_hold":   live.get(tk, {}).get("gap_and_hold", False),
                "persist_count":  _curr_persist.get(tk, 0),
                "vol_surge":      tk in vol_surge_cands,
                "pre_earnings":   tk in pre_earn_cands,
                "options_flow":   tk in bullish_options,
                "rsi_bull_div":   live.get(tk, {}).get("rsi_bull_divergence", False),
                "breakout_52w":   tk in breakout_52w_cands,
                "earnings_beat":  tk in earnings_beats,
                "ichimoku":       sum([live.get(tk,{}).get("ichimoku_above", False),
                                       live.get(tk,{}).get("ichimoku_bull_cloud", False),
                                       live.get(tk,{}).get("ichimoku_tk_bull", False),
                                       live.get(tk,{}).get("ichimoku_chikou", False)]),
                "fib_support":    live.get(tk, {}).get("fib_support", False),
                "macd_bull_div":  live.get(tk, {}).get("macd_bull_div", False),
                "mtf_aligned":    live.get(tk, {}).get("mtf_aligned", False),
                "trend_reversal": live.get(tk, {}).get("trend_reversal", False),
                "bull_flag":      live.get(tk, {}).get("bull_flag", False),
                "cup_handle":     live.get(tk, {}).get("cup_handle", False),
                "cup_pivot":      live.get(tk, {}).get("cup_handle_pivot", 0.0),
                "at_demand_zone": live.get(tk, {}).get("at_demand_zone", False),
                "mom_accel":      live.get(tk, {}).get("mom_accel", False),
                "higher_lows":    live.get(tk, {}).get("higher_lows", False),
                "double_bottom":  live.get(tk, {}).get("double_bottom", False),
                "double_bottom_neckline": live.get(tk, {}).get("double_bottom_neckline", 0.0),
                "double_top":     live.get(tk, {}).get("double_top", False),
                "poc_price":      live.get(tk, {}).get("poc_price", 0.0),
                "at_poc":         live.get(tk, {}).get("at_poc", False),
                "above_poc":      live.get(tk, {}).get("above_poc", False),
                "poc_breakout":   live.get(tk, {}).get("poc_breakout", False),
                "ema_stacked_bull": live.get(tk, {}).get("ema_stacked_bull", False),
                "ema_stacked_bear": live.get(tk, {}).get("ema_stacked_bear", False),
                "kc_breakout":      live.get(tk, {}).get("kc_breakout", False),
                "kc_oversold":      live.get(tk, {}).get("kc_oversold", False),
                "obv_rising":       live.get(tk, {}).get("obv_rising", False),
                "rvol":             live.get(tk, {}).get("rvol", 1.0),
                "rvol_surge":       live.get(tk, {}).get("rvol_surge", False),
                "mfi":              live.get(tk, {}).get("mfi", 50),
                "mfi_oversold":     live.get(tk, {}).get("mfi_oversold", False),
                "mfi_bull_div":     live.get(tk, {}).get("mfi_bull_div", False),
                "supertrend_bull":    live.get(tk, {}).get("supertrend_bull", False),
                "supertrend_stop":    live.get(tk, {}).get("supertrend_stop", 0.0),
                "force_index_div":    live.get(tk, {}).get("force_index_div", False),
                "force_index_rising": live.get(tk, {}).get("force_index_rising", False),
                "true_beta":          live.get(tk, {}).get("true_beta", 1.0),
                "true_alpha":         live.get(tk, {}).get("true_alpha", 0.0),
                "earnings_days":      get_earnings_days(tk),
                "vcp":              live.get(tk, {}).get("vcp", False),
                "grade":          momentum_grade(live.get(tk, {}), sc),
            }
            for tk, sc, sent, sec, cat in (final_scores or [])[:8]
        ]
        tlog["last_scan_rejected"] = _rejected_log[:8]

        if not final_scores:
            logger.info(f"No longs passed threshold {_eff_min_score} (base={MIN_BUY_SCORE}).")
            if candidates_buy:
                logger.info(f"  Top rejected: {' | '.join(f'{t}:{s}' for t,s in candidates_buy[:5])}")
        else:
            for tk, sc, sent, sec, catalyst in final_scores[:open_long_slots]:
                try:
                    # Earnings proximity guard: don't buy within 2 days of earnings report
                    if earnings_too_close(tk, guard_days=2):
                        logger.info(f"SKIP {tk} — earnings within 2 days (gap-risk guard)")
                        continue
                    d        = live[tk]
                    price    = d["price"]
                    atr      = d.get("atr")
                    _tk_beta = d.get("true_beta", 1.0) or 1.0
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix,
                                             macro_day=macro_day, score_val=sc,
                                             win_rate=win_rate, drawdown_pct=drawdown_pct,
                                             payoff_ratio=_payoff_ratio, true_beta=_tk_beta)
                    # Portfolio heat adjustment: if sitting on big unrealized gains ("house money"),
                    # allow slightly larger positions; if deeply underwater, shrink further
                    if _portfolio_heat > 5:
                        notional = min(notional * 1.1, portfolio_val * MAX_POSITION_PCT * 1.2)
                    elif _portfolio_heat < -5:
                        notional = notional * 0.8
                    # Size up further for strong catalysts or squeeze setups (on top of Kelly)
                    if catalyst and sent >= 5:
                        notional = min(notional * 1.4, portfolio_val * MAX_POSITION_PCT, buying_power * 0.4)
                    elif tk in squeeze_cands or tk in vol_surge_cands:
                        notional = min(notional * 1.2, portfolio_val * MAX_POSITION_PCT, buying_power * 0.35)
                    if notional < 1:
                        logger.info(f"SKIP {tk} — insufficient buying power")
                        continue
                    # ATR-based stop: 2.5x ATR below entry, capped at STOP_LOSS_PCT
                    if atr and price > 0:
                        _atr_stop_buy = min(STOP_LOSS_PCT, max(0.03, (atr / price) * 2.5))
                    else:
                        _atr_stop_buy = STOP_LOSS_PCT
                    stop_price = round(price * (1 - _atr_stop_buy), 2)
                    # Compute buy qty for bracket orders (fractional supported on paper)
                    buy_qty = round(notional / price, 4)
                    if buy_qty < 0.001:
                        logger.info(f"SKIP {tk} — calculated qty too small")
                        continue

                    # Smart order type: limit orders for value/mean-rev setups,
                    # market orders for strong momentum (we want in now, not at a limit)
                    is_strong_momo = sc >= 70 or tk in gap_ups or tk in vol_surge_cands
                    if is_strong_momo:
                        # Bracket market order: stop-loss leg protects against fast crashes
                        order_payload = {
                            "symbol": tk, "qty": str(buy_qty),
                            "side": "buy", "type": "market", "time_in_force": "day",
                            "order_class": "bracket",
                            "stop_loss": {"stop_price": str(stop_price)},
                        }
                        order_type_log = f"MARKET+STOP@${stop_price}"
                    else:
                        # Bracket limit order at +0.4% above current
                        limit_px = round(price * 1.004, 2)
                        lim_qty  = round(notional / limit_px, 4)
                        if lim_qty < 0.001:
                            logger.info(f"SKIP {tk} — calculated qty too small")
                            continue
                        order_payload = {
                            "symbol": tk, "qty": str(lim_qty),
                            "side": "buy", "type": "limit",
                            "limit_price": str(limit_px), "time_in_force": "day",
                            "order_class": "bracket",
                            "stop_loss": {"stop_price": str(stop_price)},
                        }
                        order_type_log = f"LIMIT@${limit_px}+STOP@${stop_price}"
                    logger.info(
                        f"BUY {tk} [{order_type_log}] — ${notional:.0f} @ ~${price:.2f} "
                        f"| stop ${stop_price} ({_atr_stop_buy*100:.1f}%) | score {sc} | sent {sent:+.0f}"
                        + (f" | catalyst: {catalyst}" if catalyst else "")
                    )
                    try:
                        alpaca_post("/v2/orders", order_payload)
                    except Exception as _brk_err:
                        # Bracket orders may fail on paper if unsupported for fractionals
                        # Fall back to simple market/limit order
                        logger.debug(f"Bracket order failed ({_brk_err}), falling back to simple order")
                        simple_payload = {k: v for k, v in order_payload.items()
                                          if k not in ("order_class", "stop_loss", "take_profit")}
                        alpaca_post("/v2/orders", simple_payload)
                    reason = f"score={sc} sent={sent:+.0f}"
                    if catalyst:
                        reason += f" [{catalyst}]"
                    if tk in vol_surge_cands:
                        reason += " [VOL SURGE]"
                    if tk in squeeze_cands:
                        reason += " [SQUEEZE]"
                    # Cup & Handle: log pivot target (cup depth added to pivot = technical target)
                    _d_buy = live.get(tk, {})
                    if _d_buy.get("cup_handle"):
                        _pivot = _d_buy.get("cup_handle_pivot", 0)
                        _cup_d = _d_buy.get("cup_depth_pct", 0)
                        _ch_target = round(_pivot * (1 + _cup_d / 100), 2) if _pivot and _cup_d else 0
                        reason += f" [C&H pivot=${_pivot} target=${_ch_target}]"
                    if _d_buy.get("at_demand_zone"):
                        reason += " [DEMAND ZONE]"
                    if _d_buy.get("mom_accel"):
                        reason += " [ACCEL]"
                    if _d_buy.get("double_bottom"):
                        _db_nk = _d_buy.get("double_bottom_neckline", 0)
                        reason += f" [2-BTM neckline=${_db_nk}]" if _db_nk else " [2-BTM]"
                    if _d_buy.get("poc_breakout"):
                        reason += f" [POC-BRK ${_d_buy.get('poc_price', 0)}]"
                    elif _d_buy.get("above_poc"):
                        reason += f" [abv-POC ${_d_buy.get('poc_price', 0)}]"
                    if _d_buy.get("donchian_up"):   reason += " [DON-BRK]"
                    if _d_buy.get("ha_bull"):        reason += f" [HA×{_d_buy.get('ha_consec_bull',0)}]"
                    if _d_buy.get("mfi_bull_div"):   reason += f" [MFI-div{_d_buy.get('mfi',50):.0f}]"
                    if _d_buy.get("supertrend_bull"): reason += f" [ST${_d_buy.get('supertrend_stop',0):.1f}]"
                    if _d_buy.get("rvol_surge"):      reason += f" [RVOL{_d_buy.get('rvol',1):.1f}x]"
                    log_trade(tlog, "BUY", tk, price, notional, score=sc, reason=reason,
                              signals=live.get(tk, {}))
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
                    # ADX filter for shorts: only short into confirmed downtrends
                    _adx_short = d.get("adx", 0) or 0
                    if _adx_short > 0 and _adx_short < 18:
                        logger.debug(f"SKIP SHORT {tk} — ADX={_adx_short:.0f} too low (choppy market, false breakdowns)")
                        continue
                    atr      = d.get("atr")
                    _tk_beta_s = d.get("true_beta", 1.0) or 1.0
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix,
                                             macro_day=macro_day, score_val=sc,
                                             win_rate=win_rate, drawdown_pct=drawdown_pct,
                                             payoff_ratio=_payoff_ratio, true_beta=_tk_beta_s)
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
        def _pos_signals(sym):
            """Build compact signal state for a held position."""
            sig = live.get(sym, {})
            if not sig:
                return {}
            return {
                "rsi":            round(sig.get("daily_rsi", 50), 1),
                "vwap_pos":       round(sig.get("vwap_pos", 0), 2),
                "roc5":           round(sig.get("roc5", 0), 2),
                "macd_slope":     round(sig.get("macd_slope", 0), 4),
                "vol_ratio":      round(sig.get("vol_ratio", 1), 2),
                "vwap_reclaim":   sig.get("vwap_reclaim", False),
                "adx":            round(sig.get("adx", 0), 1),
                "adx_trend":      "strong" if sig.get("adx", 0) >= 25 else ("weak" if sig.get("adx", 0) < 15 else "moderate"),
                "rs5":            round(sig.get("rs5", 0), 2),
                "rs63":           round(sig.get("rs63", 0), 2),
                "chandelier_stop": round(sig.get("chandelier_stop", 0), 2),
                "ichimoku":        sum([sig.get("ichimoku_above", False), sig.get("ichimoku_bull_cloud", False),
                                        sig.get("ichimoku_tk_bull", False), sig.get("ichimoku_chikou", False)]),
                "macd_bull_div":   sig.get("macd_bull_div", False),
                "mtf_aligned":     sig.get("mtf_aligned", False),
                "price_vs_ema200": round(sig.get("price_vs_ema200", 0), 2),
                "mfi":             round(sig.get("mfi", 50), 1),
                "mfi_overbought":  sig.get("mfi_overbought", False),
                "mfi_bull_div":    sig.get("mfi_bull_div", False),
                "supertrend_bull": sig.get("supertrend_bull", True),
                "supertrend_stop": round(sig.get("supertrend_stop", 0), 2),
                "rvol":            round(sig.get("rvol", 1), 2),
                "rvol_surge":      sig.get("rvol_surge", False),
                "force_index_rising": sig.get("force_index_rising", False),
                "force_index_div":    sig.get("force_index_div", False),
                "true_beta":       round(sig.get("true_beta", 1), 2),
                "true_alpha":      round(sig.get("true_alpha", 0), 1),
            }

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
                "earnings_days": get_earnings_days(p.get("symbol", "")),
                "live_signals": _pos_signals(p.get("symbol", "")),
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

    # Signal attribution analytics: which reason-tags correlate with wins vs losses
    _signal_analytics = {}
    try:
        _signal_tags = ["gap", "squeeze", "vol-surge", "call-flow", "earnings-beat",
                         "pre-earnings", "mean-rev", "52W-breakout", "persistent", "re-entry"]
        for tag in _signal_tags:
            _trades_with = [t for t in _closed if tag in (t.get("reason", "") or "")]
            if len(_trades_with) >= 2:
                _wins_with   = [t for t in _trades_with if t["pnl_pct"] > 0]
                _avg_pnl_tag = round(sum(t["pnl_pct"] for t in _trades_with) / len(_trades_with), 2)
                _signal_analytics[tag] = {
                    "trades":  len(_trades_with),
                    "wins":    len(_wins_with),
                    "wr":      round(len(_wins_with) / len(_trades_with) * 100, 1),
                    "avg_pnl": _avg_pnl_tag,
                }
        # Recent 10 closed trades for quick dashboard display
        _recent_closed = sorted(_closed, key=lambda t: t.get("time", ""), reverse=True)[:10]
        tlog["recent_closed_trades"] = [
            {"ticker": t.get("ticker",""), "pnl_pct": round(t["pnl_pct"],2),
             "reason": (t.get("reason","") or "")[:40], "time": t.get("time","")}
            for t in _recent_closed
        ]
    except Exception:
        pass
    tlog["signal_analytics"] = _signal_analytics

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
    tlog["sector_rotation"]    = sector_adjs   # {sector: adj_score} for dashboard heatmap
    tlog["sector_etf_trends"]  = sector_etf_trends  # {sector: {bullish, chg5d, chg1d, above_ema20}}
    tlog["portfolio_beta"]     = _port_beta_est      # estimated portfolio beta

    # Compute per-signal win rates from accumulated performance data
    try:
        _sig_perf = tlog.get("signal_performance", {})
        _sig_wr = {
            k: {
                "win_rate": round(v["wins"] / max(1, v["total"]) * 100, 1),
                "total":    v["total"],
                "avg_pnl":  round(v.get("total_pnl", 0) / max(1, v["total"]), 2),
            }
            for k, v in _sig_perf.items()
            if v.get("total", 0) >= 3  # need at least 3 trades for meaningful stats
        }
        tlog["signal_win_rates"] = dict(sorted(_sig_wr.items(), key=lambda x: -x[1]["win_rate"]))
        # Log top/bottom performers
        if _sig_wr:
            _top = sorted(_sig_wr.items(), key=lambda x: -x[1]["win_rate"])[:3]
            _bot = sorted(_sig_wr.items(), key=lambda x:  x[1]["win_rate"])[:2]
            _top_str = " | ".join(f"{k}:{v['win_rate']}%({v['total']}t)" for k, v in _top)
            logger.info(f"Signal perf (top): {_top_str}")
    except Exception:
        pass
    tlog["sharpe_ratio"]       = _sharpe_ratio
    tlog["max_drawdown"]       = round(_max_dd, 2)
    try:
        tlog["effective_min_score"] = _eff_min_score
    except NameError:
        tlog["effective_min_score"] = MIN_BUY_SCORE

    # Market quality score: 0-100, composite of VIX, breadth, regime, sector momentum
    try:
        _mq = 50  # neutral baseline
        _vix_mq = regime.get("vix", 20.0) or 20.0
        if   _vix_mq < 15:  _mq += 20
        elif _vix_mq < 18:  _mq += 12
        elif _vix_mq < 22:  _mq +=  5
        elif _vix_mq < 28:  _mq -=  5
        elif _vix_mq < 35:  _mq -= 15
        else:                _mq -= 25
        _mq += 10 if regime.get("above_200", True) else -10
        _mq += 8  if regime.get("spy_trend", 0) > 1 else (-6 if regime.get("spy_trend", 0) < -1 else 0)
        _mq += 8  if breadth.get("adv_pct", 50) > 65 else (-8 if breadth.get("adv_pct", 50) < 35 else 0)
        # Internal scan breadth contribution (our proprietary A/D ratio)
        try:
            if _scan_adv_pct > 65:  _mq += 6
            elif _scan_adv_pct < 30: _mq -= 8
        except NameError:
            pass
        _hot_sectors = sum(1 for v in sector_adjs.values() if v >= 4)
        _cold_sectors = sum(1 for v in sector_adjs.values() if v <= -4)
        _mq += (_hot_sectors - _cold_sectors) * 2
        tlog["market_quality"] = max(0, min(100, round(_mq)))
    except Exception:
        tlog["market_quality"] = 50

    # Internal scan breadth metrics
    try:
        tlog["scan_breadth_pct"] = _scan_adv_pct
        tlog["scan_breadth_poor"] = _scan_breadth_poor
    except NameError:
        tlog["scan_breadth_pct"] = None
        tlog["scan_breadth_poor"] = False

    # Plain-English summary of this cycle's decision for the dashboard
    try:
        _top_scan = tlog.get("last_scan_top", [])
        _n_pos = len(tlog.get("positions", []))
        _regime_desc = regime.get("regime", "neutral").upper()
        _vix_now = regime.get("vix", 0)
        if made_trades:
            _recent = [t for t in tlog["trades"] if t.get("action") in ("BUY", "DCA", "SELL", "COVER")][-3:]
            _acts = " · ".join(f"{t['action']} {t['ticker']}" for t in _recent)
            _last_decision = f"Bot executed: {_acts}. Regime: {_regime_desc}, VIX {_vix_now:.1f}."
        elif _open_guard:
            _last_decision = f"Market just opened — waiting for opening volatility to settle (10 min guard). VIX {_vix_now:.1f}."
        elif _close_guard:
            _last_decision = f"Near market close — no new buys in last 20 min. VIX {_vix_now:.1f}."
        elif _consecutive_losses:
            _last_decision = f"Consecutive loss guard active — last 3 trades were losses. Protecting capital. VIX {_vix_now:.1f}."
        elif vix > VIX_EXTREME_THRESH:
            _last_decision = f"VIX {_vix_now:.1f} is extreme — all buys suspended until market calms."
        elif _top_scan:
            _top_str = ", ".join(f"${s['ticker']}({s['score']})" for s in _top_scan[:3])
            _eff_thresh = tlog.get("effective_min_score", MIN_BUY_SCORE)
            _hot_sec = sorted(sector_adjs.items(), key=lambda x: -x[1])[:2] if sector_adjs else []
            _hot_str = " | ".join(f"{s}:+{v}" for s,v in _hot_sec if v > 0)
            _mq_str  = f"MktQuality={tlog.get('market_quality',50)}/100"
            _last_decision = (f"Scanned {len(candidates)} stocks. Top: {_top_str}. "
                              f"Regime: {_regime_desc}, threshold={_eff_thresh}. "
                              f"{_mq_str}. Hot sectors: {_hot_str or 'none'}.")
        else:
            _last_decision = (f"Scanned {len(candidates)} stocks. No candidates passed scoring. "
                              f"Regime: {_regime_desc}, VIX {_vix_now:.1f}, "
                              f"MktQuality={tlog.get('market_quality',50)}/100.")
        tlog["last_decision"] = _last_decision
    except Exception:
        pass

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
