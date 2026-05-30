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
_ATM_IV_CACHE:  dict  = {}   # ATM implied volatility per symbol (30m TTL)
_ATM_IV_TS:     dict  = {}   # timestamps for ATM IV cache
_SIGNAL_WIN_RATES: dict = {}   # {signal_name: {win_rate, total, avg_pnl}} loaded from tlog each cycle
_LEARNED_COLD_SECTORS: set = set()   # sectors to avoid (from accumulated loss data)
_LEARNED_HOT_SECTORS:  set = set()   # sectors that are working (from accumulated wins)
_LEARNED_WORST_HOURS:  set = set()   # UTC hours with poor historical win rates
_LEARNED_BEST_HOURS:   set = set()   # UTC hours with high historical win rates
_LEARNED_TICKER_MEMORY: dict = {}   # {ticker: score_adj} — per-ticker score modifier from history
_LEARNED_WORST_HALFHOURS: set = set()  # "HHMM" strings of 30-min windows to avoid
_LEARNED_BEST_HALFHOURS:  set = set()  # "HHMM" strings of 30-min windows that outperform
_LEARNED_MIN_BREADTH:     float = 0.0  # minimum breadth % that produces consistent wins (learned)
_LEARNED_SIGNAL_COUNT_SWEET: str  = ""   # "1-3"|"4-6"|"7-10"|"11+" — best-performing signal count bucket
_LEARNED_SPY_DOWN_PENALTY:  bool = False  # True when red SPY days consistently hurt outcomes
_LEARNED_FALLING_SCORE_PENALTY: bool = False  # True when falling-score entries consistently underperform
_LEARNED_ATR_MULTIPLIER:        float = 2.5   # learned ATR stop multiplier (starts at default)

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

    # Store catalyst type and sector at entry for performance attribution
    if action == "BUY" and signals:
        e["catalyst_type"] = signals.get("catalyst_type", "none")
        e["sector"] = SECTOR_MAP.get(sym, "other")
        # Store current regime at entry so we can analyze regime-based performance later
        _cur_regime = tlog.get("regime", {})
        e["regime"] = _cur_regime.get("regime", "unknown") if isinstance(_cur_regime, dict) else str(_cur_regime)[:20]
        # Store VIX at entry for VIX-bracket performance tracking
        _vix_entry = None
        if isinstance(_cur_regime, dict): _vix_entry = _cur_regime.get("vix")
        if _vix_entry is not None: e["vix_at_entry"] = round(float(_vix_entry), 1)
        # Store additional entry context for performance attribution neurons
        e["rvol_at_entry"]        = round(float(signals.get("rvol", 1.0) or 1.0), 2)
        e["earnings_days_at_entry"] = signals.get("earnings_days")  # None if unknown
        e["mkt_quality_at_entry"]  = int(tlog.get("market_quality", 50) or 50)
        # Portfolio concentration at entry (# held positions when we bought)
        _n_held = len([t for t in tlog.get("trades", []) if t.get("action") == "BUY"
                       and not any(s.get("ticker") == t.get("ticker") and s.get("action") in ("SELL","SELL_HALF","COVER")
                                   for s in tlog.get("trades", []) if s.get("time", "") > t.get("time", ""))])
        e["positions_at_entry"] = max(0, _n_held)
        e["concentration_bucket"] = ("1-2" if _n_held <= 2 else "3-4" if _n_held <= 4 else "5-7" if _n_held <= 7 else "8+")
        # Day of week at entry (0=Mon, 6=Sun)
        _dow = datetime.now(timezone.utc).weekday()
        e["day_of_week"] = _dow
        e["day_name"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][_dow]
        # Breadth at entry (% sectors/stocks advancing)
        _breadth_entry = tlog.get("market_breadth", {})
        e["breadth_at_entry"] = round(float(_breadth_entry.get("adv_pct", 50) or 50), 1)
        # Momentum grade at entry (A+/A/B/C/D) — compute here using score + signals
        try:
            e["grade_at_entry"] = momentum_grade(signals, score or 0)
        except Exception:
            e["grade_at_entry"] = "?"
        # Price tier at entry
        _p = float(price)
        e["price_tier"] = ("micro" if _p < 10 else "small" if _p < 30 else "mid" if _p < 100 else "large")
        # Macro event context at entry (FOMC/CPI/NFP awareness)
        try:
            _macro_ctx = get_macro_context()
            e["macro_event"] = _macro_ctx["event"]    # "FOMC", "CPI", "NFP", "none"
            e["macro_label"] = _macro_ctx["label"]    # "event_day", "day_before", "normal"
        except Exception:
            e["macro_event"] = "none"
            e["macro_label"] = "normal"
        # RSI at entry (daily RSI for swing entry quality tracking)
        e["rsi_at_entry"] = round(float(signals.get("daily_rsi", 50) or 50), 1)
        # Score trend at entry: was score rising/flat/falling across recent scans?
        e["score_trend"] = signals.get("score_trend", "flat")
        e["score_trend_delta"] = float(signals.get("score_trend_delta", 0.0) or 0.0)
        # Position size and ATR distance at entry (for Neurons 28 and 29)
        e["pos_size_pct"]    = float(signals.get("pos_size_pct", 0.0) or 0.0)
        e["pos_size_bucket"] = signals.get("pos_size_bucket", "2-5%")
        e["atr_pct_at_entry"] = float(signals.get("atr_pct_at_entry", 0.0) or 0.0)
        e["atr_bucket"]      = signals.get("atr_bucket", "1-2%")
        # Pre-market gap at entry (Neuron 30)
        _pm_gap = float(signals.get("pm_gap_pct", 0.0) or 0.0)
        e["pm_gap_pct"] = round(_pm_gap, 1)
        e["pm_gap_bucket"] = ("big_up" if _pm_gap > 3 else "small_up" if _pm_gap > 0.5 else
                              "big_down" if _pm_gap < -3 else "small_down" if _pm_gap < -0.5 else "flat")
        # Sector momentum at entry (Neuron 32): was the sector ETF accelerating?
        try:
            _sec_etf_now = tlog.get("sector_etf_trends", {}).get(SECTOR_MAP.get(sym, "other"), {})
            _sec_chg1d = float(_sec_etf_now.get("chg1d", 0.0) or 0.0)
            _sec_chg5d = float(_sec_etf_now.get("chg5d", 0.0) or 0.0)
            _sec_momentum = ("accelerating" if _sec_chg1d > 0.5 and _sec_chg5d > 1.0 else
                             "decelerating" if _sec_chg1d < -0.3 and _sec_chg5d < 0 else "neutral")
            e["sector_etf_momentum"] = _sec_momentum
            e["sector_chg1d"] = round(_sec_chg1d, 2)
        except Exception:
            e["sector_etf_momentum"] = "neutral"
        # Trend Template Neuron (33): O'Neil quality score at entry
        _tt_score = int(signals.get("trend_template", 0) or 0)
        e["tt_score_at_entry"] = _tt_score
        e["tt_bucket"] = ("elite" if _tt_score >= 7 else "good" if _tt_score >= 5 else "fair" if _tt_score >= 3 else "weak")
        # Consecutive Green Days Neuron (34): momentum confirmation days
        _cg = int(signals.get("consec_green", 0) or 0)
        e["consec_green_at_entry"] = _cg
        e["consec_green_bucket"] = ("0d" if _cg == 0 else "1d" if _cg == 1 else "2-3d" if _cg <= 3 else "4d+")
        # Institutional Accumulation Neuron (35): smart money accumulation score
        _ac = int(signals.get("accum_score", 0) or 0)
        e["accum_score_at_entry"] = _ac
        e["accum_bucket"] = ("heavy" if _ac >= 8 else "moderate" if _ac >= 5 else "light" if _ac >= 2 else "none")
        # RS Rating Neuron (36): IBD-style Relative Strength Rating at entry
        _rs_r = int(signals.get("rs_rating", 50) or 50)
        e["rs_rating_at_entry"] = _rs_r
        e["rs_bucket"] = ("elite" if _rs_r >= 90 else "strong" if _rs_r >= 75 else "average" if _rs_r >= 50 else "weak")
        # MACD Neuron (37): MACD state at entry (bullish cross, positive, negative, divergence)
        _macd_v = float(signals.get("macd", 0.0) or 0.0)
        _macd_s = float(signals.get("macd_slope", 0.0) or 0.0)
        _macd_d = bool(signals.get("macd_bull_div", False))
        e["macd_state"] = ("bull_div" if _macd_d else "rising" if _macd_v > 0 and _macd_s > 0 else
                           "recovering" if _macd_v < 0 and _macd_s > 0 else "negative")
        # TTM Squeeze Breakout Neuron (38): was a squeeze fired at entry?
        e["squeeze_fired"] = bool(signals.get("ttm_squeeze_fired", False))
        # News Catalyst Urgency Neuron (39): urgency score 0-5 from AI catalyst classification
        _urg = int(signals.get("catalyst_urg", 0) or 0)
        e["catalyst_urgency"] = _urg
        e["urgency_bucket"] = ("high" if _urg >= 4 else "medium" if _urg >= 2 else "low")
        # VWAP Position Neuron (40): was price above/below VWAP at entry?
        _vwap_p = float(signals.get("vwap_pos", 0.0) or 0.0)
        e["vwap_pos_at_entry"] = round(_vwap_p, 2)
        e["vwap_bucket"] = ("above" if _vwap_p > 0.5 else "below" if _vwap_p < -0.5 else "at_vwap")
        # POC Distance Neuron (42): how far above/below Point of Control at entry?
        # POC = price level with most volume traded = institutional anchor price.
        # Breakout entries (well above POC) = smart money control; below = risky.
        _poc_pr = float(signals.get("poc_price", 0.0) or 0.0)
        _entry_pr = float(price if price else 0.0)
        if _poc_pr > 0 and _entry_pr > 0:
            _poc_dist = round((_entry_pr - _poc_pr) / _poc_pr * 100, 2)
            e["poc_dist_pct"] = _poc_dist
            e["poc_dist_bucket"] = ("breakout" if _poc_dist > 2.0 else "above" if _poc_dist > 0.5
                                    else "at_poc" if _poc_dist >= -0.5 else "below")
        else:
            e["poc_dist_pct"] = 0.0
            e["poc_dist_bucket"] = "at_poc"
        # Intraday Momentum Neuron (43): how far from today's open at entry?
        # Stocks up 5%+ from open may be extended (chasing); flat to +2% = ideal momentum.
        # Learns: does buying early (near open) outperform chasing intraday runners?
        _d_open = float(signals.get("day_open", 0.0) or 0.0)
        if _d_open > 0 and _entry_pr > 0:
            _id_mom = round((_entry_pr - _d_open) / _d_open * 100, 2)
            e["intraday_mom_pct"] = _id_mom
            e["intraday_mom_bucket"] = ("extended" if _id_mom > 5.0 else "runner" if _id_mom > 2.0
                                        else "early" if _id_mom >= 0.0 else "pullback")
        else:
            e["intraday_mom_pct"] = 0.0
            e["intraday_mom_bucket"] = "early"
        # ADX Trend Strength Neuron (44): directional trend conviction at entry.
        # ADX >25 = strong trend (momentum algos' sweet spot), 15-25 = developing,
        # <15 = no trend / choppy — high risk for directional strategies.
        _adx_v = float(signals.get("adx", 0.0) or 0.0)
        e["adx_at_entry"] = round(_adx_v, 1)
        e["adx_bucket"] = ("strong" if _adx_v >= 25 else "developing" if _adx_v >= 15 else "weak")
        # RVOL Tier Neuron (45): actual relative volume ratio at entry.
        # 5x+ = explosive institutional interest, 2-5x = strong, 1-2x = normal, <1x = weak.
        # More granular than the binary rvol_surge signal — learns the optimal RVOL tier.
        _rvol_v = float(signals.get("rvol", 0.0) or 0.0)
        e["rvol_at_entry"] = round(_rvol_v, 2)
        e["rvol_tier"] = ("explosive" if _rvol_v >= 5.0 else "strong" if _rvol_v >= 2.0
                          else "normal" if _rvol_v >= 1.0 else "weak")
        # Stochastic Zone Neuron (46): %K at entry — overbought/neutral/oversold zone.
        # Overbought (>80) entries are late momentum chases; neutral (20-80) is the sweet spot;
        # oversold (<20) = mean reversion entry — learns which zone produces best outcomes.
        _stk_v = float(signals.get("stoch_k", 50.0) or 50.0)
        e["stoch_k_at_entry"] = round(_stk_v, 1)
        e["stoch_zone"] = ("overbought" if _stk_v > 80 else "oversold" if _stk_v < 20 else "neutral")
        # Multi-Timeframe Alignment Neuron (47): are short/medium/long timeframes all aligned?
        # full = mtf_aligned + ema_stacked_bull + ROC20>0 (all three agree = highest conviction).
        # partial = at least one confirmed. none = mixed or bearish signals.
        _mtf_ok = bool(signals.get("mtf_aligned", False))
        _ema_bull = bool(signals.get("ema_stacked_bull", False))
        _roc20_v = float(signals.get("roc20", 0.0) or 0.0)
        _mtf_score = int(_mtf_ok) + int(_ema_bull) + int(_roc20_v > 0)
        e["mtf_score_at_entry"] = _mtf_score
        e["mtf_alignment"] = ("full" if _mtf_score >= 3 else "partial" if _mtf_score >= 1 else "none")
        # Options Flow Neuron (48): smart money positioning at entry.
        # unusual_calls + options_bull + low PCR (<0.7) = institutional call buying.
        # Learns: do entries confirmed by options flow outperform unconfirmed entries?
        _uc = bool(signals.get("unusual_calls", False))
        _ob = bool(signals.get("options_bull", False))
        _pcr = float(signals.get("options_pcr", 1.0) or 1.0)
        _opt_score = int(_uc) + int(_ob) + int(_pcr < 0.7)
        e["options_flow_score"] = _opt_score
        e["options_flow_tier"] = ("confirmed" if _opt_score >= 2 else "slight" if _opt_score >= 1 else "neutral")
        # MFI Zone Neuron (49): Money Flow Index at entry (volume-weighted RSI).
        # MFI >80 = heavy distribution (overbought + volume selling); <20 = accumulation.
        # More reliable than RSI because it incorporates volume — harder to fake.
        _mfi_v = float(signals.get("mfi", 50.0) or 50.0)
        e["mfi_at_entry"] = round(_mfi_v, 1)
        e["mfi_zone"] = ("distribution" if _mfi_v > 80 else "accumulation" if _mfi_v < 30 else "neutral")

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
            # New signals (session additions)
            "three_white_soldiers", "morning_star", "bullish_engulfing", "hammer",
            "psar_bull", "price_accel_pos", "unusual_calls", "options_bull",
            "donchian_up", "ha_bull", "rvol_surge",
            "lr_below_channel",  # mean reversion buy at LR channel support
        ]
        e["entry_signals"] = list(dict.fromkeys(k for k in _SIGNAL_KEYS if signals.get(k)))
        e["signal_count_at_entry"] = len(e["entry_signals"])
        # SPY day return at entry: up/flat/down market context for trade outcome learning
        try:
            _spy_d1 = _fetch_spy_perf().get("d1", 0.0) or 0.0
            e["spy_day_return"] = round(float(_spy_d1), 2)
            e["spy_day_bucket"] = ("up" if _spy_d1 > 0.5 else "down" if _spy_d1 < -0.5 else "flat")
        except Exception:
            e["spy_day_return"] = 0.0
            e["spy_day_bucket"] = "flat"
        # Re-entry detection: was this ticker sold within the last 48h?
        try:
            _re_cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
            _prior_sell = next((t for t in tlog.get("trades", [])
                               if t.get("ticker") == sym
                               and t.get("action") in ("SELL", "SELL_HALF", "COVER")
                               and t.get("pnl_pct") is not None
                               and datetime.fromisoformat(t["time"].replace("Z", "+00:00")) > _re_cutoff), None)
            if _prior_sell:
                e["is_reentry"] = True
                e["reentry_prior_pnl"] = _prior_sell.get("pnl_pct", 0.0)
                e["reentry_type"] = "winner" if _prior_sell.get("pnl_pct", 0) > 0 else "loser"
            else:
                e["is_reentry"] = False
        except Exception:
            e["is_reentry"] = False

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
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        # Find matching BUY/SHORT for this ticker in recent trades
        for t in tlog.get("trades", []):
            if t.get("action") in ("BUY", "SHORT") and t.get("ticker") == sym and t.get("entry_signals"):
                perf = tlog.setdefault("signal_performance", {})
                _entry_sigs = t["entry_signals"]
                for sig in _entry_sigs:
                    sp = perf.setdefault(sig, {"wins": 0, "losses": 0, "total": 0,
                                               "total_pnl": 0.0, "best": 0.0, "worst": 0.0})
                    sp["total"] = sp.get("total", 0) + 1
                    sp["total_pnl"] = round(sp.get("total_pnl", 0.0) + pnl, 2)
                    if pnl > 0:
                        sp["wins"] = sp.get("wins", 0) + 1
                        sp["best"]  = max(sp.get("best", 0.0), pnl)
                    else:
                        sp["losses"] = sp.get("losses", 0) + 1
                        sp["worst"] = min(sp.get("worst", 0.0), pnl)
                    # Derived stats
                    t_cnt = sp["total"]
                    sp["win_rate"]  = round(sp["wins"] / t_cnt * 100, 1)  # stored as 0-100 for dashboard
                    sp["avg_pnl"]   = round(sp["total_pnl"] / t_cnt, 2)
                    sp["payoff_ratio"] = round(
                        sp.get("best", 0) / max(abs(sp.get("worst", -0.01)), 0.01), 2
                    ) if sp.get("worst", 0) < 0 else sp.get("best", 0)

                # ── Signal PAIR synergy tracking (neural synapse learning) ──
                # Tracks performance of signal COMBINATIONS, not just single signals.
                # When two signals fire together and a trade wins → that synapse strengthens.
                # When they fire together and lose → the synapse weakens.
                # Over time this finds the most powerful signal combinations in our system.
                if len(_entry_sigs) >= 2:
                    try:
                        _syn_perf = tlog.setdefault("signal_synergy", {})
                        _sorted_sigs = sorted(_entry_sigs[:6])  # limit pairs to top 6 signals
                        for _i in range(len(_sorted_sigs)):
                            for _j in range(_i + 1, len(_sorted_sigs)):
                                _pair_key = f"{_sorted_sigs[_i]}+{_sorted_sigs[_j]}"
                                _syn = _syn_perf.setdefault(_pair_key, {
                                    "wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0
                                })
                                _syn["total"] += 1
                                _syn["total_pnl"] = round(_syn["total_pnl"] + pnl, 2)
                                if pnl > 0: _syn["wins"] += 1
                                else:       _syn["losses"] += 1
                                _syn["win_rate"] = round(_syn["wins"] / _syn["total"] * 100, 1)
                                _syn["avg_pnl"]  = round(_syn["total_pnl"] / _syn["total"], 2)
                        # Keep only top 50 pairs by frequency to avoid unbounded growth
                        if len(_syn_perf) > 60:
                            _syn_sorted = sorted(_syn_perf.items(), key=lambda x: -x[1]["total"])
                            tlog["signal_synergy"] = dict(_syn_sorted[:50])
                    except Exception:
                        pass
                break

        # Rebuild signal_win_rates summary for the score() adaptive loop
        perf_all = tlog.get("signal_performance", {})
        tlog["signal_win_rates"] = {
            sig: {
                "win_rate": v.get("win_rate", 50.0),  # 0-100 scale for dashboard
                "total":    v.get("total", 0),
                "avg_pnl":  v.get("avg_pnl", 0),
            }
            for sig, v in perf_all.items()
            if v.get("total", 0) >= 3  # only trust with ≥3 samples
        }

    # Per-sector performance tracking (updated on every close)
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        _sector_key = SECTOR_MAP.get(sym, "other")
        _sec_perf = tlog.setdefault("sector_performance", {})
        _sp = _sec_perf.setdefault(_sector_key, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
        _sp["total"] = _sp.get("total", 0) + 1
        _sp["total_pnl"] = round(_sp.get("total_pnl", 0.0) + pnl, 2)
        if pnl > 0:
            _sp["wins"] = _sp.get("wins", 0) + 1
        else:
            _sp["losses"] = _sp.get("losses", 0) + 1
        if _sp["total"] > 0:
            _sp["win_rate"] = round(_sp["wins"] / _sp["total"] * 100, 1)
            _sp["avg_pnl"]  = round(_sp["total_pnl"] / _sp["total"], 2)

    # Catalyst-type performance tracking
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        # Find matching BUY entry with catalyst info
        for t in tlog.get("trades", []):
            if t.get("action") == "BUY" and t.get("ticker") == sym and t.get("catalyst_type"):
                _cat_key = t["catalyst_type"]
                _cat_perf = tlog.setdefault("catalyst_performance", {})
                _cp = _cat_perf.setdefault(_cat_key, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _cp["total"] = _cp.get("total", 0) + 1
                _cp["total_pnl"] = round(_cp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0:
                    _cp["wins"] = _cp.get("wins", 0) + 1
                else:
                    _cp["losses"] = _cp.get("losses", 0) + 1
                if _cp["total"] > 0:
                    _cp["win_rate"] = round(_cp["wins"] / _cp["total"] * 100, 1)
                    _cp["avg_pnl"]  = round(_cp["total_pnl"] / _cp["total"], 2)
                break

    # Regime-based performance tracking (bull/bear/choppy performance breakdown)
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            # Look up the regime at time of entry from the trade record
            _buy_entry = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _reg_key = "unknown"
            if _buy_entry:
                _reg_key = _buy_entry.get("regime", "unknown")
            else:
                # Use current regime as fallback
                _cur_reg = tlog.get("regime", {})
                _reg_key = _cur_reg if isinstance(_cur_reg, str) else (_cur_reg.get("regime", "unknown") if isinstance(_cur_reg, dict) else "unknown")
            _reg_perf = tlog.setdefault("regime_performance", {})
            _rp = _reg_perf.setdefault(str(_reg_key), {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
            _rp["total"] = _rp.get("total", 0) + 1
            _rp["total_pnl"] = round(_rp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0:
                _rp["wins"] = _rp.get("wins", 0) + 1
            else:
                _rp["losses"] = _rp.get("losses", 0) + 1
            if _rp["total"] > 0:
                _rp["win_rate"] = round(_rp["wins"] / _rp["total"] * 100, 1)
                _rp["avg_pnl"]  = round(_rp["total_pnl"] / _rp["total"], 2)
        except Exception:
            pass

    # Hold time performance tracking
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            if _buy_t and _buy_t.get("time"):
                from datetime import datetime as _hdt, timezone as _htz
                _entry_t = _hdt.fromisoformat(_buy_t["time"].replace("Z", "+00:00"))
                _hold_d  = max(0, int((_entry_t.utcnow().replace(tzinfo=_htz.utc) - _entry_t).total_seconds() / 86400))
                _ht_bucket = "0-2d" if _hold_d <= 2 else ("3-7d" if _hold_d <= 7 else ("8-14d" if _hold_d <= 14 else "15d+"))
                _ht_perf = tlog.setdefault("hold_time_performance", {})
                _htp = _ht_perf.setdefault(_ht_bucket, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _htp["total"] = _htp.get("total", 0) + 1
                _htp["total_pnl"] = round(_htp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0:
                    _htp["wins"] = _htp.get("wins", 0) + 1
                else:
                    _htp["losses"] = _htp.get("losses", 0) + 1
                if _htp["total"] > 0:
                    _htp["win_rate"] = round(_htp["wins"] / _htp["total"] * 100, 1)
                    _htp["avg_pnl"]  = round(_htp["total_pnl"] / _htp["total"], 2)
        except Exception:
            pass

    # ── Ticker Memory Neuron: per-ticker win/loss history ────────────────────
    # The bot remembers which tickers it's historically good or bad at.
    # A ticker with 70%+ WR gets a score boost next time; <35% WR gets a penalty.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _tk_mem = tlog.setdefault("ticker_memory", {})
            _tm = _tk_mem.setdefault(sym, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
            _tm["total"] = _tm.get("total", 0) + 1
            _tm["total_pnl"] = round(_tm.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _tm["wins"] = _tm.get("wins", 0) + 1
            else: _tm["losses"] = _tm.get("losses", 0) + 1
            if _tm["total"] > 0:
                _tm["win_rate"] = round(_tm["wins"] / _tm["total"] * 100, 1)
                _tm["avg_pnl"]  = round(_tm["total_pnl"] / _tm["total"], 2)
            # Trim to top 100 tickers by trade count
            if len(_tk_mem) > 120:
                tlog["ticker_memory"] = dict(sorted(_tk_mem.items(), key=lambda x: -x[1].get("total", 0))[:100])
        except Exception:
            pass

    # ── VIX Bracket Neuron: performance by volatility regime ────────────────
    # Tracks win rate in low/normal/elevated/high VIX environments.
    # The bot learns whether it should be more/less aggressive at each VIX level.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t2 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _vix_ent = _buy_t2.get("vix_at_entry") if _buy_t2 else None
            if _vix_ent is not None:
                _vbkt = ("low" if float(_vix_ent) < 14 else
                         "normal" if float(_vix_ent) < 20 else
                         "elevated" if float(_vix_ent) < 28 else "high")
                _vix_perf = tlog.setdefault("vix_bracket_performance", {})
                _vp = _vix_perf.setdefault(_vbkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _vp["total"] = _vp.get("total", 0) + 1
                _vp["total_pnl"] = round(_vp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _vp["wins"] = _vp.get("wins", 0) + 1
                else: _vp["losses"] = _vp.get("losses", 0) + 1
                if _vp["total"] > 0:
                    _vp["win_rate"] = round(_vp["wins"] / _vp["total"] * 100, 1)
                    _vp["avg_pnl"]  = round(_vp["total_pnl"] / _vp["total"], 2)
        except Exception:
            pass

    # ── 30-Minute Window Neuron: fine-grained time-of-day scoring ───────────
    # More precise than hour-level: learns "9:30 AM ET is great, 9:00 AM is rough"
    # Key "HHMM" is the UTC 30-min window when the BUY was entered.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t3 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            if _buy_t3 and _buy_t3.get("time"):
                _entry_dt3 = datetime.fromisoformat(_buy_t3["time"].replace("Z", "+00:00"))
                _hw_key = f"{_entry_dt3.hour:02d}{'30' if _entry_dt3.minute >= 30 else '00'}"
                _hw_perf = tlog.setdefault("halfhour_performance", {})
                _hwp = _hw_perf.setdefault(_hw_key, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _hwp["total"] = _hwp.get("total", 0) + 1
                _hwp["total_pnl"] = round(_hwp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _hwp["wins"] = _hwp.get("wins", 0) + 1
                else: _hwp["losses"] = _hwp.get("losses", 0) + 1
                if _hwp["total"] > 0:
                    _hwp["win_rate"] = round(_hwp["wins"] / _hwp["total"] * 100, 1)
                    _hwp["avg_pnl"]  = round(_hwp["total_pnl"] / _hwp["total"], 2)
        except Exception:
            pass

    # ── Earnings Proximity Neuron: learn near-earnings trade outcomes ─────────
    # Tracks win rate by how close to earnings the bot entered.
    # Learns: are pre-earnings drift plays working? Are day-of entries too risky?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t4 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _earn_days = _buy_t4.get("earnings_days_at_entry") if _buy_t4 else None
            if _earn_days is not None:
                _earn_bkt = ("0-2d" if _earn_days <= 2 else
                             "3-7d"  if _earn_days <= 7  else
                             "8-20d" if _earn_days <= 20 else "21d+")
                _earn_perf = tlog.setdefault("earnings_proximity_perf", {})
                _ep = _earn_perf.setdefault(_earn_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _ep["total"] = _ep.get("total", 0) + 1
                _ep["total_pnl"] = round(_ep.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _ep["wins"] = _ep.get("wins", 0) + 1
                else: _ep["losses"] = _ep.get("losses", 0) + 1
                if _ep["total"] > 0:
                    _ep["win_rate"] = round(_ep["wins"] / _ep["total"] * 100, 1)
                    _ep["avg_pnl"]  = round(_ep["total_pnl"] / _ep["total"], 2)
        except Exception:
            pass

    # ── RVOL Performance Neuron: learn optimal entry volume threshold ─────────
    # High relative volume at entry often confirms institutional participation.
    # The bot learns: does RVOL > 3x actually lead to better outcomes?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t5 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _rvol_e = float(_buy_t5.get("rvol_at_entry", 1.0) or 1.0) if _buy_t5 else None
            if _rvol_e is not None:
                _rv_bkt = ("low" if _rvol_e < 1.5 else
                           "normal" if _rvol_e < 2.5 else
                           "high"   if _rvol_e < 4.0 else "surge")
                _rvol_perf = tlog.setdefault("rvol_perf", {})
                _rvp = _rvol_perf.setdefault(_rv_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _rvp["total"] = _rvp.get("total", 0) + 1
                _rvp["total_pnl"] = round(_rvp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _rvp["wins"] = _rvp.get("wins", 0) + 1
                else: _rvp["losses"] = _rvp.get("losses", 0) + 1
                if _rvp["total"] > 0:
                    _rvp["win_rate"] = round(_rvp["wins"] / _rvp["total"] * 100, 1)
                    _rvp["avg_pnl"]  = round(_rvp["total_pnl"] / _rvp["total"], 2)
        except Exception:
            pass

    # ── Market Quality Threshold Neuron: learn minimum conditions ─────────────
    # Tracks performance by market quality at time of entry (0-100 composite).
    # Learns: below what quality score do entries fail? Above what do they thrive?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t6 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _mq_e = int(_buy_t6.get("mkt_quality_at_entry", 50) or 50) if _buy_t6 else None
            if _mq_e is not None:
                _mq_bkt = ("poor" if _mq_e < 40 else
                           "fair"      if _mq_e < 60 else
                           "good"      if _mq_e < 75 else "excellent")
                _mq_perf = tlog.setdefault("mkt_quality_perf", {})
                _mqp = _mq_perf.setdefault(_mq_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _mqp["total"] = _mqp.get("total", 0) + 1
                _mqp["total_pnl"] = round(_mqp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _mqp["wins"] = _mqp.get("wins", 0) + 1
                else: _mqp["losses"] = _mqp.get("losses", 0) + 1
                if _mqp["total"] > 0:
                    _mqp["win_rate"] = round(_mqp["wins"] / _mqp["total"] * 100, 1)
                    _mqp["avg_pnl"]  = round(_mqp["total_pnl"] / _mqp["total"], 2)
        except Exception:
            pass

    # ── Momentum Grade Performance Neuron: does A+ really outperform? ─────────
    # Tracks win rate by the momentum grade assigned at entry (A+/A/B/C/D).
    # Validates: are higher-grade setups actually more profitable?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t7 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _grd = _buy_t7.get("grade_at_entry") if _buy_t7 else None
            if _grd and _grd != "?":
                _grd_perf = tlog.setdefault("grade_perf", {})
                _gp = _grd_perf.setdefault(_grd, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _gp["total"] = _gp.get("total", 0) + 1
                _gp["total_pnl"] = round(_gp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _gp["wins"] = _gp.get("wins", 0) + 1
                else: _gp["losses"] = _gp.get("losses", 0) + 1
                if _gp["total"] > 0:
                    _gp["win_rate"] = round(_gp["wins"] / _gp["total"] * 100, 1)
                    _gp["avg_pnl"]  = round(_gp["total_pnl"] / _gp["total"], 2)
        except Exception:
            pass

    # ── Price Tier Performance Neuron: micro/small/mid/large caps ─────────────
    # Learns whether the bot does better with cheaper vs. expensive stocks.
    # Some strategies work better with small caps (more volatile, bigger moves)
    # while others need large cap liquidity.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t8 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _ptier = _buy_t8.get("price_tier") if _buy_t8 else None
            if not _ptier:
                # Fall back to estimating from sell price
                _ptier = ("micro" if float(price) < 10 else "small" if float(price) < 30 else "mid" if float(price) < 100 else "large")
            _tier_perf = tlog.setdefault("price_tier_perf", {})
            _tp = _tier_perf.setdefault(_ptier, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
            _tp["total"] = _tp.get("total", 0) + 1
            _tp["total_pnl"] = round(_tp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _tp["wins"] = _tp.get("wins", 0) + 1
            else: _tp["losses"] = _tp.get("losses", 0) + 1
            if _tp["total"] > 0:
                _tp["win_rate"] = round(_tp["wins"] / _tp["total"] * 100, 1)
                _tp["avg_pnl"]  = round(_tp["total_pnl"] / _tp["total"], 2)
        except Exception:
            pass

    # ── Market Breadth Neuron: learn optimal breadth threshold ────────────────
    # Tracks win rate by how broad the market was advancing when we entered.
    # Learns: do entries in strong breadth (>65%) days outperform narrow markets?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_t9 = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _br_e = float(_buy_t9.get("breadth_at_entry", 50) or 50) if _buy_t9 else None
            if _br_e is not None:
                _br_bkt = ("weak" if _br_e < 40 else
                           "mixed" if _br_e < 55 else
                           "broad" if _br_e < 70 else "strong")
                _br_perf = tlog.setdefault("breadth_perf", {})
                _brp = _br_perf.setdefault(_br_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _brp["total"] = _brp.get("total", 0) + 1
                _brp["total_pnl"] = round(_brp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _brp["wins"] = _brp.get("wins", 0) + 1
                else: _brp["losses"] = _brp.get("losses", 0) + 1
                if _brp["total"] > 0:
                    _brp["win_rate"] = round(_brp["wins"] / _brp["total"] * 100, 1)
                    _brp["avg_pnl"]  = round(_brp["total_pnl"] / _brp["total"], 2)
        except Exception:
            pass

    # ── DCA Intelligence Neuron: learn when averaging down helps or hurts ─────
    # When a DCA occurs and the position eventually closes, track the outcome.
    # The bot learns: in what conditions does DCA improve total returns?
    if action == "DCA" and pnl is None:  # DCA entry — tag it for tracking
        try:
            _dca_ctx = tlog.setdefault("dca_events", [])
            _dca_ctx.insert(0, {
                "ticker": sym,
                "time": datetime.now(timezone.utc).isoformat(),
                "mkt_quality": tlog.get("market_quality", 50),
                "breadth": (tlog.get("market_breadth") or {}).get("adv_pct", 50),
                "regime": (tlog.get("regime") or {}).get("regime", "unknown"),
            })
            tlog["dca_events"] = _dca_ctx[:50]  # keep last 50 DCA events
        except Exception:
            pass

    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            # Check if this position had a DCA — was DCA beneficial?
            _dca_ev = next((d for d in tlog.get("dca_events", []) if d.get("ticker") == sym), None)
            if _dca_ev:
                _dca_perf = tlog.setdefault("dca_outcome_perf", {})
                _dca_result = "win" if pnl > 0 else "loss"
                _dca_regime = _dca_ev.get("regime", "unknown")
                _dp = _dca_perf.setdefault(_dca_regime, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _dp["total"] = _dp.get("total", 0) + 1
                _dp["total_pnl"] = round(_dp.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _dp["wins"] = _dp.get("wins", 0) + 1
                else: _dp["losses"] = _dp.get("losses", 0) + 1
                if _dp["total"] > 0:
                    _dp["win_rate"] = round(_dp["wins"] / _dp["total"] * 100, 1)
                    _dp["avg_pnl"]  = round(_dp["total_pnl"] / _dp["total"], 2)
        except Exception:
            pass

    # ── RSI Entry Zone Neuron: learn optimal RSI at entry ─────────────────────
    # Tracks win rate by RSI bracket at time of purchase.
    # Learns: does buying oversold (<35) or momentum (>60) produce better results?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_ta = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _rsi_e = float(_buy_ta.get("rsi_at_entry", 50) or 50) if _buy_ta else None
            if _rsi_e is not None:
                _rsi_bkt = ("oversold" if _rsi_e < 35 else
                            "neutral"  if _rsi_e < 55 else
                            "momentum" if _rsi_e < 70 else "overbought")
                _rsi_perf = tlog.setdefault("rsi_entry_perf", {})
                _rp2 = _rsi_perf.setdefault(_rsi_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
                _rp2["total"] = _rp2.get("total", 0) + 1
                _rp2["total_pnl"] = round(_rp2.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _rp2["wins"] = _rp2.get("wins", 0) + 1
                else: _rp2["losses"] = _rp2.get("losses", 0) + 1
                if _rp2["total"] > 0:
                    _rp2["win_rate"] = round(_rp2["wins"] / _rp2["total"] * 100, 1)
                    _rp2["avg_pnl"]  = round(_rp2["total_pnl"] / _rp2["total"], 2)
        except Exception:
            pass

    # ── Macro Event Neuron: learn FOMC/CPI/NFP trade outcomes ─────────────────
    # Tracks performance of trades entered on macro event days vs. normal days.
    # Learns: should the bot avoid entering on FOMC day? Does day_before hurt?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_tb = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _macro_lbl = _buy_tb.get("macro_label", "normal") if _buy_tb else "normal"
            _macro_ev  = _buy_tb.get("macro_event", "none") if _buy_tb else "none"
            _macro_key = f"{_macro_lbl}_{_macro_ev}" if _macro_ev != "none" else "normal"
            _macro_perf = tlog.setdefault("macro_event_perf", {})
            _mp = _macro_perf.setdefault(_macro_key, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0})
            _mp["total"] = _mp.get("total", 0) + 1
            _mp["total_pnl"] = round(_mp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _mp["wins"] = _mp.get("wins", 0) + 1
            else: _mp["losses"] = _mp.get("losses", 0) + 1
            if _mp["total"] > 0:
                _mp["win_rate"] = round(_mp["wins"] / _mp["total"] * 100, 1)
                _mp["avg_pnl"]  = round(_mp["total_pnl"] / _mp["total"], 2)
        except Exception:
            pass

    # ── Signal Count Neuron: learn optimal # of confirming signals ────────────
    # Tracks whether trades with 1-3 signals vs 4-6 vs 7+ perform differently.
    # A sweet spot (e.g., 4-6 confirming signals) gets a score boost; extreme
    # counts (too few = weak, too many = late/crowded) get a penalty.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_tc = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _sc = _buy_tc.get("signal_count_at_entry", 0) if _buy_tc else 0
            _sc_bkt = ("1-3" if _sc <= 3 else "4-6" if _sc <= 6 else "7-10" if _sc <= 10 else "11+")
            _sc_perf = tlog.setdefault("signal_count_perf", {})
            _scp = _sc_perf.setdefault(_sc_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _sc_bkt})
            _scp["total"] = _scp.get("total", 0) + 1
            _scp["total_pnl"] = round(_scp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _scp["wins"] = _scp.get("wins", 0) + 1
            else: _scp["losses"] = _scp.get("losses", 0) + 1
            if _scp["total"] > 0:
                _scp["win_rate"] = round(_scp["wins"] / _scp["total"] * 100, 1)
                _scp["avg_pnl"]  = round(_scp["total_pnl"] / _scp["total"], 2)
        except Exception:
            pass

    # ── Portfolio Concentration Neuron: # positions held at entry vs outcome ────
    # Learns whether the bot performs better when it's concentrated (1-2 bets)
    # or diversified (5+ positions). This informs optimal portfolio sizing.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_cp = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _conc_bkt = _buy_cp.get("concentration_bucket", "3-4") if _buy_cp else "3-4"
            _conc_perf = tlog.setdefault("concentration_perf", {})
            _cpp = _conc_perf.setdefault(_conc_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _conc_bkt})
            _cpp["total"] = _cpp.get("total", 0) + 1
            _cpp["total_pnl"] = round(_cpp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _cpp["wins"] = _cpp.get("wins", 0) + 1
            else: _cpp["losses"] = _cpp.get("losses", 0) + 1
            if _cpp["total"] > 0:
                _cpp["win_rate"] = round(_cpp["wins"] / _cpp["total"] * 100, 1)
                _cpp["avg_pnl"]  = round(_cpp["total_pnl"] / _cpp["total"], 2)
        except Exception:
            pass

    # ── Day-of-Week Neuron: learn which weekdays produce best outcomes ────────
    # Mon morning = directional gaps from weekend news; Wed = mid-week trend confirmation;
    # Fri = end-of-week position trimming by institutions can hurt longs.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_dw = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _day_nm = _buy_dw.get("day_name", "Mon") if _buy_dw else "Mon"
            _dow_perf = tlog.setdefault("dow_perf", {})
            _dwp = _dow_perf.setdefault(_day_nm, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "day": _day_nm})
            _dwp["total"] = _dwp.get("total", 0) + 1
            _dwp["total_pnl"] = round(_dwp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _dwp["wins"] = _dwp.get("wins", 0) + 1
            else: _dwp["losses"] = _dwp.get("losses", 0) + 1
            if _dwp["total"] > 0:
                _dwp["win_rate"] = round(_dwp["wins"] / _dwp["total"] * 100, 1)
                _dwp["avg_pnl"]  = round(_dwp["total_pnl"] / _dwp["total"], 2)
        except Exception:
            pass

    # ── Re-Entry Success Neuron: track outcomes of buying back recent sells ───────
    # Splits re-entry outcomes by type: winner re-entry (sold for profit, bought back)
    # vs loser re-entry (sold for loss, bought back). Helps distinguish "adding to
    # a working theme" from "falling knife" behavior.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_re = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            if _buy_re and _buy_re.get("is_reentry"):
                _re_type = _buy_re.get("reentry_type", "unknown")
                _re_perf = tlog.setdefault("reentry_perf", {})
                _rep = _re_perf.setdefault(_re_type, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "type": _re_type})
                _rep["total"] = _rep.get("total", 0) + 1
                _rep["total_pnl"] = round(_rep.get("total_pnl", 0.0) + pnl, 2)
                if pnl > 0: _rep["wins"] = _rep.get("wins", 0) + 1
                else: _rep["losses"] = _rep.get("losses", 0) + 1
                if _rep["total"] > 0:
                    _rep["win_rate"] = round(_rep["wins"] / _rep["total"] * 100, 1)
                    _rep["avg_pnl"]  = round(_rep["total_pnl"] / _rep["total"], 2)
        except Exception:
            pass

    # ── Position Size Neuron: does bet size correlate with outcome? ──────────────
    # Tracks whether small (<2%), medium (2-5%), large (5-10%), or outsized (10%+)
    # positions produce different risk-adjusted returns. The bot learns its optimal bet.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_ps = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _ps_bkt = _buy_ps.get("pos_size_bucket", "2-5%") if _buy_ps else "2-5%"
            _ps_perf = tlog.setdefault("pos_size_perf", {})
            _psp = _ps_perf.setdefault(_ps_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _ps_bkt})
            _psp["total"] = _psp.get("total", 0) + 1
            _psp["total_pnl"] = round(_psp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _psp["wins"] = _psp.get("wins", 0) + 1
            else: _psp["losses"] = _psp.get("losses", 0) + 1
            if _psp["total"] > 0:
                _psp["win_rate"] = round(_psp["wins"] / _psp["total"] * 100, 1)
                _psp["avg_pnl"]  = round(_psp["total_pnl"] / _psp["total"], 2)
        except Exception:
            pass

    # ── News Catalyst Urgency Neuron: high-urgency catalyst vs normal entries ────
    # Tracks win rates for trades entered on high-urgency catalysts (earnings beats,
    # FDA approval, M&A) vs medium (analyst upgrade) vs low/none. Tests if AI-classified
    # urgency level predicts trade success.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_nu = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _nu_bkt = _buy_nu.get("urgency_bucket", "low") if _buy_nu else "low"
            _nu_perf = tlog.setdefault("urgency_perf", {})
            _nup = _nu_perf.setdefault(_nu_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _nu_bkt})
            _nup["total"] = _nup.get("total", 0) + 1
            _nup["total_pnl"] = round(_nup.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _nup["wins"] = _nup.get("wins", 0) + 1
            else: _nup["losses"] = _nup.get("losses", 0) + 1
            if _nup["total"] > 0:
                _nup["win_rate"] = round(_nup["wins"] / _nup["total"] * 100, 1)
                _nup["avg_pnl"]  = round(_nup["total_pnl"] / _nup["total"], 2)
        except Exception:
            pass

    # ── VWAP Position Neuron: above/below VWAP at entry vs outcome ───────────────
    # VWAP is the institutional benchmark price for the day.
    # Above VWAP = buyers in control (institutional accumulation); below = distribution.
    # The bot learns whether above-VWAP entries produce consistently better outcomes.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_vw = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _vw_bkt = _buy_vw.get("vwap_bucket", "at_vwap") if _buy_vw else "at_vwap"
            _vw_perf = tlog.setdefault("vwap_perf", {})
            _vwp = _vw_perf.setdefault(_vw_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _vw_bkt})
            _vwp["total"] = _vwp.get("total", 0) + 1
            _vwp["total_pnl"] = round(_vwp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _vwp["wins"] = _vwp.get("wins", 0) + 1
            else: _vwp["losses"] = _vwp.get("losses", 0) + 1
            if _vwp["total"] > 0:
                _vwp["win_rate"] = round(_vwp["wins"] / _vwp["total"] * 100, 1)
                _vwp["avg_pnl"]  = round(_vwp["total_pnl"] / _vwp["total"], 2)
        except Exception:
            pass

    # ── MACD State Neuron: MACD phase at entry vs trade outcome ──────────────────
    # Bull div (MACD diverging from price) = highest conviction; rising = confirming trend;
    # recovering = crossing from negative; negative = counter-trend entry risk.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_mc = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _mc_bkt = _buy_mc.get("macd_state", "negative") if _buy_mc else "negative"
            _mc_perf = tlog.setdefault("macd_state_perf", {})
            _mcp = _mc_perf.setdefault(_mc_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "state": _mc_bkt})
            _mcp["total"] = _mcp.get("total", 0) + 1
            _mcp["total_pnl"] = round(_mcp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _mcp["wins"] = _mcp.get("wins", 0) + 1
            else: _mcp["losses"] = _mcp.get("losses", 0) + 1
            if _mcp["total"] > 0:
                _mcp["win_rate"] = round(_mcp["wins"] / _mcp["total"] * 100, 1)
                _mcp["avg_pnl"]  = round(_mcp["total_pnl"] / _mcp["total"], 2)
        except Exception:
            pass

    # ── TTM Squeeze Breakout Neuron: coiled spring setups vs normal entries ──────
    # TTM Squeeze = low-volatility compression → explosive directional breakout.
    # The bot learns whether squeeze-fired entries outperform standard momentum entries.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_sq = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _sq_fired = _buy_sq.get("squeeze_fired", False) if _buy_sq else False
            _sq_key = "squeeze" if _sq_fired else "no_squeeze"
            _sq_perf = tlog.setdefault("squeeze_perf", {})
            _sqp = _sq_perf.setdefault(_sq_key, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "type": _sq_key})
            _sqp["total"] = _sqp.get("total", 0) + 1
            _sqp["total_pnl"] = round(_sqp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _sqp["wins"] = _sqp.get("wins", 0) + 1
            else: _sqp["losses"] = _sqp.get("losses", 0) + 1
            if _sqp["total"] > 0:
                _sqp["win_rate"] = round(_sqp["wins"] / _sqp["total"] * 100, 1)
                _sqp["avg_pnl"]  = round(_sqp["total_pnl"] / _sqp["total"], 2)
        except Exception:
            pass

    # ── Institutional Accumulation Neuron: smart money buying vs outcome ────────
    # Tracks whether heavy institutional accumulation at entry predicts better outcomes.
    # High accum_score = OBV rising + MFI bullish + demand zone + options flow all agree.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_ac = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _ac_bkt = _buy_ac.get("accum_bucket", "light") if _buy_ac else "light"
            _ac_perf = tlog.setdefault("accum_perf", {})
            _acp = _ac_perf.setdefault(_ac_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _ac_bkt})
            _acp["total"] = _acp.get("total", 0) + 1
            _acp["total_pnl"] = round(_acp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _acp["wins"] = _acp.get("wins", 0) + 1
            else: _acp["losses"] = _acp.get("losses", 0) + 1
            if _acp["total"] > 0:
                _acp["win_rate"] = round(_acp["wins"] / _acp["total"] * 100, 1)
                _acp["avg_pnl"]  = round(_acp["total_pnl"] / _acp["total"], 2)
        except Exception:
            pass

    # ── RS Rating Neuron: IBD-style Relative Strength at entry vs outcome ────────
    # Tracks win rates for elite RS stocks (90+) vs average (50-75) vs weak (<50).
    # IBD data shows RS90+ stocks outperform the market 4:1 — let's verify with live data.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_rs = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _rs_bkt = _buy_rs.get("rs_bucket", "average") if _buy_rs else "average"
            _rs_perf = tlog.setdefault("rs_rating_perf", {})
            _rsp = _rs_perf.setdefault(_rs_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _rs_bkt})
            _rsp["total"] = _rsp.get("total", 0) + 1
            _rsp["total_pnl"] = round(_rsp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _rsp["wins"] = _rsp.get("wins", 0) + 1
            else: _rsp["losses"] = _rsp.get("losses", 0) + 1
            if _rsp["total"] > 0:
                _rsp["win_rate"] = round(_rsp["wins"] / _rsp["total"] * 100, 1)
                _rsp["avg_pnl"]  = round(_rsp["total_pnl"] / _rsp["total"], 2)
        except Exception:
            pass

    # ── Trend Template Neuron: O'Neil quality score at entry vs outcome ──────────
    # Tracks whether high-quality (TT≥7) setups outperform fair (TT 3-4) entries.
    # The O'Neil Trend Template is the institutional-grade entry quality benchmark.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_tt = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _tt_bkt = _buy_tt.get("tt_bucket", "fair") if _buy_tt else "fair"
            _tt_perf = tlog.setdefault("tt_perf", {})
            _ttp = _tt_perf.setdefault(_tt_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _tt_bkt})
            _ttp["total"] = _ttp.get("total", 0) + 1
            _ttp["total_pnl"] = round(_ttp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _ttp["wins"] = _ttp.get("wins", 0) + 1
            else: _ttp["losses"] = _ttp.get("losses", 0) + 1
            if _ttp["total"] > 0:
                _ttp["win_rate"] = round(_ttp["wins"] / _ttp["total"] * 100, 1)
                _ttp["avg_pnl"]  = round(_ttp["total_pnl"] / _ttp["total"], 2)
        except Exception:
            pass

    # ── Consecutive Green Days Neuron: momentum confirmation vs outcome ──────────
    # Tracks performance by # of consecutive up-days before entry.
    # Learns: is 2-3 green days the sweet spot, or does it signal exhaustion (4d+)?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_cg = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _cg_bkt = _buy_cg.get("consec_green_bucket", "0d") if _buy_cg else "0d"
            _cg_perf = tlog.setdefault("consec_green_perf", {})
            _cgp = _cg_perf.setdefault(_cg_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _cg_bkt})
            _cgp["total"] = _cgp.get("total", 0) + 1
            _cgp["total_pnl"] = round(_cgp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _cgp["wins"] = _cgp.get("wins", 0) + 1
            else: _cgp["losses"] = _cgp.get("losses", 0) + 1
            if _cgp["total"] > 0:
                _cgp["win_rate"] = round(_cgp["wins"] / _cgp["total"] * 100, 1)
                _cgp["avg_pnl"]  = round(_cgp["total_pnl"] / _cgp["total"], 2)
        except Exception:
            pass

    # ── Sector Momentum Neuron: sector ETF accelerating/neutral/decelerating ──
    # Tracks whether the whole sector was accelerating when the trade was entered.
    # Learns: do entries during sector acceleration phases produce better outcomes?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_sm = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _sm_bkt = _buy_sm.get("sector_etf_momentum", "neutral") if _buy_sm else "neutral"
            _sm_perf = tlog.setdefault("sector_momentum_perf", {})
            _smp = _sm_perf.setdefault(_sm_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "momentum": _sm_bkt})
            _smp["total"] = _smp.get("total", 0) + 1
            _smp["total_pnl"] = round(_smp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _smp["wins"] = _smp.get("wins", 0) + 1
            else: _smp["losses"] = _smp.get("losses", 0) + 1
            if _smp["total"] > 0:
                _smp["win_rate"] = round(_smp["wins"] / _smp["total"] * 100, 1)
                _smp["avg_pnl"]  = round(_smp["total_pnl"] / _smp["total"], 2)
        except Exception:
            pass

    # ── Pre-Market Gap Neuron: learn if gap-up entries outperform gap-down ──────
    # Tracks performance when stock had big pre-market gap (>3%), small gap, or flat open.
    # Helps the bot avoid gap-and-fail traps while confirming gap-and-hold strength.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_pg = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _pg_bkt = _buy_pg.get("pm_gap_bucket", "flat") if _buy_pg else "flat"
            _pg_perf = tlog.setdefault("pm_gap_perf", {})
            _pgp = _pg_perf.setdefault(_pg_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _pg_bkt})
            _pgp["total"] = _pgp.get("total", 0) + 1
            _pgp["total_pnl"] = round(_pgp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _pgp["wins"] = _pgp.get("wins", 0) + 1
            else: _pgp["losses"] = _pgp.get("losses", 0) + 1
            if _pgp["total"] > 0:
                _pgp["win_rate"] = round(_pgp["wins"] / _pgp["total"] * 100, 1)
                _pgp["avg_pnl"]  = round(_pgp["total_pnl"] / _pgp["total"], 2)
        except Exception:
            pass

    # ── Exit Timing Neuron: learn which hour of day produces best exits ────────
    # Tracks P&L by exit hour (UTC). The bot learns: do early exits (9-10am ET)
    # outperform vs holding to power hour (3pm ET)?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _exit_h = datetime.now(timezone.utc).hour
            _exit_h_str = str(_exit_h)
            _exit_perf = tlog.setdefault("exit_hour_perf", {})
            _ehp = _exit_perf.setdefault(_exit_h_str, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "hour_utc": _exit_h})
            _ehp["total"] = _ehp.get("total", 0) + 1
            _ehp["total_pnl"] = round(_ehp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _ehp["wins"] = _ehp.get("wins", 0) + 1
            else: _ehp["losses"] = _ehp.get("losses", 0) + 1
            if _ehp["total"] > 0:
                _ehp["win_rate"] = round(_ehp["wins"] / _ehp["total"] * 100, 1)
                _ehp["avg_pnl"]  = round(_ehp["total_pnl"] / _ehp["total"], 2)
        except Exception:
            pass

    # ── ATR Stop Distance Neuron: learn optimal volatility at entry ───────────
    # Tracks whether high-ATR (volatile) or low-ATR (calm) entries produce better outcomes.
    # The bot learns if it should prefer tight or wide stop distances.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_at = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _at_bkt = _buy_at.get("atr_bucket", "1-2%") if _buy_at else "1-2%"
            _at_perf = tlog.setdefault("atr_perf", {})
            _atp = _at_perf.setdefault(_at_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _at_bkt})
            _atp["total"] = _atp.get("total", 0) + 1
            _atp["total_pnl"] = round(_atp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _atp["wins"] = _atp.get("wins", 0) + 1
            else: _atp["losses"] = _atp.get("losses", 0) + 1
            if _atp["total"] > 0:
                _atp["win_rate"] = round(_atp["wins"] / _atp["total"] * 100, 1)
                _atp["avg_pnl"]  = round(_atp["total_pnl"] / _atp["total"], 2)
        except Exception:
            pass

    # ── Score Trend Neuron: was the score rising or falling at entry? ────────────
    # Rising score = momentum building, flat = waiting for a move, falling = late entry.
    # The bot learns whether entries during rising score phases produce better outcomes.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_st = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _st_trend = _buy_st.get("score_trend", "flat") if _buy_st else "flat"
            _st_perf = tlog.setdefault("score_trend_perf", {})
            _stp = _st_perf.setdefault(_st_trend, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "trend": _st_trend})
            _stp["total"] = _stp.get("total", 0) + 1
            _stp["total_pnl"] = round(_stp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _stp["wins"] = _stp.get("wins", 0) + 1
            else: _stp["losses"] = _stp.get("losses", 0) + 1
            if _stp["total"] > 0:
                _stp["win_rate"] = round(_stp["wins"] / _stp["total"] * 100, 1)
                _stp["avg_pnl"]  = round(_stp["total_pnl"] / _stp["total"], 2)
        except Exception:
            pass

    # ── SPY Day Return Neuron: learn how market direction at entry affects outcome ──
    # Tracks win rates when SPY was up >0.5% (up), flat (-0.5 to 0.5%), or down <-0.5%.
    # The bot learns: do entries on red SPY days fail? Do green SPY days help?
    # This is a key market regime micro-signal that goes beyond just "bull/bear".
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_td = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _spy_bkt = _buy_td.get("spy_day_bucket", "flat") if _buy_td else "flat"
            _spy_ret = _buy_td.get("spy_day_return", 0.0) if _buy_td else 0.0
            _spy_perf = tlog.setdefault("spy_day_perf", {})
            _sdp = _spy_perf.setdefault(_spy_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0,
                                                     "bucket": _spy_bkt, "spy_returns": []})
            _sdp["total"] = _sdp.get("total", 0) + 1
            _sdp["total_pnl"] = round(_sdp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _sdp["wins"] = _sdp.get("wins", 0) + 1
            else: _sdp["losses"] = _sdp.get("losses", 0) + 1
            if _sdp["total"] > 0:
                _sdp["win_rate"] = round(_sdp["wins"] / _sdp["total"] * 100, 1)
                _sdp["avg_pnl"]  = round(_sdp["total_pnl"] / _sdp["total"], 2)
            # Keep recent SPY returns for correlation insight
            _sdp.setdefault("spy_returns", []).append(round(float(_spy_ret), 2))
            _sdp["spy_returns"] = _sdp["spy_returns"][-20:]
        except Exception:
            pass

    # ── POC Distance Neuron (42): entry distance from Volume Profile POC ────────
    # Tracks performance by how far price was above/below the Point of Control at entry.
    # POC = price level with the most volume traded = institutional price anchor.
    # Learns: do breakout entries (>2% above POC) outperform near-POC or below-POC entries?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_pc = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _pc_bkt = _buy_pc.get("poc_dist_bucket", "at_poc") if _buy_pc else "at_poc"
            _pc_perf = tlog.setdefault("poc_dist_perf", {})
            _pcp = _pc_perf.setdefault(_pc_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _pc_bkt})
            _pcp["total"] = _pcp.get("total", 0) + 1
            _pcp["total_pnl"] = round(_pcp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _pcp["wins"] = _pcp.get("wins", 0) + 1
            else:        _pcp["losses"] = _pcp.get("losses", 0) + 1
            if _pcp["total"] > 0:
                _pcp["win_rate"] = round(_pcp["wins"] / _pcp["total"] * 100, 1)
                _pcp["avg_pnl"]  = round(_pcp["total_pnl"] / _pcp["total"], 2)
        except Exception:
            pass

    # ── MFI Zone Neuron (49): Money Flow Index (volume-weighted RSI) at entry ─────
    # Tracks performance by MFI zone at entry: distribution (>80) vs neutral vs accumulation (<30).
    # MFI combines price + volume — harder to fake, more reliable than pure RSI.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_mi = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _mi_zone = _buy_mi.get("mfi_zone", "neutral") if _buy_mi else "neutral"
            _mi_perf = tlog.setdefault("mfi_zone_perf", {})
            _mip = _mi_perf.setdefault(_mi_zone, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "zone": _mi_zone})
            _mip["total"] = _mip.get("total", 0) + 1
            _mip["total_pnl"] = round(_mip.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _mip["wins"] = _mip.get("wins", 0) + 1
            else:        _mip["losses"] = _mip.get("losses", 0) + 1
            if _mip["total"] > 0:
                _mip["win_rate"] = round(_mip["wins"] / _mip["total"] * 100, 1)
                _mip["avg_pnl"]  = round(_mip["total_pnl"] / _mip["total"], 2)
        except Exception:
            pass

    # ── Options Flow Neuron (48): institutional options positioning at entry ──────
    # Tracks win rates when unusual call buying / bullish options flow confirmed the trade.
    # Learns: do flow-confirmed entries (smart money) outperform unconfirmed?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_of = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _of_tier = _buy_of.get("options_flow_tier", "neutral") if _buy_of else "neutral"
            _of_perf = tlog.setdefault("options_flow_perf", {})
            _ofp = _of_perf.setdefault(_of_tier, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "tier": _of_tier})
            _ofp["total"] = _ofp.get("total", 0) + 1
            _ofp["total_pnl"] = round(_ofp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _ofp["wins"] = _ofp.get("wins", 0) + 1
            else:        _ofp["losses"] = _ofp.get("losses", 0) + 1
            if _ofp["total"] > 0:
                _ofp["win_rate"] = round(_ofp["wins"] / _ofp["total"] * 100, 1)
                _ofp["avg_pnl"]  = round(_ofp["total_pnl"] / _ofp["total"], 2)
        except Exception:
            pass

    # ── Multi-Timeframe Alignment Neuron (47): all TFs aligned at entry? ────────
    # Tracks win rates when short/medium/long timeframes are all bullish (full alignment)
    # vs partially aligned vs mixed/none. Highest conviction trade = all agree.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_mf = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _mf_align = _buy_mf.get("mtf_alignment", "none") if _buy_mf else "none"
            _mf_perf = tlog.setdefault("mtf_align_perf", {})
            _mfp = _mf_perf.setdefault(_mf_align, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "alignment": _mf_align})
            _mfp["total"] = _mfp.get("total", 0) + 1
            _mfp["total_pnl"] = round(_mfp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _mfp["wins"] = _mfp.get("wins", 0) + 1
            else:        _mfp["losses"] = _mfp.get("losses", 0) + 1
            if _mfp["total"] > 0:
                _mfp["win_rate"] = round(_mfp["wins"] / _mfp["total"] * 100, 1)
                _mfp["avg_pnl"]  = round(_mfp["total_pnl"] / _mfp["total"], 2)
        except Exception:
            pass

    # ── Stochastic Zone Neuron (46): %K at entry — overbought/neutral/oversold ──
    # Tracks performance by Stochastic %K zone at entry time.
    # Overbought (>80) = chasing; neutral (20-80) = momentum zone; oversold (<20) = reversal.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_sk = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _sk_zone = _buy_sk.get("stoch_zone", "neutral") if _buy_sk else "neutral"
            _sk_perf = tlog.setdefault("stoch_zone_perf", {})
            _skp = _sk_perf.setdefault(_sk_zone, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "zone": _sk_zone})
            _skp["total"] = _skp.get("total", 0) + 1
            _skp["total_pnl"] = round(_skp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _skp["wins"] = _skp.get("wins", 0) + 1
            else:        _skp["losses"] = _skp.get("losses", 0) + 1
            if _skp["total"] > 0:
                _skp["win_rate"] = round(_skp["wins"] / _skp["total"] * 100, 1)
                _skp["avg_pnl"]  = round(_skp["total_pnl"] / _skp["total"], 2)
        except Exception:
            pass

    # ── RVOL Tier Neuron (45): actual relative volume ratio at entry ─────────────
    # Tracks performance by RVOL tier: explosive (5x+) vs strong (2-5x) vs normal vs weak.
    # Learns if extreme RVOL entries produce better outcomes than moderate volume surges.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_rv = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _rv_tier = _buy_rv.get("rvol_tier", "normal") if _buy_rv else "normal"
            _rv_perf = tlog.setdefault("rvol_tier_perf", {})
            _rvp = _rv_perf.setdefault(_rv_tier, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "tier": _rv_tier})
            _rvp["total"] = _rvp.get("total", 0) + 1
            _rvp["total_pnl"] = round(_rvp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _rvp["wins"] = _rvp.get("wins", 0) + 1
            else:        _rvp["losses"] = _rvp.get("losses", 0) + 1
            if _rvp["total"] > 0:
                _rvp["win_rate"] = round(_rvp["wins"] / _rvp["total"] * 100, 1)
                _rvp["avg_pnl"]  = round(_rvp["total_pnl"] / _rvp["total"], 2)
        except Exception:
            pass

    # ── ADX Trend Strength Neuron (44): directional trend conviction at entry ─────
    # Tracks performance by ADX strength at entry. Strong ADX (>25) = clear trend;
    # weak ADX (<15) = choppy, counter-trend, high risk for momentum strategies.
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_ax = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _ax_bkt = _buy_ax.get("adx_bucket", "developing") if _buy_ax else "developing"
            _ax_perf = tlog.setdefault("adx_perf", {})
            _axp = _ax_perf.setdefault(_ax_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _ax_bkt})
            _axp["total"] = _axp.get("total", 0) + 1
            _axp["total_pnl"] = round(_axp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _axp["wins"] = _axp.get("wins", 0) + 1
            else:        _axp["losses"] = _axp.get("losses", 0) + 1
            if _axp["total"] > 0:
                _axp["win_rate"] = round(_axp["wins"] / _axp["total"] * 100, 1)
                _axp["avg_pnl"]  = round(_axp["total_pnl"] / _axp["total"], 2)
        except Exception:
            pass

    # ── Intraday Momentum Neuron (43): % from open at entry vs outcome ───────────
    # Tracks performance by intraday momentum at entry time.
    # Extended (>5% from open) = chasing; Runner (2-5%) = breakout; Early (~0%) = ideal.
    # Learns: does early-day entry beat chasing afternoon runners?
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_im = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _im_bkt = _buy_im.get("intraday_mom_bucket", "early") if _buy_im else "early"
            _im_perf = tlog.setdefault("intraday_mom_perf", {})
            _imp = _im_perf.setdefault(_im_bkt, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "bucket": _im_bkt})
            _imp["total"] = _imp.get("total", 0) + 1
            _imp["total_pnl"] = round(_imp.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _imp["wins"] = _imp.get("wins", 0) + 1
            else:        _imp["losses"] = _imp.get("losses", 0) + 1
            if _imp["total"] > 0:
                _imp["win_rate"] = round(_imp["wins"] / _imp["total"] * 100, 1)
                _imp["avg_pnl"]  = round(_imp["total_pnl"] / _imp["total"], 2)
        except Exception:
            pass

    # ── Score Decay Neuron (41): did exiting on score collapse save money? ───────
    # Tracks whether trades that exited due to score decay produced better or worse P&L
    # than trades that held through score decline. Learns the optimal decay threshold.
    # Buckets: "decay_exit" (score decay was the reason), "held_with_decay" (score fell
    # but bot held based on other signals), "no_decay" (score was stable at exit).
    if action in ("SELL", "SELL_HALF", "COVER") and pnl is not None:
        try:
            _buy_sd = next((t for t in tlog.get("trades", []) if t.get("action") == "BUY" and t.get("ticker") == sym), None)
            _sd_entry_score = _buy_sd.get("score", 0) if _buy_sd else 0
            _sd_reason = reason or ""
            if "score decay exit" in _sd_reason:
                _sd_key = "decay_exit"
            elif _sd_entry_score > 0 and signals and isinstance(signals, dict):
                # Check if score decayed even though we're exiting for another reason
                _sd_curr_sc = signals.get("live_score_at_sell", 0) or 0
                if _sd_curr_sc > 0 and (_sd_entry_score - _sd_curr_sc) >= 10:
                    _sd_key = "held_with_decay"
                else:
                    _sd_key = "no_decay"
            else:
                _sd_key = "no_decay"
            _sd_perf = tlog.setdefault("score_decay_perf", {})
            _sdp2 = _sd_perf.setdefault(_sd_key, {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0, "type": _sd_key})
            _sdp2["total"] = _sdp2.get("total", 0) + 1
            _sdp2["total_pnl"] = round(_sdp2.get("total_pnl", 0.0) + pnl, 2)
            if pnl > 0: _sdp2["wins"] = _sdp2.get("wins", 0) + 1
            else:        _sdp2["losses"] = _sdp2.get("losses", 0) + 1
            if _sdp2["total"] > 0:
                _sdp2["win_rate"] = round(_sdp2["wins"] / _sdp2["total"] * 100, 1)
                _sdp2["avg_pnl"]  = round(_sdp2["total_pnl"] / _sdp2["total"], 2)
        except Exception:
            pass


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
    Returns (vwap, position_pct, vwap_zscore, vwap_reclaim, band1_up, band2_up, band1_dn, band2_dn).
    vwap_zscore: price's z-score from VWAP (>2 = overbought band, <-2 = oversold band).
    vwap_reclaim: True if price dipped below VWAP intraday and has since reclaimed it.
    band1/band2: VWAP ± 1σ and ± 2σ levels (institutional target zones).
    """
    if hourly is None:
        return None, 50.0, 0.0, False, 0.0, 0.0, 0.0, 0.0
    try:
        if "Volume" not in hourly.columns or "Close" not in hourly.columns:
            return None, 50.0, 0.0, False, 0.0, 0.0, 0.0, 0.0
        h = hourly.dropna(subset=["Close", "Volume"])
        if len(h) < 2:
            return None, 50.0, 0.0, False, 0.0, 0.0, 0.0, 0.0
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
        # VWAP band levels
        b1u = round(vwap + vwap_std, 2)
        b2u = round(vwap + 2 * vwap_std, 2)
        b1d = round(vwap - vwap_std, 2)
        b2d = round(vwap - 2 * vwap_std, 2)

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

        return round(vwap, 2), round(vwap_pos, 2), round(vwap_z, 2), vwap_reclaim, b1u, b2u, b1d, b2d
    except Exception:
        return None, 50.0, 0.0, False, 0.0, 0.0, 0.0, 0.0


def _williams_r(closes, highs, lows, period=10):
    """Williams %R: -100 to 0; -80 to -100 = oversold (buy), 0 to -20 = overbought (sell)."""
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return -50.0
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    if hh == ll:
        return -50.0
    return round(-100 * (hh - closes[-1]) / (hh - ll), 1)


def _price_acceleration(closes, period=10):
    """Price Acceleration Filter: measures rate-of-change of rate-of-change (2nd derivative).
    Positive acceleration = momentum building (institutional accumulation accelerating).
    Negative acceleration = momentum decaying (smart money distributing).
    Returns (acceleration, is_accelerating, is_decelerating).
    """
    if len(closes) < period + 5:
        return 0.0, False, False
    try:
        # ROC at t and t-5: daily pct change smoothed
        def roc(c, i, n=5):
            if i >= n and c[i-n] != 0:
                return (c[i] - c[i-n]) / c[i-n] * 100
            return 0.0
        n = len(closes)
        roc_now  = roc(closes, n-1, period)
        roc_prev = roc(closes, n-1-5, period)
        accel = roc_now - roc_prev
        is_accelerating = accel > 1.0    # momentum gaining ≥1% over 5 days
        is_decelerating = accel < -1.0   # momentum losing ≥1% over 5 days
        return round(accel, 3), is_accelerating, is_decelerating
    except Exception:
        return 0.0, False, False


def _linear_regression_channel(closes, period=20):
    """Linear regression channel: price relative to its linear trend.
    Returns (slope_pct, r_squared, above_channel, below_channel, channel_width_pct).
    - slope_pct: annualized trend slope (%)
    - r_squared: trend linearity (>0.85 = very linear, <0.5 = choppy)
    - above_channel: price >1 std dev above regression line (overbought)
    - below_channel: price <1 std dev below regression line (oversold/bouncing)
    """
    if len(closes) < period:
        return 0.0, 0.0, False, False, 0.0
    try:
        y = closes[-period:]
        x = list(range(period))
        n = period
        sx  = sum(x)
        sy  = sum(y)
        sxy = sum(x[i]*y[i] for i in range(n))
        sxx = sum(xi*xi for xi in x)
        slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
        intercept = (sy - slope*sx) / n
        fitted = [intercept + slope*i for i in x]
        residuals = [y[i] - fitted[i] for i in range(n)]
        std = (sum(r*r for r in residuals) / n) ** 0.5
        # R-squared
        y_mean = sy / n
        ss_tot = sum((yi - y_mean)**2 for yi in y)
        ss_res = sum(r*r for r in residuals)
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
        cur_price = y[-1]
        cur_fitted = fitted[-1]
        above_channel = (cur_price > cur_fitted + std) if std > 0 else False
        below_channel = (cur_price < cur_fitted - std) if std > 0 else False
        # Annualize slope: slope is per bar (daily), multiply by 252
        slope_pct = (slope / (cur_fitted if cur_fitted else 1)) * 252 * 100
        channel_width_pct = (2 * std / cur_price * 100) if cur_price > 0 else 0.0
        return round(slope_pct, 2), round(r2, 3), above_channel, below_channel, round(channel_width_pct, 2)
    except Exception:
        return 0.0, 0.0, False, False, 0.0


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


_OPTIONS_FLOW_CACHE: dict = {}

def _options_flow(sym: str, max_age_sec: int = 3600) -> dict:
    """Options flow proxy using yfinance options chain.
    Computes put/call volume ratio and flags unusual options activity.
    Returns dict: pcr (put/call ratio), unusual_calls, unusual_puts, bullish_flow, bearish_flow.
    - pcr < 0.7: bullish sentiment (more calls than puts)
    - pcr > 1.3: bearish sentiment (more puts than calls)
    - unusual_calls: call volume > 3× average open interest (institutional positioning)
    Cached for 1 hour (options chain rarely changes intraday during off-hours).
    """
    now = datetime.now(timezone.utc).timestamp()
    cache = _OPTIONS_FLOW_CACHE.get(sym)
    if cache and now - cache.get("ts", 0) < max_age_sec:
        return cache
    result = {"pcr": 1.0, "unusual_calls": False, "unusual_puts": False,
              "bullish_flow": False, "bearish_flow": False, "ts": now}
    try:
        tk = yf.Ticker(sym)
        exps = tk.options
        if not exps:
            _OPTIONS_FLOW_CACHE[sym] = result
            return result
        # Use near-term expiry (first 1-2 available) for fresh institutional signal
        total_call_vol = 0
        total_put_vol  = 0
        total_call_oi  = 0
        total_put_oi   = 0
        max_call_vcr   = 0.0  # max single-strike volume/OI ratio for calls
        max_put_vcr    = 0.0
        for exp in exps[:2]:
            chain = tk.option_chain(exp)
            calls = chain.calls
            puts  = chain.puts
            if hasattr(calls, "volume") and hasattr(calls, "openInterest"):
                cv = calls["volume"].fillna(0).sum()
                coi = calls["openInterest"].fillna(0).sum()
                total_call_vol += cv
                total_call_oi  += coi
                # Unusual call activity: any strike with vol/OI > 3
                if coi > 0:
                    vcr = (calls["volume"].fillna(0) / calls["openInterest"].replace(0, 1)).max()
                    max_call_vcr = max(max_call_vcr, vcr)
            if hasattr(puts, "volume") and hasattr(puts, "openInterest"):
                pv = puts["volume"].fillna(0).sum()
                poi = puts["openInterest"].fillna(0).sum()
                total_put_vol += pv
                total_put_oi  += poi
                if poi > 0:
                    vcr = (puts["volume"].fillna(0) / puts["openInterest"].replace(0, 1)).max()
                    max_put_vcr = max(max_put_vcr, vcr)
        pcr = total_put_vol / max(total_call_vol, 1)
        unusual_calls = max_call_vcr > 3.0 and total_call_vol > 500
        unusual_puts  = max_put_vcr  > 3.0 and total_put_vol  > 500
        result = {
            "pcr":           round(pcr, 3),
            "unusual_calls": unusual_calls,
            "unusual_puts":  unusual_puts,
            "bullish_flow":  pcr < 0.7 or unusual_calls,
            "bearish_flow":  pcr > 1.3 or unusual_puts,
            "call_vol":      int(total_call_vol),
            "put_vol":       int(total_put_vol),
            "ts":            now,
        }
    except Exception:
        pass
    _OPTIONS_FLOW_CACHE[sym] = result
    return result


_GEX_CACHE: dict = {}

def _gamma_exposure(sym: str, max_age_sec: int = 1800) -> dict:
    """Gamma Exposure (GEX) proxy from yfinance options chain.

    GEX = sum(gamma × OI × 100 × price²) for calls minus puts near ATM.
    Positive GEX = dealers are long gamma = price tends to revert (mean-reverting).
    Negative GEX = dealers are short gamma = amplified moves (trending/volatile).

    Also identifies 'gamma walls': strikes with very high OI that act as magnets.
    Returns: gex_sign (+1/-1), call_gex, put_gex, gamma_wall_up, gamma_wall_down, squeeze_potential.
    """
    import time as _time
    now = _time.time()
    if sym in _GEX_CACHE:
        cached, ts = _GEX_CACHE[sym]
        if now - ts < max_age_sec:
            return cached
    result = {
        "gex_sign": 0, "call_gex": 0.0, "put_gex": 0.0,
        "gamma_wall_up": 0.0, "gamma_wall_down": 0.0, "squeeze_potential": False,
    }
    try:
        tk = yf.Ticker(sym)
        fi = tk.fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
        if not price or float(price) <= 0:
            _GEX_CACHE[sym] = (result, now)
            return result
        price = float(price)
        exps = tk.options
        if not exps:
            _GEX_CACHE[sym] = (result, now)
            return result
        # Use nearest 2-3 expirations (most gamma exposure is near-term)
        today = datetime.now(timezone.utc).date()
        near_exps = []
        for exp in exps[:5]:
            try:
                from datetime import date as _dt_date
                exp_date = _dt_date.fromisoformat(exp)
                days_out = (exp_date - today).days
                if 1 <= days_out <= 45:
                    near_exps.append(exp)
                    if len(near_exps) >= 3:
                        break
            except Exception:
                pass
        if not near_exps:
            _GEX_CACHE[sym] = (result, now)
            return result
        call_gex = 0.0
        put_gex  = 0.0
        call_oi_by_strike: dict = {}
        put_oi_by_strike:  dict = {}
        for exp in near_exps:
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls
                puts  = chain.puts
                # Near-the-money: within 15% of current price
                ntm_calls = calls[(calls["strike"] >= price * 0.85) & (calls["strike"] <= price * 1.15)]
                ntm_puts  = puts[ (puts["strike"]  >= price * 0.85) & (puts["strike"]  <= price * 1.15)]
                for _, row in ntm_calls.iterrows():
                    g  = float(row.get("gamma", 0) or 0)
                    oi = float(row.get("openInterest", 0) or 0)
                    k  = float(row["strike"])
                    gex_contrib = g * oi * 100 * price * price
                    call_gex += gex_contrib
                    call_oi_by_strike[k] = call_oi_by_strike.get(k, 0) + oi
                for _, row in ntm_puts.iterrows():
                    g  = float(row.get("gamma", 0) or 0)
                    oi = float(row.get("openInterest", 0) or 0)
                    k  = float(row["strike"])
                    gex_contrib = g * oi * 100 * price * price
                    put_gex += gex_contrib
                    put_oi_by_strike[k] = put_oi_by_strike.get(k, 0) + oi
            except Exception:
                pass
        net_gex = call_gex - put_gex
        gex_sign = 1 if net_gex > 0 else -1
        # Gamma walls: strikes with highest combined OI above/below current price
        above_strikes = {k: v for k, v in {**call_oi_by_strike, **put_oi_by_strike}.items() if k > price}
        below_strikes = {k: v for k, v in {**call_oi_by_strike, **put_oi_by_strike}.items() if k < price}
        gamma_wall_up   = max(above_strikes, key=above_strikes.get) if above_strikes else 0.0
        gamma_wall_down = max(below_strikes, key=below_strikes.get) if below_strikes else 0.0
        # Squeeze potential: large put wall below current price + negative GEX = gamma squeeze fuel
        squeeze_potential = (put_gex > call_gex * 2 and gamma_wall_down > 0
                             and abs(price - gamma_wall_down) / price < 0.05)
        result = {
            "gex_sign":        gex_sign,
            "call_gex":        round(call_gex, 0),
            "put_gex":         round(put_gex, 0),
            "gamma_wall_up":   round(gamma_wall_up, 2),
            "gamma_wall_down": round(gamma_wall_down, 2),
            "squeeze_potential": squeeze_potential,
        }
    except Exception:
        pass
    _GEX_CACHE[sym] = (result, now)
    return result


_SHORT_DATA_CACHE: dict = {}

def _short_data(sym: str, max_age_sec: int = 7200) -> dict:
    """Fetch short interest data (cached 2 hours — updates twice monthly).
    Returns: short_float (0-1), short_ratio (days to cover), high_short (bool >15%).
    """
    import time as _time
    now = _time.time()
    if sym in _SHORT_DATA_CACHE:
        cached, ts = _SHORT_DATA_CACHE[sym]
        if now - ts < max_age_sec:
            return cached
    result = {"short_float": 0.0, "short_ratio": 0.0, "high_short": False}
    try:
        info = yf.Ticker(sym).info
        sf = float(info.get("shortPercentOfFloat") or 0)
        sr = float(info.get("shortRatio") or 0)
        result = {
            "short_float": round(sf, 3),
            "short_ratio": round(sr, 1),
            "high_short":  sf > 0.15,
        }
    except Exception:
        pass
    _SHORT_DATA_CACHE[sym] = (result, now)
    return result


_ANALYST_CACHE: dict = {}

def _analyst_revisions(sym: str, max_age_sec: int = 14400) -> dict:
    """
    Fetch analyst estimate revisions and recommendation trends (4hr cache).
    Returns:
      upgrades_30d:    int — upgrades in last 30 days
      downgrades_30d:  int — downgrades in last 30 days
      net_revisions:   int — net (upgrades - downgrades), positive = bullish
      buy_pct:         float 0-1 — fraction of analysts with buy/strong buy
      analyst_upgrade: bool — net positive revisions in last 14 days
      analyst_rev_score: int 0-3 scoring for use in score()
      price_target:    float — median analyst price target (0 if unknown)
      upside_pct:      float — % to price target
    """
    import time as _time
    now = _time.time()
    if sym in _ANALYST_CACHE:
        cached, ts = _ANALYST_CACHE[sym]
        if now - ts < max_age_sec:
            return cached
    result = {
        "upgrades_30d": 0, "downgrades_30d": 0, "net_revisions": 0,
        "buy_pct": 0.5, "analyst_upgrade": False, "analyst_rev_score": 0,
        "price_target": 0.0, "upside_pct": 0.0,
    }
    try:
        tk_obj = yf.Ticker(sym)
        # Analyst recommendations (buy/sell/hold distribution)
        try:
            recs = tk_obj.recommendations
            if recs is not None and not recs.empty:
                # Get the most recent period
                latest = recs.iloc[-1] if len(recs) > 0 else None
                if latest is not None:
                    strong_buy = int(latest.get("strongBuy", 0) or 0)
                    buy        = int(latest.get("buy", 0) or 0)
                    hold       = int(latest.get("hold", 0) or 0)
                    sell       = int(latest.get("sell", 0) or 0)
                    strong_sell= int(latest.get("strongSell", 0) or 0)
                    total = strong_buy + buy + hold + sell + strong_sell
                    if total > 0:
                        result["buy_pct"] = round((strong_buy + buy) / total, 3)
        except Exception:
            pass
        # Upgrades/downgrades in last 30 days
        try:
            upgrades = tk_obj.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                import pandas as _pd
                cutoff = _pd.Timestamp.now(tz="UTC") - _pd.Timedelta(days=30)
                cutoff14= _pd.Timestamp.now(tz="UTC") - _pd.Timedelta(days=14)
                recent = upgrades[upgrades.index >= cutoff] if hasattr(upgrades.index, 'tz') else upgrades.tail(20)
                recent14= upgrades[upgrades.index >= cutoff14] if hasattr(upgrades.index, 'tz') else upgrades.tail(10)
                _ups  = lambda df: int(df["Action"].str.upper().isin(["UPGRADE", "INIT", "REITERATED"]).sum()) if "Action" in df.columns else 0
                _dns  = lambda df: int(df["Action"].str.upper().isin(["DOWNGRADE", "DOWNGRADED"]).sum()) if "Action" in df.columns else 0
                result["upgrades_30d"]  = _ups(recent)
                result["downgrades_30d"]= _dns(recent)
                result["net_revisions"] = result["upgrades_30d"] - result["downgrades_30d"]
                # Upgrade in last 14 days = fresh signal
                result["analyst_upgrade"] = _ups(recent14) > _dns(recent14)
        except Exception:
            pass
        # Analyst price target from info
        try:
            info = tk_obj.fast_info
            pt = getattr(info, "target_price", None) or getattr(info, "analyst_target", None)
            if pt:
                result["price_target"] = round(float(pt), 2)
                cur = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
                if cur and cur > 0:
                    result["upside_pct"] = round((float(pt) - float(cur)) / float(cur) * 100, 1)
        except Exception:
            # Fallback to info dict
            try:
                info_d = tk_obj.info
                pt = info_d.get("targetMeanPrice") or info_d.get("targetMedianPrice") or 0
                cur = info_d.get("currentPrice") or info_d.get("regularMarketPrice") or 0
                if pt:
                    result["price_target"] = round(float(pt), 2)
                    if cur and cur > 0:
                        result["upside_pct"] = round((float(pt) - float(cur)) / float(cur) * 100, 1)
            except Exception:
                pass
        # Composite analyst revision score
        nr = result["net_revisions"]
        bp = result["buy_pct"]
        up_pct = result["upside_pct"]
        s = 0
        if nr >= 3:              s += 2  # strong upgrade wave
        elif nr >= 1:            s += 1  # more ups than downs
        elif nr <= -2:           s -= 1  # analyst downgrade wave
        if bp >= 0.75:           s += 1  # majority buy-rated
        if up_pct >= 15:         s += 1  # large upside to target
        elif up_pct >= 8:        s += 0  # moderate upside
        elif up_pct < -5:        s -= 1  # trading above target = risky
        result["analyst_rev_score"] = max(-2, min(3, s))
    except Exception:
        pass
    _ANALYST_CACHE[sym] = (result, now)
    return result


_FUNDAMENTAL_CACHE: dict = {}

def _get_fundamentals(sym: str, max_age_sec: int = 21600) -> dict:
    """
    Fetch fundamental quality metrics via yfinance .info (6hr cache to avoid rate limits).
    Returns earnings_growth, revenue_growth, forward_pe, profit_margin, roe, debt_equity.
    """
    import time as _ft
    _null = {"earnings_growth": None, "revenue_growth": None, "forward_pe": None,
             "profit_margin": None, "roe": None, "debt_equity": None,
             "fund_quality": 0}   # fund_quality: -2 to +3 score
    now_f = _ft.time()
    if sym in _FUNDAMENTAL_CACHE:
        cached, ts = _FUNDAMENTAL_CACHE[sym]
        if now_f - ts < max_age_sec:
            return cached
    try:
        info = yf.Ticker(sym).info
        eg = info.get("earningsGrowth")      # YoY EPS growth (decimal, e.g. 0.35 = 35%)
        rg = info.get("revenueGrowth")       # YoY revenue growth
        fpe= info.get("forwardPE")           # Forward P/E
        pm = info.get("profitMargins")       # Net profit margin
        roe= info.get("returnOnEquity")      # Return on equity
        de = info.get("debtToEquity")        # Debt/equity ratio

        # Composite fundamental quality score: -2 to +3
        fq = 0
        if eg is not None:
            if eg >= 0.25:  fq += 1   # strong earnings growth ≥25%
            elif eg <= 0:   fq -= 1   # earnings declining — avoid
        if rg is not None:
            if rg >= 0.15:  fq += 1   # strong revenue growth ≥15%
            elif rg <= -0.05: fq -= 1 # revenue shrinking
        if pm is not None and pm >= 0.15:  fq += 1  # high margin business (competitive moat)
        if roe is not None and roe >= 0.20: fq += 1 # excellent ROE ≥20% (capital efficiency)
        if de is not None and de > 2.0:   fq -= 1   # high debt = fragile in rising rate environment

        result = {"earnings_growth": eg, "revenue_growth": rg, "forward_pe": fpe,
                  "profit_margin": pm, "roe": roe, "debt_equity": de, "fund_quality": fq}
        _FUNDAMENTAL_CACHE[sym] = (result, now_f)
        return result
    except Exception:
        _FUNDAMENTAL_CACHE[sym] = (_null, now_f)
        return _null


_NEWS_VEL_CACHE: dict = {}

def _news_velocity(sym: str, max_age_sec: int = 1800) -> dict:
    """News velocity: count of yfinance news items in 24h vs prior 24h window.
    Accelerating news flow = catalyst building (often precedes big moves).
    Returns: count_24h, count_48h, velocity (ratio), accelerating (bool), headlines.
    """
    import time as _time
    now = _time.time()
    if sym in _NEWS_VEL_CACHE:
        cached, ts = _NEWS_VEL_CACHE[sym]
        if now - ts < max_age_sec:
            return cached
    result = {"count_24h": 0, "count_48h": 0, "velocity": 0.0, "accelerating": False, "headlines": []}
    try:
        news = yf.Ticker(sym).news[:15]
        if news:
            count_24h = sum(1 for n in news if (now - n.get("providerPublishTime", 0)) < 86400)
            count_48h = sum(1 for n in news if 86400 <= (now - n.get("providerPublishTime", 0)) < 172800)
            velocity  = (count_24h - count_48h) / max(1.0, count_48h)
            accel     = count_24h > count_48h + 1 and count_24h >= 3
            # Store top 4 recent headlines with timestamp
            _headlines = []
            for n in sorted(news, key=lambda x: x.get("providerPublishTime", 0), reverse=True)[:4]:
                title = n.get("title", "")
                pub   = n.get("providerPublishTime", 0)
                src   = n.get("publisher", "")
                url   = n.get("link", "") or n.get("url", "")
                if title:
                    age_h = round((now - pub) / 3600, 1) if pub else None
                    _headlines.append({"t": title[:120], "s": src[:30], "h": age_h, "u": url})
            # Classify the dominant catalyst type from recent headlines
            _all_titles = [n.get("title","") for n in news if n.get("title")]
            _cat_info = classify_catalyst(_all_titles)
            result = {
                "count_24h":      count_24h,
                "count_48h":      count_48h,
                "velocity":       round(velocity, 2),
                "accelerating":   accel,
                "headlines":      _headlines,
                "catalyst_type":  _cat_info["type"],
                "catalyst_urg":   _cat_info["urgency"],
                "catalyst_dir":   _cat_info["direction"],
            }
    except Exception:
        pass
    _NEWS_VEL_CACHE[sym] = (result, now)
    return result


_PREMARKET_CACHE: dict = {}

def _premarket_info(sym: str, max_age_sec: int = 300) -> dict:
    """Fetch pre-market price and compute gap vs prior close using yfinance fast_info.
    Returns: pre_price, gap_pct, gap_up (≥1.5%), gap_down (≤-1.5%),
             big_gap_up (≥3%), big_gap_down (≤-3%).
    """
    import time as _time
    now = _time.time()
    if sym in _PREMARKET_CACHE:
        cached, ts = _PREMARKET_CACHE[sym]
        if now - ts < max_age_sec:
            return cached
    result = {"pre_price": 0.0, "gap_pct": 0.0, "gap_up": False, "gap_down": False,
              "big_gap_up": False, "big_gap_down": False}
    try:
        fi = yf.Ticker(sym).fast_info
        pre  = (getattr(fi, "pre_market_price", None) or
                getattr(fi, "preMarketPrice", None))
        prev = (getattr(fi, "previous_close", None) or
                getattr(fi, "regularMarketPreviousClose", None) or
                getattr(fi, "last_price", None))
        if pre and prev and float(prev) > 0:
            pre, prev = float(pre), float(prev)
            gap = (pre - prev) / prev * 100
            result = {
                "pre_price":    round(pre, 2),
                "gap_pct":      round(gap, 2),
                "gap_up":       gap >=  1.5,
                "gap_down":     gap <= -1.5,
                "big_gap_up":   gap >=  3.0,
                "big_gap_down": gap <= -3.0,
            }
    except Exception:
        pass
    _PREMARKET_CACHE[sym] = (result, now)
    return result


def _parabolic_sar(highs, lows, af_start=0.02, af_max=0.20):
    """Parabolic SAR trailing stop — accelerates as price trends, tightens on reversals.
    Returns (sar_value, is_bullish) for the latest bar.
    - is_bullish=True: SAR below price (uptrend, use as trailing stop floor)
    - is_bullish=False: SAR above price (downtrend signal)
    Standard Wilder parameters: af_start=0.02, af_max=0.20.
    """
    try:
        n = len(highs)
        if n < 5:
            return 0.0, True
        sar = lows[0]
        ep  = highs[0]    # extreme point
        af  = af_start
        bull = True
        for i in range(1, n):
            prev_sar = sar
            if bull:
                sar = prev_sar + af * (ep - prev_sar)
                # SAR cannot be above prior 2 lows
                sar = min(sar, lows[i-1], lows[max(0, i-2)])
                if lows[i] < sar:
                    bull = False
                    sar  = ep           # SAR flips to EP (highest high)
                    ep   = lows[i]
                    af   = af_start
                else:
                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + af_start, af_max)
            else:
                sar = prev_sar + af * (ep - prev_sar)
                # SAR cannot be below prior 2 highs
                sar = max(sar, highs[i-1], highs[max(0, i-2)])
                if highs[i] > sar:
                    bull = True
                    sar  = ep           # SAR flips to EP (lowest low)
                    ep   = highs[i]
                    af   = af_start
                else:
                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + af_start, af_max)
        return round(float(sar), 4), bull
    except Exception:
        return 0.0, True


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

        # Sector rotation: XLK (tech) vs XLU (utilities) — risk-on vs defensive
        # When tech > utilities: growth mode. When utilities > tech: defensive/fear.
        try:
            sec_raw = yf.download("XLK XLU XLF XLV", period="15d", interval="1d",
                                   progress=False, auto_adjust=True)
            def _sec_closes(sym):
                try:
                    return list(sec_raw["Close"][sym].dropna())
                except Exception:
                    return []
            xlk_c = _sec_closes("XLK")
            xlu_c = _sec_closes("XLU")
            xlf_c = _sec_closes("XLF")
            xlv_c = _sec_closes("XLV")
            if len(xlk_c) >= 5 and len(xlu_c) >= 5:
                xlk_5d = (xlk_c[-1] - xlk_c[-5]) / xlk_c[-5] * 100
                xlu_5d = (xlu_c[-1] - xlu_c[-5]) / xlu_c[-5] * 100
                rot_score = xlk_5d - xlu_5d
                if   rot_score > 3:   score += 2   # tech crushing utilities = strong risk-on
                elif rot_score > 1:   score += 1
                elif rot_score < -3:  score -= 2   # utilities crushing tech = defensive rotation
                elif rot_score < -1:  score -= 1
            # Financials XLF: banks leading = economy expanding = bull
            if len(xlf_c) >= 5 and len(spy_closes) >= 5:
                xlf_5d = (xlf_c[-1] - xlf_c[-5]) / xlf_c[-5] * 100
                spy_5d_now = (spy_closes[-1] - spy_closes[-5]) / spy_closes[-5] * 100
                if xlf_5d > spy_5d_now + 1.5: score += 1   # financials leading = economic strength
                elif xlf_5d < spy_5d_now - 1.5: score -= 1
        except Exception:
            pass

        # SPY RSI: overbought market = headwind; oversold = opportunity
        try:
            if len(spy_closes) >= 14:
                spy_rsi = _rsi(spy_closes, 14)
                if spy_rsi < 35:  score += 1   # oversold market = contrarian buy setup
                elif spy_rsi > 75: score -= 1  # overbought market = reduced edge for longs
        except Exception:
            pass

        # SPY options put/call ratio: fear gauge
        try:
            spy_opts = _options_flow("SPY", max_age_sec=3600)
            spy_pcr = spy_opts.get("pcr", 1.0)
            if   spy_pcr < 0.7:  score += 1   # low put activity = complacency / bull
            elif spy_pcr > 1.4:  score -= 1   # high put buying = fear / protection buying
        except Exception:
            pass

        # VIX term structure: compare VIX9D (short-term) vs VIX (30-day) vs VIX3M
        # Backwardation (VIX9D > VIX) = acute fear = amplified risk; contango = normal
        vix9d = vix
        vix3m = vix
        vts_regime = "contango"   # normal; or "backwardation" / "inverted"
        try:
            vts_raw = yf.download("^VIX9D ^VIX3M", period="5d", interval="1d",
                                   progress=False, auto_adjust=True)
            def _vts_c(sym):
                try:
                    return list(vts_raw["Close"][sym].dropna())
                except Exception:
                    return []
            v9  = _vts_c("^VIX9D")
            v3m = _vts_c("^VIX3M")
            if v9:  vix9d = float(v9[-1])
            if v3m: vix3m = float(v3m[-1])
            if vix9d > vix * 1.05 and vix > vix3m:
                # Full backwardation: short-term fear > 30d fear > 3m fear
                vts_regime = "backwardation"
                score -= 2   # acute stress = reduce conviction
            elif vix9d > vix * 1.05:
                vts_regime = "inverted"
                score -= 1   # partial inversion = caution
            elif vix3m > vix * 1.05:
                vts_regime = "contango"   # healthy long-term vol > short-term = normal
                score += 1   # ideal buying conditions
        except Exception:
            pass

        # Cross-asset: DXY (Dollar Index) and TNX (10-Year Treasury Yield)
        # Dollar strength hurts multinationals; rising rates hurt growth valuations
        dxy_level = 0.0
        dxy_5d    = 0.0
        tnx_level = 0.0
        tnx_5d    = 0.0
        rate_environment = "neutral"
        try:
            ca_raw = yf.download("DX-Y.NYB ^TNX", period="15d", interval="1d",
                                  progress=False, auto_adjust=True, group_by="ticker")
            def _ca_c(sym):
                try:
                    return list(ca_raw["Close"][sym].dropna())
                except Exception:
                    return []
            dxy_c = _ca_c("DX-Y.NYB")
            tnx_c = _ca_c("^TNX")
            if len(dxy_c) >= 5:
                dxy_level = round(float(dxy_c[-1]), 2)
                dxy_5d    = round((dxy_c[-1] - dxy_c[-5]) / dxy_c[-5] * 100, 2)
                # Rapidly strengthening dollar = headwind for risk assets & commodities
                if   dxy_5d >  1.5: score -= 1
                elif dxy_5d < -1.5: score += 1   # dollar weakness = tailwind for global risk-on
            if len(tnx_c) >= 5:
                tnx_level = round(float(tnx_c[-1]), 2)
                tnx_5d    = round((tnx_c[-1] - tnx_c[-5]) / tnx_c[-5] * 100, 2)
                # High & rising rates compress growth valuations
                if   tnx_level > 5.0 and tnx_5d > 3:  score -= 2
                elif tnx_level > 4.5 and tnx_5d > 2:  score -= 1
                elif tnx_level < 4.0 and tnx_5d < -2: score += 1  # falling rates = supportive
            if tnx_level > 0:
                if   tnx_level > 5.0: rate_environment = "restrictive"
                elif tnx_level > 4.5: rate_environment = "elevated"
                elif tnx_level > 3.5: rate_environment = "neutral"
                else:                 rate_environment = "accommodative"
        except Exception:
            pass

        if score >= 2:    regime = "bull"
        elif score <= -2: regime = "bear"
        else:             regime = "neutral"

        logger.info(
            f"Market regime: {regime} | SPY trend: {spy_trend:+.1f}% | "
            f"VIX: {vix:.1f} (9d:{vix9d:.1f} 3m:{vix3m:.1f} {vts_regime}) | "
            f"DXY: {dxy_level:.1f} ({dxy_5d:+.2f}%5d) | TNX: {tnx_level:.2f}% ({rate_environment}) | "
            f"Above 200d: {above_200} | score: {score}"
        )
        return {"regime": regime, "vix": vix, "spy_trend": spy_trend,
                "score": score, "above_200": above_200,
                "vix9d": round(vix9d, 1), "vix3m": round(vix3m, 1),
                "vts_regime": vts_regime,
                "dxy_level": dxy_level, "dxy_5d": dxy_5d,
                "tnx_level": tnx_level, "tnx_5d": tnx_5d,
                "rate_environment": rate_environment}

    except Exception as e:
        logger.warning(f"Regime check failed: {e}")
        return {"regime": "neutral", "vix": 20.0, "spy_trend": 0.0,
                "score": 0, "above_200": True}


_DAY_TYPE_CACHE: list = []   # [result_dict, timestamp]

def intraday_day_type(max_age_sec: int = 300) -> dict:
    """
    Classify today's intraday character using SPY 5-min data.
    Returns:
      day_type:   'trend_up' | 'trend_down' | 'range' | 'choppy' | 'unknown'
      efficiency: 0.0-1.0 (net move / total path; high = trending)
      range_ratio:current range / 14d ATR ratio (>1.3 = expanded range)
      opening_bias: 'up' | 'down' | 'flat' (first 30min direction)
      day_score:  +2 (strong trend) to -1 (choppy/range)
      strategy_hint: 'breakout' | 'mean_reversion' | 'neutral'
    """
    import time as _time
    now_ts = _time.time()
    if len(_DAY_TYPE_CACHE) == 2 and now_ts - _DAY_TYPE_CACHE[1] < max_age_sec:
        return _DAY_TYPE_CACHE[0]
    _default = {"day_type": "unknown", "efficiency": 0.5, "range_ratio": 1.0,
                 "opening_bias": "flat", "day_score": 0, "strategy_hint": "neutral"}
    try:
        # Fetch 5-min SPY (today + 3d for ATR baseline)
        spy5 = yf.download("SPY", period="3d", interval="5m",
                            auto_adjust=True, progress=False)
        if spy5.empty or len(spy5) < 12:
            return _default
        closes5 = list(spy5["Close"].dropna())
        highs5  = list(spy5["High"].dropna())
        lows5   = list(spy5["Low"].dropna())
        opens5  = list(spy5["Open"].dropna())
        # Find today's session (last market day's bars)
        today_idx = []
        if hasattr(spy5.index, 'date'):
            last_date = spy5.index[-1].date()
            today_idx = [i for i, d in enumerate(spy5.index) if d.date() == last_date]
        if len(today_idx) < 6:
            return _default
        # Today's bars
        t_closes = [closes5[i] for i in today_idx]
        t_highs  = [highs5[i]  for i in today_idx]
        t_lows   = [lows5[i]   for i in today_idx]
        t_opens  = [opens5[i]  for i in today_idx]
        t_open   = t_opens[0] if t_opens else t_closes[0]
        t_last   = t_closes[-1]
        t_high   = max(t_highs)
        t_low    = min(t_lows)
        # Daily range
        day_range = t_high - t_low
        # Efficiency Ratio (Elder/Kaufman): net directional move / total path length
        net_move  = abs(t_last - t_open)
        path_sum  = sum(abs(t_closes[i] - t_closes[i-1]) for i in range(1, len(t_closes)))
        efficiency = round(net_move / max(path_sum, 0.0001), 3)
        # ATR baseline from prior 2 days of 5-min data (not today)
        prior_idx = [i for i in range(len(closes5)) if i not in today_idx]
        if len(prior_idx) >= 12:
            prior_h = [highs5[i]  for i in prior_idx[-48:]]
            prior_l = [lows5[i]   for i in prior_idx[-48:]]
            prior_c = [closes5[i] for i in prior_idx[-48:]]
            bars_per_day = max(len(today_idx), 12)
            # Aggregate into daily-equivalent bars for ATR estimate
            chunk = bars_per_day
            agg_ranges = []
            for st in range(0, len(prior_h) - chunk, chunk):
                agg_ranges.append(max(prior_h[st:st+chunk]) - min(prior_l[st:st+chunk]))
            avg_atr = sum(agg_ranges) / max(1, len(agg_ranges)) if agg_ranges else day_range
        else:
            avg_atr = day_range
        range_ratio = round(day_range / max(avg_atr, 0.01), 2)
        # Opening bias: first 30min (6 × 5-min bars)
        ob_bars = min(6, len(t_closes))
        open_close_ratio = (t_closes[ob_bars-1] - t_open) / max(abs(t_open) * 0.001, 0.01)
        opening_bias = "up" if open_close_ratio > 0.15 else ("down" if open_close_ratio < -0.15 else "flat")
        # Net direction of day
        net_direction = "up" if t_last > t_open * 1.001 else ("down" if t_last < t_open * 0.999 else "flat")
        # Classification logic
        if efficiency >= 0.55 and range_ratio >= 1.15:
            day_type = f"trend_{net_direction}" if net_direction != "flat" else "trend_up"
            day_score = 2
            strategy_hint = "breakout"
        elif efficiency >= 0.45 and range_ratio >= 1.05:
            day_type = f"trend_{net_direction}" if net_direction != "flat" else "trend_up"
            day_score = 1
            strategy_hint = "breakout"
        elif efficiency <= 0.25 or range_ratio <= 0.75:
            day_type = "choppy"
            day_score = -1
            strategy_hint = "neutral"   # avoid trading choppy days
        elif efficiency <= 0.38 and range_ratio <= 1.1:
            day_type = "range"
            day_score = 0
            strategy_hint = "mean_reversion"
        else:
            day_type = "neutral"
            day_score = 0
            strategy_hint = "neutral"
        result = {
            "day_type":      day_type,
            "efficiency":    efficiency,
            "range_ratio":   range_ratio,
            "opening_bias":  opening_bias,
            "day_score":     day_score,
            "strategy_hint": strategy_hint,
            "net_move_pct":  round(net_move / max(t_open, 1) * 100, 2),
        }
        _DAY_TYPE_CACHE.clear()
        _DAY_TYPE_CACHE.extend([result, now_ts])
        logger.info(f"Day type: {day_type} | eff={efficiency:.2f} | range×{range_ratio:.2f} | "
                    f"open-bias={opening_bias} | strategy={strategy_hint}")
        return result
    except Exception as e:
        logger.debug(f"Day type detection failed: {e}")
        return _default


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
    Rank sectors by 1-day, 5-day, 20-day, and 63-day ETF performance.
    Returns {sector: adj_score} where adj_score is -16 to +16.
    4-timeframe momentum: recent (1d) + short-term (5d) + medium-term (20d) + quarterly (63d).
    Hot sectors get a bonus; cold sectors get a penalty.
    """
    etfs = list(SECTOR_ETFS.values())
    try:
        kw  = dict(group_by="ticker", auto_adjust=True, progress=False)
        raw = yf.download(" ".join(etfs), period="90d", interval="1d", **kw)
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
                chg63d = (closes[-1] - closes[-63]) / closes[-63] * 100 if len(closes) >= 63 else 0
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
                # 63-day quarterly trend (weight: ±4) — persistent institutional money flows
                if   chg63d > 15.0: sc += 4   # top-quartile sector over 3 months
                elif chg63d >  8.0: sc += 2
                elif chg63d >  3.0: sc += 1
                elif chg63d < -15.0: sc -= 4  # persistent underperformance
                elif chg63d <  -8.0: sc -= 2
                elif chg63d <  -3.0: sc -= 1
                adj[sec]    = max(-16, min(16, sc))
                detail[sec] = {"1d": round(chg1d,2), "5d": round(chg5d,2),
                               "20d": round(chg20d,2), "63d": round(chg63d,2)}
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
                adj[sec] = min(16, adj.get(sec, 0) + 2)  # extra boost for accelerating sectors
        if accel_sectors:
            logger.info(f"Sector acceleration (money flowing in NOW): {', '.join(accel_sectors)}")

        # Store full detail for dashboard sector heatmap
        try:
            import builtins
            builtins._SECTOR_ROTATION_DETAIL = detail
        except Exception:
            pass

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

def get_macro_context() -> dict:
    """Returns macro event context for the current moment.
    Used to tag trade entries so we can learn macro event performance."""
    today = datetime.now(timezone.utc).date()
    for d in range(3):  # check today + next 2 days
        check = (today + timedelta(days=d)).isoformat()
        if check in _MACRO_EVENTS:
            event_type = "FOMC" if check in _FOMC_DATES else "CPI" if check in _CPI_DATES else "NFP"
            return {"event": event_type, "days_away": d,
                    "label": "event_day" if d == 0 else ("day_before" if d == 1 else "2d_before")}
    return {"event": "none", "days_away": 99, "label": "normal"}


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


_CATALYST_TYPES = {
    # High-urgency hard catalysts (binary, move stock 10-30%+)
    "earnings":   ["earnings beat", "beats estimates", "record revenue", "raised guidance",
                   "blowout quarter", "record profit", "record earnings", "misses estimates",
                   "earnings miss", "revenue miss", "guidance cut", "lowers guidance"],
    "fda":        ["fda approval", "fda approved", "fda clears", "breakthrough therapy",
                   "positive phase", "phase 3 success", "regulatory approval",
                   "fda rejects", "clinical failure", "complete response letter", "trial failure"],
    "ma":         ["merger", "acquisition", "buyout", "takeover", "going private",
                   "strategic acquisition", "deal agreed", "tender offer"],
    "insider":    ["insider buying", "insider buy", "executive purchase", "ceo buys",
                   "institutional buying", "13f", "direct purchase"],
    "analyst":    ["upgrade", "price target raised", "strong buy", "outperform", "overweight",
                   "initiates coverage", "downgrade", "price target cut", "sell rating"],
    "contract":   ["contract win", "awarded contract", "major contract", "government contract",
                   "landmark deal", "multi-year deal", "partnership"],
    "buyback":    ["share buyback", "repurchase", "dividend increase", "special dividend", "stock split"],
    "legal":      ["lawsuit", "sec investigation", "fraud", "class action", "subpoena", "antitrust", "bankruptcy"],
}
# Urgency tier: higher = more binary/immediate price impact
_CATALYST_URGENCY = {"earnings": 5, "fda": 5, "ma": 5, "insider": 4, "legal": 4,
                     "analyst": 3, "contract": 3, "buyback": 2}

def detect_catalyst(headlines: list) -> tuple[float, str]:
    """
    Fast keyword scan of headlines. Returns (boost, catalyst_label).
    boost is -15 to +15 additive score points.
    Also stores catalyst_type and urgency in thread-local for callers who need richer info.
    """
    text = " ".join(headlines).lower()
    bull_hits = [c for c in _BULL_CATALYSTS if c in text]
    bear_hits = [c for c in _BEAR_CATALYSTS if c in text]
    boost = min(15, len(bull_hits) * 6) - min(15, len(bear_hits) * 6)
    label = (bull_hits[0] if bull_hits else (bear_hits[0] if bear_hits else ""))
    return float(boost), label


def classify_catalyst(headlines: list) -> dict:
    """
    Classify the dominant catalyst type, urgency, and direction from headlines.
    Returns: {type, urgency (1-5), direction ('bull'/'bear'/'mixed'), label}
    """
    if not headlines:
        return {"type": "none", "urgency": 0, "direction": "none", "label": ""}
    text = " ".join(headlines).lower()
    best_type, best_urg = "news", 1
    for cat_type, keywords in _CATALYST_TYPES.items():
        if any(kw in text for kw in keywords):
            urg = _CATALYST_URGENCY.get(cat_type, 1)
            if urg > best_urg:
                best_type, best_urg = cat_type, urg
    bull_hits = [c for c in _BULL_CATALYSTS if c in text]
    bear_hits = [c for c in _BEAR_CATALYSTS if c in text]
    direction = "bull" if len(bull_hits) > len(bear_hits) else ("bear" if len(bear_hits) > len(bull_hits) else "mixed")
    label = bull_hits[0] if direction == "bull" and bull_hits else (bear_hits[0] if direction == "bear" and bear_hits else "")
    return {"type": best_type, "urgency": best_urg, "direction": direction, "label": label}


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
            if signals.get("mtf_triple"):
                extras.append("TRIPLE timeframe aligned (weekly+daily+hourly all bullish — rare, highest conviction)")
            elif signals.get("mtf_aligned"):
                extras.append("multi-timeframe confirmed (daily + hourly aligned)")
            # New signals
            if signals.get("three_white_soldiers"):
                extras.append("Three White Soldiers candlestick pattern — 3 consecutive strong bull bars (institutional continuation)")
            elif signals.get("morning_star"):
                extras.append("Morning Star reversal — dark candle → doji → bull candle (institutional capitulation reversal)")
            elif signals.get("bullish_engulfing"):
                extras.append("Bullish Engulfing — larger green bar fully engulfs prior red bar (buyer takeover)")
            elif signals.get("hammer"):
                extras.append("Hammer reversal — long lower wick showing buyers absorbed all selling pressure")
            if signals.get("psar_bull"):
                psar_lvl = signals.get("psar", 0)
                extras.append(f"Parabolic SAR bullish (trailing stop ${psar_lvl:.2f}) — accelerating trend confirmation")
            if signals.get("price_accel_pos"):
                pa = signals.get("price_accel", 0)
                extras.append(f"Price acceleration +{pa:.1f}% — ROC speeding up; institutional accumulation before RSI shows it")
            if signals.get("unusual_calls"):
                extras.append(f"Unusual call volume (PCR={signals.get('options_pcr',1):.2f}) — big money buying calls; institutional directional bet")
            elif signals.get("options_bull"):
                extras.append(f"Bullish options flow (PCR={signals.get('options_pcr',1):.2f}) — more calls than puts; market makers hedging bullish")
            piv_s1 = signals.get("pivot_s1", 0)
            piv_s2 = signals.get("pivot_s2", 0)
            price_v = signals.get("price", 0)
            if piv_s1 > 0 and price_v > 0 and abs(price_v - piv_s1) / price_v < 0.015:
                extras.append(f"At Pivot S1 support ${piv_s1:.2f} — institutional intraday buy zone")
            elif piv_s2 > 0 and price_v > 0 and abs(price_v - piv_s2) / price_v < 0.015:
                extras.append(f"At Pivot S2 support ${piv_s2:.2f} — deep institutional support level")
            if signals.get("news_accelerating"):
                n24 = signals.get("news_count_24h", 0)
                extras.append(f"News velocity accelerating ({n24} articles in 24h) — catalyst building, institutional awareness rising")
            if signals.get("pm_big_gap_up"):
                pm_pct = signals.get("pm_gap_pct", 0)
                extras.append(f"Pre-market gap up {pm_pct:+.1f}% — institutional positioning before open")
            if signals.get("squeeze_potential"):
                extras.append("Gamma squeeze potential — large put wall below + negative dealer GEX; explosive upside fuel")
            if signals.get("accum_score", 0) >= 8:
                acc = signals.get("accum_score", 0)
                extras.append(f"Smart accumulation score {acc}/10 — OBV+Force Index+MFI all confirm institutional buying")
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
    Enhanced market breadth using a wide ETF basket (sector + factor + size).
    Returns adv_pct, 5-day breadth trend, breadth thrust signal, and McClellan proxy.
    """
    global _BREADTH_CACHE
    if _BREADTH_CACHE:
        return _BREADTH_CACHE
    try:
        # Wide basket: 11 SPDR sectors + size/factor ETFs = 22 proxy instruments
        probe_syms = [
            "XLK","XLF","XLV","XLE","XLY","XLI","XLP","XLC","XLU","XLRE","XLB",
            "IWM","MDY","IJR","IVV","QQQ","DIA","XME","IBB","XBI","ARKK","SOXX"
        ]
        raw = yf.download(" ".join(probe_syms), period="10d", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)
        adv_series = []  # daily adv_pct over last 5 days
        for day_offset in range(-5, 0):
            day_adv = 0; day_total = 0
            for sym in probe_syms:
                try:
                    closes = list(raw["Close"][sym].dropna())
                    if len(closes) >= abs(day_offset) + 1:
                        if closes[day_offset] > closes[day_offset - 1]:
                            day_adv += 1
                        day_total += 1
                except Exception:
                    pass
            if day_total > 0:
                adv_series.append(round(day_adv / day_total * 100, 1))
        adv_pct = adv_series[-1] if adv_series else 50.0
        adv_5d_avg = round(sum(adv_series) / len(adv_series), 1) if adv_series else 50.0
        # Breadth trend: is breadth improving or deteriorating?
        breadth_trend = "neutral"
        if len(adv_series) >= 3:
            if adv_series[-1] > adv_series[-3] + 10:
                breadth_trend = "improving"
            elif adv_series[-1] < adv_series[-3] - 10:
                breadth_trend = "deteriorating"
        # Breadth thrust: rare signal — adv_pct > 70 following a period < 40 within 10 days
        breadth_thrust = adv_pct >= 68 and adv_5d_avg >= 60 and min(adv_series[:3]) < 45 if len(adv_series) >= 4 else False
        # McClellan proxy: difference between fast (3d) and slow (5d) breadth EMA
        mcl_fast = sum(adv_series[-3:]) / 3 if len(adv_series) >= 3 else adv_pct
        mcl_slow = adv_5d_avg
        mcl_osc  = round(mcl_fast - mcl_slow, 1)
        note = "broad advance" if adv_pct > 70 else "broad decline" if adv_pct < 30 else "mixed"
        total_counted = sum(1 for s in probe_syms)

        # TRIN (Arms Index): (advancing issues / declining issues) / (advancing vol / declining vol)
        # < 0.8 = strong buying, > 1.2 = selling pressure, > 2.0 = panic selling
        # Proxy using our ETF basket: avg intraday vol ratio for advancing vs declining ETFs
        trin_proxy = 1.0
        trin_signal = "neutral"
        try:
            raw_1d = yf.download(" ".join(probe_syms), period="2d", interval="1d",
                                  group_by="ticker", auto_adjust=True, progress=False)
            _adv_vols = []; _dec_vols = []
            for sym in probe_syms:
                try:
                    cl2 = list(raw_1d["Close"][sym].dropna())
                    vl2 = list(raw_1d["Volume"][sym].dropna())
                    if len(cl2) >= 2 and len(vl2) >= 2:
                        if cl2[-1] > cl2[-2]:
                            _adv_vols.append(vl2[-1])
                        else:
                            _dec_vols.append(vl2[-1])
                except Exception:
                    pass
            if _adv_vols and _dec_vols:
                _avg_adv_v = sum(_adv_vols) / len(_adv_vols)
                _avg_dec_v = sum(_dec_vols) / len(_dec_vols)
                _adv_ratio = len(_adv_vols) / max(len(_dec_vols), 1)
                _vol_ratio = _avg_adv_v / max(_avg_dec_v, 1)
                trin_proxy  = round(_adv_ratio / max(_vol_ratio, 0.01), 2)
                if   trin_proxy < 0.75: trin_signal = "strong_buy"
                elif trin_proxy < 0.90: trin_signal = "buy"
                elif trin_proxy < 1.10: trin_signal = "neutral"
                elif trin_proxy < 1.40: trin_signal = "sell"
                else:                   trin_signal = "strong_sell"
        except Exception:
            pass

        # Sector ETF returns for heatmap: 1d and 5d performance
        _sector_syms = {
            "XLK":"Tech","XLF":"Fins","XLV":"Health","XLE":"Energy","XLY":"Cons.Disc",
            "XLI":"Indust","XLP":"Staples","XLC":"Comms","XLU":"Util","XLRE":"RE","XLB":"Matrl",
        }
        _sector_perf = {}
        for _ss, _sl in _sector_syms.items():
            try:
                _sc = list(raw["Close"][_ss].dropna())
                if len(_sc) >= 2:
                    _ret1d = round((_sc[-1] - _sc[-2]) / _sc[-2] * 100, 2)
                    _ret5d = round((_sc[-1] - _sc[max(0, -6)]) / _sc[max(0, -6)] * 100, 2) if len(_sc) >= 6 else _ret1d
                    _sector_perf[_ss] = {"name": _sl, "ret1d": _ret1d, "ret5d": _ret5d}
            except Exception:
                pass

        _BREADTH_CACHE = {
            "adv_pct":       adv_pct,
            "adv_5d_avg":    adv_5d_avg,
            "adv_series":    adv_series,
            "breadth_trend": breadth_trend,
            "breadth_thrust": breadth_thrust,
            "mcl_osc":       mcl_osc,
            "note":          note,
            "total":         total_counted,
            "trin_proxy":    trin_proxy,
            "trin_signal":   trin_signal,
            "sector_perf":   _sector_perf,
        }
        logger.info(f"Market breadth: today={adv_pct}% | 5d avg={adv_5d_avg}% | trend={breadth_trend} | MCL={mcl_osc:+.1f} | TRIN~{trin_proxy:.2f}({trin_signal})")
        return _BREADTH_CACHE
    except Exception:
        return {"adv_pct": 50.0, "adv_5d_avg": 50.0, "adv_series": [], "breadth_trend": "neutral",
                "breadth_thrust": False, "mcl_osc": 0.0, "note": "unknown", "total": 0}


def ai_trade_thesis(ticker: str, score: int, signals: dict, catalyst: str = "", sentiment: int = 0) -> str:
    """
    Generate a 1-sentence trade entry thesis using Claude Haiku.
    Stored on the position card so the user understands the bot's reasoning.
    Returns a concise string like: "Entering NVDA: TT8/8 + pocket pivot + 45% EPS growth on E21 pullback"
    """
    if not ANTHROPIC_KEY:
        return ""
    try:
        # Build compact signal summary for the prompt
        sig_parts = []
        tt = signals.get("trend_template", 0) or 0
        if tt >= 6: sig_parts.append(f"TT{tt}/8 SEPA")
        if signals.get("htf"): sig_parts.append(f"HTF{signals.get('htf_consec',0)}d")
        if signals.get("pocket_pivot"): sig_parts.append("pocket pivot")
        if signals.get("ema21_pullback"): sig_parts.append("EMA21 pullback")
        if signals.get("cup_handle"): sig_parts.append("cup&handle breakout")
        if signals.get("vcp"): sig_parts.append("VCP")
        if signals.get("mtf_triple"): sig_parts.append("3TF aligned")
        if signals.get("rs_rating", 0) >= 80: sig_parts.append(f"RS{signals.get('rs_rating')}")
        if (signals.get("earnings_growth") or 0) >= 0.20:
            sig_parts.append(f"EPS+{round((signals['earnings_growth'])*100)}%")
        if signals.get("rvol_surge"): sig_parts.append(f"RVOL{signals.get('rvol',1):.1f}x")
        if signals.get("donchian_up"): sig_parts.append("20D breakout")
        if signals.get("above_avwap_52wl"): sig_parts.append("above AVWAP")
        sig_str = " + ".join(sig_parts[:6])
        cat_str = f" | catalyst: {catalyst}" if catalyst else ""
        prompt = (
            f"Write a 1-sentence (<20 words) trade thesis for entering {ticker}.\n"
            f"Score: {score}/100 | Signals: {sig_str}{cat_str}\n"
            f"Format: 'Entering {ticker}: [key reason why]'\n"
            f"Be specific, factual, no fluff. Return ONLY the sentence."
        )
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 60,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=8,
        )
        thesis = r.json()["content"][0]["text"].strip().strip('"\'')
        return thesis[:150]  # cap length
    except Exception:
        return ""


def ai_market_context(regime, top_movers, sector_adjs: dict = None, extra_ctx: dict = None):
    """
    Ask Claude for a macro market read that adjusts our overall confidence.
    Returns an adjustment score -5 to +5.
    extra_ctx: optional dict with rs_leaders, ema21_setups, scan_adv_pct, high_corr_pairs.
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
        # Extra context: RS Rating leaders, EMA21 setups, breadth
        extra_str = ""
        if extra_ctx:
            if extra_ctx.get("rs_leaders"):
                extra_str += f"\n- RS Rating leaders (≥80): {', '.join(extra_ctx['rs_leaders'][:6])}"
            if extra_ctx.get("ema21_setups"):
                extra_str += f"\n- EMA21 pullback setups: {', '.join(extra_ctx['ema21_setups'][:5])}"
            if extra_ctx.get("pocket_pivots"):
                extra_str += f"\n- Pocket pivot signals: {', '.join(extra_ctx['pocket_pivots'][:4])}"
            if extra_ctx.get("htf_stocks"):
                extra_str += f"\n- High-Tight Flag stocks: {', '.join(extra_ctx['htf_stocks'][:4])}"
            if extra_ctx.get("tt8_stocks"):
                extra_str += f"\n- Trend Template 8/8 (SEPA elite): {', '.join(extra_ctx['tt8_stocks'][:4])}"
            if extra_ctx.get("scan_adv_pct") is not None:
                extra_str += f"\n- Internal breadth: {extra_ctx['scan_adv_pct']}% of scanned stocks advancing"
            if extra_ctx.get("breadth_trend") and extra_ctx["breadth_trend"] != "neutral":
                extra_str += f"\n- Breadth trend: {extra_ctx['breadth_trend']}"
            if extra_ctx.get("breadth_thrust"):
                extra_str += "\n- BREADTH THRUST detected — rare, very bullish market condition"
            if extra_ctx.get("high_corr_pairs"):
                extra_str += f"\n- Correlated position pairs: {', '.join(extra_ctx['high_corr_pairs'][:3])}"
        prompt = (
            f"Automated US equity trader decision for today:\n"
            f"- Regime: {regime['regime']} (VIX={regime['vix']:.0f}, SPY 5d={regime['spy_trend']:+.1f}%)\n"
            f"- Market breadth: {breadth['adv_pct']}% sectors advancing ({breadth['note']})\n"
            f"- FOMC/macro event {'TODAY — be defensive' if on_macro else 'not imminent'}\n"
            f"- Top movers: {movers_str}{sec_str}{extra_str}\n\n"
            f"Should the bot be aggressive (+3 to +5 = higher scores unlock more buys) "
            f"or cautious (-3 to -5 = raise the bar, fewer buys) today?\n"
            f"Consider: VIX level, breadth, macro risk, sector leadership, RS leaders.\n"
            f"Return ONLY JSON: {{\"adj\":<-5 to 5>, \"note\":\"<10 words max>\"}}"
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
def _fetch_atm_iv(sym: str) -> float:
    """ATM implied volatility (%) from nearest 14-45 day expiry options. Cached 30 min."""
    global _ATM_IV_CACHE, _ATM_IV_TS
    import time as _time
    _now = _time.monotonic()
    if sym in _ATM_IV_CACHE and _now - _ATM_IV_TS.get(sym, 0) < 1800:
        return _ATM_IV_CACHE[sym]
    try:
        from datetime import datetime, date as _date
        tk = yf.Ticker(sym)
        exps = tk.options
        if not exps:
            _ATM_IV_CACHE[sym] = 0.0
            _ATM_IV_TS[sym] = _now
            return 0.0
        today = _date.today()
        best_exp, best_diff = None, 999
        for exp_str in exps:
            try:
                exp = datetime.strptime(exp_str, '%Y-%m-%d').date()
                days = (exp - today).days
                if 7 <= days <= 60:
                    diff = abs(days - 21)
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = exp_str
            except Exception:
                continue
        if not best_exp:
            best_exp = exps[0]
        chain = tk.option_chain(best_exp)
        calls = chain.calls
        if calls is None or calls.empty or 'impliedVolatility' not in calls.columns:
            _ATM_IV_CACHE[sym] = 0.0
            _ATM_IV_TS[sym] = _now
            return 0.0
        cur = getattr(tk.fast_info, 'last_price', 0) or 0
        calls = calls[calls['impliedVolatility'] > 0].copy()
        if calls.empty:
            _ATM_IV_CACHE[sym] = 0.0
            _ATM_IV_TS[sym] = _now
            return 0.0
        if cur > 0:
            idx = (calls['strike'] - cur).abs().idxmin()
        else:
            idx = calls.index[len(calls) // 2]
        atm_iv = round(float(calls.loc[idx, 'impliedVolatility']) * 100, 1)
        _ATM_IV_CACHE[sym] = atm_iv
        _ATM_IV_TS[sym] = _now
        return atm_iv
    except Exception:
        _ATM_IV_CACHE[sym] = 0.0
        _ATM_IV_TS[sym] = _time.monotonic()
        return 0.0


def _fetch_spy_perf() -> dict:
    """
    Fetch SPY performance over multiple timeframes once per run.
    Stored in _SPY_PERF_CACHE so individual stocks can compute relative strength, beta, and RS Rating.
    """
    global _SPY_PERF_CACHE
    if _SPY_PERF_CACHE:
        return _SPY_PERF_CACHE
    try:
        spy = yf.download("SPY", period="2y", interval="1d",
                          auto_adjust=True, progress=False)
        _spy_valid = spy["Close"].dropna()
        closes = list(_spy_valid)
        if len(closes) >= 2:
            _SPY_PERF_CACHE["d1"]     = (closes[-1] - closes[-2]) / closes[-2] * 100
            _SPY_PERF_CACHE["d5"]     = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
            _SPY_PERF_CACHE["d10"]    = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
            _SPY_PERF_CACHE["d63"]    = (closes[-1] - closes[-63]) / closes[-63] * 100 if len(closes) >= 63 else 0
            _SPY_PERF_CACHE["d126"]   = (closes[-1] - closes[-126]) / closes[-126] * 100 if len(closes) >= 126 else 0
            _SPY_PERF_CACHE["d189"]   = (closes[-1] - closes[-189]) / closes[-189] * 100 if len(closes) >= 189 else 0
            _SPY_PERF_CACHE["d252"]   = (closes[-1] - closes[-252]) / closes[-252] * 100 if len(closes) >= 252 else 0
            _SPY_PERF_CACHE["closes"] = closes  # full close history for beta regression
            # Sorted date list for RS-vs-SPY since-entry lookup
            _SPY_PERF_CACHE["date_list"] = [str(d.date()) for d in _spy_valid.index]
    except Exception:
        _SPY_PERF_CACHE = {"d1": 0.0, "d5": 0.0, "d10": 0.0, "d63": 0.0,
                           "d126": 0.0, "d189": 0.0, "d252": 0.0, "closes": [], "date_list": []}
    return _SPY_PERF_CACHE


def _pocket_pivot(closes: list, volumes: list, lookback: int = 10) -> bool:
    """O'Neil/Kacher-Morales Pocket Pivot: up day whose volume exceeds every down-day volume in prior 10 sessions."""
    if len(closes) < lookback + 2 or len(volumes) < lookback + 2:
        return False
    if closes[-1] <= closes[-2]:
        return False
    prior_down_vols = [volumes[i] for i in range(-lookback - 1, -1) if closes[i] < closes[i - 1]]
    if not prior_down_vols:
        return True  # no down days in lookback = trend dominance
    return volumes[-1] > max(prior_down_vols)


def _high_tight_flag(highs: list, closes: list, volumes: list) -> dict:
    """
    Minervini High-Tight Flag: stock rose ≥100% in ≤8 weeks, then consolidated ≤25%.
    Simplified live signal: 3+ consecutive closes within 3% of the 52-week high on rising avg volume.
    """
    if len(highs) < 20 or len(closes) < 20:
        return {"htf": False, "htf_consec": 0}
    max_high = max(highs[-min(len(highs), 252):])
    consec = 0
    for i in range(-1, -min(8, len(highs)) - 1, -1):
        if highs[i] >= max_high * 0.97:
            consec += 1
        else:
            break
    # Require volume confirmation on at least the most recent day
    avg_v = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
    vol_ok = volumes[-1] >= avg_v * 1.1 if avg_v > 0 else True
    return {"htf": consec >= 3 and vol_ok, "htf_consec": consec}


def _trend_template_score(closes: list, highs: list, lows: list, rs_rating: int = 50) -> dict:
    """
    Minervini SEPA Trend Template: 8 criteria, each worth 1 point.
    Score ≥7 = institutional-grade uptrend.
    """
    score = 0
    criteria = {}
    if len(closes) < 200:
        return {"trend_template": score, "tt_criteria": criteria, "tt_full": False}
    price = closes[-1]
    ema50  = _ema(closes, 50)[-1]
    ema150 = _ema(closes, 150)[-1]
    ema200 = _ema(closes, 200)[-1]
    # 200d slope: compare to 20 bars ago
    ema200_prev = _ema(closes[:-20], 200)[-1] if len(closes) >= 220 else ema200
    # 52W high/low from all available data (up to 252 bars)
    look = closes[-252:]
    hl252 = highs[-252:] if len(highs) >= 252 else highs
    ll252 = lows[-252:] if len(lows) >= 252 else lows
    high_52w = max(hl252)
    low_52w  = min(ll252)

    c1 = price > ema200;           score += c1; criteria["above_ema200"]   = c1
    c2 = ema200 > ema200_prev;     score += c2; criteria["ema200_trending"] = c2
    c3 = ema150 > ema200;          score += c3; criteria["ema150_gt_200"]   = c3
    c4 = ema50  > ema150;          score += c4; criteria["ema50_gt_150"]    = c4
    c5 = price  > ema50;           score += c5; criteria["above_ema50"]     = c5
    c6 = price  >= low_52w * 1.25; score += c6; criteria["25pct_off_low"]   = c6
    c7 = price  >= high_52w * 0.75;score += c7; criteria["within_25_high"]  = c7
    c8 = rs_rating >= 70;          score += c8; criteria["rs_gte70"]        = c8
    return {"trend_template": score, "tt_criteria": criteria, "tt_full": score == 8}


def _anchored_vwap(daily) -> dict:
    """VWAP anchored from the 52-week low date — acts as dynamic institutional support."""
    try:
        df = daily.tail(252).copy()
        if len(df) < 5 or "Volume" not in df.columns:
            return {"avwap_52wl": 0.0, "above_avwap_52wl": False, "avwap_dist_pct": 0.0}
        low_idx = df["Low"].idxmin()
        anchor = df[df.index >= low_idx]
        if len(anchor) < 2:
            return {"avwap_52wl": 0.0, "above_avwap_52wl": False, "avwap_dist_pct": 0.0}
        tp  = (anchor["High"] + anchor["Low"] + anchor["Close"]) / 3
        vol = anchor["Volume"].fillna(0)
        avwap = float((tp * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1]) if vol.sum() > 0 else 0.0
        cur   = float(df["Close"].iloc[-1])
        dist  = round((cur - avwap) / avwap * 100, 2) if avwap > 0 else 0.0
        return {"avwap_52wl": round(avwap, 2), "above_avwap_52wl": cur > avwap, "avwap_dist_pct": dist}
    except Exception:
        return {"avwap_52wl": 0.0, "above_avwap_52wl": False, "avwap_dist_pct": 0.0}


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

    # 52-week range position: 0% = at 52w low, 100% = at 52w high
    w52_range_pos = 0.0
    try:
        if high_52w > low_52w > 0:
            w52_range_pos = round((price - low_52w) / (high_52w - low_52w) * 100, 1)
            w52_range_pos = max(0.0, min(100.0, w52_range_pos))
    except Exception:
        pass

    # Fibonacci retracement level detection — institutional support zones
    fib_support    = False   # price near 38.2% / 50% / 61.8% retracement and holding
    fib_resistance = False   # price near 61.8% / 78.6% when in downtrend
    fib_level_382  = 0.0
    fib_level_500  = 0.0
    fib_level_618  = 0.0
    fib_level_786  = 0.0
    fib_high_ref   = 0.0
    fib_low_ref    = 0.0
    try:
        if len(daily) >= 20 and "High" in daily.columns and "Low" in daily.columns:
            _fib_high = float(daily["High"].iloc[-20:].max())
            _fib_low  = float(daily["Low"].iloc[-20:].min())
            _range    = _fib_high - _fib_low
            if _range > 0:
                fib_level_382 = round(_fib_high - 0.382 * _range, 2)
                fib_level_500 = round(_fib_high - 0.500 * _range, 2)
                fib_level_618 = round(_fib_high - 0.618 * _range, 2)
                fib_level_786 = round(_fib_high - 0.786 * _range, 2)
                fib_high_ref  = round(_fib_high, 2)
                fib_low_ref   = round(_fib_low, 2)
                # Within 1% of Fibonacci level = at the zone
                for fib_lvl in [fib_level_382, fib_level_500, fib_level_618]:
                    if abs(price - fib_lvl) / fib_lvl < 0.012:
                        dc_fib = list(daily["Close"].iloc[-5:])
                        if dc_fib and min(dc_fib[:-1]) < fib_lvl and price >= fib_lvl * 0.998:
                            fib_support = True
                        break
                if abs(price - fib_level_786) / fib_level_786 < 0.012:
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
    vwap_price        = 0.0
    vwap_z            = 0.0
    vwap_reclaim      = False
    vwap_b1u = vwap_b2u = vwap_b1d = vwap_b2d = 0.0
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

        vwap_price, vwap_pos, vwap_z, vwap_reclaim, vwap_b1u, vwap_b2u, vwap_b1d, vwap_b2d = _vwap(h)

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

    # EMA21 pullback signal: first pullback to 21-day EMA in uptrend = Minervini's highest-prob entry
    # Criteria: above 200 EMA (uptrend) + above 50 EMA + price within -3% to +0.5% of EMA21
    # + today is green (reversal) = institutional support confirmed at key moving average
    ema21_pullback = False
    ema21_touch    = False   # touching EMA21 even if no confirmed reversal yet
    try:
        if len(dc) >= 21:
            e21 = _ema(dc, 21)
            if e21 and e21 > 0 and price_vs_ema200 > 0 and price_vs_ema50 > 0:
                _dist_e21 = (dc[-1] - e21) / e21 * 100   # % above/below EMA21
                ema21_touch = -3.0 <= _dist_e21 <= 0.8    # within the pullback zone
                # Confirmed pullback: in zone AND today closed up (reversal bar)
                if ema21_touch and chg_pct > 0:
                    ema21_pullback = True
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

    # Volume Divergence: detect distribution/accumulation via price-volume relationship
    # Bearish divergence: new 10-day high but falling volume → institutions distributing
    # Bullish divergence: new 10-day low but rising volume on down days → quiet accumulation
    vol_bearish_div = False   # new high + shrinking volume = distribution
    vol_bullish_div = False   # new low + shrinking volume on sell days = quiet accumulation
    try:
        if "Close" in daily.columns and "Volume" in daily.columns and len(daily) >= 12:
            _closes_vd = list(daily["Close"])
            _vols_vd   = list(daily["Volume"])
            _n = min(10, len(_closes_vd) - 2)
            _recent_c  = _closes_vd[-_n:]
            _prior_c   = _closes_vd[-_n*2:-_n] if len(_closes_vd) >= _n*2 else _closes_vd[:_n]
            _recent_v  = _vols_vd[-_n:]
            _prior_v   = _vols_vd[-_n*2:-_n] if len(_vols_vd) >= _n*2 else _vols_vd[:_n]
            _avg_prior_v = sum(_prior_v) / len(_prior_v) if _prior_v else 1
            _avg_recent_v = sum(_recent_v) / len(_recent_v) if _recent_v else 1
            _new_high = max(_recent_c) > max(_prior_c)
            _new_low  = min(_recent_c) < min(_prior_c)
            _vol_shrinking = _avg_recent_v < _avg_prior_v * 0.75  # volume down 25%+
            if _new_high and _vol_shrinking:
                vol_bearish_div = True   # classic distribution signal
            if _new_low and not _vol_shrinking:
                # Bullish: price making new low but volume not picking up = no panic selling
                _down_vols_r = [_vols_vd[i] for i in range(-_n, 0) if _closes_vd[i] < _closes_vd[i-1]]
                _down_vols_p = [_vols_vd[i] for i in range(-_n*2, -_n) if _closes_vd[i] < _closes_vd[i-1]]
                if _down_vols_r and _down_vols_p:
                    _avg_down_r = sum(_down_vols_r) / len(_down_vols_r)
                    _avg_down_p = sum(_down_vols_p) / len(_down_vols_p)
                    if _avg_down_r < _avg_down_p * 0.80:  # sell-side volume shrinking = buyers absorbing
                        vol_bullish_div = True
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

    # Smart Accumulation Score (0-10): multi-factor composite of smart-money signals
    # Combines OBV trend, Force Index, and MFI — when all three agree = strong conviction
    accum_score = 0
    try:
        if obv_rising:      accum_score += 2
        if fi_rising:       accum_score += 2
        if fi_bull_div:     accum_score += 2
        if mfi_oversold:    accum_score += 2
        if mfi_bull_div:    accum_score += 2
        # Double-confirm: OBV + MFI together = highest conviction (institutional fingerprint)
        if obv_rising and mfi_val > 50:    accum_score += 1
        if obv_rising and mfi_bull_div:    accum_score += 1
        accum_score = min(10, accum_score)
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

    # Swing-based Support/Resistance levels — institutional memory zones
    # Find local extremes in last 60 days; cluster nearby levels for quality S/R
    key_support_1 = 0.0
    key_support_2 = 0.0
    key_resist_1  = 0.0
    key_resist_2  = 0.0
    near_key_support = False
    near_key_resist  = False
    try:
        if len(daily) >= 10 and "High" in daily.columns and "Low" in daily.columns and "Close" in daily.columns:
            _d60 = daily.iloc[-60:]
            _highs = list(_d60["High"])
            _lows  = list(_d60["Low"])
            _closes = list(_d60["Close"])
            _n = len(_highs)
            # Find swing highs (local max of 2 bars either side)
            swing_highs = []
            swing_lows  = []
            for _i in range(2, _n - 2):
                if _highs[_i] >= max(_highs[_i-2], _highs[_i-1], _highs[_i+1], _highs[_i+2]):
                    swing_highs.append(_highs[_i])
                if _lows[_i] <= min(_lows[_i-2], _lows[_i-1], _lows[_i+1], _lows[_i+2]):
                    swing_lows.append(_lows[_i])
            # Sort supports below current price, resistances above
            _sup = sorted([l for l in swing_lows if l < price], reverse=True)[:3]
            _res = sorted([h for h in swing_highs if h > price])[:3]
            if _sup:
                key_support_1 = round(_sup[0], 2)
            if len(_sup) >= 2:
                key_support_2 = round(_sup[1], 2)
            if _res:
                key_resist_1 = round(_res[0], 2)
            if len(_res) >= 2:
                key_resist_2 = round(_res[1], 2)
            # Near key level: within 1.5%
            near_key_support = key_support_1 > 0 and abs(price - key_support_1) / price < 0.015
            near_key_resist  = key_resist_1 > 0 and abs(price - key_resist_1) / price < 0.015
    except Exception:
        pass

    # Parabolic SAR: accelerating trailing stop — tightens as trend matures
    psar_val  = 0.0
    psar_bull = True
    try:
        if all(col in daily.columns for col in ["High", "Low"]) and len(daily) >= 10:
            psar_val, psar_bull = _parabolic_sar(
                list(daily["High"]), list(daily["Low"])
            )
    except Exception:
        pass

    # Price Acceleration: rate of change of ROC (2nd derivative)
    price_accel      = 0.0
    price_accel_pos  = False
    price_accel_neg  = False
    try:
        if "Close" in daily.columns and len(daily) >= 15:
            price_accel, price_accel_pos, price_accel_neg = _price_acceleration(
                list(daily["Close"]), period=10
            )
    except Exception:
        pass

    # Linear Regression Channel: price vs trend line, R-squared linearity score
    lr_slope     = 0.0
    lr_r2        = 0.0
    lr_above_ch  = False
    lr_below_ch  = False
    lr_ch_width  = 0.0
    try:
        if "Close" in daily.columns and len(daily) >= 20:
            lr_slope, lr_r2, lr_above_ch, lr_below_ch, lr_ch_width = _linear_regression_channel(
                list(daily["Close"]), period=20
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

    # Consecutive red candle count — for exit timing
    consec_red = 0
    try:
        closes = list(daily["Close"])
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                consec_red += 1
            else:
                break
    except Exception:
        pass

    # Historical Volatility ratio — expansion vs contraction regime
    hv20 = 0.0
    hv5  = 0.0
    hv_expanding = False
    hv_contracting = False
    try:
        if len(daily) >= 22 and "Close" in daily.columns:
            _c = list(daily["Close"])
            _rets20 = [(_c[i] - _c[i-1]) / _c[i-1] for i in range(len(_c)-20, len(_c))]
            _rets5  = [(_c[i] - _c[i-1]) / _c[i-1] for i in range(len(_c)-5,  len(_c))]
            import statistics
            if len(_rets20) >= 2:
                hv20 = round(statistics.stdev(_rets20) * (252 ** 0.5) * 100, 1)
            if len(_rets5) >= 2:
                hv5  = round(statistics.stdev(_rets5)  * (252 ** 0.5) * 100, 1)
            if hv20 > 0:
                _hv_ratio = hv5 / hv20
                hv_expanding   = _hv_ratio >= 1.4   # volatility surging = momentum move
                hv_contracting = _hv_ratio <= 0.65  # volatility compressing = coil setup
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
    # Time-of-day adjusted: U-shaped intraday volume profile — early RVOL is normalized
    # by the expected fraction of daily volume at the current time of day.
    # ET intraday volume fractions (empirical): open=0.13, 10am=0.25, 11am=0.38, 12pm=0.48,
    #   1pm=0.55, 2pm=0.63, 3pm=0.75, close=1.0
    _TOD_FRACTIONS = {0: 0.13, 30: 0.22, 60: 0.31, 90: 0.41, 120: 0.49,
                      150: 0.56, 180: 0.65, 210: 0.77, 240: 0.88, 270: 1.0}
    rvol = 1.0
    rvol_surge = False   # RVOL > 2.5 with positive price action
    try:
        if "Volume" in daily.columns and len(daily) >= 10:
            avg_vol_20 = float(daily["Volume"].tail(21).iloc[:-1].mean())  # exclude today
            today_vol = float(daily["Volume"].iloc[-1])
            if avg_vol_20 > 0:
                raw_rvol = today_vol / avg_vol_20
                # Time-of-day adjustment using hourly data to estimate minutes since open
                _tod_adj = 1.0
                try:
                    if hourly is not None and not hourly.empty:
                        _last_h_ts = hourly.index[-1]
                        if hasattr(_last_h_ts, "tz_convert"):
                            import pytz as _tz
                            _et = _last_h_ts.tz_convert("America/New_York")
                        elif hasattr(_last_h_ts, "tzinfo") and _last_h_ts.tzinfo:
                            _et = _last_h_ts
                        else:
                            _et = None
                        if _et is not None:
                            _mso = (_et.hour - 9) * 60 + (_et.minute - 30)
                            if 0 < _mso < 390:
                                # Find the closest TOD fraction bucket
                                _bucket = min(_TOD_FRACTIONS.keys(), key=lambda k: abs(k - _mso))
                                _frac = _TOD_FRACTIONS[_bucket]
                                if _frac > 0.05:  # avoid division by tiny fractions
                                    _tod_adj = _frac  # normalize raw RVOL by expected fraction
                except Exception:
                    pass
                # Annualized RVOL: projected full-day volume / 20d avg
                if _tod_adj < 0.99:
                    rvol = round(raw_rvol / _tod_adj, 2)  # time-adjusted RVOL
                else:
                    rvol = round(raw_rvol, 2)
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
    today_open   = 0.0  # today's open price — used for daily P&L attribution
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

    # Multi-timeframe confluence: weekly / daily / hourly all aligned?
    mtf_aligned  = False   # daily + hourly confirm same direction
    mtf_conflict = False   # daily down but hourly up = false signal risk
    mtf_triple   = False   # all three timeframes (weekly+daily+hourly) aligned
    mtf_score    = 0       # 0-3: how many of three timeframes are bullish
    _weekly_bull = False   # initialized before try so always defined
    _daily_up    = False
    _hourly_up   = False
    try:
        _daily_up   = daily_trend > 0.2 and daily_rsi > 45
        _hourly_up  = ema_cross > 0.1 and rsi_val > 45
        _daily_down = daily_trend < -0.2 and daily_rsi < 55
        mtf_aligned  = _daily_up and _hourly_up
        mtf_conflict = _daily_down and _hourly_up

        # Weekly trend: price above 5-week (25-day) EMA and trending up
        _weekly_bull = False
        try:
            dc_w = list(daily["Close"])
            if len(dc_w) >= 25:
                ema25 = _ema(dc_w, 25)
                ema50_w = _ema(dc_w, 50) if len(dc_w) >= 50 else ema25
                if ema25 and ema50_w:
                    _weekly_bull = (dc_w[-1] > ema25
                                    and ema25 > ema50_w * 0.995     # 25 EMA > 50 EMA = uptrend
                                    and dc_w[-1] > dc_w[-5])        # higher than 5 days ago
        except Exception:
            pass

        mtf_score = sum([bool(_weekly_bull), bool(_daily_up), bool(_hourly_up)])
        mtf_triple = mtf_score == 3   # all three timeframes confirmed
        # Override mtf_aligned to require at least weekly+daily OR daily+hourly
        mtf_aligned = (_daily_up and _hourly_up) or (_weekly_bull and _daily_up)
    except Exception:
        pass

    # Relative strength vs SPY (1-day, 5-day, 63-day quarterly, 252-day annual)
    spy  = _fetch_spy_perf()
    rs1  = round(chg_pct - spy.get("d1", 0), 2)   # outperformance vs SPY today
    rs5  = 0.0
    rs63 = 0.0
    rs252 = 0.0
    rs_rating = 50   # IBD-style 1-99 scale (50 = average vs market)
    try:
        dc = list(daily["Close"])
        if len(dc) >= 5:
            ret5 = (dc[-1] - dc[-5]) / dc[-5] * 100
            rs5  = round(ret5 - spy.get("d5", 0), 2)
        if len(dc) >= 63:
            ret63 = (dc[-1] - dc[-63]) / dc[-63] * 100
            rs63  = round(ret63 - spy.get("d63", 0), 2)
        # 12-month relative strength: annual trend vs SPY — IBD RS Rating backbone
        if len(dc) >= 252:
            ret252 = (dc[-1] - dc[-252]) / dc[-252] * 100
            rs252  = round(ret252 - spy.get("d252", 0), 2)
        elif len(dc) >= 126:
            # Proxy: extrapolate from 6-month data
            ret126 = (dc[-1] - dc[-126]) / dc[-126] * 100
            rs252  = round((ret126 - spy.get("d126", 0)) * 0.8, 2)
        elif len(dc) >= 63:
            rs252  = round(rs63 * 0.6, 2)   # rough proxy if only 63d available

        # IBD-style RS Rating (1-99): IBD weights recent quarter 40%, prior quarters 20% each
        # Without a full universe to rank, we map vs a calibrated range: -60 to +60
        _q1 = rs63 * 0.40           # most recent quarter (40% weight)
        _q_rest = rs252 * 0.60      # rest of 12-month (60% weight)
        _rs_composite = _q1 + _q_rest
        # Map composite to 1-99: center=50, typical range ±50
        rs_rating = max(1, min(99, round(50 + _rs_composite)))
    except Exception:
        pass

    # RS Line New High: compares stock/SPY ratio to its own 52-week high
    # IBD's single most reliable buy confirmation — RS line leading price to new highs
    rs_line_new_high = False
    rs_line_trending  = False
    try:
        _spy_closes = spy.get("closes", [])
        dc2 = list(daily["Close"]) if "Close" in daily.columns else []
        _n_rs = min(len(dc2), len(_spy_closes))
        if _n_rs >= 60:
            # align tails: last N closes for both
            _stk = dc2[-_n_rs:]
            _spx = _spy_closes[-_n_rs:]
            # RS line = stock price / SPY price (normalized so it starts at 1.0)
            _base_stk = _stk[0]; _base_spx = _spx[0]
            if _base_stk > 0 and _base_spx > 0:
                _rs_line = [(_stk[i] / _spx[i]) / (_base_stk / _base_spx) for i in range(_n_rs)]
                _rs_now  = _rs_line[-1]
                _rs_52wh = max(_rs_line[-min(252, _n_rs):])
                rs_line_new_high = _rs_now >= _rs_52wh * 0.998
                # RS line trending up: current above its own 10-bar average
                if len(_rs_line) >= 10:
                    _rs_ma10 = sum(_rs_line[-10:]) / 10
                    rs_line_trending = _rs_now > _rs_ma10
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

    # Expected Move: weekly and monthly (using HV20 as IV proxy until ATM IV is fetched)
    # Formula: price × (IV/100) × sqrt(DTE/252)
    expected_move_wk = 0.0   # ±$ expected over 5 trading days
    expected_move_mo = 0.0   # ±$ expected over 21 trading days
    expected_move_pct_wk = 0.0
    try:
        if hv20 > 0 and price > 0:
            import math as _math
            expected_move_wk     = round(price * (hv20 / 100) * _math.sqrt(5 / 252), 2)
            expected_move_mo     = round(price * (hv20 / 100) * _math.sqrt(21 / 252), 2)
            expected_move_pct_wk = round(expected_move_wk / price * 100, 2)
    except Exception:
        pass

    # ── Pocket Pivot (O'Neil / Kacher-Morales) ─────────────────────────────────
    pocket_pivot_signal = False
    try:
        if "Close" in daily.columns and "Volume" in daily.columns and len(daily) >= 12:
            pocket_pivot_signal = _pocket_pivot(list(daily["Close"]), list(daily["Volume"]), lookback=10)
    except Exception:
        pass

    # ── High-Tight Flag (Minervini) ─────────────────────────────────────────────
    htf_result = {"htf": False, "htf_consec": 0}
    try:
        if "High" in daily.columns and "Volume" in daily.columns and len(daily) >= 20:
            htf_result = _high_tight_flag(list(daily["High"]), list(daily["Close"]), list(daily["Volume"]))
    except Exception:
        pass

    # ── Minervini Trend Template Score (0-8) ────────────────────────────────────
    tt_result = {"trend_template": 0, "tt_criteria": {}, "tt_full": False}
    try:
        if "High" in daily.columns and "Low" in daily.columns and len(daily) >= 200:
            tt_result = _trend_template_score(
                list(daily["Close"]), list(daily["High"]), list(daily["Low"]), rs_rating
            )
    except Exception:
        pass

    # ── Anchored VWAP from 52W Low ──────────────────────────────────────────────
    avwap_result = {"avwap_52wl": 0.0, "above_avwap_52wl": False, "avwap_dist_pct": 0.0}
    try:
        avwap_result = _anchored_vwap(daily)
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
        "w52_range_pos":   w52_range_pos,
        "intraday":        round(intraday, 2),
        "rsi":             round(rsi_val, 1),
        "daily_rsi":       round(daily_rsi, 1),
        "daily_trend":     round(daily_trend, 3),
        "ema_cross":       round(ema_cross, 3),
        "macd":            round(macd_val, 3),
        "bb_pos":          round(bb_pos, 1),
        "vwap_pos":        round(vwap_pos, 2),
        "vwap_price":      round(vwap_price, 2),
        "vwap_b1u":        vwap_b1u,   # VWAP + 1σ (resistance)
        "vwap_b2u":        vwap_b2u,   # VWAP + 2σ (strong resistance / overbought)
        "vwap_b1d":        vwap_b1d,   # VWAP - 1σ (support)
        "vwap_b2d":        vwap_b2d,   # VWAP - 2σ (oversold bounce zone)
        "rs1":             rs1,
        "rs5":             rs5,
        "rs63":            rs63,
        "rs252":           rs252,
        "rs_rating":       rs_rating,
        "rs_line_new_high":  rs_line_new_high,
        "rs_line_trending":  rs_line_trending,
        "mtf_aligned":       mtf_aligned,
        "mtf_triple":        mtf_triple,
        "mtf_score":         mtf_score,
        "mtf_conflict":      mtf_conflict,
        "weekly_bull":       bool(_weekly_bull),
        "daily_up":          bool(_daily_up),
        "hourly_up":         bool(_hourly_up),
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
        "ema21_pullback":      ema21_pullback,
        "ema21_touch":         ema21_touch,
        "consec_green":        consec_green,
        "consec_red":          consec_red,
        # Trend Quality composite: 0-10 = how clean + strong the trend is
        # ADX(strength) + LR R²(linearity) + consecutive greens + intraday TQ
        "trend_quality_score": round(min(10, max(0,
            (min(adx_val, 50) / 50 * 3.5)         # ADX: 0-3.5 pts (max at ADX=50)
            + (lr_r2 * 2.5)                         # R²: 0-2.5 pts (linear = 2.5)
            + (min(consec_green, 4) / 4 * 2.0)      # Consecutive greens: 0-2 pts
            + (intraday_trend_quality * 2.0)         # Intraday TQ: -2 to +2
        )), 1),
        "hv20":                hv20,
        "hv5":                 hv5,
        "hv_expanding":        hv_expanding,
        "hv_contracting":      hv_contracting,
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
        "day_open":            round(today_open, 2) if (isinstance(today_open, (int, float)) and today_open > 0) else 0.0,
        "day_high":            round(float(daily["High"].iloc[-1]), 2) if ("High" in daily.columns and len(daily) > 0) else 0.0,
        "day_low":             round(float(daily["Low"].iloc[-1]),  2) if ("Low"  in daily.columns and len(daily) > 0) else 0.0,
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
        "accum_score":         accum_score,
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
        "psar":                psar_val,
        "psar_bull":           psar_bull,
        "price_accel":         price_accel,
        "price_accel_pos":     price_accel_pos,
        "price_accel_neg":     price_accel_neg,
        "lr_slope":            lr_slope,
        "lr_r2":               lr_r2,
        "lr_above_channel":    lr_above_ch,
        "lr_below_channel":    lr_below_ch,
        "lr_channel_width":    lr_ch_width,
        "fib_support":         fib_support,
        "fib_resistance":      fib_resistance,
        "fib_level_382":       fib_level_382,
        "key_support_1":       key_support_1,
        "key_support_2":       key_support_2,
        "key_resist_1":        key_resist_1,
        "key_resist_2":        key_resist_2,
        "near_key_support":    near_key_support,
        "near_key_resist":     near_key_resist,
        "fib_level_500":       fib_level_500,
        "fib_level_618":       fib_level_618,
        "fib_level_786":       fib_level_786,
        "fib_high_ref":        fib_high_ref,
        "fib_low_ref":         fib_low_ref,
        "macd_bull_div":       macd_div.get("bullish_div", False),
        "macd_bear_div":       macd_div.get("bearish_div", False),
        "chandelier_stop":     chandelier_stop,
        "kc_pos":              round(kc_pos, 1),
        "kc_breakout":         kc_breakout,
        "kc_oversold":         kc_oversold,
        "obv_rising":          obv_rising,
        "obv_slope_pct":       round(obv_slope_pct, 1),
        "vol_bearish_div":     vol_bearish_div,
        "vol_bullish_div":     vol_bullish_div,
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
        "expected_move_wk":       expected_move_wk,
        "expected_move_mo":       expected_move_mo,
        "expected_move_pct_wk":   expected_move_pct_wk,
        # ── Advanced pattern signals ────────────────────────────────────────────
        "pocket_pivot":           pocket_pivot_signal,
        "htf":                    htf_result.get("htf", False),
        "htf_consec":             htf_result.get("htf_consec", 0),
        "trend_template":         tt_result.get("trend_template", 0),
        "tt_full":                tt_result.get("tt_full", False),
        "avwap_52wl":             avwap_result.get("avwap_52wl", 0.0),
        "above_avwap_52wl":       avwap_result.get("above_avwap_52wl", False),
        "avwap_dist_pct":         avwap_result.get("avwap_dist_pct", 0.0),
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
                        # Options flow proxy: inject into sig (cached, non-blocking)
                        try:
                            opts = _options_flow(tk)
                            sig["options_pcr"]      = opts.get("pcr", 1.0)
                            sig["options_bull"]     = opts.get("bullish_flow", False)
                            sig["options_bear"]     = opts.get("bearish_flow", False)
                            sig["unusual_calls"]    = opts.get("unusual_calls", False)
                            sig["unusual_puts"]     = opts.get("unusual_puts", False)
                        except Exception:
                            sig.setdefault("options_pcr", 1.0)
                            sig.setdefault("options_bull", False)
                            sig.setdefault("options_bear", False)
                            sig.setdefault("unusual_calls", False)
                            sig.setdefault("unusual_puts", False)
                        # News velocity: cached (30 min TTL) — count of news in 24h vs prior 24h
                        try:
                            nv = _news_velocity(tk)
                            sig["news_count_24h"]   = nv.get("count_24h", 0)
                            sig["news_velocity"]    = nv.get("velocity", 0.0)
                            sig["news_accelerating"] = nv.get("accelerating", False)
                            sig["catalyst_type"]    = nv.get("catalyst_type", "none")
                            sig["catalyst_urg"]     = nv.get("catalyst_urg", 0)
                            sig["catalyst_dir"]     = nv.get("catalyst_dir", "none")
                        except Exception:
                            sig.setdefault("news_count_24h", 0)
                            sig.setdefault("news_velocity", 0.0)
                            sig.setdefault("news_accelerating", False)
                            sig.setdefault("catalyst_type", "none")
                            sig.setdefault("catalyst_urg", 0)
                            sig.setdefault("catalyst_dir", "none")
                        # Pre-market gap: cached (5 min TTL) — pre-market price vs prior close
                        try:
                            pm = _premarket_info(tk)
                            sig["pm_gap_pct"]    = pm.get("gap_pct", 0.0)
                            sig["pm_gap_up"]     = pm.get("gap_up", False)
                            sig["pm_gap_down"]   = pm.get("gap_down", False)
                            sig["pm_big_gap_up"] = pm.get("big_gap_up", False)
                            sig["pm_big_gap_down"] = pm.get("big_gap_down", False)
                            sig["pm_price"]      = pm.get("pre_price", 0.0)
                        except Exception:
                            sig.setdefault("pm_gap_pct", 0.0)
                            sig.setdefault("pm_gap_up", False)
                            sig.setdefault("pm_gap_down", False)
                            sig.setdefault("pm_big_gap_up", False)
                            sig.setdefault("pm_big_gap_down", False)
                            sig.setdefault("pm_price", 0.0)
                        # GEX: gamma exposure proxy (30 min cache) — only for top candidates
                        try:
                            gex = _gamma_exposure(tk)
                            sig["gex_sign"]          = gex.get("gex_sign", 0)
                            sig["gamma_wall_up"]     = gex.get("gamma_wall_up", 0.0)
                            sig["gamma_wall_down"]   = gex.get("gamma_wall_down", 0.0)
                            sig["squeeze_potential"] = gex.get("squeeze_potential", False)
                        except Exception:
                            sig.setdefault("gex_sign", 0)
                            sig.setdefault("gamma_wall_up", 0.0)
                            sig.setdefault("gamma_wall_down", 0.0)
                            sig.setdefault("squeeze_potential", False)
                        # Short interest: only for held positions (avoids slow .info calls for all candidates)
                        if held and tk in held:
                            try:
                                sd_info = _short_data(tk)
                                sig["short_float"] = sd_info.get("short_float", 0.0)
                                sig["short_ratio"] = sd_info.get("short_ratio", 0.0)
                                sig["high_short"]  = sd_info.get("high_short", False)
                            except Exception:
                                sig.setdefault("short_float", 0.0)
                                sig.setdefault("short_ratio", 0.0)
                                sig.setdefault("high_short", False)
                            # ATM Implied Volatility: options chain (30m cache)
                            try:
                                sig["atm_iv"] = _fetch_atm_iv(tk)
                            except Exception:
                                sig.setdefault("atm_iv", 0.0)
                            # Analyst revisions: held positions only (4hr cache, ~15s call)
                            try:
                                ar = _analyst_revisions(tk)
                                sig["analyst_upgrade"]    = ar.get("analyst_upgrade", False)
                                sig["analyst_rev_score"]  = ar.get("analyst_rev_score", 0)
                                sig["analyst_buy_pct"]    = ar.get("buy_pct", 0.5)
                                sig["analyst_net_rev"]    = ar.get("net_revisions", 0)
                                sig["analyst_price_tgt"]  = ar.get("price_target", 0.0)
                                sig["analyst_upside_pct"] = ar.get("upside_pct", 0.0)
                            except Exception:
                                sig.setdefault("analyst_upgrade", False)
                                sig.setdefault("analyst_rev_score", 0)
                                sig.setdefault("analyst_buy_pct", 0.5)
                                sig.setdefault("analyst_net_rev", 0)
                                sig.setdefault("analyst_price_tgt", 0.0)
                                sig.setdefault("analyst_upside_pct", 0.0)
                        else:
                            sig.setdefault("short_float", 0.0)
                            sig.setdefault("short_ratio", 0.0)
                            sig.setdefault("high_short", False)
                            sig.setdefault("analyst_upgrade", False)
                            sig.setdefault("analyst_rev_score", 0)
                            sig.setdefault("analyst_buy_pct", 0.5)
                            sig.setdefault("analyst_net_rev", 0)
                            sig.setdefault("analyst_price_tgt", 0.0)
                            sig.setdefault("analyst_upside_pct", 0.0)
                            sig.setdefault("atm_iv", 0.0)
                        # Fundamental quality: earnings growth, revenue growth, margins (6hr cache)
                        try:
                            fq = _get_fundamentals(tk)
                            sig["earnings_growth"]  = fq.get("earnings_growth")
                            sig["revenue_growth"]   = fq.get("revenue_growth")
                            sig["forward_pe"]       = fq.get("forward_pe")
                            sig["profit_margin"]    = fq.get("profit_margin")
                            sig["roe"]              = fq.get("roe")
                            sig["fund_quality"]     = fq.get("fund_quality", 0)
                        except Exception:
                            sig.setdefault("earnings_growth", None)
                            sig.setdefault("revenue_growth", None)
                            sig.setdefault("forward_pe", None)
                            sig.setdefault("profit_margin", None)
                            sig.setdefault("roe", None)
                            sig.setdefault("fund_quality", 0)
                        # Earnings calendar: pre-earnings drift window (5-20d) is a reliable alpha factor
                        try:
                            sig["earnings_days"] = get_earnings_days(tk)
                        except Exception:
                            sig.setdefault("earnings_days", None)
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
    if d.get("near_key_support", False):       criteria += 1  # Near swing support
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
    if d.get("ema21_pullback", False):        criteria += 2  # EMA21 pullback in uptrend (Minervini)

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
    rs_sector  = d.get("rs_sector",    0) or 0   # 63-day RS vs own sector ETF (sector leadership)
    rs_rating  = d.get("rs_rating",   50) or 50  # IBD-style 1-99 composite RS Rating

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

    # Sector leadership RS: outperforming own sector = true leader, not just market-tide float (+8/-6)
    if   rs_sector > 10: s +=  8   # clear sector leader — institutions picking this name specifically
    elif rs_sector >  5: s +=  5
    elif rs_sector >  2: s +=  2
    elif rs_sector < -10: s -= 6   # laggard even vs weak sector — avoid
    elif rs_sector <  -5: s -= 3

    # IBD RS Rating (1-99): 12-month composite rank vs market (+12/-8)
    # Stocks with RS≥90 have historically led the market; RS<40 = chronic underperformers
    if   rs_rating >= 90: s += 12   # IBD elite: top 10% RS stocks produce the biggest winners
    elif rs_rating >= 80: s +=  8   # strong leader — institutional focus
    elif rs_rating >= 70: s +=  4   # above average — worth watching
    elif rs_rating >= 60: s +=  2   # mild edge
    elif rs_rating <= 30: s -=  8   # chronic underperformer — avoid longs
    elif rs_rating <= 40: s -=  4   # below-average RS — weak relative performance

    # RS Line New High: IBD's #1 buy confirmation signal (+10)
    # When the RS line (stock/SPY ratio) hits a new 52-week high — especially before price —
    # it reveals institutional accumulation before the breakout is visible to most traders.
    if d.get("rs_line_new_high"):
        s += 10
    elif d.get("rs_line_trending"):
        s +=  4

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
    # Triple alignment (weekly+daily+hourly) is the highest-conviction setup in technical trading
    if d.get("mtf_triple", False):    s += 18   # all 3 TFs aligned = rare, high-conviction
    elif d.get("mtf_aligned", False): s += 12   # daily+hourly = high conviction
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

    # Pivot point proximity: price near S1/S2 = daily institutional support bounce (+5)
    # Pivot near R1/R2 = resistance ceiling, reduce conviction (-3)
    _pprice = d.get("price", 0) or d.get("close", 0) or 0
    _ps1 = d.get("pivot_s1", 0) or 0
    _ps2 = d.get("pivot_s2", 0) or 0
    _pr1 = d.get("pivot_r1", 0) or 0
    _pr2 = d.get("pivot_r2", 0) or 0
    if _pprice > 0 and _ps1 > 0 and abs(_pprice - _ps1) / _pprice < 0.015:
        s += 5   # price within 1.5% of S1 = daily support bounce
    elif _pprice > 0 and _ps2 > 0 and abs(_pprice - _ps2) / _pprice < 0.015:
        s += 6   # price within 1.5% of S2 = stronger support
    if _pprice > 0 and _pr2 > 0 and _pprice >= _pr2 * 0.995:
        s -= 3   # approaching/at R2 = heavy resistance overhead
    elif _pprice > 0 and _pr1 > 0 and _pprice >= _pr1 * 0.998:
        s -= 2   # at R1 = first resistance level

    # Fibonacci retracement support: bouncing off 38.2/50/61.8% level = institutional buy zone (+9)
    if d.get("fib_support", False):    s += 9
    if d.get("fib_resistance", False): s -= 5

    # Swing-based support/resistance: at historical institutional memory zone
    if d.get("near_key_support", False): s += 4   # price at swing support = high-probability bounce zone
    if d.get("near_key_resist", False):  s -= 3   # approaching swing resistance = entry timing risk

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

    # Volume Divergence: distribution vs accumulation based on price-volume relationship
    # Bearish: new high + shrinking volume = institutions distributing into retail strength (-6)
    # Bullish: new low + shrinking sell volume = institutions quietly absorbing supply (+6)
    if d.get("vol_bearish_div", False): s -= 6
    if d.get("vol_bullish_div", False): s += 6

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

    # Parabolic SAR: trend-following trailing stop alignment
    if d.get("psar_bull", True):   s += 5   # SAR below price = uptrend intact
    else:                           s -= 4   # SAR above price = downtrend signal

    # Options flow proxy: unusual call buying = institutional bullish positioning (+7/-5)
    if d.get("unusual_calls", False):    s += 7   # big money buying calls = strong directional bet
    elif d.get("options_bull", False):   s += 4   # low PCR = call skew, bullish sentiment
    if d.get("unusual_puts", False):     s -= 5   # unusual put buying = hedge or bearish bet
    elif d.get("options_bear", False):   s -= 3   # high PCR = put skew, bearish sentiment

    # Price Acceleration: momentum building (2nd derivative positive) = institutional accumulation
    if d.get("price_accel_pos", False):   s += 6   # ROC accelerating ≥1%
    elif d.get("price_accel_neg", False): s -= 5   # ROC decelerating ≥1% = fading momentum

    # Linear Regression Channel: trend quality and position within channel
    _lr_r2 = d.get("lr_r2", 0) or 0
    if d.get("lr_below_channel", False) and _lr_r2 > 0.7:
        s += 5   # Price pulled back below trend line in strong trend = mean-reversion buy
    elif d.get("lr_above_channel", False) and _lr_r2 > 0.8:
        s -= 3   # Extended above trend in very linear uptrend = caution (overbought vs trend)
    # High R² trend with positive slope = strong directional momentum
    _lr_slope = d.get("lr_slope", 0) or 0
    if _lr_r2 > 0.85 and _lr_slope > 20:   # >20% annualized, very linear
        s += 4
    elif _lr_r2 > 0.70 and _lr_slope > 10:  # >10% annualized, fairly linear
        s += 2

    # EMA21 pullback (Minervini's highest-probability entry): pullback to 21-day EMA in uptrend
    # First touch of EMA21 after a breakout run = institutions are buying the dip at this level
    # Confirmed version (green day + in zone) = +14; touching the zone = +7
    if d.get("ema21_pullback", False):   s += 14   # confirmed pullback: trend intact, reversal showing
    elif d.get("ema21_touch", False):    s +=  7   # touching EMA21 in uptrend — watch for entry

    # Higher Lows: ascending support floor = confirmed uptrend structure (+6)
    if d.get("higher_lows", False): s += 6

    # Double Bottom: W-pattern bullish reversal (+10) — confirmed break above neckline
    if d.get("double_bottom", False): s += 10

    # Double Top: M-pattern bearish reversal at resistance (-8) — reduces buy conviction
    if d.get("double_top", False): s -= 8

    # Smart Accumulation Score: composite of OBV + Force Index + MFI smart-money signals
    _accum = d.get("accum_score", 0) or 0
    if _accum >= 8:   s += 7   # very strong accumulation — rare, high conviction
    elif _accum >= 6: s += 4   # solid institutional buying pattern
    elif _accum >= 4: s += 2   # moderate accumulation

    # News Velocity: accelerating news flow = catalyst building = institutional awareness
    if d.get("news_accelerating", False):   s += 5   # 3+ articles in 24h, accelerating
    elif d.get("news_velocity", 0) > 1.5:  s += 2   # noteworthy news increase

    # Pre-market gap: real-time institutional positioning before open
    _pm_gap = d.get("pm_gap_pct", 0.0) or 0.0
    if d.get("pm_big_gap_up", False):    s += 8   # 3%+ gap up = strong institutional catalyst
    elif d.get("pm_gap_up", False):      s += 4   # 1.5%+ gap up = positive momentum
    if d.get("pm_big_gap_down", False):  s -= 7   # 3%+ gap down = distribution
    elif d.get("pm_gap_down", False):    s -= 3   # 1.5%+ gap down = weakness

    # Gamma Exposure (GEX): options market positioning
    # Negative GEX = dealers short gamma = amplified moves = trending/breakout fuel
    # Squeeze potential = large put wall below = short gamma + forced covering
    if d.get("squeeze_potential", False):   s += 8   # gamma squeeze setup — explosive fuel
    elif d.get("gex_sign", 0) == -1:        s += 3   # negative GEX = trending, momentum persists
    # Note: positive GEX = mean-reverting = slight penalty for momentum breakout plays
    elif d.get("gex_sign", 0) == 1:         s -= 1

    # Short Interest: high short float + rising price = short squeeze fuel
    # Combines with other momentum signals to detect potential short squeeze setups
    _sf  = d.get("short_float", 0.0) or 0.0
    _sr  = d.get("short_ratio", 0.0) or 0.0
    _hs  = d.get("high_short", False)
    if _hs:
        # High short + accumulation = prime squeeze setup
        if (d.get("accum_score", 0) or 0) >= 6:  s += 6
        # High short + gamma squeeze = maximum explosive setup
        elif d.get("squeeze_potential"):          s += 4
        # High short + rising OBV = quiet accumulation against shorts
        elif d.get("obv_rising"):                 s += 3
        else:                                     s += 1   # standalone high short (risky without catalyst)
        # Very high short (>25%) is riskier — could also gap down on bad news
        if _sf > 0.25:                            s -= 1   # penalty for "death by short"
    # Long days-to-cover + rising = short covering squeeze more prolonged
    if _sr >= 5 and _hs and d.get("mom_accel"):   s += 2

    # Historical Volatility regime: contracting HV = coiled spring before breakout
    if d.get("hv_contracting", False):             s += 3   # volatility coil = breakout setup
    if d.get("hv_expanding", False) and d.get("rvol_surge", False): s += 2  # vol expansion + RVOL = momentum confirmed

    # Trend Quality Score: clean, strong, linear trends earn a bonus
    _tqs = d.get("trend_quality_score", 0) or 0
    if _tqs >= 8.0:   s += 4   # pristine trend — high quality breakout candidate
    elif _tqs >= 6.0: s += 2   # solid trend quality
    elif _tqs >= 4.0: s += 1   # modest quality

    # Analyst Estimate Revisions: estimate upgrades = institutional re-rating signal
    # Fresh upgrades in last 14d = analysts just revised outlook upward = strong alpha factor
    _ar_score = d.get("analyst_rev_score", 0) or 0
    if _ar_score >= 3:           s += 5  # wave of upgrades + high buy% + big upside
    elif _ar_score >= 2:         s += 3  # solid revision trend
    elif _ar_score >= 1:         s += 1  # mild positive revisions
    elif _ar_score <= -1:        s -= 2  # downgrade wave
    if d.get("analyst_upgrade", False):  s += 2   # fresh upgrade in last 14d
    # Large upside to consensus target = still value to capture
    _up_tgt = d.get("analyst_upside_pct", 0.0) or 0.0
    if _up_tgt >= 20:            s += 2
    elif _up_tgt >= 12:          s += 1
    elif _up_tgt < -10:          s -= 2  # trading above consensus = priced for perfection

    # Pre-earnings drift: stocks tend to rise 5-20 days before earnings on anticipation
    # IBD studies: high-RS stocks in pre-earnings window outperform by 2-3× — one of the
    # most reliable calendar effects in equities. Window: 5-20 days before report.
    _earn_days = d.get("earnings_days")
    if _earn_days is not None and isinstance(_earn_days, (int, float)):
        if 5 <= _earn_days <= 20:
            # Classic pre-earnings drift: only reward stocks with positive momentum context
            if rs_rating >= 70 and roc5 > 0 and (d.get("price_vs_ema200", 0) or 0) > 0:
                s += 10   # high-RS stock in pre-earnings sweet spot — institutions positioning
            elif rs_rating >= 55 and roc5 > 0:
                s += 6    # above-average RS with momentum — moderate drift setup
            else:
                s += 3    # mild pre-earnings positioning bonus
        elif 2 <= _earn_days < 5:
            # Very close to earnings: binary risk zone — slightly penalize new entries
            s -= 4  # too close — gap risk exceeds expected drift

    # Fundamental Quality Score: earnings/revenue growth + margins + ROE
    # Minervini: "only trade quality companies with accelerating earnings" — avoids value traps
    _fq = d.get("fund_quality", 0) or 0
    if _fq >= 3:   s += 10  # exceptional fundamentals: high growth + margins + ROE
    elif _fq >= 2: s +=  6  # strong fundamentals
    elif _fq >= 1: s +=  3  # decent fundamentals
    elif _fq <= -2: s -= 8  # poor fundamentals — technical patterns fail faster here
    elif _fq <= -1: s -= 3  # weak fundamentals — caution
    # Earnings acceleration: ≥25% YoY EPS growth = IBD CAN SLIM "A" criterion
    _eg = d.get("earnings_growth")
    if _eg is not None:
        if _eg >= 0.50:   s +=  6  # exceptional earnings growth
        elif _eg >= 0.25: s +=  3  # strong — CAN SLIM threshold
        elif _eg <= -0.10: s -= 5  # declining earnings = avoid
    # Revenue growth confirms earnings quality
    _rg = d.get("revenue_growth")
    if _rg is not None:
        if _rg >= 0.30:   s +=  4  # hypergrowth revenue
        elif _rg >= 0.15: s +=  2  # healthy revenue growth
        elif _rg <= -0.05: s -= 3  # shrinking revenue

    # Pocket Pivot (O'Neil/Kacher-Morales): up day volume > every down-day volume in prior 10 sessions
    # Identifies institutional buying pressure before the big move begins (+12)
    if d.get("pocket_pivot", False): s += 12

    # High-Tight Flag (Minervini): 3+ consecutive closes near 52W high on volume → explosive setup (+14)
    _htf_c = d.get("htf_consec", 0) or 0
    if d.get("htf", False):
        s += 14
    elif _htf_c >= 2:
        s += 7  # building toward HTF — early alert

    # Minervini Trend Template: institutional-grade uptrend qualification
    # 8/8 = all criteria met → +15 (strongest structural quality bonus in the system)
    # 7/8 = near-perfect → +10, 6/8 → +6, 5/8 → +3
    _tt = d.get("trend_template", 0) or 0
    if _tt == 8:   s += 15
    elif _tt == 7: s += 10
    elif _tt == 6: s +=  6
    elif _tt == 5: s +=  3
    elif _tt <= 2: s -=  5  # very weak structure — avoid new longs

    # Anchored VWAP from 52W Low: price above institutional cost basis since the low
    # Strong confirmation that smart money is in profit → trend continuation likely (+6)
    if d.get("above_avwap_52wl", False):
        _avwap_d = d.get("avwap_dist_pct", 0.0) or 0.0
        if _avwap_d > 10:   s += 3   # extended above AVWAP — ok but less upside
        elif _avwap_d > 0:  s += 6   # healthy position above AVWAP
    elif not d.get("above_avwap_52wl", True):
        s -= 4  # below institutional cost basis — bearish anchor

    # Adaptive scoring: boost/penalize signals based on historical win rates AND avg PnL
    # ── ADAPTIVE NEURAL LAYER: learn from accumulated trade outcomes ─────────
    # Each signal is a neuron. Win/loss feedback strengthens or weakens each connection.
    # Synergy detection: multiple elite signals firing together = extra conviction boost.
    # Sector + hour-of-day context = situational awareness layer.
    if _SIGNAL_WIN_RATES:
        _adaptive_adj = 0.0
        _elite_count  = 0   # count of elite signals (≥65% WR) active right now
        _weak_count   = 0   # count of weak signals (≤38% WR) active right now
        _all_sig_keys = [
            "cup_handle", "vcp", "at_demand_zone", "mom_accel", "obv_rising",
            "kc_breakout", "higher_lows", "double_bottom", "poc_breakout",
            "ema_stacked_bull", "trend_reversal", "bull_flag", "mtf_aligned",
            "fib_support", "macd_bull_div", "rvol_surge", "mfi_bull_div",
            "supertrend_bull", "ha_bull", "donchian_up", "three_white_soldiers",
            "morning_star", "bullish_engulfing", "psar_bull", "price_accel_pos",
            "unusual_calls", "lr_below_channel", "ttm_squeeze_fired",
            "at_breakout", "pocket_pivot", "high_tight_flag",
        ]
        for sig_key in _all_sig_keys:
            if d.get(sig_key) and sig_key in _SIGNAL_WIN_RATES:
                sdata = _SIGNAL_WIN_RATES[sig_key]
                wr  = sdata.get("win_rate", 0.5)
                n   = sdata.get("total", 0)
                avg = sdata.get("avg_pnl", 0.0)
                if n >= 3:  # trust with ≥3 samples
                    wr_norm = wr / 100.0 if wr > 1.0 else wr
                    wr_adj = (wr_norm - 0.5) * 10
                    ev_adj = avg / 2.0
                    weight = min(1.0, n / 20.0)
                    contrib = (wr_adj * 0.6 + ev_adj * 0.4) * weight
                    _adaptive_adj += contrib
                    if wr_norm >= 0.65: _elite_count += 1
                    if wr_norm <= 0.38: _weak_count  += 1

        # Signal synergy boost: when 2+ elite signals fire simultaneously,
        # the combined conviction is exponentially higher than any single signal.
        # This is like multiple neurons firing in sync — a network effect.
        if _elite_count >= 3:
            _adaptive_adj += 5.0   # 3 elite signals together = very high conviction
        elif _elite_count == 2:
            _adaptive_adj += 2.5   # 2 elite signals = meaningful confirmation
        # Weak signal drag: too many weak signals dragging the score down = avoid
        if _weak_count >= 3:
            _adaptive_adj -= 4.0   # multiple weak signals = noisy/unreliable setup
        s += max(-12, min(12, round(_adaptive_adj)))  # cap total adaptive boost at ±12

    # ── SECTOR CONTEXT LAYER: sectors with poor historical win rates penalized ─
    try:
        _sector_key = SECTOR_MAP.get(tk, "other")
        if _LEARNED_COLD_SECTORS and _sector_key in _LEARNED_COLD_SECTORS:
            s -= 5   # this sector has been losing: raise the bar
        elif _LEARNED_HOT_SECTORS and _sector_key in _LEARNED_HOT_SECTORS:
            s += 3   # this sector has been winning: slight boost
    except Exception:
        pass

    # ── HOUR-OF-DAY CONTEXT LAYER: time-aware scoring from history ──────────
    try:
        _now_h_str = str(datetime.now(timezone.utc).hour)
        if _LEARNED_WORST_HOURS and _now_h_str in _LEARNED_WORST_HOURS:
            s -= 4   # historically bad entry window: reduce conviction
        elif _LEARNED_BEST_HOURS and _now_h_str in _LEARNED_BEST_HOURS:
            s += 3   # historically good entry window: boost conviction
    except Exception:
        pass

    # ── 30-MIN WINDOW LAYER: fine-grained time scoring (half-hour precision) ─
    try:
        _now_dt = datetime.now(timezone.utc)
        _hw_key = f"{_now_dt.hour:02d}{'30' if _now_dt.minute >= 30 else '00'}"
        if _LEARNED_WORST_HALFHOURS and _hw_key in _LEARNED_WORST_HALFHOURS:
            s -= 3   # this 30-min window historically bad
        elif _LEARNED_BEST_HALFHOURS and _hw_key in _LEARNED_BEST_HALFHOURS:
            s += 2   # this 30-min window historically great
    except Exception:
        pass

    # ── TICKER MEMORY LAYER: per-ticker historical performance ───────────────
    # The bot remembers individual tickers — great track record = boost,
    # repeated losses on this ticker = caution signal.
    try:
        if _LEARNED_TICKER_MEMORY and tk in _LEARNED_TICKER_MEMORY:
            _tk_adj = _LEARNED_TICKER_MEMORY[tk]
            if _tk_adj != 0:
                s += max(-5, min(4, _tk_adj))  # cap: ±5 for ticker memory
    except Exception:
        pass

    # ── COMPOUND RISK FILTER: when multiple negative factors align ────────────
    # Like a neural network's inhibitory cascade — when many neurons fire "avoid",
    # the compound signal is stronger than any individual factor.
    # This is a key differentiator vs. simpler trading systems that treat each
    # negative signal independently rather than recognizing their compound effect.
    try:
        _neg_factors = 0
        _sector_key2 = SECTOR_MAP.get(tk, "other")
        if _LEARNED_COLD_SECTORS and _sector_key2 in _LEARNED_COLD_SECTORS: _neg_factors += 1
        if _LEARNED_TICKER_MEMORY and tk in _LEARNED_TICKER_MEMORY and _LEARNED_TICKER_MEMORY[tk] <= -3: _neg_factors += 1
        _now_h2 = str(datetime.now(timezone.utc).hour)
        if _LEARNED_WORST_HOURS and _now_h2 in _LEARNED_WORST_HOURS: _neg_factors += 1
        _now_dt2 = datetime.now(timezone.utc)
        _hw_key2 = f"{_now_dt2.hour:02d}{'30' if _now_dt2.minute >= 30 else '00'}"
        if _LEARNED_WORST_HALFHOURS and _hw_key2 in _LEARNED_WORST_HALFHOURS: _neg_factors += 1
        # Apply compounded penalty when 3+ independent risk factors align
        if _neg_factors >= 3:
            s -= 8   # triple-threat: strong compounded avoidance signal
        elif _neg_factors == 2:
            s -= 3   # double-negative: meaningful compound risk
    except Exception:
        pass

    # ── SIGNAL COUNT LAYER: reward trades hitting the learned sweet spot ─────────
    # The bot has learned which # of confirming signals produces the best outcomes.
    # Too few = weak setup; too many = late/crowded; sweet spot = edge.
    try:
        if _LEARNED_SIGNAL_COUNT_SWEET:
            # Count active binary signals in d to determine current signal count
            _sc_keys = [
                "cup_handle", "at_demand_zone", "mom_accel", "vcp", "obv_rising",
                "ema_reclaim", "pullback_to_ma", "earnings_beat", "news_catalyst",
                "gap_up", "psar_bull", "rvol_surge", "ha_bull", "donchian_up",
                "lr_below_channel", "options_bull", "unusual_calls", "price_accel_pos",
            ]
            _sc_live = sum(1 for k in _sc_keys if d.get(k))
            _sc_live_bkt = ("1-3" if _sc_live <= 3 else "4-6" if _sc_live <= 6 else "7-10" if _sc_live <= 10 else "11+")
            if _sc_live_bkt == _LEARNED_SIGNAL_COUNT_SWEET:
                s += 3   # in the learned sweet spot — boost confidence
            elif _sc_live <= 1:
                s -= 2   # very weak signal confluence — reduce conviction
    except Exception:
        pass

    # ── SPY DAY RETURN LAYER: learned penalty when entering on red market days ─
    # If historical data shows red-SPY-day entries consistently fail,
    # apply a -4 penalty to make the bot more selective on down days.
    try:
        if _LEARNED_SPY_DOWN_PENALTY:
            _cur_spy_d1 = _fetch_spy_perf().get("d1", 0.0) or 0.0
            if _cur_spy_d1 < -0.5:   # SPY is down today by more than 0.5%
                s -= 4
    except Exception:
        pass

    # ── SCORE TREND LAYER: penalize entries when score has been declining ─────
    # The bot has learned that falling-score entries fail more often.
    # The score_trend is injected by the buy loop via score_history analysis.
    # Since score() itself doesn't receive this, the penalty is applied at the
    # buy-loop level via the _LEARNED_FALLING_SCORE_PENALTY flag check.
    # (Score trend is captured in signals dict and recorded at trade time.)

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

    # Trend Template: high score means strong bull structure — heavily penalize short entry
    _tt_b = d.get("trend_template", 0) or 0
    if   _tt_b == 8: s -= 15  # strongest bull structure — never short
    elif _tt_b >= 6: s -= 8   # solid bull trend — avoid shorts
    elif _tt_b <= 2: s +=  6  # weak / broken structure = short-friendly

    # Pocket Pivot: institutional buying on volume confirmation — don't short
    if d.get("pocket_pivot", False): s -= 8

    # HTF: stock at 52W highs with volume = extremely bullish — strongly avoid shorts
    if d.get("htf", False): s -= 12

    return max(0, min(100, int(s)))


# ── Position sizing ───────────────────────────────────────────────────────────
def calc_notional(portfolio_val, buying_power, price, atr, vix=20.0, macro_day=False,
                  score_val=0, win_rate=0.5, drawdown_pct=0.0, payoff_ratio=1.5,
                  true_beta=1.0, hv_ratio=1.0):
    """
    ATR-based risk sizing with full Kelly criterion, beta-adjusted, and HV-regime-adaptive sizing.
    Full Kelly: f* = (W*B - L) / B  where W=win%, L=loss%, B=avg_win/avg_loss (payoff)
    Beta adjustment: high-beta stocks get smaller positions (equal-risk sizing).
    HV ratio: hv5/hv20 — expanding vol shrinks size, contracting vol allows slightly larger.
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

    # HV regime adjustment: use individual stock's short-term vs medium-term volatility
    # hv_ratio = hv5 / hv20: >1.5 = vol surging (risky/choppy), <0.65 = vol compressed (coil)
    if hv_ratio and hv_ratio > 0:
        if   hv_ratio >= 2.5: vix_scale *= 0.65   # vol erupting: chop/blowoff, reduce hard
        elif hv_ratio >= 1.8: vix_scale *= 0.78   # vol spiking: be careful
        elif hv_ratio >= 1.4: vix_scale *= 0.88   # slightly above norm: mild caution
        elif hv_ratio <= 0.55: vix_scale *= 1.08  # vol extremely compressed: spring-loaded setup
        elif hv_ratio <= 0.70: vix_scale *= 1.04  # vol contracting: slightly favor
    vix_scale = min(1.5, max(0.2, vix_scale))

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

    # Intraday day type: classify trend vs range vs choppy (5-min SPY data)
    # Skips when market is closed (off-hours runs for crypto) — uses cached result
    day_type_info = intraday_day_type()
    _day_type     = day_type_info.get("day_type", "unknown")
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

    # 3-day consecutive SPY decline tape filter
    # Don't fight a market in a confirmed short-term downtrend — wait for at least one green day
    _spy_consec_decline = False
    _spy_tape_score_adj = 0    # score threshold adjustment for tape condition
    try:
        _spy_tap = yf.download("SPY", period="10d", interval="1d",
                               auto_adjust=True, progress=False)
        if not _spy_tap.empty and len(_spy_tap) >= 4:
            _spy_cl = list(_spy_tap["Close"].dropna())
            _last3_spy = [_spy_cl[i] < _spy_cl[i-1] for i in range(-3, 0)]
            if all(_last3_spy):
                _spy_consec_decline = True
                _spy_tape_score_adj = 8   # raise bar by 8 points when market is in 3d decline
                logger.warning(f"3-day SPY tape filter: SPY down 3 consecutive days — raising buy threshold by {_spy_tape_score_adj}")
            elif sum(_last3_spy) >= 2:
                _spy_tape_score_adj = 4
                logger.info(f"2-of-3 SPY tape caution: raising buy threshold by {_spy_tape_score_adj}")
    except Exception:
        pass

    # Portfolio drawdown guard — compute current drawdown from historical peak
    _prior_tlog  = _load(TRADES_FILE, {})
    _perf_hist   = _prior_tlog.get("perf_history", [])
    _hist_values = [h["v"] for h in _perf_hist if isinstance(h.get("v"), (int, float)) and h["v"] > 0]
    _peak_port   = max(_hist_values) if _hist_values else portfolio_val
    drawdown_pct = max(0.0, (_peak_port - portfolio_val) / _peak_port * 100) if _peak_port > 0 else 0.0
    _drawdown_halt = drawdown_pct >= 5.0  # hard halt: no new buys until portfolio recovers
    if _drawdown_halt:
        logger.warning(f"DRAWDOWN HALT: -{drawdown_pct:.1f}% from peak ${_peak_port:,.0f} — no new buys until recovery")
    elif drawdown_pct > 2:
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

    # Post-process: add rs_sector (stock 63d return vs sector ETF) using ETF data already in live
    # Formula: rs_sector = stock_rs63 - sector_etf_rs63 → no extra API calls needed
    _sector_etf_rs63: dict = {}
    for _sec, _etf in SECTOR_ETFS.items():
        if _etf in live:
            _sector_etf_rs63[_sec] = live[_etf].get("rs63", 0) or 0
    for _tk, _sig in live.items():
        _sec = SECTOR_MAP.get(_tk, "other")
        _etf_rs63 = _sector_etf_rs63.get(_sec)
        if _etf_rs63 is not None:
            _sig["rs_sector"] = round((_sig.get("rs63", 0) or 0) - _etf_rs63, 2)
        else:
            _sig.setdefault("rs_sector", 0.0)

    # Internal scan breadth: how many of our scanned stocks are trending up?
    # This is a proprietary advance/decline ratio for our universe
    _scan_up   = sum(1 for sig in live.values() if (sig.get("change_pct", 0) or 0) > 0.3)
    _scan_down = sum(1 for sig in live.values() if (sig.get("change_pct", 0) or 0) < -0.3)
    _scan_total = max(1, _scan_up + _scan_down)
    _scan_adv_pct = round(_scan_up / _scan_total * 100, 1)
    logger.info(f"Internal scan breadth: {_scan_up}/{_scan_total} ({_scan_adv_pct}%) advancing")
    # If very few stocks are advancing in our universe, be more cautious with new buys
    _scan_breadth_poor = _scan_adv_pct < 30 and _scan_total > 20

    # New 52W Highs vs Lows (from our scan universe): a genuine market health gauge.
    # High new_highs / low new_lows = healthy bull market with broad leadership.
    _new_52wh = sum(1 for sig in live.values() if sig.get("w52_range_pos", 0) >= 95)
    _new_52wl = sum(1 for sig in live.values() if sig.get("w52_range_pos", 0) <= 5)
    _nhl_ratio = round(_new_52wh / max(_new_52wl, 1), 1)  # high/low ratio (>2 = healthy)
    logger.info(f"New 52W Highs:{_new_52wh} Lows:{_new_52wl} ratio:{_nhl_ratio:.1f}")

    # Sector rotation (computed before AI context so it can be included in prompt)
    sector_adjs  = sector_rotation()   # {sector: -8..+8}

    # Sector ETF trend confirmation: identify bearish sectors to filter out buys
    sector_etf_trends = get_sector_etf_trend()

    # Market breadth (computed before AI context for richer prompt)
    breadth = get_market_breadth()

    # AI market context adjustment (use top movers from screeners)
    top_movers_for_ai = [s for s in candidates if s not in BASE_UNIVERSE][:12]
    _rs_leaders    = sorted([tk for tk, sig in live.items() if (sig.get("rs_rating", 50) or 50) >= 80],
                            key=lambda tk: -(live[tk].get("rs_rating", 50) or 50))[:8]
    _ema21_setups  = [tk for tk, sig in live.items() if sig.get("ema21_pullback", False)][:6]
    _pocket_pivots = [tk for tk, sig in live.items() if sig.get("pocket_pivot", False)][:5]
    _htf_stocks_ai = [tk for tk, sig in live.items() if sig.get("htf", False)][:4]
    _tt8_stocks_ai = [tk for tk, sig in live.items() if sig.get("tt_full", False)][:4]
    _prior_hcp     = _prior_tlog.get("portfolio_correlation", {}).get("high_corr_pairs", [])
    _high_corr_strs = [p["pair"] for p in _prior_hcp if p.get("corr", 0) >= 0.85][:4]
    _extra_ctx = {
        "rs_leaders":    _rs_leaders,
        "ema21_setups":  _ema21_setups,
        "pocket_pivots": _pocket_pivots,
        "htf_stocks":    _htf_stocks_ai,
        "tt8_stocks":    _tt8_stocks_ai,
        "scan_adv_pct":  _scan_adv_pct,
        "breadth_trend": breadth.get("breadth_trend", "neutral"),
        "breadth_thrust": breadth.get("breadth_thrust", False),
        "high_corr_pairs": _high_corr_strs,
        "new_52wh":      _new_52wh,
        "new_52wl":      _new_52wl,
        "nhl_ratio":     _nhl_ratio,
    }
    regime_adj   = ai_market_context(regime, top_movers_for_ai, sector_adjs=sector_adjs,
                                     extra_ctx=_extra_ctx)

    # Pre-market gap scan (bonus score for strong gap-up stocks)
    gap_ups = set()
    _gap_data: list = []   # full gap list for tlog storage
    if _time_ok(260):
        gaps = get_premarket_gaps(set(candidates))
        gap_ups = {sym for sym, pct, direction in gaps if direction == "up" and pct >= 3}
        if gap_ups:
            logger.info(f"Gap-up candidates: {', '.join(sorted(gap_ups))}")
        # Build richer gap data for dashboard display
        for _gs, _gp, _gd in gaps:
            _gcat = live.get(_gs, {}).get("catalyst_type", "none") if _gs in live else "none"
            _gsec = SECTOR_MAP.get(_gs, "other")
            _gap_data.append({"ticker": _gs, "gap_pct": _gp, "direction": _gd,
                              "sector": _gsec, "catalyst_type": _gcat,
                              "price": round(live.get(_gs, {}).get("price", 0), 2)})

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

    # Load self-tuned parameters from previous cycle
    _learned = tlog.get("bot_learned_params", {})
    _learned_pos_size_adj  = float(_learned.get("pos_size_adj", 1.0))
    _learned_score_adj     = int(_learned.get("base_score_adj", 0))
    _learned_elite_sigs    = set(_learned.get("elite_signals", []))
    _learned_weak_sigs     = set(_learned.get("weak_signals", []))
    _learned_cold_sectors  = set(_learned.get("cold_sectors", []))
    _learned_best_hours    = set(str(h) for h in _learned.get("best_hours_utc", []))
    _learned_worst_hours   = set(str(h) for h in _learned.get("worst_hours_utc", []))

    # Publish to module-level globals so score() can use them without parameters
    global _LEARNED_COLD_SECTORS, _LEARNED_HOT_SECTORS, _LEARNED_WORST_HOURS, _LEARNED_BEST_HOURS
    global _LEARNED_TICKER_MEMORY, _LEARNED_WORST_HALFHOURS, _LEARNED_BEST_HALFHOURS
    _LEARNED_COLD_SECTORS = _learned_cold_sectors
    _LEARNED_HOT_SECTORS  = set(_learned.get("hot_sectors", []))
    _LEARNED_WORST_HOURS  = _learned_worst_hours
    _LEARNED_BEST_HOURS   = _learned_best_hours
    # Ticker memory: {ticker: score_adj} from learned_params
    _LEARNED_TICKER_MEMORY = _learned.get("ticker_score_adjs", {})
    # Half-hour window awareness
    _LEARNED_WORST_HALFHOURS = set(str(h) for h in _learned.get("worst_halfhours_utc", []))
    _LEARNED_BEST_HALFHOURS  = set(str(h) for h in _learned.get("best_halfhours_utc", []))
    # Breadth minimum (learned from breadth_perf — if weak breadth entries fail, raise the floor)
    global _LEARNED_MIN_BREADTH
    _LEARNED_MIN_BREADTH = float(_learned.get("min_breadth_learned", 0.0) or 0.0)
    # Signal count sweet spot (the bucket with the highest win rate)
    global _LEARNED_SIGNAL_COUNT_SWEET
    _sc_perf_learned = _learned.get("signal_count_perf", [])
    if _sc_perf_learned:
        _sc_best = max(_sc_perf_learned, key=lambda x: x.get("win_rate", 0))
        _LEARNED_SIGNAL_COUNT_SWEET = _sc_best.get("bucket", "")
    else:
        _LEARNED_SIGNAL_COUNT_SWEET = ""
    # SPY day penalty: True when red SPY days consistently produce losses
    global _LEARNED_SPY_DOWN_PENALTY
    _spy_day_learned = _learned.get("spy_day_perf", [])
    _spy_down_learned = next((s for s in _spy_day_learned if s.get("bucket") == "down"), None)
    _LEARNED_SPY_DOWN_PENALTY = bool(_spy_down_learned and _spy_down_learned.get("win_rate", 50) < 40 and _spy_down_learned.get("total", 0) >= 5)
    # Score trend penalty: True when falling-score entries consistently underperform
    global _LEARNED_FALLING_SCORE_PENALTY
    _st_learned = _learned.get("score_trend_perf", [])
    _falling_learned = next((s for s in _st_learned if s.get("trend") == "falling"), None)
    _LEARNED_FALLING_SCORE_PENALTY = bool(_falling_learned and _falling_learned.get("win_rate", 50) < 40 and _falling_learned.get("total", 0) >= 5)
    # ATR stop multiplier: learned from ATR bracket performance
    global _LEARNED_ATR_MULTIPLIER
    _LEARNED_ATR_MULTIPLIER = float(_learned.get("atr_mult_learned", 2.5) or 2.5)
    # Clamp to safe range (1.5x to 4x ATR)
    _LEARNED_ATR_MULTIPLIER = max(1.5, min(4.0, _LEARNED_ATR_MULTIPLIER))

    if _learned:
        logger.info(f"Learned params loaded: score_adj={_learned_score_adj:+d}, size_adj={_learned_pos_size_adj:.2f}x, "
                    f"elite_sigs={len(_learned_elite_sigs)}, cold_sectors={sorted(_learned_cold_sectors)[:3]}, "
                    f"ticker_memory={len(_LEARNED_TICKER_MEMORY)} tickers")

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
                    dyn_trail = max(4.0, min(9.0, atr_pct * _LEARNED_ATR_MULTIPLIER))
                else:
                    dyn_trail = TRAILING_STOP_PCT * 100

            # VIX-adjusted trail: widen in fear, tighten in calm markets
            _vix_now = (tlog.get("regime") or {}).get("vix", 0) or 0
            if _vix_now >= 30:
                dyn_trail = min(dyn_trail * 1.30, dyn_trail + 1.5)   # high fear: +30% room
            elif _vix_now >= 22:
                dyn_trail = min(dyn_trail * 1.15, dyn_trail + 0.8)   # elevated fear: +15% room
            elif _vix_now > 0 and _vix_now < 14:
                dyn_trail = max(dyn_trail * 0.90, dyn_trail - 0.5)   # calm market: tighter stops

            # Pre-earnings stop tightening: 3-7 days before earnings,
            # tighten trailing stop by 50% to lock in pre-earnings drift gains.
            # Binary event risk justifies protecting whatever profit we have.
            _earn_d_close = get_earnings_days(sym)
            if _earn_d_close is not None and 2 < _earn_d_close <= 7 and pnl_pct > 1.5:
                dyn_trail = max(1.0, dyn_trail * 0.50)
                logger.debug(f"Pre-earnings tighten {sym}: trail→{dyn_trail:.1f}% (earns in {_earn_d_close}d)")

            # ── Full exit conditions ──
            reason = None
            # ATR-adaptive stop loss: learned multiplier × ATR from entry, capped at STOP_LOSS_PCT
            _atr_sig = live.get(sym, {})
            _atr_val = _atr_sig.get("atr") if _atr_sig else None
            if _atr_val and cost > 0:
                _atr_pct = _atr_val / cost * 100
                _atr_stop_pct = min(STOP_LOSS_PCT * 100, max(3.0, _atr_pct * _LEARNED_ATR_MULTIPLIER))
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
            # Parabolic SAR exit: SAR flipped from bull to bear = trend reversal confirmed
            _psar_val  = _atr_sig.get("psar", 0) or 0
            _psar_bull = _atr_sig.get("psar_bull", True)
            _use_psar  = (not _psar_bull and _psar_val > 0 and pnl_pct > 3 and age_days >= 2)
            # Pre-earnings sell: exit ANY position within 2 days of earnings to avoid binary risk
            # We rode the pre-earnings drift — now protect profits before the volatile event
            if has_earnings_soon(sym, days=2):
                reason = f"pre-earnings exit (earnings in <2d, {pnl_pct:+.1f}%)"
            elif _use_psar and not _use_chandelier and not _use_supertrend:
                reason = f"parabolic SAR reversal (SAR ${_psar_val:.2f} flipped bearish, {pnl_pct:+.1f}%)"
            elif _use_supertrend and not _use_chandelier:
                reason = f"supertrend reversal (price ${price:.2f} < stop ${_st_stop:.2f}, {pnl_pct:+.1f}%)"
            elif _use_chandelier:
                reason = f"chandelier exit (price ${price:.2f} < stop ${_chan_stop:.2f}, {pnl_pct:+.1f}%)"
            elif pnl_pct <= -_atr_stop_pct:
                reason = f"stop loss ({pnl_pct:+.1f}% ≤ -{_atr_stop_pct:.1f}%)"
            elif pnl_pct > 1.0 and age_days >= 0.5:
                # ── SCORE DECAY EXIT (Neuron 41) ──────────────────────────────────
                # If the live setup score has collapsed vs entry, the thesis is broken.
                # Learned threshold adjusts: starts at 15pts, tightens if decay exits save money.
                try:
                    _n41_entry_sc = peaks.get(sym, {}).get("entry_score", 0) or 0
                    if _n41_entry_sc > 0:
                        _n41_live_sig = live.get(sym, {})
                        _n41_live_sc = score(sym, _n41_live_sig, regime_adj=regime_adj) if _n41_live_sig else _n41_entry_sc
                        _n41_decay = _n41_entry_sc - _n41_live_sc
                        _n41_thresh = float(_learned.get("score_decay_threshold", 15) or 15) if _learned else 15.0
                        _n41_thresh = max(10.0, min(25.0, _n41_thresh))
                        if _n41_decay >= _n41_thresh and pnl_pct > 1.0:
                            reason = f"score decay exit (entry={_n41_entry_sc}→live={_n41_live_sc}, -{_n41_decay:.0f}pts, {pnl_pct:+.1f}%)"
                except Exception:
                    pass
            elif (_atr_target_ent := peaks.get(sym, {}).get("atr_at_entry", 0)) and pnl_pct >= max(PROFIT_TARGET_PCT * 100, min(22.0, _atr_target_ent / cost * 100 * 4.5 if cost > 0 else PROFIT_TARGET_PCT * 100)):
                # ATR-based take-profit: volatile stocks get wider target (4.5× ATR at entry)
                _eff_tp = max(PROFIT_TARGET_PCT * 100, min(22.0, _atr_target_ent / cost * 100 * 4.5 if cost > 0 else PROFIT_TARGET_PCT * 100))
                live_sig_ext = live.get(sym, {})
                still_strong = (live_sig_ext.get("macd_slope", 0) or 0) > 0 and (live_sig_ext.get("roc5", 0) or 0) > 3
                if still_strong and pnl_pct < 30:
                    logger.info(f"HOLD {sym} — extending target to 30% (momentum strong, {pnl_pct:+.1f}%)")
                else:
                    reason = f"profit target ({pnl_pct:+.1f}% ≥ {_eff_tp:.1f}%)"
            elif not peaks.get(sym, {}).get("atr_at_entry") and pnl_pct >= (PROFIT_TARGET_PCT * 100):
                # Fallback for positions without stored ATR (entered before this feature)
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
                # Adaptive max hold: momentum-based, then refined by learned hold period preference
                adaptive_max = MAX_HOLD_DAYS
                if m_slope_age > 0 and roc5_age > 2:
                    adaptive_max = 8   # strong uptrend: let it run
                elif m_slope_age < 0 and roc5_age < 0:
                    adaptive_max = 3   # weak momentum: exit sooner
                # Apply learned hold period preference: if history shows short holds work best, tighten
                _opt_hold = _learned.get("optimal_hold_period") if _learned else None
                if _opt_hold == "short" and adaptive_max > 2:
                    adaptive_max = min(adaptive_max, 2)   # learned: quick exits work better
                elif _opt_hold == "long" and adaptive_max < 7:
                    adaptive_max = max(adaptive_max, 7)   # learned: let winners run longer
                if age_days >= adaptive_max:
                    reason = f"stale position ({age_days:.0f}d, {pnl_pct:+.1f}%)"
            elif peaks.get(sym, {}).get("ever_hit_5pct") and pnl_pct <= 0.5:
                reason = f"breakeven lock ({pnl_pct:+.1f}%)"
            elif peaks.get(sym, {}).get("ever_hit_5pct") and pnl_pct <= -1.5 and age_days >= 1:
                reason = f"winner-turned-loser exit ({pnl_pct:+.1f}%, was up 5%+ earlier)"
            else:
                # Score degradation exit: sustained weakening = exit before full reversal
                _score_hist = [h.get("s") for h in peaks.get(sym, {}).get("score_history", [])
                               if isinstance(h.get("s"), (int, float))]
                if len(_score_hist) >= 4:
                    _score_drop = _score_hist[0] - _score_hist[-1]
                    _consec_drop = all(_score_hist[i] > _score_hist[i+1] for i in range(min(3, len(_score_hist)-1)))
                    if _score_drop >= 20 and _consec_drop and -2 <= pnl_pct <= 4 and age_days >= 2:
                        reason = f"score degradation exit ({_score_hist[0]}→{_score_hist[-1]}, -{_score_drop}pts, {pnl_pct:+.1f}%)"
                # Analyst downgrade exit: multiple recent downgrades + deteriorating momentum
                if not reason:
                    _live_sig_ar = live.get(sym, {})
                    _ar_score = _live_sig_ar.get("analyst_rev_score", 0) or 0
                    _ar_nr    = _live_sig_ar.get("analyst_net_rev", 0) or 0
                    if _ar_score <= -1 and _ar_nr <= -2 and pnl_pct < 2 and age_days >= 3:
                        reason = f"analyst downgrade wave ({_ar_nr} net downgrades, {pnl_pct:+.1f}%)"
                # Consecutive red day exit: 4+ straight down days while losing
                if not reason:
                    _live_sig_cr = live.get(sym, {})
                    _cr = _live_sig_cr.get("consec_red", 0) or 0
                    if _cr >= 4 and pnl_pct < -1.5:
                        reason = f"4 consecutive red days ({_cr}d streak, {pnl_pct:+.1f}%)"
            if reason is None:
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
                    elif live_sig.get("macd_bear_div", False) and pnl_pct > 5 and age_days >= 2:
                        # MACD bearish divergence: price made new high but MACD lower = smart money selling
                        # Very reliable exit signal when confirmed by meaningful profit cushion
                        reason = f"MACD bearish divergence (price↑ but momentum↓, {pnl_pct:+.1f}%)"
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

            # ── THESIS INVALIDATION: exit if original entry signals are gone ──
            # The bot checks if the signals that caused the BUY entry are still present.
            # If the key signals that justified entry have ALL disappeared AND the position is losing,
            # exit now rather than waiting for the stop loss. This is smart early-exit logic.
            if not reason and pnl_pct < -1.5 and age_days >= 0.5:
                try:
                    _entry_sigs_ti = []
                    for _tt in tlog.get("trades", []):
                        if _tt.get("action") == "BUY" and _tt.get("ticker") == sym:
                            _entry_sigs_ti = _tt.get("entry_signals", [])
                            break
                    if _entry_sigs_ti:
                        _live_ti = live.get(sym, {})
                        # Check which entry signals are no longer present
                        _key_sigs_ti = [s for s in _entry_sigs_ti
                                        if s in ("cup_handle", "vcp", "at_breakout", "rvol_surge",
                                                 "ttm_squeeze_fired", "donchian_up", "kc_breakout",
                                                 "at_demand_zone", "three_white_soldiers",
                                                 "ema_stacked_bull", "mtf_aligned",
                                                 "supertrend_bull", "pocket_pivot", "high_tight_flag")]
                        if len(_key_sigs_ti) >= 2:
                            _still_active = [s for s in _key_sigs_ti if _live_ti.get(s)]
                            _gone_pct = (len(_key_sigs_ti) - len(_still_active)) / len(_key_sigs_ti)
                            if _gone_pct >= 0.8 and pnl_pct < -2:
                                # 80%+ of key entry signals gone + losing = thesis broken
                                reason = (f"thesis invalidated: {len(_key_sigs_ti)-len(_still_active)}/{len(_key_sigs_ti)} "
                                          f"entry signals gone ({pnl_pct:+.1f}%)")
                except Exception:
                    pass

            # Pivot point resistance exit: price reached R2 level = institutional sell zone
            # R2 is where most short-sellers enter and longs take profits; high hit-rate exit
            if not reason:
                _piv_r2_exit = (_atr_sig.get("pivot_r2", 0) or 0)
                _piv_r1_exit = (_atr_sig.get("pivot_r1", 0) or 0)
                _live_rsi_exit = (live.get(sym, {}).get("daily_rsi", 50) or 50)
                if _piv_r2_exit > 0 and price >= _piv_r2_exit * 0.998 and pnl_pct > 4:
                    reason = f"pivot R2 resistance (price ${price:.2f} ≥ R2 ${_piv_r2_exit:.2f}, {pnl_pct:+.1f}%)"
                elif _piv_r1_exit > 0 and price >= _piv_r1_exit * 0.998 and pnl_pct > 6 and _live_rsi_exit > 72:
                    reason = f"pivot R1 + overbought (price ${price:.2f} ≥ R1 ${_piv_r1_exit:.2f}, RSI={_live_rsi_exit:.0f}, {pnl_pct:+.1f}%)"

            # P&L velocity reversal: if a winner is reversing RAPIDLY before trail stop hits
            # Detects early momentum exhaustion — don't wait for the full trailing stop
            if not reason and pnl_pct > 2 and not half_out:
                try:
                    _ph_vel = [h["p"] for h in peaks.get(sym, {}).get("pnl_history", [])
                               if isinstance(h.get("p"), (int, float))]
                    if len(_ph_vel) >= 5:
                        _recent_pnls = _ph_vel[-5:]
                        _pnl_drop_5  = _recent_pnls[0] - _recent_pnls[-1]  # drop over last 5 scans (~25 min)
                        _all_falling  = all(_recent_pnls[i] > _recent_pnls[i+1] for i in range(4))
                        if _pnl_drop_5 >= 3.5 and _all_falling and pnl_pct < _recent_pnls[0] - 2:
                            reason = f"P&L velocity reversal ({_recent_pnls[0]:+.1f}%→{pnl_pct:+.1f}%, -{_pnl_drop_5:.1f}% in 5 scans)"
                except Exception:
                    pass

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

                # Scenario C: VWAP -2σ intraday oversold bounce
                vwap_b2d_val = live_sig.get("vwap_b2d", 0) or 0
                is_vwap_oversold = (
                    vwap_b2d_val > 0 and current <= vwap_b2d_val * 1.005
                    and pnl_pct >= -8.0 and pnl_pct <= 3.0
                    and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.85
                )

                # Scenario D: Pivot S1/S2 bounce with at least 2 confirming signals
                pivot_s1_val = live_sig.get("pivot_s1", 0) or 0
                pivot_s2_val = live_sig.get("pivot_s2", 0) or 0
                at_pivot_support = (
                    ((pivot_s1_val > 0 and abs(current - pivot_s1_val) / current < 0.015) or
                     (pivot_s2_val > 0 and abs(current - pivot_s2_val) / current < 0.015))
                    and pnl_pct >= -6.0 and pnl_pct <= 2.0
                    and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.85
                )

                # Scenario E: Breakout continuation pyramid
                # Price breaks to new 20-day Donchian high on high volume — institutions
                # are buying at new highs. Classic CAN SLIM "buying strength" approach.
                _rvol_e    = live_sig.get("rvol", 1.0) or 1.0
                _st_bull_e = live_sig.get("supertrend_bull", True)
                is_breakout_pyramid = (
                    8.0 <= pnl_pct <= 25.0
                    and live_sig.get("donchian_up", False)      # new 20-day high
                    and _rvol_e >= 2.0                          # institutional volume
                    and _st_bull_e                              # supertrend aligned
                    and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.85
                    and not peaks.get(sym, {}).get("half_out", False)
                )

                # Scenario F: Smart Scale-In completion — EMA21 pullback after high-conviction entry
                # We entered at 60% initially; now complete the position when price pulls back to EMA21
                _pk_data = peaks.get(sym, {})
                is_scale_in_complete = (
                    _pk_data.get("scale_in_pending", False)
                    and (_pk_data.get("scale_in_notional", 0) or 0) >= 25
                    and live_sig.get("ema21_pullback", False)
                    and pnl_pct > -2.0 and pnl_pct < 12.0   # still in healthy range
                    and mkt_val < portfolio_val * MAX_POSITION_PCT * 0.85
                )
                if is_scale_in_complete:
                    _si_notional = min(
                        _pk_data.get("scale_in_notional", 0),
                        buying_power * 0.15,
                        portfolio_val * MAX_POSITION_PCT - mkt_val
                    )
                    if _si_notional >= 25:
                        logger.info(f"SCALE-IN {sym} — completing position ${_si_notional:.0f} at EMA21 pullback (pnl={pnl_pct:+.1f}%)")
                        r_si = alpaca_post("/v2/orders", {
                            "symbol": sym, "notional": str(round(_si_notional, 2)),
                            "side": "buy", "type": "market", "time_in_force": "day",
                        })
                        if r_si:
                            buying_power -= _si_notional
                            log_trade(tlog, "DCA", sym, current, _si_notional, score=dca_sc,
                                      reason=f"scale-in complete EMA21 pullback {pnl_pct:+.1f}%")
                            peaks[sym]["scale_in_pending"] = False
                            peaks[sym]["scale_in_notional"] = 0.0
                            made_trades = True
                    continue  # done with this position, skip other DCA scenarios

                if is_pullback_dca or is_winner_pyramid or is_vwap_oversold or at_pivot_support or is_breakout_pyramid:
                    ema50_pos = live_sig.get("price_vs_ema50", 0) or 0
                    roc5_val  = live_sig.get("roc5", 0) or 0

                    # Hard gates: don't DCA into a broken trend
                    if ema50_pos < -4 and roc5_val < -5:
                        logger.debug(f"DCA SKIP {sym} — downtrend (EMA50={ema50_pos:.1f}%, ROC5={roc5_val:.1f}%)")
                        continue
                    if live_sig.get("ema_stacked_bear", False):
                        logger.debug(f"DCA SKIP {sym} — EMA stack bearish")
                        continue
                    # Supertrend and PSAR filters: if both bearish, skip DCA
                    st_bull_dca  = live_sig.get("supertrend_bull", True)
                    psar_bull_dca = live_sig.get("psar_bull", True)
                    if not st_bull_dca and not psar_bull_dca:
                        logger.debug(f"DCA SKIP {sym} — both Supertrend and PSAR bearish")
                        continue
                    # Three Black Crows pattern: strong institutional selling
                    if live_sig.get("three_black_crows", False):
                        logger.debug(f"DCA SKIP {sym} — three black crows pattern")
                        continue
                    if is_pullback_dca and current >= cost * 1.01:
                        logger.debug(f"DCA SKIP {sym} — price above cost")
                        continue

                    dca_sc = score(sym, live_sig, regime_adj=regime_adj)
                    # Score requirements: pivot/VWAP oversold need less score (mean reversion)
                    if is_vwap_oversold or at_pivot_support:
                        min_score = 22   # oversold at key level = lower bar
                    elif is_winner_pyramid:
                        min_score = 25
                    elif is_breakout_pyramid:
                        min_score = 30   # breakout pyramid: high bar to confirm institutional move
                    else:
                        min_score = 28

                    if dca_sc >= min_score:
                        # Boost from MFI oversold: strong accumulation signal
                        mfi_dca = live_sig.get("mfi", 50) or 50
                        mfi_boost = 1.2 if mfi_dca < 25 else 1.0

                        # Position size: breakout pyramid=20%, winner pyramid=25%, VWAP/pivot=35%, pullback=50%
                        if is_breakout_pyramid:
                            size_pct = 0.20   # conservative — buying at new highs has more risk
                            dca_type = f"breakout pyramid (Donchian+RVOL={_rvol_e:.1f}x)"
                        elif is_winner_pyramid:
                            size_pct = 0.25
                            dca_type = "pyramid (VWAP reclaim)"
                        elif is_vwap_oversold:
                            size_pct = 0.35 * mfi_boost
                            dca_type = f"VWAP-2σ bounce (MFI={mfi_dca:.0f})"
                        elif at_pivot_support:
                            size_pct = 0.35 * mfi_boost
                            near_lvl = "S2" if (pivot_s2_val > 0 and abs(current - pivot_s2_val) / current < 0.015) else "S1"
                            dca_type = f"pivot-{near_lvl} bounce"
                        else:
                            size_pct = 0.50 * mfi_boost
                            dca_type = "pullback"

                        dca_notional = min(
                            mkt_val * size_pct,
                            portfolio_val * MAX_POSITION_PCT - mkt_val,
                            buying_power * 0.12,
                        )
                        if dca_notional >= 50:
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
    # Regime-aware max positions: more room in bull, tighter in bear
    _reg_str   = regime.get("regime", "neutral")
    _reg_score = regime.get("score", 0)
    if   _reg_str == "bull" and _reg_score >= 3:  _regime_max = min(MAX_POSITIONS + 3, 15)
    elif _reg_str == "bull":                       _regime_max = min(MAX_POSITIONS + 1, 14)
    elif _reg_str == "bear" and _reg_score <= -3:  _regime_max = max(MAX_POSITIONS - 6, 4)
    elif _reg_str == "bear":                       _regime_max = max(MAX_POSITIONS - 3, 6)
    else:                                          _regime_max = MAX_POSITIONS
    # Drawdown shrinks max further (recovery mode = smaller book)
    if   drawdown_pct >= 5:  _regime_max = max(_regime_max - 4, 2)
    elif drawdown_pct >= 3:  _regime_max = max(_regime_max - 2, 4)
    if _regime_max != MAX_POSITIONS:
        logger.info(f"Regime-adjusted max positions: {_regime_max} (regime={_reg_str}, score={_reg_score}, dd={drawdown_pct:.1f}%)")
    open_long_slots = _regime_max - len(longs)

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
    _last5_pnl = [t["pnl_pct"] for t in _recent_trades[:5]]
    _consecutive_losses = len(_last3_pnl) >= 3 and all(p < 0 for p in _last3_pnl)
    if _consecutive_losses:
        logger.info(f"Consecutive loss guard: last 3 trades lost ({[round(p,1) for p in _last3_pnl]}) — skipping new buys this cycle")

    # Win streak detection: 5+ consecutive wins → bot is in sync with market rhythm
    _win_streak_5 = len(_last5_pnl) >= 5 and all(p > 0 for p in _last5_pnl)
    _win_streak_3 = len(_last3_pnl) >= 3 and all(p > 0 for p in _last3_pnl)
    if _win_streak_5:
        logger.info(f"WIN STREAK: last 5 trades all won ({[round(p,1) for p in _last5_pnl]}) — bot in sync with market")

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

    # Apply self-learned score threshold adjustment from accumulated performance data
    if _learned_score_adj != 0:
        _eff_min_score += _learned_score_adj
        logger.info(f"Learned threshold adj: {_learned_score_adj:+d} → effective min={_eff_min_score}")

    # Win streak mode: 5 straight wins = bot is in sync; slightly lower bar (market is working)
    if _win_streak_5:
        _eff_min_score = max(MIN_BUY_SCORE, _eff_min_score - 3)
        logger.info(f"Win streak mode (5W): lowering threshold by 3 → {_eff_min_score}")
    elif _win_streak_3:
        _eff_min_score = max(MIN_BUY_SCORE, _eff_min_score - 1)
        logger.info(f"Win streak (3W): lowering threshold by 1 → {_eff_min_score}")

    # Internal scan breadth guard: if <30% of our universe is advancing, add +6 to threshold
    if _scan_breadth_poor:
        _eff_min_score += 6
        logger.info(f"Scan breadth guard: only {_scan_adv_pct}% advancing — raising threshold to {_eff_min_score}")

    # Enhanced breadth-driven threshold adjustment (McClellan Oscillator / Breadth Thrust)
    # Breadth Thrust: very rare signal (Zweig) — broad buying surge from oversold → -10 pts
    # (opens up the floodgates: take any setup with a valid score)
    _breadth_mcl = breadth.get("mcl_osc", 0.0) or 0.0
    _breadth_thrust = breadth.get("breadth_thrust", False)
    _breadth_trend  = breadth.get("breadth_trend", "neutral")
    if _breadth_thrust:
        _eff_min_score -= 10  # rare thrust: lower bar significantly
        logger.info(f"BREADTH THRUST — Zweig/McClellan signal fired! Lowering min score by 10 → {_eff_min_score}")
    elif _breadth_trend == "improving" and _breadth_mcl > 5:
        _eff_min_score -= 4   # improving breadth: more setups should work
        logger.info(f"Breadth improving (MCL+{_breadth_mcl:.1f}) — lowering threshold by 4 → {_eff_min_score}")
    elif _breadth_trend == "improving":
        _eff_min_score -= 2
    elif _breadth_trend == "deteriorating" and _breadth_mcl < -5:
        _eff_min_score += 6   # deteriorating breadth: much higher bar
        logger.info(f"Breadth deteriorating (MCL{_breadth_mcl:.1f}) — raising threshold by 6 → {_eff_min_score}")
    elif _breadth_trend == "deteriorating":
        _eff_min_score += 3

    # Learned breadth minimum guard: raise threshold when current breadth is below the learned floor
    _cur_breadth_adv = breadth.get("adv_pct", 50) or 50
    if _LEARNED_MIN_BREADTH > 0 and _cur_breadth_adv < _LEARNED_MIN_BREADTH:
        _eff_min_score += 5
        logger.info(f"Breadth below learned floor ({_cur_breadth_adv:.0f}% < {_LEARNED_MIN_BREADTH:.0f}%) — threshold +5 → {_eff_min_score}")

    # VIX spike guard: if VIX is spiking rapidly, add extra +8 to threshold
    if _vix_spike:
        _eff_min_score += 8
        logger.info(f"VIX spike guard active — raising threshold to {_eff_min_score}")

    # Portfolio beta guard: high-beta portfolio + new high-beta buy = double the risk
    # Add +5 to threshold when portfolio is already high-beta (>1.5 estimate)
    if _port_beta_est > 1.5:
        _eff_min_score += 5
        logger.info(f"Portfolio beta guard: beta≈{_port_beta_est:.2f} — raising threshold to {_eff_min_score}")

    # Drawdown guard: portfolio in drawdown = demand higher-quality signals before entering
    if drawdown_pct >= 5.0:
        _eff_min_score += 12
        logger.info(f"Drawdown guard (-{drawdown_pct:.1f}%): +12 to min score → {_eff_min_score}")
    elif drawdown_pct >= 3.0:
        _eff_min_score += 8
        logger.info(f"Drawdown guard (-{drawdown_pct:.1f}%): +8 to min score → {_eff_min_score}")
    elif drawdown_pct >= 1.5:
        _eff_min_score += 4
        logger.info(f"Drawdown guard (-{drawdown_pct:.1f}%): +4 to min score → {_eff_min_score}")

    # Time-of-day score adjustment: lower threshold during statistically optimal windows
    # Power Hour (3pm-3:45pm ET): institutional accumulation at day's end, strong follow-through
    # Mid-morning sweet spot (10am-11:30am): post-opening noise settled, trends confirmed
    # Avoid: lunch lull (11:30am-1pm) — low volume, choppy, mean-reverting
    _intraday_adj = 0
    if market_open and not _open_guard and not _close_guard:
        if 180 <= _minutes_since_open <= 225:  # 3pm-3:45pm: power hour
            _intraday_adj = -3  # lower threshold: strong institutional activity
            logger.info("Power Hour entry window — lowering min score by 3")
        elif 30 <= _minutes_since_open <= 90:   # 10am-11:30am: morning momentum sweet spot
            _intraday_adj = -2  # slightly lower: noise settled, trends emerging
            logger.info("Morning momentum window (10am-11:30am) — lowering min score by 2")
        elif 90 <= _minutes_since_open <= 150:  # 11:30am-1pm: lunch lull
            _intraday_adj = +4  # raise threshold: avoid choppy low-volume trades
    # Day type adjustment: trending days lower threshold for breakouts; choppy days raise it
    _day_score_adj = day_type_info.get("day_score", 0)
    if _day_score_adj > 0 and market_open and _day_type in ("trend_up", "trend_down"):
        _eff_min_score -= _day_score_adj  # trend day: breakouts more reliable
        logger.info(f"Trend day ({_day_type}, eff={day_type_info.get('efficiency',0):.2f}) — lowering min score by {_day_score_adj}")
    elif _day_score_adj < 0 and market_open:
        _eff_min_score -= _day_score_adj  # choppy day: raise threshold (subtracting negative)
        logger.info(f"Choppy day (eff={day_type_info.get('efficiency',0):.2f}) — raising min score by {abs(_day_score_adj)}")
    _eff_min_score = max(MIN_BUY_SCORE, _eff_min_score + _intraday_adj + _spy_tape_score_adj)

    if open_long_slots > 0 and vix <= VIX_EXTREME_THRESH and not _open_guard and not _close_guard and not _consecutive_losses and not _drawdown_halt:
        # Sector counts for diversification
        sector_counts = {}
        for sym in longs:
            sec = SECTOR_MAP.get(sym, "other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        # Momentum re-entry: stocks recently sold for profit get a 3-day re-entry window
        # They get a +8 bonus score to reflect high-conviction setup
        recent_sells = set()
        cutoff = now_utc - timedelta(days=3)
        # Loss cooldown: stocks sold for >2% loss within 48h are blocked from re-entry
        # Avoids "falling knife" repeats — same thesis that broke is unlikely to immediately recover
        _loss_cooldown: set = set()
        _loss_cutoff = now_utc - timedelta(hours=48)
        for t in tlog.get("trades", []):
            if t.get("action") in ("SELL", "COVER") and (t.get("pnl_pct") or 0) < -2:
                try:
                    if datetime.fromisoformat(t["time"].replace("Z", "+00:00")) > _loss_cutoff:
                        _loss_cooldown.add(t.get("ticker", ""))
                except Exception:
                    pass
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
            # Loss cooldown: don't re-enter within 48h of a >2% loss exit
            if tk in _loss_cooldown:
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": "48h loss cooldown"})
                continue
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
            # Dollar volume filter: price × avg_volume must be ≥ $500K/day to ensure fills
            _dollar_vol = _price_pre * _avg_vol_pre
            if _avg_vol_pre > 0 and _dollar_vol < 500_000:
                logger.debug(f"SKIP {tk} — dollar vol ${_dollar_vol/1e3:.0f}K < $500K minimum")
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": f"thin dollar vol ${_dollar_vol/1e3:.0f}K"})
                continue
            # Correlation guard: skip if >0.85 correlated with a held position
            _held_syms = [s for s in longs if s != tk]
            if is_correlated_with_held(tk, _held_syms, threshold=0.85):
                _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": "corr≥0.85 w/ held position"})
                continue
            # ── NEURAL GATE: accumulated learning hard-vetoes ─────────────────
            # When multiple learned negative patterns fire simultaneously, skip the
            # candidate. This uses the accumulated knowledge from all 40+ neurons
            # to avoid setups that the bot has historically failed on.
            try:
                _ng_strikes = 0
                _ng_reasons = []
                _tk_sig = live.get(tk, {})
                # Strike: in learned cold sector with <35% historical WR
                _tk_sec = SECTOR_MAP.get(tk, "other")
                _sec_wr_ng = tlog.get("sector_performance", {}).get(_tk_sec, {}).get("win_rate", 50) or 50
                _sec_n_ng  = tlog.get("sector_performance", {}).get(_tk_sec, {}).get("total", 0) or 0
                if _sec_n_ng >= 5 and _sec_wr_ng < 35:
                    _ng_strikes += 1; _ng_reasons.append(f"cold sector {_tk_sec}({_sec_wr_ng:.0f}%WR)")
                # Strike: ticker has poor history (memory score < -3)
                if _LEARNED_TICKER_MEMORY.get(tk, 0) <= -3:
                    _ng_strikes += 1; _ng_reasons.append(f"ticker memory penalty({_LEARNED_TICKER_MEMORY[tk]:+d})")
                # Strike: SPY is down today AND learned penalty is active
                if _LEARNED_SPY_DOWN_PENALTY:
                    _spy_today = _fetch_spy_perf().get("d1", 0.0) or 0.0
                    if _spy_today < -1.0:  # more than -1% SPY day
                        _ng_strikes += 1; _ng_reasons.append(f"SPY down({_spy_today:+.1f}%) + learned penalty")
                # Strike: in worst learned half-hour window
                _ng_hw = f"{datetime.now(timezone.utc).hour:02d}{'30' if datetime.now(timezone.utc).minute >= 30 else '00'}"
                if _LEARNED_WORST_HALFHOURS and _ng_hw in _LEARNED_WORST_HALFHOURS:
                    _ng_strikes += 1; _ng_reasons.append(f"worst 30min window({_ng_hw})")
                # Strike: falling score trend AND penalty active
                if _LEARNED_FALLING_SCORE_PENALTY:
                    _ng_sh = [h.get("s") for h in peaks.get(tk, {}).get("score_history", []) if isinstance(h.get("s"), (int, float))]
                    if len(_ng_sh) >= 2 and (_ng_sh[-1] - _ng_sh[0]) <= -8:
                        _ng_strikes += 1; _ng_reasons.append(f"score falling({_ng_sh[-1]-_ng_sh[0]:.0f}pts)")
                # VETO: 3+ strikes = neural gate fires
                if _ng_strikes >= 3:
                    logger.info(f"NEURAL GATE: {tk} vetoed ({_ng_strikes} strikes: {', '.join(_ng_reasons)})")
                    _rejected_log.append({"ticker": tk, "score": tech_sc, "reason": f"neural gate ({_ng_strikes} strikes)"})
                    continue
            except Exception:
                pass
            # Use Sonnet for top 3 candidates (better reasoning), Haiku for rest
            rank = len(final_scores)
            use_sonnet = (rank < 3) and _time_ok(200)
            if _time_ok(280):
                sent, catalyst = ai_sentiment(tk, use_sonnet=use_sonnet, signals=live.get(tk))
            else:
                sent, catalyst = 0, ""
            sec_adj        = sector_adjs.get(sec, 0)
            # Learned sector overlay: if historical win rate in this sector is high/low,
            # amplify or dampen the sector_rotation signal. This creates a self-reinforcing
            # feedback loop: good sector → bot wins → further boosted, bad → further avoided.
            _sec_win_rate = tlog.get("sector_performance", {}).get(sec, {}).get("win_rate", 50) or 50
            _sec_trade_n  = tlog.get("sector_performance", {}).get(sec, {}).get("total", 0) or 0
            if _sec_trade_n >= 4:
                _sec_wr_adj = round((_sec_win_rate - 50) / 10)  # -5..+5 overlay on sector adj
                sec_adj = max(-12, min(12, sec_adj + _sec_wr_adj))

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
            # Score trend guard: if history shows falling-score entries fail, require +4 for them
            if _LEARNED_FALLING_SCORE_PENALTY:
                _sh_now = [h.get("s") for h in peaks.get(tk, {}).get("score_history", []) if isinstance(h.get("s"), (int, float))]
                if len(_sh_now) >= 2 and (_sh_now[-1] - _sh_now[0]) <= -5:
                    _grade_thresh += 4  # falling score = require stronger signal

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
                "mtf_triple":     live.get(tk, {}).get("mtf_triple", False),
                "mtf_score":      live.get(tk, {}).get("mtf_score", 0),
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
                "psar":               live.get(tk, {}).get("psar", 0.0),
                "psar_bull":          live.get(tk, {}).get("psar_bull", True),
                "options_pcr":        live.get(tk, {}).get("options_pcr", 1.0),
                "options_bull":       live.get(tk, {}).get("options_bull", False),
                "unusual_calls":      live.get(tk, {}).get("unusual_calls", False),
                "unusual_puts":       live.get(tk, {}).get("unusual_puts", False),
                "price_accel":        live.get(tk, {}).get("price_accel", 0.0),
                "price_accel_pos":    live.get(tk, {}).get("price_accel_pos", False),
                "price_accel_neg":    live.get(tk, {}).get("price_accel_neg", False),
                "lr_slope":           live.get(tk, {}).get("lr_slope", 0.0),
                "lr_r2":              live.get(tk, {}).get("lr_r2", 0.0),
                "lr_below_channel":   live.get(tk, {}).get("lr_below_channel", False),
                "lr_above_channel":   live.get(tk, {}).get("lr_above_channel", False),
                "ha_bull":            live.get(tk, {}).get("ha_bull", False),
                "ha_consec_bull":     live.get(tk, {}).get("ha_consec_bull", 0),
                "donchian_up":        live.get(tk, {}).get("donchian_up", False),
                "donchian_pct":       live.get(tk, {}).get("donchian_pct", 50.0),
                "three_white_soldiers": live.get(tk, {}).get("three_white_soldiers", False),
                "morning_star":       live.get(tk, {}).get("morning_star", False),
                "bullish_engulfing":  live.get(tk, {}).get("bullish_engulfing", False),
                "hammer":             live.get(tk, {}).get("hammer", False),
                "three_black_crows":  live.get(tk, {}).get("three_black_crows", False),
                "bearish_engulfing":  live.get(tk, {}).get("bearish_engulfing", False),
                "shooting_star":      live.get(tk, {}).get("shooting_star", False),
                "pivot_r1":           live.get(tk, {}).get("pivot_r1", 0.0),
                "pivot_r2":           live.get(tk, {}).get("pivot_r2", 0.0),
                "pivot_s1":           live.get(tk, {}).get("pivot_s1", 0.0),
                "pivot_s2":           live.get(tk, {}).get("pivot_s2", 0.0),
                "earnings_days":      get_earnings_days(tk),
                "vcp":              live.get(tk, {}).get("vcp", False),
                "grade":          momentum_grade(live.get(tk, {}), sc),
                "accum_score":    live.get(tk, {}).get("accum_score", 0),
                "news_accelerating": live.get(tk, {}).get("news_accelerating", False),
                "news_velocity":  live.get(tk, {}).get("news_velocity", 0.0),
                "news_count_24h": live.get(tk, {}).get("news_count_24h", 0),
                "catalyst_type":  live.get(tk, {}).get("catalyst_type", "none"),
                "catalyst_urg":   live.get(tk, {}).get("catalyst_urg", 0),
                "catalyst_dir":   live.get(tk, {}).get("catalyst_dir", "none"),
                "pm_gap_pct":     live.get(tk, {}).get("pm_gap_pct", 0.0),
                "pm_gap_up":      live.get(tk, {}).get("pm_gap_up", False),
                "pm_gap_down":    live.get(tk, {}).get("pm_gap_down", False),
                "pm_big_gap_up":  live.get(tk, {}).get("pm_big_gap_up", False),
                "pm_price":       live.get(tk, {}).get("pm_price", 0.0),
                "mtf_triple":     live.get(tk, {}).get("mtf_triple", False),
                "mtf_score":      live.get(tk, {}).get("mtf_score", 0),
                "gex_sign":       live.get(tk, {}).get("gex_sign", 0),
                "gamma_wall_up":  live.get(tk, {}).get("gamma_wall_up", 0.0),
                "gamma_wall_down":live.get(tk, {}).get("gamma_wall_down", 0.0),
                "squeeze_potential": live.get(tk, {}).get("squeeze_potential", False),
                "trend_quality_score": live.get(tk, {}).get("trend_quality_score", 0.0),
                "consec_green":      live.get(tk, {}).get("consec_green", 0),
                "consec_red":        live.get(tk, {}).get("consec_red", 0),
                "hv20":              live.get(tk, {}).get("hv20", 0.0),
                "hv5":               live.get(tk, {}).get("hv5", 0.0),
                "hv_expanding":      live.get(tk, {}).get("hv_expanding", False),
                "hv_contracting":    live.get(tk, {}).get("hv_contracting", False),
                "key_support_1":     live.get(tk, {}).get("key_support_1", 0.0),
                "key_resist_1":      live.get(tk, {}).get("key_resist_1", 0.0),
                "near_key_support":  live.get(tk, {}).get("near_key_support", False),
                "near_key_resist":   live.get(tk, {}).get("near_key_resist", False),
                "fib_level_382":     live.get(tk, {}).get("fib_level_382", 0.0),
                "fib_level_500":     live.get(tk, {}).get("fib_level_500", 0.0),
                "fib_level_618":     live.get(tk, {}).get("fib_level_618", 0.0),
                "w52_range_pos":     live.get(tk, {}).get("w52_range_pos", 0.0),
                "rs_sector":         round(live.get(tk, {}).get("rs_sector", 0.0), 2),
                "rs252":             round(live.get(tk, {}).get("rs252", 0.0), 2),
                "rs_rating":         live.get(tk, {}).get("rs_rating", 50),
                "rs_line_new_high":  live.get(tk, {}).get("rs_line_new_high", False),
                "rs_line_trending":  live.get(tk, {}).get("rs_line_trending", False),
                "ema21_pullback":    live.get(tk, {}).get("ema21_pullback", False),
                "ema21_touch":       live.get(tk, {}).get("ema21_touch", False),
                "fund_quality":      live.get(tk, {}).get("fund_quality", 0),
                "earnings_growth":   live.get(tk, {}).get("earnings_growth"),
                "revenue_growth":    live.get(tk, {}).get("revenue_growth"),
                "pocket_pivot":      live.get(tk, {}).get("pocket_pivot", False),
                "htf":               live.get(tk, {}).get("htf", False),
                "htf_consec":        live.get(tk, {}).get("htf_consec", 0),
                "trend_template":    live.get(tk, {}).get("trend_template", 0),
                "tt_full":           live.get(tk, {}).get("tt_full", False),
                "above_avwap_52wl":  live.get(tk, {}).get("above_avwap_52wl", False),
                "avwap_52wl":        live.get(tk, {}).get("avwap_52wl", 0.0),
                "avwap_dist_pct":    live.get(tk, {}).get("avwap_dist_pct", 0.0),
                "vol_bearish_div":   live.get(tk, {}).get("vol_bearish_div", False),
                "vol_bullish_div":   live.get(tk, {}).get("vol_bullish_div", False),
                # Kelly-suggested position size as % of portfolio (half-Kelly for safety)
                "kelly_size_pct":    round(max(0, min(10.0, (
                    (win_rate * max(0.5, min(5.0, _payoff_ratio)) - (1 - win_rate)) /
                    max(0.01, max(0.5, min(5.0, _payoff_ratio))) * 50   # half-Kelly %
                ) * (1 + (sc - 60) / 100))), 1) if win_rate > 0.4 and _payoff_ratio > 0.5 else 0.0,
                "atr":               round(live.get(tk, {}).get("atr", 0.0) or 0.0, 3),
                "effective_min_score": tlog.get("effective_min_score", 60),
            }
            for tk, sc, sent, sec, cat in (final_scores or [])[:8]
        ]
        tlog["last_scan_rejected"] = _rejected_log[:8]

        # Portfolio Rotation Intelligence: when portfolio is full and a meaningfully
        # superior new candidate exists, surface the weakest held position for review.
        # This does NOT auto-trade — it flags the opportunity for the dashboard.
        tlog["rotation_suggestion"] = None
        try:
            if open_long_slots == 0 and final_scores and held:
                _top_cand = final_scores[0]
                _top_cand_sc = _top_cand[1]
                _held_scores = {}
                for _hs in held:
                    _hsig = live.get(_hs, {})
                    if _hsig:
                        _held_scores[_hs] = score(_hs, _hsig, regime_adj=regime_adj)
                if _held_scores:
                    _weakest_sym = min(_held_scores, key=_held_scores.get)
                    _weakest_sc  = _held_scores[_weakest_sym]
                    _gap = _top_cand_sc - _weakest_sc
                    if _gap >= 18:
                        _weak_pnl = 0
                        for _p_chk in (tlog.get("positions") or []):
                            if _p_chk.get("ticker") == _weakest_sym:
                                _weak_pnl = _p_chk.get("pnl_pct", 0) or 0
                                break
                        tlog["rotation_suggestion"] = {
                            "sell":       _weakest_sym,
                            "sell_score": _weakest_sc,
                            "sell_pnl":   round(_weak_pnl, 2),
                            "buy":        _top_cand[0],
                            "buy_score":  _top_cand_sc,
                            "buy_sent":   round(_top_cand[2], 1),
                            "gap":        round(_gap, 1),
                            "buy_cat":    _top_cand[4] or "",
                            "buy_sector": _top_cand[3] or "",
                        }
                        logger.info(
                            f"ROTATION SIGNAL: SELL {_weakest_sym}(score={_weakest_sc},pnl={_weak_pnl:.1f}%) "
                            f"→ BUY {_top_cand[0]}(score={_top_cand_sc}, gap={_gap:.0f}pts)"
                        )
        except Exception as _rot_e:
            logger.debug(f"Rotation check: {_rot_e}")

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
                    # Loss cooldown: don't re-enter stocks sold at >2% loss within 48h
                    # "Do not add to losers" principle — wait for the pattern to re-form
                    if tk in _loss_cooldown:
                        logger.info(f"SKIP {tk} — 48h loss cooldown (sold at >2% loss recently)")
                        continue
                    d        = live[tk]
                    price    = d["price"]
                    atr      = d.get("atr")
                    _tk_beta = d.get("true_beta", 1.0) or 1.0
                    _tk_hv5  = d.get("hv5",  0.0) or 0.0
                    _tk_hv20 = d.get("hv20", 0.0) or 0.0
                    _tk_hv_ratio = (_tk_hv5 / _tk_hv20) if _tk_hv20 > 0 else 1.0
                    notional = calc_notional(portfolio_val, buying_power, price, atr, vix,
                                             macro_day=macro_day, score_val=sc,
                                             win_rate=win_rate, drawdown_pct=drawdown_pct,
                                             payoff_ratio=_payoff_ratio, true_beta=_tk_beta,
                                             hv_ratio=_tk_hv_ratio)
                    # Portfolio heat adjustment: if sitting on big unrealized gains ("house money"),
                    # allow slightly larger positions; if deeply underwater, shrink further
                    if _portfolio_heat > 5:
                        notional = min(notional * 1.1, portfolio_val * MAX_POSITION_PCT * 1.2)
                    elif _portfolio_heat < -5:
                        notional = notional * 0.8
                    # Signal win-rate boost: if historically reliable signals are active,
                    # size up based on their tracked win rate (min 5 trades for significance)
                    _sig_wr_map_buy = tlog.get("signal_win_rates", {})
                    _wr_boost = 1.0
                    for _sig_chk in ("at_breakout", "donchian_up", "rvol_surge",
                                     "ttm_squeeze_fired", "cup_handle", "vcp",
                                     "three_white_soldiers", "morning_star"):
                        _sinfo = _sig_wr_map_buy.get(_sig_chk, {})
                        _swr   = _sinfo.get("win_rate", 0)
                        _sn    = _sinfo.get("n", 0)
                        if _sn >= 5 and live.get(tk, {}).get(_sig_chk) and _swr >= 0.65:
                            _wr_boost = min(1.35, 1.0 + (_swr - 0.50) * 1.4)
                            break
                    if _wr_boost > 1.0:
                        notional = min(notional * _wr_boost, portfolio_val * MAX_POSITION_PCT, buying_power * 0.40)
                    # Correlation-aware sizing: if new position is highly correlated to
                    # held positions (same sector/theme), reduce size to limit concentration risk.
                    # Correlation proxy: same SECTOR_MAP sector = reduce by 20-35%.
                    _corr_adj = 1.0
                    _tk_sector = SECTOR_MAP.get(tk, "other")
                    _held_same_sector = [s for s in longs if SECTOR_MAP.get(s, "other") == _tk_sector]
                    if len(_held_same_sector) >= 2:
                        _corr_adj = 0.65  # strong concentration — reduce to 65%
                        logger.info(f"CORR-ADJ {tk}: {len(_held_same_sector)} positions in {_tk_sector} → size×0.65")
                    elif len(_held_same_sector) == 1:
                        _corr_adj = 0.80  # some overlap — reduce to 80%
                    if _corr_adj < 1.0:
                        notional = round(notional * _corr_adj, 2)

                    # HIGH-CONVICTION size boost: when score, RS Rating, AND multi-timeframe all align,
                    # this is the rarest and most reliable setup — deserve a larger position.
                    # IBD research: stocks with RS≥90 + MTF triple alignment win 70%+ of the time.
                    _tk_rs_r = d.get("rs_rating", 50) or 50
                    _is_high_conv = (sc >= 80 and _tk_rs_r >= 80 and d.get("mtf_triple", False)
                                     and not d.get("hv_expanding", False))
                    if _is_high_conv:
                        # Allow up to 15% of portfolio for the highest-conviction setups
                        notional = min(notional * 1.5, portfolio_val * 0.15, buying_power * 0.45)
                        logger.info(f"HIGH-CONVICTION boost {tk}: score={sc}, RS={_tk_rs_r}, MTF triple → ${notional:.0f}")
                    # Size up further for strong catalysts or squeeze setups (on top of Kelly)
                    if catalyst and sent >= 5:
                        notional = min(notional * 1.4, portfolio_val * MAX_POSITION_PCT, buying_power * 0.4)
                    elif tk in squeeze_cands or tk in vol_surge_cands:
                        notional = min(notional * 1.2, portfolio_val * MAX_POSITION_PCT, buying_power * 0.35)
                    # Apply self-learned position size adjustment (from accumulated win/loss data)
                    if _learned_pos_size_adj != 1.0:
                        notional = round(notional * _learned_pos_size_adj, 2)

                    # Regime-Aware Dynamic Sizing Neuron: adapt size to current market regime
                    # Choppy/bear → smaller bets; strong_bull → allow slightly larger
                    _cur_regime_str = regime.get("regime", "neutral")
                    if _cur_regime_str in ("bear",):
                        notional = round(notional * 0.75, 2)   # Bear: 25% smaller
                        logger.debug(f"Bear regime sizing: {tk} notional reduced 25%")
                    elif _cur_regime_str in ("choppy", "neutral"):
                        notional = round(notional * 0.85, 2)   # Choppy: 15% smaller
                    elif _cur_regime_str == "strong_bull" and _r20_wr >= 0.60:
                        # Only size up in strong bull if recent win rate is solid
                        notional = round(notional * 1.10, 2)   # Strong bull + winning: +10%

                    if notional < 1:
                        logger.info(f"SKIP {tk} — insufficient buying power")
                        continue
                    # ATR-based stop: learned multiplier × ATR below entry, capped at STOP_LOSS_PCT
                    if atr and price > 0:
                        _atr_stop_buy = min(STOP_LOSS_PCT, max(0.03, (atr / price) * _LEARNED_ATR_MULTIPLIER))
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
                    if _d_buy.get("donchian_up"):          reason += " [DON-BRK]"
                    if _d_buy.get("ha_bull"):               reason += f" [HA×{_d_buy.get('ha_consec_bull',0)}]"
                    if _d_buy.get("mfi_bull_div"):          reason += f" [MFI-div{_d_buy.get('mfi',50):.0f}]"
                    if _d_buy.get("supertrend_bull"):       reason += f" [ST${_d_buy.get('supertrend_stop',0):.1f}]"
                    if _d_buy.get("rvol_surge"):            reason += f" [RVOL{_d_buy.get('rvol',1):.1f}x]"
                    # Candlestick confirmation tags
                    if _d_buy.get("three_white_soldiers"):  reason += " [3-WS]"
                    if _d_buy.get("morning_star"):          reason += " [M-STAR]"
                    if _d_buy.get("bullish_engulfing"):     reason += " [BULL-ENG]"
                    if _d_buy.get("hammer"):                reason += " [HAMMER]"
                    if _d_buy.get("psar_bull"):
                        _psar_entry = _d_buy.get("psar", 0) or 0
                        if _psar_entry > 0: reason += f" [SAR▲${_psar_entry:.2f}]"
                    if _d_buy.get("unusual_calls"):
                        reason += f" [OPT-CALLS pcr={_d_buy.get('options_pcr',1):.2f}]"
                    elif _d_buy.get("options_bull"):
                        reason += f" [OPT-BULL pcr={_d_buy.get('options_pcr',1):.2f}]"
                    if _d_buy.get("price_accel_pos"):
                        reason += f" [ACCEL+{_d_buy.get('price_accel',0):.1f}%]"
                    # Pivot point proximity (buying near S1/S2 = strong institutional support)
                    _piv_s1 = _d_buy.get("pivot_s1", 0) or 0
                    _piv_s2 = _d_buy.get("pivot_s2", 0) or 0
                    _piv_r1 = _d_buy.get("pivot_r1", 0) or 0
                    if _piv_s1 > 0 and abs(price - _piv_s1) / price < 0.015:
                        reason += f" [@S1${_piv_s1:.2f}]"
                    elif _piv_s2 > 0 and abs(price - _piv_s2) / price < 0.015:
                        reason += f" [@S2${_piv_s2:.2f}]"
                    if _piv_r1 > 0 and price > _piv_r1 * 0.999:
                        reason += f" [abv-R1${_piv_r1:.2f}]"
                    # New signals
                    if _d_buy.get("mtf_triple"):           reason += " [3TF✓]"
                    if _d_buy.get("news_accelerating"):
                        reason += f" [NEWS↑{_d_buy.get('news_count_24h',0)}art]"
                    if _d_buy.get("pm_big_gap_up"):
                        reason += f" [PM+{_d_buy.get('pm_gap_pct',0):.1f}%]"
                    if _d_buy.get("squeeze_potential"):    reason += " [γSQZ]"
                    if (_d_buy.get("accum_score", 0) or 0) >= 8:
                        reason += f" [ACC{_d_buy.get('accum_score',0)}]"
                    if _d_buy.get("high_short"):
                        reason += f" [SI{round((_d_buy.get('short_float',0) or 0)*100)}%]"
                    if tk in vol_surge_cands and tk in squeeze_cands:
                        reason += " [VOL+SQZ]"
                    # Generate AI trade thesis for position card (non-blocking, best-effort)
                    _entry_thesis = ""
                    try:
                        if ANTHROPIC_KEY and sc >= 65:  # only for decent-quality entries
                            _entry_thesis = ai_trade_thesis(tk, sc, _d_buy, catalyst, sent)
                    except Exception:
                        pass

                    # Smart scale-in: highest-conviction setups enter at 60% to allow for pullback add
                    _tt_buy     = _d_buy.get("trend_template", 0) or 0
                    _htf_buy    = _d_buy.get("htf", False)
                    _pp_buy     = _d_buy.get("pocket_pivot", False)
                    _is_scalein = _tt_buy >= 7 or (_htf_buy and _pp_buy)
                    if _is_scalein:
                        _full_notional = notional
                        notional = round(notional * 0.60, 2)  # enter at 60% initially
                        reason += f" [SCALE-IN 60% TT{_tt_buy}]"
                    else:
                        _full_notional = notional
                    # Inject extra context into signals for learning neurons
                    _buy_signals_merged = dict(live.get(tk, {}))
                    # Score Trend Neuron (Neuron 25)
                    try:
                        _sh_hist = [h.get("s") for h in peaks.get(tk, {}).get("score_history", []) if isinstance(h.get("s"), (int, float))]
                        if len(_sh_hist) >= 2:
                            _sh_delta = _sh_hist[-1] - _sh_hist[0]
                            _buy_signals_merged["score_trend"] = "rising" if _sh_delta >= 5 else ("falling" if _sh_delta <= -5 else "flat")
                            _buy_signals_merged["score_trend_delta"] = round(_sh_delta, 1)
                        else:
                            _buy_signals_merged["score_trend"] = "flat"
                            _buy_signals_merged["score_trend_delta"] = 0.0
                    except Exception:
                        _buy_signals_merged["score_trend"] = "flat"
                    # Position Size Neuron (Neuron 28): % of portfolio
                    try:
                        _pv_buy = float(portfolio_val) if portfolio_val > 0 else 10000.0
                        _pos_pct = round(notional / _pv_buy * 100, 1)
                        _buy_signals_merged["pos_size_pct"] = _pos_pct
                        _buy_signals_merged["pos_size_bucket"] = ("<2%" if _pos_pct < 2 else "2-5%" if _pos_pct < 5 else "5-10%" if _pos_pct < 10 else "10%+")
                    except Exception:
                        _buy_signals_merged["pos_size_pct"] = 0.0
                        _buy_signals_merged["pos_size_bucket"] = "2-5%"
                    # ATR Stop Distance Neuron (Neuron 29): ATR as % of price
                    try:
                        _atr_buy = float(atr) if atr else 0.0
                        _price_buy = float(price) if price > 0 else 1.0
                        _atr_pct = round(_atr_buy / _price_buy * 100, 1) if _atr_buy > 0 else 0.0
                        _buy_signals_merged["atr_pct_at_entry"] = _atr_pct
                        _buy_signals_merged["atr_bucket"] = ("<1%" if _atr_pct < 1 else "1-2%" if _atr_pct < 2 else "2-4%" if _atr_pct < 4 else "4%+")
                    except Exception:
                        _buy_signals_merged["atr_pct_at_entry"] = 0.0
                        _buy_signals_merged["atr_bucket"] = "1-2%"
                    log_trade(tlog, "BUY", tk, price, notional, score=sc, reason=reason,
                              signals=_buy_signals_merged)
                    peaks[tk] = {"peak": price, "time": now_utc.isoformat(), "half_out": False,
                                 "ever_hit_5pct": False, "atr_at_entry": atr or 0.0,
                                 "scale_in_pending": _is_scalein,
                                 "scale_in_notional": round(_full_notional * 0.40, 2) if _is_scalein else 0.0,
                                 "scale_in_ema21": round(_d_buy.get("avwap_52wl", 0) or price * 0.97, 2),
                                 "entry_thesis": _entry_thesis,
                                 "entry_tt": _tt_buy,
                                 "entry_score": sc}
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
    # Update score + P&L history for all held positions BEFORE saving peaks
    # Keeps a rolling 24-scan window to detect score degradation and P&L trend
    if live and held:
        for _h_sym in list(held.keys()):
            _h_sig = live.get(_h_sym, {})
            if not _h_sig:
                continue
            _h_sc = score(_h_sym, _h_sig, regime_adj=regime_adj)
            if _h_sym not in peaks or not isinstance(peaks.get(_h_sym), dict):
                peaks[_h_sym] = {"peak": _h_sig.get("price", 0), "time": now_utc.isoformat(), "half_out": False}
            _sh = peaks[_h_sym].setdefault("score_history", [])
            _sh.append({"s": _h_sc, "t": now_utc.isoformat()})
            peaks[_h_sym]["score_history"] = _sh[-24:]  # keep last 24 scans (~2hrs)
            # P&L history — track unrealized return so we can show sparkline in dashboard
            _h_cost = held[_h_sym].get("avg_entry_price", 0) or 0
            _h_price = _h_sig.get("price", 0) or 0
            if _h_cost > 0 and _h_price > 0:
                _h_pnl = round((_h_price - _h_cost) / _h_cost * 100, 2)
                _ph = peaks[_h_sym].setdefault("pnl_history", [])
                _ph.append({"p": _h_pnl, "t": now_utc.isoformat()})
                peaks[_h_sym]["pnl_history"] = _ph[-24:]
                # MAE (Maximum Adverse Excursion): worst loss from entry ever seen
                # MFE (Maximum Favorable Excursion): best gain from entry ever seen
                # Used to calibrate stops and size decisions — institutional standard
                _mae = peaks[_h_sym].get("mae", 0.0)  # most negative pnl seen (stored as negative)
                _mfe = peaks[_h_sym].get("mfe", 0.0)  # most positive pnl seen
                peaks[_h_sym]["mae"] = min(_mae, _h_pnl)   # update if new low
                peaks[_h_sym]["mfe"] = max(_mfe, _h_pnl)   # update if new high
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
                "vwap_price":     round(sig.get("vwap_price", 0), 2),
                "roc5":           round(sig.get("roc5", 0), 2),
                "macd_slope":     round(sig.get("macd_slope", 0), 4),
                "vol_ratio":      round(sig.get("vol_ratio", 1), 2),
                "vwap_reclaim":   sig.get("vwap_reclaim", False),
                "adx":            round(sig.get("adx", 0), 1),
                "adx_trend":      "strong" if sig.get("adx", 0) >= 25 else ("weak" if sig.get("adx", 0) < 15 else "moderate"),
                "rs5":            round(sig.get("rs5", 0), 2),
                "rs63":           round(sig.get("rs63", 0), 2),
                "rs_line_new_high":  sig.get("rs_line_new_high", False),
                "rs_line_trending":  sig.get("rs_line_trending", False),
                "chandelier_stop": round(sig.get("chandelier_stop", 0), 2),
                "ichimoku":        sum([sig.get("ichimoku_above", False), sig.get("ichimoku_bull_cloud", False),
                                        sig.get("ichimoku_tk_bull", False), sig.get("ichimoku_chikou", False)]),
                "macd_bull_div":   sig.get("macd_bull_div", False),
                "mtf_aligned":     sig.get("mtf_aligned", False),
                "mtf_triple":      sig.get("mtf_triple", False),
                "mtf_score":       sig.get("mtf_score", 0),
                "mtf_conflict":    sig.get("mtf_conflict", False),
                "weekly_bull":     sig.get("weekly_bull", False),
                "daily_up":        sig.get("daily_up", False),
                "hourly_up":       sig.get("hourly_up", False),
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
                "pivot":           round(sig.get("pivot", 0), 2),
                "pivot_r1":        round(sig.get("pivot_r1", 0), 2),
                "pivot_r2":        round(sig.get("pivot_r2", 0), 2),
                "pivot_s1":        round(sig.get("pivot_s1", 0), 2),
                "pivot_s2":        round(sig.get("pivot_s2", 0), 2),
                "psar":            round(sig.get("psar", 0), 3),
                "psar_bull":       sig.get("psar_bull", True),
                "options_pcr":     round(sig.get("options_pcr", 1.0), 3),
                "options_bull":    sig.get("options_bull", False),
                "options_bear":    sig.get("options_bear", False),
                "unusual_calls":   sig.get("unusual_calls", False),
                "unusual_puts":    sig.get("unusual_puts", False),
                "vwap_b1u":        round(sig.get("vwap_b1u", 0), 2),
                "vwap_b2u":        round(sig.get("vwap_b2u", 0), 2),
                "vwap_b1d":        round(sig.get("vwap_b1d", 0), 2),
                "vwap_b2d":        round(sig.get("vwap_b2d", 0), 2),
                "price_accel":     round(sig.get("price_accel", 0), 3),
                "price_accel_pos": sig.get("price_accel_pos", False),
                "lr_slope":        round(sig.get("lr_slope", 0), 1),
                "lr_r2":           round(sig.get("lr_r2", 0), 3),
                "lr_below_channel": sig.get("lr_below_channel", False),
                "lr_above_channel": sig.get("lr_above_channel", False),
                "ha_bull":         sig.get("ha_bull", False),
                "ha_consec_bull":  sig.get("ha_consec_bull", 0),
                "hammer":          sig.get("hammer", False),
                "bullish_engulfing": sig.get("bullish_engulfing", False),
                "morning_star":    sig.get("morning_star", False),
                "three_white_soldiers": sig.get("three_white_soldiers", False),
                "accum_score":     sig.get("accum_score", 0),
                "news_accelerating": sig.get("news_accelerating", False),
                "news_velocity":   sig.get("news_velocity", 0.0),
                "news_count_24h":  sig.get("news_count_24h", 0),
                "pm_gap_pct":      sig.get("pm_gap_pct", 0.0),
                "pm_gap_up":       sig.get("pm_gap_up", False),
                "pm_gap_down":     sig.get("pm_gap_down", False),
                "pm_big_gap_up":   sig.get("pm_big_gap_up", False),
                "pm_price":        sig.get("pm_price", 0.0),
                "gex_sign":        sig.get("gex_sign", 0),
                "gamma_wall_up":   sig.get("gamma_wall_up", 0.0),
                "gamma_wall_down": sig.get("gamma_wall_down", 0.0),
                "squeeze_potential": sig.get("squeeze_potential", False),
                "trend_quality_score": sig.get("trend_quality_score", 0.0),
                "consec_green":       sig.get("consec_green", 0),
                "consec_red":         sig.get("consec_red", 0),
                "hv20":               sig.get("hv20", 0.0),
                "hv5":                sig.get("hv5", 0.0),
                "hv_expanding":       sig.get("hv_expanding", False),
                "hv_contracting":     sig.get("hv_contracting", False),
                "key_support_1":      sig.get("key_support_1", 0.0),
                "key_support_2":      sig.get("key_support_2", 0.0),
                "key_resist_1":       sig.get("key_resist_1", 0.0),
                "key_resist_2":       sig.get("key_resist_2", 0.0),
                "near_key_support":   sig.get("near_key_support", False),
                "near_key_resist":    sig.get("near_key_resist", False),
                "fib_level_382":      sig.get("fib_level_382", 0.0),
                "fib_level_500":      sig.get("fib_level_500", 0.0),
                "fib_level_618":      sig.get("fib_level_618", 0.0),
                "fib_level_786":      sig.get("fib_level_786", 0.0),
                "fib_high_ref":       sig.get("fib_high_ref", 0.0),
                "fib_low_ref":        sig.get("fib_low_ref", 0.0),
                "short_float":        sig.get("short_float", 0.0),
                "short_ratio":        sig.get("short_ratio", 0.0),
                "high_short":         sig.get("high_short", False),
                "atm_iv":             sig.get("atm_iv", 0.0),
                "rs_sector":          round(sig.get("rs_sector", 0.0), 2),
                "rs252":              round(sig.get("rs252", 0.0), 2),
                "rs_rating":          sig.get("rs_rating", 50),
                "ema21_pullback":     sig.get("ema21_pullback", False),
                "ema21_touch":        sig.get("ema21_touch", False),
                "macd_bear_div":      sig.get("macd_bear_div", False),
                "macd_bull_div":      sig.get("macd_bull_div", False),
                "rsi_divergence":     sig.get("rsi_divergence", False),
                "rsi_bull_divergence": sig.get("rsi_bull_divergence", False),
                "fund_quality":        sig.get("fund_quality", 0),
                "earnings_growth":     sig.get("earnings_growth"),
                "revenue_growth":      sig.get("revenue_growth"),
                "forward_pe":          sig.get("forward_pe"),
                "profit_margin":       sig.get("profit_margin"),
                "roe":                 sig.get("roe"),
                "pocket_pivot":        sig.get("pocket_pivot", False),
                "htf":                 sig.get("htf", False),
                "htf_consec":          sig.get("htf_consec", 0),
                "trend_template":      sig.get("trend_template", 0),
                "tt_full":             sig.get("tt_full", False),
                "above_avwap_52wl":    sig.get("above_avwap_52wl", False),
                "avwap_52wl":          sig.get("avwap_52wl", 0.0),
                "avwap_dist_pct":      sig.get("avwap_dist_pct", 0.0),
                "vol_bearish_div":     sig.get("vol_bearish_div", False),
                "vol_bullish_div":     sig.get("vol_bullish_div", False),
                "w52_range_pos":      sig.get("w52_range_pos", 0.0),
                "expected_move_wk":   sig.get("expected_move_wk", 0.0),
                "expected_move_mo":   sig.get("expected_move_mo", 0.0),
                "expected_move_pct_wk": sig.get("expected_move_pct_wk", 0.0),
                "analyst_upgrade":    sig.get("analyst_upgrade", False),
                "analyst_rev_score":  sig.get("analyst_rev_score", 0),
                "analyst_buy_pct":    sig.get("analyst_buy_pct", 0.5),
                "analyst_net_rev":    sig.get("analyst_net_rev", 0),
                "analyst_price_tgt":  sig.get("analyst_price_tgt", 0.0),
                "analyst_upside_pct": sig.get("analyst_upside_pct", 0.0),
                # News headlines and catalyst info for position card
                "news_headlines":     _NEWS_VEL_CACHE.get(sym, ({}, 0))[0].get("headlines", [])
                                      if sym in _NEWS_VEL_CACHE else [],
                "catalyst_type":      _NEWS_VEL_CACHE.get(sym, ({}, 0))[0].get("catalyst_type", "none")
                                      if sym in _NEWS_VEL_CACHE else "none",
                "catalyst_urg":       _NEWS_VEL_CACHE.get(sym, ({}, 0))[0].get("catalyst_urg", 0)
                                      if sym in _NEWS_VEL_CACHE else 0,
                "catalyst_dir":       _NEWS_VEL_CACHE.get(sym, ({}, 0))[0].get("catalyst_dir", "none")
                                      if sym in _NEWS_VEL_CACHE else "none",
                # Intraday range data for range-position indicator on position cards
                "day_high":           round(sig.get("day_high", 0.0) or 0.0, 2),
                "day_low":            round(sig.get("day_low", 0.0)  or 0.0, 2),
                "day_open":           round(sig.get("day_open", 0.0) or 0.0, 2),
            }

        _pos_list_raw = []
        for p in curr:
            _sym      = p.get("symbol", "")
            _entry    = float(p.get("avg_entry_price", 0))
            _cur      = float(p.get("current_price", 0))
            _pnl_pct  = float(p.get("unrealized_plpc", 0)) * 100
            _stop     = round(_entry * (1 - STOP_LOSS_PCT), 2)
            # ATR-based dynamic target: 4.5× ATR from entry, falls back to fixed PROFIT_TARGET_PCT
            _atr_entry_pk = peaks.get(_sym, {}).get("atr_at_entry", 0) if isinstance(peaks.get(_sym), dict) else 0
            if _atr_entry_pk and _entry > 0:
                _atr_tgt_pct = min(0.22, max(PROFIT_TARGET_PCT, _atr_entry_pk / _entry * 4.5))
            else:
                _atr_tgt_pct = PROFIT_TARGET_PCT
            _tgt      = round(_entry * (1 + _atr_tgt_pct), 2)
            _init_risk_pct = (_entry - _stop) / _entry * 100 if _entry > 0 else STOP_LOSS_PCT * 100
            _r_multiple = round(_pnl_pct / _init_risk_pct, 2) if _init_risk_pct > 0 else 0.0
            # Days held from peaks.json entry timestamp
            _days_held = 0
            try:
                _pk = peaks.get(_sym, {})
                if isinstance(_pk, dict) and _pk.get("time"):
                    _et = datetime.fromisoformat(_pk["time"].replace("Z", "+00:00"))
                    _days_held = max(0, int((now_utc - _et).total_seconds() / 86400))
            except Exception:
                pass
            _pos_list_raw.append({
                "ticker":     _sym,
                "side":       "long" if float(p.get("qty", 0)) > 0 else "short",
                "qty":        abs(float(p.get("qty", 0))),
                "cost":       _entry,
                "price":      _cur,
                "pnl_pct":    _pnl_pct,
                "pnl_usd":    float(p.get("unrealized_pl",  0)),
                "market_val": float(p.get("market_value",   0)),
                "stop_price": _stop,
                "target_price": _tgt,
                "peak_price": peaks.get(_sym, {}).get("peak", 0) if isinstance(peaks.get(_sym), dict) else 0,
                "earnings_days": get_earnings_days(_sym),
                "r_multiple":  _r_multiple,
                "days_held":   _days_held,
                "live_signals": _pos_signals(_sym),
                "score_history": peaks.get(_sym, {}).get("score_history", []) if isinstance(peaks.get(_sym), dict) else [],
                "pnl_history":   peaks.get(_sym, {}).get("pnl_history", []) if isinstance(peaks.get(_sym), dict) else [],
                "mae":           peaks.get(_sym, {}).get("mae", 0.0) if isinstance(peaks.get(_sym), dict) else 0.0,
                "mfe":           peaks.get(_sym, {}).get("mfe", 0.0) if isinstance(peaks.get(_sym), dict) else 0.0,
                "atr_at_entry":  peaks.get(_sym, {}).get("atr_at_entry", 0.0) if isinstance(peaks.get(_sym), dict) else 0.0,
                "scale_in_pending": peaks.get(_sym, {}).get("scale_in_pending", False) if isinstance(peaks.get(_sym), dict) else False,
                "scale_in_notional": peaks.get(_sym, {}).get("scale_in_notional", 0.0) if isinstance(peaks.get(_sym), dict) else 0.0,
                "entry_thesis":  peaks.get(_sym, {}).get("entry_thesis", "") if isinstance(peaks.get(_sym), dict) else "",
                "entry_score":   peaks.get(_sym, {}).get("entry_score", 0) if isinstance(peaks.get(_sym), dict) else 0,
                "grade":         momentum_grade(live.get(_sym, {}), score(_sym, live.get(_sym, {}))) if live.get(_sym) else "?",
                "rs_rating":     live.get(_sym, {}).get("rs_rating", 50) if live.get(_sym) else 50,
                "rs252":         round(live.get(_sym, {}).get("rs252", 0.0), 2) if live.get(_sym) else 0.0,
                "sector":        SECTOR_MAP.get(_sym, "other"),
                "rs_sector":     round(live.get(_sym, {}).get("rs_sector", 0.0), 2) if live.get(_sym) else 0.0,
                # Kelly criterion optimal size for this position
                "kelly_size_pct": round(max(0, min(10.0, (
                    (win_rate * max(0.5, min(5.0, _payoff_ratio)) - (1 - win_rate)) /
                    max(0.01, max(0.5, min(5.0, _payoff_ratio))) * 50
                ))), 1) if win_rate > 0.4 and _payoff_ratio > 0.5 else 0.0,
            })
        tlog["positions"] = _pos_list_raw
        # Post-process: compute active trailing stop, refine EM with ATM IV, ATR position sizing
        import math as _math2
        for _pos in tlog.get("positions", []):
            _ls = _pos.get("live_signals", {})
            _pr = _pos.get("price", 0) or 0
            _atr = _ls.get("atr") or 0
            # ATR-based position sizing: shares for 1% and 2% account risk
            if _atr > 0 and portfolio_val > 0:
                _pos["atr_size_1pct"] = max(1, int(portfolio_val * 0.01 / _atr))
                _pos["atr_size_2pct"] = max(1, int(portfolio_val * 0.02 / _atr))
            else:
                _pos["atr_size_1pct"] = 0
                _pos["atr_size_2pct"] = 0
            # Override expected move with ATM IV if available (more accurate than HV20)
            _atm_iv = _ls.get("atm_iv") or 0
            if _atm_iv > 0 and _pr > 0:
                try:
                    _em_wk = round(_pr * (_atm_iv / 100) * _math2.sqrt(5 / 252), 2)
                    _em_mo = round(_pr * (_atm_iv / 100) * _math2.sqrt(21 / 252), 2)
                    _ls["expected_move_wk"]     = _em_wk
                    _ls["expected_move_mo"]     = _em_mo
                    _ls["expected_move_pct_wk"] = round(_em_wk / _pr * 100, 2)
                except Exception:
                    pass
            # Anchored VWAP from entry date — key institutional hold/exit level
            _pos["avwap_entry"] = 0.0
            _pos["avwap_entry_pct"] = 0.0
            try:
                _pk_data  = peaks.get(_pos["ticker"], {})
                _entry_ts = (_pk_data.get("time") or "") if isinstance(_pk_data, dict) else ""
                _entry_pr = _pos.get("cost", 0) or 0
                _cur_pr   = _pos.get("price", 0) or 0
                if _entry_ts and _entry_pr > 0 and _cur_pr > 0:
                    _entry_date_str = _entry_ts[:10]  # YYYY-MM-DD
                    _av_hist = yf.download(
                        _pos["ticker"], start=_entry_date_str,
                        progress=False, auto_adjust=True
                    )
                    if not _av_hist.empty and "Volume" in _av_hist.columns:
                        _av_col = "Close" if "Close" in _av_hist.columns else _av_hist.columns[-1]
                        _av_h = _av_hist["High"] if "High" in _av_hist.columns else _av_hist[_av_col]
                        _av_l = _av_hist["Low"]  if "Low"  in _av_hist.columns else _av_hist[_av_col]
                        _av_c = _av_hist[_av_col]
                        _av_v = _av_hist["Volume"]
                        _tp   = (_av_h + _av_l + _av_c) / 3
                        _vsum = float(_av_v.sum())
                        if _vsum > 0:
                            _avwap_e = float((_tp * _av_v).sum() / _vsum)
                            _pos["avwap_entry"]     = round(_avwap_e, 2)
                            _pos["avwap_entry_pct"] = round((_cur_pr - _avwap_e) / _avwap_e * 100, 2)
            except Exception:
                pass

            # RS vs SPY since entry — measures alpha generation vs market
            _pos["spy_since_entry"] = 0.0
            _pos["rs_alpha"]        = 0.0
            try:
                _rsa_pk = peaks.get(_pos["ticker"], {})
                _rsa_ts = (_rsa_pk.get("time") or "") if isinstance(_rsa_pk, dict) else ""
                if _rsa_ts:
                    _rsa_entry_date = _rsa_ts[:10]  # YYYY-MM-DD
                    _spy_cache = _fetch_spy_perf()
                    _spy_dates  = _spy_cache.get("date_list", [])
                    _spy_closes = _spy_cache.get("closes", [])
                    if _spy_dates and _spy_closes and len(_spy_dates) == len(_spy_closes):
                        _rsa_idx = next((i for i, d in enumerate(_spy_dates) if d >= _rsa_entry_date), None)
                        if _rsa_idx is not None and _spy_closes[_rsa_idx] > 0:
                            _spy_entry_p = _spy_closes[_rsa_idx]
                            _spy_cur_p   = _spy_closes[-1]
                            _spy_ret     = round((_spy_cur_p - _spy_entry_p) / _spy_entry_p * 100, 2)
                            _pos["spy_since_entry"] = _spy_ret
                            _pos["rs_alpha"] = round((_pos.get("pnl_pct", 0) or 0) - _spy_ret, 2)
            except Exception:
                pass

            # Active dynamic trailing stop: mirrors the live trading logic
            # Shows the trader exactly what stop % is currently protecting this position
            _pnl = _pos.get("pnl_pct", 0) or 0
            _peak_p = _pos.get("peak_price", 0) or _pr
            if   _pnl >= 25: _dyn_trail = 1.5
            elif _pnl >= 20: _dyn_trail = 1.8
            elif _pnl >= 15: _dyn_trail = 2.0
            elif _pnl >= 10: _dyn_trail = 3.0
            elif _pnl >=  5: _dyn_trail = TRAILING_STOP_PCT * 100
            else:
                _atr_trail = _atr / _pr * 100 * 2.5 if (_atr > 0 and _pr > 0) else TRAILING_STOP_PCT * 100
                _dyn_trail = max(4.0, min(9.0, _atr_trail))
            # VIX adjustment: widen in high fear, tighten in calm
            _vix_pos = (tlog.get("regime") or {}).get("vix", 0) or 0
            if _vix_pos >= 30:
                _dyn_trail = min(_dyn_trail * 1.30, _dyn_trail + 1.5)
            elif _vix_pos >= 22:
                _dyn_trail = min(_dyn_trail * 1.15, _dyn_trail + 0.8)
            elif 0 < _vix_pos < 14:
                _dyn_trail = max(_dyn_trail * 0.90, _dyn_trail - 0.5)
            _trail_stop_price = round(_peak_p * (1 - _dyn_trail / 100), 2) if _peak_p > 0 else 0
            _dist_from_trail  = round((_pr - _trail_stop_price) / _pr * 100, 2) if (_trail_stop_price > 0 and _pr > 0) else 0
            _pos["dyn_trail_pct"]    = round(_dyn_trail, 1)
            _pos["trail_stop_price"] = _trail_stop_price
            _pos["dist_from_trail"]  = _dist_from_trail
            # Position recommendation: comprehensive rule-based action advice for dashboard
            # Priority order: urgent sell conditions first, then management signals
            _rec_action = "HOLD"
            _rec_reason = ""
            _ls_r = _pos.get("live_signals", {})
            _sh_r = [h["s"] for h in _pos.get("score_history", []) if isinstance(h.get("s"), (int, float))]
            _age_r = _pos.get("days_held", 0)
            _earn_r = _pos.get("earnings_days")
            _rm_r = _pos.get("r_multiple", 0) or 0
            # Highest priority: Earnings risk management
            if _earn_r is not None and 2 < _earn_r <= 5 and _pnl > 3:
                _rec_action = "REDUCE"; _rec_reason = f"earnings in {_earn_r}d — protect gains"
            # High priority: multiple bearish signals converging
            elif (_ls_r.get("macd_bear_div") and _ls_r.get("vol_bearish_div") and _pnl > 3):
                _rec_action = "REDUCE"; _rec_reason = "MACD div + vol distribution — exit signal"
            elif _ls_r.get("macd_bear_div") and _pnl > 5:
                _rec_action = "REDUCE"; _rec_reason = "MACD bearish div"
            elif _ls_r.get("vol_bearish_div") and _pnl >= 12:
                _rec_action = "REDUCE"; _rec_reason = "volume distribution at new high"
            elif _dist_from_trail < 1.5 and _trail_stop_price > 0:
                _rec_action = "TIGHTEN"; _rec_reason = f"near trail stop ({_dist_from_trail:.1f}%)"
            elif _pnl >= 15 and not _ls_r.get("supertrend_bull", True):
                _rec_action = "REDUCE"; _rec_reason = "supertrend reversed, locked gains"
            # R-multiple milestones: lock profits at 3R+
            elif _rm_r >= 3 and _pnl > 0:
                _rec_action = "LOCK"; _rec_reason = f"{_rm_r:.1f}R — exceptional gain, move stop to 2R"
            # Scale-in opportunity (EMA21 pullback on high-conviction setup)
            elif _ls_r.get("ema21_pullback") and _pos.get("scale_in_pending") and _pnl > 0:
                _rec_action = "SCALE IN"; _rec_reason = f"EMA21 pullback — add ${(_pos.get('scale_in_notional',0) or 0):.0f}"
            elif _ls_r.get("ema21_pullback") and _pnl > 0 and not _ls_r.get("hv_expanding"):
                _rec_action = "ADD"; _rec_reason = "EMA21 pullback re-entry"
            # Extend on high-conviction momentum setup with strong signals
            elif _pnl >= 20 and _ls_r.get("mtf_triple") and _ls_r.get("supertrend_bull", True):
                _rec_action = "EXTEND"; _rec_reason = "high-conviction, let run"
            elif _ls_r.get("pocket_pivot") and _pnl > 0 and not _ls_r.get("vol_bearish_div"):
                _rec_action = "EXTEND"; _rec_reason = "pocket pivot re-accumulation signal"
            # Stale position reviews
            elif _age_r >= 7 and _pnl < 1.5 and _pnl > -2:
                _rec_action = "REVIEW"; _rec_reason = "dead money 7d"
            elif len(_sh_r) >= 4 and _sh_r[0] - _sh_r[-1] >= 15:
                _rec_action = "REVIEW"; _rec_reason = "score degrading"
            _pos["recommendation"] = _rec_action
            _pos["rec_reason"]     = _rec_reason
            # Live score vs entry score delta — shows if conviction is rising or falling
            _live_sc = score(_pos["ticker"], live.get(_pos["ticker"], {}), regime_adj=regime_adj)
            _entry_sc = _pos.get("entry_score", 0) or 0
            _pos["live_score"]   = round(_live_sc)
            _pos["score_delta"]  = round(_live_sc - _entry_sc)
            # R:R ratio live: (distance to target) / (distance to trail stop)
            _pr2 = _pos.get("price", 0) or 0
            _tgt2 = _pos.get("target_price", 0) or 0
            _stp2 = _pos.get("trail_stop_price", 0) or _pos.get("stop_price", 0) or 0
            if _pr2 > 0 and _tgt2 > _pr2 and _stp2 > 0 and _stp2 < _pr2:
                _rr = round((_tgt2 - _pr2) / (_pr2 - _stp2), 2)
                _pos["live_rr"] = _rr
            else:
                _pos["live_rr"] = 0.0
            # Extended/Overextended detection (O'Neil rule: >25% above ideal pivot = extended)
            # Uses cup_pivot or pivot_r1 from live signals as the "proper buy point"
            _ls_ext = _pos.get("live_signals", {})
            _cup_piv = _ls_ext.get("cup_pivot") or 0
            _piv_r1  = _ls_ext.get("pivot_r1") or 0
            _base_piv = _cup_piv if _cup_piv > 0 else (_piv_r1 if _piv_r1 > 0 else 0)
            if _base_piv > 0 and _pr2 > 0:
                _ext_pct = round((_pr2 - _base_piv) / _base_piv * 100, 1)
                _pos["pivot_pct"] = _ext_pct
                _pos["extended"]  = _ext_pct > 25
                _pos["overextended"] = _ext_pct > 40
            else:
                # Fallback: use pnl_pct as proxy — >25% gain often means extended
                _pos["pivot_pct"] = _pnl
                _pos["extended"]  = _pnl > 25
                _pos["overextended"] = _pnl > 40
            # Exit Urgency Score (0-10): composite measure of how close to an exit trigger
            try:
                _eu = 0
                _eu_ls = _pos.get("live_signals", {})
                # Distance to trail stop (biggest factor)
                _eu_dist = _pos.get("dist_from_trail", 999) or 999
                if _eu_dist < 1.0:   _eu += 4
                elif _eu_dist < 2.5: _eu += 3
                elif _eu_dist < 4.0: _eu += 2
                elif _eu_dist < 6.0: _eu += 1
                # Score degradation
                _eu_sh = [h["s"] for h in _pos.get("score_history", []) if isinstance(h.get("s"), (int, float))]
                if len(_eu_sh) >= 3:
                    _eu_drop = _eu_sh[0] - _eu_sh[-1]
                    if _eu_drop >= 20: _eu += 3
                    elif _eu_drop >= 12: _eu += 2
                    elif _eu_drop >= 6: _eu += 1
                # Bearish technical signals
                if _eu_ls.get("macd_bear_div") and _eu_ls.get("vol_bearish_div"): _eu += 2
                elif _eu_ls.get("macd_bear_div"): _eu += 1
                elif _eu_ls.get("vol_bearish_div") and _pnl >= 10: _eu += 1
                # Earnings proximity
                _eu_earn = _pos.get("earnings_days")
                if _eu_earn is not None and _eu_earn <= 2 and _pnl > 3: _eu += 2
                elif _eu_earn is not None and _eu_earn <= 5 and _pnl > 5: _eu += 1
                # Consecutive red days
                if (_eu_ls.get("consec_red") or 0) >= 3: _eu += 1
                # Negative RS alpha vs SPY
                if (_pos.get("rs_alpha") or 0) < -3: _eu += 1
                # Overextended
                if _pos.get("overextended") and _pnl > 30: _eu += 1
                _pos["exit_urgency"] = min(10, _eu)
            except Exception:
                _pos["exit_urgency"] = 0
            # Position Health Score (0-10): composite momentum + safety measure
            try:
                _ph = 5  # start neutral
                _ph_ls = _pos.get("live_signals", {})
                # Momentum signals (positive)
                if _ph_ls.get("mtf_triple"):    _ph += 2
                elif _ph_ls.get("mtf_aligned"): _ph += 1
                if _ph_ls.get("supertrend_bull", True): _ph += 1
                if (_ph_ls.get("accum_score") or 0) >= 7: _ph += 1
                if (_pos.get("rs_alpha") or 0) > 3: _ph += 1
                # Degrade score
                if _pos["exit_urgency"] >= 6:   _ph -= 3
                elif _pos["exit_urgency"] >= 4: _ph -= 2
                elif _pos["exit_urgency"] >= 2: _ph -= 1
                if not _ph_ls.get("supertrend_bull", True): _ph -= 1
                if _ph_ls.get("mtf_conflict"): _ph -= 1
                _pos["pos_health"] = max(0, min(10, _ph))
            except Exception:
                _pos["pos_health"] = 5
    except Exception as e:
        logger.warning(f"Position snapshot failed: {e}")

    # Portfolio total $ risk (sum of max loss per position at current trail stop)
    try:
        _total_risk_usd = 0.0
        for _prp in tlog.get("positions", []):
            _prp_price = _prp.get("price", 0) or 0
            _prp_stop  = _prp.get("trail_stop_price", 0) or _prp.get("stop_price", 0) or 0
            _prp_mval  = _prp.get("market_val", 0) or 0
            if _prp_price > 0 and _prp_stop > 0 and _prp_mval > 0:
                _risk_pct = max(0, (_prp_price - _prp_stop) / _prp_price)
                _total_risk_usd += _prp_mval * _risk_pct
        tlog["total_risk_usd"] = round(_total_risk_usd, 2)
        tlog["total_risk_pct"] = round(_total_risk_usd / max(portfolio_val, 1) * 100, 2)
    except Exception:
        tlog["total_risk_usd"] = 0.0
        tlog["total_risk_pct"] = 0.0

    # Rolling 20-trade win rate and portfolio correlation heuristic
    try:
        _closed_all = [t for t in tlog.get("trades", []) if t.get("action") in ("SELL","COVER") and t.get("pnl_pct") is not None]
        _recent20 = sorted(_closed_all, key=lambda t: t.get("time",""))[-20:]
        if _recent20:
            _r20_wins = sum(1 for t in _recent20 if t["pnl_pct"] > 0)
            _r20_wr   = round(_r20_wins / len(_recent20) * 100, 1)
            _r20_avg  = round(sum(t["pnl_pct"] for t in _recent20) / len(_recent20), 2)
        else:
            _r20_wr = None; _r20_avg = None
        tlog["rolling_wr_20"]  = _r20_wr
        tlog["rolling_avg_20"] = _r20_avg
        # Portfolio sector concentration: count positions per sector
        _pos_sectors = {}
        for _cp in tlog.get("positions", []):
            _csec = live.get(_cp["ticker"], {}).get("sector_key", "unknown")
            _pos_sectors[_csec] = _pos_sectors.get(_csec, 0) + 1
        _max_conc_sector = max(_pos_sectors.values()) if _pos_sectors else 0
        _max_conc_name   = max(_pos_sectors, key=_pos_sectors.get) if _pos_sectors else ""
        tlog["sector_concentration"] = {
            "max_count": _max_conc_sector,
            "max_sector": _max_conc_name,
            "distribution": _pos_sectors,
            "alert": _max_conc_sector >= 3,  # 3+ positions in same sector = concentration risk
        }
    except Exception as _rstats_e:
        logger.debug(f"Rolling stats: {_rstats_e}")

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
        # Recent 10 closed trades for quick dashboard display — enriched with entry context
        _recent_closed = sorted(_closed, key=lambda t: t.get("time", ""), reverse=True)[:10]
        # Build quick BUY lookup for enrichment (ticker → sorted BUYs descending by time)
        _jl_buy_idx: dict = {}
        for _jb in tlog.get("trades", []):
            if _jb.get("action") == "BUY":
                _jl_buy_idx.setdefault(_jb.get("ticker",""), []).append(_jb)
        for _jbl in _jl_buy_idx.values():
            _jbl.sort(key=lambda x: x.get("time",""), reverse=True)
        def _entry_for(ticker, sell_time):
            """Find most recent matching BUY entry before this SELL."""
            for _jb in _jl_buy_idx.get(ticker, []):
                if (_jb.get("time","") or "") <= (sell_time or ""):
                    return _jb
            return {}
        _enriched_closed = []
        for t in _recent_closed:
            _buy = _entry_for(t.get("ticker",""), t.get("time",""))
            _entry_sc = _buy.get("score") or 0
            _entry_t  = _buy.get("time","")
            _hold_hrs = 0
            try:
                if _entry_t and t.get("time"):
                    _dt_e = datetime.fromisoformat(_entry_t.replace("Z","+00:00"))
                    _dt_s = datetime.fromisoformat(t["time"].replace("Z","+00:00"))
                    _hold_hrs = round((_dt_s - _dt_e).total_seconds() / 3600, 1)
            except Exception:
                pass
            _entry_h = None
            try:
                if _entry_t:
                    _entry_h = datetime.fromisoformat(_entry_t.replace("Z","+00:00")).hour
            except Exception:
                pass
            _enriched_closed.append({
                "ticker":       t.get("ticker",""),
                "pnl_pct":      round(t["pnl_pct"], 2),
                "reason":       (t.get("reason","") or "")[:60],
                "time":         t.get("time",""),
                "entry_score":  _entry_sc,
                "hold_hrs":     _hold_hrs,
                "entry_hour":   _entry_h,
                "sector":       t.get("sector",""),
                "regime":       t.get("regime",""),
                "signals":      _buy.get("entry_signals", [])[:8],
            })
        tlog["recent_closed_trades"] = _enriched_closed
        # Win/loss streak from most recent trades
        _streak_len = 0
        _streak_type = "none"
        for _st in _recent_closed:
            _is_win = _st["pnl_pct"] >= 0
            if _streak_len == 0:
                _streak_type = "win" if _is_win else "loss"
                _streak_len = 1
            elif (_streak_type == "win" and _is_win) or (_streak_type == "loss" and not _is_win):
                _streak_len += 1
            else:
                break
        _streak_bonus = "hot" if _streak_type == "win" and _streak_len >= 5 else "warm" if _streak_type == "win" and _streak_len >= 3 else "cold" if _streak_type == "loss" and _streak_len >= 3 else "normal"
        tlog["trade_streak"] = {"type": _streak_type, "count": _streak_len, "mode": _streak_bonus}
    except Exception:
        pass
    tlog["signal_analytics"] = _signal_analytics

    # Build a BUY lookup index for O(1) lookups per SELL trade
    # Maps ticker → list of (time, score, entry_hour, entry_signals) sorted by time desc
    _buy_idx: dict = {}
    for _bx in tlog.get("trades", []):
        if _bx.get("action") == "BUY":
            _bxt = _bx.get("ticker","")
            try: _bxh = datetime.fromisoformat((_bx.get("time","")).replace("Z","+00:00")).hour
            except Exception: _bxh = None
            _buy_idx.setdefault(_bxt, []).append({
                "time": _bx.get("time",""),
                "score": _bx.get("score") or 0,
                "hour": _bxh,
                "signals": _bx.get("entry_signals", []),
            })
    for _bxl in _buy_idx.values():
        _bxl.sort(key=lambda x: x["time"], reverse=True)

    def _find_entry(ticker, sell_time):
        """O(n_buys_for_ticker) lookup using pre-built index."""
        for _bv in _buy_idx.get(ticker, []):
            if (_bv["time"] or "") <= (sell_time or ""):
                return _bv
        return {}

    # Score bucket accuracy: entry score range → win rate (shows if scoring is predictive)
    try:
        _score_buckets = {
            "90-100": {"trades":0,"wins":0,"pnl":0.0},
            "80-89":  {"trades":0,"wins":0,"pnl":0.0},
            "70-79":  {"trades":0,"wins":0,"pnl":0.0},
            "60-69":  {"trades":0,"wins":0,"pnl":0.0},
            "50-59":  {"trades":0,"wins":0,"pnl":0.0},
            "<50":    {"trades":0,"wins":0,"pnl":0.0},
        }
        for _bt in _closed:
            _buy_e  = _find_entry(_bt.get("ticker",""), _bt.get("time",""))
            _sc_entry = _buy_e.get("score", 0) or 0
            if   _sc_entry >= 90: _bk = "90-100"
            elif _sc_entry >= 80: _bk = "80-89"
            elif _sc_entry >= 70: _bk = "70-79"
            elif _sc_entry >= 60: _bk = "60-69"
            elif _sc_entry >= 50: _bk = "50-59"
            else:                 _bk = "<50"
            _score_buckets[_bk]["trades"] += 1
            _score_buckets[_bk]["pnl"]    = round(_score_buckets[_bk]["pnl"] + _bt["pnl_pct"], 2)
            if _bt["pnl_pct"] > 0:
                _score_buckets[_bk]["wins"] += 1
        for _bv in _score_buckets.values():
            if _bv["trades"] > 0:
                _bv["wr"]      = round(_bv["wins"] / _bv["trades"] * 100, 1)
                _bv["avg_pnl"] = round(_bv["pnl"] / _bv["trades"], 2)
        tlog["score_bucket_perf"] = _score_buckets
    except Exception:
        tlog.setdefault("score_bucket_perf", {})

    # Hour-of-day entry win rate: tracks which market hours produce the best outcomes
    try:
        _hour_perf: dict = {}
        for _ht in _closed:
            _buy_e2 = _find_entry(_ht.get("ticker",""), _ht.get("time",""))
            _bh = _buy_e2.get("hour")
            if _bh is None:
                continue
            _hb = _hour_perf.setdefault(str(_bh), {"trades":0,"wins":0,"pnl":0.0})
            _hb["trades"] += 1
            _hb["pnl"]     = round(_hb["pnl"] + _ht["pnl_pct"], 2)
            if _ht["pnl_pct"] > 0:
                _hb["wins"] += 1
        for _hv in _hour_perf.values():
            if _hv["trades"] > 0:
                _hv["wr"]      = round(_hv["wins"] / _hv["trades"] * 100, 1)
                _hv["avg_pnl"] = round(_hv["pnl"] / _hv["trades"], 2)
        tlog["hour_win_rates"] = _hour_perf
    except Exception:
        tlog.setdefault("hour_win_rates", {})

    # Pattern accuracy tracker: which chart patterns have the best win rates in our system
    # Uses closed trade 'reason' strings which contain pattern tags from the buy logic.
    try:
        _pat_tags = {
            "BRKOUT":     "Breakout",
            "SQZ":        "Squeeze",
            "C&H":        "Cup&Handle",
            "VCP":        "VCP",
            "2-BTM":      "Double Bottom",
            "EMA21":      "EMA21 Pullback",
            "RVOL":       "Volume Surge",
            "52W":        "52W Breakout",
            "PP":         "Pocket Pivot",
            "HTF":        "High-Tight Flag",
            "TT":         "Trend Template",
            "HAMMER":     "Hammer",
            "BULL-ENG":   "Bull Engulf",
            "MACD":       "MACD Signal",
            "POC":        "POC Breakout",
        }
        _pat_acc: dict = {}
        for _pat_key, _pat_lbl in _pat_tags.items():
            _pat_trades = [t for t in _closed if _pat_key in (t.get("reason", "") or "")]
            if len(_pat_trades) >= 2:
                _pat_wins = [t for t in _pat_trades if t["pnl_pct"] > 0]
                _pat_acc[_pat_lbl] = {
                    "trades":  len(_pat_trades),
                    "wins":    len(_pat_wins),
                    "wr":      round(len(_pat_wins) / len(_pat_trades) * 100, 1),
                    "avg_pnl": round(sum(t["pnl_pct"] for t in _pat_trades) / len(_pat_trades), 2),
                }
        # Sort by win rate descending
        tlog["pattern_accuracy"] = dict(sorted(_pat_acc.items(), key=lambda x: -x[1]["wr"]))
    except Exception:
        tlog.setdefault("pattern_accuracy", {})

    # Gap statistics: track performance of gap-up entries vs. non-gap entries
    # Helps calibrate whether to buy gaps or wait for pullbacks.
    try:
        _gap_counts = {"up": {"total":0,"wins":0,"pnl":0.0}, "down": {"total":0,"wins":0,"pnl":0.0}}
        _non_gap    = {"total":0,"wins":0,"pnl":0.0}
        for _gs in _closed:
            _gr = (_gs.get("reason") or "").lower()
            _gp = _gs.get("pnl_pct", 0) or 0
            _is_win = _gp > 0
            if "pm+" in _gr or "gap" in _gr:
                _gap_counts["up"]["total"] += 1
                if _is_win: _gap_counts["up"]["wins"] += 1
                _gap_counts["up"]["pnl"] = round(_gap_counts["up"]["pnl"] + _gp, 2)
            elif "gap_down" in _gr:
                _gap_counts["down"]["total"] += 1
                if _is_win: _gap_counts["down"]["wins"] += 1
                _gap_counts["down"]["pnl"] = round(_gap_counts["down"]["pnl"] + _gp, 2)
            else:
                _non_gap["total"] += 1
                if _is_win: _non_gap["wins"] += 1
                _non_gap["pnl"] = round(_non_gap["pnl"] + _gp, 2)
        for _gc in list(_gap_counts.values()) + [_non_gap]:
            if _gc["total"] > 0:
                _gc["win_rate"] = round(_gc["wins"] / _gc["total"] * 100, 1)
                _gc["avg_pnl"]  = round(_gc["pnl"] / _gc["total"], 2)
        tlog["gap_stats"] = {"gap_up": _gap_counts["up"], "gap_down": _gap_counts["down"], "no_gap": _non_gap}
    except Exception:
        tlog.setdefault("gap_stats", {})

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

    # Max drawdown, Calmar ratio, Sortino ratio from portfolio history
    _max_dd = 0.0
    _calmar_ratio = None
    _sortino_ratio = None
    try:
        _hist_v = [h["v"] for h in tlog.get("perf_history", []) if isinstance(h.get("v"), (int, float)) and h["v"] > 0]
        if len(_hist_v) >= 2:
            _peak_running = _hist_v[0]
            for v in _hist_v:
                _peak_running = max(_peak_running, v)
                _dd = (_peak_running - v) / _peak_running * 100
                _max_dd = max(_max_dd, _dd)
            # Calmar = annualized return / max drawdown (risk-adjusted return quality)
            _total_ret_pct = (_hist_v[-1] - _hist_v[0]) / _hist_v[0] * 100
            _n_snapshots = len(_hist_v)
            # Rough annualization: assume ~26 snapshots/day in market hours (5min intervals)
            _days_approx = max(1, _n_snapshots / 26)
            _ann_ret_pct = _total_ret_pct / _days_approx * 252
            if _max_dd > 0.1:
                _calmar_ratio = round(_ann_ret_pct / _max_dd, 2)
    except Exception:
        pass

    # Sortino ratio: penalizes only downside volatility (superior to Sharpe for traders)
    try:
        if len(_closed) >= 5:
            import statistics
            _pnls = [t["pnl_pct"] for t in _closed]
            _avg_pnl_s = statistics.mean(_pnls)
            _downside = [p for p in _pnls if p < 0]
            _downside_std = statistics.stdev(_downside) if len(_downside) >= 2 else 1.0
            if _downside_std > 0:
                _sortino_ratio = round(_avg_pnl_s / _downside_std, 2)
    except Exception:
        pass

    # Risk of Ruin: probability of losing 30% of capital given win rate and payoff ratio
    # Formula (Ralph Vince): RoR = ((1-edge) / (1+edge))^N where edge = W*payoff - (1-W)
    _risk_of_ruin = None
    try:
        if win_rate > 0.3 and _payoff_ratio > 0.5 and len(_closed) >= 5:
            import math as _ror_math
            _W = win_rate; _R = _payoff_ratio
            _edge = (_W * _R - (1 - _W)) / _R  # edge per trade (0-1 scale)
            if _edge > 0 and _edge < 1:
                # Ruin = 30% drawdown from current, sizing = 2% avg risk per trade
                _risk_pct   = 0.30   # ruin threshold
                _bet_frac   = 0.02   # avg % at risk per trade
                _N_to_ruin  = _risk_pct / _bet_frac  # trades needed to reach ruin if all lose
                _ror_per_trade = ((1 - _edge) / (1 + _edge))  # O'Shaughnessy formula
                _risk_of_ruin  = round(_ror_per_trade ** _N_to_ruin * 100, 2)  # as percent
    except Exception:
        pass

    tlog["last_updated"]    = now_utc.isoformat()
    tlog["portfolio_value"] = portfolio_val
    tlog["buying_power"]    = round(buying_power, 2)
    tlog["regime"]          = regime
    tlog["status"]          = "ok"
    # Rolling regime history — last 48 snapshots for timeline sparkline display
    _rh_list = tlog.get("regime_history", [])
    _rh_list.append({"r": regime, "t": now_utc.isoformat()})
    tlog["regime_history"] = _rh_list[-48:]
    tlog["macro_day"]       = macro_day
    tlog["open_positions"]  = len(tlog.get("positions", []))
    tlog["scan_universe"]   = len(candidates)
    tlog["market_open"]     = market_open
    tlog["minutes_since_open"]  = _minutes_since_open if market_open else None
    tlog["minutes_to_close"]    = _minutes_to_close   if market_open else None
    # Market timing quality: 0=avoid, 1=neutral, 2=good, 3=prime
    _timing_q = 0
    if market_open and not _open_guard and not _close_guard:
        if 180 <= _minutes_since_open <= 225:    _timing_q = 3  # power hour
        elif 30  <= _minutes_since_open <= 90:   _timing_q = 3  # morning sweet spot
        elif 90  <= _minutes_since_open <= 180:  _timing_q = 1  # lunch lull
        else:                                    _timing_q = 2  # normal
    tlog["timing_quality"]  = _timing_q
    tlog["day_type"]        = _day_type
    tlog["day_efficiency"]  = day_type_info.get("efficiency", 0.5)
    tlog["day_range_ratio"] = day_type_info.get("range_ratio", 1.0)
    tlog["day_opening_bias"]= day_type_info.get("opening_bias", "flat")
    tlog["strategy_hint"]   = day_type_info.get("strategy_hint", "neutral")
    tlog["drawdown_pct"]      = round(drawdown_pct, 2)
    tlog["drawdown_halt"]     = _drawdown_halt
    tlog["regime_max_pos"]    = _regime_max
    tlog["spy_consec_decline"]= _spy_consec_decline
    tlog["spy_tape_score_adj"]= _spy_tape_score_adj
    tlog["win_rate"]        = round(win_rate, 3)
    tlog["portfolio_peak"]  = round(_peak_port, 2)
    tlog["market_breadth"]  = breadth
    # Rolling win rate history (last 50 snapshots, taken each cycle)
    try:
        _closed_all = [t for t in tlog.get("trades", []) if t.get("action") in ("SELL","SELL_HALF","COVER") and t.get("pnl_pct") is not None]
        _wr_rolling = {}
        for _n in (10, 20, 50):
            _n_slice = _closed_all[:_n]
            _wr_rolling[f"wr{_n}"] = round(sum(1 for t in _n_slice if t["pnl_pct"] > 0) / max(len(_n_slice), 1) * 100, 1) if _n_slice else None
        tlog["rolling_win_rates"] = _wr_rolling
        # Append to rolling history for sparkline
        _rwh = tlog.get("rolling_wr_history", [])
        _rwh.append({"t": now_utc.isoformat(), "wr10": _wr_rolling.get("wr10"), "wr20": _wr_rolling.get("wr20")})
        tlog["rolling_wr_history"] = _rwh[-48:]  # 48 cycles = 4h at 5min intervals
    except Exception:
        pass
    tlog["profit_factor"]   = _profit_factor
    tlog["avg_win_pct"]     = _avg_win
    tlog["avg_loss_pct"]    = _avg_loss
    tlog["portfolio_heat"]  = round(_portfolio_heat, 2)
    tlog["sector_rotation"]    = sector_adjs   # {sector: adj_score} for dashboard heatmap
    tlog["sector_etf_trends"]  = sector_etf_trends  # {sector: {bullish, chg5d, chg1d, above_ema20}}
    # Store full sector rotation detail (1d, 5d, 20d, 63d per sector) for heatmap
    try:
        import builtins as _bt
        tlog["sector_rotation_detail"] = getattr(_bt, "_SECTOR_ROTATION_DETAIL", {})
    except Exception:
        tlog.setdefault("sector_rotation_detail", {})
    tlog["portfolio_beta"]     = _port_beta_est      # estimated portfolio beta
    if _gap_data:  # only update if we ran the gap scanner this cycle
        tlog["premarket_gaps"] = _gap_data

    # Portfolio concentration and correlation risk analysis
    try:
        _pos_list = tlog.get("positions", [])
        _held_syms = [p["ticker"] for p in _pos_list if p.get("ticker")]
        _sec_buckets: dict = {}
        for _p in _pos_list:
            _ps = live.get(_p.get("ticker",""), {}).get("sector", SECTOR_MAP.get(_p.get("ticker",""), "other"))
            _sec_buckets[_ps] = _sec_buckets.get(_ps, 0) + 1
        _max_sector_conc = max(_sec_buckets.values()) if _sec_buckets else 0
        _dominant_sector = max(_sec_buckets, key=_sec_buckets.get) if _sec_buckets else None
        _port_conc_risk = "HIGH" if _max_sector_conc >= 3 else ("MEDIUM" if _max_sector_conc == 2 else "LOW")
        tlog["portfolio_concentration"] = {
            "sector_buckets":   _sec_buckets,
            "dominant_sector":  _dominant_sector,
            "max_sector_count": _max_sector_conc,
            "risk_level":       _port_conc_risk,
            "position_count":   len(_held_syms),
        }
    except Exception:
        tlog["portfolio_concentration"] = {}

    # Portfolio correlation matrix: pairwise 20-day price correlations for held positions
    # High correlation (>0.85) = positions move in lockstep = hidden concentration risk
    try:
        _held_tks = [p["ticker"] for p in tlog.get("positions", []) if p.get("ticker")]
        if len(_held_tks) >= 2:
            _cr_raw = yf.download(_held_tks, period="30d", interval="1d",
                                   auto_adjust=True, progress=False,
                                   group_by="ticker" if len(_held_tks) > 1 else "column")
            _ret_map: dict = {}
            for _ct in _held_tks:
                try:
                    if len(_held_tks) > 1:
                        _cl = list(_cr_raw["Close"][_ct].dropna())
                    else:
                        _cl = list(_cr_raw["Close"].dropna())
                    if len(_cl) >= 10:
                        _ret_map[_ct] = [(_cl[i]-_cl[i-1])/_cl[i-1] for i in range(1, len(_cl))]
                except Exception:
                    pass
            _cm: dict = {}
            _hcp: list = []
            _valid = list(_ret_map.keys())
            for _ii in range(len(_valid)):
                for _jj in range(_ii+1, len(_valid)):
                    _t1, _t2 = _valid[_ii], _valid[_jj]
                    _a, _b = _ret_map[_t1], _ret_map[_t2]
                    _n = min(len(_a), len(_b))
                    if _n < 8: continue
                    _a, _b = _a[-_n:], _b[-_n:]
                    _m1 = sum(_a) / _n;  _m2 = sum(_b) / _n
                    _cv = sum((_a[k]-_m1)*(_b[k]-_m2) for k in range(_n))
                    _s1 = (sum((_a[k]-_m1)**2 for k in range(_n))) ** 0.5
                    _s2 = (sum((_b[k]-_m2)**2 for k in range(_n))) ** 0.5
                    _r  = round(_cv / max(_s1 * _s2, 1e-12), 3)
                    _cm[f"{_t1}/{_t2}"] = _r
                    if _r > 0.80:
                        _hcp.append({"pair": f"{_t1}/{_t2}", "corr": _r})
            tlog["portfolio_correlation"] = {
                "matrix":          _cm,
                "high_corr_pairs": sorted(_hcp, key=lambda x: -x["corr"]),
            }
            if _hcp:
                _hcp_str = " | ".join(p["pair"] + "=" + str(round(p["corr"], 2)) for p in _hcp)
                logger.info(f"High-correlation pairs (>0.80): {_hcp_str}")
        else:
            tlog["portfolio_correlation"] = {}
    except Exception as _ce:
        logger.debug(f"Portfolio correlation skipped: {_ce}")
        tlog["portfolio_correlation"] = {}

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
    tlog["sortino_ratio"]      = _sortino_ratio
    tlog["calmar_ratio"]       = _calmar_ratio
    tlog["max_drawdown"]       = round(_max_dd, 2)
    tlog["risk_of_ruin"]       = _risk_of_ruin
    # Portfolio VaR and daily volatility (parametric from perf_history returns)
    try:
        _ph_vals = [h["v"] for h in tlog.get("perf_history", []) if isinstance(h.get("v"), (int, float)) and h["v"] > 0]
        if len(_ph_vals) >= 20:
            # Daily returns from 26-snapshot (~1 day) windows — use last 60 days = 1560 snapshots
            _snap_per_day = 26
            _ph_rets = []
            for _pi in range(_snap_per_day, len(_ph_vals), _snap_per_day):
                _pr_start = _ph_vals[_pi - _snap_per_day]
                _pr_end   = _ph_vals[_pi]
                if _pr_start > 0:
                    _ph_rets.append((_pr_end - _pr_start) / _pr_start)
            if len(_ph_rets) >= 5:
                import statistics as _pf_stat
                _port_vol_daily = _pf_stat.stdev(_ph_rets) if len(_ph_rets) >= 2 else abs(_ph_rets[0])
                # Parametric VaR at 95% confidence (1-day horizon)
                _var_95 = round(portfolio_val * _port_vol_daily * 1.645, 2)
                # Expected Shortfall (CVaR): expected loss beyond VaR
                _sorted_rets = sorted(_ph_rets)
                _tail_cutoff = max(1, int(len(_sorted_rets) * 0.05))
                _cvar_ret = abs(sum(_sorted_rets[:_tail_cutoff]) / _tail_cutoff) if _tail_cutoff > 0 else _port_vol_daily * 2.0
                _cvar_95  = round(portfolio_val * _cvar_ret, 2)
                tlog["port_vol_daily_pct"]  = round(_port_vol_daily * 100, 2)
                tlog["var_95_usd"]          = _var_95
                tlog["cvar_95_usd"]         = _cvar_95
            else:
                tlog.setdefault("port_vol_daily_pct", None)
        else:
            tlog.setdefault("port_vol_daily_pct", None)
    except Exception:
        tlog.setdefault("port_vol_daily_pct", None)
    try:
        tlog["effective_min_score"] = _eff_min_score
    except NameError:
        tlog["effective_min_score"] = MIN_BUY_SCORE
    try:
        tlog["loss_cooldown_tickers"] = sorted(_loss_cooldown)
    except NameError:
        tlog.setdefault("loss_cooldown_tickers", [])

    # ── SELF-TUNING LEARNING ENGINE ──────────────────────────────────────────
    # Analyzes own trade history to continuously improve decision thresholds,
    # signal weights, and position sizing. Stored as tlog["bot_learned_params"]
    # so the next cycle can load and apply them automatically.
    try:
        _prev_learned = tlog.get("bot_learned_params", {})
        _learn_log    = []   # human-readable log of what the bot learned this cycle

        # ── 1. Recent win rate → score threshold adjustment ──────────────
        _recent_20 = [t for t in _closed[-20:]]  # last 20 closed trades
        _r20_wr = sum(1 for t in _recent_20 if t.get("pnl_pct", 0) > 0) / max(len(_recent_20), 1)
        _r20_avg_pnl = sum(t.get("pnl_pct", 0) for t in _recent_20) / max(len(_recent_20), 1)
        _base_score_adj = _prev_learned.get("base_score_adj", 0)

        if len(_recent_20) >= 10:
            if _r20_wr >= 0.68 and _r20_avg_pnl >= 1.5:
                # Winning consistently → be slightly more aggressive (lower bar by 1)
                _base_score_adj = max(-5, _base_score_adj - 1)
                _learn_log.append(f"Win rate {_r20_wr:.0%} on last {len(_recent_20)} trades — easing entry threshold by 1pt (adj={_base_score_adj:+d})")
            elif _r20_wr <= 0.40 or _r20_avg_pnl <= -1.5:
                # Losing streak → tighten up (raise bar by 2)
                _base_score_adj = min(10, _base_score_adj + 2)
                _learn_log.append(f"Win rate {_r20_wr:.0%} on last {len(_recent_20)} trades — raising entry threshold by 2pt (adj={_base_score_adj:+d})")
            elif 0.50 <= _r20_wr <= 0.58 and _base_score_adj > 0:
                # Recovering — slowly ease back toward default
                _base_score_adj = max(0, _base_score_adj - 1)
                _learn_log.append(f"Win rate improving {_r20_wr:.0%} — relaxing threshold by 1pt (adj={_base_score_adj:+d})")
        _base_score_adj = max(-5, min(12, _base_score_adj))  # hard bounds

        # ── 2. Score bucket analysis → optimal minimum score ────────────
        _bucket_perf = tlog.get("score_bucket_perf", {})
        _optimal_min = MIN_BUY_SCORE  # default
        _positive_buckets = []
        _bucket_insights = []
        for _bkt, _bkd in sorted(_bucket_perf.items()):
            if _bkd.get("trades", 0) >= 4:
                _bkwr = _bkd.get("wr", 0)
                _bkavg = _bkd.get("avg_pnl", 0)
                if _bkwr >= 55 and _bkavg > 0:
                    _positive_buckets.append(_bkt)
                if _bkwr < 40 and _bkd.get("trades", 0) >= 5:
                    _bucket_insights.append(f"Score {_bkt}: {_bkwr}% WR — underperforming, avoid")
                elif _bkwr >= 65 and _bkd.get("trades", 0) >= 5:
                    _bucket_insights.append(f"Score {_bkt}: {_bkwr}% WR — high accuracy zone")

        # ── 3. Signal quality ranking ────────────────────────────────────
        _sig_wr_data = tlog.get("signal_win_rates", {})
        _elite_signals = [k for k, v in _sig_wr_data.items() if v.get("win_rate", 0) >= 65 and v.get("total", 0) >= 5]
        _weak_signals  = [k for k, v in _sig_wr_data.items() if v.get("win_rate", 0) <= 38 and v.get("total", 0) >= 5]
        if _elite_signals:
            _learn_log.append(f"High-accuracy signals: {', '.join(_elite_signals[:5])}")
        if _weak_signals:
            _learn_log.append(f"Underperforming signals (deprioritized): {', '.join(_weak_signals[:5])}")

        # ── 4. Best trading hours from win rate data ─────────────────────
        _hwr = tlog.get("hour_win_rates", {})
        _best_hours  = sorted([h for h, d in _hwr.items() if d.get("wr", 0) >= 60 and d.get("trades", 0) >= 4], key=lambda h: -_hwr[h]["wr"])
        _worst_hours = sorted([h for h, d in _hwr.items() if d.get("wr", 0) <= 40 and d.get("trades", 0) >= 4], key=lambda h: _hwr[h]["wr"])
        _best_hours_et  = [str((int(h) - 4) % 24) + ":00 ET" for h in _best_hours[:3]]
        _worst_hours_et = [str((int(h) - 4) % 24) + ":00 ET" for h in _worst_hours[:3]]
        if _best_hours_et:
            _learn_log.append(f"Best entry hours: {', '.join(_best_hours_et)}")
        if _worst_hours_et:
            _learn_log.append(f"Avoid entries at: {', '.join(_worst_hours_et)}")

        # ── 5. Sector performance adaptation ────────────────────────────
        _sec_perf = tlog.get("sector_performance", {})
        _hot_sectors_learned  = [s for s, d in _sec_perf.items() if d.get("win_rate", 0) >= 65 and d.get("total", 0) >= 4]
        _cold_sectors_learned = [s for s, d in _sec_perf.items() if d.get("win_rate", 0) <= 38 and d.get("total", 0) >= 4]
        if _hot_sectors_learned:
            _learn_log.append(f"Hot sectors by win rate: {', '.join(_hot_sectors_learned[:3])}")
        if _cold_sectors_learned:
            _learn_log.append(f"Avoid sectors (poor win rate): {', '.join(_cold_sectors_learned[:3])}")

        # ── 6. Position sizing tuning from P&L distribution ─────────────
        _pos_size_adj = _prev_learned.get("pos_size_adj", 1.0)
        if len(_closed) >= 15:
            _all_wins = [t["pnl_pct"] for t in _closed if t.get("pnl_pct", 0) > 0]
            _all_losses = [t["pnl_pct"] for t in _closed if t.get("pnl_pct", 0) < 0]
            _avg_w = sum(_all_wins) / len(_all_wins) if _all_wins else 1
            _avg_l = abs(sum(_all_losses) / len(_all_losses)) if _all_losses else 1
            _payoff = _avg_w / max(_avg_l, 0.01)
            if _payoff >= 2.5 and _r20_wr >= 0.55:
                # Great payoff ratio and solid WR — can size slightly larger
                _pos_size_adj = min(1.3, _pos_size_adj + 0.05)
                _learn_log.append(f"Payoff ratio {_payoff:.1f}x — increasing position size by 5% (adj={_pos_size_adj:.2f}x)")
            elif _payoff < 1.0 or _r20_wr < 0.40:
                # Bad payoff or losing — reduce size
                _pos_size_adj = max(0.6, _pos_size_adj - 0.10)
                _learn_log.append(f"Payoff ratio {_payoff:.1f}x, WR {_r20_wr:.0%} — reducing position size by 10% (adj={_pos_size_adj:.2f}x)")
        _pos_size_adj = max(0.6, min(1.4, _pos_size_adj))

        # ── 7. Signal pair synergy — which combos work best ──────────────
        _syn_all = tlog.get("signal_synergy", {})
        _top_synapses = sorted(
            [(k, v) for k, v in _syn_all.items() if v.get("total", 0) >= 3],
            key=lambda x: (-x[1].get("win_rate", 0), -x[1].get("avg_pnl", 0))
        )[:5]
        _top_synapses_list = [{"pair": k, "wr": v["win_rate"], "avg_pnl": v["avg_pnl"], "n": v["total"]}
                               for k, v in _top_synapses if v.get("win_rate", 0) >= 60]
        if _top_synapses_list:
            _learn_log.append(f"Top signal synergies: {' | '.join(s['pair']+'='+str(s['wr'])+'%' for s in _top_synapses_list[:3])}")

        # ── 8. Hold period optimization from outcomes ─────────────────────
        _hold_data = [(t.get("pnl_pct", 0), t.get("hold_hrs", 0)) for t in _closed
                      if t.get("hold_hrs") is not None and t.get("hold_hrs", 0) > 0]
        _optimal_hold_days = None
        if len(_hold_data) >= 8:
            _short_hold = [p for p, h in _hold_data if h < 24]    # <1 day
            _med_hold   = [p for p, h in _hold_data if 24 <= h < 72]  # 1-3 days
            _long_hold  = [p for p, h in _hold_data if h >= 72]   # 3+ days
            _sh_avg = sum(_short_hold) / len(_short_hold) if _short_hold else None
            _mh_avg = sum(_med_hold)   / len(_med_hold)   if _med_hold   else None
            _lh_avg = sum(_long_hold)  / len(_long_hold)  if _long_hold  else None
            _best_bucket = max([("short", _sh_avg, len(_short_hold)),
                                ("1-3day", _mh_avg, len(_med_hold)),
                                ("3+day", _lh_avg, len(_long_hold))],
                               key=lambda x: x[1] if x[1] is not None else -999)
            if _best_bucket[1] is not None:
                _learn_log.append(f"Best hold period: {_best_bucket[0]} (avg P&L {_best_bucket[1]:+.1f}%, n={_best_bucket[2]})")
                _optimal_hold_days = "short" if _best_bucket[0] == "short" else ("medium" if _best_bucket[0] == "1-3day" else "long")

        # ── 9. Ticker Memory Neuron: per-ticker score adjustments ────────────
        # Tickers with great history get a small score boost; repeat losers get penalized.
        _tk_mem_raw = tlog.get("ticker_memory", {})
        _ticker_score_adjs = {}
        for _tk, _td in _tk_mem_raw.items():
            _tk_total = _td.get("total", 0)
            _tk_wr    = _td.get("win_rate", 50)
            _tk_avg   = _td.get("avg_pnl", 0)
            if _tk_total >= 3:
                if _tk_wr >= 70 and _tk_avg >= 1.5:
                    _ticker_score_adjs[_tk] = 3   # great history: boost
                elif _tk_wr >= 60 and _tk_avg >= 0.5:
                    _ticker_score_adjs[_tk] = 1   # decent history: small boost
                elif _tk_wr <= 30 and _tk_avg <= -1.0:
                    _ticker_score_adjs[_tk] = -5  # terrible history: avoid
                elif _tk_wr <= 40 and _tk_avg <= -0.5:
                    _ticker_score_adjs[_tk] = -3  # poor history: penalize
        _ticker_stars = sorted(_ticker_score_adjs.items(), key=lambda x: -x[1])
        if _ticker_stars:
            _best_tickers  = [f"{t}(+{adj})" for t, adj in _ticker_stars[:3] if adj > 0]
            _worst_tickers = [f"{t}({adj})" for t, adj in reversed(_ticker_stars) if adj < 0][:3]
            if _best_tickers:
                _learn_log.append(f"Ticker stars (score boost): {', '.join(_best_tickers)}")
            if _worst_tickers:
                _learn_log.append(f"Ticker avoid (score penalty): {', '.join(_worst_tickers)}")

        # ── 10. 30-Minute Window Neuron: sub-hour time accuracy ──────────────
        _hw_perf_raw = tlog.get("halfhour_performance", {})
        _best_halfhours  = sorted([hw for hw, hd in _hw_perf_raw.items()
                                   if hd.get("win_rate", 0) >= 65 and hd.get("total", 0) >= 4],
                                  key=lambda h: -_hw_perf_raw[h]["win_rate"])
        _worst_halfhours = sorted([hw for hw, hd in _hw_perf_raw.items()
                                   if hd.get("win_rate", 0) <= 38 and hd.get("total", 0) >= 4],
                                  key=lambda h: _hw_perf_raw[h]["win_rate"])
        if _best_halfhours:
            _learn_log.append(f"Best 30-min entry windows (UTC): {', '.join(_best_halfhours[:3])}")
        if _worst_halfhours:
            _learn_log.append(f"Worst 30-min entry windows (UTC): {', '.join(_worst_halfhours[:3])}")

        # ── 11. VIX Bracket Neuron: volatility regime performance ────────────
        _vix_bp = tlog.get("vix_bracket_performance", {})
        _vix_bracket_insights = []
        for _vbkt, _vbd in _vix_bp.items():
            if _vbd.get("total", 0) >= 4:
                _vix_bracket_insights.append({
                    "bracket": _vbkt, "win_rate": _vbd.get("win_rate", 50),
                    "avg_pnl": _vbd.get("avg_pnl", 0), "total": _vbd.get("total", 0)
                })
        _vix_bracket_insights.sort(key=lambda x: -x["win_rate"])
        if _vix_bracket_insights:
            _vix_summary = " | ".join(f"{v['bracket']}:{v['win_rate']:.0f}%WR" for v in _vix_bracket_insights[:4])
            _learn_log.append(f"VIX bracket WRs: {_vix_summary}")

        # ── 12. Earnings Proximity Neuron: near-earnings play outcomes ────────
        _earn_prox = tlog.get("earnings_proximity_perf", {})
        _earn_insights = []
        for _ebkt, _ebd in sorted(_earn_prox.items()):
            if _ebd.get("total", 0) >= 3:
                _earn_insights.append({"bucket": _ebkt, "win_rate": _ebd.get("win_rate", 50),
                                       "avg_pnl": _ebd.get("avg_pnl", 0), "total": _ebd.get("total", 0)})
        if _earn_insights:
            _earn_summary = " | ".join(f"{e['bucket']}:{e['win_rate']:.0f}%WR" for e in _earn_insights)
            _learn_log.append(f"Earnings proximity WRs: {_earn_summary}")
            # If 0-2d bucket is <40% WR, add to learn log as warning
            _very_close = next((e for e in _earn_insights if e["bucket"] == "0-2d"), None)
            if _very_close and _very_close["win_rate"] < 40:
                _learn_log.append(f"WARNING: Entries within 2d of earnings have {_very_close['win_rate']:.0f}% WR — guard reinforced")

        # ── 13. RVOL Threshold Neuron: volume confirmation analysis ──────────
        _rvol_data = tlog.get("rvol_perf", {})
        _rvol_insights = []
        for _rbkt in ("low", "normal", "high", "surge"):
            _rd = _rvol_data.get(_rbkt, {})
            if _rd.get("total", 0) >= 3:
                _rvol_insights.append({"bucket": _rbkt, "win_rate": _rd.get("win_rate", 50),
                                       "avg_pnl": _rd.get("avg_pnl", 0), "total": _rd.get("total", 0)})
        if _rvol_insights:
            _rvol_summary = " | ".join(f"{r['bucket']}:{r['win_rate']:.0f}%WR" for r in _rvol_insights)
            _learn_log.append(f"RVOL bracket WRs: {_rvol_summary}")
            # Identify best RVOL bracket
            _best_rvol = max(_rvol_insights, key=lambda x: x["win_rate"])
            if _best_rvol["win_rate"] >= 60:
                _learn_log.append(f"Best RVOL entry zone: {_best_rvol['bucket']} (avg {_best_rvol['avg_pnl']:+.1f}%)")

        # ── 14. Market Quality Threshold Neuron: minimum conditions ──────────
        _mq_data = tlog.get("mkt_quality_perf", {})
        _mq_insights = []
        for _mqbkt in ("poor", "fair", "good", "excellent"):
            _mqd = _mq_data.get(_mqbkt, {})
            if _mqd.get("total", 0) >= 3:
                _mq_insights.append({"bucket": _mqbkt, "win_rate": _mqd.get("win_rate", 50),
                                     "avg_pnl": _mqd.get("avg_pnl", 0), "total": _mqd.get("total", 0)})
        if _mq_insights:
            _mq_summary = " | ".join(f"{m['bucket']}:{m['win_rate']:.0f}%WR" for m in _mq_insights)
            _learn_log.append(f"Market quality WRs: {_mq_summary}")
            # Warn if "poor" quality entries have <40% WR
            _poor_mq = next((m for m in _mq_insights if m["bucket"] == "poor"), None)
            if _poor_mq and _poor_mq["win_rate"] < 40:
                _learn_log.append(f"Poor market quality trades failing: {_poor_mq['win_rate']:.0f}% WR — threshold raised")

        # ── 15. Momentum Grade Validation Neuron: is A+ actually best? ───────
        _grd_data = tlog.get("grade_perf", {})
        _grade_insights = []
        for _grd_key in ("A+", "A", "B", "C", "D"):
            _gd = _grd_data.get(_grd_key, {})
            if _gd.get("total", 0) >= 3:
                _grade_insights.append({"grade": _grd_key, "win_rate": _gd.get("win_rate", 50),
                                        "avg_pnl": _gd.get("avg_pnl", 0), "total": _gd.get("total", 0)})
        if _grade_insights:
            _g_summary = " | ".join(f"{g['grade']}:{g['win_rate']:.0f}%WR" for g in _grade_insights)
            _learn_log.append(f"Grade performance: {_g_summary}")
            # Warn if A+ isn't outperforming (grade inflation)
            _aplus = next((g for g in _grade_insights if g["grade"] == "A+"), None)
            _agrade = next((g for g in _grade_insights if g["grade"] == "A"), None)
            if _aplus and _agrade and _aplus["win_rate"] < _agrade["win_rate"] - 5:
                _learn_log.append(f"Grade calibration: A grades outperforming A+ — grade thresholds may need adjustment")

        # ── 16. Price Tier Intelligence Neuron: micro/small/mid/large ────────
        _tier_data = tlog.get("price_tier_perf", {})
        _tier_insights = []
        for _tier_key in ("micro", "small", "mid", "large"):
            _td_tier = _tier_data.get(_tier_key, {})
            if _td_tier.get("total", 0) >= 3:
                _tier_insights.append({"tier": _tier_key, "win_rate": _td_tier.get("win_rate", 50),
                                       "avg_pnl": _td_tier.get("avg_pnl", 0), "total": _td_tier.get("total", 0)})
        if _tier_insights:
            _t_summary = " | ".join(f"{t['tier']}:{t['win_rate']:.0f}%WR" for t in _tier_insights)
            _learn_log.append(f"Price tier WRs: {_t_summary}")
            _best_tier = max(_tier_insights, key=lambda x: x["win_rate"])
            _worst_tier = min(_tier_insights, key=lambda x: x["win_rate"])
            if _best_tier["win_rate"] - _worst_tier["win_rate"] >= 15:
                _learn_log.append(f"Clear tier edge: {_best_tier['tier']} stocks ({_best_tier['win_rate']:.0f}% WR) far outperform {_worst_tier['tier']} ({_worst_tier['win_rate']:.0f}%)")

        # ── 17. Catalyst Performance Neuron: which triggers work best ────────
        _cat_perf = tlog.get("catalyst_performance", {})
        _cat_insights = sorted(
            [{"catalyst": k, "win_rate": v.get("win_rate", 50),
              "avg_pnl": v.get("avg_pnl", 0), "total": v.get("total", 0)}
             for k, v in _cat_perf.items() if v.get("total", 0) >= 3],
            key=lambda x: -x["win_rate"]
        )
        if _cat_insights:
            _c_summary = " | ".join(f"{c['catalyst']}:{c['win_rate']:.0f}%WR" for c in _cat_insights[:5])
            _learn_log.append(f"Catalyst WRs: {_c_summary}")
            _top_cat = _cat_insights[0]
            if _top_cat["win_rate"] >= 65:
                _learn_log.append(f"Best catalyst: {_top_cat['catalyst']} ({_top_cat['win_rate']:.0f}% WR, avg {_top_cat['avg_pnl']:+.1f}%)")
            _worst_cat = _cat_insights[-1] if len(_cat_insights) > 1 else None
            if _worst_cat and _worst_cat["win_rate"] <= 40 and _worst_cat["total"] >= 5:
                _learn_log.append(f"Weak catalyst: {_worst_cat['catalyst']} only {_worst_cat['win_rate']:.0f}% WR — deprioritizing")

        # ── 18. Breadth Threshold Neuron: minimum breadth for entries ────────
        _br_data = tlog.get("breadth_perf", {})
        _br_insights = []
        _min_breadth_learned = 0.0
        for _br_bkt in ("weak", "mixed", "broad", "strong"):
            _brd = _br_data.get(_br_bkt, {})
            if _brd.get("total", 0) >= 3:
                _br_insights.append({"bucket": _br_bkt, "win_rate": _brd.get("win_rate", 50),
                                     "avg_pnl": _brd.get("avg_pnl", 0), "total": _brd.get("total", 0)})
        if _br_insights:
            _br_summary = " | ".join(f"{b['bucket']}:{b['win_rate']:.0f}%WR" for b in _br_insights)
            _learn_log.append(f"Breadth bracket WRs: {_br_summary}")
            # Learn minimum breadth: if weak breadth (<40%) has <40% WR, avoid those conditions
            _weak_br = next((b for b in _br_insights if b["bucket"] == "weak"), None)
            if _weak_br and _weak_br["win_rate"] < 40:
                _min_breadth_learned = 40.0
                _learn_log.append(f"Weak breadth entries failing ({_weak_br['win_rate']:.0f}% WR) — minimum breadth raised to 40%")
            _best_br = max(_br_insights, key=lambda x: x["win_rate"]) if _br_insights else None
            if _best_br:
                _learn_log.append(f"Best breadth entry zone: {_best_br['bucket']} ({_best_br['win_rate']:.0f}% WR)")

        # ── 19. DCA Intelligence Neuron: when does averaging down work? ──────
        _dca_out = tlog.get("dca_outcome_perf", {})
        _dca_insights = []
        for _dreg, _dd in _dca_out.items():
            if _dd.get("total", 0) >= 2:
                _dca_insights.append({"regime": _dreg, "win_rate": _dd.get("win_rate", 50),
                                      "avg_pnl": _dd.get("avg_pnl", 0), "total": _dd.get("total", 0)})
        if _dca_insights:
            _dca_summary = " | ".join(f"{d['regime']}:{d['win_rate']:.0f}%WR" for d in _dca_insights)
            _learn_log.append(f"DCA outcomes by regime: {_dca_summary}")
            _best_dca = max(_dca_insights, key=lambda x: x["win_rate"]) if _dca_insights else None
            _worst_dca = min(_dca_insights, key=lambda x: x["win_rate"]) if _dca_insights else None
            if _best_dca and _best_dca["win_rate"] >= 60:
                _learn_log.append(f"DCA works well in {_best_dca['regime']} markets ({_best_dca['win_rate']:.0f}% WR)")
            if _worst_dca and _worst_dca["win_rate"] < 40:
                _learn_log.append(f"DCA hurts in {_worst_dca['regime']} markets ({_worst_dca['win_rate']:.0f}% WR) — cut faster")

        # ── 20. RSI Entry Zone Neuron: optimal RSI at entry ───────────────────
        _rsi_data = tlog.get("rsi_entry_perf", {})
        _rsi_zone_insights = []
        for _rz_bkt in ("oversold", "neutral", "momentum", "overbought"):
            _rzd = _rsi_data.get(_rz_bkt, {})
            if _rzd.get("total", 0) >= 3:
                _rsi_zone_insights.append({"zone": _rz_bkt, "win_rate": _rzd.get("win_rate", 50),
                                           "avg_pnl": _rzd.get("avg_pnl", 0), "total": _rzd.get("total", 0)})
        if _rsi_zone_insights:
            _rz_summary = " | ".join(f"{r['zone']}:{r['win_rate']:.0f}%WR" for r in _rsi_zone_insights)
            _learn_log.append(f"RSI entry zone WRs: {_rz_summary}")
            _best_rz = max(_rsi_zone_insights, key=lambda x: x["win_rate"])
            _learn_log.append(f"Best RSI entry zone: {_best_rz['zone']} (RSI {'<35' if _best_rz['zone']=='oversold' else '35-55' if _best_rz['zone']=='neutral' else '55-70' if _best_rz['zone']=='momentum' else '>70'}): {_best_rz['win_rate']:.0f}% WR")

        # ── 21. Macro Event Neuron: FOMC/CPI/NFP trade performance ───────────
        _macro_perf_raw = tlog.get("macro_event_perf", {})
        _macro_insights = []
        for _mk, _md in _macro_perf_raw.items():
            if _md.get("total", 0) >= 2:
                _macro_insights.append({"context": _mk, "win_rate": _md.get("win_rate", 50),
                                        "avg_pnl": _md.get("avg_pnl", 0), "total": _md.get("total", 0)})
        if _macro_insights:
            _mac_sum = " | ".join(f"{m['context'].replace('_',' ')}:{m['win_rate']:.0f}%WR" for m in _macro_insights[:4])
            _learn_log.append(f"Macro event trade WRs: {_mac_sum}")
            # If event-day trades are failing, note it prominently
            _ev_day = next((m for m in _macro_insights if "event_day" in m["context"]), None)
            if _ev_day and _ev_day["win_rate"] < 40:
                _learn_log.append(f"FOMC/CPI/NFP event-day entries failing ({_ev_day['win_rate']:.0f}% WR) — will skip on event days")

        # ── 22. Signal Count Neuron: optimal # of confirming signals ──────────
        _sc_raw = tlog.get("signal_count_perf", {})
        _sc_insights = []
        _sc_best_bucket = None
        _sc_worst_bucket = None
        for _scb, _scd in _sc_raw.items():
            if _scd.get("total", 0) >= 3:
                _sc_insights.append({
                    "bucket": _scb, "win_rate": _scd.get("win_rate", 50),
                    "avg_pnl": _scd.get("avg_pnl", 0), "total": _scd.get("total", 0)
                })
        if _sc_insights:
            _sc_insights_s = sorted(_sc_insights, key=lambda x: -x["win_rate"])
            _sc_best_bucket  = _sc_insights_s[0]
            _sc_worst_bucket = _sc_insights_s[-1]
            _sc_sum = " | ".join(f"{s['bucket']}sigs:{s['win_rate']:.0f}%WR" for s in _sc_insights_s)
            _learn_log.append(f"Signal count WRs: {_sc_sum}")
            if _sc_best_bucket and _sc_worst_bucket and (_sc_best_bucket["win_rate"] - _sc_worst_bucket["win_rate"]) >= 15:
                _learn_log.append(f"Sweet spot: {_sc_best_bucket['bucket']} confirming signals ({_sc_best_bucket['win_rate']:.0f}% WR) outperforms {_sc_worst_bucket['bucket']} ({_sc_worst_bucket['win_rate']:.0f}%)")

        # ── 23. SPY Day Return Neuron: market direction at entry vs outcome ─────
        _spy_day_raw = tlog.get("spy_day_perf", {})
        _spy_day_insights = []
        for _sbk, _sd in _spy_day_raw.items():
            if _sd.get("total", 0) >= 3:
                _spy_day_insights.append({
                    "bucket": _sbk, "win_rate": _sd.get("win_rate", 50),
                    "avg_pnl": _sd.get("avg_pnl", 0), "total": _sd.get("total", 0)
                })
        if _spy_day_insights:
            _spy_day_insights_s = sorted(_spy_day_insights, key=lambda x: -x["win_rate"])
            _spy_sum = " | ".join(f"SPY_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _spy_day_insights_s)
            _learn_log.append(f"SPY day return vs outcome: {_spy_sum}")
            _spy_down = next((s for s in _spy_day_insights if s["bucket"] == "down"), None)
            _spy_up   = next((s for s in _spy_day_insights if s["bucket"] == "up"), None)
            if _spy_down and _spy_down["win_rate"] < 40:
                _learn_log.append(f"Red SPY days hurting trades ({_spy_down['win_rate']:.0f}% WR) — raising threshold on down days")
            if _spy_up and _spy_up["win_rate"] >= 65:
                _learn_log.append(f"Green SPY days boost win rate to {_spy_up['win_rate']:.0f}% — ideal entry condition confirmed")

        # ── 24. Re-Entry Success Neuron: winner vs loser re-entries ──────────
        _re_raw = tlog.get("reentry_perf", {})
        _re_insights = []
        for _retype, _red in _re_raw.items():
            if _red.get("total", 0) >= 2:
                _re_insights.append({
                    "type": _retype, "win_rate": _red.get("win_rate", 50),
                    "avg_pnl": _red.get("avg_pnl", 0), "total": _red.get("total", 0)
                })
        if _re_insights:
            _re_sum = " | ".join(f"re-entry_{s['type']}:{s['win_rate']:.0f}%WR" for s in _re_insights)
            _learn_log.append(f"Re-entry outcomes: {_re_sum}")
            _winner_re = next((r for r in _re_insights if r["type"] == "winner"), None)
            _loser_re  = next((r for r in _re_insights if r["type"] == "loser"), None)
            if _winner_re and _winner_re["win_rate"] >= 60:
                _learn_log.append(f"Winner re-entries working ({_winner_re['win_rate']:.0f}% WR) — momentum follow-through confirmed")
            if _loser_re and _loser_re["win_rate"] < 40:
                _learn_log.append(f"Loser re-entries failing ({_loser_re['win_rate']:.0f}% WR) — cooldown is correct behavior")

        # ── 39. News Catalyst Urgency Neuron: high-urgency catalyst vs outcome ────
        _nu_raw = tlog.get("urgency_perf", {})
        _nu_insights = []
        for _nubk, _nud in _nu_raw.items():
            if _nud.get("total", 0) >= 3:
                _nu_insights.append({"bucket": _nubk, "win_rate": _nud.get("win_rate", 50),
                                     "avg_pnl": _nud.get("avg_pnl", 0), "total": _nud.get("total", 0)})
        if _nu_insights:
            _nu_s = sorted(_nu_insights, key=lambda x: -x["win_rate"])
            _nu_sum = " | ".join(f"urgency_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _nu_s)
            _learn_log.append(f"Catalyst urgency WRs: {_nu_sum}")
            _high_urg = next((s for s in _nu_insights if s["bucket"] == "high"), None)
            if _high_urg and _high_urg["win_rate"] >= 65:
                _learn_log.append(f"High-urgency catalysts producing {_high_urg['win_rate']:.0f}% WR — AI urgency signal validated")

        # ── 40. VWAP Position Neuron: above/below VWAP at entry ───────────────
        _vw_raw = tlog.get("vwap_perf", {})
        _vw_insights = []
        for _vwbk, _vwd in _vw_raw.items():
            if _vwd.get("total", 0) >= 3:
                _vw_insights.append({"bucket": _vwbk, "win_rate": _vwd.get("win_rate", 50),
                                     "avg_pnl": _vwd.get("avg_pnl", 0), "total": _vwd.get("total", 0)})
        if _vw_insights:
            _vw_s = sorted(_vw_insights, key=lambda x: -x["win_rate"])
            _vw_sum = " | ".join(f"VWAP_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _vw_s)
            _learn_log.append(f"VWAP position WRs: {_vw_sum}")
            _above_vw = next((s for s in _vw_insights if s["bucket"] == "above"), None)
            _below_vw = next((s for s in _vw_insights if s["bucket"] == "below"), None)
            if _above_vw and _below_vw and (_above_vw["win_rate"] - _below_vw["win_rate"]) >= 15:
                _learn_log.append(f"Above-VWAP entries outperform by {_above_vw['win_rate']-_below_vw['win_rate']:.0f}pts — VWAP as key institutional filter confirmed")

        # ── 37. MACD State Neuron: MACD phase at entry vs outcome ─────────────────
        _mc_raw = tlog.get("macd_state_perf", {})
        _mc_insights = []
        for _mcbk, _mcd in _mc_raw.items():
            if _mcd.get("total", 0) >= 3:
                _mc_insights.append({"state": _mcbk, "win_rate": _mcd.get("win_rate", 50),
                                     "avg_pnl": _mcd.get("avg_pnl", 0), "total": _mcd.get("total", 0)})
        if _mc_insights:
            _mc_s = sorted(_mc_insights, key=lambda x: -x["win_rate"])
            _mc_sum = " | ".join(f"MACD_{s['state']}:{s['win_rate']:.0f}%WR" for s in _mc_s)
            _learn_log.append(f"MACD state WRs: {_mc_sum}")
            _best_mc = _mc_s[0]
            if _best_mc["state"] == "bull_div" and _best_mc["win_rate"] >= 60:
                _learn_log.append(f"MACD bullish divergence is strongest entry signal ({_best_mc['win_rate']:.0f}% WR)")

        # ── 38. TTM Squeeze Breakout Neuron: squeeze vs normal entries ─────────
        _sq_raw = tlog.get("squeeze_perf", {})
        _sq_squeeze = _sq_raw.get("squeeze", {})
        _sq_normal  = _sq_raw.get("no_squeeze", {})
        if _sq_squeeze.get("total", 0) >= 3 and _sq_normal.get("total", 0) >= 3:
            _sq_wr   = _sq_squeeze.get("win_rate", 50)
            _ns_wr   = _sq_normal.get("win_rate", 50)
            _learn_log.append(f"TTM Squeeze entries: {_sq_wr:.0f}% WR vs normal: {_ns_wr:.0f}% WR")
            if _sq_wr - _ns_wr >= 15:
                _learn_log.append(f"Squeeze breakouts outperform by {_sq_wr-_ns_wr:.0f}pts — coil-release thesis confirmed")
        _sq_insights = [{"type": k, "win_rate": v.get("win_rate", 50), "avg_pnl": v.get("avg_pnl", 0), "total": v.get("total", 0)}
                        for k, v in _sq_raw.items() if v.get("total", 0) >= 3]

        # ── 35. Institutional Accumulation Neuron: accum score vs outcome ────────
        _ac_raw = tlog.get("accum_perf", {})
        _ac_insights = []
        for _acbk, _acd in _ac_raw.items():
            if _acd.get("total", 0) >= 3:
                _ac_insights.append({"bucket": _acbk, "win_rate": _acd.get("win_rate", 50),
                                     "avg_pnl": _acd.get("avg_pnl", 0), "total": _acd.get("total", 0)})
        if _ac_insights:
            _ac_s = sorted(_ac_insights, key=lambda x: -x["win_rate"])
            _ac_sum = " | ".join(f"accum_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _ac_s)
            _learn_log.append(f"Accumulation score WRs: {_ac_sum}")
            _heavy = next((s for s in _ac_insights if s["bucket"] == "heavy"), None)
            _none  = next((s for s in _ac_insights if s["bucket"] == "none"), None)
            if _heavy and _heavy["win_rate"] >= 65:
                _learn_log.append(f"Heavy institutional accumulation works: {_heavy['win_rate']:.0f}% WR (avg {_heavy['avg_pnl']:+.1f}%) — smart money confirmed")

        # ── 36. RS Rating Neuron: IBD-style RS Rating at entry vs outcome ────────
        _rs_raw = tlog.get("rs_rating_perf", {})
        _rs_insights = []
        for _rsbk, _rsd in _rs_raw.items():
            if _rsd.get("total", 0) >= 3:
                _rs_insights.append({"bucket": _rsbk, "win_rate": _rsd.get("win_rate", 50),
                                     "avg_pnl": _rsd.get("avg_pnl", 0), "total": _rsd.get("total", 0)})
        if _rs_insights:
            _rs_s = sorted(_rs_insights, key=lambda x: -x["win_rate"])
            _rs_sum = " | ".join(f"RS_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _rs_s)
            _learn_log.append(f"RS rating WRs: {_rs_sum}")
            _rs_elite = next((s for s in _rs_insights if s["bucket"] == "elite"), None)
            _rs_weak  = next((s for s in _rs_insights if s["bucket"] == "weak"), None)
            if _rs_elite and _rs_weak and (_rs_elite["win_rate"] - _rs_weak["win_rate"]) >= 15:
                _learn_log.append(f"RS90+ stocks outperforming weak RS ({_rs_elite['win_rate']:.0f}% vs {_rs_weak['win_rate']:.0f}%) — IBD quality thesis validated")

        # ── 33. Trend Template Neuron: O'Neil quality score vs outcome ───────────
        _tt_raw = tlog.get("tt_perf", {})
        _tt_insights = []
        for _tbk, _td in _tt_raw.items():
            if _td.get("total", 0) >= 3:
                _tt_insights.append({"bucket": _tbk, "win_rate": _td.get("win_rate", 50),
                                     "avg_pnl": _td.get("avg_pnl", 0), "total": _td.get("total", 0)})
        if _tt_insights:
            _tt_s = sorted(_tt_insights, key=lambda x: -x["win_rate"])
            _tt_sum = " | ".join(f"TT_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _tt_s)
            _learn_log.append(f"Trend template quality WRs: {_tt_sum}")
            _tt_elite = next((s for s in _tt_insights if s["bucket"] == "elite"), None)
            _tt_weak  = next((s for s in _tt_insights if s["bucket"] == "weak"), None)
            if _tt_elite and _tt_weak and (_tt_elite["win_rate"] - _tt_weak["win_rate"]) >= 15:
                _learn_log.append(f"Elite TT setups ({_tt_elite['win_rate']:.0f}% WR) far outperform weak ({_tt_weak['win_rate']:.0f}%) — quality filter confirmed")

        # ── 34. Consecutive Green Days Neuron: momentum confirmation ──────────
        _cg_raw = tlog.get("consec_green_perf", {})
        _cg_insights = []
        for _cgbk, _cgd in _cg_raw.items():
            if _cgd.get("total", 0) >= 3:
                _cg_insights.append({"bucket": _cgbk, "win_rate": _cgd.get("win_rate", 50),
                                     "avg_pnl": _cgd.get("avg_pnl", 0), "total": _cgd.get("total", 0)})
        if _cg_insights:
            _cg_s = sorted(_cg_insights, key=lambda x: -x["win_rate"])
            _cg_sum = " | ".join(f"{s['bucket']}green:{s['win_rate']:.0f}%WR" for s in _cg_s)
            _learn_log.append(f"Consec green days WRs: {_cg_sum}")
            _cg_best = _cg_s[0]
            _cg_4d = next((s for s in _cg_insights if s["bucket"] == "4d+"), None)
            if _cg_4d and _cg_4d["win_rate"] < 40:
                _learn_log.append(f"4+ green days → exhaustion ({_cg_4d['win_rate']:.0f}% WR) — overbought extension confirmed")
            if _cg_best["bucket"] == "2-3d" and _cg_best["win_rate"] >= 60:
                _learn_log.append(f"2-3 green days is momentum sweet spot ({_cg_best['win_rate']:.0f}% WR)")

        # ── 32. Sector Momentum Neuron: sector ETF acceleration at entry ────────
        _sm_raw = tlog.get("sector_momentum_perf", {})
        _sm_insights = []
        for _smbk, _smd in _sm_raw.items():
            if _smd.get("total", 0) >= 3:
                _sm_insights.append({"momentum": _smbk, "win_rate": _smd.get("win_rate", 50),
                                     "avg_pnl": _smd.get("avg_pnl", 0), "total": _smd.get("total", 0)})
        if _sm_insights:
            _sm_s = sorted(_sm_insights, key=lambda x: -x["win_rate"])
            _sm_sum = " | ".join(f"sector_{s['momentum']}:{s['win_rate']:.0f}%WR" for s in _sm_s)
            _learn_log.append(f"Sector momentum WRs: {_sm_sum}")
            _accel = next((s for s in _sm_insights if s["momentum"] == "accelerating"), None)
            _decel = next((s for s in _sm_insights if s["momentum"] == "decelerating"), None)
            if _accel and _decel and (_accel["win_rate"] - _decel["win_rate"]) >= 15:
                _learn_log.append(f"Sector acceleration matters: accelerating={_accel['win_rate']:.0f}% vs decelerating={_decel['win_rate']:.0f}% WR")

        # ── 30. Pre-Market Gap Neuron: gap-up/down entries vs outcome ─────────
        _pg_raw = tlog.get("pm_gap_perf", {})
        _pg_insights = []
        for _pgbk, _pgd in _pg_raw.items():
            if _pgd.get("total", 0) >= 3:
                _pg_insights.append({"bucket": _pgbk, "win_rate": _pgd.get("win_rate", 50),
                                     "avg_pnl": _pgd.get("avg_pnl", 0), "total": _pgd.get("total", 0)})
        if _pg_insights:
            _pg_s = sorted(_pg_insights, key=lambda x: -x["win_rate"])
            _pg_sum = " | ".join(f"{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _pg_s)
            _learn_log.append(f"Pre-market gap WRs: {_pg_sum}")
            _big_up = next((s for s in _pg_insights if s["bucket"] == "big_up"), None)
            _big_dn = next((s for s in _pg_insights if s["bucket"] == "big_down"), None)
            if _big_up and _big_up["win_rate"] >= 65:
                _learn_log.append(f"Big pre-market gaps working ({_big_up['win_rate']:.0f}% WR) — gap-and-hold confirmed")
            if _big_dn and _big_dn["win_rate"] < 40:
                _learn_log.append(f"Big gap-downs at entry failing ({_big_dn['win_rate']:.0f}% WR) — avoid buying big gap-downs")

        # ── 31. Exit Timing Neuron: which exit hour produces best P&L ─────────
        _eh_raw = tlog.get("exit_hour_perf", {})
        _eh_insights = []
        for _ehk, _ehd in _eh_raw.items():
            if _ehd.get("total", 0) >= 3:
                _et = int(_ehk)
                _et_label = f"{(_et - 4 + 24) % 24:02d}:00ET"  # convert UTC to ET approx
                _eh_insights.append({"hour_utc": _et, "hour_label": _et_label,
                                     "win_rate": _ehd.get("win_rate", 50),
                                     "avg_pnl": _ehd.get("avg_pnl", 0), "total": _ehd.get("total", 0)})
        if _eh_insights:
            _eh_s = sorted(_eh_insights, key=lambda x: -x["win_rate"])
            _eh_sum = " | ".join(f"{s['hour_label']}:{s['win_rate']:.0f}%WR" for s in _eh_s[:5])
            _learn_log.append(f"Exit hour WRs: {_eh_sum}")
            _best_exit = _eh_s[0]
            _learn_log.append(f"Best exit window: {_best_exit['hour_label']} ({_best_exit['win_rate']:.0f}% WR, avg {_best_exit['avg_pnl']:+.1f}%)")

        # ── 28. Position Size Neuron: optimal bet size ────────────────────────
        _ps_raw = tlog.get("pos_size_perf", {})
        _ps_insights = []
        for _pbk, _pd in _ps_raw.items():
            if _pd.get("total", 0) >= 3:
                _ps_insights.append({"bucket": _pbk, "win_rate": _pd.get("win_rate", 50),
                                     "avg_pnl": _pd.get("avg_pnl", 0), "total": _pd.get("total", 0)})
        if _ps_insights:
            _ps_s = sorted(_ps_insights, key=lambda x: -x["win_rate"])
            _ps_sum = " | ".join(f"{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _ps_s)
            _learn_log.append(f"Position size WRs: {_ps_sum}")
            if _ps_s[0]["win_rate"] - _ps_s[-1]["win_rate"] >= 15:
                _learn_log.append(f"Optimal position size: {_ps_s[0]['bucket']} ({_ps_s[0]['win_rate']:.0f}% WR) vs worst {_ps_s[-1]['bucket']} ({_ps_s[-1]['win_rate']:.0f}%)")

        # ── 29. ATR Stop Distance Neuron: optimal volatility at entry ─────────
        _at_raw = tlog.get("atr_perf", {})
        _at_insights = []
        for _abk, _ad in _at_raw.items():
            if _ad.get("total", 0) >= 3:
                _at_insights.append({"bucket": _abk, "win_rate": _ad.get("win_rate", 50),
                                     "avg_pnl": _ad.get("avg_pnl", 0), "total": _ad.get("total", 0)})
        # Compute learned ATR multiplier: adjust stop width based on what's working
        _learned_atr_mult = 2.5  # default
        if _at_insights:
            _at_s = sorted(_at_insights, key=lambda x: -x["win_rate"])
            _at_sum = " | ".join(f"ATR{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _at_s)
            _learn_log.append(f"ATR bracket WRs: {_at_sum}")
            _best_atr = _at_s[0]
            if _best_atr["win_rate"] >= 60:
                _learn_log.append(f"Best ATR entry range: {_best_atr['bucket']} ({_best_atr['win_rate']:.0f}% WR, avg {_best_atr['avg_pnl']:+.1f}%)")
                # Tune the ATR multiplier: high-ATR entries performing well → widen stop tolerance
                if _best_atr["bucket"] in ("2-4%", "4%+") and _best_atr["win_rate"] >= 65:
                    _learned_atr_mult = 3.0  # wider stop for volatile setups
                elif _best_atr["bucket"] in ("<1%", "1-2%") and _best_atr["win_rate"] >= 65:
                    _learned_atr_mult = 2.0  # tighter stop for calm setups

        # ── 26. Portfolio Concentration Neuron: optimal # of held positions ───
        _conc_raw = tlog.get("concentration_perf", {})
        _conc_insights = []
        for _cbk, _cd in _conc_raw.items():
            if _cd.get("total", 0) >= 3:
                _conc_insights.append({
                    "bucket": _cbk, "win_rate": _cd.get("win_rate", 50),
                    "avg_pnl": _cd.get("avg_pnl", 0), "total": _cd.get("total", 0)
                })
        if _conc_insights:
            _conc_s = sorted(_conc_insights, key=lambda x: -x["win_rate"])
            _conc_sum = " | ".join(f"{s['bucket']}pos:{s['win_rate']:.0f}%WR" for s in _conc_s)
            _learn_log.append(f"Portfolio concentration WRs: {_conc_sum}")
            if _conc_s[0]["win_rate"] - _conc_s[-1]["win_rate"] >= 15:
                _learn_log.append(f"Optimal concentration: {_conc_s[0]['bucket']} positions ({_conc_s[0]['win_rate']:.0f}% WR)")

        # ── 27. Day-of-Week Neuron: which weekday produces best entries ────────
        _dow_raw = tlog.get("dow_perf", {})
        _dow_insights = []
        _dow_order = ["Mon","Tue","Wed","Thu","Fri"]
        for _dk in _dow_order:
            _dd = _dow_raw.get(_dk, {})
            if _dd.get("total", 0) >= 3:
                _dow_insights.append({
                    "day": _dk, "win_rate": _dd.get("win_rate", 50),
                    "avg_pnl": _dd.get("avg_pnl", 0), "total": _dd.get("total", 0)
                })
        if _dow_insights:
            _dow_best  = max(_dow_insights, key=lambda x: x["win_rate"])
            _dow_worst = min(_dow_insights, key=lambda x: x["win_rate"])
            _dow_sum = " | ".join(f"{s['day']}:{s['win_rate']:.0f}%WR" for s in _dow_insights)
            _learn_log.append(f"Day-of-week WRs: {_dow_sum}")
            if _dow_best["win_rate"] - _dow_worst["win_rate"] >= 20:
                _learn_log.append(f"Best entry day: {_dow_best['day']} ({_dow_best['win_rate']:.0f}% WR) vs worst {_dow_worst['day']} ({_dow_worst['win_rate']:.0f}%)")

        # ── 25. Score Trend Neuron: rising vs falling score at entry ──────────
        _st_raw = tlog.get("score_trend_perf", {})
        _st_insights = []
        for _stk, _std in _st_raw.items():
            if _std.get("total", 0) >= 3:
                _st_insights.append({
                    "trend": _stk, "win_rate": _std.get("win_rate", 50),
                    "avg_pnl": _std.get("avg_pnl", 0), "total": _std.get("total", 0)
                })
        if _st_insights:
            _st_sum = " | ".join(f"score_{s['trend']}:{s['win_rate']:.0f}%WR" for s in _st_insights)
            _learn_log.append(f"Score trend at entry: {_st_sum}")
            _rising = next((s for s in _st_insights if s["trend"] == "rising"), None)
            _falling = next((s for s in _st_insights if s["trend"] == "falling"), None)
            if _rising and _falling and (_rising["win_rate"] - _falling["win_rate"]) >= 15:
                _learn_log.append(f"Rising score entries ({_rising['win_rate']:.0f}% WR) outperform falling ({_falling['win_rate']:.0f}%) by {_rising['win_rate']-_falling['win_rate']:.0f}pts — momentum timing confirmed")

        # ── 41. Score Decay Warning Neuron: did decay exits save money? ─────────
        _sd_raw = tlog.get("score_decay_perf", {})
        _sd_insights = []
        for _sdk, _sdd in _sd_raw.items():
            if _sdd.get("total", 0) >= 3:
                _sd_insights.append({
                    "type": _sdk, "win_rate": _sdd.get("win_rate", 50),
                    "avg_pnl": _sdd.get("avg_pnl", 0), "total": _sdd.get("total", 0)
                })
        # Learn: if decay exits produce higher avg_pnl than held_with_decay, threshold is working
        _learned_decay_thresh = 15.0  # default
        _decay_exit_data  = next((s for s in _sd_insights if s["type"] == "decay_exit"), None)
        _held_decay_data  = next((s for s in _sd_insights if s["type"] == "held_with_decay"), None)
        if _decay_exit_data and _held_decay_data and _decay_exit_data["total"] >= 5:
            _avg_pnl_decay_exit = _decay_exit_data["avg_pnl"]
            _avg_pnl_held_decay = _held_decay_data["avg_pnl"]
            if _avg_pnl_decay_exit > _avg_pnl_held_decay + 1.0:
                # Exiting on decay is better than holding — tighten trigger (exit earlier)
                _learned_decay_thresh = max(10.0, _learned_decay_thresh - 1.5)
                _learn_log.append(f"Neuron41: decay exits avg {_avg_pnl_decay_exit:+.1f}% vs held {_avg_pnl_held_decay:+.1f}% — tightening decay threshold to {_learned_decay_thresh:.0f}pts")
            elif _avg_pnl_held_decay > _avg_pnl_decay_exit + 1.0:
                # Holding through decay works better — loosen trigger
                _learned_decay_thresh = min(25.0, _learned_decay_thresh + 1.5)
                _learn_log.append(f"Neuron41: held-with-decay avg {_avg_pnl_held_decay:+.1f}% vs exits {_avg_pnl_decay_exit:+.1f}% — loosening decay threshold to {_learned_decay_thresh:.0f}pts")
        if _sd_insights:
            _sd_sum = " | ".join(f"{s['type']}:{s['win_rate']:.0f}%WR({s['avg_pnl']:+.1f}%)" for s in _sd_insights)
            _learn_log.append(f"Score decay neuron: {_sd_sum}")

        # ── 42. POC Distance Neuron: entry vs volume POC position ─────────────
        _pc_raw = tlog.get("poc_dist_perf", {})
        _pc_insights = []
        for _pck, _pcd in _pc_raw.items():
            if _pcd.get("total", 0) >= 3:
                _pc_insights.append({
                    "bucket": _pck, "win_rate": _pcd.get("win_rate", 50),
                    "avg_pnl": _pcd.get("avg_pnl", 0), "total": _pcd.get("total", 0)
                })
        if _pc_insights:
            _pc_best = max(_pc_insights, key=lambda x: x["win_rate"])
            _pc_worst = min(_pc_insights, key=lambda x: x["win_rate"])
            _pc_sum = " | ".join(f"{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _pc_insights)
            _learn_log.append(f"POC distance WRs: {_pc_sum}")
            if _pc_best["win_rate"] - _pc_worst["win_rate"] >= 15:
                _learn_log.append(f"Best POC entry zone: {_pc_best['bucket']} ({_pc_best['win_rate']:.0f}% WR) — worst: {_pc_worst['bucket']} ({_pc_worst['win_rate']:.0f}%)")

        # ── 43. Intraday Momentum Neuron: % from open at entry ────────────────
        _im_raw = tlog.get("intraday_mom_perf", {})
        _im_insights = []
        for _imk, _imd in _im_raw.items():
            if _imd.get("total", 0) >= 3:
                _im_insights.append({
                    "bucket": _imk, "win_rate": _imd.get("win_rate", 50),
                    "avg_pnl": _imd.get("avg_pnl", 0), "total": _imd.get("total", 0)
                })
        if _im_insights:
            _im_best = max(_im_insights, key=lambda x: x["win_rate"])
            _im_worst = min(_im_insights, key=lambda x: x["win_rate"])
            _im_sum = " | ".join(f"{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _im_insights)
            _learn_log.append(f"Intraday momentum WRs: {_im_sum}")
            if _im_best["win_rate"] - _im_worst["win_rate"] >= 15:
                _learn_log.append(f"Best intraday timing: {_im_best['bucket']} ({_im_best['win_rate']:.0f}% WR avg {_im_best['avg_pnl']:+.1f}%) vs {_im_worst['bucket']} ({_im_worst['win_rate']:.0f}%)")

        # ── 44. ADX Trend Strength Neuron ─────────────────────────────────────
        _ax_raw = tlog.get("adx_perf", {})
        _ax_insights = []
        for _axk, _axd in _ax_raw.items():
            if _axd.get("total", 0) >= 3:
                _ax_insights.append({
                    "bucket": _axk, "win_rate": _axd.get("win_rate", 50),
                    "avg_pnl": _axd.get("avg_pnl", 0), "total": _axd.get("total", 0)
                })
        if _ax_insights:
            _ax_best = max(_ax_insights, key=lambda x: x["win_rate"])
            _ax_worst = min(_ax_insights, key=lambda x: x["win_rate"])
            _ax_sum = " | ".join(f"ADX_{s['bucket']}:{s['win_rate']:.0f}%WR" for s in _ax_insights)
            _learn_log.append(f"ADX trend strength WRs: {_ax_sum}")
            if _ax_best["win_rate"] - _ax_worst["win_rate"] >= 15:
                _learn_log.append(f"Strong ADX entries win rate {_ax_best['win_rate']:.0f}% vs weak {_ax_worst['win_rate']:.0f}% — trend conviction confirmed")

        # ── 45. RVOL Tier Neuron: relative volume tier at entry ───────────────
        _rv_raw = tlog.get("rvol_tier_perf", {})
        _rv_insights = []
        for _rvk, _rvd in _rv_raw.items():
            if _rvd.get("total", 0) >= 3:
                _rv_insights.append({
                    "tier": _rvk, "win_rate": _rvd.get("win_rate", 50),
                    "avg_pnl": _rvd.get("avg_pnl", 0), "total": _rvd.get("total", 0)
                })
        if _rv_insights:
            _rv_best = max(_rv_insights, key=lambda x: x["win_rate"])
            _rv_sum = " | ".join(f"RVOL_{s['tier']}:{s['win_rate']:.0f}%WR" for s in _rv_insights)
            _learn_log.append(f"RVOL tier WRs: {_rv_sum}")

        # ── 46. Stochastic Zone Neuron: %K overbought/neutral/oversold ────────
        _sk_raw = tlog.get("stoch_zone_perf", {})
        _sk_insights = []
        for _skk, _skd in _sk_raw.items():
            if _skd.get("total", 0) >= 3:
                _sk_insights.append({
                    "zone": _skk, "win_rate": _skd.get("win_rate", 50),
                    "avg_pnl": _skd.get("avg_pnl", 0), "total": _skd.get("total", 0)
                })
        if _sk_insights:
            _sk_best = max(_sk_insights, key=lambda x: x["win_rate"])
            _sk_ob = next((s for s in _sk_insights if s["zone"] == "overbought"), None)
            _sk_nt = next((s for s in _sk_insights if s["zone"] == "neutral"), None)
            _sk_sum = " | ".join(f"stoch_{s['zone']}:{s['win_rate']:.0f}%WR" for s in _sk_insights)
            _learn_log.append(f"Stoch zone WRs: {_sk_sum}")
            if _sk_ob and _sk_nt and _sk_ob["win_rate"] < 40 and _sk_ob["total"] >= 5:
                _learn_log.append(f"Overbought entries fail ({_sk_ob['win_rate']:.0f}% WR) — avoid stoch>80 buys")

        # ── 47. Multi-Timeframe Alignment Neuron ──────────────────────────────
        _mf_raw = tlog.get("mtf_align_perf", {})
        _mf_insights = []
        for _mfk, _mfd in _mf_raw.items():
            if _mfd.get("total", 0) >= 3:
                _mf_insights.append({
                    "alignment": _mfk, "win_rate": _mfd.get("win_rate", 50),
                    "avg_pnl": _mfd.get("avg_pnl", 0), "total": _mfd.get("total", 0)
                })
        if _mf_insights:
            _mf_full = next((s for s in _mf_insights if s["alignment"] == "full"), None)
            _mf_none = next((s for s in _mf_insights if s["alignment"] == "none"), None)
            _mf_sum = " | ".join(f"MTF_{s['alignment']}:{s['win_rate']:.0f}%WR" for s in _mf_insights)
            _learn_log.append(f"MTF alignment WRs: {_mf_sum}")
            if _mf_full and _mf_none and _mf_full["win_rate"] > _mf_none["win_rate"] + 15:
                _learn_log.append(f"Full MTF alignment wins {_mf_full['win_rate']:.0f}% vs none {_mf_none['win_rate']:.0f}% — high-conviction trades dominate")

        # ── 48. Options Flow Neuron ────────────────────────────────────────────
        _of_raw = tlog.get("options_flow_perf", {})
        _of_insights = []
        for _ofk, _ofd in _of_raw.items():
            if _ofd.get("total", 0) >= 3:
                _of_insights.append({
                    "tier": _ofk, "win_rate": _ofd.get("win_rate", 50),
                    "avg_pnl": _ofd.get("avg_pnl", 0), "total": _ofd.get("total", 0)
                })
        if _of_insights:
            _of_confirmed = next((s for s in _of_insights if s["tier"] == "confirmed"), None)
            _of_neutral   = next((s for s in _of_insights if s["tier"] == "neutral"), None)
            _of_sum = " | ".join(f"flow_{s['tier']}:{s['win_rate']:.0f}%WR" for s in _of_insights)
            _learn_log.append(f"Options flow WRs: {_of_sum}")
            if _of_confirmed and _of_neutral and _of_confirmed["win_rate"] > _of_neutral["win_rate"] + 15:
                _learn_log.append(f"Flow-confirmed trades win {_of_confirmed['win_rate']:.0f}% vs unconfirmed {_of_neutral['win_rate']:.0f}% — smart money signal works")

        # ── 49. MFI Zone Neuron: volume-weighted RSI zone at entry ────────────
        _mi_raw = tlog.get("mfi_zone_perf", {})
        _mi_insights = []
        for _mik, _mid in _mi_raw.items():
            if _mid.get("total", 0) >= 3:
                _mi_insights.append({
                    "zone": _mik, "win_rate": _mid.get("win_rate", 50),
                    "avg_pnl": _mid.get("avg_pnl", 0), "total": _mid.get("total", 0)
                })
        if _mi_insights:
            _mi_dist = next((s for s in _mi_insights if s["zone"] == "distribution"), None)
            _mi_nt   = next((s for s in _mi_insights if s["zone"] == "neutral"), None)
            _mi_sum = " | ".join(f"MFI_{s['zone']}:{s['win_rate']:.0f}%WR" for s in _mi_insights)
            _learn_log.append(f"MFI zone WRs: {_mi_sum}")
            if _mi_dist and _mi_nt and _mi_dist["win_rate"] < 40 and _mi_dist["total"] >= 5:
                _learn_log.append(f"MFI distribution entries fail ({_mi_dist['win_rate']:.0f}% WR) — avoid overbought volume buying")

        # Store all learned parameters for next cycle
        tlog["bot_learned_params"] = {
            "base_score_adj":      _base_score_adj,      # added to MIN_BUY_SCORE each run
            "pos_size_adj":        round(_pos_size_adj, 3),
            "elite_signals":       _elite_signals[:8],
            "weak_signals":        _weak_signals[:8],
            "hot_sectors":         _hot_sectors_learned[:5],
            "cold_sectors":        _cold_sectors_learned[:5],
            "best_hours_utc":      _best_hours[:4],
            "worst_hours_utc":     _worst_hours[:4],
            "positive_buckets":    _positive_buckets,
            "bucket_insights":     _bucket_insights,
            "top_synapses":        _top_synapses_list,   # best signal combinations
            "optimal_hold_period": _optimal_hold_days,   # short / medium / long
            "ticker_score_adjs":   _ticker_score_adjs,   # per-ticker score modifiers
            "best_halfhours_utc":  _best_halfhours[:4],  # top 30-min entry windows
            "worst_halfhours_utc": _worst_halfhours[:4], # worst 30-min entry windows
            "vix_bracket_perf":    _vix_bracket_insights,# VIX bracket outcomes
            "earnings_prox_perf":  _earn_insights,        # earnings timing outcomes
            "rvol_perf":           _rvol_insights,        # RVOL bracket outcomes
            "mkt_quality_perf":    _mq_insights,          # market quality outcomes
            "grade_perf":          _grade_insights,        # momentum grade performance
            "price_tier_perf":     _tier_insights,         # price tier performance
            "catalyst_perf":       _cat_insights[:8],      # catalyst type performance
            "breadth_perf":        _br_insights,           # breadth bracket performance
            "min_breadth_learned": _min_breadth_learned,   # learned breadth floor
            "dca_intelligence":    _dca_insights,          # DCA outcome by regime
            "rsi_entry_zones":     _rsi_zone_insights,     # RSI at entry performance
            "macro_event_perf":    _macro_insights,        # FOMC/CPI/NFP trade outcomes
            "signal_count_perf":   _sc_insights,           # optimal signal count sweet spot
            "spy_day_perf":        _spy_day_insights,      # SPY up/flat/down day vs. outcome
            "reentry_perf":        _re_insights,           # winner vs loser re-entry outcomes
            "score_trend_perf":    _st_insights,           # rising/flat/falling score at entry
            "concentration_perf":  _conc_insights,         # portfolio concentration vs outcome
            "dow_perf":            _dow_insights,           # day-of-week entry performance
            "pos_size_perf":       _ps_insights,            # position size (% portfolio) vs outcome
            "atr_perf":            _at_insights,            # ATR bracket at entry vs outcome
            "atr_mult_learned":    _learned_atr_mult,       # learned ATR stop multiplier
            "pm_gap_perf":         _pg_insights,            # pre-market gap size vs outcome
            "exit_hour_perf":      _eh_insights,            # exit timing (hour) vs P&L
            "sector_momentum_perf": _sm_insights,           # sector ETF acceleration at entry
            "tt_perf":              _tt_insights,           # O'Neil trend template quality vs outcome
            "consec_green_perf":    _cg_insights,           # consecutive green days vs outcome
            "accum_perf":           _ac_insights,           # institutional accumulation score vs outcome
            "rs_rating_perf":       _rs_insights,           # IBD RS Rating bracket vs outcome
            "macd_state_perf":      _mc_insights,           # MACD state at entry vs outcome
            "squeeze_perf":         _sq_insights,           # TTM squeeze breakout vs normal
            "urgency_perf":         _nu_insights,           # news catalyst urgency vs outcome
            "vwap_perf":            _vw_insights,           # VWAP position at entry vs outcome
            "score_decay_perf":     _sd_insights,           # decay exit vs held outcomes
            "score_decay_threshold": _learned_decay_thresh,  # learned optimal decay trigger (pts)
            "poc_dist_perf":        _pc_insights,            # POC distance at entry vs outcome
            "intraday_mom_perf":    _im_insights,            # intraday momentum (% from open) vs outcome
            "adx_perf":             _ax_insights,            # ADX trend strength at entry vs outcome
            "rvol_tier_perf":       _rv_insights,            # RVOL tier (explosive/strong/normal/weak) vs outcome
            "stoch_zone_perf":      _sk_insights,            # Stochastic %K zone at entry vs outcome
            "mtf_align_perf":       _mf_insights,            # multi-timeframe alignment at entry vs outcome
            "options_flow_perf":    _of_insights,            # options flow confirmation at entry vs outcome
            "mfi_zone_perf":        _mi_insights,            # MFI zone (volume-weighted RSI) at entry vs outcome
            "recent_wr":           round(_r20_wr, 3),
            "recent_avg_pnl":      round(_r20_avg_pnl, 2),
            "trades_analyzed":     len(_closed),
            "last_tuned":          now_utc.isoformat(),
            "learn_log":           _learn_log[-20:],     # last 20 learning observations
        }
        if _learn_log:
            logger.info(f"Self-tuning: {' | '.join(_learn_log[:3])}")
    except Exception as _ste:
        logger.debug(f"Self-tune error: {_ste}")
        tlog.setdefault("bot_learned_params", {})

    # ── Apply learned score adjustment to effective_min_score ──────────
    try:
        _learned_adj = tlog.get("bot_learned_params", {}).get("base_score_adj", 0)
        if _learned_adj != 0:
            tlog["effective_min_score"] = tlog.get("effective_min_score", MIN_BUY_SCORE) + _learned_adj
            logger.info(f"Learned score adj: {_learned_adj:+d} → effective_min_score={tlog['effective_min_score']}")
    except Exception:
        pass

    # ── Bot Conviction Meter + Strategy Mode ─────────────────────────────
    # Composite 0-100 gauge: how confident the bot is in current conditions.
    # Displayed on dashboard as a live gauge — tells the user how aggressive
    # the bot is being right now and what strategy it's running.
    try:
        _conv = 50  # neutral baseline
        _reg_name = regime.get("regime", "neutral")
        _vix_c    = regime.get("vix", 20.0) or 20.0

        # Regime layer
        if   _reg_name == "bull":    _conv += 18
        elif _reg_name == "bear":    _conv -= 22
        # VIX layer
        if   _vix_c < 15:   _conv += 12
        elif _vix_c < 18:   _conv += 6
        elif _vix_c < 22:   _conv += 0
        elif _vix_c < 28:   _conv -= 8
        elif _vix_c < 35:   _conv -= 18
        else:                _conv -= 28
        # Breadth layer
        _adv = breadth.get("adv_pct", 50) or 50
        if   _adv > 70: _conv += 10
        elif _adv > 60: _conv += 5
        elif _adv < 35: _conv -= 12
        elif _adv < 45: _conv -= 6
        # Recent win rate layer (learned)
        _lp_c = tlog.get("bot_learned_params", {})
        _rwr_c = _lp_c.get("recent_wr", 0.5) or 0.5
        if   _rwr_c >= 0.65: _conv += 8
        elif _rwr_c >= 0.55: _conv += 3
        elif _rwr_c < 0.40:  _conv -= 10
        elif _rwr_c < 0.50:  _conv -= 4
        # Drawdown penalty
        if   drawdown_pct >= 5: _conv -= 15
        elif drawdown_pct >= 3: _conv -= 8
        elif drawdown_pct >= 1.5: _conv -= 4
        # Market quality contribution
        _mq_c = tlog.get("market_quality", 50) or 50
        _conv += round((_mq_c - 50) / 10)

        _conv_final = max(5, min(98, round(_conv)))

        # Determine strategy mode from conviction + regime
        if   _conv_final >= 78 and _reg_name == "bull":
            _strat_mode = "AGGRESSIVE MOMENTUM"
            _strat_desc = "Strong conditions — taking higher-conviction breakouts with full sizing"
        elif _conv_final >= 62:
            _strat_mode = "MOMENTUM"
            _strat_desc = "Favorable market — running normal breakout playbook"
        elif _conv_final >= 48:
            _strat_mode = "SELECTIVE"
            _strat_desc = "Mixed signals — only taking highest-quality setups, reducing size"
        elif _conv_final >= 32:
            _strat_mode = "DEFENSIVE"
            _strat_desc = "Difficult environment — very selective, smaller positions, protecting capital"
        else:
            _strat_mode = "CASH CONSERVATION"
            _strat_desc = "Poor conditions — holding cash, waiting for market to stabilize"

        tlog["bot_conviction"] = _conv_final
        tlog["strategy_mode"]  = _strat_mode
        tlog["strategy_desc"]  = _strat_desc
        logger.info(f"Bot conviction: {_conv_final}/100 → {_strat_mode}")
    except Exception as _ce:
        tlog["bot_conviction"] = 50
        tlog["strategy_mode"]  = "SELECTIVE"
        tlog["strategy_desc"]  = "Normal conditions"

    # ── Last trade action summary (for dashboard) ──────────────────────
    try:
        _all_actions = [t for t in tlog.get("trades", [])
                        if t.get("action") in ("BUY","DCA","SELL","SELL_HALF","COVER","SHORT")]
        if _all_actions:
            _latest = _all_actions[-1]
            tlog["last_trade_action"] = {
                "action":  _latest.get("action"),
                "ticker":  _latest.get("ticker"),
                "price":   _latest.get("price"),
                "pnl_pct": _latest.get("pnl_pct"),
                "time":    _latest.get("time"),
                "reason":  (_latest.get("reason") or "")[:80],
                "score":   _latest.get("score"),
            }
    except Exception:
        pass

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

    # Market quality rolling history for trend display
    try:
        _mqh = tlog.get("market_quality_history", [])
        _mqh.append({"q": tlog.get("market_quality", 50), "t": now_utc.isoformat()})
        tlog["market_quality_history"] = _mqh[-48:]
    except Exception:
        pass

    # Portfolio Health Score: 0-10 composite metric for the dashboard
    # Combines position health, market conditions, risk metrics, and momentum
    try:
        _phs = 5.0  # neutral baseline
        _phs_positions = tlog.get("positions", [])
        if _phs_positions:
            # Average position health
            _ph_vals = [p.get("pos_health", 5) for p in _phs_positions if p.get("pos_health") is not None]
            if _ph_vals:
                _avg_ph = sum(_ph_vals) / len(_ph_vals)
                _phs += (_avg_ph - 5) * 0.4   # max ±2 from average position health

            # Exit urgency penalty
            _eu_vals = [p.get("exit_urgency", 0) for p in _phs_positions]
            _avg_eu = sum(_eu_vals) / len(_eu_vals) if _eu_vals else 0
            _phs -= _avg_eu * 0.3   # up to -3 from high exit urgency

            # Portfolio P&L contribution
            _pos_pnls = [p.get("pnl_pct", 0) or 0 for p in _phs_positions]
            _avg_pnl = sum(_pos_pnls) / len(_pos_pnls) if _pos_pnls else 0
            if _avg_pnl >= 10: _phs += 1.0
            elif _avg_pnl >= 5: _phs += 0.5
            elif _avg_pnl < 0: _phs -= 0.5
            elif _avg_pnl < -5: _phs -= 1.0

        # Market quality contribution
        _mq_val = tlog.get("market_quality", 50)
        if _mq_val >= 70:   _phs += 1.0
        elif _mq_val >= 55: _phs += 0.5
        elif _mq_val < 35:  _phs -= 1.0
        elif _mq_val < 45:  _phs -= 0.5

        # VaR risk contribution
        _vol_d = tlog.get("port_vol_daily_pct") or 0
        if _vol_d > 3:   _phs -= 1.0
        elif _vol_d > 2: _phs -= 0.5

        # Win rate contribution
        _wr_val = tlog.get("win_rate", 0.5) or 0.5
        if _wr_val >= 0.65:  _phs += 0.5
        elif _wr_val < 0.4:  _phs -= 0.5

        # Regime bonus/penalty
        _reg_reg = tlog.get("regime", {})
        if isinstance(_reg_reg, dict):
            if _reg_reg.get("regime") == "bull":  _phs += 0.5
            elif _reg_reg.get("regime") == "bear": _phs -= 1.0

        tlog["portfolio_health_score"] = round(max(0, min(10, _phs)), 1)
        _phs_label = ("EXCELLENT" if _phs >= 8 else "GOOD" if _phs >= 6 else
                      "FAIR" if _phs >= 4 else "WEAK" if _phs >= 2 else "POOR")
        tlog["portfolio_health_label"] = _phs_label
    except Exception as _phse:
        logger.debug(f"Portfolio health score: {_phse}")
        tlog["portfolio_health_score"] = 5.0
        tlog["portfolio_health_label"] = "FAIR"

    # Internal scan breadth metrics
    try:
        tlog["scan_breadth_pct"] = _scan_adv_pct
        tlog["scan_breadth_poor"] = _scan_breadth_poor
        tlog["new_52wh"]   = _new_52wh
        tlog["new_52wl"]   = _new_52wl
        tlog["nhl_ratio"]  = _nhl_ratio
    except NameError:
        tlog["scan_breadth_pct"] = None
        tlog["scan_breadth_poor"] = False
        tlog["new_52wh"]  = 0
        tlog["new_52wl"]  = 0
        tlog["nhl_ratio"] = 1.0

    # Plain-English summary of this cycle's decision for the dashboard
    try:
        _top_scan = tlog.get("last_scan_top", [])
        _n_pos = len(tlog.get("positions", []))
        _regime_desc = regime.get("regime", "neutral").upper()
        _vix_now = regime.get("vix", 0)
        _mq_now  = tlog.get("market_quality", 50)
        _eff_thresh = tlog.get("effective_min_score", MIN_BUY_SCORE)
        _hot_sec = sorted(sector_adjs.items(), key=lambda x: -x[1])[:2] if sector_adjs else []
        _hot_str = " | ".join(f"{s}:+{v}" for s,v in _hot_sec if v > 0)
        if made_trades:
            _recent = [t for t in tlog["trades"] if t.get("action") in ("BUY", "DCA", "SELL", "COVER")][-4:]
            _acts = " · ".join(
                f"{t['action']} ${t['ticker']} "
                f"({'+'if(t.get('pnl_pct') or 0)>=0 else ''}{(t.get('pnl_pct') or 0):.1f}%)" if t.get("action") in ("SELL","COVER")
                else f"{t['action']} ${t['ticker']} (score={t.get('score','?')})"
                for t in _recent
            )
            _last_decision = (
                f"Executed: {_acts}. "
                f"Regime: {_regime_desc} · VIX {_vix_now:.1f} · MktQuality {_mq_now}/100 · "
                f"{_n_pos} positions · threshold={_eff_thresh}. "
                f"Hot sectors: {_hot_str or 'none'}."
            )
        elif _open_guard:
            _last_decision = f"Opening guard: waiting 10 min for volatility to settle. VIX {_vix_now:.1f} · regime {_regime_desc}."
        elif _close_guard:
            _last_decision = f"Close guard: no new buys last 20 min. Managing {_n_pos} open positions. VIX {_vix_now:.1f}."
        elif _consecutive_losses:
            _last_decision = f"Loss guard: last 3 trades were losses — protecting capital. {_n_pos} open. VIX {_vix_now:.1f}."
        elif _drawdown_halt:
            _last_decision = f"Drawdown halt: -{drawdown_pct:.1f}% from peak — no new buys until recovery. {_n_pos} open. VIX {_vix_now:.1f}."
        elif vix > VIX_EXTREME_THRESH:
            _last_decision = f"VIX {_vix_now:.1f} extreme — all buys suspended. Managing exits only. MktQuality {_mq_now}/100."
        elif _top_scan:
            _top3 = _top_scan[:3]
            _top_str = " · ".join(
                f"${s['ticker']} score={s['score']} grade={s.get('grade','?')} "
                f"{'[' + s['catalyst'][:30] + ']' if s.get('catalyst') else ''}"
                for s in _top3
            )
            _last_decision = (
                f"Scanned {len(candidates)} stocks. No trigger yet. "
                f"Top candidates: {_top_str}. "
                f"Regime: {_regime_desc} · VIX {_vix_now:.1f} · MktQuality {_mq_now}/100 · "
                f"threshold={_eff_thresh}. Hot sectors: {_hot_str or 'none'}."
            )
        else:
            _last_decision = (
                f"Scanned {len(candidates)} stocks — none passed threshold {_eff_thresh}. "
                f"Regime: {_regime_desc} · VIX {_vix_now:.1f} · MktQuality {_mq_now}/100. "
                f"Hot sectors: {_hot_str or 'none'}."
            )
        tlog["last_decision"] = _last_decision
    except Exception:
        pass

    # Today's trade summary: trades made in the last 24h with running P&L
    try:
        _today_str = now_utc.strftime("%Y-%m-%d")
        _today_trades = [t for t in tlog.get("trades", []) if (t.get("time") or "")[:10] == _today_str]
        _buys_today  = [t for t in _today_trades if t.get("action") in ("BUY", "DCA")]
        _sells_today = [t for t in _today_trades if t.get("action") in ("SELL", "COVER")]
        _partial_today = [t for t in _today_trades if t.get("action") == "SELL_HALF"]
        _closed_pnl_today = sum(t.get("pnl_pct", 0) for t in _sells_today)
        _partial_pnl_today = sum(t.get("pnl_pct", 0) for t in _partial_today)
        tlog["trade_summary_today"] = {
            "date":        _today_str,
            "buys":        len(_buys_today),
            "sells":       len(_sells_today),
            "partials":    len(_partial_today),
            "closed_pnl":  round(_closed_pnl_today, 2),
            "partial_pnl": round(_partial_pnl_today, 2),
            "buy_tickers":   [t["ticker"] for t in _buys_today],
            "sell_tickers":  [t["ticker"] for t in _sells_today],
        }
    except Exception:
        pass

    # Daily P&L attribution: for each held position, compute today's USD P&L change
    # Uses market open price vs current price * shares held
    try:
        _attrib = []
        for _ap in tlog.get("positions", []):
            _asym  = _ap.get("ticker", "")
            _aprice = _ap.get("price", 0) or 0
            _acost  = _ap.get("cost", 0) or 0
            _aqty   = _ap.get("qty", 0) or 0
            _amval  = _ap.get("market_val", 0) or 0
            _apnl_pct = _ap.get("pnl_pct", 0) or 0
            # Try to get today's open price for the stock
            _aopen_price = 0.0
            try:
                _asig = live.get(_asym, {})
                _aopen_price = _asig.get("day_open", 0.0) or 0.0
            except Exception:
                pass
            if _aprice > 0 and _aqty > 0:
                if _aopen_price > 0:
                    _day_chg_pct = round((_aprice - _aopen_price) / _aopen_price * 100, 2)
                    _day_chg_usd = round(_aqty * (_aprice - _aopen_price), 2)
                else:
                    _day_chg_pct = 0.0
                    _day_chg_usd = 0.0
                _attrib.append({
                    "ticker":      _asym,
                    "day_chg_pct": _day_chg_pct,
                    "day_chg_usd": _day_chg_usd,
                    "open_price":  round(_aopen_price, 2),
                    "cur_price":   round(_aprice, 2),
                    "pnl_pct":     round(_apnl_pct, 2),
                    "mval":        round(_amval, 2),
                })
        _attrib.sort(key=lambda x: -(abs(x["day_chg_usd"])))
        tlog["daily_pnl_attribution"] = _attrib
    except Exception as _att_e:
        logger.debug(f"P&L attribution: {_att_e}")

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

    # Precompute daily P&L from perf_history for calendar heatmap
    # Groups snapshots by date, takes opening and closing values per day
    try:
        _day_map: dict = {}
        for _h in tlog["perf_history"]:
            _hday = (_h.get("t") or "")[:10]
            if not _hday or not _h.get("v"):
                continue
            if _hday not in _day_map:
                _day_map[_hday] = {"open": _h["v"], "close": _h["v"],
                                   "high": _h["v"], "low": _h["v"],
                                   "positions": _h.get("p", 0)}
            else:
                _day_map[_hday]["close"] = _h["v"]
                _day_map[_hday]["high"]  = max(_day_map[_hday]["high"], _h["v"])
                _day_map[_hday]["low"]   = min(_day_map[_hday]["low"],  _h["v"])
                _day_map[_hday]["positions"] = _h.get("p", 0)
        _daily_pnl = []
        for _dk in sorted(_day_map.keys())[-60:]:   # keep last 60 trading days
            _dv = _day_map[_dk]
            _o, _c = _dv["open"], _dv["close"]
            _ret = round((_c - _o) / _o * 100, 3) if _o > 0 else 0.0
            _daily_pnl.append({
                "date":      _dk,
                "open":      round(_o, 2),
                "close":     round(_c, 2),
                "ret_pct":   _ret,
                "positions": _dv["positions"],
            })
        tlog["daily_pnl"] = _daily_pnl
    except Exception:
        pass

    # Pre-market AI brief: runs once in the morning (7:30-9 AM ET = 12:30-14 UTC)
    # Gives trader a concise briefing on what to watch before market opens
    try:
        _pm_hour = now_utc.hour
        _is_premarket = 12 <= _pm_hour <= 14  # 7:30-9 AM ET approx
        _pm_date_key  = now_utc.strftime("%Y-%m-%d") + "_pm"
        _last_pm = tlog.get("premarket_brief_date", "")
        if _is_premarket and _pm_date_key != _last_pm and ANTHROPIC_KEY:
            import requests as _req4
            _pm_positions = tlog.get("positions", [])
            _pm_gaps = tlog.get("premarket_gaps", [])
            _pm_top  = tlog.get("last_scan_top", [])[:3]
            _spy_pm  = next((g for g in _pm_gaps if g.get("ticker") == "SPY"), {})
            _held_gaps = [g for g in _pm_gaps if g.get("ticker") in [p["ticker"] for p in _pm_positions]]
            _regime_desc = regime.get("regime", "neutral").upper()
            _vix_now = regime.get("vix", 0)
            _pm_prompt = (
                f"Pre-market briefing (2 sentences, ≤50 words total) for a momentum trader.\n"
                f"SPY PM: {_spy_pm.get('pm_gap_pct',0):+.1f}% | VIX {_vix_now:.1f} | Regime: {_regime_desc}\n"
                + (f"Held positions PM gaps: " + " | ".join(f"{g['ticker']} {g.get('pm_gap_pct',0):+.1f}%" for g in _held_gaps[:4]) + "\n" if _held_gaps else "")
                + (f"Top candidates today: " + " | ".join(f"{s['ticker']} sc{s['score']}" for s in _pm_top) + "\n" if _pm_top else "")
                + "Sentence 1: market tone + key PM gaps. Sentence 2: #1 priority action at open. Be direct."
            )
            _pmr = _req4.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 80,
                      "messages": [{"role": "user", "content": _pm_prompt}]},
                timeout=10,
            )
            if _pmr.status_code == 200:
                _pm_text = _pmr.json()["content"][0]["text"].strip()
                tlog["premarket_brief"]      = _pm_text
                tlog["premarket_brief_date"] = _pm_date_key
                logger.info(f"Pre-market brief: {_pm_text[:80]}...")
    except Exception as _pm_e:
        logger.debug(f"Pre-market brief skipped: {_pm_e}")

    # Daily AI debrief: runs once after market close (3:55-4:15 PM ET window)
    # Generates a concise post-market summary of the day's trades and key signals.
    try:
        _now_et_hr  = now_utc.hour - 5  # rough ET offset (no DST handling needed here)
        _is_close   = 20 <= now_utc.hour <= 21  # approx 3-4 PM ET in UTC-5
        _today_key  = now_utc.strftime("%Y-%m-%d")
        _last_deb   = tlog.get("daily_debrief_date", "")
        if _is_close and _today_key != _last_deb and ANTHROPIC_KEY:
            _trades_today = [t for t in tlog.get("trades", []) if (t.get("time") or "")[:10] == _today_key]
            _closed_today = [t for t in _trades_today if t.get("action") in ("SELL", "COVER") and t.get("pnl_pct") is not None]
            _buys_today   = [t for t in _trades_today if t.get("action") in ("BUY", "DCA")]
            _open_pos_today = tlog.get("positions", [])
            _day_ret = 0.0
            _dp = tlog.get("daily_pnl", [])
            if _dp and _dp[-1]["date"] == _today_key:
                _day_ret = _dp[-1]["ret_pct"]
            _top_pat = list(tlog.get("pattern_accuracy", {}).keys())[:3]
            _debrief_prompt = (
                f"Write a 3-sentence post-market debrief for a stock trader.\n"
                f"Today ({_today_key}): portfolio {_day_ret:+.2f}% | "
                f"regime={regime.get('regime','?')} VIX={regime.get('vix',0):.1f} | "
                f"breadth={breadth.get('adv_pct',50):.0f}% advancing | "
                f"buys={len(_buys_today)} sells={len(_closed_today)}\n"
                + (f"Closed trades: " + " | ".join(f"{t['ticker']} {'+'if t['pnl_pct']>=0 else ''}{t['pnl_pct']:.1f}%" for t in _closed_today[:5]) + "\n" if _closed_today else "")
                + (f"Open positions: " + " | ".join(f"{p['ticker']} {p.get('pnl_pct',0):+.1f}%" for p in _open_pos_today[:5]) + "\n" if _open_pos_today else "")
                + (f"Best patterns: {', '.join(_top_pat)}\n" if _top_pat else "")
                + "Sentence 1: what happened today (regime, key moves). "
                "Sentence 2: what worked/didn't. Sentence 3: 1 specific focus for tomorrow.\n"
                "Be specific, practical, <80 words total. No fluff."
            )
            import requests as _req2
            _dr = _req2.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 150,
                      "messages": [{"role": "user", "content": _debrief_prompt}]},
                timeout=12,
            )
            if _dr.status_code == 200:
                _debrief_text = _dr.json()["content"][0]["text"].strip()
                tlog["daily_debrief"]      = _debrief_text
                tlog["daily_debrief_date"] = _today_key
                logger.info(f"Daily debrief: {_debrief_text[:100]}...")
    except Exception as _dd_e:
        logger.debug(f"Daily debrief skipped: {_dd_e}")

    # AI Position Commentary: one-sentence Haiku assessment per held position
    # Throttled: only regenerates if commentary is > 4 hours old or score shifted >= 10 pts
    try:
        _is_market_hours = 14 <= now_utc.hour <= 22  # ~9 AM – 5 PM ET
        if ANTHROPIC_KEY and _is_market_hours and tlog.get("positions"):
            import requests as _req3
            for _pc in tlog["positions"]:
                _pc_sym   = _pc.get("ticker", "")
                _pc_pk    = peaks.get(_pc_sym, {})
                if not isinstance(_pc_pk, dict):
                    continue
                _pc_last_ts = _pc_pk.get("commentary_ts", "")
                _pc_age_hrs = 99
                if _pc_last_ts:
                    try:
                        _pc_age_hrs = (now_utc - datetime.fromisoformat(
                            _pc_last_ts.replace("Z", "+00:00"))).total_seconds() / 3600
                    except Exception:
                        pass
                _pc_entry_score = _pc.get("entry_score", 0) or 0
                _pc_live_score  = score(_pc_sym, live.get(_pc_sym, {}), regime_adj=regime_adj)
                _pc_score_delta = abs(_pc_live_score - _pc_entry_score)
                if _pc_age_hrs < 4 and _pc_score_delta < 10:
                    if _pc_pk.get("commentary"):
                        _pc["ai_commentary"] = _pc_pk["commentary"]
                    continue
                _pc_ls = _pc.get("live_signals", {})
                _pc_prompt = (
                    f"1-sentence analyst note (≤25 words) for {_pc_sym} position: "
                    f"entry ${_pc.get('cost',0):.2f} → now ${_pc.get('price',0):.2f} "
                    f"({_pc.get('pnl_pct',0):+.1f}%), "
                    f"score {_pc_live_score:.0f}, "
                    f"RSI {_pc_ls.get('rsi',50):.0f}, "
                    f"RVOL {_pc_ls.get('rvol',1):.1f}x, "
                    f"ADX {_pc_ls.get('adx',0):.0f}, "
                    f"dist-from-trail {_pc.get('dist_from_trail',0):.1f}%, "
                    f"AVWAP-entry-pct {_pc.get('avwap_entry_pct',0):+.1f}%, "
                    f"rec={_pc.get('recommendation','HOLD')}. "
                    "State action and key reason only. No fluff."
                )
                try:
                    _pc_resp = _req3.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 60,
                              "messages": [{"role": "user", "content": _pc_prompt}]},
                        timeout=8,
                    )
                    if _pc_resp.status_code == 200:
                        _pc_text = _pc_resp.json()["content"][0]["text"].strip()
                        _pc["ai_commentary"] = _pc_text
                        _pc_pk["commentary"]    = _pc_text
                        _pc_pk["commentary_ts"] = now_utc.isoformat()
                        peaks[_pc_sym] = _pc_pk
                except Exception:
                    pass
            _save(PEAKS_FILE, peaks)
    except Exception as _pc_e:
        logger.debug(f"Position commentary skipped: {_pc_e}")

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
