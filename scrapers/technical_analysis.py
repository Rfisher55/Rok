"""
Technical analysis via yfinance + pandas — RSI, MACD, Bollinger Bands, volume analysis.
No additional TA library needed; pandas comes bundled with yfinance.
"""
import logging

logger = logging.getLogger(__name__)


def _rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(close, period: int = 20, n_std: float = 2.0):
    sma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return sma + n_std * sd, sma, sma - n_std * sd


def analyze_ticker(ticker: str) -> dict:
    """Full TA snapshot for a single ticker. Returns None on failure."""
    try:
        import yfinance as yf

        hist = yf.download(ticker, period="60d", auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 26:
            return None

        close = hist["Close"].squeeze()
        volume = hist["Volume"].squeeze()
        price = round(float(close.iloc[-1]), 2)

        rsi_s = _rsi(close)
        macd_line, sig_line, histogram = _macd(close)
        bb_upper, bb_mid, bb_lower = _bollinger(close)

        rsi = round(float(rsi_s.iloc[-1]), 1) if not rsi_s.isnull().all() else None
        macd = round(float(macd_line.iloc[-1]), 4)
        macd_sig = round(float(sig_line.iloc[-1]), 4)
        macd_hist = round(float(histogram.iloc[-1]), 4)
        bb_u = round(float(bb_upper.iloc[-1]), 2)
        bb_m = round(float(bb_mid.iloc[-1]), 2)
        bb_l = round(float(bb_lower.iloc[-1]), 2)

        vol_10d = float(volume.tail(10).mean()) if len(volume) >= 10 else None
        vol_ratio = round(float(volume.iloc[-1]) / vol_10d, 2) if vol_10d and vol_10d > 0 else None

        # 52-week levels
        hist_1y = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        w52_high = round(float(hist_1y["High"].max()), 2) if not hist_1y.empty else None
        w52_low = round(float(hist_1y["Low"].min()), 2) if not hist_1y.empty else None
        pct_from_high = round((price - w52_high) / w52_high * 100, 1) if w52_high else None
        pct_from_low = round((price - w52_low) / w52_low * 100, 1) if w52_low else None

        # RSI signal
        rsi_sig = "OVERSOLD" if rsi and rsi < 32 else "OVERBOUGHT" if rsi and rsi > 68 else "NEUTRAL"

        # MACD signal (check histogram cross)
        macd_sig_label = "NEUTRAL"
        if len(histogram) >= 2:
            prev = float(histogram.iloc[-2])
            if prev < 0 and macd_hist > 0:
                macd_sig_label = "BULLISH_CROSS"
            elif prev > 0 and macd_hist < 0:
                macd_sig_label = "BEARISH_CROSS"
            elif macd_hist > 0:
                macd_sig_label = "BULLISH"
            else:
                macd_sig_label = "BEARISH"

        # Bollinger signal
        if price > bb_u:
            bb_sig = "BREAKOUT_UPPER"
        elif price < bb_l:
            bb_sig = "BREAKDOWN_LOWER"
        else:
            pct = round((price - bb_l) / (bb_u - bb_l) * 100, 0) if bb_u != bb_l else 50
            bb_sig = f"INSIDE_{int(pct)}pct"

        return {
            "ticker": ticker,
            "price": price,
            "rsi": rsi,
            "rsi_signal": rsi_sig,
            "macd": macd,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
            "macd_signal_label": macd_sig_label,
            "bb_upper": bb_u,
            "bb_mid": bb_m,
            "bb_lower": bb_l,
            "bb_signal": bb_sig,
            "volume_ratio": vol_ratio,
            "week_52_high": w52_high,
            "week_52_low": w52_low,
            "pct_from_52w_high": pct_from_high,
            "pct_from_52w_low": pct_from_low,
        }
    except Exception as e:
        logger.debug(f"TA {ticker}: {e}")
        return None


def analyze_multiple(tickers: list, max_tickers: int = 25) -> dict:
    """TA for multiple tickers. Returns dict keyed by ticker."""
    results = {}
    for t in tickers[:max_tickers]:
        r = analyze_ticker(t)
        if r:
            results[t] = r
    return results


def find_setups(ta_data: dict) -> list:
    """Identify actionable TA setups from analyzed ticker dict."""
    setups = []
    priority = {
        "MACD_BULLISH_CROSS": 0,
        "OVERSOLD_BOUNCE": 1,
        "BB_BREAKOUT": 2,
        "NEAR_52W_LOW": 3,
    }

    for ticker, d in ta_data.items():
        signals = []
        setup_type = None

        rsi = d.get("rsi")
        if rsi and rsi < 33:
            signals.append(f"RSI {rsi} — oversold territory")
            setup_type = "OVERSOLD_BOUNCE"

        if d.get("macd_signal_label") == "BULLISH_CROSS":
            signals.append("MACD histogram crossed above zero (bullish)")
            setup_type = "MACD_BULLISH_CROSS" if not setup_type else setup_type

        if d.get("bb_signal") == "BREAKOUT_UPPER":
            signals.append("Price broke above Bollinger upper band")
            setup_type = setup_type or "BB_BREAKOUT"

        vol = d.get("volume_ratio")
        if vol and vol > 2.0:
            signals.append(f"Volume {vol}x above 10-day average — unusual activity")

        pct_low = d.get("pct_from_52w_low")
        if pct_low is not None and pct_low < 12:
            signals.append(f"Near 52W low (+{pct_low:.1f}%) — potential base")
            setup_type = setup_type or "NEAR_52W_LOW"

        if signals and setup_type:
            setups.append({
                "ticker": ticker,
                "setup_type": setup_type,
                "signals": signals,
                "rsi": rsi,
                "volume_ratio": vol,
                "price": d.get("price"),
                "pct_from_52w_high": d.get("pct_from_52w_high"),
                "macd_hist": d.get("macd_hist"),
            })

    setups.sort(key=lambda x: priority.get(x["setup_type"], 99))
    return setups[:8]
