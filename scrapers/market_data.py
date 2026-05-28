"""
Enhanced market data: fear/greed, earnings, options, market breadth, put/call ratio,
short interest, indices — all free, no API key required.
"""
import logging
import requests
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import yfinance as yf

logger = logging.getLogger(__name__)

_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_JSON = {"User-Agent": "ROK-StockAdvisor/1.0 (research tool)"}


def get_fear_greed_index() -> dict:
    """CNN Fear & Greed index via their unofficial data API."""
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers=_BROWSER,
            timeout=10,
        )
        r.raise_for_status()
        fg = r.json().get("fear_and_greed", {})
        score = fg.get("score", 50)
        prev = fg.get("previous_close", score)
        return {
            "score": round(score, 1),
            "rating": fg.get("rating", "Neutral"),
            "previous_score": round(prev, 1),
            "direction": "up" if score > prev else "down" if score < prev else "flat",
        }
    except Exception as e:
        logger.warning(f"Fear/Greed: {e}")
        return {"score": 50, "rating": "Neutral", "previous_score": 50, "direction": "flat"}


def get_put_call_ratio() -> dict:
    """CBOE total equity put/call ratio from their public stats endpoint."""
    try:
        r = requests.get(
            "https://cdn.cboe.com/api/global/us_options_market_statistics/daily-market-statistics.json",
            headers=_JSON,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        pcr = data.get("data", {}).get("total_put_call_ratio")
        equity_pcr = data.get("data", {}).get("equity_put_call_ratio")
        if pcr is None:
            # Try alternate key
            for key in ["putCallRatio", "put_call_ratio", "pcr"]:
                if key in data:
                    pcr = data[key]
                    break
        return {
            "total": round(float(pcr), 3) if pcr else None,
            "equity": round(float(equity_pcr), 3) if equity_pcr else None,
            "signal": (
                "FEAR" if pcr and float(pcr) > 1.2
                else "GREED" if pcr and float(pcr) < 0.7
                else "NEUTRAL"
            ),
        }
    except Exception as e:
        logger.debug(f"Put/Call ratio: {e}")
        return {"total": None, "equity": None, "signal": "UNKNOWN"}


def get_market_breadth() -> dict:
    """
    Market breadth indicators via yfinance:
    advance/decline ratio, new highs/lows proxies using ETF data.
    """
    breadth = {}
    try:
        # Use sector ETFs as proxy for breadth
        sector_etfs = {
            "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
            "XLV": "Healthcare", "XLY": "Consumer Disc", "XLP": "Consumer Staples",
            "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
            "XLRE": "Real Estate", "XLC": "Comm Services",
        }
        up_sectors = []
        down_sectors = []
        sector_performance = {}
        for etf, name in sector_etfs.items():
            try:
                hist = yf.Ticker(etf).history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    chg = (float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])) / float(hist["Close"].iloc[-2]) * 100
                    chg = round(chg, 2)
                    sector_performance[name] = chg
                    if chg > 0:
                        up_sectors.append({"sector": name, "change_pct": chg})
                    else:
                        down_sectors.append({"sector": name, "change_pct": chg})
            except Exception:
                pass

        total = len(up_sectors) + len(down_sectors)
        breadth = {
            "advancing_sectors": len(up_sectors),
            "declining_sectors": len(down_sectors),
            "breadth_pct": round(len(up_sectors) / total * 100, 0) if total else 50,
            "top_sectors": sorted(up_sectors, key=lambda x: x["change_pct"], reverse=True)[:3],
            "worst_sectors": sorted(down_sectors, key=lambda x: x["change_pct"])[:3],
            "sector_performance": sector_performance,
        }

        # NYSE advance/decline via Yahoo (^ADVN, ^DECN)
        for sym, key in [("^ADVN", "advancers"), ("^DECN", "decliners")]:
            try:
                hist = yf.Ticker(sym).history(period="1d")
                if not hist.empty:
                    breadth[key] = int(hist["Close"].iloc[-1])
            except Exception:
                pass

        if "advancers" in breadth and "decliners" in breadth:
            tot = breadth["advancers"] + breadth["decliners"]
            breadth["advance_decline_ratio"] = round(breadth["advancers"] / breadth["decliners"], 2) if breadth["decliners"] else None
            breadth["breadth_pct"] = round(breadth["advancers"] / tot * 100, 1) if tot else 50

    except Exception as e:
        logger.warning(f"Market breadth: {e}")

    return breadth


def get_earnings_calendar(days_ahead: int = 7) -> list:
    """Upcoming earnings from Yahoo Finance earnings calendar."""
    earnings = []
    try:
        start = datetime.utcnow().strftime("%Y-%m-%d")
        end = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        r = requests.get(
            f"https://finance.yahoo.com/calendar/earnings?from={start}&to={end}",
            headers=_BROWSER,
            timeout=12,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table")
        if table:
            for row in table.find_all("tr")[1:35]:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    earnings.append({
                        "ticker": cols[0].get_text(strip=True),
                        "company": cols[1].get_text(strip=True) if len(cols) > 1 else "",
                        "date": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                        "timing": cols[3].get_text(strip=True) if len(cols) > 3 else "",
                        "eps_estimate": cols[4].get_text(strip=True) if len(cols) > 4 else "n/a",
                    })
    except Exception as e:
        logger.warning(f"Earnings calendar: {e}")
    return earnings


def get_unusual_options_activity() -> list:
    """Unusual options activity from Finviz screener."""
    activity = []
    try:
        r = requests.get(
            "https://finviz.com/screener.ashx?v=111&f=sh_opt_unusual&o=-volume",
            headers=_BROWSER,
            timeout=12,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # Try multiple selectors since Finviz changes layout
        rows = soup.select("table#screener-table tr") or soup.select("tr.styled-row") or []
        for row in rows[:20]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                ticker = cols[0].get_text(strip=True)
                if ticker and ticker.isupper() and 1 <= len(ticker) <= 5:
                    activity.append({
                        "ticker": ticker,
                        "description": " | ".join(c.get_text(strip=True) for c in cols[1:6] if c.get_text(strip=True)),
                    })
    except Exception as e:
        logger.warning(f"Unusual options: {e}")
    return activity


def get_most_active_stocks() -> list:
    """Most-active stocks by volume from Yahoo Finance."""
    stocks = []
    try:
        r = requests.get("https://finance.yahoo.com/most-active/", headers=_BROWSER, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for row in soup.select("table tbody tr")[:20]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                stocks.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True),
                    "price": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                    "change_pct": cols[4].get_text(strip=True) if len(cols) > 4 else "",
                    "volume": cols[5].get_text(strip=True) if len(cols) > 5 else "",
                })
    except Exception as e:
        logger.warning(f"Most active: {e}")
    return stocks


def get_trending_on_yahoo() -> list:
    """Trending tickers from Yahoo Finance."""
    stocks = []
    try:
        r = requests.get("https://finance.yahoo.com/trending-tickers/", headers=_BROWSER, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for row in soup.select("table tbody tr")[:15]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                stocks.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True),
                })
    except Exception as e:
        logger.warning(f"Yahoo trending: {e}")
    return stocks


def get_market_indices() -> dict:
    """SPY, QQQ, DIA, IWM, VIX — current price and daily change."""
    indices = {}
    symbols = [("SPY", "S&P 500"), ("QQQ", "NASDAQ"), ("DIA", "DOW"), ("IWM", "Russell 2000"), ("^VIX", "VIX")]
    for sym, label in symbols:
        try:
            hist = yf.Ticker(sym).history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                chg = ((price - prev) / prev * 100) if prev else 0
                indices[label] = {"price": round(price, 2), "change_pct": round(chg, 2)}
        except Exception as e:
            logger.debug(f"Index {sym}: {e}")
    return indices


def get_short_squeeze_candidates() -> list:
    """High short-interest stocks from Yahoo Finance screener."""
    candidates = []
    try:
        r = requests.get(
            "https://finance.yahoo.com/screener/predefined/short_squeeze_stocks",
            headers=_BROWSER,
            timeout=10,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for row in soup.select("table tbody tr")[:15]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                candidates.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True),
                    "short_float": cols[4].get_text(strip=True) if len(cols) > 4 else "n/a",
                })
    except Exception as e:
        logger.warning(f"Short squeeze: {e}")
    return candidates
