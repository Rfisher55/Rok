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
    """CBOE total equity put/call ratio — tries multiple endpoints."""
    urls = [
        "https://cdn.cboe.com/api/global/us_options_market_statistics/daily-market-statistics.json",
        "https://www.cboe.com/us/options/market_statistics/daily_market_statistics.json",
        "https://data.cboe.com/api/global/us_options_market_statistics/daily-market-statistics.json",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=_JSON, timeout=10)
            if not r.ok:
                continue
            data = r.json()
            # Try nested and flat formats
            pcr = (data.get("data") or data).get("total_put_call_ratio") or data.get("total_put_call_ratio")
            equity_pcr = (data.get("data") or data).get("equity_put_call_ratio") or data.get("equity_put_call_ratio")
            if pcr is None:
                # Search all keys
                flat = data.get("data", data)
                for key in flat:
                    if "put" in str(key).lower() and "call" in str(key).lower() and "total" in str(key).lower():
                        pcr = flat[key]
                        break
                    if "equity" in str(key).lower() and "put" in str(key).lower():
                        equity_pcr = equity_pcr or flat[key]
            if pcr is not None:
                pcr = float(pcr)
                return {
                    "total": round(pcr, 3),
                    "equity": round(float(equity_pcr), 3) if equity_pcr else None,
                    "signal": "FEAR" if pcr > 1.2 else "GREED" if pcr < 0.7 else "NEUTRAL",
                }
        except Exception as e:
            logger.debug(f"Put/Call ratio {url}: {e}")

    # Fallback: estimate from yfinance VIX — high VIX implies high put buying
    try:
        vix = yf.Ticker("^VIX").history(period="1d")
        if not vix.empty:
            vix_level = float(vix["Close"].iloc[-1])
            estimated_pcr = round(0.7 + vix_level / 100, 3)
            return {
                "total": estimated_pcr,
                "equity": None,
                "signal": "FEAR" if vix_level > 25 else "GREED" if vix_level < 15 else "NEUTRAL",
                "note": "estimated from VIX",
            }
    except Exception:
        pass
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
    """
    Unusual options activity — multi-source: Barchart → Market Chameleon → Yahoo high-volume options.
    Returns list of {ticker, description} sorted by options volume.
    """
    activity = _unusual_opts_barchart()
    if not activity:
        activity = _unusual_opts_market_chameleon()
    if not activity:
        activity = _unusual_opts_yahoo_screener()
    return activity[:20]


def _unusual_opts_barchart() -> list:
    """Barchart unusual options — public page, no key required."""
    activity = []
    try:
        r = requests.get(
            "https://www.barchart.com/options/unusual-activity/stocks",
            headers={**_BROWSER, "Referer": "https://www.barchart.com/"},
            timeout=14,
        )
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        for row in soup.select("table tbody tr, .bc-table tbody tr")[:25]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            ticker = cols[0].get_text(strip=True).split()[0].upper()
            if not ticker or not ticker.isalpha() or len(ticker) > 5:
                continue
            desc_parts = [c.get_text(strip=True) for c in cols[1:7] if c.get_text(strip=True)]
            activity.append({"ticker": ticker, "description": " | ".join(desc_parts)})
        return activity
    except Exception as e:
        logger.debug(f"Barchart unusual opts: {e}")
    return []


def _unusual_opts_market_chameleon() -> list:
    """Market Chameleon unusual options flow page."""
    activity = []
    try:
        r = requests.get(
            "https://marketchameleon.com/volReports/UnusualOptionVolume",
            headers=_BROWSER,
            timeout=14,
        )
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        for row in soup.select("table tbody tr")[:25]:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            ticker = cols[0].get_text(strip=True).upper()
            if not ticker or not ticker.isalpha() or len(ticker) > 5:
                continue
            desc_parts = [c.get_text(strip=True) for c in cols[1:6] if c.get_text(strip=True)]
            activity.append({"ticker": ticker, "description": " | ".join(desc_parts)})
        return activity
    except Exception as e:
        logger.debug(f"Market Chameleon unusual opts: {e}")
    return []


def _unusual_opts_yahoo_screener() -> list:
    """
    Yahoo Finance: derive unusual options candidates from stocks with high
    options implied move (high IV). Uses yfinance to get option chain volume
    for known high-volume underlyings.
    """
    activity = []
    # Broad universe of commonly active options tickers
    candidates = [
        "SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "META", "AMZN", "MSFT",
        "GOOGL", "NFLX", "BABA", "PLTR", "SOFI", "RIVN", "GME", "AMC", "MARA",
        "COIN", "HOOD", "F", "BAC", "GS", "JPM", "XOM", "CVNA", "SMCI",
    ]
    try:
        for sym in candidates[:20]:
            try:
                tk = yf.Ticker(sym)
                dates = tk.options
                if not dates:
                    continue
                chain = tk.option_chain(dates[0])
                calls = chain.calls
                puts = chain.puts
                if calls.empty and puts.empty:
                    continue
                # High volume/OI ratio = unusual
                c_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
                p_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
                total_vol = c_vol + p_vol
                if total_vol < 500:
                    continue
                pcr = round(p_vol / c_vol, 2) if c_vol else 0
                bias = "BULLISH" if pcr < 0.7 else "BEARISH" if pcr > 1.3 else "NEUTRAL"
                activity.append({
                    "ticker": sym,
                    "description": f"Vol: {total_vol:,} | P/C: {pcr} | Bias: {bias}",
                })
            except Exception:
                pass
        activity.sort(key=lambda x: int(x["description"].split("Vol: ")[1].split(" |")[0].replace(",", "")), reverse=True)
    except Exception as e:
        logger.debug(f"Yahoo options screener: {e}")
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
