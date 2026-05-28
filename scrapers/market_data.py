"""
Enhanced market data: earnings calendar, unusual options, fear/greed index,
short interest, and finviz scraping. All free, no API key required.
"""
import requests
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import yfinance as yf

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def get_fear_greed_index() -> dict:
    """Fetch CNN Fear & Greed index via their unofficial API."""
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        score = data.get("fear_and_greed", {}).get("score", 50)
        rating = data.get("fear_and_greed", {}).get("rating", "Neutral")
        prev = data.get("fear_and_greed", {}).get("previous_close", score)
        return {
            "score": round(score, 1),
            "rating": rating,
            "previous_score": round(prev, 1),
            "direction": "up" if score > prev else "down",
        }
    except Exception as e:
        logger.warning(f"Fear/Greed index fetch failed: {e}")
        return {"score": 50, "rating": "Neutral", "previous_score": 50, "direction": "neutral"}


def get_earnings_calendar(days_ahead: int = 7) -> list[dict]:
    """Scrape upcoming earnings from Yahoo Finance."""
    earnings = []
    try:
        end = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        start = datetime.utcnow().strftime("%Y-%m-%d")

        resp = requests.get(
            f"https://finance.yahoo.com/calendar/earnings?from={start}&to={end}",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.find("table")
        if not table:
            return earnings

        for row in table.find_all("tr")[1:30]:
            cols = row.find_all("td")
            if len(cols) >= 3:
                earnings.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True) if len(cols) > 1 else "",
                    "date": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                    "timing": cols[3].get_text(strip=True) if len(cols) > 3 else "",
                    "eps_estimate": cols[4].get_text(strip=True) if len(cols) > 4 else "",
                })
    except Exception as e:
        logger.warning(f"Earnings calendar fetch failed: {e}")
    return earnings


def get_unusual_options_activity() -> list[dict]:
    """Scrape unusual options activity from Finviz."""
    activity = []
    try:
        resp = requests.get(
            "https://finviz.com/screener.ashx?v=111&f=sh_opt_unusual&o=-volume",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.find("table", {"id": "screener-table"})
        if not table:
            # Finviz layout varies — try alternate selector
            rows = soup.select("tr.styled-row")
            for row in rows[:20]:
                cols = row.find_all("td")
                if cols:
                    activity.append({
                        "ticker": cols[0].get_text(strip=True),
                        "description": " ".join(c.get_text(strip=True) for c in cols[:5]),
                    })
        else:
            for row in table.find_all("tr")[1:20]:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    activity.append({
                        "ticker": cols[0].get_text(strip=True),
                        "description": " ".join(c.get_text(strip=True) for c in cols[1:6]),
                    })
    except Exception as e:
        logger.warning(f"Unusual options activity scrape failed: {e}")
    return activity


def get_most_active_stocks() -> list[dict]:
    """Get most active stocks by volume from Yahoo Finance."""
    stocks = []
    try:
        resp = requests.get(
            "https://finance.yahoo.com/most-active/",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        rows = soup.select("table tbody tr")
        for row in rows[:20]:
            cols = row.find_all("td")
            if len(cols) >= 5:
                stocks.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True),
                    "price": cols[2].get_text(strip=True),
                    "change": cols[3].get_text(strip=True),
                    "change_pct": cols[4].get_text(strip=True),
                    "volume": cols[5].get_text(strip=True) if len(cols) > 5 else "",
                })
    except Exception as e:
        logger.warning(f"Most active stocks fetch failed: {e}")
    return stocks


def get_trending_on_yahoo() -> list[dict]:
    """Get trending tickers from Yahoo Finance."""
    stocks = []
    try:
        resp = requests.get(
            "https://finance.yahoo.com/trending-tickers/",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table tbody tr")
        for row in rows[:15]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                stocks.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True),
                    "price": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                })
    except Exception as e:
        logger.warning(f"Yahoo trending tickers fetch failed: {e}")
    return stocks


def get_market_indices() -> dict:
    """Fetch SPY, QQQ, DIA, VIX current data."""
    indices = {}
    for ticker in ["SPY", "QQQ", "DIA", "^VIX"]:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                chg = ((price - prev) / prev * 100) if prev else 0
                indices[ticker.replace("^", "")] = {
                    "price": round(price, 2),
                    "change_pct": round(chg, 2),
                }
        except Exception as e:
            logger.warning(f"Index fetch failed {ticker}: {e}")
    return indices


def get_short_squeeze_candidates() -> list[dict]:
    """Pull high short-interest stocks from Yahoo Finance screener."""
    candidates = []
    try:
        resp = requests.get(
            "https://finance.yahoo.com/screener/predefined/short_squeeze_stocks",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table tbody tr")
        for row in rows[:15]:
            cols = row.find_all("td")
            if len(cols) >= 3:
                candidates.append({
                    "ticker": cols[0].get_text(strip=True),
                    "company": cols[1].get_text(strip=True),
                    "short_float": cols[4].get_text(strip=True) if len(cols) > 4 else "",
                })
    except Exception as e:
        logger.warning(f"Short squeeze candidates fetch failed: {e}")
    return candidates
