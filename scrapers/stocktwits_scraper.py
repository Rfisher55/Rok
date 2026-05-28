"""StockTwits public API — no auth required for basic read endpoints."""
import logging
import requests

logger = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "ROK-StockAdvisor/1.0 (research tool)"}


def get_trending() -> list:
    """Return trending symbols on StockTwits right now."""
    try:
        r = requests.get(
            "https://api.stocktwits.com/api/2/trending/symbols.json",
            headers=_HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return []
        return [
            {
                "ticker": s["symbol"],
                "name": s.get("title", ""),
                "watchlist_count": s.get("watchlist_count", 0),
            }
            for s in r.json().get("symbols", [])[:20]
            if s.get("symbol")
        ]
    except Exception as e:
        logger.debug(f"StockTwits trending: {e}")
        return []


def get_symbol_sentiment(ticker: str) -> dict:
    """Bull/bear breakdown for a specific ticker from recent StockTwits messages."""
    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            headers=_HEADERS,
            params={"limit": 30},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        messages = r.json().get("messages", [])
        if not messages:
            return None
        bullish = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bullish"
        )
        bearish = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bearish"
        )
        n = len(messages)
        return {
            "ticker": ticker,
            "total": n,
            "bullish": bullish,
            "bearish": bearish,
            "bullish_pct": round(bullish / n * 100) if n else 0,
            "bearish_pct": round(bearish / n * 100) if n else 0,
        }
    except Exception as e:
        logger.debug(f"StockTwits {ticker}: {e}")
        return None


def enrich_tickers(tickers: list) -> dict:
    """Get StockTwits sentiment for a list of tickers. Returns dict keyed by ticker."""
    results = {}
    for ticker in tickers[:15]:
        s = get_symbol_sentiment(ticker)
        if s:
            results[ticker] = s
    return results
