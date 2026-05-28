import yfinance as yf
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def get_stock_data(ticker: str) -> dict | None:
    """Fetch current price, fundamentals, and recent history for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        hist = stock.history(period="5d")
        if hist.empty:
            return None

        current_price = float(hist["Close"].iloc[-1])
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price else 0

        week_high = float(hist["High"].max())
        week_low = float(hist["Low"].min())
        avg_volume = int(hist["Volume"].mean())

        return {
            "ticker": ticker.upper(),
            "price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "volume": avg_volume,
            "market_cap": info.get("marketCap"),
            "week_high": round(week_high, 2),
            "week_low": round(week_low, 2),
            "pe_ratio": info.get("trailingPE"),
            "short_interest": info.get("shortPercentOfFloat"),
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "analyst_target": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey", ""),
            "snapped_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Yahoo Finance failed for {ticker}: {e}")
        return None


def get_trending_tickers() -> list[str]:
    """Return a fixed watchlist of high-activity tickers to seed the analysis."""
    return [
        "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD",
        "PLTR", "GME", "AMC", "SOFI", "RIVN", "LCID", "NIO", "MARA",
        "COIN", "HOOD", "RBLX", "SNAP", "UBER", "LYFT", "ABNB", "SHOP",
        "SPY", "QQQ", "SQQQ", "TQQQ", "VIX",
    ]
