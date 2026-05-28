import logging
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)

# Expanded seed list: mega caps, AI, finance, energy, biotech, meme, ETFs
_SEED_TICKERS = [
    # Mega caps & AI leaders
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "NFLX",
    # AI / Semis / Cloud
    "AMD", "PLTR", "CRWD", "PANW", "SNOW", "DDOG", "NET", "AI", "SMCI", "ARM",
    "MU", "INTC", "QCOM", "TXN",
    # Finance
    "JPM", "BAC", "GS", "AMP", "V", "MA", "PYPL", "SQ", "HOOD", "COIN",
    # Energy
    "XOM", "CVX", "OXY", "SLB",
    # Healthcare / Biotech
    "LLY", "MRNA", "PFE", "ABBV", "BIIB", "GILD",
    # Retail / Consumer
    "COST", "WMT", "TGT", "SHOP", "AMZN",
    # EV / Mobility
    "RIVN", "LCID", "NIO", "F", "GM",
    # Meme / Retail favorites
    "GME", "AMC", "MARA", "RBLX", "SNAP", "UBER", "LYFT", "ABNB", "SOFI",
    # ETFs & indices
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XBI",
]


def get_stock_data(ticker: str) -> dict:
    """
    Full fundamental + technical snapshot for a ticker via yfinance.
    Returns None if data is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        hist = stock.history(period="30d")
        if hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        chg = ((price - prev) / prev * 100) if prev else 0

        price_history = [round(float(p), 2) for p in hist["Close"].tolist()]

        # Analyst data
        target = info.get("targetMeanPrice") or info.get("targetHighPrice")
        upside = None
        if target and price:
            upside = round((target - price) / price * 100, 1)

        # Short interest
        short_pct = info.get("shortPercentOfFloat") or info.get("shortRatio")

        return {
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "change_pct": round(chg, 2),
            "volume": int(hist["Volume"].mean()),
            "market_cap": info.get("marketCap"),
            "week_high": round(float(hist["High"].max()), 2),
            "week_low": round(float(hist["Low"].min()), 2),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "eps": info.get("trailingEps"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "short_interest": short_pct,
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "company_name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "analyst_target": round(target, 2) if target else None,
            "upside_to_target": upside,
            "recommendation": info.get("recommendationKey", ""),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "price_history": price_history,
            "snapped_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.debug(f"Yahoo Finance {ticker}: {e}")
        return None


def get_trending_tickers() -> list:
    """Seed ticker list for analysis when social mention list is sparse."""
    return list(dict.fromkeys(_SEED_TICKERS))


def get_price_history(ticker: str, days: int = 30) -> list:
    """Return list of daily close prices for the last N trading days."""
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d")
        if hist.empty:
            return []
        return [round(float(p), 2) for p in hist["Close"].tolist()]
    except Exception as e:
        logger.debug(f"Price history {ticker}: {e}")
        return []
