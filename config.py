import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "rok-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = "sqlite:///rok.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = "claude-sonnet-4-6"

    # Reddit subreddits — broad coverage across all retail investor communities
    REDDIT_SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "investing",
        "StockMarket",
        "pennystocks",
        "options",
        "ValueInvesting",
        "thetagang",
        "Superstonk",
        "smallstreetbets",
        "dividends",
        "SecurityAnalysis",
        "RobinHoodPennyStocks",
        "ETFs",
        "CanadaStocks",
    ]

    # RSS news feeds — expanded to 20+ sources for maximum coverage
    NEWS_FEEDS = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "https://rss.cnn.com/rss/money_markets.rss",
        "https://rss.cnn.com/rss/money_latest.rss",
        "https://www.investing.com/rss/news_14.rss",
        "https://www.investing.com/rss/news_25.rss",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/companyNews",
        "https://seekingalpha.com/market_currents.xml",
        "https://www.benzinga.com/feed",
        "https://www.thestreet.com/rss/index.xml",
        "https://www.fool.com/feed/",
        "https://investorplace.com/feed/",
        "https://markets.businessinsider.com/rss/news",
        "https://www.zacks.com/rss/commentary.php",
    ]

    SCRAPE_INTERVAL_MINUTES = 60
    ANALYSIS_INTERVAL_HOURS = 6

    TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")
    TWITTER_ENABLED = bool(os.environ.get("TWITTER_BEARER_TOKEN", ""))

    REDDIT_MAX_POSTS = 75
    NEWS_MAX_ITEMS = 30

    CONGRESS_DAYS_BACK = 45
    TA_MAX_TICKERS = 25
