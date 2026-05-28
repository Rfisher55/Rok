import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "rok-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = "sqlite:///rok.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = "claude-sonnet-4-6"

    # Reddit subreddits to scrape
    REDDIT_SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "investing",
        "StockMarket",
        "pennystocks",
        "options",
        "ValueInvesting",
    ]

    # RSS news feeds (no API key needed)
    NEWS_FEEDS = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "https://rss.cnn.com/rss/money_markets.rss",
        "https://rss.cnn.com/rss/money_latest.rss",
        "https://www.investing.com/rss/news_14.rss",
        "https://www.investing.com/rss/news_25.rss",
    ]

    # Scraping schedule (cron-style)
    SCRAPE_INTERVAL_MINUTES = 60
    ANALYSIS_INTERVAL_HOURS = 6

    # Twitter/X (requires API key — set to enable)
    TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")
    TWITTER_ENABLED = bool(os.environ.get("TWITTER_BEARER_TOKEN", ""))

    # Max items per source per scrape
    REDDIT_MAX_POSTS = 50
    NEWS_MAX_ITEMS = 30
