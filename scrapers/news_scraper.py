import feedparser
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def scrape_feed(feed_url: str, max_items: int = 30) -> list[dict]:
    """Parse an RSS/Atom feed and return article metadata."""
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:max_items]:
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            articles.append({
                "source": feed.feed.get("title", feed_url),
                "title": entry.get("title", ""),
                "body": entry.get("summary", "")[:2000],
                "url": entry.get("link", ""),
                "published_at": datetime(*pub[:6]) if pub else datetime.utcnow(),
            })
    except Exception as e:
        logger.warning(f"News feed failed {feed_url}: {e}")
    return articles


def scrape_all(feed_urls: list[str], max_items: int = 30) -> list[dict]:
    all_articles = []
    for url in feed_urls:
        batch = scrape_feed(url, max_items)
        all_articles.extend(batch)
        logger.info(f"News feed {url}: +{len(batch)} articles")
    return all_articles
