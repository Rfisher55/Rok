import feedparser
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Deduplicate articles by title across all feeds
_seen_titles: set = set()


def scrape_feed(feed_url: str, max_items: int = 30) -> list:
    """Parse an RSS/Atom feed and return article metadata."""
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        source = feed.feed.get("title", feed_url.split("/")[2])
        for entry in feed.entries[:max_items]:
            title = (entry.get("title") or "").strip()
            if not title or title in _seen_titles:
                continue
            _seen_titles.add(title)
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            summary = entry.get("summary") or entry.get("description") or ""
            articles.append({
                "source": source,
                "title": title,
                "body": summary[:2000],
                "url": entry.get("link", ""),
                "published_at": datetime(*pub[:6]) if pub else datetime.utcnow(),
            })
    except Exception as e:
        logger.debug(f"Feed {feed_url}: {e}")
    return articles


def scrape_all(feed_urls: list, max_items: int = 30) -> list:
    """Scrape all feeds, deduplicating by title. Returns combined article list."""
    _seen_titles.clear()
    all_articles = []
    for url in feed_urls:
        batch = scrape_feed(url, max_items)
        all_articles.extend(batch)
        if batch:
            logger.info(f"News {url.split('/')[2]}: +{len(batch)}")
    return all_articles
