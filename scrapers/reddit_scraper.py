import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "ROK-StockAdvisor/1.0 (research tool; contact@example.com)"
}


def scrape_subreddit(subreddit: str, limit: int = 50) -> list[dict]:
    """Pull hot posts from a subreddit using Reddit's public JSON API."""
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    posts = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            posts.append({
                "source": f"reddit/{subreddit}",
                "title": post.get("title", ""),
                "body": post.get("selftext", "")[:2000],
                "url": f"https://reddit.com{post.get('permalink', '')}",
                "upvotes": post.get("ups", 0),
                "score": post.get("score", 0),
                "created_at": datetime.utcfromtimestamp(post.get("created_utc", 0)),
            })
    except Exception as e:
        logger.warning(f"Reddit scrape failed for r/{subreddit}: {e}")
    return posts


def scrape_all(subreddits: list[str], limit: int = 50) -> list[dict]:
    """Scrape multiple subreddits and return combined posts."""
    all_posts = []
    for sub in subreddits:
        all_posts.extend(scrape_subreddit(sub, limit))
        logger.info(f"Scraped r/{sub}: {len(all_posts)} total posts so far")
    return all_posts
