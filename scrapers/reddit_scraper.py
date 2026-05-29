import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Multiple User-Agents to rotate
_HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0", "Accept": "application/json"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36", "Accept": "application/json"},
    {"User-Agent": "ROK-StockAdvisor/1.0 robertcfisher3@gmail.com", "Accept": "application/json"},
]

_header_idx = 0


def _next_headers():
    global _header_idx
    h = _HEADERS_LIST[_header_idx % len(_HEADERS_LIST)]
    _header_idx += 1
    return h


def _scrape_json(subreddit: str, sort: str = "hot", limit: int = 50) -> list[dict]:
    """Try Reddit JSON API with rotating user agents."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&t=day"
    for _ in range(len(_HEADERS_LIST)):
        try:
            resp = requests.get(url, headers=_next_headers(), timeout=12)
            if resp.status_code == 429:
                continue
            if not resp.ok:
                continue
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            if not children:
                continue
            posts = []
            for child in children:
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
            return posts
        except Exception as e:
            logger.debug(f"Reddit JSON {subreddit}: {e}")
    return []


def _scrape_rss(subreddit: str) -> list[dict]:
    """Fallback: parse subreddit RSS feed."""
    try:
        import feedparser
        feed = feedparser.parse(
            f"https://www.reddit.com/r/{subreddit}/top.rss?t=day",
            request_headers={"User-Agent": _HEADERS_LIST[0]["User-Agent"]},
        )
        posts = []
        for entry in feed.entries[:30]:
            posts.append({
                "source": f"reddit/{subreddit}",
                "title": entry.get("title", ""),
                "body": (entry.get("summary") or "")[:2000],
                "url": entry.get("link", ""),
                "upvotes": 0,
                "score": 0,
                "created_at": datetime.utcnow(),
            })
        return posts
    except Exception as e:
        logger.debug(f"Reddit RSS {subreddit}: {e}")
    return []


def scrape_subreddit(subreddit: str, limit: int = 50) -> list[dict]:
    """Pull posts from a subreddit: tries JSON API first, then RSS fallback."""
    posts = _scrape_json(subreddit, "hot", limit)
    if not posts:
        posts = _scrape_json(subreddit, "top", limit)
    if not posts:
        posts = _scrape_rss(subreddit)
    return posts


def scrape_all(subreddits: list[str], limit: int = 50) -> list[dict]:
    """Scrape multiple subreddits and return combined posts."""
    all_posts = []
    for sub in subreddits:
        batch = scrape_subreddit(sub, limit)
        all_posts.extend(batch)
        if batch:
            logger.info(f"Scraped r/{sub}: +{len(batch)} posts ({len(all_posts)} total)")
        else:
            logger.debug(f"r/{sub}: no posts scraped")
    return all_posts
