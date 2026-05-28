"""
Twitter/X scraper — requires TWITTER_BEARER_TOKEN environment variable.
Set it in your .env file to enable Twitter data in ROK analysis.
"""
import requests
import logging

logger = logging.getLogger(__name__)

TWITTER_API_BASE = "https://api.twitter.com/2"

FINANCIAL_ACCOUNTS = [
    "jimcramer", "elonmusk", "chamath", "BillAckman",
    "michaeljburry", "realDonaldTrump",
]

SEARCH_QUERIES = [
    "($NVDA OR $TSLA OR $AAPL) lang:en -is:retweet",
    "(stock buy OR stock sell OR earnings) lang:en -is:retweet",
    "(short squeeze OR moon OR to the moon) lang:en -is:retweet",
]


def scrape_tweets(bearer_token: str, max_results: int = 50) -> list[dict]:
    """Search recent tweets for stock mentions."""
    if not bearer_token:
        logger.info("Twitter bearer token not set — skipping Twitter scrape")
        return []

    headers = {"Authorization": f"Bearer {bearer_token}"}
    tweets = []

    for query in SEARCH_QUERIES:
        try:
            resp = requests.get(
                f"{TWITTER_API_BASE}/tweets/search/recent",
                headers=headers,
                params={
                    "query": query,
                    "max_results": min(max_results, 100),
                    "tweet.fields": "created_at,public_metrics,author_id",
                },
                timeout=10,
            )
            if resp.status_code == 429:
                logger.warning("Twitter rate limit hit — skipping")
                break
            resp.raise_for_status()
            data = resp.json()

            for tweet in data.get("data", []):
                metrics = tweet.get("public_metrics", {})
                tweets.append({
                    "source": "twitter",
                    "title": tweet.get("text", "")[:280],
                    "body": "",
                    "url": f"https://twitter.com/i/web/status/{tweet.get('id')}",
                    "upvotes": metrics.get("like_count", 0) + metrics.get("retweet_count", 0),
                })
        except Exception as e:
            logger.warning(f"Twitter scrape failed for query '{query}': {e}")

    return tweets
