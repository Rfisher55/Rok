"""
Lightweight rule-based sentiment scoring.
No ML model required — fast keyword matching that feeds into Claude.
"""

BULLISH_WORDS = {
    "moon", "mooning", "bullish", "buy", "long", "calls", "squeeze", "breakout",
    "rally", "surge", "pump", "rocket", "soar", "beat", "outperform", "upgrade",
    "strong", "growth", "bull", "upside", "target", "accumulate", "undervalued",
    "catalyst", "momentum", "rip", "pop", "green", "run", "flying", "monster",
    "earnings beat", "guidance raised", "buyback", "dividend", "record",
}

BEARISH_WORDS = {
    "dump", "crash", "drop", "bearish", "puts", "short", "sell", "tank",
    "plunge", "collapse", "miss", "downgrade", "weak", "warning", "cut",
    "layoff", "bankrupt", "fraud", "lawsuit", "investigation", "recall",
    "guidance cut", "loss", "debt", "dilution", "overvalued", "avoid",
    "red", "bleeding", "wrecked", "rekt", "bagholding", "trap",
}


def score_text(text: str) -> float:
    """Return sentiment score: +1.0 = very bullish, -1.0 = very bearish."""
    if not text:
        return 0.0
    words = text.lower().split()
    word_set = set(words)
    bull = len(word_set & BULLISH_WORDS)
    bear = len(word_set & BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def score_posts(posts: list[dict]) -> list[dict]:
    """Add sentiment_score field to each post."""
    for p in posts:
        combined = (p.get("title", "") + " " + p.get("body", ""))
        p["sentiment_score"] = score_text(combined)
    return posts


def aggregate_sentiment(posts: list[dict]) -> dict:
    """Compute aggregate sentiment stats across all posts."""
    if not posts:
        return {"mean": 0.0, "bullish_pct": 0, "bearish_pct": 0, "neutral_pct": 100}

    scores = [score_text(p.get("title", "") + " " + p.get("body", "")) for p in posts]
    mean = sum(scores) / len(scores)
    bullish = sum(1 for s in scores if s > 0.1)
    bearish = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - bullish - bearish
    n = len(scores)

    return {
        "mean": round(mean, 3),
        "bullish_pct": round(bullish / n * 100),
        "bearish_pct": round(bearish / n * 100),
        "neutral_pct": round(neutral / n * 100),
        "total_posts": n,
    }


def per_ticker_sentiment(posts: list[dict], tickers: list[str]) -> dict[str, dict]:
    """Compute sentiment for posts mentioning each ticker."""
    result = {}
    for ticker in tickers:
        relevant = [
            p for p in posts
            if ticker.lower() in (p.get("title", "") + p.get("body", "")).lower()
            or f"${ticker}" in (p.get("title", "") + p.get("body", ""))
        ]
        result[ticker] = aggregate_sentiment(relevant)
        result[ticker]["mention_count"] = len(relevant)
    return result
