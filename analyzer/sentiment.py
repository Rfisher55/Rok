"""
Rule-based sentiment scoring with negation detection and expanded keyword sets.
Fast keyword matching that feeds additional signal context into Claude's analysis.
"""
import re

BULLISH_WORDS = {
    # Price action
    "moon", "mooning", "breakout", "rally", "surge", "pump", "rocket", "soar",
    "rip", "pop", "run", "flying", "ripping", "squeezing", "gap", "gapping",
    # Sentiment
    "bullish", "bull", "long", "calls", "call", "yolo", "apes", "diamond",
    "hold", "hodl", "accumulate", "buy", "buying", "bought",
    # Fundamentals
    "beat", "beats", "outperform", "upgrade", "upgraded", "strong", "strength",
    "growth", "growing", "upside", "target", "raised", "raise", "record",
    "guidance", "catalyst", "momentum", "undervalued", "cheap", "bargain",
    "profitable", "profit", "revenue", "earnings beat", "guidance raised",
    "buyback", "dividend", "acquisition", "merger", "partnership",
    "contract", "win", "winning", "approval", "approved",
    # Market context
    "oversold", "support", "bounce", "recovery", "rebound", "dip",
    "green", "monster", "massive",
}

BEARISH_WORDS = {
    # Price action
    "dump", "crash", "drop", "tank", "plunge", "collapse", "fall", "falling",
    "crater", "implode", "blood", "bleeding", "bleeding out",
    # Sentiment
    "bearish", "bear", "short", "puts", "put", "sell", "selling", "sold",
    "avoid", "avoid", "trap", "bagholding", "bag", "bags",
    "wrecked", "rekt", "dead", "dying",
    # Fundamentals
    "miss", "misses", "downgrade", "downgraded", "weak", "weakness", "warning",
    "cut", "cuts", "layoff", "layoffs", "bankrupt", "bankruptcy", "fraud",
    "lawsuit", "investigation", "recall", "guidance cut", "loss", "losses",
    "debt", "dilution", "diluting", "overvalued", "expensive", "bubble",
    "overbought", "resistance", "reject", "rejected",
    # Market context
    "red", "recession", "inflation", "rates", "fear", "panic", "selloff",
}

# Words that reverse sentiment when preceding a keyword ("not bullish", "no catalyst")
_NEGATIONS = {"not", "no", "never", "cannot", "can't", "won't", "don't", "nothing", "neither", "nor"}


def _tokenize(text: str) -> list:
    return re.findall(r"\b\w[\w']*\b", text.lower())


def score_text(text: str) -> float:
    """Return sentiment: +1.0 = very bullish, -1.0 = very bearish. Handles negations."""
    if not text:
        return 0.0
    tokens = _tokenize(text)
    bull = 0
    bear = 0
    for i, token in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in _NEGATIONS
        if token in BULLISH_WORDS:
            bull += -1 if negated else 1
        elif token in BEARISH_WORDS:
            bear += -1 if negated else 1

    # Multi-word phrases
    for phrase in ["earnings beat", "guidance raised", "guidance cut", "not bullish", "not bearish"]:
        if phrase in text.lower():
            if "not " in phrase:
                word = phrase.split("not ")[1]
                if word in BULLISH_WORDS:
                    bull -= 1
                elif word in BEARISH_WORDS:
                    bear -= 1
            elif "beat" in phrase or "raised" in phrase:
                bull += 1
            elif "cut" in phrase:
                bear += 1

    bull = max(bull, 0)
    bear = max(bear, 0)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def score_posts(posts: list) -> list:
    """Add sentiment_score to each post in-place."""
    for p in posts:
        text = (p.get("title", "") + " " + p.get("body", ""))
        p["sentiment_score"] = score_text(text)
    return posts


def aggregate_sentiment(posts: list) -> dict:
    """Aggregate sentiment stats across all posts."""
    if not posts:
        return {"mean": 0.0, "bullish_pct": 0, "bearish_pct": 0, "neutral_pct": 100, "total_posts": 0}
    scores = [p.get("sentiment_score", score_text(p.get("title", "") + " " + p.get("body", ""))) for p in posts]
    mean = sum(scores) / len(scores)
    bullish = sum(1 for s in scores if s > 0.08)
    bearish = sum(1 for s in scores if s < -0.08)
    neutral = len(scores) - bullish - bearish
    n = len(scores)
    return {
        "mean": round(mean, 3),
        "bullish_pct": round(bullish / n * 100),
        "bearish_pct": round(bearish / n * 100),
        "neutral_pct": round(neutral / n * 100),
        "total_posts": n,
    }


def per_ticker_sentiment(posts: list, tickers: list) -> dict:
    """Sentiment breakdown for posts mentioning each ticker."""
    result = {}
    for ticker in tickers:
        query = ticker.lower()
        relevant = [
            p for p in posts
            if query in (p.get("title", "") + " " + p.get("body", "")).lower()
            or f"${ticker}" in (p.get("title", "") + " " + p.get("body", ""))
        ]
        result[ticker] = aggregate_sentiment(relevant)
        result[ticker]["mention_count"] = len(relevant)
    return result
