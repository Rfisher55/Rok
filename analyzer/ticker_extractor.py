import re

# Common words that look like tickers but aren't
BLACKLIST = {
    "A", "I", "AM", "AN", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN",
    "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "RE", "SO", "TO",
    "UP", "US", "WE", "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "GET", "GOT",
    "HAS", "HAD", "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NEW", "NOT",
    "NOW", "OFF", "OLD", "ONE", "OUR", "OUT", "OWN", "PUT", "SAY", "SHE",
    "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO", "WHY", "YET", "YOU",
    "YOUR", "YOLO", "LMAO", "LOL", "OMG", "IMO", "DD", "OTC", "IPO", "ETF",
    "CEO", "CFO", "SEC", "FDA", "FED", "GDP", "ATH", "ATL", "EPS", "PE",
    "NOW", "IIRC", "TLDR", "FOMO", "FUD", "HODL", "MOON", "PUMP", "DUMP",
    "BEAR", "BULL", "CALL", "PUTS", "LONG", "SHORT", "SELL", "BUYS", "HOLD",
    "RISK", "HIGH", "LOW", "GOOD", "BEST", "NEXT", "LAST", "YEAR", "WEEK",
    "NEWS", "NEED", "WANT", "MAKE", "MUCH", "MORE", "LESS", "EVEN", "ALSO",
    "BACK", "THAN", "SOME", "INTO", "OVER", "LIKE", "JUST", "BEEN", "THEY",
    "SAID", "FROM", "WITH", "HAVE", "WILL", "THAT", "THIS", "WHAT", "WHEN",
    "WHERE", "THERE", "COULD", "WOULD", "SHOULD", "MIGHT", "ABOUT", "AFTER",
    "AGAIN", "BEING", "GOING", "DOING", "EVERY", "FIRST", "GIVEN", "GREAT",
    "LARGE", "LATER", "LIGHT", "MONEY", "NEVER", "NIGHT", "OFTEN", "OTHER",
    "PLACE", "PRICE", "QUICK", "QUITE", "READY", "RIGHT", "ROUND", "SMALL",
    "STILL", "STOCK", "THINK", "THOSE", "THREE", "UNDER", "UNTIL", "USING",
    "VALUE", "WATCH", "WHICH", "WHILE", "WHOLE", "WORLD", "WORTH", "WRITE",
    "YEAH", "PAST", "PART", "PLAY", "PLAN",
}

# Regex: $TICKER or ALL-CAPS 1-5 letter word
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
CAPS_RE = re.compile(r"\b([A-Z]{2,5})\b")


def extract_tickers(text: str) -> list[str]:
    """Extract likely stock tickers from text."""
    text = text or ""
    found = set()

    # $TICKER mentions are highest confidence
    for match in CASHTAG_RE.finditer(text):
        t = match.group(1).upper()
        if t not in BLACKLIST:
            found.add(t)

    # ALL-CAPS words (lower confidence, filter aggressively)
    upper_text = re.sub(r"[^A-Z\s]", " ", text.upper())
    for match in CAPS_RE.finditer(upper_text):
        t = match.group(1)
        if t not in BLACKLIST and len(t) >= 2:
            found.add(t)

    return list(found)


def count_tickers(posts: list[dict]) -> dict[str, int]:
    """Count how many times each ticker is mentioned across posts."""
    counts: dict[str, int] = {}
    for post in posts:
        combined = (post.get("title", "") + " " + post.get("body", ""))
        tickers = extract_tickers(combined)
        for t in tickers:
            counts[t] = counts.get(t, 0) + 1
    return counts


def top_tickers(posts: list[dict], n: int = 30) -> list[tuple[str, int]]:
    """Return the top N most-mentioned tickers sorted by count."""
    counts = count_tickers(posts)
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
