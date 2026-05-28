"""
ROK — Static report generator for GitHub Pages.
Run by GitHub Actions every 6 hours. Outputs docs/index.html.
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import jinja2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run():
    from config import Config
    from scrapers import reddit_scraper, news_scraper, yahoo_finance, sec_scraper
    from scrapers import market_data, twitter_scraper
    from analyzer import ticker_extractor, claude_analyzer, sentiment

    logger.info("ROK pipeline starting...")

    # --- Scrape ---
    reddit_posts = reddit_scraper.scrape_all(Config.REDDIT_SUBREDDITS, Config.REDDIT_MAX_POSTS)
    logger.info(f"Reddit: {len(reddit_posts)} posts")

    news_articles = news_scraper.scrape_all(Config.NEWS_FEEDS, Config.NEWS_MAX_ITEMS)
    logger.info(f"News: {len(news_articles)} articles")

    twitter_posts = []
    if Config.TWITTER_ENABLED:
        twitter_posts = twitter_scraper.scrape_tweets(Config.TWITTER_BEARER_TOKEN)

    all_text_posts = reddit_posts + news_articles + twitter_posts

    # --- Enhanced market data ---
    fear_greed = {}
    earnings_cal = []
    unusual_opts = []
    most_active = []
    short_squeeze = []
    market_indices = {}
    trending_yahoo = []

    for name, fn, args in [
        ("fear_greed", market_data.get_fear_greed_index, []),
        ("earnings_cal", market_data.get_earnings_calendar, [7]),
        ("unusual_opts", market_data.get_unusual_options_activity, []),
        ("most_active", market_data.get_most_active_stocks, []),
        ("short_squeeze", market_data.get_short_squeeze_candidates, []),
        ("market_indices", market_data.get_market_indices, []),
        ("trending_yahoo", market_data.get_trending_on_yahoo, []),
    ]:
        try:
            result = fn(*args)
            locals_ref = {
                "fear_greed": fear_greed, "earnings_cal": earnings_cal,
                "unusual_opts": unusual_opts, "most_active": most_active,
                "short_squeeze": short_squeeze, "market_indices": market_indices,
                "trending_yahoo": trending_yahoo,
            }
            locals_ref[name] = result
            # Re-assign to local vars
            fear_greed = locals_ref["fear_greed"]
            earnings_cal = locals_ref["earnings_cal"]
            unusual_opts = locals_ref["unusual_opts"]
            most_active = locals_ref["most_active"]
            short_squeeze = locals_ref["short_squeeze"]
            market_indices = locals_ref["market_indices"]
            trending_yahoo = locals_ref["trending_yahoo"]
            logger.info(f"{name}: ok")
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    # Re-run properly (avoid the locals hack above)
    try: fear_greed = market_data.get_fear_greed_index()
    except Exception: pass
    try: earnings_cal = market_data.get_earnings_calendar(7)
    except Exception: pass
    try: unusual_opts = market_data.get_unusual_options_activity()
    except Exception: pass
    try: most_active = market_data.get_most_active_stocks()
    except Exception: pass
    try: short_squeeze = market_data.get_short_squeeze_candidates()
    except Exception: pass
    try: market_indices = market_data.get_market_indices()
    except Exception: pass
    try: trending_yahoo = market_data.get_trending_on_yahoo()
    except Exception: pass

    # --- Sentiment + tickers ---
    all_text_posts = sentiment.score_posts(all_text_posts)
    agg_sentiment = sentiment.aggregate_sentiment(all_text_posts)
    top_tickers = ticker_extractor.top_tickers(all_text_posts, n=30)

    extra = set()
    for s in trending_yahoo[:10] + most_active[:10]:
        t = s.get("ticker", "").strip().upper()
        if t and len(t) <= 5:
            extra.add(t)

    seen = {t for t, _ in top_tickers}
    seed = yahoo_finance.get_trending_tickers()
    ticker_list = list(dict.fromkeys(
        [t for t, _ in top_tickers]
        + [t for t in extra if t not in seen]
        + [t for t in seed if t not in seen and t not in extra]
    ))[:30]

    ticker_sentiment = sentiment.per_ticker_sentiment(all_text_posts, ticker_list[:20])

    stock_data = []
    for ticker in ticker_list[:25]:
        data = yahoo_finance.get_stock_data(ticker)
        if data:
            data["sentiment"] = ticker_sentiment.get(ticker, {})
            stock_data.append(data)

    sec_filings = []
    try: sec_filings = sec_scraper.get_recent_insider_trades(7) + sec_scraper.get_recent_8k_filings(7)
    except Exception as e: logger.warning(f"SEC failed: {e}")

    # --- AI Analysis ---
    analysis = claude_analyzer.run_analysis(
        api_key=Config.ANTHROPIC_API_KEY,
        model=Config.CLAUDE_MODEL,
        ticker_mentions=top_tickers,
        reddit_posts=reddit_posts,
        news_articles=news_articles,
        stock_data=stock_data,
        sec_filings=sec_filings,
        fear_greed=fear_greed,
        earnings_calendar=earnings_cal,
        unusual_options=unusual_opts,
        short_squeeze_candidates=short_squeeze,
        market_indices=market_indices,
        aggregate_sentiment=agg_sentiment,
    )

    if not analysis:
        logger.error("Analysis returned None — aborting")
        sys.exit(1)

    # Bundle everything the template needs
    page_data = {
        "analysis_date": datetime.utcnow().strftime("%B %d, %Y %I:%M %p UTC"),
        "week_summary": analysis.get("week_summary", ""),
        "market_sentiment": analysis.get("market_sentiment", "NEUTRAL"),
        "sentiment_score": analysis.get("sentiment_score", 5),
        "buy_signals": analysis.get("buy_signals", []),
        "sell_signals": analysis.get("sell_signals", []),
        "watch_list": analysis.get("watch_list", []),
        "notable_trends": analysis.get("notable_trends", []),
        "sector_heat": analysis.get("sector_heat", {}),
        "rok_message": analysis.get("rok_message", ""),
        "short_squeeze_alerts": analysis.get("short_squeeze_alerts", []),
        "earnings_plays": analysis.get("earnings_plays", []),
        "ticker_mentions": top_tickers[:24],
        "fear_greed": fear_greed,
        "market_indices": market_indices,
        "aggregate_sentiment": agg_sentiment,
        "source_stats": {
            "reddit": len(reddit_posts),
            "news": len(news_articles),
            "stocks": len(stock_data),
            "sec": len(sec_filings),
            "earnings_upcoming": len(earnings_cal),
            "unusual_options": len(unusual_opts),
        },
    }

    # --- Render HTML ---
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("static.html")
    html = template.render(data=page_data, data_json=json.dumps(page_data))

    out = Path(__file__).parent / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {out}")
    logger.info(f"Sentiment: {page_data['market_sentiment']} | Buys: {len(page_data['buy_signals'])} | Sells: {len(page_data['sell_signals'])}")


if __name__ == "__main__":
    run()
