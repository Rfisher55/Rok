import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def run_scrape_and_analyze(app):
    """Full pipeline: scrape → extract → sentiment → analyze → store."""
    with app.app_context():
        from config import Config
        from app_factory import db
        from database.models import WeeklyAnalysis
        from scrapers import (
            reddit_scraper, news_scraper, yahoo_finance, sec_scraper,
            twitter_scraper, market_data,
            stocktwits_scraper, technical_analysis, congressional_trades,
        )
        from analyzer import ticker_extractor, claude_analyzer, sentiment

        logger.info("ROK pipeline starting...")

        # ── Social ────────────────────────────────────────────────
        reddit_posts = reddit_scraper.scrape_all(Config.REDDIT_SUBREDDITS, Config.REDDIT_MAX_POSTS)
        news_articles = news_scraper.scrape_all(Config.NEWS_FEEDS, Config.NEWS_MAX_ITEMS)
        twitter_posts = []
        if Config.TWITTER_ENABLED:
            try:
                twitter_posts = twitter_scraper.scrape_tweets(Config.TWITTER_BEARER_TOKEN)
            except Exception as e:
                logger.warning(f"Twitter: {e}")

        all_posts = reddit_posts + news_articles + twitter_posts

        # ── Market data ───────────────────────────────────────────
        def _try(fn, *args, default=None):
            try:
                return fn(*args)
            except Exception as e:
                logger.warning(f"{fn.__name__}: {e}")
                return default() if callable(default) else default

        fear_greed    = _try(market_data.get_fear_greed_index, default=dict)
        earnings_cal  = _try(market_data.get_earnings_calendar, 7, default=list)
        unusual_opts  = _try(market_data.get_unusual_options_activity, default=list)
        most_active   = _try(market_data.get_most_active_stocks, default=list)
        short_squeeze = _try(market_data.get_short_squeeze_candidates, default=list)
        market_indices= _try(market_data.get_market_indices, default=dict)
        trending_yahoo= _try(market_data.get_trending_on_yahoo, default=list)
        put_call_ratio= _try(market_data.get_put_call_ratio, default=dict)
        market_breadth= _try(market_data.get_market_breadth, default=dict)

        # ── New sources ───────────────────────────────────────────
        stocktwits_data = _try(stocktwits_scraper.get_trending, default=list)
        congress_buys   = _try(congressional_trades.get_congress_buys, Config.CONGRESS_DAYS_BACK, default=list)

        # ── Sentiment + tickers ───────────────────────────────────
        all_posts = sentiment.score_posts(all_posts)
        agg_sentiment = sentiment.aggregate_sentiment(all_posts)
        top_tickers = ticker_extractor.top_tickers(all_posts, n=40)

        extra = set()
        for s in trending_yahoo[:15] + most_active[:15]:
            t = (s.get("ticker") or "").strip().upper()
            if t and t.isalpha() and len(t) <= 5:
                extra.add(t)
        for s in stocktwits_data[:20]:
            t = (s.get("ticker") or "").strip().upper()
            if t and t.isalpha() and len(t) <= 5:
                extra.add(t)
        for c in congress_buys[:10]:
            extra.add(c["ticker"])

        seen = {t for t, _ in top_tickers}
        seed = yahoo_finance.get_trending_tickers()
        ticker_list = list(dict.fromkeys(
            [t for t, _ in top_tickers]
            + [t for t in extra if t not in seen]
            + [t for t in seed if t not in seen and t not in extra]
        ))[:40]

        ticker_sentiment = sentiment.per_ticker_sentiment(all_posts, ticker_list[:25])

        stock_data = []
        for ticker in ticker_list[:30]:
            data = _try(yahoo_finance.get_stock_data, ticker)
            if data:
                data["sentiment"] = ticker_sentiment.get(ticker, {})
                stock_data.append(data)

        # ── Technical analysis ────────────────────────────────────
        ta_data = _try(technical_analysis.analyze_multiple, ticker_list[:Config.TA_MAX_TICKERS], Config.TA_MAX_TICKERS, default=dict)
        ta_setups = technical_analysis.find_setups(ta_data or {})

        sec_filings = _try(
            lambda: sec_scraper.get_recent_insider_trades(7) + sec_scraper.get_recent_8k_filings(7),
            default=list,
        )

        # ── AI analysis ───────────────────────────────────────────
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
            stocktwits_trending=stocktwits_data,
            technical_data=ta_data,
            congressional_buys=congress_buys,
            market_breadth=market_breadth,
            put_call_ratio=put_call_ratio,
        )

        if not analysis:
            logger.error("Analysis failed — skipping DB write")
            return

        record = WeeklyAnalysis(
            analysis_date=datetime.utcnow(),
            week_summary=analysis.get("week_summary", ""),
            buy_signals=json.dumps(analysis.get("buy_signals", [])),
            sell_signals=json.dumps(analysis.get("sell_signals", [])),
            watch_list=json.dumps(analysis.get("watch_list", [])),
            market_sentiment=analysis.get("market_sentiment", "NEUTRAL"),
            notable_trends=json.dumps(analysis.get("notable_trends", [])),
            raw_data_summary=json.dumps({
                "ticker_mentions": top_tickers[:20],
                "reddit_post_count": len(reddit_posts),
                "news_article_count": len(news_articles),
                "stock_data_count": len(stock_data),
                "sec_filing_count": len(sec_filings),
                "sentiment_score": analysis.get("sentiment_score", 5),
                "sector_heat": analysis.get("sector_heat", {}),
                "rok_message": analysis.get("rok_message", ""),
                "fear_greed": fear_greed,
                "market_indices": market_indices,
                "aggregate_sentiment": agg_sentiment,
                "earnings_upcoming": len(earnings_cal),
                "unusual_options_count": len(unusual_opts),
                "congress_buys_count": len(congress_buys),
                "market_regime": analysis.get("market_regime", ""),
                "macro_risks": analysis.get("macro_risks", []),
                "congressional_plays": analysis.get("congressional_plays", []),
                "technical_breakouts": analysis.get("technical_breakouts", []),
                "put_call_ratio": put_call_ratio,
                "market_breadth": market_breadth,
            }),
        )
        db.session.add(record)
        db.session.commit()
        logger.info(f"ROK analysis saved (id={record.id})")
