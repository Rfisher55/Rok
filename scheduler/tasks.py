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
        from scrapers import reddit_scraper, news_scraper, yahoo_finance, sec_scraper, twitter_scraper
        from scrapers import market_data
        from analyzer import ticker_extractor, claude_analyzer, sentiment

        logger.info("ROK pipeline starting...")

        # --- Scrape all sources ---
        reddit_posts = reddit_scraper.scrape_all(
            Config.REDDIT_SUBREDDITS, Config.REDDIT_MAX_POSTS
        )
        news_articles = news_scraper.scrape_all(Config.NEWS_FEEDS, Config.NEWS_MAX_ITEMS)

        twitter_posts = []
        if Config.TWITTER_ENABLED:
            twitter_posts = twitter_scraper.scrape_tweets(Config.TWITTER_BEARER_TOKEN)

        all_text_posts = reddit_posts + news_articles + twitter_posts

        # --- Enhanced market data (non-blocking) ---
        fear_greed = {}
        earnings_cal = []
        unusual_opts = []
        most_active = []
        short_squeeze = []
        market_indices = {}
        trending_yahoo = []

        try:
            fear_greed = market_data.get_fear_greed_index()
            logger.info(f"Fear/Greed: {fear_greed.get('score')} ({fear_greed.get('rating')})")
        except Exception as e:
            logger.warning(f"Fear/Greed failed: {e}")

        try:
            earnings_cal = market_data.get_earnings_calendar(days_ahead=7)
            logger.info(f"Earnings calendar: {len(earnings_cal)} upcoming")
        except Exception as e:
            logger.warning(f"Earnings calendar failed: {e}")

        try:
            unusual_opts = market_data.get_unusual_options_activity()
            logger.info(f"Unusual options: {len(unusual_opts)} entries")
        except Exception as e:
            logger.warning(f"Unusual options failed: {e}")

        try:
            most_active = market_data.get_most_active_stocks()
            logger.info(f"Most active: {len(most_active)} stocks")
        except Exception as e:
            logger.warning(f"Most active failed: {e}")

        try:
            short_squeeze = market_data.get_short_squeeze_candidates()
            logger.info(f"Short squeeze candidates: {len(short_squeeze)}")
        except Exception as e:
            logger.warning(f"Short squeeze failed: {e}")

        try:
            market_indices = market_data.get_market_indices()
            logger.info(f"Market indices: {list(market_indices.keys())}")
        except Exception as e:
            logger.warning(f"Market indices failed: {e}")

        try:
            trending_yahoo = market_data.get_trending_on_yahoo()
            logger.info(f"Yahoo trending: {len(trending_yahoo)}")
        except Exception as e:
            logger.warning(f"Yahoo trending failed: {e}")

        # --- Sentiment analysis ---
        all_text_posts = sentiment.score_posts(all_text_posts)
        agg_sentiment = sentiment.aggregate_sentiment(all_text_posts)

        # --- Extract trending tickers ---
        top_tickers = ticker_extractor.top_tickers(all_text_posts, n=30)

        # Add Yahoo trending + most active to ticker pool
        extra_tickers = set()
        for s in trending_yahoo[:10]:
            t = s.get("ticker", "").strip().upper()
            if t and len(t) <= 5:
                extra_tickers.add(t)
        for s in most_active[:10]:
            t = s.get("ticker", "").strip().upper()
            if t and len(t) <= 5:
                extra_tickers.add(t)

        seen = {t for t, _ in top_tickers}
        seed_tickers = yahoo_finance.get_trending_tickers()
        all_candidate_tickers = (
            [t for t, _ in top_tickers]
            + [t for t in extra_tickers if t not in seen]
            + [t for t in seed_tickers if t not in seen and t not in extra_tickers]
        )

        ticker_list = list(dict.fromkeys(all_candidate_tickers))[:30]

        # --- Per-ticker sentiment ---
        ticker_sentiment = sentiment.per_ticker_sentiment(all_text_posts, ticker_list[:20])

        # --- Fetch stock data ---
        stock_data = []
        for ticker in ticker_list[:25]:
            data = yahoo_finance.get_stock_data(ticker)
            if data:
                # Enrich with sentiment
                sent = ticker_sentiment.get(ticker, {})
                data["sentiment"] = sent
                stock_data.append(data)

        # --- SEC filings ---
        sec_filings = sec_scraper.get_recent_insider_trades(days_back=7)
        sec_filings += sec_scraper.get_recent_8k_filings(days_back=7)

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
            logger.error("Analysis failed — skipping DB write")
            return

        # --- Persist ---
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
            }),
        )
        db.session.add(record)
        db.session.commit()
        logger.info(f"ROK analysis saved (id={record.id})")
