"""
ROK — Static report generator for GitHub Pages.
Runs via GitHub Actions every 6 hours. Outputs docs/index.html.
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


def _safe(fn, *args, default=None, label=""):
    try:
        result = fn(*args)
        logger.info(f"{label or fn.__name__}: ok ({_size(result)})")
        return result
    except Exception as e:
        logger.warning(f"{label or fn.__name__} failed: {e}")
        return default() if callable(default) else default


def _size(v):
    if isinstance(v, (list, dict)):
        return len(v)
    return "ok"


def run():
    from config import Config
    from scrapers import reddit_scraper, news_scraper, yahoo_finance, sec_scraper
    from scrapers import market_data, twitter_scraper
    from scrapers import stocktwits_scraper, technical_analysis, congressional_trades
    from analyzer import ticker_extractor, claude_analyzer, sentiment

    logger.info("=" * 60)
    logger.info("ROK PIPELINE START")
    logger.info("=" * 60)

    # ── Social scraping ──────────────────────────────────────────
    reddit_posts = _safe(
        reddit_scraper.scrape_all,
        Config.REDDIT_SUBREDDITS, Config.REDDIT_MAX_POSTS,
        default=list, label="Reddit",
    )
    news_articles = _safe(
        news_scraper.scrape_all,
        Config.NEWS_FEEDS, Config.NEWS_MAX_ITEMS,
        default=list, label="News",
    )
    twitter_posts = []
    if Config.TWITTER_ENABLED:
        twitter_posts = _safe(
            twitter_scraper.scrape_tweets,
            Config.TWITTER_BEARER_TOKEN,
            default=list, label="Twitter",
        )

    all_posts = reddit_posts + news_articles + twitter_posts

    # ── Market data ───────────────────────────────────────────────
    fear_greed   = _safe(market_data.get_fear_greed_index, default=dict, label="FearGreed")
    earnings_cal = _safe(market_data.get_earnings_calendar, 7, default=list, label="Earnings")
    unusual_opts = _safe(market_data.get_unusual_options_activity, default=list, label="Options")
    most_active  = _safe(market_data.get_most_active_stocks, default=list, label="MostActive")
    short_squeeze= _safe(market_data.get_short_squeeze_candidates, default=list, label="ShortSqueeze")
    market_indices= _safe(market_data.get_market_indices, default=dict, label="Indices")
    trending_yahoo= _safe(market_data.get_trending_on_yahoo, default=list, label="YahooTrending")
    put_call_ratio= _safe(market_data.get_put_call_ratio, default=dict, label="PutCall")
    market_breadth= _safe(market_data.get_market_breadth, default=dict, label="Breadth")

    # ── New data sources ──────────────────────────────────────────
    stocktwits_data = _safe(stocktwits_scraper.get_trending, default=list, label="StockTwits")
    congress_buys   = _safe(
        congressional_trades.get_congress_buys,
        Config.CONGRESS_DAYS_BACK,
        default=list, label="Congress",
    )

    # ── Sentiment + ticker extraction ─────────────────────────────
    all_posts = sentiment.score_posts(all_posts)
    agg_sentiment = sentiment.aggregate_sentiment(all_posts)
    top_tickers = ticker_extractor.top_tickers(all_posts, n=40)

    # Merge Yahoo trending + most active + StockTwits into ticker pool
    extra = set()
    for s in trending_yahoo[:15] + most_active[:15]:
        t = (s.get("ticker") or "").strip().upper()
        if t and t.isalpha() and len(t) <= 5:
            extra.add(t)
    for s in stocktwits_data[:20]:
        t = (s.get("ticker") or "").strip().upper()
        if t and t.isalpha() and len(t) <= 5:
            extra.add(t)
    # Add congressional buy tickers
    for c in congress_buys[:10]:
        extra.add(c["ticker"])

    seen = {t for t, _ in top_tickers}
    seed = yahoo_finance.get_trending_tickers()
    ticker_list = list(dict.fromkeys(
        [t for t, _ in top_tickers]
        + [t for t in extra if t not in seen]
        + [t for t in seed if t not in seen and t not in extra]
    ))[:60]

    # ── Per-ticker sentiment ──────────────────────────────────────
    ticker_sentiment = sentiment.per_ticker_sentiment(all_posts, ticker_list[:30])

    # ── Stock data ────────────────────────────────────────────────
    stock_data = []
    for ticker in ticker_list[:60]:
        data = _safe(yahoo_finance.get_stock_data, ticker, default=lambda: None, label=f"Stock:{ticker}")
        if data:
            data["sentiment"] = ticker_sentiment.get(ticker, {})
            stock_data.append(data)
    logger.info(f"Stock data: {len(stock_data)} tickers")

    # ── Technical analysis ────────────────────────────────────────
    ta_tickers = ticker_list[:Config.TA_MAX_TICKERS]
    ta_data = _safe(
        technical_analysis.analyze_multiple,
        ta_tickers, Config.TA_MAX_TICKERS,
        default=dict, label="TechnicalAnalysis",
    )
    ta_setups = technical_analysis.find_setups(ta_data) if ta_data else []

    # ── SEC filings ───────────────────────────────────────────────
    sec_filings = _safe(
        lambda: sec_scraper.get_recent_insider_trades(7) + sec_scraper.get_recent_8k_filings(7),
        default=list, label="SEC",
    )

    # ── AI Analysis ───────────────────────────────────────────────
    logger.info("Calling Claude AI...")
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
        logger.error("Analysis returned None — aborting")
        sys.exit(1)

    logger.info(
        f"Analysis: {analysis.get('market_sentiment')} | "
        f"Buys: {len(analysis.get('buy_signals', []))} | "
        f"Sells: {len(analysis.get('sell_signals', []))}"
    )

    # ── Build signal lookup for screener ─────────────────────────
    signal_lookup = {}
    for sig in analysis.get("buy_signals", []):
        signal_lookup[sig["ticker"]] = {"type": "buy", "strength": sig.get("signal_strength", 5)}
    for sig in analysis.get("sell_signals", []):
        signal_lookup[sig["ticker"]] = {"type": "sell", "strength": sig.get("signal_strength", 5)}
    for sig in analysis.get("watch_list", []):
        signal_lookup[sig["ticker"]] = {"type": "watch", "strength": 5}

    for stock in stock_data:
        t = stock["ticker"]
        sig = signal_lookup.get(t, {})
        stock["rok_signal"] = sig.get("type", "neutral")
        stock["signal_strength"] = sig.get("strength", 0)

    # ── Enrich signals with price sparklines ──────────────────────
    price_lookup = {s["ticker"]: s["price"] for s in stock_data}
    all_signals = (
        analysis.get("buy_signals", [])
        + analysis.get("sell_signals", [])
        + analysis.get("watch_list", [])
    )
    for sig in all_signals:
        ticker = sig.get("ticker", "")
        if ticker:
            sig["price_history"] = _safe(
                yahoo_finance.get_price_history, ticker, 30,
                default=list, label=f"Hist:{ticker}",
            )

    # ── History tracking ──────────────────────────────────────────
    history_path = Path(__file__).parent / "docs" / "history.json"
    history = {"runs": []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            pass

    history["runs"].append({
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "timestamp": datetime.utcnow().isoformat(),
        "sentiment": analysis.get("market_sentiment"),
        "regime": analysis.get("market_regime"),
        "buy_signals": [
            {"ticker": s["ticker"], "price": s.get("current_price"), "target": s.get("price_target")}
            for s in analysis.get("buy_signals", [])
        ],
        "sell_signals": [
            {"ticker": s["ticker"], "price": s.get("current_price")}
            for s in analysis.get("sell_signals", [])
        ],
    })
    # Keep last 52 runs (1 year of weekly data)
    history["runs"] = history["runs"][-52:]

    # ── Track record: compare last run's picks to current prices ──
    track_record = []
    if len(history["runs"]) >= 2:
        prev_run = history["runs"][-2]
        for sig in prev_run.get("buy_signals", []):
            ticker = sig.get("ticker")
            entry = sig.get("price")
            if not ticker or not entry:
                continue
            current = price_lookup.get(ticker)
            if not current:
                sd = _safe(yahoo_finance.get_stock_data, ticker, default=lambda: None, label=f"Track:{ticker}")
                current = sd["price"] if sd else None
            if current and entry:
                pct = round((current - entry) / entry * 100, 1)
                track_record.append({
                    "ticker": ticker,
                    "entry_price": entry,
                    "current_price": round(current, 2),
                    "pct_change": pct,
                    "target": sig.get("target"),
                    "date": prev_run.get("date", ""),
                })
        track_record.sort(key=lambda x: abs(x["pct_change"]), reverse=True)
        logger.info(f"Track record: {len(track_record)} picks evaluated")

    # ── Build page data ───────────────────────────────────────────
    page_data = {
        "analysis_date": datetime.utcnow().strftime("%B %d, %Y %I:%M %p UTC"),
        "market_regime": analysis.get("market_regime", "UNCERTAIN"),
        "week_summary": analysis.get("week_summary", ""),
        "market_sentiment": analysis.get("market_sentiment", "NEUTRAL"),
        "sentiment_score": analysis.get("sentiment_score", 5),
        "buy_signals": analysis.get("buy_signals", []),
        "sell_signals": analysis.get("sell_signals", []),
        "watch_list": analysis.get("watch_list", []),
        "notable_trends": analysis.get("notable_trends", []),
        "macro_risks": analysis.get("macro_risks", []),
        "sector_heat": analysis.get("sector_heat", {}),
        "sector_rotation": analysis.get("sector_rotation", ""),
        "rok_message": analysis.get("rok_message", ""),
        "short_squeeze_alerts": analysis.get("short_squeeze_alerts", []),
        "earnings_plays": analysis.get("earnings_plays", []),
        "congressional_plays": analysis.get("congressional_plays", []),
        "technical_breakouts": analysis.get("technical_breakouts", []) or [
            {
                "ticker": s["ticker"],
                "setup_type": s["setup_type"],
                "description": " | ".join(s["signals"]),
                "timeframe": "3-7 days",
            }
            for s in ta_setups[:4]
        ],
        "ticker_mentions": top_tickers[:24],
        "stocktwits_trending": stocktwits_data[:12],
        "fear_greed": fear_greed,
        "market_indices": market_indices,
        "aggregate_sentiment": agg_sentiment,
        "market_breadth": market_breadth,
        "put_call_ratio": put_call_ratio,
        "congressional_buys": congress_buys[:8],
        "source_stats": {
            "reddit": len(reddit_posts),
            "news": len(news_articles),
            "stocks": len(stock_data),
            "sec": len(sec_filings),
            "earnings_upcoming": len(earnings_cal),
            "unusual_options": len(unusual_opts),
            "congress_trades": len(congress_buys),
            "technical": len(ta_data),
        },
        "recent_runs": history["runs"][-8:],
        "track_record": track_record,
        "generated_timestamp": datetime.utcnow().isoformat() + "Z",
        "stock_universe": stock_data,
    }

    # ── Render HTML ───────────────────────────────────────────────
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("static.html")
    html = template.render(data=page_data, data_json=json.dumps(page_data))

    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info(f"Report written → docs/index.html")
    logger.info(f"History updated → docs/history.json ({len(history['runs'])} runs)")
    logger.info("ROK PIPELINE COMPLETE")


if __name__ == "__main__":
    run()
