"""
ROK — Static report generator for GitHub Pages.
Runs via GitHub Actions every 15 minutes. Outputs docs/index.html.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
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
    fear_greed    = _safe(market_data.get_fear_greed_index, default=dict, label="FearGreed")
    earnings_cal  = _safe(market_data.get_earnings_calendar, 7, default=list, label="Earnings")
    unusual_opts  = _safe(market_data.get_unusual_options_activity, default=list, label="Options")
    most_active   = _safe(market_data.get_most_active_stocks, default=list, label="MostActive")
    short_squeeze = _safe(market_data.get_short_squeeze_candidates, default=list, label="ShortSqueeze")
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
    seed = _safe(yahoo_finance.get_trending_tickers, default=list, label="YahooTickers")
    ticker_list = list(dict.fromkeys(
        [t for t, _ in top_tickers]
        + [t for t in extra if t not in seen]
        + [t for t in (seed or []) if t not in seen and t not in extra]
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
    insider_buys = _safe(sec_scraper.get_insider_buys, 14, default=list, label="InsiderBuys")

    # ── Load history (for last_analysis fallback) ─────────────────
    history_path = Path(__file__).parent / "docs" / "history.json"
    history = {"runs": []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            pass

    # ── AI Analysis ───────────────────────────────────────────────
    logger.info("Calling Claude AI...")
    analysis = None

    # Only call Claude if API key is present
    if Config.ANTHROPIC_API_KEY:
        analysis = _safe(
            lambda: claude_analyzer.run_analysis(
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
                insider_buys=insider_buys,
            ),
            default=None,
            label="ClaudeAI",
        )
    else:
        logger.warning("ANTHROPIC_API_KEY not set — skipping AI analysis, using cached data")

    # ── Fallback: reuse last successful analysis from history ─────
    if not analysis:
        cached = history.get("last_analysis")
        if cached:
            logger.info("Using cached last_analysis from history.json")
            analysis = cached
        else:
            logger.error("No analysis and no cached analysis available — aborting")
            sys.exit(1)
    else:
        # Save successful analysis for future fallback
        history["last_analysis"] = analysis
        logger.info("Saved analysis to history.last_analysis for fallback")

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
    stock_data_lookup = {s["ticker"]: s for s in stock_data if s}
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

    # ── Enrich signals with Yahoo Finance data ────────────────────
    for sig in all_signals:
        t = sig.get("ticker", "")
        if not t:
            continue
        sd = stock_data_lookup.get(t)
        if sd:
            # Fill missing price fields
            if not sig.get("current_price") and sd.get("price"):
                sig["current_price"] = sd["price"]
            if not sig.get("company") and sd.get("company_name"):
                sig["company"] = sd["company_name"]
            # Use analyst consensus target if AI didn't set one
            if not sig.get("price_target") and sd.get("analyst_target"):
                sig["price_target"] = sd["analyst_target"]
            # Auto-calculate stop loss at 8% below entry if not set
            if not sig.get("stop_loss") and sig.get("current_price"):
                sig["stop_loss"] = round(sig["current_price"] * 0.92, 2)
            if not sig.get("sector"):
                sig["sector"] = sd.get("sector", "")
            # Volume spike and RSI — prefer TA data (60d window) over yahoo 30d
            ta = ta_data.get(t, {}) if ta_data else {}
            if not sig.get("vol_ratio"):
                sig["vol_ratio"] = ta.get("volume_ratio") or sd.get("vol_ratio")
            if not sig.get("rsi"):
                sig["rsi"] = ta.get("rsi") or sd.get("rsi")
            # 52-week position data
            if ta:
                sig["pct_from_52w_high"] = ta.get("pct_from_52w_high")
                sig["week_52_high"] = ta.get("week_52_high")
                sig["macd_signal"] = ta.get("macd_signal_label")
            # Analyst consensus data
            if not sig.get("analyst_count") and sd.get("analyst_count"):
                sig["analyst_count"] = sd["analyst_count"]
            if not sig.get("recommendation") and sd.get("recommendation"):
                sig["recommendation"] = sd["recommendation"]
            # Earnings date warning
            if not sig.get("earnings_date") and sd.get("earnings_date"):
                sig["earnings_date"] = sd["earnings_date"]
            # Build data_signals from available sources
            if not sig.get("data_signals"):
                dsigs = []
                # Check congressional buys
                if any(c.get("ticker") == t for c in congress_buys[:20]):
                    dsigs.append("congressional")
                # Check SEC filings or insider buys for insider trades
                if any(
                    isinstance(f, dict) and f.get("ticker") == t
                    for f in (sec_filings or []) + (insider_buys or [])
                ):
                    dsigs.append("insider")
                # Check unusual options
                if any(
                    isinstance(o, dict) and o.get("ticker") == t
                    for o in (unusual_opts or [])
                ):
                    dsigs.append("options")
                # Reddit presence
                reddit_tickers = [tp[0] for tp in (top_tickers or [])]
                if t in reddit_tickers[:20]:
                    dsigs.append("reddit")
                sig["data_signals"] = dsigs
        # Apply defaults for missing fields
        if not sig.get("signal_strength"):
            sig["signal_strength"] = 6
        if not sig.get("risk_level"):
            # Calculate risk level from beta and market cap
            beta = sd.get("beta") if sd else None
            mc = sd.get("market_cap") if sd else None
            if beta and beta > 1.8:
                sig["risk_level"] = "High"
            elif beta and beta < 0.7:
                sig["risk_level"] = "Low"
            elif mc and mc >= 100_000_000_000:  # $100B+ = mega cap = lower risk
                sig["risk_level"] = "Low"
            elif mc and mc < 2_000_000_000:  # <$2B = small cap = higher risk
                sig["risk_level"] = "High"
            else:
                sig["risk_level"] = "Medium"
        if not sig.get("time_horizon"):
            sig["time_horizon"] = "1-3 months"
    logger.info("Signal enrichment complete")

    # ── History tracking ──────────────────────────────────────────
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
    # Keep last 96 runs (24 hours of 15-min data × 4)
    history["runs"] = history["runs"][-96:]

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

    # ── Compute history bullish summary ──────────────────────────
    recent_runs = history["runs"][-8:]
    bullish_count = sum(1 for r in recent_runs if (r.get("sentiment") or "").upper() == "BULLISH")
    history_summary = f"ROK was bullish {bullish_count} out of the last {len(recent_runs)} runs"

    # ── Plain-language market mood ────────────────────────────────
    sentiment = (analysis.get("market_sentiment") or "NEUTRAL").upper()
    fg_score = (fear_greed or {}).get("score", 50)
    buy_count_now = len(analysis.get("buy_signals", []))
    if sentiment == "BULLISH" and buy_count_now >= 5:
        market_mood = f"🟢 Markets are looking strong — ROK found {buy_count_now} stocks worth buying right now"
    elif sentiment == "BULLISH":
        market_mood = f"🟢 Markets are leaning bullish — ROK sees some opportunities"
    elif sentiment == "BEARISH":
        market_mood = "🔴 Markets are under pressure — ROK recommends being careful"
    elif fg_score and fg_score < 30:
        market_mood = "🟡 Fear is high but that often means buying opportunities are near"
    else:
        market_mood = "🟡 Markets are mixed — ROK is watching closely for clear signals"

    # ── Build page data ───────────────────────────────────────────
    buy_count = len(analysis.get("buy_signals", []))
    sell_count = len(analysis.get("sell_signals", []))

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
        "short_squeeze_alerts": analysis.get("short_squeeze_alerts", []) or [
            {
                "ticker": s.get("ticker", ""),
                "short_float": "High short interest",
                "setup": s.get("company", ""),
                "social_velocity": "high",
            }
            for s in (short_squeeze or [])[:6]
            if s.get("ticker") and s.get("ticker").isalpha() and len(s.get("ticker","")) <= 5
        ],
        "earnings_plays": analysis.get("earnings_plays", []) or [
            {
                "ticker": e.get("ticker", ""),
                "earnings_date": e.get("date", ""),
                "direction": "NEUTRAL",
                "play": f"Reports {e.get('timing', 'soon')}. Watch for surprise beats/misses.",
            }
            for e in (earnings_cal or [])[:6]
            if e.get("ticker")
        ],
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
            "insider_buys": len(insider_buys),
        },
        "recent_runs": recent_runs,
        "history_summary": history_summary,
        "market_mood": market_mood,
        "track_record": track_record,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "stock_universe": stock_data,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "insider_buys": insider_buys[:12],
        # Backwards-compat aliases used in template
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "run_history": history["runs"][-8:],
        "news_items": [
            {
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "sentiment": a.get("sentiment_score", 0),
                "tickers": a.get("mentioned_tickers", []),
            }
            for a in news_articles[:20]
            if a.get("title")
        ],
        "reddit_posts": reddit_posts[:10],
        "trending_tickers": [{"ticker": t, "count": c} for t, c in top_tickers[:15]],
        "portfolio": [],
        "sparklines": {},
        "fear_greed_history": [],
        "market_pulse": {
            "vix": (market_indices or {}).get("VIX", {}).get("price"),
            "put_call_ratio": (put_call_ratio or {}).get("total"),
            "sp500_change": (market_indices or {}).get("S&P 500", {}).get("change_pct"),
            "nasdaq_change": (market_indices or {}).get("NASDAQ", {}).get("change_pct"),
        },
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

    # ── Write prices.json for JS fallback ─────────────────────────
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prices_dict = {}
    for s in stock_data:
        t = s.get("ticker")
        if t:
            prices_dict[t] = {
                "price": s.get("price"),
                "change_pct": s.get("change_pct"),
                "updated": now_iso,
            }
    (docs_dir / "prices.json").write_text(json.dumps(prices_dict), encoding="utf-8")

    logger.info(f"Report written → docs/index.html ({len(html)} chars)")
    logger.info(f"Prices written → docs/prices.json ({len(prices_dict)} tickers)")
    logger.info(f"History updated → docs/history.json ({len(history['runs'])} runs)")
    logger.info(f"Summary: {buy_count} buys | {sell_count} sells")
    logger.info("ROK PIPELINE COMPLETE")


if __name__ == "__main__":
    run()
