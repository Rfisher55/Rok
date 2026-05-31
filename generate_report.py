"""
ROK — Market intelligence pipeline for GitHub Pages.
Runs via GitHub Actions every 15 minutes.
Writes docs/intel_report.json (read by the trading dashboard via JS fetch).
Does NOT overwrite docs/index.html — the trading dashboard owns that file.
"""
import json
import logging
import sys
from datetime import datetime, timezone, date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class _Encoder(json.JSONEncoder):
    """Handle datetime/date objects that scrapers sometimes return."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


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


def _sanitize(obj):
    """Recursively convert any datetime objects to ISO strings so JSON serialization never fails."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    return obj


def run():
    try:
        _run()
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        # Always write ALL three output files so git add never fails on missing paths.
        docs_dir = Path(__file__).parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        fallback = {
            "generated_at": now,
            "error": str(e),
            "market_sentiment": "UNKNOWN",
            "buy_signals": [],
            "sell_signals": [],
            "watch_list": [],
            "notable_trends": [],
            "rok_message": "Intelligence update unavailable — will retry shortly.",
        }
        (docs_dir / "intel_report.json").write_text(
            json.dumps(fallback, cls=_Encoder, indent=2), encoding="utf-8"
        )
        # Write history.json stub if it doesn't exist yet
        history_path = docs_dir / "history.json"
        if not history_path.exists():
            history_path.write_text(json.dumps({"runs": []}, indent=2), encoding="utf-8")
        # Write prices.json stub if it doesn't exist yet
        prices_path = docs_dir / "prices.json"
        if not prices_path.exists():
            prices_path.write_text(json.dumps({}), encoding="utf-8")
        logger.info("Wrote fallback output files")


def _build_weekly_bot_report(docs_dir):
    """Build a weekly performance summary from trades.json and equity.json."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    trades_path = docs_dir / "trades.json"
    equity_path = docs_dir / "equity.json"

    if not trades_path.exists():
        return None

    try:
        td = json.loads(trades_path.read_text())
    except Exception:
        return None

    all_trades = td.get("trades", [])
    lp = td.get("bot_learned_params", {})
    neurons_total = td.get("neurons_total", 90)
    neurons_active = td.get("neurons_active", 0)

    # Filter to this week's closed trades (SELL / COVER actions with pnl)
    week_trades = []
    for t in all_trades:
        if t.get("action") not in ("SELL", "SELL_HALF", "COVER"):
            continue
        ts = t.get("timestamp") or t.get("time") or ""
        try:
            trade_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except Exception:
            trade_dt = None
        if trade_dt and trade_dt >= week_ago:
            week_trades.append(t)

    wins = [t for t in week_trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in week_trades if (t.get("pnl") or 0) <= 0]
    total_pnl = round(sum(t.get("pnl", 0) or 0 for t in week_trades), 2)
    win_rate = round(len(wins) / len(week_trades) * 100, 1) if week_trades else 0
    avg_pnl = round(total_pnl / len(week_trades), 2) if week_trades else 0

    # Best and worst trades this week
    sorted_by_pnl = sorted(week_trades, key=lambda t: t.get("pnl", 0) or 0, reverse=True)
    best_trades = [{"ticker": t.get("ticker"), "pnl": t.get("pnl")} for t in sorted_by_pnl[:3]]
    worst_trades = [{"ticker": t.get("ticker"), "pnl": t.get("pnl")} for t in sorted_by_pnl[-3:] if (t.get("pnl") or 0) < 0]

    # Top active neurons (those with learned data)
    top_neurons = []
    neuron_map = {
        "vix_entry_perf": "N103 VIX Bracket",
        "entry_session_perf": "N104 Session Quality",
        "breadth_entry_perf": "N105 Market Breadth",
        "trend_template_tier_perf": "N108 Trend Template",
        "rvol_entry_tier_perf": "N109 RVOL Tier",
        "spy_vwap_entry_perf": "N111 SPY VWAP",
        "signal_density_perf": "N112 Signal Density",
        "ai_sentiment_tier_perf": "N113 AI Sentiment",
        "hold_duration_perf": "N114 Hold Duration",
        "mktcap_tier_perf": "N115 Market Cap",
        "vts_perf": "N117 VIX Term Structure",
        "macro_hold_perf": "N119 Macro Events",
        "pcr_entry_perf": "N120 Options PCR",
        "si_squeeze_perf": "N121 Short Squeeze",
        "dist_200ema_perf": "N122 200 EMA Dist",
        "sector_etf_strength_perf": "N123 Sector ETF",
        "spy_alignment_perf": "N124 SPY Alignment",
        "news_velocity_perf": "N125 News Velocity",
        "gap_entry_perf": "N126 Gap Entry",
        "rs_tier_entry_perf": "N127 RS Rating",
        "entry_score_tier_perf": "N128 Score Tier",
        "exit_trigger_perf": "N129 Exit Trigger",
        "stock_stability_perf": "N130 Stability",
        "sector_type_perf": "N147 Sector Type",
        "cap_style_perf": "N148 Cap Style",
        "futures_signal_perf": "N149 Futures Signal",
        "exit_hour_perf": "N150 Exit Hour",
        "entry_dow_perf": "N151 Day of Week",
        "vix_trend_perf": "N152 VIX Trend",
        "crowd_tier_perf": "N153 Port Crowding",
        "sector_50d_trend_perf": "N154 Sector 50d",
        "short_int_perf": "N155 Short Interest",
        "regime_duration_perf": "N156 Regime Duration",
        "orb_quality_perf": "N157 ORB Quality",
        "catalyst_type_perf": "N158 Catalyst Type",
        "spy_52wh_zone_perf": "N159 SPY 52w Zone",
        "breakout_age_perf": "N160 Breakout Age",
        "dollar_vol_perf": "N161 Dollar Volume",
        "streak_state_perf": "N162 Bot Streak",
        "score_regime_align_perf": "N163 Score-Regime",
        "sector_mom_accel_perf": "N164 Sector Accel",
        "market_correl_perf": "N165 Mkt Correlation",
        "estimate_revision_perf": "N166 Est Revisions",
        "news_sent_mom_perf": "N167 Sentiment Mom",
        "tech_confluence_perf": "N168 Tech Confluence",
        "breadth_direction_perf": "N169 Breadth Dir",
        "risk_rotation_perf": "N170 Risk Rotation",
    }
    for key, label in neuron_map.items():
        data = lp.get(key, [])
        if not isinstance(data, list) or not data:
            continue
        best = max(data, key=lambda x: x.get("win_rate", 50), default=None)
        if best:
            top_neurons.append({
                "neuron": label,
                "best_state": best.get("state", "?"),
                "win_rate": best.get("win_rate", 50),
                "samples": best.get("total", 0),
            })
    top_neurons.sort(key=lambda x: x["win_rate"], reverse=True)

    # Equity curve this week (from equity.json)
    equity_week = []
    try:
        if equity_path.exists():
            eq = json.loads(equity_path.read_text())
            snapshots = eq.get("snapshots", [])
            for snap in snapshots:
                snap_dt_str = snap.get("date") or snap.get("timestamp", "")
                try:
                    snap_dt = datetime.fromisoformat(snap_dt_str.replace("Z", "+00:00")) if snap_dt_str else None
                except Exception:
                    snap_dt = None
                if snap_dt and snap_dt >= week_ago:
                    equity_week.append({
                        "date": snap.get("date") or snap_dt_str[:10],
                        "equity": snap.get("equity"),
                        "spy_benchmark": snap.get("spy_benchmark"),
                    })
    except Exception:
        pass

    # Learn log highlights
    learn_log = lp.get("learn_log", [])[-10:]

    return {
        "generated_at": now.isoformat(),
        "period": "Last 7 days",
        "week_start": week_ago.strftime("%Y-%m-%d"),
        "week_end": now.strftime("%Y-%m-%d"),
        "trades_total": len(week_trades),
        "trades_wins": len(wins),
        "trades_losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "neurons_active": neurons_active,
        "neurons_total": neurons_total,
        "top_neurons": top_neurons[:10],
        "equity_curve": equity_week,
        "learn_log": learn_log,
        "strategy_mode": td.get("strategy_mode", "SELECTIVE"),
        "recovery_mode": td.get("recovery_mode", False),
        "effective_min_score": td.get("effective_min_score"),
        "cross_asset_risk_off": td.get("cross_asset_risk_off", False),
        "portfolio_value": td.get("portfolio_value"),
        "drawdown_pct": td.get("drawdown_pct"),
        "profit_factor": td.get("profit_factor"),
    }


def _run():
    from config import Config
    from scrapers import reddit_scraper, news_scraper, yahoo_finance, sec_scraper
    from scrapers import market_data, twitter_scraper
    from scrapers import stocktwits_scraper, technical_analysis, congressional_trades
    from analyzer import ticker_extractor, claude_analyzer
    from analyzer import sentiment as sentiment_mod

    logger.info("=" * 60)
    logger.info("ROK INTELLIGENCE PIPELINE START")
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
    all_posts = sentiment_mod.score_posts(all_posts)
    agg_sentiment = sentiment_mod.aggregate_sentiment(all_posts)
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
    seed = _safe(yahoo_finance.get_trending_tickers, default=list, label="YahooTickers")
    ticker_list = list(dict.fromkeys(
        [t for t, _ in top_tickers]
        + [t for t in extra if t not in seen]
        + [t for t in (seed or []) if t not in seen and t not in extra]
    ))[:60]

    ticker_sentiment = sentiment_mod.per_ticker_sentiment(all_posts, ticker_list[:30])

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

    # ── Load history ──────────────────────────────────────────────
    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    history_path = docs_dir / "history.json"
    history = {"runs": []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            pass

    # ── Load live trading data (positions + scan top + market context) ─
    trades_path = docs_dir / "trades.json"
    current_positions = []
    last_scan_top = []
    live_market_context = {}
    try:
        if trades_path.exists():
            td = json.loads(trades_path.read_text())
            current_positions = td.get("positions", [])
            last_scan_top     = td.get("last_scan_top", [])
            # Extract live market context for richer AI prompt
            # Extract RS leaders and EMA21 setups from last scan
            _rs_leaders = sorted(
                [e for e in last_scan_top if (e.get("rs_rating") or 50) >= 80],
                key=lambda e: -(e.get("rs_rating") or 50)
            )[:6]
            _ema21_setups = [e["ticker"] for e in last_scan_top if e.get("ema21_pullback")][:5]
            _pocket_pivots = [e["ticker"] for e in last_scan_top if e.get("pocket_pivot")][:5]
            _htf_stocks = sorted(
                [e for e in last_scan_top if e.get("htf")],
                key=lambda e: -(e.get("htf_consec") or 0)
            )[:4]
            _tt8_stocks = [e["ticker"] for e in last_scan_top if e.get("tt_full")][:5]
            _tt_leaders = sorted(
                [e for e in last_scan_top if (e.get("trend_template") or 0) >= 6],
                key=lambda e: -(e.get("trend_template") or 0)
            )[:6]
            live_market_context = {
                "market_open":     td.get("market_open"),
                "timing_quality":  td.get("timing_quality"),
                "day_type":        td.get("day_type"),
                "day_efficiency":  td.get("day_efficiency"),
                "strategy_hint":   td.get("strategy_hint"),
                "vts_regime":      td.get("regime", {}).get("vts_regime"),
                "dxy_level":       td.get("regime", {}).get("dxy_level"),
                "dxy_5d":          td.get("regime", {}).get("dxy_5d"),
                "tnx_level":       td.get("regime", {}).get("tnx_level"),
                "tnx_5d":          td.get("regime", {}).get("tnx_5d"),
                "rate_environment": td.get("regime", {}).get("rate_environment"),
                "effective_min_score": td.get("effective_min_score"),
                "win_rate":        td.get("win_rate"),
                "drawdown_pct":    td.get("drawdown_pct"),
                "drawdown_halt":   td.get("drawdown_halt", False),
                "regime_max_pos":  td.get("regime_max_pos", 12),
                "profit_factor":   td.get("profit_factor"),
                "portfolio_beta":  td.get("portfolio_beta"),
                "portfolio_heat":  td.get("portfolio_heat"),
                "market_quality":  td.get("market_quality"),
                "scan_breadth_pct": td.get("scan_breadth_pct"),
                "portfolio_concentration": td.get("portfolio_concentration", {}),
                "sector_etf_trends": td.get("sector_etf_trends", {}),
                "rs_leaders":      [{"ticker": e["ticker"], "rs_rating": e.get("rs_rating"), "score": e.get("score")} for e in _rs_leaders],
                "ema21_setups":    _ema21_setups,
                "pocket_pivots":   _pocket_pivots,
                "htf_stocks":      [{"ticker": e["ticker"], "htf_consec": e.get("htf_consec", 0)} for e in _htf_stocks],
                "tt8_stocks":      _tt8_stocks,
                "tt_leaders":      [{"ticker": e["ticker"], "trend_template": e.get("trend_template", 0)} for e in _tt_leaders],
            }
            logger.info(f"Loaded {len(current_positions)} positions and {len(last_scan_top)} scan candidates from trades.json")
    except Exception as _te:
        logger.warning(f"Could not load trades.json: {_te}")

    # Include held ticker symbols in the analysis universe
    held_tickers = [p.get("ticker", "") for p in current_positions if p.get("ticker")]
    if held_tickers:
        held_set = set(held_tickers)
        # Prepend held tickers so AI analysis covers them specifically
        for ht in reversed(held_tickers):
            if ht not in {t for t, _ in top_tickers}:
                top_tickers = [(ht, 10)] + top_tickers  # high weight for held positions
        logger.info(f"Added held positions to analysis: {', '.join(held_tickers)}")

    # ── AI Analysis ───────────────────────────────────────────────
    logger.info("Calling Claude AI...")
    analysis = None

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
                current_positions=current_positions,    # what the bot holds now
                scan_top=last_scan_top,                  # what was scanned last cycle
                live_market_context=live_market_context, # day type, timing, performance stats
            ),
            default=None,
            label="ClaudeAI",
        )
    else:
        logger.warning("ANTHROPIC_API_KEY not set — skipping AI analysis")

    # ── Fallback to cached analysis ───────────────────────────────
    if not analysis:
        cached = history.get("last_analysis")
        if cached:
            logger.info("Using cached last_analysis from history.json")
            analysis = cached
        else:
            logger.warning("No analysis and no cache — writing minimal fallback")
            analysis = {
                "market_sentiment": "UNKNOWN",
                "market_regime": "UNCERTAIN",
                "week_summary": "Intelligence data loading...",
                "buy_signals": [],
                "sell_signals": [],
                "watch_list": [],
                "notable_trends": [],
                "macro_risks": [],
                "rok_message": "Connecting to AI analysis — check back shortly.",
            }
    else:
        history["last_analysis"] = analysis

    # ── Build price lookup from stock_data ────────────────────────
    price_lookup = {s["ticker"]: s["price"] for s in stock_data if s}
    stock_data_lookup = {s["ticker"]: s for s in stock_data if s}

    # ── Enrich signals ────────────────────────────────────────────
    signal_lookup = {}
    for sig in analysis.get("buy_signals", []):
        signal_lookup[sig["ticker"]] = {"type": "buy", "strength": sig.get("signal_strength", 5)}
    for sig in analysis.get("sell_signals", []):
        signal_lookup[sig["ticker"]] = {"type": "sell", "strength": sig.get("signal_strength", 5)}

    all_signals = (
        analysis.get("buy_signals", [])
        + analysis.get("sell_signals", [])
        + analysis.get("watch_list", [])
    )
    for sig in all_signals:
        t = sig.get("ticker", "")
        if not t:
            continue
        sd = stock_data_lookup.get(t)
        if sd:
            if not sig.get("current_price") and sd.get("price"):
                sig["current_price"] = sd["price"]
            if not sig.get("company") and sd.get("company_name"):
                sig["company"] = sd["company_name"]
            if not sig.get("price_target") and sd.get("analyst_target"):
                sig["price_target"] = sd["analyst_target"]
            if not sig.get("stop_loss") and sig.get("current_price"):
                sig["stop_loss"] = round(sig["current_price"] * 0.92, 2)
            if not sig.get("sector"):
                sig["sector"] = sd.get("sector", "")
            ta = ta_data.get(t, {}) if ta_data else {}
            if not sig.get("vol_ratio"):
                sig["vol_ratio"] = ta.get("volume_ratio") or sd.get("vol_ratio")
            if not sig.get("rsi"):
                sig["rsi"] = ta.get("rsi") or sd.get("rsi")

        # Price sparkline (last 30 days)
        sig["price_history"] = _safe(
            yahoo_finance.get_price_history, t, 30,
            default=list, label=f"Hist:{t}",
        ) or []

        if not sig.get("signal_strength"):
            sig["signal_strength"] = 6
        if not sig.get("time_horizon"):
            sig["time_horizon"] = "1-3 months"
        if not sig.get("risk_level"):
            sig["risk_level"] = "Medium"

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
    history["runs"] = history["runs"][-96:]

    # ── Track record ──────────────────────────────────────────────
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
                sd2 = _safe(yahoo_finance.get_stock_data, ticker, default=lambda: None)
                current = sd2["price"] if sd2 else None
            if current and entry:
                pct = round((current - entry) / entry * 100, 1)
                track_record.append({
                    "ticker": ticker,
                    "entry_price": entry,
                    "current_price": round(current, 2),
                    "pct_change": pct,
                    "date": prev_run.get("date", ""),
                })
        track_record.sort(key=lambda x: abs(x["pct_change"]), reverse=True)

    # ── Market mood plain-language ────────────────────────────────
    mkt_sent = (analysis.get("market_sentiment") or "NEUTRAL").upper()
    fg_score = (fear_greed or {}).get("score", 50)
    buy_count = len(analysis.get("buy_signals", []))
    if mkt_sent == "BULLISH" and buy_count >= 5:
        market_mood = f"Markets are strong — ROK found {buy_count} stocks worth watching right now"
    elif mkt_sent == "BULLISH":
        market_mood = "Markets are leaning bullish — ROK sees some opportunities"
    elif mkt_sent == "BEARISH":
        market_mood = "Markets are under pressure — ROK recommends caution"
    elif fg_score and fg_score < 30:
        market_mood = "Fear is high — that often means buying opportunities are near"
    else:
        market_mood = "Markets are mixed — ROK is watching closely for clear signals"

    # ── Build intel_report output ─────────────────────────────────
    intel = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_regime": analysis.get("market_regime", "UNCERTAIN"),
        "market_sentiment": analysis.get("market_sentiment", "NEUTRAL"),
        "sentiment_score": analysis.get("sentiment_score", 5),
        "week_summary": analysis.get("week_summary", ""),
        "market_mood": market_mood,
        "rok_message": analysis.get("rok_message", ""),
        "buy_signals": analysis.get("buy_signals", []),
        "sell_signals": analysis.get("sell_signals", []),
        "watch_list": analysis.get("watch_list", []),
        "notable_trends": analysis.get("notable_trends", []),
        "macro_risks": analysis.get("macro_risks", []),
        "sector_heat": analysis.get("sector_heat", {}),
        "sector_rotation": analysis.get("sector_rotation", ""),
        "short_squeeze_alerts": analysis.get("short_squeeze_alerts", []),
        "earnings_plays": analysis.get("earnings_plays", []),
        "congressional_plays": analysis.get("congressional_plays", []),
        "technical_breakouts": analysis.get("technical_breakouts", []),
        "fear_greed": fear_greed or {},
        "market_indices": market_indices or {},
        "market_breadth": market_breadth or {},
        "put_call_ratio": put_call_ratio or {},
        "ticker_mentions": top_tickers[:24],
        "stocktwits_trending": stocktwits_data[:12],
        "congressional_buys": congress_buys[:8],
        "insider_buys": insider_buys[:12],
        "track_record": track_record[:10],
        "recent_runs": history["runs"][-8:],
        "news_items": [
            {
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "sentiment": a.get("sentiment_score", 0),
                "tickers": a.get("mentioned_tickers", []),
            }
            for a in news_articles[:30]
            if a.get("title")
        ],
        "source_stats": {
            "reddit": len(reddit_posts),
            "news": len(news_articles),
            "stocks": len(stock_data),
            "sec": len(sec_filings or []),
            "earnings_upcoming": len(earnings_cal or []),
            "unusual_options": len(unusual_opts or []),
            "congress_trades": len(congress_buys or []),
            "technical": len(ta_data or {}),
            "insider_buys": len(insider_buys or []),
        },
        "buy_count": buy_count,
        "sell_count": len(analysis.get("sell_signals", [])),
        "current_positions": current_positions[:10],  # pass-through for dashboard cross-reference
        "position_analysis": analysis.get("position_analysis", []),  # AI commentary on held positions
        "rs_leaders":      live_market_context.get("rs_leaders", []),
        "ema21_setups":    live_market_context.get("ema21_setups", []),
        "pocket_pivots":   live_market_context.get("pocket_pivots", []),
        "htf_stocks":      live_market_context.get("htf_stocks", []),
        "tt8_stocks":      live_market_context.get("tt8_stocks", []),
        "tt_leaders":      live_market_context.get("tt_leaders", []),
        "drawdown_halt":   live_market_context.get("drawdown_halt", False),
        "regime_max_pos":  live_market_context.get("regime_max_pos", 12),
        "scan_breadth_pct": live_market_context.get("scan_breadth_pct"),
    }

    # Sanitize all datetime objects before JSON serialization
    intel = _sanitize(intel)

    # ── Write output files ────────────────────────────────────────
    intel_json = json.dumps(intel, cls=_Encoder, indent=2)
    (docs_dir / "intel_report.json").write_text(intel_json, encoding="utf-8")
    logger.info(f"Intel report written → docs/intel_report.json ({len(intel_json)} chars)")

    # Update history
    history_path.write_text(json.dumps(_sanitize(history), cls=_Encoder, indent=2), encoding="utf-8")
    logger.info(f"History updated → docs/history.json ({len(history['runs'])} runs)")

    # Write prices.json for JS fallback
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
    (docs_dir / "prices.json").write_text(json.dumps(prices_dict, cls=_Encoder), encoding="utf-8")
    logger.info(f"Prices written → docs/prices.json ({len(prices_dict)} tickers)")

    # ── Weekly Bot Performance Report ────────────────────────────────
    try:
        _week_report = _build_weekly_bot_report(docs_dir)
        if _week_report:
            (docs_dir / "weekly_report.json").write_text(
                json.dumps(_sanitize(_week_report), cls=_Encoder, indent=2), encoding="utf-8"
            )
            logger.info(f"Weekly report written → docs/weekly_report.json")
    except Exception as _we:
        logger.warning(f"Weekly report failed: {_we}")

    logger.info(f"Summary: {buy_count} buys | {len(analysis.get('sell_signals', []))} sells")
    logger.info("ROK INTELLIGENCE PIPELINE COMPLETE")


if __name__ == "__main__":
    run()
