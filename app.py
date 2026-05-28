import json
import logging
import threading
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from flask import jsonify, render_template, request

from app_factory import create_app, db
from config import Config
from database.models import WeeklyAnalysis
from scheduler.tasks import run_scrape_and_analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = create_app()

scheduler = BackgroundScheduler(timezone="UTC")


def start_scheduler():
    scheduler.add_job(
        func=lambda: run_scrape_and_analyze(app),
        trigger="interval",
        minutes=Config.SCRAPE_INTERVAL_MINUTES,
        id="rok_pipeline",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"Scheduler started — pipeline every {Config.SCRAPE_INTERVAL_MINUTES} min")


@app.route("/")
def index():
    analysis = WeeklyAnalysis.query.order_by(WeeklyAnalysis.created_at.desc()).first()
    return render_template("index.html", analysis=_serialize(analysis))


@app.route("/api/latest")
def api_latest():
    analysis = WeeklyAnalysis.query.order_by(WeeklyAnalysis.created_at.desc()).first()
    return jsonify(_serialize(analysis))


@app.route("/api/history")
def api_history():
    records = WeeklyAnalysis.query.order_by(WeeklyAnalysis.created_at.desc()).limit(10).all()
    return jsonify([_serialize(r) for r in records])


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    def _run():
        run_scrape_and_analyze(app)
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"status": "ok", "message": "ROK analysis running — refresh in ~60 seconds"})


@app.route("/api/status")
def api_status():
    count = WeeklyAnalysis.query.count()
    latest = WeeklyAnalysis.query.order_by(WeeklyAnalysis.created_at.desc()).first()
    return jsonify({
        "total_analyses": count,
        "last_run": latest.created_at.isoformat() if latest else None,
        "anthropic_key_set": bool(Config.ANTHROPIC_API_KEY),
        "twitter_enabled": Config.TWITTER_ENABLED,
        "scheduler_running": scheduler.running,
    })


def _serialize(analysis: WeeklyAnalysis | None) -> dict:
    if not analysis:
        return {}

    raw = {}
    try:
        raw = json.loads(analysis.raw_data_summary or "{}")
    except json.JSONDecodeError:
        pass

    buy = json.loads(analysis.buy_signals or "[]")
    sell = json.loads(analysis.sell_signals or "[]")
    watch = json.loads(analysis.watch_list or "[]")

    # Extract extra fields that may be in buy/sell signals
    squeeze_alerts = []
    earnings_plays = []
    try:
        full = json.loads(analysis.buy_signals or "[]")
        # These might be stored in raw_data_summary for the full analysis
        data_full = raw
        squeeze_alerts = data_full.get("short_squeeze_alerts", [])
        earnings_plays = data_full.get("earnings_plays", [])
    except Exception:
        pass

    return {
        "id": analysis.id,
        "analysis_date": analysis.analysis_date.strftime("%B %d, %Y %I:%M %p UTC"),
        "week_summary": analysis.week_summary,
        "market_sentiment": analysis.market_sentiment,
        "buy_signals": buy,
        "sell_signals": sell,
        "watch_list": watch,
        "notable_trends": json.loads(analysis.notable_trends or "[]"),
        "sentiment_score": raw.get("sentiment_score", 5),
        "sector_heat": raw.get("sector_heat", {}),
        "rok_message": raw.get("rok_message", ""),
        "ticker_mentions": raw.get("ticker_mentions", []),
        "fear_greed": raw.get("fear_greed", {}),
        "market_indices": raw.get("market_indices", {}),
        "aggregate_sentiment": raw.get("aggregate_sentiment", {}),
        "short_squeeze_alerts": squeeze_alerts,
        "earnings_plays": earnings_plays,
        "source_stats": {
            "reddit": raw.get("reddit_post_count", 0),
            "news": raw.get("news_article_count", 0),
            "stocks": raw.get("stock_data_count", 0),
            "sec": raw.get("sec_filing_count", 0),
            "earnings_upcoming": raw.get("earnings_upcoming", 0),
            "unusual_options": raw.get("unusual_options_count", 0),
        },
    }


if __name__ == "__main__":
    start_scheduler()
    logger.info("Running initial ROK analysis on startup...")
    thread = threading.Thread(target=lambda: run_scrape_and_analyze(app), daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
