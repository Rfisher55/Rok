from datetime import datetime
from app_factory import db


class RawMention(db.Model):
    __tablename__ = "raw_mentions"
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(50), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)
    title = db.Column(db.Text)
    body = db.Column(db.Text)
    url = db.Column(db.String(500))
    sentiment_score = db.Column(db.Float, default=0.0)
    upvotes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scraped_at = db.Column(db.DateTime, default=datetime.utcnow)


class StockSnapshot(db.Model):
    __tablename__ = "stock_snapshots"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    price = db.Column(db.Float)
    change_pct = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    market_cap = db.Column(db.BigInteger)
    week_high = db.Column(db.Float)
    week_low = db.Column(db.Float)
    pe_ratio = db.Column(db.Float)
    short_interest = db.Column(db.Float)
    snapped_at = db.Column(db.DateTime, default=datetime.utcnow)


class SecFiling(db.Model):
    __tablename__ = "sec_filings"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10))
    company_name = db.Column(db.String(200))
    form_type = db.Column(db.String(20))
    filing_date = db.Column(db.String(20))
    description = db.Column(db.Text)
    url = db.Column(db.String(500))
    scraped_at = db.Column(db.DateTime, default=datetime.utcnow)


class WeeklyAnalysis(db.Model):
    __tablename__ = "weekly_analyses"
    id = db.Column(db.Integer, primary_key=True)
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    week_summary = db.Column(db.Text)
    buy_signals = db.Column(db.Text)
    sell_signals = db.Column(db.Text)
    watch_list = db.Column(db.Text)
    market_sentiment = db.Column(db.String(20))
    notable_trends = db.Column(db.Text)
    raw_data_summary = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
