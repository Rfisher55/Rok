"""
Congressional stock trades from House Stock Watcher + Senate Stock Watcher + QuiverQuant.
"""
import logging
import requests
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions_for_all_senators.json"
_QUIVER_URL = "https://api.quiverquant.com/beta/live/congresstrading"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime((s or "")[:10], fmt)
        except (ValueError, TypeError):
            continue
    return None


def _classify(kind: str) -> str:
    kind = (kind or "").lower()
    if "purchase" in kind or "buy" in kind:
        return "BUY"
    if "sale" in kind or "sell" in kind:
        return "SELL"
    return ""


def _fetch_house(cutoff: datetime) -> list:
    trades = []
    try:
        r = requests.get(_HOUSE_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        for t in r.json():
            date = _parse_date(t.get("transaction_date") or t.get("disclosure_date"))
            if not date or date < cutoff:
                continue
            ticker = (t.get("ticker") or "").strip().upper()
            if not ticker or ticker in ("--", "N/A") or not ticker.isalpha() or len(ticker) > 5:
                continue
            action = _classify(t.get("type", ""))
            if not action:
                continue
            trades.append({
                "ticker": ticker,
                "action": action,
                "member": t.get("representative", "Unknown"),
                "chamber": "House",
                "amount": t.get("amount", ""),
                "date": date.strftime("%Y-%m-%d"),
                "asset": (t.get("asset_description") or "")[:80],
            })
    except Exception as e:
        logger.warning(f"Congressional House: {e}")
    return trades


def _fetch_senate(cutoff: datetime) -> list:
    trades = []
    try:
        r = requests.get(_SENATE_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        for senator in r.json():
            for t in senator.get("transactions", []):
                date = _parse_date(t.get("transaction_date"))
                if not date or date < cutoff:
                    continue
                ticker = (t.get("ticker") or "").strip().upper()
                if not ticker or ticker in ("--", "N/A") or not ticker.isalpha() or len(ticker) > 5:
                    continue
                action = _classify(t.get("type", ""))
                if not action:
                    continue
                trades.append({
                    "ticker": ticker,
                    "action": action,
                    "member": senator.get("senator", "Unknown"),
                    "chamber": "Senate",
                    "amount": t.get("amount", ""),
                    "date": date.strftime("%Y-%m-%d"),
                    "asset": (t.get("asset_description") or "")[:80],
                })
    except Exception as e:
        logger.warning(f"Congressional Senate: {e}")
    return trades


def _fetch_quiverquant(cutoff: datetime) -> list:
    """Fallback: QuiverQuant public congressional trading endpoint."""
    trades = []
    try:
        r = requests.get(_QUIVER_URL, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        for t in r.json():
            date = _parse_date(t.get("Date") or t.get("TransactionDate") or "")
            if not date or date < cutoff:
                continue
            ticker = (t.get("Ticker") or "").strip().upper()
            if not ticker or not ticker.isalpha() or len(ticker) > 5:
                continue
            action_raw = (t.get("Transaction") or t.get("Type") or "").lower()
            action = "BUY" if "purchase" in action_raw or "buy" in action_raw else "SELL" if "sale" in action_raw or "sell" in action_raw else ""
            if not action:
                continue
            trades.append({
                "ticker": ticker,
                "action": action,
                "member": t.get("Representative") or t.get("Senator") or "Unknown",
                "chamber": "Congress",
                "amount": t.get("Range") or "",
                "date": date.strftime("%Y-%m-%d"),
                "asset": (t.get("Asset") or "")[:80],
            })
    except Exception as e:
        logger.debug(f"QuiverQuant congressional: {e}")
    return trades


def get_recent_trades(days_back: int = 45) -> list:
    """All recent congressional trades, sorted newest first."""
    cutoff = datetime.now() - timedelta(days=days_back)
    trades = _fetch_house(cutoff) + _fetch_senate(cutoff)
    if not trades:
        # Try QuiverQuant as fallback
        logger.info("S3 sources empty — trying QuiverQuant for congressional trades")
        trades = _fetch_quiverquant(cutoff)
    trades.sort(key=lambda x: x["date"], reverse=True)
    return trades[:200]


def get_congress_buys(days_back: int = 90) -> list:
    """
    Aggregated congressional buys by ticker — most-bought first.
    Returns list of dicts: ticker, buy_count, sell_count, member_count, members_preview, latest_date.
    """
    cutoff = datetime.now() - timedelta(days=days_back)
    all_trades = _fetch_house(cutoff) + _fetch_senate(cutoff)
    if not all_trades:
        all_trades = _fetch_quiverquant(cutoff)

    by_ticker: dict = defaultdict(lambda: {
        "ticker": "",
        "buy_count": 0,
        "sell_count": 0,
        "members": set(),
        "latest_date": "",
    })

    for t in all_trades:
        rec = by_ticker[t["ticker"]]
        rec["ticker"] = t["ticker"]
        if t["action"] == "BUY":
            rec["buy_count"] += 1
        else:
            rec["sell_count"] += 1
        rec["members"].add(t["member"])
        if t["date"] > rec["latest_date"]:
            rec["latest_date"] = t["date"]

    buys = [r for r in by_ticker.values() if r["buy_count"] > 0]
    buys.sort(key=lambda x: x["buy_count"], reverse=True)

    result = []
    for r in buys[:15]:
        members = sorted(r["members"])
        result.append({
            "ticker": r["ticker"],
            "buy_count": r["buy_count"],
            "sell_count": r["sell_count"],
            "member_count": len(members),
            "members_preview": (
                ", ".join(members[:3]) + (f" +{len(members)-3} more" if len(members) > 3 else "")
            ),
            "latest_date": r["latest_date"],
        })
    return result
