"""
SEC insider trades via EDGAR EFTS + CIK→ticker mapping.
Also fetches 8-K major event filings.
"""
import logging
import requests
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "ROK-StockAdvisor/1.0 robertcfisher3@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


@lru_cache(maxsize=1)
def _load_cik_ticker_map() -> dict:
    """Load SEC CIK → ticker mapping (cached)."""
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return {str(v["cik_str"]): v["ticker"] for v in data.values()}
    except Exception as e:
        logger.warning(f"CIK→ticker map load failed: {e}")
        return {}


def _cik_to_ticker(cik: str | int, fallback: str = "") -> str:
    mapping = _load_cik_ticker_map()
    return mapping.get(str(cik).lstrip("0"), fallback)


def get_recent_insider_trades(days_back: int = 7) -> list[dict]:
    """Fetch recent Form 4 insider trade filings with correct ticker symbols."""
    filings = []
    try:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": "",
                "forms": "4",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
            },
            headers=SEC_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        cik_map = _load_cik_ticker_map()

        for hit in data.get("hits", {}).get("hits", [])[:60]:
            src = hit.get("_source", {})
            entity_id = str(src.get("entity_id", "")).lstrip("0")
            ticker = cik_map.get(entity_id, "")
            # Skip if no ticker found or ticker looks invalid
            if not ticker or not ticker.isalpha() or len(ticker) > 6:
                continue
            filings.append({
                "ticker": ticker.upper(),
                "company_name": src.get("entity_name", ""),
                "form_type": src.get("file_type", "4"),
                "filing_date": src.get("file_date", ""),
                "description": src.get("period_of_report", ""),
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entity_id}&type=4",
            })
    except Exception as e:
        logger.warning(f"SEC EDGAR insider trade fetch failed: {e}")
    return filings


def get_recent_8k_filings(days_back: int = 7) -> list[dict]:
    """Fetch recent 8-K material event filings with ticker symbols."""
    filings = []
    try:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": "",
                "forms": "8-K",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
            },
            headers=SEC_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        cik_map = _load_cik_ticker_map()

        for hit in data.get("hits", {}).get("hits", [])[:40]:
            src = hit.get("_source", {})
            entity_id = str(src.get("entity_id", "")).lstrip("0")
            ticker = cik_map.get(entity_id, "")
            filings.append({
                "ticker": ticker.upper() if ticker else "",
                "company_name": src.get("entity_name", ""),
                "form_type": "8-K",
                "filing_date": src.get("file_date", ""),
                "description": src.get("display_names", ""),
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entity_id}&type=8-K",
            })
    except Exception as e:
        logger.warning(f"SEC EDGAR 8-K fetch failed: {e}")
    return filings
