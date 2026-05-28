import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "ROK-StockAdvisor/1.0 robertcfisher3@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"


def get_recent_insider_trades(days_back: int = 7) -> list[dict]:
    """Fetch recent Form 4 insider trade filings from SEC EDGAR."""
    filings = []
    try:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": "",
            "forms": "4",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "_source": "hits",
        }
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=SEC_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        for hit in data.get("hits", {}).get("hits", [])[:50]:
            src = hit.get("_source", {})
            filings.append({
                "company_name": src.get("entity_name", ""),
                "form_type": src.get("file_type", "4"),
                "filing_date": src.get("file_date", ""),
                "description": src.get("period_of_report", ""),
                "url": f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id', '')}/",
                "ticker": src.get("display_date_filed", ""),
            })
    except Exception as e:
        logger.warning(f"SEC EDGAR insider trade fetch failed: {e}")
    return filings


def get_recent_8k_filings(days_back: int = 7) -> list[dict]:
    """Fetch recent 8-K material event filings."""
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
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        for hit in data.get("hits", {}).get("hits", [])[:30]:
            src = hit.get("_source", {})
            filings.append({
                "company_name": src.get("entity_name", ""),
                "form_type": "8-K",
                "filing_date": src.get("file_date", ""),
                "description": src.get("display_names", ""),
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={src.get('entity_id', '')}",
            })
    except Exception as e:
        logger.warning(f"SEC EDGAR 8-K fetch failed: {e}")
    return filings
