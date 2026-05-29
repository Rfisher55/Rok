"""
SEC insider trades via EDGAR EFTS + CIK→ticker mapping.
Also fetches 8-K major event filings and OpenInsider purchase aggregates.
"""
import logging
import requests
from bs4 import BeautifulSoup
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


def get_insider_buys(days_back: int = 14) -> list[dict]:
    """
    Real CEO/CFO/director stock purchases from OpenInsider (aggregates SEC Form 4).
    Returns list of {ticker, insider_name, title, shares, value_usd, date, company}.
    """
    buys = []
    try:
        url = (
            "https://openinsider.com/screener?"
            f"s=&o=&pl=&ph=&ll=&lh=&fd={days_back}&fdr=&td=0&tdr=&fdlyl=&fdlyh="
            "&daysago=&xs=1&vl=1000&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999"
            "&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
            "&sortcol=0&cnt=30&page=1"
        )
        r = requests.get(url, headers=_BROWSER, timeout=15)
        if not r.ok:
            return _insider_buys_yahoo_fallback()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", class_="tinytable")
        if not table:
            return _insider_buys_yahoo_fallback()
        rows = table.find_all("tr")[1:]
        for row in rows[:25]:
            cols = row.find_all("td")
            if len(cols) < 10:
                continue
            ticker = cols[1].get_text(strip=True).upper()
            if not ticker or not ticker.isalpha() or len(ticker) > 5:
                continue
            company = cols[2].get_text(strip=True)
            insider_name = cols[3].get_text(strip=True)
            title = cols[4].get_text(strip=True)
            trade_date = cols[0].get_text(strip=True)
            shares_txt = cols[7].get_text(strip=True).replace(",", "").replace("+", "")
            value_txt = cols[8].get_text(strip=True).replace("$", "").replace(",", "").replace("+", "")
            try:
                shares = int(float(shares_txt)) if shares_txt else 0
                value_usd = int(float(value_txt) * 1000) if value_txt else 0
            except (ValueError, TypeError):
                shares, value_usd = 0, 0
            if shares <= 0:
                continue
            buys.append({
                "ticker": ticker,
                "company": company[:40],
                "insider_name": insider_name,
                "title": title,
                "shares": shares,
                "value_usd": value_usd,
                "date": trade_date,
                "source": "openinsider",
            })
    except Exception as e:
        logger.warning(f"OpenInsider fetch: {e}")
        return _insider_buys_yahoo_fallback()
    return buys


def _insider_buys_yahoo_fallback() -> list[dict]:
    """Fallback: Yahoo Finance insider purchases screener."""
    buys = []
    try:
        r = requests.get(
            "https://finance.yahoo.com/screener/predefined/insider_purchase_filings",
            headers={**_BROWSER, "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"},
            timeout=12,
        )
        if not r.ok:
            return buys
        soup = BeautifulSoup(r.text, "lxml")
        for row in soup.select("table tbody tr")[:20]:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            ticker = cols[0].get_text(strip=True).upper()
            if not ticker or not ticker.isalpha() or len(ticker) > 5:
                continue
            buys.append({
                "ticker": ticker,
                "company": cols[1].get_text(strip=True)[:40] if len(cols) > 1 else "",
                "insider_name": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                "title": cols[3].get_text(strip=True) if len(cols) > 3 else "",
                "shares": 0,
                "value_usd": 0,
                "date": cols[4].get_text(strip=True) if len(cols) > 4 else "",
                "source": "yahoo",
            })
    except Exception as e:
        logger.debug(f"Yahoo insider buys fallback: {e}")
    return buys


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
