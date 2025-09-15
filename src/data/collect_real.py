"""
Collect real launches and 5-year revenues using FDA, CT.gov, and SEC endpoints.
Writes normalized parquet tables satisfying G1 requirements (when data suffices).
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd

from .sources import (
    fetch_fda_drug,
    fetch_ctgov_summary,
    get_cik_for_ticker,
    fetch_company_xbrl_facts,
    extract_revenue_from_xbrl,
    fetch_10k_texts,
    extract_brand_revenue,
)


def load_brands(seed_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(seed_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_real_dataset(seed_csv: Path,
                       years: Tuple[int, int] = (2015, 2024),
                       output_dir: Path = Path("data_proc")) -> bool:
    brands = load_brands(seed_csv)

    launches_rows: List[Dict[str, Any]] = []
    revenues_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(brands):
        brand = row.get("brand", "").strip()
        company = row.get("company", "").strip()
        ticker = row.get("ticker", "").strip()
        aliases = [a.strip() for a in (row.get("aliases", "") or "").split(";") if a.strip()]
        ta = row.get("therapeutic_area", "Unknown")

        if not brand or not company:
            continue

        launch_id = f"REAL_{idx:04d}"

        # FDA info (comprehensive extraction)
        fda = fetch_fda_drug(brand)
        fda_extracted = fda.get('extracted', {})
        
        # Core approval data
        approval_date = fda_extracted.get('original_approval_date')
        review_priority = fda_extracted.get('review_priority')
        application_number = fda_extracted.get('application_number')
        
        # Drug characteristics
        route = fda_extracted.get('route')
        mechanism = fda_extracted.get('mechanism_of_action')
        dosage_form = fda_extracted.get('dosage_form')
        generic_name = fda_extracted.get('generic_name')
        
        # Competitive intelligence
        approved_supplementals_count = fda_extracted.get('approved_supplementals_count', 0)
        first_supplemental_date = fda_extracted.get('first_supplemental_date')
        
        # Market access indicators
        sponsor_name = fda_extracted.get('sponsor_name')
        marketing_status = fda_extracted.get('marketing_status')
        manufacturer_name = fda_extracted.get('manufacturer_name')

        # CT.gov summary (optional enrichment)
        ct = fetch_ctgov_summary(brand)
        
        # Use therapeutic area as indication fallback (set above in launches row)

        # SEC revenues using XBRL API (much better than text parsing)
        cik = get_cik_for_ticker(ticker) if ticker else None
        
        # Get approval year for mapping fiscal years to launch-relative years
        approval_year = None
        if approval_date:
            try:
                approval_year = int(str(approval_date)[:4])
            except Exception:
                approval_year = None

        # Extract revenues using structured XBRL data
        revenue_by_year: Dict[str, float] = {}
        if cik and approval_year:
            # Fetch structured financial data from SEC XBRL API
            xbrl_data = fetch_company_xbrl_facts(cik)
            
            # Extract revenue mapped to years since launch
            revenue_by_year = extract_revenue_from_xbrl(
                xbrl_data, 
                approval_year, 
                [brand] + aliases
            )

        # Build launches row with comprehensive FDA data
        launches_rows.append({
            # Core identifiers
            "launch_id": launch_id,
            "drug_name": brand,
            "company": company,
            
            # FDA regulatory data
            "approval_date": approval_date or "",
            "review_priority": review_priority or "",
            "application_number": application_number or "",
            "generic_name": generic_name or "",
            
            # Drug characteristics  
            "indication": ta,  # Use therapeutic area as indication
            "mechanism": mechanism or "",
            "route": route or "",
            "dosage_form": dosage_form or "",
            "therapeutic_area": ta,
            
            # Competitive intelligence
            "approved_supplementals_count": approved_supplementals_count,
            "first_supplemental_date": first_supplemental_date or "",
            
            # Market access indicators
            "sponsor_name": sponsor_name or company,  # Use CSV company as fallback
            "marketing_status": marketing_status or "",
            "manufacturer_name": manufacturer_name or "",
            
            # Fields still needing enrichment (pricing/market data)
            "eligible_patients_at_launch": 0,
            "market_size_source": "",
            "list_price_month_usd_launch": 0.0,
            "net_gtn_pct_launch": 0.0,
            "access_tier_at_launch": "",
            "price_source": "",
            "competitor_count_at_launch": 0,
            "clinical_efficacy_proxy": 0.0,
            "safety_black_box": False,
            
            # Data sources
            "source_urls": json.dumps([f"FDA:{application_number}" if application_number else "FDA:Unknown"]),
        })

        # Revenues: XBRL data is already mapped to year_since_launch
        if revenue_by_year:
            for year_since_str, amt in revenue_by_year.items():
                try:
                    year_since = int(year_since_str)
                    # Include Y0 through Y5 (launch to maturity)
                    if 0 <= year_since <= 5:
                        revenues_rows.append({
                            "launch_id": launch_id,
                            "year_since_launch": year_since,
                            "revenue_usd": float(amt),
                            "source_url": f"SEC XBRL CIK {cik}" if cik else "SEC XBRL"
                        })
                except Exception:
                    continue

    # Write outputs
    output_dir.mkdir(exist_ok=True, parents=True)
    if launches_rows:
        pd.DataFrame(launches_rows).to_parquet(output_dir / "launches.parquet", index=False)
    if revenues_rows:
        pd.DataFrame(revenues_rows).to_parquet(output_dir / "launch_revenues.parquet", index=False)

    # Analog table is optional here; can be created by existing builder later
    return bool(launches_rows)


if __name__ == "__main__":
    # Minimal manual test (expects data_raw/brands.csv)
    seed = Path("data_raw/brands.csv")
    if seed.exists():
        ok = build_real_dataset(seed)
        print("Build status:", ok)
    else:
        print("brands.csv not found; create data_raw/brands.csv first")


