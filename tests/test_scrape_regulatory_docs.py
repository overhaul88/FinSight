"""Tests for the regulatory document scraper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "scrape_regulatory_docs.py"


def load_scraper_module():
    spec = importlib.util.spec_from_file_location("scrape_regulatory_docs", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_rbi_page_extracts_pdf_title_and_date():
    scraper = load_scraper_module()
    seed = scraper.SourceSeed(
        key="rbi_kyc_master_direction",
        regulator="rbi",
        category="kyc",
        page_url="https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
        profiles=("core",),
    )
    html = """
    <table class="tablebg" width="100%">
      <tr>
        <td class="tableheader" align="left">
          <a id='APDF_MD18' target="_blank"
             href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.PDF">
             <img alt='PDF - Master Direction - Know Your Customer (KYC) Direction, 2016'>
          </a>
        </td>
      </tr>
      <tr><td align="center" class="tableheader"><b>Master Direction - Know Your Customer (KYC) Direction, 2016</b></td></tr>
      <tr class="tablecontent2">
        <td>
          <table>
            <tr>
              <td>
                <p class="head">RBI/DBR/2015-16/18</p>
                <p align="right" class="head">February 25, 2016<br>(Updated as on August 14, 2025)</p>
                <p align="center" class="head">Master Direction - Know Your Customer (KYC) Direction, 2016</p>
                <p>Body text</p>
                <p class="footnote"><a href="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12549&Mode=0">amendment</a></p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
    """

    parsed = scraper.parse_rbi_page(seed, html)

    assert parsed.asset_url.endswith(".PDF")
    assert parsed.asset_format == "txt"
    assert parsed.title == "Master Direction - Know Your Customer (KYC) Direction, 2016"
    assert parsed.published_date == "2016-02-25"
    assert "RBI/DBR/2015-16/18" in parsed.text_content
    assert "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12549&Mode=0" in parsed.related_urls


def test_parse_sebi_page_extracts_iframe_pdf_title_and_date():
    scraper = load_scraper_module()
    seed = scraper.SourceSeed(
        key="sebi_kyc_master_circular_2023",
        regulator="sebi",
        category="kyc",
        page_url="https://www.sebi.gov.in/legal/master-circulars/oct-2023/master-circular-on-know-your-client-kyc-norms-for-the-securities-market_77945.html",
        profiles=("core",),
    )
    html = """
    <html>
      <head>
        <meta name="title" content="SEBI | Master Circular on Know Your Client (KYC) norms for the securities market" />
      </head>
      <body>
        <h1>Master Circular on Know Your Client (KYC) norms for the securities market
          <div class="social-share-btn"></div>
        </h1>
        <div class='date_value'><h5>Oct 12, 2023</h5><span class='verticle_pipe'>|</span></div>
        <iframe src='../../../web/?file=https://www.sebi.gov.in/sebi_data/attachdocs/oct-2023/1697120943335.pdf'
          title="Master Circular on Know Your Client (KYC) norms for the securities market"></iframe>
      </body>
    </html>
    """

    parsed = scraper.parse_sebi_page(seed, html)

    assert parsed.asset_url == "https://www.sebi.gov.in/sebi_data/attachdocs/oct-2023/1697120943335.pdf"
    assert parsed.title == "Master Circular on Know Your Client (KYC) norms for the securities market"
    assert parsed.published_date == "2023-10-12"


def test_build_output_path_uses_loader_friendly_filename():
    scraper = load_scraper_module()
    seed = scraper.SourceSeed(
        key="rbi_gold_loans_irregular_practices_2024",
        regulator="rbi",
        category="gold_loans",
        page_url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12735&Mode=0",
        profiles=("core",),
    )
    parsed = scraper.ParsedPage(
        regulator="rbi",
        title="Gold loans - Irregular practices observed in grant of loans against pledge of gold ornaments and jewellery",
        published_date="2024-09-30",
        page_url=seed.page_url,
        asset_url="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/example.PDF",
        asset_format="txt",
        text_content="body",
    )

    path = scraper.build_output_path(ROOT / "data" / "raw", seed, parsed)

    assert path.name.startswith("RBI_2024-09-30_gold_loans_irregular_practices")
    assert path.suffix == ".txt"
