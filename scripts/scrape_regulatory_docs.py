"""Download a curated corpus of RBI and SEBI regulatory documents.

The script fetches official RBI and SEBI document pages, extracts the current
document asset URL from the page markup, downloads the asset in an ingestion-
friendly format, and records metadata in a manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


USER_AGENT = (
    "FinSightBot/1.0 (+https://github.com/openai/codex; "
    "regulatory-document-ingestion for research and compliance retrieval)"
)

DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y")


@dataclass(frozen=True)
class SourceSeed:
    key: str
    regulator: str
    category: str
    page_url: str
    profiles: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ParsedPage:
    regulator: str
    title: str
    published_date: str
    page_url: str
    asset_url: str
    asset_format: str
    text_content: str = ""
    related_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class DownloadRecord:
    key: str
    regulator: str
    category: str
    title: str
    published_date: str
    page_url: str
    asset_url: str
    local_path: str
    sha256: str
    size_bytes: int
    asset_format: str
    downloaded_at: str


DEFAULT_SOURCES: tuple[SourceSeed, ...] = (
    SourceSeed(
        key="rbi_kyc_master_direction",
        regulator="rbi",
        category="kyc",
        page_url="https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
        profiles=("core", "rbi"),
        notes="Master Direction - Know Your Customer (KYC) Direction, 2016",
    ),
    SourceSeed(
        key="rbi_digital_lending_directions_2025",
        regulator="rbi",
        category="digital_lending",
        page_url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12848&Mode=0",
        profiles=("core", "rbi"),
        notes="Reserve Bank of India (Digital Lending) Directions, 2025",
    ),
    SourceSeed(
        key="rbi_gold_silver_collateral_directions_2025",
        regulator="rbi",
        category="secured_lending",
        page_url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12859&Mode=0",
        profiles=("core", "rbi"),
        notes="Reserve Bank of India (Lending Against Gold and Silver Collateral) Directions, 2025",
    ),
    SourceSeed(
        key="rbi_gold_loans_irregular_practices_2024",
        regulator="rbi",
        category="gold_loans",
        page_url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12735&Mode=0",
        profiles=("core", "rbi"),
        notes="Gold loans - irregular practices observed in grant of loans against pledge of gold ornaments and jewellery",
    ),
    SourceSeed(
        key="sebi_kyc_master_circular_2023",
        regulator="sebi",
        category="kyc",
        page_url="https://www.sebi.gov.in/legal/master-circulars/oct-2023/master-circular-on-know-your-client-kyc-norms-for-the-securities-market_77945.html",
        profiles=("core", "sebi"),
        notes="Master Circular on Know Your Client (KYC) norms for the securities market",
    ),
    SourceSeed(
        key="sebi_aml_cft_master_circular_2024",
        regulator="sebi",
        category="aml_cft",
        page_url="https://www.sebi.gov.in/legal/master-circulars/jun-2024/guidelines-on-anti-money-laundering-aml-standards-and-combating-the-financing-of-terrorism-cft-obligations-of-securities-market-intermediaries-under-the-prevention-of-money-laundering-act-2002-a-_83942.html",
        profiles=("core", "sebi"),
        notes="Guidelines on AML/CFT obligations of securities market intermediaries",
    ),
    SourceSeed(
        key="sebi_merchant_bankers_master_circular_2023",
        regulator="sebi",
        category="intermediaries",
        page_url="https://www.sebi.gov.in/legal/master-circulars/sep-2023/master-circular-for-merchant-bankers_77368.html",
        profiles=("core", "sebi"),
        notes="Master Circular for Merchant Bankers",
    ),
)


class TextExtractor(HTMLParser):
    """Convert small HTML fragments to plain text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif tag in {"br", "p", "div", "tr", "li", "td", "h1", "h2", "h3", "h4", "h5"} and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "div", "tr", "li", "td"} and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = html.unescape("".join(self._parts))
        text = re.sub(r"\r", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


class SectionTextExtractor(TextExtractor):
    """Extract plain text from the first matching HTML section."""

    def __init__(self, target_tag: str, attr_name: str, attr_value: str) -> None:
        super().__init__()
        self.target_tag = target_tag
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = dict(attrs)
        if self.depth == 0 and tag == self.target_tag and attr_map.get(self.attr_name) == self.attr_value:
            self.depth = 1
            return

        if self.depth > 0:
            self.depth += 1
            super().handle_starttag(tag, attrs)

    def handle_startendtag(self, tag: str, attrs) -> None:
        if self.depth > 0:
            super().handle_starttag(tag, attrs)
            super().handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.depth > 0:
            super().handle_endtag(tag)
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth > 0:
            super().handle_data(data)


def slugify(value: str) -> str:
    text = html.unescape(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:120] or "document"


def normalize_date(value: str) -> str:
    cleaned = clean_fragment(value)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return cleaned


def clean_fragment(value: str) -> str:
    extractor = TextExtractor()
    extractor.feed(value)
    return extractor.get_text()


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def select_sources(profile: str, limit: int | None = None) -> list[SourceSeed]:
    selected = [seed for seed in DEFAULT_SOURCES if profile == "all" or profile in seed.profiles]
    if limit is not None:
        return selected[:limit]
    return selected


def build_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=httpx.Timeout(45.0, connect=20.0),
    )


def fetch_with_retry(client: httpx.Client, url: str, *, expect_binary: bool = False) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(1.5 * attempt)
    assert last_error is not None
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def parse_seed_page(seed: SourceSeed, html_text: str) -> ParsedPage:
    if seed.regulator == "rbi":
        return parse_rbi_page(seed, html_text)
    if seed.regulator == "sebi":
        return parse_sebi_page(seed, html_text)
    raise ValueError(f"Unsupported regulator: {seed.regulator}")


def parse_rbi_page(seed: SourceSeed, html_text: str) -> ParsedPage:
    pdf_match = re.search(
        r"<a[^>]+id=['\"]APDF_[^'\"]+['\"][^>]+href=['\"](?P<url>https://rbidocs\.rbi\.org\.in/rdocs/[^'\"]+\.pdf)['\"]",
        html_text,
        flags=re.IGNORECASE,
    )
    if not pdf_match:
        raise RuntimeError(f"No direct RBI PDF link found on {seed.page_url}")

    title = first_clean_match(
        html_text,
        (
            r"<td[^>]*class=['\"]tableheader['\"][^>]*>\s*<b>(?P<value>.*?)</b>",
            r"<img[^>]+alt=['\"]PDF - (?P<value>.*?)['\"]",
            r"<title>(?P<value>.*?)</title>",
        ),
    )
    published_date = first_date_match(
        html_text,
        (
            r"<p[^>]*align=['\"]right['\"][^>]*>(?P<value>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})",
            r"<p[^>]*class=['\"]head['\"][^>]*>.*?</p>\s*<p[^>]*align=['\"]right['\"][^>]*>(?P<value>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})",
        ),
    )
    related_urls = tuple(
        dedupe(
            re.findall(
                r"https://www\.rbi\.org\.in/Scripts/(?:NotificationUser|BS_ViewMasDirections|BS_ViewMasCirculardetails)\.aspx\?Id=\d+(?:&Mode=\d+)?",
                html_text,
                flags=re.IGNORECASE,
            )
        )
    )
    extractor = SectionTextExtractor("tr", "class", "tablecontent2")
    extractor.feed(html_text)
    text_content = extractor.get_text()
    if not text_content:
        raise RuntimeError(f"Unable to extract RBI page text from {seed.page_url}")

    return ParsedPage(
        regulator="rbi",
        title=title,
        published_date=published_date,
        page_url=seed.page_url,
        asset_url=pdf_match.group("url"),
        asset_format="txt",
        text_content=text_content,
        related_urls=related_urls,
    )


def parse_sebi_page(seed: SourceSeed, html_text: str) -> ParsedPage:
    iframe_match = re.search(
        r"<iframe[^>]+src=['\"](?P<src>[^'\"]*file=https://[^'\"]+\.pdf)['\"]",
        html_text,
        flags=re.IGNORECASE,
    )
    pdf_url = ""
    if iframe_match:
        src = html.unescape(iframe_match.group("src"))
        resolved_src = urljoin(seed.page_url, src)
        parsed = urlparse(resolved_src)
        file_values = parse_qs(parsed.query).get("file", [])
        if file_values:
            pdf_url = unquote(file_values[0])

    if not pdf_url:
        direct_match = re.search(
            r"https://www\.sebi\.gov\.in/sebi_data/attachdocs/[^'\"\s>]+\.pdf",
            html_text,
            flags=re.IGNORECASE,
        )
        if direct_match:
            pdf_url = direct_match.group(0)

    if not pdf_url:
        raise RuntimeError(f"No direct SEBI PDF link found on {seed.page_url}")

    title = first_clean_match(
        html_text,
        (
            r"<h1>(?P<value>.*?)<div class=['\"]social-share-btn",
            r"<meta[^>]+name=['\"]title['\"][^>]+content=['\"](?P<value>.*?)['\"]",
            r"<title>(?P<value>.*?)</title>",
        ),
    )
    title = re.sub(r"^SEBI\s*\|\s*", "", title).strip()
    published_date = first_date_match(
        html_text,
        (
            r"<div class=['\"]date_value['\"]>\s*<h5>(?P<value>.*?)</h5>",
            r"<h5>(?P<value>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})</h5>",
        ),
    )

    return ParsedPage(
        regulator="sebi",
        title=title,
        published_date=published_date,
        page_url=seed.page_url,
        asset_url=pdf_url,
        asset_format="pdf",
        text_content="",
        related_urls=(),
    )


def first_clean_match(html_text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = clean_fragment(match.group("value"))
            if value:
                return value
    raise RuntimeError("Unable to extract page title")


def first_date_match(html_text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = normalize_date(match.group("value"))
            if value:
                return value
    raise RuntimeError("Unable to extract publication date")


def build_output_path(output_dir: Path, seed: SourceSeed, parsed: ParsedPage) -> Path:
    date_prefix = parsed.published_date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", parsed.published_date) else "undated"
    filename = f"{seed.regulator.upper()}_{date_prefix}_{slugify(parsed.title)}.{parsed.asset_format}"
    return output_dir / seed.regulator / filename


def ensure_text_fallback(content: bytes, asset_url: str) -> tuple[bytes, str]:
    text = content.decode("utf-8", errors="replace")
    extractor = TextExtractor()
    extractor.feed(text)
    normalized = extractor.get_text()
    if not normalized:
        raise RuntimeError(f"Expected text content for {asset_url}, but nothing extractable was found.")
    return normalized.encode("utf-8"), "txt"


def download_seed(
    client: httpx.Client,
    seed: SourceSeed,
    output_dir: Path,
) -> DownloadRecord:
    page_response = fetch_with_retry(client, seed.page_url)
    page_text = page_response.text
    parsed = parse_seed_page(seed, page_text)

    asset_format = parsed.asset_format
    if parsed.asset_format == "txt":
        content = parsed.text_content.encode("utf-8")
    else:
        asset_response = fetch_with_retry(client, parsed.asset_url, expect_binary=True)
        content = asset_response.content
        content_type = asset_response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not parsed.asset_url.lower().endswith(".pdf"):
            content, asset_format = ensure_text_fallback(content, parsed.asset_url)
            parsed = ParsedPage(
                regulator=parsed.regulator,
                title=parsed.title,
                published_date=parsed.published_date,
                page_url=parsed.page_url,
                asset_url=parsed.asset_url,
                asset_format=asset_format,
                text_content=parsed.text_content,
                related_urls=parsed.related_urls,
            )

    destination = build_output_path(output_dir, seed, parsed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)

    sha256 = hashlib.sha256(content).hexdigest()
    downloaded_at = datetime.now(timezone.utc).isoformat()

    return DownloadRecord(
        key=seed.key,
        regulator=seed.regulator,
        category=seed.category,
        title=parsed.title,
        published_date=parsed.published_date,
        page_url=seed.page_url,
        asset_url=parsed.asset_url,
        local_path=str(destination.relative_to(ROOT)),
        sha256=sha256,
        size_bytes=len(content),
        asset_format=asset_format,
        downloaded_at=downloaded_at,
    )


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_manifest(path: Path, records: list[DownloadRecord]) -> None:
    existing = load_manifest(path)
    by_key = {item["key"]: item for item in existing}
    for record in records:
        by_key[record.key] = asdict(record)
    payload = [by_key[key] for key in sorted(by_key)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def scrape_sources(
    profile: str,
    output_dir: Path,
    manifest_path: Path,
    limit: int | None = None,
) -> list[DownloadRecord]:
    seeds = select_sources(profile, limit)
    if not seeds:
        raise RuntimeError(f"No sources selected for profile={profile!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[DownloadRecord] = []
    with build_client() as client:
        for seed in seeds:
            print(f"[fetch] {seed.key} -> {seed.page_url}")
            record = download_seed(client, seed, output_dir)
            records.append(record)
            print(
                f"[saved] {record.local_path} "
                f"({record.size_bytes} bytes, sha256={record.sha256[:12]}...)"
            )

    upsert_manifest(manifest_path, records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official RBI and SEBI documents into FinSight's raw corpus.")
    parser.add_argument(
        "--profile",
        default="core",
        choices=["core", "rbi", "sebi", "all"],
        help="Select which official source set to scrape.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "raw"),
        help="Directory where downloaded documents should be stored.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(ROOT / "data" / "raw" / "regulatory_manifest.json"),
        help="Path to the JSON manifest describing downloaded files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of seed sources to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    started_at = time.time()

    records = scrape_sources(
        profile=args.profile,
        output_dir=output_dir,
        manifest_path=manifest_path,
        limit=args.limit,
    )
    summary = {
        "profile": args.profile,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "documents_saved": len(records),
        "duration_seconds": round(time.time() - started_at, 2),
        "files": [asdict(record) for record in records],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
