"""EveryDB2 マニュアルスクレイピングスクリプト

https://everydb.iwinz.net/edb2_manual のデータフォーマットページとコード表をスクレイピングし、
Markdown ドキュメントとして保存する。

使い方:
  # 全ページ（データフォーマット + コード表）をスクレイピング
  python scripts/scrape_everydb2_manual.py --all --outdir docs/everydb2_manual

  # データフォーマットページのみ
  python scripts/scrape_everydb2_manual.py --formats --outdir docs/everydb2_manual

  # コード表のみ
  python scripts/scrape_everydb2_manual.py --codes --outdir docs/everydb2_manual

  # 特定ページのみ（カンマ区切りで複数指定可）
  python scripts/scrape_everydb2_manual.py --pages 03,04 --outdir docs/everydb2_manual

  # インデックスページのみ生成（スクレイピングなし）
  python scripts/scrape_everydb2_manual.py --index-only --outdir docs/everydb2_manual

  # ヘッドレスモードをオフ（ブラウザを表示）
  python scripts/scrape_everydb2_manual.py --all --no-headless
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://everydb.iwinz.net/edb2_manual"
FORMAT_LIST_SLUG = "12-3-00-FormatList.html"
CODE_PAGE_SLUG = "12-3-99-CODE.html"

CODE_REF_RE = re.compile(r"コード表\s*(\d{4})\.([^<>\s]+)")
RECORD_SPEC_RE = re.compile(r"(\w+)\s*をセット")

# 59 data format pages (slug → human-readable table name)
PAGES: dict[str, str] = {
    "01-TOKU_RACE": "特別レース定義",
    "02-TOKU": "特別登録馬",
    "03-RACE": "レース",
    "04-UMA_RACE": "出走馬",
    "05-HARAI": "払戻",
    "06-HYOSU": "票数",
    "07-HYOSU_TANPUKU": "票数(単複)",
    "08-HYOSU_WAKU": "票数(枠連)",
    "09-HYOSU_UMARENWIDE": "票数(馬連ワイド)",
    "10-HYOSU_UMATAN": "票数(馬単)",
    "11-HYOSU_SANREN": "票数(3連複)",
    "12-HYOSU2": "票数2",
    "13-HYOSU_SANRENTAN": "票数(3連単)",
    "14-ODDS_TANPUKUWAKU_HEAD": "オッズ単複枠ヘッダ",
    "15-ODDS_TANPUKU": "オッズ単複",
    "16-ODDS_WAKU": "オッズ枠連",
    "17-ODDS_UMAREN_HEAD": "オッズ馬連ヘッダ",
    "18-ODDS_UMAREN": "オッズ馬連",
    "19-ODDS_WIDE_HEAD": "オッズワイドヘッダ",
    "20-ODDS_WIDE": "オッズワイド",
    "21-ODDS_UMATAN_HEAD": "オッズ馬単ヘッダ",
    "22-ODDS_UMATAN": "オッズ馬単",
    "23-ODDS_SANREN_HEAD": "オッズ3連複ヘッダ",
    "24-ODDS_SANREN": "オッズ3連複",
    "25-ODDS_SANRENTAN_HEAD": "オッズ3連単ヘッダ",
    "26-ODDS_SANRENTAN": "オッズ3連単",
    "27-UMA": "競走馬マスタ",
    "28-KISYU": "騎手マスタ",
    "29-KISYU_SEISEKI": "騎手成績",
    "30-CHOKYO": "調教師マスタ",
    "31-CHOKYO_SEISEKI": "調教師成績",
    "32-SEISAN": "生産者マスタ",
    "33-BANUSI": "馬主マスタ",
    "34-HANSYOKU": "繁殖馬マスタ",
    "35-SANKU": "産駒マスタ",
    "36-RECORD": "レコード",
    "37-HANRO": "馬齢",
    "38-BATAIJYU": "馬体重",
    "39-TENKO_BABA": "天候馬場",
    "40-TORIKESI_JYOGAI": "取消除外",
    "41-KISYU_CHANGE": "騎手変更",
    "42-HASSOU_JIKOKU_CHANGE": "発走時刻変更",
    "43-COURSE_CHANGE": "コース変更",
    "44-MINING": "マイニング",
    "45-SCHEDULE": "スケジュール",
    "46-JODDS_TANPUKUWAKU_HEAD": "時系列オッズ単複枠ヘッダ",
    "47-JODDS_TANPUKU": "時系列オッズ単複",
    "48-JODDS_WAKU": "時系列オッズ枠連",
    "49-JODDS_UMAREN_HEAD": "時系列オッズ馬連ヘッダ",
    "50-JODDS_UMAREN": "時系列オッズ馬連",
    "51-SALE": "セール",
    "52-BAMEIORIGIN": "馬名のオリジナル",
    "53-KEITO": "系統",
    "54-COURSE": "コース",
    "55-TAISENGATA_MINING": "対戦型マイニング",
    "56-JYUSYOSIKI_HEAD": "受賞式ヘッダ",
    "57-JYUSYOSIKI": "受賞式",
    "58-JOGAIBA": "除外馬",
    "59-WOOD_CHIP": "ウッドチップ",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FieldDef:
    """Single field definition row from a data format table."""

    row_no: int
    is_key: bool
    name_ja: str
    field_name: str
    col_type: str
    size: str
    default_val: str
    description: str
    code_refs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class FormatPage:
    """Parsed data format page."""

    slug: str
    table_name_ja: str
    table_name_en: str
    record_spec: str
    fields: list[FieldDef]


@dataclass
class CodeEntry:
    """Single code value entry."""

    code_value: str
    short_name: str = ""
    full_name: str = ""


@dataclass
class CodeTable:
    """One code table from the CODE page."""

    code_id: str
    table_name: str
    meta_info: str = ""
    headers: list[str] = field(default_factory=list)
    entries: list[CodeEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EveryDB2 マニュアルスクレイピング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", dest="scrape_all", help="全ページをスクレイピング"
    )
    group.add_argument("--formats", action="store_true", help="データフォーマットページのみ")
    group.add_argument("--codes", action="store_true", help="コード表のみ")
    group.add_argument(
        "--pages",
        type=str,
        help="特定ページのみ (例: 03,04 または 03-04-UMA_RACE,27-UMA)",
    )
    group.add_argument("--index-only", action="store_true", help="INDEX.md のみ生成")

    parser.add_argument(
        "--outdir",
        type=str,
        default="docs/everydb2_manual",
        help="出力ディレクトリ (default: docs/everydb2_manual)",
    )
    parser.add_argument("--no-headless", action="store_true", help="ブラウザを表示する")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="ページ間の待機秒数 (default: 1.0)",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG ログを出力")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Browser management
# ---------------------------------------------------------------------------


def create_browser(headless: bool = True) -> Any:
    """Create and return a Playwright browser instance.

    Returns a tuple of (playwright_instance, browser, context).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error(
            "playwright がインストールされていません。"
            "pip install playwright && playwright install chromium を実行してください。"
        )
        sys.exit(1)

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=headless)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 900},
    )
    return pw, browser, context


def close_browser(pw: Any, browser: Any) -> None:
    """Close browser and Playwright instance."""
    browser.close()
    pw.stop()


# ---------------------------------------------------------------------------
# Page discovery
# ---------------------------------------------------------------------------


def discover_format_slugs(context: Any) -> list[str]:
    """Discover all format page slugs from the FormatList page.

    Returns list of slugs like ['01-TOKU_RACE', '02-TOKU', ...].
    """
    url = f"{BASE_URL}/{FORMAT_LIST_SLUG}"
    logger.info("FormatList ページを取得: %s", url)
    page = context.new_page()
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("table", timeout=10000)

        slugs: list[str] = []
        links = page.query_selector_all("table a[href]")
        for link in links:
            href = link.get_attribute("href") or ""
            match = re.match(r"12-3-(\d{2})-([^.]+)\.html", href)
            if match:
                slug = f"{match.group(1)}-{match.group(2)}"
                slugs.append(slug)

        logger.info("FormatList から %d ページを発見", len(slugs))
        return slugs
    finally:
        page.close()


def resolve_page_slugs(pages_arg: str) -> list[str]:
    """Resolve --pages argument to a list of slugs.

    Accepts formats like '03,04' or '03-04-UMA_RACE,27-UMA'.
    """
    slugs: list[str] = []
    for part in pages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        # Try direct slug match (e.g., '03-04-UMA_RACE')
        if part in PAGES:
            slugs.append(part)
            continue
        # Try numeric prefix match (e.g., '03' -> '03-RACE')
        matched = [s for s in PAGES if s.startswith(f"{part}-")]
        if len(matched) == 1:
            slugs.append(matched[0])
        elif len(matched) > 1:
            logger.warning("ambiguous page prefix '%s': %s", part, matched)
            slugs.append(matched[0])
        else:
            logger.warning("page not found: '%s'", part)
    return slugs


# ---------------------------------------------------------------------------
# Data format page parsing
# ---------------------------------------------------------------------------


def _extract_record_spec(description_text: str) -> str:
    """Extract RecordSpec from row-1 description text."""
    m = RECORD_SPEC_RE.search(description_text)
    return m.group(1) if m else ""


def _parse_code_refs(cell_html: str) -> list[tuple[str, str]]:
    """Extract code references from a cell's inner HTML.

    Returns list of (code_id, code_name) tuples.
    """
    refs: list[tuple[str, str]] = []
    for m in CODE_REF_RE.finditer(cell_html):
        refs.append((m.group(1), m.group(2)))
    return refs


def _normalize_text(text: str) -> str:
    """Normalize whitespace in extracted text."""
    return re.sub(r"\s+", " ", text).strip()


def parse_format_page(context: Any, slug: str) -> FormatPage:
    """Parse a single data format page.

    Returns a FormatPage with all field definitions.
    """
    url = f"{BASE_URL}/12-3-{slug}.html"
    logger.info("データフォーマットページを取得: %s", url)
    page = context.new_page()
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("table", timeout=10000)

        # Extract page title
        title_el = page.query_selector("h2, h3, .page-header h2, .page-header h1")
        title_text = _normalize_text(title_el.inner_text()) if title_el else slug

        # Find the main data table (usually the first large table)
        tables = page.query_selector_all("table")
        data_table = None
        for tbl in tables:
            header_row = tbl.query_selector("tr")
            if header_row is None:
                continue
            header_cells = header_row.query_selector_all("th, td")
            if len(header_cells) >= 6:
                data_table = tbl
                break
        if data_table is None:
            logger.warning("テーブルが見つかりません: %s", slug)
            return FormatPage(
                slug=slug,
                table_name_ja=title_text,
                table_name_en="",
                record_spec="",
                fields=[],
            )

        rows = data_table.query_selector_all("tr")
        fields: list[FieldDef] = []
        record_spec = ""
        table_name_en = ""

        for row_idx, row in enumerate(rows):
            cells = row.query_selector_all("th, td")
            if row_idx == 0:
                # Header row - skip
                continue

            if len(cells) < 6:
                continue

            cell_texts = [_normalize_text(c.inner_text()) for c in cells]
            cell_htmls = [c.inner_html() for c in cells]

            row_no_str = cell_texts[0]
            if not row_no_str.strip():
                continue

            try:
                row_no = int(row_no_str)
            except ValueError:
                # Might be a sub-header or continuation row; skip
                continue

            # Extract key marker
            is_key = "〇" in cell_texts[1] or "○" in cell_texts[1] or "Ｐ" in cell_texts[1]
            # Some pages use "P" to mark primary key
            if cell_texts[1].strip().upper() == "P":
                is_key = True

            name_ja = cell_texts[2]
            field_name = cell_texts[3]
            col_type = cell_texts[4] if len(cell_texts) > 4 else ""
            size = cell_texts[5] if len(cell_texts) > 5 else ""
            default_val = cell_texts[6] if len(cell_texts) > 6 else ""
            description = cell_texts[7] if len(cell_texts) > 7 else ""

            # Extract code references from description cell
            desc_html = cell_htmls[7] if len(cell_htmls) > 7 else ""
            code_refs = _parse_code_refs(desc_html)

            # Extract RecordSpec from row 1 description
            if row_no == 1:
                record_spec = _extract_record_spec(description)
                if not table_name_en:
                    table_name_en = record_spec

            fields.append(
                FieldDef(
                    row_no=row_no,
                    is_key=is_key,
                    name_ja=name_ja,
                    field_name=field_name,
                    col_type=col_type,
                    size=size,
                    default_val=default_val,
                    description=description,
                    code_refs=code_refs,
                )
            )

        return FormatPage(
            slug=slug,
            table_name_ja=title_text,
            table_name_en=table_name_en,
            record_spec=record_spec,
            fields=fields,
        )
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Markdown generation for format pages
# ---------------------------------------------------------------------------


def _escape_md_table(text: str) -> str:
    """Escape pipe characters for use in Markdown tables."""
    return text.replace("|", "\\|").replace("\n", " ")


def format_page_to_markdown(fp: FormatPage) -> str:
    """Convert a FormatPage to Markdown string."""
    lines: list[str] = []

    # Title
    ja_name = fp.table_name_ja or fp.slug
    en_name = fp.table_name_en
    lines.append(f"# {ja_name}")
    lines.append("")

    if en_name:
        lines.append(f"**テーブル:** `{en_name}`")
        lines.append("")

    if fp.record_spec:
        lines.append(f"**RecordSpec:** `{fp.record_spec}`")
        lines.append("")

    lines.append(f"**フィールド数:** {len(fp.fields)}")
    lines.append("")

    # Table header
    lines.append("| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |")
    lines.append("|---:|:---:|---|---|---|---:|---|---|")

    for f in fp.fields:
        key_mark = "PK" if f.is_key else ""
        desc = _escape_md_table(f.description)
        # Add code refs inline if present
        if f.code_refs:
            ref_strs = [f"[コード表{cid}.{cname}](CODE.md#{cid})" for cid, cname in f.code_refs]
            desc = desc.rstrip()
            if desc:
                desc += " "
            desc += ", ".join(ref_strs)

        lines.append(
            f"| {f.row_no} "
            f"| {key_mark} "
            f"| {_escape_md_table(f.name_ja)} "
            f"| `{_escape_md_table(f.field_name)}` "
            f"| {f.col_type} "
            f"| {f.size} "
            f"| {_escape_md_table(f.default_val)} "
            f"| {desc} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code table scraping
# ---------------------------------------------------------------------------


def scrape_code_tables(context: Any) -> list[CodeTable]:
    """Scrape all code tables from the CODE page.

    The CODE page contains multiple independent tables, each preceded by
    a title row indicating the code ID and name.
    """
    url = f"{BASE_URL}/{CODE_PAGE_SLUG}"
    logger.info("コード表ページを取得: %s", url)
    page = context.new_page()
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("table", timeout=10000)

        tables = page.query_selector_all("table")
        code_tables: list[CodeTable] = []

        for table_idx, table_el in enumerate(tables):
            rows = table_el.query_selector_all("tr")
            if len(rows) < 2:
                continue

            # Row 0: title row with a single <th> spanning all columns
            # e.g., "コード表 2001 競馬場コード"
            title_cell = rows[0].query_selector("th, td")
            if title_cell is None:
                continue
            title_text = _normalize_text(title_cell.inner_text())

            # Extract code ID and name from title
            code_id_match = re.search(r"(\d{4})\s+(.+)", title_text)
            if not code_id_match:
                logger.debug("テーブル %d: コードIDを検出できません ('%s')", table_idx, title_text)
                continue
            code_id = code_id_match.group(1)
            table_name = code_id_match.group(2).strip()

            # Determine header structure
            # Row 1 may be meta headers, row 2 may be sub-headers
            # We need to find the actual data rows
            header_row_idx = 1
            if len(rows) > 2:
                # Check if row 1 has meta info and row 2 has actual headers
                row1_cells = rows[1].query_selector_all("th, td")
                row1_texts = [_normalize_text(c.inner_text()) for c in row1_cells]
                # If row 1 looks like meta info (single cell or few cells), skip it
                if len(row1_cells) <= 2 and any("コード" in t for t in row1_texts):
                    header_row_idx = 2

            # Extract headers
            if header_row_idx < len(rows):
                header_cells = rows[header_row_idx].query_selector_all("th, td")
                headers = [_normalize_text(c.inner_text()) for c in header_cells]
            else:
                headers = []

            # Extract data rows (everything after header row)
            entries: list[CodeEntry] = []
            for row_idx in range(header_row_idx + 1, len(rows)):
                cells = rows[row_idx].query_selector_all("td")
                if not cells:
                    continue
                cell_texts = [_normalize_text(c.inner_text()) for c in cells]

                if not cell_texts or not cell_texts[0].strip():
                    continue

                code_value = cell_texts[0]
                short_name = cell_texts[1] if len(cell_texts) > 1 else ""
                full_name = cell_texts[2] if len(cell_texts) > 2 else short_name

                entries.append(
                    CodeEntry(
                        code_value=code_value,
                        short_name=short_name,
                        full_name=full_name,
                    )
                )

            if entries:
                code_tables.append(
                    CodeTable(
                        code_id=code_id,
                        table_name=table_name,
                        headers=headers,
                        entries=entries,
                    )
                )
                logger.info("コード表 %s (%s): %dエントリ", code_id, table_name, len(entries))

        logger.info("合計 %d コード表を取得", len(code_tables))
        return code_tables
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Markdown generation for code tables
# ---------------------------------------------------------------------------


def code_tables_to_markdown(tables: list[CodeTable]) -> str:
    """Convert all code tables to a single Markdown string."""
    lines: list[str] = []
    lines.append("# EveryDB2 コード表")
    lines.append("")
    lines.append("EveryDB2 マニュアルから抽出したコード表一覧。")
    lines.append("")

    # TOC
    lines.append("## 目次")
    lines.append("")
    for ct in tables:
        anchor = ct.code_id
        lines.append(f"- [{ct.code_id} {ct.table_name}](#{anchor})")
    lines.append("")

    # Individual tables
    for ct in tables:
        lines.append(f'<a id="{ct.code_id}"></a>')
        lines.append("")
        lines.append(f"## {ct.code_id} {ct.table_name}")
        lines.append("")

        # Determine how many meaningful columns we have
        n_cols = len(ct.headers) if ct.headers else 3
        # Ensure we have at least 3 columns
        n_cols = max(n_cols, 3)

        # Build header
        if ct.headers and len(ct.headers) >= 2:
            # Use actual headers
            hdrs = ct.headers[:n_cols]
            # Pad if needed
            while len(hdrs) < n_cols:
                hdrs.append("")
            lines.append("| " + " | ".join(hdrs) + " |")
            lines.append("|" + "|".join(["---"] * n_cols) + "|")
        else:
            lines.append("| コード | 略称 | 名称 |")
            lines.append("|---|---|---|")

        for entry in ct.entries:
            cells = [entry.code_value, entry.short_name, entry.full_name]
            # Pad or truncate to n_cols
            while len(cells) < n_cols:
                cells.append("")
            cells = cells[:n_cols]
            escaped = [_escape_md_table(c) for c in cells]
            lines.append("| " + " | ".join(escaped) + " |")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# INDEX.md generation
# ---------------------------------------------------------------------------


def generate_index(
    outdir: str,
    format_pages: dict[str, FormatPage],
    code_tables: list[CodeTable],
) -> str:
    """Generate INDEX.md with links to all scraped pages."""
    lines: list[str] = []
    lines.append("# EveryDB2 マニュアル - インデックス")
    lines.append("")
    lines.append(
        "EveryDB2 (JRA-VAN DataLab) データベースのテーブル定義とコード表を"
        "スクレイピングして生成したドキュメント。"
    )
    lines.append("")

    # Data formats
    lines.append("## データフォーマット")
    lines.append("")
    lines.append("| No | テーブル名 | 日本語名 | RecordSpec | フィールド数 |")
    lines.append("|---:|---|---|---|---:|")

    for slug, name_ja in PAGES.items():
        fp = format_pages.get(slug)
        if fp:
            link = f"[{name_ja}]({slug}.md)"
            spec = f"`{fp.record_spec}`" if fp.record_spec else "-"
            n_fields = len(fp.fields)
        else:
            link = name_ja
            spec = "-"
            n_fields = 0
        lines.append(f"| {slug.split('-')[0]} | {link} | {name_ja} | {spec} | {n_fields} |")

    lines.append("")

    # Code tables
    if code_tables:
        lines.append("## コード表")
        lines.append("")
        lines.append("| コードID | 名称 | エントリ数 |")
        lines.append("|---|---|---:|")
        for ct in code_tables:
            anchor = ct.code_id
            n_entries = len(ct.entries)
            lines.append(f"| {ct.code_id} | [{ct.table_name}](CODE.md#{anchor}) | {n_entries} |")
        lines.append("")

    # Generation info
    lines.append("---")
    lines.append("")
    lines.append(f"- 生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- ソース: {BASE_URL}")
    lines.append("")

    content = "\n".join(lines)
    index_path = Path(outdir) / "INDEX.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(content, encoding="utf-8")
    logger.info("INDEX.md を生成: %s (%d bytes)", index_path, len(content))
    return str(index_path)


# ---------------------------------------------------------------------------
# Metadata for cross-run validation
# ---------------------------------------------------------------------------


def save_meta(
    outdir: str,
    format_pages: dict[str, FormatPage],
    code_tables: list[CodeTable],
) -> None:
    """Save scrape metadata for cross-run validation."""
    meta_path = Path(outdir) / ".scrape_meta.json"
    prev_path = Path(outdir) / ".prev_scrape_meta.json"

    # Rotate previous meta
    if meta_path.exists():
        if prev_path.exists():
            prev_path.unlink()
        meta_path.rename(prev_path)

    meta: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "format_pages": {},
        "code_tables": {},
    }
    for slug, fp in format_pages.items():
        meta["format_pages"][slug] = {
            "table_name_en": fp.table_name_en,
            "record_spec": fp.record_spec,
            "n_fields": len(fp.fields),
        }
    for ct in code_tables:
        meta["code_tables"][ct.code_id] = {
            "table_name": ct.table_name,
            "n_entries": len(ct.entries),
        }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("メタデータを保存: %s", meta_path)


def check_meta(outdir: str) -> dict[str, Any] | None:
    """Load previous scrape metadata if available."""
    meta_path = Path(outdir) / ".scrape_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Check previous metadata
    prev_meta = check_meta(outdir)
    if prev_meta:
        logger.info("前回のスクレイプメタデータを検出: %s", prev_meta.get("timestamp", "unknown"))

    # Load any existing scraped data for --index-only mode
    format_pages: dict[str, FormatPage] = {}
    code_tables: list[CodeTable] = []

    if args.index_only:
        # Try to load from metadata to generate index
        logger.info("INDEX.md のみ生成（スクレイピングなし）")
        if prev_meta:
            # Reconstruct minimal FormatPage objects from metadata
            for slug, info in prev_meta.get("format_pages", {}).items():
                format_pages[slug] = FormatPage(
                    slug=slug,
                    table_name_ja=PAGES.get(slug, slug),
                    table_name_en=info.get("table_name_en", ""),
                    record_spec=info.get("record_spec", ""),
                    fields=[],  # We don't need fields for index
                )
            for cid, info in prev_meta.get("code_tables", {}).items():
                code_tables.append(
                    CodeTable(
                        code_id=cid,
                        table_name=info.get("table_name", ""),
                        entries=[],
                    )
                )
        generate_index(outdir, format_pages, code_tables)
        return

    # Create browser
    pw, browser, context = create_browser(headless=not args.no_headless)
    try:
        # Determine which pages to scrape
        target_slugs: list[str] = []

        if args.scrape_all or args.formats:
            target_slugs = list(PAGES.keys())
        elif args.pages:
            target_slugs = resolve_page_slugs(args.pages)

        # Scrape data format pages
        if target_slugs:
            logger.info("%d データフォーマットページをスクレイピング", len(target_slugs))
            for i, slug in enumerate(target_slugs):
                logger.info("[%d/%d] %s", i + 1, len(target_slugs), slug)
                try:
                    fp = parse_format_page(context, slug)
                    format_pages[slug] = fp

                    # Write individual markdown
                    md = format_page_to_markdown(fp)
                    md_path = Path(outdir) / f"{slug}.md"
                    md_path.write_text(md, encoding="utf-8")
                    logger.info("  → %s (%d fields)", md_path.name, len(fp.fields))
                except Exception as e:
                    logger.error("  スクレイプ失敗 %s: %s", slug, e)

                if args.delay > 0 and i < len(target_slugs) - 1:
                    time.sleep(args.delay)

        # Scrape code tables
        if args.scrape_all or args.codes:
            logger.info("コード表をスクレイピング")
            try:
                code_tables = scrape_code_tables(context)
                md = code_tables_to_markdown(code_tables)
                md_path = Path(outdir) / "CODE.md"
                md_path.write_text(md, encoding="utf-8")
                logger.info("  → CODE.md (%d tables)", len(code_tables))
            except Exception as e:
                logger.error("コード表スクレイプ失敗: %s", e)

        # Generate index
        generate_index(outdir, format_pages, code_tables)

        # Save metadata
        save_meta(outdir, format_pages, code_tables)

        # Summary
        logger.info("=== 完了 ===")
        logger.info("データフォーマット: %d ページ", len(format_pages))
        logger.info("コード表: %d テーブル", len(code_tables))
        logger.info("出力先: %s", outdir)
    finally:
        close_browser(pw, browser)


if __name__ == "__main__":
    main()
