# EveryDB2 Manual Scraping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scrape EveryDB2 manual Chapter 12 (59 data format pages + code tables) into organized markdown files under `docs/everydb2/`.

**Architecture:** Single Python script using Playwright (standalone, not MCP). Playwright launches Chromium, navigates each HTML page, extracts table data via DOM queries, converts to markdown, and writes files. Three-phase approach: (1) discover pages from FormatList, (2) scrape data format pages + code tables, (3) generate INDEX.md.

**Tech Stack:** Python 3.11, Playwright (standalone), argparse, re, pathlib, json

**Spec:** `docs/superpowers/specs/2026-03-29-everydb2-manual-scraping-design.md`

**Test policy:** No automated tests (scraping script, manual execution). Verify via `--dry-run` and visual inspection.

**Mypy note:** Playwright is an untyped dependency. Add `# type: ignore[import-untyped]` at the import site. Do NOT run `mypy src/` on this script.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add `scraping` optional dependency |
| `scripts/scrape_everydb2_manual.py` | Create | Main scraping script (single file) |
| `docs/everydb2/` | Create (at runtime) | Output directory for generated markdown |
| `docs/everydb2/INDEX.md` | Create (at runtime) | Master index of all data formats |
| `docs/everydb2/12-3-XX-*.md` | Create (at runtime) | Individual data format pages |
| `docs/everydb2/codes/INDEX.md` | Create (at runtime) | Code table index |
| `docs/everydb2/codes/C*.md` | Create (at runtime) | Individual code tables |

---

### Task 1: Add playwright dependency

**Files:**
- Modify: `pyproject.toml:21-26`

- [ ] **Step 1: Add `scraping` optional dependency to pyproject.toml**

Add after the `dev` optional dependency block (after line 26):

```toml
scraping = [
    "playwright>=1.40",
]
```

- [ ] **Step 2: Install the dependency**

```bash
pip install -e ".[scraping]"
playwright install chromium
```

- [ ] **Step 3: Verify playwright is installed**

```bash
playwright --version
```

Expected: version >= 1.40

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add playwright as scraping optional dependency"
```

---

### Task 2: Create script skeleton with browser management

**Files:**
- Create: `scripts/scrape_everydb2_manual.py`

- [ ] **Step 1: Write script skeleton with argparse CLI, browser lifecycle, and retry logic**

Key fixes from review:
- `import os` removed (unused)
- `import json` at module level (not inside functions)
- `--delay` bounds check added
- Playwright import has `# type: ignore`

```python
"""EveryDB2マニュアル 12章データフォーマット スクレイピング

使い方:
  # 全ページスクレイピング
  python scripts/scrape_everydb2_manual.py

  # 特定ページのみ
  python scripts/scrape_everydb2_manual.py --pages 03 04 05

  # コード表のみ
  python scripts/scrape_everydb2_manual.py --codes-only

  # ドライラン
  python scripts/scrape_everydb2_manual.py --dry-run

  # リクエスト間隔調整
  python scripts/scrape_everydb2_manual.py --delay 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# プロジェクトルート
ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://everydb.iwinz.net/edb2_manual"
OUTPUT_DIR = ROOT / "docs" / "everydb2"
CODES_DIR = OUTPUT_DIR / "codes"

# コード表参照の正規表現: コード表 NNNN.NAME
CODE_REF_RE = re.compile(r"コード表\s*(\d{4})\.([^<>\s]+)")

# ハードコード: ページ一覧 (フォールバック用)
PAGES: list[dict[str, str]] = [
    {"no": "01", "slug": "12-3-01-TOKU_RACE", "name": "特別レース"},
    {"no": "02", "slug": "12-3-02-TOKU", "name": "特別登録馬"},
    {"no": "03", "slug": "12-3-03-RACE", "name": "レース詳細"},
    {"no": "04", "slug": "12-3-04-UMA_RACE", "name": "馬毎レース情報"},
    {"no": "05", "slug": "12-3-05-HARAI", "name": "払戻"},
    {"no": "06", "slug": "12-3-06-HYOSU", "name": "票数"},
    {"no": "07", "slug": "12-3-07-HYOSU_TANPUKU", "name": "票数_単複"},
    {"no": "08", "slug": "12-3-08-HYOSU_WAKU", "name": "票数_枠連"},
    {"no": "09", "slug": "12-3-09-HYOSU_UMARENWIDE", "name": "票数_馬連_ワイド"},
    {"no": "10", "slug": "12-3-10-HYOSU_UMATAN", "name": "票数_馬単"},
    {"no": "11", "slug": "12-3-11-HYOSU_SANREN", "name": "票数_3連複"},
    {"no": "12", "slug": "12-3-12-HYOSU2", "name": "票数2"},
    {"no": "13", "slug": "12-3-13-HYOSU_SANRENTAN", "name": "票数_3連単"},
    {"no": "14", "slug": "12-3-14-ODDS_TANPUKUWAKU_HEAD", "name": "オッズ_単複枠_ヘッダ"},
    {"no": "15", "slug": "12-3-15-ODDS_TANPUKU", "name": "オッズ_単複"},
    {"no": "16", "slug": "12-3-16-ODDS_WAKU", "name": "オッズ_枠連"},
    {"no": "17", "slug": "12-3-17-ODDS_UMAREN_HEAD", "name": "オッズ_馬連_ヘッダ"},
    {"no": "18", "slug": "12-3-18-ODDS_UMAREN", "name": "オッズ_馬連"},
    {"no": "19", "slug": "12-3-19-ODDS_WIDE_HEAD", "name": "オッズ_ワイド_ヘッダ"},
    {"no": "20", "slug": "12-3-20-ODDS_WIDE", "name": "オッズ_ワイド"},
    {"no": "21", "slug": "12-3-21-ODDS_UMATAN_HEAD", "name": "オッズ_馬単_ヘッダ"},
    {"no": "22", "slug": "12-3-22-ODDS_UMATAN", "name": "オッズ_馬単"},
    {"no": "23", "slug": "12-3-23-ODDS_SANREN_HEAD", "name": "オッズ_3連複_ヘッダ"},
    {"no": "24", "slug": "12-3-24-ODDS_SANREN", "name": "オッズ_3連複"},
    {"no": "25", "slug": "12-3-25-ODDS_SANRENTAN_HEAD", "name": "オッズ_3連単_ヘッダ"},
    {"no": "26", "slug": "12-3-26-ODDS_SANRENTAN", "name": "オッズ_3連単"},
    {"no": "27", "slug": "12-3-27-UMA", "name": "競走馬マスタ"},
    {"no": "28", "slug": "12-3-28-KISYU", "name": "騎手マスタ"},
    {"no": "29", "slug": "12-3-29-KISYU_SEISEKI", "name": "騎手マスタ_成績"},
    {"no": "30", "slug": "12-3-30-CHOKYO", "name": "調教師マスタ"},
    {"no": "31", "slug": "12-3-31-CHOKYO_SEISEKI", "name": "調教師マスタ_成績"},
    {"no": "32", "slug": "12-3-32-SEISAN", "name": "生産者マスタ"},
    {"no": "33", "slug": "12-3-33-BANUSI", "name": "馬主マスタ"},
    {"no": "34", "slug": "12-3-34-HANSYOKU", "name": "繁殖馬マスタ"},
    {"no": "35", "slug": "12-3-35-SANKU", "name": "産駒マスタ"},
    {"no": "36", "slug": "12-3-36-RECORD", "name": "レコードマスタ"},
    {"no": "37", "slug": "12-3-37-HANRO", "name": "坂路調教"},
    {"no": "38", "slug": "12-3-38-BATAIJYU", "name": "馬体重"},
    {"no": "39", "slug": "12-3-39-TENKO_BABA", "name": "天候馬場状態"},
    {"no": "40", "slug": "12-3-40-TORIKESI_JYOGAI", "name": "出走取消・競走除外"},
    {"no": "41", "slug": "12-3-41-KISYU_CHANGE", "name": "騎手変更"},
    {"no": "42", "slug": "12-3-42-HASSOU_JIKOKU_CHANGE", "name": "発走時刻変更"},
    {"no": "43", "slug": "12-3-43-COURSE_CHANGE", "name": "コース変更"},
    {"no": "44", "slug": "12-3-44-MINING", "name": "データマイニング予想"},
    {"no": "45", "slug": "12-3-45-SCHEDULE", "name": "開催スケジュール"},
    {"no": "46", "slug": "12-3-46-JODDS_TANPUKUWAKU_HEAD", "name": "時系列オッズ_単複枠_ヘッダ"},
    {"no": "47", "slug": "12-3-47-JODDS_TANPUKU", "name": "時系列オッズ_単複"},
    {"no": "48", "slug": "12-3-48-JODDS_WAKU", "name": "時系列オッズ_枠連"},
    {"no": "49", "slug": "12-3-49-JODDS_UMAREN_HEAD", "name": "時系列オッズ_馬連_ヘッダ"},
    {"no": "50", "slug": "12-3-50-JODDS_UMAREN", "name": "時系列オッズ_馬連"},
    {"no": "51", "slug": "12-3-51-SALE", "name": "競走馬市場取引価格"},
    {"no": "52", "slug": "12-3-52-BAMEIORIGIN", "name": "馬名の意味由来"},
    {"no": "53", "slug": "12-3-53-KEITO", "name": "系統情報"},
    {"no": "54", "slug": "12-3-54-COURSE", "name": "コース情報"},
    {"no": "55", "slug": "12-3-55-TAISENGATA_MINING", "name": "対戦型データマイニング予想"},
    {"no": "56", "slug": "12-3-56-JYUSYOSIKI_HEAD", "name": "重勝式_ヘッダ"},
    {"no": "57", "slug": "12-3-57-JYUSYOSIKI", "name": "重勝式"},
    {"no": "58", "slug": "12-3-58-JOGAIBA", "name": "競走馬除外情報"},
    {"no": "59", "slug": "12-3-59-WOOD_CHIP", "name": "ウッドチップ調教"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EveryDB2マニュアル 12章スクレイピング")
    parser.add_argument("--pages", nargs="+", help="特定ページのみ (例: 03 04 05)")
    parser.add_argument("--codes-only", action="store_true", help="コード表のみスクレイプ")
    parser.add_argument("--dry-run", action="store_true", help="アクセスのみ、ファイル出力なし")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="リクエスト間隔(秒, デフォルト1)"
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be >= 0")
    return args


def create_browser() -> tuple[Any, Any]:
    """Playwrightブラウザを起動して (playwright_instance, browser) を返す。"""
    from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    return pw, browser


def fetch_page(page: Any, url: str, retries: int = 3) -> bool:
    """ページにアクセス。成功=True、リトライ上限=False。"""
    for attempt in range(retries):
        try:
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            return True
        except Exception as e:
            logger.warning("  リトライ %d/%d: %s", attempt + 1, retries, e)
            time.sleep(2)
    return False


def discover_page_links(page: Any) -> list[dict[str, str]]:
    """12-3-00-FormatList.html からページ一覧を動的に取得。失敗時はPAGESにフォールバック。"""
    url = f"{BASE_URL}/12-3-00-FormatList.html"
    logger.info("ページ一覧取得: %s", url)

    if not fetch_page(page, url):
        logger.warning("FormatList ページにアクセス失敗。ハードコードリストを使用。")
        return PAGES

    links = page.evaluate("""() => {
        const anchors = document.querySelectorAll('a[href*="12-3-"]');
        const result = [];
        for (const a of anchors) {
            const href = a.getAttribute('href') || '';
            const match = href.match(/12-3-(\\d{2})-([^.]+)\\.html/);
            if (!match) continue;
            const no = match[1];
            const name = a.textContent.trim()
                .replace(/^12-3-\\d{2}\\.\\s*/, '');
            const slug = '12-3-' + no + '-' + match[2];
            result.push({no, slug, name});
        }
        return result;
    }""")

    if not links or len(links) < 10:
        logger.warning(
            "FormatList から十分なリンクを取得できません(%d件)。ハードコードリストを使用。",
            len(links) if links else 0,
        )
        return PAGES

    logger.info("FormatList から %d ページを検出", len(links))
    return links


def main() -> None:
    args = parse_args()

    pw, browser = create_browser()
    page = browser.new_page()

    try:
        if args.codes_only:
            logger.info("コード表のみスクレイピング")
            scrape_code_tables(page, args)
        elif args.pages:
            logger.info("指定ページ: %s", args.pages)
            filtered = [p for p in PAGES if p["no"] in args.pages]
            scrape_data_formats(page, filtered, args)
            generate_index(args)
        else:
            logger.info("全ページスクレイピング開始")
            discovered = discover_page_links(page)
            scrape_data_formats(page, discovered, args)
            scrape_code_tables(page, args)
            generate_index(args)
    finally:
        browser.close()
        pw.stop()

    logger.info("完了")


# --- 以下、Task 3-7で実装する関数のプレースホルダ ---

def scrape_data_formats(
    page: Any, pages: list[dict[str, str]], args: argparse.Namespace
) -> None:
    """データフォーマットページ群をスクレイピング。"""
    raise NotImplementedError("Task 3-4で実装")


def scrape_code_tables(page: Any, args: argparse.Namespace) -> None:
    """コード表ページをスクレイピング。"""
    raise NotImplementedError("Task 5-6で実装")


def generate_index(args: argparse.Namespace) -> None:
    """INDEX.md を生成。"""
    raise NotImplementedError("Task 7で実装")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script loads without errors**

```bash
python scripts/scrape_everydb2_manual.py --help
```

Expected: usage message showing --pages, --codes-only, --dry-run, --delay options

- [ ] **Step 3: Verify ruff passes**

```bash
ruff check scripts/scrape_everydb2_manual.py
```

Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add scripts/scrape_everydb2_manual.py
git commit -m "feat(scrape): add script skeleton with CLI, browser lifecycle, page discovery"
```

---

### Task 3: Implement data format page scraping + markdown generation

**Files:**
- Modify: `scripts/scrape_everydb2_manual.py` (add `_parse_data_format_page`, `_replace_code_refs`, `_write_data_format_md`, `_validate_fields`, replace `scrape_data_formats`)

Each data format page has a single `<table>` with 8 columns:
`No, キー, 項目, フィールド名, 型, サイズ, 初期値, 説明文`

The `<h3>` element contains the page title (e.g. `12-3-03. レース詳細（RACE）`).

- [ ] **Step 1: Add parsing, validation, and markdown functions**

Add these functions before `scrape_data_formats`:

```python
def _validate_fields(rows: list[dict[str, str]], page_slug: str) -> None:
    """フィールド名の重複をチェック。"""
    field_names = [r.get("field_name", "") for r in rows if r.get("field_name")]
    seen: set[str] = set()
    for name in field_names:
        if name in seen:
            logger.warning("  %s: フィールド名重複 '%s'", page_slug, name)
        seen.add(name)


def _parse_data_format_page(page: Any) -> dict[str, Any] | None:
    """データフォーマットページをパースして構造化データを返す。

    Returns:
        dict with keys: title, record_spec, rows, code_refs
        Returns None if parsing fails.
    """
    title_el = page.query_selector("h3")
    if not title_el:
        logger.error("  <h3>タイトルが見つかりません")
        return None
    title = title_el.inner_text().strip()

    rows_data = page.evaluate("""() => {
        const table = document.querySelector('table');
        if (!table) return null;
        const rows = table.querySelectorAll('tr');
        const result = [];
        for (let i = 1; i < rows.length; i++) {
            const cells = rows[i].querySelectorAll('th, td');
            if (cells.length < 7) continue;
            result.push({
                no: cells[0]?.textContent?.trim() || '',
                key: cells[1]?.textContent?.trim() || '',
                item: cells[2]?.textContent?.trim() || '',
                field_name: cells[3]?.textContent?.trim() || '',
                type: cells[4]?.textContent?.trim() || '',
                size: cells[5]?.textContent?.trim() || '',
                default_val: cells[6]?.textContent?.trim() || '',
                description: cells[7]?.textContent?.trim() || '',
            });
        }
        return result;
    }""")

    if not rows_data:
        logger.error("  テーブルデータが見つかりません: %s", title)
        return None

    # RecordSpec抽出 (行1の説明文から)
    record_spec = ""
    if rows_data and "をセット" in rows_data[0].get("description", ""):
        desc = rows_data[0]["description"]
        match = re.match(r"(\w+)\s*をセット", desc)
        if match:
            record_spec = match.group(1)

    # コード表参照を検出
    code_refs: list[tuple[str, str, str]] = []
    for row in rows_data:
        field_name = row.get("field_name", "")
        desc = row.get("description", "")
        for m in CODE_REF_RE.finditer(desc):
            code_refs.append((field_name, m.group(1), m.group(2)))

    return {
        "title": title,
        "record_spec": record_spec,
        "rows": rows_data,
        "code_refs": code_refs,
    }


def _replace_code_refs(
    text: str, resolved_codes: set[str] | None = None
) -> str:
    """説明文中のコード表参照をマークダウンリンクに変換。

    Args:
        resolved_codes: スクレイピング済みのコードID集合。
            存在しない場合は⚠️マーク付きリンクにする。
    """
    def _repl(m: re.Match[str]) -> str:
        code_id = m.group(1)
        code_name = m.group(2)
        if resolved_codes is not None and code_id not in resolved_codes:
            return (
                f"[コード表{code_id} {code_name}](codes/C{code_id}.md)"
                " ⚠️リンク先未確認"
            )
        return f"[コード表{code_id} {code_name}](codes/C{code_id}.md)"
    return CODE_REF_RE.sub(_repl, text)


def _write_data_format_md(
    pg: dict[str, str],
    data: dict[str, Any],
    resolved_codes: set[str] | None = None,
) -> None:
    """データフォーマット1ページ分のマークダウンを書き出す。"""
    lines: list[str] = []
    lines.append(f"# {data['title']}")
    lines.append("")
    if data["record_spec"]:
        lines.append(f"**RecordSpec:** {data['record_spec']}")
        lines.append("")
    lines.append(
        "| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明文 |"
    )
    lines.append("|---:|:---:|---|---|---|---:|---|---|")

    for row in data["rows"]:
        no = row["no"]
        key = "○" if row["key"] else ""
        item = row["item"]
        field = row["field_name"]
        typ = row["type"]
        size = row["size"]
        default = row["default_val"] if row["default_val"] else "-"
        desc = _replace_code_refs(
            row["description"].replace("\n", " "), resolved_codes
        )
        lines.append(
            f"| {no} | {key} | {item} | {field} | {typ} | {size} "
            f"| {default} | {desc} |"
        )

    if data["code_refs"]:
        lines.append("")
        lines.append("## コード参照")
        for field_name, code_id, code_name in data["code_refs"]:
            suffix = ""
            if resolved_codes is not None and code_id not in resolved_codes:
                suffix = " ⚠️リンク先未確認"
            lines.append(
                f"- `{field_name}` → "
                f"[コード表{code_id} {code_name}](codes/C{code_id}.md){suffix}"
            )

    lines.append("")

    out_path = OUTPUT_DIR / f"{pg['slug']}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  出力: %s", out_path.name)
```

- [ ] **Step 2: Replace `scrape_data_formats` placeholder with implementation**

```python
def scrape_data_formats(
    page: Any, pages: list[dict[str, str]], args: argparse.Namespace
) -> None:
    """データフォーマットページ群をスクレイピング。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    # 既存メタデータの読み込み (行数差分チェック用)
    prev_meta_path = OUTPUT_DIR / ".prev_scrape_meta.json"
    prev_field_counts: dict[str, int] = {}
    if prev_meta_path.exists() and not args.dry_run:
        with open(prev_meta_path, encoding="utf-8") as f:
            prev_data = json.load(f)
            prev_field_counts = {
                r["slug"]: r["field_count"] for r in prev_data if "field_count" in r
            }

    for i, pg in enumerate(pages):
        url = f"{BASE_URL}/{pg['slug']}.html"
        logger.info("[%d/%d] %s (%s)", i + 1, len(pages), pg["name"], pg["slug"])

        if not fetch_page(page, url):
            errors.append(f"アクセス失敗: {pg['slug']}")
            continue

        data = _parse_data_format_page(page)
        if data is None:
            errors.append(f"パース失敗: {pg['slug']}")
            continue

        field_count = len(data["rows"])

        # バリデーション: フィールド名重複
        _validate_fields(data["rows"], pg["slug"])

        # バリデーション: 行数差分
        prev_fc = prev_field_counts.get(pg["slug"])
        if prev_fc is not None and field_count != prev_fc:
            logger.warning(
                "  %s: フィールド数変化 %d→%d", pg["slug"], prev_fc, field_count
            )

        logger.info(
            "  RecordSpec=%s, フィールド数=%d, コード参照=%d",
            data["record_spec"],
            field_count,
            len(data["code_refs"]),
        )

        result_entry = {
            "no": pg["no"],
            "name": pg["name"],
            "slug": pg["slug"],
            "record_spec": data["record_spec"],
            "field_count": field_count,
        }
        results.append(result_entry)

        if not args.dry_run:
            _write_data_format_md(pg, data)

        if i < len(pages) - 1:
            time.sleep(args.delay)

    if errors:
        logger.warning("エラー (%d件):", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    logger.info("データフォーマット: %d/%d 成功", len(results), len(pages))

    if not args.dry_run:
        meta_path = OUTPUT_DIR / ".scrape_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # 前回メタデータを更新
        meta_path.rename(prev_meta_path) if not prev_meta_path.exists() else None
```

Wait, the rename logic is wrong. Let me fix: after writing the new meta, we should update prev for next run. Actually the simplest approach: always write the new `.scrape_meta.json` and let `generate_index` rename it to `.prev_scrape_meta.json` after reading.

Let me re-read and fix. The corrected `scrape_data_formats` tail:

```python
    if not args.dry_run:
        meta_path = OUTPUT_DIR / ".scrape_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
```

And in `generate_index` (Task 7), after reading `.scrape_meta.json`, rename it to `.prev_scrape_meta.json` instead of deleting it. This way the prev counts are available for the next run.

- [ ] **Step 3: Verify with dry-run on a single page**

```bash
python scripts/scrape_everydb2_manual.py --pages 03 --dry-run
```

Expected: logs showing `[1/1] レース詳細 (12-3-03-RACE)`, RecordSpec=RA, フィールド数=~110

- [ ] **Step 4: Commit**

```bash
git add scripts/scrape_everydb2_manual.py
git commit -m "feat(scrape): implement data format page parsing and markdown generation"
```

---

### Task 4: Implement code table scraping + markdown generation

**Files:**
- Modify: `scripts/scrape_everydb2_manual.py` (add `_parse_code_tables`, `_write_code_table_md`, `_write_codes_index`, replace `scrape_code_tables`)

The code table page (`12-3-99-CODE.html`) has 20 independent `<table>` elements. Each table:
- Row 0: Title `<th>` with colspan (e.g., `2001.競馬場コード`)
- Row 1: Meta headers (`バイト数`, `値`, `内容`)
- Row 2: Sub-headers (varies per table)
- Row 3+: Data rows

- [ ] **Step 1: Add code table functions and replace placeholder**

```python
def _parse_code_tables(page: Any) -> list[dict[str, Any]]:
    """コード表ページから全コードテーブルをパースする。"""
    tables_data = page.evaluate("""() => {
        const tables = document.querySelectorAll('table');
        const result = [];
        for (const table of tables) {
            const rows = table.querySelectorAll('tr');
            if (rows.length < 4) continue;
            const titleCell = rows[0].querySelector('th');
            if (!titleCell) continue;
            const title = titleCell.textContent.trim();
            const titleMatch = title.match(/^(\\d{4})\\.(.+)$/);
            if (!titleMatch) continue;
            const subHeaderCells = rows[2].querySelectorAll('th, td');
            const headers = Array.from(subHeaderCells)
                .map(c => c.textContent.trim());
            const dataRows = [];
            for (let i = 3; i < rows.length; i++) {
                const cells = rows[i].querySelectorAll('th, td');
                const cellTexts = Array.from(cells)
                    .map(c => c.textContent.trim());
                dataRows.push(cellTexts);
            }
            result.push({
                codeId: titleMatch[1],
                codeName: titleMatch[2],
                headers: headers,
                rows: dataRows
            });
        }
        return result;
    }""")
    return tables_data or []


def _write_code_table_md(table: dict[str, Any]) -> None:
    """コード表1つ分のマークダウンを書き出す。"""
    code_id = table["codeId"]
    code_name = table["codeName"]
    headers = table["headers"]
    rows = table["rows"]

    lines: list[str] = []
    lines.append(f"# C{code_id}. {code_name}")
    lines.append("")

    col_count = max((len(r) for r in rows), default=len(headers))
    col_count = max(col_count, len(headers))

    header_cells = list(headers)
    while len(header_cells) < col_count:
        header_cells.append("")
    lines.append("| " + " | ".join(header_cells[:col_count]) + " |")
    lines.append("|" + "|".join(["---"] * col_count) + "|")

    for row in rows:
        cells = list(row)
        while len(cells) < col_count:
            cells.append("")
        cells = [
            c.replace("\n", " ").replace("|", "｜") for c in cells[:col_count]
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    out_path = CODES_DIR / f"C{code_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  出力: C%s %s (%d行)", code_id, code_name, len(rows))


def _write_codes_index(tables: list[dict[str, Any]]) -> None:
    """コード表のINDEX.mdを生成。"""
    lines: list[str] = []
    lines.append("# EveryDB2 コード表一覧")
    lines.append("")
    lines.append("12章データフォーマットのコード表。")
    lines.append("")
    lines.append("| コード | 名称 | 行数 |")
    lines.append("|:---|---|---:|")

    for t in sorted(tables, key=lambda x: x["codeId"]):
        cid = t["codeId"]
        cname = t["codeName"]
        nrows = len(t["rows"])
        lines.append(f"| [C{cid}](C{cid}.md) | {cname} | {nrows} |")

    lines.append("")
    (CODES_DIR / "INDEX.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("  出力: codes/INDEX.md")


def scrape_code_tables(page: Any, args: argparse.Namespace) -> None:
    """コード表ページをスクレイピング。"""
    CODES_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/12-3-99-CODE.html"
    logger.info("コード表スクレイピング: %s", url)

    if not fetch_page(page, url):
        logger.error("コード表ページにアクセスできません")
        return

    tables = _parse_code_tables(page)
    logger.info("  %d 個のコード表を検出", len(tables))

    if args.dry_run:
        for t in tables:
            logger.info(
                "  C%s %s (%d行)", t["codeId"], t["codeName"], len(t["rows"])
            )
        return

    for t in tables:
        _write_code_table_md(t)
    _write_codes_index(tables)
    logger.info("コード表: %d 個出力完了", len(tables))
```

- [ ] **Step 2: Verify with dry-run**

```bash
python scripts/scrape_everydb2_manual.py --codes-only --dry-run
```

Expected: list of ~20 code tables with IDs, names, row counts

- [ ] **Step 3: Commit**

```bash
git add scripts/scrape_everydb2_manual.py
git commit -m "feat(scrape): implement code table parsing and markdown generation"
```

---

### Task 5: Implement INDEX.md generation

**Files:**
- Modify: `scripts/scrape_everydb2_manual.py` (replace `generate_index` placeholder)

- [ ] **Step 1: Replace `generate_index` with implementation**

After reading `.scrape_meta.json`, rename it to `.prev_scrape_meta.json` for cross-run validation.

```python
def generate_index(args: argparse.Namespace) -> None:
    """INDEX.md を生成。"""
    if args.dry_run or args.codes_only:
        return

    meta_path = OUTPUT_DIR / ".scrape_meta.json"
    if not meta_path.exists():
        logger.warning(".scrape_meta.json が見つかりません。INDEX.md をスキップ")
        return

    with open(meta_path, encoding="utf-8") as f:
        results = json.load(f)

    lines: list[str] = []
    lines.append("# EveryDB2 データフォーマット一覧")
    lines.append("")
    lines.append("12章データフォーマットの全テーブル定義。")
    lines.append("")
    lines.append("| No | ページ | RecordSpec | フィールド数 |")
    lines.append("|---:|---|:---:|---:|")

    for r in results:
        no = r["no"]
        name = r["name"]
        slug = r["slug"]
        rs = r.get("record_spec", "-") or "-"
        fc = r.get("field_count", "-")
        lines.append(f"| {no} | [{name}]({slug}.md) | {rs} | {fc} |")

    lines.append("")
    (OUTPUT_DIR / "INDEX.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    logger.info("INDEX.md を生成 (%d エントリ)", len(results))

    # 次回実行時の差分チェック用にリネーム
    prev_path = OUTPUT_DIR / ".prev_scrape_meta.json"
    if prev_path.exists():
        prev_path.unlink()
    meta_path.rename(prev_path)
```

- [ ] **Step 2: Test with a few pages**

```bash
python scripts/scrape_everydb2_manual.py --pages 03 04 05
cat docs/everydb2/INDEX.md
```

Expected: INDEX.md with 3 entries (RACE, UMA_RACE, HARAI)

- [ ] **Step 3: Clean up test output and commit**

```bash
rm -rf docs/everydb2/
git add scripts/scrape_everydb2_manual.py
git commit -m "feat(scrape): implement INDEX.md generation with cross-run validation"
```

---

### Task 6: Full integration run

**Files:**
- Run: `scripts/scrape_everydb2_manual.py` (full scrape)

- [ ] **Step 1: Run full scrape**

```bash
python scripts/scrape_everydb2_manual.py --delay 1
```

Expected: ~60 pages scraped, takes ~2-3 minutes. Logs show progress per page.

- [ ] **Step 2: Verify output completeness**

```bash
ls docs/everydb2/*.md | wc -l        # ~60 files (59 data + INDEX)
ls docs/everydb2/codes/*.md | wc -l   # ~21 files (20 codes + INDEX)
```

- [ ] **Step 3: Verify sample files**

```bash
head -30 docs/everydb2/12-3-03-RACE.md
head -20 docs/everydb2/codes/C2003.md
cat docs/everydb2/INDEX.md
```

Expected:
- RACE.md: title, RecordSpec: RA, ~110 field rows, code references section
- C2003.md: grade codes (G1, G2, G3, etc.)
- INDEX.md: all 59 entries

- [ ] **Step 4: Verify lint**

```bash
ruff check scripts/scrape_everydb2_manual.py
```

Expected: No errors

- [ ] **Step 5: Commit output**

```bash
git add docs/everydb2/ scripts/scrape_everydb2_manual.py
git commit -m "docs: EveryDB2マニュアル12章データフォーマットをスクレイピング

全59データフォーマット + 20コード表をmarkdownに変換。
scripts/scrape_everydb2_manual.py で再実行可能。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | What | Key changes |
|------|------|-------------|
| 1 | Add playwright dependency | `pyproject.toml` |
| 2 | Script skeleton + CLI + discovery | Full script skeleton with `discover_page_links()` |
| 3 | Data format parsing + markdown | `_parse_data_format_page()`, `_validate_fields()`, `_replace_code_refs()`, `_write_data_format_md()`, `scrape_data_formats()` |
| 4 | Code table parsing + markdown | `_parse_code_tables()`, `_write_code_table_md()`, `_write_codes_index()`, `scrape_code_tables()` |
| 5 | INDEX.md generation | `generate_index()` with `.prev_scrape_meta.json` for delta detection |
| 6 | Full integration run | Run + verify + commit output |
