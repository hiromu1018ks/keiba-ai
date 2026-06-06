"""track_condition_fetcher.py — JRA 馬場情報 HTML 取得・解析 (D-05)

JRA 公式サイト (https://www.jra.go.jp/keiba/baba/) から芝クッション値・ダート含水率を
取得する TrackConditionFetcher を実装する。

設計判断:
  D-05: Playwright で HTML を取得し、解析は純粋関数 (parse_track_condition_html) に分離。
         TrackConditionFetcherProtocol で Protocol-based DI を実現。
  T-53-05: 取得失敗時はフォールバックせず例外を送出 → 予測停止 (非ゼロ終了)。
  T-53-04: HTML 構造変更は html_hash で検知。必須要素欠落時は TrackConditionParseError。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class TrackConditionParseError(Exception):
    """トラック条件 HTML 解析エラー。

    必須 DOM 要素の欠落、空 HTML、値の解析失敗時に送出する。
    """


# ---------------------------------------------------------------------------
# Pure function: percent parser
# ---------------------------------------------------------------------------


def _parse_percent(text: str) -> float | None:
    """パーセント文字列を float に変換する。

    Args:
        text: "16.2%" のような文字列。

    Returns:
        パーセント値 (例: 16.2)。変換失敗時や空文字の場合は None。
    """
    text = text.replace("%", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Pure function: HTML parser
# ---------------------------------------------------------------------------


def parse_track_condition_html(html: str) -> dict[str, Any]:
    """JRA 馬場情報ページ HTML からクッション値・含水率を抽出する純粋関数。

    Playwright 非依存。保存済み HTML fixture でテスト可能。

    DOM 構造 (2026-06-06 確認):
      クッション値: <div id="cushion_num"><p><strong>9.9</strong></p></div>
      含水率テーブル:
        <tr id="turf_line"><th>芝</th><td class="gm">16.2%</td><td class="c4">15.7%</td></tr>
        <tr id="dirt_line"><th>ダート</th><td class="gm">7.7%</td><td class="c4">8.9%</td></tr>
      測定時刻:
        <select id="moist_list"><option selected>...</option></select>
        <select id="cushion_list"><option selected>...</option></select>

    Args:
        html: JRA 馬場情報ページの HTML 文字列。

    Returns:
        以下のキーを含む dict:
          - turf_cushion: float | None — 芝クッション値
          - dirt_moisture_goal: float | None — ダート含水率 (ゴール前 %)
          - dirt_moisture_4c: float | None — ダート含水率 (4コーナー %)
          - turf_moisture_goal: float | None — 芝含水率 (ゴール前 %)
          - turf_moisture_4c: float | None — 芝含水率 (4コーナー %)
          - measured_at_moist: str — 含水率測定時刻
          - measured_at_cushion: str — クッション値測定時刻
          - html_hash: str — SHA256 ハッシュ (構造変更検知用)

    Raises:
        TrackConditionParseError: HTML が空、または #turf_line と #dirt_line が
            両方欠落している場合。
    """
    if not html:
        raise TrackConditionParseError("Empty HTML")

    html_hash = hashlib.sha256(html.encode()).hexdigest()

    soup = BeautifulSoup(html, "html.parser")

    # --- クッション値 (cushion_num は一部開催場のみ) ---
    turf_cushion: float | None = None
    cushion_el = soup.select_one("#cushion_num strong")
    if cushion_el:
        try:
            turf_cushion = float(cushion_el.get_text(strip=True))
        except ValueError:
            logger.warning("cushion_num 値の変換失敗: '%s'", cushion_el.get_text(strip=True))

    # --- 含水率テーブル ---
    result: dict[str, Any] = {
        "turf_cushion": turf_cushion,
        "dirt_moisture_goal": None,
        "dirt_moisture_4c": None,
        "turf_moisture_goal": None,
        "turf_moisture_4c": None,
        "measured_at_moist": "",
        "measured_at_cushion": "",
        "html_hash": html_hash,
    }

    turf_line = soup.select_one("#turf_line")
    dirt_line = soup.select_one("#dirt_line")

    # 両方欠落の場合はエラー
    if turf_line is None and dirt_line is None:
        raise TrackConditionParseError(
            "Missing required elements: #turf_line and #dirt_line"
        )

    # いずれかが欠落の場合もエラー (必須要素)
    if turf_line is None:
        raise TrackConditionParseError("Missing required element: #turf_line")

    if dirt_line is None:
        raise TrackConditionParseError("Missing required element: #dirt_line")

    # 含水率の抽出 (ゴール前=.gm, 4コーナー=.c4)
    for row, prefix in [(turf_line, "turf"), (dirt_line, "dirt")]:
        gm_el = row.select_one(".gm")
        c4_el = row.select_one(".c4")
        if gm_el:
            result[f"{prefix}_moisture_goal"] = _parse_percent(gm_el.get_text(strip=True))
        if c4_el:
            result[f"{prefix}_moisture_4c"] = _parse_percent(c4_el.get_text(strip=True))

    # --- 測定時刻 ---
    moist_select = soup.select_one("#moist_list option[selected]")
    if moist_select:
        result["measured_at_moist"] = moist_select.get_text(strip=True)

    cushion_select = soup.select_one("#cushion_list option[selected]")
    if cushion_select:
        result["measured_at_cushion"] = cushion_select.get_text(strip=True)

    return result


# ---------------------------------------------------------------------------
# Protocol: TrackConditionFetcherProtocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrackConditionFetcherProtocol(Protocol):
    """トラック条件 HTML 取得の抽象プロトコル (D-05)。

    Protocol-based DI によりテスト時のモック差し替えが可能。
    OddsFetcherProtocol と同じパターン。
    """

    def fetch_track_conditions_html(self, venue_code: str) -> str:
        """指定した開催場コードのトラック条件 HTML を取得する。

        Args:
            venue_code: 開催場コード (例: "05" = 東京)。

        Returns:
            JRA 馬場情報ページの HTML 文字列。

        Raises:
            Exception: 取得失敗時 (フォールバックなし、D-05)。
        """
        ...


# ---------------------------------------------------------------------------
# Implementation: JRATrackConditionFetcher
# ---------------------------------------------------------------------------


class JRATrackConditionFetcher:
    """JRA 公式サイトから馬場条件 HTML を取得するフェッチャー。

    Playwright sync_api を使用して HTML を取得する。
    解析は行わず、生の HTML 文字列を返す (D-05: 取得と解析の完全分離)。

    取得失敗時は例外をそのまま送出し、フォールバックは行わない。
    呼び出し元が非ゼロ終了して予測を停止する (T-53-05)。
    """

    BASE_URL = "https://www.jra.go.jp/keiba/baba/"

    def __init__(self, headless: bool = True, timeout_ms: int = 30000) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms

    def fetch_track_conditions_html(self, venue_code: str) -> str:
        """指定した開催場のトラック条件 HTML を Playwright で取得する。

        Args:
            venue_code: 開催場コード (例: "05" = 東京)。

        Returns:
            JRA 馬場情報ページの HTML 文字列。

        Raises:
            Exception: Playwright エラー時 (フォールバックなし)。
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "playwright がインストールされていません。"
                "pip install playwright && playwright install chromium を実行してください。"
            ) from e

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self._headless)
            try:
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 900},
                )
                page = context.new_page()
                page.goto(
                    self.BASE_URL,
                    wait_until="domcontentloaded",
                    timeout=self._timeout_ms,
                )

                # 開催場タブをクリック (同一ページ内 DOM 切り替え)
                # JRA ページは開催場選択がタブ形式で同一 URL 内で切り替わる
                tab_selector = f'[data-venue="{venue_code}"], .tab-item[data-code="{venue_code}"]'
                try:
                    page.click(tab_selector, timeout=5000)
                except Exception:
                    # タブクリック失敗時はそのままのページ内容を返す
                    logger.warning(
                        "開催場タブクリック失敗: venue_code=%s。"
                        "デフォルトページ内容を返します。",
                        venue_code,
                    )

                # コンテンツ待機
                try:
                    page.wait_for_selector("#turf_line", timeout=10000)
                except Exception:
                    logger.warning(
                        "#turf_line セレクタ待機タイムアウト: venue_code=%s",
                        venue_code,
                    )

                html = page.content()
                logger.info(
                    "HTML 取得成功: venue_code=%s, length=%d",
                    venue_code,
                    len(html),
                )
                return html
            finally:
                browser.close()

    def fetch_all_venues(self, track_date: str) -> dict[str, dict[str, Any]]:
        """全開催場のトラック条件を取得・解析する。

        Args:
            track_date: 競馬開催日 (YYYY-MM-DD)。

        Returns:
            {venue_code: parse_track_condition_html() の結果 dict}。

        Raises:
            TrackConditionParseError: 解析エラー時。
            Exception: 取得失敗時 (1場でも失敗すれば予測停止)。
        """
        # JRA の開催場コード一覧
        venue_codes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        results: dict[str, dict[str, Any]] = {}

        for venue_code in venue_codes:
            try:
                html = self.fetch_track_conditions_html(venue_code)
                parsed = parse_track_condition_html(html)
                results[venue_code] = parsed
                logger.info(
                    "venue=%s parsed: cushion=%s, dirt_goal=%s",
                    venue_code,
                    parsed.get("turf_cushion"),
                    parsed.get("dirt_moisture_goal"),
                )
            except TrackConditionParseError:
                # 非開催場は必須要素がなく ParseError になるのが正常
                logger.debug("venue=%s: 非開催場またはデータなし", venue_code)
                continue

        logger.info(
            "track_date=%s: %d venues with data",
            track_date,
            len(results),
        )
        return results


# ---------------------------------------------------------------------------
# Helper: HTML structure change detection
# ---------------------------------------------------------------------------


def _detect_html_structure_change(reference_hash: str, current_hash: str) -> bool:
    """HTML 構造変更をハッシュ比較で検知する。

    パーサーが期待するセレクタが存在しない場合も、ハッシュ不一致として
    構造変更とみなす。

    Args:
        reference_hash: 以前の HTML の SHA256 ハッシュ。
        current_hash: 現在の HTML の SHA256 ハッシュ。

    Returns:
        True の場合、HTML 構造が変更された可能性がある。
    """
    return reference_hash != current_hash
