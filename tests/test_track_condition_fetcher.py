"""test_track_condition_fetcher.py — TrackConditionFetcherProtocol + パーサーのユニットテスト

53-02 TDD: RED phase — parse_track_condition_html() / JRATrackConditionFetcher のテスト。
DOM 構造は 53-RESEARCH.md の確認済み JRA ページ構造に基づく。
"""

from __future__ import annotations

import hashlib
import re
from unittest.mock import MagicMock

import pytest

from ingestion.track_condition_fetcher import (
    JRATrackConditionFetcher,
    TrackConditionFetcherProtocol,
    TrackConditionParseError,
    _detect_html_structure_change,
    _parse_percent,
    parse_track_condition_html,
)

# ---------------------------------------------------------------------------
# HTML Fixtures (53-RESEARCH.md 確認済み DOM 構造)
# ---------------------------------------------------------------------------

VALID_HTML = """\
<html>
<body>
<div id="cushion_num"><p><strong>9.9</strong></p></div>
<table>
  <tr id="turf_line"><th>芝</th><td class="gm">16.2%</td><td class="c4">15.7%</td></tr>
  <tr id="dirt_line"><th>ダート</th><td class="gm">7.7%</td><td class="c4">8.9%</td></tr>
</table>
<select id="moist_list"><option selected>6月6日（土曜）5時00分</option></select>
<select id="cushion_list"><option selected>6月6日（土曜）4時30分</option></select>
</body>
</html>
"""

HTML_NO_CUSHION = """\
<html>
<body>
<table>
  <tr id="turf_line"><th>芝</th><td class="gm">16.2%</td><td class="c4">15.7%</td></tr>
  <tr id="dirt_line"><th>ダート</th><td class="gm">7.7%</td><td class="c4">8.9%</td></tr>
</table>
<select id="moist_list"><option selected>6月6日（土曜）5時00分</option></select>
</body>
</html>
"""

HTML_NO_TURF_LINE = """\
<html>
<body>
<div id="cushion_num"><p><strong>9.9</strong></p></div>
<table>
  <tr id="dirt_line"><th>ダート</th><td class="gm">7.7%</td><td class="c4">8.9%</td></tr>
</table>
</body>
</html>
"""

HTML_NO_DIRT_LINE = """\
<html>
<body>
<div id="cushion_num"><p><strong>9.9</strong></p></div>
<table>
  <tr id="turf_line"><th>芝</th><td class="gm">16.2%</td><td class="c4">15.7%</td></tr>
</table>
</body>
</html>
"""

HTML_NO_BOTH_LINES = """\
<html>
<body>
<div id="cushion_num"><p><strong>9.9</strong></p></div>
<table>
  <tr><th>その他</th><td>値なし</td></tr>
</table>
</body>
</html>
"""

HTML_NO_MEASURED_AT = """\
<html>
<body>
<table>
  <tr id="turf_line"><th>芝</th><td class="gm">16.2%</td><td class="c4">15.7%</td></tr>
  <tr id="dirt_line"><th>ダート</th><td class="gm">7.7%</td><td class="c4">8.9%</td></tr>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Test 1: 有効な HTML fixture から正しい値を抽出
# ---------------------------------------------------------------------------
class TestParseValidHtml:
    def test_turf_cushion(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["turf_cushion"] == 9.9

    def test_dirt_moisture_goal(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["dirt_moisture_goal"] == 7.7

    def test_dirt_moisture_4c(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["dirt_moisture_4c"] == 8.9

    def test_turf_moisture_goal(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["turf_moisture_goal"] == 16.2

    def test_turf_moisture_4c(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["turf_moisture_4c"] == 15.7


# ---------------------------------------------------------------------------
# Test 2: cushion_num 要素なしでも turf_cushion=None で成功
# ---------------------------------------------------------------------------
class TestParseCushionMissing:
    def test_turf_cushion_is_none(self) -> None:
        result = parse_track_condition_html(HTML_NO_CUSHION)
        assert result["turf_cushion"] is None

    def test_other_values_present(self) -> None:
        result = parse_track_condition_html(HTML_NO_CUSHION)
        assert result["dirt_moisture_goal"] == 7.7
        assert result["turf_moisture_goal"] == 16.2


# ---------------------------------------------------------------------------
# Test 3: #turf_line 欠落で TrackConditionParseError
# ---------------------------------------------------------------------------
class TestParseTurfLineMissing:
    def test_raises_parse_error(self) -> None:
        with pytest.raises(TrackConditionParseError):
            parse_track_condition_html(HTML_NO_TURF_LINE)


# ---------------------------------------------------------------------------
# Test 4: #dirt_line 欠落で TrackConditionParseError
# ---------------------------------------------------------------------------
class TestParseDirtLineMissing:
    def test_raises_parse_error(self) -> None:
        with pytest.raises(TrackConditionParseError):
            parse_track_condition_html(HTML_NO_DIRT_LINE)


# ---------------------------------------------------------------------------
# Test (plan step g): 両方欠落で TrackConditionParseError
# ---------------------------------------------------------------------------
class TestParseBothLinesMissing:
    def test_raises_parse_error(self) -> None:
        with pytest.raises(TrackConditionParseError):
            parse_track_condition_html(HTML_NO_BOTH_LINES)


# ---------------------------------------------------------------------------
# Test 5: _parse_percent ヘルパー
# ---------------------------------------------------------------------------
class TestParsePercent:
    def test_normal_percent(self) -> None:
        assert _parse_percent("16.2%") == 16.2

    def test_empty_string(self) -> None:
        assert _parse_percent("") is None

    def test_no_percent_sign(self) -> None:
        assert _parse_percent("7.7") == 7.7

    def test_invalid_text(self) -> None:
        assert _parse_percent("abc") is None


# ---------------------------------------------------------------------------
# Test 7: html_hash が SHA256 ハッシュ文字列
# ---------------------------------------------------------------------------
class TestHtmlHash:
    def test_html_hash_present(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert "html_hash" in result
        assert isinstance(result["html_hash"], str)

    def test_html_hash_is_sha256(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        expected = hashlib.sha256(VALID_HTML.encode()).hexdigest()
        assert result["html_hash"] == expected

    def test_html_hash_64_hex_chars(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert len(result["html_hash"]) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", result["html_hash"])


# ---------------------------------------------------------------------------
# Test 8: 測定時刻が #moist_list option[selected] から抽出される
# ---------------------------------------------------------------------------
class TestMeasuredAtExtraction:
    def test_measured_at_moist(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["measured_at_moist"] == "6月6日（土曜）5時00分"

    def test_measured_at_cushion(self) -> None:
        result = parse_track_condition_html(VALID_HTML)
        assert result["measured_at_cushion"] == "6月6日（土曜）4時30分"

    def test_measured_at_missing(self) -> None:
        result = parse_track_condition_html(HTML_NO_MEASURED_AT)
        assert result["measured_at_moist"] == ""
        assert result["measured_at_cushion"] == ""


# ---------------------------------------------------------------------------
# Test 9: Protocol conformance (isinstance チェック)
# ---------------------------------------------------------------------------
class TestProtocolConformance:
    def test_jra_fetcher_satisfies_protocol(self) -> None:
        fetcher = JRATrackConditionFetcher()
        assert isinstance(fetcher, TrackConditionFetcherProtocol)

    def test_mock_satisfies_protocol(self) -> None:
        mock_fetcher = MagicMock(spec=TrackConditionFetcherProtocol)
        assert isinstance(mock_fetcher, TrackConditionFetcherProtocol)


# ---------------------------------------------------------------------------
# Test: 空文字 HTML → TrackConditionParseError
# ---------------------------------------------------------------------------
class TestParseEmptyHtml:
    def test_empty_string_raises_error(self) -> None:
        with pytest.raises(TrackConditionParseError):
            parse_track_condition_html("")


# ---------------------------------------------------------------------------
# Test: _detect_html_structure_change
# ---------------------------------------------------------------------------
class TestDetectHtmlStructureChange:
    def test_same_hash_no_change(self) -> None:
        h = hashlib.sha256(b"test").hexdigest()
        assert _detect_html_structure_change(h, h) is False

    def test_different_hash_detected(self) -> None:
        h1 = hashlib.sha256(b"old").hexdigest()
        h2 = hashlib.sha256(b"new").hexdigest()
        assert _detect_html_structure_change(h1, h2) is True


# ---------------------------------------------------------------------------
# Test: _venue_name_to_code — リンクテキストから開催場コード逆引き
# ---------------------------------------------------------------------------
class TestVenueNameToCode:
    def test_tokyo_single_char_prefix(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回東京") == "05"

    def test_hanshin(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回阪神") == "09"

    def test_nakayama(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("2回中山") == "06"

    def test_chukyo(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回中京") == "07"

    def test_kyoto(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("3回京都") == "08"

    def test_sapporo(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回札幌") == "01"

    def test_hakodate(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回函館") == "02"

    def test_fukushima(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("2回福島") == "03"

    def test_niigata(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回新潟") == "04"

    def test_kokura(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("1回小倉") == "10"

    def test_unknown_venue_returns_none(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("不明な会場") is None

    def test_empty_string_returns_none(self) -> None:
        assert JRATrackConditionFetcher._venue_name_to_code("") is None


# ---------------------------------------------------------------------------
# Test: _discover_venue_links — 開催場リンク発見 (mock Playwright Page)
# ---------------------------------------------------------------------------
class TestDiscoverVenueLinks:
    @staticmethod
    def _make_mock_page(links_html: str) -> MagicMock:
        """指定したリンク HTML を持つモック Playwright Page を作成する。

        query_selector_all はセレクタに関わらず links_html をパースして
        <a> 要素のモックリストを返す。
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(links_html, "html.parser")
        anchor_tags = soup.find_all("a")

        mock_elements = []
        for tag in anchor_tags:
            el = MagicMock()
            el.get_attribute.return_value = tag.get("href")
            el.inner_text.return_value = tag.get_text()
            mock_elements.append(el)

        page = MagicMock()
        page.query_selector_all.return_value = mock_elements
        return page

    def test_discovers_two_venues(self) -> None:
        html = """
        <div class="nav tab">
          <div class="current"><a href="index.html">1回東京</a></div>
          <div><a href="index2.html">1回阪神</a></div>
        </div>
        """
        fetcher = JRATrackConditionFetcher()
        page = self._make_mock_page(html)
        venues = fetcher._discover_venue_links(page)

        assert len(venues) == 2
        assert venues[0] == ("05", "https://www.jra.go.jp/keiba/baba/index.html")
        assert venues[1] == ("09", "https://www.jra.go.jp/keiba/baba/index2.html")

    def test_no_venues_on_non_race_day(self) -> None:
        page = MagicMock()
        page.query_selector_all.return_value = []
        fetcher = JRATrackConditionFetcher()
        venues = fetcher._discover_venue_links(page)

        assert venues == []

    def test_absolute_url_preserved(self) -> None:
        html = '<a href="https://other.example.com/baba/">1回中山</a>'
        fetcher = JRATrackConditionFetcher()
        page = self._make_mock_page(html)
        venues = fetcher._discover_venue_links(page)

        assert len(venues) == 1
        assert venues[0] == ("06", "https://other.example.com/baba/")

    def test_skips_link_without_href(self) -> None:
        el = MagicMock()
        el.get_attribute.return_value = None
        el.inner_text.return_value = "1回東京"

        page = MagicMock()
        page.query_selector_all.return_value = [el]

        fetcher = JRATrackConditionFetcher()
        venues = fetcher._discover_venue_links(page)

        assert venues == []

    def test_skips_unrecognized_venue_name(self) -> None:
        html = '<a href="index3.html">謎の開催場</a>'
        fetcher = JRATrackConditionFetcher()
        page = self._make_mock_page(html)
        venues = fetcher._discover_venue_links(page)

        assert venues == []

    def test_three_venues(self) -> None:
        html = """
        <div class="nav tab">
          <div class="current"><a href="index.html">1回東京</a></div>
          <div><a href="index2.html">1回阪神</a></div>
          <div><a href="index3.html">1回中京</a></div>
        </div>
        """
        fetcher = JRATrackConditionFetcher()
        page = self._make_mock_page(html)
        venues = fetcher._discover_venue_links(page)

        assert len(venues) == 3
        codes = [v[0] for v in venues]
        assert codes == ["05", "09", "07"]
