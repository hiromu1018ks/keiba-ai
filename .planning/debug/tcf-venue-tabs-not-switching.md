---
slug: tcf-venue-tabs-not-switching
created: 2026-06-06T12:00:00Z
resolved: 2026-06-06T12:30:00Z
status: resolved
severity: critical
component: src/ingestion/track_condition_fetcher.py
---

# Bug: TrackConditionFetcher returns identical data for all venues

## Resolution

**Commit:** 318c3d2
**Fixed:** 2026-06-06

### Root Cause
JRA 馬場情報ページは同一ページ内 JavaScript タブではなく、各開催場が別 HTML ページ
(index.html, index2.html) で構成されていた。セレクタ [data-venue] / .tab-item が DOM に存在せず、
全10場で同一のベースページ HTML を返していた。また wait_until=domcontentloaded で
JS レンダリング完了前にパースしていたため値が空になっていた。

### Fix Applied
- fetch_all_venues: div.nav.tab > div > a からリンクを発見し各ページにナビゲート
- _discover_venue_links: DOM からアクティブな開催場リンクを抽出
- _venue_name_to_code: リンクテキストから jyocd を逆引き
- wait_until=networkidle + wait_for_selector(#turf_line .gm) でレンダリング完了を確実に待機

### Verification
- 43/43 tests pass
- Live: 東京(05) cushion=9.9 dirt=7.7/8.9 turf=16.2/15.7
- Live: 阪神(09) cushion=9.9 dirt=5.8/6.6 turf=11.4/10.4
- 2場で異なる値が正しく取得できることを確認
