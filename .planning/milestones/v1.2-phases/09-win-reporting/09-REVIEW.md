---
phase: 09-win-reporting
reviewed: 2026-05-04T12:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/backtest/engine.py
  - src/backtest/report.py
  - src/backtest/templates/report.html
  - scripts/run_backtest.py
  - tests/test_backtest_report.py
findings:
  critical: 1
  warning: 6
  info: 4
  total: 11
status: issues_found
---

# Phase 09: Code Review Report

**Reviewed:** 2026-05-04T12:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

フェーズ9 (win reporting) の実装5ファイルを standard depth でレビューした。`src/backtest/report.py` に新規追加された `BacktestReportGenerator` と `MultiYearReportGenerator`、HTMLテンプレート、テストファイルが対象。

HTMLテンプレート (`report.html`) に1件の BLOCKER を発見した: `betting_target != "win"` の場合、ベット数・投資額・払戻額の基本KPIが表示されない。これは place/wide モードでレポートを生成した際に基本統計が欠落するロジックバグである。

また、`_parse_kumi` の曖昧な3文字パース、デッドコード（`monthly_returns` 初期化、`_build_race_features`/`_generate_bets` メソッド）、テストスキップ、`save_ai_diagnostics` のテスト不足などの WARNING/INFO を計10件発見した。

## Critical Issues

### CR-01: KPI cards が win モード以外で表示されない

**File:** `src/backtest/templates/report.html:110-123`
**Issue:** ベット数・投資額・払戻額の3つのKPIカードが `{% if betting_target == "win" %}` の条件ブロック内にある。`place` または `wide` モードでレポートを生成した場合、これらの基本統計がHTMLに表示されない。ROI・的最大DD・最終資金は表示されるが、ベット数・投資額・払戻額が欠落するため、ユーザーがplace/wideモードのレポートで基本情報を確認できない。
**Fix:**
```html
<!-- line 110: 条件ブロックを削除し、全モードで表示 -->
<div class="kpi-card">
    <div class="kpi-label">ベット数</div>
    <div class="kpi-value">{{ summary.total_bets }}</div>
</div>
<div class="kpi-card">
    <div class="kpi-label">投資額</div>
    <div class="kpi-value">&yen;{{ summary.total_stake|format_number }}</div>
</div>
<div class="kpi-card">
    <div class="kpi-label">払戻額</div>
    <div class="kpi-value">&yen;{{ summary.total_return|format_number }}</div>
</div>
```

## Warnings

### WR-01: `_parse_kumi` の3文字パースに曖昧さ — lo/hi ソートなし

**File:** `src/backtest/engine.py:165-174`
**Issue:** 3文字の `kumi` 文字列（例: "112"）を `(1, 12)` と解釈するが、実際のデータが `(11, 2)` を意味する可能性がある。また、返り値が常に `lo <= hi` の順序でない場合、呼び出し側 (line 1006) の `min/max` ソートと不一致になる。ただし、docstring の例 "513" = 馬5+馬13 が示す規則に従えば、小さい馬番が先に来るため `(1, 12)` は正しい可能性がある。`_parse_kumi` 内で `min/max` を取ることで安全性を担保すべき。
**Fix:**
```python
def _parse_kumi(kumi_str: str) -> tuple[int, int] | None:
    n = len(kumi_str)
    raw: tuple[int, int] | None = None
    if n == 4:
        raw = (int(kumi_str[:2]), int(kumi_str[2:]))
    elif n == 3:
        raw = (int(kumi_str[:1]), int(kumi_str[1:]))
    elif n == 2:
        raw = (int(kumi_str[:1]), int(kumi_str[1:]))
    if raw is None:
        return None
    return (min(raw), max(raw))
```

### WR-02: `monthly_returns` dict が初期化だけで更新されない (デッドコード)

**File:** `src/backtest/engine.py:454`
**Issue:** `monthly_returns: dict[str, float] = {}` として初期化されるが、`run()` メソッドのループ内で一度も値が追加されず、そのまま空の dict として `BacktestResult` に渡される。`BacktestResult.monthly_returns` フィールド自体が使用されていない可能性がある。
**Fix:** `monthly_returns` 変数の初期化と `BacktestResult` への渡しを削除するか、月次ROIの計算を追加する。

### WR-03: `_derive_fields` が `race_id` の長さを検証しない

**File:** `src/backtest/report.py:207`
**Issue:** `d["race_date"] = f"{bet['race_id'][:4]}-{bet['race_id'][4:6]}-{bet['race_id'][6:8]}"` において、`race_id` が8文字未満の場合に不正な日付文字列が生成される。`engine.py` では `if len(race_id) >= 8` のガードがあるが、`report.py` は任意の bet_history を受け取る公開メソッドのため、防御的チェックが必要。
**Fix:**
```python
rid = str(bet.get("race_id", ""))
if len(rid) >= 8:
    d["race_date"] = f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}"
else:
    d["race_date"] = ""
```

### WR-04: スキップされたテストの理由が古い

**File:** `tests/test_backtest_report.py:445`
**Issue:** `@pytest.mark.skip(reason="run_backtest.py still references deleted DataRepository")` とあるが、実際の `run_backtest.py` は `DataRepository` を参照していない（`ParquetStore` のみ使用）。スキップ理由が古く、テストが実行可能になっている可能性がある。このテストが実行されないまま放置されていることで回帰を検出できない。
**Fix:** スキップデコレータを削除し、テストが実行できるか確認する。mock のパス `db.repository.DataRepository` も不要なので削除する。

### WR-05: `save_ai_diagnostics` のテストが存在しない

**File:** `tests/test_backtest_report.py` (該当テストなし)
**Issue:** `BacktestReportGenerator.save_ai_diagnostics()` は Phase 9 で新規追加された主要機能（D-06, D-08）だが、ユニットテストが1つも存在しない。ハイライト自動特定（best/worst band）、月別トレンド検出、regime breakdown、オッズ倍率帯の集計など、複雑なロジックがテストされていない。
**Fix:** `TestSaveAiDiagnostics` クラスを追加し、正常系（win モードでのフル出力）、空データ、place モード（None 返却）、トレンド判定ロジックのテストを追加する。

### WR-06: `_build_race_features` と `_generate_bets` がデッドコード

**File:** `src/backtest/engine.py:913-998`
**Issue:** これら2つのメソッドは「互換性のため残している」とコメントにあるが、`BacktestEngine.run()` からは呼ばれておらず、外部からも参照されていない。約85行のデッドコードがメンテナンス負荷となっている。
**Fix:** デッドコードを削除する。テストで参照されている場合は、テストを `RacePredictor` 側に移行する。

## Info

### IN-01: ハードコードされた `before_roi = 0.638` 比較値

**File:** `scripts/run_backtest.py:258`
**Issue:** 改善前ROI比較値 `0.638` がハードコードされている。設定ファイルやコマンドライン引数で指定できず、将来の改善サイクルで値が陳腐化する。
**Fix:** `config/settings.yaml` または `--before-roi` 引数で外部化する。

### IN-02: 外部CDNスクリプトにSRI (Subresource Integrity) なし

**File:** `src/backtest/templates/report.html:7-10`
**Issue:** Chart.js、jQuery、DataTables をCDNから読み込んでいるが、`integrity` 属性と `crossorigin` 属性が設定されていない。中間者攻撃のリスクがあるが、ローカル専用レポートのため実用的リスクは低い。
**Fix:** 各CDNスクリプトに `integrity` と `crossorigin="anonymous"` を追加する。

### IN-03: `BacktestReportGenerator.generate()` のデフォルト `betting_target="place"`

**File:** `src/backtest/report.py:30`
**Issue:** `generate()` のデフォルト `betting_target="place"` だが、プロジェクトのデフォルトは `win`（`run_backtest.py` line 87: `default="win"`）。呼び出し側が明示的に渡しているため現在は問題ないが、デフォルト値の不一致が将来のバグの原因になる可能性がある。
**Fix:** デフォルトを `"win"` に変更する。

### IN-04: `_compute_bankroll_series` が `bankroll_after` キー欠落時に KeyError

**File:** `src/backtest/report.py:452`
**Issue:** `b["bankroll_after"]` がキー存在チェックなしでアクセスされている。engine.py の bet_history には常に含まれるため実用上は問題ないが、防御的ではない。
**Fix:** `bal = b.get("bankroll_after", 0.0)` に変更する。

---

_Reviewed: 2026-05-04T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
