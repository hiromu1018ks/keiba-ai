---
phase: 41-shadow-comparison-framework
reviewed: 2026-05-28T12:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - src/backtest/shadow_comparison.py
  - src/backtest/race_predictor.py
  - tests/test_shadow_comparison.py
  - src/backtest/shadow_report.py
  - src/backtest/templates/shadow_comparison_report.html
  - scripts/run_shadow_comparison.py
  - tests/test_shadow_report.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 41: Code Review Report

**Reviewed:** 2026-05-28T12:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Shadow Comparison Framework の7ファイル (framework, race_predictor flag injection, report generator, HTML template, CLI script, 2 test files) を standard depth でレビューした。

2つの Critical issue を発見した。(1) `_align_horse_level` が3-way以上のバリアント比較で列名衝突により merge が壊れる。(2) `metrics_by_selected_changed` がグループ化されたサブセットのメトリクスではなく全体のメトリクスを出力する。また、race_predictor.py の `import traceback` (line 238) は production コードに残るデバッグ用 import である。

## Critical Issues

### CR-01: _align_horse_level breaks with 3+ variants due to column name collision

**File:** `src/backtest/shadow_comparison.py:684-704`
**Issue:** N-way (3 variants) 比較の際、2番目以降の shadow variant を merge するループで、列名が `"shadow_p_win_final"`, `"shadow_selected"` 等とハードコードされている。2番目の variant (variant_names[2]) を処理する際、1回目のループで既に `"shadow_p_win_final"` 列が `merged` DataFrame に存在するため、pandas は `shadow_p_win_final_x`, `shadow_p_win_final_y` のように suffix を付与し、列名が破壊される。以降のメトリクス計算 (`compute_metrics`) は `"baseline_p_win_final"` / `"shadow_p_win_final"` 列を期待しているため、3番目以降の variant の確率品質メトリクスは計算不能または不正確になる。

`_align_race_level` (line 612) は `suffixes=(f"_{baseline_name}", f"_{vname}")` を使用してこの問題を回避しているが、`_align_horse_level` は variant name を使用していない。
**Fix:**
```python
# In _align_horse_level, for each vname in variant_names[1:]:
for vname in variant_names[1:]:
    shadow_df = dfs[vname]
    shadow_subset = shadow_df[key_cols].copy()
    for col in align_cols:
        if col in shadow_df.columns:
            shadow_subset[f"{vname}_{col}"] = shadow_df[col].values
        else:
            shadow_subset[f"{vname}_{col}"] = np.nan
    shadow_subset[f"{vname}_selected"] = (
        shadow_df["stake"].astype(float) > 0
        if "stake" in shadow_df.columns
        else False
    )
    merge_cols = key_cols + [c for c in shadow_subset.columns if c not in key_cols]
    merged = merged.merge(
        shadow_subset[merge_cols],
        on=key_cols,
        how="outer",
    )
```
`compute_metrics` 側でも variant-specific column name (`f"{vname}_p_win_final"`) を使用するよう修正が必要。

### CR-02: metrics_by_selected_changed outputs total metrics, not subset metrics

**File:** `src/backtest/shadow_comparison.py:142-152`
**Issue:** `save_results` 内で `metrics_by_selected_changed` を計算する際、`changed_df` (グループ化されたサブセット) を反復処理しているが、実際のメトリクスとして `cr.metrics` (全体メトリクス) をそのまま使用している。結果として `"changed"` グループと `"unchanged"` グループは常に同じ値を出力し、グループ間の差異分析が全く意味を持たない。このデータは JSON artifact に保存され、HTML レポートで表示されるため、利用者に誤った情報を提供する。
**Fix:**
```python
# In save_results(), lines 142-152:
if not cr.race_diff.empty and "selected_changed" in cr.race_diff.columns:
    changed_groups: dict[str, dict[str, Any]] = {}
    for changed_val, changed_df in cr.race_diff.groupby(
        "selected_changed", observed=True,
    ):
        label = "changed" if changed_val else "unchanged"
        # Filter bet_history by race_ids in this group, then compute metrics
        group_race_ids = set(changed_df["race_id"])
        _fw = ShadowComparisonFramework(variants=[])
        group_metrics: dict[str, dict[str, Any]] = {}
        for vname, vr in cr.variants.items():
            group_bh = [
                b for b in vr.backtest_result.bet_history
                if b.get("race_id") in group_race_ids
            ]
            group_metrics[vname] = _metrics_to_dict(
                _fw.compute_metrics(
                    pd.DataFrame(), pd.DataFrame(), vname, group_bh,
                )
            )
        changed_groups[label] = group_metrics
    fold_entry["metrics_by_selected_changed"] = changed_groups
```

## Warnings

### WR-01: ECE last bin excludes p=1.0 boundary

**File:** `src/backtest/shadow_comparison.py:899-900`
**Issue:** `_compute_ece` は `y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1]` でビンを判定している。最後のビン (0.9-1.0) は `y_pred < 1.0` が条件のため、`y_pred == 1.0` の予測値はどのビンにも含まれず、ECE 計算から完全に除外される。`valid_mask` (line 767) が `p_vals < 1` でフィルタしているため、p==1.0 は既に除外されているが、浮動小数点の丸めで 0.9999... が最後のビンに含まれる境界ケースでは、`<` と `<=` の違いがサンプルの帰属先ビンに影響する可能性がある。一般的に ECE の最後のビンは `<=` を使用する。
**Fix:** 最後のビン (i == n_bins - 1) では上限を `<=` に変更する:
```python
if i == n_bins - 1:
    mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
else:
    mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
```

### WR-02: Lazy import of traceback in production code

**File:** `src/backtest/race_predictor.py:238`
**Issue:** `predict()` メソッドの exception handler 内で `import traceback` を遅延インポートしている。これは production コードに残るデバッグ用パターンであり、`logger.error(..., exc_info=True)` で同等のスタックトレース出力が可能。`traceback` モジュールのインポート自体は遅延されるためパフォーマンス影響は最小だが、コードベース全体での一貫性を損なう。
**Fix:**
```python
# Replace lines 237-241:
except Exception as e:
    logger.error("Market prediction failed: %s", e, exc_info=True)
    return pd.DataFrame()
```

### WR-03: _shadow_flags injected via dynamic attribute on dataclass

**File:** `src/backtest/shadow_comparison.py:500-503`
**Issue:** `TrainedModelsV5` は `@dataclass` で定義されており、`_shadow_flags` は定義に存在しない属性である。Python dataclass は frozen=False であれば動的属性を許容するが、mypy strict mode (`disallow_untyped_defs = true`) では、`models._shadow_flags` への代入は `type: ignore[attr-defined]` で抑制されている。このアプローチは fragile であり、`TrainedModelsV5` が将来的に `frozen=True` になった場合や、`__slots__` が追加された場合に実行時エラーとなる。`RacePredictor.__init__` (line 133) の `getattr(models, "_shadow_flags", None)` も同様に fragile。
**Fix:** `_shadow_flags` を `TrainedModelsV5` dataclass に正式な optional field として追加する:
```python
@dataclass
class TrainedModelsV5:
    submodels: dict[str, SubmodelSet]
    quality_screener: RaceQualityScreener
    regime_detector: RegimeDetector
    train_period: tuple[str, str] = field(default=("2020-01-01", "2023-12-31"))
    _shadow_flags: dict[str, bool] | None = None
```

### WR-04: compute_metrics falls back to "baseline_" prefix for non-baseline variants

**File:** `src/backtest/shadow_comparison.py:752-755, 795-799`
**Issue:** `compute_metrics` で variant-specific column (`f"{variant_name}_p_win_final"`) が存在しない場合、`"baseline_p_win_final"` にフォールバックする (line 753-755)。`investment_score` も同様 (line 795-799)。このフォールバックは2バリアント比較では正常に動作するが、N-way 比較で `_align_horse_level` が修正された後 (CR-01)、variant 固有の列名が使用されるため、このフォールバックは不要かつ混乱を招く可能性がある。誤ったバリアントの確率値で Brier/logloss を計算するリスクがある。
**Fix:** フォールバックを削除し、variant-specific column が存在しない場合は Brier/logloss を計算しない (デフォルト 0.0 のまま) ようにする:
```python
p_col = f"{variant_name}_p_win_final"
if not aligned_horse.empty and p_col in aligned_horse.columns:
    # ... compute Brier, logloss, ECE
```

## Info

### IN-01: Repeated test helper construction in test_shadow_report.py

**File:** `tests/test_shadow_report.py:154-188, 193-228, 230-265` (etc.)
**Issue:** `test_generate_html_*` テストメソッド群で、`variant_configs` と `metrics_json` の構築が各メソッドで繰り返されている。pytest fixture またはクラスレベルの setup に抽出することで重複を減らせる。
**Fix:** `conftest.py` またはクラス fixtures に共通セットアップを抽出する。

### IN-02: HTML template uses Jinja2 autoescape but outputs raw dict for odds band

**File:** `src/backtest/templates/shadow_comparison_report.html:243`
**Issue:** Odds Band Breakdown セクションで `{{ band_metrics }}` を直接出力している。Jinja2 autoescape が有効なため XSS リスクはないが、dict の `__repr__` が出力され、ユーザ可読性が低い。構造化テーブルとして表示すべき。
**Fix:** band_metrics の各キー (roi, brier等) を個別の td 要素として出力するよう template を修正する。

### IN-03: CLI script uses sys.path manipulation instead of proper package installation

**File:** `scripts/run_shadow_comparison.py:33-35`
**Issue:** `ROOT` ベースの `sys.path.insert` でモジュール解決を行っている。CLAUDE.md では `pip install -e ".[dev]"` でインストール可能と記載されており、他のスクリプトも同パターンを使用しているため一貫性はあるが、一般的には避けるべきパターン。
**Fix:** パッケージインストール経由で実行できるエントリポイントを設定する (既存パターンとの整合性を優先する場合は修正不要)。

---

_Reviewed: 2026-05-28T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
