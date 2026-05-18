---
phase: 31-race-level-aggregation-features
reviewed: 2026-05-18T12:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - src/features/race_level_features.py
  - tests/test_race_level_features.py
  - src/models/stage1_ability_model.py
  - src/models/market_model.py
  - src/models/regime_detector.py
  - src/models/place_ability_model.py
  - src/models/race_quality_screener.py
  - src/models/wide_two_stage_model.py
  - src/models/two_stage_return_model.py
  - src/models/ev_correction_model.py
  - tests/test_post_race_leakage.py
  - src/features/feature_engine.py
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 31: Code Review Report

**Reviewed:** 2026-05-18T12:00:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Phase 31 の成果物 (6つの `rl_*` 特徴量モジュール + `implied_prob_hhi`/`odds_skewness` の全12モデル FEATURE_COLS 昇格) を standard depth でレビューした。

POST_RACE データリーク防止は適切に実装されており、`rl_*` 特徴量は `tanodds` (発走前スナップショット) のみを使用している。`implied_prob_hhi` と `odds_skewness` は全12モデルの FEATURE_COLS に重複なく正しく追加されている。NaN フォールバックも堅牢。

3件の WARNING と 4件の INFO を検出した。Critical はなし。

## Warnings

### WR-01: `rl_*` 特徴量 (RLF-01~06) がどのモデルの FEATURE_COLS にも含まれていない

**File:** `src/features/race_level_features.py` / 関連全モデルファイル
**Issue:** `compute_race_level_features()` は6つの `rl_*` 特徴量 (`rl_log_odds_entropy`, `rl_odds_dispersion`, `rl_top3_odds_gap`, `rl_top1_odds`, `rl_favorite_rank_gap`, `rl_n_horses`) を計算して DataFrame に追加するが、これらの特徴量名はいずれのモデルの FEATURE_COLS にも存在しない。各モデルの `_prepare_features()` は `available_cols = [c for c in self.FEATURE_COLS if c in df.columns]` で特徴量を抽出するため、rl_* 列は常に破棄される。つまり、計算コストをかけているがモデル学習・推論に一切使用されていない。

Phase 31 Plan では Task 2 で `implied_prob_hhi`/`odds_skewness` の昇格のみを規定しており、rl_* の FEATURE_COLS 追加は意図的にスコープ外だった可能性がある。しかし、その場合 `rl_*` 特徴量を計算する意義が不明確である。

**Fix:** 意図的な設計であれば、Phase 32 等で rl_* を FEATURE_COLS に追加する計画を立てる。そうでなければ、モデルの FEATURE_COLS に rl_* 特徴量を追加する。

### WR-02: `build_features()` (推論パス) に `compute_market_bias()` / `compute_flb_slope()` 呼び出しがない

**File:** `src/features/feature_engine.py:455-464`
**Issue:** `build_features()` の推論パスでは `compute_race_level_features()` のみが呼び出されるが、`compute_market_bias()` や `compute_flb_slope()` は呼び出されない。`compute_flb_slope()` は `implied_prob_hhi` と `odds_skewness` を生成する関数である。Phase 31 でこれら2特徴量が全モデルの FEATURE_COLS に追加されたため、推論時に `implied_prob_hhi` と `odds_skewness` が欠損 (NaN) になる。

これは LightGBM の NaN 処理でクラッシュはしないが、学習時と推論時で特徴量分布が大きく異なる可能性がある。`build_all()` パスでは `compute_flb_slope()` が呼ばれるため、これら特徴量の学習時有効率は100%だが、推論時は0%になる。

**Fix:** `build_features()` に `compute_market_bias()` + `compute_flb_slope()` の呼び出しを追加する。または、推論パスで `implied_prob_hhi`/`odds_skewness` を計算する軽量関数を追加する。

### WR-03: `_compute_for_multi_race()` 内で `race_ids` が `df["race_id"]` の参照 (copy ではない) ための潜在的問題

**File:** `src/features/race_level_features.py:154`
**Issue:** `race_ids = df["race_id"]` は DataFrame の列の直接参照 (view) である。その後、`df["rl_log_odds_entropy"]` 等の列代入で DataFrame に新列が追加される際、pandas の内部実装によっては `df` のインデックスや列管理が再構成され、`race_ids` の参照が無効になるエッジケースが存在する。現在の pandas (2.x) では列追加による既存 view の無効化は稀だが、pandas の実装詳細に依存するコードである。

`_compute_for_single_race()` ではこの問題はない (`race_ids` を使用しない)。

**Fix:** `race_ids = df["race_id"].copy()` または `race_ids = df["race_id"].values` で値を確保する。

## Info

### IN-01: `_compute_for_multi_race()` の `tanodds.where(valid_mask)` が無意味

**File:** `src/features/race_level_features.py:144-153`
**Issue:** `valid_mask = tanodds.notna()` で NaN を検出し、`tanodds_valid = tanodds.where(valid_mask)` で NaN を NaN に置換している。`tanodds` は既に `pd.to_numeric(..., errors="coerce").replace(0, np.nan)` で前処理されているため、`where(valid_mask)` は何も変更しない。`has_any_valid` のチェックのために `valid_mask` 自体は必要だが、`tanodds.where(valid_mask)` の戻り値は `tanodds` と同一。

**Fix:** `tanodds_valid = tanodds` に簡略化する。`has_any_valid = tanodds.notna().any()` はそのまま維持。

### IN-02: 複数レーステストで `rl_log_odds_entropy` の値検証が欠落

**File:** `tests/test_race_level_features.py:260-287`
**Issue:** `test_multi_race_different_values()` は `rl_top1_odds` と `rl_n_horses` のみを検証し、`rl_log_odds_entropy`、`rl_odds_dispersion`、`rl_top3_odds_gap`、`rl_favorite_rank_gap` の値を検証していない。単一レーステスト (`test_3horse_race_all_features_computed`) は全特徴量を検証するが、groupby パス (`_compute_for_multi_race`) のロジックが単一レースパスと異なるため、複数レースでの値検証が望ましい。

**Fix:** 複数レーステストに `rl_log_odds_entropy`、`rl_odds_dispersion`、`rl_top3_odds_gap`、`rl_favorite_rank_gap` の期待値アサーションを追加する。

### IN-03: `_assign_n_horses_grouped()` で `field_size` が同一レース内で不一致の場合に `rl_n_horses` が馬ごとに異なる値になる可能性

**File:** `src/features/race_level_features.py:259-266`
**Issue:** `np.where(fs > 0, fs, fallback)` は行ごとに評価される。もし `field_size` 列が同一レース内の異なる馬で異なる値を持つ場合 (例: 一部の馬で `field_size > 0`、他の馬で `field_size = 0`)、`rl_n_horses` が同一レース内で馬ごとに異なる値になる。これは通常は発生しない (`_map_basic_features` でレース単位に設定される) が、防御的なチェックがない。

**Fix:** レース単位で `field_size` の最初の有効値を使用するように `field_size` をグループ内で `ffill`/`bfill` する、または `groupby("race_id").transform("first")` で正規化する。

### IN-04: `test_tanodds_with_zero_and_nan` で `rl_n_horses` の期待値がドキュメントと不一致

**File:** `tests/test_race_level_features.py:164`
**Issue:** テストコメントに「rl_n_horses = 4 (field_size)」とあるが、実際のアサーションでは `rl_n_horses` の値を検証していない。有効オッズが2頭のみの場合に `field_size=4` が使用されるか `n_valid=2` が使用されるかは `_assign_n_horses` のロジックに依存し、`field_size` が優先される。テストはコメント通りの動作を検証すべき。

**Fix:** `assert result["rl_n_horses"].iloc[0] == 4` のアサーションを追加する。

---

_Reviewed: 2026-05-18T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
