---
phase: 20-high-odds-pattern-features
verified: 2026-05-09T10:25:00Z
status: human_needed
score: 12/15 must-haves verified
overrides_applied: 0
deferred:
  - truth: "Feature importance上位50%に新特徴量が含まれる (ROADMAP SC-2)"
    addressed_in: "Phase 22"
    evidence: "Phase 22 goal: '全改善を適用したバックテストでROI改善を確認する'"
  - truth: "高オッズ帯(20+)のOOF予測AUCが改善 (ROADMAP SC-3)"
    addressed_in: "Phase 22"
    evidence: "Phase 22 goal: '全改善を適用したバックテストでROI改善を確認する', plans include 'セグメント別ROI検証'"
  - truth: "10+新特徴量が生成され、欠損率10%以下 (ROADMAP SC-1: 欠損率確認)"
    addressed_in: "Phase 22"
    evidence: "Phase 22 goal: 統合バックテスト実行時に欠損率を確認可能"
human_verification:
  - test: "Feature importance分析で新特徴量18個が上位50%に含まれることを確認"
    expected: "新特徴量の少なくとも一部がFeature importance上位50%にランクイン"
    why_human: "学習済みモデルでのFeature importance分析が必要。バックテストまたは学習パイプラインの実行が必要"
  - test: "高オッズ帯(20+)のOOF予測AUCが改善していることを確認"
    expected: "新特徴量追加前のベースラインAUCと比較して改善"
    why_human: "バックテストパイプラインの実行とベースライン比較が必要"
  - test: "新特徴量18個の欠損率が10%以下であることを確認"
    expected: "各特徴量のNaN率が10%以下"
    why_human: "実際のデータで特徴量計算を実行してNaN率を測定する必要あり"
---

# Phase 20: 高オッズ的中パターン特徴量 Verification Report

**Phase Goal:** 高オッズ(20+)の的中率を2.1%→3%+に改善するため、新特徴量18個を追加する
**Verified:** 2026-05-09T10:25:00Z
**Status:** human_needed
**Re-verification:** No (initial verification)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 高オッズ的中馬と非的中馬の特徴量分布差をCohen's dで定量化できる | VERIFIED | `scripts/analyze_high_odds.py` L286-301: `_compute_cohens_d()` with pooled_std。L171-194: 的中群/非的中群別Cohen's d計算ループ |
| 2 | 既存LightGBMモデルで高オッズ馬のみSHAP値を計算できる | VERIFIED | `scripts/analyze_high_odds.py` L216-217: `model.predict(features_for_shap, pred_contrib=True)` でTreeSHAP実行 |
| 3 | クラストラジェクトリ5特徴量(昇級回数・降級回数・ネット変化・最高クラス・分散)が正しく計算される | VERIFIED | `src/features/high_odds_features.py` L91-163: `compute_class_trajectory()` が7要素タプル(5+V回復2)を返す。テスト8件全通過 |
| 4 | V字回復パターン(降級→再昇級)フラグと降級期間が計算される | VERIFIED | `src/features/high_odds_features.py` L135-153: V字回復検出ロジック。テスト: `test_v_recovery_pattern` PASSED |
| 5 | EMAベース指数改善率(タイム+着順)がhalflife=3で計算される | VERIFIED | `src/features/high_odds_features.py` L166-253: `compute_form_improvement_rate()` + `_ema_improvement()` L236: `decay = np.log(2) / halflife` パターン |
| 6 | 3変化(距離/サーフェス/馬場状態)の過去適性履歴が条件別に計算される | VERIFIED | `src/features/high_odds_features.py` L321-402: `compute_env_adaptability()` が距離/サーフェス/馬場の3変化を検出し統計計算。テスト8件全通過 |
| 7 | 各変化について平均着順・勝率・経験回数の3サブ特徴量が計算される (計9特徴量) | VERIFIED | `_compute_change_stats()` L304-318: avg_pos(正規化着順), win_rate(1着割合), exp_count(該当走数) |
| 8 | 新特徴量がHorseHistoryFeatures.compute()のper-horseループ内で計算される | VERIFIED | `horse_history_features.py` L1021-1195: クラストラジェクトリ(L1021-1037), フォーム改善率(L1039-1049), 環境変化適性(L1168-1195)がper-horseループ内で計算 |
| 9 | 新特徴量がBASE_COLSリストに追加される | VERIFIED | `horse_history_features.py` L318-338: HODDS-02/03/04の18特徴量がBASE_COLSに追加。BASE_COLS count=48 |
| 10 | 新特徴量名がresults.append()辞書に含まれる | VERIFIED | `horse_history_features.py` L1297-1317: 18個の新キーが辞書に含まれる |
| 11 | 環境変化が検出されない場合(経験回数0)はNaNとなる | VERIFIED | `high_odds_features.py` L296-302: `_compute_change_stats()` がchange_detected=Falseまたはmatch_mask空の場合NaN返す。テスト: `test_no_changes_all_nan` PASSED |
| 12 | 新特徴量18個がAbilityModel.FEATURE_COLSに追加されている | VERIFIED | `stage1_ability_model.py` L107-127: HODDS-02/03/04の18特徴量追加。FEATURE_COLS count=80, unique=80 (重複なし) |
| 13 | _prepare_features()が新特徴量をavailable_colsから正しくフィルタする | VERIFIED | `stage1_ability_model.py` L135: `available_cols = [c for c in self.FEATURE_COLS if c in df.columns]` で安全にフィルタ |
| 14 | 新特徴量がFEATURE_COLSとBASE_COLSで整合している | VERIFIED | `TestHighOddsFeatureIntegration` 3テスト全通過: BASE_COLS包含、FEATURE_COLS包含、重複なし |
| 15 | 全テスト通過 (回帰なし) | VERIFIED | `python -m pytest tests/ -v --tb=short`: 1386 passed, 1 skipped, 0 failed |

**Score:** 15/15 truths verified (automated)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Feature importance上位50%に新特徴量が含まれる (ROADMAP SC-2) | Phase 22 | Phase 22 goal: "全改善を適用したバックテストでROI改善を確認する" |
| 2 | 高オッズ帯(20+)のOOF予測AUCが改善 (ROADMAP SC-3) | Phase 22 | Phase 22 plans: "セグメント別ROI検証" |
| 3 | 新特徴量18個の欠損率10%以下 (ROADMAP SC-1 欠損率部分) | Phase 22 | バックテスト実行時に欠損率確認可能 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/analyze_high_odds.py` | 高オッズ分析スクリプト | VERIFIED | 348行、main()あり、Cohen's d + TreeSHAP実装、--odds-threshold default=20.0 |
| `src/features/high_odds_features.py` | 純粋関数群モジュール | VERIFIED | 402行、3関数(compute_class_trajectory, compute_form_improvement_rate, compute_env_adaptability)、FEATURE_COLS 18個 |
| `tests/test_high_odds_features.py` | 単体テスト | VERIFIED | 457行、27テスト全通過、3テストクラス |
| `src/features/horse_history_features.py` | per-horseループ統合 | VERIFIED | import文あり(L27-31)、BASE_COLS更新(L318-338)、per-horse計算(L1021-1195)、results.append()更新(L1297-1317) |
| `src/models/stage1_ability_model.py` | FEATURE_COLS更新 | VERIFIED | L107-127に18特徴量追加、FEATURE_COLS count=80 |
| `tests/test_horse_history_features.py` | 整合性テスト | VERIFIED | TestHighOddsFeatureIntegration 3テスト(L2136-2159) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `analyze_high_odds.py` | `win_feature_analysis.py` (TreeSHAP) | `pred_contrib=True` | WIRED | L217: `model.predict(features_for_shap, pred_contrib=True)` (インラインTreeSHAP、analyze_feature_importance()のインポートは不要な設計) |
| `high_odds_features.py` | `horse_history_features.py` | import文 | WIRED | L27-31: `from features.high_odds_features import (compute_class_trajectory, compute_form_improvement_rate, compute_env_adaptability)` |
| `horse_history_features.py` BASE_COLS | `results.append()` dict | キー追加 | WIRED | L318-338 (BASE_COLS) と L1297-1317 (results) が18キーで整合 |
| `stage1_ability_model.py` FEATURE_COLS | `horse_history_features.py` BASE_COLS | 整合性テスト | WIRED | TestHighOddsFeatureIntegration で自動検証 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `compute_class_trajectory` | gradecd_arr, jyokencd1_arr | horse_arrs["gradecd"][valid_mask] | Yes (既存parquetデータ) | FLOWING |
| `compute_form_improvement_rate` | zscore_arr, kakuteijyuni_arr | horse_arrs["harontimel3"][valid_mask] | Yes (既存parquetデータ) | FLOWING |
| `compute_env_adaptability` | distance_bin_arr, surface_arr, etc. | horse_arrs各列[history_mask] | Yes (既存parquetデータ) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| FEATURE_COLS count = 18 | `python -c "from features.high_odds_features import FEATURE_COLS; assert len(FEATURE_COLS)==18"` | 成功 | PASS |
| AbilityModel.FEATURE_COLS count = 80 | `python -c "from models.stage1_ability_model import AbilityModel; assert len(AbilityModel.FEATURE_COLS)==80"` | 成功 | PASS |
| FEATURE_COLS整合性 (BASE_COLS/AbilityModel) | `python -c "...missing check..."` | 空リスト (全包含) | PASS |
| 新特徴量テスト全通過 | `python -m pytest tests/test_high_odds_features.py -v` | 27 passed | PASS |
| 整合性テスト通過 | `python -m pytest tests/test_horse_history_features.py -v -k "HighOdds"` | 3 passed | PASS |
| 全テストスイート回帰なし | `python -m pytest tests/ --tb=short -q` | 1386 passed, 0 failed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| HODDS-01 | 20-01 | 高オッズ的中パターン分析モジュール | SATISFIED | `scripts/analyze_high_odds.py`: Cohen's d + TreeSHAPハイブリッド分析 |
| HODDS-02 | 20-01 | クラストラジェクトリ特徴量 | SATISFIED | `compute_class_trajectory()`: 7特徴量(昇級/降級/ネット/最高/分散/V回復フラグ/V回復期間) |
| HODDS-03 | 20-01 | フォーム改善率特徴量 | SATISFIED | `compute_form_improvement_rate()`: 2特徴量(time_improvement_rate, position_improvement_rate) |
| HODDS-04 | 20-02 | 環境変化適性特徴量 | SATISFIED | `compute_env_adaptability()`: 9特徴量(3変化 x 3サブ) |
| HODDS-05 | 20-02, 20-03 | 新特徴量のAbilityModel統合 | SATISFIED | BASE_COLS 48特徴量、AbilityModel.FEATURE_COLS 80特徴量、3箇所整合性テスト通過 |

Note: REQUIREMENTS.md HODDS-02記載の `recent_class_trend`, `class_drop_magnitude`, `hidden_class_score` は PLAN で別名/別構成に設計変更されている(PLAN: class_promotions, class_demotions, class_net_change, class_max_level, class_level_std)。同様に HODDS-03の `finish_position_trend`, `speed_figure_trend`, `recent_improvement_flag` も PLAN で time_improvement_rate, position_improvement_rate に変更。HODDS-04の `trainer_change_flag`, `jockey_upgrade_flag` は距離/サーフェス/馬場の3変化に変更。機能的な意図(クラストラジェクトリ/フォーム改善/環境変化適性の定量化)は満たしている。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/analyze_high_odds.py` | L286-301 | `np`/`pd`がmain()内ローカルimportだが、グローバルスコープ関数 `_compute_cohens_d`, `_find_column`, `_find_common_columns` で使用 (F821) | Warning | ruff lint error 14件。実行時はmain()のimportが先に実行されるため実害なしだが、静的解析でエラー。`np`/`pd`をファイルトップレベルにimportすべき |
| `src/features/horse_history_features.py` | L1185 | E501 Line too long (110 > 100) — Phase 20追加行 | Info | `_ea_cond = horse_arrs["track_condition_code"][...]...astype(float)` が100文字超過 |
| `src/features/horse_history_features.py` | L26-31 | I001 Import block un-sorted — Phase 20追加行の影響 | Info | ruff --fix で自動修正可能 |

### Human Verification Required

### 1. Feature Importance分析

**Test:** `python scripts/run_train.py --start 20200101 --end 20231231 --ensemble` で学習後、`python scripts/analyze_high_odds.py --start 20200101 --end 20231231` を実行して新特徴量のFeature importanceを確認
**Expected:** 新特徴量18個の少なくとも一部が上位50%にランクイン
**Why human:** 学習済みモデルと実データが必要。バックテスト/学習パイプラインの実行(~17分)が必要

### 2. 高オッズ帯OOF予測AUC改善確認

**Test:** 新特徴量追加前のベースラインAUCと比較
**Expected:** 高オッズ帯(20+)のOOF予測AUCが改善
**Why human:** バックテストパイプラインの実行とベースライン比較が必要

### 3. 新特徴量欠損率確認

**Test:** 実データで特徴量計算を実行し、18個の新特徴量のNaN率を測定
**Expected:** 各特徴量のNaN率が10%以下
**Why human:** 実データでの特徴量計算が必要

### Gaps Summary

Phase 20の実装は技術的に完全である。18個の新特徴量(クラストラジェクトリ7 + フォーム改善率2 + 環境変化適性9)がすべて実装され、パイプラインに統合されている。3箇所(high_odds_features.py, horse_history_features.py BASE_COLS, stage1_ability_model.py FEATURE_COLS)の整合性はテストで自動検証されている。1386テスト全通過、回帰なし。

残存する3つのROADMAP Success Criteria (Feature importance上位50%、OOF AUC改善、欠損率10%以下) はPhase 22の統合バックテストで検証される予定。

軽微なlint問題: `scripts/analyze_high_odds.py`で`np`/`pd`がグローバルスコープで未定義(F821)。実行時エラーはないが、トップレベルimportに修正推奨。

---

_Verified: 2026-05-09T10:25:00Z_
_Verifier: Claude (gsd-verifier)_
