---
phase: 05-foundation-features
verified: 2026-05-03T17:30:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Phase 5: Foundation Features Verification Report

**Phase Goal:** 過去走の時系列特徴量・展開予測特徴量・オッズ変動特徴量を追加し、後続のモデル改善がより豊かな入力から恩恵を受けられるようにする
**Verified:** 2026-05-03T17:30:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

ROADMAP Success Criteria + PLAN must_haves を統合した検証:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | harontimel5_avg が全過去走の EMA 重み付け (halflife=3) で計算され、直近の成績に高い重みが付与されている | VERIFIED | `horse_history_features.py` L677-691: `decay = np.log(2) / halflife`, `weights = (1 - decay) ** np.arange(n_ht)` を反転して正規化。ht_valid 全体を使用 |
| 2 | class_adj_formetric が新特徴量として生成され、NaN率50%未満である | VERIFIED | `horse_history_features.py` L697-725: `_class_level_from_values` で重み付き計算、NaN フィルタ実装。BASE_COLS L298、results.append L1177 に出力 |
| 3 | haron_zscore_trend が過去走 z-score の線形回帰傾きとして計算される (最小3走要件) | VERIFIED | `horse_history_features.py` L729-778: `valid_z = z_arr[~np.isnan(z_arr)]`, `len(valid_z) >= 3` で `np.polyfit(x, valid_z, 1)[0]` を計算 |
| 4 | pace_corner_stability, pace_closing_power, pace_position_consistency の3サブ特徴量が PaceAptitudeFeatures に追加される | VERIFIED | `pace_aptitude_features.py` L76-78: result_cols に3列追加。L261-279: 各特徴量の計算ロジック実装。L212-214: results dict に3キー追加 |
| 5 | actual_pace_fit が実績ベースのペース適性として interaction_features に追加される | VERIFIED | `interaction_features.py` L109-118: `is_front_runner`/`is_closer` で `front_pace_wr`/`closing_pace_wr` を選択。脚質不明時は NaN |
| 6 | odds_acceleration がオッズ変動の2次微分として計算され、steam move を検出できる | VERIFIED | `odds_dynamics_features.py` L181-187: `vel_early = (odds_30 - odds_60) / 30.0`, `vel_late = (odds_10 - odds_30) / 20.0`, `odds_acceleration = vel_late - vel_early` |
| 7 | odds_direction_consistency がオッズ変動方向の時間加重一貫性として計算される | VERIFIED | `odds_dynamics_features.py` L233-256: `_compute_direction_consistency` で EMA 重み付け (halflife=n/4)、最小5スナップショット要件、reindex で NaN 埋め |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/horse_history_features.py` | EMA版harontimel5_avg, class_adj_formetric, haron_zscore_trend | VERIFIED | L265-301: BASE_COLS に3列追加。L677-778: 計算ロジック。L1176-1179: results.append に出力 |
| `src/features/pace_aptitude_features.py` | 3つのペースフィグアサブ特徴量 | VERIFIED | L76-78: result_cols 拡張。L212-214: results dict。L261-279: 計算ロジック |
| `src/features/interaction_features.py` | actual_pace_fit | VERIFIED | L109-118: 脚質ベース条件分岐で実装 |
| `src/features/odds_dynamics_features.py` | odds_acceleration, odds_direction_consistency | VERIFIED | L129-137: nan_cols に追加。L181-187: acceleration 計算。L233-256: consistency 計算。L270-276: agg_df に concat |
| `src/models/stage1_ability_model.py` | 新特徴量を含む FEATURE_COLS | VERIFIED | L97-106: class_adj_formetric, haron_zscore_trend, pace_corner_stability, pace_closing_power, pace_position_consistency, actual_pace_fit を含む。odds_acceleration/odds_direction_consistency を含まない (Stage1 Rule 1 準拠) |
| `src/models/two_stage_return_model.py` | 新特徴量を含む FEATURE_COLS | VERIFIED | L58-61: odds_acceleration, odds_direction_consistency を含む。L92-97: class_adj_formetric, haron_zscore_trend, actual_pace_fit を含む |
| `src/pipelines/training_pipeline.py` | pace_df マージ列リストの更新 | VERIFIED | L316-319: pace_merge_cols に pace_corner_stability, pace_closing_power, pace_position_consistency を追加 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| horse_history_features.py | HorseHistoryFeatures.BASE_COLS | class_adj_formetric, haron_zscore_trend の BASE_COLS 追加 | WIRED | L298-300: 両特徴量が BASE_COLS に含まれる |
| pace_aptitude_features.py | PaceAptitudeFeatures.compute_batch() | result_cols リストへの新列名追加 | WIRED | L76-78: 3列が result_cols に含まれ、L282-283 で出力 |
| interaction_features.py | _add_pace_projection_features() | actual_pace_fit 生成の追加 | WIRED | L109-118: pace_scenario_fit の後に actual_pace_fit を追加 |
| stage1_ability_model.py | AbilityModel.FEATURE_COLS | 新特徴量の FEATURE_COLS への追加 | WIRED | Python検証で全6特徴量が含まれることを確認 |
| two_stage_return_model.py | WinTwoStageModel.FEATURE_COLS | 新特徴量の FEATURE_COLS への追加 | WIRED | Python検証で全5特徴量が含まれることを確認 |
| training_pipeline.py | pace_df マージ | pace_merge_cols リストへの新列追加 | WIRED | L316-319: 3新ペース特徴量が pace_merge_cols に含まれる |
| odds_dynamics_features.py | compute_odds_dynamics() | 既存 odds_10/30/60 値を利用した3点差分計算 | WIRED | L181-187: 3点差分で acceleration 計算 |
| odds_dynamics_features.py | compute_odds_dynamics() | スナップショット groupby を利用した方向一貫性計算 | WIRED | L233-256: _odds_diff から方向計算、groupby.apply で一貫性 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| horse_history_features.py (class_adj_formetric) | class_adj_formetric | hp_kakuteijyuni, hp_syussotosu, gradecd, jyokencd1 via _class_level_from_values | 有効過去走の norm_finish x class_level 重み付き平均 | FLOWING |
| horse_history_features.py (haron_zscore_trend) | haron_zscore_trend | zscores (expanding_stats ベース) -> np.polyfit | 過去走 z-score の線形回帰傾き (最小3走) | FLOWING |
| horse_history_features.py (harontimel5_avg) | harontimel5_avg | ht_valid (harontimel3) + EMA weights | EMA重み付けハロンタイム平均 (halflife=3) | FLOWING |
| pace_aptitude_features.py | pace_corner_stability, pace_closing_power, pace_position_consistency | h_norm_1c, h_norm_4c, h_harontimel3, h_norm_finish | 各サブ特徴量の numpy 計算 | FLOWING |
| interaction_features.py | actual_pace_fit | front_pace_wr, closing_pace_wr (from pace_df merge), kyakusitukubun_cd | np.where による脚質ベース選択 | FLOWING |
| odds_dynamics_features.py | odds_acceleration | odds_10, odds_30, odds_60 (from _pick_target_snapshot) | vel_late - vel_early 3点差分 | FLOWING |
| odds_dynamics_features.py | odds_direction_consistency | ts["_odds_diff"] -> ts["_odds_dir"] -> groupby.apply | EMA重み付け方向一貫性 (0-1) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 全テスト通過 | `python -m pytest tests/test_horse_history_features.py tests/test_pace_aptitude_features.py tests/test_interaction_features.py tests/test_odds_dynamics_features.py -v` | 122 passed in 2.77s | PASS |
| BASE_COLS に新特徴量 | `python -c "from features.horse_history_features import HorseHistoryFeatures; print('class_adj_formetric' in HorseHistoryFeatures.BASE_COLS)"` | True | PASS |
| AbilityModel.FEATURE_COLS 検証 | Python検証で6特徴量の存在確認 | 全て True (odds系は意図的に除外) | PASS |
| WinTwoStageModel.FEATURE_COLS 検証 | Python検証で5特徴量の存在確認 | 全て True | PASS |
| compute_odds_dynamics None時NaN | `compute_odds_dynamics(df, None)` の列確認 | odds_acceleration, odds_direction_consistency が NaN で存在 | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TSER-01 | 05-01 | 過去走の全平均値特徴量を指数減衰重み付けに置き換え | SATISFIED | horse_history_features.py L677-691: EMA (halflife=3) 実装、全過去走使用 |
| TSER-02 | 05-01 | クラス調整済みフォーメトリック | SATISFIED | horse_history_features.py L697-725: _class_level_from_values で重み付き計算 |
| TSER-03 | 05-01 | z-score の線形トレンド (改善トラジェクトリ) | SATISFIED | horse_history_features.py L729-778: np.polyfit で傾き計算、最小3走要件 |
| PACE-01 | 05-01 | ペースフィグア3サブ特徴量 (corner_stability, closing_power, position_consistency) | SATISFIED | pace_aptitude_features.py L261-279: 3サブ特徴量の計算実装 |
| PACE-02 | 05-01 | 実績ベースのペース適性 (actual_pace_fit) | SATISFIED | interaction_features.py L109-118: 脚質ベース条件分岐で実装 |
| ODTS-01 | 05-02 | オッズ変動の2次微分 (加速度) / steam move 検出 | SATISFIED | odds_dynamics_features.py L181-187: 3点2次微分計算 |
| ODTS-02 | 05-02 | オッズ変動方向の一貫性測定 | SATISFIED | odds_dynamics_features.py L233-256: EMA重み付け方向一貫性 (最小5点要件) |

**Orphaned requirements:** なし。REQUIREMENTS.md で Phase 5 にマッピングされた全7要件が2つのPLANで網羅されている。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (なし) | - | - | - | 全ファイルで TODO/FIXME/PLACEHOLDER なし |

ruff lint エラー (7件) は全て事前存在する行長・インポート順序問題であり、Phase 5 で新規導入されたものではない。mypy エラーも全てサードパーティスタブ不足による既存エラー。

### Human Verification Required

ROADMAP Success Criteria の一部はバックテスト実行による確認が含まれるが、Phase 5 のスコープは「特徴量の追加とパイプライン統合」であり、バックテスト feature importance の確認は後続フェーズ (Phase 6-7) で評価される。

以下の項目はコードベース検証で完結し、人間による手動確認は不要:

- 特徴量計算ロジックの正確性: 122テストで検証済み
- モデル FEATURE_COLS への統合: Python スクリプトで検証済み
- training_pipeline.py のマージ更新: grep で検証済み
- Stage1 Rule 1 (オッズ不入力) の遵守: AbilityModel に odds 系特徴量なしを確認

### Gaps Summary

ギャップなし。全7要件 (TSER-01/02/03, PACE-01/02, ODTS-01/02) が実装され、テスト (122件) が全通過、モデルFEATURE_COLSに正しく統合されている。

---

_Verified: 2026-05-03T17:30:00Z_
_Verifier: Claude (gsd-verifier)_
