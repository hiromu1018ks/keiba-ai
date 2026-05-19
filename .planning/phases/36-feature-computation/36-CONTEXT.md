# Phase 36: Feature Computation - Context

**Gathered:** 2026-05-19
**Status:** Ready for planning

<domain>
## Phase Boundary

All new turf-focused features (relative ranks, conditional interactions, haron/lap history) are computed PIT-safe and registered across all 12 models.

**In scope:**
- TRF-01: form_trend_race_rank, blood_total_wr_race_rank, blood_surface_wr_race_rank を add_race_transforms() に追加
- TRF-02: weighted_recent_form (直近3走重み付き成績) を horse_history_features.py に追加
- TRF-03: 全TRF特徴量を12モデルのFEATURE_COLSに登録
- INT-01: grade_x_form_trend (grade_code × form_trend) 交互作用を interaction_features.py に追加
- INT-02: distance_x_closing_index (kyori × closing_index_avg) 交互作用を追加
- INT-03: grade_x_blood_prize_log (grade_code × blood_prize_log) 交互作用を追加
- INT-04: 全INT特徴量を12モデルのFEATURE_COLSに登録
- HLF-01: 過走上がり3F/4Fの平均・z-score・トレンド特徴量をPIT-safeに計算 (expanding_stats + race_date < target_date)
- HLF-02: 上がりタイムのレース内相対ランキング (harontime_l3/l4/unified_race_rank) を計算
- HLF-03: LapTime1~25からレースペース特徴量(前半/中盤/後半ペース比)を計算 (過走レースのみ)
- HLF-04: 全HLF特徴量を12モデルのFEATURE_COLSに登録
- HLF-05: TrainingPipeline._train_submodel() と BettingOrchestrator.build_features() の両パスでHLF特徴量が計算されることを確認
- harontime_last3f統合列の作成(距離別L3/L4自動選択)
- POST_RACE安全性: 全新特徴量が3層CI漏洩テストを通過することを確認

**Out of scope:**
- ETL実行 (Phase 35)
- バックテスト実行 (Phase 38)
- モデル再学習・ハイパーパラメータチューニング
- EVキャリブレーション (Phase 37)
- コーナー通過順位からの展開特徴量 (将来フェーズ HLF-06)
- ペースプロファイル分類 (将来フェーズ HLF-07)

</domain>

<decisions>
## Implementation Decisions

### HaronTime統合ロジック (Phase 35 D-05委譲)
- **D-01:** 距離別自動選択で統合列`harontime_last3f`を作成。距離閾値はPhase 35品質確認後に決定(閾値未定時は2000mをデフォルトとする)
- **D-02:** L3とL4の両方について独立に履歴統計(avg/zscore/trend)を計算。統合列`harontime_last3f`からも同様に計算
- **D-03:** HaronTime特徴量は4統計量(avg/zscore/trend/race_rank) × 3列(L3/L4/unified) = 12特徴量
  - avg: EMA加重(halflife=3)平均 — 既存harontimel5_avgと同じ方式
  - zscore: expanding_stats hierarchical fallback (FALLBACK_LEVELS流用)
  - trend: 直近3走の線形回帰傾き
  - race_rank: groupby("race_id").rank(pct=True) (HLF-02)

### LapTimeペース特徴量 (HLF-03)
- **D-04:** 25個のLapTimeを等分3分割(各1/3)。ラップ数 = kyori/200。例: 2400m=12ラップ→前半4/中盤4/後半4
- **D-05:** ペース比特徴量: pace_ratio = 後半平均ペース / 前半平均ペース (< 1.0 = 末脚速い, > 1.0 = 一定ペース)
- **D-06:** LapTime履歴特徴量: 過走レースのpace_ratioのavg/zscore/trend + 各セグメント(early/mid/late)のavg。PIT-safe (race_date < target_date)

### weighted_recent_form (TRF-02)
- **D-07:** 加重方式はEMA(halflife=3)。既存harontimel5_avgと整合
- **D-08:** 2指標を計算:
  - `weighted_recent_form_finish`: EMA(norm_finish_logit, halflife=3, 直近3走)
  - `weighted_recent_form_time`: EMA(timediff, halflife=3, 直近3走)
- **D-09:** norm_finish_logitは頭数正規化済み(logit変換)、timediffは勝馬との着差(秒)。2指標は相補的

### 交互作用ベース列 (INT-01~03)
- **D-10:** INT-01: `grade_code × form_trend` (グレード別調子トレンド)。grade_codeは0~6の数値
- **D-11:** INT-02: `kyori × closing_index_avg` (距離別追込力)。kyoriは連続値距離(m)
- **D-12:** INT-03: `grade_code × blood_prize_log` (グレード別血統賞金)。既存NaN安全パターン(.where(notna))を使用

### Turf相対特徴量 (TRF-01)
- **D-13:** add_race_transforms()に3列追加: form_trend_race_rank, blood_total_wr_race_rank, blood_surface_wr_race_rank。既存7列パターンに準拠
- **D-14:** race_rankはtraining pathではgroupby("race_id").rank()、backtest pathでは単一race内で直接.rank() (既存パターン)

### FEATURE_COLS登録 (TRF-03/INT-04/HLF-04)
- **D-15:** 全新特徴量を12モデル全てのFEATURE_COLSに登録:
  - AbilityModel, WinTwoStageModel, PlaceTwoStageModel, ConformalEVModel, EVCorrectionModel, PlaceEVCorrectionModel, MarketModel, PlaceAbilityModel, RaceQualityScreener, WideTwoStageModel, RegimeDetector, StackedEnsemble

### 双方パス対応 (HLF-05)
- **D-16:** 全HLF/TRF特徴量が両方のパスで計算されることを確認:
  - Training path: `_train_submodel()` (TrainingPipeline)
  - Inference path: `RacePredictor.predict()` (BacktestEngine) + `PaperPredictor`

### Claude's Discretion
- 距離閾値のデフォルト値(Phase 35品質確認前の初期値)
- LapTime特徴量の具体的なexpanding_stats実装(セグメント計算ロジック)
- 各特徴量のNaNハンドリング(過走0走の場合のデフォルト値)
- テストケースの設計(PIT安全性・双方向パス一致性・FEATURE_COLS完全性)
- harontime_last3f統合列の具体的なcoalesceロジック
- LapTime列名の正規化(Phase 35 ETL出力との整合)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 特徴量モジュール (変更対象)
- `src/features/horse_history_features.py` — HorseHistoryFeatures.compute()、add_race_transforms()、BASE_COLS、expanding_stats。HLF-01/02、TRF-01/02の主要変更対象
- `src/features/interaction_features.py` — compute_interaction_features()、INTERACTION_COLS。INT-01~03の変更対象
- `src/features/feature_engine.py` — FeatureEngine.build_all() と build_features()。特徴量モジュール呼び出しのオーケストレーション

### FEATURE_COLS定義 (12モデル登録)
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS
- `src/models/two_stage_return_model.py` — WinTwoStageModel / PlaceTwoStageModel FEATURE_COLS (HIT + RETURN)
- `src/models/ev_correction_model.py` — EVCorrectionModel / PlaceEVCorrectionModel FEATURE_COLS
- `src/models/conformal_ev_model.py` — ConformalEVModel.FEATURE_COLS
- `src/models/market_model.py` — MarketModel.FEATURE_COLS
- `src/models/stacked_ensemble.py` — StackedEnsemble.FEATURE_COLS
- `src/models/regime_detector.py` — RegimeDetector.FEATURE_COLS
- `src/models/place_ability_model.py` — PlaceAbilityModel.FEATURE_COLS
- `src/models/race_quality_screener.py` — RaceQualityScreener.FEATURE_COLS
- `src/models/wide_two_stage_model.py` — WideTwoStageModel.FEATURE_COLS

### 双方パス対応
- `src/pipelines/training_pipeline.py` lines 280-400 — _train_submodel() 特徴量計算パス
- `src/backtest/engine.py` lines 653-791 — BacktestEngine 事前特徴量計算
- `src/backtest/race_predictor.py` lines 51-228 — RacePredictor.predict() 推論時特徴量計算

### POST_RACE安全性
- `src/domain/types.py` lines 38-55 — POST_RACE_COLS定義 (41列、Phase 35で拡張)
- `tests/test_post_race_leakage.py` — 3層CI漏洩検出テスト

### データソース (Phase 35 ETL出力)
- `data/raw/entries.parquet` — HaronTimeL3/L4 (float64, sentinel NaN化済み)
- `data/raw/races.parquet` — LapTime1~25 (float64, sentinel NaN化済み)

### 要件定義
- `.planning/REQUIREMENTS.md` §HLF — HLF-01~05
- `.planning/REQUIREMENTS.md` §TRF — TRF-01~03
- `.planning/REQUIREMENTS.md` §INT — INT-01~04

### Prior Phase Context
- `.planning/phases/35-etl-data-foundation/35-CONTEXT.md` — ETL基盤(sentinel NaN化、POST_RACE拡張、HaronTime相互排他性)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/features/horse_history_features.py::expanding_stats`: 階層型FALLBACK_LEVELS(L1~L4)のexpanding stats。HaronTime/LapTimeのzscore計算にそのまま流用
- `src/features/horse_history_features.py::add_race_transforms()`: groupby("race_id").rank(pct=True)パターン。TRF-01の3列追加は既存7列への追加のみ
- `src/features/horse_history_features.py::compute()`: 過走配列をdict-of-numpyに変換→searchsortedでPIT-safe検索の最適化パターン。LapTime検索にも適用
- `src/features/interaction_features.py`: .where(notna) NaN安全な数値積パターン。INT-01~03は既存12交互作用への追加
- `src/features/horse_history_features.py::BASE_COLS`: 52列の履歴特徴量一覧。新特徴量をここに追加

### Established Patterns
- PIT-safeパターン: searchsorted(target_date, side="left") on sorted race_date arrays → strictly past data only
- EMA加重: halflife=3 で w[i] = (1/2)^(i/halflife)、正規化後に加重平均
- race_rank: training=groupby("race_id").rank(pct=True)、backtest=direct .rank(pct=True) on single race
- FEATURE_COLS登録: 全12モデルのclass-level listに追加 → [c for c in FEATURE_COLS if c in df.columns] で安全選択
- 双方パス: _train_submodelとRacePredictor.predictで同一特徴量セットを生成。Historyはbatch pre-compute、race_rank/interactionはper-race

### Integration Points
- `src/features/horse_history_features.py`: HLF-01/02、TRF-01/02の主要変更対象。compute()内にHaronTime/LapTime集計ロジックを追加
- `src/features/interaction_features.py`: INT-01~03の変更対象。compute_interaction_features()に3つの新積を追加
- `src/features/horse_history_features.py::add_race_transforms()`: TRF-01の3列をrace_rank_colsに追加
- 12モデルのFEATURE_COLS: 全モデルに新特徴量名を追加
- `src/pipelines/training_pipeline.py::_train_submodel()`: HaronTime/LapTime列のmerge追加が必要な可能性
- `src/backtest/race_predictor.py::predict()`: 推論パスの特徴量計算に新列を追加

### HaronTime L3/L4の現状
- **HaronTimeL3**: harontimel5_avg/zscore等に既に統合済み。harontimel5_* は harontimel3 + harontimel5 のEMA加重
- **HaronTimeL4**: 未使用(float64化のみ完了)。HLF-01で新規に履歴統計を計算
- **相互排他性**: Phase 35品質確認で検証予定。L3のみ/L4のみ/両方/なしの4分類分布を確認

</code_context>

<specifics>
## Specific Ideas

- harontime_last3f統合列の距離閾値はPhase 35品質確認後に決定。初期値2000m (短距離<middle<->では3F、中長距離では4F)
- LapTimeはraces Parquet内に格納(Phase 35 ETL出力)。HorseHistoryFeatures.compute()内でentriesのkettonum→過走racesのLapTimeをlookupする必要あり
- HaronTimeはentries Parquet内のため、過走entriesから直接lookup可能(LapTimeはracesのためjoinが必要)
- LapTime pace_ratio計算は過走レースのみ(PIT-safe)。当該レースのLapTimeはPOST_RACEのため使用不可
- HaronTime trendは既存harontime_late_trendと同パターン(直近3走の線形回帰傾き)
- weighted_recent_form_finishは既存form_trendとは異なる信号: form_trend=傾き(方向性)、weighted_recent_form_finish=加重平均(絶対レベル)
- 全ての新特徴量はPOST_RACEデータから派生しているため、PIT安全性の3層CI漏洩テストの対象

</specifics>

<deferred>
## Deferred Ideas

- harontime_last3f統合列の距離閾値の最終決定 — Phase 35品質確認の相互排他性検証結果に依存
- コーナー通過順位(Jyuni1c~4c)からの展開特徴量 — 将来フェーズ HLF-06
- ペースプロファイル分類(スロー/ミドル/ハイペース) — 将来フェーズ HLF-07
- 末脚指数(上がり3F - レース平均上がり) — 将来フェーズ HLF-08
- LapTimeの中盤セグメント単体特徴量(early/mid/lateの3つ全て) — pace_ratioとセグメントavgに含むが、個別のexpanding_statsは将来検討

---

*Phase: 36-Feature Computation*
*Context gathered: 2026-05-19*
