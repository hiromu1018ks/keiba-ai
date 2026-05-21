# Phase 6: Odds Deviation EV - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

モデル予測確率と市場オッズの乖離をEV信号としてモデルに直接組み込み、Conformal予測区間でベット選択の信頼性を最適化する。

**In scope (from ROADMAP.md):**
- ODDS-01: p_market/p_ability比率をStage2特徴量カラムとして追加し、ROIに直結するエッジ信号をモデルに学習させる
- ODDS-02: スタッキング出力がBenterGate→WinSelectionGateに正しく流れることを検証し、EV計算パイプラインの整合性を確保する
- ODDS-03: Conformal予測区間をEV区間に変換し、エッジの信頼性に基づいてベット選択を最適化する

**Out of scope:**
- LightGBM/XGBoost/CatBoostスタッキング (Phase 7)
- ベッティング戦略の変更 (Kelly/RegimeDetector — v1.2以降)
- 複勝/ワイドモデルの変更
- 新データ源の導入

**Plans:** 1 plan
- 06-01: Odds deviation EV features and pipeline integration (ODDS-01, ODDS-02, ODDS-03)

</domain>

<decisions>
## Implementation Decisions

### 乖離信号の拡張 (ODDS-01)
- **D-01:** `odds_to_ability_ratio`（既存、`p_market_win_adj / p_ability_win`）に加えて、`deviation_rank`（レース内乖離順位）と`deviation_zscore`（レース内標準化乖離）の2信号を追加。絶対乖離フラグは不要（LightGBMが自動閾値学習）
- **D-02:** `compute_odds_deviation_features(df)`をstandalone関数として作成。MarketModelの`predict_and_calc_error()`パターンに倣い、TrainingPipelineとRacePredictorの両方でAbilityModel後に呼び出す
- **D-03:** `deviation_rank`と`deviation_zscore`はレース内相対評価（`race_id`でgroupby）。`odds_to_ability_ratio`が絶対値、rank/z-scoreが相対値を担う構成
- **D-04:** 3信号をWinTwoStageModelのFEATURE_COLSに追加。odds_to_ability_ratioは既存、2信号を新規追加

### EV区間とベット選択 (ODDS-03)
- **D-05:** 既存`RobustConfidenceEstimator`（`robust_confidence_estimator.py`）を拡張。`predict_lower_bound()`を`predict_interval()`に拡張し、EV区間（下限＋上限）を計算
- **D-06:** EV下限（profitability filter）と区間幅（confidence filter）の両方を合成。`conformal_confidence_score`としてWinSelectionGateの`score()`に統合
- **D-07:** 90%と80%の2段階信頼水準を採用。80%区間（高信頼）と90%区間（最低基準）で段階的信頼性評価。nonconformity scoresの再利用で追加コストほぼゼロ
- **D-08:** conformal_confidence_scoreはEV下限と区間幅の合成指標。WinSelectionGateのquantile-bin scoring systemに新次元として追加

### パイプライン検証戦略 (ODDS-02)
- **D-09:** 三層テスト戦略を採用: (1) 単体テスト、(2) 統合テスト、(3) 数値的一貫性チェック
- **D-10:** 新規テストファイル`tests/test_odds_deviation.py`で乖離特徴量とEV区間のテストを集約
- **D-11:** `RobustConfidenceEstimator`拡張（predict_interval）のテストは既存テストファイルに追加
- **D-12:** パイプライン統合テストは既存`tests/test_race_predictor.py`に追加。RacePredictor.predict()の全フローをmockデータで検証
- **D-13:** 数値的一貫性チェック項目: 確率正規化（sum=1.0）、EV区間順序性（lower < point < upper）、NaN率検証

### Claude's Discretion
- `compute_odds_deviation_features()`の具体的な配置先ファイル（既存モジュール or 新規ユーティリティ）
- conformal_confidence_scoreの合成式の詳細（EV下限と区間幅の重み付け）
- WinSelectionGateへのconformal_confidence_score統合方法（既存quantile-binに追加 vs 新スコアリング次元）
- deviation_rank/z-scoreのNaN処理（デフォルトNaN、LightGBMネイティブ処理 — Phase 5 D-23と同一方針）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Stage2モデルと乖離特徴量
- `src/models/two_stage_return_model.py` lines 47-98 — WinTwoStageModel FEATURE_COLS。odds_to_ability_ratio（line 91）とodds_to_ability_ratio計算（lines 149-157）
- `src/models/market_model.py` lines 21-32, 82-217 — MarketModel。predict_and_calc_error()でsigned_log_error_win等を生成。get_stage2_features()はerror列のみ公開
- `src/models/ev_correction_model.py` lines 135-168 — EVCorrectionModel FEATURE_COLS。p_x_e_interaction等の市場歪曲特徴量

### BenterGateとWinSelectionGate
- `src/models/benter_combination.py` line 88 — BenterCombination。logit-space確率ブレンディング（alpha, beta, gamma）
- `src/models/win_selection_gate.py` lines 19, 434-481, 915, 936 — WinSelectionGateModel。build_win_selection_ev(), threshold grid search, score(), soft_pass_mask()
- `src/models/win_benter_gate.py` — WinBenterGate。EV correction後のBenter適用ポイント

### Conformal予測区間
- `src/models/robust_confidence_estimator.py` — RobustConfidenceEstimator。calibrate()（line 44）, predict_lower_bound()（line 96）。race-condition-dependent quantiles（per surface/distance_bin）
- `src/backtest/race_predictor.py` lines 51-219 — RacePredictor.predict()の推論フロー。全ステージの呼び出し順序

### データソース
- `src/db/readers.py` — load_odds_time_series_range()等のデータ読み込み
- `src/features/odds_dynamics_features.py` — compute_odds_dynamics()。Phase 5で追加したODTS-01/02の実装

### 既存テスト
- `tests/` — 既存テストファイル群（mock、DB不要）

### 要件定義
- `.planning/REQUIREMENTS.md` — ODDS-01, ODDS-02, ODDS-03の要件定義
- `.planning/ROADMAP.md` — Phase 6 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **WinTwoStageModel._prepare_features()** (`two_stage_return_model.py:146-157`): odds_to_ability_ratio計算の既存パターン。`p_market_win_adj / p_ability_win`。deviation特徴量も同じデータソースを利用
- **MarketModel.predict_and_calc_error()** (`market_model.py:82`): モデル出力に基づく特徴量追加のパターン。非raw-data特徴量の追加方法の参照実装
- **RobustConfidenceEstimator** (`robust_confidence_estimator.py`): conformal interval計算の完全な実装。nonconformity scores、CP quantiles、race-condition-dependent quantiles。predict_interval()拡張のベース
- **WinSelectionGate.quantile-bin scoring** (`win_selection_gate.py:738-811`): walk-forward OOF scoring system。conformal_confidence_scoreを新しいスコアリング次元として追加可能
- **MarketModel.get_stage2_features()** (`market_model.py:217`): error列のみStage2に公開する設計パターン

### Established Patterns
- **モデル後特徴量追加パターン**: MarketModel → error features、AbilityModel → p_ability_win。どちらもモデル出力に基づく特徴量をDataFrameに直接追加。deviation featuresは同じパターン
- **FeatureEngine → Pipeline呼び出しパターン**: FeatureEngine.build_all()はraw data featuresのみ。モデル依存特徴量はTrainingPipeline/RacePredictorで個別呼び出し
- **NaN-safe処理**: `pd.Series(np.nan, index=df.index, dtype=float)` でデフォルトNaN。LightGBMネイティブ処理（Phase 5 D-23）
- **groupby race_id パターン**: market_error_rank_in_race等のレース内相対特徴量は既に確立済み

### Integration Points
- **TrainingPipeline._train_submodel()** (`training_pipeline.py:282`): サブモデル学習フロー。AbilityModel後にcompute_odds_deviation_features()を追加
- **RacePredictor.predict()** (`race_predictor.py:89-141`): 推論チェーン。AbilityModel（step b）後にdeviation features追加、confidence.predict_lower_bound()（step k）をpredict_interval()に拡張
- **WinTwoStageModel.FEATURE_COLS** (`two_stage_return_model.py:47-98`): deviation_rank, deviation_zscoreを追加
- **WinSelectionGate.score()** (`win_selection_gate.py:915`): conformal_confidence_scoreをスコアリングに統合
- **RobustConfidenceEstimator** (`robust_confidence_estimator.py`): predict_interval()追加、既存predict_lower_bound()は内部でpredict_interval()を呼ぶ形にリファクタ可能

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「難易度は問わない」方針。品質優先で実装する
- 乖離信号は3信号構成: odds_to_ability_ratio（絶対値）、deviation_rank（順序）、deviation_zscore（標準化連続値）。直交性が高くLightGBMに適する
- EV区間は2段階信頼水準（80%/90%）でプロのベッティング運用に倣う。80%で高信頼判定、90%で最低基準判定
- 三層テスト戦略はMLパイプラインの品質保証のベストプラクティス。数値的不変量チェックが最も価値が高い

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 6-Odds Deviation EV*
*Context gathered: 2026-05-03*
