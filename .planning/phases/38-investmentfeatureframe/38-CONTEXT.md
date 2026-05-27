# Phase 38: InvestmentFeatureFrame - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

投資判断用統合特徴量フレーム (80-150列) を構築する。Phase 37のOOF健全性基盤の上に、モデル出力・市場データ・OOF予測を統合し、投資判断に特化した構造化特徴量フレームを生成する。

**In scope:**
- 全9カテゴリの投資特徴量: model_prob, market_prob, model_market_gap, race_relative, odds_band, late_odds, ability_form, course_pace, uncertainty
- デュアルモード対応: build_frame(df, mode=Literal["train", "infer"]) — 学習モードはOOF-safe列のみ、推論モードは本番列
- 出力スキーマ同一性: train/inferで同じ列名・列順・dtype
- Graceful degradation: 必須列は欠損時にfail-fast、オプション列は*_missing指標 + デフォルト値
- Parquetキャッシュ: source artifact identityをキーにしたsidecar manifest付きキャッシュ
- スキーマレジストリ: InvestmentFeatureSpec frozen dataclassによる型安全な定義
- 投資フレームartifact manifest: feature_version, schema_hash, source_artifact_hash, OOF health manifest path, builder_version
- 漏洩テスト: InvestmentFeatureFrame対象のPOST_RACE除外、OOF-safe train mode検証
- 新規パッケージ: src/investment/ (feature_frame.py, schema_registry.py, manifest.py, cache.py, leakage.py)

**Out of scope:**
- CAL-01~05 (人気帯キャリブレーション + レジーム伝播) → Phase 39
- レジーム依存動作 → v2.0全体でスコープ外
- ROI閾値ゲート → Phase 39/40統合検証信号
- Race-Level Ranker → Phase 40
- BT 2024全量実行 → Phase 39/40
- モデル再学習・ハイパーパラメータチューニング
- FeatureEngineの変更
- 既存モデルのFEATURE_COLS変更

</domain>

<decisions>
## Implementation Decisions

### フェーズ範囲定義
- **D-01:** Phase 38はInvestmentFeatureFrame構築のみ。CAL-01~05はPhase 39(MarketAwareWinCalibratorのセグメント条件付け)、RankerはPhase 40に移行
- **D-02:** レジーム伝播/regime-dependent calibrationはv2.0全体でスコープ外。過去実験で信頼性が確認されていないため
- **D-03:** ROI検証はPhase 38の成功基準ではない。Phase 39/40の統合検証信号として扱う
- **D-04:** 軽量スモークテストのみ許可(パイプラインインターフェース破損確認用)。ROI閾値なし、特徴量選択/係数調整への使用禁止

### 特徴量カテゴリ設計
- **D-05:** 全9カテゴリ実装: model_prob(8-12), market_prob(6-10), model_market_gap(10-16), race_relative(12-18), odds_band(6-10), late_odds(8-12), ability_form(15-25), course_pace(10-18), uncertainty(10-16)
- **D-06:** Required-core + optional-extension設計。初期ターゲット90-130列、上限150列
- **D-07:** 信号密度重視の配分: model_market_gap, race_relative, uncertainty, ability_formに多くの容量。market_prob, odds_bandはコンパクト
- **D-08:** FeatureEngine出力のパススルーではない。投資判断に特化した選別+派生のみ。AbilityModel/ConformalEVModelの全特徴量は流さない
- **D-09:** 各特徴量にメタデータ必須: category, source columns, train/infer source behavior, missing behavior, leakage classification, dtype, stable output name

### デュアルモード設計
- **D-10:** 単一API: `build_frame(df, mode=Literal["train", "infer"])` + thin convenience wrappers (build_train_frame, build_inference_frame)
- **D-11:** モード自動検出禁止。明示的mode引数必須。train誤用防止(p_win_pred等のin-sample列をtrain modeで拒否)
- **D-12:** 出力スキーマ同一性: 同じ列名・列順・dtype・feature_version・missing-indicator動作。モードはsource priorityのみ制御
- **D-13:** train mode: OOF-safe列(p_win_oof, p_win_final_oof)のみ使用。p_win_predはOOF明示マークな限り拒否
- **D-14:** infer mode: 本番列(p_win_pred, p_win_final)を使用
- **D-15:** テストでtrain/inferスキーマ同一性をアサート

### スキーマレジストリ
- **D-16:** InvestmentFeatureSpec frozen dataclass: name, category, dtype, train_sources, infer_sources, required, default_value, missing_indicator, leakage_class, description
- **D-17:** コードが真実の情報源(YAMLはドキュメント用ミラーとして後で生成可能)
- **D-18:** Builderはモード別にソース解決: train→train_sources, infer→infer_sources
- **D-19:** required featureのsource欠損時はfail-fast。optionalはdefault_value + *_missing indicator
- **D-20:** テスト検証: 全featureにspecあり、train sourceにin-sample-only列なし、train/infer同一スキーマ、required feature fail-fast

### キャッシュ・決定性
- **D-21:** Parquetキャッシュ + sidecar manifest JSON
- **D-22:** キャッシュパス: `data/features/investment_frame/{mode}/{feature_version}_{source_artifact_hash}_{schema_hash}.parquet`
- **D-23:** キャッシュキー: mode, feature_version, source_artifact_hash, source_schema_hash, output_schema_hash, source OOF health manifest path/hash (train mode), builder_version
- **D-24:** キャッシュ読込時はsidecar manifest + output schema_hash検証。失敗時は再計算
- **D-25:** メモリキャッシュは補助的のみ。Parquet+manifestが真実の情報源
- **D-26:** 決定性要件: 同一入力 + 同一builder_version + 同一feature_version → byte/value等価出力
- **D-27:** テスト: stable row order (race_id/umaban), stable column order

### バリデーション
- **D-28:** Phase 38成功基準: スキーマ正確性、OOF安全性、POST_RACE非混入、決定性、キャッシュ/manifest正確性、実行時間測定
- **D-29:** VAL-01(漏洩テスト)はInvestmentFeatureFrame対象にスコープ: POST_RACE列除外、train mode OOF-safe確認、train/infer同一スキーマ
- **D-30:** 新規manifest要件: feature_version, schema_hash, source_artifact_hash, source OOF health manifest path, builder_version, mode, generated_at
- **D-31:** VAL-02~05(芝IC, pop4-12 ratio, ROI, Turf conservative) → Phase 39/40に移行
- **D-32:** VAL-06(v1.8 manifest凍結) → 廃止。v2.0 artifact manifest要件に置き換え

### モジュール配置
- **D-33:** 新規パッケージ `src/investment/`: __init__.py, feature_frame.py (or frame_builder.py), schema_registry.py, manifest.py, cache.py, leakage.py
- **D-34:** 公開API: InvestmentFeatureFrameBuilder, InvestmentFeatureSpec, InvestmentFrameManifest, build_frame()
- **D-35:** FeatureEngineとモデルレイヤーから独立。FeatureEngineがモデル出力に依存しない構造を維持

### ROADMAP/REQUIREMENTS更新
- **D-36:** ROADMAP.md Phase 38の要件と成功基準を更新(現在TBD)
- **D-37:** REQUIREMENTS.md: CAL-01~05 → Phase 39に移行、レジーム除外を記録、VAL要件を再配分
- **D-38:** Phase 36.1.1の先送りROI検証はPhase 39/40統合検証に移行または廃止

### Claude's Discretion
- 各カテゴリの具体的な特徴量選定(model_prob 8-12列のどの列か)
- 派生特徴量の計算式(race-relative, uncertainty等)
- キャッシュinvalidationの具体ロジック
- sidecar manifest JSONの完全スキーマ
- テストケースの設計詳細
- ビルダーの内部アーキテクチャ(パイプライン化、バッチ処理等)
- frame_builder.py vs feature_frame.py のファイル名
- OOF health manifest との統合インターフェース
- Phase 37のOOFHealthValidatorとの接続方法

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 37成果物 (直前フェーズ)
- `src/validation/oof_health_validator.py` — OOFHealthValidator, ValidationResult, load_validated_oof。Phase 38のtrain modeはvalidated OOF artifactsを消費
- `src/validation/__init__.py` — OOF_PREDICTIONS_PROFILE, WIN_SELECTION_OOF_PROFILE。train mode source artifactのprofile定義
- `.planning/phases/37-ev-calibration-layers/37-CONTEXT.md` — OOF health基盤の設計決定
- `.planning/phases/37-ev-calibration-layers/37-VERIFICATION.md` — Phase 37完了確認

### モデル出力 (投資特徴量のソース)
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS (144列)。model_prob/ability_formカテゴリのソース
- `src/models/two_stage_return_model.py` — WinTwoStageModel/PlaceTwoStageModel FEATURE_COLS。model_probカテゴリのソース
- `src/models/ev_correction_model.py` — EVCorrectionModel/PlaceEVCorrectionModel (59列)。uncertaintyカテゴリのソース
- `src/models/market_model.py` — MarketModel.FEATURE_COLS (20列)。market_probカテゴリのソース
- `src/models/conformal_ev_model.py` — ConformalEVModel.FEATURE_COLS (165列)。uncertaintyカテゴリのソース
- `src/models/regime_detector.py` — RegimeDetector.FEATURE_COLS (44列)。参考(レジームはv2.0スコープ外)
- `src/models/race_quality_screener.py` — RaceQualityScreener.FEATURE_COLS (40列)。race_relativeカテゴリのソース

### バックテストパイプライン (推論パス)
- `src/backtest/race_predictor.py` — RacePredictor.predict()。推論モードでのsource columns (p_win_pred等)
- `src/backtest/engine.py` — BacktestEngine。hist_df_allマージ箇所、regime_state検出ループ
- `src/betting/orchestrator.py` — BettingOrchestrator。build_features()推論パス

### 学習パイプライン (学習パス)
- `src/pipelines/training_pipeline.py` — TrainingPipeline。OOF予測生成、fit_ev_calibration()。train mode source columns

### 既存特徴量モジュール (派生元)
- `src/features/feature_engine.py` — FeatureEngine.build_all()。投資特徴量のソース特徴量
- `src/features/horse_history_features.py` — 履歴特徴量(144列中の多く)。ability_form/course_paceカテゴリのソース
- `src/features/interaction_features.py` — 交互作用特徴量。race_relativeカテゴリのソース

### データ・ドメイン
- `src/domain/types.py` — POST_RACE_COLS (漏洩チェック対象)
- `src/db/readers.py` — DataReaders (Parquet読込)

### 安全性テスト
- `tests/test_post_race_leakage.py` — 3層CI漏洩テスト。VAL-01参考

### キャリブレーション参考
- `src/betting/odds_band_filter.py` — OddsBandFilter.BANDS。odds_bandカテゴリの閾値参考

### 要件・ロードマップ
- `.planning/ROADMAP.md` — Phase 38定義 (更新予定)
- `.planning/REQUIREMENTS.md` — VAL-01~06 (再配分予定)
- `.planning/PROJECT.md` — プロジェクト全体コンテキスト

### Prior Phase Context
- `.planning/phases/36.1.1-marketmodel-racequality-phase36-ev-tail-calibration/36.1.1-CONTEXT.md` — MarketModel/RaceQuality配線修正
- `.planning/phases/36.1-harontime-l4-laptime-feature-redesign-backtest-engine-hist-f/36.1-CONTEXT.md` — HaronTime/LapTime再設計

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `OOFHealthValidator` (validation/oof_health_validator.py): Phase 38 train modeがOOF artifactの健全性を確認する際に利用。load_validated_oof()でvalidated DataFrameを取得可能
- `fit_ev_calibration()` (training_pipeline.py): 5-fold OOF → Isotonic + odds-band scales。train modeのOOF予測ソース
- `POST_RACE_COLS` (domain/types.py): 41列のPOST_RACE定義。漏洩チェックの基準
- `AbilityModel.FEATURE_COLS` (144列): model_prob/ability_formカテゴリの主要ソース。全てを投資フレームに流すのではなく選別
- `MarketModel.FEATURE_COLS` (20列): market_probカテゴリのソース。コンパクトに設計済み
- `ConformalEVModel.FEATURE_COLS` (165列): uncertaintyカテゴリのソース。予測区間・キャリブレーション関連

### Established Patterns
- FEATURE_COLS選択: `[c for c in FEATURE_COLS if c in df.columns]` パターンで安全選択
- OOF予測生成: 5-fold expanding window → Isotonic calibration → band scaling
- ParquetStore: pyarrow predicate pushdown対応のParquet I/O
- Manifest生成: hashlib.sha256 + json.dumps(sort_keys=True, indent=2)で決定的JSON
- 3層CI漏洩検出: POST_RACE列のFEATURE_COLS混入→検出→fail

### Integration Points
- `src/investment/feature_frame.py` → `src/validation/oof_health_validator.py` (train mode OOF validation)
- `src/investment/feature_frame.py` → `src/models/*.py` (FEATURE_COLS reference, NOT import)
- `src/investment/feature_frame.py` → `src/features/feature_engine.py` (source feature selection)
- `src/investment/cache.py` → `src/db/parquet_store.py` (Parquet I/O pattern)
- `src/investment/manifest.py` → Phase 37 manifest format (artifact_version, schema_hash等のフォーマット踏襲)
- `src/investment/leakage.py` → `src/domain/types.py` (POST_RACE_COLS reference)

</code_context>

<specifics>
## Specific Ideas

- model_probカテゴリ: p_win_oof(infer=p_win_pred), calibrated_ev, ev_correction, isotonic_calibrated等
- market_probカテゴリ: implied_prob, odds_skewness, overround, market_entropy, popularity_rank等(MarketModel FEATURESから選別)
- model_market_gapカテゴリ: logit(p_model) - logit(p_market), devition_rank, devition_zscore, edge等
- race_relativeカテゴリ: 各特徴量のrace内rank(pct), top3_gap, field_dispersion等
- odds_bandカテゴリ: odds_band_id, band_median_ev, band_count等
- late_oddsカテゴリ: odds_acceleration, direction_consistency, late_money_ratio等
- ability_formカテゴリ: ability_score, form_trend, closing_index, blood_wr等(AbilityModel FEATURESから選別)
- course_paceカテゴリ: pace_ratio, closing_speed_ratio, haron_race_gap等(Phase 36/36.1特徴量)
- uncertaintyカテゴリ: conformal_lower, conformal_upper, interval_width, calibration_residual等

- Graceful degradation例: conformal_lower/upperが欠損 → default=NaN + conformal_missing=1。必須列(model_prob)が欠損 → ValueError
- キャッシュsidecar例: `{"feature_version": "v2.0", "schema_hash": "abc123...", "source_artifact_hash": "def456...", "mode": "train", "generated_at": "2026-05-27T18:00:00Z"}`

</specifics>

<deferred>
## Deferred Ideas

- 人気帯キャリブレーション (CAL-01~05) → Phase 39 MarketAwareWinCalibratorのセグメント条件付けとして実装
- レジーム伝播 → v2.0全体でスコープ外(信頼性未確認)
- Race-Level Ranker → Phase 40
- BT 2024 ROI検証 (VAL-04) → Phase 39/40統合検証信号
- 芝IC b_difference確認 (VAL-02) → Phase 39/40診断
- 芝pop 4-12 ratio (VAL-03) → Phase 39/40 shadow comparison
- Turf conservative ROI (VAL-05) → 廃止またはPhase 39/40 non-regime baseline
- v1.8 Manifest凍結 (VAL-06) → 廃止、v2.0 artifact manifestに置き換え
- Phase 36.1.1先送りROI >= 97.8%確認 → Phase 39/40統合検証または廃止

</deferred>

---

*Phase: 38-InvestmentFeatureFrame*
*Context gathered: 2026-05-27*
