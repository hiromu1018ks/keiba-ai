# Phase 33: Gain per Depth Diagnostic - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

LightGBMの木構造をdepth別に分析し、Market/Fundamental/Categorical 3分類で特徴量のgain寄与率を可視化する診断ツール。暗黙的Two-Stage構造（上位depth=Market、下位depth=Fundamental）の仮説をデータで検証する。

**In scope:**
- GPD-01: LightGBM trees_to_dataframe() でdepth別gain寄与率を集計する機能
- GPD-02: Market/Fundamental/Categorical 3分類でdepth別シェアを可視化する機能（matplotlib PNG）
- GPD-03: StackedEnsemble内LightGBMモデルへのアクセスと分析機能
- GPD-04: 暗黙的Two-Stage構造の検証（Market Dominance Ratio + Fundamental Activation Depth指標）
- 特徴量3分類の明示的マッピングdict（単一ソース・オブ・トゥルース）
- JSON + Console + matplotlib PNG の3層出力
- CLIスクリプト (scripts/run_gpd.py)
- テスト作成（分類dict妥当性検証 + GPD計算ロジック + グラフ生成）

**Out of scope:**
- バックテスト実行 (Phase 34)
- IC評価・ベースライン比較 (Phase 34)
- 新特徴量の設計・実装 (Phase 31/32 complete)
- モデル構造の変更
- ETL拡張 (Phase 29 complete)
- XGBoost/CatBoostの木構造分析（将来拡張として文書化のみ）
- モデル再学習

</domain>

<decisions>
## Implementation Decisions

### 特徴量3分類の定義
- **D-01:** 特徴量分類は `gpd_diagnostics.py` 内に `FEATURE_CATEGORY_MAP: dict[str, str]` として明示的に定義。キー=特徴量名、値="market"|"fundamental"|"categorical"の3値
- **D-02:** テストで「全モデルのFEATURE_COLSの全特徴量がFEATURE_CATEGORY_MAPに登録されているか」を自動検証。未登録特徴量があればFAIL
- **D-03:** 3分類の境界基準:
  - **Market**: オッズ系(tanodds, implied_prob, popularity_rank等)、市場構造(rl_odds_dispersion, rl_log_odds_entropy等)、市場クロス整合性(Harville ratio等)、FLB/overround等の市場由来特徴量
  - **Fundamental**: 過去成績・血統・調教・馬体・フォームサイクル・コース適性・ペース適性・高オッズ的中パターン・EMA等の「馬の能力」由来特徴量
  - **Categorical**: 騎手・調教師・種牡馬・TE(ターゲットエンコーディング)・レース条件(距離/芝ダ/グレード/頭数)等の「カテゴリ」特徴量

### 分析対象モデルの範囲
- **D-04:** SubmodelSet内全LightGBM Boosterを分析対象とし、出力を階層化:
  - **主要分析**: AbilityModel(Stage1), WinTwoStage(hit+return), MarketModel, StackedEnsemble内LightGBM → GPD-04 Two-Stage仮説検証に直接関連
  - **詳細分析**: Place/Wide TwoStage, EV Correction(P+E), ConformalEV(q_low+q_high) → 補足的分析
  - **除外**: RegimeDetector, RaceQualityScreener → 分類/ルーティング目的で予測本質ではない
- **D-05:** LightGBMを主分析対象。XGBoost/CatBoostの分析は将来拡張として文書化のみ（今回は実装しない）
- **D-06:** コードはモデルタイプ別に抽象化し、将来XGBoost追加時の変更を最小化する設計

### 出力形式と可視化
- **D-07:** 3層出力: JSONレポート(data/gpd/gpd_report.json) + console_summary() + matplotlib PNGグラフ
- **D-08:** グラフはmatplotlib使用（既存依存関係）。モデル毎に個別PNG生成
- **D-09:** グラフ内容はClaudeの判断で最適設計。推奨: depth別3分類stacked bar + cumulative gain line

### Two-Stage仮説の検証方法
- **D-10:** 連続depth分析を採用（二分法や5段階の恣意的閾値なし）。全depthレベルのMarket/Fundamental/Categorical gainシェアを連続的に表示
- **D-11:** 仮説検証のための2つの定量的指標を自動計算:
  - **Market Dominance Ratio**: depth 1-3のMarket gain share vs depth 4+のMarket gain share
  - **Fundamental Activation Depth**: Fundamental gain shareがMarket gain shareを初めて超えるdepth
- **D-12:** 判定は人間が行う。指標を出力しconsole_summary()で視覚的に提示。自動ALERT/WARN/PASS判定は実装しない

### モジュール構成
- **D-13:** 新規 `src/models/gpd_diagnostics.py` を作成。既存診断モジュール(ev_diagnostics.py, drift_diagnostics.py)のパターンを踏襲:
  - 関数ベース（クラスではない）
  - JSON出力 + console_summary()
  - logging.getLogger(__name__)
- **D-14:** CLIスクリプト `scripts/run_gpd.py` を作成。学習済みモデルを読み込んでGPD診断を実行

### Claude's Discretion
- グラフの具体的なデザイン（stacked barの配置、色、サブプロット構成等）
- gpd_diagnostics.pyの内部関数構成
- FEATURE_CATEGORY_MAPの完全な内容（各特徴量の3分類マッピング）
- モデル名→Boosterアクセスパスの抽象化方法
- テストケースの具体的な設計
- JSON出力のスキーマ詳細
- モデル毎のPNGファイル命名規則

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 既存診断モジュール (パターン参照)
- `src/models/ev_diagnostics.py` — EV推定精度診断。関数ベース、JSON出力、console_summary()パターンの参照実装
- `src/models/drift_diagnostics.py` — ドリフト診断。ks_2samp/wasserstein_distance、JSON出力パターン

### モデルアクセス (Booster取得方法)
- `src/domain/models.py` — SubmodelSet (line 230) / TrainedModelsV5 (line 268)。全LightGBM Boosterのアクセスパス
- `src/models/stacked_ensemble.py` — StackedEnsemble。self.lgbm_model でLightGBM Boosterにアクセス
- `src/models/stage1_ability_model.py` — AbilityModel。self.models["turf"/"dirt"] でlgb.Boosterにアクセス
- `src/models/two_stage_return_model.py` — WinTwoStageModel / PlaceTwoStageModel。self.hit_model / self.return_model
- `src/models/market_model.py` — MarketModel。self.model でlgb.Boosterにアクセス
- `src/models/wide_two_stage_model.py` — WideTwoStageModel。self.hit_model / self.return_model
- `src/models/ev_correction_model.py` — EVCorrectionModel。self.p_correction_model / self.e_correction_model
- `src/models/conformal_ev_model.py` — ConformalEVModel。self.q_low_model / self.q_high_model

### FEATURE_COLS (分類対象)
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS (95列)
- `src/models/two_stage_return_model.py` — WinTwoStageModel / PlaceTwoStageModel FEATURE_COLS
- `src/models/ev_correction_model.py` — EVCorrectionModel / PlaceEVCorrectionModel FEATURE_COLS
- `src/models/conformal_ev_model.py` — ConformalEVModel.FEATURE_COLS
- `src/models/market_model.py` — MarketModel.FEATURE_COLS (7列)
- `src/models/stacked_ensemble.py` — StackedEnsemble.FEATURE_COLS

### 特徴量分析パターン (既存のfeature importance)
- `src/features/win_feature_analysis.py` — feature_importance("gain")、SHAP、permutation importance。dict[str, lgb.Booster]受け取りパターン
- `src/models/walk_forward_cv.py` — extract_feature_ranking() でgain-based feature ranking

### モデルロード (CLIスクリプト用)
- `src/db/model_loader.py` — ModelLoader。TrainedModelsV5のロード方法
- `src/pipelines/training_pipeline.py` — _save_models_local()。モデル保存形式

### 要件定義
- `.planning/REQUIREMENTS.md` §Gain per Depth Diagnostic — GPD-01~04

### Prior Phase Context
- `.planning/phases/31-race-level-aggregation-features/31-CONTEXT.md` — race-level特徴量(rl_*)の定義
- `.planning/phases/32-market-cross-consistency-features/32-CONTEXT.md` — market-cross特徴量(Harville ratio等)の定義

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/models/ev_diagnostics.py`: 関数ベース診断モジュールのテンプレート。`_compute_ece()`, `_brier_decomposition()` 等の関数構成、JSON出力、console_summary()パターン
- `src/models/drift_diagnostics.py`: ドリフト診断の参照。`_compute_column_stats()`, `_compare_columns()` 等のパターン
- `src/features/win_feature_analysis.py`: dict[str, lgb.Booster]を受け取りfeature_importanceを計算するパターン
- `lightgbm.Booster.trees_to_dataframe()`: Returns DataFrame with columns [tree_index, node_depth, node_index, left_child, right_child, parent_index, split_feature, split_gain, threshold, decision_type, missing_direction, missing_type, value, weight, count]
- `lightgbm.Booster.feature_importance(importance_type="gain")`: 既存のfeature importance取得方法

### Established Patterns
- 診断モジュールパターン: モジュールレベル定数 → 計算関数 → run_* オーケストレーション関数 → JSON出力 → console_summary()
- JSON出力パターン: `json.dump(result, f, indent=2, ensure_ascii=False)` + `_json_default()` for numpy/pandas types
- loggingパターン: `logging.getLogger(__name__)` でモジュールロガー取得
- CLIスクリプトパターン: argparse → ModelLoader → 診断関数 → JSON出力

### Integration Points
- `src/db/model_loader.py::ModelLoader`: CLIスクリプトからのモデルロード
- `src/domain/models.py::SubmodelSet`: 全LightGBM Boosterのアクセスエントリポイント
- `src/models/gpd_diagnostics.py` → `scripts/run_gpd.py`: CLIラッパーからの関数呼び出し
- `data/gpd/gpd_report.json`: GPD診断結果の保存先(新規)
- `data/gpd/`: matplotlib PNGグラフの保存先(新規)

</code_context>

<specifics>
## Specific Ideas

- trees_to_dataframe()の返すDataFrameのsplit_feature列に特徴量名が入る。これをFEATURE_CATEGORY_MAPで3分類にマッピングし、depth×categoryでgainを集計する
- node_depth列でdepth別にグループ化し、split_gainをcategory別に合計することでdepth別gainシェアが計算できる
- Market Dominance Ratio = (Market gain at depth 1-3) / (Total gain at depth 1-3) - (Market gain at depth 4+) / (Total gain at depth 4+). 正の値 = Marketがshallowで支配的
- Fundamental Activation Depthは「MarketからFundamentalへの遷移点」を示す。このdepthが小さいほどFundamental特徴量がモデル全体で重要
- 各モデルの木数(num_trees)と平均木深度をレポートに含めることで、モデル間の比較が可能
- レース条件系特徴量(距離/芝ダ/グレード)をCategoricalに分類する理由: LightGBMではcategorical扱いではなく、one-hot/multi-hot的な分岐になるが、本質的には「カテゴリ」情報

</specifics>

<deferred>
## Deferred Ideas

- XGBoost trees_to_dataframe() による同様のdepth別分析 — 将来フェーズで実装。コードは拡張可能設計にする
- CatBoost 木構造分析 — CatBoost API制約(plot_tree等のみ)のため将来検討
- GPD-05 多次元直交IC (win+wide+umaren同時直交化) — REQUIREMENTS.md Future要件

</deferred>

---

*Phase: 33-Gain per Depth Diagnostic*
*Context gathered: 2026-05-18*
