# Phase 7: Ensemble Enhancement - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

3モデルスタッキング(LightGBM+XGBoost+CatBoost)の多様性を強制するハイパーパラメータ最適化・early stopping・特徴量サブセット分割を実装し、予測精度を最大化する。

**In scope (from ROADMAP.md):**
- ENS-01: 各ベースモデルに異なるハイパーパラメータ(lr, depth, rounds)を設定し、モデル間の多様性を確保
- ENS-02: 各ベースモデルにバリデーションベースのearly stoppingを追加し、過学習を防止
- ENS-03: feature_fraction/colsample_bytree/rsmで各モデルに異なる特徴量サブセットを与え、アンサンブル多様性を向上

**Out of scope:**
- Stage1 (AbilityModel Ranker) のスタッキング (ENS-04 — v2以降)
- ベッティング戦略の変更 (Kelly/RegimeDetector — v1.2以降)
- 複勝/ワイドモデルの変更
- 複雑メタラーナー(GBM/NN) — Ridgeが最適(REQUIREMENTS.md Out of Scope)

**Plans:** 1 plan
- 07-01: Ensemble stacking enhancement with forced diversity (ENS-01, ENS-02, ENS-03)

</domain>

<decisions>
## Implementation Decisions

### ハイパーパラメータ差別化 (ENS-01)
- **D-01:** Optuna個別最適化で各モデル(LightGBM/XGBoost/CatBoost)のハイパーパラメータを最適化。固定値ではなくデータ駆動で最適なパラメータを見つける
- **D-02:** 探索空間分離 — 各モデルに意図的に異なるパラメータ範囲を設定:
  - LightGBM: 浅い木(num_leaves 31-63)、中程度のlr
  - XGBoost: 中程度の深さ(max_depth 4-8)、高めのlr
  - CatBoost: 深い木(depth 6-10)、低めのlr
  これにより各モデルが異なる表現空間を学習し多様性が確保される
- **D-03:** チューニングはStackedEnsembleクラス内に完結させる。TrainingPipeline.run()への変更は最小限(use_ensemble=True時の動作のみ)。自己完結した設計で責務を明確化

### Early Stopping適用範囲 (ENS-02)
- **D-04:** 全フェーズ適用 — K-fold OOF内の各foldとfinalモデルの両方でvalidation-based early stoppingを実装。K-fold内の過学習も防止する包括的アプローチ
- **D-05:** K-fold OOF内の各foldで学習データを80/20に分割し、validationデータを確保。OOF予測はvalidation部のみを使用し、学習部はearly stoppingのmonitoringに使用
- **D-06:** stopping_roundsはメインパイプライン(WinTwoStageModel)と同じ100を採用。実績のある値で安定性を確保

### 特徴量サブセット分割 (ENS-03)
- **D-07:** Optunaチューニングにfeature_fraction(LightGBM)、colsample_bytree(XGBoost)、rsm(CatBoost)を含めて各モデルの最適比率を最適化。ハイパーパラメータ最適化と統合した効率的なアプローチ
- **D-08:** 探索範囲は0.3-0.9を想定。0.3未満は情報損失が大きすぎ、0.9以上は多様性効果が薄いため

### 多様性検証手法
- **D-09:** OOF予測のペアワイズ相関(LGB-XGB, LGB-CAT, XGB-CAT)を計算し、全ペアで<0.95を確認。Success Criteriaの「予測相関0.95未満」を直接検証
- **D-10:** feature importanceのSpearman順位相関も計算。モデルが「同じ特徴量に依存している」場合も検出し、真の多様性を評価
- **D-11:** 相関≥0.95またはimportance順位相関が高い場合(>0.8)は警告ログを出力。自動再調整は行わない(パラメータ再検討のトリガーとして使用)

### Claude's Discretion
- Optunaの具体的な試行回数(n_trials)とタイムアウト設定
- 各モデルの探索空間の具体的な範囲(lr範囲、depth範囲、rounds範囲)
- OOF内80/20分割の実装詳細(train_size=0.8)
- feature_fraction探索の具体的な範囲とステップ
- Ridgeメタラーナーのalpha値(デフォルト1.0のまま、チューニング不要)
- 多様性評価結果のログ出力フォーマット
- Optunaのobjective関数の設計(AUC vs logloss等)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### StackedEnsemble既存実装
- `src/models/stacked_ensemble.py` — StackedEnsembleクラス。K-fold OOF + Ridge メタラーナーの完全実装。Phase 7の主たる変更対象
- `src/domain/models.py` lines 212-227 — TwoStageConfig dataclass。既存HPデフォルト値(hit_lr=0.03, hit_leaves=15, hit_rounds=300)
- `src/domain/models.py` lines 229-250 — SubmodelSet dataclass。use_ensemble フラグとモデル格納

### パイプライン統合
- `src/pipelines/training_pipeline.py` lines 84-97 — TrainingPipelineV5.run(use_ensemble=bool)。エントリポイント
- `src/pipelines/training_pipeline.py` lines 281-506 — _train_submodel()。StackedEnsemble使用箇所(win_hit, place_hit)
- `src/db/model_loader.py` lines 445-472 — _load_hit_model()。StackedEnsembleのjoblibロード

### 既存チューニングインフラ
- `src/tuning/optuna_tuner.py` — OptunaTuner。既存のOptunaチューニング実装。パターン参照

### モデルハイパーパラメータ参照
- `src/models/two_stage_return_model.py` lines 181-204 — WinTwoStageModel.train_hit_model()。feature_fraction=0.7, early_stopping(stopping_rounds=100)の既存パターン
- `src/models/two_stage_return_model.py` lines 427-450 — PlaceTwoStageModel.train_hit_model()。同パターン

### 要件定義
- `.planning/REQUIREMENTS.md` — ENS-01, ENS-02, ENS-03の要件定義
- `.planning/ROADMAP.md` — Phase 7 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **StackedEnsemble** (`src/models/stacked_ensemble.py`): 既存の3モデルスタッキング実装。K-fold OOF + Ridge メタラーナーの基本構造は完成済み。Phase 7ではこのクラスを拡張(HP多様化、early stopping、feature_fraction追加)
- **OptunaTuner** (`src/tuning/optuna_tuner.py`): 既存のOptunaチューニング実装。objective関数設計、MLflowログ、study作成パターンを参照
- **WinTwoStageModel.train_hit_model()** (`two_stage_return_model.py:181`): early_stopping(stopping_rounds=100) + feature_fraction=0.7 の実績あるパターン。StackedEnsembleの各foldでも同じパターンを採用

### Established Patterns
- **lgb.train + callbacks パターン**: LightGBMのearly stoppingはcallbacks=[lgb.early_stopping(stopping_rounds=N)]で実装。StackedEnsembleの_train_lgbm_fold/fullに追加
- **XGBoost early stopping**: xgb.trainのevals引数 + early_stopping_roundsパラメータ
- **CatBoost early stopping**: CatBoostClassifierのearly_stopping_roundsパラメータ + eval_set
- **Optuna objective パターン**: def objective(trial) → trial.suggest_* → モデル学習 → 評価指標返却。OptunaTunerで確立済み
- **joblib保存/ロード**: StackedEnsembleはjoblib.dump/load (.joblib)。TrainingPipeline._save_models_local()とModelLoader._load_hit_model()で対応済み

### Integration Points
- **StackedEnsemble.train()** (`stacked_ensemble.py:41`): 主たる変更対象。Optunaチューニングの追加、early stoppingの追加、feature_fractionの追加
- **StackedEnsemble._train_lgbm_fold/full** (`stacked_ensemble.py:115-130`): HP固定→Optuna最適化値に変更、early stopping callback追加、feature_fraction追加
- **StackedEnsemble._train_xgb_fold/full** (`stacked_ensemble.py:133-153`): 同様の変更。colsample_bytree, early_stopping_rounds追加
- **StackedEnsemble._train_cat_fold/full** (`stacked_ensemble.py:156-177`): 同様の変更。rsm, early_stopping_rounds追加
- **TrainingPipelineV5._train_submodel()** (`training_pipeline.py:282`): StackedEnsemble呼び出し箇所。HPチューニング結果の受け渡し方法を検討

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・精度を最優先
- Optuna + 探索空間分離はML ensemble stackingのstate-of-the-art。学術論文(Nguyen et al. 2024等)でも推奨される手法
- 多様性検証はOOF予測相関 + feature importance相関の二重チェックが最も堅牢
- Ridgeメタラーナーは3特徴量(各モデル予測)に対して最適(REQUIREMENTS.mdで確認済み)。変更不要
- K-fold OOF内のearly stoppingは、各foldでの過学習を防止し、OOF予測の品質を向上させる重要な改善

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 7-Ensemble Enhancement*
*Context gathered: 2026-05-03*
