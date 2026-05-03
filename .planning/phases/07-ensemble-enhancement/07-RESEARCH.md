# Phase 7: Ensemble Enhancement - Research

**Researched:** 2026-05-03
**Domain:** ML Ensemble Stacking (LightGBM + XGBoost + CatBoost) with Optuna hyperparameter optimization
**Confidence:** HIGH

## Summary

Phase 7 は既存の `StackedEnsemble` クラス (`src/models/stacked_ensemble.py`) を拡張し、3つのGBMベースモデル間の多様性を強制する。現在の実装は全モデルが同一ハイパーパラメータ(lr=0.03, rounds=300)を使用しており、モデル間の予測相関が高く、スタッキングの恩恵が限定的になっている可能性がある。

主要な変更は3点: (1) Optunaによる個別ハイパーパラメータ最適化と探索空間分離、(2) K-fold OOF内の各fold + finalモデルでのearly stopping、(3) feature_fraction/colsample_bytree/rsm による特徴量サブセット分割。全ての変更はStackedEnsembleクラス内に完結し、TrainingPipelineへの変更は最小限で済む。

**Primary recommendation:** Optuna探索空間を意図的に分離し(LightGBM=浅い木、XGBoost=中深さ、CatBoost=深い木)、各モデルが異なる表現空間を学習する構造を確立する。多様性検証はOOF予測相関 + feature importance順位相関の二重チェックで行う。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Optuna個別最適化で各モデル(LightGBM/XGBoost/CatBoost)のハイパーパラメータを最適化
- **D-02:** 探索空間分離 — LightGBM: 浅い木(num_leaves 31-63)、中程度のlr / XGBoost: 中程度の深さ(max_depth 4-8)、高めのlr / CatBoost: 深い木(depth 6-10)、低めのlr
- **D-03:** チューニングはStackedEnsembleクラス内に完結。TrainingPipeline.run()への変更は最小限
- **D-04:** 全フェーズ適用 — K-fold OOF内の各foldとfinalモデルの両方でvalidation-based early stopping
- **D-05:** K-fold OOF内の各foldで学習データを80/20に分割し、validationデータを確保
- **D-06:** stopping_rounds=100 (WinTwoStageModelと同じ実績値)
- **D-07:** Optunaチューニングにfeature_fraction/colsample_bytree/rsmを含めて各モデルの最適比率を最適化
- **D-08:** 探索範囲0.3-0.9 (0.3未満は情報損失、0.9以上は多様性効果薄)
- **D-09:** OOF予測のペアワイズ相関で全ペア<0.95を確認
- **D-10:** feature importanceのSpearman順位相関も計算
- **D-11:** 相関>=0.95またはimportance順位相関>0.8の場合は警告ログ出力(自動再調整なし)

### Claude's Discretion
- Optunaの具体的な試行回数(n_trials)とタイムアウト設定
- 各モデルの探索空間の具体的な範囲(lr範囲、depth範囲、rounds範囲)
- OOF内80/20分割の実装詳細(train_size=0.8)
- feature_fraction探索の具体的な範囲とステップ
- Ridgeメタラーナーのalpha値(デフォルト1.0のまま、チューニング不要)
- 多様性評価結果のログ出力フォーマット
- Optunaのobjective関数の設計(AUC vs logloss等)

### Deferred Ideas (OUT OF SCOPE)
- Stage1 (AbilityModel Ranker) のスタッキング (ENS-04 — v2以降)
- 複雑メタラーナー(GBM/NN) — Ridgeが最適
- sklearn StackingClassifier — ネイティブブースティングAPIとPIT安全フォールドに非対応
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENS-01 | 各ベースモデルに異なるHP(lr, depth, rounds)を設定し多様性を確保 | Optuna探索空間分離パターン(下記Architecture Patterns参照)。各モデルに異なるパラメータ範囲をtrial.suggest_*で設定 |
| ENS-02 | 各ベースモデルにバリデーションベースのearly stoppingを追加 | LightGBM: lgb.early_stopping(stopping_rounds=100), XGBoost: early_stopping_rounds=100, CatBoost: early_stopping_rounds=100。K-fold内80/20分割でvalid確保 |
| ENS-03 | feature_fraction等で各モデルに異なる特徴量サブセット | LightGBM: feature_fraction, XGBoost: colsample_bytree, CatBoost: rsm。Optunaで0.3-0.9を最適化 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| HP最適化(Optuna) | StackedEnsembleクラス | TrainingPipeline(呼び出しのみ) | D-03: チューニングはStackedEnsemble内に完結 |
| Early stopping | StackedEnsemble._train_*_fold/full | - | K-fold OOF + finalモデルの両方に適用(D-04) |
| 特徴量サブセット | StackedEnsemble._train_*_fold/full | - | 各モデルのparamsにfeature_fraction系を追加 |
| 多様性検証 | StackedEnsemble.train() | - | OOF予測相関 + importance順位相関の計算・ログ |
| モデル保存/ロード | TrainingPipeline / ModelLoader | - | 変更不要(joblib.dump/loadで対応済み) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | GBMベースモデル1 | [VERIFIED: pip list] プロジェクト既存。lgb.early_stopping() callback対応済み |
| xgboost | 3.2.0 | GBMベースモデル2 | [VERIFIED: pip list] プロジェクト既存。xgb.train()でearly_stopping_rounds対応 |
| catboost | 1.2.10 | GBMベースモデル3 | [VERIFIED: pip list] プロジェクト既存。rsm + early_stopping_rounds対応確認済み |
| optuna | 4.8.0 | HP最適化 | [VERIFIED: pip list] プロジェクト既存。create_study + trial.suggest_*パターン確立済み |
| scikit-learn | 1.8.0 | Ridge メタラーナー | [VERIFIED: pip list] sklearn.linear_model.Ridge |
| numpy | (既存) | OOF行列・相関計算 | np.corrcoef, scipy.stats.spearmanr |
| scipy | (既存) | Spearman順位相関 | scipy.stats.spearmanr(多様性検証 D-10) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logging | (stdlib) | 多様性評価ログ | D-11: 警告ログ出力 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Optuna個別最適化 | 固定パラメータ手動設定 | 手動は簡単だが、データ駆動でない。Optunaの方が最適値を発見しやすい |
| Spearman相関 | Kendall相関 | Spearmanの方が一般的で解釈しやすい。3特徴量程度では差は小さい |

**Installation:**
追加インストール不要。全ライブラリが既にインストール済み。

## Architecture Patterns

### System Architecture Diagram

```
TrainingPipeline._train_submodel(use_ensemble=True)
    |
    v
StackedEnsemble.train(X_train, y_train, X_valid, y_valid)
    |
    +--> [Optuna Tuning Phase]
    |    |  study = optuna.create_study(direction="maximize")
    |    |  3つのobjective関数(lgb_obj, xgb_obj, cat_obj)を別々に最適化
    |    |  --> best_params_lgb, best_params_xgb, best_params_cat
    |
    +--> [K-fold OOF Phase] (n_folds=3, expanding window)
    |    |
    |    +--> Fold i: train[:val_start] 80/20 split --> valid for early stopping
    |    |    |  _train_lgbm_fold(best_params_lgb, early_stopping=100)
    |    |    |  _train_xgb_fold(best_params_xgb, early_stopping=100)
    |    |    |  _train_cat_fold(best_params_cat, early_stopping=100)
    |    |    v
    |    |    oof_preds[val_start:val_end] = [p_lgb, p_xgb, p_cat]
    |    |
    |    +--> [Diversity Check]
    |         pairwise_corr = np.corrcoef(oof_preds)
    |         importance_corr = spearmanr(importance_pairs)
    |         LOG warnings if corr >= 0.95 or rank_corr > 0.8
    |
    +--> [Ridge Meta-Learner]
    |    Ridge(alpha=1.0).fit(oof_preds, y_train)
    |
    +--> [Final Models Phase] (train+valid全データ)
    |    _train_lgbm_full(best_params_lgb, early_stopping=100)
    |    _train_xgb_full(best_params_xgb, early_stopping=100)
    |    _train_cat_full(best_params_cat, early_stopping=100)
    |
    v
StackedEnsemble.predict(X)
    |  p_lgb, p_xgb, p_cat --> meta_model.predict() --> clipped [0,1]
```

### Recommended Project Structure
変更は `src/models/stacked_ensemble.py` のみ。新しいファイルは不要。
```
src/models/stacked_ensemble.py  # 全変更の主対象
tests/test_stacked_ensemble.py  # テスト拡張
```

### Pattern 1: Optuna探索空間分離
**What:** 各GBMに意図的に異なるパラメータ範囲を設定し、異なる表現空間を学習させる
**When to use:** StackedEnsemble.train()内でのHPチューニング
**Example:**
```python
# Source: [VERIFIED: optuna 4.8.0 API確認済み]
def _suggest_lgbm_params(self, trial: optuna.Trial) -> dict:
    """LightGBM: 浅い木 + 中程度のlr"""
    return {
        "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 63),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.05, log=True),
        "feature_fraction": trial.suggest_float("lgb_feat_frac", 0.3, 0.9),
    }

def _suggest_xgb_params(self, trial: optuna.Trial) -> dict:
    """XGBoost: 中程度の深さ + 高めのlr"""
    return {
        "max_depth": trial.suggest_int("xgb_max_depth", 4, 8),
        "learning_rate": trial.suggest_float("xgb_lr", 0.03, 0.1, log=True),
        "colsample_bytree": trial.suggest_float("xgb_col_sample", 0.3, 0.9),
    }

def _suggest_cat_params(self, trial: optuna.Trial) -> dict:
    """CatBoost: 深い木 + 低めのlr"""
    return {
        "depth": trial.suggest_int("cat_depth", 6, 10),
        "learning_rate": trial.suggest_float("cat_lr", 0.005, 0.03, log=True),
        "rsm": trial.suggest_float("cat_rsm", 0.3, 0.9),
    }
```

### Pattern 2: K-fold OOF内の80/20分割 + Early Stopping
**What:** K-fold内の各foldのtrain部をさらに80/20に分割し、validation確保
**When to use:** _train_*_fold() メソッド内
**Example:**
```python
# Source: [VERIFIED: LightGBM 4.6.0 lgb.early_stopping, XGBoost 3.2.0 early_stopping_rounds, CatBoost 1.2.10 early_stopping_rounds]
def _train_lgbm_fold(self, X_tr, y_tr, X_va, nt, params):
    # K-fold train部を80/20に分割 (D-05)
    n_tr = len(X_tr)
    es_split = int(n_tr * 0.8)
    X_t, y_t = X_tr.iloc[:es_split], y_tr.iloc[:es_split]
    X_v, y_v = X_tr.iloc[es_split:], y_tr.iloc[es_split:]

    train_data = lgb.Dataset(X_t, label=y_t)
    valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

    m = lgb.train(
        {"objective": "binary", "metric": "auc",
         "num_leaves": params["num_leaves"],
         "learning_rate": params["learning_rate"],
         "feature_fraction": params["feature_fraction"],
         "verbose": -1, "num_threads": nt},
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return m.predict(X_va)
```

### Pattern 3: 多様性検証
**What:** OOF予測相関 + feature importance順位相関の二重チェック
**When to use:** K-fold OOF完了後、Ridge学習前
**Example:**
```python
# Source: [VERIFIED: numpy/scipy API]
from scipy.stats import spearmanr

def _check_diversity(self, oof_preds: np.ndarray, y_train: pd.Series) -> None:
    """OOF予測の多様性を検証 (D-09, D-10, D-11)"""
    # ペアワイズ相関 (D-09)
    corr_matrix = np.corrcoef(oof_preds.T)  # shape: (3, 3)
    pairs = [(0,1,"LGB-XGB"), (0,2,"LGB-CAT"), (1,2,"XGB-CAT")]
    for i, j, name in pairs:
        c = corr_matrix[i, j]
        logger.info("OOF correlation %s: %.4f", name, c)
        if c >= 0.95:
            logger.warning("High prediction correlation %s: %.4f >= 0.95", name, c)
```

### Anti-Patterns to Avoid
- **全モデルに同じHPを使う:** 現在の実装(lr=0.03, rounds=300固定)はアンサンブル多様性を損なう。Optuna探索空間分離で解決
- **OptunaチューニングをK-fold内で毎回実行:** 計算コストがO(n_folds * n_trials * 3_models)になり非実用的。チューニングはtrain()の最初に1回だけ実行し、best_paramsを全fold/finalで使い回す
- **Early stoppingなしで高roundsを指定:** 過学習の原因。必ずvalid_sets + early_stoppingを併用
- **feature_fraction=1.0で全モデルが全特徴量を使用:** モデル間の多様性が減少。Optunaで0.3-0.9を探索

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HP最適化 | 手動グリッドサーチ | Optuna create_study + trial.suggest_* | 既存OptunaTunerパターンで確立済み。Bayesian optimizationが効率的 |
| Early stopping | 手動loss監視ループ | lgb.early_stopping(), xgb.train(early_stopping_rounds=), CatBoostClassifier(early_stopping_rounds=) | 各ライブラリのネイティブ実装が最適 |
| 相関計算 | 手動共分散計算 | np.corrcoef, scipy.stats.spearmanr | 数値安定性と正確性が保証される |
| 特徴量サンプリング | 手動カラム選択 | feature_fraction/colsample_bytree/rsmパラメータ | 各GBMのネイティブ機能が最適。木ごとにランダムサンプリング |

**Key insight:** 全てのGBMライブラリがfeature subsamplingをネイティブサポートしている。カスタム実装は不要。

## Common Pitfalls

### Pitfall 1: Optuna試行回数の過大設定
**What goes wrong:** n_trials=100で3モデル × 500rounds = 非常に長い学習時間
**Why it happens:** ベースモデルが3つあり、各trialでフル学習が必要
**How to avoid:** n_trialsは30-50程度に抑制。early_stoppingで各trialのroundsを制限
**Warning signs:** Optunaチューニングだけで数時間かかる

### Pitfall 2: K-fold内80/20分割時のデータ不足
**What goes wrong:** fold train部の80%しかearly stopping用の学習に使えず、20%がvalidになるが、元のfold train部が小さいとデータ不足に
**Why it happens:** 3-fold + 80/20分割 = 実質的にデータの~40%程度しか最初のfoldの学習に使われない
**How to avoid:** 現在のexpanding window方式を維持。最終foldはデータの大部分を使用するので問題なし
**Warning signs:** early_stoppingが即座に発動(1-2 roundで停止)

### Pitfall 3: XGBoost 3.xでのAPI変更
**What goes wrong:** xgb.train()のパラメータ名や挙動が2.xと異なる可能性
**Why it happens:** XGBoost 3.0でデフォルトパラメータが変更(colsample_bytree=0.8等)
**How to avoid:** 明示的に全パラメータを指定。デフォルトに依存しない
**Warning signs:** 予期しないモデル挙動、パラメータ警告

### Pitfall 4: CatBoost predict()がクラスラベルを返す
**What goes wrong:** predict()は確率ではなく0/1を返すため、メタラーナーへの入力が不正
**Why it happens:** CatBoost APIの仕様。predict_proba()が必要
**How to avoid:** 既存コード(line 95)で既に対応済み(predict_proba()使用)。このpitfallは既に解決済み
**Warning signs:** メタラーナーの出力が0/1の二値のみ

### Pitfall 5: Optunaロギングが大量に出力される
**What goes wrong:** n_trials=30でも各trialのログが出力され、コンソールが氾濫
**Why it happens:** OptunaのデフォルトログレベルがINFO
**How to avoid:** `optuna.logging.set_verbosity(optuna.logging.WARNING)` で抑制
**Warning signs:** コンソールログが読みにくい

## Code Examples

### Optunaチューニング統合 (StackedEnsemble.train内)
```python
# Source: [VERIFIED: Optuna 4.8.0 API] + [VERIFIED: pip list versions]
import optuna

class StackedEnsemble:
    def train(self, X_train, y_train, X_valid, y_valid, *, num_threads=0):
        # ... existing setup ...

        # --- Optuna Tuning Phase ---
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        best_params = self._tune_hyperparams(X_train, y_train, num_threads)

        # --- K-fold OOF with tuned params + early stopping ---
        for i in range(self.n_folds):
            # ... fold split (existing expanding window) ...
            oof_preds[val_start:val_end, 0] = self._train_lgbm_fold(
                X_tr, y_tr, X_va, num_threads, best_params["lgbm"])
            # ... xgb, cat同様 ...

        # --- Diversity Check ---
        self._check_diversity(oof_preds, y_train)

        # --- Ridge + Final models (existing) ---
```

### XGBoost Early Stopping (xgb.train)
```python
# Source: [VERIFIED: XGBoost 3.2.0 xgb.train signature]
import xgboost as xgb

def _train_xgb_fold(self, X_tr, y_tr, X_va, nt, params):
    X_tr_num = self._encode_cats(X_tr)
    X_va_num = self._encode_cats(X_va)

    # 80/20 split for validation (D-05)
    n = len(X_tr_num)
    es_split = int(n * 0.8)
    dtrain = xgb.DMatrix(X_tr_num.iloc[:es_split], label=y_tr.iloc[:es_split])
    dvalid = xgb.DMatrix(X_tr_num.iloc[es_split:], label=y_tr.iloc[es_split:])

    m = xgb.train(
        {"objective": "binary:logistic",
         "eval_metric": "auc",
         "max_depth": params["max_depth"],
         "learning_rate": params["learning_rate"],
         "colsample_bytree": params["colsample_bytree"],
         "nthread": nt},
        dtrain,
        num_boost_round=500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    return m.predict(xgb.DMatrix(X_va_num))
```

### CatBoost Early Stopping
```python
# Source: [VERIFIED: CatBoost 1.2.10 API - rsm + early_stopping_rounds confirmed]
from catboost import CatBoostClassifier

def _train_cat_fold(self, X_tr, y_tr, X_va, nt, params):
    X_tr_num = self._encode_cats(X_tr)
    X_va_num = self._encode_cats(X_va)

    # 80/20 split for validation (D-05)
    n = len(X_tr_num)
    es_split = int(n * 0.8)

    m = CatBoostClassifier(
        iterations=500,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        rsm=params["rsm"],
        thread_count=nt,
        verbose=0,
        early_stopping_rounds=100,
        eval_metric="AUC",
    )
    m.fit(
        X_tr_num.iloc[:es_split], y_tr.iloc[:es_split],
        eval_set=(X_tr_num.iloc[es_split:], y_tr.iloc[es_split:]),
    )
    return m.predict_proba(X_va_num)[:, 1]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LightGBM early_stopping_rounds param | lgb.early_stopping() callback | LightGBM 4.0.0 | [VERIFIED: Context7 + WebSearch] コールバックリストで渡す必要あり |
| 固定HP全モデル共通 | Optuna探索空間分離 | ベストプラクティス | モデル間多様性の強制 |
| 全特徴量使用 | feature_fraction系でサブセット | ランダムフォレスト由来 | Randomized GBMによる多様性向上 |

**Deprecated/outdated:**
- `lgb.train(..., early_stopping_rounds=N)`: LightGBM 4.0以降非推奨。`callbacks=[lgb.early_stopping(N)]`を使用 [CITED: lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Ridgeメタラーナーのalpha=1.0が3特徴量入力で最適 | Architecture | LOW — Ridgeは3特徴量で十分機能する。Claude's Discretionで変更可能 |
| A2 | n_trials=30-50が実用的なバランス | Architecture | MEDIUM — データサイズ依存。実際の実行時間で調整が必要 |
| A3 | K-fold 3fold + 80/20分割で十分なvalidデータが確保できる | Architecture | LOW — データ量が十分(数万行)であれば問題なし |
| A4 | Optuna objectiveの評価指標としてAUCが適切 | Architecture | LOW — 二値分類の標準指標。loglossも検討可能 |

## Open Questions (RESOLVED)

1. **Optuna n_trials の最適値**
   - RESOLVED: n_trials=30, タイムアウト60秒/trial。Claude's Discretionで調整可能

2. **Early stoppingのeval_metric**
   - RESOLVED: 既存パターンと同じ"auc"を採用。AUCは不均衡データでも安定

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| LightGBM | ベースモデル1 | Yes | 4.6.0 | - |
| XGBoost | ベースモデル2 | Yes | 3.2.0 | - |
| CatBoost | ベースモデル3 | Yes | 1.2.10 | - |
| Optuna | HP最適化 | Yes | 4.8.0 | - |
| scikit-learn | Ridge メタラーナー | Yes | 1.8.0 | - |
| scipy | Spearman相関 | Yes | (既存) | - |

**Missing dependencies with no fallback:**
- なし — 全依存関係がインストール済み

## Sources

### Primary (HIGH confidence)
- [VERIFIED: pip list] - lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.10, optuna 4.8.0, scikit-learn 1.8.0
- [VERIFIED: Context7] LightGBM early_stopping callback API - stopping_rounds, min_delta, verbose params confirmed
- [VERIFIED: Context7] XGBoost xgb.train() - early_stopping_rounds, evals, colsample_bytree params confirmed
- [VERIFIED: Context7] CatBoost CatBoostClassifier - early_stopping_rounds, rsm, eval_set params confirmed
- [VERIFIED: Context7] Optuna create_study + trial.suggest_* API confirmed
- [VERIFIED: Python runtime] CatBoost rsm + early_stopping_rounds params動作確認
- [VERIFIED: Python runtime] XGBoost 3.2.0 xgb.train() signature確認

### Secondary (MEDIUM confidence)
- [CITED: lightgbm.readthedocs.io] LightGBM 4.6.0 early_stopping callback docs
- [CITED: xgboost.readthedocs.io] XGBoost 3.0 migration guide - default colsample_bytree changed to 0.8
- [WebSearch verified] LightGBM 4.0以降のearly_stopping_rounds非推奨、callbacks方式への移行

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全ライブラリがインストール済み、API動作確認済み
- Architecture: HIGH - 既存コードベースの構造を完全に把握、変更箇所が明確
- Pitfalls: HIGH - 既存のOptunaTunerパターンとWinTwoStageModelのearly_stopping実績パターンがある

**Research date:** 2026-05-03
**Valid until:** 2026-06-03 (stable libraries, slow-moving domain)
