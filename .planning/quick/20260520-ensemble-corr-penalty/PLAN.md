---
name: ensemble-corr-penalty
description: Optuna目的関数に予測相関ペナルティを追加してアンサンブル多様性を向上
created: 2026-05-20
---

# Ensemble Correlation Penalty

## 概要
`StackedEnsemble._tune_hyperparams()` で各モデルを逐次最適化する際、前のモデルの予測との相関をペナルティ項として目的関数に組み込む。

## 設計

### 目的関数
```
objective = AUC - λ × max(0, mean_corr - threshold)
```
- `λ` (corr_penalty_weight): ペナルティ強度。デフォルト0.5
- `threshold`: この相関まではペナルティなし。デフォルト0.85
- `mean_corr`: 前のモデルの予測との平均Pearson相関

### 流れ
1. LGB最適化（ペナルティなし、最初のモデル）
2. LGB best_paramsで参照モデルを(X_t, y_t)に学習 → X_v予測を`ref_preds_list`に格納
3. XGB最適化: 各trialでcorr(xgb_preds, ref_lgb_preds)を計算 → ペナルティ加算
4. XGB best_paramsで参照モデルを学習 → `ref_preds_list`に追加
5. CAT最適化: corr(cat_preds, ref_lgb)とcorr(cat_preds, ref_xgb)の平均 → ペナルティ加算

### 変更ファイル
- `src/models/stacked_ensemble.py` のみ

### 変更箇所

#### 1. `__init__` にパラメータ追加
```python
def __init__(self, cat_cols=None, n_folds=3, n_trials=30, corr_penalty_weight=0.5, corr_threshold=0.85):
    ...
    self.corr_penalty_weight = corr_penalty_weight
    self.corr_threshold = corr_threshold
```

#### 2. `_tune_hyperparams` の変更
現在のループを改修:
```python
best_params = {}
ref_preds_list: list[np.ndarray] = []

for model_name, suggest_fn, eval_fn, ref_train_fn in [
    ("lgbm", self._suggest_lgbm_params, self._eval_lgbm, self._train_ref_lgbm),
    ("xgb", self._suggest_xgb_params, self._eval_xgb, self._train_ref_xgb),
    ("cat", self._suggest_cat_params, self._eval_cat, self._train_ref_cat),
]:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial, fn=suggest_fn, tf=eval_fn: tf(
            trial, fn, X_t, y_t, X_v, y_v, num_threads,
            ref_preds_list=ref_preds_list,
            corr_penalty_weight=self.corr_penalty_weight,
            corr_threshold=self.corr_threshold,
        ),
        n_trials=self.n_trials,
    )
    best_params[model_name] = study.best_params

    # 参照モデルを学習して次モデルのペナルティ計算に使う
    ref_preds = ref_train_fn(X_t, y_t, X_v, num_threads, study.best_params)
    ref_preds_list.append(ref_preds)
```

#### 3. `_eval_lgbm` は変更不要（最初のモデル、ref_preds_list空なのでペナルティ0）

#### 4. `_eval_xgb` にペナルティ追加
```python
def _eval_xgb(self, trial, suggest_fn, X_t, y_t, X_v, y_v, num_threads,
              ref_preds_list=None, corr_penalty_weight=0.5, corr_threshold=0.85):
    ...  # 既存のAUC計算
    auc = float(roc_auc_score(y_v, preds))

    # 相関ペナルティ
    penalty = 0.0
    if ref_preds_list:
        corrs = [np.corrcoef(preds, ref)[0, 1] for ref in ref_preds_list]
        mean_corr = np.mean(corrs)
        penalty = corr_penalty_weight * max(0.0, mean_corr - corr_threshold)

    return auc - penalty
```

#### 5. `_eval_cat` も同様にペナルティ追加

#### 6. 参照モデル学習関数を追加
各モデルのbest_paramsで(X_t)に学習し、X_v予測を返す軽量関数:
- `_train_ref_lgbm(X_t, y_t, X_v, num_threads, best_params) -> np.ndarray`
- `_train_ref_xgb(...)`
- `_train_ref_cat(...)`

### テスト
- `python -m pytest tests/ -v -k "stacked_ensemble"`
- `ruff check src/models/stacked_ensemble.py`
