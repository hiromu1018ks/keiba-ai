---
phase: 07-ensemble-enhancement
reviewed: 2026-05-03T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - src/models/stacked_ensemble.py
  - tests/test_stacked_ensemble.py
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-05-03
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

2ファイル（`src/models/stacked_ensemble.py` 553行、`tests/test_stacked_ensemble.py` 351行）をstandard depthでレビューした。

StackedEnsembleにOptuna HP tuning、K-fold OOF、特徴量サブセット、多様性検証を追加する実装。全体的に構造は良好だが、**HPチューニングとK-fold OOFの間のデータリーク**（BLOCKER）を1件発見した。またデッドコード（`_cat_codes`）や、カテゴリエンコーディングの未知値処理などWARNINGレベルの問題が3件ある。

## Critical Issues

### CR-01: HPチューニングvalidationとK-fold OOF最終foldのデータリーク

**File:** `src/models/stacked_ensemble.py:176-179` および `72-89`
**Issue:** `_tune_hyperparams()` は `X_train` の最後の20% (`X_train.iloc[int(n*0.8):]`) をHPチューニング用validationとして使用する。その後、K-fold OOFの最後のfold（`n_folds=3`の場合、`val_start=int(n*0.75)`, `val_end=n`）のvalidationデータは、データの最後の25%をカバーする。HPチューニングのvalidation（最後の20%）とK-fold最終foldのvalidation（最後の25%）が完全に重なる。

結果として、HPはこのvalidation区間に過剰最適化され、同じデータでOOF予測を生成するため、OOF予測が過度に楽観的になる。これによりRidgeメタラーナーの学習が不正確になり、アンサンブルの汎化性能見積もりが偏向する。

**Fix:** HPチューニングをK-foldの外で独立したホールドアウトで行うか、K-fold OOF生成時にHPチューニングに使ったデータ区間をOOF対象から除外する。例えば:

```python
def _tune_hyperparams(
    self, X_train: pd.DataFrame, y_train: pd.Series, num_threads: int,
) -> dict[str, dict[str, Any]]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(X_train)
    # K-fold OOF の最終fold validation区間を除外してHPチューニング
    # (n_folds=3の場合、最後の25%をOOF用に予約)
    oob_start = int(n * self.n_folds / (self.n_folds + 1))  # 最終foldのval_start
    split = int(oob_start * 0.8)  # HPチューニングはOOF対象外データ内で80/20
    X_t, y_t = X_train.iloc[:split], y_train.iloc[:split]
    X_v, y_v = X_train.iloc[split:oob_start], y_train.iloc[split:oob_start]
    # ... 以降同じ
```

または、HPチューニング自体を内側K-foldで実行し、OOF区間と完全に分離する。

## Warnings

### WR-01: `_cat_codes` がデッドコード — 学習された値がどこでも使用されない

**File:** `src/models/stacked_ensemble.py:44` および `135-139`
**Issue:** `self._cat_codes` は `__init__` で初期化され（行44）、`_learn_cat_codes()` で値が設定される（行139）が、この辞書はどこからも参照されない。`_encode_cats()` は `_cat_codes` を使わず、`X_out[col].cat.codes` を直接使用している。`_learn_cat_codes()` メソッド全体が不要なデッドコード。

**Fix:** `_cat_codes` フィールドと `_learn_cat_codes()` メソッドを削除する。または、`_encode_cats()` で `_cat_codes` を使って未知カテゴリを処理するように変更する（WR-02参照）。

### WR-02: `_encode_cats()` が未知カテゴリ値を -1 に変換し、予測時のモデル挙動が未定義

**File:** `src/models/stacked_ensemble.py:127-133`
**Issue:** `_encode_cats()` は `X_out[col].cat.codes` を直接使用する。pandasの`cat.codes`は、学習時に存在しなかったカテゴリ値に対して `-1` を返す。予測時（`predict()`）に新しいカテゴリ値が現れた場合、XGBoostとCatBoostは `-1` を特徴量値として受け取り、モデルの挙動が未定義になる可能性がある。

`_cat_codes` が正しく使われていれば、未知値を既知のカテゴリにマップする処理が可能だったが、現在はデッドコード（WR-01）。

**Fix:** `_encode_cats()` を修正し、`_cat_codes` を使って未知カテゴリを処理する:

```python
def _encode_cats(self, X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    for col in self.cat_cols:
        if col in X_out.columns and col in self._cat_codes:
            codes = self._cat_codes[col]
            X_out[col] = X_out[col].map(codes).fillna(-1).astype(float)
    return X_out
```

### WR-03: `test_exploration_space_separation` で同じtrialにsuggest関数を2回呼び出している

**File:** `tests/test_stacked_ensemble.py:177-180`
**Issue:** 行177のoptimize内で `_suggest_lgbm_params(t)` を呼び出し、行180でも同じ `study_lgb.best_trial` に対して再度 `_suggest_lgbm_params()` を呼び出している。Optunaは同じtrialで同じパラメータ名のsuggestを冪等に処理するが、これは意図が不明瞭で、テストがOptunaの内部挙動に依存している。

同様にXGBoost（行184-188）とCatBoost（行192-196）も同じパターン。

**Fix:** optimize内でsuggestの戻り値を保持し、best_trialからの再呼び出しを避ける:

```python
captured_params = {}
def objective(t):
    captured_params.update(ensemble._suggest_lgbm_params(t))
    return 0.0  # objective valueは使用しない

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective, n_trials=1)
lgb_params = captured_params
assert lgb_params["lgb_num_leaves"] <= 63
```

## Info

### IN-01: モジュールレベルでのxgboostインポートが欠落

**File:** `src/models/stacked_ensemble.py:117`
**Issue:** `predict()` メソッド内で `import xgboost as xgb` を行っている。同じインポートが `_eval_xgb`（行239）、`_train_xgb_fold`（行372）、`_train_xgb_full`（行406）でも繰り返されている。Pythonはモジュールキャッシュがあるためパフォーマンス問題はないが、各メソッドでの重複インポートは可読性を下げる。CatBoost、Optunaも同様。

**Fix:** ファイル先頭またはクラス定義直後に一度だけインポートする。

### IN-02: テストで実際のMLモデルを学習しており実行時間が長い

**File:** `tests/test_stacked_ensemble.py:67-76` および `328-336`
**Issue:** `test_optuna_tuning_produces_different_params`、`test_feature_fraction_in_lgbm_params`、`test_base_models_have_different_hp` などが実際のLightGBM/XGBoost/CatBoostを学習する。n=300-500の小データでもOptuna n_trials=3 + K-fold 3foldで9回のモデル学習が発生し、テスト全体の実行時間を押し上げる。

**Fix:** テストの高速化が必要な場合は、Optuna自体をmockするか、`n_trials=1` に減らすことを検討。ただし現在のテストは統合テストとしての価値があるため、CIでの実行時間とのトレードオフで判断。

---

_Reviewed: 2026-05-03_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
