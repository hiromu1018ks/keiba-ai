# 学習パイプライン高速化 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** run_train.py の実行時間を 2時間+ → 30分以内に短縮する (精度完全維持)

**Architecture:** Phase 0 (プロファイリング) → Phase 1 (早期停止 + スレッド最適化) → Phase 2 (特徴量ベクトル化) → Phase 3 (並列化、必要に応じて) の段階的アプローチ。各Phase後に効果を計測し、30分目標達成で完了。

**Tech Stack:** Python 3.11, LightGBM 4.6.0, pandas 2.3.3, numpy 2.4.3

**Spec:** `docs/superpowers/specs/2026-04-04-training-speedup-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/utils/timing.py` | タイミング計測ユーティリティ (context manager) |
| Modify | `src/pipelines/training_pipeline.py` | プロファイリング、num_threads動的調整、並列化 |
| Modify | `src/features/feature_engine.py` | プロファイリング (build_all内5モジュール) |
| Modify | `src/features/horse_history_features.py` | プロファイリング + iterrowsベクトル化 |
| Modify | `src/features/odds_dynamics_features.py` | apply→aggベクトル化 |
| Modify | `src/features/market_bias_features.py` | compute_flb_slopeのapply最適化 |
| Modify | `src/models/wide_pair_builder.py` | combinations→merge最適化 |
| Modify | `src/models/stage1_ability_model.py` | 早期停止 (finalのみ、OOF除く) + predict修正 |
| Modify | `src/models/market_model.py` | 早期停止 + predict修正 |
| Modify | `src/models/two_stage_return_model.py` | Win/Place TwoStage 早期停止 + predict修正 |
| Modify | `src/models/wide_two_stage_model.py` | 早期停止 + predict修正 |
| Modify | `src/models/ev_correction_model.py` | 早期停止 + predict修正 |
| Modify | `src/models/race_quality_screener.py` | 早期停止 + predict修正 + num_threads追加 |
| Modify | `src/models/regime_detector.py` | 早期停止 + predict修正 + num_threads追加 |
| Modify | `src/models/place_ability_model.py` | num_threads動的調整 (sklearn API) |
| Modify | `tests/test_timing.py` | タイミングユーティリティのテスト (新規) |
| Modify | 各 `tests/test_*.py` | mock期待値の更新 |

---

## Task 1: Phase 0 — タイミング計測ユーティリティ

**Files:**
- Create: `src/utils/timing.py`
- Create: `tests/test_timing.py`

- [ ] **Step 1: テストを書く**

```python
# tests/test_timing.py
import logging
import time

from utils.timing import timed, TimingContext


def test_timed_context_manager_logs_elapsed(caplog):
    """TimingContext が [TIMING] プレフィクスで経過時間をログ出力する"""
    with caplog.at_level(logging.INFO):
        with TimingContext("my_step"):
            time.sleep(0.01)

    assert any("[TIMING] my_step:" in r.message for r in caplog.records)
    # 経過時間が0.01秒以上であることを確認
    timing_records = [r for r in caplog.records if "[TIMING]" in r.message]
    assert len(timing_records) == 1
    elapsed = float(timing_records[0].message.split(":")[-1].strip().rstrip("s"))
    assert elapsed >= 0.01


def test_timed_decorator_logs_elapsed(caplog):
    """@timed デコレータが関数実行時間をログ出力する"""
    @timed("decorated_func")
    def slow_func():
        time.sleep(0.01)
        return 42

    with caplog.at_level(logging.INFO):
        result = slow_func()

    assert result == 42
    assert any("[TIMING] decorated_func:" in r.message for r in caplog.records)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_timing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.timing'`

- [ ] **Step 3: 実装を書く**

```python
# src/utils/timing.py
"""軽量タイミング計測ユーティリティ。"""
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class TimingContext:
    """with文でコードブロックの実行時間を計測するコンテキストマネージャ。"""

    def __init__(self, step_name: str) -> None:
        self._step_name = step_name
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info("[TIMING] %s: %.1fs", self._step_name, elapsed)


def timed(step_name: str) -> Callable[..., Any]:
    """関数の実行時間を計測するデコレータ。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TimingContext(step_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_timing.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: コミット**

```bash
git add src/utils/timing.py tests/test_timing.py
git commit -m "feat: 軽量タイミング計測ユーティリティ (Phase 0)"
```

---

## Task 2: Phase 0 — プロファイリング計測ポイントの追加

**Files:**
- Modify: `src/pipelines/training_pipeline.py` (行71-287)
- Modify: `src/features/feature_engine.py` (行29-112)

- [ ] **Step 1: feature_engine.py に TimingContext を追加**

`build_all()` 内の5つのモジュール呼び出しを `TimingContext` でラップする:

```python
# src/features/feature_engine.py — build_all() 内
# ファイル先頭に import を追加:
from utils.timing import TimingContext

# 各モジュール呼び出しを TimingContext でラップ:
with TimingContext("build_all/intra_race"):
    compute_intra_race_features(df)       # 行84-86
with TimingContext("build_all/odds_dynamics"):
    compute_odds_dynamics(df, odds_ts_df) # 行88-90
with TimingContext("build_all/market_bias"):
    compute_market_bias(df)               # 行92-94
with TimingContext("build_all/difficulty"):
    compute_difficulty_score(df)          # 行96-98
# BloodlineFeatures は store is not None ガード内にあるため、
# ガードの内側で TimingContext をラップする:
if store is not None:
    with TimingContext("build_all/bloodline"):
        BloodlineFeatures(store).compute(df)  # 行101-106
```

- [ ] **Step 2: training_pipeline.py に TimingContext を追加**

`_train_submodel()` 内の各ステップを `TimingContext` でラップする:

```python
# src/pipelines/training_pipeline.py — _train_submodel() 内
from utils.timing import TimingContext

# 各ステップをラップ (合計21ポイント):
with TimingContext(f"{surface}/horse_history"):          # 行191
    hist.compute(df, surface)
with TimingContext(f"{surface}/add_race_transforms"):    # 行193
    HorseHistoryFeatures.add_race_transforms(df)
with TimingContext(f"{surface}/interaction"):             # 行198
    compute_interaction_features(df)
with TimingContext(f"{surface}/market_model"):            # 行209-211
    mm.train(df); mm.predict_and_calc_error(df)
with TimingContext(f"{surface}/ability_oof"):             # 行219-220
    ability.train_oof(df, n_folds=3)
with TimingContext(f"{surface}/place_ability_train"):     # 行227-228
    pam.train(df)
with TimingContext(f"{surface}/place_ability_predict"):   # 行229
    pam.predict(df)
with TimingContext(f"{surface}/win_hit"):                 # 行232
    wm.train_hit_model(df)
with TimingContext(f"{surface}/win_return"):              # 行233
    wm.train_return_model(df)
with TimingContext(f"{surface}/win_predict"):             # 行234-235
    wm.predict_ev(df)
with TimingContext(f"{surface}/jockey_ctx"):              # 行241-243
    JockeyContextFeatures.compute(df)
with TimingContext(f"{surface}/trainer_ctx"):             # 行245-247
    TrainerContextFeatures.compute(df)
with TimingContext(f"{surface}/ev_correction"):           # 行250-252
    evm.train(df); evm.correct_ev(df)
with TimingContext(f"{surface}/place_hit"):               # 行255
    plm.train_hit_model(df)
with TimingContext(f"{surface}/place_return"):            # 行256
    plm.train_return_model(df)
with TimingContext(f"{surface}/place_predict"):           # 行257-258
    plm.predict_ev(df)
with TimingContext(f"{surface}/wide_pair_build"):         # 行261
    wpb.build(df)
with TimingContext(f"{surface}/wide_hit"):                # 行264
    wdm.train_hit_model(df)
with TimingContext(f"{surface}/wide_return"):             # 行265
    wdm.train_return_model(df)
with TimingContext(f"{surface}/wide_predict"):            # 行266
    wdm.predict_score(df)
with TimingContext(f"{surface}/confidence"):              # 行268-276
    rce.calibrate(df)
```

`run()` 内にも追加 (5ポイント):

```python
with TimingContext("race_level_features"):
    self._build_race_level_features(...)
with TimingContext("quality_screener"):
    self.quality_screener.train(race_df)
with TimingContext("regime_detector"):
    self.regime_detector.train(race_df)
```

- [ ] **Step 3: テストが通ることを確認**

Run: `python -m pytest tests/test_feature_engine.py tests/test_training_pipeline.py -v`
Expected: PASS (mockがインポートを許容するため)

- [ ] **Step 4: コミット**

```bash
git add src/features/feature_engine.py src/pipelines/training_pipeline.py
git commit -m "feat: Phase 0 — 全ステップにタイミング計測を追加 (26ポイント)"
```

---

## Task 3: Phase 1 — 早期停止ヘルパー + MarketModel

**Files:**
- Modify: `src/models/market_model.py` (行40-120)
- Modify: `tests/test_market_model.py`

**背景**: MarketModel は `lgb.train` (native API) を使用。`predict_and_calc_error()` が `self.model.predict(features)` を呼び出す。

- [ ] **Step 1: テストを更新 — early stopping callback が追加されることを確認**

テストで `lgb.train` のモックが `callbacks` 引数を受け取るようにする。

```python
# tests/test_market_model.py の該当テスト
# モックの side_effect で lgb.train が callbacks パラメータを受け取ることを確認:
# lgb.train(params, dataset, num_boost_round=300, callbacks=...)
# モックされた lgb.Booster に best_iteration 属性を追加:
mock_booster.best_iteration = 150
```

- [ ] **Step 2: テストを実行して現在の失敗を確認**

Run: `python -m pytest tests/test_market_model.py -v`
Expected: テストが既存のモックと互換性があるため PASS の可能性。次ステップで壊す。

- [ ] **Step 3: MarketModel.train() に早期停止を追加**

```python
# src/models/market_model.py — train() メソッド (行54-66)
# 変更前:
self.model = lgb.train(
    {...params...},
    lgb.Dataset(features, label=target),
    num_boost_round=300,
)

# 変更後:
import numpy as np

n = len(features)
perm = np.random.RandomState(42).permutation(n)
split = int(n * 0.8)
train_idx, valid_idx = perm[:split], perm[split:]

train_data = lgb.Dataset(features.iloc[train_idx], label=target.iloc[train_idx])
valid_data = lgb.Dataset(features.iloc[valid_idx], label=target.iloc[valid_idx],
                         reference=train_data)

self.model = lgb.train(
    {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "num_threads": max(1, (os.cpu_count() or 4) // 2),
        "verbose": -1,
    },
    train_data,
    num_boost_round=300,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
```

- [ ] **Step 4: MarketModel.predict_and_calc_error() に num_iteration を追加**

```python
# src/models/market_model.py — predict_and_calc_error() (行82)
# 変更前:
raw_pred = self.model.predict(features)
# 変更後:
raw_pred = self.model.predict(features, num_iteration=self.model.best_iteration)
```

- [ ] **Step 5: テストを更新して通す**

モックの `lgb.train` が `valid_sets` と `callbacks` を受け取るようにする。
モックの booster に `best_iteration = 150` を設定。

Run: `python -m pytest tests/test_market_model.py -v`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/models/market_model.py tests/test_market_model.py
git commit -m "feat: MarketModel に早期停止 + predict num_iteration を追加"
```

---

## Task 4: Phase 1 — AbilityModel 早期停止 (finalのみ)

**Files:**
- Modify: `src/models/stage1_ability_model.py` (行71-153)
- Modify: `tests/test_stage1_ability.py`

**重要**: OOF fold モデルには早期停止を適用しない。`train()` メソッド (final model) のみに適用。

- [ ] **Step 1: AbilityModel.train() に早期停止を追加**

```python
# src/models/stage1_ability_model.py — train() (行71-111)
# lambdarank は group 情報が必要なため、分割時に group も分割する:

n = len(features)
perm = np.random.RandomState(42).permutation(n)
split = int(n * 0.8)
train_idx, valid_idx = perm[:split], perm[split:]

# group を features と同じインデックスで分割
train_groups = groups[train_idx]  # groups は numpy array
valid_groups = groups[valid_idx]

train_data = lgb.Dataset(features.iloc[train_idx], label=y.iloc[train_idx],
                         group=train_groups)
valid_data = lgb.Dataset(features.iloc[valid_idx], label=y.iloc[valid_idx],
                         group=valid_groups, reference=train_data)

self.models[key] = lgb.train(
    {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "num_threads": max(1, (os.cpu_count() or 4) // 2),
        "verbose": -1,
    },
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
```

- [ ] **Step 2: AbilityModel.add_ability_probs() に num_iteration を追加**

```python
# src/models/stage1_ability_model.py — add_ability_probs() (行140)
# 変更前:
raw_scores = self.models[key].predict(features)
# 変更後:
raw_scores = self.models[key].predict(features,
                                       num_iteration=self.models[key].best_iteration)
```

- [ ] **Step 3: AbilityModel.train_oof() に num_threads パラメータを追加**

`train_oof()` は内部で `AbilityModel()` の新しいインスタンスを作成して `train()` を呼び出す。
このインスタンスにも `num_threads` が伝播するよう、`train_oof()` にも `num_threads` パラメータを追加し、
内部の `AbilityModel()` コンストラクタに渡す。

**注意**: OOF fold モデルには早期停止を適用しない。`num_threads` のみ調整。

- [ ] **Step 4: テストを更新**

モックの `lgb.train` が `valid_sets`, `callbacks`, `group` を受け取るようにする。
モックの booster に `best_iteration` 属性を設定。

Run: `python -m pytest tests/test_stage1_ability.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/stage1_ability_model.py tests/test_stage1_ability.py
git commit -m "feat: AbilityModel final に早期停止を追加 (OOF除く)"
```

---

## Task 5: Phase 1 — Win/Place TwoStageModel 早期停止

**Files:**
- Modify: `src/models/two_stage_return_model.py` (行13-199)
- Modify: `tests/test_two_stage_return_model.py`

- [ ] **Step 1: 共通のバリデーション分割ヘルパーを追加 (ファイル内プライベート関数)**

```python
# src/models/two_stage_return_model.py の先頭付近に追加:
import numpy as np

def _train_valid_split(features: pd.DataFrame, label: pd.Series,
                       valid_ratio: float = 0.2, seed: int = 42):
    """学習データを train/valid にランダム分割して (train_data, valid_data) を返す。"""
    n = len(features)
    perm = np.random.RandomState(seed).permutation(n)
    split = int(n * (1 - valid_ratio))
    train_idx, valid_idx = perm[:split], perm[split:]

    train_data = lgb.Dataset(features.iloc[train_idx], label=label.iloc[train_idx])
    valid_data = lgb.Dataset(features.iloc[valid_idx], label=label.iloc[valid_idx],
                             reference=train_data)
    return train_data, valid_data
```

- [ ] **Step 2: WinTwoStageModel.train_hit_model() に早期停止を追加 (行62-80)**

```python
# train_data, valid_data = _train_valid_split(features, y) に置換
# callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] を追加
# valid_sets=[valid_data] を追加
```

- [ ] **Step 3: WinTwoStageModel.train_return_model() に早期停止を追加 (行82-109)**

同様。ただし return model は勝利馬のみのデータなのでサンプル数に注意。

- [ ] **Step 4: WinTwoStageModel.predict_ev() に num_iteration を追加 (行111-121)**

```python
# 行118-119:
p_hit = self.hit_model.predict(features, num_iteration=self.hit_model.best_iteration)
e_return = self.return_model.predict(features, num_iteration=self.return_model.best_iteration)
```

- [ ] **Step 5: PlaceTwoStageModel も同様に変更 (行124-199)**

`train_hit_model()`, `train_return_model()`, `predict_ev()` を同じパターンで変更。

- [ ] **Step 6: テストを更新して通す**

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: PASS

- [ ] **Step 7: コミット**

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py
git commit -m "feat: Win/Place TwoStageModel に早期停止を追加"
```

---

## Task 6: Phase 1 — WideTwoStageModel 早期停止

**Files:**
- Modify: `src/models/wide_two_stage_model.py` (行15-141)
- Modify: `tests/test_wide_two_stage_model.py`

- [ ] **Step 1: train_hit_model() に早期停止を追加 (行41-77)**

`_train_valid_split` を追加し、`valid_sets` と `callbacks` を追加。

- [ ] **Step 2: train_return_model() に早期停止を追加 (行79-118)**

同様。

- [ ] **Step 3: predict_score() に num_iteration を追加 (行120-141)**

```python
# 行134-135:
p_hit = self.hit_model.predict(features, num_iteration=self.hit_model.best_iteration)
e_return = self.return_model.predict(features, num_iteration=self.return_model.best_iteration)
```

- [ ] **Step 4: テストを更新**

Run: `python -m pytest tests/test_wide_two_stage_model.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/wide_two_stage_model.py tests/test_wide_two_stage_model.py
git commit -m "feat: WideTwoStageModel に早期停止を追加"
```

---

## Task 7: Phase 1 — EVCorrectionModel 早期停止

**Files:**
- Modify: `src/models/ev_correction_model.py` (行74-155)
- Modify: `tests/test_ev_correction.py`

**注意**: P correction model は `init_score` パラメータを使用。分割時にも `init_score` を維持する必要がある。

- [ ] **Step 1: train() の P correction モデルに早期停止を追加 (行89-102)**

```python
# init_score 付きのバリデーション分割:
n = len(features)
perm = np.random.RandomState(42).permutation(n)
split = int(n * 0.8)
train_idx, valid_idx = perm[:split], perm[split:]

train_data = lgb.Dataset(
    features.iloc[train_idx], label=y_p.iloc[train_idx],
    init_score=init_score[train_idx],
)
valid_data = lgb.Dataset(
    features.iloc[valid_idx], label=y_p.iloc[valid_idx],
    init_score=init_score[valid_idx],
    reference=train_data,
)
```

- [ ] **Step 2: train() の E correction モデルに早期停止を追加 (行114-130)**

同様。`weight` も分割する。

- [ ] **Step 3: correct_ev() の predict に num_iteration を追加 (行146, 150)**

```python
p_corrected = self.p_correction_model.predict(features,
    num_iteration=self.p_correction_model.best_iteration)
e_corrected = self.e_correction_model.predict(features,
    num_iteration=self.e_correction_model.best_iteration)
```

- [ ] **Step 4: テストを更新**

Run: `python -m pytest tests/test_ev_correction.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/ev_correction_model.py tests/test_ev_correction.py
git commit -m "feat: EVCorrectionModel に早期停止を追加"
```

---

## Task 8: Phase 1 — RaceQualityScreener + RegimeDetector 早期停止

**Files:**
- Modify: `src/models/race_quality_screener.py` (行86-110)
- Modify: `src/models/regime_detector.py` (行54-138)
- Modify: `tests/test_race_quality_screener.py`
- Modify: `tests/test_regime_detector.py`

**注意**: この2モデルは現在 `num_threads` が未設定。ついでに追加。

- [ ] **Step 1: RaceQualityScreener.train() に早期停止 + num_threads を追加**

```python
# src/models/race_quality_screener.py — train() (行86-102)
# 時系列ベースのバリデーション (最終20%):
n = len(features)
split = int(n * 0.8)
train_data = lgb.Dataset(features.iloc[:split], label=y.iloc[:split])
valid_data = lgb.Dataset(features.iloc[split:], label=y.iloc[split:],
                         reference=train_data)

self.model = lgb.train(
    {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "num_threads": max(1, (os.cpu_count() or 4) // 2),
        "verbose": -1,
    },
    train_data,
    num_boost_round=200,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)
```

- [ ] **Step 2: RaceQualityScreener の全predictに num_iteration を追加**

```python
# should_bet() (行109):
quality_score = self.model.predict(features, num_iteration=self.model.best_iteration)[0]

# calibrate_threshold() (行122付近) — 見落とし注意:
# このメソッドも self.model.predict(features) を呼び出すため、
# 同様に num_iteration=self.model.best_iteration を追加する必要がある
probs = self.model.predict(features, num_iteration=self.model.best_iteration)
```

- [ ] **Step 3: RegimeDetector も同様に変更**

```python
# 時系列ベース最終20%で分割
# num_threads を追加
# early_stopping(stopping_rounds=20) (100ラウンドなので短め)
# detect() の predict に num_iteration を追加
```

- [ ] **Step 4: テストを更新**

Run: `python -m pytest tests/test_race_quality_screener.py tests/test_regime_detector.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/race_quality_screener.py src/models/regime_detector.py \
        tests/test_race_quality_screener.py tests/test_regime_detector.py
git commit -m "feat: RaceQualityScreener/RegimeDetector に早期停止 + num_threads を追加"
```

---

## Task 9: Phase 1 — num_threads 動的調整

**Files:**
- Modify: `src/pipelines/training_pipeline.py` (行130)
- Modify: `src/models/stage1_ability_model.py` — `train()`, `train_oof()` のnum_threads
- Modify: `src/models/market_model.py` — `train()` のnum_threads
- Modify: `src/models/two_stage_return_model.py` — Win/Place `train_*()` のnum_threads
- Modify: `src/models/ev_correction_model.py` — `train()` のnum_threads
- Modify: `src/models/wide_two_stage_model.py` — `train_*()` のnum_threads
- Modify: `src/models/race_quality_screener.py` — `train()` のnum_threads (Task 8で追加済み)
- Modify: `src/models/regime_detector.py` — `train()` のnum_threads (Task 8で追加済み)
- Modify: `src/models/place_ability_model.py` — **`n_jobs`** (sklearn API、注意)
- Modify: 各 `tests/test_*.py` — モックシグネチャ更新

**注意**: このタスクは9ファイルにまたがるインターフェース変更。各モデルの `train()` メソッドシグネチャに
`num_threads` パラメータを追加し、全テストのモック期待値を更新する必要がある。

- [ ] **Step 1: _get_num_threads() ヘルパーを追加**

```python
# src/pipelines/training_pipeline.py に追加:
def _get_num_threads(parallel_workers: int = 1) -> int:
    """並列ワーカー数に応じて最適なスレッド数を返す。"""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count // (parallel_workers + 1))
```

- [ ] **Step 2: run() の ThreadPoolExecutor で並列数を渡す**

```python
# 行130付近:
with ThreadPoolExecutor(max_workers=2) as executor:
    num_threads = _get_num_threads(parallel_workers=2)
    futures = {
        executor.submit(self._train_submodel, s, df, num_threads=num_threads): s
        for s in surfaces_to_train
    }
```

- [ ] **Step 3: _train_submodel() が num_threads を受け取って各モデルに渡す**

各モデルの `train()` メソッドが `num_threads` パラメータを受け取るように変更。

- [ ] **Step 4: lgb.train (native API) モデルの num_threads を引数化**

以下の7ファイルの `train()` メソッドに `num_threads` パラメータを追加:

| ファイル | メソッド | 現在の値 | 変更 |
|---------|---------|---------|------|
| `stage1_ability_model.py` | `train()`, `train_oof()` | `max(1, (os.cpu_count() or 4) // 2)` | 引数化 |
| `market_model.py` | `train()` | 同上 | 引数化 |
| `two_stage_return_model.py` | Win/Place `train_hit/return()` | 同上 | 引数化 |
| `ev_correction_model.py` | `train()` | 同上 | 引数化 |
| `wide_two_stage_model.py` | `train_hit/return()` | 同上 | 引数化 |
| `race_quality_screener.py` | `train()` | なし (Task 8で追加) | 引数化 |
| `regime_detector.py` | `train()` | なし (Task 8で追加) | 引数化 |

```python
# 変更前:
"num_threads": max(1, (os.cpu_count() or 4) // 2),
# 変更後 (メソッド引数から取得):
"num_threads": num_threads,
```

- [ ] **Step 5: PlaceAbilityModel の n_jobs を引数化 (sklearn API)**

`PlaceAbilityModel` は **`lgb.LGBMClassifier`** (sklearn API) を使用するため、
`num_threads` ではなく **`n_jobs`** パラメータを使用する点に注意:

```python
# src/models/place_ability_model.py — train() (行104-117)
# 変更前:
n_jobs=max(1, (os.cpu_count() or 4) // 2),
# 変更後:
n_jobs=num_threads,  # _train_submodel() から受け取った値
```

- [ ] **Step 6: テストを更新 (全モックシグネチャ更新)**

各モデルのテストファイルで、`train()` メソッドのモックが `num_threads` 引数を
受け取るようにする。これには以下のテストファイルの更新が必要:

- `tests/test_stage1_ability.py` (train + train_oof)
- `tests/test_market_model.py`
- `tests/test_two_stage_return_model.py`
- `tests/test_ev_correction.py`
- `tests/test_wide_two_stage_model.py`
- `tests/test_race_quality_screener.py`
- `tests/test_regime_detector.py`
- `tests/test_place_ability_model.py` (n_jobs)
- `tests/test_training_pipeline.py` (_train_submodel)

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: コミット**

```bash
git add -A
git commit -m "feat: num_threads を並列数に応じて動的調整 (Phase 1b)"
```

---

## Task 10: Phase 2 — HorseHistoryFeatures ベクトル化 (最大のボトルネック)

**Files:**
- Modify: `src/features/horse_history_features.py` (行174-470)
- Modify: `tests/test_horse_history_features.py`

**これは最も複雑なタスク。行267の外側 iterrows() と行319-337の内側 iterrows() を両方ベクトル化する。**

- [ ] **Step 1: テストを書く — ベクトル化前後で同一の特徴量が生成されることを確認**

```python
# tests/test_horse_history_features.py に追加:
def test_vectorized_compute_matches_original():
    """ベクトル化後も従来と同一の特徴量値が得られることを確認するテスト。
    小規模なダミーデータで、iterrows版とベクトル版の出力を比較する。"""
    # ダミーデータを作成 (数頭の馬、数レース分)
    # compute() の結果の数値列を比較
    # 許容誤差: atol=1e-10
```

- [ ] **Step 2: 外側 iterrows() をベクトル化**

現在の `compute()` メソッドの行267-470を書き換え:

```python
# 戦略:
# 1. 全馬の過去レースを一つの DataFrame に結合
# 2. searchsorted で日付カットオフを一括取得
# 3. cumcount ベースで直近3走をフィルタ
# 4. groupby().agg() で集約特徴量を一括計算

# 具体的アプローチ:
# - horses (対象出走馬) の kettonum 一覧を取得
# - past_races 辞書から該当する過去レースを一括取得し、
#   searchsorted でカットオフインデックスを算出
# - 各馬の直近3走をフィルタ (cumcount)
# - 集約特徴量 (着順平均、タイムz-score等) を groupby().agg() で一括計算
# - 結果を horses DataFrame に merge
```

実装の詳細はコードの構造に依存するため、実装時に `horse_history_features.py` を注意深く読んで適切なベクトル化を行う。

- [ ] **Step 3: 内側 iterrows() (距離bin z-score) をベクトル化**

```python
# 行319-337のネストされたループを置換:
# FALLBACK_LEVELS を使用した階層フォールバックを保持
# groupby().transform() でベクトル化
# groupby().transform('count') でサンプル数を確認し、
# 閾値未満なら次レベルのフォールバックに進む
```

- [ ] **Step 4: テストを実行**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: PASS (特徴量値が同一)

- [ ] **Step 5: コミット**

```bash
git add src/features/horse_history_features.py tests/test_horse_history_features.py
git commit -m "perf: HorseHistoryFeatures の iterrows をベクトル化 (Phase 2a)"
```

---

## Task 11: Phase 2 — compute_odds_dynamics ベクトル化

**Files:**
- Modify: `src/features/odds_dynamics_features.py` (行22-140)
- Modify: `tests/test_odds_dynamics_features.py`

- [ ] **Step 1: テストを更新 — ベクトル化前後で同一の値が得られることを確認**

小規模なテストデータで `apply` 版と `agg` 版の結果を比較。

- [ ] **Step 2: _calc_velocity を agg ベースに置換**

```python
# 行87-101 の groupby().apply(_calc_velocity) を置換:
# 一次回帰係数を中間統計量で計算:
# slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)

stats = ts.groupby(["race_id", "umaban"]).agg(
    count=("tanodds", "count"),
    first_odds=("tanodds", "first"),
    last_odds=("tanodds", "last"),
)
# x = インデックス (0, 1, 2, ...)、y = tanodds
# 各グループの中間統計量を計算
```

- [ ] **Step 3: _calc_volatility を std() に置換**

```python
# groupby().apply(_calc_volatility) → groupby().std()
volatility = ts.groupby(["race_id", "umaban"])["tanodds"].std().reset_index()
```

- [ ] **Step 4: _get_mid_odds を last()/first() に置換**

```python
# groupby().apply(_get_mid_odds) → groupby().agg(first=, last=, nth)
```

- [ ] **Step 5: テストを実行**

Run: `python -m pytest tests/test_odds_dynamics_features.py -v`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/features/odds_dynamics_features.py tests/test_odds_dynamics_features.py
git commit -m "perf: compute_odds_dynamics の apply を agg にベクトル化 (Phase 2b)"
```

---

## Task 12: Phase 2 — WideJointPairBuilder + market_bias 最適化

**Files:**
- Modify: `src/models/wide_pair_builder.py` (行26-84)
- Modify: `src/features/market_bias_features.py` (行57-99)
- Modify: `tests/test_wide_pair_builder.py`
- Modify: `tests/test_market_bias_features.py`

- [ ] **Step 1: WideJointPairBuilder.build() を自己結合に置換**

```python
# src/models/wide_pair_builder.py — build() (行26-84)
# 行33の groupby ループ + 行63の combinations を置換:

# 全レース一括処理:
pairs = pd.merge(entry_df, entry_df, on="race_id", suffixes=("_1", "_2"))
pairs = pairs[pairs["umaban_1"] < pairs["umaban_2"]].copy()

# 列名を従来のフォーマットに調整
```

- [ ] **Step 2: compute_flb_slope() の apply をベクトル化**

```python
# src/features/market_bias_features.py — compute_flb_slope() (行93)
# groupby("race_id").apply(_race_flb) を置換:
# 各レースの polyfit を groupby().agg() で計算
```

- [ ] **Step 3: テストを実行**

Run: `python -m pytest tests/test_wide_pair_builder.py tests/test_market_bias_features.py -v`
Expected: PASS

- [ ] **Step 4: コミット**

```bash
git add src/models/wide_pair_builder.py src/features/market_bias_features.py \
        tests/test_wide_pair_builder.py tests/test_market_bias_features.py
git commit -m "perf: WidePairBuilder と market_bias をベクトル化 (Phase 2c)"
```

---

## Task 13: Phase 3 — 条件付き並列化 (Phase 1-2の効果に依存)

**Files:**
- Modify: `src/pipelines/training_pipeline.py`

**判断基準**: Phase 1-2 完了後に30分目標を達成していれば、このTaskはスキップ。

- [ ] **Step 1: Phase 1-2 完了後に `python scripts/run_train.py` を実行して所要時間を計測**

- [ ] **Step 2 (条件付き): 30分未満ならコミットして完了。30分以上なら以下を実装**

Win/Place/Wide Three TwoStageModel を並列実行:

```python
# _train_submodel() 内で、AbilityModel完了後:
with ThreadPoolExecutor(max_workers=3) as model_executor:
    model_executor.submit(self._train_win_pipeline, ...)
    model_executor.submit(self._train_place_pipeline, ...)
    model_executor.submit(self._train_wide_pipeline, ...)
```

- [ ] **Step 3: テスト・コミット**

---

## Task 14: 全体テスト + 最終効果確認

**Files:** なし (検証のみ)

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: リント・型チェック**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: PASS

- [ ] **Step 3: `run_train.py` を実行して実際のタイムブレークダウンを確認**

Run: `python scripts/run_train.py --start 20200101 --end 20231231`
Expected: [TIMING] ログから各フェーズの所要時間を確認。合計30分以内。

- [ ] **Step 4: 最終コミット**

```bash
git commit --allow-empty -m "perf: 学習パイプライン高速化完了 — Phase 0-3 全実装"
```
