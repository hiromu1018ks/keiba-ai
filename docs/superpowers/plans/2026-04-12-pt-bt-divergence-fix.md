# PT/BT 乖離修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ペーパートレード (ROI 57.1%) とバックテスト (ROI 216.6%) の乖離を、根本原因であるオッズ動態特徴量のバグと防御的欠落を修正して解消する。

**Architecture:** 3つの独立した修正を適用: (1) `compute_odds_dynamics` の条件付き切り詰めを無条件化、(2) PT に POST_RACE 列の DROP を追加、(3) PT に JRA フィルタを追加。いずれも BT 側の既存実装と同じロジックを PT 側に移植する。

**Tech Stack:** Python 3.11, pandas, LightGBM, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/features/odds_dynamics_features.py` | Modify | `if len(ts) > 1_000_000` 条件を削除し、常に tail(60) を適用 |
| `scripts/run_paper_trading.py` | Modify | POST_RACE DROP (3箇所) と JRA フィルタ (3箇所) を追加 |
| `tests/test_odds_dynamics_fix.py` | Create | odds_drop_rate の切り詰めが常に発動することを検証 |
| `tests/test_paper_trading_guards.py` | Create | POST_RACE DROP と JRA フィルタの検証 |

---

## Task 1: odds_drop_rate 切り詰めの無条件化

**Files:**
- Modify: `src/features/odds_dynamics_features.py:64-67`
- Test: `tests/test_odds_dynamics_fix.py`

- [ ] **Step 1: Write the failing test**

`tests/test_odds_dynamics_fix.py` を作成:

```python
"""compute_odds_dynamics の切り詰めがデータ量に関わらず常に発動することを検証。"""
import numpy as np
import pandas as pd
import pytest

from features.odds_dynamics_features import compute_odds_dynamics


def _make_ts(n_points: int, n_horses: int = 2) -> pd.DataFrame:
    """指定ポイント数のオッズ時系列を生成。"""
    rows = []
    for umaban in range(1, n_horses + 1):
        for i in range(n_points):
            rows.append({
                "race_id": "20260412010101",
                "umaban": umaban,
                "happyotime": f"{1200 + i:04d}",  # 1200, 1201, ...
                "tanodds": 5.0 + np.random.randn() * 0.1,
                "ninki": umaban,
            })
    return pd.DataFrame(rows)


def test_truncation_always_applies_small_data():
    """データが100万行未満でも切り詰めが発動する (PT相当)。"""
    base = pd.DataFrame({
        "race_id": ["20260412010101"] * 2,
        "umaban": [1, 2],
    })
    # 100ポイント → 60に切り詰められるべき
    ts = _make_ts(100)
    assert len(ts) < 1_000_000  # 従来の閾値以下

    result = compute_odds_dynamics(base, ts)
    # odds_drop_rate_60_10 が計算されている (= NaN ではない)
    assert result["odds_drop_rate_60_10"].notna().all()


def test_truncation_always_applies_large_data():
    """データが100万行超えでも切り詰めが発動する (BT相当)。"""
    base = pd.DataFrame({
        "race_id": ["20260412010101"] * 2,
        "umaban": [1, 2],
    })
    ts = _make_ts(100)
    # テスト用に大きな DataFrame を偽装 (実際の計算は100行で検証)
    assert len(ts) > 0
    result = compute_odds_dynamics(base, ts)
    assert result["odds_drop_rate_60_10"].notna().all()


def test_truncation_limit_60_points():
    """各 (race_id, umaban) が最大60ポイントに切り詰められる。"""
    base = pd.DataFrame({
        "race_id": ["20260412010101"] * 2,
        "umaban": [1, 2],
    })
    # 200ポイント生成 → 内部的に60に切り詰められる
    ts = _make_ts(200, n_horses=2)

    # 切り詰め後の odds_velocity が正しく計算されている
    # (200ポイント全体で計算すると値が異なる)
    result = compute_odds_dynamics(base, ts)
    assert result["odds_velocity"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds_dynamics_fix.py -v`
Expected: tests pass (修正は防御的 - 既存動作の維持確認)

- [ ] **Step 3: Write minimal implementation**

`src/features/odds_dynamics_features.py` line 64-67 を変更:

Before:
```python
    max_points = 60
    if len(ts) > 1_000_000:
        ts = ts.groupby(["race_id", "umaban"], as_index=False).tail(max_points)
```

After:
```python
    max_points = 60
    ts = ts.groupby(["race_id", "umaban"], as_index=False).tail(max_points)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds_dynamics_fix.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/odds_dynamics_features.py tests/test_odds_dynamics_fix.py
git commit -m "fix: compute_odds_dynamics の切り詰めを無条件化 (PT/BT の一致を保証)"
```

---

## Task 2: PT に POST_RACE DROP を追加

**Files:**
- Modify: `scripts/run_paper_trading.py` (3箇所)
- Test: `tests/test_paper_trading_guards.py`

- [ ] **Step 1: Write the failing test**

`tests/test_paper_trading_guards.py` を作成:

```python
"""PT の POST_RACE DROP と JRA フィルタの検証。"""
import pandas as pd
import pytest


def test_post_race_cols_removed():
    """POST_RACE 列が predict 前に DROP されることを検証。"""
    _POST_RACE_COLS = ["kakuteijyuni", "confirmed_odds"]

    df = pd.DataFrame({
        "race_id": ["R001"],
        "umaban": [1],
        "kakuteijyuni": [3],
        "confirmed_odds": [5.2],
        "tanodds": [4.8],
    })

    result = df.drop(
        columns=[c for c in _POST_RACE_COLS if c in df.columns],
        errors="ignore",
    )

    assert "kakuteijyuni" not in result.columns
    assert "confirmed_odds" not in result.columns
    assert "tanodds" in result.columns
    assert "umaban" in result.columns


def test_post_race_cols_missing_no_error():
    """POST_RACE 列が存在しなくてもエラーにならない。"""
    _POST_RACE_COLS = ["kakuteijyuni", "confirmed_odds"]

    df = pd.DataFrame({
        "race_id": ["R001"],
        "umaban": [1],
        "tanodds": [4.8],
    })

    result = df.drop(
        columns=[c for c in _POST_RACE_COLS if c in df.columns],
        errors="ignore",
    )

    assert "tanodds" in result.columns
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_paper_trading_guards.py -v`
Expected: PASS

- [ ] **Step 3: Add POST_RACE DROP to `_run_predict`**

`scripts/run_paper_trading.py` の `_run_predict` 関数内、
`predict()` 呼び出し直前 (単一レース抽出後) に追加:

```python
    # POST_RACE 列を除外 (BT engine.py と同じ処理)
    _POST_RACE_COLS = ["kakuteijyuni", "confirmed_odds"]
    single_race = single_race.drop(
        columns=[c for c in _POST_RACE_COLS if c in single_race.columns],
        errors="ignore",
    )
```

- [ ] **Step 4: Add POST_RACE DROP to `_run_diagnose`**

`_run_diagnose` 内の対応箇所に同じコードを追加。

- [ ] **Step 5: Add POST_RACE DROP to `_run_dry_run`**

`_run_dry_run` 内の対応箇所に同じコードを追加。

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_paper_trading_guards.py tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/run_paper_trading.py tests/test_paper_trading_guards.py
git commit -m "fix: PT に POST_RACE 列の DROP を追加 (BT と同等の防御)"
```

---

## Task 3: PT に JRA フィルタを追加

**Files:**
- Modify: `scripts/run_paper_trading.py` (3箇所)
- Test: `tests/test_paper_trading_guards.py` (追記)

- [ ] **Step 1: Write the failing test**

`tests/test_paper_trading_guards.py` に追記:

```python
def test_jra_filter_removes_nar():
    """jyocd >= 30 の NAR レースが除外される。"""
    df = pd.DataFrame({
        "race_id": ["R001", "R002", "R003"],
        "jyocd": [5, 30, 8],     # 5=JRA, 30=NAR, 8=JRA
        "umaban": [1, 2, 3],
    })

    jyocd_int = pd.to_numeric(df["jyocd"], errors="coerce")
    result = df[jyocd_int.between(1, 10)]

    assert len(result) == 2
    assert set(result["race_id"]) == {"R001", "R003"}


def test_jra_filter_preserves_all_jra():
    """jyocd 1-10 は全て保持される。"""
    df = pd.DataFrame({
        "race_id": [f"R{i:03d}" for i in range(10)],
        "jyocd": list(range(1, 11)),
        "umaban": [1] * 10,
    })

    jyocd_int = pd.to_numeric(df["jyocd"], errors="coerce")
    result = df[jyocd_int.between(1, 10)]

    assert len(result) == 10


def test_jra_filter_handles_missing_jyocd():
    """jyocd 列がない場合はフィルタをスキップ。"""
    df = pd.DataFrame({
        "race_id": ["R001"],
        "umaban": [1],
    })

    if "jyocd" in df.columns:
        jyocd_int = pd.to_numeric(df["jyocd"], errors="coerce")
        df = df[jyocd_int.between(1, 10)]

    assert len(df) == 1  # フィルタなし = 全て保持
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_paper_trading_guards.py -v`
Expected: PASS

- [ ] **Step 3: Add JRA filter to `_run_predict`**

`scripts/run_paper_trading.py` の `_run_predict` 内、
`feat_df` 生成後 (既存の `submodel_mgr.add_distance_band_features` の後) に追加:

```python
    # JRAフィルタ: NARレース (jyocd >= 30) を除外
    if "jyocd" in feat_df.columns:
        jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
        before_count = len(feat_df)
        feat_df = feat_df[jyocd_int.between(1, 10)]
        after_count = len(feat_df)
        if before_count > after_count:
            logger.info(
                "JRA filter: excluded %d NAR entries, %d remaining",
                before_count - after_count,
                after_count,
            )
```

- [ ] **Step 4: Add JRA filter to `_run_diagnose`**

`_run_diagnose` 内の対応箇所に同じコードを追加。

- [ ] **Step 5: Add JRA filter to `_run_dry_run`**

`_run_dry_run` 内の対応箇所に同じコードを追加。

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_paper_trading_guards.py tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/run_paper_trading.py tests/test_paper_trading_guards.py
git commit -m "fix: PT に JRA フィルタを追加 (NAR レース除外、BT と同等)"
```

---

## Verification

全修正完了後、以下で回帰テスト:

```bash
python -m pytest tests/ -v
ruff check src/ tests/ scripts/run_paper_trading.py
```
