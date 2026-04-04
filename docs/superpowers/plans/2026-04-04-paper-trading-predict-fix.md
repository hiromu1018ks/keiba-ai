# Paper Trading 予測障害修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ペーパートレードの predict/dry-run でオッズ時系列特徴量が全馬 NaN になる障害を修正し、正常な予測・EV計算を可能にする。

**Architecture:** 4つの独立した Fix を順次適用する。(1) readers.py のフォールバックロジック修正 → (2) odds_dynamics_features.py の異常値ガード → (3) run_paper_trading.py への odds_ts_df 渡し → (4) bamei デコード。各 Fix は独立しており、順序依存はないが、Fix 2 を先に完了させると Fix 1 の動作確認が容易になる。

> **NOTE:** Task 順序は Spec の Fix 番号と異なります。Spec では Fix 1 (odds_ts_df 渡し) が最優先ですが、フォールバック (Fix 2) を先に実装することで Fix 1 の動作確認時に 2025-2026 データのフォールバックも検証できます。

**Tech Stack:** Python 3.11, pandas, numpy, pytest, unittest.mock

---

### Task 1: readers.py 空結果フォールバック追加

**Files:**
- Modify: `src/db/readers.py:1-133`
- Test: `tests/test_readers.py`

- [ ] **Step 1: フォールバックの失敗テストを書く**

`tests/test_readers.py` の `TestLoadOddsTimeSeriesRange` クラスに追加:

```python
def test_falls_back_to_jodds_tanpuku_when_time_series_empty(self):
    """time_series が空の場合、jodds_tanpuku にフォールバックする。"""
    store = MagicMock()
    store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
    empty_df = pd.DataFrame()
    fallback_df = pd.DataFrame({
        "race_id": ["20260401010101"], "happyotime": ["03241000"],
        "umaban": [1], "tanodds": [3.0],
    })
    store.read.side_effect = [empty_df, fallback_df]
    result = load_odds_time_series_range(store, "20260401", "20260401")
    assert store.read.call_count == 2
    assert len(result) == 1

def test_no_fallback_when_time_series_has_data(self):
    """time_series にデータがある場合、フォールバックしない。"""
    store = MagicMock()
    store.exists.return_value = True
    valid_df = pd.DataFrame({
        "race_id": ["20240701010101"], "happyotime": ["03241000"],
        "umaban": [1], "tanodds": [5.4],
    })
    store.read.return_value = valid_df
    result = load_odds_time_series_range(store, "20240701", "20240701")
    assert store.read.call_count == 1
    assert len(result) == 1
```

`TestLoadOddsTimeSeries` クラスにも単一 race_id 版のフォールバックテストを追加:

```python
def test_falls_back_to_jodds_tanpuku_when_empty(self):
    """time_series が空の場合、jodds_tanpuku にフォールバックする。"""
    store = MagicMock()
    store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
    empty_df = pd.DataFrame()
    fallback_df = pd.DataFrame({
        "race_id": ["20260401010101"], "happyotime": ["03241000"],
        "umaban": [1], "tanodds": [3.0],
    })
    store.read.side_effect = [empty_df, fallback_df]
    result = load_odds_time_series(store, "20260401010101")
    assert store.read.call_count == 2
    assert len(result) == 1
```

`test_readers.py` の先頭 (line 6 `import pytest` の後) に `import pandas as pd` を追加:

```python
import pandas as pd
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_readers.py::TestLoadOddsTimeSeriesRange::test_falls_back_to_jodds_tanpuku_when_time_series_empty tests/test_readers.py::TestLoadOddsTimeSeriesRange::test_no_fallback_when_time_series_has_data tests/test_readers.py::TestLoadOddsTimeSeries::test_falls_back_to_jodds_tanpuku_when_empty -v`
Expected: FAIL (フォールバックロジック未実装のため `store.read.call_count == 2` が失敗)

- [ ] **Step 3: readers.py にフォールバックを実装**

`src/db/readers.py` に以下を追加:

1. ファイル先頭に `import logging` と `logger = logging.getLogger(__name__)` を追加（既存 import の下）

2. `load_odds_time_series_range()` (line 114-115) の `store.read` 呼び出しの直後にフォールバックを追加:

```python
    subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
    df = store.read("odds", subpath, filters=filters)
    # time_series が要求範囲のデータを持たない場合、jodds_tanpuku にフォールバック
    # jodds_tanpuku も year/month パーティションなので同一 filters が適用可能
    if df.empty and subpath == "time_series" and store.exists("odds", "jodds_tanpuku"):
        logger.debug("time_series empty for %s-%s, falling back to jodds_tanpuku", start, end)
        df = store.read("odds", "jodds_tanpuku", filters=filters)
    df = _coerce_types(df)
```

3. `load_odds_time_series()` (line 127) の `store.read` 呼び出しの直後に同様のフォールバックを追加:

```python
    subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
    df = store.read("odds", subpath, filters=[("race_id", "==", race_id)])
    if df.empty and subpath == "time_series" and store.exists("odds", "jodds_tanpuku"):
        logger.debug("time_series empty for %s, falling back to jodds_tanpuku", race_id)
        df = store.read("odds", "jodds_tanpuku", filters=[("race_id", "==", race_id)])
    df = _coerce_types(df)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_readers.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/readers.py tests/test_readers.py
git commit -m "fix: load_odds_time_series が空結果時に jodds_tanpuku にフォールバックするよう修正"
```

---

### Task 2: odds_dynamics_features.py 異常値ガード追加

**Files:**
- Modify: `src/features/odds_dynamics_features.py:52`
- Test: `tests/test_odds_dynamics_features.py`

- [ ] **Step 1: 異常値ガードの失敗テストを書く**

`tests/test_odds_dynamics_features.py` の `TestOddsDynamicsFeatures` クラスに追加:

```python
def test_out_of_range_tanodds_produces_nan_features(self, base_df: pd.DataFrame):
    """tanodds が範囲外 (1.0未満, 999.9超) の場合、特徴量が NaN になる。"""
    odds_ts = pd.DataFrame({
        "race_id": ["R1", "R1"], "umaban": [1, 1],
        "happyotime": [1, 2], "tanodds": [0.5, 1500.0],
    })
    result = compute_odds_dynamics(base_df, odds_ts)
    assert pd.isna(result["odds_drop_rate_60_10"].iloc[0])
```

回帰防止テスト (実装前でも PASS するが、ガード追加後に正しく動作することを確認する用):

```python
def test_valid_range_tanodds_computes_normally(self, base_df: pd.DataFrame):
    """tanodds が範囲内 (1.0-999.9) の場合、特徴量が正常に計算される。"""
    odds_ts = pd.DataFrame({
        "race_id": ["R1", "R1"], "umaban": [1, 1],
        "happyotime": [1, 2], "tanodds": [5.0, 3.0],
    })
    result = compute_odds_dynamics(base_df, odds_ts)
    assert not pd.isna(result["odds_drop_rate_60_10"].iloc[0])
```

- [ ] **Step 2: 失敗テストを実行して失敗を確認**

Run: `python -m pytest tests/test_odds_dynamics_features.py::TestOddsDynamicsFeatures::test_out_of_range_tanodds_produces_nan_features -v`
Expected: FAIL (ガード未実装のため `odds_drop_rate_60_10` が NaN にならない)

> NOTE: `test_valid_range_tanodds_computes_normally` はガード追加前でも PASS します (回帰防止テスト)。

- [ ] **Step 3: ガードを実装**

`src/features/odds_dynamics_features.py` の `compute_odds_dynamics()` 内、line 52 (`ts = odds_ts.sort_values(...)`) の直後、line 56 (`if "tanninki" in ts.columns`) の前に追加:

```python
    # 合理的オッズ範囲外を NaN にする (1.0-999.9)
    ts.loc[ts["tanodds"] < 1.0, "tanodds"] = np.nan
    ts.loc[ts["tanodds"] > 999.9, "tanodds"] = np.nan
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_odds_dynamics_features.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/odds_dynamics_features.py tests/test_odds_dynamics_features.py
git commit -m "fix: compute_odds_dynamics に tanodds 異常値ガードを追加"
```

---

### Task 3: run_paper_trading.py へ odds_ts_df を渡す

**Files:**
- Modify: `scripts/run_paper_trading.py:204,229,479,519`

- [ ] **Step 1: _run_predict に import と odds_ts_df 読み込みを追加**

`scripts/run_paper_trading.py` の `_run_predict()` (line 204):

1. line 204 の import に `load_odds_time_series_range` を追加:
```python
    from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races
```

2. line 219 (`odds_df = load_odds_snapshots(store, ymd, ymd)`) の直後に追加:
```python
    odds_ts_df = load_odds_time_series_range(store, ymd, ymd)
```

3. line 229 の `build_all` 呼び出しを変更:
```python
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

- [ ] **Step 2: _run_dry_run に import と odds_ts_df 読み込みを追加**

`scripts/run_paper_trading.py` の `_run_dry_run()` (line 479):

1. line 479 の import に `load_odds_time_series_range` を追加:
```python
    from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races
```

2. line 511 (`odds_df = load_odds_snapshots(store, all_start, all_end)`) の直後に追加:
```python
    odds_ts_df = load_odds_time_series_range(store, all_start, all_end)
```

3. line 519 の `build_all` 呼び出しを変更:
```python
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

- [ ] **Step 3: 全テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_paper_trading.py
git commit -m "fix: ペーパートレード predict/dry-run でオッズ時系列データを特徴量生成に渡す"
```

---

### Task 4: bamei 文字化け修正

**Files:**
- Modify: `scripts/run_paper_trading.py`

- [ ] **Step 1: _decode_bamei ヘルパーを追加**

`scripts/run_paper_trading.py` の `_run_setup()` の直前 (line 133 の前) に追加:

```python
def _decode_bamei(name: object) -> str:
    """Shift-JIS バイト列の bamei をデコードする。"""
    if not isinstance(name, str):
        return str(name)
    try:
        return name.encode("latin-1").decode("shift_jis")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name
```

- [ ] **Step 2: _run_predict の bamei 使用箇所を修正**

line 266 の `horse_name = horse.iloc[0]["bamei"]` を:
```python
                    horse_name = _decode_bamei(horse.iloc[0]["bamei"]) if not horse.empty else ""
```

- [ ] **Step 3: _run_dry_run の bamei 使用箇所を修正**

`_run_dry_run` では `bamei` を直接使用していないが、将来の拡張用にメモ。現状では変更なし。ただし `result_df` に `bamei` 列が含まれており、ログ出力等で使用される可能性があるため、必要に応じて追加。

- [ ] **Step 4: 全テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_paper_trading.py
git commit -m "fix: ペーパートレードで bamei の Shift-JIS 文字化けを修正"
```

---

### Task 5: 動作確認 (manual)

- [ ] **Step 1: 既存テストで回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: dry-run で 2024 データをテスト (time_series 経由)**

Run: `python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13`
Expected: `e_return_place_pred` が馬ごとに異なる値を持つ。`odds_drop_rate_*`, `odds_velocity` が NaN でない。

- [ ] **Step 3: dry-run で 2026 データをテスト (jodds_tanpuku フォールバック)**

Run: `python scripts/run_paper_trading.py --mode dry-run --date 2026-04-05`
Expected: フォールバックログ出力。オッズ動態特徴量が計算される。
