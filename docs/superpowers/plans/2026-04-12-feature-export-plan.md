# 特徴量 Parquet 出力 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DiagnosticLogger に horse-level 特徴量 parquet 出力を追加し、バックテストとペーパートレードの特徴量を比較可能にする。

**Architecture:** 既存の DiagnosticLogger に `feature_records` リストと `log_horse_features()` を追加。各パイプラインの `result_df` ループ内で呼び出し、`save()` 時に parquet 出力。既存の CSV 出力・API は変更なし。

**Tech Stack:** Python 3.11, pandas, pyarrow (parquet)

**Spec:** `docs/superpowers/specs/2026-04-12-feature-export-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/backtest/diagnostic_logger.py` | Modify | `feature_records`, `log_horse_features()`, parquet 出力追加 |
| `tests/test_diagnostic_logger.py` | Modify | `log_horse_features()` のテスト追加 |
| `src/backtest/engine.py` | Modify | `log_horse_features()` 呼び出し 2箇所に追加 |
| `scripts/run_paper_trading.py` | Modify | `log_horse_features()` 呼び出し 4箇所に追加 (predict 2箇所, diagnose 2箇所) |

---

### Task 1: DiagnosticLogger に `log_horse_features()` を追加

**Files:**
- Modify: `src/backtest/diagnostic_logger.py`
- Modify: `tests/test_diagnostic_logger.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_diagnostic_logger.py` の `TestDiagnosticLogger` クラスに追加:

```python
def test_log_horse_features_adds_record(self):
    logger = DiagnosticLogger()
    logger.log_horse_features({"race_id": "20240101010111", "umaban": 5, "ev_place": 1.5})
    assert len(logger.feature_records) == 1
    assert logger.feature_records[0]["race_id"] == "20240101010111"
    assert logger.feature_records[0]["umaban"] == 5
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_diagnostic_logger.py::TestDiagnosticLogger::test_log_horse_features_adds_record -v`
Expected: FAIL — `AttributeError: 'DiagnosticLogger' object has no attribute 'log_horse_features'`

- [ ] **Step 3: 最小実装を書く**

`src/backtest/diagnostic_logger.py` の `DiagnosticLogger.__init__` に `self.feature_records: list[dict[str, Any]] = []` を追加し、新メソッドを追加:

```python
from typing import Any

# __init__ に追加:
self.feature_records: list[dict[str, Any]] = []

# 新メソッド:
def log_horse_features(self, row: dict[str, Any]) -> None:
    """result_df の1行（特徴量+予測値+判定）を収集する。"""
    self.feature_records.append(row)
```

注意: `Any` の import は `from typing import Any` または `from __future__ import annotations` 経由。ファイル冒頭に既に `from __future__ import annotations` があるので、型アノテーション内の `Any` は文字列として評価されるが、`list[dict[str, Any]]` の runtime 評価のため `from typing import Any` を追加する必要があるか確認。既存コードが `from dataclasses import asdict, dataclass` のみを import しているので、`from typing import Any` を追加。

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_diagnostic_logger.py::TestDiagnosticLogger::test_log_horse_features_adds_record -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/backtest/diagnostic_logger.py tests/test_diagnostic_logger.py
git commit -m "feat: add log_horse_features() to DiagnosticLogger"
```

---

### Task 2: DiagnosticLogger.save() に parquet 出力を追加

**Files:**
- Modify: `src/backtest/diagnostic_logger.py`
- Modify: `tests/test_diagnostic_logger.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_save_creates_parquet_when_features_logged(self):
    logger = DiagnosticLogger()
    logger.log_horse_features({
        "race_id": "20240101010111",
        "umaban": 5,
        "ev_place": 1.5,
        "surface": "turf",
    })
    logger.log_horse_features({
        "race_id": "20240101010111",
        "umaban": 8,
        "ev_place": 0.9,
        "surface": "dirt",
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        logger.save(outdir, prefix="test")

        parquet_path = outdir / "test_horse_features.parquet"
        assert parquet_path.exists()

        df = pd.read_parquet(parquet_path)
        assert len(df) == 2
        assert "race_id" in df.columns
        assert "umaban" in df.columns
        assert "ev_place" in df.columns
        assert "surface" in df.columns

def test_save_creates_no_parquet_when_no_features(self):
    logger = DiagnosticLogger()
    logger.log_race("20240101010111", "AGGRESSIVE", 1.10, True, 0.6, 3, 2)
    logger.log_horse("20240101010111", 5, 0.35, 4.5, 1.575, 4.2, True)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        logger.save(outdir, prefix="test")
        assert not (outdir / "test_horse_features.parquet").exists()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_diagnostic_logger.py::TestDiagnosticLogger::test_save_creates_parquet_when_features_logged -v`
Expected: FAIL — parquet file does not exist

- [ ] **Step 3: `save()` に parquet 出力を追加**

`src/backtest/diagnostic_logger.py` の `save()` メソッド末尾に追加:

```python
if self.feature_records:
    path = outdir / f"{prefix}_horse_features.parquet"
    pd.DataFrame(self.feature_records).to_parquet(path, index=False)
    logger.info(
        "Feature diagnostics saved: %d records -> %s",
        len(self.feature_records),
        path,
    )
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_diagnostic_logger.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add src/backtest/diagnostic_logger.py tests/test_diagnostic_logger.py
git commit -m "feat: add parquet output to DiagnosticLogger.save()"
```

---

### Task 3: ネスト型除外のテストを追加

**Files:**
- Modify: `tests/test_diagnostic_logger.py`

- [ ] **Step 1: テストを書く**

```python
def test_parquet_excludes_nested_types(self):
    """list/dict 値を持つ列が parquet に出力されないことを確認。"""
    logger = DiagnosticLogger()
    logger.log_horse_features({
        "race_id": "20240101010111",
        "umaban": 5,
        "top3_finishers": [{"umaban": 1}, {"umaban": 2}],  # list[dict] → 除外
        "nested": {"key": "value"},  # dict → 除外
        "ev_place": 1.5,  # float → 保持
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        logger.save(outdir, prefix="test")

        df = pd.read_parquet(outdir / "test_horse_features.parquet")
        assert "top3_finishers" not in df.columns
        assert "nested" not in df.columns
        assert "ev_place" in df.columns
        assert "race_id" in df.columns
```

注意: このテストは呼び出し側（engine.py / run_paper_trading.py）が dict 内包でフィルタすることを前提としている。DiagnosticLogger 自体はフィルタしない。呼び出し側のフィルタが正しく動作することを確認するためのテスト。

- [ ] **Step 2: テストが通ることを確認**

このテストは DiagnosticLogger が生の dict を保存する設計なので、呼び出し側がフィルタしないと `to_parquet` がエラーになる可能性がある。もしエラーになる場合は、`log_horse_features()` 側でフィルタする設計に変更する。

Run: `python -m pytest tests/test_diagnostic_logger.py::TestDiagnosticLogger::test_parquet_excludes_nested_types -v`
Expected: フィルタが呼び出し側にある設計なら、このテストは parquet 書き出し時にエラーになるはず。

**設計判断:** 呼び出し側でフィルタするか、`log_horse_features()` 内でフィルタするか。

- 呼び出し側フィルタ: `{k: v for k, v in hr.items() if not isinstance(v, (list, dict))}` — 4箇所に同じコードが必要
- `log_horse_features()` 内フィルタ: 1箇所で済む → こちらを採用

`log_horse_features()` の実装を以下に変更:

```python
def log_horse_features(self, row: dict[str, Any]) -> None:
    """result_df の1行（特徴量+予測値+判定）を収集する。list/dict 値は除外。"""
    self.feature_records.append(
        {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
    )
```

テストも修正: 生の dict を渡して、フィルタ後の結果を検証。

- [ ] **Step 3: テストが通ることを確認**

Run: `python -m pytest tests/test_diagnostic_logger.py -v`
Expected: 全テスト PASS

- [ ] **Step 4: コミット**

```bash
git add tests/test_diagnostic_logger.py
git commit -m "test: add nested-type exclusion test for log_horse_features()"
```

---

### Task 4: BacktestEngine に `log_horse_features()` を追加

**Files:**
- Modify: `src/backtest/engine.py`

**テスト方針:** 既存の BacktestEngine テストは mock を使用しており、`log_horse_features()` の
呼び出し自体は検証しない。コールサイトのテストは本タスクのスコープ外。
`log_horse_features()` の動作（フィルタ・parquet出力）は Task 1-3 で単体テスト済み。

- [ ] **Step 1: engine.py の should_bet=False ブロックに追加**

`src/backtest/engine.py:348-358` — `log_horse()` の直後に追加:

```python
# 既存コード (line 348-358):
if "ev_place" in result_df.columns:
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            race_id=race_id,
            umaban=int(hr["umaban"]),
            p_place_pred=float(hr.get("p_place_pred", 0)),
            e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
            ev_place=float(hr.get("ev_place", 0)),
            fukuoddslow=float(hr.get("fukuoddslow", 0)),
            is_bet=False,
        )
        # ↓ NEW:
        diag_logger.log_horse_features(hr.to_dict())
```

- [ ] **Step 2: engine.py の should_bet=True ブロックに追加**

`src/backtest/engine.py:385-396` — 同じパターンで追加:

```python
# 既存コード (line 385-396):
if "ev_place" in result_df.columns:
    bet_umabans = {b.umaban for b in bets}
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            race_id=race_id,
            umaban=int(hr["umaban"]),
            ...
            is_bet=int(hr["umaban"]) in bet_umabans,
        )
        # ↓ NEW:
        diag_logger.log_horse_features(hr.to_dict())
```

- [ ] **Step 3: 既存テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: 全テスト PASS（既存テストは mock を使用、`log_horse_features()` が呼ばれても問題なし）

- [ ] **Step 4: コミット**

```bash
git add src/backtest/engine.py
git commit -m "feat: add log_horse_features() calls to BacktestEngine"
```

---

### Task 5: ペーパートレード predict モードに `log_horse_features()` を追加

**Files:**
- Modify: `scripts/run_paper_trading.py`

**テスト方針:** Task 4 と同様、コールサイトのテストはスコープ外。

- [ ] **Step 1: should_bet=False ブロック (line 368-378) に追加**

`scripts/run_paper_trading.py:368-378`:

```python
# 既存コード:
if "ev_place" in result_df.columns:
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            race_id=race_id,
            umaban=int(hr["umaban"]),
            p_place_pred=float(hr.get("p_place_pred", 0)),
            e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
            ev_place=float(hr.get("ev_place", 0)),
            fukuoddslow=float(hr.get("fukuoddslow", 0)),
            is_bet=False,
        )
        # ↓ NEW:
        diag_logger.log_horse_features(hr.to_dict())
```

- [ ] **Step 2: should_bet=True ブロック (line 394-404) に追加**

```python
# 既存コード:
if "ev_place" in result_df.columns:
    bet_umabans = {b.umaban for b in bets}
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            ...
            is_bet=int(hr["umaban"]) in bet_umabans,
        )
        # ↓ NEW:
        diag_logger.log_horse_features(hr.to_dict())
```

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: add log_horse_features() to paper trading predict mode"
```

---

### Task 6: ペーパートレード diagnose モードに `log_horse_features()` を追加

**Files:**
- Modify: `scripts/run_paper_trading.py`

**テスト方針:** Task 4 と同様、コールサイトのテストはスコープ外。

- [ ] **Step 1: should_bet=False ブロック (line 639-649) に追加**

```python
# 既存コード:
if "ev_place" in result_df.columns:
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            ...
            is_bet=False,
        )
        # ↓ NEW:
        diag_logger.log_horse_features(hr.to_dict())
```

- [ ] **Step 2: should_bet=True ブロック (line 662-673) に追加**

```python
# 既存コード:
if "ev_place" in result_df.columns:
    bet_umabans = {b.umaban for b in bets}
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            ...
            is_bet=int(hr["umaban"]) in bet_umabans,
        )
        # ↓ NEW:
        diag_logger.log_horse_features(hr.to_dict())
```

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: add log_horse_features() to paper trading diagnose mode"
```

---

### Task 7: 全テスト実行でリグレッション確認

- [ ] **Step 1: テストスイート全体を実行**

Run: `python -m pytest tests/ -v`
Expected: 全テスト PASS

- [ ] **Step 2: ruff チェック**

Run: `ruff check src/ tests/`
Expected: エラーなし

- [ ] **Step 3: mypy チェック**

Run: `mypy src/`
Expected: エラーなし
