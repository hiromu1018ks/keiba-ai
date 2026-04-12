# バックテストスクリプト統合 & モデル分離 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** バックテスト実行時の本番モデル上書きを防止し、単一・マルチ年度のバックテストを1スクリプトに統合する

**Architecture:** TrainingPipelineV5 に model_dir パラメータを追加してバックテスト用モデルを data/models-backtest/ に分離。run_backtest.py に --years/--train-window オプションを追加してマルチ年度対応。各年度の分析結果を parquet で出力。

**Tech Stack:** Python 3.11, LightGBM, pandas, pyarrow, pytest

**Spec:** `docs/superpowers/specs/2026-04-12-backtest-script-unification-design.md`

---

## File Structure

### Modify
| File | Responsibility |
|---|---|
| `src/pipelines/training_pipeline.py` | `model_dir` パラメータ追加 + `_save_models_local` のインスタンスメソッド化 |
| `src/backtest/validation_suite.py` | `model_dir` を渡して本番モデルを保護 (1行) |
| `src/backtest/engine.py` | `diag_prefix` パラメータ追加（年度別診断出力） |
| `scripts/run_backtest.py` | マルチ年度モード + model_dir 分離 + parquet 出力（完全書き換え） |
| `tests/test_training_pipeline.py` | `__new__()` パターンに `model_dir` 追加 (6箇所) |
| `tests/test_mlflow_logging.py` | `__new__()` パターンに `model_dir` 追加 (2箇所) |
| `tests/test_backtest_engine.py` | `__new__()` パターンに `model_dir` 追加 (1箇所) + diag_prefix テスト追加 |

### Create
| File | Responsibility |
|---|---|
| `tests/test_run_backtest_args.py` | CLI 引数検証のテスト |

### Delete
| File | Reason |
|---|---|
| `scripts/run_multi_year_backtest.py` | run_backtest.py に統合 |

---

### Task 1: TrainingPipelineV5 に model_dir パラメータを追加

**Files:**
- Modify: `src/pipelines/training_pipeline.py:63-72` (__init__), `712-802` (_save_models_local)
- Test: `tests/test_training_pipeline.py`

- [ ] **Step 1: model_dir デフォルト値とカスタム値のテストを追加**

`tests/test_training_pipeline.py` の末尾に追加:

```python
class TestModelDir:
    """TrainingPipelineV5.model_dir のテスト"""

    def test_default_model_dir(self) -> None:
        """model_dir を省略した場合のデフォルトは Path('data/models')"""
        from pathlib import Path

        from pipelines.training_pipeline import TrainingPipelineV5

        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
        pipeline.store = MagicMock()
        pipeline.db = None
        pipeline.feature_engine = MagicMock()
        pipeline.submodel_mgr = MagicMock()
        # __init__ を呼ばずに直接検証はできないので、
        # __init__ を呼び出して確認
        pipeline2 = TrainingPipelineV5()
        assert pipeline2.model_dir == Path("data/models")

    def test_custom_model_dir(self) -> None:
        """カスタム model_dir が設定できる"""
        from pathlib import Path

        from pipelines.training_pipeline import TrainingPipelineV5

        pipeline = TrainingPipelineV5(model_dir=Path("data/models-backtest"))
        assert pipeline.model_dir == Path("data/models-backtest")

    def test_model_dir_none_uses_default(self) -> None:
        """model_dir=None の場合はデフォルト値が使用される"""
        from pathlib import Path

        from pipelines.training_pipeline import TrainingPipelineV5

        pipeline = TrainingPipelineV5(model_dir=None)
        assert pipeline.model_dir == Path("data/models")
```

- [ ] **Step 2: テストを実行して失敗を確認**

```bash
python -m pytest tests/test_training_pipeline.py::TestModelDir -v
```

Expected: FAIL — `TrainingPipelineV5.__init__()` は `model_dir` 引数を受け取らないため `TypeError`

- [ ] **Step 3: `__init__` に `model_dir` を追加**

`src/pipelines/training_pipeline.py` の `__init__` (63-72行):

```python
# 変更前:
def __init__(
    self,
    store: ParquetStore | None = None,
    db: DatabaseConnection | None = None,
    settings_path: str | None = None,
) -> None:
    self.store = store or ParquetStore()
    self.db = db
    self.feature_engine = FeatureEngine()
    self.submodel_mgr = SubModelManager()

# 変更後:
def __init__(
    self,
    store: ParquetStore | None = None,
    db: DatabaseConnection | None = None,
    settings_path: str | None = None,
    model_dir: Path | None = None,
) -> None:
    self.store = store or ParquetStore()
    self.db = db
    self.feature_engine = FeatureEngine()
    self.submodel_mgr = SubModelManager()
    self.model_dir = model_dir or Path("data/models")
```

`Path` の import がファイル上部にない場合、`from pathlib import Path` を追加。

- [ ] **Step 4: `_save_models_local` を `@staticmethod` → インスタンスメソッドに変換**

`src/pipelines/training_pipeline.py` 712行目付近:

```python
# 変更前:
    @staticmethod
    def _save_models_local(
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> Path:
        models_dir = Path("data/models")

# 変更後:
    def _save_models_local(
        self,
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> Path:
        models_dir = self.model_dir
```

変更点:
- `@staticmethod` デコレータを削除
- 第一引数に `self` を追加
- `Path("data/models")` → `self.model_dir`

`_log_to_mlflow` からの呼び出し (710行目) は `self._save_models_local(...)` のまま変更不要（Python は `@staticmethod` も `self.` で呼べる）。

- [ ] **Step 5: テストを実行して成功を確認**

```bash
python -m pytest tests/test_training_pipeline.py::TestModelDir -v
```

Expected: PASS

- [ ] **Step 6: 既存テストの `__new__()` パターンに `model_dir` を追加 (9箇所)**

以下のすべての `__new__()` パターンの `pipeline.submodel_mgr = ...` の直後に `pipeline.model_dir = Path("data/models")` を追加:

**`tests/test_training_pipeline.py`** (6箇所: 246, 288, 324, 341, 441, 559行目):
```python
# 各箇所の pipeline.submodel_mgr = SubModelManager() の直後に追加:
pipeline.model_dir = Path("data/models")
```

ファイル上部に `from pathlib import Path` がなければ追加。

**`tests/test_mlflow_logging.py`** (2箇所: 56, 118行目):
```python
# pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5) の直後の行に追加:
pipeline.model_dir = Path("data/models")
```

**`tests/test_backtest_engine.py`** (1箇所: 732行目):
```python
# pipeline.submodel_mgr = SubModelManager() の直後に追加:
pipeline.model_dir = Path("data/models")
```

- [ ] **Step 7: 全テストを実行して回帰なしを確認**

```bash
python -m pytest tests/ -v
```

Expected: 全テスト PASS

- [ ] **Step 8: コミット**

```bash
git add src/pipelines/training_pipeline.py tests/test_training_pipeline.py tests/test_mlflow_logging.py tests/test_backtest_engine.py
git commit -m "feat: TrainingPipelineV5 に model_dir パラメータを追加

- __init__ に model_dir: Path | None = None を追加 (デフォルト: data/models)
- _save_models_local を @staticmethod → インスタンスメソッドに変換
- self.model_dir でモデル保存先をカスタマイズ可能に
- 全テストの __new__() パターンに model_dir を追加 (9箇所)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: BacktestValidationSuite に model_dir を渡す

**Files:**
- Modify: `src/backtest/validation_suite.py:587`
- Test: `tests/test_validation_suite.py`

- [ ] **Step 1: 実装を変更（1行）**

`src/backtest/validation_suite.py` の587行目:

```python
# 変更前:
pipeline = TrainingPipelineV5(store=self.store)

# 変更後:
pipeline = TrainingPipelineV5(store=self.store, model_dir=Path("data/models-validation"))
```

`from pathlib import Path` が import にない場合は追加。

- [ ] **Step 2: 全テストを実行して回帰なしを確認**

```bash
python -m pytest tests/ -v
```

Expected: 全テスト PASS（テストは mock 使用のため実動作に影響なし）

- [ ] **Step 3: コミット**

```bash
git add src/backtest/validation_suite.py
git commit -m "fix: BacktestValidationSuite で model_dir を分離し本番モデルを保護

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: BacktestEngine に年度別診断プレフィックスを追加

**Files:**
- Modify: `src/backtest/engine.py:100-124` (__init__), `501` (diag_logger.save)
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: diag_prefix パラメータのテストを追加**

`tests/test_backtest_engine.py` の `TestBacktestEngine` クラスに追加:

```python
def test_init_with_diag_prefix(self, mock_models: MagicMock) -> None:
    """diag_prefix パラメータを設定できる"""
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(models=mock_models, diag_prefix="bt_2024")
    assert engine.diag_prefix == "bt_2024"

def test_init_diag_prefix_default(self, mock_models: MagicMock) -> None:
    """diag_prefix のデフォルトは 'bt'"""
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(models=mock_models)
    assert engine.diag_prefix == "bt"
```

- [ ] **Step 2: テストを実行して失敗を確認**

```bash
python -m pytest tests/test_backtest_engine.py::TestBacktestEngine::test_init_with_diag_prefix tests/test_backtest_engine.py::TestBacktestEngine::test_init_diag_prefix_default -v
```

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'diag_prefix'`

- [ ] **Step 3: BacktestEngine に diag_prefix パラメータを追加**

**3a. `__init__` シグネチャ (100-105行):**

```python
# 変更前:
def __init__(
    self,
    models: TrainedModelsV5,
    initial_bankroll: float = 100_000,
    store: ParquetStore | None = None,
    betting_mode: str = "flat",
) -> None:

# 変更後:
def __init__(
    self,
    models: TrainedModelsV5,
    initial_bankroll: float = 100_000,
    store: ParquetStore | None = None,
    betting_mode: str = "flat",
    diag_prefix: str = "bt",
) -> None:
```

**3b. `__init__` 本体 (112行目付近、`self.betting_mode` の直後):**

```python
self.diag_prefix = diag_prefix
```

**3c. `diag_logger.save` の呼び出し (501行):**

```python
# 変更前:
diag_logger.save(Path("data/backtest"), prefix="bt")

# 変更後:
diag_logger.save(Path("data/backtest"), prefix=self.diag_prefix)
```

- [ ] **Step 4: テストを実行して成功を確認**

```bash
python -m pytest tests/test_backtest_engine.py -v
```

Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: BacktestEngine に diag_prefix パラメータを追加（年度別診断出力）

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: run_backtest.py の CLI をマルチ年度対応に再構築

**Files:**
- Create: `tests/test_run_backtest_args.py`
- Modify: `scripts/run_backtest.py` (完全書き換え)

- [ ] **Step 1: 引数検証のテストを新規ファイルに追加**

`tests/test_run_backtest_args.py` を作成:

```python
"""run_backtest.py CLI 引数解析のテスト"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_backtest(args: list[str]) -> subprocess.CompletedProcess[str]:
    """run_backtest.py を subprocess で実行し、結果を返す"""
    return subprocess.run(
        [sys.executable, "scripts/run_backtest.py"] + args,
        capture_output=True,
        text=True,
        timeout=10,
    )


class TestSingleYearMode:
    """単一年度モード: --train-start/end + --test-start/end"""

    def test_single_year_args_accepted(self) -> None:
        """4つの日付引数がすべて指定されていれば引数解析はパスする"""
        result = _run_backtest([
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_single_year_with_betting_mode(self) -> None:
        """--betting-mode が指定できる"""
        result = _run_backtest([
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
            "--betting-mode", "kelly",
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_single_year_with_ensemble(self) -> None:
        """--ensemble が指定できる"""
        result = _run_backtest([
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
            "--ensemble",
        ])
        assert "unrecognized arguments" not in result.stderr


class TestMultiYearMode:
    """マルチ年度モード: --years"""

    def test_years_args_accepted(self) -> None:
        """--years が指定されていればマルチ年度モードとして動作する"""
        result = _run_backtest(["--years", "2023", "2024"])
        assert "unrecognized arguments" not in result.stderr

    def test_years_with_train_window(self) -> None:
        """--years と --train-window が同時に指定できる"""
        result = _run_backtest([
            "--years", "2023", "2024", "2025",
            "--train-window", "5",
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_years_with_all_options(self) -> None:
        """--years とすべてのオプションが同時に指定できる"""
        result = _run_backtest([
            "--years", "2024", "2025",
            "--train-window", "4",
            "--betting-mode", "flat",
            "--ensemble",
            "--report",
        ])
        assert "unrecognized arguments" not in result.stderr


class TestErrorCases:
    """エラーケース"""

    def test_no_args_error(self) -> None:
        """引数なし → エラーメッセージ"""
        result = _run_backtest([])
        assert result.returncode != 0

    def test_partial_single_year_args_error(self) -> None:
        """--train-start だけ指定 → エラー"""
        result = _run_backtest(["--train-start", "20200101"])
        assert result.returncode != 0

    def test_train_without_test_error(self) -> None:
        """--train-start/end だけ指定 (--test-start/end なし) → エラー"""
        result = _run_backtest([
            "--train-start", "20200101",
            "--train-end", "20231231",
        ])
        assert result.returncode != 0


class TestTrainWindowDefault:
    """--train-window のデフォルト値"""

    def test_default_train_window_is_four(self) -> None:
        """デフォルトは 4"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--train-window", type=int, default=4)
        args = parser.parse_args([])
        assert args.train_window == 4

    def test_custom_train_window(self) -> None:
        """カスタム値を指定できる"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--train-window", type=int, default=4)
        args = parser.parse_args(["--train-window", "5"])
        assert args.train_window == 5
```

- [ ] **Step 2: テストを実行して失敗を確認**

```bash
python -m pytest tests/test_run_backtest_args.py -v
```

Expected: `TestSingleYearMode` と `TestMultiYearMode` は一部 PASS、`TestErrorCases` は FAIL（現在の run_backtest.py は `required=True` なので `--years` を認識しない）

- [ ] **Step 3: run_backtest.py を完全書き換え**

`scripts/run_backtest.py` を以下の内容で完全に置き換え:

```python
"""バックテスト計測スクリプト

使い方:
  # モード1: 単一年度 (従来互換)
  python scripts/run_backtest.py \\
    --train-start 20200101 --train-end 20231231 \\
    --test-start 20240101 --test-end 20241231

  # モード2: マルチ年度
  python scripts/run_backtest.py \\
    --years 2023 2024 2025 \\
    --train-window 4

  # 共通オプション
    --betting-mode flat|kelly   (デフォルト: flat)
    --ensemble                  (アンサンブル有効化)
    --report                    (HTMLレポート + JSON + parquet 生成)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

# Windows cp932 環境で ¥ が表示できない問題を回避
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def to_dash_date(yyyymmdd: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def build_parser() -> argparse.ArgumentParser:
    """引数パーサーを構築"""
    parser = argparse.ArgumentParser(description="バックテスト")
    parser.add_argument("--train-start", required=False, help="学習開始日 (YYYYMMDD)")
    parser.add_argument("--train-end", required=False, help="学習終了日 (YYYYMMDD)")
    parser.add_argument("--test-start", required=False, help="テスト開始日 (YYYYMMDD)")
    parser.add_argument("--test-end", required=False, help="テスト終了日 (YYYYMMDD)")
    parser.add_argument("--years", nargs="+", type=int, help="マルチ年度指定 (テスト年度)")
    parser.add_argument(
        "--train-window", type=int, default=4,
        help="マルチ年度の学習年数 (デフォルト: 4)",
    )
    parser.add_argument("--report", action="store_true", help="HTMLレポート + parquet を生成")
    parser.add_argument(
        "--betting-mode", choices=["flat", "kelly"], default="flat",
        help="ベット額計算モード (flat=100円固定, kelly=Fractional Kelly)",
    )
    parser.add_argument("--ensemble", action="store_true", help="アンサンブル (B1) を有効化")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """引数の排他バリデーション"""
    if args.years:
        return  # マルチ年度モード — OK
    single_year_args = [args.train_start, args.train_end, args.test_start, args.test_end]
    if all(single_year_args):
        return  # 単一年度モード — OK
    parser.error(
        "単一年度モードには --train-start, --train-end, --test-start, --test-end が必要です"
    )


def save_year_parquet(year: int, result: object) -> None:
    """年度別 parquet 出力: horse_diagnostics + bet_history を結合して保存

    注意: HorseDiagnostic に含まれないフィールド (race_date, bamei, surface, kyori,
    grade_code 等) は bet_history 側にのみ存在する。ベット対象外の馬 (is_bet=False) は
    これらのフィールドが NaN になる。
    """
    pred_dir = Path(ROOT) / "data" / "backtest" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 年度別プレフィックスで診断 CSV を読み込む
    diag_path = Path(ROOT) / "data" / "backtest" / f"bt_{year}_horse_diagnostics.csv"
    if not diag_path.exists():
        logger.warning("診断CSVが見つかりません: %s", diag_path)
        return

    diag_df = pd.read_csv(diag_path)

    if not result.bet_history:
        merged = diag_df
    else:
        bet_df = pd.DataFrame(result.bet_history)
        # bet_history 側の付加情報を horse_diagnostics に left-join
        # bet_cols: ベット対象のみに存在するフィールド (非ベット馬は NaN)
        bet_only_cols = [
            "bet_type", "stake", "odds", "final_odds", "result",
            "ev", "popularity", "bankroll_after",
            "race_date", "surface", "kyori", "grade_code",
            "race_name", "bamei", "kisyu",
            "kakuteijyuni", "track_condition_code",
        ]
        # 存在する列のみ選択
        available_cols = ["race_id", "umaban"] + [c for c in bet_only_cols if c in bet_df.columns]
        bet_subset = bet_df[available_cols].copy()
        merged = diag_df.merge(
            bet_subset, on=["race_id", "umaban"], how="left", suffixes=("", "_bet")
        )

    out_path = pred_dir / f"{year}.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info("Parquet保存: %s (%d rows)", out_path, len(merged))


def display_single_year_result(
    result: object,
    elapsed_train: float,
    elapsed_test: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, Any]:
    """単一年度の結果を表示し、JSON用dictを返す"""
    print()
    print("=" * 50)
    print("  結果")
    print("=" * 50)
    print(f"  レース数:       {result.total_bets:>8,}")
    print(f"  投資額:         {result.total_stake:>10,.0f} 円")
    print(f"  払戻額:         {result.total_return:>10,.0f} 円")
    print(f"  利益:           {result.profit:>10,.0f} 円")
    print(f"  ROI:            {result.total_roi:>9.1%}")
    print(f"  最大DD:         {result.max_drawdown:>9.1%}")
    print(f"  最終資金:       {result.final_bankroll:>10,.0f} 円")
    print(f"  学習時間:       {elapsed_train:>7,.0f} 秒")
    print(f"  テスト時間:     {elapsed_test:>7,.0f} 秒")

    before_roi = 0.638
    diff = result.total_roi - before_roi
    status = "目標達成!" if result.total_roi >= 1.01 else "未達"
    print()
    print("=" * 50)
    print("  Before vs After")
    print("=" * 50)
    print(f"  改善前 ROI:     {before_roi:.1%}")
    print(f"  改善後 ROI:     {result.total_roi:.1%}")
    print(f"  差分:           {diff:+.1%}")
    print(f"  判定:           {status}")

    return {
        "before_roi": before_roi,
        "total_roi": result.total_roi,
        "total_bets": result.total_bets,
        "total_stake": result.total_stake,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "final_bankroll": result.final_bankroll,
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "train_seconds": round(elapsed_train),
        "test_seconds": round(elapsed_test),
    }


def _run_single_year(args: argparse.Namespace) -> None:
    """単一年度バックテスト"""
    train_start = to_dash_date(args.train_start)
    train_end = to_dash_date(args.train_end)
    test_start = to_dash_date(args.test_start)
    test_end = to_dash_date(args.test_end)

    from db.parquet_store import ParquetStore
    from pipelines.training_pipeline import TrainingPipelineV5

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # 学習
    logger.info("=" * 50)
    logger.info("  学習期間: %s ~ %s", train_start, train_end)
    logger.info("=" * 50)
    t0 = time.time()

    pipeline = TrainingPipelineV5(store=store, model_dir=Path("data/models-backtest"))
    try:
        models = pipeline.run(train_start, train_end, use_ensemble=args.ensemble)
    except KeyboardInterrupt:
        logger.warning("学習が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error("学習失敗: %s", e)
        sys.exit(1)

    elapsed_train = time.time() - t0
    logger.info("学習完了 (%.0f秒)", elapsed_train)

    # バックテスト
    logger.info("=" * 50)
    logger.info("  テスト期間: %s ~ %s", test_start, test_end)
    logger.info("=" * 50)
    t1 = time.time()

    from backtest.engine import BacktestEngine

    test_year = int(test_start[:4])
    engine = BacktestEngine(
        models=models, store=store, betting_mode=args.betting_mode,
        diag_prefix=f"bt_{test_year}",
    )
    result = engine.run(test_start, test_end)
    elapsed_test = time.time() - t1
    logger.info("バックテスト完了 (%.0f秒)", elapsed_test)

    # 結果表示
    out = display_single_year_result(
        result, elapsed_train, elapsed_test,
        train_start, train_end, test_start, test_end,
    )

    # 出力
    if args.report:
        from backtest.report import BacktestReportGenerator

        output_dir = Path(ROOT) / "data" / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)

        gen = BacktestReportGenerator(output_dir=output_dir)
        bet_history_path = gen.save_bet_history(result.bet_history)
        print(f"\nbet_history保存: {bet_history_path}")

        result_path = output_dir / "backtest_result.json"
        result_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"結果保存: {result_path}")

        report_path = gen.generate(
            result, result.bet_history,
            train_period=f"{train_start} ~ {train_end}",
            test_period=f"{test_start} ~ {test_end}",
        )
        print(f"レポート生成: {report_path}")

        save_year_parquet(test_year, result)
    else:
        outpath = os.path.join(ROOT, "backtest_result.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n結果保存: {outpath}")


def _run_multi_year(args: argparse.Namespace) -> None:
    """マルチ年度バックテスト"""
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    logger.info("ParquetStore OK")

    all_results: dict[int, Any] = {}
    all_metadata: dict[int, dict[str, str]] = {}

    for test_year in args.years:
        train_start = f"{test_year - args.train_window}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print()
        print("=" * 50)
        print(f"  {test_year}年 (学習: {train_start[:4]}-{train_end[:4]})")
        print("=" * 50)

        # 学習
        t0 = time.time()
        try:
            from pipelines.training_pipeline import TrainingPipelineV5

            pipeline = TrainingPipelineV5(store=store, model_dir=Path("data/models-backtest"))
            models = pipeline.run(train_start, train_end, use_ensemble=args.ensemble)
        except KeyboardInterrupt:
            logger.warning("中断されました")
            sys.exit(1)
        except Exception as e:
            logger.error("%d年 学習失敗: %s — スキップ", test_year, e)
            continue
        elapsed_train = time.time() - t0

        # バックテスト
        t1 = time.time()
        try:
            from backtest.engine import BacktestEngine

            engine = BacktestEngine(
                models=models, store=store, betting_mode=args.betting_mode,
                diag_prefix=f"bt_{test_year}",
            )
            result = engine.run(test_start, test_end)
        except Exception as e:
            logger.error("%d年 テスト失敗: %s — スキップ", test_year, e)
            continue
        elapsed_test = time.time() - t1

        all_results[test_year] = result
        all_metadata[test_year] = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_seconds": str(round(elapsed_train)),
            "test_seconds": str(round(elapsed_test)),
        }

        # マルチ年度では常に parquet 出力
        save_year_parquet(test_year, result)

        profit = result.profit
        print(f"  学習完了 ({elapsed_train:.0f}秒)")
        print(f"  テスト完了 ({elapsed_test:.0f}秒)")
        print(
            f"  ベット数: {result.total_bets:>8,} | "
            f"投資額: ¥{result.total_stake:>10,.0f} | "
            f"払戻: ¥{result.total_return:>10,.0f}"
        )
        print(
            f"  ROI: {result.total_roi:>8.1%} | "
            f"利益: ¥{profit:>+10,.0f} | "
            f"最大DD: {result.max_drawdown:>6.1%}"
        )

    if not all_results:
        logger.error("全年度失敗。レポートは生成しません。")
        sys.exit(1)

    # 全体サマリー
    print()
    print("=" * 50)
    print("  全体サマリー")
    print("=" * 50)
    total_bets = sum(r.total_bets for r in all_results.values())
    total_stake = sum(r.total_stake for r in all_results.values())
    total_return = sum(r.total_return for r in all_results.values())
    total_profit = total_return - total_stake
    total_roi = total_return / total_stake if total_stake > 0 else 0.0
    best_year = max(all_results, key=lambda y: all_results[y].total_roi)
    worst_year = min(all_results, key=lambda y: all_results[y].total_roi)

    print(f"  総ベット数:  {total_bets:>10,}")
    print(f"  総投資額:  ¥{total_stake:>12,.0f}")
    print(f"  総払戻額:  ¥{total_return:>12,.0f}")
    print(f"  総利益:    ¥{total_profit:>+12,.0f}")
    print(f"  合計 ROI:   {total_roi:>10.1%}")
    print(f"  最良年度:  {best_year} ({all_results[best_year].total_roi:.1%})")
    print(f"  最悪年度:  {worst_year} ({all_results[worst_year].total_roi:.1%})")

    # --report 時の出力
    if args.report:
        output_dir = Path(ROOT) / "data" / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)

        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=output_dir)
        report_path = gen.generate(all_results, all_metadata)
        print(f"\n  レポート生成: {report_path}")

        json_data: dict[str, Any] = {
            "overall": {
                "total_bets": total_bets,
                "total_stake": total_stake,
                "total_return": total_return,
                "profit": total_profit,
                "roi": total_roi,
                "best_year": best_year,
                "worst_year": worst_year,
            },
            "years": {},
        }
        for year, r in all_results.items():
            json_data["years"][str(year)] = {
                "total_bets": r.total_bets,
                "total_stake": r.total_stake,
                "total_return": r.total_return,
                "roi": r.total_roi,
                "profit": r.profit,
                "max_drawdown": r.max_drawdown,
                "metadata": all_metadata[year],
            }
        json_path = output_dir / "multi_year_result.json"
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  JSON保存: {json_path}")

        all_bets: list[dict[str, Any]] = []
        for year, r in all_results.items():
            for bet in r.bet_history:
                all_bets.append({**bet, "_test_year": year})
        bets_path = output_dir / "multi_year_bet_history.json"
        bets_path.write_text(
            json.dumps(all_bets, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  bet_history保存: {bets_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.years:
        _run_multi_year(args)
    else:
        _run_single_year(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストを実行して成功を確認**

```bash
python -m pytest tests/test_run_backtest_args.py -v
```

Expected: PASS

- [ ] **Step 5: 全テストを実行して回帰なしを確認**

```bash
python -m pytest tests/ -v
```

Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add scripts/run_backtest.py tests/test_run_backtest_args.py
git commit -m "feat: run_backtest.py をマルチ年度対応に統合

- --years/--train-window でマルチ年度バックテスト対応
- model_dir=Path('data/models-backtest') で本番モデルを保護
- 年度別 parquet 出力 (data/backtest/predictions/{year}.parquet)
- diag_prefix で年度別診断CSV出力
- --report フラグで単一/マルチ両対応のレポート生成
- run_multi_year_backtest.py の機能を統合

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: run_multi_year_backtest.py を削除 + ドキュメント更新

**Files:**
- Delete: `scripts/run_multi_year_backtest.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: run_multi_year_backtest.py を削除**

```bash
git rm scripts/run_multi_year_backtest.py
```

- [ ] **Step 2: CLAUDE.md の Pipeline Scripts セクションを更新**

`CLAUDE.md` の以下を変更:

**バックテストコマンド (Step 3):**

```markdown
# 変更前:
# Step 3: バックテスト — 学習 + テスト期間のシミュレーション
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231

# 変更後:
# Step 3a: バックテスト (単一年度)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231

# Step 3b: バックテスト (マルチ年度)
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4
```

**スクリプト詳細テーブル:**

```markdown
# 変更前:
| `scripts/run_backtest.py` | 学習+バックテスト | Parquet + 学習済みモデル | `backtest_result.json` | ~57分 |

# 変更後:
| `scripts/run_backtest.py` | 学習+バックテスト (単一年度/マルチ年度) | Parquet + 学習済みモデル | `backtest_result.json` または `data/backtest/multi_year_result.json` | ~57分/年 |
```

**備考行:**

```markdown
# 変更前:
- `run_backtest.py` は毎回学習し直す設計（再現性保証）。モデルの保存/読み込みはMLflow経由

# 変更後:
- `run_backtest.py` は毎回学習し直す設計（再現性保証）。モデルは `data/models-backtest/` に保存 (本番 `data/models/` は上書きしない)
- `run_multi_year_backtest.py` は廃止。`run_backtest.py --years` に統合済み
```

- [ ] **Step 3: 全テストを実行して回帰なしを確認**

```bash
python -m pytest tests/ -v
```

Expected: PASS

- [ ] **Step 4: コミット**

```bash
git add scripts/run_multi_year_backtest.py CLAUDE.md
git commit -m "chore: run_multi_year_backtest.py を削除し CLAUDE.md を更新

- run_multi_year_backtest.py を削除 (run_backtest.py --years に統合済み)
- CLAUDE.md にマルチ年度モードのドキュメントを追加
- モデル分離 (data/models-backtest/) を記載

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
