# Dev Workflow Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 3 standalone scripts (ETL, Train, Backtest) with argparse that cover the development pipeline.

**Architecture:** Each script is self-contained — initializes its own `ParquetStore`/`DataRepository`, accepts CLI arguments, handles errors. `run_backtest.py` refactors the existing hardcoded script.

**Tech Stack:** Python 3.11, argparse, existing keiba-ai modules (db, pipelines, backtest)

**Spec:** `docs/superpowers/specs/2026-03-28-dev-workflow-scripts-design.md`

---

### Task 1: `scripts/run_etl.py` — ETLスクリプト

**Files:**
- Create: `scripts/run_etl.py`

- [ ] **Step 1: `run_etl.py` を作成**

```python
"""EveryDB2外部テーブル → Parquetエクスポート

使い方:
  python scripts/run_etl.py --start 20150101 --end 20241231
"""

import argparse
import logging
import os
import sys
import time

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="EveryDB2 → Parquet ETL")
    parser.add_argument("--start", required=True, help="開始日 (YYYYMMDD)")
    parser.add_argument("--end", required=True, help="終了日 (YYYYMMDD)")
    args = parser.parse_args()

    # DB接続
    from db.connection import DatabaseConnection
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    db = DatabaseConnection()

    logger.info("ETL開始: %s ~ %s", args.start, args.end)
    t0 = time.time()

    try:
        counts = db.etl_to_parquet(store, args.start, args.end)
    except KeyboardInterrupt:
        logger.warning("ETLが中断されました")
        sys.exit(1)
    except Exception as e:
        if "could not connect" in str(e).lower() or "connection refused" in str(e).lower():
            logger.error("PostgreSQLに接続できません。localhost:5432 が実行中か確認してください。")
        else:
            logger.error("ETL失敗: %s", e)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("ETL完了 (%.0f秒)", elapsed)

    for table, n in counts.items():
        logger.info("  %s: %d行", table, n)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: ヘルプ表示を確認**

Run: `python scripts/run_etl.py --help`
Expected: argparse usage message with `--start` and `--end`

- [ ] **Step 3: コミット**

```bash
git add scripts/run_etl.py
git commit -m "feat: add scripts/run_etl.py — EveryDB2 to Parquet ETL script"
```

---

### Task 2: `scripts/run_train.py` — 学習スクリプト

**Files:**
- Create: `scripts/run_train.py`

- [ ] **Step 1: `run_train.py` を作成**

```python
"""モデル学習スクリプト

使い方:
  python scripts/run_train.py --start 20200101 --end 20231231
  python scripts/run_train.py --start 20200101 --end 20231231 --experiment keiba-v5.5
"""

import argparse
import logging
import os
import sys
import time

# プロジェクトルートをパスに追加
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


def main() -> None:
    parser = argparse.ArgumentParser(description="モデル学習")
    parser.add_argument("--start", required=True, help="学習開始日 (YYYYMMDD)")
    parser.add_argument("--end", required=True, help="学習終了日 (YYYYMMDD)")
    parser.add_argument(
        "--experiment", default="keiba-v5", help="MLflow実験名 (default: keiba-v5)"
    )
    args = parser.parse_args()

    train_start = to_dash_date(args.start)
    train_end = to_dash_date(args.end)

    # データ検証
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # MLflow実験名設定
    import mlflow

    mlflow.set_experiment(args.experiment)

    # 学習
    from db.repository import DataRepository
    from pipelines.training_pipeline import TrainingPipelineV5

    repo = DataRepository(store)
    pipeline = TrainingPipelineV5(repo=repo)

    logger.info("学習開始: %s ~ %s", train_start, train_end)
    t0 = time.time()

    try:
        models = pipeline.run(train_start, train_end)
    except KeyboardInterrupt:
        logger.warning("学習が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error("学習失敗: %s", e)
        sys.exit(1)

    elapsed = time.time() - t0
    n_surfaces = len(models.submodels)
    logger.info("学習完了 (%.0f秒) — %dサーフェスモデル", elapsed, n_surfaces)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: ヘルプ表示を確認**

Run: `python scripts/run_train.py --help`
Expected: argparse usage message with `--start`, `--end`, `--experiment`

- [ ] **Step 3: コミット**

```bash
git add scripts/run_train.py
git commit -m "feat: add scripts/run_train.py — model training script with argparse"
```

---

### Task 3: `scripts/run_backtest.py` — バックテストスクリプト（リファクタ）

**Files:**
- Modify: `scripts/run_backtest.py` (完全書き換え)

- [ ] **Step 1: `run_backtest.py` をリファクタ**

既存のハードコード版をargparse版に書き換える。構造は既存コードを踏襲し、日付範囲をCLI引数化する。

```python
"""バックテスト計測スクリプト

使い方:
  python scripts/run_backtest.py \\
    --train-start 20200101 --train-end 20231231 \\
    --test-start 20240101 --test-end 20241231
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
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


def main() -> None:
    parser = argparse.ArgumentParser(description="バックテスト")
    parser.add_argument("--train-start", required=True, help="学習開始日 (YYYYMMDD)")
    parser.add_argument("--train-end", required=True, help="学習終了日 (YYYYMMDD)")
    parser.add_argument("--test-start", required=True, help="テスト開始日 (YYYYMMDD)")
    parser.add_argument("--test-end", required=True, help="テスト終了日 (YYYYMMDD)")
    args = parser.parse_args()

    train_start = to_dash_date(args.train_start)
    train_end = to_dash_date(args.train_end)
    test_start = to_dash_date(args.test_start)
    test_end = to_dash_date(args.test_end)

    # データ検証
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # データリポジトリ
    from db.repository import DataRepository

    repo = DataRepository(store)
    logger.info("DataRepository OK")

    # 学習
    logger.info("=" * 50)
    logger.info("  学習期間: %s ~ %s", train_start, train_end)
    logger.info("=" * 50)
    t0 = time.time()

    from pipelines.training_pipeline import TrainingPipelineV5

    pipeline = TrainingPipelineV5(repo=repo)
    try:
        models = pipeline.run(train_start, train_end)
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

    engine = BacktestEngine(models=models, repo=repo)
    result = engine.run(test_start, test_end)
    elapsed_test = time.time() - t1
    logger.info("バックテスト完了 (%.0f秒)", elapsed_test)

    # 結果表示
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
    print(f"  学習時間:       {elapsed_train:>7.0f} 秒")
    print(f"  テスト時間:     {elapsed_test:>7.0f} 秒")

    # 改善前との比較
    BEFORE_ROI = 0.638
    diff = result.total_roi - BEFORE_ROI
    status = "目標達成!" if result.total_roi >= 1.01 else "未達"
    print()
    print("=" * 50)
    print("  Before vs After")
    print("=" * 50)
    print(f"  改善前 ROI:     {BEFORE_ROI:.1%}")
    print(f"  改善後 ROI:     {result.total_roi:.1%}")
    print(f"  差分:           {diff:+.1%}")
    print(f"  判定:           {status}")

    # JSON保存
    out = {
        "before_roi": BEFORE_ROI,
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
    outpath = os.path.join(ROOT, "backtest_result.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {outpath}")


if __name__ == "__main__":
    main()
```

**既存コードとの主な変更点:**
1. ハードコード日付 → argparse引数
2. `result.roi` → `result.total_roi` (BacktestResultの正しいフィールド名)
3. `print()` → `logger.info()` (ETL・学習ログ) + `print()` (最終結果)
4. `BacktestEngine(models=models, repo=repo)` キーワード引数で安全に呼び出し
5. ハードコード `PGPASSWORD` を削除（環境変数または`config/settings.yaml`で設定）

- [ ] **Step 2: ヘルプ表示を確認**

Run: `python scripts/run_backtest.py --help`
Expected: argparse usage with `--train-start`, `--train-end`, `--test-start`, `--test-end`

- [ ] **Step 3: コミット**

```bash
git add scripts/run_backtest.py
git commit -m "refactor: add argparse to run_backtest.py, fix total_roi field name"
```

---

### Task 4: 検証

- [ ] **Step 1: 全スクリプトの --help が動作することを確認**

```bash
python scripts/run_etl.py --help
python scripts/run_train.py --help
python scripts/run_backtest.py --help
```

- [ ] **Step 2: ruff でリント**

```bash
ruff check scripts/
```

- [ ] **Step 3: 既存テストが通ることを確認**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: 最終コミット（必要な場合のみ）**

```bash
git add -A scripts/
git commit -m "chore: verify dev workflow scripts pass lint and tests"
```
