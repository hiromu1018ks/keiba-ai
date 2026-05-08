"""バックテストパイプラインの各コンポーネント実行時間を計測するスクリプト。

使い方:
  python .planning/spikes/002-backtest-perf/measure_bottlenecks.py

各Phaseのwall timeを個別に計測し、結果をJSONで出力する。
DBアクセスなし (mock使用) の軽量版と、実データ使用版の両方に対応。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np


def measure_data_loading() -> dict:
    """Bottleneck #3: データロード時間を計測"""
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    results = {}

    for table, subpath in [
        ("raw", "races"),
        ("raw", "entries"),
        ("odds", "snapshots"),
        ("odds", "jodds_tanpuku"),
        ("raw", "payouts"),
    ]:
        if not store.exists(table, subpath):
            print(f"  SKIP {table}/{subpath} (not found)")
            continue

        t0 = time.time()
        df = store.read(table, subpath)
        elapsed = time.time() - t0

        # 型情報を収集
        object_cols = [c for c in df.columns if df[c].dtype == object]
        results[f"{table}/{subpath}"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "object_cols": len(object_cols),
            "object_col_names": object_cols[:10],
            "size_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "load_seconds": round(elapsed, 2),
        }
        print(f"  {table}/{subpath}: {len(df):,} rows, {elapsed:.2f}s, {len(object_cols)} object cols")

    return results


def measure_coerce_types() -> dict:
    """Bottleneck #3: _coerce_types のオーバーヘッドを計測"""
    from db.parquet_store import ParquetStore
    from db.readers import _coerce_types

    store = ParquetStore()

    # odds time series をロードして _coerce_types を計測
    subpath = "jodds_tanpuku" if store.exists("odds", "jodds_tanpuku") else "time_series"
    filters = [("year", ">=", 2020), ("year", "<=", 2024)]

    t0 = time.time()
    df = store.read("odds", subpath, filters=filters)
    load_time = time.time() - t0
    print(f"  odds/{subpath}: {len(df):,} rows loaded in {load_time:.2f}s")

    t1 = time.time()
    df_coerced = _coerce_types(df.copy())
    coerce_time = time.time() - t1
    print(f"  _coerce_types: {coerce_time:.2f}s")

    # 新ETLで既に正しい型かチェック
    object_cols_before = [c for c in df.columns if df[c].dtype == object]
    object_cols_after = [c for c in df_coerced.columns if df_coerced[c].dtype == object]
    already_correct = len(object_cols_before) == len(object_cols_after)

    return {
        "load_seconds": round(load_time, 2),
        "coerce_seconds": round(coerce_time, 2),
        "object_cols_before": len(object_cols_before),
        "object_cols_after": len(object_cols_after),
        "already_correct_types": already_correct,
        "potential_saving_pct": round(coerce_time / (load_time + coerce_time) * 100, 1) if coerce_time > 0 else 0,
    }


def measure_feature_computation() -> dict:
    """Bottleneck #2/#5: 特徴量計算のオーバーヘッドを計測"""
    from db.parquet_store import ParquetStore
    from db.readers import load_races, load_entries

    store = ParquetStore()
    results = {}

    # データロード
    t0 = time.time()
    races = load_races(store, "2020-01-01", "2024-12-31")
    entries = load_entries(store, "2020-01-01", "2024-12-31")
    print(f"  Data loaded: {len(races):,} races, {len(entries):,} entries in {time.time()-t0:.2f}s")

    # object列のカーディナリティを計測
    key_cols = ["race_id", "kettonum", "kisyucode", "chokyosicode"]
    for col in key_cols:
        if col in entries.columns:
            nunique = entries[col].nunique()
            dtype = str(entries[col].dtype)
            print(f"  {col}: dtype={dtype}, nunique={nunique:,}")
            results[f"entries_{col}"] = {"dtype": dtype, "nunique": nunique}

    # Categorical変換の効果を計測
    for col in ["race_id", "kettonum"]:
        if col not in entries.columns:
            continue
        # object dtypeでの groupby
        if entries[col].dtype == object:
            t0 = time.time()
            _ = entries.groupby(col).size()
            object_time = time.time() - t0

            # categorical dtypeでの groupby
            entries_cat = entries.copy()
            entries_cat[col] = entries_cat[col].astype("category")
            t1 = time.time()
            _ = entries_cat.groupby(col).size()
            cat_time = time.time() - t1

            speedup = object_time / cat_time if cat_time > 0 else float("inf")
            print(f"  groupby({col}): object={object_time:.3f}s, category={cat_time:.3f}s, speedup={speedup:.1f}x")
            results[f"groupby_{col}"] = {
                "object_seconds": round(object_time, 3),
                "category_seconds": round(cat_time, 3),
                "speedup": round(speedup, 1),
            }

    return results


def measure_mlflow_overhead() -> dict:
    """Bottleneck #4: MLflow log_model の pip推論時間を計測"""
    try:
        import mlflow
    except ImportError:
        return {"error": "mlflow not installed"}

    import tempfile
    import lightgbm as lgb

    # 最小モデルで log_model の時間を計測
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = lgb.train(
        {"objective": "regression", "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")

        with mlflow.start_run():
            # デフォルト (pip推論あり)
            t0 = time.time()
            mlflow.lightgbm.log_model(model, "test_default")
            default_time = time.time() - t0

            # 明示的 pip指定
            t1 = time.time()
            mlflow.lightgbm.log_model(
                model, "test_explicit",
                pip_requirements=["lightgbm", "scikit-learn", "pandas", "numpy"],
            )
            explicit_time = time.time() - t1

    speedup = default_time / explicit_time if explicit_time > 0 else float("inf")
    per_call_saving = default_time - explicit_time
    total_saving_26calls = per_call_saving * 26

    print(f"  Default log_model: {default_time:.2f}s")
    print(f"  Explicit pip: {explicit_time:.2f}s")
    print(f"  Per-call saving: {per_call_saving:.2f}s")
    print(f"  Total saving (26 calls): {total_saving_26calls:.1f}s")

    return {
        "default_seconds": round(default_time, 2),
        "explicit_pip_seconds": round(explicit_time, 2),
        "per_call_saving": round(per_call_saving, 2),
        "total_saving_26calls": round(total_saving_26calls, 1),
    }


def main() -> None:
    print("=" * 60)
    print("  Backtest Performance Bottleneck Measurement")
    print("=" * 60)

    all_results: dict = {}

    print("\n--- Bottleneck #3: Data Loading ---")
    try:
        all_results["data_loading"] = measure_data_loading()
    except Exception as e:
        all_results["data_loading"] = {"error": str(e)}

    print("\n--- Bottleneck #3: _coerce_types ---")
    try:
        all_results["coerce_types"] = measure_coerce_types()
    except Exception as e:
        all_results["coerce_types"] = {"error": str(e)}

    print("\n--- Bottleneck #2/#5: Feature Computation ---")
    try:
        all_results["feature_computation"] = measure_feature_computation()
    except Exception as e:
        all_results["feature_computation"] = {"error": str(e)}

    print("\n--- Bottleneck #4: MLflow Overhead ---")
    try:
        all_results["mlflow_overhead"] = measure_mlflow_overhead()
    except Exception as e:
        all_results["mlflow_overhead"] = {"error": str(e)}

    # 結果保存
    output_path = Path(__file__).parent / "measurement_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
