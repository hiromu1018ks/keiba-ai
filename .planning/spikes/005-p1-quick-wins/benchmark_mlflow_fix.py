"""MLflow pip推論ベンチマーク（Windows互換版）。"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np


def benchmark_mlflow_pip() -> dict:
    """MLflow log_model の pip推論オーバーヘッドを計測。"""
    import mlflow

    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = lgb.train(
        {"objective": "regression", "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )

    results = {}
    tmpdir = tempfile.mkdtemp()
    # Windows互換: バックスラッシュをスラッシュに変換 + ドライブレター対応
    tracking_uri = f"file:///{tmpdir.replace(chr(92), '/')}/mlruns"

    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run():
        # デフォルト (pip推論あり) — 3回計測
        times_default = []
        for i in range(3):
            t0 = time.time()
            mlflow.lightgbm.log_model(model, f"test_default_{i}")
            times_default.append(time.time() - t0)

        # 明示的 pip指定 — 3回計測
        pip_reqs = ["lightgbm", "scikit-learn", "pandas", "numpy", "joblib"]
        times_explicit = []
        for i in range(3):
            t1 = time.time()
            mlflow.lightgbm.log_model(
                model, f"test_explicit_{i}",
                pip_requirements=pip_reqs,
            )
            times_explicit.append(time.time() - t1)

    avg_default = sum(times_default) / len(times_default)
    avg_explicit = sum(times_explicit) / len(times_explicit)
    per_call_saving = avg_default - avg_explicit

    results = {
        "default_avg_seconds": round(avg_default, 3),
        "explicit_avg_seconds": round(avg_explicit, 3),
        "per_call_saving_seconds": round(per_call_saving, 3),
        "total_saving_26_calls_seconds": round(per_call_saving * 26, 1),
        "default_runs": [round(t, 3) for t in times_default],
        "explicit_runs": [round(t, 3) for t in times_explicit],
    }

    print(f"  Default (pip inference): {avg_default:.3f}s/call")
    print(f"  Explicit pip_requirements: {avg_explicit:.3f}s/call")
    print(f"  Per-call saving: {per_call_saving:.3f}s")
    print(f"  Total saving (26 calls): {per_call_saving * 26:.1f}s")

    return results


def benchmark_odds_caching() -> dict:
    """odds時系列データの重複ロードを計測。"""
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    results = {}

    # odds時系列データ
    subpaths = [
        ("odds", "jodds_tanpuku"),
        ("odds", "time_series"),
        ("odds", "snapshots"),
    ]

    for table, subpath in subpaths:
        if not store.exists(table, subpath):
            continue

        # コールドロード
        t0 = time.time()
        df = store.read(table, subpath)
        cold_time = time.time() - t0

        size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        results[f"{table}/{subpath}"] = {
            "rows": len(df),
            "size_mb": round(size_mb, 1),
            "cold_load_seconds": round(cold_time, 2),
            "saving_if_cached_2x": round(cold_time * 2, 2),
        }
        print(f"  {table}/{subpath}: {len(df):,} rows, {size_mb:.1f}MB, "
              f"load={cold_time:.2f}s, saving(2x)={cold_time * 2:.2f}s")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  Spike 005: MLflow + Odds Caching (Windows Fix)")
    print("=" * 70)

    all_results: dict = {}

    print("\n--- MLflow pip inference ---")
    try:
        all_results["mlflow_pip"] = benchmark_mlflow_pip()
    except Exception as e:
        all_results["mlflow_pip"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n--- Odds time series caching ---")
    try:
        all_results["odds_caching"] = benchmark_odds_caching()
    except Exception as e:
        all_results["odds_caching"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # 結果保存
    output_path = Path(__file__).parent / "mlflow_odds_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")
