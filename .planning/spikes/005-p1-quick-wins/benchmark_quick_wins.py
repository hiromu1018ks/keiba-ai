"""Spike 005: P1 quick-wins の効果を検証。

使い方:
  python .planning/spikes/005-p1-quick-wins/benchmark_quick_wins.py

3つの検証:
  1. MLflow pip推論 vs 明示的pip指定
  2. _coerce_types のオーバーヘッドと早期return効果
  3. LRU cache によるデータ再利用効果
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# Experiment 1: MLflow pip推論 vs 明示的指定
# ============================================================

def benchmark_mlflow_pip() -> dict:
    """MLflow log_model の pip推論オーバーヘッドを計測。"""
    try:
        import mlflow
    except ImportError:
        return {"error": "mlflow not installed"}

    # 最小モデル
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
            for i in range(5):
                mlflow.lightgbm.log_model(model, f"test_default_{i}")
            default_time = time.time() - t0

            # 明示的 pip指定
            pip_reqs = ["lightgbm", "scikit-learn", "pandas", "numpy", "joblib"]
            t1 = time.time()
            for i in range(5):
                mlflow.lightgbm.log_model(
                    model, f"test_explicit_{i}",
                    pip_requirements=pip_reqs,
                )
            explicit_time = time.time() - t1

    per_call_default = default_time / 5
    per_call_explicit = explicit_time / 5
    per_call_saving = per_call_default - per_call_explicit
    total_saving_26 = per_call_saving * 26

    results = {
        "default_per_call_seconds": round(per_call_default, 3),
        "explicit_per_call_seconds": round(per_call_explicit, 3),
        "per_call_saving_seconds": round(per_call_saving, 3),
        "total_saving_26_calls_seconds": round(total_saving_26, 1),
        "speedup": round(per_call_default / per_call_explicit, 2) if per_call_explicit > 0 else 0,
    }

    print(f"  Default: {per_call_default:.3f}s/call")
    print(f"  Explicit: {per_call_explicit:.3f}s/call")
    print(f"  Per-call saving: {per_call_saving:.3f}s")
    print(f"  Total saving (26 calls): {total_saving_26:.1f}s")
    print(f"  Speedup: {results['speedup']:.2f}x")

    return results


# ============================================================
# Experiment 2: _coerce_types オーバーヘッド
# ============================================================

def benchmark_coerce_types() -> dict:
    """_coerce_types の実行時間と早期return効果を計測。"""
    from db.readers import _coerce_types

    # テストデータ: object列を含む (旧ETL形式をシミュレート)
    n = 100_000
    df_old = pd.DataFrame({
        "race_id": [f"R{i:010d}" for i in range(n)],
        "race_date": pd.date_range("2020-01-01", periods=n, freq="h"),
        "kyori": np.random.randint(1000, 3600, n).astype(str),  # object (should be numeric)
        "kisyucode": [f"J{i % 1000:06d}" for i in range(n)],
        "score": np.random.rand(n).astype(str),  # object (should be numeric)
        "bamei": [f"horse_{i}" for i in range(n)],  # string (should stay string)
    })

    # テストデータ: 既に正しい型 (新ETL形式)
    df_new = df_old.copy()
    df_new["kyori"] = pd.to_numeric(df_new["kyori"])
    df_new["score"] = pd.to_numeric(df_new["score"])

    results = {}

    # 旧ETL形式での _coerce_types
    t0 = time.time()
    for _ in range(5):
        _ = _coerce_types(df_old.copy())
    old_time = (time.time() - t0) / 5
    results["old_etl_seconds"] = round(old_time, 4)
    print(f"  Old ETL (object cols): {old_time:.4f}s/call")

    # 新ETL形式での _coerce_types
    t1 = time.time()
    for _ in range(5):
        _ = _coerce_types(df_new.copy())
    new_time = (time.time() - t1) / 5
    results["new_etl_seconds"] = round(new_time, 4)
    print(f"  New ETL (correct types): {new_time:.4f}s/call")

    # 早期return の効果をシミュレート
    has_object = any(df_new[c].dtype == object for c in df_new.columns
                     if c not in _coerce_types.__code__.co_consts or True)
    # 簡易チェック: 全列が正しい型なら早期return可能か
    string_cols = {"race_id", "kisyucode", "bamei"}  # _STRING_COLUMNS の一部
    all_correct = all(
        df_new[col].dtype != object
        for col in df_new.columns
        if col not in string_cols and col != "race_date"
    )
    results["can_skip_coerce"] = all_correct
    results["potential_saving_per_call"] = round(old_time - new_time, 4)
    results["calls_per_backtest"] = 6  # load_races, load_entries, load_odds x2, load_payouts, etc.
    results["total_saving"] = round((old_time - new_time) * 6, 4)
    print(f"  Can skip: {all_correct}")
    print(f"  Per-call saving: {old_time - new_time:.4f}s")
    print(f"  Total saving (6 calls): {(old_time - new_time) * 6:.4f}s")

    return results


# ============================================================
# Experiment 3: LRU cache 効果
# ============================================================

def benchmark_data_caching() -> dict:
    """ParquetStore の繰り返し読み込みコストとキャッシュ効果を計測。"""
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    results = {}

    # 実データの読み込み時間を計測
    tables_to_test = [
        ("raw", "races"),
        ("raw", "entries"),
        ("raw", "payouts"),
    ]

    for table, subpath in tables_to_test:
        if not store.exists(table, subpath):
            print(f"  SKIP {table}/{subpath} (not found)")
            continue

        # 初回ロード (コールド)
        t0 = time.time()
        df1 = store.read(table, subpath)
        cold_time = time.time() - t0

        # 2回目ロード (OSファイルキャッシュ)
        t1 = time.time()
        df2 = store.read(table, subpath)
        warm_time = time.time() - t1

        # メモリキャッシュ (単純dict)
        cache: dict = {}
        key = f"{table}/{subpath}"

        def cached_read(k: str) -> pd.DataFrame:
            if k not in cache:
                cache[k] = store.read(table, subpath)
            return cache[k]

        t2 = time.time()
        for _ in range(10):
            _ = cached_read(key)
        cached_time = (time.time() - t2) / 10

        results[f"{table}/{subpath}"] = {
            "rows": len(df1),
            "cold_seconds": round(cold_time, 3),
            "warm_seconds": round(warm_time, 3),
            "cached_seconds": round(cached_time, 6),
            "cold_to_warm_speedup": round(cold_time / warm_time, 1) if warm_time > 0 else 0,
            "cold_to_cached_speedup": round(cold_time / max(cached_time, 0.0001), 0),
        }
        print(f"  {table}/{subpath}: {len(df1):,} rows, "
              f"cold={cold_time:.3f}s, warm={warm_time:.3f}s, "
              f"cached={cached_time:.6f}s")

    return results


def main() -> None:
    print("=" * 70)
    print("  Spike 005: P1 Quick-Wins Verification")
    print("=" * 70)

    all_results: dict = {}

    print("\n--- Experiment 1: MLflow pip inference ---")
    try:
        all_results["mlflow_pip"] = benchmark_mlflow_pip()
    except Exception as e:
        all_results["mlflow_pip"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n--- Experiment 2: _coerce_types overhead ---")
    try:
        all_results["coerce_types"] = benchmark_coerce_types()
    except Exception as e:
        all_results["coerce_types"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n--- Experiment 3: Data caching ---")
    try:
        all_results["data_caching"] = benchmark_data_caching()
    except Exception as e:
        all_results["data_caching"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # 結果保存
    output_path = Path(__file__).parent / "quick_wins_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    # サマリー
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if "mlflow_pip" in all_results and "error" not in all_results["mlflow_pip"]:
        ml = all_results["mlflow_pip"]
        print(f"  MLflow pip saving: {ml['total_saving_26_calls_seconds']:.1f}s "
              f"({ml['speedup']:.1f}x per call)")

    if "coerce_types" in all_results and "error" not in all_results["coerce_types"]:
        ct = all_results["coerce_types"]
        print(f"  _coerce_types: old={ct['old_etl_seconds']:.4f}s, "
              f"new={ct['new_etl_seconds']:.4f}s, "
              f"can_skip={ct['can_skip_coerce']}")

    if "data_caching" in all_results and "error" not in all_results["data_caching"]:
        dc = all_results["data_caching"]
        for key, data in dc.items():
            if isinstance(data, dict) and "cold_seconds" in data:
                print(f"  {key}: cold={data['cold_seconds']:.3f}s, "
                      f"cached={data['cached_seconds']:.6f}s, "
                      f"speedup={data.get('cold_to_cached_speedup', 'N/A')}x")

    # 推定累積効果
    ml_saving = 0
    ct_saving = 0
    cache_saving = 0

    if "mlflow_pip" in all_results and isinstance(all_results["mlflow_pip"], dict):
        ml_saving = all_results["mlflow_pip"].get("total_saving_26_calls_seconds", 0)
    if "coerce_types" in all_results and isinstance(all_results["coerce_types"], dict):
        ct_saving = all_results["coerce_types"].get("total_saving", 0)
    if "data_caching" in all_results and isinstance(all_results["data_caching"], dict):
        for key, data in all_results["data_caching"].items():
            if isinstance(data, dict) and "cold_seconds" in data:
                # odds時系列の重複ロード (3回 → 1回)
                cache_saving += data["cold_seconds"] * 2  # 2回分節約

    total_saving = ml_saving + ct_saving + cache_saving
    print(f"\n  Estimated total P1 saving: ~{total_saving:.0f}s ({total_saving/60:.1f}min)")


if __name__ == "__main__":
    main()
