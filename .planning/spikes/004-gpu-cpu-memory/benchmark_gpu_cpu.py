"""Spike 004: GPU/CPU/メモリ効率化のベンチマーク。

使い方:
  python .planning/spikes/004-gpu-cpu-memory/benchmark_gpu_cpu.py

3つの実験を実行:
  1. LightGBM CPU vs GPU 訓練速度比較
  2. Categorical dtype による groupby/merge 高速化
  3. ThreadPoolExecutor 並列度の最適化
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd


def _make_dataset(n_rows: int, n_features: int, n_categorical: int = 0) -> tuple:
    """ベンチマーク用データセット生成。"""
    X = np.random.rand(n_rows, n_features)
    # binary target (like hit model)
    y = (np.random.rand(n_rows) > 0.85).astype(np.float64)
    feature_names = [f"f{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # categorical列を追加 (高カーディナリティ)
    for i in range(n_categorical):
        n_cats = 500 + i * 200
        df[f"cat_{i}"] = np.random.choice([f"id_{j}" for j in range(n_cats)], n_rows)

    return df, feature_names


# ============================================================
# Experiment 1: LightGBM CPU vs GPU
# ============================================================

def benchmark_lgbm_device(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """CPU vs GPU でLightGBM訓練を比較。"""
    results = {}
    X = df[feature_names].values
    y = df["target"].values

    # 代表的な訓練パラメータ (keiba-ai の hit model に近い設定)
    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "is_unbalance": True,
        "feature_fraction": 0.7,
        "verbose": -1,
    }

    dataset_sizes = [
        ("small", 50_000, 50),
        ("medium", 200_000, 50),
        ("large", 200_000, 200),
    ]

    for label, n_rows, n_features in dataset_sizes:
        X = np.random.rand(n_rows, n_features)
        y = (np.random.rand(n_rows) > 0.85).astype(np.float64)
        ds = lgb.Dataset(X, label=y)

        size_results = {}

        for device in ["cpu", "gpu"]:
            params = {**base_params, "device": device}

            # Warmup
            try:
                lgb.train(params, ds, num_boost_round=5)
            except Exception as e:
                size_results[device] = {"error": str(e)}
                continue

            # Benchmark (3 runs)
            times = []
            for _ in range(3):
                t0 = time.time()
                model = lgb.train(params, ds, num_boost_round=100)
                elapsed = time.time() - t0
                times.append(elapsed)

            avg = sum(times) / len(times)
            size_results[device] = {
                "avg_seconds": round(avg, 3),
                "runs": [round(t, 3) for t in times],
                "boost_rounds": 100,
            }

            # 予測値の一致確認 (CPU vs GPU)
            if device == "gpu" and "cpu" in size_results:
                cpu_pred = lgb.train(
                    {**base_params, "device": "cpu", "verbose": -1},
                    ds, num_boost_round=100,
                ).predict(X[:100])
                gpu_pred = model.predict(X[:100])
                # deterministic seedなし → 完全一致は期待できないが相関を確認
                corr = np.corrcoef(cpu_pred, gpu_pred)[0, 1]
                size_results["correlation"] = round(corr, 6)

        if "cpu" in size_results and "gpu" in size_results:
            cpu_t = size_results["cpu"]["avg_seconds"]
            gpu_t = size_results["gpu"]["avg_seconds"]
            size_results["speedup"] = round(cpu_t / gpu_t, 2) if gpu_t > 0 else 0

        results[label] = size_results
        print(f"  {label} ({n_rows:,} rows × {n_features} features): {size_results}")

    return results


# ============================================================
# Experiment 2: Categorical dtype高速化
# ============================================================

def benchmark_categorical(n_rows: int = 500_000) -> dict:
    """object vs category dtype で groupby/merge を比較。"""
    results = {}

    # 高カーディナリティ文字列列
    n_cats_race = 20_000
    n_cats_horse = 5_000
    n_cats_jockey = 1_000

    df = pd.DataFrame({
        "race_id": [f"R{i % n_cats_race:06d}" for i in range(n_rows)],
        "kettonum": [f"H{i % n_cats_horse:06d}" for i in range(n_rows)],
        "kisyucode": [f"J{i % n_cats_jockey:04d}" for i in range(n_rows)],
        "value": np.random.rand(n_rows),
    })

    benchmarks = {
        "groupby_single": lambda d: d.groupby("race_id")["value"].mean(),
        "groupby_multi": lambda d: d.groupby(["race_id", "kettonum"])["value"].mean(),
        "merge": lambda d: d.merge(
            d.groupby("race_id")["value"].mean().reset_index(),
            on="race_id", how="left",
        ),
        "isin": lambda d: d[d["race_id"].isin(d["race_id"].unique()[:1000])],
        "value_counts": lambda d: d["kisyucode"].value_counts(),
    }

    for name, fn in benchmarks.items():
        # object dtype
        df_obj = df.copy()
        for col in ["race_id", "kettonum", "kisyucode"]:
            df_obj[col] = df_obj[col].astype(str)

        t0 = time.time()
        _ = fn(df_obj)
        obj_time = time.time() - t0

        # category dtype
        df_cat = df.copy()
        for col in ["race_id", "kettonum", "kisyucode"]:
            df_cat[col] = df_cat[col].astype("category")

        t1 = time.time()
        _ = fn(df_cat)
        cat_time = time.time() - t1

        speedup = obj_time / cat_time if cat_time > 0 else float("inf")
        results[name] = {
            "object_seconds": round(obj_time, 4),
            "category_seconds": round(cat_time, 4),
            "speedup": round(speedup, 2),
        }
        print(f"  {name}: object={obj_time:.4f}s, category={cat_time:.4f}s, "
              f"speedup={speedup:.1f}x")

    # メモリ使用量比較
    mem_obj = df.memory_usage(deep=True).sum() / 1024 / 1024
    df_cat = df.copy()
    for col in ["race_id", "kettonum", "kisyucode"]:
        df_cat[col] = df_cat[col].astype("category")
    mem_cat = df_cat.memory_usage(deep=True).sum() / 1024 / 1024
    results["memory_reduction"] = {
        "object_mb": round(mem_obj, 1),
        "category_mb": round(mem_cat, 1),
        "reduction_pct": round((1 - mem_cat / mem_obj) * 100, 1),
    }
    print(f"  memory: object={mem_obj:.1f}MB, category={mem_cat:.1f}MB, "
          f"reduction={results['memory_reduction']['reduction_pct']:.1f}%")

    return results


# ============================================================
# Experiment 3: ThreadPoolExecutor 並列度
# ============================================================

def benchmark_parallelism(n_tasks: int = 4) -> dict:
    """LightGBM訓練の並列度をベンチマーク。"""
    results = {}

    # 典型的なkeiba-aiサブモデル訓練をシミュレート
    X = np.random.rand(100_000, 50)
    y = (np.random.rand(100_000) > 0.85).astype(np.float64)
    ds = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.7,
        "verbose": -1,
    }

    def train_one(task_id: int) -> float:
        t0 = time.time()
        lgb.train({**params, "num_threads": 2}, ds, num_boost_round=100)
        return time.time() - t0

    for max_workers in [1, 2, 4]:
        t0 = time.time()
        if max_workers == 1:
            times = [train_one(i) for i in range(n_tasks)]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                times = list(pool.map(train_one, range(n_tasks)))
        total = time.time() - t0

        results[f"workers_{max_workers}"] = {
            "total_seconds": round(total, 2),
            "per_task_avg": round(sum(times) / len(times), 2),
            "num_tasks": n_tasks,
        }
        print(f"  workers={max_workers}: total={total:.2f}s, "
              f"per_task={sum(times)/len(times):.2f}s")

    # Sequential vs 2-worker の比較
    seq_time = results["workers_1"]["total_seconds"]
    par2_time = results["workers_2"]["total_seconds"]
    results["parallel_speedup"] = round(seq_time / par2_time, 2) if par2_time > 0 else 0
    print(f"  Parallel speedup (2 workers): {results['parallel_speedup']:.2f}x")

    return results


def main() -> None:
    print("=" * 70)
    print("  Spike 004: GPU/CPU/Memory Performance Benchmark")
    print("=" * 70)

    all_results: dict = {}

    print("\n--- Experiment 1: LightGBM CPU vs GPU ---")
    try:
        df, feat_names = _make_dataset(100_000, 50)
        all_results["lgbm_device"] = benchmark_lgbm_device(df, feat_names)
    except Exception as e:
        all_results["lgbm_device"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n--- Experiment 2: Categorical dtype ---")
    try:
        all_results["categorical"] = benchmark_categorical()
    except Exception as e:
        all_results["categorical"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n--- Experiment 3: ThreadPoolExecutor parallelism ---")
    try:
        all_results["parallelism"] = benchmark_parallelism()
    except Exception as e:
        all_results["parallelism"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # 結果保存
    output_path = Path(__file__).parent / "benchmark_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    # サマリー
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if "lgbm_device" in all_results and "error" not in all_results["lgbm_device"]:
        for size, data in all_results["lgbm_device"].items():
            if "speedup" in data:
                print(f"  LightGBM GPU speedup ({size}): {data['speedup']:.2f}x")

    if "categorical" in all_results and "error" not in all_results["categorical"]:
        cat = all_results["categorical"]
        avg_speedup = sum(
            v["speedup"] for k, v in cat.items()
            if isinstance(v, dict) and "speedup" in v
        ) / max(1, sum(1 for k, v in cat.items() if isinstance(v, dict) and "speedup" in v))
        print(f"  Categorical avg speedup: {avg_speedup:.1f}x")
        if "memory_reduction" in cat:
            print(f"  Memory reduction: {cat['memory_reduction']['reduction_pct']:.1f}%")

    if "parallelism" in all_results and "error" not in all_results["parallelism"]:
        par = all_results["parallelism"]
        print(f"  2-worker speedup: {par.get('parallel_speedup', 'N/A')}x")


if __name__ == "__main__":
    main()
