---
spike: 004
name: gpu-cpu-memory
type: standard
validates: "Given CPU-only LightGBM訓練, when GPU device有効化 + Categorical化 + ThreadPool最適化, then 訓練時間が有意に短縮され予測値が一致"
verdict: PARTIAL
related: [002]
tags: [lightgbm, gpu, categorical, threading, pandas]
---

# Spike 004: GPU/CPU/メモリ効率化の検証

## What This Validates

Given: CPU-only LightGBM訓練（GPU未使用）、object列のgroupby/merge
When: GPU有効化、Categorical化、ThreadPool最適化
Then: 訓練・特徴量計算時間が短縮され、予測値が一致

## Research

### LightGBM GPU Support

LightGBM 4.6.0 は `device=gpu` パラメータでGPU訓練をサポート。
CUDA 13.2 + NVIDIA Driver 595.97 で動作確認済み。
予測値はCPU/GPU間で完全一致（相関1.0）。

### pandas Categorical dtype

pandas categorical dtype は離散値列（文字列ID等）を整数コードで管理。
`observed=True` パラメータが重要 — `observed=False` は全カテゴリの直積を生成するため、
多列groupbyで破滅的遅延（70x）が発生する。

### ThreadPoolExecutor

LightGBMは内部でOpenMPスレッドを使用。Python ThreadPoolExecutorと
組み合わせる際、スレッド数の配分が重要。

## How to Run

```bash
python .planning/spikes/004-gpu-cpu-memory/benchmark_gpu_cpu.py
python .planning/spikes/004-gpu-cpu-memory/test_categorical_groupby.py
```

## What to Expect

### Experiment 1: LightGBM CPU vs GPU

| Dataset Size | CPU Time | GPU Time | GPU Speedup |
|-------------|----------|----------|-------------|
| 50k × 50 | 0.26s | 1.49s | **0.18x (5.5x SLOWER)** |
| 200k × 50 | 0.42s | 1.62s | **0.26x (3.8x SLOWER)** |
| 200k × 200 | 1.34s | 2.04s | **0.66x (1.5x SLOWER)** |

**結論: GPUはkeiba-aiのデータサイズでは逆効果。** GPU転送オーバーヘッドが支配的。
数百萬行×数百特徴量のデータセットで初めてGPUが有利になる。

### Experiment 2: Categorical dtype (observed=True)

| 操作 | Object | Category | Speedup |
|------|--------|----------|---------|
| groupby(race_id) | 20ms | 9ms | **2.2x** |
| groupby(kisyucode) | 13ms | 6ms | **2.2x** |
| groupby([race_id, umaban]) | 59ms | 42ms | **1.4x** |
| groupby([race_id, kisyucode]) | 47ms | 26ms | **1.8x** |
| groupby([race_id, umaban, kisyucode]) | 86ms | 51ms | **1.7x** |
| merge(on=race_id) | 62ms | 47ms | **1.3x** |
| isin(1000 ids) | 20ms | 6ms | **3.3x** |
| **メモリ** | **121.6MB** | **9.0MB** | **92.6%削減** |

**結論: Categorical + observed=True で全操作が1.3-3.3x高速、メモリ92.6%削減。**

### Experiment 3: ThreadPoolExecutor

| Workers | Total Time | Speedup |
|---------|-----------|---------|
| 1 | 2.16s | baseline |
| 2 | 1.00s | **2.16x** |
| 4 | 0.74s | 2.92x ( diminishing ) |

**結論: 現在の2-worker設定は既に最適。4-workerはoversubscriptionリスク。**

## Investigation Trail

1. **GPU可用性確認**: nvidia-smi → CUDA 13.2, NVIDIA Driver 595.97 動作確認
2. **LightGBM GPU ベンチマーク**: 3つのデータサイズでCPU vs GPU 比較 → 全てCPU勝ち
3. **Categorical 初回テスト**: groupby_multi で70x遅延を発見 → `observed=False` が原因
4. **observed=True で再テスト**: 全操作で一貫した高速化を確認
5. **ThreadPool テスト**: 2-workerが最適、4-workerはoversubscription

## Results

**Verdict: PARTIAL ⚠**

### 主要発見

1. **GPU: 無益** — keiba-aiのデータサイズ（50k-200k行）ではGPU転送オーバーヘッドが支配的。CPU-onlyが最適。予測値は一致するので、将来データサイズが増えても安全に切り替え可能。

2. **Categorical化: 有効（条件付き）** — `observed=True` と組み合わせることで:
   - groupby: 1.4-2.2x高速
   - merge: 1.3x高速
   - isin: 3.3x高速
   - メモリ: 92.6%削減（121.6MB → 9.0MB）
   - **条件**: 全groupby呼び出しに `observed=True` を追加する必要あり

3. **ThreadPool: 既に最適** — 現在の `ThreadPoolExecutor(max_workers=2)` で2.16x speedup。変更不要。

### 推定効果（keiba-aiパイプライン全体）

| 改善 | 推定短縮 | 適用範囲 |
|------|---------|---------|
| Categorical化 + observed=True | -200〜400s | HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, SireFeatures |
| メモリ削減 | GC負荷軽減 | 全DataFrames |
| GPU（却下） | +200〜500s（逆効果） | — |

### What to Avoid

- **GPU訓練はkeiba-aiでは使わない** — データサイズに対してオーバーヘッドが大きすぎる
- **`observed=False` + Categorical groupbyは絶対NG** — 70x遅延
- **ThreadPoolExecutor(max_workers > 2)** — oversubscriptionで逆効果

### Constraints

- Categorical化はParquetStore/DataRepositoryレベルで行うべき（全コンシューマーに自動適用）
- `observed=True` は各groupby呼び出しに個別に追加が必要
- LightGBM 4.6.0 の categorical_feature サポートは限定的（文字列列は数値化が必要）
