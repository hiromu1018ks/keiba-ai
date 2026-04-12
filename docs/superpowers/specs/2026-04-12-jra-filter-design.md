# JRA-only Filter Design

Date: 2026-04-12

## Problem

DirtサブモデルのMarket Model (`market_dirt.lgb`) が1木しか学習されず、
連鎖的に `place_ret_dirt`, `ev_corrector_p_dirt` も1木で早期停止する。

### Root Cause

Dirtレースの59%がNAR (地方競馬) であり、これらにオッズ時系列データがほぼ存在しない:

| 指標 | Dirt (全体) | Turf (全体) |
|------|------------|------------|
| 総レース数 | 16,413 | 6,853 |
| NARレース | 9,742 (59%) | 366 (5%) |
| TS保有レース | 5,024 (31%) | 4,965 (73%) |
| tanodds有効率 | 36.5% | 73.5% |

NARレースが大量に混入 → `p_market_win_adj` がNaN → Market Modelの
ターゲット欠損 → LightGBMが少数の有効データだけで学習 → 1木で早期停止。

### JRA-onlyに絞った場合

| 指標 | Dirt (JRA) | Turf (JRA) |
|------|-----------|-----------|
| レース数 | 6,671 | 6,654 |
| エントリ数 | 94,886 | 89,462 |
| tanodds有効率 (2022-2024) | **~97%** | **~97%** |

Dirt/Turfがほぼ同量でバランスが良く、オッズカバレッジも97%に改善。

## Solution

学習・バックテスト両方でNARレース (jyocd 30以上) を除外する。

### Changes

#### 1. Training Pipeline (`src/pipelines/training_pipeline.py`)

`train()` メソッド内、surface分割の前にJRAフィルタを追加:

```python
# NARレース除外 (jyocd 01-10 = JRA競馬場のみ)
if "jyocd" in feat_df.columns:
    jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
    before = len(feat_df)
    feat_df = feat_df[jyocd_int.between(1, 10)]
    after = len(feat_df)
    if after < before:
        logger.info("JRA filter: %d -> %d entries (removed %d NAR)",
                     before, after, before - after)
```

**位置**: surface分割ループ (`for surface in ["turf", "dirt"]`) の直前。

#### 2. Backtest Engine (`src/backtest/engine.py`)

`run()` メソッド内、特徴量生成後にJRAフィルタを追加:

```python
# NARレース除外
if "jyocd" in feat_df.columns:
    jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
    feat_df = feat_df[jyocd_int.between(1, 10)]
```

**位置**: `add_distance_band_features()` の直後、レースループの前。

### Not Changed

- **推論パイプライン** (`src/backtest/race_predictor.py`): surfaceでサブモデルを
  選択するだけ。NARレースは `surface="dirt"` でdirtモデルが使われる。
  学習が正常になれば予測も改善される。

- **Market Model特徴量**: 変更なし。JRAフィルタでカバレッジ97%になれば
  正常に学習できると予想。

- **オッズパイプライン**: スナップショット置換ロジックは変更しない。
  (Snapshotは確定オッズと同一のため、pre-post oddsを優先する現行ロジックが正しい)

## Expected Outcome

- `market_dirt.lgb`: 1木 → 100+木 (正常な学習)
- `place_ret_dirt.lgb`: 1木 → 100+木 (正常な学習)
- `ev_corrector_p_dirt.lgb`: 1木 → 10+木 (正常な学習)
- Dirt/Turfの予測精度がバランスよく向上

## Verification

バックテストで以下を確認:
1. Market Model の木数が正常 (>1) であること
2. Dirtモデルの予測に馬個別の差異が反映されていること
3. バックテストROIが改善していること

## Rollback

JRAフィルタの追加のみなので、`git revert` で簡単にロールバック可能。
