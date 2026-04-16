# Place EV Hit/Return 分離修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PlaceTwoStageModel の hit model (Stage A) と return model (Stage B) の特徴量を分離し、hit model の確率判別力を復旧させることで ROI 31% → 100%+ を達成する。

**Architecture:** `FEATURE_COLS` の単一リストを `HIT_FEATURE_COLS` と `RETURN_FEATURE_COLS` の2つに分割する。HIT_FEATURE_COLS は `fukuoddslow`, `tanodds` を含まず元の判別力を維持する。RETURN_FEATURE_COLS は `fukuoddslow`, `tanodds` を含み配当予測を正確化する。`_prepare_features()` は `use_cols` 引数を受け取るように変更し、各メソッドが適切な特徴量リストを使用する。

**Tech Stack:** LightGBM, pandas, numpy, pytest

---

## 根本原因の再確認

前回の修正 (`bc8dc7d`) で `PlaceTwoStageModel.FEATURE_COLS` に `fukuoddslow`, `tanodds` を追加した。この特徴量リストは hit model (Stage A: 確率分類) と return model (Stage B: 配当回帰) の両方で共有されている。

`fukuoddslow` は市場の複勝確率評価を直接エンコードするため、binary classifier (hit model) にとって強力すぎる信号となる。モデルがこの単一特徴量に過度依存し、他の特徴量の微細な信号を無視。結果として p_place_pred が 0.17-0.40 の範囲に圧縮され、確率判別力が完全に失われた。

| 指標 | 修正前 | 前回修正後 (異常) |
|------|--------|-----------------|
| p_place_pred 範囲 | 0.02-0.95 | 0.175-0.399 |
| 平均人気順位 | 2.9 | 13.4 (大穴) |
| 的中率 | 69.8% | 3.1% |
| ROI | 72.8% | 31.2% |

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/two_stage_return_model.py` | Modify | FEATURE_COLS → HIT_FEATURE_COLS + RETURN_FEATURE_COLS に分割 |
| `tests/test_two_stage_return_model.py` | Modify | テストの特徴量リスト検証を更新 |

---

## Task 1: PlaceTwoStageModel の特徴量リストを HIT/RETURN 分離

**Files:**
- Modify: `src/models/two_stage_return_model.py:171-299`
- Modify: `tests/test_two_stage_return_model.py`

**現在のコード構造 (lines 171-299):**
- `FEATURE_COLS` (line 181): 23列。hit と return で共用
- `_prepare_features(df)` (line 214): `self.FEATURE_COLS` を参照
- `train_hit_model(df)` (line 225): `self._prepare_features(df)` を使用
- `train_return_model(df)` (line 250): `self._prepare_features(df)` を使用
- `predict_ev(df)` (line 287): `self._prepare_features(df)` を使用

### 1-A: テストファースト

- [ ] **Step 1: テストフィクスチャの確認**

`tests/test_two_stage_return_model.py` の `feature_df` フィクスチャは既に `fukuoddslow`, `tanodds`, `p_ability_place` を含んでいる (前回の修正で追加済み)。このまま使用可能。

- [ ] **Step 2: 特徴量分離の検証テストを追加**

`tests/test_two_stage_return_model.py` の `TestPlaceTwoStageModel` クラスに追加:

```python
def test_hit_and_return_features_separated(self):
    """Hit model と Return model で特徴量が分離されていること"""
    # Return model のみがオッズ特徴量を持つ
    assert "fukuoddslow" in PlaceTwoStageModel.RETURN_FEATURE_COLS
    assert "tanodds" in PlaceTwoStageModel.RETURN_FEATURE_COLS
    # Hit model はオッズ特徴量を持たない
    assert "fukuoddslow" not in PlaceTwoStageModel.HIT_FEATURE_COLS
    assert "tanodds" not in PlaceTwoStageModel.HIT_FEATURE_COLS
    # p_ability_place は両方に含まれる
    assert "p_ability_place" in PlaceTwoStageModel.HIT_FEATURE_COLS
    assert "p_ability_place" in PlaceTwoStageModel.RETURN_FEATURE_COLS
```

- [ ] **Step 3: 旧テストの更新**

`test_place_feature_cols_include_place_specific` (前回追加) を更新して RETURN_FEATURE_COLS を検証するように変更:

```python
def test_place_return_feature_cols_include_place_specific(self):
    """Return model should have place-specific features"""
    assert "fukuoddslow" in PlaceTwoStageModel.RETURN_FEATURE_COLS
    assert "tanodds" in PlaceTwoStageModel.RETURN_FEATURE_COLS
    assert "p_ability_place" in PlaceTwoStageModel.RETURN_FEATURE_COLS
    # Win特徴量も全て含む
    for col in WinTwoStageModel.FEATURE_COLS:
        assert col in PlaceTwoStageModel.RETURN_FEATURE_COLS
```

- [ ] **Step 4: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_hit_and_return_features_separated -v`
Expected: FAIL — `PlaceTwoStageModel` has no attribute `HIT_FEATURE_COLS`

### 1-B: 実装

- [ ] **Step 5: FEATURE_COLS を HIT_FEATURE_COLS + RETURN_FEATURE_COLS に分割**

`src/models/two_stage_return_model.py` line 181 の `FEATURE_COLS` を2つのリストに置換:

```python
    # --- Hit model (Stage A): 確率分類用 ---
    # fukuoddslow, tanodds は含めない (確率判別力を維持するため)
    HIT_FEATURE_COLS: list[str] = [
        # Stage1 出力
        "p_ability_win",
        "p_ability_place",             # PlaceAbilityModel 出力
        # Market Model 正規化差分
        "signed_log_error_win",
        "abs_log_error_win",
        # オッズ変化率
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        "popularity_change_30_10",
        # 市場歪み
        "market_entropy",
        "popularity_rank",
        "overround",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        # FLB slope
        "odds_skewness",
    ]

    # --- Return model (Stage B): 配当回帰用 ---
    # fukuoddslow はターゲットに近いため最も重要な特徴量
    RETURN_FEATURE_COLS: list[str] = [
        # Stage1 出力
        "p_ability_win",
        "p_ability_place",             # PlaceAbilityModel 出力
        # Market Model 正規化差分
        "signed_log_error_win",
        "abs_log_error_win",
        # 複勝・単勝オッズ (return model のみ)
        "fukuoddslow",                 # 複勝オッズ (最重要特徴量)
        "tanodds",                     # 単勝オッズ
        # オッズ変化率
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        "popularity_change_30_10",
        # 市場歪み
        "market_entropy",
        "popularity_rank",
        "overround",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        # FLB slope
        "odds_skewness",
    ]

    # 後方互換: FEATURE_COLS は return model のリストを返す (最も情報量が多いため)
    FEATURE_COLS: list[str] = RETURN_FEATURE_COLS
```

- [ ] **Step 6: _prepare_features() に use_cols 引数を追加**

`src/models/two_stage_return_model.py` line 214 を変更:

```python
    # Before (line 214):
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.FEATURE_COLS].copy()

    # After:
    def _prepare_features(
        self, df: pd.DataFrame, *, use_cols: list[str] | None = None
    ) -> pd.DataFrame:
        cols = use_cols or self.FEATURE_COLS
        features = df[cols].copy()
```

メソッドの残りの部分 (Int64→float64 変換, category 変換) は変更不要。`use_cols` に含まれる列だけが処理される。

- [ ] **Step 7: train_hit_model で HIT_FEATURE_COLS を使用**

`src/models/two_stage_return_model.py` line 229 を変更:

```python
    # Before (line 229):
    features = self._prepare_features(df)

    # After:
    features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
```

- [ ] **Step 8: train_return_model で RETURN_FEATURE_COLS を使用**

`src/models/two_stage_return_model.py` line 256 を変更:

```python
    # Before (line 256):
    features = self._prepare_features(hit_df)

    # After:
    features = self._prepare_features(hit_df, use_cols=self.RETURN_FEATURE_COLS)
```

- [ ] **Step 9: predict_ev で各モデルに適切な特徴量を渡す**

`src/models/two_stage_return_model.py` line 290 を変更:

```python
    # Before (line 290):
    features = self._prepare_features(df)

    # After:
    hit_features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
    ret_features = self._prepare_features(df, use_cols=self.RETURN_FEATURE_COLS)
```

そして line 296-297 を変更:

```python
    # Before (lines 296-297):
    df["p_place_pred"] = self.hit_model.predict(features, num_iteration=hit_iter)
    df["e_return_place_pred"] = self.return_model.predict(features, num_iteration=ret_iter)

    # After:
    df["p_place_pred"] = self.hit_model.predict(hit_features, num_iteration=hit_iter)
    df["e_return_place_pred"] = self.return_model.predict(ret_features, num_iteration=ret_iter)
```

### 1-C: 検証

- [ ] **Step 10: テストを実行して通過を確認**

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: ALL PASS

- [ ] **Step 11: 全テストを実行して回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 12: コミット**

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py
git commit -m "fix: PlaceTwoStageModel の HIT/RETURN 特徴量を分離 (hit model の判別力復旧)"
```

---

## Task 2: バックテストで検証

**Files:** (変更なし、検証のみ)

- [ ] **Step 1: バックテストを実行**

```bash
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4 \
  --report
```

所要時間: ~57分/年 × 3年 ≈ 3時間

- [ ] **Step 2: 結果を確認**

`data/backtest/multi_year_result.json` で以下を確認:

| 確認項目 | 合格基準 |
|----------|---------|
| p_place_pred の範囲 | 0.02-0.95 (判別力あり) |
| 平均人気順位 | 1-4位の範囲 |
| 的中率 | > 50% |
| 全体 ROI | > 100% |
| e_return_place_pred / actual 比 | 0.8-1.2 (過大評価の是正) |

- [ ] **Step 3: 結果を記録**

確認結果をメモリーに保存。改善の推移:
- 初期: ROI 73% (e_return 2.2倍過大評価)
- 前回修正: ROI 31% (hit model 判別力喪失)
- 今回修正: ROI ???% (hit model 復旧 + return model 是正)
