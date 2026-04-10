# MLモデル改善 (A群) 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MarketModelのデータリーク修正、体重変化・休養期間特徴量の追加、バックテスト/本番ベットロジックの切替可能化

**Architecture:** 4つの独立した改善を順次適用。A1(バグ修正) → A2(特徴量) → A3(特徴量) → A4(ベットロジック)。各タスクはテスト→実装→コミットのTDDサイクル。

**Tech Stack:** Python 3.11, LightGBM, pandas, numpy, pytest, unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-10-model-improvement-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/market_model.py` | Modify | A1: random split → time-based split |
| `src/pipelines/training_pipeline.py` | Modify | A1: sort by race_date before market.train() |
| `src/features/horse_history_features.py` | Modify | A2/A3: weight stats, rest period computation |
| `src/features/feature_engine.py` | Modify | A2: weight_zscore, weight_change_zone mapping |
| `src/models/stage1_ability_model.py` | Modify | A2/A3: FEATURE_COLS に新特徴量追加 |
| `src/models/place_ability_model.py` | Modify | A2/A3: FEATURE_COLS に新特徴量追加 |
| `src/backtest/race_predictor.py` | Modify | A4: betting_mode分岐 |
| `src/backtest/engine.py` | Modify | A4: betting_mode パラメータ伝播 |
| `scripts/run_backtest.py` | Modify | A4: --betting-mode CLI引数 |
| `tests/test_market_model.py` | Modify | A1: 時間ベース分割テスト追加 |
| `tests/test_horse_history_features.py` | Modify | A2/A3: 体重統計・休養期間テスト追加 |
| `tests/test_race_predictor.py` | Modify | A4: flat/kelly モードテスト |
| `tests/test_backtest_engine.py` | Modify | A4: betting_mode 伝播テスト |

---

## Task 1: A1 — MarketModel 時間ベース分割

**Files:**
- Modify: `src/models/market_model.py:56-60`
- Modify: `src/pipelines/training_pipeline.py:246-249`
- Modify: `tests/test_market_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_market_model.py` の `TestMarketModelTrain` に追加:

```python
def test_train_uses_time_based_split_not_random(self) -> None:
    """train() が時間ベースの分割を使用し、ランダム置換を使わないことを確認"""
    model = MarketModel()
    n = 100
    df = pd.DataFrame({
        "race_id": ["R1"] * 50 + ["R2"] * 50,
        "surface": ["turf"] * n,
        "distance_bin": ["sprint"] * n,
        "track_condition_code": [1] * n,
        "grade_code": [0] * n,
        "field_size": [10] * n,
        "weight_diff_from_mean": [0.0] * n,
        "difficulty_score": [0.5] * n,
        "p_market_win_adj": np.linspace(0.1, 0.5, n),
    })

    with patch("models.market_model.lgb.Dataset") as mock_ds, \
         patch("models.market_model.lgb.train") as mock_train:
        mock_train.return_value = MagicMock(best_iteration=50)

        model.train(df)

        # lgb.train が呼ばれたことを確認
        assert mock_train.called

        # train_idx が [0, 79]、valid_idx が [80, 99] であることを確認
        # (時間ベースの最初80% = 学習、最後20% = 検証)
        call_args = mock_ds.call_args_list
        train_features = call_args[0][0][0]  # first lgb.Dataset call = train data
        assert len(train_features) == 80  # 最初の80%
        valid_features = call_args[1][0][0]  # second lgb.Dataset call = valid data
        assert len(valid_features) == 20  # 最後の20%
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_model.py::TestMarketModelTrain::test_train_uses_time_based_split_not_random -v`
Expected: FAIL (現在の実装はランダム分割なので、常に80/20になるが中身がシャッフルされている)

- [ ] **Step 3: Modify `src/models/market_model.py:56-60`**

```python
# Before (lines 56-60):
        # 80/20 train/valid split (再現性のため固定seed)
        n = len(features)
        perm = np.random.RandomState(42).permutation(n)
        split = int(n * 0.8)
        train_idx, valid_idx = perm[:split], perm[split:]

# After:
        # 80/20 time-based split (過去→未来、リーク防止)
        n = len(features)
        split = int(n * 0.8)
        train_idx, valid_idx = np.arange(split), np.arange(split, n)
```

- [ ] **Step 4: Modify `src/pipelines/training_pipeline.py` — ソート追加**

`_train_submodel()` の `market.train(df, ...)` 呼び出しの直前 (line 246 の直前) にソートを追加:

```python
        with TimingContext(f"{surface}/market_model"):
            # 時間ベース分割の前提: race_date でソート
            df = df.sort_values("race_date").reset_index(drop=True)
            market = MarketModel()
            market.train(df, num_threads=num_threads)
            df = market.predict_and_calc_error(df)
```

- [ ] **Step 5: Run all market model tests**

Run: `python -m pytest tests/test_market_model.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: ALL PASS (既存テストに影響なし)

- [ ] **Step 7: Commit**

```bash
git add src/models/market_model.py src/pipelines/training_pipeline.py tests/test_market_model.py
git commit -m "fix: MarketModel のランダム分割を時間ベース分割に変更 (データリーク修正)"
```

---

## Task 2: A2 — 体重変化特徴量 (weight_zscore, weight_change_zone)

**Files:**
- Modify: `src/features/horse_history_features.py:279-292` (cols_horse), `:627-654` (results dict)
- Modify: `src/features/feature_engine.py:202-252` (_map_basic_features)
- Modify: `src/models/stage1_ability_model.py:28-65` (FEATURE_COLS)
- Modify: `src/models/place_ability_model.py:26-65` (FEATURE_COLS)
- Modify: `tests/test_horse_history_features.py`

- [ ] **Step 1: Write the failing tests for weight_change_zone**

`tests/test_feature_engine.py` に新しいクラスを追加:

```python
class TestWeightChangeZone:
    """A2: weight_change_zone のユニットテスト (_map_basic_features内)"""

    def _make_df_with_zogen(self, zogen_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "race_id": ["R1"] * len(zogen_values),
            "umaban": list(range(1, len(zogen_values) + 1)),
            "surface": ["turf"] * len(zogen_values),
            "kyori": [1600] * len(zogen_values),
            "gradecd": [0] * len(zogen_values),
            "syussotosu": [10] * len(zogen_values),
            "ninki": list(range(1, len(zogen_values) + 1)),
            "kyakusitukubun": [1] * len(zogen_values),
            "zogen_sa": zogen_values,
        })

    def test_golden_zone(self) -> None:
        from features.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = self._make_df_with_zogen([5.0, 8.0, 12.0])
        result = fe._map_basic_features(df)
        assert "weight_change_zone" in result.columns
        assert (result["weight_change_zone"] == 2).all()

    def test_stable_zone(self) -> None:
        from features.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = self._make_df_with_zogen([0.0, -3.0, 3.0, 4.0])
        result = fe._map_basic_features(df)
        # -3, 0, 3 → stable (1); 4 → golden boundary (2)
        assert result["weight_change_zone"].iloc[0] == 1  # 0.0 → stable
        assert result["weight_change_zone"].iloc[1] == 1  # -3.0 → stable
        assert result["weight_change_zone"].iloc[2] == 1  # 3.0 → stable

    def test_caution_zone(self) -> None:
        from features.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = self._make_df_with_zogen([-5.0, 13.0])
        result = fe._map_basic_features(df)
        assert (result["weight_change_zone"] == 0).all()

    def test_danger_zone(self) -> None:
        from features.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = self._make_df_with_zogen([15.0, -15.0])
        result = fe._map_basic_features(df)
        assert (result["weight_change_zone"] == -1).all()

    def test_missing_zogen_sa(self) -> None:
        """zogen_sa 列がない場合はNaN"""
        from features.feature_engine import FeatureEngine
        fe = FeatureEngine()
        df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "surface": ["turf"],
            "kyori": [1600], "gradecd": [0], "syussotosu": [10],
            "ninki": [1], "kyakusitukubun": [1],
        })
        result = fe._map_basic_features(df)
        assert "weight_change_zone" in result.columns
        assert result["weight_change_zone"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feature_engine.py::TestWeightChangeZone -v`
Expected: FAIL (`weight_change_zone` column does not exist yet)

- [ ] **Step 3: Modify `src/features/horse_history_features.py`**

**2a: `cols_horse` に `bataijyu` を追加** (line 279-292):

```python
        cols_horse = [
            "race_date",
            "valid_field",
            "kakuteijyuni",
            "syussotosu",
            "harontimel3",
            "distance_bin",
            "surface",
            "baba_cd",
            "timediff",
            "jyuni1c",
            "jyuni4c",
            "kyakusitukubun",
            "bataijyu",  # A2: 体重統計計算用
        ]
```

**2b: results dict の前に体重統計を計算** (line 636 の直前、`weight_absolute` の後に追加):

```python
            # weight_zscore — 馬個体の体重分布に対する正規化
            # 注意: 全履歴 [:idx] を使用 (直近3走 [start:idx] ではない)
            # 理由: 平均・標準偏差は可能な限り多くのサンプルから計算すべき
            if n_past > 0 and "bataijyu" in horse_arrs:
                past_weights = horse_arrs["bataijyu"][valid_mask][:idx].astype(float)
                past_valid_w = past_weights[~np.isnan(past_weights)]
                if len(past_valid_w) >= 2 and pd.notna(weight_absolute):
                    w_mean = float(past_valid_w.mean())
                    w_std = float(past_valid_w.std())
                    if w_std > 0:
                        weight_zscore: float = float((weight_absolute - w_mean) / w_std)
                    else:
                        weight_zscore = 0.0
                else:
                    weight_zscore = float("nan")
            else:
                weight_zscore = float("nan")
```

**2c: results dict に `weight_zscore` を追加** (line 652 の `weight_absolute` の次に):

```python
                    "weight_absolute": weight_absolute,
                    "weight_zscore": weight_zscore,
```

- [ ] **Step 3: Modify `src/features/feature_engine.py` — weight_change_zone マッピング**

`_map_basic_features()` の末尾 (line 250付近) に追加:

```python
        # A2: weight_change_zone — 体重変化カテゴリ (zogen_sa ベース、数値エンコード)
        if "zogen_sa" in df.columns:
            zogen = df["zogen_sa"].astype(float)
            zone = pd.Series(1, index=df.index)  # default: stable (-4 ~ +4)
            zone[(zogen >= 4) & (zogen <= 12)] = 2   # golden
            zone[(zogen >= -14) & (zogen < -4)] = 0   # caution (下側)
            zone[(zogen > 12) & (zogen <= 14)] = 0    # caution (上側)
            zone[(zogen < -14) | (zogen > 14)] = -1    # danger
            df["weight_change_zone"] = zone.astype(float)
        else:
            df["weight_change_zone"] = float("nan")
```

- [ ] **Step 4: Modify FEATURE_COLS**

`src/models/stage1_ability_model.py` の FEATURE_COLS (line 64 の直前、`weight_absolute` の後に追加):

```python
        # 馬体 (2)
        "weight_absolute",
        "weight_zscore",
        "weight_change_zone",
```

同様に `src/models/place_ability_model.py` の FEATURE_COLS (line 62 の直後):

```python
        # 馬体 (2)
        "weight_absolute",
        "weight_zscore",
        "weight_change_zone",
```

- [ ] **Step 5: Write weight_zscore tests in test_horse_history_features.py**

```python
class TestWeightZscore:
    """A2: weight_zscore が results DataFrame に含まれることを確認"""

    def _make_mock_store_with_weights(self) -> MagicMock:
        """過去出走データに bataijyu 列を含むモックストア"""
        store = MagicMock(spec=ParquetStore)

        entries_hist = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "kettonum": ["K1", "K1", "K1"],
            "umaban": [1, 1, 1],
            "kakuteijyuni": [3, 5, 2],
            "syussotosu": [10, 12, 8],
            "harontimel3": [35.0, 36.0, 34.5],
            "distance_bin": ["mile", "mile", "sprint"],
            "surface": ["turf", "turf", "turf"],
            "baba_cd": [1, 2, 1],
            "timediff": [0.3, -0.2, 0.5],
            "jyuni1c": [5, 8, 3],
            "jyuni4c": [4, 6, 2],
            "kyakusitukubun": [2, 2, 1],
            "bataijyu": [480.0, 482.0, 484.0],
        })

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        return store

    def test_weight_zscore_in_output_columns(self) -> None:
        """compute() の出力に weight_zscore 列が含まれる"""
        from features.horse_history_features import HorseHistoryFeatures
        store = self._make_mock_store_with_weights()
        hhf = HorseHistoryFeatures(store=store)

        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-04-01"]),
            "surface": ["turf"],
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"],
            "umaban": [1],
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "bataijyu": [486.0],
            "kakuteijyuni": [1],
            "syussotosu": [10],
        })

        result = hhf.compute(race_df, entry_df)
        assert "weight_zscore" in result.columns
```

- [ ] **Step 6: Run tests**

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_horse_history_features.py tests/test_feature_engine.py -v`
Expected: ALL PASS

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/features/horse_history_features.py src/features/feature_engine.py src/models/stage1_ability_model.py src/models/place_ability_model.py tests/test_horse_history_features.py tests/test_feature_engine.py
git commit -m "feat: 体重変化特徴量を追加 (weight_zscore, weight_change_zone)"
```

---

## Task 3: A3 — 休養期間特徴量 (days_since_last_race, rest_category)

**Files:**
- Modify: `src/features/horse_history_features.py:432-448` (n_past判定後), `:638-654` (results dict)
- Modify: `src/models/stage1_ability_model.py` (FEATURE_COLS)
- Modify: `src/models/place_ability_model.py` (FEATURE_COLS)
- Modify: `tests/test_horse_history_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_horse_history_features.py` に追加:

```python
class TestRestPeriodFeatures:
    """A3: days_since_last_race, rest_category のテスト"""

    def _make_mock_store_with_dates(self) -> MagicMock:
        """過去出走データに race_date を含むモックストア"""
        store = MagicMock(spec=ParquetStore)

        entries_hist = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-15", "2024-03-01", "2024-05-10"]),
            "kettonum": ["K1", "K1", "K1"],
            "umaban": [1, 1, 1],
            "kakuteijyuni": [3, 5, 2],
            "syussotosu": [10, 12, 8],
            "harontimel3": [35.0, 36.0, 34.5],
            "distance_bin": ["mile", "mile", "sprint"],
            "surface": ["turf", "turf", "turf"],
            "baba_cd": [1, 2, 1],
            "timediff": [0.3, -0.2, 0.5],
            "jyuni1c": [5, 8, 3],
            "jyuni4c": [4, 6, 2],
            "kyakusitukubun": [2, 2, 1],
            "bataijyu": [480.0, 482.0, 484.0],
        })

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        return store

    def test_days_since_last_race_in_output(self) -> None:
        """compute() の出力に days_since_last_race 列が含まれる"""
        from features.horse_history_features import HorseHistoryFeatures
        store = self._make_mock_store_with_dates()
        hhf = HorseHistoryFeatures(store=store)

        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-07-01"]),
            "surface": ["turf"],
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"],
            "umaban": [1],
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "bataijyu": [486.0],
            "kakuteijyuni": [1],
            "syussotosu": [10],
        })

        result = hhf.compute(race_df, entry_df)
        assert "days_since_last_race" in result.columns
        assert "rest_category" in result.columns
        # 2024-05-10 → 2024-07-01 = 52日 → rest_category = 3 (medium: 31-90日)
        assert result["rest_category"].iloc[0] == 3.0
        assert result["days_since_last_race"].iloc[0] == 52.0

    def test_rest_category_nan_for_no_history(self) -> None:
        """過去データなしの場合はNaN"""
        from features.horse_history_features import HorseHistoryFeatures
        store = MagicMock(spec=ParquetStore)
        store.read = MagicMock(return_value=pd.DataFrame())
        hhf = HorseHistoryFeatures(store=store)

        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-07-01"]),
            "surface": ["turf"],
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"],
            "umaban": [1],
            "kettonum": ["K99"],  # 履歴なし
            "kisyucode": ["J1"],
            "bataijyu": [486.0],
            "kakuteijyuni": [1],
            "syussotosu": [10],
        })

        result = hhf.compute(race_df, entry_df)
        assert np.isnan(result["days_since_last_race"].iloc[0])
        assert np.isnan(result["rest_category"].iloc[0])
```

- [ ] **Step 2: Modify `src/features/horse_history_features.py`**

`n_past` 判定ブロック (lines 432-448) の後、`norm_finish_logit_avg` 計算 (line 450) の前に追加:

```python
            # A3: days_since_last_race + rest_category
            if n_past > 0:
                last_race_date = horse_arrs["race_date"][valid_mask][idx - 1]
                if isinstance(last_race_date, np.datetime64):
                    days_since: float = float(
                        (np.datetime64(race_date, "ns") - last_race_date.astype("datetime64[ns]"))
                        / np.timedelta64(1, "D")
                    )
                else:
                    days_since = float("nan")
                # rest_category (数値エンコード、LightGBM用)
                if days_since <= 7:
                    rest_cat: float = 1.0   # consecutive
                elif days_since <= 30:
                    rest_cat = 2.0           # short
                elif days_since <= 90:
                    rest_cat = 3.0           # medium
                elif days_since <= 180:
                    rest_cat = 4.0           # long
                else:
                    rest_cat = 5.0           # return
            else:
                days_since = float("nan")
                rest_cat = float("nan")
```

results dict (line 652付近) に追加:

```python
                    "weight_zscore": weight_zscore,
                    "days_since_last_race": days_since,
                    "rest_category": rest_cat,
```

- [ ] **Step 3: Modify FEATURE_COLS**

`src/models/stage1_ability_model.py` の FEATURE_COLS に追加:

```python
        "weight_change_zone",
        # 休養期間 (2)
        "days_since_last_race",
        "rest_category",
```

`src/models/place_ability_model.py` の FEATURE_COLS に同様に追加。

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/horse_history_features.py src/models/stage1_ability_model.py src/models/place_ability_model.py tests/test_horse_history_features.py
git commit -m "feat: 休養期間特徴量を追加 (days_since_last_race, rest_category)"
```

---

## Task 4: A4 — バックテスト/本番ベットロジック統一 (flat/kelly切替)

**Files:**
- Modify: `src/backtest/race_predictor.py:24-25` (__init__), `:104-140` (select_bets)
- Modify: `src/backtest/engine.py:68-77` (__init__)
- Modify: `scripts/run_backtest.py:39-45` (argparse)
- Modify: `tests/test_race_predictor.py`
- Modify: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing test for RacePredictor kelly mode**

`tests/test_race_predictor.py` に追加:

```python
def test_select_bets_flat_mode_uses_100_yen(self, mock_models: MagicMock) -> None:
    """flat モード (デフォルト) は100円固定ベット"""
    from backtest.race_predictor import RacePredictor

    predictor = RacePredictor(mock_models)
    race_df = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "umaban": [1, 2],
        "ev_place": [1.5, 1.3],
        "fukuoddslow": [3.0, 2.5],
        "surface": ["turf", "turf"],
    })
    bets = predictor.select_bets(race_df, bankroll=100000)
    assert len(bets) > 0
    assert all(b.stake == 100.0 for b in bets)


def test_select_bets_kelly_mode_uses_stake_calculator(self, mock_models: MagicMock) -> None:
    """kelly モードは StakeCalculator を使用する"""
    from unittest.mock import MagicMock
    from backtest.race_predictor import RacePredictor
    from betting.stake_calculator import StakeCalculator
    from betting.drawdown_controller import DrawdownController

    stake_calc = MagicMock(spec=StakeCalculator)
    stake_calc.calc_stake.return_value = 200.0

    dd_ctrl = MagicMock(spec=DrawdownController)
    dd_ctrl.adjust_stake.return_value = 200.0

    predictor = RacePredictor(
        mock_models,
        stake_calculator=stake_calc,
        dd_controller=dd_ctrl,
    )
    predictor._betting_mode = "kelly"

    race_df = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "umaban": [1, 2],
        "ev_place": [1.5, 1.3],
        "EV_lower_place": [1.4, 1.2],
        "fukuoddslow": [3.0, 2.5],
        "surface": ["turf", "turf"],
    })
    bets = predictor.select_bets(race_df, bankroll=100000)
    # kellyモードなのでStakeCalculatorが呼ばれる
    assert stake_calc.calc_stake.called or dd_ctrl.adjust_stake.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_race_predictor.py::test_select_bets_kelly_mode_uses_stake_calculator -v`
Expected: FAIL (RacePredictor に stake_calculator/dd_controller 引数がない)

- [ ] **Step 3: Modify `src/backtest/race_predictor.py`**

**3a: __init__ にオプション引数を追加** (line 24):

```python
    def __init__(
        self,
        models: TrainedModelsV5,
        *,
        stake_calculator: object | None = None,
        dd_controller: object | None = None,
    ) -> None:
        self.models = models
        self.stake_calc = stake_calculator
        self.dd_ctrl = dd_controller
        self._betting_mode = "kelly" if stake_calculator is not None else "flat"
```

**3b: select_bets にモード分岐を追加** (line 104-140):

```python
    def select_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
    ) -> list[Bet]:
        """EV > 閾値 の馬をベット候補として抽出。"""
        regime = self.models.regime_detector.current_regime
        regime_params = self.models.regime_detector.get_strategy_params(regime)

        bets: list[Bet] = []
        ev_threshold = regime_params.get("ev_threshold", 1.10)
        max_bets = regime_params.get("max_bets_per_race", 3)

        # EV列の選択
        ev_col = "EV_lower_place" if self._betting_mode == "kelly" else "ev_place"
        if ev_col not in race_df.columns or "fukuoddslow" not in race_df.columns:
            return bets

        candidates = race_df[race_df[ev_col].fillna(0) >= ev_threshold].copy()
        candidates = candidates.nlargest(max_bets, ev_col)

        for _, row in candidates.iterrows():
            if self._betting_mode == "kelly" and self.stake_calc is not None:
                stake = self.stake_calc.calc_stake(
                    ev_lower=float(row[ev_col]),
                    odds=float(row["fukuoddslow"]),
                    bankroll=bankroll,
                    bet_type=BetType.PLACE,
                )
                if self.dd_ctrl is not None:
                    stake = self.dd_ctrl.adjust_stake(stake, bankroll)
            else:
                stake = 100.0

            if bankroll >= stake:
                bets.append(
                    Bet(
                        race_id=row["race_id"],
                        umaban=int(row["umaban"]),
                        bet_type=BetType.PLACE,
                        odds=float(row["fukuoddslow"]),
                        ev_lower_corrected=float(row.get(ev_col, 0)),
                        stake=stake,
                    )
                )

        return bets
```

- [ ] **Step 4: Modify `src/backtest/engine.py` — betting_mode パラメータ追加**

**4a: __init__ に betting_mode を追加** (line 68):

```python
    def __init__(
        self,
        models: TrainedModelsV5,
        initial_bankroll: float = 100_000,
        store: ParquetStore | None = None,
        betting_mode: str = "flat",
    ) -> None:
        self.models = models
        self.initial_bankroll = initial_bankroll
        self.store = store or ParquetStore()
        self.betting_mode = betting_mode

        if betting_mode == "kelly":
            from betting.stake_calculator import StakeCalculator
            from betting.drawdown_controller import DrawdownController

            self._race_predictor = RacePredictor(
                models,
                stake_calculator=StakeCalculator(),
                dd_controller=DrawdownController(peak_bankroll=initial_bankroll),
            )
        else:
            self._race_predictor = RacePredictor(models)
```

- [ ] **Step 5: Modify `scripts/run_backtest.py` — CLI引数追加**

argparse (line 39付近) に追加:

```python
    parser.add_argument(
        "--betting-mode",
        choices=["flat", "kelly"],
        default="flat",
        help="ベット額計算モード (flat=100円固定, kelly=Fractional Kelly)",
    )
```

`BacktestEngine` のインスタンス化 (line 92付近) を変更:

```python
    engine = BacktestEngine(models=models, store=store, betting_mode=args.betting_mode)
```

- [ ] **Step 6: Write test for BacktestEngine betting_mode propagation**

`tests/test_backtest_engine.py` に追加:

```python
    def test_engine_kelly_mode_creates_predictor_with_stake_calc(self, mock_models: MagicMock) -> None:
        """betting_mode='kelly' の場合、RacePredictor に StakeCalculator が注入される"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, betting_mode="kelly")
        assert engine._race_predictor._betting_mode == "kelly"
        assert engine._race_predictor.stake_calc is not None
        assert engine._race_predictor.dd_ctrl is not None

    def test_engine_flat_mode_default(self, mock_models: MagicMock) -> None:
        """デフォルトはflatモード"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        assert engine._race_predictor._betting_mode == "flat"
        assert engine._race_predictor.stake_calc is None
```

- [ ] **Step 7: Run all affected tests**

Run: `python -m pytest tests/test_race_predictor.py tests/test_backtest_engine.py -v`
Expected: ALL PASS

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/backtest/race_predictor.py src/backtest/engine.py scripts/run_backtest.py tests/test_race_predictor.py tests/test_backtest_engine.py
git commit -m "feat: バックテストのベットロジックをflat/kelly切替可能に変更"
```

---

## Task 5: 全体動作確認

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 2: Run lint and type check**

Run: `ruff check src/ tests/ && mypy src/`
Expected: No errors

- [ ] **Step 3: Dry-run with flat mode (smoke test)**

```bash
python scripts/run_backtest.py \
  --train-start 20240101 --train-end 20241231 \
  --test-start 20250101 --test-end 20250331 \
  --betting-mode flat
```

Expected: 正常完了、ROI表示

- [ ] **Step 4: Dry-run with kelly mode (smoke test)**

```bash
python scripts/run_backtest.py \
  --train-start 20240101 --train-end 20241231 \
  --test-start 20250101 --test-end 20250331 \
  --betting-mode kelly
```

Expected: 正常完了、ROI表示 (flatと異なる値)

- [ ] **Step 5: Final commit (if any lint fixes needed)**

```bash
git add -A
git commit -m "chore: A群改善の最終調整"
```
