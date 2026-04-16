# Place EV 過大評価修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PlaceTwoStageModel の Stage B (e_return_place_pred) が 2.2 倍過大評価されている問題を修正し、ROI 73% → 100%+ を達成する。

**Architecture:** 2段階修正。(1) PlaceTwoStageModel の FEATURE_COLS に複勝特徴量 (fukuoddslow, p_ability_place, tanodds) を追加し、Stage B 回帰モデルが複勝オッズスケールで正しく学習できるようにする。(2) PlaceEVCorrectionModel クラスを追加し、P/E 分解による EV 補正を複勝にも適用する。既存の Win EVCorrectionModel と同じパターンで、target と init_score を place 用に変更したもの。

**Tech Stack:** LightGBM, pandas, numpy, pytest, unittest.mock

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/two_stage_return_model.py` | Modify | PlaceTwoStageModel の FEATURE_COLS を複勝専用に変更 |
| `src/models/ev_correction_model.py` | Modify | PlaceEVCorrectionModel クラスを追加 |
| `src/domain/models.py` | Modify | SubmodelSet に place_ev_corrector フィールドを追加 |
| `src/pipelines/training_pipeline.py` | Modify | PlaceEVCorrectionModel の学習・適用・保存を追加 |
| `src/backtest/race_predictor.py` | Modify | 推論チェーンに place EV 補正を追加、閾値列を ev_place_corrected に変更 |
| `src/backtest/engine.py` | Modify | _generate_bets 閾値列を ev_place_corrected に変更 |
| `src/db/model_loader.py` | Modify | PlaceEVCorrectionModel の読込・SubmodelSet 組立を追加 |
| `tests/test_two_stage_return_model.py` | Modify | 新特徴量に対応したテスト更新 |
| `tests/test_ev_correction.py` | Modify | PlaceEVCorrectionModel のテスト追加 |
| `tests/test_parameter_freeze.py` | Modify | SubmodelSet 構築箇所 3件 に place_ev_corrector 追加 |
| `tests/test_training_pipeline.py` | Modify | SubmodelSet 構築箇所 5件 に place_ev_corrector 追加 |

### 前提知識: 推論チェーンの実行順序

推論時 (`race_predictor.py:76-114`) の実行順序:
1. `market.predict_and_calc_error(df)` — 市場モデル log_error 特徴量
2. `stage1.add_ability_probs(df)` — `p_ability_win` 追加
3. **`place_ability.predict(df)`** — `p_ability_place` 追加 ★ここで利用可能になる
4. `win.predict_ev(df)` — `p_win_pred`, `e_return_win_pred`, `ev_win` 追加
5. jockey/trainer/JT context マージ — 騎手・調教師特徴量
6. `ev_corrector.correct_ev(df)` — WIN EV 補正のみ
7. **`place.predict_ev(df)`** — `p_place_pred`, `e_return_place_pred`, `ev_place` 追加
8. [NEW] **`place_ev_corrector.correct_ev(df)`** — Place EV 補正

各ステップで DataFrame に列が累積していく。Step 3 時点で `fukuoddslow` (FeatureEngine 出力)、`p_ability_place` (PlaceAbilityModel 出力) が利用可能。

学習パイプライン (`training_pipeline.py:270-513`) も同順序。PlaceAbilityModel (line 395) → WinTwoStageModel (line 402) → EVCorrectionModel (line 448) → PlaceTwoStageModel (line 454)。

---

## Task 1: PlaceTwoStageModel に複勝特徴量を追加 (最重要)

**Files:**
- Modify: `src/models/two_stage_return_model.py:171-181`
- Modify: `tests/test_two_stage_return_model.py:18-46, 230-248`

**背景:** 現在 `PlaceTwoStageModel.FEATURE_COLS = WinTwoStageModel.FEATURE_COLS` (line 181) により、Stage B 回帰モデルが単勝市場の特徴量のみで複勝配当 (`fukuoddslow`) を予測している。これが e_return_place_pred の 2.2 倍過大評価の根本原因。複勝オッズそのもの (`fukuoddslow`) を特徴量に追加することで、モデルが複勝オッズスケールで正しく学習できるようになる。

- [ ] **Step 1: テストフィクスチャに新特徴量列を追加**

`tests/test_two_stage_return_model.py` の `feature_df` フィクスチャ (lines 18-46) に以下の列を追加:

```python
"fukuoddslow": [1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0],
"tanodds": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0],
"p_ability_place": [0.55, 0.48, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15],
```

これらは PlaceTwoStageModel の新 FEATURE_COLS に含まれる列。フィクスチャの全列は `FEATURE_COLS` に含まれる必要がある。

- [ ] **Step 2: 新特徴量の包含を検証するテストを追加**

`tests/test_two_stage_return_model.py` の `TestPlaceTwoStageModel` クラスに追加:

```python
def test_place_feature_cols_include_place_specific(self):
    """Place model should have place-specific features beyond win features"""
    assert "fukuoddslow" in PlaceTwoStageModel.FEATURE_COLS
    assert "p_ability_place" in PlaceTwoStageModel.FEATURE_COLS
    assert "tanodds" in PlaceTwoStageModel.FEATURE_COLS
    # Win特徴量も全て含む
    for col in WinTwoStageModel.FEATURE_COLS:
        assert col in PlaceTwoStageModel.FEATURE_COLS
    # Place固有特徴量が追加されている
    assert len(PlaceTwoStageModel.FEATURE_COLS) > len(WinTwoStageModel.FEATURE_COLS)
```

- [ ] **Step 3: 既存の共有テストを更新**

`test_shared_feature_cols_with_win` テスト (line ~230) を更新。現在は `assert PlaceTwoStageModel.FEATURE_COLS == WinTwoStageModel.FEATURE_COLS` だが、新設計では Place が Win のスーパーセットになる:

```python
# Before:
def test_shared_feature_cols_with_win(self, trained_place_model):
    assert PlaceTwoStageModel.FEATURE_COLS == WinTwoStageModel.FEATURE_COLS

# After: 削除して Step 2 の test_place_feature_cols_include_place_specific に統合
```

このテスト関数を削除し、Step 2 の新テストで代替する。

- [ ] **Step 4: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_place_feature_cols_include_place_specific -v`
Expected: FAIL — `fukuoddslow` not in FEATURE_COLS

- [ ] **Step 5: PlaceTwoStageModel に複勝専用 FEATURE_COLS を定義**

`src/models/two_stage_return_model.py` line 181 を変更:

```python
# Before (line 181):
FEATURE_COLS: list[str] = WinTwoStageModel.FEATURE_COLS

# After:
FEATURE_COLS: list[str] = [
    # Stage1 出力
    "p_ability_win",
    "p_ability_place",             # PlaceAbilityModel 出力
    # Market Model 正規化差分
    "signed_log_error_win",
    "abs_log_error_win",
    # 複勝・単勝オッズ
    "fukuoddslow",                 # 複勝オッズ (return model 最重要特徴量)
    "tanodds",                     # 単勝オッズ (win-place spread の文脈)
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
```

`_prepare_features()` メソッド (line 186-195) は変更不要。`fukuoddslow`, `tanodds`, `p_ability_place` は全て float64 であり、既存の Int64→float64 変換と category 変換で正しく処理される。

- [ ] **Step 6: テストを実行して通過を確認**

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: ALL PASS

- [ ] **Step 7: コミット**

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py
git commit -m "feat: PlaceTwoStageModel に複勝特徴量 (fukuoddslow, p_ability_place, tanodds) を追加"
```

---

## Task 2: PlaceEVCorrectionModel クラスを追加

**Files:**
- Modify: `src/models/ev_correction_model.py` (末尾に追加)
- Modify: `tests/test_ev_correction.py` (末尾に追加)

**背景:** 既存の `EVCorrectionModel` は WIN 用の P/E 分解補正のみを行う。Place 用の EV 補正は存在せず、`ev_place_corrected = ev_place` (passthrough) となっている。PlaceEVCorrectionModel を追加して、Stage B の過大評価を事後補正する二重安全ネットを構築する。

### 2-A: テストファースト

- [ ] **Step 1: テスト用 mock booster ヘルパーとフィクスチャを追加**

`tests/test_ev_correction.py` の末尾に追加。既存の `_make_mock_booster()` (line ~10) を再利用:

```python
# --- PlaceEVCorrectionModel tests ---

@pytest.fixture
def pre_place_ev_df():
    """PlaceEVCorrectionModel.correct_ev() の入力 DataFrame (8行)"""
    n = 8
    return pd.DataFrame({
        "race_id": ["R001"] * n,
        "umaban": list(range(1, n + 1)),
        "kakuteijyuni": [1, 2, 3, 4, 5, 6, 7, 8],
        "p_place_pred": [0.65, 0.55, 0.50, 0.40, 0.30, 0.25, 0.20, 0.10],
        "e_return_place_pred": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0],
        "fukuoddslow": [1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 5.0],
        "p_ability_place": [0.60, 0.52, 0.45, 0.38, 0.30, 0.22, 0.18, 0.10],
        "signed_log_error_win": [0.1, -0.05, 0.02, -0.1, 0.15, -0.08, 0.03, -0.12],
        "abs_log_error_win": [0.1, 0.05, 0.02, 0.1, 0.15, 0.08, 0.03, 0.12],
        "market_entropy": [2.5] * n,
        "popularity_rank": list(range(1, n + 1)),
        "surface": ["turf"] * n,
        "distance_bin": [2] * n,
        "track_condition_code": [1] * n,
        "field_size": [n] * n,
        "jockey_wr_overall": [0.3] * n,
        "jockey_wr_distance": [0.25] * n,
        "jockey_wr_venue": [0.28] * n,
        "jockey_prize_log": [13.0] * n,
        "trainer_wr_overall": [0.2] * n,
        "trainer_wr_distance": [0.18] * n,
        "trainer_wr_venue": [0.22] * n,
        "trainer_prize_log": [12.0] * n,
        "jt_combo_wr": [0.15] * n,
        "jt_combo_place_rate": [0.4] * n,
        "jt_combo_starts": [50] * n,
        "jt_combo_prize_log": [12.0] * n,
        "implied_prob_hhi": [0.15] * n,
    })
```

- [ ] **Step 2: correct_ev 出力列テストを追加**

```python
class TestPlaceEVCorrectionModel:
    def test_correct_ev_outputs_place_columns(self, pre_place_ev_df):
        """correct_ev should output p_place_corrected, e_return_place_corrected, ev_place_corrected"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.array([0.1, -0.05, 0.02, -0.03, 0.08, -0.01, 0.04, -0.06]))
        model.e_correction_model = _make_mock_booster(np.array([0.05, -0.02, 0.01, -0.04, 0.03, -0.01, 0.02, -0.03]))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        assert "p_place_corrected" in result.columns
        assert "e_return_place_corrected" in result.columns
        assert "ev_place_corrected" in result.columns
```

- [ ] **Step 3: P補正の境界テストを追加**

```python
    def test_place_p_corrected_bounds(self, pre_place_ev_df):
        """P(place) corrected should be in [0, 1]"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.array([2.0, -3.0, 1.0, -1.0, 1.5, -2.0, 0.5, -1.5]))
        model.e_correction_model = _make_mock_booster(np.zeros(8))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        assert (result["p_place_corrected"] >= 0).all()
        assert (result["p_place_corrected"] <= 1).all()
```

- [ ] **Step 4: E補正の正値テストを追加**

```python
    def test_place_e_corrected_positive(self, pre_place_ev_df):
        """E(return|place) corrected should always be positive"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.zeros(8))
        model.e_correction_model = _make_mock_booster(np.array([-0.1, 0.1, -0.05, 0.05, -0.08, 0.08, -0.03, 0.03]))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        assert (result["e_return_place_corrected"] > 0).all()
```

- [ ] **Step 5: EV 分解テストを追加**

```python
    def test_place_ev_decomposition(self, pre_place_ev_df):
        """ev_place_corrected = p_place_corrected * e_return_place_corrected"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.array([0.1, -0.05, 0.02, -0.03, 0.08, -0.01, 0.04, -0.06]))
        model.e_correction_model = _make_mock_booster(np.array([0.05, -0.02, 0.01, -0.04, 0.03, -0.01, 0.02, -0.03]))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        expected = result["p_place_corrected"] * result["e_return_place_corrected"]
        assert np.allclose(result["ev_place_corrected"], expected, atol=1e-10)
```

- [ ] **Step 6: 未学習フォールバックテストを追加**

```python
    def test_untrained_fallback_passes_through(self, pre_place_ev_df):
        """Untrained model should pass through ev_place as ev_place_corrected"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()  # _trained = False

        # ev_place 列を事前に設定
        pre_place_ev_df["ev_place"] = pre_place_ev_df["p_place_pred"] * pre_place_ev_df["e_return_place_pred"]
        result = model.correct_ev(pre_place_ev_df)

        # フォールバック: ev_place_corrected == ev_place
        assert "ev_place_corrected" in result.columns
        assert np.allclose(result["ev_place_corrected"], result["ev_place"])
```

- [ ] **Step 7: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_ev_correction.py::TestPlaceEVCorrectionModel -v`
Expected: FAIL — ImportError: cannot import name 'PlaceEVCorrectionModel'

### 2-B: 実装

- [ ] **Step 8: PlaceEVCorrectionModel クラスを実装**

`src/models/ev_correction_model.py` の末尾に追加。`EVCorrectionModel` と同じ P/E 分解パターン。Win 版との差分:

| 項目 | EVCorrectionModel (Win) | PlaceEVCorrectionModel (Place) |
|------|------------------------|-------------------------------|
| P-target | `kakuteijyuni == 1` | `kakuteijyuni <= 3` |
| P-init_score | `logit(p_win_pred)` | `logit(p_place_pred)` |
| E-target | `log(confirmed_odds) - log(e_return_win_pred)` | `log(fukuoddslow) - log(e_return_place_pred)` |
| E-filter | winners only (`kakuteijyuni == 1`) | placed only (`kakuteijyuni <= 3`) |
| E-weight | `1/sqrt(p_win_pred)` | `1/sqrt(p_place_pred)` |
| 出力列 | `p_win_corrected`, `e_return_win_corrected`, `ev_win_corrected` | `p_place_corrected`, `e_return_place_corrected`, `ev_place_corrected` |
| interaction | `p_x_e = p_win_pred * e_return_win_pred` | `p_x_e = p_place_pred * e_return_place_pred` |

```python
class PlaceEVCorrectionModel:
    """複勝EV補正モデル — P補正(二値分類) × E補正(log-ratio 回帰)

    EVCorrectionModel の複勝版。
    Stage B (e_return_place_pred) の過大評価を事後補正する。
    """

    E_CLIP_FLOOR: float = 1.0

    FEATURE_COLS: list[str] = [
        "e_return_place_pred",
        "fukuoddslow",
        "p_ability_place",
        # Market features
        "signed_log_error_win",
        "abs_log_error_win",
        "market_entropy",
        "popularity_rank",
        # Race conditions
        "surface",
        "distance_bin",
        "track_condition_code",
        "field_size",
        # Jockey context
        "jockey_wr_overall",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        # Trainer context
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        # Jockey-Trainer combo
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
        # FLB slope (市場集中度)
        "implied_prob_hhi",
    ]

    def __init__(self) -> None:
        self.p_correction_model: lgb.Booster | StackedEnsemble | None = None
        self.e_correction_model: lgb.Booster | StackedEnsemble | None = None
        self._trained: bool = False

    @staticmethod
    def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["p_x_e_interaction_place"] = df["p_place_pred"] * df["e_return_place_pred"]
        df["p_minus_e_gap_place"] = np.abs(
            np.log(df["p_place_pred"].clip(1e-4, 1 - 1e-4))
            - np.log(df["e_return_place_pred"].clip(PlaceEVCorrectionModel.E_CLIP_FLOOR))
        )
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "p_x_e_interaction_place" not in df.columns:
            df = self._add_interaction_features(df)
        features = df[self.FEATURE_COLS + ["p_x_e_interaction_place", "p_minus_e_gap_place"]].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P補正 + E補正モデルを学習"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)

        # --- P-correction: P(place) の補正 ---
        features = self._prepare_features(df)
        y_p = (df["kakuteijyuni"] <= 3).astype(int)
        p_pred = df["p_place_pred"].clip(1e-4, 1 - 1e-4)
        init_score = np.log(p_pred / (1 - p_pred))

        n = len(features)
        split = int(n * 0.8)
        train_data = lgb.Dataset(
            features.iloc[:split], label=y_p.iloc[:split],
            init_score=init_score.iloc[:split],
        )
        valid_data = lgb.Dataset(
            features.iloc[split:], label=y_p.iloc[split:],
            init_score=init_score.iloc[split:], reference=train_data,
        )

        self.p_correction_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        # --- E-correction: E(return|place) の補正 ---
        placed_df = df[df["kakuteijyuni"] <= 3].copy()
        placed_features = self._prepare_features(placed_df)
        e_pred = placed_df["e_return_place_pred"].clip(self.E_CLIP_FLOOR)
        actual_return = placed_df["fukuoddslow"].clip(self.E_CLIP_FLOOR)
        log_e_correction = np.log(actual_return) - np.log(e_pred)
        sample_weight = 1.0 / np.sqrt(placed_df["p_place_pred"].clip(0.01))

        n_placed = len(placed_features)
        if n_placed < 10:
            self.e_correction_model = lgb.train(
                {
                    "objective": "regression_l1",
                    "metric": "mae",
                    "learning_rate": 0.03,
                    "num_leaves": 15,
                    "feature_fraction": 0.7,
                    "num_threads": num_threads,
                    "verbose": -1,
                },
                lgb.Dataset(placed_features, label=log_e_correction, weight=sample_weight),
                num_boost_round=300,
            )
        else:
            split_e = int(n_placed * 0.8)
            train_e = lgb.Dataset(
                placed_features.iloc[:split_e],
                label=log_e_correction.iloc[:split_e],
                weight=sample_weight.iloc[:split_e],
            )
            valid_e = lgb.Dataset(
                placed_features.iloc[split_e:],
                label=log_e_correction.iloc[split_e:],
                weight=sample_weight.iloc[split_e:],
                reference=train_e,
            )
            self.e_correction_model = lgb.train(
                {
                    "objective": "regression_l1",
                    "metric": "mae",
                    "learning_rate": 0.03,
                    "num_leaves": 15,
                    "feature_fraction": 0.7,
                    "num_threads": num_threads,
                    "verbose": -1,
                },
                train_e,
                num_boost_round=300,
                valid_sets=[valid_e],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )

        self._trained = True

    def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """P/E 分解補正を適用して ev_place_corrected を出力"""
        if not self._trained:
            df["ev_place_corrected"] = df.get("ev_place", 0.0)
            return df

        features = self._prepare_features(df)

        # P-correction
        p_iter = self.p_correction_model.best_iteration if self.p_correction_model.best_iteration > 0 else None
        p_correction_logit = self.p_correction_model.predict(features, num_iteration=p_iter)
        p_pred = df["p_place_pred"].clip(1e-4, 1 - 1e-4)
        df["p_place_corrected"] = 1.0 / (1.0 + np.exp(-(np.log(p_pred / (1 - p_pred)) + p_correction_logit)))
        df["p_place_corrected"] = df["p_place_corrected"].clip(0.0, 1.0)

        # E-correction
        e_iter = self.e_correction_model.best_iteration if self.e_correction_model.best_iteration > 0 else None
        log_e_correction = self.e_correction_model.predict(features, num_iteration=e_iter)
        df["e_return_place_corrected"] = df["e_return_place_pred"] * np.exp(log_e_correction)
        df["e_return_place_corrected"] = df["e_return_place_corrected"].clip(0.01)

        # EV
        df["ev_place_corrected"] = df["p_place_corrected"] * df["e_return_place_corrected"]

        return df
```

- [ ] **Step 9: テストを実行して通過を確認**

Run: `python -m pytest tests/test_ev_correction.py -v`
Expected: ALL PASS (既存 Win テスト + 新規 Place テスト)

- [ ] **Step 10: コミット**

```bash
git add src/models/ev_correction_model.py tests/test_ev_correction.py
git commit -m "feat: PlaceEVCorrectionModel クラスを追加 (複勝EV補正)"
```

---

## Task 3: SubmodelSet + TrainingPipeline + RacePredictor + ModelLoader に統合

**Files:**
- Modify: `src/domain/models.py:219-234`
- Modify: `src/pipelines/training_pipeline.py:447-876` (学習・保存)
- Modify: `src/backtest/race_predictor.py:102-140` (推論 + 閾値列変更)
- Modify: `src/backtest/engine.py:629-644` (閾値列変更)
- Modify: `src/db/model_loader.py:102-104, 153-162, 390-396, 433-442` (読込)
- Modify: `tests/test_parameter_freeze.py:28-37, 75-84, 108-117` (3箇所)
- Modify: `tests/test_training_pipeline.py:60-69, 77-86, 99-118, 627-636` (5箇所)

### 3-A: SubmodelSet

- [ ] **Step 1: SubmodelSet に place_ev_corrector フィールドを追加**

`src/domain/models.py` line ~228 に追加:

```python
# import 文 (ファイル上部) に追加:
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel

# SubmodelSet dataclass (line 220):
@dataclass
class SubmodelSet:
    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    place_ev_corrector: PlaceEVCorrectionModel    # ← NEW
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    use_ensemble: bool = False
```

### 3-B: TrainingPipeline

- [ ] **Step 2: import を追加**

`src/pipelines/training_pipeline.py` の import セクションに追加:

```python
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
```

- [ ] **Step 3: _train_submodel() に PlaceEVCorrectionModel 学習を追加**

`src/pipelines/training_pipeline.py` の `_train_submodel()` メソッド内、PlaceTwoStageModel 学習 (line ~477) の直後に追加:

```python
        # 6. Place EV補正 (P/E decomposition)
        with TimingContext(f"{surface}/place_ev_correction"):
            place_ev_corrector = PlaceEVCorrectionModel()
            place_ev_corrector.train(df_oof, num_threads=num_threads)
            df_oof = place_ev_corrector.correct_ev(df_oof)
```

- [ ] **Step 4: 信頼区間キャリブレーションを更新**

同メソッド内、confidence calibration 部分 (line ~500):

```python
# Before:
place_calib_df["ev_place_corrected"] = df_oof["ev_place"]  # passthrough

# After:
place_calib_df["ev_place_corrected"] = df_oof["ev_place_corrected"]
```

- [ ] **Step 5: SubmodelSet 組立を更新**

同メソッドの return 文 (line ~503):

```python
return SubmodelSet(
    market=market,
    stage1=stage1,
    place_ability=place_ability,
    win=win_2s,
    ev_corrector=ev_corrector,
    place=place_2s,
    place_ev_corrector=place_ev_corrector,    # ← NEW
    wide=wide_2s,
    confidence=conf,
    use_ensemble=use_ensemble,
)
```

- [ ] **Step 6: _save_models_local に place_ev_corrector の保存を追加**

`_save_models_local()` メソッド (line ~786) 内で、既存の `ev_corrector` P/E booster 保存パターン (line ~816-817) に倣って追加:

```python
# ev_corrector の保存の直後 (line ~817 の次):
saved[f"place_ev_corrector_p_{surface}"] = sub.place_ev_corrector.p_correction_model
saved[f"place_ev_corrector_e_{surface}"] = sub.place_ev_corrector.e_correction_model
```

これで `_save_models_local` 内の既存の汎用ループが `.lgb` または `.joblib` として自動保存する。

- [ ] **Step 7: _log_to_mlflow に place_ev_corrector のロギングを追加**

`_log_to_mlflow()` メソッド (line ~674) 内で、既存の `ev_corrector` MLflow ロギングパターン (line ~709-717) に倣って追加:

```python
# EVCorrectionModel (win) のロギングの直後:
mlflow.lightgbm.log_model(
    sub.place_ev_corrector.p_correction_model,
    name=f"place_ev_corrector_p_{surface}",
)
mlflow.lightgbm.log_model(
    sub.place_ev_corrector.e_correction_model,
    name=f"place_ev_corrector_e_{surface}",
)
```

### 3-C: ModelLoader

- [ ] **Step 8: MLflow ロードパスに place_ev_corrector を追加**

`src/db/model_loader.py` の MLflow ロードメソッド内 (line ~102-104 の直後) に追加:

```python
# ev_corrector 読込の直後:
place_ev_corr = PlaceEVCorrectionModel()
place_ev_corr.p_correction_model = self._load_lgbm(
    f"{artifact_uri}/place_ev_corrector_p_{surface}"
)
place_ev_corr.e_correction_model = self._load_lgbm(
    f"{artifact_uri}/place_ev_corrector_e_{surface}"
)
place_ev_corr._trained = True
```

同メソッド内の SubmodelSet 構築 (line ~153-162) に `place_ev_corrector=place_ev_corr,` を追加。

- [ ] **Step 9: ローカルロードパスに place_ev_corrector を追加 (フォールバック付き)**

`src/db/model_loader.py` のローカルロードメソッド内 (line ~390-396 の直後) に追加:

```python
# ev_corrector 読込の直後:
place_ev_corr_file = models_dir / f"place_ev_corrector_p_{surface}.lgb"
if place_ev_corr_file.exists():
    place_ev_corr = PlaceEVCorrectionModel()
    place_ev_corr.p_correction_model = self._load_lgbm(
        str(models_dir / f"place_ev_corrector_p_{surface}.lgb")
    )
    place_ev_corr.e_correction_model = self._load_lgbm(
        str(models_dir / f"place_ev_corrector_e_{surface}.lgb")
    )
    place_ev_corr._trained = True
else:
    # 古いモデルとの後方互換: ファイルがなければ未学習モデルを作成
    place_ev_corr = PlaceEVCorrectionModel()
```

同メソッド内の SubmodelSet 構築 (line ~433-442) に `place_ev_corrector=place_ev_corr,` を追加。

### 3-D: RacePredictor + BacktestEngine (閾値列の変更)

- [ ] **Step 10: 推論チェーンに PlaceEVCorrectionModel を追加**

`src/backtest/race_predictor.py` lines 102-107 を変更:

```python
# Before (lines 102-107):
# 6. EV補正 + Place推論
df = submodel.ev_corrector.correct_ev(df)
df = submodel.place.predict_ev(df)
if "ev_place_corrected" not in df.columns:
    df["ev_place_corrected"] = df.get("ev_place", 0.0)

# After:
# 6. EV補正 + Place推論 + Place EV補正
df = submodel.ev_corrector.correct_ev(df)
df = submodel.place.predict_ev(df)
df = submodel.place_ev_corrector.correct_ev(df)
```

`if "ev_place_corrected" not in df.columns:` の fallback は不要。`PlaceEVCorrectionModel.correct_ev()` が `_trained=False` 時にパススルーを返すため。

- [ ] **Step 11: ベッティング閾値列を ev_place_corrected に変更**

`src/backtest/race_predictor.py` line 140 を変更:

```python
# Before:
ev_col = "ev_place"

# After:
ev_col = "ev_place_corrected"
```

**理由:** `ev_place_corrected` は PlaceEVCorrectionModel による補正済み EV。補正により過大評価が是正されるため、閾値判定に補正済み値を使うことで正確なベット選択が可能になる。コメント (lines 128-131) も更新:

```python
# Before:
# 閾値判定は常に点推定 (ev_place)、kellyの賭け金のみ信頼区間下限を使用

# After:
# 閾値判定は補正済み EV (ev_place_corrected)、kellyの賭け金のみ信頼区間下限を使用
```

- [ ] **Step 12: BacktestEngine._generate_bets の閾値列も変更**

`src/backtest/engine.py` lines 629-632 を変更:

```python
# Before:
if "ev_place" in race_df.columns and "fukuoddslow" in race_df.columns:
    candidates = race_df[race_df["ev_place"].fillna(0) >= ev_threshold].copy()
    ...
    candidates = candidates.nlargest(max_bets, "ev_place")

# After:
ev_col = "ev_place_corrected" if "ev_place_corrected" in race_df.columns else "ev_place"
if ev_col in race_df.columns and "fukuoddslow" in race_df.columns:
    candidates = race_df[race_df[ev_col].fillna(0) >= ev_threshold].copy()
    ...
    candidates = candidates.nlargest(max_bets, ev_col)
```

フォールバック (`ev_place_corrected` がなければ `ev_place` を使用) により、古いモデルとの後方互換を維持。

### 3-E: 壊れるテストの修正

SubmodelSet dataclass に必須フィールドが追加されるため、以下の **8箇所** の `SubmodelSet(` 呼び出しに `place_ev_corrector=...` を追加する必要がある。

- [ ] **Step 13: test_parameter_freeze.py (3箇所)**

`tests/test_parameter_freeze.py`:

| 行 | コンテキスト | 追加する値 |
|----|-------------|-----------|
| 28-37 | `mock_models` fixture | `place_ev_corrector=MagicMock(),` |
| 75-84 | `test_detect_change_after_freeze` | `place_ev_corrector=MagicMock(),` |
| 108-117 | `test_context_manager_detects_violation` | `place_ev_corrector=MagicMock(),` |

各 SubmodelSet 構築で `wide=MagicMock(),` の前に `place_ev_corrector=MagicMock(),` を追加。

- [ ] **Step 14: test_training_pipeline.py (5箇所)**

`tests/test_training_pipeline.py`:

| 行 | コンテキスト | 追加する値 |
|----|-------------|-----------|
| 60-69 | `test_submodel_set_holds_models` | `place_ev_corrector=None,` |
| 77-86 | `test_trained_models_v5_structure` (turf) | `place_ev_corrector=None,` |
| 99-107 | `test_trained_models_v5_supports_both_surfaces` (turf) | `place_ev_corrector=None,` |
| 109-118 | `test_trained_models_v5_supports_both_surfaces` (dirt) | `place_ev_corrector=None,` |
| 627-636 | `TestJRAFilterTraining._run_pipeline_with_mocks` | `place_ev_corrector=MagicMock(),` |

各 SubmodelSet 構築で `wide=...,` の前に `place_ev_corrector=...,` を追加。

### 3-F: テスト実行とコミット

- [ ] **Step 15: テストを実行して全通過を確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 16: コミット**

```bash
git add src/domain/models.py src/pipelines/training_pipeline.py src/backtest/race_predictor.py \
        src/backtest/engine.py src/db/model_loader.py \
        tests/test_parameter_freeze.py tests/test_training_pipeline.py
git commit -m "feat: PlaceEVCorrectionModel をパイプラインに統合 (学習・推論・保存・読込)"
```

---

## Task 4: バックテストで検証

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
| 全体 ROI | > 100% |
| 各年 ROI | 全年 > 100% |
| 最大ドローダウン | < 15% |

`data/backtest/bt_*_horse_diagnostics.csv` で以下を確認:

| 確認項目 | 合格基準 |
|----------|---------|
| e_return_place_pred / actual 比 | 0.8 ~ 1.2 (2.2倍過大評価からの改善) |
| EV デシル別 ROI | 高 EV → 高 ROI の単調増加傾向 |
| ベット数 | 10,431件から大幅減少 (EV 閾値が正しく機能) |

- [ ] **Step 3: 結果を記録**

確認結果をメモリーに保存し、改善前 (ROI 73%) との比較を記録。
