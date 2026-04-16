# Place Prediction Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three structural problems causing poor ROI: (1) double-counting of market odds, (2) missing calibration on PlaceTwoStageModel, (3) information bottleneck from lack of horse-level features.

**Architecture:** Remove direct odds features from Place hit model, add horse-level features for genuine alpha, fit Benter logit combination (fundamental + market) on validation data, apply isotonic calibration as post-processing.

**Tech Stack:** LightGBM, scikit-learn (IsotonicRegression), scipy.optimize (Benter MLE), numpy, pandas

---

### Task 1: PlaceTwoStageModel — 特徴量更新 + バリデーションデータ保存

**Files:**
- Modify: `src/models/two_stage_return_model.py:182-290`

- [ ] **Step 1: HIT_FEATURE_COLS からオッズ特徴量を削除し馬レベル特徴量を追加**

`HIT_FEATURE_COLS` (lines 185-213) を以下に置き換える:
- 削除: `fukuoddslow`, `tanodds`
- 追加: 13個の馬レベル特徴量 (df_oof に学習時点でマージ済み)

```python
HIT_FEATURE_COLS: list[str] = [
    # Stage1 出力
    "p_ability_win",
    "p_ability_place",
    # Market Model 正規化差分 (間接的市場情報)
    "signed_log_error_win",
    "abs_log_error_win",
    # --- 馬レベル特徴量 (新規) ---
    "norm_finish_logit_avg",
    "harontimel5_zscore",
    "closing_index_avg",
    "weight_zscore",
    "days_since_last_race",
    "rest_category",
    "form_trend",
    "form_consistency",
    "blood_surface_wr",
    "blood_distance_wr",
    "jockey_wr_overall",
    "trainer_wr_overall",
    "jt_combo_place_rate",
    "course_wr",
    # --- 間接的市場情報 (既存) ---
    "odds_drop_rate_60_10",
    "odds_drop_rate_30_10",
    "odds_velocity",
    "odds_volatility",
    "popularity_change_30_10",
    "market_entropy",
    "popularity_rank",
    "overround",
    "surface",
    "distance_bin",
    "track_condition_code",
    "grade_code",
    "field_size",
    "odds_skewness",
]
```

- [ ] **Step 2: train_hit_model にバリデーションデータ保存を追加**

`train_hit_model()` メソッド (line 267-290) の末尾、`self.hit_model = lgb.train(...)` の後に追加:

```python
        # バリデーション予測を保存 (Benter combination + isotonic fitting 用)
        n = len(features)
        split = int(n * 0.8)
        self._val_p_raw = self.hit_model.predict(
            features.iloc[split:], num_iteration=hit_iter
        )
        self._val_y = y.iloc[split:].values
        self._val_fukuoddslow = df["fukuoddslow"].iloc[split:].values
```

- [ ] **Step 3: テストを実行**

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: PASS (feature list が変更されたテストがあれば更新)

- [ ] **Step 4: コミット**

```bash
git add src/models/two_stage_return_model.py
git commit -m "feat: remove odds from place hit model, add horse-level features"
```

---

### Task 2: SubmodelSet — benter_lr → benter_combo に変更

**Files:**
- Modify: `src/domain/models.py:222-239`

- [ ] **Step 1: SubmodelSet のフィールドを更新**

`src/domain/models.py` の import に追加:
```python
from sklearn.isotonic import IsotonicRegression
```

SubmodelSet (line 222-239) の `benter_lr` を置き換え:
```python
@dataclass
class SubmodelSet:
    """サブモデル（芝/ダート）のセット"""

    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    place_ev_corrector: PlaceEVCorrectionModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    use_ensemble: bool = False
    benter_combo: BenterCombination | None = None
    isotonic_calibrator: IsotonicRegression | None = None
```

TYPE_CHECKING block に import を追加:
```python
if TYPE_CHECKING:
    from models.benter_combination import BenterCombination
```

- [ ] **Step 2: テストを実行**

Run: `python -m pytest tests/test_domain.py -v`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add src/domain/models.py
git commit -m "refactor: replace benter_lr with benter_combo + isotonic_calibrator in SubmodelSet"
```

---

### Task 3: Training Pipeline — Benter + Isotonic 学習・保存

**Files:**
- Modify: `src/pipelines/training_pipeline.py:459-526` (学習部分)
- Modify: `src/pipelines/training_pipeline.py:804-890` (保存部分)

- [ ] **Step 1: Place 学習後に Benter + Isotonic を fit する**

`_train_submodel()` メソッド内、`place_2s.predict_ev(df_oof)` の後 (line 483付近) に追加:

```python
        # 5b. Benter Combination + Isotonic Calibration
        benter_combo = None
        isotonic_cal = None
        if hasattr(place_2s, "_val_p_raw") and len(place_2s._val_p_raw) >= 500:
            from models.benter_combination import BenterCombination
            from sklearn.isotonic import IsotonicRegression

            val_p = place_2s._val_p_raw
            val_p_market = np.where(
                place_2s._val_fukuoddslow > 0,
                1.0 / place_2s._val_fukuoddslow,
                0.5,
            )
            val_y = place_2s._val_y

            with TimingContext(f"{surface}/benter"):
                benter_combo = BenterCombination.fit(val_p, val_p_market, val_y)
                logger.info(
                    "Benter params: alpha=%.3f, beta=%.3f, gamma=%.3f",
                    benter_combo.alpha, benter_combo.beta, benter_combo.gamma,
                )

            with TimingContext(f"{surface}/isotonic"):
                val_p_combined = benter_combo.combine(val_p, val_p_market)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(val_p_combined, val_y)
                isotonic_cal = iso
                logger.info("Isotonic calibrator fitted on %d samples", len(val_p))
```

- [ ] **Step 2: SubmodelSet 構築に benter_combo + isotonic_cal を追加**

`return SubmodelSet(...)` (line 515-526) を更新:

```python
        return SubmodelSet(
            market=market,
            stage1=stage1,
            place_ability=place_ability,
            win=win_2s,
            ev_corrector=ev_corrector,
            place=place_2s,
            place_ev_corrector=place_ev_corrector,
            wide=wide_2s,
            confidence=conf,
            use_ensemble=use_ensemble,
            benter_combo=benter_combo,
            isotonic_calibrator=isotonic_cal,
        )
```

- [ ] **Step 3: _save_models_local に Benter + Isotonic 保存を追加**

`_save_models_local()` メソッド (line 804付近) の `for surface, sub in models.items()` ループ内に追加:

```python
            # Benter Combination (JSON)
            if sub.benter_combo is not None:
                sub.benter_combo.save(models_dir / f"benter_combo_{surface}.json")

            # Isotonic Calibrator (joblib)
            if sub.isotonic_calibrator is not None:
                import joblib
                joblib.dump(
                    sub.isotonic_calibrator,
                    models_dir / f"isotonic_place_{surface}.joblib",
                )
```

- [ ] **Step 4: テストを実行**

Run: `python -m pytest tests/test_training_pipeline.py -v`
Expected: PASS (古い SubmodelSet 構築を参照するテストがあれば更新)

- [ ] **Step 5: コミット**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "feat: fit Benter combination + isotonic calibration in training pipeline"
```

---

### Task 4: RacePredictor — Benter 推論統合

**Files:**
- Modify: `src/backtest/race_predictor.py:117-132`

- [ ] **Step 1: predict() のエッジ計算を Benter + Isotonic に更新**

Lines 117-132 を以下に置き換える:

```python
        # --- Benter Combination + Isotonic Calibration ---
        # p_place_pred は fundamental model 出力 (オッズ特徴量なし)
        # Benter: logit(p_c) = alpha*logit(p_fund) + beta*logit(p_market) + gamma
        p_market = np.where(
            df["fukuoddslow"] > 0,
            1.0 / df["fukuoddslow"],
            np.nan,
        )
        df["p_market"] = p_market

        benter = submodel.benter_combo
        if benter is not None:
            p_market_clipped = np.clip(
                np.where(df["fukuoddslow"] > 0, 1.0 / df["fukuoddslow"], 0.5),
                0.01, 0.99,
            )
            df["p_place_combined"] = benter.combine(
                df["p_place_pred"].values, p_market_clipped
            )

            # Isotonic calibration (optional post-processing)
            cal = submodel.isotonic_calibrator
            if cal is not None:
                df["p_place_combined"] = cal.transform(df["p_place_combined"])
        else:
            # フォールバック: Benter なし → raw p_place_pred を使用
            df["p_place_combined"] = df["p_place_pred"]

        # Edge = p_combined * odds - 1.0
        df["edge_place"] = df["p_place_combined"] * df["fukuoddslow"] - 1.0
        df["ev_place_direct"] = df["p_place_combined"] * df["fukuoddslow"]
```

- [ ] **Step 2: select_bets() のドキュメントを更新**

Line 144-147 の docstring を更新:
```python
    def select_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
    ) -> list[Bet]:
        """Benter Value Betting: edge >= threshold の馬を選択。

        edge = p_place_combined * fukuoddslow - 1.0
        p_place_combined = Benter(p_fundamental, p_market) + isotonic calibration
        """
```

- [ ] **Step 3: テストを実行**

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: PASS (エッジ計算の変更を反映)

- [ ] **Step 4: コミット**

```bash
git add src/backtest/race_predictor.py
git commit -m "feat: integrate Benter combination + isotonic calibration in RacePredictor"
```

---

### Task 5: Model Loader — Benter JSON + Isotonic joblib 読み込み

**Files:**
- Modify: `src/db/model_loader.py:472-492`

- [ ] **Step 1: benter_lr 読み込みを benter_combo 読み込みに変更**

Lines 472-492 を以下に置き換える:

```python
            # Benter Combination (JSON)
            benter_combo = None
            benter_file = models_dir / f"benter_combo_{surface}.json"
            if benter_file.is_file():
                try:
                    from models.benter_combination import BenterCombination
                    benter_combo = BenterCombination.load(benter_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", benter_file)

            # Isotonic Calibrator (joblib)
            isotonic_calibrator = None
            iso_file = models_dir / f"isotonic_place_{surface}.joblib"
            if iso_file.is_file():
                try:
                    isotonic_calibrator = joblib.load(iso_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", iso_file)

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                place_ev_corrector=place_ev_corr,
                wide=wide,
                confidence=confidence,
                benter_combo=benter_combo,
                isotonic_calibrator=isotonic_calibrator,
            )
```

- [ ] **Step 2: 未使用 import を削除**

`LogisticRegression` import が他で使われていなければ削除。

- [ ] **Step 3: テストを実行**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: PASS

- [ ] **Step 4: コミット**

```bash
git add src/db/model_loader.py
git commit -m "feat: load Benter combination (JSON) + isotonic calibrator (joblib)"
```

---

### Task 6: Paper Trading Reconciler — 精算を実際の配当に変更

**Files:**
- Modify: `src/paper_trading/reconciler.py:82-89`

- [ ] **Step 1: 複勝精算を実際の配当に変更**

Lines 86-89 を以下に置き換える:

```python
            # 複勝的中判定 — 実際の配当 (payfukusyopay) を使用
            payout = 0.0
            if bet_type == "place" and 1 <= finish_pos <= 3:
                # payfukusyopay があれば実際の配当を使用、なければオッズ
                actual_payout = result_row.iloc[0].get("payfukusyopay", None)
                if pd.notna(actual_payout) and actual_payout > 0:
                    payout = bet_row["stake"] * float(actual_payout) / 100.0
                else:
                    payout = bet_row["stake"] * bet_row["odds"]
                n_wins += 1
```

- [ ] **Step 2: EveryDB2 クエリに payfukusyopay を含めることを確認**

`everydb2_queries.get_race_results()` が `payfukusyopay` を返すか確認。
返さない場合は、クエリに追加する必要あり。

- [ ] **Step 3: テストを実行**

Run: `python -m pytest tests/test_paper_reconciler.py -v`
Expected: PASS

- [ ] **Step 4: コミット**

```bash
git add src/paper_trading/reconciler.py
git commit -m "fix: use actual payout (payfukusyopay) for paper trading settlement"
```

---

### Task 7: テスト更新 + 全体検証

**Files:**
- Modify: `tests/test_two_stage_return_model.py` (feature list 更新)
- Modify: `tests/test_race_predictor.py` (edge 計算更新)
- Modify: `tests/test_model_loader.py` (benter_combo 読み込み更新)
- Modify: `tests/test_training_pipeline.py` (SubmodelSet 更新)

- [ ] **Step 1: 全テストを実行**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -50`
Expected: 全テスト PASS (失敗するテストがあれば個別に修正)

- [ ] **Step 2: リント + 型チェック**

Run: `ruff check src/ tests/`
Run: `ruff format --check src/ tests/`
Run: `mypy src/`

- [ ] **Step 3: 失敗したテストを修正**

テスト失敗の原因に応じて:
- Feature list の変更 → テストデータの特徴量列を更新
- SubmodelSet フィールド名変更 → `benter_lr` → `benter_combo`
- Edge 計算変更 → 期待値を更新

- [ ] **Step 4: 最終コミット**

```bash
git add tests/
git commit -m "test: update tests for place prediction overhaul"
```

---

## Summary of Changes

| File | Change | Impact |
|------|--------|--------|
| `src/models/two_stage_return_model.py` | Remove odds from HIT, add 13 horse features, save val data | Fix #1 (double-counting) + Fix #3 (bottleneck) |
| `src/domain/models.py` | `benter_lr` → `bender_combo` + `isotonic_calibrator` | Type system update |
| `src/pipelines/training_pipeline.py` | Fit Benter + Isotonic, save artifacts | Training integration |
| `src/backtest/race_predictor.py` | Benter combine + isotonic + edge formula | Inference integration |
| `src/db/model_loader.py` | Load JSON + joblib artifacts | Model loading |
| `src/paper_trading/reconciler.py` | Use actual payouts | Fix settlement mismatch |
| `tests/*.py` | Update for new feature lists and APIs | Test correctness |

## Expected Outcome

- Fix #1: Edge differentiation restored (no more double-counting)
- Fix #2: Isotonic calibration reduces 2.38x overestimation
- Fix #3: Horse-level features provide genuine alpha beyond market
- Fix #4: Paper trading matches backtest settlement
- Target: ROI improvement from ~65% toward 85-105%
