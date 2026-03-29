# Pipeline Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vectorize feature computation and parallelize model training to achieve 60-80% pipeline speedup without changing model outputs.

**Architecture:** Replace Python for-loops (iterrows) with pandas/numpy vectorized operations. Pre-index past data by horse/jockey keys for O(log n) lookup. Use ThreadPoolExecutor for independent turf/dirt model training. Pre-compute backtest features once instead of per-race.

**Tech Stack:** pandas, numpy, concurrent.futures (stdlib)

**Spec:** `docs/superpowers/specs/2026-03-29-pipeline-performance-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `src/features/bloodline_features.py` | Modify | Vectorize compute() — replace iterrows with column ops |
| `src/features/jockey_context_features.py` | Modify | Vectorize compute() — replace iterrows with column ops |
| `src/features/trainer_context_features.py` | Modify | Vectorize compute() — replace iterrows with column ops |
| `src/models/wide_pair_builder.py` | Modify | Optimize pair generation with numpy arrays |
| `src/features/horse_history_features.py` | Modify | Vectorize compute() — pre-index + searchsorted |
| `src/backtest/engine.py` | Modify | Pre-compute features before race loop |
| `src/pipelines/training_pipeline.py` | Modify | Parallel surface training with ThreadPoolExecutor |

Test files (existing, no new test files needed):
- `tests/test_bloodline_features.py`
- `tests/test_jockey_context_features.py`
- `tests/test_trainer_context_features.py`
- `tests/test_wide_pair_builder.py`
- `tests/test_horse_history_features.py`
- `tests/test_backtest_engine.py`
- `tests/test_training_pipeline.py`

---

### Task 1: BloodlineFeatures Vectorization (S4a)

**Files:**
- Modify: `src/features/bloodline_features.py:73-138`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_bloodline_features.py -v`
Expected: All 17 tests PASS

- [ ] **Step 2: Vectorize `compute()` method**

Replace the entire `compute()` method body (lines 73-138) in `src/features/bloodline_features.py` with vectorized column operations. The `_smoothed_wr` static method stays the same but is used with numpy arrays instead of scalar calls.

```python
def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
    """entry_df (race_id, umaban, ketto_num) -> 血統特徴量 DataFrame。"""
    horses_df = self._load_horses()

    if "ketto_num" not in entry_df.columns or horses_df.empty:
        return entry_df[["race_id", "umaban"]].assign(
            **{c: float("nan") for c in FEATURE_COLS}
        )

    merged = entry_df[["race_id", "umaban", "ketto_num"]].merge(
        horses_df, left_on="ketto_num", right_on="kettonum", how="left"
    )

    result = merged[["race_id", "umaban"]].copy()

    # --- 馬場別勝率 (芝 = ba1) ---
    ba_cols = [f"ba1chakukaisu{i}" for i in range(1, 7)]
    ba_data = merged[ba_cols].fillna(0).astype(float)
    ba1_wins = ba_data["ba1chakukaisu1"]
    ba1_total = ba_data[ba_cols].sum(axis=1)
    result["blood_surface_wr"] = np.where(
        ba1_total == 0, np.nan, (ba1_wins + ALPHA_PRIOR) / (ba1_total + TOTAL_OFFSET)
    )

    # --- 距離別勝率 (短距離 = kyori1) ---
    ky_cols = [f"kyori1chakukaisu{i}" for i in range(1, 7)]
    ky_data = merged[ky_cols].fillna(0).astype(float)
    ky1_wins = ky_data["kyori1chakukaisu1"]
    ky1_total = ky_data[ky_cols].sum(axis=1)
    result["blood_distance_wr"] = np.where(
        ky1_total == 0, np.nan, (ky1_wins + ALPHA_PRIOR) / (ky1_total + TOTAL_OFFSET)
    )

    # --- 馬場状態別勝率 — Phase 2 ---
    result["blood_condition_wr"] = np.nan

    # --- 総合成績勝率 (中央 = chuo) ---
    ch_cols = [f"chuochakukaisu{i}" for i in range(1, 7)]
    ch_data = merged[ch_cols].fillna(0).astype(float)
    ch_wins = ch_data["chuochakukaisu1"]
    ch_total = ch_data[ch_cols].sum(axis=1)
    result["blood_total_wr"] = np.where(
        ch_total == 0, np.nan, (ch_wins + ALPHA_PRIOR) / (ch_total + TOTAL_OFFSET)
    )

    # --- 累計賞金 (log変換) ---
    prize = pd.to_numeric(merged["ruikeihonsyoheiti"], errors="coerce")
    result["blood_prize_log"] = np.where(
        prize.fillna(0) > 0, np.log1p(prize.fillna(0)), np.nan
    )

    # --- 系統コード — Phase 2 ---
    result["blood_keito_cd"] = np.nan

    return result[["race_id", "umaban"] + FEATURE_COLS]
```

- [ ] **Step 3: Run existing tests to verify regression**

Run: `python -m pytest tests/test_bloodline_features.py -v`
Expected: All 11 tests PASS (identical numerical results)

- [ ] **Step 4: Commit**

```bash
git add src/features/bloodline_features.py
git commit -m "perf: BloodlineFeaturesをベクトル化 (iterrows→列演算)"
```

---

### Task 2: JockeyContextFeatures Vectorization (S4b)

**Files:**
- Modify: `src/features/jockey_context_features.py:44-111`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_jockey_context_features.py -v`
Expected: All 15 tests PASS

- [ ] **Step 2: Vectorize `compute()` method**

Replace the iterrows loop (lines 79-104) with vectorized operations after `latest` DataFrame is built. The `latest` groupby logic stays the same; only the feature computation changes.

Replace lines 79-111 with:

```python
        # Vectorized feature computation
        heichi_cols = [f"heichichakukaisu{i}" for i in range(1, 7)]
        heichi_data = latest[heichi_cols].fillna(0).astype(float)
        wins = heichi_data["heichichakukaisu1"]
        total = heichi_data[heichi_cols].sum(axis=1)
        result = latest[["race_id", "umaban"]].copy()
        result["jockey_wr_overall"] = np.where(
            total == 0, np.nan, (wins + 1) / (total + 11)
        )

        ky_cols = [f"kyori1chakukaisu{i}" for i in range(1, 7)]
        ky_data = latest[ky_cols].fillna(0).astype(float)
        ky1_w = ky_data["kyori1chakukaisu1"]
        ky1_t = ky_data[ky_cols].sum(axis=1)
        result["jockey_wr_distance"] = np.where(
            ky1_t == 0, np.nan, (ky1_w + 1) / (ky1_t + 11)
        )

        j5_cols = [f"jyo5chakukaisu{i}" for i in range(1, 7)]
        j5_data = latest[j5_cols].fillna(0).astype(float)
        j5_w = j5_data["jyo5chakukaisu1"]
        j5_t = j5_data[j5_cols].sum(axis=1)
        result["jockey_wr_venue"] = np.where(
            j5_t == 0, np.nan, (j5_w + 1) / (j5_t + 11)
        )

        prize = pd.to_numeric(latest["honsyokinheichi"], errors="coerce")
        result["jockey_prize_log"] = np.log1p(prize.fillna(0))

        return result[["race_id", "umaban"] + FEATURE_COLS]
```

- [ ] **Step 3: Run existing tests to verify regression**

Run: `python -m pytest tests/test_jockey_context_features.py -v`
Expected: All 15 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/features/jockey_context_features.py
git commit -m "perf: JockeyContextFeaturesをベクトル化 (iterrows→列演算)"
```

---

### Task 3: TrainerContextFeatures Vectorization (S4b)

**Files:**
- Modify: `src/features/trainer_context_features.py:44-111`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_trainer_context_features.py -v`
Expected: All 13 tests PASS

- [ ] **Step 2: Vectorize `compute()` method**

Same pattern as Task 2. Replace lines 79-111 with identical vectorized code, adjusting column names (`chokyosicode` instead of `kisyucode`, `trainer_wr_*` instead of `jockey_wr_*`):

```python
        # Vectorized feature computation
        heichi_cols = [f"heichichakukaisu{i}" for i in range(1, 7)]
        heichi_data = latest[heichi_cols].fillna(0).astype(float)
        wins = heichi_data["heichichakukaisu1"]
        total = heichi_data[heichi_cols].sum(axis=1)
        result = latest[["race_id", "umaban"]].copy()
        result["trainer_wr_overall"] = np.where(
            total == 0, np.nan, (wins + 1) / (total + 11)
        )

        ky_cols = [f"kyori1chakukaisu{i}" for i in range(1, 7)]
        ky_data = latest[ky_cols].fillna(0).astype(float)
        ky1_w = ky_data["kyori1chakukaisu1"]
        ky1_t = ky_data[ky_cols].sum(axis=1)
        result["trainer_wr_distance"] = np.where(
            ky1_t == 0, np.nan, (ky1_w + 1) / (ky1_t + 11)
        )

        j5_cols = [f"jyo5chakukaisu{i}" for i in range(1, 7)]
        j5_data = latest[j5_cols].fillna(0).astype(float)
        j5_w = j5_data["jyo5chakukaisu1"]
        j5_t = j5_data[j5_cols].sum(axis=1)
        result["trainer_wr_venue"] = np.where(
            j5_t == 0, np.nan, (j5_w + 1) / (j5_t + 11)
        )

        prize = pd.to_numeric(latest["honsyokinheichi"], errors="coerce")
        result["trainer_prize_log"] = np.log1p(prize.fillna(0))

        return result[["race_id", "umaban"] + FEATURE_COLS]
```

- [ ] **Step 3: Run existing tests to verify regression**

Run: `python -m pytest tests/test_trainer_context_features.py -v`
Expected: All 13 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/features/trainer_context_features.py
git commit -m "perf: TrainerContextFeaturesをベクトル化 (iterrows→列演算)"
```

---

### Task 4: WidePairBuilder Optimization (S4c)

**Files:**
- Modify: `src/models/wide_pair_builder.py:25-56`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_wide_pair_builder.py -v`
Expected: All 7 tests PASS (TestWideJointPairBuilder class)

- [ ] **Step 2: Optimize `build()` method**

Replace the nested loop in `build()` with pre-extracted numpy arrays and `itertools.combinations`. Keep `_build_pair` logic but inline it for speed.

Replace the `build()` method (lines 25-56):

```python
    def build(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """全レースの馬ペアを構築"""
        if entry_df.empty:
            return pd.DataFrame()

        all_pairs: list[dict[str, Any]] = []

        for _, group in entry_df.groupby("race_id"):
            horses = group.sort_values("umaban").reset_index(drop=True)
            n = len(horses)
            if n < 2:
                continue

            # Pre-extract as numpy arrays for fast access
            umabans = horses["umaban"].values.astype(int)
            finish_positions = horses["finish_pos"].values.astype(int)
            popularity_ranks = horses["popularity_rank"].values.astype(int)
            running_styles = horses["running_style"].values.astype(int)

            # Get wide odds columns from first row
            first_row = horses.iloc[0]
            race_shared = {
                "race_id": first_row["race_id"],
                "surface": first_row["surface"],
                "distance_bin": first_row["distance_bin"],
                "track_condition_code": first_row["track_condition_code"],
                "grade_code": first_row["grade_code"],
                "field_size": first_row["field_size"],
            }

            # Build wide_odds lookup from first row columns
            wide_odds_cache: dict[str, float] = {}
            for col in horses.columns:
                if col.startswith("wide_odds_"):
                    val = horses[col].iloc[0]
                    wide_odds_cache[col] = float(val) if not pd.isna(val) else 0.0

            from itertools import combinations  # noqa: move to top of function for efficiency

            for i, j in combinations(range(n), 2):
                lo, hi = min(umabans[i], umabans[j]), max(umabans[i], umabans[j])
                odds_col = f"wide_odds_{lo}_{hi}"

                all_pairs.append({
                    **race_shared,
                    "umaban_a": int(umabans[i]),
                    "umaban_b": int(umabans[j]),
                    "joint_hit": int(finish_positions[i] <= 3 and finish_positions[j] <= 3),
                    "popularity_sum": int(popularity_ranks[i] + popularity_ranks[j]),
                    "running_style_combo": int(running_styles[i] + running_styles[j]),
                    "wide_odds": wide_odds_cache.get(odds_col, 0.0),
                })

        if not all_pairs:
            return pd.DataFrame()

        pair_df = pd.DataFrame(all_pairs)
        logger.info(f"Built {len(pair_df)} pairs from {entry_df['race_id'].nunique()} races")
        return pair_df
```

Note: Remove the now-unused `_build_pair` and `_lookup_wide_odds` methods.

- [ ] **Step 3: Run existing tests to verify regression**

Run: `python -m pytest tests/test_wide_pair_builder.py -v`
Expected: All 7 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/models/wide_pair_builder.py
git commit -m "perf: WidePairBuilderを最適化 (numpy配列化+itertools)"
```

---

### Task 5: HorseHistoryFeatures Vectorization (S1) — CRITICAL PATH

This is the largest and most impactful task. The approach: pre-index past data by ketto_num/kisyu_code for O(1) lookup + O(log n) searchsorted instead of O(M) filtering per horse.

**Files:**
- Modify: `src/features/horse_history_features.py:44-55,161-394`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: All 15 tests PASS

- [ ] **Step 2: Add vectorized `_norm_finish_logit_vec` function**

Add after the existing `_norm_finish_logit` function (around line 55):

```python
def _norm_finish_logit_vec(finish_pos: np.ndarray, field_size: np.ndarray) -> np.ndarray:
    """Vectorized version of _norm_finish_logit."""
    score = 1.0 - (finish_pos - 1) / np.maximum(field_size - 1, 1)
    score = np.clip(score, CLIP_LO, CLIP_HI)
    result = np.log(score / (1.0 - score))
    result[field_size < 8] = np.nan
    return result
```

- [ ] **Step 3: Rewrite `compute()` method with pre-indexed lookup**

Replace the `compute()` method (lines 161-394). Key changes:
- Pre-index `past_df` by `ketto_num` and `kisyu_code` into sorted dicts
- Use `searchsorted` for fast date-based lookup
- Vectorize feature computation on the small (≤3 row) result

```python
    def compute(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        target_race_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """過去成績特徴量を計算"""
        if target_race_ids is not None:
            entry_df = entry_df[entry_df["race_id"].isin(target_race_ids)]

        horses = entry_df[["race_id", "umaban", "ketto_num", "kisyu_code"]].copy()
        if "race_date" not in horses.columns:
            date_map = race_df.set_index("race_id")["race_date"]
            horses["race_date"] = horses["race_id"].map(date_map)

        unique_ketto = horses["ketto_num"].unique().tolist()
        unique_kisyu = horses["kisyu_code"].unique().tolist()

        if not unique_ketto:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        entries_hist, races_hist = self._get_history()

        ketto_set = set(unique_ketto)
        kisyu_set = set(unique_kisyu)

        entries_filtered = entries_hist[
            entries_hist["ketto_num"].isin(ketto_set) | entries_hist["kisyu_code"].isin(kisyu_set)
        ].copy()

        if entries_filtered.empty:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        race_cols = ["race_id", "field_size", "race_date", "track_cd", "distance"]
        races_subset = races_hist[races_hist["race_id"].isin(entries_filtered["race_id"].unique())]
        entries_no_date = entries_filtered.drop(columns=["race_date"], errors="ignore")
        past_df = entries_no_date.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )

        # Add distance_bin
        if "distance_bin" not in past_df.columns and "track_cd" in past_df.columns and "distance" in past_df.columns:
            is_turf = (past_df["track_cd"] >= 10) & (past_df["track_cd"] <= 22)
            dist = past_df["distance"]
            past_df["distance_bin"] = "unknown"
            past_df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
            past_df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
            past_df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
            past_df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
            past_df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"

        past_df["valid_field"] = (past_df["field_size"] >= 8).astype(int)

        # Pre-index past data by ketto_num (sorted by race_date)
        past_df_sorted = past_df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
        past_by_ketto: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted.groupby("ketto_num")
        }

        # Pre-index past data by kisyu_code for jockey_surprise (finish_pos > 0 AND win_odds > 0)
        past_by_kisyu: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted[
                (past_df_sorted["finish_pos"] > 0) & (past_df_sorted["win_odds"] > 0)
            ].groupby("kisyu_code")
        }

        # Pre-index past data by kisyu_code for jockey_cond_wr (finish_pos > 0 only, no win_odds filter)
        past_by_kisyu_all: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted[
                (past_df_sorted["finish_pos"] > 0)
            ].groupby("kisyu_code")
        }

        # Weight column name
        weight_col = "ba_taijyu" if "ba_taijyu" in entry_df.columns else "weight"

        total = len(horses)
        results: list[dict] = []
        empty_past = pd.DataFrame()

        for i, (_, row) in enumerate(horses.iterrows()):
            if i % 200 == 0:
                print(
                    f"  HorseHistoryFeatures: {i}/{total} ({i / max(total, 1) * 100:.0f}%)",
                    flush=True,
                )
            race_date = row["race_date"]
            ketto = row["ketto_num"]
            kisyu = row["kisyu_code"]

            # --- Horse features: O(1) lookup + O(log m) searchsorted ---
            horse_past_all = past_by_ketto.get(ketto, empty_past)
            if len(horse_past_all) > 0:
                valid_past = horse_past_all[
                    (horse_past_all["valid_field"] == 1)
                    & (horse_past_all["finish_pos"] > 0)
                ]
                # searchsorted for date cutoff
                idx = valid_past["race_date"].values.searchsorted(race_date, side="left")
                horse_past = valid_past.iloc[max(0, idx - 3):idx]
            else:
                horse_past = empty_past

            # norm_finish_logit_avg
            if len(horse_past) > 0:
                logits = _norm_finish_logit_vec(
                    horse_past["finish_pos"].values.astype(float),
                    horse_past["field_size"].values.astype(float),
                )
                norm_finish_logit_avg: float = float(np.nanmean(logits))
            else:
                norm_finish_logit_avg = float("nan")

            # haron_time_l3_avg
            if "haron_time_l3" in horse_past.columns and len(horse_past) > 0:
                ht_vals = horse_past["haron_time_l3"].dropna()
                haron_time_l3_avg: float = float(ht_vals.tail(3).mean()) if len(ht_vals) > 0 else float("nan")
            else:
                haron_time_l3_avg = float("nan")

            # haron_time_l3_zscore
            if "haron_time_l3" in horse_past.columns and "distance_bin" in horse_past.columns and len(horse_past) > 0:
                ht = horse_past["haron_time_l3"]
                db = horse_past["distance_bin"]
                valid = ht.notna() & db.notna()
                if valid.sum() > 0:
                    grp_stats = horse_past.loc[valid].groupby("distance_bin")["haron_time_l3"].agg(["mean", "std"])
                    zscores: list[float] = []
                    for _, r in horse_past.loc[valid].iterrows():
                        bin_key = r["distance_bin"]
                        if bin_key in grp_stats.index and not pd.isna(grp_stats.loc[bin_key, "std"]):
                            z = (r["haron_time_l3"] - grp_stats.loc[bin_key, "mean"]) / grp_stats.loc[bin_key, "std"]
                            zscores.append(z)
                        else:
                            zscores.append(float("nan"))
                    haron_time_l3_zscore: float = float(pd.Series(zscores).tail(3).mean()) if zscores else float("nan")
                else:
                    haron_time_l3_zscore = float("nan")
            else:
                haron_time_l3_zscore = float("nan")

            # time_diff_avg
            if "time_diff" in horse_past.columns and len(horse_past) > 0:
                td_vals = horse_past["time_diff"].dropna()
                time_diff_avg: float = float(td_vals.tail(3).mean()) if len(td_vals) > 0 else float("nan")
            else:
                time_diff_avg = float("nan")

            # corner_1c_avg
            if "corner_1c" in horse_past.columns and len(horse_past) > 0:
                c1_vals = horse_past["corner_1c"].dropna()
                corner_1c_avg: float = float(c1_vals.tail(3).mean()) if len(c1_vals) > 0 else float("nan")
            else:
                corner_1c_avg = float("nan")

            # corner_4c_avg
            if "corner_4c" in horse_past.columns and len(horse_past) > 0:
                c4_vals = horse_past["corner_4c"].dropna()
                corner_4c_avg: float = float(c4_vals.tail(3).mean()) if len(c4_vals) > 0 else float("nan")
            else:
                corner_4c_avg = float("nan")

            # closing_index_avg
            if all(c in horse_past.columns for c in ["corner_4c", "finish_pos", "field_size"]) and len(horse_past) > 0:
                valid_ci = horse_past.dropna(subset=["corner_4c", "finish_pos", "field_size"])
                valid_ci = valid_ci[valid_ci["field_size"] > 1]
                if len(valid_ci) > 0:
                    norm_4c = (valid_ci["corner_4c"] - 1) / (valid_ci["field_size"] - 1)
                    norm_finish = (valid_ci["finish_pos"] - 1) / (valid_ci["field_size"] - 1)
                    closing_indices = norm_4c - norm_finish
                    closing_index_avg: float = float(closing_indices.tail(3).mean())
                else:
                    closing_index_avg = float("nan")
            else:
                closing_index_avg = float("nan")

            # kyakusitu_cd
            if "kyakusitu" in horse_past.columns and len(horse_past) > 0:
                kt_vals = horse_past["kyakusitu"].dropna()
                kyakusitu_cd: float | int = int(kt_vals.iloc[-1]) if len(kt_vals) > 0 else float("nan")
            else:
                kyakusitu_cd = float("nan")

            # --- Jockey features: O(1) lookup + O(log m) searchsorted ---
            jockey_past_all = past_by_kisyu.get(kisyu, empty_past)
            if len(jockey_past_all) > 0:
                idx = jockey_past_all["race_date"].values.searchsorted(race_date, side="left")
                jockey_past = jockey_past_all.iloc[max(0, idx - 100):idx]
            else:
                jockey_past = empty_past

            if len(jockey_past) >= 30:
                expected = (PAYOUT_RATE / jockey_past["win_odds"].clip(lower=1.1)).sum()
                actual = int((jockey_past["finish_pos"] == 1).sum())
                jockey_surprise: float = _compute_jockey_surprise(
                    actual, len(jockey_past), expected
                )
            else:
                jockey_surprise = float("nan")

            # jockey_cond_wr — uses past_by_kisyu_all (finish_pos > 0 only, no win_odds filter)
            jockey_all_past = past_by_kisyu_all.get(kisyu, empty_past)
            if len(jockey_all_past) > 0:
                idx = jockey_all_past["race_date"].values.searchsorted(race_date, side="left")
                jockey_all = jockey_all_past.iloc[:idx]
                total_rides = len(jockey_all)
            else:
                jockey_all = empty_past
                total_rides = 0
            total_wins = int((jockey_all["finish_pos"] == 1).sum()) if total_rides > 0 else 0

            k_smooth = 25
            if total_rides >= 10:
                cond_wr = total_wins / max(total_rides, 1)
                global_wr = total_wins / max(total_rides, 1)
                w = min(total_rides / (total_rides + k_smooth), 1.0)
                jockey_cond_wr: float = float(w * cond_wr + (1 - w) * global_wr)
            else:
                jockey_cond_wr = float("nan")

            # weight_absolute
            weight_val = entry_df.loc[
                (entry_df["race_id"] == row["race_id"]) & (entry_df["umaban"] == row["umaban"]),
                weight_col,
            ].values
            weight_absolute: float = (
                float(weight_val[0])
                if len(weight_val) > 0 and pd.notna(weight_val[0])
                else float("nan")
            )

            results.append({
                "race_id": row["race_id"],
                "umaban": row["umaban"],
                "norm_finish_logit_avg": norm_finish_logit_avg,
                "haron_time_l3_avg": haron_time_l3_avg,
                "haron_time_l3_zscore": haron_time_l3_zscore,
                "time_diff_avg": time_diff_avg,
                "corner_1c_avg": corner_1c_avg,
                "corner_4c_avg": corner_4c_avg,
                "closing_index_avg": closing_index_avg,
                "kyakusitu_cd": kyakusitu_cd,
                "jockey_surprise": jockey_surprise,
                "jockey_cond_wr": jockey_cond_wr,
                "weight_absolute": weight_absolute,
            })

        print(f"  HorseHistoryFeatures: done ({len(results)} rows)", flush=True)
        return pd.DataFrame(results)
```

**Key optimization points:**
1. `past_by_ketto` dict: O(1) lookup instead of O(M) DataFrame filtering per horse
2. `searchsorted` on sorted dates: O(log m) instead of O(m) linear scan
3. `_norm_finish_logit_vec`: numpy vectorized instead of per-row `math.log`
4. `past_by_kisyu` dict: same optimization for jockey features

- [ ] **Step 4: Run existing tests to verify regression**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/horse_history_features.py
git commit -m "perf: HorseHistoryFeaturesをpre-index+searchsortedで高速化"
```

---

### Task 6: Backtest Single-Pass Feature Computation (S2)

**Files:**
- Modify: `src/backtest/engine.py:89-226`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: All 7 tests PASS

- [ ] **Step 2: Pre-compute HorseHistoryFeatures before race loop**

In `src/backtest/engine.py`, after the feature generation section (around line 112) and before the race loop (line 121), add pre-computation. Replace lines 138-149 (the per-race HorseHistoryFeatures/JockeyContext/TrainerContext instantiation) with pre-computed merges.

Add before `for race_id in race_ids:` loop (after line 118):

```python
        # Pre-compute HorseHistoryFeatures for all races (single-pass)
        # Note: These imports are inline (not top-level) to match existing pattern
        # in the codebase and avoid circular import issues
        from features.horse_history_features import HorseHistoryFeatures

        hist = HorseHistoryFeatures(repo=self.repo)
        hist_all = hist.compute(race_df, entry_df)

        # Pre-compute Jockey/Trainer context for all races
        from features.jockey_context_features import JockeyContextFeatures
        from features.trainer_context_features import TrainerContextFeatures

        jockey_ctx = JockeyContextFeatures(self.repo)
        jockey_all = jockey_ctx.compute(feat_df)
        trainer_ctx = TrainerContextFeatures(self.repo)
        trainer_all = trainer_ctx.compute(feat_df)
```

Then in the loop body, replace lines 138-177 (per-race feature computation) with:

```python
            # 3c. HorseHistoryFeatures — pre-computed, merge from hist_all
            race_hist = hist_all[hist_all["race_id"] == race_id]
            race_df_single = race_df_single.merge(race_hist, on=["race_id", "umaban"], how="left")
            race_df_single = HorseHistoryFeatures.add_race_transforms(race_df_single)

            # Group E: 交互作用特徴量
            from features.interaction_features import compute_interaction_features

            race_df_single = compute_interaction_features(race_df_single)
```

And for Jockey/Trainer context in the loop (replace lines 164-177):

```python
            # Group C/D: 騎手/調教師コンテキスト — pre-computed
            race_jockey = jockey_all[jockey_all["race_id"] == race_id]
            race_df_single = race_df_single.merge(
                race_jockey, on=["race_id", "umaban"], how="left"
            )
            race_trainer = trainer_all[trainer_all["race_id"] == race_id]
            race_df_single = race_df_single.merge(
                race_trainer, on=["race_id", "umaban"], how="left"
            )
```

- [ ] **Step 3: Run existing tests to verify regression**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: All 7 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/backtest/engine.py
git commit -m "perf: バックテストの特徴量計算をループ前一括化"
```

---

### Task 7: Parallel Model Training (S3)

**Files:**
- Modify: `src/pipelines/training_pipeline.py:1-10,101-111`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_training_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 2: Add ThreadPoolExecutor import and parallelize surface loop**

Add import at the top of `src/pipelines/training_pipeline.py` (after line 9):

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Replace the sequential surface loop (lines 102-111):

```python
        # 3. 各 surface ごとに学習 (parallel)
        models: dict[str, SubmodelSet] = {}
        surfaces_to_train: list[tuple[str, pd.DataFrame]] = []
        for surface in ["turf", "dirt"]:
            subset_df = feat_df[feat_df["surface"] == surface].copy()
            if len(subset_df) < 1000:
                logger.warning(f"Skipping {surface}: insufficient data ({len(subset_df)})")
                continue
            surfaces_to_train.append((surface, subset_df))

        if len(surfaces_to_train) == 1:
            # Single surface — no parallelism needed
            surface, subset_df = surfaces_to_train[0]
            sub = self._train_submodel(subset_df)
            models[surface] = sub
            logger.info(f"Trained {surface} submodel")
        elif len(surfaces_to_train) >= 2:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(self._train_submodel, subset_df): surface
                    for surface, subset_df in surfaces_to_train
                }
                for future in as_completed(futures):
                    surface = futures[future]
                    try:
                        models[surface] = future.result()
                        logger.info(f"Trained {surface} submodel (parallel)")
                    except Exception as e:
                        logger.error(f"Failed to train {surface} submodel: {e}")
                        raise
```

**Thread safety note:** Each `_train_submodel` call operates on its own `subset_df` copy and creates fresh model instances. `self.repo` caches are read-only after initial load. `self._race_df` and `self._entry_df` are read-only. No shared mutable state.

**LightGBM num_threads:** When running 2 threads, each LightGBM model should use `num_threads = max(1, os.cpu_count() // 2)` to avoid oversubscription. This should be set in each model's training params (e.g., in `MarketModel.train()`, `AbilityModel.train()`, etc.). Add `n_jobs = max(1, (os.cpu_count() or 4) // 2)` to all LightGBM model constructors.

- [ ] **Step 3: Run existing tests to verify regression**

Run: `python -m pytest tests/test_training_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "perf: 芝/ダートモデル学習をThreadPoolExecutorで並列化"
```

---

### Task 8: Full Test Suite Verification

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

- [ ] **Step 3: Run type checker**

Run: `mypy src/`
Expected: No new errors (existing errors are OK)

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "perf: パイプライン最適化後のlint/型チェック修正"
```
