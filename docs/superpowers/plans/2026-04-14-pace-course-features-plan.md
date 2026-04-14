# PaceAptitude & CourseFeatures 活用化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pace_aptitude と course_features を実際に計算し、モデル学習に活用する

**Architecture:** HorseHistoryFeatures と同様のパターンで compute_batch() メソッドを追加し、TrainingPipeline で呼び出す

**Tech Stack:** Python 3.11, pandas, numpy, LightGBM, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/features/pace_aptitude_features.py` | Modify | compute_batch() メソッド追加 |
| `src/features/course_features.py` | Modify | compute_batch() メソッド追加 |
| `src/pipelines/training_pipeline.py` | Modify | _train_submodel() に呼び出し追加 |
| `src/features/feature_engine.py` | Modify | build_all() のプレースホルダーNaN削除 |
| `tests/test_pace_aptitude_features.py` | Modify | compute_batch() のテスト追加 |
| `tests/test_course_features.py` | Modify | compute_batch() のテスト追加 |

---

## Task 1: PaceAptitudeFeatures.compute_batch() 実装

**Files:**
- Modify: `src/features/pace_aptitude_features.py`
- Modify: `tests/test_pace_aptitude_features.py`

**既存パターン (SireFeatures.compute_batch):**
- 入力 DataFrame に kettonum, race_date, surface, kyori, jyocd などを含む
- sire_id ごとの groupby + searchsorted で lookup
- 結果を DataFrame で返す

**変更方針 (HorseHistoryFeatures パターン):**
- 入力: `df` (kettonum, race_id, race_date, surface, distance_bin, jyocd を含む)
- `load_history_entries()` / `load_history_races()` で過去走データを取得
- kettonum ごとに過去走を抽出 → `self.compute()` 呼び出し
- 結果を df にマージ

- [ ] **Step 1: テストを書く**

`tests/test_pace_aptitude_features.py` に追加:

```python
class TestPaceAptitudeComputeBatch:
    def test_compute_batch_returns_three_columns(self):
        """compute_batch が pace_aptitude, front_pace_wr, closing_pace_wr を返す"""
        from features.pace_aptitude_features import PaceAptitudeFeatures
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            "kettonum": ["K1", "K1", "K2"],
            "race_id": ["R1", "R2", "R1"],
            "race_date": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-15"]),
            "surface": ["turf", "turf", "dirt"],
            "distance_bin": ["mile", "mile", "sprint"],
            "jyocd": ["01", "01", "02"],  # 競馬場コード
        })
        
        feat = PaceAptitudeFeatures()
        # モストの簡易化のため、ここでは過去走データなしで NaN を返す
        # 実際の Integration Test で history データありのパターンを検証
        result = feat.compute_batch(df)
        
        assert "pace_aptitude" in result.columns
        assert "front_pace_wr" in result.columns
        assert "closing_pace_wr" in result.columns
        assert len(result) == 3  # 入力と同じ行数

    def test_compute_batch_with_history_data(self):
        """過去走データがある場合、正しく計算される"""
        # これは integration test 的なテスト
        # 実際の load_history_entries/load_history_races を mock する必要がある
        # TODO: 実装後に詳細を追加
        pass
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_pace_aptitude_features.py::TestPaceAptitudeComputeBatch::test_compute_batch_returns_three_columns -v`
Expected: FAIL (compute_batch が存在しない)

- [ ] **Step 3: compute_batch() を実装**

`src/features/pace_aptitude_features.py` に追加:

```python
    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """全馬のペース適性特徴量を一括計算する
        
        Args:
            df: kettonum, race_id, race_date, surface, distance_bin, jyocd 列を持つ DataFrame
        
        Returns:
            pace_aptitude, front_pace_wr, closing_pace_wr 列を持つ DataFrame
        """
        from db.readers import load_history_entries, load_history_races
        
        # 過去走データをロード（HorseHistoryFeatures と同じパターン）
        entries_hist = load_history_entries(self.store)
        races_hist = load_history_races(self.store)
        
        # 必要列の結合
        race_cols = ["race_id", "trackcd", "kyori", "surface", "track_condition_code"]
        races_subset = races_hist[races_hist["race_id"].isin(entries_hist["race_id"])]
        past_df = entries_hist.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left"
        )
        
        # distance_bin 追加 (race_df と同じマッピング)
        if "distance_bin" not in past_df.columns and "kyori" in past_df.columns:
            is_turf = past_df["surface"] == "turf"
            dist = past_df["kyori"]
            past_df["distance_bin"] = "unknown"
            past_df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
            past_df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
            past_df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
            past_df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
            past_df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"
        
        # jyocd を文字列化
        if "jyocd" in past_df.columns:
            past_df["jyocd"] = past_df["jyocd"].astype(str).str.zfill(2, "0")
        
        # syussotosu > 0 のみ有効な出走のみ対象（HorseHistoryFeatures と同じ）
        past_df["valid_field"] = (past_df["syussotosu"].fillna(-1) >= 8).astype(int)
        
        # kettonum ごとの特徴量計算
        results = []
        for kettonum in df["kettonum"].unique():
            target_races = df[df["kettonum"] == kettonum]["race_id"].unique()
            
            # 該当馬の過去走を抽出
            horse_past = past_df[
                (past_df["kettonum"] == kettonum) &
                (past_df["syussotosu"].fillna(-1) >= 8)  # 有効な出走のみ
            ].copy()
            
            # 各対象レースの特徴量を計算
            for target_id in target_races:
                target_date = df[df["race_id"] == target_id]["race_date"].values[0]
                
                # 該当馬の過去走で、対象レースより前のデータのみ
                past_before_target = horse_past[horse_past["race_date"] < target_date]
                
                # compute() を呼び出し
                feat_dict = self.compute(past_before_target, target_date)
                
                # 結果を保存（該当馬・レースの行）
                row_mask = (df["kettonum"] == kettonum) & (df["race_id"] == target_id)
                for col, val in feat_dict.items():
                    df.loc[row_mask, col] = val
        
        # 結果列のみを返す
        return df[["kettonum", "race_id", "pace_aptitude", "front_pace_wr", "closing_pace_wr"]].copy()
```

**PIT注意点:**
- `self.compute()` 内で `history[history["race_date"] < target_date]` フィルタがある
- 呼び出し側でもさらにフィルタリングしている → 二重ガードで安全

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_pace_aptitude_features.py::TestPaceAptitudeComputeBatch -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/pace_aptitude_features.py tests/test_pace_aptitude_features.py
git commit -m "feat: PaceAptitudeFeatures.compute_batch() メソッド追加"
```

---

## Task 2: CourseFeatures.compute_batch() 実装

**Files:**
- Modify: `src/features/course_features.py`
- Modify: `tests/test_course_features.py`

- [ ] **Step 1: テストを書く**

`tests/test_course_features.py` に追加:

```python
class TestCourseFeaturesComputeBatch:
    def test_compute_batch_returns_two_columns(self):
        """compute_batch が course_wr, course_distance_wr を返す"""
        from features.course_features import CourseFeatures
        import pandas as pd
        
        df = pd.DataFrame({
            "kettonum": ["K1", "K1", "K2"],
            "race_id": ["R1", "R2", "R1"],
            "race_date": pd.to_datetime(["2024-06-01", "2024-06-15", "2024-06-15"]),
            "surface": ["turf", "turf", "dirt"],
            "distance_bin": ["mile", "sprint", "sprint"],
            "jyocd": ["01", "01", "02"],  # 競馬場コード
        })
        
        feat = CourseFeatures()
        result = feat.compute_batch(df)
        
        assert "course_wr" in result.columns
        assert "course_distance_wr" in result.columns
        assert len(result) == 3

    def test_compute_batch_filters_by_jyocd(self):
        """jyocd ごとの正しいフィルタリング"""
        # TODO: 実装後に詳細を追加
        pass
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_course_features.py::TestCourseFeaturesComputeBatch::test_compute_batch_returns_two_columns -v`
Expected: FAIL

- [ ] **Step 3: compute_batch() を実装**

`src/features/course_features.py` に追加:

```python
    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """全馬のコース適性特徴量を一括計算する
        
        Args:
            df: kettonum, race_id, race_date, surface, distance_bin, jyocd 列を持つ DataFrame
        
        Returns:
            course_wr, course_distance_wr 列を持つ DataFrame
        """
        from db.readers import load_history_entries, load_history_races
        
        # 過去走データをロード
        entries_hist = load_history_entries(self.store)
        races_hist = load_history_races(self.store)
        
        # 必要列の結合
        race_cols = ["race_id", "trackcd", "kyori", "surface", "track_condition_code"]
        races_subset = races_hist[races_hist["race_id"].isin(entries_hist["race_id"])]
        past_df = entries_hist.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left"
        )
        
        # distance_bin 追加
        if "distance_bin" not in past_df.columns and "kyori" in past_df.columns:
            is_turf = past_df["surface"] == "turf"
            dist = past_df["kyori"]
            past_df["distance_bin"] = "unknown"
            past_df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
            past_df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
            past_df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
            past_df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
            past_df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"
        
        # jyocd を文字列化（2桁ゼロ埋め）
        if "jyocd" in past_df.columns:
            past_df["jyocd"] = past_df["jyocd"].astype(str).str.zfill(2, "0")
        
        # syussotosu > 0 のみ有効な出走のみ対象
        past_df["valid_field"] = (past_df["syussotosu"].fillna(-1) >= 8).astype(int)
        
        # kettonum ごとの特徴量計算
        for kettonum in df["kettonum"].unique():
            target_races = df[df["kettonum"] == kettonum]["race_id"].unique()
            
            # 該当馬の過去走を抽出
            horse_past = past_df[
                (past_df["kettonum"] == kettonum) &
                (past_df["syussotosu"].fillna(-1) >= 8)
            ].copy()
            
            # 各対象レースの特徴量を計算
            for target_id in target_races:
                target_date = df[df["race_id"] == target_id]["race_date"].values[0]
                
                # 該当馬の過去走で、対象レースより前のデータのみ
                past_before_target = horse_past[horse_past["race_date"] < target_date]
                
                # race_df から jyocd, distance_bin を取得（同一レース内では全馬同じ値）
                race_row = df[df["race_id"] == target_id].iloc[0]
                jyocd = race_row.get("jyocd", "")
                distance_bin = race_row.get("distance_bin", "")
                
                # compute() を呼び出し
                feat_dict = self.compute(past_before_target, jyocd, distance_bin, target_date)
                
                # 結果を保存
                row_mask = (df["kettonum"] == kettonum) & (df["race_id"] == target_id)
                for col, val in feat_dict.items():
                    df.loc[row_mask, col] = val
        
        # 結果列のみを返す
        return df[["kettonum", "race_id", "course_wr", "course_distance_wr"]].copy()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_course_features.py::TestCourseFeaturesComputeBatch -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/course_features.py tests/test_course_features.py
git commit -m "feat: CourseFeatures.compute_batch() メソッド追加"
```

---

## Task 3: TrainingPipeline 統合

**Files:**
- Modify: `src/pipelines/training_pipeline.py`

- [ ] **Step 1: _train_submodel に呼び出しを追加**

`_train_submodel()` メソッド内、HorseHistoryFeatures 呼び出しの直後（line 285 の後）に追加:

```python
        # Group C: ペース適性特徴量 (HorseHistoryFeatures の直後)
        from features.pace_aptitude_features import PaceAptitudeFeatures

        with TimingContext(f"{surface}/pace_aptitude"):
            pace_feat = PaceAptitudeFeatures()
            pace_df = pace_feat.compute_batch(df)
            df = df.merge(pace_df[["kettonum", "race_id", "pace_aptitude", "front_pace_wr", "closing_pace_wr"]],
                          on=["kettonum", "race_id"], how="left")
        
        # Group D: コース別適性特徴量 (pace_aptitude の直後)
        from features.course_features import CourseFeatures

        with TimingContext(f"{surface}/course_features"):
            course_feat = CourseFeatures()
            course_df = course_feat.compute_batch(df)
            df = df.merge(course_df[["kettonum", "race_id", "course_wr", "course_distance_wr"]],
                          on=["kettonum", "race_id"], how="left")
```

**注意点:**
- `race_df` と `entry_df` は既に `_train_submodel` 呼び出し時に self._race_df, self._entry_df に保存済み（line 277 参照）
- df には既に kettonum, race_id, surface, distance_bin, jyocd 列が含まれている

- [ ] **Step 2: 全テスト実行**

Run: `python -m pytest tests/test_training_pipeline.py -v -k "test_train"`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "feat: TrainingPipeline に pace_aptitude と course_features の計算を統合"
```

---

## Task 4: feature_engine.py のプレースホルダー削除

**Files:**
- Modify: `src/features/feature_engine.py`

- [ ] **Step 1: build_all() のプレースホルダーを削除**

Line 135-144 の Group C ブロックを削除:

```python
        # Group C: ペース適性特徴量 (削除 - TrainingPipeline で計算)
        # with TimingContext("build_all/pace_aptitude"):
        #     from features.pace_aptitude_features import PaceAptitudeFeatures
        #     pace_feat = PaceAptitudeFeatures()
        #     df["pace_aptitude"] = np.nan
        #     df["front_pace_wr"] = np.nan
        #     df["closing_pace_wr"] = np.nan
```

Line 146-150 の Group D ブロックも削除:

```python
        # Group D: コース別適性特徴量 (削除 - TrainingPipelineで計算)
        # with TimingContext("build_all/course_features"):
        #     from features.course_features import CourseFeatures
        #     df["course_wr"] = np.nan
        #     df["course_distance_wr"] = _np.nan
```

- [ ] **Step 2: 全テスト実行**

Run: `python -m pytest tests/test_feature_engine.py -v`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add src/features/feature_engine.py
git commit -m "refactor: build_all() の pace/course プレースホルダーを削除 (TrainingPipelineで計算するよう変更)"
```

---

## Task 5: 統合テストとバックテスト

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: バックテスト実行**

Run: `python scripts/run_backtest.py --train-start 20210101 --train-end 20241231 --test-start 20250101 --test-end 20251231 --ensemble`

- [ ] **Step 3: 結果を記録**

`docs/backlog/2026-04-14-pace-course-features-result.md` に結果を記録
- 比較: Phase 4 単体 (ROI 75.0%) vs 今回 (pace+course 活用後)
- 期待: ROI 向上（新特徴量5列が実際に計算されるため）

- [ ] **Step 4: 最終コミット**

```bash
git add docs/backlog/2026-04-14-pace-course-features-result.md
git commit -m "docs: pace_aptitude + course_features 活用化の結果を記録"
```
