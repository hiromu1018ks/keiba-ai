# JRA-only Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NARレース (jyocd >= 30) を学習・バックテストの両方から除外し、Dirt Market Model の正常な学習を復旧させる。

**Architecture:** 学習パイプラインとバックテストエンジンの両方で、surface分割前に `jyocd` 列が 1-10 (JRA) のエントリのみを保持するフィルタを追加する。推論パイプラインは変更なし。

**Tech Stack:** Python 3.11, pandas, LightGBM

---

## File Structure

| File | Action | Description |
|------|--------|-------------|
| `src/pipelines/training_pipeline.py` | Modify (line ~150) | JRAフィルタを surface分割ループの直前に追加 |
| `src/backtest/engine.py` | Modify (line ~193) | JRAフィルタを `add_distance_band_features()` の直後に追加 |
| `tests/test_training_pipeline.py` | Modify | JRAフィルタのテストを追加 |
| `tests/test_backtest_engine.py` | Modify | JRAフィルタのテストを追加 |

---

### Task 1: Training Pipeline — JRAフィルタのテストと実装 (TDD)

**Files:**
- Modify: `tests/test_training_pipeline.py` (テストクラス追加)
- Modify: `src/pipelines/training_pipeline.py:150-162` (フィルタ追加)

**前提:** `_make_feature_df()` は `jyocd` 列を含まない。テスト内で `jyocd` 列を追加する。
既存テストは `jyocd` がないためフィルタが実行されず、後方互換性が保たれる。

- [ ] **Step 1: JRAフィルタのテストを書く**

`tests/test_training_pipeline.py` に `TestJRAFilterTraining` クラスを追加。
`_train_submodel` をモックして、フィルタ後のデータにNARエントリが含まれないことを検証する。

```python
class TestJRAFilterTraining:
    """学習パイプラインのJRAフィルタ テスト"""

    @patch("pipelines.training_pipeline.mlflow")
    def test_nar_entries_filtered_before_surface_split(
        self, mock_mlflow: MagicMock
    ) -> None:
        """NARエントリ (jyocd >= 30) が surface分割前に除外される"""
        feat_df = _make_feature_df(8000, 800)
        # jyocd 列を追加: デフォルトはJRA (05)
        feat_df["jyocd"] = "05"
        # dirt エントリの後半をNARに変更
        dirt_mask = feat_df["surface"] == "dirt"
        dirt_indices = feat_df[dirt_mask].index
        nar_count = len(dirt_indices) // 2
        feat_df.loc[dirt_indices[:nar_count], "jyocd"] = "35"

        mock_store = _make_mock_store()
        mock_sub = SubmodelSet(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=MagicMock(),
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=MagicMock(),
            wide=MagicMock(),
            confidence=MagicMock(),
        )

        with patch.object(FeatureEngine, "build_all", return_value=feat_df):
            with patch.object(
                SubModelManager,
                "add_distance_band_features",
                side_effect=lambda df: df.copy(),
            ):
                with patch.object(
                    TrainingPipelineV5,
                    "_train_submodel",
                    return_value=mock_sub,
                ) as mock_train:
                    with patch(
                        "pipelines.training_pipeline.TrainingPipelineV5._save_models_local"
                    ):
                        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                        pipeline.store = mock_store
                        pipeline.db = None
                        pipeline.feature_engine = FeatureEngine()
                        pipeline.submodel_mgr = SubModelManager()

                        pipeline.run("2020-01-01", "2023-12-31")

        # _train_submodel に渡された DataFrame にNARエントリがないことを検証
        for call_args in mock_train.call_args_list:
            args, kwargs = call_args
            df = args[0]  # 最初の位置引数 = feat_df (surfaceで分割済み)
            if "jyocd" in df.columns:
                jyocd_int = pd.to_numeric(df["jyocd"], errors="coerce")
                nar_found = (jyocd_int >= 30).sum()
                assert nar_found == 0, (
                    f"NAR entries should be filtered, found {nar_found} with jyocd >= 30"
                )

    @patch("pipelines.training_pipeline.mlflow")
    def test_no_jyocd_column_skips_filter(
        self, mock_mlflow: MagicMock
    ) -> None:
        """jyocd列がない場合はフィルタを実行しない (後方互換)"""
        feat_df = _make_feature_df(8000, 800)
        # jyocd 列なし → フィルタが実行されない

        mock_store = _make_mock_store()
        mock_sub = SubmodelSet(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=MagicMock(),
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=MagicMock(),
            wide=MagicMock(),
            confidence=MagicMock(),
        )

        with patch.object(FeatureEngine, "build_all", return_value=feat_df):
            with patch.object(
                SubModelManager,
                "add_distance_band_features",
                side_effect=lambda df: df.copy(),
            ):
                with patch.object(
                    TrainingPipelineV5,
                    "_train_submodel",
                    return_value=mock_sub,
                ) as mock_train:
                    with patch(
                        "pipelines.training_pipeline.TrainingPipelineV5._save_models_local"
                    ):
                        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                        pipeline.store = mock_store
                        pipeline.db = None
                        pipeline.feature_engine = FeatureEngine()
                        pipeline.submodel_mgr = SubModelManager()

                        pipeline.run("2020-01-01", "2023-12-31")

        # 全データがそのまま渡される (8エントリ/レース相当)
        assert mock_train.call_count >= 1, "Should train at least 1 submodel"
```

- [ ] **Step 2: テストを実行して失敗を確認**

```bash
python -m pytest tests/test_training_pipeline.py::TestJRAFilterTraining -v
```

Expected: `test_nar_entries_filtered_before_surface_split` FAILS (フィルタ未実装のためNARエントリが残る)
Expected: `test_no_jyocd_column_skips_filter` PASSES (フィルタなしでも動作)

- [ ] **Step 3: Training Pipeline に JRAフィルタを実装**

`src/pipelines/training_pipeline.py` の `run()` メソッド内、
`add_distance_band_features()` の直後 (line 150) に挿入:

```python
        feat_df = self.submodel_mgr.add_distance_band_features(feat_df)

        # JRAフィルタ: NARレース (jyocd 30以上) を除外
        if "jyocd" in feat_df.columns:
            jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
            before = len(feat_df)
            feat_df = feat_df[jyocd_int.between(1, 10)]
            after = len(feat_df)
            if after < before:
                logger.info(
                    "JRA filter: %d -> %d entries (removed %d NAR)",
                    before, after, before - after,
                )
```

**位置:** line 150 (`add_distance_band_features`) と line 152 (`# 2b. ワイドオッズ`) の間。

- [ ] **Step 4: テストを実行して通過を確認**

```bash
python -m pytest tests/test_training_pipeline.py::TestJRAFilterTraining -v
```

Expected: 2 tests PASS

- [ ] **Step 5: 既存テストの回帰確認**

```bash
python -m pytest tests/test_training_pipeline.py -v
```

Expected: 全テスト PASS (既存テストは `jyocd` 列なしでフィルタ非実行)

- [ ] **Step 6: コミット**

```bash
git add tests/test_training_pipeline.py src/pipelines/training_pipeline.py
git commit -m "feat: add JRA-only filter to training pipeline (NAR exclusion)"
```

---

### Task 2: Backtest Engine — JRAフィルタのテストと実装 (TDD)

**Files:**
- Modify: `tests/test_backtest_engine.py` (テストクラス追加)
- Modify: `src/backtest/engine.py:193` (フィルタ追加)

**方針:** `TestBetHistoryEnrichment` と同じモックパターンを使用するが、
`feat_df` の `jyocd` を NAR 値 (35) に設定し、フィルタで除外されることを検証。

- [ ] **Step 1: JRAフィルタのテストを書く**

`tests/test_backtest_engine.py` に `TestJRAFilterBacktest` クラスを追加。
2つのテスト:
1. `test_nar_race_excluded`: NARレースは0ベットになる
2. `test_jra_race_included`: JRAレースは通常通りベットされる

```python
class TestJRAFilterBacktest:
    """バックテストエンジン JRAフィルタのテスト"""

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_nar_race_excluded_from_backtest(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """NARレース (jyocd >= 30) はバックテストから除外される"""
        # --- load mocks ---
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [2.4]}
        )

        # --- feat_df with NAR jyocd ---
        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["dirt"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [2.4],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "jyocd": ["35"],  # NAR — フィルタで除外されるべき
                "racenum": [1],
                "grade_code": ["E"],
                "hondai": ["地方レース"],
                "bamei": ["テスト馬"],
                "kisyuryakusyo": ["テスト騎手"],
                "track_condition_code": [1],
                "p_place_pred": [0.65],
                "e_return_place_pred": [1.80],
            }
        )

        # --- FeatureEngine mock ---
        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        # --- SubModelManager mock ---
        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        # --- pre-computation mocks ---
        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # --- run engine ---
        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        # --- NARレースはフィルタで除外されるためベット0件 ---
        assert result.total_bets == 0, (
            "NAR race (jyocd=35) should be excluded from backtest"
        )

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_jra_race_included_in_backtest(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """JRAレース (jyocd 01-10) は通常通りバックテスト対象"""
        # --- load mocks (same as NAR test) ---
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [2.4]}
        )

        # --- feat_df with JRA jyocd ---
        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [2.4],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "jyocd": ["05"],  # JRA — フィルタを通過する
                "racenum": [11],
                "grade_code": ["E"],
                "hondai": ["JRAレース"],
                "bamei": ["テスト馬"],
                "kisyuryakusyo": ["テスト騎手"],
                "track_condition_code": [1],
                "p_place_pred": [0.65],
                "e_return_place_pred": [1.80],
            }
        )

        # --- FeatureEngine mock ---
        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        # --- SubModelManager mock ---
        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        # --- pre-computation mocks ---
        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # --- run engine ---
        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        # --- JRAレースは通常通りベットされる ---
        assert result.total_bets >= 1, (
            "JRA race (jyocd=05) should be included in backtest"
        )
```

- [ ] **Step 2: テストを実行して失敗を確認**

```bash
python -m pytest tests/test_backtest_engine.py::TestJRAFilterBacktest -v
```

Expected: `test_nar_race_excluded_from_backtest` FAILS (フィルタ未実装のためNARレースがベットされる)
Expected: `test_jra_race_included_in_backtest` PASSES (JRAレースは常にベットされる)

- [ ] **Step 3: Backtest Engine に JRAフィルタを実装**

`src/backtest/engine.py` の `run()` メソッド内、
`add_distance_band_features()` の直後 (line 193) に挿入:

```python
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # JRAフィルタ: NARレース (jyocd 30以上) を除外
        if "jyocd" in feat_df.columns:
            jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
            feat_df = feat_df[jyocd_int.between(1, 10)]
```

**位置:** line 193 (`add_distance_band_features`) と line 195 (`# 3. 特徴量の一括事前計算`) の間。

- [ ] **Step 4: テストを実行して通過を確認**

```bash
python -m pytest tests/test_backtest_engine.py::TestJRAFilterBacktest -v
```

Expected: 2 tests PASS

- [ ] **Step 5: 既存テストの回帰確認**

```bash
python -m pytest tests/test_backtest_engine.py -v
```

Expected: 全テスト PASS

- [ ] **Step 6: コミット**

```bash
git add tests/test_backtest_engine.py src/backtest/engine.py
git commit -m "feat: add JRA-only filter to backtest engine (NAR exclusion)"
```

---

### Task 3: 統合テスト — 全テストスイートの実行

**Files:**
- なし (確認のみ)

- [ ] **Step 1: 全テストを実行して回帰がないことを確認**

```bash
python -m pytest tests/ -v
```

Expected: 全テスト PASS。既存テストは `jyocd` 列なしで動作するため影響なし。

- [ ] **Step 2: リント確認**

```bash
ruff check src/pipelines/training_pipeline.py src/backtest/engine.py
```

Expected: No errors

- [ ] **Step 3: 最終確認 — 最終コミット不要 (Task 1/2 でコミット済み)**

---

## Rollback

JRAフィルタの追加のみなので、各コミットを `git revert` でロールバック可能:

```bash
git log --oneline -3  # JRAフィルタの2コミットを確認
git revert <commit-hash>  # 各コミットを個別にリバート
```

## Verification (バックテスト実行後)

本番バックテストで以下を確認:
1. `market_dirt.lgb`: 1木 → 100+木 (正常な学習)
2. `place_ret_dirt.lgb`: 1木 → 100+木
3. `ev_corrector_p_dirt.lgb`: 1木 → 10+木
4. バックテストROIが改善していること
