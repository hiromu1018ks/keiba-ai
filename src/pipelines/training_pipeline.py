"""学習パイプライン v5.4 (§11)

Phase C の全モデルを正しい順序で学習し、TrainedModelsV5 に格納。
MLflow に実験を記録。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import mlflow
import numpy as np
import pandas as pd

from db.parquet_store import ParquetStore
from db.readers import (
    load_entries,
    load_odds_snapshots,
    load_odds_time_series_range,
    load_races,
    load_wide_odds,
)
from domain.models import SubmodelSet, TrainedModelsV5
from utils.timing import TimingContext

if TYPE_CHECKING:
    from db.connection import DatabaseConnection

from features.feature_engine import FeatureEngine
from models.ev_correction_model import EVCorrectionModel
from models.market_model import MarketModel
from models.race_quality_screener import RaceQualityScreener
from models.regime_detector import RegimeDetector
from models.robust_confidence_estimator import RobustConfidenceEstimator
from models.stage1_ability_model import AbilityModel
from models.submodel_manager import SubModelManager
from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
from models.wide_pair_builder import WideJointPairBuilder
from models.wide_two_stage_model import WideTwoStageModel

logger = logging.getLogger(__name__)


def _get_num_threads(parallel_workers: int = 1) -> int:
    """並列ワーカー数に応じて最適なスレッド数を返す。"""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count // (parallel_workers + 1))


class TrainingPipelineV5:
    """学習パイプライン (§11)

    データロード → 特徴量生成 → モデル学習 → MLflow記録
    """

    def __init__(
        self,
        store: ParquetStore | None = None,
        db: DatabaseConnection | None = None,
        settings_path: str | None = None,
    ) -> None:
        self.store = store or ParquetStore()
        self.db = db  # kept for etl_to_parquet if needed, can be None
        self.feature_engine = FeatureEngine()
        self.submodel_mgr = SubModelManager()

    @staticmethod
    def _to_yyyymmdd(date_str: str) -> str:
        """YYYY-MM-DD → YYYYMMDD"""
        return date_str.replace("-", "")

    def run(
        self, train_start: str, train_end: str, *, use_ensemble: bool = False
    ) -> TrainedModelsV5:
        """全モデルを学習し TrainedModelsV5 を返す

        Args:
            train_start: 学習開始日 (YYYY-MM-DD)
            train_end: 学習終了日 (YYYY-MM-DD)
            use_ensemble: アンサンブル (B1) を有効化

        Returns:
            学習済みモデルのコンテナ
        """
        self.use_ensemble = use_ensemble
        start = self._to_yyyymmdd(train_start)
        end = self._to_yyyymmdd(train_end)

        # 1. データロード
        logger.info(f"Loading data: {train_start} ~ {train_end}")
        race_df = load_races(self.store, start, end)
        entry_df = load_entries(self.store, start, end)
        # datakubun違い ('0'確定 vs 'B'速報) で同一 (race_id, umaban) が
        # 複数行存在する場合があるため、確定データ('0') を優先して重複除去
        if "datakubun" in entry_df.columns:
            entry_df = entry_df.sort_values("datakubun", na_position="last")
        entry_df = entry_df.drop_duplicates(subset=["race_id", "umaban"], keep="first")
        odds_df = load_odds_snapshots(self.store, start, end)  # フォールバック用

        # NEW: _train_submodel 内で HorseHistoryFeatures が使用するため保存
        self._race_df = race_df
        self._entry_df = entry_df
        # オッズ時系列データ — Stage2 (WinTwoStageModel) のオッズ動的特徴量に必須
        odds_ts_df = load_odds_time_series_range(self.store, start, end)

        # 発走5分前オッズの抽出 (本番と同じ時点のデータを使用)
        # 年ごとに処理してメモリ使用量を抑制
        if not odds_ts_df.empty and "hassotime" in race_df.columns:
            from db.odds_extractor import extract_pre_post_odds

            start_year = int(start[:4])
            end_year = int(end[:4])
            pre_post_frames: list[pd.DataFrame] = []
            for year in range(start_year, end_year + 1):
                year_ts = odds_ts_df[odds_ts_df["year"] == year]
                if year_ts.empty:
                    continue
                pp = extract_pre_post_odds(
                    year_ts, race_df, minutes_before=5
                )
                if not pp.empty:
                    pre_post_frames.append(pp)
            if pre_post_frames:
                odds_df = pd.concat(pre_post_frames, ignore_index=True)
                logger.info(
                    "Using pre-post odds (5min before): %d rows",
                    len(odds_df),
                )
            else:
                logger.warning(
                    "extract_pre_post_odds empty, using snapshots"
                )
        else:
            logger.warning(
                "No time-series data or hassotime, using snapshots"
            )

        # 2. 特徴量生成
        logger.info("Building features")
        feat_df = self.feature_engine.build_all(
            race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=self.store
        )
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

        # 2b. ワイドオッズを pivot して特徴量に merge
        wide_odds_df = load_wide_odds(self.store, start, end)
        if wide_odds_df is not None and not wide_odds_df.empty:
            wide_pivot = wide_odds_df.pivot_table(index="race_id", columns="kumi", values="oddslow")
            wide_pivot.columns = [
                f"wide_odds_{kumi.replace('-', '_')}" for kumi in wide_pivot.columns
            ]
            wide_pivot = wide_pivot.reset_index()
            feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")

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
            sub = self._train_submodel(
                subset_df, num_threads=_get_num_threads(1), use_ensemble=self.use_ensemble
            )
            models[surface] = sub
            logger.info(f"Trained {surface} submodel")
        elif len(surfaces_to_train) >= 2:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        self._train_submodel,
                        subset_df,
                        num_threads=_get_num_threads(2),
                        use_ensemble=self.use_ensemble,
                    ): surface
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

        # 4. feat_df の object 数値列を float64 に統一
        for col in feat_df.columns:
            if feat_df[col].dtype == object:
                try:
                    feat_df[col] = feat_df[col].astype(float)
                except (ValueError, TypeError):
                    pass

        # 4b. Market Model の出力を feat_df にも反映
        # (サブモデル学習で market model が作成した予測誤差を feat_df にマージ)
        for surface, sub in models.items():
            mask = feat_df["surface"] == surface
            if mask.any():
                result_df = sub.market.predict_and_calc_error(feat_df.loc[mask].copy())
                # 新規列のみ feat_df に反映 (列数不一致を避ける)
                new_cols = [c for c in result_df.columns if c not in feat_df.columns]
                for c in new_cols:
                    feat_df[c] = np.nan
                feat_df.loc[mask, new_cols] = result_df[new_cols].astype(float).values

        # 5. レースレベル特徴量を構築
        required_cols = ["signed_log_error_win", "abs_log_error_win"]
        missing = [c for c in required_cols if c not in feat_df.columns]
        if missing:
            logger.warning("Market model columns missing: %s — filling with 0.0", missing)
            for c in missing:
                feat_df[c] = 0.0
        with TimingContext("race_level_features"):
            race_feat_df = self._build_race_level_features(feat_df)

        # 5. RaceQualityScreener
        with TimingContext("quality_screener"):
            quality_screen = RaceQualityScreener()
            quality_screen.train(race_feat_df, num_threads=_get_num_threads(1))
            quality_screen.calibrate_threshold(race_feat_df, target_investment_rate=0.40)
        logger.info("Trained RaceQualityScreener")

        # 6. RegimeDetector
        with TimingContext("regime_detector"):
            regime_stats_df = self._build_regime_stats(race_feat_df, feat_df)
            regime_det = RegimeDetector()
            regime_det.train(regime_stats_df, num_threads=_get_num_threads(1))
        logger.info("Trained RegimeDetector")

        # 7. MLflow 記録
        self._log_to_mlflow(models, quality_screen, regime_det, train_start, train_end)

        return TrainedModelsV5(
            submodels=models,
            quality_screener=quality_screen,
            regime_detector=regime_det,
            train_period=(train_start, train_end),
        )

    def _train_submodel(
        self, df: pd.DataFrame, *, num_threads: int = 0, use_ensemble: bool = False
    ) -> SubmodelSet:
        """単一 surface のサブモデル群を学習"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        surface = df["surface"].iloc[0] if "surface" in df.columns else "unknown"

        # NEW: 馬過去成績特徴量
        from features.horse_history_features import HorseHistoryFeatures

        with TimingContext(f"{surface}/horse_history"):
            hist = HorseHistoryFeatures(store=self.store, n_past=5)
            hist_df = hist.compute(self._race_df, self._entry_df, df["race_id"].unique())
            df = df.merge(hist_df, on=["race_id", "umaban"], how="left")
        with TimingContext(f"{surface}/add_race_transforms"):
            df = HorseHistoryFeatures.add_race_transforms(df)

        # Group E: 交互作用特徴量 (HorseHistoryFeatures 後に実行 — kyakusitu_cd が必要)
        from features.interaction_features import compute_interaction_features

        with TimingContext(f"{surface}/interaction"):
            df = compute_interaction_features(df)

        # 1. Market Model (正規化差分 log_error のみ出力)
        # object型の数値列 (pd.NA含む) → float64 (2回目のsurface処理でpd.NAが混入するため)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    pass

        with TimingContext(f"{surface}/market_model"):
            # 時間ベース分割の前提: race_date でソート
            df = df.sort_values("race_date").reset_index(drop=True)
            market = MarketModel()
            market.train(df, num_threads=num_threads)
            df = market.predict_and_calc_error(df)

        # nullable int (Int64) → float64 (market model が Int64 を追加するため)
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)

        # 2. Stage1: OOF predictions (リーク防止)
        with TimingContext(f"{surface}/ability_oof"):
            stage1 = AbilityModel()
            df = stage1.train_oof(df, n_folds=3, num_threads=num_threads)
        oof_mask = df["p_ability_win"].notna()
        df_oof = df[oof_mask].copy()

        # NEW: PlaceAbilityModel
        from models.place_ability_model import PlaceAbilityModel

        with TimingContext(f"{surface}/place_ability_train"):
            place_ability = PlaceAbilityModel()
            place_ability.train(df_oof, n_jobs=num_threads)
        with TimingContext(f"{surface}/place_ability_predict"):
            df_oof = place_ability.predict(df_oof)

        # 3. 単勝 2段階モデル
        win_2s = WinTwoStageModel()
        if use_ensemble:
            from models.stacked_ensemble import StackedEnsemble

            with TimingContext(f"{surface}/win_hit_ensemble"):
                features = win_2s._prepare_features(df_oof)
                y = (df_oof["kakuteijyuni"] == 1).astype(int)
                split = int(len(features) * 0.8)
                ensemble = StackedEnsemble(cat_cols=["surface", "distance_bin", "grade_code"])
                ensemble.train(
                    features.iloc[:split], y.iloc[:split],
                    features.iloc[split:], y.iloc[split:],
                    num_threads=num_threads,
                )
                win_2s.hit_model = ensemble
        else:
            with TimingContext(f"{surface}/win_hit"):
                win_2s.train_hit_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/win_return"):
            win_2s.train_return_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/win_predict"):
            df_oof = win_2s.predict_ev(df_oof)

        # Group C/D: 騎手/調教師コンテキスト (Stage2)
        from features.jockey_context_features import JockeyContextFeatures
        from features.trainer_context_features import TrainerContextFeatures

        with TimingContext(f"{surface}/jockey_ctx"):
            jockey_ctx = JockeyContextFeatures(self.store)
            jockey_df = jockey_ctx.compute(df_oof)
            df_oof = pd.merge(df_oof, jockey_df, on=["race_id", "umaban"], how="left")

        with TimingContext(f"{surface}/trainer_ctx"):
            trainer_ctx = TrainerContextFeatures(self.store)
            trainer_df = trainer_ctx.compute(df_oof)
            df_oof = pd.merge(df_oof, trainer_df, on=["race_id", "umaban"], how="left")

        # B4: 騎手-調教師コンビコンテキスト (Stage2)
        from features.jockey_trainer_combo import JockeyTrainerComboFeatures

        with TimingContext(f"{surface}/jt_combo"):
            jt_combo = JockeyTrainerComboFeatures(self.store)
            jt_df = jt_combo.compute(df_oof)
            df_oof = pd.merge(df_oof, jt_df, on=["race_id", "umaban"], how="left")

        # 4. EV補正モデル (P/E分解)
        with TimingContext(f"{surface}/ev_correction"):
            ev_corrector = EVCorrectionModel()
            ev_corrector.train(df_oof, num_threads=num_threads)
            df_oof = ev_corrector.correct_ev(df_oof)

        # 5. 複勝 2段階モデル
        place_2s = PlaceTwoStageModel()
        if use_ensemble:
            from models.stacked_ensemble import StackedEnsemble

            with TimingContext(f"{surface}/place_hit_ensemble"):
                features = place_2s._prepare_features(df_oof)
                y = (df_oof["kakuteijyuni"] <= 3).astype(int)
                split = int(len(features) * 0.8)
                ensemble_place = StackedEnsemble(
                    cat_cols=["surface", "distance_bin", "grade_code"]
                )
                ensemble_place.train(
                    features.iloc[:split], y.iloc[:split],
                    features.iloc[split:], y.iloc[split:],
                    num_threads=num_threads,
                )
                place_2s.hit_model = ensemble_place
        else:
            with TimingContext(f"{surface}/place_hit"):
                place_2s.train_hit_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/place_return"):
            place_2s.train_return_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/place_predict"):
            df_oof = place_2s.predict_ev(df_oof)

        # 6. ワイド 2段階モデル
        with TimingContext(f"{surface}/wide_pair_build"):
            pair_df = WideJointPairBuilder().build(df_oof)
        wide_2s = WideTwoStageModel()
        if len(pair_df) > 0:
            with TimingContext(f"{surface}/wide_hit"):
                wide_2s.train_hit_model(pair_df, num_threads=num_threads)
            with TimingContext(f"{surface}/wide_return"):
                wide_2s.train_return_model(pair_df, num_threads=num_threads)

        # 7. 信頼区間キャリブレーション
        with TimingContext(f"{surface}/confidence"):
            conf = RobustConfidenceEstimator()
            win_calib_df = df_oof.copy()
            win_calib_df["actual_ev_win"] = df_oof["confirmed_odds"] * (
                df_oof["kakuteijyuni"] == 1
            ).astype(int)
            place_calib_df = df_oof.copy()
            place_calib_df["actual_ev_place"] = df_oof["fukuoddslow"] * (
                df_oof["kakuteijyuni"] <= 3
            ).astype(int)
            place_calib_df["ev_place_corrected"] = df_oof["ev_place"]
            conf.calibrate(win_calib_df, place_calib_df)

        return SubmodelSet(
            market=market,
            stage1=stage1,
            place_ability=place_ability,
            win=win_2s,
            ev_corrector=ev_corrector,
            place=place_2s,
            wide=wide_2s,
            confidence=conf,
            use_ensemble=use_ensemble,
        )

    def _build_race_level_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """馬レベル特徴量 → レースレベル特徴量に集約

        RaceQualityScreener.FEATURE_COLS (19列) に対応。
        v5.5 leak-fix: favorite_win_rate を expanding window で計算 (C3)。
        """
        race_feat = (
            feat_df.groupby("race_id")
            .agg(
                surface=("surface", "first"),
                distance_bin=("distance_bin", "first"),
                track_condition_code=("track_condition_code", "first"),
                grade_code=("grade_code", "first"),
                field_size=("field_size", "first"),
                difficulty_score=("difficulty_score", "first"),
                # 市場エラー統計
                market_log_error_mean=("signed_log_error_win", "mean"),
                market_log_error_std=("signed_log_error_win", "std"),
                market_log_error_abs_mean=("abs_log_error_win", "mean"),
                # 分布特徴量
                n_positive_errors=("signed_log_error_win", lambda x: (x > 0).sum()),
                top_k_error_sum=("signed_log_error_win", lambda x: x.nlargest(3).sum()),
                positive_error_ratio=(
                    "signed_log_error_win",
                    lambda x: (x > 0).sum() / max(len(x), 1),
                ),
                # 市場構造
                market_entropy_mean=("market_entropy", "first"),
                overround_mean=("overround", "first"),
            )
            .reset_index()
        )

        # C3 fix: favorite_win_rate を expanding window で計算
        if "race_date" in feat_df.columns:
            date_map = feat_df.groupby("race_id")["race_date"].first()
            race_feat["race_date"] = race_feat["race_id"].map(date_map)
            race_feat = race_feat.sort_values("race_date").reset_index(drop=True)

        if "kakuteijyuni" in feat_df.columns and "popularity_rank" in feat_df.columns:
            fav_df = feat_df[feat_df["popularity_rank"] == 1][["race_id", "kakuteijyuni"]].copy()
            fav_df["fav_won"] = (fav_df["kakuteijyuni"] == 1).astype(float)
            race_feat = race_feat.merge(fav_df[["race_id", "fav_won"]], on="race_id", how="left")
            race_feat["fav_won"] = race_feat["fav_won"].fillna(0.0)
            race_feat["favorite_win_rate"] = (
                race_feat["fav_won"].shift(1).expanding(min_periods=10).mean()
            )
            race_feat["favorite_win_rate"] = race_feat["favorite_win_rate"].fillna(0.3)
            race_feat = race_feat.drop(columns=["fav_won"])
        else:
            race_feat["favorite_win_rate"] = 0.3

        # 結果ベース proxy (初期値) — favorite_win_rate は expanding 済み
        race_feat["hist_hit_rate_topk"] = race_feat["favorite_win_rate"]
        race_feat["hist_roi_topk"] = 1.0
        race_feat["hist_positive_return_ratio"] = 0.3

        # compute_hist_features が必要とする列を追加
        race_feat["distance_band"] = race_feat["distance_bin"]
        race_feat["market_entropy"] = race_feat["market_entropy_mean"]
        race_feat["topk_hit"] = 0
        race_feat["topk_roi"] = 1.0
        race_feat["positive_return"] = 0.0
        race_feat["is_winner"] = 0

        # RaceQualityScreener が必要とする列を補完
        race_feat["market_log_error_max_abs"] = race_feat["market_log_error_abs_mean"] * 2.0
        race_feat["market_log_error_top_q75"] = race_feat["market_log_error_abs_mean"] * 1.5
        race_feat["market_entropy"] = race_feat["market_entropy_mean"]
        race_feat["overround"] = race_feat["overround_mean"]
        race_feat["overround_deviation"] = 0.0
        race_feat["hist_win_rate_same_condition"] = race_feat["favorite_win_rate"]
        race_feat["hist_market_entropy_avg"] = race_feat["market_entropy_mean"]

        # 履歴特徴量 (expanding window — リークフリー)
        if "race_date" in race_feat.columns:
            try:
                from features.info_asymmetry_features import compute_hist_features

                race_feat = compute_hist_features(race_feat)
            except Exception as e:
                logger.debug("hist_features skipped: %s", e)

        return race_feat

    def _build_regime_stats(
        self, race_feat_df: pd.DataFrame, feat_df: pd.DataFrame
    ) -> pd.DataFrame:
        """RegimeDetector 用の rolling 統計を構築

        RegimeDetector.FEATURE_COLS (8列) に対応。
        直近200レースの window 統計。全て発走前情報のみ使用。
        """
        if "race_date" in race_feat_df.columns:
            race_feat_df = race_feat_df.sort_values("race_date").reset_index(drop=True)

        window = 200
        stats = race_feat_df.copy()

        # 基本列マッピング (MarketModel 由来)
        stats["market_error_std"] = stats["market_log_error_std"].fillna(0.2)
        stats["market_error_mean"] = stats["market_log_error_mean"].fillna(0.0)
        stats["field_size_mean"] = stats["field_size"].fillna(14.0).astype(float)

        # overround_rolling: overround の rolling mean
        if "overround_mean" in stats.columns:
            stats["overround_rolling"] = (
                stats["overround_mean"].rolling(window=window, min_periods=50).mean()
            )
        else:
            stats["overround_rolling"] = 0.20

        # entropy_rolling: market_entropy の rolling mean
        if "market_entropy_mean" in stats.columns:
            stats["entropy_rolling"] = (
                stats["market_entropy_mean"].rolling(window=window, min_periods=50).mean()
            )
        else:
            stats["entropy_rolling"] = 2.0

        # favorite_implied_prob_rolling
        if all(c in feat_df.columns for c in ["race_id", "tanodds", "popularity_rank"]):
            fav_df = (
                feat_df[feat_df["popularity_rank"] == 1][["race_id", "tanodds"]].copy()
            )
            fav_df["fav_implied"] = 1.0 / fav_df["tanodds"].replace(0, np.nan)
            race_fav_implied = fav_df.groupby("race_id")["fav_implied"].first()
            stats["fav_implied"] = stats["race_id"].map(race_fav_implied).fillna(0.3)
            stats["favorite_implied_prob_rolling"] = (
                stats["fav_implied"].rolling(window=window, min_periods=50).mean()
            )
            stats = stats.drop(columns=["fav_implied"])
        else:
            stats["favorite_implied_prob_rolling"] = 0.3

        # odds_skewness_rolling
        if all(c in feat_df.columns for c in ["race_id", "tanodds"]):
            race_skew = feat_df.groupby("race_id")["tanodds"].skew()
            stats["odds_skew"] = stats["race_id"].map(race_skew).fillna(0.0)
            stats["odds_skewness_rolling"] = (
                stats["odds_skew"].rolling(window=window, min_periods=50).mean()
            )
            stats = stats.drop(columns=["odds_skew"])
        else:
            stats["odds_skewness_rolling"] = 0.0

        # odds_volatility_mean
        if "odds_volatility" in feat_df.columns:
            race_vol = feat_df.groupby("race_id")["odds_volatility"].mean()
            stats["race_vol"] = stats["race_id"].map(race_vol).fillna(0.1)
            stats["odds_volatility_mean"] = (
                stats["race_vol"].rolling(window=window, min_periods=50).mean()
            )
            stats = stats.drop(columns=["race_vol"])
        else:
            stats["odds_volatility_mean"] = 0.1

        return stats

    def _log_to_mlflow(
        self,
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> None:
        """MLflow に全モデルとメトリクスを記録 (Paper Trading対応)"""
        with mlflow.start_run(run_name=f"v5.5_{train_end}"):
            for surface, sub in models.items():
                # Stage1 (AbilityModel per surface)
                stage1_model = sub.stage1.models.get(surface)
                if stage1_model is not None:
                    mlflow.lightgbm.log_model(stage1_model, name=f"stage1_{surface}")

                # MarketModel
                mlflow.lightgbm.log_model(sub.market.model, name=f"market_{surface}")

                # WinTwoStageModel
                if sub.use_ensemble:
                    # StackedEnsemble — joblib pickle で保存
                    _se_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as _f:
                            _se_tmp = _f.name
                            joblib.dump(sub.win.hit_model, _f.name)
                        mlflow.log_artifact(_se_tmp, f"win_hit_{surface}")
                    finally:
                        if _se_tmp and os.path.exists(_se_tmp):
                            os.unlink(_se_tmp)
                else:
                    mlflow.lightgbm.log_model(sub.win.hit_model, name=f"win_hit_{surface}")
                mlflow.lightgbm.log_model(sub.win.return_model, name=f"win_ret_{surface}")

                # EVCorrectionModel
                mlflow.lightgbm.log_model(
                    sub.ev_corrector.p_correction_model,
                    name=f"ev_corrector_p_{surface}",
                )
                mlflow.lightgbm.log_model(
                    sub.ev_corrector.e_correction_model,
                    name=f"ev_corrector_e_{surface}",
                )

                # PlaceTwoStageModel
                if sub.use_ensemble:
                    _se_tmp2: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as _f2:
                            _se_tmp2 = _f2.name
                            joblib.dump(sub.place.hit_model, _f2.name)
                        mlflow.log_artifact(_se_tmp2, f"place_hit_{surface}")
                    finally:
                        if _se_tmp2 and os.path.exists(_se_tmp2):
                            os.unlink(_se_tmp2)
                else:
                    mlflow.lightgbm.log_model(
                        sub.place.hit_model, name=f"place_hit_{surface}"
                    )
                mlflow.lightgbm.log_model(
                    sub.place.return_model, name=f"place_ret_{surface}"
                )

                # PlaceAbilityModel (sklearn CalibratedClassifierCV → joblib)
                calibrated = sub.place_ability._calibrated or sub.place_ability._model
                if calibrated is not None:
                    _tmp_path: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                            _tmp_path = f.name
                            joblib.dump(calibrated, f.name)
                        mlflow.log_artifact(_tmp_path, f"place_ability_{surface}")
                    finally:
                        if _tmp_path and os.path.exists(_tmp_path):
                            os.unlink(_tmp_path)

                # WideTwoStageModel
                mlflow.lightgbm.log_model(sub.wide.hit_model, name=f"wide_hit_{surface}")
                mlflow.lightgbm.log_model(
                    sub.wide.return_model, name=f"wide_ret_{surface}"
                )

            # RaceQualityScreener
            mlflow.lightgbm.log_model(quality_screen.model, name="race_quality")
            mlflow.log_param("quality_threshold", quality_screen.threshold)

            # RegimeDetector
            mlflow.lightgbm.log_model(regime_det.model, name="regime_detector")

            # RobustConfidenceEstimator キャリブレーション値 (JSON)
            first_sub = next(iter(models.values()))
            conf = first_sub.confidence
            if hasattr(conf, "_calibrated") and conf._calibrated:
                conf_params = {
                    "alpha": conf.alpha,
                    "rolling_window": conf.rolling_window,
                    "win_cp_quantile": conf._win_cp_quantile,
                    "place_cp_quantile": conf._place_cp_quantile,
                    "win_rolling_quantile": conf._win_rolling_quantile,
                    "place_rolling_quantile": conf._place_rolling_quantile,
                }
                mlflow.log_dict(conf_params, "confidence_params.json")

            mlflow.log_param("train_start", train_start)
            mlflow.log_param("train_end", train_end)
            mlflow.log_param("n_surfaces", str(len(models)))
            mlflow.log_param("pipeline_version", "v5.5")

            # ローカルにもモデル保存 (MLflow Model Registry不使用時のフォールバック)
            self._save_models_local(models, quality_screen, regime_det, train_start, train_end)

    @staticmethod
    def _save_models_local(
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> Path:
        """全モデルをローカルディレクトリに保存 (MLflow非依存)"""
        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # 古いモデルファイルを完全に削除 (アンサンブル/非アンサンブル間の不整合を防止)
        for old_file in models_dir.glob("*.lgb"):
            old_file.unlink()
        for old_file in models_dir.glob("*.joblib"):
            old_file.unlink()
        for old_file in models_dir.glob("*.json"):
            old_file.unlink()

        saved: dict[str, object] = {}
        ensemble_keys: set[str] = set()
        for surface, sub in models.items():
            saved[f"stage1_{surface}"] = sub.stage1.models.get(surface)
            saved[f"market_{surface}"] = sub.market.model
            if sub.use_ensemble:
                ensemble_keys.add(f"win_hit_{surface}")
                ensemble_keys.add(f"place_hit_{surface}")
            saved[f"win_hit_{surface}"] = sub.win.hit_model
            saved[f"win_ret_{surface}"] = sub.win.return_model
            saved[f"ev_corrector_p_{surface}"] = sub.ev_corrector.p_correction_model
            saved[f"ev_corrector_e_{surface}"] = sub.ev_corrector.e_correction_model
            saved[f"place_hit_{surface}"] = sub.place.hit_model
            saved[f"place_ret_{surface}"] = sub.place.return_model
            saved[f"wide_hit_{surface}"] = sub.wide.hit_model
            saved[f"wide_ret_{surface}"] = sub.wide.return_model
            # PlaceAbilityModel (sklearn) は joblib で保存
            calibrated = sub.place_ability._calibrated or sub.place_ability._model
            if calibrated is not None:
                import joblib

                joblib.dump(
                    calibrated,
                    models_dir / f"place_ability_{surface}.joblib",
                )

        saved["race_quality"] = quality_screen.model
        saved["regime_detector"] = regime_det.model

        # LightGBM モデルを model.lgb として保存; StackedEnsemble は joblib
        for name, model in saved.items():
            if model is None:
                continue
            if name.startswith("place_ability"):
                continue  # joblib で既に保存済み
            if name in ensemble_keys:
                # StackedEnsemble — joblib pickle で保存
                joblib.dump(model, models_dir / f"{name}.joblib")
            elif hasattr(model, "save_model"):
                model.save_model(str(models_dir / f"{name}.lgb"))

        # RobustConfidenceEstimator パラメータ保存
        for surface, sub in models.items():
            conf = sub.confidence
            if conf._calibrated:
                conf_data = {
                    "alpha": conf.alpha,
                    "rolling_window": conf.rolling_window,
                    "win_cp_quantile": conf._win_cp_quantile,
                    "place_cp_quantile": conf._place_cp_quantile,
                    "win_rolling_quantile": conf._win_rolling_quantile,
                    "place_rolling_quantile": conf._place_rolling_quantile,
                }
                # 各surfaceごとに保存 (最後のsurfaceの値が使われる)
                with open(models_dir / "confidence_params.json", "w", encoding="utf-8") as f:
                    json.dump(conf_data, f, indent=2)

        # メタ情報
        meta = {
            "train_start": train_start,
            "train_end": train_end,
            "surfaces": list(models.keys()),
            "quality_threshold": quality_screen.threshold,
            "saved_at": pd.Timestamp.now().isoformat(),
            "use_ensemble": all(sub.use_ensemble for sub in models.values()),
        }
        with open(models_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Models saved to %s", models_dir)
        return models_dir
