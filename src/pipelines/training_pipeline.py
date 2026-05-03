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
from features.odds_dynamics_features import compute_roi_ema
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
from models.market_model import MarketModel
from models.place_selection_gate import PlaceSelectionGateModel, ensure_place_selection_columns
from models.race_quality_screener import RaceQualityScreener
from models.win_selection_gate import WinSelectionGateModel, ensure_win_selection_columns
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
        model_dir: Path | None = None,
    ) -> None:
        self.store = store or ParquetStore()
        self.db = db  # kept for etl_to_parquet if needed, can be None
        self.feature_engine = FeatureEngine()
        self.submodel_mgr = SubModelManager()
        self.model_dir = model_dir or Path("data/models")

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

        # JRA 学習対象に絞ってからオッズ抽出/特徴量生成へ進む。
        if "jyocd" in race_df.columns:
            jyocd_int = pd.to_numeric(race_df["jyocd"], errors="coerce")
            jra_race_ids = race_df.loc[jyocd_int.between(1, 10), "race_id"].drop_duplicates()
            race_df = race_df[race_df["race_id"].isin(jra_race_ids)].copy()
            entry_df = entry_df[entry_df["race_id"].isin(jra_race_ids)].copy()
            odds_df = odds_df[odds_df["race_id"].isin(jra_race_ids)].copy()

        # NEW: _train_submodel 内で HorseHistoryFeatures が使用するため保存
        self._race_df = race_df
        self._entry_df = entry_df
        # オッズ時系列データ — Stage2 (WinTwoStageModel) のオッズ動的特徴量に必須
        odds_ts_df = load_odds_time_series_range(self.store, start, end)
        if not odds_ts_df.empty:
            odds_ts_df = odds_ts_df[odds_ts_df["race_id"].isin(race_df["race_id"])].copy()

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
                pp = extract_pre_post_odds(year_ts, race_df, minutes_before=5)
                if not pp.empty:
                    pre_post_frames.append(pp)
            if pre_post_frames:
                odds_df = pd.concat(pre_post_frames, ignore_index=True)
                logger.info(
                    "Using pre-post odds (5min before): %d rows",
                    len(odds_df),
                )
            else:
                logger.warning("extract_pre_post_odds empty, using snapshots")
        else:
            logger.warning("No time-series data or hassotime, using snapshots")

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
                    before,
                    after,
                    before - after,
                )

        # 2b. ワイドオッズを pivot して特徴量に merge
        wide_odds_df = load_wide_odds(self.store, start, end)
        if wide_odds_df is not None and not wide_odds_df.empty:
            wide_pivot = wide_odds_df.pivot_table(index="race_id", columns="kumi", values="oddslow")
            # ゼロ埋め解除: kumi "0102" → "1_2" (WideJointPairBuilder の lookup 形式に合わせる)
            new_cols = []
            for kumi in wide_pivot.columns:
                lo = int(str(kumi)[:2])
                hi = int(str(kumi)[2:])
                new_cols.append(f"wide_odds_{lo}_{hi}")
            wide_pivot.columns = new_cols
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
                        import traceback as tb

                        logger.error(f"Failed to train {surface} submodel: {e}\n{tb.format_exc()}")
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

        # Group C: ペース適性特徴量 (HorseHistoryFeatures の直後)
        from features.pace_aptitude_features import PaceAptitudeFeatures

        with TimingContext(f"{surface}/pace_aptitude"):
            pace_feat = PaceAptitudeFeatures(store=self.store)
            pace_df = pace_feat.compute_batch(df)
            # df には既に特徴量列が含まれている可能性があるため削除
            _pace_drop_cols = [
                "pace_aptitude", "front_pace_wr", "closing_pace_wr",
                "pace_corner_stability", "pace_closing_power", "pace_position_consistency",
            ]
            for col in _pace_drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            if not pace_df.empty:
                pace_merge_cols = [c for c in
                    ["kettonum", "race_id", "pace_aptitude", "front_pace_wr", "closing_pace_wr",
                     "pace_corner_stability", "pace_closing_power", "pace_position_consistency"]
                    if c in pace_df.columns
                ]
                df = df.merge(
                    pace_df[pace_merge_cols],
                    on=["kettonum", "race_id"],
                    how="left",
                )
            else:
                # 空の場合は結果列を NaN で追加
                df["pace_aptitude"] = np.nan
                df["front_pace_wr"] = np.nan
                df["closing_pace_wr"] = np.nan
                df["pace_corner_stability"] = np.nan
                df["pace_closing_power"] = np.nan
                df["pace_position_consistency"] = np.nan

        # Group D: コース別適性特徴量 (pace_aptitude の直後)
        from features.course_features import CourseFeatures

        with TimingContext(f"{surface}/course_features"):
            course_feat = CourseFeatures(store=self.store)
            course_df = course_feat.compute_batch(df)
            # df には既に特徴量列が含まれている可能性があるため削除
            for col in ["course_wr", "course_distance_wr"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            if not course_df.empty:
                df = df.merge(
                    course_df[["kettonum", "race_id", "course_wr", "course_distance_wr"]],
                    on=["kettonum", "race_id"],
                    how="left",
                )
            else:
                # 空の場合は結果列を NaN で追加
                df["course_wr"] = np.nan
                df["course_distance_wr"] = np.nan

        # 種牡馬産駒特徴量の追加 (ベクトル化)
        from db.readers import load_horses, load_sire_stats
        from features.sire_features import SireFeatures

        with TimingContext(f"{surface}/sire_features"):
            sire_stats = load_sire_stats(self.store)
            if not sire_stats.empty:
                horses_df = load_horses(self.store)
                sire_feat = SireFeatures(sire_stats)
                # entry_df に sire_id / bms_id 列を追加
                sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
                df["sire_id"] = df["kettonum"].map(sire_map)
                bms_map = horses_df.set_index("kettonum")["ketto3infohansyokunum3"]
                df["bms_id"] = df["kettonum"].map(bms_map)
                # ベクトル化一括計算
                sire_result = sire_feat.compute_batch(df)
                # モデルで使用する5列のみを反映 (sire_place_rate は未使用のため除外)
                _sire_cols_needed = {
                    "sire_wr",
                    "sire_surface_wr",
                    "sire_distance_wr",
                    "sire_prize_avg",
                    "bms_wr",
                }
                for col in _sire_cols_needed:
                    if col in sire_result.columns:
                        df[col] = sire_result[col].values

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
            # OOF予測で学習データのリークを除去
            # 注: _train_submodel は学習データのみを受け取る (呼び出し側で既にフィルタ済み)。
            #      よって df 全体に OOF を適用してよい。test_start_date の分割は不要。
            df = market.predict_oof(df, n_splits=5, num_threads=num_threads)

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

        # FEAT-02: odds_to_ability_ratio -- 市場確率と能力確率の比
        # p_market_win_adj は FeatureEngine.build_all() -> compute_market_bias() で生成済み
        # p_ability_win は直上の AbilityModel.train_oof() で生成済み
        #
        # 【依存順序の意図】この計算は PlaceAbilityModel.train() の前に実行する必要がある。
        # odds_to_ability_ratio は p_ability_win (単勝能力確率) のみに依存し、
        # p_ability_place (複勝能力確率) には依存しない。もし将来 place 確率に依存する
        # ように変更する場合は、PlaceAbilityModel.predict() の後に移動すること。
        if "p_market_win_adj" in df_oof.columns and "p_ability_win" in df_oof.columns:
            p_market = df_oof["p_market_win_adj"].clip(lower=1e-6)
            p_ability = df_oof["p_ability_win"].clip(lower=1e-6)
            df_oof["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

        # ODDS-01: deviation features (after odds_to_ability_ratio computed)
        from features.odds_deviation_features import compute_odds_deviation_features
        df_oof = compute_odds_deviation_features(df_oof)

        # NEW: PlaceAbilityModel
        from models.place_ability_model import PlaceAbilityModel

        with TimingContext(f"{surface}/place_ability_train"):
            place_ability = PlaceAbilityModel()
            place_ability.train(df_oof, n_jobs=num_threads)
        with TimingContext(f"{surface}/place_ability_predict"):
            df_oof = place_ability.predict(df_oof)

        # 3. 単勝 2段階モデル
        win_2s = WinTwoStageModel()
        # PITリーク防止: WinTwoStageModel 学習前にソート
        if "race_date" in df_oof.columns:
            df_oof = df_oof.sort_values("race_date").reset_index(drop=True)
        if use_ensemble:
            from models.stacked_ensemble import StackedEnsemble

            with TimingContext(f"{surface}/win_hit_ensemble"):
                features = win_2s._prepare_features(df_oof)
                y = (df_oof["kakuteijyuni"] == 1).astype(int)
                split = int(len(features) * 0.8)
                ensemble = StackedEnsemble(cat_cols=["surface", "distance_bin", "grade_code"])
                ensemble.train(
                    features.iloc[:split],
                    y.iloc[:split],
                    features.iloc[split:],
                    y.iloc[split:],
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

        # PITリーク防止: merge操作で順序が破壊された可能性があるため再ソート
        if "race_date" in df_oof.columns:
            df_oof = df_oof.sort_values("race_date").reset_index(drop=True)

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
                features = place_2s._prepare_features(df_oof, use_cols=place_2s.HIT_FEATURE_COLS)
                y = (df_oof["kakuteijyuni"] <= 3).astype(int)
                split = int(len(features) * 0.8)
                ensemble_place = StackedEnsemble(cat_cols=["surface", "distance_bin", "grade_code"])
                ensemble_place.train(
                    features.iloc[:split],
                    y.iloc[:split],
                    features.iloc[split:],
                    y.iloc[split:],
                    num_threads=num_threads,
                )
                place_2s.hit_model = ensemble_place
                # バリデーション予測を保存 (Benter combination + isotonic fitting 用)
                ensemble_val_pred = ensemble_place.predict(features.iloc[split:])
                place_2s._val_p_raw = ensemble_val_pred
                place_2s._val_y = y.iloc[split:].values
                place_2s._val_fukuoddslow = df_oof["fukuoddslow"].iloc[split:].values
        else:
            # PITリーク防止: PlaceTwoStageModel 学習前にソート
            if "race_date" in df_oof.columns:
                df_oof = df_oof.sort_values("race_date").reset_index(drop=True)
            with TimingContext(f"{surface}/place_hit"):
                place_2s.train_hit_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/place_return"):
            place_2s.train_return_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/place_predict"):
            df_oof = place_2s.predict_ev(df_oof)

        # 5b. Benter Combination + Isotonic Calibration + Temperature Scaling
        benter_combo = None
        isotonic_cal = None
        temp_scaler = None
        if hasattr(place_2s, "_val_p_raw") and len(place_2s._val_p_raw) >= 500:
            from sklearn.isotonic import IsotonicRegression

            from models.benter_combination import BenterCombination, TemperatureScaling

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

            # v5: Temperature Scaling — Isotonic後の過信/過少評価を補正
            with TimingContext(f"{surface}/temperature"):
                val_p_isotonic = isotonic_cal.transform(val_p_combined)
                temp_scaler = TemperatureScaling.fit(val_p_isotonic, val_y)
                logger.info(
                    "Temperature Scaling: T=%.4f", temp_scaler.temperature
                )

        # 5c. Win Benter Combination (D-11, D-04, D-13)
        win_benter = None
        win_isotonic_cal = None
        win_temp_scaler = None
        if "tanodds" in df_oof.columns and len(df_oof) >= 500:
            from models.benter_combination import BenterCombination
            from models.win_benter_gate import generate_win_oof_predictions

            with TimingContext(f"{surface}/win_oof"):
                oof_p_fund, oof_p_market, oof_y = generate_win_oof_predictions(
                    df_oof,
                    win_model_cls=WinTwoStageModel,
                    ev_corrector=ev_corrector,
                    n_splits=5,
                    num_threads=num_threads,
                )

            if len(oof_p_fund) >= 500:
                from itertools import product as iter_product

                from scipy.optimize import minimize as scipy_minimize

                # Grid search for initial parameters (D-13)
                best_nll = float("inf")
                best_benter = None
                alpha_grid = [0.3, 0.5, 0.7, 1.0]
                beta_grid = [0.3, 0.5, 0.7, 1.0]
                gamma_grid = [-1.0, 0.0, 1.0]

                for a0, b0, g0 in iter_product(alpha_grid, beta_grid, gamma_grid):
                    try:
                        logit_f = BenterCombination._logit(oof_p_fund)
                        logit_m = BenterCombination._logit(oof_p_market)
                        y_arr = oof_y.astype(float)

                        def _nll(params: np.ndarray) -> float:
                            alpha, beta, gamma = params
                            logit_c = alpha * logit_f + beta * logit_m + gamma
                            p_c = 1.0 / (1.0 + np.exp(-logit_c))
                            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
                            return float(
                                -np.sum(
                                    y_arr * np.log(p_c)
                                    + (1 - y_arr) * np.log(1 - p_c)
                                )
                            )

                        res = scipy_minimize(
                            _nll,
                            x0=[a0, b0, g0],
                            method="L-BFGS-B",
                            bounds=[(0.01, 5.0), (0.20, 5.0), (-5.0, 5.0)],
                        )
                        if res.fun < best_nll:
                            best_nll = res.fun
                            best_benter = BenterCombination(
                                alpha=float(res.x[0]),
                                beta=float(res.x[1]),
                                gamma=float(res.x[2]),
                            )
                    except Exception:
                        continue

                if best_benter is not None:
                    win_benter = best_benter
                    logger.info(
                        "Win Benter (grid): alpha=%.3f, beta=%.3f, gamma=%.3f, NLL=%.2f",
                        win_benter.alpha,
                        win_benter.beta,
                        win_benter.gamma,
                        best_nll,
                    )
                else:
                    # Fallback to standard fit
                    with TimingContext(f"{surface}/win_benter"):
                        win_benter = BenterCombination.fit(
                            oof_p_fund, oof_p_market, oof_y
                        )
                    logger.info(
                        "Win Benter (fallback): alpha=%.3f, beta=%.3f, gamma=%.3f",
                        win_benter.alpha,
                        win_benter.beta,
                        win_benter.gamma,
                    )
            else:
                logger.warning(
                    "Win OOF samples < 500 (%d), skipping Win Benter", len(oof_p_fund)
                )
        else:
            logger.info("tanodds not in df_oof or df too small, skipping Win Benter")

        # 5d. Win Calibration Comparison (D-05, D-07, D-08)
        if win_benter is not None and len(oof_p_fund) >= 500:
            from models.win_benter_gate import compare_calibrations, generate_reliability_data

            # Get Benter-combined probabilities for calibration
            oof_p_combined = win_benter.combine(oof_p_fund, oof_p_market)

            with TimingContext(f"{surface}/win_calibration"):
                cal_result = compare_calibrations(oof_p_combined, oof_y, train_ratio=0.8)

            # Log reliability diagram data
            reliability = generate_reliability_data(oof_y, oof_p_combined, n_bins=10)
            logger.info(
                "Win Reliability: bins=%s, positives=%s",
                [f"{v:.3f}" for v in reliability["mean_predicted_value"]],
                [f"{v:.3f}" for v in reliability["fraction_of_positives"]],
            )

            # Select calibrator based on comparison
            winner = cal_result["winner"]
            if winner == "beta":
                win_isotonic_cal = cal_result["beta_calibrator"]
                logger.info("Win calibration: Beta selected (Brier=%.6f)", cal_result["beta_brier"])
            elif winner == "isotonic":
                win_isotonic_cal = cal_result["iso_calibrator"]
                logger.info(
                    "Win calibration: Isotonic selected (Brier=%.6f)", cal_result["iso_brier"]
                )
            else:
                win_isotonic_cal = None
                logger.info("Win calibration: none selected (insufficient data)")

            # Temperature scaling (D-06: optional, apply only if it improves Brier Score)
            if win_isotonic_cal is not None:
                from sklearn.metrics import brier_score_loss

                from models.benter_combination import TemperatureScaling

                # Get calibrated probabilities on full OOF data
                oof_p_calibrated = np.asarray(
                    win_isotonic_cal.transform(oof_p_combined), dtype=float
                )
                brier_before_temp = float(
                    brier_score_loss(oof_y, np.clip(oof_p_calibrated, 1e-10, 1 - 1e-10))
                )

                try:
                    win_temp_scaler = TemperatureScaling.fit(oof_p_calibrated, oof_y)
                    oof_p_temp = win_temp_scaler.transform(oof_p_calibrated)
                    brier_after_temp = float(
                        brier_score_loss(oof_y, np.clip(oof_p_temp, 1e-10, 1 - 1e-10))
                    )

                    # Only keep TempScale if it improves Brier Score (D-06)
                    if brier_after_temp >= brier_before_temp:
                        logger.info(
                            "Win TempScale: no improvement (%.6f -> %.6f), skipping",
                            brier_before_temp,
                            brier_after_temp,
                        )
                        win_temp_scaler = None
                    else:
                        logger.info(
                            "Win TempScale: T=%.4f improved Brier (%.6f -> %.6f)",
                            win_temp_scaler.temperature,
                            brier_before_temp,
                            brier_after_temp,
                        )
                except Exception:
                    logger.warning("Win TempScale failed, skipping")
                    win_temp_scaler = None

        # 5a. Place EV補正 (P/E decomposition)
        with TimingContext(f"{surface}/place_ev_correction"):
            place_ev_corrector = PlaceEVCorrectionModel()
            place_ev_corrector.train(df_oof, num_threads=num_threads)
            df_oof = place_ev_corrector.correct_ev(df_oof)

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
            place_calib_df["ev_place_corrected"] = df_oof["ev_place_corrected"]
            conf.calibrate(win_calib_df, place_calib_df)

        with TimingContext(f"{surface}/place_selection_gate"):
            gate_train_df = df_oof.copy()
            _, gate_place_df = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
            if "EV_lower_place" in gate_place_df.columns:
                gate_train_df["EV_lower_place"] = gate_place_df["EV_lower_place"].values
            gate_train_df = ensure_place_selection_columns(gate_train_df)
            place_selection_gate = PlaceSelectionGateModel()
            place_selection_gate.train(gate_train_df)

        # --- WinSelectionGate training (SELC-01, D-01) ---
        with TimingContext(f"{surface}/win_selection_gate"):
            wsg_train_df = df_oof.copy()
            wsg_win_df, _ = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
            if "EV_lower_win_corrected" in wsg_win_df.columns:
                wsg_train_df["EV_lower_win_corrected"] = wsg_win_df["EV_lower_win_corrected"].values
            wsg_train_df = ensure_win_selection_columns(wsg_train_df)
            win_selection_gate = WinSelectionGateModel()
            win_selection_gate.train(wsg_train_df)

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
            place_selection_gate=place_selection_gate,
            use_ensemble=use_ensemble,
            benter_combo=benter_combo,
            isotonic_calibrator=isotonic_cal,
            temperature_scaler=temp_scaler,
            win_benter=win_benter,
            win_isotonic_calibrator=win_isotonic_cal,
            win_temperature_scaler=win_temp_scaler,
            win_selection_gate=win_selection_gate,
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

        # 結果ベース proxy — 人気馬の実際の結果から expanding window で計算
        race_feat["hist_hit_rate_topk"] = race_feat["favorite_win_rate"]

        # 人気馬 (popularity_rank==1) の実際の成績を race_id 単位で集計
        if (
            "kakuteijyuni" in feat_df.columns
            and "popularity_rank" in feat_df.columns
            and "tanodds" in feat_df.columns
        ):
            fav_df = feat_df[feat_df["popularity_rank"] == 1][
                ["race_id", "kakuteijyuni", "tanodds"]
            ].copy()
            fav_df["fav_won"] = (fav_df["kakuteijyuni"] == 1).astype(float)
            fav_df["fav_placed"] = (fav_df["kakuteijyuni"] <= 3).astype(float)
            fav_df["fav_roi"] = np.where(
                fav_df["fav_won"] == 1,
                fav_df["tanodds"].astype(float) - 1.0,
                -1.0,
            )
            fav_df["fav_positive"] = (fav_df["fav_roi"] > 0).astype(float)
            fav_agg = fav_df[["race_id", "fav_placed", "fav_roi", "fav_positive", "fav_won"]].copy()
            race_feat = race_feat.merge(fav_agg, on="race_id", how="left")
            race_feat["topk_hit"] = race_feat["fav_placed"].fillna(0.0)
            race_feat["topk_roi"] = race_feat["fav_roi"].fillna(-1.0)
            race_feat["positive_return"] = race_feat["fav_positive"].fillna(0.0)
            race_feat["is_winner"] = race_feat["fav_won"].fillna(0.0)
            # expanding window で履歴統計を計算 (リーク防止: shift(1))
            race_feat["hist_roi_topk"] = (
                race_feat["topk_roi"].shift(1).expanding(min_periods=10).mean().fillna(1.0)
            )
            race_feat["hist_positive_return_ratio"] = (
                race_feat["positive_return"].shift(1).expanding(min_periods=10).mean().fillna(0.3)
            )
            race_feat = race_feat.drop(
                columns=["fav_placed", "fav_roi", "fav_positive", "fav_won"],
                errors="ignore",
            )
        else:
            race_feat["hist_roi_topk"] = 1.0
            race_feat["hist_positive_return_ratio"] = 0.3

        # compute_hist_features が必要とする列を追加
        race_feat["distance_band"] = race_feat["distance_bin"]
        race_feat["market_entropy"] = race_feat["market_entropy_mean"]

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

        # v5.6: EMA 平滑化市場指標 (horse-level feat_df から計算し race-level に展開)
        if "tanodds" in feat_df.columns:
            ema_df = compute_roi_ema(
                feat_df[
                    ["race_id", "tanodds", "popularity_rank"]
                    + (["race_date"] if "race_date" in feat_df.columns else [])
                ]
            )
            # race_id 単位で EMA 値を race_feat にマージ
            for ema_col in ["overround_ema", "entropy_ema"]:
                if ema_col in ema_df.columns:
                    ema_map = ema_df.groupby("race_id")[ema_col].first()
                    race_feat[ema_col] = race_feat["race_id"].map(ema_map).fillna(0.0)
        else:
            race_feat["overround_ema"] = 0.0
            race_feat["entropy_ema"] = 0.0

        return race_feat

    def _build_regime_stats(
        self, race_feat_df: pd.DataFrame, feat_df: pd.DataFrame
    ) -> pd.DataFrame:
        """RegimeDetector 用のレース統計を構築

        RegimeDetector.FEATURE_COLS (8列) に対応。
        推論パスと整合させるため rolling を行わず、レース毎の生値を使用。
        全て発走前情報のみ使用。
        """
        if "race_date" in race_feat_df.columns:
            race_feat_df = race_feat_df.sort_values("race_date").reset_index(drop=True)

        stats = race_feat_df.copy()

        # 基本列マッピング (MarketModel 由来) — 元々 rolling 不使用
        stats["market_error_std"] = stats["market_log_error_std"].fillna(0.2)
        stats["market_error_mean"] = stats["market_log_error_mean"].fillna(0.0)
        stats["field_size_mean"] = stats["field_size"].fillna(14.0).astype(float)

        # overround_rolling: overround の生値 (推論パスと整合)
        if "overround_mean" in stats.columns:
            stats["overround_rolling"] = stats["overround_mean"]
        else:
            stats["overround_rolling"] = 0.20

        # entropy_rolling: market_entropy の生値 (推論パスと整合)
        if "market_entropy_mean" in stats.columns:
            stats["entropy_rolling"] = stats["market_entropy_mean"]
        else:
            stats["entropy_rolling"] = 2.0

        # favorite_implied_prob_rolling: 1番人気オッズの逆数 (レース毎生値)
        if all(c in feat_df.columns for c in ["race_id", "tanodds", "popularity_rank"]):
            fav_df = feat_df[feat_df["popularity_rank"] == 1][["race_id", "tanodds"]].copy()
            fav_df["fav_implied"] = 1.0 / fav_df["tanodds"].replace(0, np.nan)
            race_fav_implied = fav_df.groupby("race_id")["fav_implied"].first()
            stats["favorite_implied_prob_rolling"] = (
                stats["race_id"].map(race_fav_implied).fillna(0.3)
            )
        else:
            stats["favorite_implied_prob_rolling"] = 0.3

        # odds_skewness_rolling: レース毎のオッズ歪度 (生値)
        if all(c in feat_df.columns for c in ["race_id", "tanodds"]):
            race_skew = feat_df.groupby("race_id")["tanodds"].skew()
            stats["odds_skewness_rolling"] = stats["race_id"].map(race_skew).fillna(0.0)
        else:
            stats["odds_skewness_rolling"] = 0.0

        # odds_volatility_mean: レース毎の odds_volatility 平均 (生値)
        if "odds_volatility" in feat_df.columns:
            race_vol = feat_df.groupby("race_id")["odds_volatility"].mean()
            stats["odds_volatility_mean"] = stats["race_id"].map(race_vol).fillna(0.1)
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

                # PlaceEVCorrectionModel
                mlflow.lightgbm.log_model(
                    sub.place_ev_corrector.p_correction_model,
                    name=f"place_ev_corrector_p_{surface}",
                )
                mlflow.lightgbm.log_model(
                    sub.place_ev_corrector.e_correction_model,
                    name=f"place_ev_corrector_e_{surface}",
                )
                if (
                    sub.place_selection_gate is not None
                    and sub.place_selection_gate.is_trained
                ):
                    gate_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".joblib",
                            delete=False,
                        ) as gate_file:
                            gate_tmp = gate_file.name
                        sub.place_selection_gate.save(Path(gate_tmp))
                        mlflow.log_artifact(gate_tmp, f"place_selection_gate_{surface}")
                    finally:
                        if gate_tmp and os.path.exists(gate_tmp):
                            os.unlink(gate_tmp)

                # --- WinSelectionGate (MLflow) ---
                if (
                    sub.win_selection_gate is not None
                    and sub.win_selection_gate.is_trained
                ):
                    wsg_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as wsg_file:
                            wsg_tmp = wsg_file.name
                        sub.win_selection_gate.save(Path(wsg_tmp))
                        mlflow.log_artifact(wsg_tmp, f"win_selection_gate_{surface}")
                    finally:
                        if wsg_tmp and os.path.exists(wsg_tmp):
                            os.unlink(wsg_tmp)

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
                    mlflow.lightgbm.log_model(sub.place.hit_model, name=f"place_hit_{surface}")
                mlflow.lightgbm.log_model(sub.place.return_model, name=f"place_ret_{surface}")

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
                mlflow.lightgbm.log_model(sub.wide.return_model, name=f"wide_ret_{surface}")

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
                    "win_cp_quantile_by_condition": conf._win_cp_quantile_by_condition,
                }
                mlflow.log_dict(conf_params, "confidence_params.json")

            mlflow.log_param("train_start", train_start)
            mlflow.log_param("train_end", train_end)
            mlflow.log_param("n_surfaces", str(len(models)))
            mlflow.log_param("pipeline_version", "v5.5")

            # ローカルにもモデル保存 (MLflow Model Registry不使用時のフォールバック)
            self._save_models_local(models, quality_screen, regime_det, train_start, train_end)

    def _save_models_local(
        self,
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> Path:
        """全モデルをローカルディレクトリに保存 (MLflow非依存)"""
        models_dir = self.model_dir
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
            saved[f"place_ev_corrector_p_{surface}"] = sub.place_ev_corrector.p_correction_model
            saved[f"place_ev_corrector_e_{surface}"] = sub.place_ev_corrector.e_correction_model
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
            if sub.place_selection_gate is not None and sub.place_selection_gate.is_trained:
                sub.place_selection_gate.save(
                    models_dir / f"place_selection_gate_{surface}.joblib"
                )

            # --- WinSelectionGate (local) ---
            if sub.win_selection_gate is not None and sub.win_selection_gate.is_trained:
                sub.win_selection_gate.save(
                    models_dir / f"win_selection_gate_{surface}.joblib"
                )

            # Benter Combination (JSON)
            if sub.benter_combo is not None:
                sub.benter_combo.save(models_dir / f"benter_combo_{surface}.json")

            # Isotonic Calibrator (joblib)
            if sub.isotonic_calibrator is not None:
                joblib.dump(
                    sub.isotonic_calibrator,
                    models_dir / f"isotonic_place_{surface}.joblib",
                )

            # v5: Temperature Scaler (JSON)
            if sub.temperature_scaler is not None:
                sub.temperature_scaler.save(models_dir / f"temp_scale_{surface}.json")

            # Win Benter Combination (JSON)
            if sub.win_benter is not None:
                sub.win_benter.save(models_dir / f"benter_combo_win_{surface}.json")

            # Win Isotonic Calibrator (joblib)
            if sub.win_isotonic_calibrator is not None:
                joblib.dump(
                    sub.win_isotonic_calibrator,
                    models_dir / f"isotonic_win_{surface}.joblib",
                )

            # Win Temperature Scaler (JSON)
            if sub.win_temperature_scaler is not None:
                sub.win_temperature_scaler.save(
                    models_dir / f"temp_scale_win_{surface}.json"
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
                    "win_cp_quantile_by_condition": conf._win_cp_quantile_by_condition,
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
