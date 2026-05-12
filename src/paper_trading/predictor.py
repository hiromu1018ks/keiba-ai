"""Paper Trading 日次予測ロジック"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from db.odds_extractor import extract_pre_post_odds
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races

if TYPE_CHECKING:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from db.parquet_store import ParquetStore
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


class PaperPredictor:
    """Paper Trading の予測コア。

    setup() で事前特徴量を生成し、predict_race() で当日データをマージして推論。
    """

    def __init__(
        self,
        store: ParquetStore,
        race_predictor: RacePredictor,
        models: TrainedModelsV5,
        output_dir: Path = Path("data/paper_trading"),
    ) -> None:
        self.store = store
        self.race_predictor = race_predictor
        self.models = models
        self.output_dir = output_dir

    def setup(
        self,
        target_date: date,
        everydb2: EveryDB2Queries,
    ) -> list[dict[str, Any]]:
        """当日の出走表を取得し、履歴特徴量を生成。

        Returns:
            レーススケジュール (race_id, venue, race_num, post_time, surface, distance)。
            事前計算済み特徴量を predictions/YYYYMMDD_pre.parquet に保存。
        """
        from features.feature_engine import FeatureEngine
        from features.horse_history_features import HorseHistoryFeatures
        from features.jockey_context_features import JockeyContextFeatures
        from features.jockey_trainer_combo import JockeyTrainerComboFeatures
        from features.trainer_context_features import TrainerContextFeatures
        from models.submodel_manager import SubModelManager

        # 1. スケジュール取得
        schedule = everydb2.get_race_schedule(target_date)
        if not schedule:
            logger.info("No races on %s", target_date)
            return []

        # 2. Parquet から特徴量を生成
        ymd = target_date.strftime("%Y%m%d")

        race_df = load_races(self.store, ymd, ymd)
        entry_df = load_entries(self.store, ymd, ymd)
        odds_df = load_odds_snapshots(self.store, ymd, ymd)

        if race_df.empty:
            logger.warning("No race data in Parquet for %s", target_date)
            return []

        # 発走前オッズを優先使用 (フォールバック: 確定オッズ)
        odds_ts_df = load_odds_time_series_range(self.store, ymd, ymd)
        if not odds_ts_df.empty and "hassotime" in race_df.columns:
            pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
            if not pre_post_odds.empty:
                logger.info("Using pre-race odds for %s (%d entries)", ymd, len(pre_post_odds))
                odds_df = pre_post_odds
            else:
                logger.info("Pre-race odds empty for %s, falling back to confirmed odds", ymd)
        else:
            logger.info("No time-series odds for %s, using confirmed odds", ymd)

        feat_engine = FeatureEngine()
        submodel_mgr = SubModelManager()
        feat_df = feat_engine.build_all(
            race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=self.store
        )
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # 3. 事前特徴量の計算
        race_ids = feat_df["race_id"].unique()
        hist_all = HorseHistoryFeatures(store=self.store)
        hist_df = hist_all.compute(race_df, entry_df, race_ids)

        jockey_ctx = JockeyContextFeatures(self.store)
        jockey_df = jockey_ctx.compute(entry_df)

        trainer_ctx = TrainerContextFeatures(self.store)
        trainer_df = trainer_ctx.compute(entry_df)

        jt_combo = JockeyTrainerComboFeatures(self.store)
        jt_combo_df = jt_combo.compute(entry_df)

        # マージして保存
        for col_df in [hist_df, jockey_df, trainer_df, jt_combo_df]:
            if not col_df.empty:
                common_cols = [c for c in col_df.columns if c in ["race_id", "umaban"]]
                merge_cols = [
                    c for c in col_df.columns if c not in feat_df.columns or c in common_cols
                ]
                feat_df = feat_df.merge(col_df[merge_cols], on=["race_id", "umaban"], how="left")

        # 事前計算済み特徴量を保存
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pre_path = pred_dir / f"{ymd}_pre.parquet"
        feat_df.to_parquet(pre_path, index=False)
        logger.info("Pre-computed features saved: %s (%d races)", pre_path, len(race_ids))

        return cast(list[dict[str, Any]], schedule)

    def predict_race(
        self,
        race_id: str,
        pre_computed_features: pd.DataFrame,
        horse_weights: pd.DataFrame,
        odds: pd.DataFrame,
        bankroll: float,
    ) -> list[dict[str, Any]]:
        """1レース分の予測。馬体重+オッズ特徴量をマージして推論。

        Returns:
            bet_history と同じスキーマの dict リスト。
        """
        race_df = pre_computed_features[pre_computed_features["race_id"] == race_id].copy()

        if race_df.empty:
            logger.warning("No pre-computed features for %s", race_id)
            return []

        # 馬体重マージ
        if horse_weights is not None and not horse_weights.empty:
            weight_map = dict(zip(horse_weights["umaban"], horse_weights["weight"]))
            race_df["bataijyu"] = race_df["umaban"].map(weight_map)
            if "weight_absolute" in race_df.columns:
                race_df["weight_absolute"] = race_df["umaban"].map(weight_map)

        # オッズマージ
        if odds is not None and not odds.empty:
            odds_map = dict(zip(odds["umaban"], odds["fukuoddslow"]))
            race_df["fukuoddslow"] = race_df["umaban"].map(odds_map)
            tan_map = dict(zip(odds["umaban"], odds["tanodds"]))
            race_df["win_odds"] = race_df["umaban"].map(tan_map)

        # 推論
        result_df = self.race_predictor.predict(race_df)
        if result_df.empty:
            return []

        # Quality screening
        if not self.race_predictor.should_bet(result_df):
            logger.info("Race %s skipped by quality screener", race_id)
            return []

        # ベット選定
        bets = self.race_predictor.select_bets(result_df, bankroll)

        # dict リストに変換 (bet_history スキーマ)
        bet_dicts: list[dict[str, Any]] = []
        surface = result_df["surface"].iloc[0]
        race_date = pd.Timestamp(f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}")
        for bet in bets:
            bet_dicts.append(
                {
                    "race_id": race_id,
                    "bet_type": bet.bet_type.value,
                    "umaban": bet.umaban,
                    "stake": bet.stake,
                    "odds": bet.odds,
                    "result": 0.0,  # 未確定
                    "surface": surface,
                    "kyori": int(result_df["kyori"].iloc[0]) if "kyori" in result_df.columns else 0,
                    "ev": float(bet.ev_lower_corrected),
                    "edge": float(bet.edge),  # Value Betting edge
                    "popularity": 0,
                    "bankroll_after": bankroll - bet.stake,
                    "race_date": race_date,
                    "horse_name": "",
                    "is_paper": True,
                }
            )

        return bet_dicts
