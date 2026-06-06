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
        live_track_conditions: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """当日の出走表を取得し、履歴特徴量を生成。

        Args:
            target_date: 予測対象日。
            everydb2: EveryDB2 クエリインターフェース。
            live_track_conditions: JRAから取得したライブトラック条件。

        Returns:
            レーススケジュール (race_id, venue, race_num, post_time, surface, distance)。
            事前計算済み特徴量を predictions/YYYYMMDD_pre.parquet に保存。
        """
        from features.feature_builder import FeatureBuilder
        from features.feature_manifest import FeatureState

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

        # FeatureBuilder: 13エンリッチメントモジュールを一括実行 (Phase 52)
        # PT は推論パス: build_for_inference() を使用し、FeatureState は
        # SubmodelSet から取得。7ギャップ (Sire/PaceAptitude/Course/DamPedigree/
        # Record/Mining/Interaction) が FeatureBuilder で解消される。
        builder = FeatureBuilder(store=self.store)

        # 全 surface の submodel から FeatureState を構築してビルド
        # surface_key ("turf"/"dirt") に基づいて race_df をフィルタリングし、
        # 重複行の生成を防止する (WR-02)
        feat_dfs: list[pd.DataFrame] = []
        for _surface_key, submodel in self.models.submodels.items():
            try:
                feature_state = FeatureState.from_submodel_set(submodel, version="1.0")
            except ValueError:
                # track_stats が未設定の場合はスキップ (学習済みモデルのみ使用)
                logger.warning("Skipping surface %s: track_stats not available", _surface_key)
                continue

            # surface_key に該当するレースのみをフィルタリング
            if "surface" in race_df.columns:
                surf_race_df = race_df[race_df["surface"] == _surface_key].copy()
                surf_race_ids = set(surf_race_df["race_id"].unique())
                surf_entry_df = entry_df[entry_df["race_id"].isin(surf_race_ids)].copy()
                surf_odds_df = odds_df[odds_df["race_id"].isin(surf_race_ids)].copy()
                if odds_ts_df is not None and not odds_ts_df.empty:
                    surf_odds_ts_df = odds_ts_df[odds_ts_df["race_id"].isin(surf_race_ids)].copy()
                else:
                    surf_odds_ts_df = None
            else:
                # surface 列がない場合は全レースを使用 (フォールバック)
                surf_race_df = race_df
                surf_entry_df = entry_df
                surf_odds_df = odds_df
                surf_odds_ts_df = odds_ts_df

            if surf_race_df.empty:
                logger.info("No %s races for %s, skipping", _surface_key, target_date)
                continue

            result = builder.build_for_inference(
                surf_race_df,
                surf_entry_df,
                surf_odds_df,
                feature_state=feature_state,
                odds_ts_df=surf_odds_ts_df,
                live_track_conditions=live_track_conditions,
            )
            feat_dfs.append(result.frame)

        if feat_dfs:
            feat_df = pd.concat(feat_dfs, ignore_index=True)
        else:
            # フォールバック: FeatureState が利用できない場合は build_for_training を使用
            logger.warning("No FeatureState available, falling back to build_for_training")
            result = builder.build_for_training(
                race_df,
                entry_df,
                odds_df,
                odds_ts_df=odds_ts_df,
            )
            feat_df = result.frame

        race_ids = feat_df["race_id"].unique()

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
        if len(race_id) < 8:
            logger.warning("Invalid race_id format (too short): %s", race_id)
            return []
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
