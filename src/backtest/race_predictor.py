"""1レース分の推論パイプライン (BacktestEngine と PaperPredictor の共通コンポーネント)

BacktestEngine.run() のレース別ループ (4a-4g) を抽出。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from domain.models import Bet, BetType

if TYPE_CHECKING:
    from betting.drawdown_controller import DrawdownController
    from betting.stake_calculator import StakeCalculator
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


class RacePredictor:
    """1レース分の特徴量→推論→ベット候補生成を担当する共通コンポーネント"""

    def __init__(
        self,
        models: TrainedModelsV5,
        *,
        stake_calculator: StakeCalculator | None = None,
        dd_controller: DrawdownController | None = None,
        alpha: float = 0.4,
    ) -> None:
        self.models = models
        self.stake_calc = stake_calculator
        self.dd_ctrl = dd_controller
        self._betting_mode = "kelly" if stake_calculator is not None else "flat"
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def predict(
        self,
        race_df: pd.DataFrame,
        hist_features: pd.DataFrame | None = None,
        jockey_features: pd.DataFrame | None = None,
        trainer_features: pd.DataFrame | None = None,
        jt_combo_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """1レースの推論パイプラインを実行。

        Returns:
            推論結果列 (EV, ev_lower_corrected等) を追加した DataFrame。
            サーフェスが不明な場合は空 DataFrame を返す。
        """
        from features.horse_history_features import HorseHistoryFeatures
        from features.interaction_features import compute_interaction_features

        if race_df.empty:
            return race_df

        # 1. サブモデル選択
        surface_key = race_df["surface"].iloc[0]
        if surface_key not in self.models.submodels:
            logger.debug("Unknown surface: %s, skipping", surface_key)
            return pd.DataFrame()
        submodel = self.models.submodels[surface_key]

        df = race_df.copy()

        # 2. HorseHistoryFeatures マージ + race_transforms
        if hist_features is not None:
            df = df.merge(hist_features, on=["race_id", "umaban"], how="left")
        df = HorseHistoryFeatures.add_race_transforms(df)

        # 3. interaction_features (kyakusitu_cd が必要なため HorseHistoryFeatures 後)
        df = compute_interaction_features(df)

        # 4. 推論チェーン
        try:
            df = submodel.market.predict_and_calc_error(df)
        except Exception as e:
            import traceback

            logger.error("Market prediction failed: %s\n%s", e, traceback.format_exc())
            return pd.DataFrame()
        df = submodel.stage1.add_ability_probs(df)
        df = submodel.place_ability.predict(df)
        df = submodel.win.predict_ev(df)

        # 5. 騎手/調教師コンテキスト マージ
        if jockey_features is not None:
            jockey_race = jockey_features[jockey_features["race_id"] == race_df["race_id"].iloc[0]]
            df = df.merge(jockey_race, on=["race_id", "umaban"], how="left")
        if trainer_features is not None:
            trainer_race = trainer_features[
                trainer_features["race_id"] == race_df["race_id"].iloc[0]
            ]
            df = df.merge(trainer_race, on=["race_id", "umaban"], how="left")
        if jt_combo_features is not None:
            jt_race = jt_combo_features[
                jt_combo_features["race_id"] == race_df["race_id"].iloc[0]
            ]
            df = df.merge(jt_race, on=["race_id", "umaban"], how="left")

        # 6. EV補正 + Place推論
        df = submodel.ev_corrector.correct_ev(df)
        df = submodel.place.predict_ev(df)
        df = submodel.place_ev_corrector.correct_ev(df)

        # 7. 信頼区間
        win_df, place_df = submodel.confidence.predict_lower_bound(df, df)
        df = win_df
        if "EV_lower_place" in place_df.columns:
            df["EV_lower_place"] = place_df["EV_lower_place"].values

        # --- Value Betting with Benter combined probability ---
        # logit(p_combined) = alpha * logit(p_model) + (1-alpha) * logit(p_market)
        # edge = p_combined - p_market
        p_market = np.where(
            df["fukuoddslow"] > 0,
            1.0 / df["fukuoddslow"],
            np.nan,
        )

        # Clip to avoid logit(0) or logit(1) = ±inf
        p_model = np.clip(df["p_place_pred"], 1e-6, 1 - 1e-6)
        p_mkt = np.clip(p_market, 1e-6, 1 - 1e-6)

        logit_model = np.log(p_model / (1 - p_model))
        logit_market = np.log(p_mkt / (1 - p_mkt))

        logit_combined = self.alpha * logit_model + (1 - self.alpha) * logit_market
        p_combined = 1.0 / (1.0 + np.exp(-logit_combined))

        df["p_place_combined"] = p_combined
        df["edge_place"] = p_combined - p_mkt

        return df

    def should_bet(self, race_df: pd.DataFrame) -> bool:
        """RaceQualityScreener でベット対象か判定"""
        features = self.build_race_features(race_df)
        return bool(self.models.quality_screener.should_bet(features))

    def select_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
    ) -> list[Bet]:
        """Value Betting: edge = p_combined - p_market >= threshold の馬を選択。

        p_combined は Benter combined probability:
          logit(p_c) = alpha * logit(p_model) + (1-alpha) * logit(p_market)
        賭け金計算は StakeCalculator.calc_stake(edge=...) で直接 edge を渡す。
        """
        regime = self.models.regime_detector.current_regime
        regime_params = self.models.regime_detector.get_strategy_params(regime)

        bets: list[Bet] = []
        edge_threshold = regime_params.get("edge_threshold", 0.03)
        max_bets = regime_params.get("max_bets_per_race", 3)

        # Value Betting criterion: filter by edge (not EV)
        edge_col = "edge_place"
        if edge_col not in race_df.columns or "fukuoddslow" not in race_df.columns:
            return bets

        candidates = race_df[race_df[edge_col].fillna(0) >= edge_threshold].copy()
        candidates = candidates.nlargest(max_bets, edge_col)

        for _, row in candidates.iterrows():
            edge_val = float(row[edge_col])
            odds_val = float(row["fukuoddslow"])

            if self._betting_mode == "kelly" and self.stake_calc is not None:
                stake = self.stake_calc.calc_stake(
                    edge=edge_val,
                    odds=odds_val,
                    bankroll=bankroll,
                    bet_type=BetType.PLACE,
                )
                if self.dd_ctrl is not None:
                    stake = self.dd_ctrl.adjust_stake(stake, bankroll)
                    stake = max(0, math.floor(stake / 100) * 100)
            else:
                stake = 100.0

            if stake < 100:
                continue

            bets.append(
                Bet(
                    race_id=row["race_id"],
                    umaban=int(row["umaban"]),
                    bet_type=BetType.PLACE,
                    odds=odds_val,
                    ev_lower_corrected=float(row.get("ev_place_corrected", 0)),
                    stake=stake,
                    edge=edge_val,
                )
            )

        return bets

    @staticmethod
    def build_race_features(race_df: pd.DataFrame) -> dict[str, Any]:
        """レースレベル特徴量を dict に変換 (QualityScreener 用)。

        BacktestEngine._build_race_features() から移行。
        """
        row = race_df.iloc[0]
        signed_error = (
            race_df["signed_log_error_win"]
            if "signed_log_error_win" in race_df.columns
            else pd.Series([0.0])
        )
        abs_error = (
            race_df["abs_log_error_win"]
            if "abs_log_error_win" in race_df.columns
            else pd.Series([0.0])
        )
        return {
            "surface": row.get("surface", "turf"),
            "distance_bin": row.get("distance_bin", "mile"),
            "track_condition_code": row.get("track_condition_code", 2),
            "grade_code": row.get("grade_code", "C"),
            "field_size": row.get("field_size", 10),
            "difficulty_score": row.get("difficulty_score", 0.5),
            "market_log_error_mean": float(signed_error.mean()),
            "market_log_error_std": float(signed_error.std()) if len(signed_error) > 1 else 0.0,
            "market_log_error_abs_mean": float(abs_error.mean()),
            "market_log_error_max_abs": float(abs_error.max()) if len(abs_error) > 0 else 0.0,
            "market_log_error_top_q75": float(abs_error.quantile(0.75))
            if len(abs_error) > 1
            else 0.0,
            "n_positive_errors": int((signed_error > 0).sum()),
            "top_k_error_sum": float(signed_error.nlargest(3).sum())
            if len(signed_error) >= 3
            else 0.0,
            "positive_error_ratio": float((signed_error > 0).sum()) / max(len(signed_error), 1),
            "market_entropy": row.get("market_entropy", 2.0),
            "overround": row.get("overround", 0.20),
            "overround_deviation": 0.0,
            "hist_hit_rate_topk": row.get("hist_hit_rate_topk", 0.3),
            "hist_roi_topk": row.get("hist_roi_topk", 1.0),
            "hist_positive_return_ratio": row.get("hist_positive_return_ratio", 0.3),
            "hist_win_rate_same_condition": row.get("hist_hit_rate_topk", 0.3),
            "hist_market_entropy_avg": row.get("market_entropy", 2.0),
            # v5.6: EMA平滑化市場指標
            "overround_ema": row.get("overround_ema", 0.20),
            "entropy_ema": row.get("entropy_ema", 2.0),
        }
