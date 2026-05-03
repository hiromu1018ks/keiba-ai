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
from domain.types import RegimeState
from models.place_selection_gate import build_place_selection_ev, ensure_place_selection_columns
from models.win_selection_gate import build_win_selection_ev, ensure_win_selection_columns

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
        self.alpha = alpha  # kept for backwards compatibility / fallback

    @staticmethod
    def _build_place_selection_ev(df: pd.DataFrame) -> pd.Series:
        return build_place_selection_ev(df)

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
        # ODDS-01: deviation features (after AbilityModel, before WinTwoStageModel)
        from features.odds_deviation_features import compute_odds_deviation_features
        df = compute_odds_deviation_features(df)
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
            jt_race = jt_combo_features[jt_combo_features["race_id"] == race_df["race_id"].iloc[0]]
            df = df.merge(jt_race, on=["race_id", "umaban"], how="left")

        # 6. EV補正 + Place推論
        df = submodel.ev_corrector.correct_ev(df)  # Win EV補正は維持

        # --- Win Benter Combination + Race Normalization (D-11) ---
        if getattr(submodel, "win_benter", None) is not None:
            from models.win_benter_gate import WinBenterGate

            win_gate = WinBenterGate(
                benter=submodel.win_benter,
                calibrator=getattr(submodel, "win_isotonic_calibrator", None),
                temp_scaler=getattr(submodel, "win_temperature_scaler", None),
            )
            df = win_gate.apply(df)

        # --- WinSelectionGate (SELC-01, D-14: after Benter, before Place) ---
        df_winsel = ensure_win_selection_columns(df)
        if "win_selection_ev" not in df.columns:
            df = df_winsel
        win_gate_model = getattr(submodel, "win_selection_gate", None)
        win_gate_enabled = bool(
            win_gate_model is not None and getattr(win_gate_model, "is_trained", False) is True
        )
        if win_gate_enabled:
            assert win_gate_model is not None
            df = win_gate_model.score(df)
            win_annotate = getattr(win_gate_model, "annotate_race_context", None)
            if callable(win_annotate):
                df = win_annotate(df)

        df = submodel.place.predict_ev(df)
        # place_ev_corrector: 補正EVと下限EVの両方をベット選択に使う
        df = submodel.place_ev_corrector.correct_ev(df)

        # 7. 信頼区間
        win_df, place_df = submodel.confidence.predict_lower_bound(df, df)
        df = win_df
        if "EV_lower_place" in place_df.columns:
            df["EV_lower_place"] = place_df["EV_lower_place"].values

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
                0.01,
                0.99,
            )
            df["p_place_combined"] = benter.combine(df["p_place_pred"].values, p_market_clipped)

            # Isotonic calibration (optional post-processing)
            # v5.6: Isotonic post-Benter is too aggressive (pushes mean 0.224 vs true ~0.375).
            # Benter combination already calibrates; skip additional isotonic correction.
            # cal = submodel.isotonic_calibrator
            # if cal is not None:
            #     df["p_place_combined"] = cal.transform(df["p_place_combined"])

            # v5: Temperature Scaling (optional post-isotonic)
            temp = submodel.temperature_scaler
            if temp is not None:
                df["p_place_combined"] = temp.transform(df["p_place_combined"])
        else:
            # フォールバック: Benter なし → raw p_place_pred を使用
            df["p_place_combined"] = df["p_place_pred"]

        # Edge = p_combined * odds - 1.0
        p_combined = pd.to_numeric(df["p_place_combined"], errors="coerce")
        df["p_place_combined"] = p_combined
        df["edge_place"] = p_combined * df["fukuoddslow"] - 1.0
        df["ev_place_direct"] = p_combined * df["fukuoddslow"]

        # Corrected edge from PlaceEVCorrectionModel (if available)
        if "ev_place_corrected" in df.columns:
            df["edge_place_corrected"] = df["ev_place_corrected"] - 1.0
        if "EV_lower_place" in df.columns:
            df["edge_place_lower"] = pd.to_numeric(df["EV_lower_place"], errors="coerce") - 1.0

        selection_ev = self._build_place_selection_ev(df)
        df["place_selection_ev"] = selection_ev
        df["place_selection_edge"] = selection_ev - 1.0
        if "p_place_corrected" in df.columns:
            selection_prob = pd.to_numeric(df["p_place_corrected"], errors="coerce")
        elif "p_place_combined" in df.columns:
            selection_prob = pd.to_numeric(df["p_place_combined"], errors="coerce")
        else:
            selection_prob = pd.to_numeric(df.get("p_place_pred"), errors="coerce")
        df["place_selection_prob"] = selection_prob
        gate_model = getattr(submodel, "place_selection_gate", None)
        gate_enabled = bool(
            gate_model is not None and getattr(gate_model, "is_trained", False) is True
        )
        if gate_enabled:
            assert gate_model is not None
            df = gate_model.score(df)
            annotate_race_context = getattr(gate_model, "annotate_race_context", None)
            if callable(annotate_race_context):
                df = annotate_race_context(df)

        return df

    @staticmethod
    def _ensure_place_selection_columns(race_df: pd.DataFrame) -> pd.DataFrame:
        return ensure_place_selection_columns(race_df)

    @staticmethod
    def _gate_rank_metrics(
        race_df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        gate_scores = pd.to_numeric(race_df["place_gate_score"], errors="coerce")
        score_rank = gate_scores.groupby(race_df["race_id"]).rank(
            method="first",
            ascending=False,
        )
        score_gap = gate_scores.groupby(race_df["race_id"]).transform("max") - gate_scores
        return score_rank, score_gap

    @staticmethod
    def _race_first_numeric(race_df: pd.DataFrame, column: str) -> pd.Series:
        if column not in race_df.columns:
            return pd.Series(np.nan, index=race_df.index, dtype=float)
        values = pd.to_numeric(race_df[column], errors="coerce")
        return values.groupby(race_df["race_id"]).transform("first")

    @staticmethod
    def _favorite_implied_prob(race_df: pd.DataFrame) -> pd.Series:
        if "race_id" not in race_df.columns:
            return pd.Series(np.nan, index=race_df.index, dtype=float)
        if "popularity_rank" not in race_df.columns or "odds" not in race_df.columns:
            return pd.Series(np.nan, index=race_df.index, dtype=float)
        popularity_rank = pd.to_numeric(race_df["popularity_rank"], errors="coerce")
        odds = pd.to_numeric(race_df["odds"], errors="coerce")
        favorite_odds = odds.where(popularity_rank.eq(1))
        favorite_odds = favorite_odds.groupby(race_df["race_id"]).transform("min")
        favorite_prob = pd.Series(np.nan, index=race_df.index, dtype=float)
        valid_favorite = favorite_odds.notna() & favorite_odds.gt(0.0)
        favorite_prob.loc[valid_favorite] = 1.0 / favorite_odds.loc[valid_favorite]
        return favorite_prob

    @staticmethod
    def _market_condition_score(race_df: pd.DataFrame) -> pd.Series:
        favorite_prob = RacePredictor._favorite_implied_prob(race_df)
        overround = RacePredictor._race_first_numeric(race_df, "overround")
        overround_adj = 1.0 - np.clip(overround - 0.20, 0.0, 0.15) / 0.15
        return favorite_prob * overround_adj

    @staticmethod
    def _aggressive_runner_up_mask(
        race_df: pd.DataFrame,
        *,
        current_mask: pd.Series,
        selection_edge: pd.Series,
        selection_prob: pd.Series,
        odds: pd.Series,
        max_place_odds: float,
        regime_params: dict[str, Any],
    ) -> pd.Series:
        if "place_gate_score" not in race_df.columns or "race_id" not in race_df.columns:
            return pd.Series(False, index=race_df.index, dtype=bool)

        score_rank, score_gap = RacePredictor._gate_rank_metrics(race_df)
        valid_odds = odds.notna() & (odds > 0) & (odds <= max_place_odds)
        selected_races = race_df["race_id"].isin(race_df.loc[current_mask, "race_id"])
        second_rank_mask = score_rank.eq(2)
        extra_mask = pd.Series(False, index=race_df.index, dtype=bool)

        quality_second_margin = regime_params.get("quality_second_margin")
        if isinstance(quality_second_margin, (int, float)):
            quality_second_min_edge = float(
                regime_params.get(
                    "quality_second_min_edge",
                    regime_params.get("edge_threshold", 0.0),
                )
            )
            quality_second_min_prob = float(regime_params.get("quality_second_min_prob", 0.0))
            extra_mask |= (
                selected_races
                & second_rank_mask
                & score_gap.le(float(quality_second_margin))
                & selection_edge.ge(quality_second_min_edge)
                & selection_prob.ge(quality_second_min_prob)
                & valid_odds
            )

        runner_up_rescue_margin = regime_params.get("runner_up_rescue_margin")
        if isinstance(runner_up_rescue_margin, (int, float)):
            runner_up_rescue_min_edge = float(regime_params.get("runner_up_rescue_min_edge", 0.0))
            runner_up_rescue_min_prob = float(regime_params.get("runner_up_rescue_min_prob", 0.0))
            extra_mask |= (
                ~selected_races
                & second_rank_mask
                & score_gap.le(float(runner_up_rescue_margin))
                & selection_edge.ge(runner_up_rescue_min_edge)
                & selection_prob.ge(runner_up_rescue_min_prob)
                & valid_odds
            )

        rerank_market_condition_max = regime_params.get("runner_up_rerank_market_condition_max")
        rerank_entropy_min = regime_params.get("runner_up_rerank_entropy_min")
        rerank_entropy_max = regime_params.get("runner_up_rerank_entropy_max")
        if (
            isinstance(rerank_market_condition_max, (int, float))
            and isinstance(rerank_entropy_min, (int, float))
            and isinstance(rerank_entropy_max, (int, float))
        ):
            rerank_min_edge = float(regime_params.get("runner_up_rerank_min_edge", 0.0))
            rerank_min_prob = float(regime_params.get("runner_up_rerank_min_prob", 0.0))
            rerank_max_odds = float(
                regime_params.get("runner_up_rerank_max_odds", max_place_odds)
            )
            market_condition_score = RacePredictor._market_condition_score(race_df)
            market_entropy = RacePredictor._race_first_numeric(race_df, "market_entropy")
            rerank_valid_odds = (
                odds.notna()
                & odds.gt(0.0)
                & odds.le(min(max_place_odds, rerank_max_odds))
            )
            extra_mask |= (
                ~selected_races
                & second_rank_mask
                & market_condition_score.le(float(rerank_market_condition_max))
                & market_entropy.ge(float(rerank_entropy_min))
                & market_entropy.le(float(rerank_entropy_max))
                & selection_edge.ge(rerank_min_edge)
                & selection_prob.ge(rerank_min_prob)
                & rerank_valid_odds
            )

        return extra_mask

    @staticmethod
    def _prune_place_candidates(
        candidates: pd.DataFrame,
        *,
        regime: RegimeState,
        regime_params: dict[str, Any],
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        prepared = candidates.copy()
        prune_reason = pd.Series("", index=prepared.index, dtype=object)
        selection_reason = prepared.get(
            "place_selection_reason",
            pd.Series("", index=prepared.index, dtype=object),
        ).astype(str)
        selection_prob = pd.to_numeric(prepared.get("place_selection_prob"), errors="coerce")
        selection_edge = pd.to_numeric(prepared.get("place_selection_edge"), errors="coerce")
        aggressive_tier = prepared.get(
            "aggressive_tier",
            pd.Series("weak", index=prepared.index, dtype=object),
        ).astype(str)
        surface = prepared.get(
            "surface",
            pd.Series("", index=prepared.index, dtype=object),
        ).astype(str)

        weak_prob_prune_threshold = regime_params.get("weak_prob_prune_threshold")
        if isinstance(weak_prob_prune_threshold, (int, float)):
            weak_prob_mask = aggressive_tier.eq("weak") & selection_prob.ge(
                float(weak_prob_prune_threshold)
            )
            prune_reason.loc[weak_prob_mask] = "weak_high_prob"

        if regime == RegimeState.CONSERVATIVE and bool(
            regime_params.get("prune_turf_candidates", False)
        ):
            conservative_turf_mask = surface.eq("turf")
            prune_reason.loc[conservative_turf_mask & prune_reason.eq("")] = "conservative_turf"

        add_second_keep_min_edge = regime_params.get("add_second_keep_min_edge")
        add_second_keep_max_edge = regime_params.get("add_second_keep_max_edge")
        if isinstance(add_second_keep_min_edge, (int, float)) and isinstance(
            add_second_keep_max_edge,
            (int, float),
        ):
            keep_add_second_mask = selection_edge.ge(float(add_second_keep_min_edge)) & (
                selection_edge.lt(float(add_second_keep_max_edge))
            )
            add_second_prune_mask = selection_reason.eq("add_second") & ~keep_add_second_mask
            prune_reason.loc[add_second_prune_mask & prune_reason.eq("")] = "add_second_band"

        prepared["place_prune_reason"] = prune_reason.replace("", pd.NA)
        return prepared.loc[prune_reason.eq("")].copy()

    def get_place_candidates(
        self,
        race_df: pd.DataFrame,
        *,
        regime_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if regime_params is None:
            regime = self.models.regime_detector.current_regime
            regime_params = self.models.regime_detector.get_strategy_params(regime)
        else:
            regime = self.models.regime_detector.current_regime

        prepared = self._ensure_place_selection_columns(race_df)
        edge_col = "place_selection_edge"
        ev_col = "place_selection_ev"
        prob_col = "place_selection_prob"
        if edge_col not in prepared.columns or "fukuoddslow" not in prepared.columns:
            return prepared.iloc[0:0].copy()

        edge_threshold = float(regime_params.get("edge_threshold", 0.03))
        min_place_prob = float(regime_params.get("min_place_prob", 0.0))
        max_place_odds = float(regime_params.get("max_place_odds", float("inf")))

        selection_prob = pd.to_numeric(prepared[prob_col], errors="coerce")
        odds = pd.to_numeric(prepared["fukuoddslow"], errors="coerce")
        selection_edge = pd.to_numeric(prepared[edge_col], errors="coerce")
        surface_key = (
            prepared["surface"].iloc[0]
            if "surface" in prepared.columns and not prepared.empty
            else None
        )
        submodel = self.models.submodels.get(surface_key) if surface_key is not None else None
        gate_model = getattr(submodel, "place_selection_gate", None)
        gate_enabled = bool(
            gate_model is not None and getattr(gate_model, "is_trained", False) is True
        )
        if gate_enabled:
            assert gate_model is not None
            prepared = gate_model.score(prepared)
            annotate_race_context = getattr(gate_model, "annotate_race_context", None)
            if callable(annotate_race_context):
                prepared = annotate_race_context(prepared)
            prepared["place_selection_reason"] = "rejected"
            hard_mask = prepared["place_gate_pass"].fillna(False).astype(bool)
            hard_mask &= selection_edge.fillna(float("-inf")) >= 0.0
            hard_mask &= selection_prob.fillna(0.0) >= min_place_prob
            hard_mask &= odds.notna() & (odds > 0) & (odds <= max_place_odds)
            soft_mask = pd.Series(False, index=prepared.index, dtype=bool)
            soft_pass_mask = getattr(gate_model, "soft_pass_mask", None)
            if callable(soft_pass_mask):
                soft_mask = soft_pass_mask(
                    prepared,
                    edge_floor=0.0,
                    min_prob=min_place_prob,
                    max_odds=max_place_odds,
                    max_per_race=1,
                )
            base_mask = hard_mask | soft_mask
            selected_races = prepared["race_id"].isin(prepared.loc[base_mask, "race_id"])

            runner_up_reason = pd.Series("", index=prepared.index, dtype=object)
            runner_up_candidate_reason = getattr(gate_model, "runner_up_candidate_reason", None)
            if callable(runner_up_candidate_reason) and regime == RegimeState.AGGRESSIVE:
                runner_up_reason = runner_up_candidate_reason(
                    prepared,
                    selected_races=selected_races,
                    max_odds=max_place_odds,
                )
            elif regime == RegimeState.AGGRESSIVE:
                fallback_mask = self._aggressive_runner_up_mask(
                    prepared,
                    current_mask=base_mask,
                    selection_edge=selection_edge.fillna(float("-inf")),
                    selection_prob=selection_prob.fillna(0.0),
                    odds=odds,
                    max_place_odds=max_place_odds,
                    regime_params=regime_params,
                )
                runner_up_reason.loc[fallback_mask & selected_races] = "add_second"
                runner_up_reason.loc[fallback_mask & ~selected_races] = "rescue"

            mask = base_mask | runner_up_reason.ne("")
            prepared.loc[hard_mask, "place_selection_reason"] = "hard_gate"
            prepared.loc[soft_mask & ~hard_mask, "place_selection_reason"] = "soft_gate"
            prepared.loc[
                runner_up_reason.eq("add_second"),
                "place_selection_reason",
            ] = "add_second"
            prepared.loc[runner_up_reason.eq("rescue"), "place_selection_reason"] = "rescue"
            prepared["place_selection_reason"] = prepared["place_selection_reason"].fillna(
                "rejected"
            )
        else:
            prepared["place_selection_reason"] = "rejected"
            mask = selection_edge.fillna(0.0) >= edge_threshold
            mask &= selection_prob.fillna(0.0) >= min_place_prob
            mask &= odds.notna() & (odds > 0) & (odds <= max_place_odds)
            prepared.loc[mask, "place_selection_reason"] = "threshold"

        candidates = prepared.loc[mask].copy()
        candidates = self._prune_place_candidates(
            candidates,
            regime=regime,
            regime_params=regime_params,
        )
        if gate_enabled and "place_gate_score" in candidates.columns:
            candidates = candidates.sort_values(
                ["place_gate_score", edge_col, ev_col, prob_col],
                ascending=[False, False, False, False],
            )
        elif "place_selection_prob" in candidates.columns:
            candidates = candidates.sort_values(
                [edge_col, ev_col, prob_col],
                ascending=[False, False, False],
            )
        else:
            candidates = candidates.sort_values([edge_col, ev_col], ascending=[False, False])
        return candidates

    def should_bet(self, race_df: pd.DataFrame) -> bool:
        """RaceQualityScreener でベット対象か判定"""
        features = self.build_race_features(race_df)
        return bool(self.models.quality_screener.should_bet(features))

    def select_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
        *,
        candidates: pd.DataFrame | None = None,
    ) -> list[Bet]:
        """Benter Value Betting: edge >= threshold の馬を選択 + ワイドペア生成。

        v5: 下限EV (EV_lower_place) を最優先し、未利用時のみ補正EVへフォールバック。
        Place: edge = selection_ev - 1.0
        Wide: WideTwoStageModel でスコアリング
        """
        regime = self.models.regime_detector.current_regime
        regime_params = self.models.regime_detector.get_strategy_params(regime)

        bets: list[Bet] = []
        max_bets = regime_params.get("max_bets_per_race", 3)
        ev_col = "place_selection_ev"
        if candidates is None:
            candidates = self.get_place_candidates(race_df, regime_params=regime_params)
        if candidates.empty:
            return bets

        # --- Place bets ---
        if max_bets < 2 and "place_selection_reason" in candidates.columns:
            has_add_second = candidates["place_selection_reason"].eq("add_second").any()
            if has_add_second:
                max_bets = 2
        if max_bets < 2 and "aggressive_tier" not in candidates.columns:
            soft_second_margin = regime_params.get("soft_gate_second_margin")
            soft_second_min_edge = float(
                regime_params.get(
                    "soft_gate_second_min_edge",
                    regime_params.get("edge_threshold", 0.03),
                )
            )
            quality_second_margin = regime_params.get("quality_second_margin")
            quality_second_min_edge = float(
                regime_params.get("quality_second_min_edge", soft_second_min_edge)
            )
            quality_second_min_prob = float(regime_params.get("quality_second_min_prob", 0.0))
            if (
                isinstance(soft_second_margin, (int, float))
                and len(candidates) >= 2
                and "place_gate_score" in candidates.columns
            ):
                gate_scores = pd.to_numeric(candidates["place_gate_score"], errors="coerce")
                top_score = gate_scores.iloc[0]
                second_score = gate_scores.iloc[1]
                second_edge = float(candidates["place_selection_edge"].iloc[1])
                second_prob = float(candidates["place_selection_prob"].iloc[1])
                score_gap = (
                    float(top_score) - float(second_score)
                    if pd.notna(top_score) and pd.notna(second_score)
                    else float("inf")
                )
                margin_rule = (
                    pd.notna(top_score)
                    and pd.notna(second_score)
                    and score_gap <= float(soft_second_margin)
                    and second_edge >= soft_second_min_edge
                )
                quality_rule = (
                    isinstance(quality_second_margin, (int, float))
                    and pd.notna(top_score)
                    and pd.notna(second_score)
                    and score_gap <= float(quality_second_margin)
                    and second_edge >= quality_second_min_edge
                    and second_prob >= quality_second_min_prob
                )
                if margin_rule or quality_rule:
                    max_bets = 2
        candidates = candidates.head(max_bets)

        for _, row in candidates.iterrows():
            edge_val = float(row["place_selection_edge"])
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
                    ev_lower_corrected=float(row.get(ev_col, row.get("ev_place_corrected", 0))),
                    stake=stake,
                    edge=edge_val,
                )
            )

        if bool(regime_params.get("wide_enabled", False)):
            bets.extend(self._select_wide_bets(race_df, bankroll, bets))

        return bets

    def _select_wide_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
        place_bets: list[Bet],
    ) -> list[Bet]:
        """Wide ベットペアを生成。

        Place ベット対象馬を中心に、corrected edge >= 0.03 の馬も候補に含める。
        ペアの少なくとも1頭は place ベット対象馬であること。
        """
        edge_col = "place_selection_edge"
        if edge_col not in race_df.columns:
            if "edge_place" in race_df.columns:
                race_df = race_df.copy()
                race_df[edge_col] = pd.to_numeric(race_df["edge_place"], errors="coerce")
            else:
                return []

        # Place ベット対象馬
        place_umabans = {b.umaban for b in place_bets if b.bet_type == BetType.PLACE}
        if not place_umabans:
            return []

        # 候補馬: place ベット馬 + 下限/補正EVで十分な馬
        wide_edge_col = edge_col
        min_wide_edge = 0.04
        candidate_df = race_df[race_df[wide_edge_col].fillna(0) >= min_wide_edge].copy()
        candidate_umabans = set(int(u) for u in candidate_df["umaban"])

        # 少なくとも1頭は place ベット対象
        pair_candidates = candidate_umabans
        if len(pair_candidates) < 2:
            return []

        # 行データを収集
        rows_map: dict[int, pd.Series] = {}
        for _, row in race_df.iterrows():
            u = int(row["umaban"])
            if u in pair_candidates:
                rows_map[u] = row

        pair_list = sorted(pair_candidates)
        race_id = str(race_df.iloc[0]["race_id"])
        wide_bets: list[Bet] = []

        for i in range(len(pair_list)):
            for j in range(i + 1, len(pair_list)):
                umaban_a, umaban_b = pair_list[i], pair_list[j]
                # At least one must be a place-bet horse
                if umaban_a not in place_umabans and umaban_b not in place_umabans:
                    continue

                lo, hi = min(umaban_a, umaban_b), max(umaban_a, umaban_b)
                odds_col = f"wide_odds_{lo}_{hi}"
                wide_odds = 0.0
                if odds_col in race_df.columns:
                    odds_vals = race_df[race_df["umaban"] == umaban_a][odds_col]
                    if not odds_vals.empty and pd.notna(odds_vals.iloc[0]):
                        wide_odds = float(odds_vals.iloc[0])
                if wide_odds <= 0:
                    continue

                row_a = rows_map.get(umaban_a)
                row_b = rows_map.get(umaban_b)
                if row_a is None or row_b is None:
                    continue

                fuku_a = float(row_a.get("fukuoddslow", 0))
                fuku_b = float(row_b.get("fukuoddslow", 0))
                if fuku_a <= 0 or fuku_b <= 0:
                    continue

                edge_a = float(row_a[edge_col])
                edge_b = float(row_b[edge_col])
                p_a = (edge_a + 1.0) / fuku_a
                p_b = (edge_b + 1.0) / fuku_b

                ev_wide = p_a * p_b * wide_odds
                # Wide stake: 200 yen (higher ROI bet type gets larger stake)
                wide_stake = 200.0
                wide_bets.append(
                    Bet(
                        race_id=race_id,
                        umaban=umaban_a,
                        bet_type=BetType.WIDE,
                        odds=wide_odds,
                        ev_lower_corrected=ev_wide,
                        stake=wide_stake,
                        edge=ev_wide - 1.0,
                        umaban_b=umaban_b,
                    )
                )

        wide_bets.sort(key=lambda b: b.edge, reverse=True)
        return wide_bets[:3]

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
