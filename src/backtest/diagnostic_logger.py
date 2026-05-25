"""バックテスト・ペーパートレード診断ログ出力"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RaceDiagnostic:
    """1レースごとの診断情報"""

    race_id: str
    regime: str
    ev_threshold: float
    edge_threshold: float = 0.0
    quality_passed: bool = True
    quality_score: float = 0.0
    n_candidates: int = 0
    n_bets: int = 0
    aggressive_strength: float | None = None
    aggressive_tier: str | None = None
    market_condition_score: float | None = None


@dataclass
class HorseDiagnostic:
    """1馬ごとの診断情報"""

    race_id: str
    umaban: int
    p_place_pred: float
    e_return_place_pred: float
    ev_place: float
    fukuoddslow: float
    is_bet: bool
    is_actual_bet: bool = False
    p_place_corrected: float | None = None
    e_return_place_corrected: float | None = None
    ev_place_corrected: float | None = None
    ev_lower_place: float | None = None
    place_selection_ev: float | None = None
    place_selection_edge: float | None = None
    place_selection_prob: float | None = None
    place_bucket_multiplier: float | None = None
    place_gate_score: float | None = None
    place_gate_pass: bool | None = None
    place_gate_rank: float | None = None
    place_gate_score_gap: float | None = None
    market_condition_score: float | None = None
    aggressive_strength: float | None = None
    aggressive_tier: str | None = None
    place_selection_reason: str | None = None
    p_win_pred: float | None = None
    p_win_corrected: float | None = None
    p_win_final: float | None = None
    e_return_win_pred: float | None = None
    e_return_win_corrected: float | None = None
    win_selection_ev: float | None = None
    win_selection_ev_tail_calibrated: float | None = None
    win_selection_edge: float | None = None
    win_selection_prob: float | None = None
    win_gate_score: float | None = None
    win_gate_pass: bool | None = None
    win_gate_odds_score: float | None = None
    win_gate_prob_score: float | None = None
    win_gate_edge_score: float | None = None
    win_gate_edge_odds_score: float | None = None
    p_market_win_raw: float | None = None
    p_market_win_norm: float | None = None
    win_market_residual: float | None = None
    win_market_logit_edge: float | None = None
    win_market_prob_ratio: float | None = None
    win_market_value_ratio: float | None = None
    win_market_selection_score: float | None = None
    win_profit_score: float | None = None
    win_profit_selector_pass: bool | None = None
    win_profit_rank: float | None = None
    win_profit_stake_scale: float | None = None
    win_profit_reason: str | None = None
    win_late_odds_drop_z: float | None = None
    win_late_odds_drop_weight: float | None = None
    win_ev_tail_pressure: float | None = None
    win_ev_tail_penalty_weight: float | None = None
    win_log_odds: float | None = None
    win_log_odds_penalty: float | None = None
    win_model_prob_rank: float | None = None
    win_prob_rank_bonus: float | None = None
    win_market_risk_penalty: float | None = None
    risk_flags: str | None = None
    tanodds: float | None = None
    closing_win_odds: float | None = None
    clv: float | None = None
    final_odds: float | None = None
    stake: float | None = None
    result: float | None = None
    excluded_reason: str | None = None
    filter_pass_flags: str | None = None
    candidate_count_before_filter: int | None = None
    candidate_count_after_filter: int | None = None
    selected_rank_by_p_win_final: float | None = None
    selected_rank_by_win_selection_ev: float | None = None
    selected_rank_by_win_market_logit_edge: float | None = None
    selected_rank_by_win_market_score: float | None = None


class DiagnosticLogger:
    """レース・馬単位の診断情報を収集し、CSVに出力する。"""

    def __init__(self) -> None:
        self.race_records: list[RaceDiagnostic] = []
        self.horse_records: list[HorseDiagnostic] = []
        self.feature_records: list[dict[str, Any]] = []

    def log_race(
        self,
        race_id: str,
        regime: str,
        ev_threshold: float,
        edge_threshold: float = 0.0,
        quality_passed: bool = True,
        quality_score: float = 0.0,
        n_candidates: int = 0,
        n_bets: int = 0,
        aggressive_strength: float | None = None,
        aggressive_tier: str | None = None,
        market_condition_score: float | None = None,
    ) -> None:
        self.race_records.append(
            RaceDiagnostic(
                race_id=race_id,
                regime=regime,
                ev_threshold=ev_threshold,
                edge_threshold=edge_threshold,
                quality_passed=quality_passed,
                quality_score=quality_score,
                n_candidates=n_candidates,
                n_bets=n_bets,
                aggressive_strength=aggressive_strength,
                aggressive_tier=aggressive_tier,
                market_condition_score=market_condition_score,
            )
        )

    def log_horse(
        self,
        race_id: str,
        umaban: int,
        p_place_pred: float,
        e_return_place_pred: float,
        ev_place: float,
        fukuoddslow: float,
        is_bet: bool,
        is_actual_bet: bool | None = None,
        p_place_corrected: float | None = None,
        e_return_place_corrected: float | None = None,
        ev_place_corrected: float | None = None,
        ev_lower_place: float | None = None,
        place_selection_ev: float | None = None,
        place_selection_edge: float | None = None,
        place_selection_prob: float | None = None,
        place_bucket_multiplier: float | None = None,
        place_gate_score: float | None = None,
        place_gate_pass: bool | None = None,
        place_gate_rank: float | None = None,
        place_gate_score_gap: float | None = None,
        market_condition_score: float | None = None,
        aggressive_strength: float | None = None,
        aggressive_tier: str | None = None,
        place_selection_reason: str | None = None,
        p_win_pred: float | None = None,
        p_win_corrected: float | None = None,
        p_win_final: float | None = None,
        e_return_win_pred: float | None = None,
        e_return_win_corrected: float | None = None,
        win_selection_ev: float | None = None,
        win_selection_ev_tail_calibrated: float | None = None,
        win_selection_edge: float | None = None,
        win_selection_prob: float | None = None,
        win_gate_score: float | None = None,
        win_gate_pass: bool | None = None,
        win_gate_odds_score: float | None = None,
        win_gate_prob_score: float | None = None,
        win_gate_edge_score: float | None = None,
        win_gate_edge_odds_score: float | None = None,
        p_market_win_raw: float | None = None,
        p_market_win_norm: float | None = None,
        win_market_residual: float | None = None,
        win_market_logit_edge: float | None = None,
        win_market_prob_ratio: float | None = None,
        win_market_value_ratio: float | None = None,
        win_market_selection_score: float | None = None,
        win_profit_score: float | None = None,
        win_profit_selector_pass: bool | None = None,
        win_profit_rank: float | None = None,
        win_profit_stake_scale: float | None = None,
        win_profit_reason: str | None = None,
        win_late_odds_drop_z: float | None = None,
        win_late_odds_drop_weight: float | None = None,
        win_ev_tail_pressure: float | None = None,
        win_ev_tail_penalty_weight: float | None = None,
        win_log_odds: float | None = None,
        win_log_odds_penalty: float | None = None,
        win_model_prob_rank: float | None = None,
        win_prob_rank_bonus: float | None = None,
        win_market_risk_penalty: float | None = None,
        risk_flags: str | None = None,
        tanodds: float | None = None,
        closing_win_odds: float | None = None,
        clv: float | None = None,
        final_odds: float | None = None,
        stake: float | None = None,
        result: float | None = None,
        excluded_reason: str | None = None,
        filter_pass_flags: str | None = None,
        candidate_count_before_filter: int | None = None,
        candidate_count_after_filter: int | None = None,
        selected_rank_by_p_win_final: float | None = None,
        selected_rank_by_win_selection_ev: float | None = None,
        selected_rank_by_win_market_logit_edge: float | None = None,
        selected_rank_by_win_market_score: float | None = None,
    ) -> None:
        self.horse_records.append(
            HorseDiagnostic(
                race_id=race_id,
                umaban=umaban,
                p_place_pred=p_place_pred,
                e_return_place_pred=e_return_place_pred,
                ev_place=ev_place,
                fukuoddslow=fukuoddslow,
                is_bet=is_bet,
                is_actual_bet=is_bet if is_actual_bet is None else is_actual_bet,
                p_place_corrected=p_place_corrected,
                e_return_place_corrected=e_return_place_corrected,
                ev_place_corrected=ev_place_corrected,
                ev_lower_place=ev_lower_place,
                place_selection_ev=place_selection_ev,
                place_selection_edge=place_selection_edge,
                place_selection_prob=place_selection_prob,
                place_bucket_multiplier=place_bucket_multiplier,
                place_gate_score=place_gate_score,
                place_gate_pass=place_gate_pass,
                place_gate_rank=place_gate_rank,
                place_gate_score_gap=place_gate_score_gap,
                market_condition_score=market_condition_score,
                aggressive_strength=aggressive_strength,
                aggressive_tier=aggressive_tier,
                place_selection_reason=place_selection_reason,
                p_win_pred=p_win_pred,
                p_win_corrected=p_win_corrected,
                p_win_final=p_win_final,
                e_return_win_pred=e_return_win_pred,
                e_return_win_corrected=e_return_win_corrected,
                win_selection_ev=win_selection_ev,
                win_selection_ev_tail_calibrated=win_selection_ev_tail_calibrated,
                win_selection_edge=win_selection_edge,
                win_selection_prob=win_selection_prob,
                win_gate_score=win_gate_score,
                win_gate_pass=win_gate_pass,
                win_gate_odds_score=win_gate_odds_score,
                win_gate_prob_score=win_gate_prob_score,
                win_gate_edge_score=win_gate_edge_score,
                win_gate_edge_odds_score=win_gate_edge_odds_score,
                p_market_win_raw=p_market_win_raw,
                p_market_win_norm=p_market_win_norm,
                win_market_residual=win_market_residual,
                win_market_logit_edge=win_market_logit_edge,
                win_market_prob_ratio=win_market_prob_ratio,
                win_market_value_ratio=win_market_value_ratio,
                win_market_selection_score=win_market_selection_score,
                win_profit_score=win_profit_score,
                win_profit_selector_pass=win_profit_selector_pass,
                win_profit_rank=win_profit_rank,
                win_profit_stake_scale=win_profit_stake_scale,
                win_profit_reason=win_profit_reason,
                win_late_odds_drop_z=win_late_odds_drop_z,
                win_late_odds_drop_weight=win_late_odds_drop_weight,
                win_ev_tail_pressure=win_ev_tail_pressure,
                win_ev_tail_penalty_weight=win_ev_tail_penalty_weight,
                win_log_odds=win_log_odds,
                win_log_odds_penalty=win_log_odds_penalty,
                win_model_prob_rank=win_model_prob_rank,
                win_prob_rank_bonus=win_prob_rank_bonus,
                win_market_risk_penalty=win_market_risk_penalty,
                risk_flags=risk_flags,
                tanodds=tanodds,
                closing_win_odds=closing_win_odds,
                clv=clv,
                final_odds=final_odds,
                stake=stake,
                result=result,
                excluded_reason=excluded_reason,
                filter_pass_flags=filter_pass_flags,
                candidate_count_before_filter=candidate_count_before_filter,
                candidate_count_after_filter=candidate_count_after_filter,
                selected_rank_by_p_win_final=selected_rank_by_p_win_final,
                selected_rank_by_win_selection_ev=selected_rank_by_win_selection_ev,
                selected_rank_by_win_market_logit_edge=selected_rank_by_win_market_logit_edge,
                selected_rank_by_win_market_score=selected_rank_by_win_market_score,
            )
        )

    def log_horse_features(self, row: dict[str, Any]) -> None:
        """result_df の1行（特徴量+予測値+判定）を収集する。list/dict 値は除外。"""
        self.feature_records.append(
            {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
        )

    def save(self, outdir: Path, prefix: str = "diag") -> None:
        """診断レコードをCSVに出力。レコードが0件ならファイルを作成しない。"""
        outdir.mkdir(parents=True, exist_ok=True)

        if self.race_records:
            path = outdir / f"{prefix}_race_diagnostics.csv"
            pd.DataFrame([asdict(r) for r in self.race_records]).to_csv(path, index=False)
            logger.info("Race diagnostics saved: %d records -> %s", len(self.race_records), path)

        if self.horse_records:
            path = outdir / f"{prefix}_horse_diagnostics.csv"
            pd.DataFrame([asdict(r) for r in self.horse_records]).to_csv(path, index=False)
            logger.info("Horse diagnostics saved: %d records -> %s", len(self.horse_records), path)

        if self.feature_records:
            path = outdir / f"{prefix}_horse_features.parquet"
            pd.DataFrame(self.feature_records).to_parquet(path, index=False)
            logger.info(
                "Feature diagnostics saved: %d records -> %s",
                len(self.feature_records),
                path,
            )
