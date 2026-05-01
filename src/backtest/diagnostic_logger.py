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
