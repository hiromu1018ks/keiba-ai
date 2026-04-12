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
    quality_passed: bool
    quality_score: float
    n_candidates: int
    n_bets: int


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
        quality_passed: bool,
        quality_score: float,
        n_candidates: int,
        n_bets: int,
    ) -> None:
        self.race_records.append(
            RaceDiagnostic(
                race_id=race_id,
                regime=regime,
                ev_threshold=ev_threshold,
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
            )
        )

    def log_horse_features(self, row: dict[str, Any]) -> None:
        """result_df の1行（特徴量+予測値+判定）を収集する。"""
        self.feature_records.append(row)

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
