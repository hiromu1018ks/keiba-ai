"""D-06/D-07: 動的オッズバンドフィルター。トレーニング期間ROI < 100%のバンドを除外。"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class OddsBandFilter:
    """D-06: 動的オッズバンドフィルター。トレーニング期間ROI < 100%のバンドを除外 (D-07)。"""

    BANDS: list[tuple[float, float]] = [
        (1.0, 3.0),
        (3.0, 10.0),
        (10.0, 30.0),
        (30.0, float("inf")),
    ]
    BAND_NAMES: list[str] = ["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"]

    def __init__(self) -> None:
        self._excluded_bands: set[str] = set()
        self._band_roi: dict[str, float] = {}
        self._band_counts: dict[str, int] = {}

    @staticmethod
    def _get_band_name(odds: float) -> str:
        """オッズ値からバンド名を返す (report.py _band_stats と同じ境界)"""
        for (lo, hi), name in zip(OddsBandFilter.BANDS, OddsBandFilter.BAND_NAMES):
            if lo <= odds < hi:
                return name
        return "30.0+"  # fallback for inf/edge cases

    def calibrate(self, bet_history: list[dict[str, Any]]) -> None:
        """D-05: トレーニング期間ベットデータから各バンドROIを計算。D-07: ROI < 100% を除外。

        bet_history items must have keys: "odds" (decision-time odds), "result" (payout), "stake".
        """
        if not bet_history:
            return

        # Group by band and compute ROI
        band_data: dict[str, dict[str, float]] = {}
        for name in self.BAND_NAMES:
            band_data[name] = {"total_stake": 0.0, "total_return": 0.0, "count": 0}

        for bet in bet_history:
            odds = float(bet.get("odds", 0))
            if odds < 1.0:
                continue  # Skip bets with invalid/missing odds
            band = self._get_band_name(odds)
            band_data[band]["total_stake"] += float(bet.get("stake", 0))
            band_data[band]["total_return"] += float(bet.get("result", 0))
            band_data[band]["count"] += 1

        self._excluded_bands = set()
        self._band_roi = {}
        self._band_counts = {}
        for name, data in band_data.items():
            count = int(data["count"])
            self._band_counts[name] = count
            if count == 0:
                continue
            roi = data["total_return"] / data["total_stake"] if data["total_stake"] > 0 else 0.0
            self._band_roi[name] = roi
            if roi < 1.0:  # D-07: ROI < 100% → exclude
                self._excluded_bands.add(name)

        logger.info(
            "OddsBandFilter calibration: excluded_bands=%s, band_roi=%s, band_counts=%s",
            self._excluded_bands,
            self._band_roi,
            self._band_counts,
        )

    def filter(self, candidate_df: pd.DataFrame, odds_col: str = "tanodds") -> pd.DataFrame:
        """除外バンドに該当する候補を除外してDataFrameを返す。"""
        if candidate_df.empty or not self._excluded_bands:
            return candidate_df
        if odds_col not in candidate_df.columns:
            return candidate_df

        odds = pd.to_numeric(candidate_df[odds_col], errors="coerce")
        mask = pd.Series([True] * len(candidate_df), index=candidate_df.index)
        for band_name in self._excluded_bands:
            band_idx = self.BAND_NAMES.index(band_name)
            lo, hi = self.BANDS[band_idx]
            band_mask = (odds >= lo) & (odds < hi)
            mask &= ~band_mask

        excluded = candidate_df.loc[~mask]
        if not excluded.empty:
            logger.info(
                "OddsBandFilter excluded %d candidates in bands: %s",
                len(excluded),
                self._excluded_bands,
            )

        return candidate_df.loc[mask].copy()

    @property
    def excluded_bands(self) -> dict[str, dict[str, Any]]:
        """D-08: 除外バンド情報を返す (band_name -> {roi, count})"""
        return {
            name: {"roi": self._band_roi.get(name, 0.0), "count": self._band_counts.get(name, 0)}
            for name in self._excluded_bands
        }
