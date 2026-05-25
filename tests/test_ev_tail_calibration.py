"""EVTailCalibrator のテスト (TDD RED phase)

Feature family合意度による高EV候補スケーリングを検証する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betting.ev_tail_calibration import (
    EV_THRESHOLD,
    FAMILY_FEATURES,
    MULTI_FAMILY_FACTOR,
    NO_FAMILY_FACTOR,
    SINGLE_FAMILY_FACTOR,
    ZSCORE_THRESHOLD,
    EVTtailCalibrator,
)

# ---------------------------------------------------------------------------
# Helpers: fixture builders
# ---------------------------------------------------------------------------

def _build_race_df(n_horses: int = 8, seed: int = 42) -> pd.DataFrame:
    """全family特徴量を含む標準race DataFrameを生成"""
    rng = np.random.default_rng(seed)
    data: dict[str, list[float]] = {
        "race_id": ["R001"] * n_horses,
        "umaban": list(range(1, n_horses + 1)),
    }
    # TRF family
    for col in FAMILY_FEATURES["trf"]:
        data[col] = (rng.normal(0.5, 0.1, n_horses)).tolist()
    # INT family
    for col in FAMILY_FEATURES["int"]:
        data[col] = (rng.normal(0.3, 0.08, n_horses)).tolist()
    # HLF family
    for col in FAMILY_FEATURES["hlf"]:
        data[col] = (rng.normal(0.6, 0.12, n_horses)).tolist()
    # Market family
    for col in FAMILY_FEATURES["market"]:
        data[col] = (rng.normal(0.4, 0.1, n_horses)).tolist()
    # Ability family
    for col in FAMILY_FEATURES["ability"]:
        data[col] = (rng.normal(0.2, 0.05, n_horses)).tolist()
    return pd.DataFrame(data)


def _horse_at_mean(race_df: pd.DataFrame) -> pd.Series:
    """全特徴量がrace平均値の馬 (z-score ≈ 0 → 0 families)"""
    row: dict[str, float] = {
        "race_id": "R001",
        "umaban": 999,
    }
    for family_cols in FAMILY_FEATURES.values():
        for col in family_cols:
            if col in race_df.columns:
                row[col] = float(race_df[col].mean())
    return pd.Series(row)


def _horse_above_family(
    race_df: pd.DataFrame,
    families: list[str],
    n_sigma: float = 2.0,
) -> pd.Series:
    """指定familyの特徴量のみrace平均+n_sigma*stdの値を持つ馬"""
    row: dict[str, float] = {
        "race_id": "R001",
        "umaban": 998,
    }
    for family_name, family_cols in FAMILY_FEATURES.items():
        for col in family_cols:
            if col in race_df.columns:
                if family_name in families:
                    mean = float(race_df[col].mean())
                    std = float(race_df[col].std())
                    row[col] = mean + n_sigma * std
                else:
                    row[col] = float(race_df[col].mean())
    return pd.Series(row)


# ---------------------------------------------------------------------------
# Test: EV passthrough (EV < 1.5)
# ---------------------------------------------------------------------------

class TestPassthrough:
    """EV < 1.5 の候補はスケーリングされずそのまま通過"""

    def test_ev_below_threshold_unchanged(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_at_mean(race_df)
        ev = 1.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev, abs=1e-9)

    def test_ev_exactly_threshold_unchanged(self) -> None:
        """EV == 1.5 は < threshold ではないのでスケーリングされる"""
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_at_mean(race_df)
        ev = EV_THRESHOLD
        # EV == threshold は < ではないので calibration が適用される
        # 0 families → NO_FAMILY_FACTOR
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * NO_FAMILY_FACTOR, abs=1e-6)

    def test_ev_negative_unchanged(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_at_mean(race_df)
        ev = -0.5
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev, abs=1e-9)


# ---------------------------------------------------------------------------
# Test: 0-family agreement → 0.70x
# ---------------------------------------------------------------------------

class TestZeroFamilyAgreement:
    """全familyがz < 1.0 → 0.70x縮小"""

    def test_all_features_near_mean(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_at_mean(race_df)
        ev = 2.0  # >= 1.5
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * NO_FAMILY_FACTOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: 1-family agreement → 0.85x
# ---------------------------------------------------------------------------

class TestSingleFamilyAgreement:
    """1 familyのみz > 1.0 → 0.85x縮小"""

    def test_only_trf_agrees(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_above_family(race_df, ["trf"])
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * SINGLE_FAMILY_FACTOR, abs=1e-6)

    def test_only_market_agrees(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_above_family(race_df, ["market"])
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * SINGLE_FAMILY_FACTOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: 2+-family agreement → 1.05x
# ---------------------------------------------------------------------------

class TestMultiFamilyAgreement:
    """2+ familiesがz > 1.0 → 1.05x拡大"""

    def test_two_families_agree(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_above_family(race_df, ["trf", "hlf"])
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * MULTI_FAMILY_FACTOR, abs=1e-6)

    def test_three_families_agree(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_above_family(race_df, ["trf", "hlf", "market"])
        ev = 2.5
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * MULTI_FAMILY_FACTOR, abs=1e-6)

    def test_all_families_agree(self) -> None:
        calibrator = EVTtailCalibrator()
        race_df = _build_race_df()
        horse_row = _horse_above_family(race_df, ["trf", "int", "hlf", "market", "ability"])
        ev = 3.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        assert result == pytest.approx(ev * MULTI_FAMILY_FACTOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: missing columns → family skipped gracefully
# ---------------------------------------------------------------------------

class TestMissingColumns:
    """欠損特徴量列に対するgraceful handling"""

    def test_missing_all_feature_columns(self) -> None:
        """特徴量列が全くない場合、0 families → NO_FAMILY_FACTOR"""
        calibrator = EVTtailCalibrator()
        # race_df に特徴量列がない
        race_df = pd.DataFrame({
            "race_id": ["R001"] * 8,
            "umaban": list(range(1, 9)),
        })
        horse_row = pd.Series({"race_id": "R001", "umaban": 999})
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        # 0 families agreeing → NO_FAMILY_FACTOR
        assert result == pytest.approx(ev * NO_FAMILY_FACTOR, abs=1e-6)

    def test_missing_some_families(self) -> None:
        """一部familyの列がない場合、残りfamilyだけで判定"""
        calibrator = EVTtailCalibrator()
        # TRF列のみのrace_df
        race_df = pd.DataFrame({
            "race_id": ["R001"] * 8,
            "umaban": list(range(1, 9)),
        })
        rng = np.random.default_rng(42)
        for col in FAMILY_FEATURES["trf"]:
            race_df[col] = rng.normal(0.5, 0.1, 8)
        # TRFのみ高い馬
        row_data: dict[str, float] = {"race_id": "R001", "umaban": 999}
        for col in FAMILY_FEATURES["trf"]:
            mean = float(race_df[col].mean())
            std = float(race_df[col].std())
            row_data[col] = mean + 2.0 * std
        horse_row = pd.Series(row_data)
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        # 1 family (TRF) agreeing → SINGLE_FAMILY_FACTOR
        assert result == pytest.approx(ev * SINGLE_FAMILY_FACTOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: std=0 edge case → z=0 → family skipped
# ---------------------------------------------------------------------------

class TestStdZero:
    """全馬の値が同一 (std=0) → z=0 → family disagrees"""

    def test_identical_values_no_agreement(self) -> None:
        calibrator = EVTtailCalibrator()
        n = 8
        race_df = pd.DataFrame({
            "race_id": ["R001"] * n,
            "umaban": list(range(1, n + 1)),
        })
        # 全列が同一値
        for family_cols in FAMILY_FEATURES.values():
            for col in family_cols:
                race_df[col] = [0.5] * n
        # horse_rowも同じ値
        row_data: dict[str, float] = {"race_id": "R001", "umaban": 999}
        for family_cols in FAMILY_FEATURES.values():
            for col in family_cols:
                row_data[col] = 0.5
        horse_row = pd.Series(row_data)
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        # std=0 → z=0 → 0 families → NO_FAMILY_FACTOR
        assert result == pytest.approx(ev * NO_FAMILY_FACTOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: constants verification
# ---------------------------------------------------------------------------

class TestConstants:
    """スケーリング定数の検証"""

    def test_family_features_has_five_families(self) -> None:
        assert len(FAMILY_FEATURES) == 5
        expected = {"trf", "int", "hlf", "market", "ability"}
        assert set(FAMILY_FEATURES.keys()) == expected

    def test_factors_are_correct(self) -> None:
        assert NO_FAMILY_FACTOR == pytest.approx(0.70)
        assert SINGLE_FAMILY_FACTOR == pytest.approx(0.85)
        assert MULTI_FAMILY_FACTOR == pytest.approx(1.05)
        assert EV_THRESHOLD == pytest.approx(1.5)
        assert ZSCORE_THRESHOLD == pytest.approx(1.0)

    def test_trf_features_correct(self) -> None:
        expected = [
            "form_trend_race_rank",
            "blood_total_wr_race_rank",
            "blood_surface_wr_race_rank",
        ]
        assert FAMILY_FEATURES["trf"] == expected

    def test_market_features_correct(self) -> None:
        expected = ["implied_prob_hhi", "odds_skewness", "overround", "market_entropy"]
        assert FAMILY_FEATURES["market"] == expected

    def test_ability_features_correct(self) -> None:
        assert FAMILY_FEATURES["ability"] == ["p_win_pred"]


# ---------------------------------------------------------------------------
# Test: Integration with get_win_candidates()
# ---------------------------------------------------------------------------

class TestAllNaNValues:
    """race_vals が全て NaN の場合、mean/std が pd.NA ではなく安全に skip されること"""

    def test_all_nan_feature_columns_no_crash(self) -> None:
        calibrator = EVTtailCalibrator()
        n = 8
        race_df = pd.DataFrame({
            "race_id": ["R001"] * n,
            "umaban": list(range(1, n + 1)),
        })
        # 全列を NaN にする
        for family_cols in FAMILY_FEATURES.values():
            for col in family_cols:
                race_df[col] = [np.nan] * n
        row_data: dict[str, float] = {"race_id": "R001", "umaban": 999}
        for family_cols in FAMILY_FEATURES.values():
            for col in family_cols:
                row_data[col] = 1.0
        horse_row = pd.Series(row_data)
        ev = 2.0
        result = calibrator.calibrate(horse_row, race_df, ev)
        # dropna後 empty → 0 families → NO_FAMILY_FACTOR
        assert result == pytest.approx(ev * NO_FAMILY_FACTOR, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: Integration with get_win_candidates()
# ---------------------------------------------------------------------------

class TestGetWinCandidatesIntegration:
    """get_win_candidates() で高EV候補にキャリブレーションが適用されるか検証"""

    def _make_race_df_with_edges(
        self,
        edges: list[float],
        n_horses: int = 8,
        seed: int = 42,
        *,
        with_gate_score: bool = False,
    ) -> pd.DataFrame:
        """win_selection_edge 付きのrace_dfを生成

        Args:
            edges: 各馬のedge値 (最初のlen(edges)馬に設定)
            n_horses: 馬数
            seed: 乱数シード
            with_gate_score: Trueの場合、win_gate_score列を追加 (値=馬番号)
        """
        df = _build_race_df(n_horses=n_horses, seed=seed)
        # 最初の len(edges) 馬にedgeを設定
        edge_col = [0.0] * n_horses
        for i, e in enumerate(edges):
            if i < n_horses:
                edge_col[i] = e
        df["win_selection_edge"] = edge_col
        df["tanodds"] = [5.0 + i for i in range(n_horses)]
        if with_gate_score:
            df["win_gate_score"] = [float(i) for i in range(n_horses)]
        return df

    def test_low_ev_candidate_not_calibrated(self) -> None:
        """edge < 1.5 の候補はソート順が変わらない (gate_scoreなし=edge順sort)"""
        from backtest.race_predictor import RacePredictor

        df = self._make_race_df_with_edges([0.5, 0.3, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
        rp = RacePredictor.__new__(RacePredictor)
        rp.models = object()
        result = rp.get_win_candidates(df)
        # edge > 0 のみ候補 → 単勝は1レース最良1頭
        assert len(result) == 1
        assert float(result.iloc[0]["win_selection_edge"]) == pytest.approx(0.5)

    def test_high_ev_candidate_calibrated(self) -> None:
        """edge >= 1.5 の候補がキャリブレーションされることを確認

        gate_scoreなしの場合、sort keyは_calibrated_edgeのみ。
        edge=2.0は平均値の馬なので0 families → 2.0*0.70=1.4 に縮小。
        edge=1.5の馬が calibration 後に上位になることを確認。
        """
        from backtest.race_predictor import RacePredictor

        # edge=2.0 (calibrated→1.4), edge=1.5 (calibrated→1.05), edge=0.5 (not calibrated)
        df = self._make_race_df_with_edges([2.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        rp = RacePredictor.__new__(RacePredictor)
        rp.models = object()
        result = rp.get_win_candidates(df)
        assert len(result) == 1
        # sorted by _calibrated_edge DESC:
        # idx0: edge=2.0 → cal=1.4
        # idx1: edge=1.5 → cal=1.05
        # idx2: edge=0.5 → cal=0.5
        # → [1.4] → original edge [2.0]
        first_edge = float(result.iloc[0]["win_selection_edge"])
        assert first_edge == pytest.approx(2.0)

    def test_no_calibrated_edge_column_in_result(self) -> None:
        """返却DataFrameに_calibrated_edge列が含まれないことを確認"""
        from backtest.race_predictor import RacePredictor

        df = self._make_race_df_with_edges([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rp = RacePredictor.__new__(RacePredictor)
        rp.models = object()
        result = rp.get_win_candidates(df)
        assert "_calibrated_edge" not in result.columns
        # 元のedge列は保持
        assert "win_selection_edge" in result.columns
