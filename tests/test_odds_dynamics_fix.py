"""compute_odds_dynamics の切り詰めがデータ量に関わらず常に発動することを検証。"""
import numpy as np
import pandas as pd
import pytest

from features.odds_dynamics_features import compute_odds_dynamics


def _make_ts(n_points: int, n_horses: int = 2) -> pd.DataFrame:
    """指定ポイント数のオッズ時系列を生成。"""
    rows = []
    for umaban in range(1, n_horses + 1):
        for i in range(n_points):
            rows.append({
                "race_id": "20260412010101",
                "umaban": umaban,
                "happyotime": f"{1200 + i:04d}",
                "tanodds": 5.0 + np.random.randn() * 0.1,
                "ninki": umaban,
            })
    return pd.DataFrame(rows)


def _make_ts_with_early_spike(
    n_points: int,
    spike_count: int,
    spike_odds: float,
    normal_odds: float,
    n_horses: int = 2,
) -> pd.DataFrame:
    """先頭 spike_count ポイントに高いオッズ (spike_odds)、残りに正常オッズ (normal_odds) を設定。

    切り詰めが正しく動作すれば、spike部分は除外され first_odds ≈ normal_odds になる。
    切り詰めが動作しなければ first_odds ≈ spike_odds となり、drop_rate が大きくなる。
    """
    rows = []
    for umaban in range(1, n_horses + 1):
        for i in range(n_points):
            odds = spike_odds if i < spike_count else normal_odds
            rows.append({
                "race_id": "20260412010101",
                "umaban": umaban,
                "happyotime": f"{1200 + i:04d}",
                "tanodds": odds,
                "ninki": umaban,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("n_points", [100, 200])
def test_truncation_always_applies(n_points: int):
    """切り詰め (max_points=60) はデータ量に関わらず常に発動し、特徴量が計算される。"""
    base = pd.DataFrame({
        "race_id": ["20260412010101"] * 2,
        "umaban": [1, 2],
    })
    ts = _make_ts(n_points)
    result = compute_odds_dynamics(base, ts)
    assert result["odds_drop_rate_60_10"].notna().all()


def test_truncation_limit_60_points():
    """各 (race_id, umaban) が最大60ポイントに切り詰められることを drop_rate で検証。

    100ポイントのうち先頭40ポイントは tanodds=100.0、後半60ポイントは tanodds=5.0。
    切り詰めが正しく動作すれば、先頭40ポイントは除外され first_odds ≈ 5.0 となるため
    drop_rate_60_10 ≈ (5.0 - 5.0) / 5.0 = 0.0。
    切り詰めが動作しなければ first_odds = 100.0 となり drop_rate ≈ 0.95。
    """
    base = pd.DataFrame({
        "race_id": ["20260412010101"] * 2,
        "umaban": [1, 2],
    })
    # 100 points: first 40 at 100.0, last 60 at 5.0
    ts = _make_ts_with_early_spike(
        n_points=100, spike_count=40, spike_odds=100.0, normal_odds=5.0, n_horses=2,
    )
    result = compute_odds_dynamics(base, ts)

    # After truncation to 60, only the last 60 points (all 5.0) remain
    # so drop_rate should be ~0, NOT ~0.95
    drop_rates = result["odds_drop_rate_60_10"].values
    assert len(drop_rates) == 2
    for rate in drop_rates:
        assert abs(rate) < 0.1, (
            f"drop_rate should be near 0 after truncation, got {rate}"
        )
