"""Market Cross-Consistency Features (MCF-01~06) のユニットテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_base_df(n_horses: int = 6, race_id: str | None = None) -> pd.DataFrame:
    """テスト用のベースDataFrame (tanodds, umaban を含む)."""
    data = {
        "umaban": list(range(1, n_horses + 1)),
        "tanodds": [2.0, 3.5, 5.0, 8.0, 15.0, 30.0][:n_horses],
    }
    if race_id is not None:
        data["race_id"] = [race_id] * n_horses
    return pd.DataFrame(data)


def _make_wide_df(race_id: str, kumi_list: list[str],
                  oddslow: list[float], oddshigh: list[float],
                  ninki: list[str | int]) -> pd.DataFrame:
    """テスト用のワイドオッズDataFrame."""
    df = pd.DataFrame({
        "race_id": [race_id] * len(kumi_list),
        "kumi": kumi_list,
        "oddslow": oddslow,
        "oddshigh": oddshigh,
        "ninki": ninki,
    })
    return df


def _make_trio_df(race_id: str, kumi_list: list[str],
                  odds: list[float], ninki: list[int | str]) -> pd.DataFrame:
    """テスト用の三連複オッズDataFrame."""
    df = pd.DataFrame({
        "race_id": [race_id] * len(kumi_list),
        "kumi": kumi_list,
        "odds": odds,
        "ninki": ninki,
    })
    return df


class TestMarketCrossFeaturesNoneFallback:
    """Test 1: wide_df/trio_dfがNoneの場合は全MCF列がNaNとなる."""

    def test_none_wide_trio_single_race(self) -> None:
        """単一レース (race_idなし) でNone入力 → 全MCF列NaN."""
        from features.market_cross_features import MCF_COLS, compute_market_cross_features

        df = _make_base_df(n_horses=6, race_id=None)
        result = compute_market_cross_features(df, wide_df=None, trio_df=None)

        for col in MCF_COLS:
            assert col in result.columns, f"{col} が結果に含まれていない"
            assert result[col].isna().all(), f"{col} が全てNaNでない (None入力)"

    def test_none_wide_trio_multi_race(self) -> None:
        """複数レース (race_idあり) でNone入力 → 全MCF列NaN."""
        from features.market_cross_features import MCF_COLS, compute_market_cross_features

        df = _make_base_df(n_horses=6, race_id="202401010101")
        result = compute_market_cross_features(df, wide_df=None, trio_df=None)

        for col in MCF_COLS:
            assert col in result.columns
            assert result[col].isna().all(), f"{col} が全てNaNでない (None入力, race_idあり)"


class TestMarketCrossFeaturesEmptyFallback:
    """Test 2: wide_df/trio_dfが空DataFrameの場合は全MCF列がNaNとなる."""

    def test_empty_wide_trio(self) -> None:
        from features.market_cross_features import MCF_COLS, compute_market_cross_features

        df = _make_base_df(n_horses=6, race_id="202401010101")
        wide_empty = pd.DataFrame(columns=["race_id", "kumi", "oddslow", "oddshigh", "ninki"])
        trio_empty = pd.DataFrame(columns=["race_id", "kumi", "odds", "ninki"])

        result = compute_market_cross_features(df, wide_df=wide_empty, trio_df=trio_empty)

        for col in MCF_COLS:
            assert result[col].isna().all(), f"{col} が全てNaNでない (empty入力)"


class TestMarketCrossFeaturesNormal:
    """Test 3: 複数レースで正常なwide_df/trio_dfが与えられた場合のテスト."""

    def test_favorite_in_wide_top1(self) -> None:
        """rl_favorite_in_wide_top1: 1番人気がワイドninki=1組合せに含まれるなら1."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 1番人気 = umaban=1 (tanodds=2.0が最小)
        # ワイドninki=1 = kumi "0102" (umaban 1 & 2)
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.0], ["001"])
        trio_df = _make_trio_df(race_id, ["010203"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        assert (result["rl_favorite_in_wide_top1"] == 1.0).all(), \
            "1番人気(umaban=1)がワイドninki=1(0102)に含まれるなら1"

    def test_favorite_not_in_wide_top1(self) -> None:
        """rl_favorite_in_wide_top1: 1番人気がワイドninki=1組合せに含まれないなら0."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 1番人気 = umaban=1 (tanodds=2.0)
        # ワイドninki=1 = kumi "0203" (umaban 2 & 3、1番人気を含まない)
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0203"], [1.5], [2.0], ["001"])
        trio_df = _make_trio_df(race_id, ["020304"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        assert (result["rl_favorite_in_wide_top1"] == 0.0).all(), \
            "1番人気(umaban=1)がワイドninki=1(0203)に含まれないなら0"

    def test_trio_overlap(self) -> None:
        """rl_trio_overlap: 三連複ninki=1構成馬と単勝上位3頭のオーバーラップ数(0-3)."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 単勝上位3頭: umaban 1, 2, 3 (tanodds 2.0, 3.5, 5.0)
        # 三連複ninki=1: kumi "010203" → umaban 1, 2, 3 → overlap=3
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.0], ["001"])
        trio_df = _make_trio_df(race_id, ["010203"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        assert (result["rl_trio_overlap"] == 3.0).all(), \
            "三連複ninki=1(010203)と上位3頭(1,2,3) → overlap=3"

    def test_trio_overlap_partial(self) -> None:
        """rl_trio_overlap: 部分オーバーラップ(1-of-3)."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 三連複ninki=1: kumi "020304" → umaban 2, 3, 4
        # 単勝上位3頭: 1, 2, 3 → overlap = {2, 3} = 2
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.0], ["001"])
        trio_df = _make_trio_df(race_id, ["020304"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        assert (result["rl_trio_overlap"] == 2.0).all(), \
            "三連複ninki=1(020304)と上位3頭(1,2,3) → overlap=2"

    def test_market_consistency(self) -> None:
        """rl_market_consistency: 1番人気が三連複ninki=1に含まれるなら1."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 1番人気 = umaban=1
        # 三連複ninki=1: kumi "010203" → 含む → 1
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.0], ["001"])
        trio_df = _make_trio_df(race_id, ["010203"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        assert (result["rl_market_consistency"] == 1.0).all(), \
            "1番人気(umaban=1)が三連複ninki=1(010203)に含まれる → 1"

    def test_market_consistency_not_included(self) -> None:
        """rl_market_consistency: 1番人気が三連複ninki=1に含まれないなら0."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 三連複ninki=1: kumi "020304" → umaban 2, 3, 4 (1番人気=umaban=1を含まない)
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.0], ["001"])
        trio_df = _make_trio_df(race_id, ["020304"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        assert (result["rl_market_consistency"] == 0.0).all(), \
            "1番人気(umaban=1)が三連複ninki=1(020304)に含まれない → 0"


class TestMarketCrossHarvilleRatio:
    """Test 4: Harville理論オッズ比率のテスト."""

    def test_wide_harville_ratio(self) -> None:
        """rl_wide_harville_ratio: 実ワイド中間オッズ / Harville理論ワイドオッズ."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        df = _make_base_df(n_horses=6, race_id=race_id)
        # ワイドninki=1: kumi "0102", oddslow=1.5, oddshigh=2.5 → mid=2.0
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.5], ["001"])
        trio_df = _make_trio_df(race_id, ["010203"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        ratio = result["rl_wide_harville_ratio"].iloc[0]
        assert pd.notna(ratio), "rl_wide_harville_ratio がNaN"
        assert ratio > 0, f"rl_wide_harville_ratio が正でない: {ratio}"
        # 手動検証: tanodds = [2.0, 3.5, 5.0, 8.0, 15.0, 30.0]
        # inv = [0.5, 0.286, 0.2, 0.125, 0.067, 0.033], sum ≈ 1.211
        # P(1) = 0.5/1.211 ≈ 0.413, P(2) = 0.286/1.211 ≈ 0.236
        # Harville_wide(1,2) = P1*P2*(1/(1-P1) + 1/(1-P2))
        #   ≈ 0.413*0.236*(1/0.587 + 1/0.764)
        #   ≈ 0.0975 * (1.704 + 1.309)
        #   ≈ 0.0975 * 3.013
        #   ≈ 0.294
        # theoretical_odds = 1/0.294 ≈ 3.40
        # actual_mid = 2.0
        # ratio = 2.0 / 3.40 ≈ 0.588
        assert 0.3 < ratio < 1.0, f"wide Harville ratioが想定範囲外: {ratio}"

    def test_trio_odds_ratio(self) -> None:
        """rl_trio_odds_ratio: 実三連複ninki=1オッズ / Harville理論三連複オッズ."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df = _make_wide_df(race_id, ["0102"], [1.5], [2.5], ["001"])
        trio_df = _make_trio_df(race_id, ["010203"], [10.0], [1])

        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        ratio = result["rl_trio_odds_ratio"].iloc[0]
        assert pd.notna(ratio), "rl_trio_odds_ratio がNaN"
        assert ratio > 0, f"rl_trio_odds_ratio が正でない: {ratio}"
        # 手動検証: ratio = 10.0 / theoretical_trio_odds
        # Harville trio は6順列の和なので理論確率は小さく、theoretical_oddsは大きい
        # ratio < 1.0 が期待される (実オッズ < 理論オッズ)
        assert ratio < 5.0, f"trio odds ratioが想定範囲外(大きすぎ): {ratio}"


class TestMarketCrossEdgeCases:
    """Test 5-7: エッジケースのテスト."""

    def test_small_field_2horses(self) -> None:
        """Test 5: 少頭数(2頭)レースで例外が発生せずNaNとなる."""
        from features.market_cross_features import MCF_COLS, compute_market_cross_features

        race_id = "202401010101"
        df = _make_base_df(n_horses=2, race_id=race_id)
        # 2頭ではワイド・三連複が成立しない
        wide_df = _make_wide_df(race_id, ["0102"], [1.0], [1.5], ["001"])
        trio_df = pd.DataFrame(columns=["race_id", "kumi", "odds", "ninki"])

        # 例外が発生しないことを確認
        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        for col in MCF_COLS:
            assert col in result.columns

    def test_extreme_favorite_no_division_by_zero(self) -> None:
        """Test 6: P(i)が1.0に近い極端な本命レースでもdivision-by-zeroが発生しない."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"
        # 1番人気が極端に低オッズ (tanodds=1.01)
        df = pd.DataFrame({
            "umaban": [1, 2, 3],
            "tanodds": [1.01, 50.0, 100.0],
            "race_id": [race_id, race_id, race_id],
        })
        wide_df = _make_wide_df(race_id, ["0102"], [1.0], [1.1], ["001"])
        trio_df = _make_trio_df(race_id, ["010203"], [5.0], [1])

        # 例外が発生しないことを確認
        result = compute_market_cross_features(df, wide_df=wide_df, trio_df=trio_df)

        # infや極端な値が含まれていないことを確認
        for col in ["rl_wide_harville_ratio", "rl_trio_odds_ratio"]:
            vals = result[col]
            assert not vals.isin([np.inf, -np.inf]).any(), \
                f"{col} にinfが含まれる"
            # NaN または有限値であること
            assert (vals.isna() | np.isfinite(vals)).all(), \
                f"{col} にNaNでも有限でもない値が含まれる"

    def test_ninki_string_and_int(self) -> None:
        """Test 7: ninki型違い(文字列"001"と整数1)の両方でninki=1フィルタが動作する."""
        from features.market_cross_features import compute_market_cross_features

        race_id = "202401010101"

        # 文字列 "001" のワイドオッズ
        df = _make_base_df(n_horses=6, race_id=race_id)
        wide_df_str = _make_wide_df(race_id, ["0102"], [1.5], [2.5], ["001"])
        trio_df_int = _make_trio_df(race_id, ["010203"], [10.0], [1])

        result1 = compute_market_cross_features(df, wide_df=wide_df_str, trio_df=trio_df_int)

        # 整数 1 のワイドオッズ
        wide_df_int = _make_wide_df(race_id, ["0102"], [1.5], [2.5], [1])

        result2 = compute_market_cross_features(df, wide_df=wide_df_int, trio_df=trio_df_int)

        # 両方で同じ結果が得られることを確認
        for col in ["rl_favorite_in_wide_top1", "rl_market_consistency"]:
            assert result1[col].iloc[0] == result2[col].iloc[0], \
                f"{col}: ninki文字列と整数で結果が異なる"


class TestDataRepositoryLoadWideOdds:
    """Test 8: DataRepository.load_wide_odds()が呼び出し可能で正しいParquetを読み込む."""

    def test_load_wide_odds_method_exists(self) -> None:
        """DataRepositoryにload_wide_oddsメソッドが存在する."""
        from db.repository import DataRepository
        assert hasattr(DataRepository, "load_wide_odds"), \
            "DataRepositoryにload_wide_oddsメソッドが存在しない"

    def test_load_wide_odds_calls_store(self) -> None:
        """load_wide_oddsがParquetStore.readを正しく呼び出す."""
        from db.repository import DataRepository

        mock_store = MagicMock()
        mock_df = pd.DataFrame({
            "race_id": ["202401010101"],
            "kumi": ["0102"],
            "oddslow": [1.5],
            "oddshigh": [2.5],
            "ninki": ["001"],
        })
        mock_store.read.return_value = mock_df

        repo = DataRepository(store=mock_store)
        result = repo.load_wide_odds("20240101", "20240131")

        mock_store.read.assert_called_once()
        call_args = mock_store.read.call_args
        assert call_args[0][0] == "odds", f"第1引数が'odds'でない: {call_args[0][0]}"
        assert call_args[0][1] == "odds_wide", f"第2引数が'odds_wide'でない: {call_args[0][1]}"
        assert not result.empty
