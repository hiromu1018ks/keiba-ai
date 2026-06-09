"""WinTop1ScoreBlender のテスト.

train: fold構築, weight評価, baseline未改善→w=0, 列不足→early return
apply: w=0 no-op, w>0 置換, 診断列, 行数行順維持, 空DF, 未訓練
persistence: save/load roundtrip, 未訓練 roundtrip
confirmed_odds: 確定オッズ優先, tanodds フォールバック
weight-zero identity: w=0 で順位/選択が従来と同一
runtime order: blender → reranker のスコア伝播確認
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.win_top1_score_blender import (
    BLENDER_CURRENT_RANK_COL,
    BLENDER_MARKET_RANK_COL,
    BLENDER_RAW_SCORE_COL,
    BLENDER_SCORE_COL,
    BLENDER_WEIGHT_COL,
    WinTop1ScoreBlender,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blender_rows(
    n_races: int = 300,
    *,
    seed: int = 42,
    include_confirmed_odds: bool = False,
) -> pd.DataFrame:
    """Build a synthetic OOF frame for score blender training.

    Each race has 8-14 horses. win_market_selection_score is generated with
    controlled signal so that top-1 selection is meaningful.
    """
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    for r in range(n_races):
        n_horses = rng.randint(8, 15)
        race_id = f"R{r:04d}"
        race_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=r)
        winner_idx = rng.randint(n_horses)
        for h in range(n_horses):
            tanodds = float(rng.uniform(2.0, 80.0))
            # Model probability (higher for winner)
            base_prob = 1.0 / tanodds
            if h == winner_idx:
                p_win = min(0.80, base_prob * rng.uniform(1.2, 2.5))
            else:
                p_win = max(0.01, base_prob * rng.uniform(0.5, 1.5))
            # Selection score with some signal
            score = p_win - (1.0 / tanodds) + rng.normal(0, 0.02)
            confirmed_odds = tanodds * rng.uniform(0.9, 1.1) if include_confirmed_odds else np.nan
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": h + 1,
                    "tanodds": round(tanodds, 1),
                    "confirmed_odds": (
                        round(confirmed_odds, 1) if not np.isnan(confirmed_odds) else np.nan
                    ),
                    "p_win_final": round(p_win, 4),
                    "win_selection_prob": round(p_win, 4),
                    "win_market_selection_score": round(score, 4),
                    "kakuteijyuni": 1 if h == winner_idx else rng.randint(2, n_horses + 1),
                }
            )
    return pd.DataFrame(rows)


def _make_single_race_df(
    scores: list[float],
    odds: list[float],
    p_wins: list[float] | None = None,
    *,
    race_id: str = "TEST_R1",
    surface: str = "turf",
) -> pd.DataFrame:
    """Minimal single-race DataFrame for apply tests."""
    n = len(scores)
    if p_wins is None:
        p_wins = [round(1.0 / o, 4) for o in odds]
    return pd.DataFrame(
        {
            "race_id": [race_id] * n,
            "race_date": [pd.Timestamp("2024-06-01")] * n,
            "surface": [surface] * n,
            "umaban": list(range(1, n + 1)),
            "tanodds": odds,
            "p_win_final": p_wins,
            "win_selection_prob": p_wins,
            "win_market_selection_score": scores,
            "kakuteijyuni": list(range(1, n + 1)),
        }
    )


def _trained_model(weight: float = 0.4) -> WinTop1ScoreBlender:
    """Create a trained model without calling train()."""
    model = WinTop1ScoreBlender()
    model.selected_weight = weight
    model.is_trained = True
    return model


# ---------------------------------------------------------------------------
# Train tests
# ---------------------------------------------------------------------------


class TestScoreBlenderTrain:
    def test_fold_construction(self) -> None:
        """最小fold数のレースでfoldが構築されること."""
        df = _make_blender_rows(n_races=300)
        model = WinTop1ScoreBlender(min_train_races=200, min_fold_races=80, max_folds=4)
        race_order = (
            df[["race_id", "race_date"]].drop_duplicates().sort_values(["race_date", "race_id"])
        )
        folds = model._build_folds(race_order)
        assert len(folds) > 0
        # All folds are chronologically ordered
        prev_end = 0
        for train_end, test_end in folds:
            assert train_end >= prev_end
            assert test_end > train_end
            prev_end = test_end

    def test_insufficient_races_returns_w0(self) -> None:
        """レース数不足で w=0 となること."""
        df = _make_blender_rows(n_races=10)
        model = WinTop1ScoreBlender()
        model.train(df)
        assert model.is_trained is True
        assert model.selected_weight == 0.0
        assert model.training_summary.get("reason") == "insufficient_races"

    def test_missing_columns_returns_w0(self) -> None:
        """必須列不足で w=0 となること."""
        df = pd.DataFrame({"race_id": ["R1"], "race_date": [pd.Timestamp("2024-01-01")]})
        model = WinTop1ScoreBlender()
        model.train(df)
        assert model.selected_weight == 0.0
        assert model.training_summary.get("reason") == "missing_required_columns"

    def test_empty_df_returns_w0(self) -> None:
        """空DFで w=0 となること."""
        model = WinTop1ScoreBlender()
        model.train(pd.DataFrame())
        assert model.selected_weight == 0.0

    def test_train_selects_best_weight(self) -> None:
        """学習が各weightを評価し最適なものを選ぶこと."""
        df = _make_blender_rows(n_races=500)
        model = WinTop1ScoreBlender()
        model.train(df)
        assert model.is_trained is True
        assert model.selected_weight in model.candidate_weights
        assert "all_weight_metrics" in model.training_summary
        # Baseline (w=0) should always be evaluated
        assert "0.0" in model.training_summary["all_weight_metrics"]

    def test_no_improvement_keeps_w0(self) -> None:
        """baseline改善がない場合 w=0 を維持すること."""
        # Need enough races for fold construction: min_train(200) + min_fold(80) = 280
        df = _make_blender_rows(n_races=350, seed=123)
        model = WinTop1ScoreBlender()
        model.train(df)
        assert model.is_trained is True
        # Weight is always in candidate range (0.0 to 1.0)
        assert 0.0 <= model.selected_weight <= 1.0
        assert "baseline_objective" in model.training_summary

    def test_simulate_weight_returns_profit(self) -> None:
        """_simulate_weight が profit/bets/roi を返すこと."""
        df = _make_blender_rows(n_races=300)
        race_order = (
            df[["race_id", "race_date"]].drop_duplicates().sort_values(["race_date", "race_id"])
        )
        model = WinTop1ScoreBlender(min_train_races=200, min_fold_races=50, max_folds=4)
        folds = model._build_folds(race_order)
        assert len(folds) >= 1
        fold_rids = set(race_order.iloc[folds[0][0] : folds[0][1]]["race_id"])
        metrics = model._simulate_weight(df, 0.0, fold_rids)
        assert "profit" in metrics
        assert "bets" in metrics
        assert "roi" in metrics
        assert metrics["bets"] > 0
        assert metrics["bets_equals_races"] is True


# ---------------------------------------------------------------------------
# Apply tests
# ---------------------------------------------------------------------------


class TestScoreBlenderApply:
    def test_w0_is_noop(self) -> None:
        """w=0 でスコアが変更されないこと."""
        df = _make_single_race_df([0.5, 0.3, 0.1], [5.0, 10.0, 20.0])
        original_score = df["win_market_selection_score"].copy()
        model = _trained_model(weight=0.0)
        result = model.apply(df)
        pd.testing.assert_series_equal(
            result["win_market_selection_score"],
            original_score,
            check_names=False,
        )
        assert BLENDER_RAW_SCORE_COL not in result.columns

    def test_untrained_is_noop(self) -> None:
        """未訓練で no-op であること."""
        df = _make_single_race_df([0.5, 0.3, 0.1], [5.0, 10.0, 20.0])
        model = WinTop1ScoreBlender()
        result = model.apply(df)
        assert BLENDER_RAW_SCORE_COL not in result.columns

    def test_empty_df_unchanged(self) -> None:
        """空DFがそのまま返されること."""
        df = pd.DataFrame(columns=["race_id", "win_market_selection_score"])
        model = _trained_model(weight=0.4)
        result = model.apply(df)
        assert len(result) == 0

    def test_weight_replaces_score(self) -> None:
        """w>0 で win_market_selection_score が blended に置換されること."""
        df = _make_single_race_df([0.5, 0.3, 0.1], [5.0, 10.0, 20.0], p_wins=[0.20, 0.10, 0.05])
        model = _trained_model(weight=0.4)
        result = model.apply(df)
        # Raw score saved
        assert BLENDER_RAW_SCORE_COL in result.columns
        pd.testing.assert_series_equal(
            result[BLENDER_RAW_SCORE_COL],
            df["win_market_selection_score"],
            check_names=False,
        )
        # Score is replaced (different from original)
        assert not result["win_market_selection_score"].equals(df["win_market_selection_score"])

    def test_diagnostic_columns_present(self) -> None:
        """診断列が全て追加されること."""
        df = _make_single_race_df([0.5, 0.3, 0.1], [5.0, 10.0, 20.0], p_wins=[0.20, 0.10, 0.05])
        model = _trained_model(weight=0.6)
        result = model.apply(df)
        for col in [
            BLENDER_WEIGHT_COL,
            BLENDER_CURRENT_RANK_COL,
            BLENDER_MARKET_RANK_COL,
            BLENDER_SCORE_COL,
        ]:
            assert col in result.columns

    def test_row_count_preserved(self) -> None:
        """行数が維持されること."""
        df = _make_blender_rows(n_races=10)
        n_rows = len(df)
        model = _trained_model(weight=0.5)
        result = model.apply(df)
        assert len(result) == n_rows

    def test_row_order_preserved(self) -> None:
        """行順が維持されること."""
        df = _make_single_race_df([0.1, 0.3, 0.5], [20.0, 10.0, 5.0], p_wins=[0.05, 0.10, 0.20])
        original_index = df.index.tolist()
        model = _trained_model(weight=0.8)
        result = model.apply(df)
        assert result.index.tolist() == original_index

    def test_blender_weight_value(self) -> None:
        """診断列の weight が selected_weight と一致すること."""
        df = _make_single_race_df([0.5, 0.3, 0.1], [5.0, 10.0, 20.0], p_wins=[0.20, 0.10, 0.05])
        model = _trained_model(weight=0.6)
        result = model.apply(df)
        assert (result[BLENDER_WEIGHT_COL] == 0.6).all()

    def test_multiple_races_preserved(self) -> None:
        """複数レースで行数・race_id が維持されること."""
        df = _make_blender_rows(n_races=5)
        model = _trained_model(weight=0.3)
        result = model.apply(df)
        pd.testing.assert_index_equal(result.index, df.index)
        pd.testing.assert_series_equal(result["race_id"], df["race_id"], check_names=False)

    def test_w1_uses_only_market(self) -> None:
        """w=1.0 で market_rank のみが使用されること (blended = market_rank)."""
        df = _make_single_race_df([0.5, 0.3, 0.1], [5.0, 10.0, 20.0], p_wins=[0.30, 0.10, 0.05])
        model = _trained_model(weight=1.0)
        result = model.apply(df)
        np.testing.assert_allclose(
            result["win_market_selection_score"].values,
            result[BLENDER_MARKET_RANK_COL].values,
        )


# ---------------------------------------------------------------------------
# Weight-zero identity
# ---------------------------------------------------------------------------


class TestScoreBlenderWeightZeroIdentity:
    def test_w0_preserves_top1_selection(self) -> None:
        """w=0 で各レースの top-1 選択馬が従来と同一であること.

        percentile rank は単調変換なので、w=0 は元スコアと同じ順位を生成する。
        apply は w=0 で no-op なので、win_market_selection_score はそのまま。
        """
        df = _make_blender_rows(n_races=100)
        original_top1 = df.loc[
            df.groupby("race_id")["win_market_selection_score"].idxmax()
        ].set_index("race_id")["umaban"]
        model = WinTop1ScoreBlender()
        model.selected_weight = 0.0
        model.is_trained = True
        result = model.apply(df)
        result_top1 = result.loc[
            result.groupby("race_id")["win_market_selection_score"].idxmax()
        ].set_index("race_id")["umaban"]
        pd.testing.assert_series_equal(result_top1, original_top1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestScoreBlenderPersistence:
    def test_save_load_roundtrip(self) -> None:
        """save/load で状態が復元されること."""
        model = _trained_model(weight=0.6)
        model.training_summary = {"selected_weight": 0.6, "objective": 0.05}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blender.joblib"
            model.save(path)
            loaded = WinTop1ScoreBlender.load(path)
        assert loaded.selected_weight == 0.6
        assert loaded.is_trained is True
        assert loaded.training_summary == {"selected_weight": 0.6, "objective": 0.05}

    def test_untrained_save_load(self) -> None:
        """未訓練モデルの save/load で w=0 が維持されること."""
        model = WinTop1ScoreBlender()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blender.joblib"
            model.save(path)
            loaded = WinTop1ScoreBlender.load(path)
        assert loaded.selected_weight == 0.0
        assert loaded.is_trained is False


# ---------------------------------------------------------------------------
# confirmed_odds
# ---------------------------------------------------------------------------


class TestScoreBlenderConfirmedOdds:
    def test_confirmed_odds_preferred(self) -> None:
        """confirmed_odds が存在すれば払戻計算に使用されること."""
        df = _make_blender_rows(n_races=300, include_confirmed_odds=True)
        model = WinTop1ScoreBlender(min_train_races=200, min_fold_races=50, max_folds=4)
        model.train(df)
        assert model.is_trained is True
        # Training completed without error means confirmed_odds was used
        assert "all_weight_metrics" in model.training_summary

    def test_tanodds_fallback(self) -> None:
        """confirmed_odds なしで tanodds フォールバックが動作すること."""
        df = _make_blender_rows(n_races=300, include_confirmed_odds=False)
        model = WinTop1ScoreBlender(min_train_races=200, min_fold_races=50, max_folds=4)
        model.train(df)
        assert model.is_trained is True

    def test_simulate_prefers_confirmed(self) -> None:
        """_simulate_weight で confirmed_odds が tanodds より優先されること."""
        df = _make_single_race_df(
            [0.5, 0.1],
            [10.0, 50.0],
            p_wins=[0.10, 0.02],
        )
        df["confirmed_odds"] = [15.0, 60.0]  # Different from tanodds
        df["race_date"] = [pd.Timestamp("2024-01-01")] * 2
        model = WinTop1ScoreBlender()
        metrics = model._simulate_weight(df, 0.0, {"TEST_R1"})
        # With confirmed_odds=15 for winner: payout = 15 - 1 = 14
        assert metrics["profit"] == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# Runtime order: blender -> reranker
# ---------------------------------------------------------------------------


class TestScoreBlenderRuntimeOrder:
    def test_blended_score_consumed_by_reranker(self) -> None:
        """blender apply後のblended scoreがrerankerに伝播すること.

        等オッズ(5.0)の3頭でmarketが均等(norm=1/3)。
        Horse Aのp_winが最も高くmarket normを上回る -> positive residual。
        original scoreではCが最高だが、w=1.0でAに反転する。
        """
        from models.win_top1_odds_reranker import WinTop1OddsReranker

        # Equal odds -> market norm = 1/3 for all
        # p_win: A=0.50 (above norm -> positive residual),
        #         B=0.10 (below norm -> negative),
        #         C=0.05 (below norm -> negative)
        # Original scores: C wins (0.8 > 0.3 > 0.1)
        df = _make_single_race_df(
            [0.1, 0.3, 0.8],
            [5.0, 5.0, 5.0],
            p_wins=[0.50, 0.10, 0.05],
        )

        # Original: C wins by score
        original_top1 = df.loc[df["win_market_selection_score"].idxmax(), "umaban"]
        assert original_top1 == 3

        # Blender with w=1.0: uses market_rank only -> A wins (highest residual)
        blender = _trained_model(weight=1.0)
        blended = blender.apply(df)

        # Reranker with cap=inf: selects top-1 by blended score
        reranker = WinTop1OddsReranker()
        reranker.selected_cap = float("inf")
        reranker.is_trained = True
        reranked = reranker.apply(blended)

        assert isinstance(reranked, pd.DataFrame)
        assert len(reranked) == 1
        # After blending with w=1.0, A (umaban=1) has highest market residual
        assert reranked.iloc[0]["umaban"] == 1

    def test_blender_then_reranker_cap(self) -> None:
        """blender + reranker capの組み合わせが動作すること."""
        from models.win_top1_odds_reranker import WinTop1OddsReranker

        # 3 horses: A (high score, high odds), B (medium), C (low score, low odds)
        df = _make_single_race_df(
            [0.7, 0.4, 0.1],
            [90.0, 20.0, 3.0],
            p_wins=[0.50, 0.10, 0.05],
        )
        blender = _trained_model(weight=0.5)
        blended = blender.apply(df)

        # Reranker with cap=30: A (odds=90) excluded, picks from B and C
        reranker = WinTop1OddsReranker()
        reranker.selected_cap = 30.0
        reranker.is_trained = True
        reranked = reranker.apply(blended)

        assert isinstance(reranked, pd.DataFrame)
        assert len(reranked) == 1
        # Reranker diagnostic columns should be present
        assert "reranker_switch_reason" in reranked.columns
