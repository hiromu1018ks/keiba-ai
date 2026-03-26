"""バックテストエンジン (§13)

学習済みモデルで履歴データをシミュレーションし、投資成績を評価。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from db.connection import DatabaseConnection
from domain.models import Bet, BetType

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """バックテスト結果"""

    total_bets: int = 0
    total_stake: float = 0.0
    total_return: float = 0.0
    winning_bets: int = 0
    total_roi: float = 0.0
    max_drawdown: float = 0.0
    final_bankroll: float = 0.0
    monthly_returns: dict[str, float] = field(default_factory=dict)
    bet_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def profit(self) -> float:
        return self.total_return - self.total_stake

    def summary(self) -> str:
        return (
            f"Backtest Result:\n"
            f"  Bets: {self.total_bets}\n"
            f"  Stake: ¥{self.total_stake:,.0f}\n"
            f"  Return: ¥{self.total_return:,.0f}\n"
            f"  ROI: {self.total_roi:.3%}\n"
            f"  Max DD: {self.max_drawdown:.3%}\n"
            f"  Final Bankroll: ¥{self.final_bankroll:,.0f}"
        )


class BacktestEngine:
    """バックテストエンジン

    TrainedModelsV5 を使用して、指定期間のレースをシミュレーション。

    Args:
        models: 学習済みモデル
        initial_bankroll: 初期資金 (デフォルト 100,000円)
        db: データベース接続 (省略時は新規接続)
    """

    def __init__(
        self,
        models: TrainedModelsV5,
        initial_bankroll: float = 100_000,
        db: DatabaseConnection | None = None,
    ) -> None:
        self.models = models
        self.initial_bankroll = initial_bankroll
        self.db = db or DatabaseConnection()

    def run(
        self,
        test_start: str,
        test_end: str,
    ) -> BacktestResult:
        """バックテストを実行

        Args:
            test_start: テスト開始日 (YYYY-MM-DD)
            test_end: テスト終了日 (YYYY-MM-DD)

        Returns:
            BacktestResult
        """
        # 1. データロード
        race_df = self.db.load_races(test_start, test_end)
        entry_df = self.db.load_entries_with_results(test_start, test_end)
        odds_df = self.db.load_odds_snapshots(test_start, test_end)

        if race_df.empty:
            logger.warning(f"No races found in {test_start} ~ {test_end}")
            return BacktestResult(final_bankroll=self.initial_bankroll)

        # 2. 特徴量生成
        from features.feature_engine import FeatureEngine
        from models.submodel_manager import SubModelManager

        feat_engine = FeatureEngine()
        submodel_mgr = SubModelManager()
        feat_df = feat_engine.build_all(race_df, entry_df, odds_df)
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # 3. レースごとにシミュレーション
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_dd = 0.0
        bet_history: list[dict[str, Any]] = []
        monthly_returns: dict[str, float] = {}

        race_ids = feat_df["race_id"].unique()

        for race_id in race_ids:
            race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
            if race_df_single.empty:
                continue

            # 3a. サブモデル選択
            surface_key = race_df_single["surface_key"].iloc[0]
            if surface_key not in self.models.submodels:
                continue
            submodel = self.models.submodels[surface_key]

            # 3b. レジーム検知
            regime = self.models.regime_detector.current_regime
            regime_params = self.models.regime_detector.get_strategy_params(regime)

            # 3c. 特徴量 → 予測
            try:
                race_df_single = submodel.market.predict_and_calc_error(race_df_single)
            except Exception as e:
                logger.debug("Skipping race %s: market prediction failed: %s", race_id, e)
                continue
            race_df_single = submodel.stage1.add_ability_probs(race_df_single)
            race_df_single = submodel.win.predict_ev(race_df_single)
            race_df_single = submodel.ev_corrector.correct_ev(race_df_single)
            race_df_single = submodel.place.predict_ev(race_df_single)

            # 信頼区間
            win_df, place_df = submodel.confidence.predict_lower_bound(
                race_df_single, race_df_single
            )
            race_df_single = win_df

            # 3d. RaceQualityScreener
            race_features = self._build_race_features(race_df_single)
            if not self.models.quality_screener.should_bet(race_features):
                continue

            # 3e. ベット生成
            bets = self._generate_bets(race_df_single, bankroll, regime_params)

            # 3f. 結果判定
            for bet in bets:
                bet_result = self._settle_bet(bet, race_df_single)
                bet_history.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet.bet_type.value,
                        "umaban": bet.umaban,
                        "stake": bet.stake,
                        "odds": bet.odds,
                        "result": bet_result,
                    }
                )

                bankroll -= bet.stake
                if bet_result > 0:
                    bankroll += bet_result

                # DD 追跡
                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_dd = max(max_dd, dd)

        # 4. ROI 計算
        total_stake = sum(b["stake"] for b in bet_history)
        total_return = sum(b["result"] for b in bet_history if b["result"] > 0)
        total_bets = len(bet_history)
        winning_bets = sum(1 for b in bet_history if b["result"] > 0)

        return BacktestResult(
            total_bets=total_bets,
            total_stake=total_stake,
            total_return=total_return,
            winning_bets=winning_bets,
            total_roi=total_return / total_stake if total_stake > 0 else 0.0,
            max_drawdown=max_dd,
            final_bankroll=bankroll,
            monthly_returns=monthly_returns,
            bet_history=bet_history,
        )

    def _build_race_features(self, race_df: pd.DataFrame) -> dict[str, Any]:
        """レースレベル特徴量を dict に変換 (QualityScreener 用)

        RaceQualityScreener.FEATURE_COLS (20列) に対応。
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
        }

    def _generate_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
        regime_params: dict[str, Any],
    ) -> list[Bet]:
        """簡易ベット生成 (EV条件を満たす馬にベット)"""
        bets: list[Bet] = []
        ev_threshold = regime_params.get("ev_threshold", 1.20)

        # 複勝ベット
        if "ev_place" in race_df.columns and "place_odds_actual" in race_df.columns:
            for _, row in race_df.iterrows():
                if row.get("ev_place", 0) >= ev_threshold:
                    stake = 100.0  # 固定100円ベット (簡易版)
                    if bankroll >= stake:
                        bets.append(
                            Bet(
                                race_id=row["race_id"],
                                umaban=int(row["umaban"]),
                                bet_type=BetType.PLACE,
                                odds=float(row["place_odds_actual"]),
                                ev_lower_corrected=float(row.get("ev_place", 0)),
                                stake=stake,
                            )
                        )

        return bets

    def _settle_bet(self, bet: Bet, race_df: pd.DataFrame) -> float:
        """ベットの結果を判定"""
        horse = race_df[race_df["umaban"] == bet.umaban]
        if horse.empty:
            return 0.0

        finish_pos = int(horse.iloc[0]["finish_pos"])

        if bet.bet_type == BetType.PLACE:
            if 1 <= finish_pos <= 3:
                return float(bet.stake * bet.odds)
        elif bet.bet_type == BetType.WIN:
            if finish_pos == 1:
                return float(bet.stake * bet.odds)
        elif bet.bet_type == BetType.WIDE:
            if 1 <= finish_pos <= 3:
                pair_b = getattr(bet, "umaban_b", None)
                if pair_b is not None:
                    pair_horse = race_df[race_df["umaban"] == pair_b]
                    if not pair_horse.empty and int(pair_horse.iloc[0]["finish_pos"]) <= 3:
                        return float(bet.stake * bet.odds)

        return 0.0
