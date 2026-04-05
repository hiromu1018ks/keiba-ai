"""バックテストエンジン (§13)

学習済みモデルで履歴データをシミュレーションし、投資成績を評価。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtest.diagnostic_logger import DiagnosticLogger
from backtest.race_predictor import RacePredictor
from db.parquet_store import ParquetStore
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races
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
        store: ParquetStore (省略時は新規インスタンス)
    """

    def __init__(
        self,
        models: TrainedModelsV5,
        initial_bankroll: float = 100_000,
        store: ParquetStore | None = None,
    ) -> None:
        self.models = models
        self.initial_bankroll = initial_bankroll
        self.store = store or ParquetStore()
        self._race_predictor = RacePredictor(models)

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
        start = test_start.replace("-", "")
        end = test_end.replace("-", "")
        race_df = load_races(self.store, start, end)
        entry_df = load_entries(self.store, start, end)
        odds_df = load_odds_snapshots(self.store, start, end)

        if race_df.empty:
            logger.warning(f"No races found in {test_start} ~ {test_end}")
            return BacktestResult(final_bankroll=self.initial_bankroll)

        # 2. 特徴量生成
        from features.feature_engine import FeatureEngine
        from models.submodel_manager import SubModelManager

        feat_engine = FeatureEngine()
        submodel_mgr = SubModelManager()
        odds_ts_df = load_odds_time_series_range(self.store, start, end)
        feat_df = feat_engine.build_all(
            race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=self.store
        )
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # 3. 特徴量の一括事前計算 (ループ外で全レース分を一度に計算)
        from features.horse_history_features import HorseHistoryFeatures
        from features.jockey_context_features import JockeyContextFeatures
        from features.trainer_context_features import TrainerContextFeatures

        race_ids = feat_df["race_id"].unique()

        logger.info("Pre-computing HorseHistoryFeatures for %d races...", len(race_ids))
        hist_all = HorseHistoryFeatures(store=self.store)
        hist_df_all = hist_all.compute(race_df, entry_df, race_ids)

        logger.info("Pre-computing JockeyContextFeatures for %d entries...", len(entry_df))
        jockey_ctx = JockeyContextFeatures(self.store)
        jockey_df_all = jockey_ctx.compute(entry_df)

        logger.info("Pre-computing TrainerContextFeatures for %d entries...", len(entry_df))
        trainer_ctx = TrainerContextFeatures(self.store)
        trainer_df_all = trainer_ctx.compute(entry_df)

        # 4. レースごとにシミュレーション (推論は RacePredictor に委譲)
        diag_logger = DiagnosticLogger()
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_dd = 0.0
        bet_history: list[dict[str, Any]] = []
        monthly_returns: dict[str, float] = {}

        for race_id in race_ids:
            race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
            if race_df_single.empty:
                continue

            # 事前計算済み特徴量をマージ
            hist_df_race = hist_df_all[hist_df_all["race_id"] == race_id]
            jockey_df_race = jockey_df_all[jockey_df_all["race_id"] == race_id]
            trainer_df_race = trainer_df_all[trainer_df_all["race_id"] == race_id]

            # RacePredictor に委譲
            result_df = self._race_predictor.predict(
                race_df_single,
                hist_features=hist_df_race,
                jockey_features=jockey_df_race,
                trainer_features=trainer_df_race,
            )
            if result_df.empty:
                continue

            # Quality screening
            regime = self.models.regime_detector.current_regime
            regime_params = self.models.regime_detector.get_strategy_params(regime)
            ev_threshold = regime_params.get("ev_threshold", 1.10)
            n_candidates = (
                int((result_df["ev_place"].fillna(0) >= ev_threshold).sum())
                if "ev_place" in result_df.columns
                else 0
            )

            if not self._race_predictor.should_bet(result_df):
                diag_logger.log_race(
                    race_id=race_id,
                    regime=str(regime),
                    ev_threshold=ev_threshold,
                    quality_passed=False,
                    quality_score=0.0,
                    n_candidates=n_candidates,
                    n_bets=0,
                )
                if "ev_place" in result_df.columns:
                    for _, hr in result_df.iterrows():
                        diag_logger.log_horse(
                            race_id=race_id,
                            umaban=int(hr["umaban"]),
                            p_place_pred=float(hr.get("p_place_pred", 0)),
                            e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                            ev_place=float(hr.get("ev_place", 0)),
                            fukuoddslow=float(hr.get("fukuoddslow", 0)),
                            is_bet=False,
                        )
                continue

            # Bet generation
            surface_key = result_df["surface"].iloc[0]
            bets = self._race_predictor.select_bets(result_df, bankroll)

            # Log diagnostics for quality-passed race
            diag_logger.log_race(
                race_id=race_id,
                regime=str(regime),
                ev_threshold=ev_threshold,
                quality_passed=True,
                quality_score=0.0,
                n_candidates=n_candidates,
                n_bets=len(bets),
            )
            if "ev_place" in result_df.columns:
                bet_umabans = {b.umaban for b in bets}
                for _, hr in result_df.iterrows():
                    diag_logger.log_horse(
                        race_id=race_id,
                        umaban=int(hr["umaban"]),
                        p_place_pred=float(hr.get("p_place_pred", 0)),
                        e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                        ev_place=float(hr.get("ev_place", 0)),
                        fukuoddslow=float(hr.get("fukuoddslow", 0)),
                        is_bet=int(hr["umaban"]) in bet_umabans,
                    )

            # Settlement (BacktestEngine 固有)
            for bet in bets:
                bet_result = self._settle_bet(bet, result_df)
                bankroll -= bet.stake
                if bet_result > 0:
                    bankroll += bet_result

                horse_rows = result_df[result_df["umaban"] == bet.umaban]
                pop_val = (
                    horse_rows["popularity_rank"].iloc[0]
                    if not horse_rows.empty and "popularity_rank" in horse_rows.columns
                    else 0
                )

                bet_history.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet.bet_type.value,
                        "umaban": bet.umaban,
                        "stake": bet.stake,
                        "odds": bet.odds,
                        "result": bet_result,
                        "surface": surface_key,
                        "kyori": (
                            int(result_df["kyori"].iloc[0]) if "kyori" in result_df.columns else 0
                        ),
                        "ev": float(bet.ev_lower_corrected),
                        "popularity": int(pop_val) if pd.notna(pop_val) else 0,
                        "bankroll_after": round(bankroll, 2),
                    }
                )

                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_dd = max(max_dd, dd)

        # 5. 診断ログ保存
        diag_logger.save(Path("data/backtest"), prefix="bt")

        # 6. ROI 計算
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

        NOTE: 現在は RacePredictor.build_race_features() に委譲されている。
        互換性のため残している (テスト等で参照される可能性がある)。

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
        """簡易ベット生成 (EV条件を満たす馬にベット)

        NOTE: 現在は RacePredictor.select_bets() に委譲されている。
        互換性のため残している (テスト等で参照される可能性がある)。
        """
        bets: list[Bet] = []
        ev_threshold = regime_params.get("ev_threshold", 1.10)
        max_bets = regime_params.get("max_bets_per_race", 3)

        # 複勝ベット
        if "ev_place" in race_df.columns and "fukuoddslow" in race_df.columns:
            candidates = race_df[race_df["ev_place"].fillna(0) >= ev_threshold].copy()
            # ev_place 降順でソートし、上位 max_bets 頭のみベット
            candidates = candidates.nlargest(max_bets, "ev_place")

            for _, row in candidates.iterrows():
                stake = 100.0  # 固定100円ベット (簡易版)
                if bankroll >= stake:
                    bets.append(
                        Bet(
                            race_id=row["race_id"],
                            umaban=int(row["umaban"]),
                            bet_type=BetType.PLACE,
                            odds=float(row["fukuoddslow"]),
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

        finish_pos = int(horse.iloc[0]["kakuteijyuni"])

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
                    if not pair_horse.empty and int(pair_horse.iloc[0]["kakuteijyuni"]) <= 3:
                        return float(bet.stake * bet.odds)

        return 0.0
