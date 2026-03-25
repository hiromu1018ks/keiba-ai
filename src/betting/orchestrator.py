"""メインオーケストレーター + finalize_bets (設計書 §12)"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from domain.models import Bet, Race
from domain.types import BetType

logger = logging.getLogger(__name__)


@runtime_checkable
class StakeCalculatorProtocol(Protocol):
    def calc_stake(
        self, ev_lower: float, odds: float,
        bankroll: float, bet_type: BetType,
    ) -> float: ...
    def check_race_exposure(
        self, bets: list[Bet], bankroll: float,
    ) -> list[Bet]: ...


@runtime_checkable
class GateKeeperProtocol(Protocol):
    def filter_bets(
        self, bets: list[Bet], ev_threshold: float,
    ) -> list[Bet]: ...


@runtime_checkable
class MetaSwitcherProtocol(Protocol):
    def get_strategy_params(self) -> dict[str, object]: ...
    def should_retrain(self) -> bool: ...


@runtime_checkable
class BetStrategyProtocol(Protocol):
    def generate(
        self, feats: dict, bankroll: float,
        ev_threshold: float, max_bets: int = 3,
    ) -> list[Bet]: ...


@runtime_checkable
class WideStrategyProtocol(Protocol):
    def select_bets(
        self, scored_pairs: list[dict], ev_threshold: float,
        score_threshold: float, max_bets: int = 3,
    ) -> list[dict]: ...


@runtime_checkable
class LateMoneyFilterProtocol(Protocol):
    def process_last_minute(
        self, pending_bets: list[Bet],
        odds_t3_snapshot: dict[int, float],
        odds_t10_snapshot: dict[int, float],
        stage2_predictions: object,
    ) -> tuple[list[Bet], list[Bet]]: ...


@runtime_checkable
class QualityScreenerProtocol(Protocol):
    def should_bet(self, race_features: dict) -> bool: ...


@runtime_checkable
class DrawdownControllerProtocol(Protocol):
    def adjust_stake(
        self, base_stake: float, bankroll: float,
    ) -> float: ...


class BettingOrchestrator:
    """
    ベッティング決定の12ステップフローを統括する。

    設計書 §12 のオーケストレーターに対応。
    process_race() で ①〜⑩ を実行し、finalize_bets() で ⑫ t-3min
    直前キャンセルチェックを行う。

    注: モデル推論 (③〜⑦) は TrainingPipeline 側で実行され、
    その結果 (feats dict) を受け取る設計。Orchestrator はベッティング
    決定ロジックのみを担当する。
    """

    def __init__(
        self,
        stake_calculator: StakeCalculatorProtocol,
        gate_keeper: GateKeeperProtocol,
        meta_switcher: MetaSwitcherProtocol,
        place_strategy: BetStrategyProtocol,
        win_strategy: BetStrategyProtocol,
        wide_strategy: WideStrategyProtocol,
        late_money_filter: LateMoneyFilterProtocol,
        quality_screener: QualityScreenerProtocol,
    ) -> None:
        self.stake_calculator = stake_calculator
        self.gate_keeper = gate_keeper
        self.meta_switcher = meta_switcher
        self.place_strategy = place_strategy
        self.win_strategy = win_strategy
        self.wide_strategy = wide_strategy
        self.late_money_filter = late_money_filter
        self.quality_screener = quality_screener

    def process_race(
        self,
        race: Race,
        feats: dict,
        bankroll: float,
        dd_ctrl: DrawdownControllerProtocol,
    ) -> list[Bet]:
        """
        レースのベット候補を生成する（設計書 §12 ステップ ①〜⑩）。

        Args:
            race: レース情報
            feats: モデル推論済み特徴量dict（TrainingPipeline出力）
            bankroll: 現在の資金
            dd_ctrl: DrawdownControllerインスタンス

        Returns:
            pending_bets: 最終ベット候補リスト
        """
        # ② レジームパラメータ取得
        params = self.meta_switcher.get_strategy_params()
        ev_threshold = params["ev_threshold"]
        score_threshold = params["score_threshold"]
        max_bets = params["max_bets_per_race"]
        logger.info(f"Regime: {params.get('description', 'unknown')}")

        # 再学習トリガー確認
        if self.meta_switcher.should_retrain():
            logger.warning("COLLAPSED状態が連続 → 再学習をトリガー")
            self._trigger_retrain()

        # ⑦ RaceQualityScreener（レースレベル特徴量を抽出）
        race_features = self._build_race_features(feats)
        if not self.quality_screener.should_bet(race_features):
            logger.info(f"Skipping by QualityScreener: {race.race_id}")
            return []

        # ⑧ ベット候補生成
        place_bets = self.place_strategy.generate(
            feats, bankroll, ev_threshold, max_bets=max_bets,
        )
        win_bets = self.win_strategy.generate(
            feats, bankroll, ev_threshold, max_bets=max_bets,
        )
        wide_pairs = self.wide_strategy.select_bets(
            feats.get("wide_scored_pairs", []),
            ev_threshold,
            score_threshold,
            max_bets=max_bets,
        )
        # ワイドペアをBetに変換
        wide_bets = self._pairs_to_bets(wide_pairs)

        all_bets = place_bets + wide_bets + win_bets

        # GateKeeper: EV下限値で最終足切り
        all_bets = self.gate_keeper.filter_bets(all_bets, ev_threshold)

        # ⑨ 賭け金計算
        for bet in all_bets:
            base_stake = self.stake_calculator.calc_stake(
                bet.ev_lower_corrected, bet.odds, bankroll, bet.bet_type,
            )
            bet.stake = dd_ctrl.adjust_stake(base_stake, bankroll)

        # ⑩ 1レース露出キャップ（2%）
        all_bets = self.stake_calculator.check_race_exposure(all_bets, bankroll)

        # ⑪ SafetyGuard（Phase F で実装。この段階ではスキップ）
        # if not self.safety_guard.check(bankroll).can_bet:
        #     return []

        # 最小投票額フィルタ
        pending_bets = [b for b in all_bets if b.stake >= 100]

        return pending_bets

    def finalize_bets(
        self,
        race: Race,
        pending_bets: list[Bet],
        odds_t3_snapshot: dict[int, float],
        odds_t10_snapshot: dict[int, float],
    ) -> list[Bet]:
        """
        発走3分前に実行。t-3min オッズで最終キャンセルチェック。

        設計書 §12 ステップ ⑫。
        """
        approved, cancelled = self.late_money_filter.process_last_minute(
            pending_bets=pending_bets,
            odds_t3_snapshot=odds_t3_snapshot,
            odds_t10_snapshot=odds_t10_snapshot,
            stage2_predictions=None,
        )

        if cancelled:
            logger.info(
                f"[{race.race_id}] {len(cancelled)} bets cancelled by t-3min trigger"
            )

        return approved

    def _build_race_features(self, feats: dict) -> dict:
        """レースレベル特徴量を構築（QualityScreener用）"""
        # feats dictからrace_id単位で集計
        return {
            "race_id": feats.get("race_id", ["unknown"])[0],
            "field_size": len(feats.get("umaban", [])),
        }

    def _pairs_to_bets(self, pairs: list[dict]) -> list[Bet]:
        """ワイドペアdictをBetリストに変換"""
        bets: list[Bet] = []
        for pair in pairs:
            bets.append(
                Bet(
                    race_id=pair["race_id"],
                    umaban=pair["umaban_a"],  # 代表馬番
                    bet_type=BetType.WIDE,
                    odds=pair["wide_odds"],
                    ev_lower_corrected=pair["ev_wide"],
                    stake=0.0,
                )
            )
        return bets

    def _trigger_retrain(self) -> None:
        """再学習トリガー（Phase E/F で実装）"""
        logger.warning("Retrain trigger called (not yet implemented)")
