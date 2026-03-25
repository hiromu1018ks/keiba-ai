"""複勝戦略"""

from __future__ import annotations

import math

from domain.models import Bet
from domain.types import BetType


class PlaceStrategy:
    """
    複勝ベット候補を生成する。

    2段階モデル出力 (p_place × e_return_place = ev_place) の
    信頼区間下限値を基準に候補を抽出し、簡易ケリー基準で賭け金を算出する。

    stake計算は StakeCalculator と同様のロジック:
      edge = ev_lower - 1.0
      kelly_fraction = edge / (odds - 1.0)
      raw_stake = bankroll × kelly_fraction
      stake = floor(raw_stake / 100) × 100  (100円単位)
    """

    MIN_STAKE: int = 100  # 最低投票額
    KELLY_FRACTION_CAP: float = 0.25  # Kelly fraction の最大値

    def generate(
        self,
        feats: dict,
        bankroll: float,
        ev_threshold: float,
        max_bets: int = 3,
    ) -> list[Bet]:
        """
        複勝ベット候補を生成する。

        Args:
            feats: 特徴量dict（DataFrame互換、キーでアクセス）
            bankroll: 現在の資金
            ev_threshold: EV下限閾値
            max_bets: 1レースあたりの最大ベット数

        Returns:
            Bet リスト
        """
        ev_lower_list = feats["ev_lower_place"]
        odds_list = feats["place_odds"]
        race_ids = feats["race_id"]
        umabans = feats["umaban"]

        candidates: list[tuple[float, float, str, int]] = []
        for i, ev_lower in enumerate(ev_lower_list):
            if ev_lower >= ev_threshold:
                candidates.append((ev_lower, odds_list[i], race_ids[i], umabans[i]))

        # EV降順でソート → 上位 max_bets 件
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_bets]

        bets: list[Bet] = []
        for ev_lower, odds, race_id, umaban in candidates:
            stake = self._calc_stake(ev_lower, odds, bankroll)
            bets.append(
                Bet(
                    race_id=race_id,
                    umaban=umaban,
                    bet_type=BetType.PLACE,
                    odds=odds,
                    ev_lower_corrected=ev_lower,
                    stake=stake,
                )
            )

        return bets

    def _calc_stake(self, ev_lower: float, odds: float, bankroll: float) -> float:
        """簡易ケリー基準で賭け金を計算する（100円単位）。"""
        if bankroll <= 0 or odds <= 1.0:
            return 0.0

        edge = ev_lower - 1.0
        kelly_fraction = edge / (odds - 1.0)
        kelly_fraction = min(kelly_fraction, self.KELLY_FRACTION_CAP)

        raw_stake = bankroll * kelly_fraction
        stake = max(0, int(math.floor(raw_stake / self.MIN_STAKE)) * self.MIN_STAKE)
        return float(stake)
