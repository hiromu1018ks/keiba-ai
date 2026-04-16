"""複勝戦略"""

from __future__ import annotations

import math

from domain.models import Bet
from domain.types import BetType


class PlaceStrategy:
    """
    複勝ベット候補を生成する。

    Value Betting (edge) を基準に候補を抽出し、
    簡易ケリー基準で賭け金を算出する。

    stake計算は StakeCalculator と同様のロジック:
      kelly_fraction = edge / (odds - 1.0)
      ただし edge = p_model * odds - 1 (期待利益/単位投資額)
      half-Kelly: kelly_fraction *= 0.5
      cap: min(kelly_fraction, 0.125)
      raw_stake = bankroll × kelly_fraction
      stake = floor(raw_stake / 100) × 100  (100円単位)
      stake = min(stake, 10000)
    """

    MIN_STAKE: int = 100  # 最低投票額
    MAX_STAKE: int = 10000  # 最大投票額
    KELLY_FRACTION_CAP: float = 0.25  # Kelly fraction の最大値
    FRACTIONAL_KELLY: float = 0.5  # half-Kelly

    def generate(
        self,
        feats: dict,
        bankroll: float,
        ev_threshold: float = 1.10,      # kept for API compat, not used
        max_bets: int = 3,
        edge_threshold: float = 0.03,    # Value Betting threshold
    ) -> list[Bet]:
        """
        複勝ベット候補を生成する。

        Args:
            feats: 特徴量dict（DataFrame互換、キーでアクセス）
            bankroll: 現在の資金
            ev_threshold: EV下限閾値（API互換のため残す、未使用）
            max_bets: 1レースあたりの最大ベット数
            edge_threshold: edge閾値（Value Betting）

        Returns:
            Bet リスト
        """
        edge_list = feats["edge_place"]
        odds_list = feats["place_odds"]
        race_ids = feats["race_id"]
        umabans = feats["umaban"]

        candidates: list[tuple[float, float, str, int]] = []
        for i, edge in enumerate(edge_list):
            if edge >= edge_threshold:
                candidates.append((edge, odds_list[i], race_ids[i], umabans[i]))

        # edge降順でソート → 上位 max_bets 件
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_bets]

        bets: list[Bet] = []
        for edge, odds, race_id, umaban in candidates:
            stake = self._calc_stake(edge, odds, bankroll)
            bets.append(
                Bet(
                    race_id=race_id,
                    umaban=umaban,
                    bet_type=BetType.PLACE,
                    odds=odds,
                    ev_lower_corrected=0.0,  # edge-based: EV lower no longer used
                    stake=stake,
                    edge=float(edge),
                )
            )

        return bets

    def _calc_stake(self, edge: float, odds: float, bankroll: float) -> float:
        """Value Betting Kelly で賭け金を計算する（StakeCalculator と同ロジック）。"""
        if bankroll <= 0 or odds <= 1.0:
            return 0.0

        # VB Kelly: f* = edge / (odds - 1)
        # edge = p*odds - 1 のとき、f = (p*odds - 1)/(odds-1) は標準Kelly公式
        kelly_fraction = edge / (odds - 1.0)
        kelly_fraction = min(kelly_fraction, self.KELLY_FRACTION_CAP)
        kelly_fraction *= self.FRACTIONAL_KELLY  # half-Kelly

        raw_stake = bankroll * kelly_fraction
        stake = max(0, int(math.floor(raw_stake / self.MIN_STAKE)) * self.MIN_STAKE)
        stake = min(stake, self.MAX_STAKE)
        return float(stake)
