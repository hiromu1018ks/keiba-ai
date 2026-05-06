"""バックテストエンジン (§13)

学習済みモデルで履歴データをシミュレーションし、投資成績を評価。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from backtest.diagnostic_logger import DiagnosticLogger
from backtest.race_predictor import RacePredictor
from betting.odds_band_filter import OddsBandFilter
from db.parquet_store import ParquetStore
from db.readers import (
    load_entries,
    load_odds_snapshots,
    load_odds_time_series_range,
    load_payouts,
    load_races,
    load_wide_odds,
)
from domain.models import Bet, BetType
from models.regime_detector import calc_favorite_implied_prob, calc_odds_skewness

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)

POST_RACE_COLS: list[str] = [
    "kakuteijyuni",
    "confirmed_odds",
    "ninki",
    "kyakusitukubun",
    "time",
    "timediff",
    "harontimel3",
    "harontimel4",
    "jyuni1c",
    "jyuni2c",
    "jyuni3c",
    "jyuni4c",
    "honsyokin",
    "chakusacd",
    "dmjyuni",
    "dmtime",
]


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
    bet_history: list[dict[str, Any]] = field(default_factory=list)
    n_pre_post_odds_bets: int = 0   # 発走前オッズでベットした件数
    n_fallback_odds_bets: int = 0   # フォールバック（確定オッズ）でベットした件数
    avg_edge: float = 0.0           # Value Betting 平均 edge
    min_edge: float = 0.0           # Value Betting 最小 edge
    max_edge: float = 0.0           # Value Betting 最大 edge
    # Phase 11: Bet selection filter exclusion stats
    n_collapsed_skipped: int = 0          # D-11: COLLAPSED regime skip count
    n_ev_excluded: int = 0                # D-01: EV filter exclusion count
    n_odds_band_excluded: int = 0         # D-06: OddsBandFilter exclusion count
    exclusion_stats: dict[str, Any] = field(default_factory=dict)  # Full exclusion breakdown

    @property
    def profit(self) -> float:
        return self.total_return - self.total_stake

    @property
    def monthly_returns(self) -> dict[str, float]:
        """bet_historyから月別ROIを集計して返す。"""
        if not self.bet_history:
            return {}
        monthly: dict[str, list[float]] = {}
        for b in self.bet_history:
            date_str = b.get("race_date", "")
            month_key = date_str[:7]  # "YYYY-MM"
            if not month_key:
                continue
            if month_key not in monthly:
                monthly[month_key] = [0.0, 0.0]  # [total_stake, total_return]
            monthly[month_key][0] += b.get("stake", 0)
            result_val = b.get("result", 0)
            if result_val > 0:
                monthly[month_key][1] += result_val
        return {
            k: (ret / stk if stk > 0 else 0.0)
            for k, (stk, ret) in monthly.items()
        }

    def summary(self) -> str:
        lines = [
            "Backtest Result:",
            f"  Bets: {self.total_bets}",
            f"  Stake: ¥{self.total_stake:,.0f}",
            f"  Return: ¥{self.total_return:,.0f}",
            f"  ROI: {self.total_roi:.3%}",
            f"  Max DD: {self.max_drawdown:.3%}",
            f"  Final Bankroll: ¥{self.final_bankroll:,.0f}",
        ]
        if self.n_pre_post_odds_bets + self.n_fallback_odds_bets > 0:
            total = self.n_pre_post_odds_bets + self.n_fallback_odds_bets
            fallback_pct = self.n_fallback_odds_bets / total * 100
            lines.append(
                f"  Odds fallback: {self.n_fallback_odds_bets}/{total} ({fallback_pct:.1f}%)"
            )
        if self.avg_edge > 0:
            lines.append(
                f"  Edge: avg={self.avg_edge:.4f}, "
                f"min={self.min_edge:.4f}, max={self.max_edge:.4f}"
            )
        return "\n".join(lines)


def build_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) → odds_multiplier のマップを構築。

    payfukusyopay は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: melt + groupby で一括処理。同一 (race_id, umaban) の最大値を保持。
    """
    if payouts_df.empty:
        return {}
    id_vars = ["race_id"]
    maban_cols = [f"payfukusyoumaban{i}" for i in range(1, 6)]
    pay_cols = [f"payfukusyopay{i}" for i in range(1, 6)]

    maban_melted = payouts_df[id_vars + maban_cols].melt(
        id_vars=id_vars, value_vars=maban_cols, value_name="umaban",
    )
    pay_melted = payouts_df[id_vars + pay_cols].melt(
        id_vars=id_vars, value_vars=pay_cols, value_name="pay",
    )

    combined = pd.DataFrame({
        "race_id": maban_melted["race_id"].values,
        "umaban": maban_melted["umaban"].values,
        "pay": pay_melted["pay"].values,
    })
    combined = combined.dropna(subset=["umaban", "pay"])
    combined["umaban"] = combined["umaban"].astype(int)
    combined["pay_100"] = combined["pay"] / 100.0

    # 同一 (race_id, umaban) の最大値を保持
    idx = combined.groupby(["race_id", "umaban"])["pay_100"].idxmax()
    deduped = combined.loc[idx]

    payout_map: dict[tuple[str, int], float] = {}
    for race_id, umaban, pay_100 in zip(
        deduped["race_id"].values, deduped["umaban"].values, deduped["pay_100"].values
    ):
        payout_map[(str(race_id), int(umaban))] = float(pay_100)
    return payout_map


def build_win_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) → odds_multiplier のマップを構築 (単勝用)。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: dropna → astype → dict comprehension。
    """
    if payouts_df.empty:
        return {}
    df = payouts_df.dropna(subset=["paytansyoumaban1", "paytansyopay1"]).copy()
    if df.empty:
        return {}
    df["umaban"] = df["paytansyoumaban1"].astype(int)
    df["pay_100"] = df["paytansyopay1"] / 100.0
    return {
        (str(race_id), int(umaban)): float(pay_100)
        for (race_id, umaban), pay_100 in df.set_index(["race_id", "umaban"])[
            "pay_100"
        ].items()
    }


def build_wide_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int, int], float]:
    """payouts DataFrame から (race_id, umaban_lo, umaban_hi) → odds_multiplier のマップを構築。

    ワイド払戻は paywidekumi1-7 と paywidepay1-7 (100円あたり円) を使用。
    kumi 形式は非ゼロ埋め: "513" = 馬5+馬13, "1113" = 馬11+馬13, "15" = 馬1+馬5。
    ベクトル化: melt + str vectorized ops で一括処理。
    """
    if payouts_df.empty:
        return {}

    id_vars = ["race_id"]
    kumi_cols = [f"paywidekumi{i}" for i in range(1, 8)]
    pay_cols = [f"paywidepay{i}" for i in range(1, 8)]

    kumi_melted = payouts_df[id_vars + kumi_cols].melt(
        id_vars=id_vars, value_vars=kumi_cols, value_name="kumi",
    )
    pay_melted = payouts_df[id_vars + pay_cols].melt(
        id_vars=id_vars, value_vars=pay_cols, value_name="pay",
    )

    combined = pd.DataFrame({
        "race_id": kumi_melted["race_id"].values,
        "kumi": kumi_melted["kumi"].values,
        "pay": pay_melted["pay"].values,
    })
    combined = combined.dropna(subset=["kumi", "pay"])
    # BUG-FIX: Parquet may store kumi as float64 (e.g. 513.0).
    # Convert to str and strip trailing ".0" from float-as-string, preserving zero-padded strings.
    combined["kumi"] = combined["kumi"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    combined = combined[combined["kumi"] != ""]

    if combined.empty:
        return {}

    # Vectorized kumi parsing based on string length
    lengths = combined["kumi"].str.len()

    # Initialize lo/hi columns
    lo = pd.Series(np.nan, index=combined.index, dtype=float)
    hi = pd.Series(np.nan, index=combined.index, dtype=float)

    # Length 2: "XY" → lo=X, hi=Y (e.g., "15" → 1, 5)
    mask2 = lengths == 2
    if mask2.any():
        lo.loc[mask2] = combined.loc[mask2, "kumi"].str.slice(0, 1).astype(int)
        hi.loc[mask2] = combined.loc[mask2, "kumi"].str.slice(1, 2).astype(int)

    # Length 3: "XYZ" — ambiguous: could be (X, YZ) or (XY, Z)
    # Heuristic: if int(XY) <= 18, use (XY, Z); else use (X, YZ)
    mask3 = lengths == 3
    if mask3.any():
        first_two = combined.loc[mask3, "kumi"].str.slice(0, 2).astype(int)
        use_first_two = first_two <= 18
        idx3 = combined.index[mask3]

        # Where first two digits form a valid horse number (1-18)
        split_at_2 = idx3[use_first_two]
        if len(split_at_2) > 0:
            lo.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(0, 2).astype(int)
            hi.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(2, 3).astype(int)

        # Otherwise split at 1
        split_at_1 = idx3[~use_first_two]
        if len(split_at_1) > 0:
            lo.loc[split_at_1] = combined.loc[split_at_1, "kumi"].str.slice(0, 1).astype(int)
            hi.loc[split_at_1] = combined.loc[split_at_1, "kumi"].str.slice(1, 3).astype(int)

    # Length 4: "XXYY" → lo=XX, hi=YY (e.g., "1113" → 11, 13)
    mask4 = lengths == 4
    if mask4.any():
        lo.loc[mask4] = combined.loc[mask4, "kumi"].str.slice(0, 2).astype(int)
        hi.loc[mask4] = combined.loc[mask4, "kumi"].str.slice(2, 4).astype(int)

    # Length 5: "XXYYZ" (rare, e.g. zero-padded "01113") → treat as (XX, YYZ) or (XXX, YZ)
    mask5 = lengths >= 5
    if mask5.any():
        # Use same 2+3 or 3+2 logic based on first 2 digits
        first_two = combined.loc[mask5, "kumi"].str.slice(0, 2).astype(int)
        use_first_two = first_two <= 18
        idx5 = combined.index[mask5]
        kumi5 = combined.loc[mask5, "kumi"]
        kumi5_len = lengths[mask5]

        split_at_2 = idx5[use_first_two]
        if len(split_at_2) > 0:
            lo.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(0, 2).astype(int)
            hi.loc[split_at_2] = (
                combined.loc[split_at_2, "kumi"]
                .str.slice(2).astype(int)
            )

        split_at_3 = idx5[~use_first_two]
        if len(split_at_3) > 0:
            lo.loc[split_at_3] = (
                combined.loc[split_at_3, "kumi"]
                .str.slice(0, -2).astype(int)
            )
            hi.loc[split_at_3] = combined.loc[split_at_3, "kumi"].str.slice(-2).astype(int)

    combined["lo"] = lo
    combined["hi"] = hi
    combined = combined.dropna(subset=["lo", "hi"])
    combined["lo"] = combined["lo"].astype(int)
    combined["hi"] = combined["hi"].astype(int)
    combined["pay_100"] = combined["pay"] / 100.0

    # Ensure lo <= hi
    combined["_lo"] = np.minimum(combined["lo"], combined["hi"])
    combined["_hi"] = np.maximum(combined["lo"], combined["hi"])
    combined["lo"] = combined["_lo"]
    combined["hi"] = combined["_hi"]
    combined = combined.drop(columns=["_lo", "_hi", "kumi"])

    # Keep max payout per key
    idx = combined.groupby(["race_id", "lo", "hi"])["pay_100"].idxmax()
    deduped = combined.loc[idx]

    wide_payout_map: dict[tuple[str, int, int], float] = {}
    for race_id, lo_val, hi_val, pay_100 in zip(
        deduped["race_id"].values,
        deduped["lo"].values,
        deduped["hi"].values,
        deduped["pay_100"].values,
    ):
        wide_payout_map[(str(race_id), int(lo_val), int(hi_val))] = float(pay_100)
    return wide_payout_map


def build_race_groups(
    df: pd.DataFrame,
    group_col: str = "race_id",
    name: str = "",
) -> dict[str, pd.DataFrame]:
    """DataFrame を group_col でグループ化し dict に変換。

    pandas>=2.0 の groupby は view を返すため、実質的なメモリ増加は元の1.1〜1.2倍程度。
    """
    if df.empty:
        logger.warning("[%s] empty DataFrame, returning empty dict", name)
        return {}
    groups: dict[str, pd.DataFrame] = {}
    for key, group in df.groupby(group_col):
        groups[str(key)] = group
    empty_count = sum(1 for g in groups.values() if g.empty)
    if empty_count > 0:
        logger.warning("[%s] %d empty groups in %d total", name, empty_count, len(groups))
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info("[%s] %d groups, %d rows, %.1f MB", name, len(groups), len(df), mem_mb)
    return groups


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
        betting_mode: str = "flat",
        diag_prefix: str = "bt",
        betting_target: str = "win",
        strategy_params: dict[str, Any] | None = None,
    ) -> None:
        if betting_mode not in ("flat", "kelly"):
            raise ValueError(f"betting_mode must be 'flat' or 'kelly', got '{betting_mode}'")
        if betting_target not in ("win", "place", "wide"):
            raise ValueError(f"betting_target must be 'win', 'place', or 'wide', got '{betting_target}'")
        self.models = models
        self.initial_bankroll = initial_bankroll
        self.store = store or ParquetStore()
        self.betting_mode = betting_mode
        self.diag_prefix = diag_prefix
        self.betting_target = betting_target
        self.strategy_params = strategy_params

        # Phase 11: Bet selection filters
        self._odds_band_filter: OddsBandFilter | None = None
        if betting_target == "win":
            _roi_thresh = (strategy_params or {}).get("roi_threshold", 1.0)
            self._odds_band_filter = OddsBandFilter(roi_threshold=_roi_thresh)

        if betting_mode == "kelly":
            from betting.drawdown_controller import DDConfig, DrawdownController
            from betting.stake_calculator import StakeCalculator

            if strategy_params is not None:
                # Optuna最適化済みパラメータで注入
                dd_cfg = strategy_params.get("dd_config", DDConfig())
                stake_calc = StakeCalculator(
                    fractional_kelly=strategy_params.get("fractional_kelly", 0.5),
                    target_ev=strategy_params.get("target_ev", 1.10),
                    max_scale=strategy_params.get("max_scale", 2.0),
                )
                dd_ctrl = DrawdownController(
                    peak_bankroll=initial_bankroll,
                    cfg=dd_cfg,
                )
            else:
                stake_calc = StakeCalculator()
                dd_ctrl = DrawdownController(peak_bankroll=initial_bankroll)

            self._race_predictor = RacePredictor(
                models,
                stake_calculator=stake_calc,
                dd_controller=dd_ctrl,
            )
        else:
            self._race_predictor = RacePredictor(models)

    def _generate_training_bet_history(
        self,
    ) -> list[dict[str, Any]] | None:
        """デフォルトパラメータでトレーニング期間バックテストを実行し、bet_historyを生成。

        D-05/D-06: run()内でtraining_bet_historyがNoneの場合に自動呼び出し。
        D-07: トレーニング期間はself.models.train_periodから取得 (test_start/test_endは使用しない)。
        Pitfall 3回避: 内部BacktestEngineはbetting_target="place"で構築し、
        OddsBandFilterを持たせない（再帰calibrate防止）。
        """
        from betting.default_strategy import build_default_strategy_config

        try:
            # D-07: models.train_periodからトレーニング期間を取得
            # TrainedModelsV5.train_period = (train_start, train_end)
            train_start, train_end = self.models.train_period
            default_config = build_default_strategy_config()
            # Pitfall 3: 内部エンジンはOddsBandFilterを持たない
            # betting_target="place"を指定して_odds_band_filter=Noneにする
            inner_engine = BacktestEngine(
                models=self.models,
                initial_bankroll=self.initial_bankroll,
                store=self.store,
                betting_mode=self.betting_mode,
                diag_prefix=f"{self.diag_prefix}_train",
                betting_target="place",  # OddsBandFilterはwin専用 -> 再帰防止
                strategy_params=default_config,
            )
            train_result = inner_engine.run(train_start, train_end)
            logger.info(
                "自動training_bet_history生成完了: %d bets, ROI=%.1f%% (%s ~ %s)",
                train_result.total_bets,
                train_result.total_roi * 100,
                train_start, train_end,
            )
            return train_result.bet_history
        except Exception as e:
            logger.warning(
                "自動training_bet_history生成失敗: %s — OddsBandFilterキャリブレーションスキップ", e,
            )
            return None

    def run(
        self,
        test_start: str,
        test_end: str,
        training_bet_history: list[dict[str, Any]] | None = None,  # D-05
    ) -> BacktestResult:
        """バックテストを実行

        Args:
            test_start: テスト開始日 (YYYY-MM-DD)
            test_end: テスト終了日 (YYYY-MM-DD)
            training_bet_history: トレーニング期間のベット履歴 (OddsBandFilter キャリブレーション用)

        Returns:
            BacktestResult
        """
        # 1. データロード
        start = test_start.replace("-", "")
        end = test_end.replace("-", "")
        race_df = load_races(self.store, start, end)
        entry_df = load_entries(self.store, start, end)
        final_odds_df = load_odds_snapshots(self.store, start, end)  # 確定オッズ（精算用）

        if race_df.empty:
            logger.warning(f"No races found in {test_start} ~ {test_end}")
            return BacktestResult(final_bankroll=self.initial_bankroll)

        if "jyocd" in race_df.columns:
            jyocd_int = pd.to_numeric(race_df["jyocd"], errors="coerce")
            jra_race_ids = race_df.loc[jyocd_int.between(1, 10), "race_id"].drop_duplicates()
            race_df = race_df[race_df["race_id"].isin(jra_race_ids)].copy()
            entry_df = entry_df[entry_df["race_id"].isin(jra_race_ids)].copy()
            final_odds_df = final_odds_df[final_odds_df["race_id"].isin(jra_race_ids)].copy()

        # 2. 特徴量生成
        from db.odds_extractor import extract_pre_post_odds
        from features.feature_engine import FeatureEngine
        from models.submodel_manager import SubModelManager

        feat_engine = FeatureEngine()
        submodel_mgr = SubModelManager()
        odds_ts_df = load_odds_time_series_range(self.store, start, end)
        if not odds_ts_df.empty:
            odds_ts_df = odds_ts_df[odds_ts_df["race_id"].isin(race_df["race_id"])].copy()

        # 発走前オッズの抽出（フォールバックなし: 時系列オッズがない場合は全レーススキップ）
        if odds_ts_df.empty:
            logger.warning(
                "No time-series odds data for %s ~ %s, skipping all races", test_start, test_end
            )
            return BacktestResult(final_bankroll=self.initial_bankroll)

        if "hassotime" not in race_df.columns:
            logger.warning(
                "hassotime column missing, cannot extract pre-race odds, skipping all races"
            )
            return BacktestResult(final_bankroll=self.initial_bankroll)

        pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
        if pre_post_odds.empty:
            logger.warning(
                "extract_pre_post_odds returned empty for %s ~ %s, skipping all races",
                test_start,
                test_end,
            )
            return BacktestResult(final_bankroll=self.initial_bankroll)

        # 確定オッズマップを構築（精算用。FeatureEngine の列フィルタ回避）
        final_odds_map: dict[tuple[str, int], float] = {}
        if not final_odds_df.empty:
            _odds = final_odds_df.dropna(subset=["fukuoddslow"])
            if not _odds.empty:
                for (race_id, umaban), odds in (
                    _odds.set_index(["race_id", "umaban"])["fukuoddslow"].items()
                ):
                    final_odds_map[(str(race_id), int(umaban))] = float(odds)

        # 確定配当マップを構築（精算用。実際の払戻金額を使用）
        # BUG-FIX: betting_target に応じて必要な払戻マップのみ構築
        payouts_df = load_payouts(self.store, start, end)

        needs_place = self.betting_target in ("place", "wide")
        needs_win = self.betting_target in ("win", "wide")
        needs_wide = self.betting_target == "wide"

        if needs_place:
            self.payout_map = build_payout_map(payouts_df)
            logger.info("Loaded payout map: %d entries", len(self.payout_map))
        else:
            self.payout_map = {}

        if needs_win:
            self.win_payout_map = build_win_payout_map(payouts_df)
            logger.info("Loaded win payout map: %d entries", len(self.win_payout_map))
        else:
            self.win_payout_map = {}

        if needs_wide:
            self.wide_payout_map = build_wide_payout_map(payouts_df)
            logger.info("Loaded wide payout map: %d entries", len(self.wide_payout_map))
        else:
            self.wide_payout_map = {}

        feat_df = feat_engine.build_all(
            race_df, entry_df, pre_post_odds, odds_ts_df=odds_ts_df, store=self.store
        )
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # ワイドオッズを pivot して特徴量にマージ（WideJointPairBuilder 用）
        wide_odds_df = load_wide_odds(self.store, start, end)
        if wide_odds_df is not None and not wide_odds_df.empty:
            # kumi "0102" → int変換で "1_2" 形式（WideJointPairBuilder の lookup に合わせる）
            _wide = wide_odds_df[["race_id", "kumi", "oddslow"]].dropna(subset=["oddslow"])
            if not _wide.empty:
                wide_pivot = _wide.pivot_table(index="race_id", columns="kumi", values="oddslow")
                # ゼロ埋めを解除: "0102" → "1_2", "0211" → "2_11"
                new_cols = []
                for c in wide_pivot.columns:
                    lo = int(c[:2])
                    hi = int(c[2:])
                    new_cols.append(f"wide_odds_{lo}_{hi}")
                wide_pivot.columns = new_cols
                wide_pivot = wide_pivot.reset_index()
                feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")
                logger.info("Merged wide odds: %d pair-columns", len(wide_pivot.columns) - 1)

        # Safety check: verify feature generation did not introduce NAR entries
        # (already filtered at data load above; this catches pipeline bugs)
        if "jyocd" in feat_df.columns:
            jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
            nar_count = (~jyocd_int.between(1, 10)).sum()
            if nar_count > 0:
                logger.warning(
                    "NAR entries leaked into feat_df: %d (feature pipeline bug?)",
                    int(nar_count),
                )
                feat_df = feat_df[jyocd_int.between(1, 10)]

        # 3. 特徴量の一括事前計算 (ループ外で全レース分を一度に計算)
        from features.horse_history_features import HorseHistoryFeatures
        from features.jockey_context_features import JockeyContextFeatures
        from features.jockey_trainer_combo import JockeyTrainerComboFeatures
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

        logger.info("Pre-computing JockeyTrainerComboFeatures for %d entries...", len(entry_df))
        jt_combo = JockeyTrainerComboFeatures(self.store)
        jt_df_all = jt_combo.compute(entry_df)

        # 種牡馬産駒特徴量の追加 (推論パス — 学習と同一ロジック)
        from db.readers import load_horses, load_sire_stats
        from features.sire_features import SireFeatures

        logger.info("Computing SireFeatures for backtest inference...")
        sire_stats_bt = load_sire_stats(self.store)
        if not sire_stats_bt.empty:
            horses_bt = load_horses(self.store)
            sire_feat_bt = SireFeatures(sire_stats_bt)
            sire_map_bt = horses_bt.set_index("kettonum")["ketto3infohansyokunum1"]
            feat_df["sire_id"] = feat_df["kettonum"].map(sire_map_bt)
            bms_map_bt = horses_bt.set_index("kettonum")["ketto3infohansyokunum3"]
            feat_df["bms_id"] = feat_df["kettonum"].map(bms_map_bt)
            sire_result_bt = sire_feat_bt.compute_batch(feat_df)
            _sire_cols_needed = {
                "sire_wr",
                "sire_surface_wr",
                "sire_distance_wr",
                "sire_prize_avg",
                "bms_wr",
            }
            for col in _sire_cols_needed:
                if col in sire_result_bt.columns:
                    feat_df[col] = sire_result_bt[col].values

        # 4. PaceAptitude + CourseFeatures の事前計算 (推論パスでも必要)
        from features.course_features import CourseFeatures
        from features.pace_aptitude_features import PaceAptitudeFeatures

        logger.info("Pre-computing PaceAptitudeFeatures...")
        pace_feat = PaceAptitudeFeatures(store=self.store)
        pace_df = pace_feat.compute_batch(feat_df)
        # BUG-FIX: 学習パイプラインと同様に全6列をマージ (PACE-01 の3列が漏れていた)
        _pace_cols = [
            c
            for c in [
                "pace_aptitude", "front_pace_wr", "closing_pace_wr",
                "pace_corner_stability", "pace_closing_power", "pace_position_consistency",
            ]
            if c in pace_df.columns
        ]
        if _pace_cols:
            feat_df = feat_df.drop(columns=_pace_cols, errors="ignore").merge(
                pace_df[["kettonum", "race_id"] + _pace_cols],
                on=["kettonum", "race_id"],
                how="left",
            )

        logger.info("Pre-computing CourseFeatures...")
        course_feat = CourseFeatures(store=self.store)
        course_df = course_feat.compute_batch(feat_df)
        _course_cols = [c for c in ["course_wr", "course_distance_wr"] if c in course_df.columns]
        if _course_cols:
            feat_df = feat_df.drop(columns=_course_cols, errors="ignore").merge(
                course_df[["kettonum", "race_id"] + _course_cols],
                on=["kettonum", "race_id"],
                how="left",
            )

        # 5. レースごとにシミュレーション (推論は RacePredictor に委譲)
        # Groupby dict preprocessing — O(1) race lookups per D-07
        feat_groups = build_race_groups(feat_df, name="features")
        hist_groups = build_race_groups(hist_df_all, name="history")
        jockey_groups = build_race_groups(jockey_df_all, name="jockey")
        trainer_groups = build_race_groups(trainer_df_all, name="trainer")
        jt_groups = build_race_groups(jt_df_all, name="jockey_trainer")

        diag_logger = DiagnosticLogger()
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_dd = 0.0
        bet_history: list[dict[str, Any]] = []
        n_pre_post_odds_bets = 0
        n_fallback_odds_bets = 0

        # RegimeDetector 用: 直近200レースの統計を蓄積
        recent_stats_list: list[dict[str, float]] = []

        # Phase 11: Bet selection filter counters
        n_collapsed_skipped = 0
        n_ev_excluded = 0
        n_odds_band_excluded = 0
        n_total_candidates = 0

        # D-05/D-06/D-07: OddsBandFilter キャリブレーション
        # training_bet_historyがNoneの場合、engine内で自動生成する
        if self._odds_band_filter is not None:
            if training_bet_history is None:
                # D-05: run()内で自動的にtraining_bet_historyを生成
                # D-06: engine.py内部で完結させる
                # D-07: トレーニング期間はmodels.train_periodから取得
                training_bet_history = self._generate_training_bet_history()
            if training_bet_history:
                self._odds_band_filter.calibrate(training_bet_history)

        for race_id in race_ids:
            race_id = str(race_id)
            race_df_single = feat_groups.get(race_id)
            if race_df_single is None:
                continue
            race_df_single = race_df_single.copy()

            # --- レースメタデータ抽出 (bet_history拡張用) ---
            race_row = race_df_single.iloc[0]
            race_date_str = (
                f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}" if len(race_id) >= 8 else ""
            )
            _jyocd = (
                str(race_row.get("jyocd", "")).zfill(2) if pd.notna(race_row.get("jyocd")) else ""
            )
            _racenum = int(race_row.get("racenum", 0)) if pd.notna(race_row.get("racenum")) else 0
            _grade_code = (
                str(race_row.get("grade_code", "_"))
                if pd.notna(race_row.get("grade_code"))
                else "_"
            )
            _race_name = str(race_row.get("hondai", "")) if pd.notna(race_row.get("hondai")) else ""
            _track_condition = (
                int(race_row.get("track_condition_code", 0))
                if pd.notna(race_row.get("track_condition_code"))
                else 0
            )

            # top3_finishers: kakuteijyuni でソートした上位3頭
            _valid = race_df_single[
                race_df_single["kakuteijyuni"].notna() & (race_df_single["kakuteijyuni"] > 0)
            ].nsmallest(3, "kakuteijyuni")
            _top3: list[dict[str, Any]] = []
            for r in _valid.itertuples(index=False):
                _top3.append(
                    {
                        "umaban": int(r.umaban),
                        "bamei": str(r.bamei) if pd.notna(r.bamei) else "",
                        "kisyuryakusyo": (
                            str(r.kisyuryakusyo)
                            if pd.notna(r.kisyuryakusyo)
                            else ""
                        ),
                        "kakuteijyuni": int(r.kakuteijyuni),
                    }
                )

            # 事前計算済み特徴量をマージ (groupby dict O(1) lookup)
            hist_df_race = hist_groups.get(race_id)
            jockey_df_race = jockey_groups.get(race_id)
            trainer_df_race = trainer_groups.get(race_id)
            jt_df_race = jt_groups.get(race_id)

            # M3 fix: POST_RACE 列を predict() に渡さない
            predict_df = race_df_single.drop(
                columns=[c for c in POST_RACE_COLS if c in race_df_single.columns],
                errors="ignore",
            )
            # RacePredictor に委譲
            result_df = self._race_predictor.predict(
                predict_df,
                hist_features=hist_df_race,
                jockey_features=jockey_df_race,
                trainer_features=trainer_df_race,
                jt_combo_features=jt_df_race,
            )
            if result_df.empty:
                continue

            # 精算・bet_history 用に POST_RACE 列を復元 (kakuteijyuni, confirmed_odds のみ)
            for col in POST_RACE_COLS[:2]:
                if col in race_df_single.columns and col not in result_df.columns:
                    result_df = result_df.merge(
                        race_df_single[["umaban", col]],
                        on="umaban",
                        how="left",
                    )

            # Quality screening — RegimeDetector.detect() でレジーム更新
            recent_stats_df = pd.DataFrame(recent_stats_list[-200:])
            if len(recent_stats_df) >= self.models.regime_detector.cfg.min_samples:
                regime = self.models.regime_detector.detect(recent_stats_df)
            else:
                regime = self.models.regime_detector.current_regime
            regime_params = self.models.regime_detector.get_strategy_params(regime)
            # D-11: レジーム別 fractional_kelly を StakeCalculator に注入
            if self._race_predictor.stake_calc is not None:
                fk = float(regime_params.get("fractional_kelly", 0.5))
                self._race_predictor.stake_calc.fractional_kelly = fk
            edge_threshold = regime_params.get("edge_threshold", 0.03)

            # D-11: 統計を蓄積 (COLLAPSEDスキップ前 — レジーム遷移に必要, Pitfall 3)
            row_data = result_df.iloc[0] if not result_df.empty else {}
            recent_stats_list.append({
                "market_error_std": (
                    float(result_df["signed_log_error_win"].std())
                    if "signed_log_error_win" in result_df.columns and len(result_df) > 1
                    else 0.2
                ),
                "market_error_mean": (
                    float(result_df["signed_log_error_win"].mean())
                    if "signed_log_error_win" in result_df.columns
                    else 0.0
                ),
                "overround_rolling": float(row_data.get("overround", 0.20))
                    if not result_df.empty else 0.20,
                "entropy_rolling": float(row_data.get("market_entropy", 2.0))
                    if not result_df.empty else 2.0,
                "odds_skewness_rolling": calc_odds_skewness(result_df),
                "favorite_implied_prob_rolling": calc_favorite_implied_prob(result_df),
                "odds_volatility_mean": (
                    float(result_df["odds_volatility"].mean())
                    if "odds_volatility" in result_df.columns and not result_df.empty
                    else 0.1
                ),
                "field_size_mean": float(row_data.get("field_size", 14.0))
                    if not result_df.empty else 14.0,
            })

            # D-11: COLLAPSED regime skip (race-level, D-09: filter order #1)
            if regime_params.get("skip", False):
                n_collapsed_skipped += 1
                continue
            if self.betting_target == "win":
                candidate_df = self._race_predictor.get_win_candidates(result_df)
                n_ev_excluded += int(candidate_df.attrs.get("n_ev_excluded", 0))
            else:
                candidate_df = self._race_predictor.get_place_candidates(
                    result_df,
                    regime_params=regime_params,
                )
            n_candidates = len(candidate_df)

            # D-09: OddsBandFilter (candidate-level, filter order #3)
            n_candidates_before_band = n_candidates
            if (
                self._odds_band_filter is not None
                and not candidate_df.empty
            ):
                candidate_df = self._odds_band_filter.filter(candidate_df)
                n_odds_band_excluded += n_candidates_before_band - len(candidate_df)
            n_total_candidates += n_candidates_before_band
            # BUG-FIX: place_selection_reason は place/wide 候補にのみ存在。
            # win モード時は get_win_candidates() がこの列を生成しないため、列存在チェックで保護。
            _reason_cols = ["race_id", "umaban"]
            if "place_selection_reason" in candidate_df.columns:
                _reason_cols.append("place_selection_reason")
            candidate_reason_df = candidate_df[_reason_cols].copy()
            candidate_reason_df["umaban"] = candidate_reason_df["umaban"].astype(
                result_df["umaban"].dtype
            )
            result_df = result_df.merge(
                candidate_reason_df.drop_duplicates(subset=["race_id", "umaban"]),
                on=["race_id", "umaban"],
                how="left",
            )
            race_aggressive_strength = float(
                result_df.get("aggressive_strength", pd.Series([np.nan])).iloc[0]
            )
            race_aggressive_tier = result_df.get("aggressive_tier", pd.Series([None])).iloc[0]
            race_market_condition = float(
                result_df.get("market_condition_score", pd.Series([np.nan])).iloc[0]
            )

            if not self._race_predictor.should_bet(result_df):
                diag_logger.log_race(
                    race_id=race_id,
                    regime=str(regime),
                    ev_threshold=regime_params.get("ev_threshold", 1.10),
                    edge_threshold=edge_threshold,
                    quality_passed=False,
                    quality_score=0.0,
                    n_candidates=n_candidates,
                    n_bets=0,
                    aggressive_strength=race_aggressive_strength,
                    aggressive_tier=(
                        str(race_aggressive_tier) if pd.notna(race_aggressive_tier) else None
                    ),
                    market_condition_score=race_market_condition,
                )
                if "ev_place" in result_df.columns:
                    for hr in result_df.itertuples(index=False):
                        diag_logger.log_horse(
                            race_id=race_id,
                            umaban=int(hr.umaban),
                            p_place_pred=float(getattr(hr, "p_place_pred", 0)),
                            e_return_place_pred=float(getattr(hr, "e_return_place_pred", 0)),
                            ev_place=float(getattr(hr, "ev_place", 0)),
                            fukuoddslow=float(getattr(hr, "fukuoddslow", 0)),
                            is_bet=False,
                            p_place_corrected=float(getattr(hr, "p_place_corrected", float("nan"))),
                            e_return_place_corrected=float(
                                getattr(hr, "e_return_place_corrected", float("nan"))
                            ),
                            ev_place_corrected=float(getattr(hr, "ev_place_corrected", float("nan"))),
                            ev_lower_place=float(getattr(hr, "EV_lower_place", float("nan"))),
                            place_selection_ev=float(getattr(hr, "place_selection_ev", float("nan"))),
                            place_selection_edge=float(
                                getattr(hr, "place_selection_edge", float("nan"))
                            ),
                            place_selection_prob=float(
                                getattr(hr, "place_selection_prob", float("nan"))
                            ),
                            place_bucket_multiplier=float(
                                getattr(hr, "place_bucket_multiplier", float("nan"))
                            ),
                            place_gate_score=float(getattr(hr, "place_gate_score", float("nan"))),
                            place_gate_pass=bool(getattr(hr, "place_gate_pass", False)),
                            place_gate_rank=float(getattr(hr, "place_gate_rank", float("nan"))),
                            place_gate_score_gap=float(
                                getattr(hr, "place_gate_score_gap", float("nan"))
                            ),
                            market_condition_score=float(
                                getattr(hr, "market_condition_score", float("nan"))
                            ),
                            aggressive_strength=float(
                                getattr(hr, "aggressive_strength", float("nan"))
                            ),
                            aggressive_tier=(
                                str(getattr(hr, "aggressive_tier"))
                                if pd.notna(getattr(hr, "aggressive_tier", None))
                                else None
                            ),
                            place_selection_reason=(
                                str(getattr(hr, "place_selection_reason"))
                                if pd.notna(getattr(hr, "place_selection_reason", None))
                                else None
                            ),
                        )
                        diag_logger.log_horse_features(hr._asdict())
                continue

            # Bet generation
            surface_key = result_df["surface"].iloc[0]
            bets = self._race_predictor.select_bets(
                result_df, bankroll, candidates=candidate_df,
                betting_target=self.betting_target,
            )

            # v5: セグメント除外フィルタ全削除 — モデル自身がedgeを低に見積もるように改善する
            # (旧v4の14個の除外フィルタは全て削除)

            # Bet に確定オッズを設定（place/win のみ。wide は wide_payout_map で精算）
            updated_bets = []
            for bet in bets:
                if bet.bet_type == BetType.WIDE:
                    updated_bets.append(bet)
                else:
                    fo = final_odds_map.get((bet.race_id, bet.umaban), bet.odds)
                    updated_bets.append(replace(bet, final_odds=fo))
            bets = updated_bets

            # メトリクス集計 (全ベットが発走前オッズ)
            n_pre_post_odds_bets += len(bets)

            # Log diagnostics for quality-passed race
            diag_logger.log_race(
                race_id=race_id,
                regime=str(regime),
                ev_threshold=regime_params.get("ev_threshold", 1.10),
                edge_threshold=edge_threshold,
                quality_passed=True,
                quality_score=0.0,
                n_candidates=n_candidates,
                n_bets=len(bets),
                aggressive_strength=race_aggressive_strength,
                aggressive_tier=(
                    str(race_aggressive_tier) if pd.notna(race_aggressive_tier) else None
                ),
                market_condition_score=race_market_condition,
            )
            if "ev_place" in result_df.columns:
                bet_umabans = {b.umaban for b in bets}
                for hr in result_df.itertuples(index=False):
                    diag_logger.log_horse(
                        race_id=race_id,
                        umaban=int(hr.umaban),
                        p_place_pred=float(getattr(hr, "p_place_pred", 0)),
                        e_return_place_pred=float(getattr(hr, "e_return_place_pred", 0)),
                        ev_place=float(getattr(hr, "ev_place", 0)),
                        fukuoddslow=float(getattr(hr, "fukuoddslow", 0)),
                        is_bet=int(hr.umaban) in bet_umabans,
                        p_place_corrected=float(getattr(hr, "p_place_corrected", float("nan"))),
                        e_return_place_corrected=float(
                            getattr(hr, "e_return_place_corrected", float("nan"))
                        ),
                        ev_place_corrected=float(getattr(hr, "ev_place_corrected", float("nan"))),
                        ev_lower_place=float(getattr(hr, "EV_lower_place", float("nan"))),
                        place_selection_ev=float(getattr(hr, "place_selection_ev", float("nan"))),
                        place_selection_edge=float(
                            getattr(hr, "place_selection_edge", float("nan"))
                        ),
                        place_selection_prob=float(
                            getattr(hr, "place_selection_prob", float("nan"))
                        ),
                        place_bucket_multiplier=float(
                            getattr(hr, "place_bucket_multiplier", float("nan"))
                        ),
                        place_gate_score=float(getattr(hr, "place_gate_score", float("nan"))),
                        place_gate_pass=bool(getattr(hr, "place_gate_pass", False)),
                        place_gate_rank=float(getattr(hr, "place_gate_rank", float("nan"))),
                        place_gate_score_gap=float(
                            getattr(hr, "place_gate_score_gap", float("nan"))
                        ),
                        market_condition_score=float(
                            getattr(hr, "market_condition_score", float("nan"))
                        ),
                        aggressive_strength=float(
                            getattr(hr, "aggressive_strength", float("nan"))
                        ),
                        aggressive_tier=(
                            str(getattr(hr, "aggressive_tier"))
                            if pd.notna(getattr(hr, "aggressive_tier", None))
                            else None
                        ),
                        place_selection_reason=(
                            str(getattr(hr, "place_selection_reason"))
                            if pd.notna(getattr(hr, "place_selection_reason", None))
                            else None
                        ),
                    )
                    diag_logger.log_horse_features(hr._asdict())

            # Settlement (BacktestEngine 固有)
            for bet in bets:
                bet_result = self._settle_bet(bet, result_df)
                bankroll -= bet.stake
                if bet_result > 0:
                    bankroll += bet_result

                # A4: DD Controller にバンクロールをフィードバック
                if self._race_predictor.dd_ctrl is not None:
                    self._race_predictor.dd_ctrl.update(bankroll)

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
                        "final_odds": bet.final_odds,
                        "result": bet_result,
                        "surface": surface_key,
                        "kyori": (
                            int(result_df["kyori"].iloc[0]) if "kyori" in result_df.columns else 0
                        ),
                        "ev": float(bet.ev_lower_corrected),
                        "edge": float(bet.edge),
                        "popularity": int(pop_val) if pd.notna(pop_val) else 0,
                        "bankroll_after": round(bankroll, 2),
                        # --- 拡張フィールド ---
                        "race_date": race_date_str,
                        "jyocd": _jyocd,
                        "racenum": _racenum,
                        "grade_code": _grade_code,
                        "race_name": _race_name,
                        "bamei": (
                            str(horse_rows.iloc[0].get("bamei", ""))
                            if not horse_rows.empty and pd.notna(horse_rows.iloc[0].get("bamei"))
                            else ""
                        ),
                        "kisyu": (
                            str(horse_rows.iloc[0].get("kisyuryakusyo", ""))
                            if not horse_rows.empty
                            and pd.notna(horse_rows.iloc[0].get("kisyuryakusyo"))
                            else ""
                        ),
                        "kakuteijyuni": (
                            int(horse_rows.iloc[0]["kakuteijyuni"])
                            if not horse_rows.empty
                            and pd.notna(horse_rows.iloc[0].get("kakuteijyuni"))
                            else 0
                        ),
                        "track_condition_code": _track_condition,
                        "regime": str(regime),
                        "p_place_pred": (
                            float(horse_rows.iloc[0].get("p_place_pred", 0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "e_return_place_pred": (
                            float(horse_rows.iloc[0].get("e_return_place_pred", 0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "top3_finishers": _top3,
                        "umaban_b": getattr(bet, "umaban_b", None),
                    }
                )

                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_dd = max(max_dd, dd)

        # 5. 診断ログ保存
        diag_logger.save(Path("data/backtest"), prefix=self.diag_prefix)

        # 6. ROI 計算
        total_stake = sum(b["stake"] for b in bet_history)
        total_return = sum(b["result"] for b in bet_history if b["result"] > 0)
        total_bets = len(bet_history)
        winning_bets = sum(1 for b in bet_history if b["result"] > 0)

        # Edge statistics for Value Betting
        result_data: dict[str, Any] = {}
        if bet_history:
            edges = [b["edge"] for b in bet_history if "edge" in b]
            if edges:
                result_data["avg_edge"] = sum(edges) / len(edges)
                result_data["min_edge"] = min(edges)
                result_data["max_edge"] = max(edges)

        # Phase 11: Bet selection filter summary logging
        logger.info(
            "Bet Selection Filters: EV_excluded=%d, COLLAPSED_skipped=%d",
            n_ev_excluded,
            n_collapsed_skipped,
        )

        # D-08: OddsBandFilter除外ログ
        if n_odds_band_excluded > 0:
            logger.info(
                "OddsBandFilter excluded %d candidates in bands: %s",
                n_odds_band_excluded,
                self._odds_band_filter.excluded_bands if self._odds_band_filter else {},
            )

        # D-10: Bet count guard
        if total_bets > 0:
            try:
                from datetime import datetime as dt
                t_start = dt.strptime(test_start, "%Y-%m-%d")
                t_end = dt.strptime(test_end, "%Y-%m-%d")
                n_years = max(1, (t_end - t_start).days / 365.25)
            except (ValueError, TypeError):
                n_years = 1.0
            bets_per_year = total_bets / n_years
            if bets_per_year < 1000:
                logger.warning(
                    "Bet count guard: %.0f bets/year (below 1000 threshold). "
                    "Consider parameter adjustment in Phase 13.",
                    bets_per_year,
                )

        return BacktestResult(
            total_bets=total_bets,
            total_stake=total_stake,
            total_return=total_return,
            winning_bets=winning_bets,
            total_roi=total_return / total_stake if total_stake > 0 else 0.0,
            max_drawdown=max_dd,
            final_bankroll=bankroll,
            bet_history=bet_history,
            n_pre_post_odds_bets=n_pre_post_odds_bets,
            n_fallback_odds_bets=n_fallback_odds_bets,
            avg_edge=result_data.get("avg_edge", 0.0),
            min_edge=result_data.get("min_edge", 0.0),
            max_edge=result_data.get("max_edge", 0.0),
            n_collapsed_skipped=n_collapsed_skipped,
            n_ev_excluded=n_ev_excluded,
            n_odds_band_excluded=n_odds_band_excluded,
            exclusion_stats={
                "collapsed_skipped": n_collapsed_skipped,
                "ev_excluded": n_ev_excluded,
                "odds_band_excluded": n_odds_band_excluded,
                "total_candidates_evaluated": n_total_candidates,
                "odds_band_filter_excluded": (
                    self._odds_band_filter.excluded_bands
                    if self._odds_band_filter
                    else {}
                ),
            },
        )

    def _settle_bet(self, bet: Bet, race_df: pd.DataFrame) -> float:
        """ベットの結果を判定"""
        # ワイド: wide_payout_map（確定配当）から精算
        if bet.bet_type == BetType.WIDE:
            pair_b = getattr(bet, "umaban_b", None)
            if pair_b is not None and hasattr(self, "wide_payout_map"):
                lo, hi = min(bet.umaban, pair_b), max(bet.umaban, pair_b)
                wide_key = (bet.race_id, lo, hi)
                if wide_key in self.wide_payout_map:
                    return float(bet.stake * self.wide_payout_map[wide_key])
            # フォールバック: 着順ベースの簡易精算
            horse = race_df[race_df["umaban"] == bet.umaban]
            if horse.empty:
                return 0.0
            finish_pos = int(horse.iloc[0]["kakuteijyuni"])
            if 1 <= finish_pos <= 3 and pair_b is not None:
                pair_horse = race_df[race_df["umaban"] == pair_b]
                if not pair_horse.empty and int(pair_horse.iloc[0]["kakuteijyuni"]) <= 3:
                    settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds
                    return float(bet.stake * settle_odds)
            return 0.0

        # 単勝: win_payout_map（確定配当）から精算
        if bet.bet_type == BetType.WIN:
            win_key = (bet.race_id, bet.umaban)
            if hasattr(self, "win_payout_map") and win_key in self.win_payout_map:
                return float(bet.stake * self.win_payout_map[win_key])
            logger.warning(
                "Win payout missing for %s umaban=%d, using odds fallback",
                bet.race_id, bet.umaban,
            )
            horse = race_df[race_df["umaban"] == bet.umaban]
            if horse.empty:
                return 0.0
            finish_pos = int(horse.iloc[0]["kakuteijyuni"])
            if finish_pos == 1:
                settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds
                return float(bet.stake * settle_odds)
            return 0.0

        # 複勝: payout_map（確定配当）から精算
        payout_key = (bet.race_id, bet.umaban)
        if hasattr(self, "payout_map") and payout_key in self.payout_map:
            return float(bet.stake * self.payout_map[payout_key])

        horse = race_df[race_df["umaban"] == bet.umaban]
        if horse.empty:
            return 0.0

        finish_pos = int(horse.iloc[0]["kakuteijyuni"])
        settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds

        if bet.bet_type == BetType.PLACE:
            if 1 <= finish_pos <= 3:
                return float(bet.stake * settle_odds)

        return 0.0
