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
    monthly_returns: dict[str, float] = field(default_factory=dict)
    bet_history: list[dict[str, Any]] = field(default_factory=list)
    n_pre_post_odds_bets: int = 0   # 発走前オッズでベットした件数
    n_fallback_odds_bets: int = 0   # フォールバック（確定オッズ）でベットした件数
    avg_edge: float = 0.0           # Value Betting 平均 edge
    min_edge: float = 0.0           # Value Betting 最小 edge
    max_edge: float = 0.0           # Value Betting 最大 edge

    @property
    def profit(self) -> float:
        return self.total_return - self.total_stake

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
    """
    payout_map: dict[tuple[str, int], float] = {}
    if payouts_df.empty:
        return payout_map
    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        for i in range(1, 6):
            umaban = row.get(f"payfukusyoumaban{i}")
            pay = row.get(f"payfukusyopay{i}")
            if pd.notna(umaban) and pd.notna(pay):
                try:
                    key = (race_id, int(umaban))
                    val = float(pay) / 100.0
                    if key not in payout_map or val > payout_map[key]:
                        payout_map[key] = val
                except (ValueError, TypeError):
                    continue
    return payout_map


def build_win_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) → odds_multiplier のマップを構築 (単勝用)。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    単勝は1着のみ払戻しがあるため、ループは1回のみ。
    """
    win_payout_map: dict[tuple[str, int], float] = {}
    if payouts_df.empty:
        return win_payout_map
    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        umaban = row.get("paytansyoumaban1")
        pay = row.get("paytansyopay1")
        if pd.notna(umaban) and pd.notna(pay):
            try:
                key = (race_id, int(umaban))
                val = float(pay) / 100.0
                win_payout_map[key] = val
            except (ValueError, TypeError):
                continue
    return win_payout_map


def build_wide_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int, int], float]:
    """payouts DataFrame から (race_id, umaban_lo, umaban_hi) → odds_multiplier のマップを構築。

    ワイド払戻は paywidekumi1-7 と paywidepay1-7 (100円あたり円) を使用。
    kumi 形式は非ゼロ埋め: "513" = 馬5+馬13, "1113" = 馬11+馬13, "15" = 馬1+馬5。
    """
    wide_payout_map: dict[tuple[str, int, int], float] = {}
    if payouts_df.empty:
        return wide_payout_map

    def _parse_kumi(kumi_str: str) -> tuple[int, int] | None:
        """非ゼロ埋め kumi を (lo, hi) にパース。馬番は 1-18 を前提。"""
        n = len(kumi_str)
        if n == 4:
            lo, hi = int(kumi_str[:2]), int(kumi_str[2:])
        elif n == 3:
            # Two possible splits: X|YZ or XY|Z
            split_a = (int(kumi_str[:1]), int(kumi_str[1:]))
            split_b = (int(kumi_str[:2]), int(kumi_str[2:]))
            # Valid horse numbers are 1-18; pick the valid split
            valid_a = all(1 <= v <= 18 for v in split_a)
            valid_b = all(1 <= v <= 18 for v in split_b)
            if valid_a and not valid_b:
                lo, hi = split_a
            elif valid_b and not valid_a:
                lo, hi = split_b
            elif valid_a:
                # Both valid -- use first split (X|YZ) as convention
                lo, hi = split_a
            else:
                return None
        elif n == 2:
            lo, hi = int(kumi_str[:1]), int(kumi_str[1:])
        else:
            return None
        return (min(lo, hi), max(lo, hi))

    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        for i in range(1, 8):
            kumi = row.get(f"paywidekumi{i}")
            pay = row.get(f"paywidepay{i}")
            if pd.notna(kumi) and pd.notna(pay) and str(kumi).strip():
                try:
                    parsed = _parse_kumi(str(kumi).strip())
                    if parsed is None:
                        continue
                    umaban_lo, umaban_hi = parsed
                    val = float(pay) / 100.0
                    key = (race_id, umaban_lo, umaban_hi)
                    if key not in wide_payout_map or val > wide_payout_map[key]:
                        wide_payout_map[key] = val
                except (ValueError, TypeError):
                    continue
    return wide_payout_map


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

        if betting_mode == "kelly":
            from betting.drawdown_controller import DrawdownController
            from betting.stake_calculator import StakeCalculator

            self._race_predictor = RacePredictor(
                models,
                stake_calculator=StakeCalculator(),
                dd_controller=DrawdownController(peak_bankroll=initial_bankroll),
            )
        else:
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
            for _, r in final_odds_df.iterrows():
                key = (str(r["race_id"]), int(r["umaban"]))
                if pd.notna(r.get("fukuoddslow")):
                    final_odds_map[key] = float(r["fukuoddslow"])

        # 確定配当マップを構築（精算用。実際の払戻金額を使用）
        payouts_df = load_payouts(self.store, start, end)
        self.payout_map = build_payout_map(payouts_df)
        logger.info("Loaded payout map: %d entries", len(self.payout_map))

        # ワイド払戻マップを構築（精算用）
        self.wide_payout_map = build_wide_payout_map(payouts_df)
        logger.info("Loaded wide payout map: %d entries", len(self.wide_payout_map))

        # 単勝払戻マップを構築（精算用）
        self.win_payout_map = build_win_payout_map(payouts_df)
        logger.info("Loaded win payout map: %d entries", len(self.win_payout_map))

        feat_df = feat_engine.build_all(
            race_df, entry_df, pre_post_odds, odds_ts_df=odds_ts_df, store=self.store
        )
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # 単勝確定オッズマップを構築（精算用。tanodds列を使用）
        final_win_odds_map: dict[tuple[str, int], float] = {}
        if not feat_df.empty and "tanodds" in feat_df.columns:
            for _, r in feat_df.iterrows():
                key = (str(r["race_id"]), int(r["umaban"]))
                if pd.notna(r.get("tanodds")):
                    final_win_odds_map[key] = float(r["tanodds"])

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

        # JRAフィルタ: NARレース (jyocd 30以上) を除外
        if "jyocd" in feat_df.columns:
            jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
            before_count = len(feat_df)
            feat_df = feat_df[jyocd_int.between(1, 10)]
            after_count = len(feat_df)
            if before_count > after_count:
                logger.info(
                    "JRA filter: excluded %d NAR entries (jyocd >= 30), %d remaining",
                    before_count - after_count,
                    after_count,
                )

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
        _pace_cols = [
            c
            for c in ["pace_aptitude", "front_pace_wr", "closing_pace_wr"]
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
        diag_logger = DiagnosticLogger()
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_dd = 0.0
        bet_history: list[dict[str, Any]] = []
        n_pre_post_odds_bets = 0
        n_fallback_odds_bets = 0

        # RegimeDetector 用: 直近200レースの統計を蓄積
        recent_stats_list: list[dict[str, float]] = []

        for race_id in race_ids:
            race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
            if race_df_single.empty:
                continue

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
            for _, r in _valid.iterrows():
                _top3.append(
                    {
                        "umaban": int(r["umaban"]),
                        "bamei": str(r.get("bamei", "")) if pd.notna(r.get("bamei")) else "",
                        "kisyuryakusyo": (
                            str(r.get("kisyuryakusyo", ""))
                            if pd.notna(r.get("kisyuryakusyo"))
                            else ""
                        ),
                        "kakuteijyuni": int(r["kakuteijyuni"]),
                    }
                )

            # 事前計算済み特徴量をマージ
            hist_df_race = hist_df_all[hist_df_all["race_id"] == race_id]
            jockey_df_race = jockey_df_all[jockey_df_all["race_id"] == race_id]
            trainer_df_race = trainer_df_all[trainer_df_all["race_id"] == race_id]
            jt_df_race = jt_df_all[jt_df_all["race_id"] == race_id]

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
            edge_threshold = regime_params.get("edge_threshold", 0.03)
            if self.betting_target == "win":
                get_win = getattr(self._race_predictor, "get_win_candidates", None)
                if callable(get_win):
                    candidate_df = get_win(result_df)
                else:
                    candidate_df = self._race_predictor.get_place_candidates(
                        result_df,
                        regime_params=regime_params,
                    )
            else:
                candidate_df = self._race_predictor.get_place_candidates(
                    result_df,
                    regime_params=regime_params,
                )
            n_candidates = len(candidate_df)
            # place_selection_reason is only available in place mode
            if "place_selection_reason" in candidate_df.columns:
                candidate_reason_df = candidate_df[
                    ["race_id", "umaban", "place_selection_reason"]
                ].copy()
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
                    for _, hr in result_df.iterrows():
                        diag_logger.log_horse(
                            race_id=race_id,
                            umaban=int(hr["umaban"]),
                            p_place_pred=float(hr.get("p_place_pred", 0)),
                            e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                            ev_place=float(hr.get("ev_place", 0)),
                            fukuoddslow=float(hr.get("fukuoddslow", 0)),
                            is_bet=False,
                            p_place_corrected=float(hr.get("p_place_corrected", float("nan"))),
                            e_return_place_corrected=float(
                                hr.get("e_return_place_corrected", float("nan"))
                            ),
                            ev_place_corrected=float(hr.get("ev_place_corrected", float("nan"))),
                            ev_lower_place=float(hr.get("EV_lower_place", float("nan"))),
                            place_selection_ev=float(hr.get("place_selection_ev", float("nan"))),
                            place_selection_edge=float(
                                hr.get("place_selection_edge", float("nan"))
                            ),
                            place_selection_prob=float(
                                hr.get("place_selection_prob", float("nan"))
                            ),
                            place_bucket_multiplier=float(
                                hr.get("place_bucket_multiplier", float("nan"))
                            ),
                            place_gate_score=float(hr.get("place_gate_score", float("nan"))),
                            place_gate_pass=bool(hr.get("place_gate_pass", False)),
                            place_gate_rank=float(hr.get("place_gate_rank", float("nan"))),
                            place_gate_score_gap=float(
                                hr.get("place_gate_score_gap", float("nan"))
                            ),
                            market_condition_score=float(
                                hr.get("market_condition_score", float("nan"))
                            ),
                            aggressive_strength=float(hr.get("aggressive_strength", float("nan"))),
                            aggressive_tier=(
                                str(hr.get("aggressive_tier"))
                                if pd.notna(hr.get("aggressive_tier"))
                                else None
                            ),
                            place_selection_reason=(
                                str(hr.get("place_selection_reason"))
                                if pd.notna(hr.get("place_selection_reason"))
                                else None
                            ),
                        )
                        diag_logger.log_horse_features(hr.to_dict())
                continue

            # Bet generation
            surface_key = result_df["surface"].iloc[0]
            bets = self._race_predictor.select_bets(
                result_df, bankroll, candidates=candidate_df,
                betting_target=self.betting_target,
            )

            # v5: セグメント除外フィルタ全削除 — モデル自身がedgeを低く見積もるように改善する
            # (旧v4の14個の除外フィルタは全て削除)

            # Bet に確定オッズを設定（place/win のみ。wide は wide_payout_map で精算）
            updated_bets = []
            for bet in bets:
                if bet.bet_type == BetType.WIDE:
                    updated_bets.append(bet)
                elif bet.bet_type == BetType.WIN:
                    fo = final_win_odds_map.get((bet.race_id, bet.umaban), bet.odds)
                    updated_bets.append(replace(bet, final_odds=fo))
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
                for _, hr in result_df.iterrows():
                    diag_logger.log_horse(
                        race_id=race_id,
                        umaban=int(hr["umaban"]),
                        p_place_pred=float(hr.get("p_place_pred", 0)),
                        e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                        ev_place=float(hr.get("ev_place", 0)),
                        fukuoddslow=float(hr.get("fukuoddslow", 0)),
                        is_bet=int(hr["umaban"]) in bet_umabans,
                        p_place_corrected=float(hr.get("p_place_corrected", float("nan"))),
                        e_return_place_corrected=float(
                            hr.get("e_return_place_corrected", float("nan"))
                        ),
                        ev_place_corrected=float(hr.get("ev_place_corrected", float("nan"))),
                        ev_lower_place=float(hr.get("EV_lower_place", float("nan"))),
                        place_selection_ev=float(hr.get("place_selection_ev", float("nan"))),
                        place_selection_edge=float(
                            hr.get("place_selection_edge", float("nan"))
                        ),
                        place_selection_prob=float(
                            hr.get("place_selection_prob", float("nan"))
                        ),
                        place_bucket_multiplier=float(
                            hr.get("place_bucket_multiplier", float("nan"))
                        ),
                        place_gate_score=float(hr.get("place_gate_score", float("nan"))),
                        place_gate_pass=bool(hr.get("place_gate_pass", False)),
                        place_gate_rank=float(hr.get("place_gate_rank", float("nan"))),
                        place_gate_score_gap=float(
                            hr.get("place_gate_score_gap", float("nan"))
                        ),
                        market_condition_score=float(
                            hr.get("market_condition_score", float("nan"))
                        ),
                        aggressive_strength=float(hr.get("aggressive_strength", float("nan"))),
                        aggressive_tier=(
                            str(hr.get("aggressive_tier"))
                            if pd.notna(hr.get("aggressive_tier"))
                            else None
                        ),
                        place_selection_reason=(
                            str(hr.get("place_selection_reason"))
                            if pd.notna(hr.get("place_selection_reason"))
                            else None
                        ),
                    )
                    diag_logger.log_horse_features(hr.to_dict())

            # Settlement (BacktestEngine 固有)
            for bet in bets:
                bet_result = self._settle_bet(bet, result_df)
                bankroll -= bet.stake
                if bet_result > 0:
                    bankroll += bet_result

                # A4: DD Controller に ROI 比 (bet_result / stake) をフィードバック
                if self._race_predictor.dd_ctrl is not None:
                    roi = bet_result / bet.stake if bet.stake > 0 else 0.0
                    self._race_predictor.dd_ctrl.update(bankroll, roi)

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
                        # --- Win-specific fields (D-09, RPT-01) ---
                        "win_selection_ev": (
                            float(horse_rows.iloc[0].get("win_selection_ev", 0.0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "win_selection_edge": (
                            float(horse_rows.iloc[0].get("win_selection_edge", 0.0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "win_selection_prob": (
                            float(horse_rows.iloc[0].get("win_selection_prob", 0.0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "win_gate_score": (
                            float(horse_rows.iloc[0].get("win_gate_score", float("nan")))
                            if not horse_rows.empty
                            else float("nan")
                        ),
                        "conformal_confidence_score": (
                            float(horse_rows.iloc[0].get("conformal_confidence_score", 0.0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "tanoddslow": (
                            float(horse_rows.iloc[0].get("tanoddslow", 0.0))
                            if not horse_rows.empty
                            else 0.0
                        ),
                        "regime": str(regime),
                    }
                )

                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_dd = max(max_dd, dd)

            # 統計を蓄積 (発走前情報のみ)
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

        return BacktestResult(
            total_bets=total_bets,
            total_stake=total_stake,
            total_return=total_return,
            winning_bets=winning_bets,
            total_roi=total_return / total_stake if total_stake > 0 else 0.0,
            max_drawdown=max_dd,
            final_bankroll=bankroll,
            monthly_returns={},
            bet_history=bet_history,
            n_pre_post_odds_bets=n_pre_post_odds_bets,
            n_fallback_odds_bets=n_fallback_odds_bets,
            avg_edge=result_data.get("avg_edge", 0.0),
            min_edge=result_data.get("min_edge", 0.0),
            max_edge=result_data.get("max_edge", 0.0),
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
        ev_col = "ev_place_corrected" if "ev_place_corrected" in race_df.columns else "ev_place"
        if ev_col in race_df.columns and "fukuoddslow" in race_df.columns:
            candidates = race_df[race_df[ev_col].fillna(0) >= ev_threshold].copy()
            # ev_col 降順でソートし、上位 max_bets 頭のみベット
            candidates = candidates.nlargest(max_bets, ev_col)

            for _, row in candidates.iterrows():
                stake = 100.0  # 固定100円ベット (簡易版)
                if bankroll >= stake:
                    bets.append(
                        Bet(
                            race_id=row["race_id"],
                            umaban=int(row["umaban"]),
                            bet_type=BetType.PLACE,
                            odds=float(row["fukuoddslow"]),
                            final_odds=float(row["fukuoddslow"]),  # レガシー: 同一オッズ
                            ev_lower_corrected=float(row.get(ev_col, 0)),
                            stake=stake,
                        )
                    )

        return bets

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
            # D-04: フォールバック — 着順ベース
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
