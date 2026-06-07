"""バックテストエンジン (§13)

学習済みモデルで履歴データをシミュレーションし、投資成績を評価。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from backtest.diagnostic_logger import DiagnosticLogger
from backtest.parameter_freeze_protocol import (
    ParameterFreezeProtocol,
    verify_strategy_manifest,
)
from backtest.race_predictor import RacePredictor
from betting.odds_band_filter import OddsBandFilter
from betting.payout_maps import build_payout_map, build_wide_payout_map, build_win_payout_map
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
from domain.types import POST_RACE_COLS, RegimeState
from models.regime_detector import calc_favorite_implied_prob, calc_odds_skewness

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
    bet_history: list[dict[str, Any]] = field(default_factory=list)
    n_pre_post_odds_bets: int = 0  # 発走前オッズでベットした件数
    n_fallback_odds_bets: int = 0  # フォールバック（確定オッズ）でベットした件数
    avg_edge: float = 0.0  # Value Betting 平均 edge
    min_edge: float = 0.0  # Value Betting 最小 edge
    max_edge: float = 0.0  # Value Betting 最大 edge
    # Phase 11: Bet selection filter exclusion stats
    n_collapsed_skipped: int = 0  # D-11: COLLAPSED regime skip count
    n_ev_excluded: int = 0  # D-01: EV filter exclusion count
    n_odds_band_excluded: int = 0  # D-06: OddsBandFilter exclusion count
    n_win_ev_odds_excluded: int = 0  # select_bets後 EV/Oddsフィルター除外 bet数
    n_win_stake_increased: int = 0  # tiered stake増額 bet数
    total_win_stake_increased: float = 0.0  # tiered stake増額総額
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
        return {k: (ret / stk if stk > 0 else 0.0) for k, (stk, ret) in monthly.items()}

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
                f"  Edge: avg={self.avg_edge:.4f}, min={self.min_edge:.4f}, max={self.max_edge:.4f}"
            )
        return "\n".join(lines)


@dataclass
class BacktestPreparedData:
    """P4: fold-level pre-computed data (model-independent).

    prepare_data() は常にこのオブジェクトを返す（空データでも None は返さない）。
    run() 側で len(race_ids) == 0 をチェックして早期リターンする。

    Attributes:
        race_ids: 対象レースID配列 (空の場合あり)
        feat_df: FeatureEngine + バッチ特徴量適用済みの全レース DataFrame
        jockey_df_all: JockeyContextFeatures の全計算結果
        trainer_df_all: TrainerContextFeatures の全計算結果
        jt_df_all: JockeyTrainerComboFeatures の全計算結果
        final_odds_map: (race_id, umaban) → 確定複勝オッズ
        closing_win_odds_map: (race_id, umaban) → 確定単勝オッズ
        payout_map: (race_id, umaban) → 複勝配当倍率
        win_payout_map: (race_id, umaban) → 単勝配当倍率
        wide_payout_map: (race_id, umaban_lo, umaban_hi) → ワイド配当倍率
    """

    race_ids: np.ndarray
    feat_df: pd.DataFrame
    jockey_df_all: pd.DataFrame
    trainer_df_all: pd.DataFrame
    jt_df_all: pd.DataFrame
    final_odds_map: dict[tuple[str, int], float]
    closing_win_odds_map: dict[tuple[str, int], float]
    payout_map: dict[tuple[str, int], float]
    win_payout_map: dict[tuple[str, int], float]
    wide_payout_map: dict[tuple[str, int, int], float]

    @classmethod
    def empty(cls) -> BacktestPreparedData:
        """空データのインスタンスを返す (None は返さない設計)."""
        return cls(
            race_ids=np.array([]),
            feat_df=pd.DataFrame(),
            jockey_df_all=pd.DataFrame(),
            trainer_df_all=pd.DataFrame(),
            jt_df_all=pd.DataFrame(),
            final_odds_map={},
            closing_win_odds_map={},
            payout_map={},
            win_payout_map={},
            wide_payout_map={},
        )


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
    for key, group in df.groupby(group_col, observed=True):
        groups[str(key)] = group
    empty_count = sum(1 for g in groups.values() if g.empty)
    if empty_count > 0:
        logger.warning("[%s] %d empty groups in %d total", name, empty_count, len(groups))
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info("[%s] %d groups, %d rows, %.1f MB", name, len(groups), len(df), mem_mb)
    return groups


def _optional_float(value: Any) -> float | None:
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    float_value = _optional_float(value)
    if float_value is None:
        return None
    return int(float_value)


def _optional_bool(value: Any) -> bool | None:
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _horse_win_diagnostic_kwargs(row: Any) -> dict[str, Any]:
    return {
        "is_actual_bet": bool(getattr(row, "is_actual_bet", False)),
        "p_win_pred": _optional_float(getattr(row, "p_win_pred", None)),
        "p_win_corrected": _optional_float(getattr(row, "p_win_corrected", None)),
        "p_win_final": _optional_float(getattr(row, "p_win_final", None)),
        "e_return_win_pred": _optional_float(getattr(row, "e_return_win_pred", None)),
        "e_return_win_corrected": _optional_float(getattr(row, "e_return_win_corrected", None)),
        "win_selection_ev": _optional_float(getattr(row, "win_selection_ev_raw", None)),
        "win_selection_ev_tail_calibrated": _optional_float(
            getattr(row, "win_selection_ev_tail_calibrated", None)
        ),
        "win_selection_edge": _optional_float(getattr(row, "win_selection_edge", None)),
        "win_selection_prob": _optional_float(getattr(row, "win_selection_prob", None)),
        "win_gate_score": _optional_float(getattr(row, "win_gate_score", None)),
        "win_gate_pass": _optional_bool(getattr(row, "win_gate_pass", None)),
        "win_gate_odds_score": _optional_float(getattr(row, "win_gate_odds_score", None)),
        "win_gate_prob_score": _optional_float(getattr(row, "win_gate_prob_score", None)),
        "win_gate_edge_score": _optional_float(getattr(row, "win_gate_edge_score", None)),
        "win_gate_edge_odds_score": _optional_float(getattr(row, "win_gate_edge_odds_score", None)),
        "p_market_win_raw": _optional_float(getattr(row, "p_market_win_raw", None)),
        "p_market_win_norm": _optional_float(getattr(row, "p_market_win_norm", None)),
        "win_market_residual": _optional_float(getattr(row, "win_market_residual", None)),
        "win_market_logit_edge": _optional_float(getattr(row, "win_market_logit_edge", None)),
        "win_market_prob_ratio": _optional_float(getattr(row, "win_market_prob_ratio", None)),
        "win_market_value_ratio": _optional_float(getattr(row, "win_market_value_ratio", None)),
        "win_market_selection_score": _optional_float(
            getattr(row, "win_market_selection_score", None)
        ),
        "win_profit_score": _optional_float(getattr(row, "win_profit_score", None)),
        "win_profit_selector_pass": _optional_bool(getattr(row, "win_profit_selector_pass", None)),
        "win_profit_rank": _optional_float(getattr(row, "win_profit_rank", None)),
        "win_profit_stake_scale": _optional_float(getattr(row, "win_profit_stake_scale", None)),
        "win_profit_reason": _optional_str(getattr(row, "win_profit_reason", None)),
        "win_late_odds_drop_z": _optional_float(getattr(row, "win_late_odds_drop_z", None)),
        "win_late_odds_drop_weight": _optional_float(
            getattr(row, "win_late_odds_drop_weight", None)
        ),
        "win_ev_tail_pressure": _optional_float(getattr(row, "win_ev_tail_pressure", None)),
        "win_ev_tail_penalty_weight": _optional_float(
            getattr(row, "win_ev_tail_penalty_weight", None)
        ),
        "win_log_odds": _optional_float(getattr(row, "win_log_odds", None)),
        "win_log_odds_penalty": _optional_float(getattr(row, "win_log_odds_penalty", None)),
        "win_model_prob_rank": _optional_float(getattr(row, "win_model_prob_rank", None)),
        "win_prob_rank_bonus": _optional_float(getattr(row, "win_prob_rank_bonus", None)),
        "win_market_risk_penalty": _optional_float(getattr(row, "win_market_risk_penalty", None)),
        "risk_flags": _optional_str(getattr(row, "risk_flags", None)),
        "tanodds": _optional_float(getattr(row, "tanodds", None)),
        "closing_win_odds": _optional_float(getattr(row, "closing_win_odds", None)),
        "clv": _optional_float(getattr(row, "clv", None)),
        "final_odds": _optional_float(getattr(row, "final_odds", None)),
        "stake": _optional_float(getattr(row, "stake", None)),
        "result": _optional_float(getattr(row, "result", None)),
        "excluded_reason": _optional_str(getattr(row, "excluded_reason", None)),
        "filter_pass_flags": _optional_str(getattr(row, "filter_pass_flags", None)),
        "candidate_count_before_filter": _optional_int(
            getattr(row, "candidate_count_before_filter", None)
        ),
        "candidate_count_after_filter": _optional_int(
            getattr(row, "candidate_count_after_filter", None)
        ),
        "selected_rank_by_p_win_final": _optional_float(
            getattr(row, "selected_rank_by_p_win_final", None)
        ),
        "selected_rank_by_win_selection_ev": _optional_float(
            getattr(row, "selected_rank_by_win_selection_ev", None)
        ),
        "selected_rank_by_win_market_logit_edge": _optional_float(
            getattr(row, "selected_rank_by_win_market_logit_edge", None)
        ),
        "selected_rank_by_win_market_score": _optional_float(
            getattr(row, "selected_rank_by_win_market_score", None)
        ),
    }


def _annotate_actual_bets(
    result_df: pd.DataFrame,
    settlements: list[tuple[Bet, float]],
) -> pd.DataFrame:
    annotated = result_df.copy()
    annotated["is_actual_bet"] = False
    annotated["stake"] = np.nan
    annotated["result"] = np.nan
    annotated["final_odds"] = np.nan
    annotated["clv"] = np.nan

    for bet, bet_result in settlements:
        if bet.stake <= 0:
            continue
        mask = pd.to_numeric(annotated["umaban"], errors="coerce").eq(int(bet.umaban))
        annotated.loc[mask, "is_actual_bet"] = True
        annotated.loc[mask, "stake"] = float(bet.stake)
        annotated.loc[mask, "result"] = float(bet_result)
        annotated.loc[mask, "final_odds"] = float(
            bet.final_odds if bet.final_odds > 0 else bet.odds
        )
        if "tanodds" in annotated.columns:
            pre_odds = pd.to_numeric(annotated.loc[mask, "tanodds"], errors="coerce")
            if "closing_win_odds" in annotated.columns:
                close_odds = pd.to_numeric(
                    annotated.loc[mask, "closing_win_odds"],
                    errors="coerce",
                )
            else:
                close_odds = pd.to_numeric(annotated.loc[mask, "final_odds"], errors="coerce")
            valid_clv = pre_odds.gt(0.0) & close_odds.gt(0.0)
            annotated.loc[mask, "clv"] = np.where(
                valid_clv,
                close_odds / pre_odds - 1.0,
                np.nan,
            )

    return annotated


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
        manifest_path: Path | None = None,
        preloaded_race_df: pd.DataFrame | None = None,
        preloaded_entry_df: pd.DataFrame | None = None,
        preloaded_final_odds_df: pd.DataFrame | None = None,
        preloaded_payouts_df: pd.DataFrame | None = None,
        preloaded_odds_ts: pd.DataFrame | None = None,
        min_bets_per_year: int = 1000,
        min_win_ev: float = 0.0,
        min_win_odds: float = 0.0,
        win_ev_stake_threshold: float = 0.0,
        win_ev_stake_multiplier: float = 1.0,
    ) -> None:
        if betting_mode not in ("flat", "kelly"):
            raise ValueError(f"betting_mode must be 'flat' or 'kelly', got '{betting_mode}'")
        if betting_target not in ("win", "place", "wide"):
            raise ValueError(
                f"betting_target must be 'win', 'place', or 'wide', got '{betting_target}'"
            )
        self.models = models
        self.initial_bankroll = initial_bankroll
        self.store = store or ParquetStore()
        self.betting_mode = betting_mode
        self.diag_prefix = diag_prefix
        self.betting_target = betting_target
        self.strategy_params = strategy_params
        self._manifest_path = manifest_path
        self._pfp: ParameterFreezeProtocol | None = None
        self._preloaded_odds_ts = preloaded_odds_ts  # P1: odds時系列データ受け渡し
        self._preloaded_race_df = preloaded_race_df  # P2: fold単位共有ロード
        self._preloaded_entry_df = preloaded_entry_df
        self._preloaded_final_odds_df = preloaded_final_odds_df
        self._preloaded_payouts_df = preloaded_payouts_df
        self._min_bets_per_year = min_bets_per_year
        # 意思決定EV/Oddsフィルター + tiered stake sizing
        self._min_win_ev = min_win_ev
        self._min_win_odds = min_win_odds
        self._win_ev_stake_threshold = win_ev_stake_threshold
        self._win_ev_stake_multiplier = win_ev_stake_multiplier
        # Phase 43.5: 内部エンジンフラグ — OddsBandFilterキャリブレーションを
        # スキップし、再帰的な _generate_training_bet_history 呼び出しを防止する。
        self._skip_odds_band_calibration: bool = False

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
                betting_target=betting_target,
            )
        else:
            self._race_predictor = RacePredictor(models, betting_target=betting_target)

    def _generate_training_bet_history(
        self,
    ) -> list[dict[str, Any]] | None:
        """デフォルトパラメータでトレーニング期間バックテストを実行し、bet_historyを生成。

        D-05/D-06: run()内でtraining_bet_historyがNoneの場合に自動呼び出し。
        D-07: トレーニング期間はself.models.train_periodから取得 (test_start/test_endは使用しない)。
        Phase 43.5 FIX: 内部エンジンは同じbetting_targetを使用し、
        _skip_odds_band_calibration=Trueで再帰calibrate防止。
        """
        try:
            # D-07: models.train_periodからトレーニング期間を取得
            # TrainedModelsV5.train_period = (train_start, train_end)
            train_start, train_end = self.models.train_period
            # Phase 43.5 FIX: 内部エンジンは同じbetting_targetで生成し、
            # _skip_odds_band_calibration=TrueでOddsBandFilterキャリブレーションを
            # スキップ (再帰防止)。これによりwin-onlyモデルでもbetが生成される。
            inner_engine = BacktestEngine(
                models=self.models,
                initial_bankroll=self.initial_bankroll,
                store=self.store,
                betting_mode=self.betting_mode,
                diag_prefix=f"{self.diag_prefix}_train",
                betting_target=self.betting_target,
                strategy_params=None,  # default OddsBandFilter (roi_threshold=1.0)
            )
            inner_engine._skip_odds_band_calibration = True  # noqa: SLF001
            train_result = inner_engine.run(train_start, train_end)
            logger.info(
                "自動training_bet_history生成完了: %d bets, ROI=%.1f%% (%s ~ %s)",
                train_result.total_bets,
                train_result.total_roi * 100,
                train_start,
                train_end,
            )
            return train_result.bet_history
        except Exception as e:
            logger.warning(
                "自動training_bet_history生成失敗: %s — OddsBandFilterキャリブレーションスキップ",
                e,
            )
            return None

    def _calibrate_odds_band_filter(
        self,
        training_bet_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        """OddsBandFilterキャリブレーションを実行。

        D-05/D-06/D-07: training_bet_historyがNoneの場合はengine内で自動生成する。
        キャリブレーション後のtraining_bet_historyを返す (後続処理で使用しないが、
        テストで呼び出しを検証可能にするため)。

        Args:
            training_bet_history: 外部から提供されたbet_history。Noneの場合は自動生成。

        Returns:
            キャリブレーションに使用したtraining_bet_history、またはNone
        """
        # Phase 43.5: 内部エンジンからの再帰呼び出しをスキップ
        if self._skip_odds_band_calibration:
            return None
        if self._odds_band_filter is not None:
            if training_bet_history is None:
                # D-05: run()内で自動的にtraining_bet_historyを生成
                # D-06: engine.py内部で完結させる
                # D-07: トレーニング期間はmodels.train_periodから取得
                training_bet_history = self._generate_training_bet_history()
            if training_bet_history:
                self._odds_band_filter.calibrate(training_bet_history)
        return training_bet_history

    def _verify_pfp(self) -> None:
        """D-03(2): PFP verify -- OOS期間終了時のモデル不変性確認。

        失敗時はRuntimeError(D-04)。run()の各returnパスで呼び出す。
        """
        if self._pfp is not None:
            pfp_result = self._pfp.verify()
            if not pfp_result["passed"]:
                raise RuntimeError(pfp_result["message"])  # D-04: 即時停止
            logger.info("PFP verification passed: %s", pfp_result["message"])

    # ------------------------------------------------------------------
    # P4: prepare_data — model-independent data + feature preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_data(
        store: ParquetStore,
        betting_target: str,
        test_start: str,
        test_end: str,
        preloaded_race_df: pd.DataFrame | None = None,
        preloaded_entry_df: pd.DataFrame | None = None,
        preloaded_final_odds_df: pd.DataFrame | None = None,
        preloaded_payouts_df: pd.DataFrame | None = None,
        preloaded_odds_ts: pd.DataFrame | None = None,
    ) -> BacktestPreparedData:
        """P4: モデル非依存のデータ準備 + 特徴量生成.

        run() の前半 (データロード → 特徴量生成) を抽出したメソッド。
        self.models / _shadow_flags に依存しないコードのみを含む。
        常に BacktestPreparedData を返す (空データでも None は返さない)。

        Args:
            store: ParquetStore インスタンス
            betting_target: "win" | "place" | "wide"
            test_start: テスト開始日 (YYYY-MM-DD)
            test_end: テスト終了日 (YYYY-MM-DD)
            preloaded_race_df: fold単位で共有済みのレース DataFrame (省略時はDB読み込み)
            preloaded_entry_df: fold単位で共有済みの出走 DataFrame
            preloaded_final_odds_df: fold単位で共有済みの確定オッズ DataFrame
            preloaded_payouts_df: fold単位で共有済みの配当 DataFrame
            preloaded_odds_ts: fold単位で共有済みのオッズ時系列 DataFrame

        Returns:
            BacktestPreparedData (空データ時は race_ids が空配列)
        """
        _empty = BacktestPreparedData.empty

        # 1. データロード
        start = test_start.replace("-", "")
        end = test_end.replace("-", "")
        race_df = (
            preloaded_race_df.copy()
            if preloaded_race_df is not None
            else load_races(store, start, end)
        )
        entry_df = (
            preloaded_entry_df.copy()
            if preloaded_entry_df is not None
            else load_entries(store, start, end)
        )
        final_odds_df = (
            preloaded_final_odds_df.copy()
            if preloaded_final_odds_df is not None
            else load_odds_snapshots(store, start, end)
        )

        if race_df.empty:
            logger.warning("No races found in %s ~ %s [prepare_data]", test_start, test_end)
            return _empty()

        if "jyocd" in race_df.columns:
            jyocd_int = pd.to_numeric(race_df["jyocd"], errors="coerce")
            jra_race_ids = race_df.loc[jyocd_int.between(1, 10), "race_id"].drop_duplicates()
            race_df = race_df[race_df["race_id"].isin(jra_race_ids)].copy()
            entry_df = entry_df[entry_df["race_id"].isin(jra_race_ids)].copy()
            final_odds_df = final_odds_df[final_odds_df["race_id"].isin(jra_race_ids)].copy()

        # 2. 特徴量生成
        from db.odds_extractor import extract_pre_post_odds

        # P1: odds時系列データ
        if preloaded_odds_ts is not None:
            odds_ts_df = preloaded_odds_ts
            s_dt = pd.Timestamp(start)
            e_dt = pd.Timestamp(end)
            if "race_date" in odds_ts_df.columns:
                mask = (odds_ts_df["race_date"] >= s_dt) & (odds_ts_df["race_date"] <= e_dt)
                odds_ts_df = odds_ts_df[mask]
            logger.debug(
                "Using preloaded odds_ts (%d rows for %s ~ %s) [prepare_data]",
                len(odds_ts_df),
                start,
                end,
            )
        else:
            odds_ts_df = load_odds_time_series_range(store, start, end)
        if not odds_ts_df.empty:
            odds_ts_df = odds_ts_df[odds_ts_df["race_id"].isin(race_df["race_id"])].copy()

        # 発走前オッズの抽出
        if odds_ts_df.empty:
            logger.warning(
                "No time-series odds data for %s ~ %s [prepare_data]", test_start, test_end
            )
            return _empty()

        if "hassotime" not in race_df.columns:
            logger.warning("hassotime column missing [prepare_data]")
            return _empty()

        pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
        if pre_post_odds.empty:
            logger.warning(
                "extract_pre_post_odds returned empty for %s ~ %s [prepare_data]",
                test_start,
                test_end,
            )
            return _empty()

        # 確定オッズマップを構築
        final_odds_map: dict[tuple[str, int], float] = {}
        if not final_odds_df.empty:
            _odds = final_odds_df.dropna(subset=["fukuoddslow"])
            if not _odds.empty:
                for (race_id, umaban), odds in _odds.set_index(["race_id", "umaban"])[
                    "fukuoddslow"
                ].items():
                    final_odds_map[(str(race_id), int(umaban))] = float(odds)
        closing_win_odds_map: dict[tuple[str, int], float] = {}
        if not final_odds_df.empty and {"race_id", "umaban", "tanodds"}.issubset(
            final_odds_df.columns
        ):
            _win_odds = final_odds_df.dropna(subset=["tanodds"])
            if not _win_odds.empty:
                for (race_id, umaban), odds in _win_odds.set_index(["race_id", "umaban"])[
                    "tanodds"
                ].items():
                    closing_win_odds_map[(str(race_id), int(umaban))] = float(odds)

        # 確定配当マップを構築
        payouts_df = (
            preloaded_payouts_df.copy()
            if preloaded_payouts_df is not None
            else load_payouts(store, start, end)
        )

        needs_place = betting_target in ("place", "wide")
        needs_win = betting_target in ("win", "wide")
        needs_wide = betting_target == "wide"

        payout_map = build_payout_map(payouts_df) if needs_place else {}
        if needs_place:
            logger.info("Loaded payout map: %d entries [prepare_data]", len(payout_map))

        win_payout_map = build_win_payout_map(payouts_df) if needs_win else {}
        if needs_win:
            logger.info("Loaded win payout map: %d entries [prepare_data]", len(win_payout_map))

        wide_payout_map = build_wide_payout_map(payouts_df) if needs_wide else {}
        if needs_wide:
            logger.info("Loaded wide payout map: %d entries [prepare_data]", len(wide_payout_map))

        # FeatureBuilder: 13エンリッチメントモジュールを一括実行 (Phase 52 D-10)
        from features.feature_builder import FeatureBuilder

        builder = FeatureBuilder(store=store)
        build_result = builder.build_for_training(
            race_df,
            entry_df,
            pre_post_odds,
            odds_ts_df=odds_ts_df,
            preserve_columns=["kakuteijyuni", "confirmed_odds"],
        )
        feat_df = build_result.frame

        race_ids = feat_df["race_id"].unique()

        # ワイドペア専用オッズ
        if betting_target == "wide":
            wide_odds_df = load_wide_odds(store, start, end)
            if wide_odds_df is not None and not wide_odds_df.empty:
                _wide = wide_odds_df[["race_id", "kumi", "oddslow"]].dropna(subset=["oddslow"])
                if not _wide.empty:
                    wide_pivot = _wide.pivot_table(
                        index="race_id",
                        columns="kumi",
                        values="oddslow",
                    )
                    new_cols = []
                    for c in wide_pivot.columns:
                        lo = int(c[:2])
                        hi = int(c[2:])
                        new_cols.append(f"wide_odds_{lo}_{hi}")
                    wide_pivot.columns = new_cols
                    wide_pivot = wide_pivot.reset_index()
                    feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")
                    logger.info("Merged wide odds: %d pair-columns", len(wide_pivot.columns) - 1)
        else:
            logger.info("Skipping wide odds pivot for betting_target=%s", betting_target)

        # Safety check: NAR filter
        if "jyocd" in feat_df.columns:
            jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
            nar_count = (~jyocd_int.between(1, 10)).sum()
            if nar_count > 0:
                logger.warning(
                    "NAR entries leaked into feat_df: %d (feature pipeline bug?)",
                    int(nar_count),
                )
                feat_df = feat_df[jyocd_int.between(1, 10)]

        # jockey/trainer/jt は FeatureBuilder で feat_df に既にマージ済み
        # 空DataFrame を返す (BacktestPreparedData 構造は維持)
        jockey_df_all = pd.DataFrame()
        trainer_df_all = pd.DataFrame()
        jt_df_all = pd.DataFrame()

        logger.info(
            "prepare_data complete: %d races, %d features, betting_target=%s",
            len(race_ids),
            len(feat_df.columns),
            betting_target,
        )
        return BacktestPreparedData(
            race_ids=race_ids,
            feat_df=feat_df,
            jockey_df_all=jockey_df_all,
            trainer_df_all=trainer_df_all,
            jt_df_all=jt_df_all,
            final_odds_map=final_odds_map,
            closing_win_odds_map=closing_win_odds_map,
            payout_map=payout_map,
            win_payout_map=win_payout_map,
            wide_payout_map=wide_payout_map,
        )

    def run(
        self,
        test_start: str,
        test_end: str,
        training_bet_history: list[dict[str, Any]] | None = None,  # D-05
        prepared_data: BacktestPreparedData | None = None,  # P4
    ) -> BacktestResult:
        """バックテストを実行

        Args:
            test_start: テスト開始日 (YYYY-MM-DD)
            test_end: テスト終了日 (YYYY-MM-DD)
            training_bet_history: トレーニング期間のベット履歴 (OddsBandFilter キャリブレーション用)
            prepared_data: P4: fold単位で共有済みのデータ+特徴量 (None時は内部で生成)

        Returns:
            BacktestResult
        """
        # --- D-03: PFP二重検証 (SHA256 + ParameterFreezeProtocol) ---
        if self._manifest_path is not None:
            # D-03(1): SHA256再検証
            verify_strategy_manifest(self._manifest_path)
            logger.info("Manifest SHA256 verified in engine.run(): %s", self._manifest_path)

            # D-03(2): PFP freeze -- OOS期間開始時のモデルスナップショット
            self._pfp = ParameterFreezeProtocol(self.models)
            self._pfp.freeze()

        # --- P4: prepared_data 分岐 ---
        # prepared_data is not None → 共有データを使用 (copy して安全に分離)
        # prepared_data is None     → 従来通り内部でデータロード + 特徴量生成
        if prepared_data is not None:
            # P4: 空データチェック -- _verify_pfp() は run() 側で必ず呼ぶ
            if len(prepared_data.race_ids) == 0:
                logger.warning(
                    "Prepared data has no races for %s ~ %s", test_start, test_end
                )
                self._verify_pfp()
                return BacktestResult(final_bankroll=self.initial_bankroll)

            race_ids = prepared_data.race_ids
            feat_df = prepared_data.feat_df.copy()

            # --- モデル固有の track_stats で track_condition_features を補正 ---
            # prepare_data() はモデル読み込み前に実行されるため build_for_training() を使用。
            # ここで学習期間の track_stats で上書きすることで正しい z-score を復元。
            from features.feature_manifest import FeatureState as _FS

            try:
                _feat_state = _FS.from_models(self.models)
            except ValueError as e:
                raise ValueError(
                    f"Cannot run backtest: model has no track_stats. "
                    f"Re-train with the latest pipeline. Detail: {e}"
                ) from e

            from features.track_condition_features import (
                compute_race_condition_features,
                compute_track_condition_features,
            )

            feat_df = compute_track_condition_features(
                feat_df,
                track_stats=_feat_state.track_stats,
                track_month_stats=_feat_state.track_month_stats,
            )
            feat_df = compute_race_condition_features(feat_df)

            feat_groups = build_race_groups(feat_df, name="features")
            jockey_groups = build_race_groups(
                prepared_data.jockey_df_all.copy(), name="jockey"
            )
            trainer_groups = build_race_groups(
                prepared_data.trainer_df_all.copy(), name="trainer"
            )
            jt_groups = build_race_groups(
                prepared_data.jt_df_all.copy(), name="jockey_trainer"
            )
            final_odds_map = dict(prepared_data.final_odds_map)
            closing_win_odds_map = dict(prepared_data.closing_win_odds_map)
            self.payout_map = dict(prepared_data.payout_map)
            self.win_payout_map = dict(prepared_data.win_payout_map)
            self.wide_payout_map = dict(prepared_data.wide_payout_map)
            logger.info(
                "P4: Using prepared_data: %d races, %d features",
                len(race_ids),
                len(prepared_data.feat_df.columns),
            )
        else:
            # --- 既存パス: 内部でデータロード + 特徴量生成 ---
            # 1. データロード
            start = test_start.replace("-", "")
            end = test_end.replace("-", "")
            race_df = (
                self._preloaded_race_df.copy()
                if self._preloaded_race_df is not None
                else load_races(self.store, start, end)
            )
            entry_df = (
                self._preloaded_entry_df.copy()
                if self._preloaded_entry_df is not None
                else load_entries(self.store, start, end)
            )
            final_odds_df = (
                self._preloaded_final_odds_df.copy()
                if self._preloaded_final_odds_df is not None
                else load_odds_snapshots(self.store, start, end)  # 確定オッズ（精算用）
            )

            if race_df.empty:
                logger.warning(f"No races found in {test_start} ~ {test_end}")
                self._verify_pfp()
                return BacktestResult(final_bankroll=self.initial_bankroll)

            if "jyocd" in race_df.columns:
                jyocd_int = pd.to_numeric(race_df["jyocd"], errors="coerce")
                jra_race_ids = race_df.loc[jyocd_int.between(1, 10), "race_id"].drop_duplicates()
                race_df = race_df[race_df["race_id"].isin(jra_race_ids)].copy()
                entry_df = entry_df[entry_df["race_id"].isin(jra_race_ids)].copy()
                final_odds_df = final_odds_df[final_odds_df["race_id"].isin(jra_race_ids)].copy()

            # 2. 特徴量生成
            from db.odds_extractor import extract_pre_post_odds

            # P1: odds時系列データ — preloaded_odds_tsがあれば再利用、なければロード
            if self._preloaded_odds_ts is not None:
                odds_ts_df = self._preloaded_odds_ts
                s_dt = pd.Timestamp(start)
                e_dt = pd.Timestamp(end)
                if "race_date" in odds_ts_df.columns:
                    mask = (odds_ts_df["race_date"] >= s_dt) & (odds_ts_df["race_date"] <= e_dt)
                    odds_ts_df = odds_ts_df[mask]
                logger.debug(
                    "Using preloaded odds_ts (%d rows for %s ~ %s)",
                    len(odds_ts_df),
                    start,
                    end,
                )
            else:
                odds_ts_df = load_odds_time_series_range(self.store, start, end)
            if not odds_ts_df.empty:
                odds_ts_df = odds_ts_df[odds_ts_df["race_id"].isin(race_df["race_id"])].copy()

            # 発走前オッズの抽出（フォールバックなし: 時系列オッズがない場合は全レーススキップ）
            if odds_ts_df.empty:
                logger.warning(
                    "No time-series odds data for %s ~ %s, skipping all races", test_start, test_end
                )
                self._verify_pfp()
                return BacktestResult(final_bankroll=self.initial_bankroll)

            if "hassotime" not in race_df.columns:
                logger.warning(
                    "hassotime column missing, cannot extract pre-race odds, skipping all races"
                )
                self._verify_pfp()
                return BacktestResult(final_bankroll=self.initial_bankroll)

            pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
            if pre_post_odds.empty:
                logger.warning(
                    "extract_pre_post_odds returned empty for %s ~ %s, skipping all races",
                    test_start,
                    test_end,
                )
                self._verify_pfp()
                return BacktestResult(final_bankroll=self.initial_bankroll)

            # 確定オッズマップを構築（精算用。FeatureEngine の列フィルタ回避）
            final_odds_map: dict[tuple[str, int], float] = {}
            if not final_odds_df.empty:
                _odds = final_odds_df.dropna(subset=["fukuoddslow"])
                if not _odds.empty:
                    for (race_id, umaban), odds in _odds.set_index(["race_id", "umaban"])[
                        "fukuoddslow"
                    ].items():
                        final_odds_map[(str(race_id), int(umaban))] = float(odds)
            closing_win_odds_map: dict[tuple[str, int], float] = {}
            if not final_odds_df.empty and {"race_id", "umaban", "tanodds"}.issubset(
                final_odds_df.columns
            ):
                _win_odds = final_odds_df.dropna(subset=["tanodds"])
                if not _win_odds.empty:
                    for (race_id, umaban), odds in _win_odds.set_index(["race_id", "umaban"])[
                        "tanodds"
                    ].items():
                        closing_win_odds_map[(str(race_id), int(umaban))] = float(odds)

            # 確定配当マップを構築（精算用。実際の払戻金額を使用）
            # BUG-FIX: betting_target に応じて必要な払戻マップのみ構築
            payouts_df = (
                self._preloaded_payouts_df.copy()
                if self._preloaded_payouts_df is not None
                else load_payouts(self.store, start, end)
            )

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

            # FeatureBuilder: 推論パイプラインで正しい track_stats を使用
            from features.feature_builder import FeatureBuilder
            from features.feature_manifest import FeatureState

            # --- 精算列の事前保存 (entry_df から抽出) ---
            _settlement_cols = ["race_id", "umaban", "kakuteijyuni", "odds"]
            _avail = [c for c in _settlement_cols if c in entry_df.columns]
            settlement_df = entry_df[_avail].copy() if _avail else pd.DataFrame()
            if "odds" in settlement_df.columns:
                settlement_df = settlement_df.rename(columns={"odds": "confirmed_odds"})

            # --- FeatureState 構築 (フォールバック禁止) ---
            feat_state = FeatureState.from_models(self.models)

            builder = FeatureBuilder(store=self.store)
            build_result = builder.build_for_inference(
                race_df,
                entry_df,
                pre_post_odds,
                feat_state,
                odds_ts_df=odds_ts_df,
            )
            feat_df = build_result.frame

            # --- 精算列の結合 (race_id + umaban キー) ---
            # build_for_inference は POST_RACE 列を削除するが、
            # モックや旧パスで残っている場合は上書きする
            if not settlement_df.empty:
                _dup_cols = [c for c in settlement_df.columns if c in feat_df.columns and c not in ("race_id", "umaban")]
                if _dup_cols:
                    feat_df = feat_df.drop(columns=_dup_cols)
                feat_df = feat_df.merge(settlement_df, on=["race_id", "umaban"], how="left")

            race_ids = feat_df["race_id"].unique()

            # ワイドペア専用オッズを pivot して特徴量にマージ（WideJointPairBuilder 用）
            if self.betting_target == "wide":
                wide_odds_df = load_wide_odds(self.store, start, end)
                if wide_odds_df is not None and not wide_odds_df.empty:
                    _wide = wide_odds_df[["race_id", "kumi", "oddslow"]].dropna(subset=["oddslow"])
                    if not _wide.empty:
                        wide_pivot = _wide.pivot_table(
                            index="race_id",
                            columns="kumi",
                            values="oddslow",
                        )
                        new_cols = []
                        for c in wide_pivot.columns:
                            lo = int(c[:2])
                            hi = int(c[2:])
                            new_cols.append(f"wide_odds_{lo}_{hi}")
                        wide_pivot.columns = new_cols
                        wide_pivot = wide_pivot.reset_index()
                        feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")
                        logger.info(
                            "Merged wide odds: %d pair-columns",
                            len(wide_pivot.columns) - 1,
                        )
            else:
                logger.info("Skipping wide odds pivot for betting_target=%s", self.betting_target)

            # Safety check: verify feature generation did not introduce NAR entries
            if "jyocd" in feat_df.columns:
                jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
                nar_count = (~jyocd_int.between(1, 10)).sum()
                if nar_count > 0:
                    logger.warning(
                        "NAR entries leaked into feat_df: %d (feature pipeline bug?)",
                        int(nar_count),
                    )
                    feat_df = feat_df[jyocd_int.between(1, 10)]

            # 5. レースごとにシミュレーション (推論は RacePredictor に委譲)
            # jockey/trainer/jt は FeatureBuilder で feat_df に既にマージ済み
            feat_groups = build_race_groups(feat_df, name="features")
            jockey_groups: dict[str, pd.DataFrame] = {}
            trainer_groups: dict[str, pd.DataFrame] = {}
            jt_groups: dict[str, pd.DataFrame] = {}

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
        n_win_ev_odds_excluded = 0
        n_win_stake_increased = 0
        total_win_stake_increased = 0.0

        # D-05/D-06/D-07: OddsBandFilter キャリブレーション
        # training_bet_historyがNoneの場合、engine内で自動生成する
        training_bet_history = self._calibrate_odds_band_filter(training_bet_history)

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
                            str(r.kisyuryakusyo) if pd.notna(r.kisyuryakusyo) else ""
                        ),
                        "kakuteijyuni": int(r.kakuteijyuni),
                    }
                )

            # 事前計算済み特徴量をマージ (groupby dict O(1) lookup)
            jockey_df_race = jockey_groups.get(race_id)
            trainer_df_race = trainer_groups.get(race_id)
            jt_df_race = jt_groups.get(race_id)

            # M3 fix: POST_RACE 列を predict() に渡さない
            predict_df = race_df_single.drop(
                columns=[c for c in POST_RACE_COLS if c in race_df_single.columns],
                errors="ignore",
            )
            # RacePredictor に委譲
            # D-11: hist_features=None — 既に feat_df に事前マージ済み (二重マージ回避)
            result_df = self._race_predictor.predict(
                predict_df,
                hist_features=None,
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
            # TODO: Regime動的に戻す場合はコメントアウト解除
            # recent_stats_df = pd.DataFrame(recent_stats_list[-200:])
            # if len(recent_stats_df) >= self.models.regime_detector.cfg.min_samples:
            #     regime = self.models.regime_detector.detect(recent_stats_df)
            # else:
            #     regime = self.models.regime_detector.current_regime
            regime = RegimeState.AGGRESSIVE  # TODO: Regime動的に戻す場合はコメントアウト解除
            regime_params = self.models.regime_detector.get_strategy_params(regime)
            # D-11: レジーム別 fractional_kelly を StakeCalculator に注入
            if self._race_predictor.stake_calc is not None:
                fk = float(regime_params.get("fractional_kelly", 0.5))
                self._race_predictor.stake_calc.fractional_kelly = fk
            edge_threshold = regime_params.get("edge_threshold", 0.03)

            # D-11: 統計を蓄積 (COLLAPSEDスキップ前 — レジーム遷移に必要, Pitfall 3)
            row_data = result_df.iloc[0] if not result_df.empty else {}
            recent_stats_list.append(
                {
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
                    "overround_rolling": (
                        float(row_data["overround"])
                        if "overround" in row_data.index and pd.notna(row_data.get("overround"))
                        else 0.20
                    )
                    if not result_df.empty
                    else 0.20,
                    "entropy_rolling": (
                        float(row_data["market_entropy"])
                        if (
                            "market_entropy" in row_data.index
                            and pd.notna(row_data.get("market_entropy"))
                        )
                        else 2.0
                    )
                    if not result_df.empty
                    else 2.0,
                    "odds_skewness_rolling": calc_odds_skewness(result_df),
                    "favorite_implied_prob_rolling": calc_favorite_implied_prob(result_df),
                    "odds_volatility_mean": (
                        float(result_df["odds_volatility"].mean())
                        if "odds_volatility" in result_df.columns and not result_df.empty
                        else 0.1
                    ),
                    "field_size_mean": (
                        float(row_data["field_size"])
                        if "field_size" in row_data.index and pd.notna(row_data.get("field_size"))
                        else 14.0
                    )
                    if not result_df.empty
                    else 14.0,
                }
            )

            # D-11: COLLAPSED regime skip (race-level, D-09: filter order #1)
            if regime_params.get("skip", False):
                n_collapsed_skipped += 1
                continue
            if self.betting_target == "win":
                candidate_df = self._race_predictor.get_win_candidates(result_df)
                win_diag_df = candidate_df.attrs.get("win_diagnostic_df")
                if isinstance(win_diag_df, pd.DataFrame):
                    result_df = win_diag_df
                n_ev_excluded += int(candidate_df.attrs.get("n_ev_excluded", 0))
            else:
                candidate_df = self._race_predictor.get_place_candidates(
                    result_df,
                    regime_params=regime_params,
                )
            n_candidates = len(candidate_df)

            # D-09: OddsBandFilter (candidate-level, filter order #3)
            n_candidates_before_band = n_candidates
            if self._odds_band_filter is not None and not candidate_df.empty:
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
            race_aggressive_strength = (
                float(result_df["aggressive_strength"].iloc[0])
                if "aggressive_strength" in result_df.columns
                and pd.notna(result_df["aggressive_strength"].iloc[0])
                else float("nan")
            )
            race_aggressive_tier = result_df.get("aggressive_tier", pd.Series([None])).iloc[0]
            race_market_condition = (
                float(result_df["market_condition_score"].iloc[0])
                if "market_condition_score" in result_df.columns
                and pd.notna(result_df["market_condition_score"].iloc[0])
                else float("nan")
            )

            _quality_score = self._race_predictor.get_quality_score(result_df)
            _quality_passed = self._race_predictor.should_bet(result_df)

            # 単勝はベット数を削らない方針のため、RaceQualityScreener は診断に留める。
            # 複勝/ワイドでは既存どおり品質ゲートを適用する。
            if not _quality_passed and self.betting_target != "win":
                result_df = _annotate_actual_bets(result_df, [])
                diag_logger.log_race(
                    race_id=race_id,
                    regime=str(regime),
                    ev_threshold=regime_params.get("ev_threshold", 1.10),
                    edge_threshold=edge_threshold,
                    quality_passed=False,
                    quality_score=_quality_score,
                    n_candidates=n_candidates,
                    n_bets=0,
                    aggressive_strength=race_aggressive_strength,
                    aggressive_tier=(
                        str(race_aggressive_tier) if pd.notna(race_aggressive_tier) else None
                    ),
                    market_condition_score=race_market_condition,
                )
                if "ev_place" in result_df.columns or self.betting_target == "win":
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
                            ev_place_corrected=float(
                                getattr(hr, "ev_place_corrected", float("nan"))
                            ),
                            ev_lower_place=float(getattr(hr, "EV_lower_place", float("nan"))),
                            place_selection_ev=float(
                                getattr(hr, "place_selection_ev", float("nan"))
                            ),
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
                            **_horse_win_diagnostic_kwargs(hr),
                        )
                        diag_logger.log_horse_features(hr._asdict())
                continue

            # Bet generation
            surface_key = result_df["surface"].iloc[0]
            bets = self._race_predictor.select_bets(
                result_df,
                bankroll,
                candidates=candidate_df,
                betting_target=self.betting_target,
            )

            # v5: セグメント除外フィルタ全削除 — モデル自身がedgeを低に見積もるように改善する
            # (旧v4の14個の除外フィルタは全て削除)

            # --- 意思決定EV/Oddsフィルター (select_bets後、bet確定前) ---
            # bet.odds = tanodds (5分前), bet.ev_lower_corrected = win_selection_ev
            # closing_win_odds, final_odds, result, kakuteijyuni は不使用
            if (
                self.betting_target == "win"
                and (self._min_win_ev > 0.0 or self._min_win_odds > 0.0)
                and bets
            ):
                _kept: list[Bet] = []
                for bet in bets:
                    if bet.bet_type != BetType.WIN:
                        _kept.append(bet)
                        continue
                    _exclude = False
                    # EV閾値チェック: NaNは除外
                    if self._min_win_ev > 0.0:
                        _ev = bet.ev_lower_corrected
                        if pd.isna(_ev) or _ev < self._min_win_ev:
                            _exclude = True
                    # オッズ閾値チェック: NaNは除外
                    if self._min_win_odds > 0.0:
                        if pd.isna(bet.odds) or bet.odds < self._min_win_odds:
                            _exclude = True
                    if _exclude:
                        n_win_ev_odds_excluded += 1
                    else:
                        _kept.append(bet)
                bets = _kept

            # --- tiered stake sizing (flat mode + win only) ---
            # 100円単位の天井丸め (math.ceil)
            if (
                self.betting_target == "win"
                and self.betting_mode == "flat"
                and self._win_ev_stake_threshold > 0.0
                and self._win_ev_stake_multiplier > 1.0
                and bets
            ):
                _updated: list[Bet] = []
                for bet in bets:
                    if (
                        bet.bet_type == BetType.WIN
                        and not pd.isna(bet.ev_lower_corrected)
                        and bet.ev_lower_corrected >= self._win_ev_stake_threshold
                    ):
                        new_stake = (
                            math.ceil(bet.stake * self._win_ev_stake_multiplier / 100.0)
                            * 100.0
                        )
                        new_stake = max(100.0, new_stake)
                        extra = new_stake - bet.stake
                        if extra > 0:
                            n_win_stake_increased += 1
                            total_win_stake_increased += extra
                            _updated.append(replace(bet, stake=new_stake))
                        else:
                            _updated.append(bet)
                    else:
                        _updated.append(bet)
                bets = _updated

            # Bet に確定オッズを設定（place/win のみ。wide は wide_payout_map で精算）
            updated_bets = []
            for bet in bets:
                if bet.bet_type == BetType.WIDE:
                    updated_bets.append(bet)
                elif bet.bet_type == BetType.WIN:
                    fo = self.win_payout_map.get((bet.race_id, bet.umaban), bet.odds)
                    updated_bets.append(replace(bet, final_odds=fo))
                else:
                    fo = final_odds_map.get((bet.race_id, bet.umaban), bet.odds)
                    updated_bets.append(replace(bet, final_odds=fo))
            bets = updated_bets
            settlements = [(bet, self._settle_bet(bet, result_df)) for bet in bets]
            if closing_win_odds_map and "umaban" in result_df.columns:
                result_df = result_df.copy()
                result_df["closing_win_odds"] = [
                    closing_win_odds_map.get((race_id, int(umaban)), np.nan)
                    if pd.notna(umaban)
                    else np.nan
                    for umaban in pd.to_numeric(result_df["umaban"], errors="coerce")
                ]
            result_df = _annotate_actual_bets(result_df, settlements)

            # メトリクス集計 (全ベットが発走前オッズ)
            n_pre_post_odds_bets += len(bets)

            # Log diagnostics for processed race. In win mode, quality_passed may be False
            # because the quality model is diagnostic-only.
            diag_logger.log_race(
                race_id=race_id,
                regime=str(regime),
                ev_threshold=regime_params.get("ev_threshold", 1.10),
                edge_threshold=edge_threshold,
                quality_passed=_quality_passed,
                quality_score=_quality_score,
                n_candidates=n_candidates,
                n_bets=len(bets),
                aggressive_strength=race_aggressive_strength,
                aggressive_tier=(
                    str(race_aggressive_tier) if pd.notna(race_aggressive_tier) else None
                ),
                market_condition_score=race_market_condition,
            )
            if "ev_place" in result_df.columns or self.betting_target == "win":
                for hr in result_df.itertuples(index=False):
                    is_actual_bet = bool(getattr(hr, "is_actual_bet", False))
                    diag_logger.log_horse(
                        race_id=race_id,
                        umaban=int(hr.umaban),
                        p_place_pred=float(getattr(hr, "p_place_pred", 0)),
                        e_return_place_pred=float(getattr(hr, "e_return_place_pred", 0)),
                        ev_place=float(getattr(hr, "ev_place", 0)),
                        fukuoddslow=float(getattr(hr, "fukuoddslow", 0)),
                        is_bet=is_actual_bet,
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
                        aggressive_strength=float(getattr(hr, "aggressive_strength", float("nan"))),
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
                        **_horse_win_diagnostic_kwargs(hr),
                    )
                    diag_logger.log_horse_features(hr._asdict())

            # Settlement (BacktestEngine 固有)
            for bet, bet_result in settlements:
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
                horse_row = horse_rows.iloc[0] if not horse_rows.empty else pd.Series(dtype=object)

                bet_history.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet.bet_type.value,
                        "umaban": bet.umaban,
                        "stake": bet.stake,
                        "odds": bet.odds,
                        "tanodds": _optional_float(horse_row.get("tanodds", bet.odds)),
                        "closing_win_odds": _optional_float(
                            horse_row.get("closing_win_odds", None)
                        ),
                        "clv": _optional_float(horse_row.get("clv", None)),
                        "final_odds": bet.final_odds,
                        "fuku_odds_low": bet.final_odds,
                        "result": bet_result,
                        "is_actual_bet": bet.stake > 0,
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
                            and horse_rows.iloc[0].get("p_place_pred", 0) is not pd.NA
                            and not pd.isna(horse_rows.iloc[0].get("p_place_pred", 0))
                            else 0.0
                        ),
                        "e_return_place_pred": (
                            float(horse_rows.iloc[0].get("e_return_place_pred", 0))
                            if not horse_rows.empty
                            and horse_rows.iloc[0].get("e_return_place_pred", 0) is not pd.NA
                            and not pd.isna(horse_rows.iloc[0].get("e_return_place_pred", 0))
                            else 0.0
                        ),
                        "p_win_pred": _optional_float(horse_row.get("p_win_pred", None)),
                        "p_win_corrected": _optional_float(horse_row.get("p_win_corrected", None)),
                        "p_win_final": _optional_float(horse_row.get("p_win_final", None)),
                        "e_return_win_pred": _optional_float(
                            horse_row.get("e_return_win_pred", None)
                        ),
                        "e_return_win_corrected": _optional_float(
                            horse_row.get("e_return_win_corrected", None)
                        ),
                        "win_selection_ev": _optional_float(
                            horse_row.get(
                                "win_selection_ev_raw",
                                horse_row.get("win_selection_ev", None),
                            )
                        ),
                        "win_selection_ev_tail_calibrated": _optional_float(
                            horse_row.get("win_selection_ev_tail_calibrated", None)
                        ),
                        "win_selection_edge": _optional_float(
                            horse_row.get("win_selection_edge", None)
                        ),
                        "win_selection_prob": _optional_float(
                            horse_row.get("win_selection_prob", None)
                        ),
                        "win_gate_score": _optional_float(horse_row.get("win_gate_score", None)),
                        "win_gate_pass": _optional_bool(horse_row.get("win_gate_pass", None)),
                        "win_gate_odds_score": _optional_float(
                            horse_row.get("win_gate_odds_score", None)
                        ),
                        "win_gate_prob_score": _optional_float(
                            horse_row.get("win_gate_prob_score", None)
                        ),
                        "win_gate_edge_score": _optional_float(
                            horse_row.get("win_gate_edge_score", None)
                        ),
                        "win_gate_edge_odds_score": _optional_float(
                            horse_row.get("win_gate_edge_odds_score", None)
                        ),
                        "p_market_win_raw": _optional_float(
                            horse_row.get("p_market_win_raw", None)
                        ),
                        "p_market_win_norm": _optional_float(
                            horse_row.get("p_market_win_norm", None)
                        ),
                        "win_market_residual": _optional_float(
                            horse_row.get("win_market_residual", None)
                        ),
                        "win_market_logit_edge": _optional_float(
                            horse_row.get("win_market_logit_edge", None)
                        ),
                        "win_market_prob_ratio": _optional_float(
                            horse_row.get("win_market_prob_ratio", None)
                        ),
                        "win_market_value_ratio": _optional_float(
                            horse_row.get("win_market_value_ratio", None)
                        ),
                        "win_market_selection_score": _optional_float(
                            horse_row.get("win_market_selection_score", None)
                        ),
                        "win_profit_score": _optional_float(
                            horse_row.get("win_profit_score", None)
                        ),
                        "win_profit_selector_pass": _optional_bool(
                            horse_row.get("win_profit_selector_pass", None)
                        ),
                        "win_profit_rank": _optional_float(horse_row.get("win_profit_rank", None)),
                        "win_profit_stake_scale": _optional_float(
                            horse_row.get("win_profit_stake_scale", None)
                        ),
                        "win_profit_reason": _optional_str(
                            horse_row.get("win_profit_reason", None)
                        ),
                        "win_late_odds_drop_z": _optional_float(
                            horse_row.get("win_late_odds_drop_z", None)
                        ),
                        "win_late_odds_drop_weight": _optional_float(
                            horse_row.get("win_late_odds_drop_weight", None)
                        ),
                        "win_ev_tail_pressure": _optional_float(
                            horse_row.get("win_ev_tail_pressure", None)
                        ),
                        "win_ev_tail_penalty_weight": _optional_float(
                            horse_row.get("win_ev_tail_penalty_weight", None)
                        ),
                        "win_log_odds": _optional_float(horse_row.get("win_log_odds", None)),
                        "win_log_odds_penalty": _optional_float(
                            horse_row.get("win_log_odds_penalty", None)
                        ),
                        "win_model_prob_rank": _optional_float(
                            horse_row.get("win_model_prob_rank", None)
                        ),
                        "win_prob_rank_bonus": _optional_float(
                            horse_row.get("win_prob_rank_bonus", None)
                        ),
                        "win_market_risk_penalty": _optional_float(
                            horse_row.get("win_market_risk_penalty", None)
                        ),
                        "risk_flags": _optional_str(horse_row.get("risk_flags", None)),
                        "excluded_reason": _optional_str(horse_row.get("excluded_reason", None)),
                        "filter_pass_flags": _optional_str(
                            horse_row.get("filter_pass_flags", None)
                        ),
                        "candidate_count_before_filter": _optional_int(
                            horse_row.get("candidate_count_before_filter", None)
                        ),
                        "candidate_count_after_filter": _optional_int(
                            horse_row.get("candidate_count_after_filter", None)
                        ),
                        "selected_rank_by_p_win_final": _optional_float(
                            horse_row.get("selected_rank_by_p_win_final", None)
                        ),
                        "selected_rank_by_win_selection_ev": _optional_float(
                            horse_row.get("selected_rank_by_win_selection_ev", None)
                        ),
                        "selected_rank_by_win_market_logit_edge": _optional_float(
                            horse_row.get("selected_rank_by_win_market_logit_edge", None)
                        ),
                        "selected_rank_by_win_market_score": _optional_float(
                            horse_row.get("selected_rank_by_win_market_score", None)
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

        # 意思決定EV/Oddsフィルター + tiered stake ログ
        if n_win_ev_odds_excluded > 0:
            logger.info(
                "Win EV/Odds filter excluded %d bets (min_ev=%.2f, min_odds=%.1f)",
                n_win_ev_odds_excluded,
                self._min_win_ev,
                self._min_win_odds,
            )
        if n_win_stake_increased > 0:
            logger.info(
                "Win tiered stake: %d bets increased, +%.0f yen total",
                n_win_stake_increased,
                total_win_stake_increased,
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
            if bets_per_year < self._min_bets_per_year:
                logger.warning(
                    "Bet count guard WARNING: %.0f bets/year (below %d threshold). "
                    "Ultra-selective betting produces unreliable ROI estimates. "
                    "Total bets: %d, Period: %.1f years.",
                    bets_per_year,
                    self._min_bets_per_year,
                    total_bets,
                    n_years,
                )

        # --- D-03(2): PFP verify -- OOS期間終了時のモデル不変性確認 ---
        pfp_result: dict[str, Any] | None = None
        if self._pfp is not None:
            pfp_result = self._pfp.verify()
            if not pfp_result["passed"]:
                raise RuntimeError(pfp_result["message"])  # D-04: 即時停止
            logger.info("PFP verification passed: %s", pfp_result["message"])

        backtest_result = BacktestResult(
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
            n_win_ev_odds_excluded=n_win_ev_odds_excluded,
            n_win_stake_increased=n_win_stake_increased,
            total_win_stake_increased=total_win_stake_increased,
            exclusion_stats={
                "collapsed_skipped": n_collapsed_skipped,
                "ev_excluded": n_ev_excluded,
                "odds_band_excluded": n_odds_band_excluded,
                "total_candidates_evaluated": n_total_candidates,
                "odds_band_filter_excluded": (
                    self._odds_band_filter.excluded_bands if self._odds_band_filter else {}
                ),
                "win_ev_odds_excluded": n_win_ev_odds_excluded,
                "win_stake_increased": n_win_stake_increased,
                "total_win_stake_increased": total_win_stake_increased,
            },
        )

        # --- D-07/D-08: 検証結果JSON出力 ---
        try:
            from backtest.validation_report import generate_validation_report

            train_start_val = ""
            train_end_val = ""
            if hasattr(self.models, "train_period") and self.models.train_period:
                train_start_val, train_end_val = self.models.train_period

            report = generate_validation_report(
                result=backtest_result,
                test_start=test_start,
                test_end=test_end,
                train_start=train_start_val,
                train_end=train_end_val,
                manifest_path=self._manifest_path,
                pfp_result=pfp_result,
            )

            validation_dir = Path("data/validation")
            validation_dir.mkdir(parents=True, exist_ok=True)
            report_path = validation_dir / "validation_report.json"
            import json as _json

            report_path.write_text(
                _json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info("Validation report saved: %s", report_path)
        except Exception as e:
            logger.warning("Validation report generation failed: %s", e)

        # P3: HorseHistoryFeaturesクラスキャッシュをクリア (メモリリーク防止)
        from features.horse_history_features import HorseHistoryFeatures

        HorseHistoryFeatures.clear_class_cache()

        return backtest_result

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

        # 単勝: 着順確認 → 1着のみ payout lookup → 確定配当で精算
        if bet.bet_type == BetType.WIN:
            horse = race_df[race_df["umaban"] == bet.umaban]
            if horse.empty:
                return 0.0
            finish_pos = int(horse.iloc[0]["kakuteijyuni"])
            if finish_pos != 1:
                return 0.0
            win_key = (bet.race_id, bet.umaban)
            if hasattr(self, "win_payout_map") and win_key in self.win_payout_map:
                return float(bet.stake * self.win_payout_map[win_key])
            # WIN fallback: tanodds を使用 (fukuoddslow は単勝精算に不適切)
            logger.warning(
                "Win payout missing for %s umaban=%d, using tanodds fallback=%.1f",
                bet.race_id,
                bet.umaban,
                bet.odds,
            )
            return float(bet.stake * bet.odds)

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
