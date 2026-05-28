"""Shadow Comparison Framework (SHD-01~03)

BacktestEngine を2回 (baseline vs shadow) 実行し、同一テスト期間で
事後アライメント・メトリクス計算を行う比較基盤。

D-01~D-06, D-12~D-15, D-18~D-19, D-21 を実装。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldDefinition:
    """単一 fold の定義 (D-05)."""

    year: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    @staticmethod
    def create_folds(
        fold_years: list[int],
        train_window: int = 4,
    ) -> list[FoldDefinition]:
        """fold_years から FoldDefinition リストを生成.

        D-05: Fold 2024 = train 2020-2023, test 2024.
        """
        folds: list[FoldDefinition] = []
        for year in fold_years:
            train_start_year = year - train_window
            folds.append(
                FoldDefinition(
                    year=year,
                    train_start=f"{train_start_year}-01-01",
                    train_end=f"{year - 1}-12-31",
                    test_start=f"{year}-01-01",
                    test_end=f"{year}-12-31",
                )
            )
        return folds


@dataclass(frozen=True)
class VariantConfig:
    """比較バリアントの設定 (D-01, D-06)."""

    variant_name: str
    model_dir: Path
    enable_market_aware_calibrator: bool
    enable_race_level_ranker: bool


@dataclass
class ComparisonMetrics:
    """比較メトリクス (D-12, D-14, D-15)."""

    brier: float = 0.0
    logloss: float = 0.0
    ece: float = 0.0
    roi: float = 0.0
    hit_rate: float = 0.0
    bet_count: int = 0
    avg_odds: float = 0.0
    max_drawdown: float = 0.0
    clv: float | None = None
    clv_available: bool = False
    selection_agreement: float | None = None
    avg_investment_score: float | None = None
    actual_predicted_ratio: float = 0.0


@dataclass
class VariantResult:
    """バリアントごとの BacktestResult とフラグ状態."""

    variant_name: str
    backtest_result: Any  # BacktestResult
    flag_states: dict[str, bool] = field(default_factory=dict)


@dataclass
class ShadowComparisonResult:
    """単一 fold の比較結果 (D-01)."""

    fold: FoldDefinition
    variants: dict[str, VariantResult] = field(default_factory=dict)
    race_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    horse_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, ComparisonMetrics] = field(default_factory=dict)
    alignment_succeeded: bool = False


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------


class ShadowComparisonFramework:
    """Shadow Comparison Framework (D-01).

    N-way バリアント対応の BacktestEngine 比較基盤.
    """

    def __init__(
        self,
        variants: list[VariantConfig],
        store: Any | None = None,
        betting_target: str = "win",
        betting_mode: str = "flat",
        strategy_params: dict[str, Any] | None = None,
        min_bets_per_year: int = 1000,
    ) -> None:
        self.variants = variants
        self.store = store
        self.betting_target = betting_target
        self.betting_mode = betting_mode
        self.strategy_params = strategy_params
        self.min_bets_per_year = min_bets_per_year

    # ------------------------------------------------------------------
    # D-21: Strict mode artifact validation
    # ------------------------------------------------------------------

    def _validate_artifacts(
        self,
        variant_name: str,
        models: Any,
        model_dir: Path,
    ) -> None:
        """D-21: flag=True なのに artifact が無い場合は ValueError."""
        variant_cfg = next(
            (v for v in self.variants if v.variant_name == variant_name), None,
        )
        if variant_cfg is None:
            return

        if variant_cfg.enable_market_aware_calibrator:
            has_mawc = any(
                getattr(sm, "market_aware_win_calibrator", None) is not None
                and getattr(sm.market_aware_win_calibrator, "is_trained", False)
                for sm in models.submodels.values()
            )
            if not has_mawc:
                raise ValueError(
                    f"Variant '{variant_name}': enable_market_aware_calibrator=True "
                    f"but no trained MAWC artifact in {model_dir}"
                )

        if variant_cfg.enable_race_level_ranker:
            has_ranker = any(
                getattr(sm, "win_race_level_ranker", None) is not None
                and getattr(sm.win_race_level_ranker, "is_trained", False)
                for sm in models.submodels.values()
            )
            if not has_ranker:
                raise ValueError(
                    f"Variant '{variant_name}': enable_race_level_ranker=True "
                    f"but no trained ranker artifact in {model_dir}"
                )

    # ------------------------------------------------------------------
    # run_fold: BacktestEngine x N variants
    # ------------------------------------------------------------------

    def run_fold(self, fold: FoldDefinition) -> ShadowComparisonResult:
        """単一 fold で N-way 比較を実行 (D-01, D-02, D-03)."""
        from backtest.engine import BacktestEngine
        from db.model_loader import ModelLoader

        results: dict[str, Any] = {}  # variant_name -> BacktestResult
        variant_results: dict[str, VariantResult] = {}

        for variant_cfg in self.variants:
            # D-05: model_dir / year でロード
            model_dir = variant_cfg.model_dir / str(fold.year)
            loader = ModelLoader()
            loaded_models, _info = loader.load_from_dir(model_dir)

            # D-18: train_period を設定
            loaded_models.train_period = (fold.train_start, fold.train_end)

            # D-21: strict mode validation
            self._validate_artifacts(variant_cfg.variant_name, loaded_models, model_dir)

            # D-19: _shadow_flags を注入 (RacePredictor が読み取る)
            loaded_models._shadow_flags = {  # type: ignore[attr-defined]
                "enable_market_aware_calibrator": variant_cfg.enable_market_aware_calibrator,
                "enable_race_level_ranker": variant_cfg.enable_race_level_ranker,
            }

            engine = BacktestEngine(
                models=loaded_models,
                betting_mode=self.betting_mode,
                betting_target=self.betting_target,
                strategy_params=self.strategy_params,
                diag_prefix=f"shadow_{variant_cfg.variant_name}",
                min_bets_per_year=self.min_bets_per_year,
            )
            bt_result = engine.run(
                test_start=fold.test_start,
                test_end=fold.test_end,
            )
            results[variant_cfg.variant_name] = bt_result
            variant_results[variant_cfg.variant_name] = VariantResult(
                variant_name=variant_cfg.variant_name,
                backtest_result=bt_result,
                flag_states={
                    "enable_market_aware_calibrator": variant_cfg.enable_market_aware_calibrator,
                    "enable_race_level_ranker": variant_cfg.enable_race_level_ranker,
                },
            )

        # D-03: Post-hoc alignment
        race_diff = self._align_race_level(results)
        horse_diff = self._align_horse_level(results)
        alignment_ok = not race_diff.empty or not horse_diff.empty

        # D-12: Metrics computation
        metrics: dict[str, ComparisonMetrics] = {}
        for vname, bt_result in results.items():
            metrics[vname] = self.compute_metrics(
                race_diff, horse_diff, vname, bt_result.bet_history,
                bt_result=bt_result,
            )

        return ShadowComparisonResult(
            fold=fold,
            variants=variant_results,
            race_diff=race_diff,
            horse_diff=horse_diff,
            metrics=metrics,
            alignment_succeeded=alignment_ok,
        )

    # ------------------------------------------------------------------
    # run: multi-fold entry point
    # ------------------------------------------------------------------

    def run(
        self,
        folds: list[FoldDefinition] | None = None,
    ) -> list[ShadowComparisonResult]:
        """全 fold を実行. デフォルトは 2024/2025."""
        if folds is None:
            folds = FoldDefinition.create_folds([2024, 2025])
        return [self.run_fold(fold) for fold in folds]

    # ------------------------------------------------------------------
    # D-03: Post-hoc alignment
    # ------------------------------------------------------------------

    def _align_race_level(
        self,
        results: dict[str, Any],
    ) -> pd.DataFrame:
        """race_id ごとに baseline vs shadow をアライメント (D-03)."""
        variant_names = list(results.keys())
        if len(variant_names) < 2:
            return pd.DataFrame()

        # 各バリアントの bet_history を DataFrame に変換
        dfs: dict[str, pd.DataFrame] = {}
        for vname, bt_result in results.items():
            if bt_result.bet_history:
                dfs[vname] = pd.DataFrame(bt_result.bet_history)

        if len(dfs) < 2:
            return pd.DataFrame()

        baseline_name = variant_names[0]
        baseline_df = dfs.get(baseline_name, pd.DataFrame())
        if baseline_df.empty:
            return pd.DataFrame()

        # レースごとに選択された馬 (stake > 0) を特定
        def _pick_selected_per_race(bh_df: pd.DataFrame) -> pd.DataFrame:
            """各 race_id で stake > 0 の馬を抽出."""
            bet_rows = bh_df[bh_df["stake"].astype(float) > 0].copy()
            if bet_rows.empty:
                return pd.DataFrame()
            # レースごとに最初のベット馬を代表値として選択
            return bet_rows.groupby("race_id", observed=True).first().reset_index()

        all_rows: list[dict[str, Any]] = []
        baseline_selected = _pick_selected_per_race(baseline_df)

        for vname in variant_names[1:]:
            shadow_df = dfs.get(vname, pd.DataFrame())
            if shadow_df.empty:
                continue
            shadow_selected = _pick_selected_per_race(shadow_df)

            # race_id の和集合でマージ
            merged = baseline_selected.merge(
                shadow_selected,
                on="race_id",
                how="outer",
                suffixes=(f"_{baseline_name}", f"_{vname}"),
            )

            for _, row in merged.iterrows():
                bl_umaban = row.get(f"umaban_{baseline_name}", row.get("umaban"))
                sh_umaban = row.get(f"umaban_{vname}", row.get("umaban"))
                if pd.isna(bl_umaban):
                    bl_umaban = row.get("umaban", None)
                if pd.isna(sh_umaban):
                    sh_umaban = row.get("umaban", None)

                changed = bl_umaban != sh_umaban
                diff_row: dict[str, Any] = {
                    "race_id": row["race_id"],
                    "baseline_selected_umaban": bl_umaban,
                    "shadow_selected_umaban": sh_umaban,
                    "selected_changed": changed,
                }

                # 各バリアントのメトリクス列を転記
                for col_key in ["tanodds", "p_win_final", "win_selection_ev",
                                "win_market_selection_score", "result", "stake",
                                "closing_win_odds", "investment_score"]:
                    bl_col = f"{col_key}_{baseline_name}"
                    sh_col = f"{col_key}_{vname}"
                    diff_row[f"baseline_{col_key}"] = row.get(bl_col, row.get(col_key))
                    diff_row[f"shadow_{col_key}"] = row.get(sh_col, row.get(col_key))

                all_rows.append(diff_row)

        if not all_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_rows)

    def _align_horse_level(
        self,
        results: dict[str, Any],
    ) -> pd.DataFrame:
        """race_id + umaban ごとに baseline vs shadow をアライメント (D-03)."""
        variant_names = list(results.keys())
        if len(variant_names) < 2:
            return pd.DataFrame()

        dfs: dict[str, pd.DataFrame] = {}
        for vname, bt_result in results.items():
            if bt_result.bet_history:
                dfs[vname] = pd.DataFrame(bt_result.bet_history)

        if len(dfs) < 2:
            return pd.DataFrame()

        baseline_name = variant_names[0]
        baseline_df = dfs[baseline_name]

        # baseline を基準にして shadow をマージ
        key_cols = ["race_id", "umaban"]
        align_cols = ["p_win_final", "investment_score", "stake",
                       "win_market_selection_score"]

        merged = baseline_df[key_cols].copy()
        for col in align_cols:
            if col in baseline_df.columns:
                merged[f"baseline_{col}"] = baseline_df[col].values
            else:
                merged[f"baseline_{col}"] = np.nan

        merged["baseline_selected"] = (
            baseline_df["stake"].astype(float) > 0
            if "stake" in baseline_df.columns
            else False
        )

        for vname in variant_names[1:]:
            shadow_df = dfs[vname]
            shadow_subset = shadow_df[key_cols].copy()
            for col in align_cols:
                if col in shadow_df.columns:
                    shadow_subset[f"shadow_{col}"] = shadow_df[col].values
                else:
                    shadow_subset[f"shadow_{col}"] = np.nan
            shadow_subset["shadow_selected"] = (
                shadow_df["stake"].astype(float) > 0
                if "stake" in shadow_df.columns
                else False
            )

            # Merge on key_cols
            merge_cols = key_cols + [c for c in shadow_subset.columns if c not in key_cols]
            merged = merged.merge(
                shadow_subset[merge_cols],
                on=key_cols,
                how="outer",
            )

        # kakuteijyuni があればマージ
        if "kakuteijyuni" in baseline_df.columns:
            kakutei = baseline_df[key_cols + ["kakuteijyuni"]].drop_duplicates(subset=key_cols)
            merged = merged.merge(kakutei, on=key_cols, how="left")

        return merged

    # ------------------------------------------------------------------
    # D-12: Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        aligned_race: pd.DataFrame,
        aligned_horse: pd.DataFrame,
        variant_name: str,
        bet_history: list[dict],
        *,
        bt_result: Any | None = None,
    ) -> ComparisonMetrics:
        """メトリクスを計算 (D-12, D-14, D-15)."""
        metrics = ComparisonMetrics()

        # --- Bet-level metrics ---
        if bet_history:
            total_stake = sum(b.get("stake", 0) for b in bet_history)
            total_return = sum(b.get("result", 0) for b in bet_history if b.get("result", 0) > 0)
            n_bets = len(bet_history)
            n_wins = sum(1 for b in bet_history if b.get("result", 0) > 0)

            metrics.bet_count = n_bets
            metrics.roi = total_return / total_stake - 1.0 if total_stake > 0 else 0.0
            metrics.hit_rate = n_wins / n_bets if n_bets > 0 else 0.0

            odds_vals = [b.get("tanodds", b.get("odds", 0)) for b in bet_history]
            metrics.avg_odds = float(np.mean(odds_vals)) if odds_vals else 0.0

            if bt_result is not None:
                metrics.max_drawdown = bt_result.max_drawdown

        # --- CLV (D-14) ---
        clv_val, clv_avail = self._compute_clv(bet_history)
        metrics.clv = clv_val
        metrics.clv_available = clv_avail

        # --- Probability quality metrics (horse-level) ---
        p_col = f"{variant_name}_p_win_final"
        if p_col not in aligned_horse.columns and "baseline_p_win_final" in aligned_horse.columns:
            # Use baseline columns for single-variant metrics
            p_col = "baseline_p_win_final"

        if not aligned_horse.empty and p_col in aligned_horse.columns:
            p_vals = pd.to_numeric(aligned_horse[p_col], errors="coerce")
            # kakuteijyuni may not exist in all horse_diff DataFrames
            if "kakuteijyuni" in aligned_horse.columns:
                is_win = (aligned_horse["kakuteijyuni"] == 1).astype(float)
            else:
                is_win = pd.Series(0.0, index=aligned_horse.index)
            # Ensure same length
            p_vals = p_vals.reset_index(drop=True)
            is_win = is_win.reset_index(drop=True)
            valid_mask = p_vals.notna() & (p_vals > 0) & (p_vals < 1)

            if valid_mask.sum() > 0:
                p_valid = p_vals[valid_mask].values
                y_valid = is_win[valid_mask].values

                # Brier
                metrics.brier = float(np.mean((p_valid - y_valid) ** 2))

                # Logloss
                eps = 1e-15
                p_clipped = np.clip(p_valid, eps, 1 - eps)
                metrics.logloss = float(
                    -np.mean(y_valid * np.log(p_clipped) + (1 - y_valid) * np.log(1 - p_clipped))
                )

                # ECE (10 equal-width bins)
                metrics.ece = self._compute_ece(p_valid, y_valid, n_bins=10)

                # Actual/predicted ratio
                mean_actual = float(np.mean(y_valid))
                mean_pred = float(np.mean(p_valid))
                metrics.actual_predicted_ratio = (
                    mean_actual / mean_pred if mean_pred > 0 else 0.0
                )

        # --- Investment score average ---
        inv_col = f"{variant_name}_investment_score"
        if (
            inv_col not in aligned_horse.columns
            and "baseline_investment_score" in aligned_horse.columns
        ):
            inv_col = "baseline_investment_score"
        if not aligned_horse.empty and inv_col in aligned_horse.columns:
            inv_vals = pd.to_numeric(aligned_horse[inv_col], errors="coerce")
            if inv_vals.notna().any():
                metrics.avg_investment_score = float(inv_vals.mean())

        # --- Selection agreement (D-15) ---
        if not aligned_race.empty and "selected_changed" in aligned_race.columns:
            metrics.selection_agreement = float(
                1.0 - aligned_race["selected_changed"].mean()
            )

        return metrics

    def compute_metrics_by_group(
        self,
        aligned_horse: pd.DataFrame,
        group_col: str,
    ) -> dict[str, ComparisonMetrics]:
        """グループ別メトリクス (D-13)."""
        result: dict[str, ComparisonMetrics] = {}
        if aligned_horse.empty:
            return result

        if group_col == "odds_band":
            aligned_horse = aligned_horse.copy()
            odds = pd.to_numeric(
                aligned_horse.get("closing_win_odds", pd.Series(dtype=float)),
                errors="coerce",
            )
            aligned_horse["odds_band"] = pd.cut(
                odds,
                bins=[0, 3, 5, 10, 30, float("inf")],
                labels=["1-3", "3-5", "5-10", "10-30", "30+"],
            )
            group_col = "odds_band"
        elif group_col == "prob_rank_band":
            # Would need prob rank data - skip if not available
            return result

        if group_col not in aligned_horse.columns:
            return result

        for group_key, group_df in aligned_horse.groupby(group_col, observed=True):
            group_metrics = self.compute_metrics(
                pd.DataFrame(), group_df, "baseline", [],
            )
            result[str(group_key)] = group_metrics

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_clv(bet_history: list[dict]) -> tuple[float | None, bool]:
        """D-14: CLV = closing_odds / betting_odds - 1."""
        if not bet_history:
            return None, False

        clvs: list[float] = []
        for b in bet_history:
            betting = b.get("tanodds", b.get("odds", 0))
            closing = b.get("closing_win_odds", None)
            if (
                closing is not None
                and not pd.isna(closing)
                and betting is not None
                and not pd.isna(betting)
                and float(betting) > 0
                and float(closing) > 0
            ):
                clvs.append(float(closing) / float(betting) - 1.0)

        if len(clvs) < max(1, len(bet_history) * 0.1):
            return None, False

        return float(np.mean(clvs)), True

    @staticmethod
    def _compute_selection_agreement(race_diff: pd.DataFrame) -> float:
        """D-15: selection agreement = fraction of races with same selection."""
        if race_diff.empty or "selected_changed" not in race_diff.columns:
            return 0.0
        return float(1.0 - race_diff["selected_changed"].mean())

    @staticmethod
    def _compute_ece(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_pred)
        if total == 0:
            return 0.0

        for i in range(n_bins):
            mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            avg_pred = y_pred[mask].mean()
            avg_true = y_true[mask].mean()
            ece += abs(avg_pred - avg_true) * (n_in_bin / total)

        return float(ece)
