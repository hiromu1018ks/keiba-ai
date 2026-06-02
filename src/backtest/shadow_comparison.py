"""Shadow Comparison Framework (SHD-01~03)

BacktestEngine を2回 (baseline vs shadow) 実行し、同一テスト期間で
事後アライメント・メトリクス計算を行う比較基盤。

D-01~D-06, D-12~D-15, D-18~D-19, D-21 を実装。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output artifact helpers (D-08, D-10, D-11, D-13, D-20, D-22)
# ---------------------------------------------------------------------------


def _compute_sha256(path: Path) -> str:
    """ファイルのSHA256ハッシュを計算."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics_to_dict(m: ComparisonMetrics) -> dict[str, Any]:
    """ComparisonMetrics を JSON 互換 dict に変換."""
    return {
        "brier": m.brier,
        "logloss": m.logloss,
        "ece": m.ece,
        "roi": m.roi,
        "hit_rate": m.hit_rate,
        "bet_count": m.bet_count,
        "avg_odds": m.avg_odds,
        "max_drawdown": m.max_drawdown,
        "clv": m.clv,
        "clv_available": m.clv_available,
        "selection_agreement": m.selection_agreement,
        "avg_investment_score": m.avg_investment_score,
        "actual_predicted_ratio": m.actual_predicted_ratio,
    }


def save_results(
    comparison_results: list[ShadowComparisonResult],
    output_dir: Path,
) -> dict[str, Path]:
    """全 fold の比較結果を出力 (D-08, D-10, D-11, D-13).

    Returns:
        artifact name -> Path のマッピング.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. shadow_comparison_result.json ---
    folds_data: dict[str, Any] = {}
    all_race_dfs: list[pd.DataFrame] = []
    all_horse_dfs: list[pd.DataFrame] = []

    for cr in comparison_results:
        year_str = str(cr.fold.year)
        fold_entry: dict[str, Any] = {
            "metrics": {vname: _metrics_to_dict(m) for vname, m in cr.metrics.items()},
        }

        # -- Grouped metrics (D-13) --
        if not cr.horse_diff.empty:
            # Dummy framework instance for compute_metrics (no state dependency)
            _fw = ShadowComparisonFramework(variants=[])

            # by surface
            if "surface" in cr.horse_diff.columns:
                surface_groups: dict[str, dict[str, Any]] = {}
                for surf_val, surf_df in cr.horse_diff.groupby("surface", observed=True):
                    surface_groups[str(surf_val)] = {
                        vname: _metrics_to_dict(
                            _fw.compute_metrics(
                                pd.DataFrame(), surf_df, vname, [],
                            )
                        )
                        for vname in cr.metrics
                    }
                fold_entry["metrics_by_surface"] = surface_groups

            # by odds_band
            odds_groups = _fw.compute_metrics_by_group(
                cr.horse_diff, "odds_band",
            )
            if odds_groups:
                fold_entry["metrics_by_odds_band"] = {
                    k: _metrics_to_dict(v) for k, v in odds_groups.items()
                }

            # by prob_rank_band
            prob_groups = _fw.compute_metrics_by_group(
                cr.horse_diff, "prob_rank_band",
            )
            if prob_groups:
                fold_entry["metrics_by_prob_rank_band"] = {
                    k: _metrics_to_dict(v) for k, v in prob_groups.items()
                }

            # by value_score_band (if investment_score exists)
            inv_cols = [c for c in cr.horse_diff.columns if "investment_score" in c]
            if inv_cols:
                for inv_col in inv_cols:
                    inv_vals = pd.to_numeric(cr.horse_diff[inv_col], errors="coerce")
                    if inv_vals.notna().any():
                        band_df = cr.horse_diff.copy()
                        band_df["value_score_band"] = pd.cut(
                            inv_vals,
                            bins=[0, 0.3, 0.5, 0.7, 1.0, float("inf")],
                            labels=["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0", "1.0+"],
                        )
                        vs_groups: dict[str, dict[str, Any]] = {}
                        for band_val, band_df_sub in band_df.groupby(
                            "value_score_band", observed=True,
                        ):
                            vs_groups[str(band_val)] = {
                                vname: _metrics_to_dict(
                                    _fw.compute_metrics(
                                        pd.DataFrame(), band_df_sub, vname, [],
                                    )
                                )
                                for vname in cr.metrics
                            }
                        if vs_groups:
                            fold_entry["metrics_by_value_score_band"] = vs_groups
                        break  # Use first investment_score column found

        # by selected_changed (D-13)
        if not cr.race_diff.empty and "selected_changed" in cr.race_diff.columns:
            _fw_sc = ShadowComparisonFramework(variants=[])
            changed_groups: dict[str, dict[str, Any]] = {}
            for changed_val, changed_df in cr.race_diff.groupby(
                "selected_changed", observed=True,
            ):
                label = "changed" if changed_val else "unchanged"
                group_race_ids = set(changed_df["race_id"])
                group_metrics: dict[str, dict[str, Any]] = {}
                for vname, vr in cr.variants.items():
                    group_bh = [
                        b for b in vr.backtest_result.bet_history
                        if b.get("race_id") in group_race_ids
                    ]
                    group_metrics[vname] = _metrics_to_dict(
                        _fw_sc.compute_metrics(
                            pd.DataFrame(), pd.DataFrame(), vname, group_bh,
                        )
                    )
                changed_groups[label] = group_metrics
            fold_entry["metrics_by_selected_changed"] = changed_groups

        # selection agreement
        if not cr.race_diff.empty and "selected_changed" in cr.race_diff.columns:
            fold_entry["selection_agreement"] = float(
                1.0 - cr.race_diff["selected_changed"].mean()
            )
        else:
            fold_entry["selection_agreement"] = None

        # bet counts
        fold_entry["bet_counts"] = {
            vname: m.bet_count for vname, m in cr.metrics.items()
        }

        folds_data[year_str] = fold_entry

        # Collect DataFrames for concatenation
        if not cr.race_diff.empty:
            rd = cr.race_diff.copy()
            rd["fold_year"] = cr.fold.year
            all_race_dfs.append(rd)
        if not cr.horse_diff.empty:
            hd = cr.horse_diff.copy()
            hd["fold_year"] = cr.fold.year
            all_horse_dfs.append(hd)

    # Overall metrics (aggregate across folds)
    overall_metrics: dict[str, Any] = {}
    all_variant_names: set[str] = set()
    for cr in comparison_results:
        all_variant_names.update(cr.metrics.keys())

    for vname in sorted(all_variant_names):
        # Pool bet_history across folds
        pooled_bh: list[dict] = []
        pooled_dd: list[float] = []
        for cr in comparison_results:
            vr = cr.variants.get(vname)
            if vr is not None:
                pooled_bh.extend(vr.backtest_result.bet_history)
                pooled_dd.append(vr.backtest_result.max_drawdown)

        if pooled_bh:
            _fw_overall = ShadowComparisonFramework(variants=[])
            overall_metrics[vname] = _metrics_to_dict(
                _fw_overall.compute_metrics(
                    pd.DataFrame(), pd.DataFrame(), vname, pooled_bh,
                )
            )
            overall_metrics[vname]["max_drawdown"] = max(pooled_dd) if pooled_dd else 0.0

    # Build final JSON structure
    result_json: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": folds_data,
        "overall": {
            "metrics": overall_metrics,
        },
    }

    metrics_path = output_dir / "shadow_comparison_result.json"
    metrics_path.write_text(
        json.dumps(result_json, indent=2, sort_keys=True, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- 2. shadow_race_diff.parquet + .csv ---
    race_diff_path = output_dir / "shadow_race_diff.parquet"
    csv_path = output_dir / "shadow_race_diff.csv"

    if all_race_dfs:
        combined_race = pd.concat(all_race_dfs, ignore_index=True)
        combined_race.to_parquet(race_diff_path, index=False, engine="pyarrow")
        combined_race.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_parquet(race_diff_path, index=False, engine="pyarrow")
        pd.DataFrame().to_csv(csv_path, index=False, encoding="utf-8-sig")

    # --- 3. shadow_horse_diff.parquet ---
    horse_diff_path = output_dir / "shadow_horse_diff.parquet"
    if all_horse_dfs:
        combined_horse = pd.concat(all_horse_dfs, ignore_index=True)
        combined_horse.to_parquet(horse_diff_path, index=False, engine="pyarrow")
    else:
        pd.DataFrame().to_parquet(horse_diff_path, index=False, engine="pyarrow")

    return {
        "metrics_json": metrics_path,
        "race_diff_parquet": race_diff_path,
        "race_diff_csv": csv_path,
        "horse_diff_parquet": horse_diff_path,
    }


def save_manifest(
    comparison_results: list[ShadowComparisonResult],
    variant_configs: list[VariantConfig],
    output_dir: Path,
    artifact_paths: dict[str, Path],
) -> Path:
    """shadow_manifest.json を書き出す (D-20, D-22)."""
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework_version": "1.0",
        "variants": [],
        "folds": [],
        "artifacts": {},
        "metric_definitions": {
            "brier": "mean((p_win_final - is_win)^2)",
            "logloss": "mean(-y*log(p) - (1-y)*log(1-p))",
            "ece": "Expected Calibration Error, 10-bin equal width",
            "selection_agreement": "fraction of races where same horse selected",
            "clv": "closing_odds / betting_odds - 1 (diagnostic only)",
        },
    }

    # Variants
    for vc in variant_configs:
        variant_entry: dict[str, Any] = {
            "variant_name": vc.variant_name,
            "model_dir": str(vc.model_dir),
            "flag_states": {
                "enable_market_aware_calibrator": vc.enable_market_aware_calibrator,
                "enable_race_level_ranker": vc.enable_race_level_ranker,
            },
        }
        if vc.variant_name == "baseline":
            variant_entry["baseline_definition"] = (
                "MAWC/ranker disabled, existing p_win_final + existing selector stack (D-22)"
            )
        manifest["variants"].append(variant_entry)

    # Folds
    for cr in comparison_results:
        manifest["folds"].append({
            "year": cr.fold.year,
            "train_start": cr.fold.train_start,
            "train_end": cr.fold.train_end,
            "test_start": cr.fold.test_start,
            "test_end": cr.fold.test_end,
        })

    # Artifacts with SHA256
    artifact_key_to_filename: dict[str, str] = {
        "metrics_json": "shadow_comparison_result.json",
        "race_diff_parquet": "shadow_race_diff.parquet",
        "race_diff_csv": "shadow_race_diff.csv",
        "horse_diff_parquet": "shadow_horse_diff.parquet",
    }
    for key, filename in artifact_key_to_filename.items():
        path = artifact_paths.get(key)
        if path and path.exists():
            manifest["artifacts"][key] = {
                "path": filename,
                "sha256": _compute_sha256(path),
            }

    manifest_path = output_dir / "shadow_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


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
        from db.parquet_store import ParquetStore
        from db.readers import (
            load_entries,
            load_odds_snapshots,
            load_odds_time_series_range,
            load_payouts,
            load_races,
        )

        results: dict[str, Any] = {}  # variant_name -> BacktestResult
        variant_results: dict[str, VariantResult] = {}

        # P1 + P2: fold単位で全データを1回だけロードし全variantで共有
        store = self.store or ParquetStore()
        start = fold.test_start.replace("-", "")
        end = fold.test_end.replace("-", "")
        preloaded_race_df = load_races(store, start, end)
        preloaded_entry_df = load_entries(store, start, end)
        preloaded_final_odds_df = load_odds_snapshots(store, start, end)
        preloaded_payouts_df = load_payouts(store, start, end)
        preloaded_odds_ts = load_odds_time_series_range(store, start, end)
        logger.info(
            "Preloaded fold %d data: races=%d entries=%d final_odds=%d payouts=%d odds_ts=%d",
            fold.year,
            len(preloaded_race_df),
            len(preloaded_entry_df),
            len(preloaded_final_odds_df),
            len(preloaded_payouts_df),
            len(preloaded_odds_ts),
        )

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
            loaded_models._shadow_flags = {
                "enable_market_aware_calibrator": variant_cfg.enable_market_aware_calibrator,
                "enable_race_level_ranker": variant_cfg.enable_race_level_ranker,
            }

            engine = BacktestEngine(
                models=loaded_models,
                store=store,
                betting_mode=self.betting_mode,
                betting_target=self.betting_target,
                strategy_params=self.strategy_params,
                diag_prefix=f"shadow_{variant_cfg.variant_name}",
                min_bets_per_year=self.min_bets_per_year,
                preloaded_race_df=preloaded_race_df,
                preloaded_entry_df=preloaded_entry_df,
                preloaded_final_odds_df=preloaded_final_odds_df,
                preloaded_payouts_df=preloaded_payouts_df,
                preloaded_odds_ts=preloaded_odds_ts,
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
        """race_id + umaban ごとに baseline vs shadow をアライメント (D-03).

        Phase 43.5 FIX (P0-4): kakuteijyuni を全 variant DataFrames からマージ。
        Phase 43.5 FIX (P0-5): surface, tanodds, closing_win_odds, popularity を追加。
        """
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
                    shadow_subset[f"{vname}_{col}"] = shadow_df[col].values
                else:
                    shadow_subset[f"{vname}_{col}"] = np.nan
            shadow_subset[f"{vname}_selected"] = (
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

        # Phase 43.5 FIX (P0-4): kakuteijyuni を全 variant DataFrames からマージ。
        # baseline 側に NaN がある場合 (outer join で shadow-only 馬)、shadow から補完。
        kakutei_parts: list[pd.DataFrame] = []
        for vname, vdf in dfs.items():
            if "kakuteijyuni" in vdf.columns:
                part = vdf[key_cols + ["kakuteijyuni"]].drop_duplicates(subset=key_cols)
                kakutei_parts.append(part)
        if kakutei_parts:
            # 最初の non-NaN 値で補完するため、全 variant を結合して dropna
            combined_kakutei = pd.concat(kakutei_parts, ignore_index=True)
            combined_kakutei = combined_kakutei.dropna(subset=["kakuteijyuni"])
            combined_kakutei = combined_kakutei.drop_duplicates(subset=key_cols, keep="first")
            if "kakuteijyuni" in merged.columns:
                merged = merged.drop(columns=["kakuteijyuni"])
            merged = merged.merge(combined_kakutei, on=key_cols, how="left")

        # Phase 43.5 FIX (P0-5 rev2): surface, tanodds, closing_win_odds, popularity を追加。
        # baseline_df のみから取得すると shadow-only 馬 (outer join で追加された馬) が NaN になる。
        # kakuteijyuni と同様に、全 variant DataFrames から concat して first non-null で補完。
        extra_cols = ["surface", "tanodds", "closing_win_odds", "popularity"]
        for col in extra_cols:
            parts: list[pd.DataFrame] = []
            for vname, vdf in dfs.items():
                if col in vdf.columns:
                    part = vdf[key_cols + [col]].drop_duplicates(subset=key_cols)
                    parts.append(part)
            if parts:
                combined = pd.concat(parts, ignore_index=True)
                combined = combined.dropna(subset=[col])
                combined = combined.drop_duplicates(subset=key_cols, keep="first")
                if col in merged.columns:
                    merged = merged.drop(columns=[col])
                merged = merged.merge(combined, on=key_cols, how="left")

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
            if i == n_bins - 1:
                mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
            else:
                mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            avg_pred = y_pred[mask].mean()
            avg_true = y_true[mask].mean()
            ece += abs(avg_pred - avg_true) * (n_in_bin / total)

        return float(ece)
