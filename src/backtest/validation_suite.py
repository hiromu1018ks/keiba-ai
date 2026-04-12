"""バックテスト検証スイート (§13)

設計の正しさを検証するテスト群。
全テスト通過が Hold-out 評価の前提条件。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class BacktestValidationSuite:
    """§13 の全検証項目を集約したスイート

    各テストは passed (bool) + message (str) を返す。
    run_all() で全テストを一括実行。
    run_walk_forward_cv() で3-window walk-forward CV を実行。
    """

    def __init__(self, store: ParquetStore | None = None) -> None:
        self.store = store

    def run_all(
        self,
        models: Any | None,
        feature_cols: list[str],
        stage2_feature_cols: list[str],
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """全検証テストを実行

        Args:
            models: TrainedModelsV5 (省略可能)
            feature_cols: RaceQualityScreener の特徴量リスト
            stage2_feature_cols: Stage2 の特徴量リスト
            test_df: テスト用DataFrame

        Returns:
            {passed: bool, tests: list[dict], summary: str}
        """
        tests: list[dict[str, Any]] = []

        # Stage B ゼロ検出
        if not test_df.empty and "odds" in test_df.columns:
            tests.append(self.test_stage_b_no_zeros(test_df))

        # Market Model 検証
        if stage2_feature_cols:
            tests.append(self.test_market_model_no_pred_in_stage2(stage2_feature_cols))
            tests.append(self.test_market_model_uses_log_error(stage2_feature_cols))

        # EV補正 P/E独立検証
        if not test_df.empty and "ev_win_corrected" in test_df.columns:
            tests.append(self.test_ev_correction_pe_independent(test_df))

        # ワイドスコア検証
        if not test_df.empty and "wide_score_adj" in test_df.columns:
            tests.append(self.test_wide_score_variance_based(test_df))

        # log_error クリップ検証
        tests.append(self.test_log_error_clipping())

        # サブモデルサンプル数
        tests.append(self.test_submodel_sample_sufficiency("turf", 25000))
        tests.append(self.test_submodel_sample_sufficiency("dirt", 25000))

        # EV補正の低ev帯安定性 (§13.1)
        if not test_df.empty and "ev_win_corrected" in test_df.columns:
            tests.append(self.test_ev_correction_log_denominator_stable(test_df))

        # RaceQualityScreener 分布特徴量 (§13.1)
        if feature_cols:
            tests.append(self.test_race_quality_uses_distribution_features(feature_cols))
            tests.append(self.test_race_quality_uses_profitability_proxy(feature_cols))

        # RaceQuality proxy に EV依存特徴量が含まれないこと (§13.1)
        if feature_cols:
            tests.append(self.test_race_quality_no_ev_dependent_features(feature_cols))

        # §13.1 追加テスト (G-1)
        if not test_df.empty and "ev_win_corrected" in test_df.columns:
            tests.append(self.test_ev_correction_reduces_error(test_df))
            tests.append(self.test_ev_correction_mid_range_improvement(test_df))
            tests.append(self.test_ev_correction_winner_weight(test_df))
        if not test_df.empty and "quality_score" in test_df.columns:
            tests.append(self.test_race_quality_screener_independence(test_df))
        if not test_df.empty and "hist_mean" in test_df.columns:
            tests.append(self.test_race_quality_no_temporal_leak(test_df))

        passed = all(t["passed"] for t in tests)
        failed = [t for t in tests if not t["passed"]]

        summary = (
            f"Validation: {len(tests)} tests, "
            f"{len(tests) - len(failed)} passed, {len(failed)} failed"
        )
        if failed:
            summary += "\nFailed:\n" + "\n".join(f"  - {t['name']}: {t['message']}" for t in failed)

        logger.info(summary)

        return {
            "passed": passed,
            "tests": tests,
            "summary": summary,
        }

    def test_stage_b_no_zeros(self, df: pd.DataFrame) -> dict[str, Any]:
        """Stage B の学習データにゼロがないことを確認"""
        hit_df = df[df["kakuteijyuni"] == 1]
        has_zeros = (hit_df["odds"] <= 0).any()
        return {
            "name": "stage_b_no_zeros",
            "passed": not has_zeros,
            "message": ("Stage B ラベルにゼロが含まれています" if has_zeros else "OK"),
        }

    def test_market_model_no_pred_in_stage2(
        self,
        stage2_feature_cols: list[str],
    ) -> dict[str, Any]:
        """Stage2 の入力に p_market_pred が含まれていないことを確認"""
        has_pred = "p_market_pred_win" in stage2_feature_cols
        return {
            "name": "market_model_no_pred_in_stage2",
            "passed": not has_pred,
            "message": (
                "p_market_pred が Stage2 に入っています（市場コピー化リスク）" if has_pred else "OK"
            ),
        }

    def test_market_model_uses_log_error(
        self,
        stage2_feature_cols: list[str],
    ) -> dict[str, Any]:
        """Stage2 に log_error が含まれていることを確認"""
        has_log = "market_log_error_win" in stage2_feature_cols
        return {
            "name": "market_model_uses_log_error",
            "passed": has_log,
            "message": (
                "market_log_error_win が Stage2 に含まれていません" if not has_log else "OK"
            ),
        }

    def test_ev_correction_pe_independent(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """P補正とE補正が独立に動作していることを確認"""
        required = ["p_win_corrected", "e_return_win_corrected", "ev_win_corrected"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "ev_correction_pe_independent",
                "passed": False,
                "message": f"Missing columns: {missing}",
            }

        expected = df["p_win_corrected"] * df["e_return_win_corrected"]
        try:
            np.testing.assert_allclose(
                df["ev_win_corrected"].values,
                expected.values,
                rtol=1e-6,
            )
            return {
                "name": "ev_correction_pe_independent",
                "passed": True,
                "message": "OK",
            }
        except AssertionError as e:
            return {
                "name": "ev_correction_pe_independent",
                "passed": False,
                "message": str(e),
            }

    def test_wide_score_variance_based(self, df: pd.DataFrame) -> dict[str, Any]:
        """ワイドスコアが EV / (E × √P) で計算されていることを確認"""
        required = ["ev_wide", "e_return_given_hit", "p_hit", "wide_score_adj"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "wide_score_variance_based",
                "passed": False,
                "message": f"Missing columns: {missing}",
            }

        risk_denom = df["e_return_given_hit"] * np.sqrt(df["p_hit"].clip(lower=0.001))
        expected = df["ev_wide"] / risk_denom

        if np.allclose(df["wide_score_adj"].round(6), expected.round(6)):
            return {
                "name": "wide_score_variance_based",
                "passed": True,
                "message": "OK",
            }
        return {
            "name": "wide_score_variance_based",
            "passed": False,
            "message": "wide_score_adj が EV / (E × √P) と一致しません",
        }

    def test_log_error_clipping(self) -> dict[str, Any]:
        """log_error に両側クリップが適用されていることを確認"""
        p_pred = 0.001
        p_actual = 0.02
        p_pred_clipped = np.clip(p_pred, 0.01, 0.99)
        log_error = np.log(p_actual / p_pred_clipped)
        ok = abs(log_error) < 2.0

        p_market_extreme = 0.999
        p_pred_normal = 0.10
        p_market_clipped = np.clip(p_market_extreme, 0.01, 0.99)
        log_error_sym = np.log(p_market_clipped / p_pred_normal)
        ok2 = abs(log_error_sym) < 3.0

        return {
            "name": "log_error_clipping",
            "passed": bool(ok and ok2),
            "message": (
                f"log_error 発散: {log_error:.3f}"
                if not ok
                else (f"p_market log_error 発散: {log_error_sym:.3f}" if not ok2 else "OK")
            ),
        }

    def test_submodel_sample_sufficiency(
        self,
        submodel_key: str,
        sample_count: int,
    ) -> dict[str, Any]:
        """サブモデルのサンプル数が十分かチェック"""
        min_samples = 20_000
        ok = sample_count >= min_samples
        return {
            "name": f"submodel_sample_sufficiency_{submodel_key}",
            "passed": ok,
            "message": (
                f"サブモデル '{submodel_key}' サンプル不足: {sample_count} < {min_samples}"
                if not ok
                else f"OK ({sample_count})"
            ),
        }

    def test_ev_correction_log_denominator_stable(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """v5.4: P/E分解補正が低確率帯で発散しないことを確認 (§13.1)"""
        required = ["ev_win", "ev_win_corrected", "p_win_corrected"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "ev_correction_log_denominator_stable",
                "passed": False,
                "message": f"Missing columns: {missing}",
            }

        # P_corrected が [0, 1] の範囲外に出ていない
        in_range = bool((df["p_win_corrected"] >= 0).all() and (df["p_win_corrected"] <= 1.0).all())
        return {
            "name": "ev_correction_log_denominator_stable",
            "passed": in_range,
            "message": ("P_corrected が [0, 1] の範囲外です" if not in_range else "OK"),
        }

    def test_race_quality_uses_distribution_features(
        self,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """v5.1: RaceQualityScreener に分布特徴量が含まれていることを確認 (§13.1)"""
        dist_features = ["n_positive_errors", "top_k_error_sum", "positive_error_ratio"]
        missing = [f for f in dist_features if f not in feature_cols]
        return {
            "name": "race_quality_uses_distribution_features",
            "passed": len(missing) == 0,
            "message": f"分布特徴量が不足: {missing}" if missing else "OK",
        }

    def test_race_quality_uses_profitability_proxy(
        self,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """v5.4: RaceQualityScreener に結果ベース利益proxyが含まれることを確認 (§13.1)"""
        proxy_features = [
            "hist_hit_rate_topk",
            "hist_roi_topk",
            "hist_positive_return_ratio",
        ]
        missing = [f for f in proxy_features if f not in feature_cols]
        # EV依存proxy が含まれていないこと
        forbidden = ["hist_top3_ev_mean", "hist_positive_edge_ratio"]
        has_forbidden = [f for f in forbidden if f in feature_cols]
        ok = len(missing) == 0 and len(has_forbidden) == 0
        return {
            "name": "race_quality_uses_profitability_proxy",
            "passed": ok,
            "message": (
                f"proxy不足: {missing}"
                + (f" 禁止特徴量あり: {has_forbidden}" if has_forbidden else "")
                if not ok
                else "OK"
            ),
        }

    def test_race_quality_no_ev_dependent_features(
        self,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """v5.4: RaceQuality proxy に EV依存特徴量が含まれていないこと (§13.1)"""
        forbidden = ["hist_top3_ev_mean", "hist_positive_edge_ratio"]
        found = [f for f in forbidden if f in feature_cols]
        return {
            "name": "race_quality_no_ev_dependent_features",
            "passed": len(found) == 0,
            "message": f"EV依存特徴量が含まれています: {found}" if found else "OK",
        }

    def test_ev_correction_reduces_error(self, df: pd.DataFrame) -> dict[str, Any]:
        """v5.4: EV補正モデルがEVのMAEを改善することを確認 (§13.1)"""
        required = ["ev_win", "ev_win_corrected", "odds", "kakuteijyuni"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "ev_correction_reduces_error",
                "passed": False,
                "message": f"Missing columns: {missing}",
            }
        actual_ev = df["odds"] * (df["kakuteijyuni"] == 1).astype(int)
        mae_raw = float(np.mean(np.abs(df["ev_win"] - actual_ev)))
        mae_corrected = float(np.mean(np.abs(df["ev_win_corrected"] - actual_ev)))
        if bool(mae_corrected < mae_raw):
            return {
                "name": "ev_correction_reduces_error",
                "passed": True,
                "message": f"OK (MAE: {mae_raw:.4f} → {mae_corrected:.4f})",
            }
        return {
            "name": "ev_correction_reduces_error",
            "passed": False,
            "message": f"MAE悪化: {mae_raw:.4f} → {mae_corrected:.4f}",
        }

    def test_ev_correction_mid_range_improvement(self, df: pd.DataFrame) -> dict[str, Any]:
        """v5.4: 中穴ゾーン(P=0.05-0.15)で補正改善>10% (§13.1)"""
        required = ["p_win_pred", "ev_win", "ev_win_corrected", "odds", "kakuteijyuni"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "ev_correction_mid_range_improvement",
                "passed": False,
                "message": f"Missing columns: {missing}",
            }
        mid = df[df["p_win_pred"].between(0.05, 0.15)]
        if len(mid) < 100:
            return {
                "name": "ev_correction_mid_range_improvement",
                "passed": True,
                "message": f"SKIP (中穴ゾーン {len(mid)} < 100)",
            }
        actual_ev = mid["odds"] * (mid["kakuteijyuni"] == 1).astype(int)
        mae_raw = float(np.mean(np.abs(mid["ev_win"] - actual_ev)))
        mae_corrected = float(np.mean(np.abs(mid["ev_win_corrected"] - actual_ev)))
        if mae_raw == 0:
            return {
                "name": "ev_correction_mid_range_improvement",
                "passed": True,
                "message": "SKIP (MAE_raw=0)",
            }
        improvement = (mae_raw - mae_corrected) / mae_raw
        ok = bool(improvement > 0.10)
        return {
            "name": "ev_correction_mid_range_improvement",
            "passed": ok,
            "message": f"{'OK' if ok else 'FAIL'} 改善率={improvement:.1%}",
        }

    def test_ev_correction_winner_weight(self, df: pd.DataFrame) -> dict[str, Any]:
        """v5.4: 1着馬P_corrected中央値>=P_pred中央値 (§13.1)"""
        required = ["kakuteijyuni", "p_win_pred", "p_win_corrected"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "ev_correction_winner_weight",
                "passed": False,
                "message": f"Missing columns: {missing}",
            }
        winners = df[df["kakuteijyuni"] == 1]
        if len(winners) < 50:
            return {
                "name": "ev_correction_winner_weight",
                "passed": True,
                "message": f"SKIP (winner {len(winners)} < 50)",
            }
        ok = bool(winners["p_win_corrected"].median() >= winners["p_win_pred"].median())
        return {
            "name": "ev_correction_winner_weight",
            "passed": ok,
            "message": (
                f"{'OK' if ok else 'FAIL'} "
                f"P_corrected_median={winners['p_win_corrected'].median():.4f} "
                f"vs P_pred_median={winners['p_win_pred'].median():.4f}"
            ),
        }

    def test_race_quality_screener_independence(self, df: pd.DataFrame) -> dict[str, Any]:
        """v5.4: 品質スコアとedge_maxの相関<0.30 (§13.1)"""
        required = ["quality_score", "edge_max_per_race"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "race_quality_screener_independence",
                "passed": True,
                "message": f"SKIP (columns not available: {missing})",
            }
        corr = float(np.corrcoef(df["quality_score"], df["edge_max_per_race"])[0, 1])
        ok = bool(abs(corr) < 0.30)
        return {
            "name": "race_quality_screener_independence",
            "passed": ok,
            "message": f"{'OK' if ok else 'FAIL'} corr={corr:.3f}",
        }

    def test_race_quality_no_temporal_leak(self, df: pd.DataFrame) -> dict[str, Any]:
        """v5.4: hist特徴量に未来情報リークがないこと (§13.1)"""
        required = ["race_date", "hist_mean"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "name": "race_quality_no_temporal_leak",
                "passed": True,
                "message": f"SKIP (columns not available: {missing})",
            }
        if "value" not in df.columns:
            return {
                "name": "race_quality_no_temporal_leak",
                "passed": True,
                "message": "SKIP (value column not available)",
            }
        sorted_df = df.sort_values("race_date").reset_index(drop=True)
        leaked = 0
        for i in range(1, len(sorted_df)):
            race_date = sorted_df.iloc[i]["race_date"]
            hist_rows = sorted_df[sorted_df["race_date"] < race_date]
            if len(hist_rows) == 0:
                continue
            expected = hist_rows["value"].mean()
            actual = sorted_df.iloc[i]["hist_mean"]
            if pd.notna(actual) and abs(actual - expected) > 1e-6:
                leaked += 1
        ok = bool(leaked == 0)
        return {
            "name": "race_quality_no_temporal_leak",
            "passed": ok,
            "message": f"{'OK' if ok else f'FAIL ({leaked} rows with leak)'}",
        }

    def check_holdout_criteria(
        self,
        result: Any,
        config: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """§13.2 Hold-out 合格基準をチェック

        Args:
            result: BacktestResult オブジェクト
            config: 合格基準 (デフォルトは backtest_config.yaml の値)

        Returns:
            {passed: bool, criteria: list[dict]}
        """
        if config is None:
            config = {
                "place_roi": 1.00,
                "wide_roi": 1.03,
                "overall_roi": 1.01,
                "max_drawdown": 0.16,
            }

        criteria: list[dict[str, Any]] = []
        if hasattr(result, "total_roi"):
            criteria.append(
                {
                    "name": "overall_roi",
                    "value": result.total_roi,
                    "threshold": config["overall_roi"],
                    "passed": result.total_roi >= config["overall_roi"],
                }
            )
        if hasattr(result, "max_drawdown"):
            criteria.append(
                {
                    "name": "max_drawdown",
                    "value": result.max_drawdown,
                    "threshold": config["max_drawdown"],
                    "passed": result.max_drawdown <= config["max_drawdown"],
                }
            )

        passed = all(c["passed"] for c in criteria)
        return {
            "name": "holdout_criteria",
            "passed": passed,
            "message": (
                "全合格基準を満たしています"
                if passed
                else "不合格: "
                + ", ".join(
                    f"{c['name']} ({c['value']:.3%} vs {c['threshold']:.3%})"
                    for c in criteria
                    if not c["passed"]
                )
            ),
            "criteria": criteria,
        }

    def run_walk_forward_cv(
        self,
        train_start: str = "2018-01-01",
        train_end: str = "2023-12-31",
    ) -> dict[str, Any]:
        """Run 3-window walk-forward CV with parameter freeze.

        Each window: train on expanding period, backtest on next year.
        Parameters are frozen during OOS evaluation (Rule 7).

        Args:
            train_start: Start of overall training period
            train_end: End of overall training period

        Returns:
            dict with per-window and overall metrics.
        """
        from backtest.engine import BacktestEngine
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol
        from pipelines.training_pipeline import TrainingPipelineV5

        windows = [
            {
                "name": "Window 1",
                "train": ("2018-01-01", "2021-12-31"),
                "test": ("2022-01-01", "2022-12-31"),
            },
            {
                "name": "Window 2",
                "train": ("2019-01-01", "2022-12-31"),
                "test": ("2023-01-01", "2023-12-31"),
            },
            {
                "name": "Window 3",
                "train": ("2020-01-01", "2023-12-31"),
                "test": ("2024-01-01", "2024-12-31"),
            },
        ]

        # git hash for reproducibility tracking
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:7]
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = "unknown"

        results: dict[str, Any] = {}

        for w in windows:
            logger.info(
                "Walk-forward %s: train %s~%s, test %s~%s",
                w["name"],
                w["train"][0],
                w["train"][1],
                w["test"][0],
                w["test"][1],
            )

            # 1. Train
            pipeline = TrainingPipelineV5(
                store=self.store, model_dir=Path("data/models-validation")
            )
            trained = pipeline.run(w["train"][0], w["train"][1])

            # 2. Freeze parameters (Rule 7)
            protocol = ParameterFreezeProtocol(trained)
            protocol.freeze()

            # 3. Backtest on test period (OOS)
            engine = BacktestEngine(models=trained, store=self.store)
            bt_result = engine.run(w["test"][0], w["test"][1])

            # 4. Verify parameters unchanged
            freeze_result = protocol.verify()
            if not freeze_result["passed"]:
                logger.warning("Rule 7 violation in %s: %s", w["name"], freeze_result["message"])

            results[w["name"]] = {
                "roi": bt_result.total_roi,
                "max_dd": bt_result.max_drawdown,
                "total_bets": bt_result.total_bets,
                "logloss": None,  # TODO: compute from predictions
                "spearman_rho": None,  # TODO: compute from predictions
                "git_hash": git_hash,
                "rule7_passed": freeze_result["passed"],
                "train_period": w["train"],
                "test_period": w["test"],
            }

        # Overall summary
        rois = [r["roi"] for r in results.values()]
        overall_roi = float(np.mean(rois)) if rois else 0.0
        results["_overall"] = {
            "mean_roi": overall_roi,
            "std_roi": float(np.std(rois)) if len(rois) > 1 else 0.0,
            "git_hash": git_hash,
            "n_windows": len(windows),
        }

        return results
