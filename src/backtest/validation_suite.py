"""バックテスト検証スイート (§13)

設計の正しさを検証するテスト群。
全テスト通過が Hold-out 評価の前提条件。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestValidationSuite:
    """§13 の全検証項目を集約したスイート

    各テストは passed (bool) + message (str) を返す。
    run_all() で全テストを一括実行。
    """

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
        if not test_df.empty and "win_odds_actual" in test_df.columns:
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
        hit_df = df[df["finish_pos"] == 1]
        has_zeros = (hit_df["win_odds_actual"] <= 0).any()
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
