"""WalkForwardCV のテスト"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from domain.models import TrainedModelsV5


@dataclass
class MockBacktestResult:
    """WalkForwardCV の結果 (テスト用)"""

    fold: int
    train_period: tuple[str, str]
    test_period: tuple[str, str]
    roi: float
    total_roi: float = 1.05
    max_drawdown: float = 0.10


class TestWalkForwardCV:
    """WalkForwardCV のテスト"""

    def test_generate_folds_expanding_window(self) -> None:
        """expanding window でフォールドが正しく生成される"""
        from models.walk_forward_cv import WalkForwardCV

        cv = WalkForwardCV(
            train_years=2,
            test_years=1,
            step_years=1,
        )
        folds = cv.generate_folds("2015-01-01", "2023-12-31")
        assert len(folds) == 3
        # 各フォールドの期間が正しく設定される
        for i, fold in enumerate(folds):
            assert fold.train_end < fold.test_start
            assert fold.test_start <= fold.test_end
            # フォールドインデックスが連続
            assert fold.fold_idx == i

    def test_generate_folds_with_step(self) -> None:
        """step_years でフォールド間のステップが制御される"""
        from models.walk_forward_cv import WalkForwardCV

        cv = WalkForwardCV(
            train_years=1,
            test_years=1,
            step_years=2,
        )
        folds = cv.generate_folds("2015-01-01", "2023-12-31")
        assert len(folds) >= 1
        # step=2 の場合、フォールド数は step=1 より少ない
        cv_step1 = WalkForwardCV(train_years=1, test_years=1, step_years=1)
        folds_step1 = cv_step1.generate_folds("2015-01-01", "2023-12-31")
        assert len(folds) <= len(folds_step1)

    def test_generate_folds_short_period(self) -> None:
        """期間が短すぎる場合は空リストを返す"""
        from models.walk_forward_cv import WalkForwardCV

        cv = WalkForwardCV(train_years=10, test_years=1, step_years=1)
        folds = cv.generate_folds("2020-01-01", "2020-12-31")
        assert len(folds) == 0

    def test_run_executes_all_folds(self) -> None:
        """全フォールドが実行される"""
        from models.walk_forward_cv import WalkForwardCV

        mock_pipeline = MagicMock()
        mock_models = MagicMock(spec=TrainedModelsV5)
        mock_pipeline.run.return_value = mock_models

        mock_engine = MagicMock()
        mock_engine.run.return_value = MockBacktestResult(
            fold=0, train_period=("", ""), test_period=("", ""), roi=1.05
        )

        cv = WalkForwardCV(
            pipeline=mock_pipeline,
            backtest_engine_factory=lambda m: mock_engine,
            train_years=1,
            test_years=1,
            step_years=1,
        )
        results = cv.run("2015-01-01", "2023-12-31")

        assert len(results.folds) == 5
        assert mock_pipeline.run.call_count == 5
        assert mock_engine.run.call_count == 5

    def test_run_creates_engine_per_fold(self) -> None:
        """各フォールドで独立したエンジンが生成される"""
        from models.walk_forward_cv import WalkForwardCV

        mock_pipeline = MagicMock()
        mock_models = MagicMock(spec=TrainedModelsV5)
        mock_pipeline.run.return_value = mock_models

        engine_calls: list[Any] = []

        def factory(models: Any) -> MagicMock:
            engine_calls.append(models)
            engine = MagicMock()
            engine.run.return_value = MockBacktestResult(
                fold=0, train_period=("", ""), test_period=("", ""), roi=1.05
            )
            return engine

        cv = WalkForwardCV(
            pipeline=mock_pipeline,
            backtest_engine_factory=factory,
            train_years=1,
            test_years=1,
            step_years=1,
        )
        cv.run("2015-01-01", "2023-12-31")

        # 各フォールドでファクトリが呼ばれる
        assert len(engine_calls) == 5

    def test_rule7_no_parameter_change_in_oos(self) -> None:
        """Rule 7: OOS期間でパラメータ変更が禁止される"""
        from models.walk_forward_cv import WalkForwardCV

        call_log: list[str] = []

        mock_pipeline = MagicMock()

        def mock_run(start: str, end: str) -> MagicMock:
            call_log.append(f"train:{start}-{end}")
            return MagicMock(spec=TrainedModelsV5)

        mock_pipeline.run.side_effect = mock_run

        mock_engine = MagicMock()
        mock_engine.run.return_value = MockBacktestResult(
            fold=0, train_period=("", ""), test_period=("", ""), roi=1.05
        )

        cv = WalkForwardCV(
            pipeline=mock_pipeline,
            backtest_engine_factory=lambda m: mock_engine,
            train_years=1,
            test_years=1,
            step_years=1,
        )
        cv.run("2015-01-01", "2023-12-31")

        # 学習はフォールドごとに1回だけ
        train_calls = [c for c in call_log if c.startswith("train:")]
        assert len(train_calls) == 5

        # バックテストは学習と同数
        assert mock_engine.run.call_count == len(train_calls)

    def test_run_without_pipeline_raises(self) -> None:
        """pipeline なしで run() を呼ぶと RuntimeError"""
        import pytest

        from models.walk_forward_cv import WalkForwardCV

        cv = WalkForwardCV()
        with pytest.raises(RuntimeError, match="pipeline is required"):
            cv.run("2015-01-01", "2020-12-31")

    def test_run_without_engine_factory_skips_backtest(self) -> None:
        """engine_factory なしでもエラーにならない (バックテストをスキップ)"""
        from models.walk_forward_cv import WalkForwardCV

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(spec=TrainedModelsV5)

        cv = WalkForwardCV(pipeline=mock_pipeline, train_years=1, test_years=1, step_years=1)
        result = cv.run("2015-01-01", "2023-12-31")

        # フォールドは生成されるがバックテスト結果はない
        assert len(result.folds) == 5
        assert len(result.fold_results) == 0
        assert result.mean_roi == 0.0

    def test_cv_result_aggregation(self) -> None:
        """CVResult の集計が正しい"""
        from models.walk_forward_cv import WalkForwardCV

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(spec=TrainedModelsV5)

        fold_idx = 0

        def make_engine(models: Any) -> MagicMock:
            nonlocal fold_idx
            engine = MagicMock()
            engine.run.return_value = MockBacktestResult(
                fold=fold_idx,
                train_period=("", ""),
                test_period=("", ""),
                roi=1.0,
                total_roi=1.05 + fold_idx * 0.01,
                max_drawdown=0.05 + fold_idx * 0.02,
            )
            fold_idx += 1
            return engine

        cv = WalkForwardCV(
            pipeline=mock_pipeline,
            backtest_engine_factory=make_engine,
            train_years=1,
            test_years=1,
            step_years=1,
        )
        result = cv.run("2015-01-01", "2023-12-31")

        # max_drawdown は全フォールドの最大値
        assert result.max_drawdown > 0.0
        # mean_roi は正の値
        assert result.mean_roi > 1.0
        # std_roi が計算される
        assert result.std_roi > 0.0

    def test_fold_order(self) -> None:
        """フォールドが時系列順に生成される"""
        from models.walk_forward_cv import WalkForwardCV

        cv = WalkForwardCV(train_years=2, test_years=1, step_years=1)
        folds = cv.generate_folds("2015-01-01", "2023-12-31")

        for i in range(1, len(folds)):
            assert folds[i].train_start >= folds[i - 1].train_start


class TestCVResult:
    """CVResult データクラスのテスト"""

    def test_cv_result_default_values(self) -> None:
        """デフォルト値が正しい"""
        from models.walk_forward_cv import CVResult

        result = CVResult()
        assert result.folds == []
        assert result.fold_results == []
        assert result.mean_roi == 0.0
        assert result.std_roi == 0.0
        assert result.max_drawdown == 0.0

    def test_cv_result_summary(self) -> None:
        """summary() が文字列を返す"""
        from models.walk_forward_cv import CVResult

        result = CVResult(mean_roi=1.05, std_roi=0.02, max_drawdown=0.10)
        summary = result.summary()
        assert "Mean ROI" in summary
        assert "105.000%" in summary
        assert "Std ROI" in summary
        assert "Max DD" in summary

    def test_cv_result_summary_zero_folds(self) -> None:
        """0フォールドの summary が正しい"""
        from models.walk_forward_cv import CVResult

        result = CVResult()
        summary = result.summary()
        assert "0 folds" in summary


class TestFold:
    """Fold データクラスのテスト"""

    def test_fold_creation(self) -> None:
        """Fold が正しく作成される"""
        from models.walk_forward_cv import Fold

        fold = Fold(
            fold_idx=0,
            train_start="2020-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
        )
        assert fold.fold_idx == 0
        assert fold.train_start == "2020-01-01"
        assert fold.test_end == "2024-12-31"


class TestAddYearsDt:
    """_add_years_dt ヘルパーのテスト"""

    def test_add_years(self) -> None:
        from datetime import datetime

        from models.walk_forward_cv import _add_years_dt

        dt = datetime(2020, 1, 1)
        result = _add_years_dt(dt, 1)
        # 365.25 * 1 = 365 days -> Dec 31, 2020
        assert result.year == 2020
        assert result.month == 12
        assert result.day == 31

    def test_add_years_4(self) -> None:
        from datetime import datetime

        from models.walk_forward_cv import _add_years_dt

        dt = datetime(2015, 1, 1)
        result = _add_years_dt(dt, 4)
        # 365.25 * 4 = 1461 days -> Jan 1, 2019 (2016 is leap year)
        assert result.year == 2019

    def test_add_zero_years(self) -> None:
        from datetime import datetime

        from models.walk_forward_cv import _add_years_dt

        dt = datetime(2020, 6, 15)
        result = _add_years_dt(dt, 0)
        assert result == dt


# ---------------------------------------------------------------------------
# Phase 4: Walk-Forward Validation Infrastructure Tests
# ---------------------------------------------------------------------------


class TestFoldResult:
    """FoldResult データクラスのテスト"""

    def test_fold_result_defaults(self) -> None:
        from models.walk_forward_cv import FoldResult

        fr = FoldResult(
            fold_idx=0, train_start="2020-01-01", train_end="2023-12-31",
            test_start="2024-01-01", test_end="2024-12-31",
        )
        assert fr.train_roi == 0.0
        assert fr.top_features == []
        assert fr.feature_ranking == {}

    def test_fold_result_roi_gap(self) -> None:
        from models.walk_forward_cv import FoldResult

        fr = FoldResult(
            fold_idx=0, train_start="2020-01-01", train_end="2023-12-31",
            test_start="2024-01-01", test_end="2024-12-31",
            train_roi=1.30, test_roi=1.05, roi_gap=0.25,
        )
        assert fr.roi_gap == 0.25


class TestWFValidationResult:
    """WFValidationResult データクラスのテスト"""

    def test_defaults(self) -> None:
        from models.walk_forward_cv import WFValidationResult

        r = WFValidationResult()
        assert r.folds == []
        assert r.pool_roi == 0.0
        assert r.overall_verdict == "PASS"


class TestExtractFeatureRanking:
    """extract_feature_ranking のテスト"""

    def test_returns_top_features(self) -> None:
        from models.walk_forward_cv import extract_feature_ranking

        import lightgbm as lgb
        import numpy as np

        data = np.random.rand(100, 5)
        ds = lgb.Dataset(
            data, label=np.random.randint(0, 2, 100),
            feature_name=["f1", "f2", "f3", "f4", "f5"],
        )
        model = lgb.train(
            {"objective": "binary", "verbose": -1, "num_leaves": 4},
            ds, num_boost_round=10,
        )
        ranking, top = extract_feature_ranking(model, top_n=3)
        assert len(top) == 3
        assert len(ranking) == 3
        assert all(f in ranking for f in top)


class TestComputeFeatureStability:
    """compute_feature_stability のテスト"""

    def test_identical_rankings(self) -> None:
        from models.walk_forward_cv import compute_feature_stability

        r1 = {"a": 0, "b": 1, "c": 2, "d": 3}
        r2 = {"a": 0, "b": 1, "c": 2, "d": 3}
        rho = compute_feature_stability([r1, r2])
        assert rho == 1.0

    def test_reversed_rankings(self) -> None:
        from models.walk_forward_cv import compute_feature_stability

        r1 = {"a": 0, "b": 1, "c": 2, "d": 3}
        r2 = {"a": 3, "b": 2, "c": 1, "d": 0}
        rho = compute_feature_stability([r1, r2])
        assert rho == -1.0

    def test_single_ranking_returns_nan(self) -> None:
        import math

        from models.walk_forward_cv import compute_feature_stability

        rho = compute_feature_stability([{"a": 0, "b": 1}])
        assert math.isnan(rho)


class TestJudgeOverfitting:
    """judge_overfitting のテスト (Per D-08, D-13)"""

    def _make_result(
        self, train_roi: float = 1.2, test_roi: float = 1.1, spearman_rho: float = 0.8,
    ) -> "WFValidationResult":
        from models.walk_forward_cv import FoldResult, WFValidationResult

        return WFValidationResult(
            folds=[
                FoldResult(
                    fold_idx=0, train_start="2020-01-01", train_end="2023-12-31",
                    test_start="2024-01-01", test_end="2024-12-31",
                    train_roi=train_roi, test_roi=test_roi,
                    roi_gap=train_roi - test_roi,
                ),
                FoldResult(
                    fold_idx=1, train_start="2021-01-01", train_end="2024-12-31",
                    test_start="2025-01-01", test_end="2025-12-31",
                    train_roi=train_roi, test_roi=test_roi,
                    roi_gap=train_roi - test_roi,
                ),
            ],
            spearman_rho=spearman_rho,
        )

    def test_all_pass(self) -> None:
        from models.walk_forward_cv import judge_overfitting

        r = self._make_result(train_roi=1.15, test_roi=1.10, spearman_rho=0.8)
        judge_overfitting(r)
        assert r.overall_verdict == "PASS"

    def test_roi_gap_warning(self) -> None:
        """D-08: ROI gap 20-30% -> WARNING"""
        from models.walk_forward_cv import judge_overfitting

        r = self._make_result(train_roi=1.35, test_roi=1.10, spearman_rho=0.8)
        judge_overfitting(r)
        assert r.roi_gap_verdict == "WARNING"

    def test_roi_gap_fail(self) -> None:
        """D-08: ROI gap > 30% -> FAIL"""
        from models.walk_forward_cv import judge_overfitting

        r = self._make_result(train_roi=1.50, test_roi=1.10, spearman_rho=0.8)
        judge_overfitting(r)
        assert r.roi_gap_verdict == "FAIL"
        assert r.overall_verdict == "FAIL"

    def test_consistency_warning(self) -> None:
        """D-07: 一方のみ>100% -> WARNING"""
        from models.walk_forward_cv import FoldResult, WFValidationResult, judge_overfitting

        r = WFValidationResult(
            folds=[
                FoldResult(
                    fold_idx=0, train_start="", train_end="", test_start="", test_end="",
                    train_roi=1.1, test_roi=1.05, roi_gap=0.05,
                ),
                FoldResult(
                    fold_idx=1, train_start="", train_end="", test_start="", test_end="",
                    train_roi=1.1, test_roi=0.95, roi_gap=0.15,
                ),
            ],
            spearman_rho=0.8,
        )
        judge_overfitting(r)
        assert r.consistency_verdict == "WARNING"

    def test_stability_warning(self) -> None:
        """D-09: rho < 0.5 -> WARNING"""
        from models.walk_forward_cv import judge_overfitting

        r = self._make_result(train_roi=1.15, test_roi=1.10, spearman_rho=0.3)
        judge_overfitting(r)
        assert r.stability_verdict == "WARNING"
