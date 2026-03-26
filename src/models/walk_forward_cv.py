"""ウォークフォワード交差検証 (Rule 7: OOS期間パラメータ固定)

expanding window で時系列交差検証を実行。
各フォールド: train → freeze parameters → test (no modification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from backtest.engine import BacktestEngine
    from domain.models import TrainedModelsV5
    from pipelines.training_pipeline import TrainingPipelineV5

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """単一フォールドの期間定義"""

    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class CVResult:
    """WalkForwardCV の集計結果"""

    folds: list[Fold] = field(default_factory=list)
    fold_results: list[Any] = field(default_factory=list)
    mean_roi: float = 0.0
    std_roi: float = 0.0
    max_drawdown: float = 0.0

    def summary(self) -> str:
        return (
            f"WalkForwardCV: {len(self.folds)} folds\n"
            f"  Mean ROI: {self.mean_roi:.3%}\n"
            f"  Std ROI:  {self.std_roi:.3%}\n"
            f"  Max DD:   {self.max_drawdown:.3%}"
        )


class WalkForwardCV:
    """ウォークフォワード交差検証

    Rule 7: out-of-sample期間ではパラメータ変更を一切行わない。
    各フォールドで独立した学習→評価を行う。

    Args:
        pipeline: 学習パイプライン
        backtest_engine_factory: BacktestEngine を生成するファクトリ関数。
            TrainedModelsV5 を受け取り BacktestEngine を返す。
            例: ``lambda models: BacktestEngine(models=models)``
        train_years: 学習期間 (年)
        test_years: テスト期間 (年)
        step_years: フォールド間のステップ (年)
    """

    def __init__(
        self,
        pipeline: TrainingPipelineV5 | None = None,
        backtest_engine_factory: Callable[[TrainedModelsV5], BacktestEngine] | None = None,
        train_years: int = 4,
        test_years: int = 1,
        step_years: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self.backtest_engine_factory = backtest_engine_factory
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years

    def generate_folds(
        self,
        start_date: str,
        end_date: str,
    ) -> list[Fold]:
        """フォールド期間のリストを生成

        Args:
            start_date: 全体開始日 (YYYY-MM-DD)
            end_date: 全体終了日 (YYYY-MM-DD)

        Returns:
            Fold のリスト
        """
        folds: list[Fold] = []
        current_start = datetime.strptime(start_date, "%Y-%m-%d")
        overall_end = datetime.strptime(end_date, "%Y-%m-%d")
        fold_idx = 0

        while True:
            train_start = current_start
            train_end_dt = _add_years_dt(train_start, self.train_years) - timedelta(days=1)
            test_start_dt = train_end_dt + timedelta(days=1)
            test_end_dt = _add_years_dt(test_start_dt, self.test_years) - timedelta(days=1)

            if test_start_dt > overall_end:
                break

            # test_end を overall_end でキャップ
            test_end_dt = min(test_end_dt, overall_end)

            fold = Fold(
                fold_idx=fold_idx,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end_dt.strftime("%Y-%m-%d"),
                test_start=test_start_dt.strftime("%Y-%m-%d"),
                test_end=test_end_dt.strftime("%Y-%m-%d"),
            )
            folds.append(fold)

            # 次のフォールドの開始 = test_start + step
            current_start = _add_years_dt(test_start_dt, self.step_years)
            fold_idx += 1

        return folds

    def run(
        self,
        start_date: str,
        end_date: str,
    ) -> CVResult:
        """全フォールドを実行

        各フォールド:
          1. pipeline.run(train_start, train_end) で学習
          2. backtest_engine_factory(models).run(test_start, test_end) で評価
          3. 結果を記録 (OOS期間ではパラメータ変更なし)

        Note:
            pipeline は必須 (未設定時は RuntimeError)。
            backtest_engine_factory は任意 (未設定時はバックテストをスキップ)。

        Args:
            start_date: 全体開始日
            end_date: 全体終了日

        Returns:
            CVResult (全フォールドの集計)
        """
        folds = self.generate_folds(start_date, end_date)
        result = CVResult(folds=folds)

        rois: list[float] = []

        for fold in folds:
            logger.info(
                f"Fold {fold.fold_idx}: train {fold.train_start}~{fold.train_end}, "
                f"test {fold.test_start}~{fold.test_end}"
            )

            # 1. 学習 (パラメータを凍結)
            if self.pipeline is not None:
                models = self.pipeline.run(fold.train_start, fold.train_end)
            else:
                raise RuntimeError("pipeline is required for run()")

            # 2. バックテスト (OOS — パラメータ変更なし)
            if self.backtest_engine_factory is not None:
                engine = self.backtest_engine_factory(models)
                fold_result = engine.run(fold.test_start, fold.test_end)
                result.fold_results.append(fold_result)

                # ROI を収集
                if hasattr(fold_result, "total_roi"):
                    rois.append(fold_result.total_roi)
                if hasattr(fold_result, "max_drawdown"):
                    result.max_drawdown = max(result.max_drawdown, fold_result.max_drawdown)

        # 3. 集計
        if rois:
            import numpy as np

            result.mean_roi = float(np.mean(rois))
            result.std_roi = float(np.std(rois))

        logger.info(result.summary())
        return result


def _add_years_dt(dt: datetime, years: int) -> datetime:
    """datetime に年を加算 (簡易: 365.25日/年)"""
    return dt + timedelta(days=int(365.25 * years))
