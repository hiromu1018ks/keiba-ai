"""ウォークフォワード交差検証 (Rule 7: OOS期間パラメータ固定)

expanding window で時系列交差検証を実行。
各フォールド: train → freeze parameters → test (no modification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

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
            result.mean_roi = float(np.mean(rois))
            result.std_roi = float(np.std(rois))

        logger.info(result.summary())
        return result


def _add_years_dt(dt: datetime, years: int) -> datetime:
    """datetime に年を加算 (簡易: 365.25日/年)"""
    return dt + timedelta(days=int(365.25 * years))


# ---------------------------------------------------------------------------
# Phase 4: Walk-Forward Validation Infrastructure
# ---------------------------------------------------------------------------

import lightgbm as lgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


@dataclass
class FoldResult:
    """単一フォールドのWF検証結果"""

    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_roi: float = 0.0
    test_roi: float = 0.0
    roi_gap: float = 0.0
    train_bets: int = 0
    test_bets: int = 0
    train_stake: float = 0.0
    test_stake: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    top_features: list[str] = field(default_factory=list)
    feature_ranking: dict[str, int] = field(default_factory=dict)


@dataclass
class WFValidationResult:
    """Walk-forward検証の全体結果"""

    folds: list[FoldResult] = field(default_factory=list)
    pool_roi: float = 0.0
    weighted_roi: float = 0.0
    total_stake: float = 0.0
    total_return: float = 0.0
    total_bets: int = 0
    roi_gap_verdict: str = "PASS"
    consistency_verdict: str = "PASS"
    stability_verdict: str = "PASS"
    overall_verdict: str = "PASS"
    spearman_rho: float = 0.0
    roi_gap_max: float = 0.0
    git_hash: str = ""


def extract_feature_ranking(
    model: lgb.Booster, top_n: int = 10,
) -> tuple[dict[str, int], list[str]]:
    """LightGBMモデルからtop-N特徴量の順位を取得

    Args:
        model: 学習済みlgb.Booster
        top_n: 上位特徴量数

    Returns:
        (ranking_dict, top_features_list)
        ranking_dict: {feature_name: rank} (0=top, 1=2nd, ...)
        top_features_list: top-N特徴量名のリスト(順位順)
    """
    feature_names = model.feature_name()
    gain = model.feature_importance(importance_type="gain")
    sorted_features = sorted(zip(feature_names, gain), key=lambda x: -x[1])
    top_features = [f for f, _ in sorted_features[:top_n]]
    ranking = {f: rank for rank, (f, _) in enumerate(sorted_features[:top_n])}
    return ranking, top_features


def compute_feature_stability(
    rankings: list[dict[str, int]], top_n: int = 10,
) -> float:
    """複数フォールド間の特徴量順位相関(平均)を計算

    共通特徴量が3未満の場合はNaNを返す。
    """
    if len(rankings) < 2:
        return float("nan")

    all_features: set[str] = set()
    for r in rankings:
        top = sorted(r, key=r.get)[:top_n]  # type: ignore[arg-type]
        all_features.update(top)

    if len(all_features) < 3:
        return float("nan")

    rhos: list[float] = []
    for i in range(len(rankings) - 1):
        r1 = rankings[i]
        r2 = rankings[i + 1]
        common = [f for f in all_features if f in r1 and f in r2]
        if len(common) < 3:
            continue
        ranks1 = [r1[f] for f in common]
        ranks2 = [r2[f] for f in common]
        rho, _ = spearmanr(ranks1, ranks2)
        rhos.append(float(rho))

    return float(np.mean(rhos)) if rhos else float("nan")


def judge_overfitting(
    result: WFValidationResult,
    warning_gap: float = 0.20,
    fail_gap: float = 0.30,
    min_rho: float = 0.5,
) -> None:
    """3基準の自動判定を実行し、結果をresultに反映

    基準1 ROI gap (train - test の最大値):
      < 20% -> PASS, 20-30% -> WARNING, > 30% -> FAIL
    基準2 両年度ROI一貫性:
      全年度test_roi > 100% -> PASS, 一方のみ -> WARNING, 全<100% -> FAIL
    基準3 Feature importance安定性:
      rho >= 0.5 -> PASS, rho < 0.5 -> WARNING
    全PASS -> overall PASS, 一つでもFAIL -> overall FAIL, WARNINGのみ -> overall WARNING
    """
    # 基準1: ROI gap
    gaps = [f.roi_gap for f in result.folds]
    max_gap = max(gaps) if gaps else 0.0
    result.roi_gap_max = max_gap
    if max_gap > fail_gap:
        result.roi_gap_verdict = "FAIL"
    elif max_gap > warning_gap:
        result.roi_gap_verdict = "WARNING"
    else:
        result.roi_gap_verdict = "PASS"

    # 基準2: 一貫性
    test_rois = [f.test_roi for f in result.folds]
    above_100 = sum(1 for r in test_rois if r > 1.0)
    if len(test_rois) > 0 and above_100 == len(test_rois):
        result.consistency_verdict = "PASS"
    elif above_100 > 0:
        result.consistency_verdict = "WARNING"
    else:
        result.consistency_verdict = "FAIL"

    # 基準3: 安定性
    if not np.isnan(result.spearman_rho):
        if result.spearman_rho >= min_rho:
            result.stability_verdict = "PASS"
        else:
            result.stability_verdict = "WARNING"

    # 総合判定
    verdicts = [result.roi_gap_verdict, result.consistency_verdict, result.stability_verdict]
    if "FAIL" in verdicts:
        result.overall_verdict = "FAIL"
    elif "WARNING" in verdicts:
        result.overall_verdict = "WARNING"
    else:
        result.overall_verdict = "PASS"
