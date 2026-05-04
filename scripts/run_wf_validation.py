"""Walk-forward検証スクリプト (Phase 4)

2フォールド(2024, 2025テスト)のウォークフォワード検証を実行し、
過学習検出・ROI検証・feature importance安定性評価を自動実行する。

Usage:
    python scripts/run_wf_validation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fold definitions (Per D-02, D-03)
# ---------------------------------------------------------------------------
FOLDS: list[dict[str, str]] = [
    {
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
]


def _save_intermediate_result(result_dict: dict[str, Any], path: Path) -> None:
    """フォールド完了ごとに途中結果をJSONに書き出す (Per Pitfall 4)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _extract_all_feature_rankings(models: Any) -> tuple[dict[str, int], list[str]]:
    """芝/ダート両方のstage1 + win.hit_modelからfeature importanceを統合して取得

    各サーフェスの各モデルからtop-10を取得し、統合ランキングを作成。
    複数モデルで出現する特徴量は最高順位を採用。
    """
    from models.walk_forward_cv import extract_feature_ranking

    all_rankings: dict[str, int] = {}
    all_top_features: list[str] = []

    for _surface, sub in models.submodels.items():
        # Stage1 (AbilityModel) — models dict contains lgb.Booster per surface
        if hasattr(sub.stage1, "models") and sub.stage1.models:
            for _key, booster in sub.stage1.models.items():
                ranking, top = extract_feature_ranking(booster, top_n=10)
                for f, r in ranking.items():
                    if f not in all_rankings or r < all_rankings[f]:
                        all_rankings[f] = r
                all_top_features.extend(top)

        # Win hit model (lgb.Booster)
        if hasattr(sub.win, "hit_model") and sub.win.hit_model is not None:
            ranking, top = extract_feature_ranking(sub.win.hit_model, top_n=10)
            for f, r in ranking.items():
                if f not in all_rankings or r < all_rankings[f]:
                    all_rankings[f] = r
            all_top_features.extend(top)

    # top_features: 出現頻度順でソート(複数モデルで出現する特徴量を優先)
    freq = Counter(all_top_features)
    top_features = [f for f, _ in freq.most_common(10)]

    return all_rankings, top_features


def main() -> None:
    """WF検証のメインループ"""
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument(
        "--betting-target",
        choices=["win", "place", "wide"],
        default="win",
        help="ベッティング対象 (デフォルト: win)",
    )
    args = parser.parse_args()

    # git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"

    from models.walk_forward_cv import (
        FoldResult,
        WFValidationResult,
        compute_feature_stability,
        judge_overfitting,
    )

    wf_result = WFValidationResult(git_hash=git_hash)
    output_path = Path(ROOT) / "data" / "backtest" / "wf_validation_result.json"

    # 予想時間表示 (Per Pitfall 4)
    est_per_fold_min = 120  # ~2時間/フォールド (train bt + test bt)
    print("=== Walk-Forward Validation ===")
    print(f"Folds: {len(FOLDS)}")
    print(f"Estimated time: ~{est_per_fold_min * len(FOLDS)} min")
    print()

    fold_rankings: list[dict[str, int]] = []

    for i, fold_def in enumerate(FOLDS):
        print(
            f"--- Fold {i}: train {fold_def['train_start']}~{fold_def['train_end']}, "
            f"test {fold_def['test_start']}~{fold_def['test_end']} ---",
        )
        logger.info(
            "Fold %d: train %s~%s, test %s~%s",
            i, fold_def["train_start"], fold_def["train_end"],
            fold_def["test_start"], fold_def["test_end"],
        )

        # 1. 学習
        t0 = time.time()
        year_model_dir = Path("data/models-wf-validation") / str(i)
        year_model_dir.mkdir(parents=True, exist_ok=True)

        from pipelines.training_pipeline import TrainingPipelineV5

        pipeline = TrainingPipelineV5(store=store, model_dir=year_model_dir)
        models = pipeline.run(fold_def["train_start"], fold_def["train_end"])
        elapsed_train = time.time() - t0
        logger.info("Fold %d: 学習完了 (%.0f秒)", i, elapsed_train)

        # 2. Feature importance抽出
        ranking, top_features = _extract_all_feature_rankings(models)
        fold_rankings.append(ranking)
        logger.info("Fold %d: top features = %s", i, top_features[:5])

        # 3a. Test期間バックテスト
        t1 = time.time()
        from backtest.engine import BacktestEngine

        test_engine = BacktestEngine(
            models=models, store=store, diag_prefix=f"wf_{i}_test",
            betting_target=args.betting_target,
        )
        test_result = test_engine.run(fold_def["test_start"], fold_def["test_end"])
        elapsed_test = time.time() - t1
        logger.info(
            "Fold %d: テストBT完了 (%.0f秒) ROI=%.1f%%",
            i, elapsed_test, test_result.total_roi * 100,
        )

        # 3b. Train期間バックテスト (Per D-05)
        # 別インスタンスで状態汚染を回避 (Per Pitfall 2)
        t2 = time.time()
        train_engine = BacktestEngine(
            models=models, store=store, diag_prefix=f"wf_{i}_train",
            betting_target=args.betting_target,
        )
        train_result = train_engine.run(fold_def["train_start"], fold_def["train_end"])
        elapsed_train_bt = time.time() - t2
        logger.info(
            "Fold %d: 学習BT完了 (%.0f秒) ROI=%.1f%%",
            i, elapsed_train_bt, train_result.total_roi * 100,
        )

        # 4. FoldResult作成
        fold_result = FoldResult(
            fold_idx=i,
            train_start=fold_def["train_start"],
            train_end=fold_def["train_end"],
            test_start=fold_def["test_start"],
            test_end=fold_def["test_end"],
            train_roi=train_result.total_roi,
            test_roi=test_result.total_roi,
            roi_gap=train_result.total_roi - test_result.total_roi,
            train_bets=train_result.total_bets,
            test_bets=test_result.total_bets,
            train_stake=train_result.total_stake,
            test_stake=test_result.total_stake,
            train_return=train_result.total_return,
            test_return=test_result.total_return,
            top_features=top_features,
            feature_ranking=ranking,
        )
        wf_result.folds.append(fold_result)

        # 5. 途中結果をセーブ (Per Pitfall 4)
        result_dict = asdict(wf_result)
        _save_intermediate_result(result_dict, output_path)
        logger.info("Fold %d: 途中結果保存 -> %s", i, output_path)

        print(
            f"  学習: %.0f秒 | テストBT: %.0f秒 | 学習BT: %.0f秒"
            % (elapsed_train, elapsed_test, elapsed_train_bt),
        )
        print(
            f"  Train ROI: %.1f%% | Test ROI: %.1f%% | Gap: %.1f%%"
            % (
                train_result.total_roi * 100,
                test_result.total_roi * 100,
                (train_result.total_roi - test_result.total_roi) * 100,
            ),
        )
        print(
            f"  Test bets: %d | Test stake: %.0f | Test return: %.0f"
            % (test_result.total_bets, test_result.total_stake, test_result.total_return),
        )
        print()

    # 6. 集計 (Per D-10, D-11)
    total_test_stake = sum(f.test_stake for f in wf_result.folds)
    total_test_return = sum(f.test_return for f in wf_result.folds)
    total_test_bets = sum(f.test_bets for f in wf_result.folds)
    wf_result.total_stake = total_test_stake
    wf_result.total_return = total_test_return
    wf_result.total_bets = total_test_bets

    # D-10: プールROI
    wf_result.pool_roi = (
        total_test_return / total_test_stake if total_test_stake > 0 else 0.0
    )

    # D-11: ベット数加重ROI
    if total_test_bets > 0:
        wf_result.weighted_roi = (
            sum(f.test_roi * f.test_bets for f in wf_result.folds) / total_test_bets
        )
    else:
        wf_result.weighted_roi = 0.0

    # 7. Feature importance安定性 (Per D-09)
    rho = compute_feature_stability(fold_rankings)
    wf_result.spearman_rho = rho
    logger.info("Feature stability Spearman rho = %.3f", rho)

    # 8. 過学習判定 (Per D-07, D-08, D-13)
    judge_overfitting(wf_result)

    # 9. MLflow記録 (Per D-12)
    mlflow.set_experiment("wf_validation")
    with mlflow.start_run(
        run_name=f"wf_{FOLDS[0]['test_start'][:4]}_{FOLDS[-1]['test_end'][:4]}",
    ):
        mlflow.log_params({
            "n_folds": len(FOLDS),
            "train_years": 4,
            "test_years": 1,
            "git_hash": git_hash,
        })
        mlflow.log_metrics({
            "pool_roi": wf_result.pool_roi,
            "weighted_roi": wf_result.weighted_roi,
            "spearman_rho": float(rho) if not np.isnan(rho) else -1.0,
            "roi_gap_max": wf_result.roi_gap_max,
            "total_bets": wf_result.total_bets,
        })
        for fold_idx, fold in enumerate(wf_result.folds):
            mlflow.log_metrics({
                f"fold_{fold_idx}_train_roi": fold.train_roi,
                f"fold_{fold_idx}_test_roi": fold.test_roi,
                f"fold_{fold_idx}_roi_gap": fold.roi_gap,
                f"fold_{fold_idx}_test_bets": fold.test_bets,
            })
        mlflow.set_tag("verdict", wf_result.overall_verdict)

    # 10. 最終結果保存 (Per D-14)
    result_dict = asdict(wf_result)
    _save_intermediate_result(result_dict, output_path)

    # 11. 結果表示
    print("=" * 60)
    print("  Walk-Forward Validation Result")
    print("=" * 60)
    for fold in wf_result.folds:
        print(
            f"  Fold {fold.fold_idx}: Train ROI={fold.train_roi:.1%} | "
            f"Test ROI={fold.test_roi:.1%} | Gap={fold.roi_gap:.1%}",
        )
    print(f"  Pool ROI (D-10):         {wf_result.pool_roi:.1%}")
    print(f"  Weighted ROI (D-11):     {wf_result.weighted_roi:.1%}")
    print(f"  Feature Stability rho:   {wf_result.spearman_rho:.3f}")
    print(f"  ROI Gap Max:             {wf_result.roi_gap_max:.1%}")
    print()
    print(f"  ROI Gap Verdict:         {wf_result.roi_gap_verdict}")
    print(f"  Consistency Verdict:     {wf_result.consistency_verdict}")
    print(f"  Stability Verdict:       {wf_result.stability_verdict}")
    print(f"  *** Overall Verdict:     {wf_result.overall_verdict} ***")
    print()
    print(f"  Result saved: {output_path}")
    print("  MLflow experiment: wf_validation")


if __name__ == "__main__":
    main()
