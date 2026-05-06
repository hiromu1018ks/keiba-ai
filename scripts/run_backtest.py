"""バックテスト計測スクリプト

使い方:
  # モード1: 単一年度 (従来互換)
  python scripts/run_backtest.py \\
    --train-start 20200101 --train-end 20231231 \\
    --test-start 20240101 --test-end 20241231

  # モード2: マルチ年度
  python scripts/run_backtest.py \\
    --years 2023 2024 2025 \\
    --train-window 4

  # 共通オプション
    --betting-mode flat|kelly   (デフォルト: flat)
    --ensemble                  (アンサンブル有効化)
    --report                    (HTMLレポート + JSON + parquet 生成)
    --strategy-manifest PATH    (Optuna最適化済み戦略パラメータJSON)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backtest.engine import BacktestResult

import pandas as pd

warnings.filterwarnings("ignore")

# Windows cp932 環境で ¥ が表示できない問題を回避
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Phase 7 バックテスト (2024年テスト、複勝モード、フラットベット) のベースラインROI
BASELINE_ROI = 0.638


def to_dash_date(yyyymmdd: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def build_parser() -> argparse.ArgumentParser:
    """引数パーサーを構築"""
    parser = argparse.ArgumentParser(description="バックテスト")
    parser.add_argument("--train-start", required=False, help="学習開始日 (YYYYMMDD)")
    parser.add_argument("--train-end", required=False, help="学習終了日 (YYYYMMDD)")
    parser.add_argument("--test-start", required=False, help="テスト開始日 (YYYYMMDD)")
    parser.add_argument("--test-end", required=False, help="テスト終了日 (YYYYMMDD)")
    parser.add_argument("--years", nargs="+", type=int, help="マルチ年度指定 (テスト年度)")
    parser.add_argument(
        "--train-window",
        type=int,
        default=4,
        help="マルチ年度の学習年数 (デフォルト: 4)",
    )
    parser.add_argument("--report", action="store_true", help="HTMLレポート + parquet を生成")
    parser.add_argument(
        "--betting-mode",
        choices=["flat", "kelly"],
        default="flat",
        help="ベット額計算モード (flat=100円固定, kelly=Fractional Kelly)",
    )
    parser.add_argument("--ensemble", action="store_true", help="アンサンブル (B1) を有効化")
    parser.add_argument(
        "--betting-target",
        choices=["win", "place", "wide"],
        default="win",
        help="ベッティング対象 (win=単勝, place=複勝, wide=ワイド, デフォルト: win)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="学習をスキップし、キャッシュ済みモデルをロードしてテストのみ実行",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable pyinstrument profiling (outputs to data/profiles/)",
    )
    parser.add_argument(
        "--strategy-manifest",
        type=str,
        default=None,
        help="Optuna最適化済み戦略パラメータ manifest JSON (parameter_freeze_protocol形式)",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """引数の排他バリデーション"""
    if args.skip_train and not args.ensemble:
        parser.error("--skip-train requires --ensemble (キャッシュモデルはensemble前提)")
    if args.years:
        return  # マルチ年度モード — OK
    single_year_args = [args.train_start, args.train_end, args.test_start, args.test_end]
    if all(single_year_args):
        return  # 単一年度モード — OK
    parser.error(
        "単一年度モードには --train-start, --train-end, --test-start, --test-end が必要です"
    )


def _load_cached_models(model_dir: Path) -> Any:
    """キャッシュ済みモデルをディレクトリからロードする。

    Args:
        model_dir: meta.json が含まれるディレクトリ (例: data/models-backtest/ または
                   data/models-backtest/2025/)

    Returns:
        TrainedModelsV5
    """
    from db.model_loader import ModelLoader

    loader = ModelLoader()
    models, info = loader.load_from_dir(model_dir, use_ensemble_override=True)
    logger.info(
        "キャッシュモデルをロード: %s (学習期間: %s ~ %s)",
        model_dir,
        info.train_start,
        info.train_end,
    )
    return models


def _get_model_dir(base_dir: Path, test_year: int | None = None) -> Path:
    """年度別モデルディレクトリを解決。

    優先順位:
    1. {base_dir}/{year}/ (年度別サブディレクトリ)
    2. {base_dir}/ (フラット構成 — 従来互換)
    """
    if test_year is not None:
        year_dir = base_dir / str(test_year)
        if (year_dir / "meta.json").is_file():
            return year_dir
    if (base_dir / "meta.json").is_file():
        return base_dir
    raise FileNotFoundError(
        f"キャッシュモデルが見つかりません: {base_dir} "
        f"(test_year={test_year} の meta.json が存在しません。"
        f"先に --skip-train なしで実行してください。)"
    )


def _load_strategy_params(manifest_path: str | None) -> dict[str, Any] | None:
    """--strategy-manifest から戦略パラメータをロード。

    Args:
        manifest_path: manifest JSON ファイルパス (None の場合は None を返す)

    Returns:
        strategy_params dict、または None
    """
    if manifest_path is None:
        return None

    path = Path(manifest_path)
    if not path.exists():
        logger.warning("Strategy manifest が見つかりません: %s — デフォルトパラメータを使用", path)
        return None

    try:
        from backtest.parameter_freeze_protocol import verify_strategy_manifest

        params = verify_strategy_manifest(path)
        logger.info("Strategy manifest ロード完了: %s (SHA256検証OK)", path)
        return params
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Strategy manifest 検証失敗: %s — デフォルトパラメータを使用", e)
        return None


def _build_strategy_config_from_manifest(
    params: dict[str, Any],
) -> dict[str, Any]:
    """manifest params (Optuna best_params 形式) を BacktestEngine strategy_config に変換。

    betting.default_strategy.build_strategy_config_from_params に委譲。
    """
    from betting.default_strategy import build_strategy_config_from_params
    return build_strategy_config_from_params(params)


def _collect_training_bet_history(
    models: Any,
    store: Any,
    train_start: str,
    train_end: str,
    betting_mode: str,
    betting_target: str,
    strategy_params: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """トレーニング期間のバックテストを実行し、OddsBandFilter キャリブレーション用の
    bet_history を収集する。

    常にデフォルトパラメータ (build_default_strategy_config) を使用する。
    strategy_params引数は呼び出し側インターフェース互換のために保持するが、
    関数内部では使用しない (ルックアヘッド防止)。

    Args:
        models: 学習済みモデル
        store: ParquetStore
        train_start: 学習開始日 (YYYY-MM-DD)
        train_end: 学習終了日 (YYYY-MM-DD)
        betting_mode: ベッティングモード
        betting_target: ベッティング対象
        strategy_params: 戦略パラメータ (使用しない — デフォルトパラメータ優先)

    Returns:
        bet_history list
    """
    from backtest.engine import BacktestEngine
    from betting.default_strategy import build_default_strategy_config

    # D-07: デフォルトパラメータでtraining_bet_historyを生成 (ルックアヘッド防止)
    default_train_config = build_default_strategy_config()

    logger.info("トレーニング期間バックテスト (OddsBandFilter キャリブレーション用): %s ~ %s",
                train_start, train_end)
    train_engine = BacktestEngine(
        models=models,
        store=store,
        betting_mode=betting_mode,
        diag_prefix="bt_train",
        betting_target=betting_target,
        strategy_params=default_train_config,
    )
    train_result = train_engine.run(train_start, train_end)
    logger.info(
        "トレーニング期間バックテスト完了: %d bets, ROI=%.1f%%",
        train_result.total_bets,
        train_result.total_roi * 100,
    )
    return train_result.bet_history


def save_year_parquet(year: int, result: BacktestResult) -> None:
    """年度別 parquet 出力: horse_diagnostics + bet_history を結合して保存

    注意: HorseDiagnostic に含まれないフィールド (race_date, bamei, surface, kyori,
    grade_code 等) は bet_history 側にのみ存在する。ベット対象外の馬 (is_bet=False) は
    これらのフィールドが NaN になる。
    """
    pred_dir = Path(ROOT) / "data" / "backtest" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 年度別プレフィックスで診断 CSV を読み込む
    diag_path = Path(ROOT) / "data" / "backtest" / f"bt_{year}_horse_diagnostics.csv"
    if not diag_path.exists():
        logger.warning("診断CSVが見つかりません: %s", diag_path)
        return

    diag_df = pd.read_csv(diag_path)

    if not result.bet_history:
        merged = diag_df
    else:
        bet_df = pd.DataFrame(result.bet_history)
        # bet_history 側の付加情報を horse_diagnostics に left-join
        # bet_cols: ベット対象のみに存在するフィールド (非ベット馬は NaN)
        bet_only_cols = [
            "bet_type",
            "stake",
            "odds",
            "final_odds",
            "result",
            "ev",
            "popularity",
            "bankroll_after",
            "race_date",
            "surface",
            "kyori",
            "grade_code",
            "race_name",
            "bamei",
            "kisyu",
            "kakuteijyuni",
            "track_condition_code",
        ]
        # 存在する列のみ選択
        available_cols = ["race_id", "umaban"] + [c for c in bet_only_cols if c in bet_df.columns]
        bet_subset = bet_df[available_cols].copy()
        # merge key の型不一致を解消 (CSV は string, bet_history は int64)
        bet_subset["race_id"] = bet_subset["race_id"].astype(str)
        bet_subset["umaban"] = bet_subset["umaban"].astype(str)
        diag_df_merge = diag_df.copy()
        diag_df_merge["race_id"] = diag_df_merge["race_id"].astype(str)
        diag_df_merge["umaban"] = diag_df_merge["umaban"].astype(str)
        merged = diag_df_merge.merge(
            bet_subset, on=["race_id", "umaban"], how="left", suffixes=("", "_bet")
        )

    out_path = pred_dir / f"{year}.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info("Parquet保存: %s (%d rows)", out_path, len(merged))


def display_single_year_result(
    result: BacktestResult,
    elapsed_train: float,
    elapsed_test: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    betting_target: str = "win",
) -> dict[str, Any]:
    """単一年度の結果を表示し、JSON用dictを返す"""
    print()
    print("=" * 50)
    print("  結果")
    print("=" * 50)
    print(f"  レース数:       {result.total_bets:>8,}")
    print(f"  投資額:         {result.total_stake:>10,.0f} 円")
    print(f"  払戻額:         {result.total_return:>10,.0f} 円")
    print(f"  利益:           {result.profit:>10,.0f} 円")
    print(f"  ROI:            {result.total_roi:>9.1%}")
    print(f"  最大DD:         {result.max_drawdown:>9.1%}")
    print(f"  最終資金:       {result.final_bankroll:>10,.0f} 円")
    print(f"  学習時間:       {elapsed_train:>7,.0f} 秒")
    print(f"  テスト時間:     {elapsed_test:>7,.0f} 秒")

    if betting_target == "win":
        print()
        print("=" * 50)
        print("  単勝ベット詳細")
        print("=" * 50)
        wr = result.winning_bets / result.total_bets if result.total_bets > 0 else 0
        print(f"  的中率:         {wr:>9.1%}")
        print(f"  的中数:         {result.winning_bets:>8,} / {result.total_bets:,}")
        avg_odds = (
            sum(b.get("final_odds", b.get("odds", 0)) for b in result.bet_history)
            / len(result.bet_history)
            if result.bet_history
            else 0.0
        )
        print(f"  平均オッズ:     {avg_odds:>9.1f}")
        if result.avg_edge > 0:
            print(f"  平均Edge:       {result.avg_edge:>9.3f}")
            print(f"  Edge範囲:       {result.min_edge:.3f} ~ {result.max_edge:.3f}")

    before_roi = BASELINE_ROI
    diff = result.total_roi - before_roi
    status = "目標達成!" if result.total_roi >= 1.01 else "未達"
    print()
    print("=" * 50)
    print("  Before vs After")
    print("=" * 50)
    print(f"  改善前 ROI:     {before_roi:.1%}")
    print(f"  改善後 ROI:     {result.total_roi:.1%}")
    print(f"  差分:           {diff:+.1%}")
    print(f"  判定:           {status}")

    return {
        "before_roi": before_roi,
        "total_roi": result.total_roi,
        "total_bets": result.total_bets,
        "total_stake": result.total_stake,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "final_bankroll": result.final_bankroll,
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "train_seconds": round(elapsed_train),
        "test_seconds": round(elapsed_test),
    }


def _run_single_year(args: argparse.Namespace) -> None:
    """単一年度バックテスト"""
    train_start = to_dash_date(args.train_start)
    train_end = to_dash_date(args.train_end)
    test_start = to_dash_date(args.test_start)
    test_end = to_dash_date(args.test_end)

    from db.parquet_store import ParquetStore
    from pipelines.training_pipeline import TrainingPipelineV5

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # Strategy manifest ロード
    strategy_params = _load_strategy_params(args.strategy_manifest)
    if strategy_params is not None:
        strategy_params = _build_strategy_config_from_manifest(strategy_params)

    # 学習 または キャッシュロード
    t0 = time.time()
    model_dir = Path("data/models-backtest")

    if args.skip_train:
        logger.info("学習スキップ (--skip-train)")
        try:
            models = _load_cached_models(model_dir)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)
        elapsed_train = 0.0
    else:
        logger.info("=" * 50)
        logger.info("  学習期間: %s ~ %s", train_start, train_end)
        logger.info("=" * 50)

        pipeline = TrainingPipelineV5(store=store, model_dir=model_dir)
        try:
            models = pipeline.run(train_start, train_end, use_ensemble=args.ensemble)
        except KeyboardInterrupt:
            logger.warning("学習が中断されました")
            sys.exit(1)
        except Exception as e:
            logger.error("学習失敗: %s", e)
            sys.exit(1)

        elapsed_train = time.time() - t0
        logger.info("学習完了 (%.0f秒)", elapsed_train)

    # トレーニング期間 bet_history 収集 (OddsBandFilter キャリブレーション用)
    training_bet_history = _collect_training_bet_history(
        models=models,
        store=store,
        train_start=train_start,
        train_end=train_end,
        betting_mode=args.betting_mode,
        betting_target=args.betting_target,
        strategy_params=strategy_params,
    )

    # バックテスト
    logger.info("=" * 50)
    logger.info("  テスト期間: %s ~ %s", test_start, test_end)
    logger.info("=" * 50)
    t1 = time.time()

    from backtest.engine import BacktestEngine

    test_year = int(test_start[:4])
    engine = BacktestEngine(
        models=models,
        store=store,
        betting_mode=args.betting_mode,
        diag_prefix=f"bt_{test_year}",
        betting_target=args.betting_target,
        strategy_params=strategy_params,
    )
    result = engine.run(test_start, test_end, training_bet_history=training_bet_history)
    elapsed_test = time.time() - t1
    logger.info("バックテスト完了 (%.0f秒)", elapsed_test)

    # 結果表示
    out = display_single_year_result(
        result,
        elapsed_train,
        elapsed_test,
        train_start,
        train_end,
        test_start,
        test_end,
        betting_target=args.betting_target,
    )

    # 出力
    if args.report:
        from backtest.report import BacktestReportGenerator

        output_dir = Path(ROOT) / "data" / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)

        gen = BacktestReportGenerator(output_dir=output_dir)
        bet_history_path = gen.save_bet_history(result.bet_history)
        print(f"\nbet_history保存: {bet_history_path}")

        if args.betting_target == "win":
            diag_path = gen.save_ai_diagnostics(
                gen._derive_fields(result.bet_history),
                result,
                betting_target=args.betting_target,
            )
            if diag_path:
                print(f"AI診断JSON: {diag_path}")

        result_path = output_dir / "backtest_result.json"
        result_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"結果保存: {result_path}")

        report_path = gen.generate(
            result,
            result.bet_history,
            train_period=f"{train_start} ~ {train_end}",
            test_period=f"{test_start} ~ {test_end}",
            betting_target=args.betting_target,
        )
        print(f"レポート生成: {report_path}")

        save_year_parquet(test_year, result)
    else:
        outpath = os.path.join(ROOT, "backtest_result.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n結果保存: {outpath}")


def _run_multi_year(args: argparse.Namespace) -> None:
    """マルチ年度バックテスト"""
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    logger.info("ParquetStore OK")

    # Strategy manifest ロード
    strategy_params = _load_strategy_params(args.strategy_manifest)
    if strategy_params is not None:
        strategy_params = _build_strategy_config_from_manifest(strategy_params)

    all_results: dict[int, Any] = {}
    all_metadata: dict[int, dict[str, str]] = {}
    base_model_dir = Path("data/models-backtest")

    for test_year in args.years:
        train_start = f"{test_year - args.train_window}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print()
        print("=" * 50)
        print(f"  {test_year}年 (学習: {train_start[:4]}-{train_end[:4]})")
        print("=" * 50)

        # 学習 または キャッシュロード
        t0 = time.time()

        if args.skip_train:
            logger.info("%d年: 学習スキップ (--skip-train)", test_year)
            try:
                models = _load_cached_models(
                    _get_model_dir(base_model_dir, test_year)
                )
            except FileNotFoundError as e:
                logger.error("%s — スキップ", e)
                continue
            elapsed_train = 0.0
        else:
            # 年度別サブディレクトリに保存 (マルチ年度で各年度のモデルを保持)
            year_model_dir = base_model_dir / str(test_year)
            year_model_dir.mkdir(parents=True, exist_ok=True)
            try:
                from pipelines.training_pipeline import TrainingPipelineV5

                pipeline = TrainingPipelineV5(store=store, model_dir=year_model_dir)
                models = pipeline.run(train_start, train_end, use_ensemble=args.ensemble)
            except KeyboardInterrupt:
                logger.warning("中断されました")
                sys.exit(1)
            except Exception as e:
                logger.error("%d年 学習失敗: %s — スキップ", test_year, e)
                continue
            elapsed_train = time.time() - t0

        # トレーニング期間 bet_history 収集 (OddsBandFilter キャリブレーション用)
        training_bet_history = _collect_training_bet_history(
            models=models,
            store=store,
            train_start=train_start,
            train_end=train_end,
            betting_mode=args.betting_mode,
            betting_target=args.betting_target,
            strategy_params=strategy_params,
        )

        # バックテスト
        t1 = time.time()
        try:
            from backtest.engine import BacktestEngine

            engine = BacktestEngine(
                models=models,
                store=store,
                betting_mode=args.betting_mode,
                diag_prefix=f"bt_{test_year}",
                betting_target=args.betting_target,
                strategy_params=strategy_params,
            )
            result = engine.run(test_start, test_end, training_bet_history=training_bet_history)
        except Exception as e:
            logger.error("%d年 テスト失敗: %s — スキップ", test_year, e)
            continue
        elapsed_test = time.time() - t1

        all_results[test_year] = result
        all_metadata[test_year] = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_seconds": str(round(elapsed_train)),
            "test_seconds": str(round(elapsed_test)),
        }

        # マルチ年度では常に parquet 出力
        save_year_parquet(test_year, result)

        profit = result.profit
        print(f"  学習完了 ({elapsed_train:.0f}秒)")
        print(f"  テスト完了 ({elapsed_test:.0f}秒)")
        print(
            f"  ベット数: {result.total_bets:>8,} | "
            f"投資額: ¥{result.total_stake:>10,.0f} | "
            f"払戻: ¥{result.total_return:>10,.0f}"
        )
        print(
            f"  ROI: {result.total_roi:>8.1%} | "
            f"利益: ¥{profit:>+10,.0f} | "
            f"最大DD: {result.max_drawdown:>6.1%}"
        )

    if not all_results:
        logger.error("全年度失敗。レポートは生成しません。")
        sys.exit(1)

    # 全体サマリー
    print()
    print("=" * 50)
    print("  全体サマリー")
    print("=" * 50)
    total_bets = sum(r.total_bets for r in all_results.values())
    total_stake = sum(r.total_stake for r in all_results.values())
    total_return = sum(r.total_return for r in all_results.values())
    total_profit = total_return - total_stake
    total_roi = total_return / total_stake if total_stake > 0 else 0.0
    best_year = max(all_results, key=lambda y: all_results[y].total_roi)
    worst_year = min(all_results, key=lambda y: all_results[y].total_roi)

    print(f"  総ベット数:  {total_bets:>10,}")
    print(f"  総投資額:  ¥{total_stake:>12,.0f}")
    print(f"  総払戻額:  ¥{total_return:>12,.0f}")
    print(f"  総利益:    ¥{total_profit:>+12,.0f}")
    print(f"  合計 ROI:   {total_roi:>10.1%}")
    print(f"  最良年度:  {best_year} ({all_results[best_year].total_roi:.1%})")
    print(f"  最悪年度:  {worst_year} ({all_results[worst_year].total_roi:.1%})")

    # --report 時の出力
    if args.report:
        output_dir = Path(ROOT) / "data" / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)

        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=output_dir)
        report_path = gen.generate(all_results, all_metadata, betting_target=args.betting_target)
        print(f"\n  レポート生成: {report_path}")

        json_data: dict[str, Any] = {
            "overall": {
                "total_bets": total_bets,
                "total_stake": total_stake,
                "total_return": total_return,
                "profit": total_profit,
                "roi": total_roi,
                "best_year": best_year,
                "worst_year": worst_year,
            },
            "years": {},
        }
        for year, r in all_results.items():
            json_data["years"][str(year)] = {
                "total_bets": r.total_bets,
                "total_stake": r.total_stake,
                "total_return": r.total_return,
                "roi": r.total_roi,
                "profit": r.profit,
                "max_drawdown": r.max_drawdown,
                "metadata": all_metadata[year],
            }
        json_path = output_dir / "multi_year_result.json"
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON保存: {json_path}")

        all_bets: list[dict[str, Any]] = []
        for year, r in all_results.items():
            for bet in r.bet_history:
                all_bets.append({**bet, "_test_year": year})
        bets_path = output_dir / "multi_year_bet_history.json"
        bets_path.write_text(json.dumps(all_bets, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  bet_history保存: {bets_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    from utils.profiling import ProfileContext

    with ProfileContext(enabled=args.profile, label="backtest"):
        if args.years:
            _run_multi_year(args)
        else:
            _run_single_year(args)


if __name__ == "__main__":
    main()
