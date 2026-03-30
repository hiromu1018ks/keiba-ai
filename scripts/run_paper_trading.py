"""Paper Trading メインスクリプト

使い方:
  python scripts/run_paper_trading.py --mode setup --date 2026-04-05
  python scripts/run_paper_trading.py --mode watch --date 2026-04-05
  python scripts/run_paper_trading.py --mode reconcile --date 2026-04-05
  python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13
  python scripts/run_paper_trading.py --mode dry-run --start 2024-07-01 --end 2024-07-31
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.repository import DataRepository
    from domain.models import TrainedModelsV5
    from paper_trading.config import PaperTradingConfig

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Trading")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["setup", "watch", "reconcile", "dry-run"],
        help="実行モード",
    )
    parser.add_argument("--date", help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--start", help="期間開始 (YYYYMMDD, dry-run用)")
    parser.add_argument("--end", help="期間終了 (YYYYMMDD, dry-run用)")
    parser.add_argument("--run-id", help="MLflow run ID (省略時は最新)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> "PaperTradingConfig":
    from paper_trading.config import PaperTradingConfig

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set, notifications disabled")

    db_password = os.environ.get("PGPASSWORD", "")
    conn_str = f"postgresql://postgres:{db_password}@localhost:5432/everydb2"

    config = PaperTradingConfig(
        slack_webhook_url=webhook_url,
        everydb2_connection_string=conn_str,
        mlflow_run_id=args.run_id,
    )
    config.ensure_dirs()
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args)

    # --- モデルロード ---
    from db.model_loader import ModelLoader

    t0 = time.time()
    loader = ModelLoader(tracking_uri=config.mlflow_tracking_uri)
    models, model_info = loader.load(run_id=config.mlflow_run_id)
    logger.info(
        "Model loaded: %s (train: %s ~ %s) in %.1fs",
        model_info.mlflow_run_id,
        model_info.train_start,
        model_info.train_end,
        time.time() - t0,
    )

    # model_info.json を保存
    info_path = config.paper_trading_dir / "model" / "model_info.json"
    info_path.write_text(
        json.dumps(
            {
                "mlflow_run_id": model_info.mlflow_run_id,
                "train_start": model_info.train_start,
                "train_end": model_info.train_end,
                "loaded_at": model_info.loaded_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- リポジトリ ---
    from db.parquet_store import ParquetStore
    from db.repository import DataRepository

    repo = DataRepository(ParquetStore())

    if args.mode == "setup":
        _run_setup(args, config, models, repo)

    elif args.mode == "watch":
        _run_watch(args, config, models, repo)

    elif args.mode == "reconcile":
        _run_reconcile(args, config, repo, models)

    elif args.mode == "dry-run":
        _run_dry_run(args, config, models, repo)


def _run_setup(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    repo: "DataRepository",
) -> None:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from paper_trading.predictor import PaperPredictor

    target_date = date.fromisoformat(args.date)
    race_predictor = RacePredictor(models)
    predictor = PaperPredictor(
        repo=repo,
        race_predictor=race_predictor,
        models=models,
        output_dir=config.paper_trading_dir,
    )
    everydb2 = EveryDB2Queries(connection_string=config.everydb2_connection_string)

    t0 = time.time()
    schedule = predictor.setup(target_date, everydb2)
    logger.info("Setup complete: %d races (%.1fs)", len(schedule), time.time() - t0)

    # schedule.json を保存
    schedule_path = config.paper_trading_dir / "schedule.json"
    schedule_path.write_text(
        json.dumps({"date": args.date, "races": schedule}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Schedule saved: %s", schedule_path)

    # Slack 通知
    if config.slack_webhook_url:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url=config.slack_webhook_url)
        notifier.send(f"Setup complete: {len(schedule)} races scheduled for {args.date}")


def _run_watch(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    repo: "DataRepository",
) -> None:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from monitoring.notifier import CompositeNotifier, LoggingNotifier, SlackNotifier
    from paper_trading.predictor import PaperPredictor
    from paper_trading.watcher import RaceWatcher

    target_date = date.fromisoformat(args.date)
    race_predictor = RacePredictor(models)
    predictor = PaperPredictor(
        repo=repo,
        race_predictor=race_predictor,
        models=models,
        output_dir=config.paper_trading_dir,
    )
    everydb2 = EveryDB2Queries(connection_string=config.everydb2_connection_string)

    # 通知設定
    notifiers: list["LoggingNotifier | SlackNotifier"] = [LoggingNotifier()]
    if config.slack_webhook_url:
        notifiers.append(SlackNotifier(webhook_url=config.slack_webhook_url))
    notifier = CompositeNotifier(notifiers)

    # スケジュール読み込み
    schedule_path = config.paper_trading_dir / "schedule.json"
    if not schedule_path.exists():
        logger.error("schedule.json not found. Run --mode setup first.")
        sys.exit(1)
    schedule_data = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = schedule_data["races"]

    watcher = RaceWatcher(
        predictor=predictor,
        everydb2=everydb2,
        notifier=notifier,
        predictions_dir=config.paper_trading_dir / "predictions",
        retry_count=config.retry_count,
        retry_interval_seconds=config.retry_interval_seconds,
        watch_lead_minutes=config.watch_lead_minutes,
    )

    logger.info("Watch mode started for %s (%d races)", args.date, len(schedule))
    bets = watcher.watch(target_date, schedule, bankroll=config.initial_bankroll)
    logger.info("Watch complete: %d bets placed", len(bets))


def _run_reconcile(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    repo: "DataRepository",
    models: "TrainedModelsV5 | None" = None,
) -> None:
    from db.everydb2_queries import EveryDB2Queries
    from paper_trading.reconciler import PaperReconciler
    from paper_trading.report import PaperTradingReport

    target_date = date.fromisoformat(args.date)
    everydb2 = EveryDB2Queries(connection_string=config.everydb2_connection_string)

    reconciler = PaperReconciler(
        repo=repo,
        bets_path=config.paper_trading_dir / "bets.parquet",
        everydb2=everydb2,
    )

    t0 = time.time()
    result = reconciler.reconcile(target_date)
    logger.info("Reconcile complete (%.1fs): %s", time.time() - t0, result)

    # 日次サマリー保存
    summary_dir = config.paper_trading_dir / "daily_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{target_date.strftime('%Y%m%d')}.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # HTML レポート更新
    if config.paper_trading_dir.joinpath("bets.parquet").exists():
        import pandas as pd

        bets_df = pd.read_parquet(config.paper_trading_dir / "bets.parquet")
        report = PaperTradingReport(output_dir=config.paper_trading_dir)
        report.generate(bets_df.to_dict("records"), result)
        logger.info("Report updated")

    # Slack 通知
    if config.slack_webhook_url:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url=config.slack_webhook_url)
        notifier.send_daily_result(result)


def _run_dry_run(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    repo: "DataRepository",
) -> None:
    """過去データで本番パイプラインの動作確認"""
    from backtest.race_predictor import RacePredictor
    from features.feature_engine import FeatureEngine
    from features.horse_history_features import HorseHistoryFeatures
    from features.jockey_context_features import JockeyContextFeatures
    from features.trainer_context_features import TrainerContextFeatures
    from models.submodel_manager import SubModelManager

    # 期間決定
    if args.date:
        dates: list[date] = [date.fromisoformat(args.date)]
    elif args.start and args.end:
        start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]))
        end = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:8]))
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
    else:
        logger.error("--date or --start/--end required for dry-run")
        sys.exit(1)

    race_predictor = RacePredictor(models)

    # 特徴量を一括生成
    all_start = dates[0].strftime("%Y%m%d")
    all_end = dates[-1].strftime("%Y%m%d")

    logger.info("Loading data: %s ~ %s", all_start, all_end)
    race_df = repo.load_races(all_start, all_end)
    entry_df = repo.load_entries(all_start, all_end)
    odds_df = repo.load_odds_snapshots(all_start, all_end)

    if race_df.empty:
        logger.error("No race data found")
        sys.exit(1)

    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, repo=repo)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(repo=repo).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(repo).compute(entry_df)
    trainer_all = TrainerContextFeatures(repo).compute(entry_df)

    # 日次シミュレーション
    total_bets = 0
    total_stake = 0.0
    total_return = 0.0
    dry_run_dir = config.paper_trading_dir / "dry_run"
    dry_run_dir.mkdir(parents=True, exist_ok=True)

    bankroll = config.initial_bankroll

    for target_date in dates:
        ymd = target_date.strftime("%Y%m%d")
        day_races = [rid for rid in race_ids if rid[:8] == ymd]

        if not day_races:
            continue

        day_bets: list[dict[str, object]] = []
        for race_id in day_races:
            race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
            hist_race = hist_all[hist_all["race_id"] == race_id]
            jockey_race = jockey_all[jockey_all["race_id"] == race_id]
            trainer_race = trainer_all[trainer_all["race_id"] == race_id]

            result_df = race_predictor.predict(race_df_single, hist_race, jockey_race, trainer_race)
            if result_df.empty:
                continue

            if not race_predictor.should_bet(result_df):
                continue

            bets = race_predictor.select_bets(result_df, bankroll)
            for bet in bets:
                horse = result_df[result_df["umaban"] == bet.umaban]
                if not horse.empty:
                    finish_pos = int(horse.iloc[0]["finish_pos"])
                    payout = 0.0
                    if bet.bet_type.value == "place" and 1 <= finish_pos <= 3:
                        payout = bet.stake * bet.odds
                    bankroll -= bet.stake
                    if payout > 0:
                        bankroll += payout
                    total_stake += bet.stake
                    total_return += payout
                    total_bets += 1
                    day_bets.append(
                        {
                            "race_id": race_id,
                            "umaban": bet.umaban,
                            "odds": bet.odds,
                            "ev": bet.ev_lower_corrected,
                            "stake": bet.stake,
                            "payout": payout,
                            "bankroll": bankroll,
                        }
                    )

        # 日次結果保存
        day_result = {
            "date": ymd,
            "n_bets": len(day_bets),
            "bankroll": bankroll,
        }
        (dry_run_dir / f"{ymd}.json").write_text(
            json.dumps(day_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    roi = total_return / total_stake if total_stake > 0 else 0.0
    print(f"\nDry-run Results ({all_start} ~ {all_end}):")
    print(f"  Bets:    {total_bets}")
    print(f"  Stake:   \u00a5{total_stake:,.0f}")
    print(f"  Return:  \u00a5{total_return:,.0f}")
    print(f"  ROI:     {roi:.1%}")
    print(f"  Bankroll:\u00a5{bankroll:,.0f}")


if __name__ == "__main__":
    main()
