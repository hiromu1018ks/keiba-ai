"""Paper Trading メインスクリプト (Parquet-based)

使い方:
  # 準備: ETL deltaで最新データをParquetに取り込む
  python scripts/run_etl.py --mode delta

  # setup: 当日のレース一覧を確認
  python scripts/run_paper_trading.py --mode setup --date 2026-04-05

  # predict: 特徴量生成→推論→ベット保存→Slack通知
  python scripts/run_paper_trading.py --mode predict --date 2026-04-05

  # reconcile: レース結果と照合→ROI計算→レポート更新
  python scripts/run_paper_trading.py --mode reconcile --date 2026-04-05

  # dry-run: 過去データでパイプライン動作確認
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
    from db.parquet_store import ParquetStore
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
        choices=["setup", "predict", "reconcile", "dry-run"],
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


def _load_models(config: "PaperTradingConfig") -> tuple["TrainedModelsV5", object]:
    """MLflowから学習済みモデルをロード"""
    from db.model_loader import ModelInfo, ModelLoader

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
    return models, model_info


def _send_slack(config: "PaperTradingConfig", message: str) -> None:
    """Slackに通知 (エラーを無視)"""
    if not config.slack_webhook_url:
        return
    try:
        from monitoring.notifier import SlackNotifier

        SlackNotifier(webhook_url=config.slack_webhook_url).send(message)
    except Exception as e:
        logger.warning("Slack通知失敗: %s", e)


def _decode_bamei(name: object) -> str:
    """Shift-JIS バイト列の bamei をデコードする。"""
    if not isinstance(name, str):
        return str(name)
    try:
        return name.encode("latin-1").decode("shift_jis")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


# ─────────────────────────────────────────────────
# setup: 当日のレース一覧を確認
# ─────────────────────────────────────────────────


def _run_setup(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    store: "ParquetStore",
) -> None:
    from db.everydb2_queries import EveryDB2Queries
    from db.readers import load_entries_from_db, load_races_from_db

    target_date = date.fromisoformat(args.date)
    ymd = target_date.strftime("%Y%m%d")

    db = EveryDB2Queries(config.everydb2_connection_string)
    race_df = load_races_from_db(db, ymd)
    entry_df = load_entries_from_db(db, ymd)

    if race_df.empty:
        logger.warning("No races found for %s", args.date)
        return

    # レーススケジュールを構築
    schedule = []
    for race_id in race_df["race_id"].unique():
        race = race_df[race_df["race_id"] == race_id].iloc[0]
        entries = entry_df[entry_df["race_id"] == race_id]
        n_with_results = entries["kakuteijyuni"].notna().sum()
        schedule.append(
            {
                "race_id": race_id,
                "surface": race.get("surface", ""),
                "distance": int(race.get("kyori", 0)),
                "post_time": str(race.get("hassotime", "")),
                "n_horses": len(entries),
                "n_with_results": int(n_with_results),
                "tenkocd": race.get("tenkocd", ""),
                "track_condition_code": race.get("track_condition_code", ""),
            }
        )

    # 保存
    schedule_path = config.paper_trading_dir / "schedule.json"
    schedule_path.write_text(
        json.dumps({"date": args.date, "races": schedule}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Setup: %d races found for %s", len(schedule), args.date)
    for s in schedule:
        logger.info(
            "  %s %s %dm %s頭 (発走%s 結果%d件)",
            s["race_id"],
            s["surface"],
            s["distance"],
            s["n_horses"],
            s["post_time"],
            s["n_with_results"],
        )

    _send_slack(config, f"Setup: {len(schedule)} races scheduled for {args.date}")


# ─────────────────────────────────────────────────
# predict: 特徴量生成→推論→ベット保存→Slack通知
# ─────────────────────────────────────────────────


def _run_predict(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    store: "ParquetStore",
) -> None:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from db.readers import (
        load_entries_from_db,
        load_odds_snapshots_from_db,
        load_odds_time_series_from_db,
        load_races_from_db,
    )
    from features.bloodline_features import BloodlineFeatures
    from features.feature_engine import FeatureEngine
    from features.horse_history_features import HorseHistoryFeatures
    from features.jockey_context_features import JockeyContextFeatures
    from features.trainer_context_features import TrainerContextFeatures
    from models.submodel_manager import SubModelManager

    target_date = date.fromisoformat(args.date)
    ymd = target_date.strftime("%Y%m%d")

    # EveryDB2からデータ読み込み
    logger.info("Loading data for %s...", ymd)
    db = EveryDB2Queries(config.everydb2_connection_string)
    race_df = load_races_from_db(db, ymd)
    entry_df = load_entries_from_db(db, ymd)
    odds_df = load_odds_snapshots_from_db(db, ymd)
    odds_ts_df = load_odds_time_series_from_db(db, ymd)

    if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
        logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
        return

    # 特徴量生成 (dry-runと同じパイプライン)
    logger.info("Generating features...")
    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(store=store).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(store).compute(entry_df)
    trainer_all = TrainerContextFeatures(store).compute(entry_df)
    blood_all = BloodlineFeatures(store=store).compute(entry_df)

    # 血統特徴量を feat_df にマージ
    feat_df = feat_df.merge(
        blood_all[["race_id", "umaban"] + [c for c in blood_all.columns if c.startswith("blood_")]],
        on=["race_id", "umaban"],
        how="left",
    )

    # 推論
    race_predictor = RacePredictor(models)
    bankroll = config.initial_bankroll
    all_bets: list[dict[str, object]] = []

    for race_id in race_ids:
        single_race = feat_df[feat_df["race_id"] == race_id].copy()
        hist_race = hist_all[hist_all["race_id"] == race_id]
        jockey_race = jockey_all[jockey_all["race_id"] == race_id]
        trainer_race = trainer_all[trainer_all["race_id"] == race_id]

        result_df = race_predictor.predict(single_race, hist_race, jockey_race, trainer_race)
        if result_df.empty:
            continue

        if not race_predictor.should_bet(result_df):
            continue

        bets = race_predictor.select_bets(result_df, bankroll)
        for bet in bets:
            horse = result_df[result_df["umaban"] == bet.umaban]
            horse_name = _decode_bamei(horse.iloc[0]["bamei"]) if not horse.empty else ""
            all_bets.append(
                {
                    "race_id": race_id,
                    "bet_type": bet.bet_type.value,
                    "umaban": bet.umaban,
                    "horse_name": horse_name,
                    "stake": bet.stake,
                    "odds": bet.odds,
                    "ev": bet.ev_lower_corrected,
                    "result": 0.0,  # 未確定
                    "surface": result_df.iloc[0].get("surface", ""),
                    "distance": result_df.iloc[0].get("kyori", 0),
                    "bankroll_after": bet.stake,  # reconcileで更新
                    "race_date": ymd,
                    "is_paper": True,
                }
            )
            bankroll -= bet.stake

    if not all_bets:
        logger.info("No bets generated for %s", args.date)
        _send_slack(config, f"Predict: 0 bets for {args.date}")
        return

    # 予測結果を保存
    import pandas as pd

    pred_path = config.paper_trading_dir / "predictions" / f"{ymd}.parquet"
    pred_df = pd.DataFrame(all_bets)
    pred_df.to_parquet(pred_path, index=False)
    logger.info("Predictions saved: %d bets → %s", len(all_bets), pred_path)

    # コンソール出力 (Windows cp932 対応)
    import io

    _venue_map = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }

    def _fmt_race_id(rid: str) -> str:
        jyocd = rid[8:10]
        racenum = rid[14:16]
        venue = _venue_map.get(jyocd, jyocd)
        return f"{venue}{int(racenum):2d}R"

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  Predict: {args.date}  -  {len(all_bets)} bets")
    lines.append("=" * 60)
    for b in all_bets:
        lines.append(
            f"  {_fmt_race_id(b['race_id'])}  "
            f"馬番{int(b['umaban']):2d}  {b['horse_name']:16s}  "
            f"複勝{b['odds']:.1f}  EV={b['ev']:.2f}"
        )
    lines.append("")
    text = "\n".join(lines)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    # Slack通知
    slack_msg = f"Predict: {len(all_bets)} bets for {args.date}\n" + "\n".join(
        f"  {_fmt_race_id(b['race_id'])} 馬番{b['umaban']} {b['horse_name']} "
        f"複勝{b['odds']:.1f} EV={b['ev']:.2f}"
        for b in all_bets
    )
    _send_slack(config, slack_msg)
    logger.info("Predict complete: %d bets", len(all_bets))


# ─────────────────────────────────────────────────
# reconcile: レース結果と照合→ROI計算
# ─────────────────────────────────────────────────


def _run_reconcile(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5 | None" = None,
    store: "ParquetStore | None" = None,
) -> None:
    import pandas as pd

    if store is None:
        from db.parquet_store import ParquetStore

        store = ParquetStore()

    target_date = date.fromisoformat(args.date)
    ymd = target_date.strftime("%Y%m%d")

    # 予測を読み込み
    pred_path = config.paper_trading_dir / "predictions" / f"{ymd}.parquet"
    if not pred_path.exists():
        logger.error("Predictions not found: %s", pred_path)
        return
    pred_df = pd.read_parquet(pred_path)

    # 未確定のベットのみ処理
    unsettled = pred_df[pred_df["result"] == 0.0]
    if unsettled.empty:
        logger.info("All bets already settled for %s", args.date)
        return

    # レース結果をParquetから取得
    from db.readers import load_entries, load_payouts

    entry_df = load_entries(store, ymd, ymd)
    payout_df = load_payouts(store, ymd, ymd)

    if entry_df.empty:
        logger.warning("No entry data for %s — races may not have been run yet", args.date)
        return

    n_settled = 0
    n_wins = 0

    for idx, row in unsettled.iterrows():
        race_id = row["race_id"]
        umaban = int(row["umaban"])

        # 着順を取得
        race_entries = entry_df[entry_df["race_id"] == race_id]
        horse_entry = race_entries[race_entries["umaban"] == umaban]

        if horse_entry.empty:
            continue

        finish_pos = horse_entry.iloc[0].get("kakuteijyuni")
        if pd.isna(finish_pos) or finish_pos == 0:
            continue  # レース未確定

        # 複勝的中判定: 1着〜3着
        if 1 <= finish_pos <= 3:
            # 払戻をpayouts.parquetから取得
            payout = 0.0
            if not payout_df.empty:
                race_payouts = payout_df[payout_df["race_id"] == race_id]
                if not race_payouts.empty:
                    po = race_payouts.iloc[0]
                    for i in range(1, 6):
                        maban_col = f"payfukusyoumaban{i}"
                        pay_col = f"payfukusyopay{i}"
                        if maban_col in po.index and pay_col in po.index:
                            if po[maban_col] == umaban:
                                payout = po[pay_col]
                                break

            if payout > 0:
                n_wins += 1
            else:
                # fallback: 予測時オッズを使用
                payout = row["stake"] * row["odds"]

            pred_df.at[idx, "result"] = payout
        else:
            pred_df.at[idx, "result"] = 0.0

        n_settled += 1

    # 確定した予測を書き戻し
    pred_df.to_parquet(pred_path, index=False)

    # bets.parquet に追記
    bets_path = config.paper_trading_dir / "bets.parquet"
    if bets_path.exists():
        existing = pd.read_parquet(bets_path)
        combined = pd.concat([existing, pred_df], ignore_index=True)
    else:
        combined = pred_df
    combined.to_parquet(bets_path, index=False)

    # 累積統計を計算
    total_stake = combined["stake"].sum()
    total_return = combined["result"].sum()
    cumulative_roi = total_return / total_stake if total_stake > 0 else 0.0
    n_total_wins = (combined["result"] > 0).sum()

    result = {
        "date": ymd,
        "n_bets": len(pred_df),
        "n_settled": n_settled,
        "n_new_wins": n_wins,
        "total_stake": total_stake,
        "total_return": total_return,
        "cumulative_roi": cumulative_roi,
        "bankroll": config.initial_bankroll + total_return - total_stake,
        "n_total_bets": len(combined),
        "n_total_wins": int(n_total_wins),
    }

    # 日次サマリー保存
    summary_dir = config.paper_trading_dir / "daily_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{ymd}.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # HTMLレポート更新
    try:
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(output_dir=config.paper_trading_dir)
        report.generate(combined.to_dict("records"), result)
    except Exception as e:
        logger.warning("Report generation failed: %s", e)

    # Slack通知
    slack_msg = (
        f"Reconcile {ymd}: {n_settled} settled, {n_wins} wins\n"
        f"  Daily ROI: {total_return / (n_settled * 100) if n_settled else 0:.1%}\n"
        f"  Cumulative: {len(combined)} bets, ROI={cumulative_roi:.1%}"
    )
    _send_slack(config, slack_msg)

    logger.info(
        "Reconcile: %d settled, %d wins, cumulative ROI=%.1f%% (%d total bets)",
        n_settled,
        n_wins,
        cumulative_roi * 100,
        len(combined),
    )


# ─────────────────────────────────────────────────
# dry-run: 過去データでパイプライン動作確認
# ─────────────────────────────────────────────────


def _run_dry_run(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    store: "ParquetStore",
) -> None:
    import pandas as pd

    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from db.readers import (
        load_entries_from_db,
        load_odds_snapshots_from_db,
        load_odds_time_series_from_db,
        load_races_from_db,
    )
    from features.bloodline_features import BloodlineFeatures
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
    db = EveryDB2Queries(config.everydb2_connection_string)
    race_frames, entry_frames, odds_frames, odds_ts_frames = [], [], [], []
    for d in dates:
        ymd_d = d.strftime("%Y%m%d")
        race_frames.append(load_races_from_db(db, ymd_d))
        entry_frames.append(load_entries_from_db(db, ymd_d))
        odds_frames.append(load_odds_snapshots_from_db(db, ymd_d))
        odds_ts_frames.append(load_odds_time_series_from_db(db, ymd_d))

    race_df = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
    entry_df = pd.concat(entry_frames, ignore_index=True) if entry_frames else pd.DataFrame()
    odds_df = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()
    odds_ts_df = pd.concat(odds_ts_frames, ignore_index=True) if odds_ts_frames else pd.DataFrame()

    if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
        logger.error("EveryDB2 からデータ取得失敗: %s ~ %s", all_start, all_end)
        return

    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(store=store).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(store).compute(entry_df)
    trainer_all = TrainerContextFeatures(store).compute(entry_df)
    blood_all = BloodlineFeatures(store=store).compute(entry_df)

    # 血統特徴量を feat_df にマージ
    feat_df = feat_df.merge(
        blood_all[["race_id", "umaban"] + [c for c in blood_all.columns if c.startswith("blood_")]],
        on=["race_id", "umaban"],
        how="left",
    )

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
                    finish_pos = int(horse.iloc[0]["kakuteijyuni"])
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
    print(f"  Stake:   {total_stake:,.0f} yen")
    print(f"  Return:  {total_return:,.0f} yen")
    print(f"  ROI:     {roi:.1%}")
    print(f"  Bankroll: {bankroll:,.0f} yen")


# ─────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    config = load_config(args)

    # --- モデルロード ---
    models, model_info = _load_models(config)

    # --- ParquetStore ---
    from db.parquet_store import ParquetStore

    store = ParquetStore()

    if args.mode == "setup":
        _run_setup(args, config, models, store)

    elif args.mode == "predict":
        _run_predict(args, config, models, store)

    elif args.mode == "reconcile":
        _run_reconcile(args, config, store=store)

    elif args.mode == "dry-run":
        _run_dry_run(args, config, models, store)


if __name__ == "__main__":
    main()
