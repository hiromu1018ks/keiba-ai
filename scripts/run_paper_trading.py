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

  # diagnose: Parquetデータで診断推論 (EveryDB2 バイパス)
  python scripts/run_paper_trading.py --mode diagnose --start 2024-07-01 --end 2024-07-31
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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
        choices=["setup", "predict", "reconcile", "dry-run", "diagnose"],
        help="実行モード",
    )
    parser.add_argument("--date", help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--start", help="期間開始 (YYYY-MM-DD, diagnose/dry-run用)")
    parser.add_argument("--end", help="期間終了 (YYYY-MM-DD, diagnose/dry-run用)")
    parser.add_argument("--run-id", help="MLflow run ID (省略時は最新)")
    parser.add_argument(
        "--ensemble", action="store_true",
        help="StackedEnsemble (.joblib) モデルをロード",
    )
    parser.add_argument(
        "--minutes-before", type=int, default=5,
        help="発走何分前のオッズを使用するか (デフォルト: 5)",
    )
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


def _load_models(
    config: "PaperTradingConfig", *, use_ensemble: bool = False
) -> tuple["TrainedModelsV5", object]:
    """MLflowから学習済みモデルをロード"""
    from db.model_loader import ModelInfo, ModelLoader

    t0 = time.time()
    loader = ModelLoader(tracking_uri=config.mlflow_tracking_uri)
    models, model_info = loader.load(run_id=config.mlflow_run_id, use_ensemble=use_ensemble)
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
# 発走N分前オッズスナップショット抽出
# ─────────────────────────────────────────────────


def _extract_pre_post_odds(
    odds_ts_df: pd.DataFrame,
    race_df: pd.DataFrame,
    minutes_before: int = 5,
    max_staleness_minutes: int = 60,
    *,
    _now: datetime | None = None,
) -> pd.DataFrame:
    """各レースの発走N分前時点のオッズスナップショットを抽出.

    Parameters
    ----------
    odds_ts_df : DataFrame
        時系列オッズ。happyotime (str "MMDDHHmm"), year, umaban 等を含む。
    race_df : DataFrame
        レース情報。hassotime (int "hhmm"), race_id 等を含む。
    minutes_before : int
        発走何分前のオッズを使うか (デフォルト: 5)。
    max_staleness_minutes : int
        cutoff から何分以上前のスナップショットを除外するか (デフォルト: 60)。
    _now : datetime, optional
        現在時刻のオーバーライド (テスト用)。未指定時は datetime.now()。

    Returns
    -------
    DataFrame
        build_all() と互換のスキーマ:
        race_id, umaban, tanodds, fukuoddslow, tanninki
    """
    from datetime import datetime, timedelta

    import pandas as pd

    if odds_ts_df.empty or race_df.empty:
        return pd.DataFrame(
            columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]
        )

    # 1. race_id -> post_datetime のマッピング
    post_time_map: dict[str, datetime] = {}
    for _, r in race_df.iterrows():
        ht = r.get("hassotime")
        if pd.isna(ht) or str(ht).strip() == "":
            continue
        ht_str = f"{int(ht):04d}"  # 930 -> "0930"
        # race_id の先頭8桁 = YYYYMMDD
        rid = r["race_id"]
        race_date_str = rid[:8]
        post_time_map[rid] = datetime(
            int(race_date_str[:4]),
            int(race_date_str[4:6]),
            int(race_date_str[6:8]),
            int(ht_str[:2]),
            int(ht_str[2:]),
        )

    # 2. odds_ts_df の各行について happyotime -> datetime
    def _parse_happyotime(row: pd.Series) -> datetime | None:
        ht = row.get("happyotime")
        if pd.isna(ht):
            return None
        ht = str(ht).zfill(8)  # "4110930" -> "04110930"
        if len(ht) != 8:
            return None
        year = int(row["year"])
        month = int(ht[:2])
        day = int(ht[2:4])
        hour = int(ht[4:6])
        minute = int(ht[6:8])
        return datetime(year, month, day, hour, minute)

    odds_ts_df = odds_ts_df.copy()
    odds_ts_df["_ht_datetime"] = odds_ts_df.apply(_parse_happyotime, axis=1)
    odds_ts_df = odds_ts_df[odds_ts_df["_ht_datetime"].notna()]

    # 3. 各行に cutoff を付与し、cutoff 以前のエントリのみ残す
    now = _now or datetime.now()

    def _is_before_cutoff(row: pd.Series) -> bool:
        post_time = post_time_map.get(row["race_id"])
        if post_time is None:
            return False
        cutoff = post_time - timedelta(minutes=minutes_before)
        # cutoff時刻に達していないレースはまだオッズが確定していない → 除外
        if cutoff > now:
            return False
        min_cutoff = cutoff - timedelta(minutes=max_staleness_minutes)
        ht_dt = row["_ht_datetime"]
        return min_cutoff <= ht_dt <= cutoff

    mask = odds_ts_df.apply(_is_before_cutoff, axis=1)
    valid = odds_ts_df[mask]

    if valid.empty:
        return pd.DataFrame(
            columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]
        )

    # 4. (race_id, umaban) ごとに最新エントリを取得
    idx = valid.groupby(["race_id", "umaban"])["_ht_datetime"].idxmax()
    snapshot = valid.loc[idx]

    # 5. build_all() と互換のスキーマで返す
    result = snapshot[["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]].copy()
    result = result.reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────
# predict: 特徴量生成→推論→ベット保存→Slack通知
# ─────────────────────────────────────────────────


def _run_predict(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    store: "ParquetStore",
) -> None:
    from backtest.diagnostic_logger import DiagnosticLogger
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
    from features.jockey_trainer_combo import JockeyTrainerComboFeatures
    from features.trainer_context_features import TrainerContextFeatures
    from models.submodel_manager import SubModelManager

    import pandas as pd

    target_date = date.fromisoformat(args.date)
    ymd = target_date.strftime("%Y%m%d")

    # EveryDB2からデータ読み込み
    logger.info("Loading data for %s...", ymd)
    db = EveryDB2Queries(config.everydb2_connection_string)
    race_df = load_races_from_db(db, ymd)
    entry_df = load_entries_from_db(db, ymd)
    odds_snapshot_df = load_odds_snapshots_from_db(db, ymd)  # fallback用
    odds_ts_df = load_odds_time_series_from_db(db, ymd)

    if race_df.empty or entry_df.empty:
        logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
        return

    # 発走時刻マッピングを先に構築 (_extract_pre_post_odds のログと出力で使用)
    _race_time_map: dict[str, str] = {}
    for _, r in race_df.iterrows():
        ht = r.get("hassotime", "")
        if pd.notna(ht) and str(ht).strip():
            ht_str = f"{int(ht):04d}"
            _race_time_map[r["race_id"]] = f"{ht_str[:2]}:{ht_str[2:]}"

    # 発走N分前のオッズスナップショットを生成
    minutes_before = getattr(args, "minutes_before", 5)
    if odds_ts_df.empty:
        logger.warning("No odds time series for %s, falling back to snapshots", ymd)
        odds_df = odds_snapshot_df
    else:
        odds_df = _extract_pre_post_odds(odds_ts_df, race_df, minutes_before=minutes_before)
        if odds_df.empty:
            logger.warning("No pre-post odds extracted for %s, falling back to snapshots", ymd)
            odds_df = odds_snapshot_df

    # オッズスナップショットがないレースの race_id を特定
    if odds_df.empty:
        logger.error("No odds data available for %s (time series and snapshots both empty)", ymd)
        return
    all_race_ids = set(race_df["race_id"].unique())
    covered_race_ids = set(odds_df["race_id"].unique())
    skipped_race_ids = all_race_ids - covered_race_ids
    for rid in sorted(skipped_race_ids):
        post_time = _race_time_map.get(rid, "??")
        logger.info("Skipping %s: no odds snapshot (post_time=%s)", rid, post_time)

    # 特徴量生成 (odds_df を発走N分前スナップショットに差し替え。
    # odds_ts_df はそのまま渡す → odds_dynamics 特徴量は完全時系列から計算)
    logger.info("Generating features...")
    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()

    hist_all = HorseHistoryFeatures(store=store).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(store).compute(entry_df)
    trainer_all = TrainerContextFeatures(store).compute(entry_df)
    jt_all = JockeyTrainerComboFeatures(store).compute(entry_df)
    blood_all = BloodlineFeatures(store=store).compute(entry_df)

    # 血統特徴量を feat_df にマージ
    feat_df = feat_df.merge(
        blood_all[["race_id", "umaban"] + [c for c in blood_all.columns if c.startswith("blood_")]],
        on=["race_id", "umaban"],
        how="left",
    )

    # 既存予測の読み込み (重複回避)
    pred_path = config.paper_trading_dir / "predictions" / f"{ymd}.parquet"
    existing_pred_df = pd.DataFrame()
    existing_race_ids: set[str] = set()
    if pred_path.exists():
        existing_pred_df = pd.read_parquet(pred_path)
        existing_race_ids = set(existing_pred_df["race_id"].unique())

    # 推論
    race_predictor = RacePredictor(models)
    bankroll = config.initial_bankroll
    diag_logger = DiagnosticLogger()
    all_bets: list[dict[str, object]] = []

    for race_id in race_ids:
        if race_id in skipped_race_ids:
            continue  # 発走前オッズスナップショットなし → スキップ
        if race_id in existing_race_ids:
            continue  # 既に予測済み (重複回避)
        single_race = feat_df[feat_df["race_id"] == race_id].copy()
        hist_race = hist_all[hist_all["race_id"] == race_id]
        jockey_race = jockey_all[jockey_all["race_id"] == race_id]
        trainer_race = trainer_all[trainer_all["race_id"] == race_id]
        jt_race = jt_all[jt_all["race_id"] == race_id]

        result_df = race_predictor.predict(single_race, hist_race, jockey_race, trainer_race, jt_combo_features=jt_race)
        if result_df.empty:
            continue

        # Regime info for diagnostics
        regime = models.regime_detector.current_regime
        regime_params = models.regime_detector.get_strategy_params(regime)
        ev_threshold = regime_params.get("ev_threshold", 1.10)
        if "ev_place" in result_df.columns:
            n_candidates = int((result_df["ev_place"].fillna(0) >= ev_threshold).sum())
        else:
            n_candidates = 0

        if not race_predictor.should_bet(result_df):
            # Log race diagnostic: quality check failed
            diag_logger.log_race(
                race_id=race_id,
                regime=str(regime),
                ev_threshold=ev_threshold,
                quality_passed=False,
                quality_score=0.0,
                n_candidates=n_candidates,
                n_bets=0,
            )
            # Log horse diagnostics: none selected
            if "ev_place" in result_df.columns:
                for _, hr in result_df.iterrows():
                    diag_logger.log_horse(
                        race_id=race_id,
                        umaban=int(hr["umaban"]),
                        p_place_pred=float(hr.get("p_place_pred", 0)),
                        e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                        ev_place=float(hr.get("ev_place", 0)),
                        fukuoddslow=float(hr.get("fukuoddslow", 0)),
                        is_bet=False,
                    )
            continue

        bets = race_predictor.select_bets(result_df, bankroll)

        # Log race diagnostic: quality check passed
        diag_logger.log_race(
            race_id=race_id,
            regime=str(regime),
            ev_threshold=ev_threshold,
            quality_passed=True,
            quality_score=0.0,
            n_candidates=n_candidates,
            n_bets=len(bets),
        )
        # Log horse diagnostics with bet selection info
        if "ev_place" in result_df.columns:
            bet_umabans = {b.umaban for b in bets}
            for _, hr in result_df.iterrows():
                diag_logger.log_horse(
                    race_id=race_id,
                    umaban=int(hr["umaban"]),
                    p_place_pred=float(hr.get("p_place_pred", 0)),
                    e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                    ev_place=float(hr.get("ev_place", 0)),
                    fukuoddslow=float(hr.get("fukuoddslow", 0)),
                    is_bet=int(hr["umaban"]) in bet_umabans,
                )

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
                    "post_time": _race_time_map.get(race_id, ""),
                    "is_paper": True,
                    "predicted_at": datetime.now().isoformat(),
                }
            )
            bankroll -= bet.stake

    # Save diagnostics
    diag_logger.save(config.paper_trading_dir, prefix=f"diag_{ymd}")

    if not all_bets and existing_pred_df.empty:
        logger.info("No bets generated for %s", args.date)
        _send_slack(config, f"Predict: 0 bets for {args.date}")
        return

    # 予測結果を保存 (追記、重複なし)
    if all_bets:
        new_pred_df = pd.DataFrame(all_bets)
        if not existing_pred_df.empty:
            combined_pred_df = pd.concat([existing_pred_df, new_pred_df], ignore_index=True)
        else:
            combined_pred_df = new_pred_df
        combined_pred_df.to_parquet(pred_path, index=False)
        logger.info("Predictions saved: %d new + %d existing → %s",
                     len(all_bets), len(existing_pred_df), pred_path)
    elif not existing_pred_df.empty:
        # No new bets but existing ones remain
        logger.info("No new bets for %s, %d existing predictions preserved", args.date, len(existing_pred_df))

    # コンソール出力 (Windows cp932 対応)
    _venue_map = {
        "01": "札幌",
        "02": "函館",
        "03": "福島",
        "04": "新潟",
        "05": "東京",
        "06": "中山",
        "07": "中京",
        "08": "京都",
        "09": "阪神",
        "10": "小倉",
    }

    def _fmt_race_id(rid: str) -> str:
        jyocd = rid[8:10]
        racenum = rid[14:16]
        venue = _venue_map.get(jyocd, jyocd)
        return f"{venue}{int(racenum):2d}R"

    # New vs Previous bets
    new_bets = [b for b in all_bets if b["race_id"] not in existing_race_ids]
    prev_bets_from_df = existing_pred_df.to_dict("records") if not existing_pred_df.empty else []

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  Predict: {args.date}  -  {len(new_bets)} new bets  ({len(skipped_race_ids)} races skipped)")
    lines.append("=" * 60)

    if new_bets:
        lines.append("  --- New Predictions ---")
        new_bets.sort(key=lambda b: b.get("post_time", "99:99"))
        prev_rid: str = ""
        for b in new_bets:
            rid = b["race_id"]
            if rid != prev_rid:
                t = b.get("post_time", "--:--")
                lines.append(f"  ── {t}  {_fmt_race_id(rid)} ──")
                prev_rid = rid
            lines.append(
                f"      馬番{int(b['umaban']):2d}  {b['horse_name']:<16s}  "
                f"複勝{b['odds']:5.1f}  EV={b['ev']:.2f}"
            )

    if prev_bets_from_df:
        lines.append(f"  --- Previous Predictions ({len(prev_bets_from_df)} bets) ---")
        prev_bets_from_df.sort(key=lambda b: b.get("post_time", "99:99"))
        prev_rid = ""
        for b in prev_bets_from_df:
            rid = b["race_id"]
            if rid != prev_rid:
                t = b.get("post_time", "--:--")
                lines.append(f"  ── {t}  {_fmt_race_id(rid)} ──")
                prev_rid = rid
            name = b.get("horse_name", "")
            lines.append(
                f"      馬番{int(b['umaban']):2d}  {name:<16s}  "
                f"複勝{b['odds']:5.1f}  EV={b['ev']:.2f}"
            )

    lines.append("")
    text = "\n".join(lines)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    # Slack通知
    slack_msg = f"Predict: {len(new_bets)} new bets for {args.date} ({len(skipped_race_ids)} skipped)\n"
    if new_bets:
        slack_msg += "--- New ---\n"
        new_bets.sort(key=lambda b: b.get("post_time", "99:99"))
        for b in new_bets:
            slack_msg += (
                f"  {b.get('post_time', '--:--')} {_fmt_race_id(b['race_id'])} "
                f"馬番{b['umaban']} {b['horse_name']} 複勝{b['odds']:.1f} EV={b['ev']:.2f}\n"
            )
    if prev_bets_from_df:
        slack_msg += f"--- Previous ({len(prev_bets_from_df)} bets) ---\n"
    _send_slack(config, slack_msg)
    logger.info("Predict complete: %d new bets, %d existing", len(new_bets), len(prev_bets_from_df))


# ─────────────────────────────────────────────────
# diagnose: Parquet データで診断推論 (EveryDB2 バイパス)
# ─────────────────────────────────────────────────


def _run_diagnose(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    store: "ParquetStore",
) -> None:
    """Parquet データを使って診断推論を実行 (EveryDB2 バイパス)"""
    from backtest.diagnostic_logger import DiagnosticLogger
    from backtest.race_predictor import RacePredictor
    from db.readers import (
        load_entries,
        load_odds_snapshots,
        load_odds_time_series_range,
        load_races,
    )
    from features.feature_engine import FeatureEngine
    from features.horse_history_features import HorseHistoryFeatures
    from features.jockey_context_features import JockeyContextFeatures
    from features.jockey_trainer_combo import JockeyTrainerComboFeatures
    from features.trainer_context_features import TrainerContextFeatures
    from models.submodel_manager import SubModelManager

    start_ymd = args.start.replace("-", "")
    end_ymd = args.end.replace("-", "")

    # ParquetStore からデータロード (EveryDB2 バイパス)
    logger.info("Loading data from Parquet: %s ~ %s", args.start, args.end)
    race_df = load_races(store, start_ymd, end_ymd)
    entry_df = load_entries(store, start_ymd, end_ymd)
    odds_df = load_odds_snapshots(store, start_ymd, end_ymd)

    if race_df.empty or entry_df.empty or odds_df.empty:
        logger.error("No Parquet data for %s ~ %s", args.start, args.end)
        return

    # 特徴量生成 (_run_predict と同じパイプライン)
    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    odds_ts_df = load_odds_time_series_range(store, start_ymd, end_ymd)
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=store)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(store=store).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(store).compute(entry_df)
    trainer_all = TrainerContextFeatures(store).compute(entry_df)
    jt_all = JockeyTrainerComboFeatures(store).compute(entry_df)

    # 推論 + 診断ログ
    race_predictor = RacePredictor(models)
    diag_logger = DiagnosticLogger()

    for race_id in race_ids:
        single_race = feat_df[feat_df["race_id"] == race_id].copy()
        hist_race = hist_all[hist_all["race_id"] == race_id]
        jockey_race = jockey_all[jockey_all["race_id"] == race_id]
        trainer_race = trainer_all[trainer_all["race_id"] == race_id]
        jt_race = jt_all[jt_all["race_id"] == race_id]

        result_df = race_predictor.predict(single_race, hist_race, jockey_race, trainer_race, jt_combo_features=jt_race)
        if result_df.empty:
            continue

        regime = models.regime_detector.current_regime
        regime_params = models.regime_detector.get_strategy_params(regime)
        ev_threshold = regime_params.get("ev_threshold", 1.10)
        if "ev_place" in result_df.columns:
            n_candidates = int((result_df["ev_place"].fillna(0) >= ev_threshold).sum())
        else:
            n_candidates = 0

        if not race_predictor.should_bet(result_df):
            diag_logger.log_race(
                race_id=race_id,
                regime=str(regime),
                ev_threshold=ev_threshold,
                quality_passed=False,
                quality_score=0.0,
                n_candidates=n_candidates,
                n_bets=0,
            )
            if "ev_place" in result_df.columns:
                for _, hr in result_df.iterrows():
                    diag_logger.log_horse(
                        race_id=race_id,
                        umaban=int(hr["umaban"]),
                        p_place_pred=float(hr.get("p_place_pred", 0)),
                        e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                        ev_place=float(hr.get("ev_place", 0)),
                        fukuoddslow=float(hr.get("fukuoddslow", 0)),
                        is_bet=False,
                    )
            continue

        bets = race_predictor.select_bets(result_df, bankroll=0)
        diag_logger.log_race(
            race_id=race_id,
            regime=str(regime),
            ev_threshold=ev_threshold,
            quality_passed=True,
            quality_score=0.0,
            n_candidates=n_candidates,
            n_bets=len(bets),
        )
        if "ev_place" in result_df.columns:
            bet_umabans = {b.umaban for b in bets}
            for _, hr in result_df.iterrows():
                diag_logger.log_horse(
                    race_id=race_id,
                    umaban=int(hr["umaban"]),
                    p_place_pred=float(hr.get("p_place_pred", 0)),
                    e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                    ev_place=float(hr.get("ev_place", 0)),
                    fukuoddslow=float(hr.get("fukuoddslow", 0)),
                    is_bet=int(hr["umaban"]) in bet_umabans,
                )

    prefix = f"diag_parquet_{start_ymd}_{end_ymd}"
    diag_logger.save(config.paper_trading_dir, prefix=prefix)
    logger.info(
        "Diagnose complete: %d races, %d horses logged",
        len(diag_logger.race_records),
        len(diag_logger.horse_records),
    )


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

    from db.everydb2_queries import EveryDB2Queries

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

    # EveryDB2 払戻テーブルから複勝結果を取得
    db = EveryDB2Queries(config.everydb2_connection_string)
    payout_df = db.get_payouts(ymd)

    if payout_df.empty:
        logger.warning("No payout data for %s -- races may not have finished yet", args.date)
        return

    # race_id → 払戻情報 のルックアップ辞書を構築
    payout_map: dict[str, dict[int, float]] = {}
    for _, row in payout_df.iterrows():
        rid = row["race_id"]
        winners: dict[int, float] = {}
        for i in range(1, 6):
            umaban_str = row.get(f"payfukusyoumaban{i}")
            pay_str = row.get(f"payfukusyopay{i}")
            if pd.isna(umaban_str) or pd.isna(pay_str):
                continue
            umaban_str = str(umaban_str).strip()
            pay_str = str(pay_str).strip()
            if not umaban_str or umaban_str == "00" or not pay_str or pay_str == "0":
                continue
            umaban_int = int(umaban_str)
            # 払戻金は「100円あたりの円」→ オッズ倍率 = pay / 100
            pay_yen = int(pay_str)
            winners[umaban_int] = pay_yen / 100.0
        payout_map[rid] = winners

    logger.info("Payout data: %d races loaded", len(payout_map))

    n_settled = 0
    n_wins = 0

    for idx, row in unsettled.iterrows():
        race_id = row["race_id"]
        umaban = int(row["umaban"])

        winners = payout_map.get(race_id)
        if winners is None:
            continue  # 払戻データなし = レース未確定

        n_settled += 1
        if umaban in winners:
            actual_odds = winners[umaban]
            payout = row["stake"] * actual_odds
            pred_df.at[idx, "result"] = payout
            n_wins += 1
        else:
            pred_df.at[idx, "result"] = 0.0

    if n_settled == 0:
        logger.info("No races settled yet for %s", args.date)
        return

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

    # 日次統計
    day_settled = pred_df[pred_df["result"] != 0.0]
    day_total_stake = pred_df["stake"].sum()
    day_total_return = pred_df["result"].sum()

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

    # コンソール出力
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  Reconcile: {args.date}  -  {n_settled} settled, {n_wins} wins")
    lines.append("=" * 60)

    _venue_map = {
        "01": "札幌",
        "02": "函館",
        "03": "福島",
        "04": "新潟",
        "05": "東京",
        "06": "中山",
        "07": "中京",
        "08": "京都",
        "09": "阪神",
        "10": "小倉",
    }

    def _fmt_race_id(rid: str) -> str:
        jyocd = rid[8:10]
        racenum = rid[14:16]
        venue = _venue_map.get(jyocd, jyocd)
        return f"{venue}{int(racenum):2d}R"

    prev_rid = ""
    for _, row in pred_df.iterrows():
        rid = row["race_id"]
        if rid != prev_rid:
            post_time = str(row.get("post_time", ""))
            lines.append(f"  ── {post_time}  {_fmt_race_id(rid)} ──")
            prev_rid = rid
        res_mark = "---"
        actual_pay = 0.0
        winners = payout_map.get(row["race_id"], {})
        if int(row["umaban"]) in winners:
            res_mark = "WIN"
            actual_pay = row["result"]
        elif row["race_id"] in payout_map:
            res_mark = "LOSE"

        lines.append(
            f"      馬番{int(row['umaban']):2d}  "
            f"予測Odds={row['odds']:.1f}  "
            f"{res_mark:4s}  "
            f"払戻{actual_pay:,.0f}円"
        )

    lines.append("")
    lines.append(
        f"  Day:  Stake={day_total_stake:,.0f}  Return={day_total_return:,.0f}  "
        f"ROI={day_total_return / day_total_stake if day_total_stake else 0:.1%}"
    )
    lines.append(f"  Cum:  {len(combined)} bets  ROI={cumulative_roi:.1%}")
    lines.append("")
    text = "\n".join(lines)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    # Slack通知
    slack_msg = (
        f"Reconcile {ymd}: {n_settled} settled, {n_wins} wins\n"
        f"  Day ROI: {day_total_return / day_total_stake if day_total_stake else 0:.1%}\n"
        f"  Cumulative: {len(combined)} bets, ROI={cumulative_roi:.1%}"
    )
    _send_slack(config, slack_msg)

    logger.info(
        "Reconcile: %d settled, %d wins, day ROI=%.1f%%, cumulative ROI=%.1f%% (%d total bets)",
        n_settled,
        n_wins,
        day_total_return / day_total_stake * 100 if day_total_stake else 0,
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
    from features.jockey_trainer_combo import JockeyTrainerComboFeatures
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
    jt_all = JockeyTrainerComboFeatures(store).compute(entry_df)
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
            jt_race = jt_all[jt_all["race_id"] == race_id]

            result_df = race_predictor.predict(race_df_single, hist_race, jockey_race, trainer_race, jt_combo_features=jt_race)
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
    models, model_info = _load_models(config, use_ensemble=args.ensemble)

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

    elif args.mode == "diagnose":
        _run_diagnose(args, config, models, store)


if __name__ == "__main__":
    main()
