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
from pathlib import Path
import time
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore
    from domain.models import TrainedModelsV5
    from paper_trading.config import PaperTradingConfig

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from domain.types import POST_RACE_COLS  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _drop_post_race_cols(df: pd.DataFrame) -> pd.DataFrame:
    """POST_RACE 列を除外 (BT engine.py と同じ処理)。"""
    return df.drop(
        columns=[c for c in POST_RACE_COLS if c in df.columns],
        errors="ignore",
    )


def _apply_jra_filter(
    feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """JRAフィルタ: NARレース (jyocd >= 30) を除外 (BT engine.py と同じ処理)。"""
    if "jyocd" not in feat_df.columns:
        return feat_df
    jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
    before_count = len(feat_df)
    feat_df = feat_df[jyocd_int.between(1, 10)]
    after_count = len(feat_df)
    if before_count > after_count:
        logger.info(
            "JRA filter: excluded %d NAR entries, %d remaining",
            before_count - after_count,
            after_count,
        )
    return feat_df


def _build_race_stats(result_df: pd.DataFrame) -> dict[str, float]:
    """RegimeDetector 用のレース統計を構築 (BT engine.py と同等)。"""
    from models.regime_detector import calc_favorite_implied_prob, calc_odds_skewness

    row_data = result_df.iloc[0] if not result_df.empty else {}
    return {
        "market_error_std": (
            float(result_df["signed_log_error_win"].std())
            if "signed_log_error_win" in result_df.columns and len(result_df) > 1
            else 0.2
        ),
        "market_error_mean": (
            float(result_df["signed_log_error_win"].mean())
            if "signed_log_error_win" in result_df.columns
            else 0.0
        ),
        "overround_rolling": float(row_data.get("overround", 0.20))
        if not result_df.empty
        else 0.20,
        "entropy_rolling": float(row_data.get("market_entropy", 2.0))
        if not result_df.empty
        else 2.0,
        "odds_skewness_rolling": calc_odds_skewness(result_df),
        "favorite_implied_prob_rolling": calc_favorite_implied_prob(result_df),
        "odds_volatility_mean": (
            float(result_df["odds_volatility"].mean())
            if "odds_volatility" in result_df.columns and not result_df.empty
            else 0.1
        ),
        "field_size_mean": float(row_data.get("field_size", 14.0))
        if not result_df.empty
        else 14.0,
    }


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
        "--ensemble",
        action="store_true",
        help="StackedEnsemble (.joblib) モデルをロード",
    )
    parser.add_argument(
        "--minutes-before",
        type=int,
        default=5,
        help="発走何分前のオッズを使用するか (デフォルト: 5)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Git dirty 状態でも PT 実行を許可 (開発用)",
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
    from db.model_loader import ModelLoader

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
    from db.odds_extractor import extract_pre_post_odds
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
    from paper_trading.reconciler import PaperReconciler

    import pandas as pd

    target_date = date.fromisoformat(args.date)
    ymd = target_date.strftime("%Y%m%d")

    # Session ID generation / crash recovery (D-02)
    sessions_dir = config.paper_trading_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / f"{ymd}.json"
    if session_file.exists():
        session_data = json.loads(session_file.read_text(encoding="utf-8"))
        session_id = session_data["session_id"]
    else:
        import uuid
        session_id = uuid.uuid4().hex[:16]
        session_file.write_text(
            json.dumps({"session_id": session_id, "created_at": datetime.now().isoformat()}),
            encoding="utf-8",
        )

    # ── PT Startup Verification (D-06, D-07, D-08) ──
    from features.data_cutoff_manifest import DataCutoffManifest
    from features.feature_manifest import FeatureManifest, FeatureState
    from features.pipeline_consistency import PFPVerifier
    from features.session_manifest import SessionManifest, get_code_version, write_session_manifest

    # D-06: Git dirty 状態検出
    code_version = get_code_version()
    if code_version["git_dirty"] and not getattr(args, "allow_dirty", False):
        logger.error(
            "Git dirty state detected — uncommitted changes in src/scripts/config. "
            "Use --allow-dirty to override (development only)."
        )
        sys.exit(1)
    elif code_version["git_dirty"]:
        logger.warning(
            "Git dirty state detected but --allow-dirty is set. "
            "Proceeding with warning (diff_hash=%s...)",
            (code_version.get("dirty_diff_hash") or "N/A")[:8],
        )

    # D-07: DataCutoffManifest — 全データソースの最終日付検証
    strategy_manifest_path = (
        Path(ROOT) / "data" / "strategy_manifest.json"
    )
    cutoff_manifest = DataCutoffManifest.from_config(
        prediction_date=args.date,
        models=models,
        strategy_manifest_path=strategy_manifest_path if strategy_manifest_path.exists() else None,
    )
    actual_cutoff = {
        "model_train_end": cutoff_manifest.model_train_end,
        "stats_fit_end": cutoff_manifest.stats_fit_end,
        "odds_band_calibration_end": cutoff_manifest.odds_band_calibration_end,
        "strategy_optimization_end": cutoff_manifest.strategy_optimization_end,
    }
    cutoff_manifest.verify_strict(actual_cutoff)
    logger.info("Data cutoff verification passed for %s", args.date)

    # D-08: PFPVerifier — パラメータ不変性検証の準備
    first_submodel = next(iter(models.submodels.values()))
    try:
        feature_state = FeatureState.from_submodel_set(first_submodel, version="1.0")
    except ValueError:
        feature_state = FeatureState(
            track_stats={}, track_month_stats={}, feature_version="1.0"
        )
        logger.warning("Using empty FeatureState for PFP verification")
    feature_manifest = FeatureManifest(
        column_names=tuple(), column_dtypes=tuple(), feature_version="1.0"
    )
    pfp_verifier = PFPVerifier(
        models, feature_manifest, feature_state,
        betting_target="place",
        betting_mode="flat",
    )
    pfp_verifier.freeze()

    # D-09: SessionManifest — 実行記録
    session_manifest = SessionManifest(
        session_id=session_id,
        prediction_date=args.date,
    )
    session_manifest.set_code_version(code_version)
    session_manifest.set_model_identity(
        run_id=config.mlflow_run_id or "",
        training_start=models.train_period[0],
        training_end=models.train_period[1],
        manifest_hash=feature_manifest.compute_hash(),
    )
    session_manifest_path = config.paper_trading_dir / "session_manifest.json"
    write_session_manifest(session_manifest, session_manifest_path)

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

    # 発走時刻マッピングを先に構築 (extract_pre_post_odds のログと出力で使用)
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
        odds_df = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=minutes_before)
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

    # JRAフィルタ: NARレースを除外 (BT と同等)
    feat_df = _apply_jra_filter(feat_df)

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

    # 種牡馬特徴量 (SireFeatures)
    from db.readers import load_horses, load_sire_stats
    from features.sire_features import SireFeatures

    sire_stats_pt = load_sire_stats(store)
    if not sire_stats_pt.empty:
        horses_pt = load_horses(store)
        sire_feat_pt = SireFeatures(sire_stats_pt)
        sire_map_pt = horses_pt.set_index("kettonum")["ketto3infohansyokunum1"]
        bms_source_col_pt = (
            "ketto3infohansyokunum5"
            if "ketto3infohansyokunum5" in horses_pt.columns
            else "ketto3infohansyokunum3"
        )
        bms_map_pt = horses_pt.set_index("kettonum")[bms_source_col_pt]
        feat_df["sire_id"] = feat_df["kettonum"].map(sire_map_pt)
        feat_df["bms_id"] = feat_df["kettonum"].map(bms_map_pt)
        sire_result_pt = sire_feat_pt.compute_batch(feat_df)
        _sire_cols = {
            "sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_prize_avg",
            "bms_wr", "bms_distance_wr", "bms_surface_wr", "bms_has_history",
            "bms_starts_log", "bms_surface_starts_log", "bms_distance_starts_log",
        }
        for sc in _sire_cols:
            if sc in sire_result_pt.columns:
                feat_df[sc] = sire_result_pt[sc].values

    # ペース適性 + コース別適性特徴量
    from features.course_features import CourseFeatures
    from features.pace_aptitude_features import PaceAptitudeFeatures

    pace_feat = PaceAptitudeFeatures(store=store)
    pace_df = pace_feat.compute_batch(feat_df)
    _pace_cols = [c for c in ["pace_aptitude", "front_pace_wr", "closing_pace_wr"] if c in pace_df.columns]
    if _pace_cols:
        feat_df = feat_df.drop(columns=_pace_cols, errors="ignore").merge(
            pace_df[["kettonum", "race_id"] + _pace_cols], on=["kettonum", "race_id"], how="left"
        )

    course_feat = CourseFeatures(store=store)
    course_df = course_feat.compute_batch(feat_df)
    _course_cols = [c for c in ["course_wr", "course_distance_wr"] if c in course_df.columns]
    if _course_cols:
        feat_df = feat_df.drop(columns=_course_cols, errors="ignore").merge(
            course_df[["kettonum", "race_id"] + _course_cols], on=["kettonum", "race_id"], how="left"
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

    # RegimeDetector 用: 直近200レースの統計を蓄積 (BT と同等)
    recent_stats_list: list[dict[str, float]] = []

    for race_id in race_ids:
        if race_id in skipped_race_ids:
            continue  # 発走前オッズスナップショットなし → スキップ
        if race_id in existing_race_ids:
            continue  # 既に予測済み (重複回避)

        # D-08: Pre-race PFP verification
        pfp_result = pfp_verifier.verify()
        if not pfp_result["passed"]:
            logger.error(
                "PFP verification failed before race %s: %s",
                race_id, pfp_result["message"],
            )
            session_manifest.set_pfp_result(pfp_result)
            session_manifest.set_status("aborted", exit_code=1)
            write_session_manifest(session_manifest, session_manifest_path)
            sys.exit(1)

        single_race = feat_df[feat_df["race_id"] == race_id].copy()
        hist_race = hist_all[hist_all["race_id"] == race_id]
        jockey_race = jockey_all[jockey_all["race_id"] == race_id]
        trainer_race = trainer_all[trainer_all["race_id"] == race_id]
        jt_race = jt_all[jt_all["race_id"] == race_id]

        # POST_RACE 列を除外 (BT engine.py と同じ処理)
        single_race = _drop_post_race_cols(single_race)

        result_df = race_predictor.predict(
            single_race, hist_race, jockey_race, trainer_race, jt_combo_features=jt_race
        )
        if result_df.empty:
            continue

        # 統計を蓄積してレジーム判定 (BT engine.py と同等)
        recent_stats_list.append(_build_race_stats(result_df))
        recent_stats_df = pd.DataFrame(recent_stats_list[-200:])
        if len(recent_stats_df) >= models.regime_detector.cfg.min_samples:
            regime = models.regime_detector.detect(recent_stats_df)
        else:
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
                    diag_logger.log_horse_features(hr.to_dict())
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
                diag_logger.log_horse_features(hr.to_dict())

        for bet in bets:
            horse = result_df[result_df["umaban"] == bet.umaban]
            horse_name = _decode_bamei(horse.iloc[0]["bamei"]) if not horse.empty else ""
            bet_id = PaperReconciler.compute_bet_id(
                session_id, race_id, bet.bet_type.value, bet.umaban,
            )
            all_bets.append(
                {
                    "bet_id": bet_id,
                    "session_id": session_id,
                    "schema_version": 2,
                    "settlement_status": "pending",
                    "outcome": None,
                    "payout": None,
                    "race_id": race_id,
                    "bet_type": bet.bet_type.value,
                    "umaban": bet.umaban,
                    "horse_name": horse_name,
                    "stake": bet.stake,
                    "odds": bet.odds,
                    "ev": bet.ev_lower_corrected,
                    "surface": result_df.iloc[0].get("surface", ""),
                    "distance": result_df.iloc[0].get("kyori", 0),
                    "bankroll_after": round(bankroll - bet.stake, 2),
                    "race_date": pd.Timestamp(ymd),
                    "post_time": _race_time_map.get(race_id, ""),
                    "is_paper": True,
                    "predicted_at": datetime.now().isoformat(),
                }
            )
            bankroll -= bet.stake

    # Save diagnostics
    diag_logger.save(config.paper_trading_dir, prefix=f"diag_{ymd}")

    # D-08: End-of-run PFP verification
    pfp_result = pfp_verifier.verify()
    session_manifest.set_pfp_result(pfp_result)
    if not pfp_result["passed"]:
        logger.error("PFP verification failed at end of run: %s", pfp_result["message"])
        session_manifest.set_status("failed", exit_code=1)
    else:
        session_manifest.set_status("completed", exit_code=0)
    write_session_manifest(session_manifest, session_manifest_path)

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
        logger.info(
            "Predictions saved: %d new + %d existing → %s",
            len(all_bets),
            len(existing_pred_df),
            pred_path,
        )
    elif not existing_pred_df.empty:
        # No new bets but existing ones remain
        logger.info(
            "No new bets for %s, %d existing predictions preserved",
            args.date,
            len(existing_pred_df),
        )

    # Append new bets to cumulative bets.parquet (source of truth, D-08)
    if all_bets:
        new_bet_rows = pd.DataFrame(all_bets)
        bets_path = config.paper_trading_dir / "bets.parquet"
        if bets_path.exists():
            existing_bets = pd.read_parquet(bets_path)
            # Old schema rejection (D-18)
            if "result" in existing_bets.columns and "payout" not in existing_bets.columns:
                raise ValueError(
                    "Old schema detected in bets.parquet: 'result' column present without 'payout'. "
                    "Migration not supported -- recreate bets from predictions."
                )
            combined_bets = pd.concat([existing_bets, new_bet_rows], ignore_index=True)
            # Dedup by bet_id (D-02)
            combined_bets = combined_bets.drop_duplicates(subset=["bet_id"], keep="last")
        else:
            combined_bets = new_bet_rows

        # Schema validation (D-20)
        errors = PaperReconciler._validate_bet_schema(combined_bets)
        if errors:
            raise ValueError(f"Bet schema validation failed: {'; '.join(errors)}")

        # Atomic write
        PaperReconciler._atomic_write_parquet(combined_bets, bets_path)
        logger.info(
            "Bets appended to cumulative bets.parquet: %d new → %s",
            len(all_bets), bets_path,
        )

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
    lines.append(
        f"  Predict: {args.date}  -  {len(new_bets)} new bets  ({len(skipped_race_ids)} races skipped)"
    )
    lines.append("=" * 60)

    if new_bets:
        lines.append("  --- New Predictions ---")
        new_bets.sort(key=lambda b: b.get("post_time", "99:99"))
        prev_rid: str = ""
        for b in new_bets:
            rid = b["race_id"]
            if rid != prev_rid:
                if prev_rid:
                    lines.append("")
                t = b.get("post_time", "--:--")
                lines.append(f"  ── {t}  {_fmt_race_id(rid)} ──")
                prev_rid = rid
            lines.append(
                f"      馬番{int(b['umaban']):2d}  {b['horse_name']:<16s}  "
                f"複勝{b['odds']:5.1f}  EV={b['ev']:.2f}"
            )

    if prev_bets_from_df:
        lines.append("")
        lines.append(f"  --- Previous Predictions ({len(prev_bets_from_df)} bets) ---")
        prev_bets_from_df.sort(key=lambda b: b.get("post_time", "99:99"))
        prev_rid = ""
        for b in prev_bets_from_df:
            rid = b["race_id"]
            if rid != prev_rid:
                if prev_rid:
                    lines.append("")
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
    slack_msg = (
        f"Predict: {len(new_bets)} new bets for {args.date} ({len(skipped_race_ids)} skipped)\n"
    )
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
    from db.odds_extractor import extract_pre_post_odds
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

    # 発走前オッズを優先使用 (フォールバック: 確定オッズ)
    odds_ts_df = load_odds_time_series_range(store, start_ymd, end_ymd)
    minutes_before = getattr(args, "minutes_before", 5)
    if not odds_ts_df.empty and "hassotime" in race_df.columns:
        pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=minutes_before)
        if not pre_post_odds.empty:
            logger.info("Using pre-race odds for diagnose (%d entries)", len(pre_post_odds))
            odds_df = pre_post_odds
        else:
            logger.info("Pre-race odds empty, falling back to confirmed odds")
    else:
        logger.info("No time-series odds, using confirmed odds")

    # 特徴量生成 (_run_predict と同じパイプライン)
    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=store)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    # JRAフィルタ: NARレースを除外 (BT と同等)
    feat_df = _apply_jra_filter(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(store=store).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(store).compute(entry_df)
    trainer_all = TrainerContextFeatures(store).compute(entry_df)
    jt_all = JockeyTrainerComboFeatures(store).compute(entry_df)

    # ペース適性 + コース別適性 + 種牡馬特徴量 (予測パスで必要)
    from db.readers import load_horses, load_sire_stats
    from features.course_features import CourseFeatures
    from features.pace_aptitude_features import PaceAptitudeFeatures
    from features.sire_features import SireFeatures

    pace_feat2 = PaceAptitudeFeatures(store=store)
    pace_df2 = pace_feat2.compute_batch(feat_df)
    _pace_cols2 = [c for c in ["pace_aptitude", "front_pace_wr", "closing_pace_wr"] if c in pace_df2.columns]
    if _pace_cols2:
        feat_df = feat_df.drop(columns=_pace_cols2, errors="ignore").merge(
            pace_df2[["kettonum", "race_id"] + _pace_cols2], on=["kettonum", "race_id"], how="left"
        )

    course_feat2 = CourseFeatures(store=store)
    course_df2 = course_feat2.compute_batch(feat_df)
    _course_cols2 = [c for c in ["course_wr", "course_distance_wr"] if c in course_df2.columns]
    if _course_cols2:
        feat_df = feat_df.drop(columns=_course_cols2, errors="ignore").merge(
            course_df2[["kettonum", "race_id"] + _course_cols2], on=["kettonum", "race_id"], how="left"
        )

    sire_stats_pt2 = load_sire_stats(store)
    if not sire_stats_pt2.empty:
        horses_pt2 = load_horses(store)
        sire_feat2 = SireFeatures(sire_stats_pt2)
        sire_map2 = horses_pt2.set_index("kettonum")["ketto3infohansyokunum1"]
        bms_source_col2 = (
            "ketto3infohansyokunum5"
            if "ketto3infohansyokunum5" in horses_pt2.columns
            else "ketto3infohansyokunum3"
        )
        bms_map2 = horses_pt2.set_index("kettonum")[bms_source_col2]
        feat_df["sire_id"] = feat_df["kettonum"].map(sire_map2)
        feat_df["bms_id"] = feat_df["kettonum"].map(bms_map2)
        sire_result2 = sire_feat2.compute_batch(feat_df)
        for sc2 in {
            "sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_prize_avg",
            "bms_wr", "bms_distance_wr", "bms_surface_wr", "bms_has_history",
            "bms_starts_log", "bms_surface_starts_log", "bms_distance_starts_log",
        }:
            if sc2 in sire_result2.columns:
                feat_df[sc2] = sire_result2[sc2].values

    # 推論 + 診断ログ
    race_predictor = RacePredictor(models)
    diag_logger = DiagnosticLogger()

    # RegimeDetector 用: 直近200レースの統計を蓄積 (BT と同等)
    recent_stats_list: list[dict[str, float]] = []

    for race_id in race_ids:
        single_race = feat_df[feat_df["race_id"] == race_id].copy()
        hist_race = hist_all[hist_all["race_id"] == race_id]
        jockey_race = jockey_all[jockey_all["race_id"] == race_id]
        trainer_race = trainer_all[trainer_all["race_id"] == race_id]
        jt_race = jt_all[jt_all["race_id"] == race_id]

        # POST_RACE 列を除外 (BT engine.py と同じ処理)
        single_race = _drop_post_race_cols(single_race)

        result_df = race_predictor.predict(
            single_race, hist_race, jockey_race, trainer_race, jt_combo_features=jt_race
        )
        if result_df.empty:
            continue

        # 統計を蓄積してレジーム判定 (BT engine.py と同等)
        recent_stats_list.append(_build_race_stats(result_df))
        recent_stats_df = pd.DataFrame(recent_stats_list[-200:])
        if len(recent_stats_df) >= models.regime_detector.cfg.min_samples:
            regime = models.regime_detector.detect(recent_stats_df)
        else:
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
                    diag_logger.log_horse_features(hr.to_dict())
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
                diag_logger.log_horse_features(hr.to_dict())

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
    """Thin CLI wrapper for PaperReconciler.reconcile() (D-01)."""
    from db.everydb2_queries import EveryDB2Queries
    from paper_trading.reconciler import PaperReconciler

    target_date = date.fromisoformat(args.date)
    ymd = target_date.strftime("%Y%m%d")

    # Construct dependencies
    everydb2 = EveryDB2Queries(config.everydb2_connection_string)
    bets_path = config.paper_trading_dir / "bets.parquet"

    reconciler = PaperReconciler(
        bets_path=bets_path,
        everydb2=everydb2,
        retry_interval=60,
        retry_timeout=600,
    )

    # Execute reconciliation
    result = reconciler.reconcile(target_date)

    # Retry if requested and pending remain
    if getattr(args, "retry", False) and result.get("n_pending", 0) > 0:
        result = reconciler.retry_pending(target_date)

    # Display results
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    n_settled = result.get("n_settled", 0)
    n_wins = result.get("n_new_wins", 0)
    lines.append(f"  Reconcile: {args.date}  -  {n_settled} settled, {n_wins} wins")
    lines.append("=" * 60)
    lines.append(f"  Cum: {result.get('n_bets', 0)} bets  "
                 f"Effective Stake={result.get('effective_stake', 0):,.0f}  "
                 f"Return={result.get('total_return', 0):,.0f}  "
                 f"ROI={result.get('cumulative_roi', 0):.1%}")
    if result.get("n_pending", 0) > 0:
        lines.append(f"  Pending: {result['n_pending']} bets still unsettled")
    lines.append("")
    text = "\n".join(lines)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    # Save daily summary
    summary_dir = config.paper_trading_dir / "daily_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{ymd}.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Slack notification
    slack_msg = (
        f"Reconcile {ymd}: {n_settled} settled, {n_wins} wins\n"
        f"  ROI={result.get('cumulative_roi', 0):.1%}  "
        f"Pending={result.get('n_pending', 0)}"
    )
    _send_slack(config, slack_msg)
    logger.info(
        "Reconcile: %d settled, %d wins, ROI=%.1f%%, %d pending",
        n_settled, n_wins, result.get("cumulative_roi", 0) * 100,
        result.get("n_pending", 0),
    )

    # Exit code 2 if pending remain (D-06)
    if result.get("exit_code", 0) == 2 or result.get("n_pending", 0) > 0:
        sys.exit(2)


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
    from db.odds_extractor import extract_pre_post_odds
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
        start = date.fromisoformat(args.start.replace("-", ""))
        end = date.fromisoformat(args.end.replace("-", ""))
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

    # 発走前オッズを優先使用 (フォールバック: 確定オッズ)
    minutes_before = getattr(args, "minutes_before", 5)
    if not odds_ts_df.empty and "hassotime" in race_df.columns:
        pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=minutes_before)
        if not pre_post_odds.empty:
            logger.info("Using pre-race odds for dry-run (%d entries)", len(pre_post_odds))
            odds_df = pre_post_odds
        else:
            logger.info("Pre-race odds empty, falling back to confirmed odds")
    else:
        logger.info("No time-series odds, using confirmed odds")

    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    # JRAフィルタ: NARレースを除外 (BT と同等)
    feat_df = _apply_jra_filter(feat_df)

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

    # 種牡馬特徴量 (SireFeatures)
    from db.readers import load_horses, load_sire_stats
    from features.sire_features import SireFeatures

    sire_stats_pt3 = load_sire_stats(store)
    if not sire_stats_pt3.empty:
        horses_pt3 = load_horses(store)
        sire_feat3 = SireFeatures(sire_stats_pt3)
        sire_map3 = horses_pt3.set_index("kettonum")["ketto3infohansyokunum1"]
        bms_source_col3 = (
            "ketto3infohansyokunum5"
            if "ketto3infohansyokunum5" in horses_pt3.columns
            else "ketto3infohansyokunum3"
        )
        bms_map3 = horses_pt3.set_index("kettonum")[bms_source_col3]
        feat_df["sire_id"] = feat_df["kettonum"].map(sire_map3)
        feat_df["bms_id"] = feat_df["kettonum"].map(bms_map3)
        sire_result3 = sire_feat3.compute_batch(feat_df)
        for sc3 in {
            "sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_prize_avg",
            "bms_wr", "bms_distance_wr", "bms_surface_wr", "bms_has_history",
            "bms_starts_log", "bms_surface_starts_log", "bms_distance_starts_log",
        }:
            if sc3 in sire_result3.columns:
                feat_df[sc3] = sire_result3[sc3].values

    # ペース適性 + コース別適性特徴量
    from features.course_features import CourseFeatures
    from features.pace_aptitude_features import PaceAptitudeFeatures

    pace_feat3 = PaceAptitudeFeatures(store=store)
    pace_df3 = pace_feat3.compute_batch(feat_df)
    _pace_cols3 = [c for c in ["pace_aptitude", "front_pace_wr", "closing_pace_wr"] if c in pace_df3.columns]
    if _pace_cols3:
        feat_df = feat_df.drop(columns=_pace_cols3, errors="ignore").merge(
            pace_df3[["kettonum", "race_id"] + _pace_cols3], on=["kettonum", "race_id"], how="left"
        )

    course_feat3 = CourseFeatures(store=store)
    course_df3 = course_feat3.compute_batch(feat_df)
    _course_cols3 = [c for c in ["course_wr", "course_distance_wr"] if c in course_df3.columns]
    if _course_cols3:
        feat_df = feat_df.drop(columns=_course_cols3, errors="ignore").merge(
            course_df3[["kettonum", "race_id"] + _course_cols3], on=["kettonum", "race_id"], how="left"
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
            race_df_full = feat_df[feat_df["race_id"] == race_id].copy()
            hist_race = hist_all[hist_all["race_id"] == race_id]
            jockey_race = jockey_all[jockey_all["race_id"] == race_id]
            trainer_race = trainer_all[trainer_all["race_id"] == race_id]
            jt_race = jt_all[jt_all["race_id"] == race_id]

            # POST_RACE 列を除外 (BT engine.py と同じ処理)
            race_df_single = _drop_post_race_cols(race_df_full.copy())

            result_df = race_predictor.predict(
                race_df_single, hist_race, jockey_race, trainer_race, jt_combo_features=jt_race
            )
            if result_df.empty:
                continue

            if not race_predictor.should_bet(result_df):
                continue

            bets = race_predictor.select_bets(result_df, bankroll)
            for bet in bets:
                horse_full = race_df_full[race_df_full["umaban"] == bet.umaban]
                if not horse_full.empty and "kakuteijyuni" in horse_full.columns:
                    finish_pos = int(horse_full.iloc[0]["kakuteijyuni"])
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
