"""RunModeOrchestrator -- full PT lifecycle for --mode run (AUT-01).

Encapsulates: schedule -> TC fetch -> sequential per-race predict -> reconcile -> aggregate.
Supports crash resume via RaceProgress state tracking (AUT-02).
Produces structured exit codes per D-17 taxonomy (AUT-03).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from paper_trading.exit_codes import ExitCode, determine_final_exit_code
from paper_trading.race_progress import RaceProgress, RaceState

if TYPE_CHECKING:
    import argparse

    from db.parquet_store import ParquetStore
    from domain.models import TrainedModelsV5
    from features.session_manifest import SessionManifest
    from paper_trading.config import PaperTradingConfig

logger = logging.getLogger(__name__)


class RunModeOrchestrator:
    """Orchestrates the full PT lifecycle for --mode run.

    Phases (in order):
        1. _ensure_schedule   — load or fetch race schedule
        2. _fetch_track_conditions — fetch live TC (Phase 53 pattern)
        3. _predict_races     — sequential per-race prediction
        4. _reconcile         — settle bets after last race
        5. _aggregate_and_report — generate JSON/HTML reports
    """

    def __init__(
        self,
        config: PaperTradingConfig,
        models: TrainedModelsV5,
        store: ParquetStore,
        args: argparse.Namespace,
        strategy_config: dict[str, Any],
        session_manifest: SessionManifest,
    ) -> None:
        self.config = config
        self.models = models
        self.store = store
        self.args = args
        self.strategy_config = strategy_config
        self.session_manifest = session_manifest

        # Derived paths
        self.target_date: date = date.fromisoformat(args.date)
        self.ymd: str = self.target_date.strftime("%Y%m%d")
        self.session_id: str = getattr(session_manifest, "session_id", f"{self.ymd}_run")
        self.session_dir: Path = config.paper_trading_dir / "sessions" / self.session_id
        self.bets_path: Path = config.paper_trading_dir / "bets.parquet"
        self.race_progress_path: Path = self.session_dir / "race_progress.json"

        # State
        self._cancelled: bool = False
        self.errors: list[ExitCode] = []
        self._schedule: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self) -> ExitCode:
        """Run the full lifecycle. Returns final ExitCode."""
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)

            if not self._ensure_schedule():
                return self._determine_exit_code()

            if not self._fetch_track_conditions():
                # Non-fatal: TC fetch failure is logged but we continue
                # (some venues may not have TC data)
                logger.warning("Track condition fetch had issues, continuing")

            self._predict_races()
            self._reconcile()
            self._aggregate_and_report()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down gracefully")
            self._cancelled = True
        except Exception as e:
            logger.error("Unexpected error in run mode: %s", e)
            self.errors.append(ExitCode.GENERAL_ERROR)

        return self._determine_exit_code()

    # ------------------------------------------------------------------
    # Phase 1: Schedule
    # ------------------------------------------------------------------

    def _ensure_schedule(self) -> bool:
        """Load existing schedule or fetch from DB (D-03).

        Returns False on DB error.
        """
        schedule_path = self.config.paper_trading_dir / "schedule.json"

        # Try to reuse existing schedule
        if schedule_path.exists():
            try:
                data = json.loads(schedule_path.read_text(encoding="utf-8"))
                if data.get("date") == self.args.date and data.get("races"):
                    self._schedule = data["races"]
                    logger.info(
                        "Reusing existing schedule: %d races (D-03)", len(self._schedule)
                    )
                    return True
                else:
                    logger.info("Schedule date mismatch, re-fetching")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read schedule.json: %s", e)

        # Fetch from DB
        return self._run_setup_logic()

    def _run_setup_logic(self) -> bool:
        """Fetch schedule from EveryDB2 (extracted from _run_setup)."""
        try:
            from db.everydb2_queries import EveryDB2Queries
            from db.readers import load_entries_from_db, load_races_from_db
        except ImportError:
            logger.error("DB modules not available")
            self.errors.append(ExitCode.DB_FETCH_ERROR)
            return False

        try:
            db = EveryDB2Queries(self.config.everydb2_connection_string)
            race_df = load_races_from_db(db, self.ymd)
            entry_df = load_entries_from_db(db, self.ymd)
        except Exception as e:
            logger.error("DB fetch error during schedule setup: %s", e)
            self.errors.append(ExitCode.DB_FETCH_ERROR)
            return False

        if race_df.empty:
            logger.warning("No races found for %s", self.args.date)
            self._schedule = []
            return True

        # Build schedule
        schedule: list[dict[str, Any]] = []
        for race_id in race_df["race_id"].unique():
            race = race_df[race_df["race_id"] == race_id].iloc[0]
            entries = entry_df[entry_df["race_id"] == race_id]
            schedule.append(
                {
                    "race_id": race_id,
                    "surface": race.get("surface", ""),
                    "distance": int(race.get("kyori", 0)),
                    "post_time": str(race.get("hassotime", "")),
                    "n_horses": len(entries),
                    "n_with_results": int(entries["kakuteijyuni"].notna().sum()),
                }
            )

        # Save schedule
        schedule_path = self.config.paper_trading_dir / "schedule.json"
        schedule_path.write_text(
            json.dumps(
                {"date": self.args.date, "races": schedule},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._schedule = schedule
        logger.info("Fetched %d races for %s", len(schedule), self.args.date)
        return True

    # ------------------------------------------------------------------
    # Phase 2: Track conditions
    # ------------------------------------------------------------------

    def _fetch_track_conditions(self) -> bool:
        """Fetch live track conditions for all venues (D-04).

        Reuses Phase 53 JRATrackConditionFetcher pattern.
        On resume, reuses existing session TC snapshot.
        Returns True on success or non-fatal skip.
        """
        tc_path = self.session_dir / "track_conditions.parquet"

        # Resume: reuse existing TC snapshot
        if tc_path.exists():
            logger.info("Reusing existing track conditions snapshot (D-04)")
            return True

        try:
            from ingestion.track_condition_fetcher import JRATrackConditionFetcher

            fetcher = JRATrackConditionFetcher()
            tc_df = fetcher.fetch_all_venues(self.target_date)

            if tc_df.empty:
                logger.warning("No track condition data fetched for %s", self.args.date)
                return True  # Non-fatal: some days have no TC data

            # Validate: measurement time should be from today
            self.session_dir.mkdir(parents=True, exist_ok=True)
            tc_df.to_parquet(tc_path, index=False)
            logger.info("Track conditions saved: %d records", len(tc_df))
            return True

        except ImportError:
            logger.warning("TrackConditionFetcher not available, skipping TC fetch")
            return True
        except Exception as e:
            logger.error("Track condition fetch failed: %s", e)
            self.errors.append(ExitCode.DATA_INTEGRITY_ERROR)
            return False

    # ------------------------------------------------------------------
    # Phase 3: Sequential per-race prediction
    # ------------------------------------------------------------------

    def _predict_races(self) -> None:
        """Process each race sequentially with resume support (D-05, D-06)."""
        if not self._schedule:
            logger.info("No races to predict")
            return

        progress = RaceProgress.load(self.race_progress_path)

        # Load existing bets for cross-validation
        bets_df = self._load_bets()

        # Sort races by post_time
        sorted_schedule = sorted(self._schedule, key=lambda r: r.get("post_time", "99:99"))

        for race_info in sorted_schedule:
            race_id = race_info["race_id"]

            # Check cancellation
            if self._cancelled:
                logger.info("Cancelled, stopping prediction")
                break

            current_state = progress.get_state(race_id)

            # Skip completed races (D-06)
            if current_state in (str(RaceState.PREDICTED), str(RaceState.NO_BET)):
                # Cross-validate (D-08)
                if current_state == str(RaceState.PREDICTED):
                    if not self._cross_validate_race(race_id, bets_df, progress):
                        logger.warning(
                            "Cross-validation failed for %s, marking for reprocessing (D-08)",
                            race_id,
                        )
                        progress.mark(race_id, RaceState.FAILED, failure_reason="cross_validation")
                        # Fall through to reprocess
                    else:
                        logger.debug("Skipping already predicted race: %s", race_id)
                        continue
                else:
                    logger.debug("Skipping no_bet race: %s", race_id)
                    continue

            # Process PENDING / FAILED / PROCESSING / untracked races
            post_time_str = race_info.get("post_time", "")
            if post_time_str:
                try:
                    target_time = self._parse_post_time(self.target_date, post_time_str)
                    wait_minutes = getattr(self.args, "minutes_before", 5)
                    wait_target = target_time - timedelta(minutes=wait_minutes)

                    # Wait until N minutes before post (with cancel check)
                    if not self._wait_until_with_cancel(wait_target):
                        logger.info("Cancelled while waiting for race %s", race_id)
                        break
                except (ValueError, AttributeError):
                    logger.warning("Invalid post_time '%s' for %s", post_time_str, race_id)

            # Mark PROCESSING
            progress.mark(race_id, RaceState.PROCESSING)

            try:
                # Attempt prediction
                self._predict_single_race(race_id, race_info, progress)
            except Exception as e:
                logger.error("Prediction failed for %s: %s", race_id, e)
                progress.mark(race_id, RaceState.FAILED, failure_reason=str(e))
                self.errors.append(ExitCode.GENERAL_ERROR)

    def _predict_single_race(
        self,
        race_id: str,
        race_info: dict[str, Any],
        progress: RaceProgress,
    ) -> None:
        """Predict a single race and save results.

        This follows the same pattern as _run_predict in run_paper_trading.py.
        """
        from backtest.race_predictor import RacePredictor

        race_predictor = RacePredictor(self.models)

        # Load race data for this single race
        feat_df = self._load_features_for_race(race_id)
        if feat_df is None or feat_df.empty:
            logger.warning("No features for race %s, marking no_bet", race_id)
            progress.mark(race_id, RaceState.NO_BET)
            return

        # Drop POST_RACE columns
        from domain.types import POST_RACE_COLS
        feat_df = feat_df.drop(
            columns=[c for c in POST_RACE_COLS if c in feat_df.columns],
            errors="ignore",
        )

        # Run prediction
        result_df = race_predictor.predict(feat_df)
        if result_df.empty:
            progress.mark(race_id, RaceState.NO_BET)
            return

        if not race_predictor.should_bet(result_df):
            progress.mark(race_id, RaceState.NO_BET)
            return

        # Select bets
        bankroll = self._compute_current_bankroll()
        bets = race_predictor.select_bets(result_df, bankroll)

        if not bets:
            progress.mark(race_id, RaceState.NO_BET)
            return

        # Save bets
        bet_records = []
        bet_ids = []
        for bet in bets:
            from paper_trading.reconciler import PaperReconciler

            bet_id = PaperReconciler.compute_bet_id(
                self.session_id, race_id, bet.bet_type.value, bet.umaban,
            )
            bet_ids.append(bet_id)

            horse = result_df[result_df["umaban"] == bet.umaban]
            horse_name = ""
            if not horse.empty and "bamei" in horse.columns:
                try:
                    horse_name = horse.iloc[0]["bamei"]
                    if not isinstance(horse_name, str):
                        horse_name = str(horse_name)
                except Exception:
                    horse_name = ""

            bankroll -= bet.stake
            bet_records.append(
                {
                    "bet_id": bet_id,
                    "race_id": race_id,
                    "umaban": bet.umaban,
                    "horse_name": horse_name,
                    "bet_type": bet.bet_type.value,
                    "stake": bet.stake,
                    "odds": bet.odds,
                    "ev": getattr(bet, "ev_lower_corrected", bet.ev),
                    "schema_version": 2,
                    "settlement_status": "pending",
                    "outcome": None,
                    "payout": None,
                    "race_date": pd.Timestamp(self.target_date),
                    "session_id": self.session_id,
                    "bankroll_after": bankroll,
                    "is_paper": True,
                    "predicted_at": datetime.now().isoformat(),
                }
            )

        # Append to bets.parquet atomically
        self._append_bets(bet_records)

        # Save input snapshot (D-09)
        odds_df = self._load_odds_for_race(race_id)
        fallback_odds = odds_df if odds_df is not None else pd.DataFrame()
        self._save_input_snapshot(race_id, feat_df, fallback_odds)

        # Mark PREDICTED with bet_ids
        progress.mark(race_id, RaceState.PREDICTED, bet_ids=bet_ids)
        logger.info("Predicted %d bets for race %s", len(bet_records), race_id)

    def _load_features_for_race(self, race_id: str) -> pd.DataFrame | None:
        """Load features for a single race from Parquet data.

        Uses the same feature pipeline as _run_predict.
        """
        try:
            feat_path = self.config.paper_trading_dir / "features" / f"{self.ymd}.parquet"
            if feat_path.exists():
                feat_df = pd.read_parquet(feat_path)
                return feat_df[feat_df["race_id"] == race_id]
        except Exception as e:
            logger.warning("Failed to load features for %s: %s", race_id, e)
        return None

    def _load_odds_for_race(self, race_id: str) -> pd.DataFrame | None:
        """Load odds data for a single race."""
        try:
            odds_path = self.config.paper_trading_dir / "odds" / f"{self.ymd}.parquet"
            if odds_path.exists():
                odds_df = pd.read_parquet(odds_path)
                return odds_df[odds_df["race_id"] == race_id]
        except Exception:
            pass
        return None

    def _compute_current_bankroll(self) -> float:
        """Compute current bankroll from bets.parquet."""
        initial = self.config.initial_bankroll
        bets_df = self._load_bets()
        if bets_df.empty:
            return initial
        if "bankroll_after" in bets_df.columns:
            return float(bets_df["bankroll_after"].iloc[-1])
        return initial - float(bets_df["stake"].sum())

    def _load_bets(self) -> pd.DataFrame:
        """Load bets.parquet or return empty DataFrame."""
        if not self.bets_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.bets_path)
        except Exception:
            return pd.DataFrame()

    def _append_bets(self, new_records: list[dict[str, Any]]) -> None:
        """Append bet records to bets.parquet atomically (D-07)."""
        from paper_trading.reconciler import PaperReconciler

        new_df = pd.DataFrame(new_records)

        if self.bets_path.exists():
            existing = pd.read_parquet(self.bets_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        PaperReconciler._atomic_write_parquet(combined, self.bets_path)

    # ------------------------------------------------------------------
    # Phase 4: Reconciliation
    # ------------------------------------------------------------------

    def _reconcile(self) -> None:
        """Reconcile bets after all races (D-02)."""
        if not self._schedule:
            return

        # Determine last race time
        last_time_str = max(
            (r.get("post_time", "00:00") for r in self._schedule),
            default="00:00",
        )
        try:
            last_race_time = self._parse_post_time(self.target_date, last_time_str)
        except (ValueError, AttributeError):
            logger.warning("Could not parse last race time, proceeding to reconcile")
            last_race_time = datetime.now()

        # Wait until last race + 5 minutes
        wait_target = last_race_time + timedelta(minutes=5)
        if datetime.now() < wait_target:
            if not self._wait_until_with_cancel(wait_target):
                logger.info("Cancelled while waiting for last race to finish")
                return

        # Create reconciler and reconcile
        try:
            from db.everydb2_queries import EveryDB2Queries
            from paper_trading.reconciler import PaperReconciler

            db = EveryDB2Queries(self.config.everydb2_connection_string)
            reconciler = PaperReconciler(
                bets_path=self.bets_path,
                everydb2=db,
                retry_interval=30,
                retry_timeout=300,
            )

            # Reconcile
            reconciler.reconcile(self.target_date)

            # Retry pending
            result = reconciler.retry_pending(self.target_date, last_time_str)

            # Check for pending remain
            pending_count = result.get("n_pending", 0)
            if pending_count > 0:
                logger.warning("%d bets still pending after reconcile (D-02)", pending_count)
                self.errors.append(ExitCode.PENDING_REMAIN)

        except ImportError:
            logger.warning("DB modules not available for reconciliation")
        except Exception as e:
            logger.error("Reconciliation failed: %s", e)
            self.errors.append(ExitCode.GENERAL_ERROR)

    # ------------------------------------------------------------------
    # Phase 5: Reporting
    # ------------------------------------------------------------------

    def _aggregate_and_report(self) -> None:
        """Generate reports via PaperTradingReportAggregator (D-14, D-15)."""
        try:
            from paper_trading.report_aggregator import PaperTradingReportAggregator

            aggregator = PaperTradingReportAggregator(
                bets_path=self.bets_path,
                output_dir=self.config.paper_trading_dir,
                session_manifest=self.session_manifest,
            )
            paths = aggregator.save_outputs(self.target_date)
            logger.info("Reports saved: %s", list(paths.keys()))

        except Exception as e:
            logger.error("Report generation failed (D-16): %s", e)
            # D-16: report failure does NOT roll back bets/reconciliation
            self.errors.append(ExitCode.REPORT_ERROR)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_until_with_cancel(self, target_time: datetime) -> bool:
        """Wait until target_time, checking _cancelled flag every second.

        Returns True if target_time reached, False if cancelled.
        """
        while datetime.now() < target_time:
            if self._cancelled:
                return False
            remaining = (target_time - datetime.now()).total_seconds()
            sleep_time = min(1.0, max(0.0, remaining))
            if sleep_time > 0:
                _time.sleep(sleep_time)
        return True

    def _cross_validate_race(
        self,
        race_id: str,
        bets_df: pd.DataFrame,
        progress: RaceProgress,
    ) -> bool:
        """Cross-validate PREDICTED race against bets.parquet (D-08).

        Returns True if valid, False if mismatch detected.
        """
        if bets_df.empty:
            # No bets file at all -- might be a no_bet race that was
            # incorrectly marked. Check bet_ids.
            return progress.verify_bet_ids_present(race_id, bets_df)

        return progress.verify_bet_ids_present(race_id, bets_df)

    def _determine_exit_code(self) -> ExitCode:
        """Determine final exit code from errors list (D-17, D-18)."""
        if self._cancelled:
            return ExitCode.SIGINT
        return determine_final_exit_code(self.errors)

    @staticmethod
    def _parse_post_time(target_date: date, post_time_str: str) -> datetime:
        """Parse 'HH:MM' or 'HHMM' format post time to datetime."""
        post_time_str = post_time_str.strip()
        if ":" in post_time_str:
            h, m = map(int, post_time_str.split(":"))
        elif len(post_time_str) == 4:
            h = int(post_time_str[:2])
            m = int(post_time_str[2:])
        else:
            raise ValueError(f"Invalid post_time format: '{post_time_str}'")
        return datetime.combine(target_date, time(h, m))

    # ------------------------------------------------------------------
    # Input snapshots (D-09)
    # ------------------------------------------------------------------

    def _save_input_snapshot(
        self,
        race_id: str,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> None:
        """Save input snapshot with replay-ready metadata (D-09).

        Adds _snapshot_hash, _parent_session_id, _source_info columns.
        """
        inputs_dir = self.session_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = inputs_dir / f"{race_id}.parquet"

        # Merge features and odds for the snapshot
        if not odds_df.empty:
            # Combine on race_id + umaban if both exist
            key_cols = ["race_id", "umaban"]
            merge_cols = [
                c for c in key_cols
                if c in features_df.columns and c in odds_df.columns
            ]
            if merge_cols:
                snapshot_df = features_df.merge(
                    odds_df, on=merge_cols, how="left",
                    suffixes=("", "_odds"),
                )
            else:
                snapshot_df = features_df.copy()
        else:
            snapshot_df = features_df.copy()

        # Compute SHA256 hash of the raw data (before adding metadata columns)
        raw_bytes = features_df.to_parquet(index=False)
        snapshot_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Add metadata columns (D-09)
        source_info = json.dumps(
            {
                "race_id": race_id,
                "target_date": str(self.target_date),
                "fetch_timestamp": datetime.now().isoformat(),
                "model_run_id": getattr(self.session_manifest, "model_run_id", ""),
            },
            ensure_ascii=False,
        )

        n_rows = len(snapshot_df)
        snapshot_df["_snapshot_hash"] = [snapshot_hash] * n_rows
        snapshot_df["_parent_session_id"] = [self.session_id] * n_rows
        snapshot_df["_source_info"] = [source_info] * n_rows

        # Atomic write
        from paper_trading.reconciler import PaperReconciler
        PaperReconciler._atomic_write_parquet(snapshot_df, snapshot_path)

        logger.debug("Input snapshot saved: %s (hash=%s...)", race_id, snapshot_hash[:12])

    def _build_race_predictor(self) -> Any:
        """Build RacePredictor from models (delegates to same pattern as run_paper_trading.py)."""
        from backtest.race_predictor import RacePredictor
        return RacePredictor(self.models)
