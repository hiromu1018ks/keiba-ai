"""学習パイプライン v5.4 (§11)

Phase C の全モデルを正しい順序で学習し、TrainedModelsV5 に格納。
MLflow に実験を記録。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from db.parquet_store import ParquetStore
from db.readers import (
    load_entries,
    load_odds_snapshots,
    load_odds_time_series_range,
    load_races,
    load_wide_odds,
    save_features,
)
from domain.models import SubmodelSet, TrainedModelsV5
from utils.timing import TimingContext

if TYPE_CHECKING:
    from db.connection import DatabaseConnection

from features.feature_engine import FeatureEngine
from features.odds_dynamics_features import compute_roi_ema
from models.conformal_ev_model import (
    ConformalEVModel,  # Phase 21: CQR-based EV prediction intervals
)
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
from models.market_model import MarketModel
from models.place_selection_gate import PlaceSelectionGateModel, ensure_place_selection_columns
from models.race_quality_screener import RaceQualityScreener
from models.regime_detector import RegimeDetector
from models.stage1_ability_model import AbilityModel
from models.submodel_manager import SubModelManager
from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
from models.wide_pair_builder import WideJointPairBuilder
from models.wide_two_stage_model import WideTwoStageModel
from models.win_profit_selector import WinProfitSelector
from models.win_segment_calibrator import WinSegmentCalibrator
from models.win_selection_gate import WinSelectionGateModel, ensure_win_selection_columns
from models.win_selection_policy import WinSelectionPolicy
from validation.oof_health_validator import (
    OOFHealthValidator,
    OOF_PREDICTIONS_PROFILE,
    _update_index,
)

logger = logging.getLogger(__name__)

# MLflow pip高速化: pip freeze子プロセス起動を回避 (57.6x高速化, Spike 005 VALIDATED)
_MLFLOW_PIP_REQS: list[str] = ["lightgbm", "scikit-learn", "pandas", "numpy", "joblib"]
WIN_SELECTION_OOF_MAX_TOP1_HIT_RATE = 0.35
WIN_SELECTION_OOF_MAX_TOP1_ROI = 2.0
WIN_SELECTION_OOF_MIN_GUARD_RACES = 30


def _valid_ev_band_scales(scales: dict[str, float] | None) -> bool:
    if not scales:
        return False
    values = np.array([float(v) for v in scales.values()], dtype=float)
    return bool(np.isfinite(values).all() and not np.allclose(values, 0.0))


def _get_num_threads(parallel_workers: int = 1) -> int:
    """並列ワーカー数に応じて最適なスレッド数を返す。"""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count // (parallel_workers + 1))


def _prepare_oof_artifact(df: pd.DataFrame) -> pd.DataFrame:
    """OOF成果物をfeature cacheと区別できる形に整える。"""
    oof_df = df.copy()
    oof_df["is_oof"] = True
    oof_df["oof_artifact_version"] = 1
    oof_df["oof_row_id"] = np.arange(len(oof_df), dtype=np.int64)
    return oof_df


def _normalize_race_id_for_artifact(series: pd.Series) -> pd.Series:
    """Parquet artifactでrace_idをfloat化させないため文字列に正規化する。"""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").round()
        return numeric.astype("Int64").astype(str).replace("<NA>", pd.NA)
    text = series.astype("string")
    return text.str.replace(r"\.0$", "", regex=True)


def _prepare_win_selection_oof_artifact(df: pd.DataFrame) -> pd.DataFrame:
    """最終単勝選定の検証に必要なOOF列だけを保存する。"""
    wanted_cols = [
        "race_id",
        "race_date",
        "surface",
        "umaban",
        "kakuteijyuni",
        "tanodds",
        "confirmed_odds",
        "p_win_oof",
        "p_win_pred",
        "p_win_corrected",
        "p_win_combined",
        "p_win_final",
        "p_win_final_oof",
        "win_selection_prob",
        "win_selection_ev",
        "win_selection_edge",
        "win_market_logit_edge",
        "win_market_value_ratio",
        "win_market_selection_score",
        "win_profit_score",
        "win_profit_selector_pass",
        "win_profit_rank",
        "win_profit_stake_scale",
        "win_profit_reason",
        "win_late_odds_drop_z",
        "win_late_odds_drop_weight",
        "win_ev_tail_pressure",
        "win_ev_tail_penalty_weight",
        "win_log_odds",
        "win_log_odds_penalty",
        "win_model_prob_rank",
        "win_prob_rank_bonus",
        "odds_drop_rate_30_10",
        "odds_drop_rate_60_10",
        "odds_velocity",
        "popularity_change_30_10",
        "win_gate_score",
        "win_gate_pass",
        "win_selection_oof_fold",
        "is_win",
        "win_return",
        "win_return_unit",
    ]
    cols = [col for col in wanted_cols if col in df.columns]
    oof_df = df.loc[:, cols].copy()
    if "race_id" in oof_df.columns:
        oof_df["race_id"] = _normalize_race_id_for_artifact(oof_df["race_id"])
    if "umaban" in oof_df.columns:
        oof_df["umaban"] = pd.to_numeric(oof_df["umaban"], errors="coerce").astype("Int64")
    if "p_win_final_oof" not in oof_df.columns:
        if "p_win_final" in oof_df.columns:
            oof_df["p_win_final_oof"] = pd.to_numeric(oof_df["p_win_final"], errors="coerce")
        elif "win_selection_prob" in oof_df.columns:
            oof_df["p_win_final_oof"] = pd.to_numeric(
                oof_df["win_selection_prob"],
                errors="coerce",
            )
        elif "p_win_oof" in oof_df.columns:
            oof_df["p_win_final_oof"] = pd.to_numeric(oof_df["p_win_oof"], errors="coerce")
    if "kakuteijyuni" in oof_df.columns and "is_win" not in oof_df.columns:
        oof_df["is_win"] = pd.to_numeric(oof_df["kakuteijyuni"], errors="coerce").eq(1)
    if "win_return" not in oof_df.columns and "is_win" in oof_df.columns:
        odds_col = "confirmed_odds" if "confirmed_odds" in oof_df.columns else "tanodds"
        if odds_col in oof_df.columns:
            odds = pd.to_numeric(oof_df[odds_col], errors="coerce").fillna(0.0)
            if odds.dropna().quantile(0.75) > 100.0:
                odds = odds / 100.0
            oof_df["win_return_unit"] = odds.where(oof_df["is_win"], 0.0)
            oof_df["win_return"] = oof_df["win_return_unit"] * 100.0
    oof_df["is_oof"] = True
    oof_df["oof_artifact_version"] = 4
    oof_df["oof_row_id"] = np.arange(len(oof_df), dtype=np.int64)
    return oof_df


def _win_selection_oof_return_unit(df: pd.DataFrame) -> pd.Series:
    """Return realized win payout per 100-yen stake for OOF ROI checks."""
    if "win_return_unit" in df.columns:
        unit_return = pd.to_numeric(df["win_return_unit"], errors="coerce")
        if unit_return.notna().any():
            return unit_return.clip(lower=0.0).fillna(0.0)
    if "win_return" in df.columns:
        yen_return = pd.to_numeric(df["win_return"], errors="coerce")
        if yen_return.notna().any():
            return (yen_return.clip(lower=0.0).fillna(0.0) / 100.0)

    odds_col = "confirmed_odds" if "confirmed_odds" in df.columns else "tanodds"
    if odds_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    odds = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0)
    if odds.dropna().quantile(0.75) > 100.0:
        odds = odds / 100.0
    hit = pd.to_numeric(df["kakuteijyuni"], errors="coerce").eq(1)
    return odds.where(hit, 0.0).clip(lower=0.0)


def _walk_forward_race_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    min_train_races: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """race_id単位のexpanding walk-forward splitを返す。"""
    if df.empty or "race_id" not in df.columns:
        return []
    race_cols = ["race_id"]
    if "race_date" in df.columns:
        race_cols.append("race_date")
    race_order = df[race_cols].drop_duplicates()
    if "race_date" in race_order.columns:
        race_order = race_order.sort_values(["race_date", "race_id"])
    else:
        race_order = race_order.sort_values("race_id")
    race_order = race_order.reset_index(drop=True)
    n_races = len(race_order)
    if n_races <= min_train_races + 1:
        return []

    effective_splits = min(n_splits, max(2, n_races // max(1, min_train_races)))
    if n_races <= effective_splits:
        return []

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    for train_race_idx, val_race_idx in splitter.split(race_order):
        if len(train_race_idx) < min_train_races or len(val_race_idx) == 0:
            continue
        train_races = set(race_order.iloc[train_race_idx]["race_id"])
        val_races = set(race_order.iloc[val_race_idx]["race_id"])
        train_idx = df.index[df["race_id"].isin(train_races)].to_numpy()
        val_idx = df.index[df["race_id"].isin(val_races)].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


def _validate_win_selection_oof_health(
    df: pd.DataFrame,
    *,
    context: str,
    score_col: str = "win_market_selection_score",
    max_top1_hit_rate: float = WIN_SELECTION_OOF_MAX_TOP1_HIT_RATE,
    max_top1_roi: float = WIN_SELECTION_OOF_MAX_TOP1_ROI,
    min_guard_races: int = WIN_SELECTION_OOF_MIN_GUARD_RACES,
) -> dict[str, float | int | str]:
    """OOF選定artifactがin-sample級に異常でないことを検証する。"""
    if df.empty:
        return {"context": context, "n_races": 0, "top1_hit_rate": 0.0, "top1_roi": 0.0}
    required = {"race_id", "kakuteijyuni", score_col}
    missing = required - set(df.columns)
    if missing:
        return {
            "context": context,
            "n_races": 0,
            "top1_hit_rate": 0.0,
            "top1_roi": 0.0,
            "missing": ",".join(sorted(missing)),
        }

    has_return_col = bool({"win_return_unit", "win_return"} & set(df.columns))
    has_odds_col = bool({"confirmed_odds", "tanodds"} & set(df.columns))
    if not has_return_col and not has_odds_col:
        return {
            "context": context,
            "n_races": 0,
            "top1_hit_rate": 0.0,
            "top1_roi": 0.0,
            "missing": "win_return_unit|win_return|confirmed_odds|tanodds",
        }

    scored = df.copy()
    scored["_oof_score"] = pd.to_numeric(scored[score_col], errors="coerce").fillna(float("-inf"))
    top1 = (
        scored.sort_values(["race_id", "_oof_score"], ascending=[True, False])
        .groupby("race_id", observed=True)
        .head(1)
    )
    n_races = int(top1["race_id"].nunique())
    hit = pd.to_numeric(top1["kakuteijyuni"], errors="coerce").eq(1)
    unit_return = _win_selection_oof_return_unit(top1)
    top1_hit_rate = float(hit.mean()) if n_races > 0 else 0.0
    top1_roi = float(unit_return.sum() / max(1, n_races))
    metrics: dict[str, float | int | str] = {
        "context": context,
        "n_races": n_races,
        "top1_hit_rate": top1_hit_rate,
        "top1_roi": top1_roi,
    }
    if n_races >= min_guard_races and (
        top1_hit_rate > max_top1_hit_rate or top1_roi > max_top1_roi
    ):
        raise ValueError(
            f"Win selection OOF leakage guard failed ({context}): "
            f"top1_hit_rate={top1_hit_rate:.3f}, top1_roi={top1_roi:.3f}, "
            f"n_races={n_races}. Refusing to train/save selection models on this artifact."
        )
    return metrics


def _unpack_train_submodel_result(
    result: tuple[SubmodelSet, pd.DataFrame] | tuple[SubmodelSet, pd.DataFrame, pd.DataFrame],
) -> tuple[SubmodelSet, pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible unpacking for tests/custom overrides."""
    if len(result) == 2:
        sub, sub_oof = result
        return sub, sub_oof, pd.DataFrame()
    return result


class TrainingPipelineV5:
    """学習パイプライン (§11)

    データロード → 特徴量生成 → モデル学習 → MLflow記録
    """

    def __init__(
        self,
        store: ParquetStore | None = None,
        db: DatabaseConnection | None = None,
        settings_path: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.store = store or ParquetStore()
        self.db = db  # kept for etl_to_parquet if needed, can be None
        self.feature_engine = FeatureEngine()
        self.submodel_mgr = SubModelManager()
        self.model_dir = model_dir or Path("data/models")

    @staticmethod
    def _to_yyyymmdd(date_str: str) -> str:
        """YYYY-MM-DD → YYYYMMDD"""
        return date_str.replace("-", "")

    def run(
        self,
        train_start: str,
        train_end: str,
        *,
        use_ensemble: bool = False,
        betting_target: str = "place",
    ) -> TrainedModelsV5:
        """全モデルを学習し TrainedModelsV5 を返す

        Args:
            train_start: 学習開始日 (YYYY-MM-DD)
            train_end: 学習終了日 (YYYY-MM-DD)
            use_ensemble: アンサンブル (B1) を有効化
            betting_target: "win"/"place"/"wide" — 不要なモデルの学習をスキップ

        Returns:
            学習済みモデルのコンテナ
        """
        self.use_ensemble = use_ensemble
        self._betting_target = betting_target
        start = self._to_yyyymmdd(train_start)
        end = self._to_yyyymmdd(train_end)

        # 1. データロード
        logger.info(f"Loading data: {train_start} ~ {train_end}")
        race_df = load_races(self.store, start, end)
        entry_df = load_entries(self.store, start, end)
        # datakubun違い ('0'確定 vs 'B'速報) で同一 (race_id, umaban) が
        # 複数行存在する場合があるため、確定データ('0') を優先して重複除去
        if "datakubun" in entry_df.columns:
            entry_df = entry_df.sort_values("datakubun", na_position="last")
        entry_df = entry_df.drop_duplicates(subset=["race_id", "umaban"], keep="first")
        odds_df = load_odds_snapshots(self.store, start, end)  # フォールバック用

        # JRA 学習対象に絞ってからオッズ抽出/特徴量生成へ進む。
        if "jyocd" in race_df.columns:
            jyocd_int = pd.to_numeric(race_df["jyocd"], errors="coerce")
            jra_race_ids = race_df.loc[jyocd_int.between(1, 10), "race_id"].drop_duplicates()
            race_df = race_df[race_df["race_id"].isin(jra_race_ids)].copy()
            entry_df = entry_df[entry_df["race_id"].isin(jra_race_ids)].copy()
            odds_df = odds_df[odds_df["race_id"].isin(jra_race_ids)].copy()

        # NEW: _train_submodel 内で HorseHistoryFeatures が使用するため保存
        self._race_df = race_df
        self._entry_df = entry_df
        # オッズ時系列データ — Stage2 (WinTwoStageModel) のオッズ動的特徴量に必須
        odds_ts_df = load_odds_time_series_range(self.store, start, end)
        if not odds_ts_df.empty:
            odds_ts_df = odds_ts_df[odds_ts_df["race_id"].isin(race_df["race_id"])].copy()

        # 発走5分前オッズの抽出 (本番と同じ時点のデータを使用)
        # 年ごとに処理してメモリ使用量を抑制
        if not odds_ts_df.empty and "hassotime" in race_df.columns:
            from db.odds_extractor import extract_pre_post_odds

            start_year = int(start[:4])
            end_year = int(end[:4])
            pre_post_frames: list[pd.DataFrame] = []
            for year in range(start_year, end_year + 1):
                year_ts = odds_ts_df[odds_ts_df["year"] == year]
                if year_ts.empty:
                    continue
                pp = extract_pre_post_odds(year_ts, race_df, minutes_before=5)
                if not pp.empty:
                    pre_post_frames.append(pp)
            if pre_post_frames:
                odds_df = pd.concat(pre_post_frames, ignore_index=True)
                logger.info(
                    "Using pre-post odds (5min before): %d rows",
                    len(odds_df),
                )
            else:
                logger.warning("extract_pre_post_odds empty, using snapshots")
        else:
            logger.warning("No time-series data or hassotime, using snapshots")

        # 2. 特徴量生成
        # 学習パイプラインでは kakuteijyuni (ターゲット変数) と confirmed_odds (EV計算)
        # を保持する。build_all() の SAFE-01 はこれらを除外してドロップする。
        logger.info("Building features")
        feat_df = self.feature_engine.build_all(
            race_df,
            entry_df,
            odds_df,
            odds_ts_df=odds_ts_df,
            store=self.store,
            preserve_columns=["kakuteijyuni", "confirmed_odds"],
        )
        feat_df = self.submodel_mgr.add_distance_band_features(feat_df)

        # JRAフィルタ: NARレース (jyocd 30以上) を除外
        if "jyocd" in feat_df.columns:
            jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
            before = len(feat_df)
            feat_df = feat_df[jyocd_int.between(1, 10)]
            after = len(feat_df)
            if after < before:
                logger.info(
                    "JRA filter: %d -> %d entries (removed %d NAR)",
                    before,
                    after,
                    before - after,
                )

        # 2b. ワイドペア専用オッズを pivot して特徴量に merge
        # build_all() 内の市場クロス特徴量は別経路で計算済み。wide_odds_* は
        # WideJointPairBuilder だけが参照するため、単勝/複勝学習では不要。
        if self._betting_target == "wide":
            wide_odds_df = load_wide_odds(self.store, start, end)
            if wide_odds_df is not None and not wide_odds_df.empty:
                wide_pivot = wide_odds_df.pivot_table(
                    index="race_id",
                    columns="kumi",
                    values="oddslow",
                )
                # ゼロ埋め解除: kumi "0102" → "1_2" (WideJointPairBuilder の lookup 形式に合わせる)
                new_cols = []
                for kumi in wide_pivot.columns:
                    lo = int(str(kumi)[:2])
                    hi = int(str(kumi)[2:])
                    new_cols.append(f"wide_odds_{lo}_{hi}")
                wide_pivot.columns = new_cols
                wide_pivot = wide_pivot.reset_index()
                feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")
        else:
            logger.info(
                "Skipping wide odds pivot for betting_target=%s",
                self._betting_target,
            )

        # 3. 各 surface ごとに学習 (parallel)
        models: dict[str, SubmodelSet] = {}
        oof_dfs: list[pd.DataFrame] = []
        win_selection_oof_dfs: list[pd.DataFrame] = []
        surfaces_to_train: list[tuple[str, pd.DataFrame]] = []
        for surface in ["turf", "dirt"]:
            subset_df = feat_df[feat_df["surface"] == surface].copy()
            if len(subset_df) < 1000:
                logger.warning(f"Skipping {surface}: insufficient data ({len(subset_df)})")
                continue
            surfaces_to_train.append((surface, subset_df))

        if len(surfaces_to_train) == 1:
            # Single surface — no parallelism needed
            surface, subset_df = surfaces_to_train[0]
            sub, sub_oof, sub_win_selection_oof = _unpack_train_submodel_result(
                self._train_submodel(
                    subset_df,
                    num_threads=_get_num_threads(1),
                    use_ensemble=self.use_ensemble,
                    betting_target=self._betting_target,
                )
            )
            models[surface] = sub
            oof_dfs.append(sub_oof)
            if not sub_win_selection_oof.empty:
                win_selection_oof_dfs.append(sub_win_selection_oof)
            logger.info(f"Trained {surface} submodel")
        elif len(surfaces_to_train) >= 2:
            # Sequential training to avoid segfault from LightGBM/XGBoost
            # native library conflicts under ThreadPoolExecutor
            # Use same num_threads as parallel mode to maintain model parity
            num_threads = _get_num_threads(2)
            for surface, subset_df in surfaces_to_train:
                sub, sub_oof, sub_win_selection_oof = _unpack_train_submodel_result(
                    self._train_submodel(
                        subset_df,
                        num_threads=num_threads,
                        use_ensemble=self.use_ensemble,
                        betting_target=self._betting_target,
                    )
                )
                models[surface] = sub
                oof_dfs.append(sub_oof)
                if not sub_win_selection_oof.empty:
                    win_selection_oof_dfs.append(sub_win_selection_oof)
                logger.info(f"Trained {surface} submodel (sequential)")

        # 3b. 全サーフェスの完全特徴量を保存 (feature audit 用)
        # _train_submodel で追加された horse_history / pace / course / sire /
        # interaction / jockey_context / trainer_context 等を含む
        if oof_dfs:
            full_features_df = pd.concat(oof_dfs, ignore_index=True)
            if full_features_df.empty:
                logger.warning(
                    "Skipping feature/OOF artifact save because no OOF rows were produced"
                )
            else:
                save_features(self.store, full_features_df)
                logger.info(
                    "Saved full feature set: %d rows, %d cols -> "
                    "data/features/horse_features.parquet",
                    len(full_features_df),
                    len(full_features_df.columns),
                )

            # 3c. OOF予測Parquet保存 — producer-side validation (D-13)
            if not full_features_df.empty and "race_date" in full_features_df.columns:
                oof_path = Path("data/oof/oof_predictions.parquet")
                oof_path.parent.mkdir(parents=True, exist_ok=True)

                # Add OOF metadata columns before validation
                full_features_df["is_oof"] = True
                full_features_df["oof_artifact_version"] = 1

                # D-13: validate before save
                train_date_range = (
                    str(full_features_df["race_date"].min()),
                    str(full_features_df["race_date"].max()),
                )
                oof_validator = OOFHealthValidator()
                oof_result = oof_validator.validate(
                    full_features_df,
                    OOF_PREDICTIONS_PROFILE,
                    train_date_range=train_date_range,
                    expected_row_count=len(full_features_df),
                )
                if oof_result["status"] != "PASS":
                    raise ValueError(
                        f"OOF health check failed for oof_predictions: "
                        f"{oof_result['failures']}"
                    )

                full_features_df.to_parquet(oof_path, index=False)

                # D-08: health manifest
                artifact_hash = hashlib.sha256(oof_path.read_bytes()).hexdigest()
                manifest = oof_validator.generate_manifest(
                    full_features_df,
                    OOF_PREDICTIONS_PROFILE,
                    artifact_hash,
                    train_date_range=train_date_range,
                )
                manifest_path = Path("data/oof/manifests/oof_predictions.health.json")
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _update_index("oof_predictions", str(manifest_path), artifact_hash)
                logger.info(
                    "Saved OOF predictions: %d rows -> %s (validated, manifest=%s)",
                    len(full_features_df),
                    oof_path,
                    manifest_path,
                )
            else:
                logger.info(
                    "Skipping OOF predictions save: empty df or no race_date column"
                )

        if win_selection_oof_dfs:
            win_oof_path = Path("data/oof/win_selection_oof.parquet")
            win_oof_path.parent.mkdir(parents=True, exist_ok=True)
            win_oof_df = _prepare_win_selection_oof_artifact(
                pd.concat(win_selection_oof_dfs, ignore_index=True)
            )
            metrics = _validate_win_selection_oof_health(
                win_oof_df,
                context="combined_win_selection_oof_save",
            )
            logger.info("Win selection OOF guard passed: %s", metrics)
            win_oof_df.to_parquet(win_oof_path, index=False)
            logger.info(
                "Saved win selection OOF predictions: %d rows, %d cols -> %s",
                len(win_oof_df),
                len(win_oof_df.columns),
                win_oof_path,
            )

        # 4. feat_df の object 数値列を float64 に統一
        for col in feat_df.columns:
            if feat_df[col].dtype == object:
                try:
                    feat_df[col] = feat_df[col].astype(float)
                except (ValueError, TypeError):
                    pass

        # 4b. Market Model の出力を feat_df にも反映
        # (サブモデル学習で market model が作成した予測誤差を feat_df にマージ)
        for surface, sub in models.items():
            mask = feat_df["surface"] == surface
            if mask.any():
                result_df = sub.market.predict_and_calc_error(feat_df.loc[mask].copy())
                # 新規列のみ feat_df に反映 (列数不一致を避ける)
                new_cols = [c for c in result_df.columns if c not in feat_df.columns]
                for c in new_cols:
                    feat_df[c] = np.nan
                feat_df.loc[mask, new_cols] = result_df[new_cols].astype(float).values

        # 5. レースレベル特徴量を構築
        required_cols = ["signed_log_error_win", "abs_log_error_win"]
        missing = [c for c in required_cols if c not in feat_df.columns]
        if missing:
            logger.warning("Market model columns missing: %s — filling with 0.0", missing)
            for c in missing:
                feat_df[c] = 0.0
        with TimingContext("race_level_features"):
            race_feat_df = self._build_race_level_features(feat_df)

        # 5. RaceQualityScreener
        with TimingContext("quality_screener"):
            quality_screen = RaceQualityScreener()
            quality_screen.train(race_feat_df, num_threads=_get_num_threads(1))
            quality_screen.calibrate_threshold(race_feat_df, target_investment_rate=0.40)
        logger.info("Trained RaceQualityScreener")

        # 6. RegimeDetector
        with TimingContext("regime_detector"):
            regime_stats_df = self._build_regime_stats(race_feat_df, feat_df)
            regime_det = RegimeDetector()
            regime_det.train(regime_stats_df, num_threads=_get_num_threads(1))
        logger.info("Trained RegimeDetector")

        # 7. MLflow 記録
        self._log_to_mlflow(models, quality_screen, regime_det, train_start, train_end)

        return TrainedModelsV5(
            submodels=models,
            quality_screener=quality_screen,
            regime_detector=regime_det,
            train_period=(train_start, train_end),
        )

    @staticmethod
    def _compute_ev_threshold(
        df_oof: pd.DataFrame,
        surface: str,
        fallback: float = 1.0,
    ) -> float:
        """D-01/D-02: OOF positive-edge winnersの25th percentileから動的閾値を計算.

        positive-edge winners = kakuteijyuni==1 AND win_selection_edge > 0
        """
        surf_df = df_oof[df_oof["surface"] == surface]
        if "EV_lower_win_corrected" not in surf_df.columns:
            return fallback
        winners = surf_df[(surf_df["kakuteijyuni"] == 1) & (surf_df["win_selection_edge"] > 0)]
        ev_lower_values = pd.to_numeric(winners["EV_lower_win_corrected"], errors="coerce").dropna()
        if len(ev_lower_values) < 30:
            logger.info(
                "EV threshold for %s: too few positive-edge winners (%d), using fallback %.2f",
                surface,
                len(ev_lower_values),
                fallback,
            )
            return fallback
        if TrainingPipelineV5._ev_lower_distribution_degenerate(ev_lower_values):
            calibrated = pd.to_numeric(
                winners.get("ev_win_calibrated", pd.Series(np.nan, index=winners.index)),
                errors="coerce",
            )
            blended_source = pd.DataFrame(
                {
                    "ev_lower": pd.to_numeric(
                        winners["EV_lower_win_corrected"],
                        errors="coerce",
                    ),
                    "ev_calibrated": calibrated,
                }
            ).dropna()
            if len(blended_source) >= 30 and float(blended_source["ev_calibrated"].std()) > 1e-6:
                calibrated_cap = float(blended_source["ev_calibrated"].quantile(0.90))
                ev_lower_values = 0.70 * blended_source["ev_lower"] + 0.30 * blended_source[
                    "ev_calibrated"
                ].clip(upper=calibrated_cap)
                logger.info(
                    "EV threshold for %s: EV_lower distribution degenerate; "
                    "using blended CQR/calibrated EV distribution",
                    surface,
                )
            # Second fallback: use full OOF population if still degenerate
            if TrainingPipelineV5._ev_lower_distribution_degenerate(ev_lower_values):
                all_ev_lower = pd.to_numeric(
                    surf_df["EV_lower_win_corrected"], errors="coerce"
                ).dropna()
                if len(all_ev_lower) >= 30:
                    ev_lower_values = all_ev_lower
                    logger.info(
                        "EV threshold for %s: blended still degenerate; "
                        "falling back to full OOF population (%d samples)",
                        surface,
                        len(ev_lower_values),
                    )
        threshold = float(ev_lower_values.quantile(0.25))
        logger.info(
            "EV threshold for %s: %.4f (from %d positive-edge winners, q25=%.4f q50=%.4f q75=%.4f)",
            surface,
            threshold,
            len(ev_lower_values),
            float(ev_lower_values.quantile(0.25)),
            float(ev_lower_values.quantile(0.50)),
            float(ev_lower_values.quantile(0.75)),
        )
        return threshold

    @staticmethod
    def _ev_lower_distribution_degenerate(values: pd.Series) -> bool:
        """EV_lower の分布が定数化しているか判定する."""
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if len(clean) < 30:
            return False
        q25 = float(clean.quantile(0.25))
        q50 = float(clean.quantile(0.50))
        q75 = float(clean.quantile(0.75))
        scale = max(abs(q50), 1.0)
        unique_ratio = clean.round(6).nunique() / len(clean)
        return bool((q75 - q25) <= scale * 1e-4 or unique_ratio < 0.02)

    @staticmethod
    def _shrunken_group_mean(
        values: pd.Series,
        groups: pd.Series,
        *,
        prior: float,
        prior_weight: float,
    ) -> pd.Series:
        """グループ平均を全体平均へ縮約して少サンプル帯の過大振れを抑える."""
        work = pd.DataFrame(
            {
                "value": pd.to_numeric(values, errors="coerce"),
                "group": groups,
            },
            index=values.index,
        )
        valid = work["value"].notna() & work["group"].notna()
        if int(valid.sum()) == 0:
            return pd.Series(prior, index=values.index, dtype=float)

        stats = work.loc[valid].groupby("group", observed=True)["value"].agg(["sum", "count"])
        means = (stats["sum"] + prior_weight * prior) / (stats["count"] + prior_weight)
        return work["group"].map(means).fillna(prior).astype(float)

    @staticmethod
    def _build_cqr_actual_ev_target(df: pd.DataFrame) -> pd.Series:
        """CQR用の連続的な実現EV教師信号を作る.

        point-wise の odds * I(win) はゼロが支配的で、単純なdecile平均は
        逆に定数化しやすい。EV階層・オッズ帯の縮約平均に、winsorizeした
        実現払戻を少量混ぜて、期待値の滑らかさと実現ノイズの両方を残す。
        """
        from betting.odds_band_filter import OddsBandFilter

        if "confirmed_odds" not in df.columns or "kakuteijyuni" not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)

        odds = pd.to_numeric(df["confirmed_odds"], errors="coerce").fillna(0.0)
        point_ev = odds * (df["kakuteijyuni"] == 1).astype(float)
        global_mean = float(point_ev.mean()) if len(point_ev) else 0.0
        if not np.isfinite(global_mean):
            global_mean = 0.0

        ev_cal = pd.to_numeric(
            df.get("ev_win_calibrated", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        try:
            ev_bins = pd.qcut(
                ev_cal.rank(method="first"),
                q=10,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            ev_bins = pd.Series(0, index=df.index, dtype=int)
        ev_bin_expected = TrainingPipelineV5._shrunken_group_mean(
            point_ev,
            pd.Series(ev_bins, index=df.index),
            prior=global_mean,
            prior_weight=50.0,
        )

        odds_for_band = pd.to_numeric(
            df.get("odds", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        odds_band = pd.Series("missing", index=df.index, dtype=object)
        for (lo, hi), band_name in zip(OddsBandFilter.BANDS, OddsBandFilter.BAND_NAMES):
            mask = (odds_for_band >= lo) & (odds_for_band < hi)
            odds_band.loc[mask] = band_name
        odds_band_expected = TrainingPipelineV5._shrunken_group_mean(
            point_ev,
            odds_band,
            prior=global_mean,
            prior_weight=75.0,
        )

        positive_point = point_ev[point_ev > 0]
        if len(positive_point) > 0:
            point_cap = max(float(positive_point.quantile(0.995)), global_mean)
        else:
            point_cap = global_mean
        point_component = point_ev.clip(upper=point_cap)

        target = (0.65 * ev_bin_expected) + (0.25 * odds_band_expected) + (0.10 * point_component)
        return target.clip(lower=0.0).fillna(global_mean).astype(float)

    def _train_submodel(
        self,
        df: pd.DataFrame,
        *,
        num_threads: int = 0,
        use_ensemble: bool = False,
        betting_target: str = "place",
    ) -> tuple[SubmodelSet, pd.DataFrame, pd.DataFrame]:
        """単一 surface のサブモデル群を学習

        Returns:
            (SubmodelSet, df_oof, win_selection_oof) のタプル。
            df_oof は全特徴量を含むDataFrame。
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        surface = df["surface"].iloc[0] if "surface" in df.columns else "unknown"

        # NEW: 馬過去成績特徴量
        from features.horse_history_features import HorseHistoryFeatures

        with TimingContext(f"{surface}/horse_history"):
            hist = HorseHistoryFeatures(store=self.store, n_past=5)
            hist_df = hist.compute(self._race_df, self._entry_df, df["race_id"].unique())
            df = df.merge(hist_df, on=["race_id", "umaban"], how="left")
        with TimingContext(f"{surface}/add_race_transforms"):
            df = HorseHistoryFeatures.add_race_transforms(df)

        # Group C: ペース適性特徴量 (HorseHistoryFeatures の直後)
        from features.pace_aptitude_features import PaceAptitudeFeatures

        with TimingContext(f"{surface}/pace_aptitude"):
            pace_feat = PaceAptitudeFeatures(store=self.store)
            pace_df = pace_feat.compute_batch(df)
            # df には既に特徴量列が含まれている可能性があるため削除
            _pace_drop_cols = [
                "pace_aptitude",
                "front_pace_wr",
                "closing_pace_wr",
                "pace_corner_stability",
                "pace_closing_power",
                "pace_position_consistency",
            ]
            for col in _pace_drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            if not pace_df.empty:
                pace_merge_cols = [
                    c
                    for c in [
                        "kettonum",
                        "race_id",
                        "pace_aptitude",
                        "front_pace_wr",
                        "closing_pace_wr",
                        "pace_corner_stability",
                        "pace_closing_power",
                        "pace_position_consistency",
                    ]
                    if c in pace_df.columns
                ]
                df = df.merge(
                    pace_df[pace_merge_cols],
                    on=["kettonum", "race_id"],
                    how="left",
                )
            else:
                # 空の場合は結果列を NaN で追加
                df["pace_aptitude"] = np.nan
                df["front_pace_wr"] = np.nan
                df["closing_pace_wr"] = np.nan
                df["pace_corner_stability"] = np.nan
                df["pace_closing_power"] = np.nan
                df["pace_position_consistency"] = np.nan

        # Group D: コース別適性特徴量 (pace_aptitude の直後)
        from features.course_features import CourseFeatures

        with TimingContext(f"{surface}/course_features"):
            course_feat = CourseFeatures(store=self.store)
            course_df = course_feat.compute_batch(df)
            # df には既に特徴量列が含まれている可能性があるため削除
            for col in ["course_wr", "course_distance_wr"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            if not course_df.empty:
                df = df.merge(
                    course_df[["kettonum", "race_id", "course_wr", "course_distance_wr"]],
                    on=["kettonum", "race_id"],
                    how="left",
                )
            else:
                # 空の場合は結果列を NaN で追加
                df["course_wr"] = np.nan
                df["course_distance_wr"] = np.nan

        # 種牡馬産駒特徴量の追加 (ベクトル化)
        from db.readers import load_horses, load_sire_stats
        from features.sire_features import SireFeatures

        with TimingContext(f"{surface}/sire_features"):
            sire_stats = load_sire_stats(self.store)
            if not sire_stats.empty:
                horses_df = load_horses(self.store)
                sire_feat = SireFeatures(sire_stats)
                # entry_df に sire_id / bms_id 列を追加
                sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
                df["sire_id"] = df["kettonum"].map(sire_map)
                bms_source_col = (
                    "ketto3infohansyokunum5"
                    if "ketto3infohansyokunum5" in horses_df.columns
                    else "ketto3infohansyokunum3"
                )
                bms_map = horses_df.set_index("kettonum")[bms_source_col]
                df["bms_id"] = df["kettonum"].map(bms_map)
                # ベクトル化一括計算
                sire_result = sire_feat.compute_batch(df)
                # モデルで使用する7列のみを反映 (sire_place_rate は未使用のため除外)
                _sire_cols_needed = {
                    "sire_wr",
                    "sire_surface_wr",
                    "sire_distance_wr",
                    "sire_prize_avg",
                    "bms_wr",
                    "bms_distance_wr",
                    "bms_surface_wr",
                    "bms_has_history",
                    "bms_starts_log",
                    "bms_surface_starts_log",
                    "bms_distance_starts_log",
                }
                for col in _sire_cols_needed:
                    if col in sire_result.columns:
                        df[col] = sire_result[col].values

        # Group B-2: 繁殖牝馬産駒特徴量 (sire features の後)
        from features.dam_pedigree_features import FEATURE_COLS as DAM_PED_FEATURE_COLS
        from features.dam_pedigree_features import DamPedigreeFeatures

        with TimingContext(f"{surface}/dam_pedigree"):
            dam_ped = DamPedigreeFeatures(self.store)
            dam_ped_df = dam_ped.compute(df)
            _dam_drop_cols = [c for c in DAM_PED_FEATURE_COLS if c in df.columns]
            if _dam_drop_cols:
                df.drop(columns=_dam_drop_cols, inplace=True)
            if not dam_ped_df.empty:
                df = df.merge(dam_ped_df, on=["race_id", "umaban"], how="left")
            else:
                for col in DAM_PED_FEATURE_COLS:
                    df[col] = np.nan

        # Group B-3: コースレコード特徴量
        from features.record_features import FEATURE_COLS as RECORD_FEATURE_COLS
        from features.record_features import RecordFeatures

        with TimingContext(f"{surface}/record_features"):
            record_feat = RecordFeatures(self.store)
            record_df = record_feat.compute(df)
            assert record_df.empty or record_df["race_id"].is_unique, (
                f"record_df has duplicate race_ids: {record_df['race_id'].duplicated().sum()}"
            )
            _record_drop_cols = [c for c in RECORD_FEATURE_COLS if c in df.columns]
            if _record_drop_cols:
                df.drop(columns=_record_drop_cols, inplace=True)
            if not record_df.empty:
                df = df.merge(record_df, on=["race_id"], how="left")
            else:
                for col in RECORD_FEATURE_COLS:
                    df[col] = np.nan

        # Group E: 交互作用特徴量 (HorseHistoryFeatures 後に実行 — kyakusitu_cd が必要)
        from features.interaction_features import compute_interaction_features

        with TimingContext(f"{surface}/interaction"):
            df = compute_interaction_features(df)

        # Group F: n_mining予想特徴量 (interaction features の後)
        from features.mining_features import FEATURE_COLS as MINING_FEATURE_COLS
        from features.mining_features import MiningFeatures

        with TimingContext(f"{surface}/mining_features"):
            mining_feat = MiningFeatures(self.store)
            mining_df = mining_feat.compute(df)
            _mining_drop_cols = [c for c in MINING_FEATURE_COLS if c in df.columns]
            if _mining_drop_cols:
                df.drop(columns=_mining_drop_cols, inplace=True)
            if not mining_df.empty:
                df = df.merge(mining_df, on=["race_id", "umaban"], how="left")
            else:
                for col in MINING_FEATURE_COLS:
                    df[col] = np.nan

        # Group G: レース内相対比較特徴量 (all per-horse features の後)
        from features.relative_features import compute_relative_features

        with TimingContext(f"{surface}/relative_features"):
            df = compute_relative_features(df)

        # 1. Market Model (正規化差分 log_error のみ出力)
        # object型の数値列 (pd.NA含む) → float64 (2回目のsurface処理でpd.NAが混入するため)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    pass

        with TimingContext(f"{surface}/market_model"):
            # 時間ベース分割の前提: race_date でソート
            df = df.sort_values("race_date").reset_index(drop=True)
            market = MarketModel()
            market.train(df, num_threads=num_threads)
            df = market.predict_and_calc_error(df)
            # OOF予測で学習データのリークを除去
            # 注: _train_submodel は学習データのみを受け取る (呼び出し側で既にフィルタ済み)。
            #      よって df 全体に OOF を適用してよい。test_start_date の分割は不要。
            df = market.predict_oof(df, n_splits=5, num_threads=num_threads)

        # nullable int (Int64) → float64 (market model が Int64 を追加するため)
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)

        # 2. Stage1: OOF predictions (リーク防止)
        with TimingContext(f"{surface}/ability_oof"):
            stage1 = AbilityModel()
            df = stage1.train_oof(df, n_folds=3, num_threads=num_threads)
        oof_mask = df["p_ability_win"].notna()
        df_oof = df[oof_mask].copy()

        # INTER-03: Target Encoding (OOF-safe, expanding window)
        # 血統系統・騎手・調教師の高カーディナリティカテゴリをTE化。
        # Stage1 OOF直後に実行: target (kakuteijyuni==1) が利用可能。
        # Stage1には追加しない (TE target == Stage1 targetでリークの可能性)。
        te_encoder = None
        with TimingContext(f"{surface}/target_encoding"):
            from features.target_encoding import TargetEncoder

            te_cat_cols = [
                c for c in ["blood_keito_cd", "kisyucode", "chokyosicode"] if c in df_oof.columns
            ]
            if te_cat_cols:
                te_encoder = TargetEncoder(
                    cat_cols=te_cat_cols,
                    target_col="kakuteijyuni",
                )
                df_oof = te_encoder.fit_transform_oof(df_oof)

        # FEAT-02: odds_to_ability_ratio -- 市場確率と能力確率の比
        # p_market_win_adj は FeatureEngine.build_all() -> compute_market_bias() で生成済み
        # p_ability_win は直上の AbilityModel.train_oof() で生成済み
        #
        # 【依存順序の意図】この計算は PlaceAbilityModel.train() の前に実行する必要がある。
        # odds_to_ability_ratio は p_ability_win (単勝能力確率) のみに依存し、
        # p_ability_place (複勝能力確率) には依存しない。もし将来 place 確率に依存する
        # ように変更する場合は、PlaceAbilityModel.predict() の後に移動すること。
        if "p_market_win_adj" in df_oof.columns and "p_ability_win" in df_oof.columns:
            p_market = df_oof["p_market_win_adj"].clip(lower=1e-6)
            p_ability = df_oof["p_ability_win"].clip(lower=1e-6)
            df_oof["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

        # INTER-01: Stage2相対特徴量 (p_ability_win / odds_to_ability_ratio依存)
        from features.relative_features import compute_stage2_relative_features

        df_oof = compute_stage2_relative_features(df_oof)

        # ODDS-01: deviation features (after odds_to_ability_ratio computed)
        from features.odds_deviation_features import compute_odds_deviation_features

        df_oof = compute_odds_deviation_features(df_oof)

        # NEW: PlaceAbilityModel
        place_ability = None
        if betting_target != "win":
            from models.place_ability_model import PlaceAbilityModel

            with TimingContext(f"{surface}/place_ability_train"):
                place_ability = PlaceAbilityModel()
                place_ability.train(df_oof, n_jobs=num_threads)
            with TimingContext(f"{surface}/place_ability_predict"):
                df_oof = place_ability.predict(df_oof)
        else:
            logger.info("Skipping PlaceAbilityModel for %s (betting_target=win)", surface)

        # 3. 単勝 2段階モデル
        win_2s = WinTwoStageModel()
        # PITリーク防止: WinTwoStageModel 学習前にソート
        if "race_date" in df_oof.columns:
            df_oof = df_oof.sort_values("race_date").reset_index(drop=True)
        if use_ensemble:
            from models.stacked_ensemble import StackedEnsemble

            with TimingContext(f"{surface}/win_hit_ensemble"):
                features = win_2s._prepare_features(df_oof)
                # Per-surface: drop constant columns that carry no information
                _const_cols = ["surface"] + [
                    c for c in features.columns if c.startswith("surface_x_")
                ]
                features = features.drop(columns=[c for c in _const_cols if c in features.columns])
                y = (df_oof["kakuteijyuni"] == 1).astype(int)
                split = int(len(features) * 0.8)
                _cat_cols = [c for c in ["distance_bin", "grade_code"] if c in features.columns]
                ensemble = StackedEnsemble(cat_cols=_cat_cols)
                ensemble.train(
                    features.iloc[:split],
                    y.iloc[:split],
                    features.iloc[split:],
                    y.iloc[split:],
                    num_threads=num_threads,
                )
                win_2s.hit_model = ensemble
        else:
            with TimingContext(f"{surface}/win_hit"):
                win_2s.train_hit_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/win_return"):
            win_2s.train_return_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/win_predict"):
            df_oof = win_2s.predict_ev(df_oof)

        # Group C/D: 騎手/調教師コンテキスト (Stage2)
        from features.jockey_context_features import JockeyContextFeatures
        from features.trainer_context_features import TrainerContextFeatures

        with TimingContext(f"{surface}/jockey_ctx"):
            jockey_ctx = JockeyContextFeatures(self.store)
            jockey_df = jockey_ctx.compute(df_oof)
            df_oof = pd.merge(df_oof, jockey_df, on=["race_id", "umaban"], how="left")

        with TimingContext(f"{surface}/trainer_ctx"):
            trainer_ctx = TrainerContextFeatures(self.store)
            trainer_df = trainer_ctx.compute(df_oof)
            df_oof = pd.merge(df_oof, trainer_df, on=["race_id", "umaban"], how="left")

        # B4: 騎手-調教師コンビコンテキスト (Stage2)
        from features.jockey_trainer_combo import JockeyTrainerComboFeatures

        with TimingContext(f"{surface}/jt_combo"):
            jt_combo = JockeyTrainerComboFeatures(self.store)
            jt_df = jt_combo.compute(df_oof)
            df_oof = pd.merge(df_oof, jt_df, on=["race_id", "umaban"], how="left")

        # PITリーク防止: merge操作で順序が破壊された可能性があるため再ソート
        if "race_date" in df_oof.columns:
            df_oof = df_oof.sort_values("race_date").reset_index(drop=True)

        # 4. EV補正モデル (P/E分解)
        with TimingContext(f"{surface}/ev_correction"):
            ev_corrector = EVCorrectionModel()
            ev_corrector.train(df_oof, num_threads=num_threads)
            df_oof = ev_corrector.correct_ev(df_oof)

        # --- Phase 19: EV Isotonic Calibration + Odds Band Residual Scaling (EVC-01/EVC-02) ---
        ev_isotonic_calibrator = None
        ev_odds_band_scales = None
        if len(df_oof) >= 500 and "confirmed_odds" in df_oof.columns:
            with TimingContext(f"{surface}/ev_isotonic_oof"):
                oof_ev, oof_actual, oof_odds, ev_fold_full = self.generate_ev_oof_predictions(
                    df_oof,
                    n_splits=5,
                    num_threads=num_threads,
                )
            # D-05: record ev_oof_fold on df_oof before df_oof_for_save copy
            df_oof["ev_oof_fold"] = pd.array(ev_fold_full, dtype=pd.Int64Dtype())
            if np.isfinite(oof_ev).sum() >= 200:
                with TimingContext(f"{surface}/ev_isotonic_fit"):
                    ev_isotonic_calibrator, ev_odds_band_scales = self.fit_ev_calibration(
                        oof_ev,
                        oof_actual,
                        oof_odds,
                    )
                if not _valid_ev_band_scales(ev_odds_band_scales):
                    logger.warning("Ignoring degenerate EV odds band scales for %s", surface)
                    ev_odds_band_scales = None
                ev_corrector.ev_isotonic_calibrator = ev_isotonic_calibrator
                ev_corrector.ev_odds_band_scales = ev_odds_band_scales
                df_oof = ev_corrector.correct_ev(df_oof)
                logger.info(
                    "EV Isotonic fitted for %s: %d OOF samples, band_scales=%s",
                    surface,
                    int(np.isfinite(oof_ev).sum()),
                    ev_odds_band_scales,
                )
            else:
                logger.warning(
                    "EV Isotonic: insufficient valid OOF samples (%d) for %s",
                    int(np.isfinite(oof_ev).sum()),
                    surface,
                )
        else:
            logger.info(
                "Skipping EV Isotonic for %s: len=%d, has_confirmed_odds=%s",
                surface,
                len(df_oof),
                "confirmed_odds" in df_oof.columns,
            )

        # 5. 複勝 2段階モデル
        place_2s: PlaceTwoStageModel | None = None
        if betting_target != "win":
            place_2s = PlaceTwoStageModel()
            if use_ensemble:
                from models.stacked_ensemble import StackedEnsemble

                with TimingContext(f"{surface}/place_hit_ensemble"):
                    features = place_2s._prepare_features(
                        df_oof,
                        use_cols=place_2s.HIT_FEATURE_COLS,
                    )
                    # Per-surface: drop constant columns that carry no information
                    _const_cols = ["surface"] + [
                        c for c in features.columns if c.startswith("surface_x_")
                    ]
                    features = features.drop(
                        columns=[c for c in _const_cols if c in features.columns]
                    )
                    y = (df_oof["kakuteijyuni"] <= 3).astype(int)
                    split = int(len(features) * 0.8)
                    _place_cat_cols = [
                        c for c in ["distance_bin", "grade_code"] if c in features.columns
                    ]
                    ensemble_place = StackedEnsemble(cat_cols=_place_cat_cols)
                    ensemble_place.train(
                        features.iloc[:split],
                        y.iloc[:split],
                        features.iloc[split:],
                        y.iloc[split:],
                        num_threads=num_threads,
                    )
                    place_2s.hit_model = ensemble_place
                    # バリデーション予測を保存 (Benter combination + isotonic fitting 用)
                    ensemble_val_pred = ensemble_place.predict(features.iloc[split:])
                    place_2s._val_p_raw = ensemble_val_pred
                    place_2s._val_y = y.iloc[split:].values
                    place_2s._val_fukuoddslow = df_oof["fukuoddslow"].iloc[split:].values
            else:
                # PITリーク防止: PlaceTwoStageModel 学習前にソート
                if "race_date" in df_oof.columns:
                    df_oof = df_oof.sort_values("race_date").reset_index(drop=True)
                with TimingContext(f"{surface}/place_hit"):
                    place_2s.train_hit_model(df_oof, num_threads=num_threads)
            with TimingContext(f"{surface}/place_return"):
                place_2s.train_return_model(df_oof, num_threads=num_threads)
            with TimingContext(f"{surface}/place_predict"):
                df_oof = place_2s.predict_ev(df_oof)
        else:
            logger.info("Skipping place model training for %s (betting_target=win)", surface)

        # 5b. Benter Combination + Isotonic Calibration + Temperature Scaling
        benter_combo = None
        isotonic_cal = None
        temp_scaler = None
        if (
            place_2s is not None
            and hasattr(place_2s, "_val_p_raw")
            and len(place_2s._val_p_raw) >= 500
        ):
            from sklearn.isotonic import IsotonicRegression

            from models.benter_combination import BenterCombination, TemperatureScaling

            val_p = place_2s._val_p_raw
            val_p_market = np.where(
                place_2s._val_fukuoddslow > 0,
                1.0 / place_2s._val_fukuoddslow,
                0.5,
            )
            val_y = place_2s._val_y

            with TimingContext(f"{surface}/benter"):
                benter_combo = BenterCombination.fit(val_p, val_p_market, val_y)
                logger.info(
                    "Benter params: alpha=%.3f, beta=%.3f, gamma=%.3f",
                    benter_combo.alpha,
                    benter_combo.beta,
                    benter_combo.gamma,
                )

            with TimingContext(f"{surface}/isotonic"):
                val_p_combined = benter_combo.combine(val_p, val_p_market)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(val_p_combined, val_y)
                isotonic_cal = iso
                logger.info("Isotonic calibrator fitted on %d samples", len(val_p))

            # v5: Temperature Scaling — Isotonic後の過信/過少評価を補正
            with TimingContext(f"{surface}/temperature"):
                val_p_isotonic = isotonic_cal.transform(val_p_combined)
                temp_scaler = TemperatureScaling.fit(val_p_isotonic, val_y)
                logger.info("Temperature Scaling: T=%.4f", temp_scaler.temperature)

        # 5c. Win Benter Combination (D-11, D-04, D-13)
        win_benter = None
        win_isotonic_cal = None
        win_temp_scaler = None
        if "tanodds" in df_oof.columns and len(df_oof) >= 500:
            from models.benter_combination import BenterCombination
            from models.win_benter_gate import generate_win_oof_predictions

            with TimingContext(f"{surface}/win_oof"):
                oof_p_fund, oof_p_market, oof_y = generate_win_oof_predictions(
                    df_oof,
                    win_model_cls=WinTwoStageModel,
                    ev_corrector=ev_corrector,
                    n_splits=5,
                    num_threads=num_threads,
                )

            if len(oof_p_fund) >= 500:
                from itertools import product as iter_product

                from scipy.optimize import minimize as scipy_minimize

                # Grid search for initial parameters (D-13)
                best_nll = float("inf")
                best_benter = None
                alpha_grid = [0.3, 0.5, 0.7, 1.0]
                beta_grid = [0.3, 0.5, 0.7, 1.0]
                gamma_grid = [-1.0, 0.0, 1.0]

                for a0, b0, g0 in iter_product(alpha_grid, beta_grid, gamma_grid):
                    try:
                        logit_f = BenterCombination._logit(oof_p_fund)
                        logit_m = BenterCombination._logit(oof_p_market)
                        y_arr = oof_y.astype(float)

                        def _nll(params: np.ndarray) -> float:
                            alpha, beta, gamma = params
                            logit_c = alpha * logit_f + beta * logit_m + gamma
                            p_c = 1.0 / (1.0 + np.exp(-logit_c))
                            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
                            return float(
                                -np.sum(y_arr * np.log(p_c) + (1 - y_arr) * np.log(1 - p_c))
                            )

                        res = scipy_minimize(
                            _nll,
                            x0=[a0, b0, g0],
                            method="L-BFGS-B",
                            bounds=[(0.01, 5.0), (0.20, 5.0), (-5.0, 5.0)],
                        )
                        if res.fun < best_nll:
                            best_nll = res.fun
                            best_benter = BenterCombination(
                                alpha=float(res.x[0]),
                                beta=float(res.x[1]),
                                gamma=float(res.x[2]),
                            )
                    except Exception:
                        continue

                if best_benter is not None:
                    win_benter = best_benter
                    logger.info(
                        "Win Benter (grid): alpha=%.3f, beta=%.3f, gamma=%.3f, NLL=%.2f",
                        win_benter.alpha,
                        win_benter.beta,
                        win_benter.gamma,
                        best_nll,
                    )
                else:
                    # Fallback to standard fit
                    with TimingContext(f"{surface}/win_benter"):
                        win_benter = BenterCombination.fit(oof_p_fund, oof_p_market, oof_y)
                    logger.info(
                        "Win Benter (fallback): alpha=%.3f, beta=%.3f, gamma=%.3f",
                        win_benter.alpha,
                        win_benter.beta,
                        win_benter.gamma,
                    )
            else:
                logger.warning("Win OOF samples < 500 (%d), skipping Win Benter", len(oof_p_fund))
        else:
            logger.info("tanodds not in df_oof or df too small, skipping Win Benter")

        # 5d. Win Calibration Comparison (D-05, D-07, D-08)
        if win_benter is not None and len(oof_p_fund) >= 500:
            from models.win_benter_gate import compare_calibrations, generate_reliability_data

            # Get Benter-combined probabilities for calibration
            oof_p_combined = win_benter.combine(oof_p_fund, oof_p_market)

            with TimingContext(f"{surface}/win_calibration"):
                cal_result = compare_calibrations(oof_p_combined, oof_y, train_ratio=0.8)

            # Log reliability diagram data
            reliability = generate_reliability_data(oof_y, oof_p_combined, n_bins=10)
            logger.info(
                "Win Reliability: bins=%s, positives=%s",
                [f"{v:.3f}" for v in reliability["mean_predicted_value"]],
                [f"{v:.3f}" for v in reliability["fraction_of_positives"]],
            )

            # Select calibrator based on comparison
            winner = cal_result["winner"]
            if winner == "beta":
                win_isotonic_cal = cal_result["beta_calibrator"]
                logger.info("Win calibration: Beta selected (Brier=%.6f)", cal_result["beta_brier"])
            elif winner == "isotonic":
                win_isotonic_cal = cal_result["iso_calibrator"]
                logger.info(
                    "Win calibration: Isotonic selected (Brier=%.6f)", cal_result["iso_brier"]
                )
            else:
                win_isotonic_cal = None
                logger.info("Win calibration: none selected (insufficient data)")

            # Temperature scaling (D-06: optional, apply only if it improves Brier Score)
            if win_isotonic_cal is not None:
                from sklearn.metrics import brier_score_loss

                from models.benter_combination import TemperatureScaling

                # Get calibrated probabilities on full OOF data
                oof_p_calibrated = np.asarray(
                    win_isotonic_cal.transform(oof_p_combined), dtype=float
                )
                brier_before_temp = float(
                    brier_score_loss(oof_y, np.clip(oof_p_calibrated, 1e-10, 1 - 1e-10))
                )

                try:
                    win_temp_scaler = TemperatureScaling.fit(oof_p_calibrated, oof_y)
                    oof_p_temp = win_temp_scaler.transform(oof_p_calibrated)
                    brier_after_temp = float(
                        brier_score_loss(oof_y, np.clip(oof_p_temp, 1e-10, 1 - 1e-10))
                    )

                    # Only keep TempScale if it improves Brier Score (D-06)
                    if brier_after_temp >= brier_before_temp:
                        logger.info(
                            "Win TempScale: no improvement (%.6f -> %.6f), skipping",
                            brier_before_temp,
                            brier_after_temp,
                        )
                        win_temp_scaler = None
                    else:
                        logger.info(
                            "Win TempScale: T=%.4f improved Brier (%.6f -> %.6f)",
                            win_temp_scaler.temperature,
                            brier_before_temp,
                            brier_after_temp,
                        )
                except Exception:
                    logger.warning("Win TempScale failed, skipping")
                    win_temp_scaler = None

        # 5a. Place EV補正 (P/E decomposition)
        place_ev_corrector: PlaceEVCorrectionModel | None = None
        if betting_target != "win":
            with TimingContext(f"{surface}/place_ev_correction"):
                place_ev_corrector = PlaceEVCorrectionModel()
                place_ev_corrector.train(df_oof, num_threads=num_threads)
                df_oof = place_ev_corrector.correct_ev(df_oof)

        # 6. ワイド 2段階モデル
        wide_2s: WideTwoStageModel | None = None
        if betting_target == "wide":
            with TimingContext(f"{surface}/wide_pair_build"):
                pair_df = WideJointPairBuilder().build(df_oof)
            wide_2s = WideTwoStageModel()
            if len(pair_df) > 0:
                with TimingContext(f"{surface}/wide_hit"):
                    wide_2s.train_hit_model(pair_df, num_threads=num_threads)
                with TimingContext(f"{surface}/wide_return"):
                    wide_2s.train_return_model(pair_df, num_threads=num_threads)

        # 7. Conformal EV Prediction Interval (CQR) per D-07/D-08/D-09
        conformal_ev: ConformalEVModel | None = None
        # ★ SAVE-01: confirmed_odds 削除前に df_oof を保存 (feature audit 用)
        df_oof_for_save = df_oof.copy()
        if len(df_oof) >= 500 and "ev_win_calibrated" in df_oof.columns:
            with TimingContext(f"{surface}/conformal_ev"):
                df_oof["actual_ev_win"] = self._build_cqr_actual_ev_target(df_oof).values

                # actual_ev_win 計算後、POST_RACE列を明示的に削除
                # (下流モデルが誤ってconfirmed_oddsを使用するのを防止)
                df_oof = df_oof.drop(columns=["confirmed_odds"], errors="ignore")

                # ★ SAFE-01: ConformalEVModel uses whitelist FEATURE_COLS internally
                # No explicit feature_cols needed — train() auto-selects from FEATURE_COLS
                conformal_ev = ConformalEVModel(alpha=0.1)
                conformal_ev.train(df_oof, num_threads=num_threads)
                if not conformal_ev._calibrated:
                    logger.warning("Conformal EV training incomplete for %s", surface)
                    conformal_ev = None
                else:
                    logger.info(
                        "Conformal EV fitted for %s: Q_90=%.4f, Q_80=%.4f",
                        surface,
                        conformal_ev._calibration_quantile_90,
                        conformal_ev._calibration_quantile_80,
                    )
        else:
            logger.info(
                "Skipping Conformal EV for %s: len=%d, has_ev_calibrated=%s",
                surface,
                len(df_oof),
                "ev_win_calibrated" in df_oof.columns,
            )

        place_selection_gate: PlaceSelectionGateModel | None = None
        if betting_target != "win":
            with TimingContext(f"{surface}/place_selection_gate"):
                gate_train_df = df_oof.copy()
                if conformal_ev is not None:
                    _, gate_place_df = conformal_ev.predict_interval(df_oof.copy(), df_oof.copy())
                    if "EV_lower_place" in gate_place_df.columns:
                        gate_train_df["EV_lower_place"] = gate_place_df["EV_lower_place"].values
                gate_train_df = ensure_place_selection_columns(gate_train_df)
                place_selection_gate = PlaceSelectionGateModel()
                place_selection_gate.train(gate_train_df)

        # --- WinSelectionGate training (SELC-01, D-01) ---
        with TimingContext(f"{surface}/win_selection_oof"):
            wsg_train_df = self.generate_win_selection_oof_frame(
                df_oof_for_save,
                n_splits=5,
                num_threads=num_threads,
            )
            if wsg_train_df.empty:
                logger.warning(
                    "Win selection OOF generation returned empty for %s; "
                    "selection models will remain conservative/untrained.",
                    surface,
                )
                wsg_train_df = df_oof.iloc[0:0].copy()
            wsg_train_df = ensure_win_selection_columns(wsg_train_df)

        # --- Drift diagnostics (GATE-02, D-01/D-02/D-03) ---
        if use_ensemble:
            with TimingContext(f"{surface}/drift_diagnostics"):
                from models.drift_diagnostics import compute_drift_diagnostics, console_summary

                drift_output_path = Path("data/backtest") / f"drift_diagnostics_{surface}.json"
                drift_result = compute_drift_diagnostics(
                    wsg_train_df,
                    output_path=drift_output_path,
                    surface=surface,
                )
                console_summary(drift_result)

        # --- EV diagnostics (EVF-02, D-04/D-05) ---
        if use_ensemble:
            with TimingContext(f"{surface}/ev_diagnostics"):
                from models.ev_diagnostics import compute_ev_diagnostics as compute_ev_diag
                from models.ev_diagnostics import console_summary as ev_console_summary

                ev_output_path = Path("data/backtest") / f"ev_diagnostics_{surface}.json"
                ev_result = compute_ev_diag(
                    wsg_train_df,
                    output_path=ev_output_path,
                    surface=surface,
                )
                ev_console_summary(ev_result)

        with TimingContext(f"{surface}/win_selection_gate_train"):
            win_selection_gate = WinSelectionGateModel()
            win_selection_gate.train(wsg_train_df)

        with TimingContext(f"{surface}/win_selection_policy_train"):
            win_selection_policy = WinSelectionPolicy()
            win_selection_policy.train(wsg_train_df)
            wsg_train_df = win_selection_policy.apply(wsg_train_df)
            oof_guard_metrics = _validate_win_selection_oof_health(
                wsg_train_df,
                context=f"{surface}/win_selection_train",
            )
            logger.info("Win selection OOF guard passed for %s: %s", surface, oof_guard_metrics)
            logger.info(
                "WinSelectionPolicy trained for %s: late_odds_drop_weight=%.4f summary=%s",
                surface,
                win_selection_policy.late_odds_drop_weight,
                win_selection_policy.training_summary,
            )

        with TimingContext(f"{surface}/win_profit_selector_train"):
            win_profit_selector = WinProfitSelector()
            win_profit_selector.train(wsg_train_df)
            wsg_train_df = win_profit_selector.score(wsg_train_df)
            logger.info(
                "WinProfitSelector trained for %s: trained=%s summary=%s",
                surface,
                win_profit_selector.is_trained,
                win_profit_selector.training_summary,
            )

        win_segment_calibrator = None
        if surface == "turf":
            with TimingContext(f"{surface}/win_segment_calibrator_train"):
                win_segment_calibrator = WinSegmentCalibrator(target_surface=surface)
                win_segment_calibrator.train(wsg_train_df)
                logger.info(
                    "WinSegmentCalibrator trained for %s: trained=%s summary=%s",
                    surface,
                    win_segment_calibrator.is_trained,
                    win_segment_calibrator.training_summary,
                )
        else:
            logger.info(
                "WinSegmentCalibrator skipped for %s: turf-only probability shrinkage is enabled",
                surface,
            )

        # --- D-08 Part 2: Runtime check (ensemble mode only) ---
        if use_ensemble and not win_selection_gate.is_trained:
            logger.warning(
                "WinSelectionGate did not train for surface=%s "
                "(check debug logs for reason: empty data / insufficient races / "
                "no profitable threshold)",
                surface,
            )

        # --- D-01/D-02: Dynamic EV_lower threshold from OOF winners ---
        # D-03 fallback values: conservative defaults below 1.0
        # wsg_train_df contains EV_lower_win_corrected from confidence estimator
        # BUGFIX: wsg_train_df is already filtered to a single surface, so we must
        # compute the threshold for the *current* surface only. The other surface
        # gets the same value since each SubmodelSet is surface-specific.
        ev_threshold = self._compute_ev_threshold(
            wsg_train_df,
            surface=surface,
            fallback=0.75,
        )
        ev_threshold_turf = ev_threshold
        ev_threshold_dirt = ev_threshold

        sub = SubmodelSet(
            market=market,
            stage1=stage1,
            place_ability=place_ability,
            win=win_2s,
            ev_corrector=ev_corrector,
            place=place_2s,
            place_ev_corrector=place_ev_corrector,
            wide=wide_2s,
            conformal_ev_model=conformal_ev,  # Phase 21: CQR model
            place_selection_gate=place_selection_gate,
            use_ensemble=use_ensemble,
            benter_combo=benter_combo,
            isotonic_calibrator=isotonic_cal,
            temperature_scaler=temp_scaler,
            win_benter=win_benter,
            win_isotonic_calibrator=win_isotonic_cal,
            win_temperature_scaler=win_temp_scaler,
            win_selection_gate=win_selection_gate,
            win_selection_policy=win_selection_policy,
            win_profit_selector=win_profit_selector,
            win_segment_calibrator=win_segment_calibrator,
            ev_lower_threshold_turf=ev_threshold_turf,
            ev_lower_threshold_dirt=ev_threshold_dirt,
            ev_isotonic_calibrator=ev_isotonic_calibrator,
            ev_odds_band_scales=ev_odds_band_scales,
            target_encoder=te_encoder,
        )
        # Wire Isotonic + band scales into ev_corrector for correct_ev() to apply
        sub.ev_corrector.ev_isotonic_calibrator = ev_isotonic_calibrator
        sub.ev_corrector.ev_odds_band_scales = ev_odds_band_scales
        return sub, df_oof_for_save, wsg_train_df

    @staticmethod
    def generate_win_selection_oof_frame(
        df: pd.DataFrame,
        *,
        n_splits: int = 5,
        num_threads: int = 0,
    ) -> pd.DataFrame:
        """Win selection用のfold外予測DataFrameを生成する。

        WinSelectionGate/Policy/ProfitSelector は選定順位を学習するため、
        full trainに再予測した確率を使うとROIが極端に上振れする。
        ここではrace単位のwalk-forwardで、各検証foldを過去データだけで
        学習したWinTwoStageModel + EVCorrectionModelに通す。
        """
        if df.empty:
            return pd.DataFrame()
        sort_cols = [col for col in ["race_date", "race_id", "umaban"] if col in df.columns]
        sorted_df = (
            df.sort_values(sort_cols).reset_index(drop=True)
            if sort_cols
            else df.reset_index(drop=True)
        )
        splits = _walk_forward_race_splits(sorted_df, n_splits=n_splits)
        if not splits:
            return pd.DataFrame()

        oof_frames: list[pd.DataFrame] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_train = sorted_df.iloc[train_idx].copy()
            fold_val_source = sorted_df.iloc[val_idx].copy()

            fold_win = WinTwoStageModel()
            try:
                fold_win.train_hit_model(fold_train, num_threads=num_threads)
                fold_win.train_return_model(fold_train, num_threads=num_threads)

                fold_train_pred = fold_win.predict_ev(fold_train.copy())
                fold_ev_corr = EVCorrectionModel()
                fold_ev_corr.train(fold_train_pred, num_threads=num_threads)

                fold_val = fold_win.predict_ev(fold_val_source)
                fold_val["p_win_oof"] = fold_val["p_win_pred"]
                fold_val = fold_ev_corr.correct_ev(fold_val, probability_col="p_win_oof")
            except Exception as exc:
                logger.warning("Skipping win selection OOF fold %d: %s", fold_idx, exc)
                continue
            fold_val["win_selection_oof_fold"] = fold_idx
            oof_frames.append(fold_val)

        if not oof_frames:
            return pd.DataFrame()
        result = pd.concat(oof_frames, ignore_index=True)
        return result.sort_values(sort_cols).reset_index(drop=True) if sort_cols else result

    @staticmethod
    def generate_ev_oof_predictions(
        df: pd.DataFrame,
        *,
        n_splits: int = 5,
        num_threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """EVC-01/D-05/D-09: OOF EV予測をK-foldで生成.

        学習チェーン: WinTwoStage predict_ev → EVCorrection correct_ev
        Returns: (oof_ev_corrected, oof_actual_return, oof_odds, oof_fold_assignments)
          — first 3 are NaN-masked (valid only), oof_fold_assignments is full-length
        """
        df = df.sort_values("race_date").reset_index(drop=True)
        splits = _walk_forward_race_splits(df, n_splits=n_splits)
        oof_ev_corrected = np.full(len(df), np.nan)
        oof_actual_return = np.full(len(df), np.nan)
        oof_odds = np.full(len(df), np.nan)
        oof_fold_assignments = np.full(len(df), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_win = WinTwoStageModel()
            try:
                fold_win.train_hit_model(df.iloc[train_idx], num_threads=num_threads)
                fold_win.train_return_model(df.iloc[train_idx], num_threads=num_threads)

                fold_ev_corr = EVCorrectionModel()
                fold_train = fold_win.predict_ev(df.iloc[train_idx].copy())
                fold_ev_corr.train(fold_train, num_threads=num_threads)
                fold_val = fold_win.predict_ev(df.iloc[val_idx].copy())
                fold_val = fold_ev_corr.correct_ev(fold_val)
            except Exception as exc:
                logger.warning("Skipping EV OOF fold: %s", exc)
                continue

            oof_ev_corrected[val_idx] = fold_val["ev_win_corrected"].values

            odds_col = "confirmed_odds" if "confirmed_odds" in fold_val.columns else "odds"
            odds_vals = fold_val.get(odds_col, pd.Series(0.0, index=fold_val.index))
            oof_odds[val_idx] = pd.to_numeric(odds_vals, errors="coerce").values
            oof_actual_return[val_idx] = (
                pd.to_numeric(odds_vals, errors="coerce")
                * (fold_val["kakuteijyuni"] == 1).astype(float)
            ).values
            oof_fold_assignments[val_idx] = fold_idx

        return oof_ev_corrected, oof_actual_return, oof_odds, oof_fold_assignments

    @staticmethod
    def fit_ev_calibration(
        oof_ev: np.ndarray,
        oof_actual: np.ndarray,
        oof_odds: np.ndarray,
    ) -> tuple[IsotonicRegression, dict[str, float]]:
        """OOF EVをactual_returnへ校正し、オッズ帯別残差スケールを推定する.

        Returns: (isotonic_model, odds_band_scales)
        """
        from sklearn.isotonic import IsotonicRegression

        from betting.odds_band_filter import OddsBandFilter

        valid = np.isfinite(oof_ev) & np.isfinite(oof_actual) & (oof_ev > 0)
        if valid.sum() < 10:
            iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
            iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            return iso, {name: 1.0 for name in OddsBandFilter.BAND_NAMES}

        # Isotonic fit on EV buckets to avoid sparse point EV collapsing to zero.
        ev_valid = oof_ev[valid].astype(float)
        actual_valid = oof_actual[valid].astype(float)
        try:
            bin_id = pd.qcut(
                ev_valid,
                q=min(20, max(2, valid.sum() // 100)),
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            bin_id = pd.Series(np.zeros(len(ev_valid), dtype=int))
        bucket_df = pd.DataFrame(
            {"ev": ev_valid, "actual": actual_valid, "bin": np.asarray(bin_id)}
        )
        grouped = (
            bucket_df.groupby("bin", observed=True)
            .agg(ev_mean=("ev", "mean"), actual_mean=("actual", "mean"), n=("actual", "size"))
            .sort_values("ev_mean")
        )
        if len(grouped) < 2:
            grouped = pd.DataFrame(
                {
                    "ev_mean": [float(np.nanmin(ev_valid)), float(np.nanmax(ev_valid))],
                    "actual_mean": [
                        float(np.nanmean(actual_valid)),
                        float(np.nanmean(actual_valid)),
                    ],
                }
            )

        iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
        iso.fit(grouped["ev_mean"].to_numpy(), grouped["actual_mean"].to_numpy())

        ev_calibrated = np.copy(oof_ev)
        ev_calibrated[valid] = iso.transform(oof_ev[valid])

        # オッズバンド別残差スケーリング (D-10, Pattern 3 from RESEARCH)
        # BUGFIX: point-wise ratio median is 0 because 93% of actual_ev_win = 0.
        # Use aggregate ratio sum(actual)/sum(calibrated) instead, which correctly
        # estimates E[actual|band] / E[predicted|band] and averages out the zeros.
        bands = OddsBandFilter.BANDS
        band_names = OddsBandFilter.BAND_NAMES
        min_samples = 50

        band_scales: dict[str, float] = {}
        for band_name, (lo, hi) in zip(band_names, bands):
            mask = (oof_odds >= lo) & (oof_odds < hi) & valid
            if mask.sum() >= min_samples:
                actual_sum = float(np.sum(oof_actual[mask]))
                calibrated_sum = float(np.sum(np.clip(ev_calibrated[mask], 1e-6, None)))
                if actual_sum <= 0 or calibrated_sum <= 0:
                    band_scales[band_name] = 1.0
                else:
                    band_scales[band_name] = float(np.clip(actual_sum / calibrated_sum, 0.25, 3.0))
            else:
                band_scales[band_name] = 1.0

        return iso, band_scales

    def _build_race_level_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """馬レベル特徴量 → レースレベル特徴量に集約

        RaceQualityScreener.FEATURE_COLS (19列) に対応。
        v5.5 leak-fix: favorite_win_rate を expanding window で計算 (C3)。
        """
        race_feat = (
            feat_df.groupby("race_id", observed=True)
            .agg(
                surface=("surface", "first"),
                distance_bin=("distance_bin", "first"),
                track_condition_code=("track_condition_code", "first"),
                grade_code=("grade_code", "first"),
                field_size=("field_size", "first"),
                difficulty_score=("difficulty_score", "first"),
                # 市場エラー統計
                market_log_error_mean=("signed_log_error_win", "mean"),
                market_log_error_std=("signed_log_error_win", "std"),
                market_log_error_abs_mean=("abs_log_error_win", "mean"),
                # 分布特徴量
                n_positive_errors=("signed_log_error_win", lambda x: (x > 0).sum()),
                top_k_error_sum=("signed_log_error_win", lambda x: x.nlargest(3).sum()),
                positive_error_ratio=(
                    "signed_log_error_win",
                    lambda x: (x > 0).sum() / max(len(x), 1),
                ),
                # 市場構造
                market_entropy_mean=("market_entropy", "first"),
                overround_mean=("overround", "first"),
            )
            .reset_index()
        )

        # C3 fix: favorite_win_rate を expanding window で計算
        if "race_date" in feat_df.columns:
            date_map = feat_df.groupby("race_id", observed=True)["race_date"].first()
            race_feat["race_date"] = self._map_race_values(race_feat["race_id"], date_map)
            race_feat = race_feat.sort_values("race_date").reset_index(drop=True)

        if "kakuteijyuni" in feat_df.columns and "popularity_rank" in feat_df.columns:
            fav_df = feat_df[feat_df["popularity_rank"] == 1][["race_id", "kakuteijyuni"]].copy()
            fav_df["fav_won"] = (fav_df["kakuteijyuni"] == 1).astype(float)
            race_feat = race_feat.merge(fav_df[["race_id", "fav_won"]], on="race_id", how="left")
            race_feat["fav_won"] = race_feat["fav_won"].fillna(0.0)
            race_feat["favorite_win_rate"] = (
                race_feat["fav_won"].shift(1).expanding(min_periods=10).mean()
            )
            race_feat["favorite_win_rate"] = race_feat["favorite_win_rate"].fillna(0.3)
            race_feat = race_feat.drop(columns=["fav_won"])
        else:
            race_feat["favorite_win_rate"] = 0.3

        # 結果ベース proxy — 人気馬の実際の結果から expanding window で計算
        race_feat["hist_hit_rate_topk"] = race_feat["favorite_win_rate"]

        # 人気馬 (popularity_rank==1) の実際の成績を race_id 単位で集計
        if (
            "kakuteijyuni" in feat_df.columns
            and "popularity_rank" in feat_df.columns
            and "tanodds" in feat_df.columns
        ):
            fav_df = feat_df[feat_df["popularity_rank"] == 1][
                ["race_id", "kakuteijyuni", "tanodds"]
            ].copy()
            fav_df["fav_won"] = (fav_df["kakuteijyuni"] == 1).astype(float)
            fav_df["fav_placed"] = (fav_df["kakuteijyuni"] <= 3).astype(float)
            fav_df["fav_roi"] = np.where(
                fav_df["fav_won"] == 1,
                fav_df["tanodds"].astype(float) - 1.0,
                -1.0,
            )
            fav_df["fav_positive"] = (fav_df["fav_roi"] > 0).astype(float)
            fav_agg = fav_df[["race_id", "fav_placed", "fav_roi", "fav_positive", "fav_won"]].copy()
            race_feat = race_feat.merge(fav_agg, on="race_id", how="left")
            race_feat["topk_hit"] = race_feat["fav_placed"].fillna(0.0)
            race_feat["topk_roi"] = race_feat["fav_roi"].fillna(-1.0)
            race_feat["positive_return"] = race_feat["fav_positive"].fillna(0.0)
            race_feat["is_winner"] = race_feat["fav_won"].fillna(0.0)
            # expanding window で履歴統計を計算 (リーク防止: shift(1))
            race_feat["hist_roi_topk"] = (
                race_feat["topk_roi"].shift(1).expanding(min_periods=10).mean().fillna(1.0)
            )
            race_feat["hist_positive_return_ratio"] = (
                race_feat["positive_return"].shift(1).expanding(min_periods=10).mean().fillna(0.3)
            )
            race_feat = race_feat.drop(
                columns=["fav_placed", "fav_roi", "fav_positive", "fav_won"],
                errors="ignore",
            )
        else:
            race_feat["hist_roi_topk"] = 1.0
            race_feat["hist_positive_return_ratio"] = 0.3

        # compute_hist_features が必要とする列を追加
        race_feat["distance_band"] = race_feat["distance_bin"]
        race_feat["market_entropy"] = race_feat["market_entropy_mean"]

        # RaceQualityScreener が必要とする列を補完
        race_feat["market_log_error_max_abs"] = race_feat["market_log_error_abs_mean"] * 2.0
        race_feat["market_log_error_top_q75"] = race_feat["market_log_error_abs_mean"] * 1.5
        race_feat["market_entropy"] = race_feat["market_entropy_mean"]
        race_feat["overround"] = race_feat["overround_mean"]
        race_feat["overround_deviation"] = 0.0
        race_feat["hist_win_rate_same_condition"] = race_feat["favorite_win_rate"]
        race_feat["hist_market_entropy_avg"] = race_feat["market_entropy_mean"]

        # rl_* columns: propagate race-constant features from horse-level feat_df
        # These are generated by compute_race_level_features() and compute_market_cross_features()
        # in build_all(), and are constant within each race, so "first" aggregation is correct.
        _rl_cols = [
            # RLF-01~06 (race_level_features.py)
            "rl_log_odds_entropy",
            "rl_odds_dispersion",
            "rl_top3_odds_gap",
            "rl_top1_odds",
            "rl_favorite_rank_gap",
            "rl_n_horses",
            # MCF-07 (market_cross_features.py)
            "rl_favorite_in_wide_top1",
            "rl_trio_overlap",
            "rl_market_consistency",
            "rl_trio_odds_ratio",
            "rl_wide_harville_ratio",
            # FLB slope (market_bias_features.py)
            "implied_prob_hhi",
            "odds_skewness",
        ]
        for _col in _rl_cols:
            if _col in feat_df.columns:
                _map = feat_df.groupby("race_id", observed=True)[_col].first()
                race_feat[_col] = self._map_race_numeric(race_feat["race_id"], _map)
            else:
                race_feat[_col] = np.nan

        # 履歴特徴量 (expanding window — リークフリー)
        if "race_date" in race_feat.columns:
            try:
                from features.info_asymmetry_features import compute_hist_features

                race_feat = compute_hist_features(race_feat)
            except Exception as e:
                logger.debug("hist_features skipped: %s", e)

        # v5.6: EMA 平滑化市場指標 (horse-level feat_df から計算し race-level に展開)
        if "tanodds" in feat_df.columns:
            ema_df = compute_roi_ema(
                feat_df[
                    ["race_id", "tanodds", "popularity_rank"]
                    + (["race_date"] if "race_date" in feat_df.columns else [])
                ]
            )
            # race_id 単位で EMA 値を race_feat にマージ
            for ema_col in ["overround_ema", "entropy_ema"]:
                if ema_col in ema_df.columns:
                    ema_map = ema_df.groupby("race_id", observed=True)[ema_col].first()
                    race_feat[ema_col] = self._map_race_numeric(
                        race_feat["race_id"],
                        ema_map,
                        fill_value=0.0,
                    )
        else:
            race_feat["overround_ema"] = 0.0
            race_feat["entropy_ema"] = 0.0

        # Phase36 race-level aggregates (RTG-02/03)
        _col_csr = "closing_speed_ratio_avg"
        if _col_csr in feat_df.columns:
            _csr = feat_df.groupby("race_id", observed=True)[_col_csr]
            race_feat["phase36_top1_strength"] = self._map_race_numeric(
                race_feat["race_id"],
                _csr.max(),
                fill_value=0.0,
            )
            _top2_gap = _csr.apply(
                lambda x: x.nlargest(2).diff().iloc[-1] if x.notna().sum() >= 2 else 0.0
            )
            race_feat["phase36_top1_top2_gap"] = self._map_race_numeric(
                race_feat["race_id"],
                _top2_gap,
                fill_value=0.0,
            )
            race_feat["phase36_field_dispersion"] = self._map_race_numeric(
                race_feat["race_id"],
                _csr.std(),
                fill_value=0.0,
            )
        else:
            race_feat["phase36_top1_strength"] = 0.0
            race_feat["phase36_top1_top2_gap"] = 0.0
            race_feat["phase36_field_dispersion"] = 0.0

        _col_ftr = "form_trend_race_rank"
        if _col_ftr in feat_df.columns:
            _ftr = feat_df.groupby("race_id", observed=True)[_col_ftr]
            race_feat["phase36_form_signal_dispersion"] = self._map_race_numeric(
                race_feat["race_id"], _ftr.std(), fill_value=0.0
            )
        else:
            race_feat["phase36_form_signal_dispersion"] = 0.0

        _col_wrf = "weighted_recent_form_finish"
        if _col_wrf in feat_df.columns:
            _wrf = feat_df.groupby("race_id", observed=True)[_col_wrf]
            race_feat["phase36_weighted_form_mean"] = self._map_race_numeric(
                race_feat["race_id"], _wrf.mean(), fill_value=0.0
            )
        else:
            race_feat["phase36_weighted_form_mean"] = 0.0

        return race_feat

    def _build_regime_stats(
        self, race_feat_df: pd.DataFrame, feat_df: pd.DataFrame
    ) -> pd.DataFrame:
        """RegimeDetector 用のレース統計を構築

        RegimeDetector.FEATURE_COLS (8列) に対応。
        推論パスと整合させるため rolling を行わず、レース毎の生値を使用。
        全て発走前情報のみ使用。
        """
        if "race_date" in race_feat_df.columns:
            race_feat_df = race_feat_df.sort_values("race_date").reset_index(drop=True)

        stats = race_feat_df.copy()
        # race_id や条件列が Categorical のまま map されないよう通常 dtype に戻す。
        for col in stats.columns:
            if isinstance(stats[col].dtype, pd.CategoricalDtype):
                stats[col] = stats[col].astype(str)

        # 基本列マッピング (MarketModel 由来) — 元々 rolling 不使用
        stats["market_error_std"] = stats["market_log_error_std"].fillna(0.2)
        stats["market_error_mean"] = stats["market_log_error_mean"].fillna(0.0)
        stats["field_size_mean"] = stats["field_size"].fillna(14.0).astype(float)

        # overround_rolling: overround の生値 (推論パスと整合)
        if "overround_mean" in stats.columns:
            stats["overround_rolling"] = stats["overround_mean"]
        else:
            stats["overround_rolling"] = 0.20

        # entropy_rolling: market_entropy の生値 (推論パスと整合)
        if "market_entropy_mean" in stats.columns:
            stats["entropy_rolling"] = stats["market_entropy_mean"]
        else:
            stats["entropy_rolling"] = 2.0

        # favorite_implied_prob_rolling: 1番人気オッズの逆数 (レース毎生値)
        if all(c in feat_df.columns for c in ["race_id", "tanodds", "popularity_rank"]):
            fav_df = feat_df[feat_df["popularity_rank"] == 1][["race_id", "tanodds"]].copy()
            fav_df["fav_implied"] = 1.0 / fav_df["tanodds"].replace(0, np.nan)
            race_fav_implied = fav_df.groupby("race_id", observed=True)["fav_implied"].first()
            stats["favorite_implied_prob_rolling"] = self._map_race_numeric(
                stats["race_id"],
                race_fav_implied,
                fill_value=0.3,
            )
        else:
            stats["favorite_implied_prob_rolling"] = 0.3

        # odds_skewness_rolling: レース毎のオッズ歪度 (生値)
        if all(c in feat_df.columns for c in ["race_id", "tanodds"]):
            race_skew = feat_df.groupby("race_id", observed=True)["tanodds"].skew()
            stats["odds_skewness_rolling"] = self._map_race_numeric(
                stats["race_id"],
                race_skew,
                fill_value=0.0,
            )
        else:
            stats["odds_skewness_rolling"] = 0.0

        # odds_volatility_mean: レース毎の odds_volatility 平均 (生値)
        if "odds_volatility" in feat_df.columns:
            race_vol = feat_df.groupby("race_id", observed=True)["odds_volatility"].mean()
            stats["odds_volatility_mean"] = self._map_race_numeric(
                stats["race_id"],
                race_vol,
                fill_value=0.1,
            )
        else:
            stats["odds_volatility_mean"] = 0.1

        return stats

    @staticmethod
    def _map_race_values(race_ids: pd.Series, values: pd.Series) -> pd.Series:
        """race_id が Categorical でも通常 dtype の Series として map する。"""
        mapping = {str(k): v for k, v in values.items()}
        return race_ids.astype(str).map(mapping)

    @staticmethod
    def _map_race_numeric(
        race_ids: pd.Series,
        values: pd.Series,
        *,
        fill_value: float | None = None,
    ) -> pd.Series:
        """race_id map の結果を必ず数値 dtype にする。

        pandas は Categorical Series に map() した結果を category dtype に保つ場合があり、
        そのまま LightGBM に渡すと学習時/推論時の categorical_feature が不一致になる。
        """
        mapped = TrainingPipelineV5._map_race_values(race_ids, values)
        numeric = pd.to_numeric(mapped, errors="coerce")
        if fill_value is not None:
            numeric = numeric.fillna(fill_value)
        return numeric.astype(float)

    def _log_to_mlflow(
        self,
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> None:
        """MLflow に全モデルとメトリクスを記録 (Paper Trading対応)"""
        with mlflow.start_run(run_name=f"v5.5_{train_end}"):
            for surface, sub in models.items():
                # Stage1 (AbilityModel per surface)
                stage1_model = sub.stage1.models.get(surface)
                if stage1_model is not None:
                    mlflow.lightgbm.log_model(
                        stage1_model,
                        name=f"stage1_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )

                # MarketModel
                mlflow.lightgbm.log_model(
                    sub.market.model,
                    name=f"market_{surface}",
                    pip_requirements=_MLFLOW_PIP_REQS,
                )

                # WinTwoStageModel
                if sub.use_ensemble:
                    # StackedEnsemble — joblib pickle で保存
                    _se_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as _f:
                            _se_tmp = _f.name
                            joblib.dump(sub.win.hit_model, _f.name)
                        mlflow.log_artifact(_se_tmp, f"win_hit_{surface}")
                    finally:
                        if _se_tmp and os.path.exists(_se_tmp):
                            os.unlink(_se_tmp)
                else:
                    mlflow.lightgbm.log_model(
                        sub.win.hit_model,
                        name=f"win_hit_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )
                mlflow.lightgbm.log_model(
                    sub.win.return_model,
                    name=f"win_ret_{surface}",
                    pip_requirements=_MLFLOW_PIP_REQS,
                )

                # EVCorrectionModel
                mlflow.lightgbm.log_model(
                    sub.ev_corrector.p_correction_model,
                    name=f"ev_corrector_p_{surface}",
                    pip_requirements=_MLFLOW_PIP_REQS,
                )
                mlflow.lightgbm.log_model(
                    sub.ev_corrector.e_correction_model,
                    name=f"ev_corrector_e_{surface}",
                    pip_requirements=_MLFLOW_PIP_REQS,
                )

                # PlaceEVCorrectionModel
                if sub.place_ev_corrector is not None:
                    mlflow.lightgbm.log_model(
                        sub.place_ev_corrector.p_correction_model,
                        name=f"place_ev_corrector_p_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )
                    mlflow.lightgbm.log_model(
                        sub.place_ev_corrector.e_correction_model,
                        name=f"place_ev_corrector_e_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )
                if sub.place_selection_gate is not None and sub.place_selection_gate.is_trained:
                    gate_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".joblib",
                            delete=False,
                        ) as gate_file:
                            gate_tmp = gate_file.name
                        sub.place_selection_gate.save(Path(gate_tmp))
                        mlflow.log_artifact(gate_tmp, f"place_selection_gate_{surface}")
                    finally:
                        if gate_tmp and os.path.exists(gate_tmp):
                            os.unlink(gate_tmp)

                # --- WinSelectionGate (MLflow) ---
                if sub.win_selection_gate is not None and sub.win_selection_gate.is_trained:
                    wsg_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".joblib",
                            delete=False,
                        ) as wsg_file:
                            wsg_tmp = wsg_file.name
                        sub.win_selection_gate.save(Path(wsg_tmp))
                        mlflow.log_artifact(wsg_tmp, f"win_selection_gate_{surface}")
                    finally:
                        if wsg_tmp and os.path.exists(wsg_tmp):
                            os.unlink(wsg_tmp)

                if sub.win_selection_policy is not None and sub.win_selection_policy.is_trained:
                    wsp_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".joblib",
                            delete=False,
                        ) as wsp_file:
                            wsp_tmp = wsp_file.name
                        sub.win_selection_policy.save(Path(wsp_tmp))
                        mlflow.log_artifact(wsp_tmp, f"win_selection_policy_{surface}")
                    finally:
                        if wsp_tmp and os.path.exists(wsp_tmp):
                            os.unlink(wsp_tmp)

                if sub.win_profit_selector is not None and sub.win_profit_selector.is_trained:
                    wps_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".joblib",
                            delete=False,
                        ) as wps_file:
                            wps_tmp = wps_file.name
                        sub.win_profit_selector.save(Path(wps_tmp))
                        mlflow.log_artifact(wps_tmp, f"win_profit_selector_{surface}")
                    finally:
                        if wps_tmp and os.path.exists(wps_tmp):
                            os.unlink(wps_tmp)

                if sub.win_segment_calibrator is not None and sub.win_segment_calibrator.is_trained:
                    wsc_tmp: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".joblib",
                            delete=False,
                        ) as wsc_file:
                            wsc_tmp = wsc_file.name
                        sub.win_segment_calibrator.save(Path(wsc_tmp))
                        mlflow.log_artifact(wsc_tmp, f"win_segment_calibrator_{surface}")
                    finally:
                        if wsc_tmp and os.path.exists(wsc_tmp):
                            os.unlink(wsc_tmp)

                # PlaceTwoStageModel
                if sub.place is not None:
                    if sub.use_ensemble:
                        _se_tmp2: str | None = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as _f2:
                                _se_tmp2 = _f2.name
                                joblib.dump(sub.place.hit_model, _f2.name)
                            mlflow.log_artifact(_se_tmp2, f"place_hit_{surface}")
                        finally:
                            if _se_tmp2 and os.path.exists(_se_tmp2):
                                os.unlink(_se_tmp2)
                    else:
                        mlflow.lightgbm.log_model(
                            sub.place.hit_model,
                            name=f"place_hit_{surface}",
                            pip_requirements=_MLFLOW_PIP_REQS,
                        )
                    mlflow.lightgbm.log_model(
                        sub.place.return_model,
                        name=f"place_ret_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )

                # PlaceAbilityModel (sklearn CalibratedClassifierCV → joblib)
                calibrated = None
                if sub.place_ability is not None:
                    calibrated = sub.place_ability._calibrated or sub.place_ability._model
                if calibrated is not None:
                    _tmp_path: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                            _tmp_path = f.name
                            joblib.dump(calibrated, f.name)
                        mlflow.log_artifact(_tmp_path, f"place_ability_{surface}")
                    finally:
                        if _tmp_path and os.path.exists(_tmp_path):
                            os.unlink(_tmp_path)

                # WideTwoStageModel
                if sub.wide is not None:
                    mlflow.lightgbm.log_model(
                        sub.wide.hit_model,
                        name=f"wide_hit_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )
                    mlflow.lightgbm.log_model(
                        sub.wide.return_model,
                        name=f"wide_ret_{surface}",
                        pip_requirements=_MLFLOW_PIP_REQS,
                    )

            # RaceQualityScreener
            mlflow.lightgbm.log_model(
                quality_screen.model,
                name="race_quality",
                pip_requirements=_MLFLOW_PIP_REQS,
            )
            mlflow.log_param("quality_threshold", quality_screen.threshold)

            # RegimeDetector
            mlflow.lightgbm.log_model(
                regime_det.model,
                name="regime_detector",
                pip_requirements=_MLFLOW_PIP_REQS,
            )

            # Phase 21: ConformalEVModel MLflow保存
            for surface, sub in models.items():
                if sub.conformal_ev_model is not None:
                    # CQR LightGBMモデルをMLflowに記録
                    if sub.conformal_ev_model.q_low_model is not None:
                        mlflow.lightgbm.log_model(
                            sub.conformal_ev_model.q_low_model,
                            name=f"cqr_quantile_low_{surface}",
                            pip_requirements=_MLFLOW_PIP_REQS,
                        )
                    if sub.conformal_ev_model.q_high_model is not None:
                        mlflow.lightgbm.log_model(
                            sub.conformal_ev_model.q_high_model,
                            name=f"cqr_quantile_high_{surface}",
                            pip_requirements=_MLFLOW_PIP_REQS,
                        )
                    # CQR paramsをアーティファクトとして記録
                    cqr_params = {
                        "alpha": sub.conformal_ev_model.alpha,
                        "calibration_quantile_90": sub.conformal_ev_model._calibration_quantile_90,
                        "calibration_quantile_80": sub.conformal_ev_model._calibration_quantile_80,
                        "residual_quantile_90": sub.conformal_ev_model._residual_quantile_90,
                        "residual_quantile_80": sub.conformal_ev_model._residual_quantile_80,
                        "_calibrated": sub.conformal_ev_model._calibrated,
                    }
                    if sub.conformal_ev_model.feature_cols is not None:
                        cqr_params["feature_cols"] = sub.conformal_ev_model.feature_cols
                    mlflow.log_dict(cqr_params, f"cqr_params_{surface}.json")

            mlflow.log_param("train_start", train_start)
            mlflow.log_param("train_end", train_end)
            mlflow.log_param("n_surfaces", str(len(models)))
            mlflow.log_param("pipeline_version", "v5.5")

            # ローカルにもモデル保存 (MLflow Model Registry不使用時のフォールバック)
            self._save_models_local(models, quality_screen, regime_det, train_start, train_end)

    def _save_models_local(
        self,
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_start: str,
        train_end: str,
    ) -> Path:
        """全モデルをローカルディレクトリに保存 (MLflow非依存)"""
        models_dir = self.model_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        # 古いモデルファイルを完全に削除 (アンサンブル/非アンサンブル間の不整合を防止)
        for old_file in models_dir.glob("*.lgb"):
            old_file.unlink()
        for old_file in models_dir.glob("*.joblib"):
            old_file.unlink()
        for old_file in models_dir.glob("*.json"):
            old_file.unlink()

        saved: dict[str, object] = {}
        ensemble_keys: set[str] = set()
        for surface, sub in models.items():
            saved[f"stage1_{surface}"] = sub.stage1.models.get(surface)
            saved[f"market_{surface}"] = sub.market.model
            if sub.use_ensemble:
                ensemble_keys.add(f"win_hit_{surface}")
                if sub.place is not None:
                    ensemble_keys.add(f"place_hit_{surface}")
            saved[f"win_hit_{surface}"] = sub.win.hit_model
            saved[f"win_ret_{surface}"] = sub.win.return_model
            saved[f"ev_corrector_p_{surface}"] = sub.ev_corrector.p_correction_model
            saved[f"ev_corrector_e_{surface}"] = sub.ev_corrector.e_correction_model
            if sub.place_ev_corrector is not None:
                saved[f"place_ev_corrector_p_{surface}"] = sub.place_ev_corrector.p_correction_model
                saved[f"place_ev_corrector_e_{surface}"] = sub.place_ev_corrector.e_correction_model
            if sub.place is not None:
                saved[f"place_hit_{surface}"] = sub.place.hit_model
                saved[f"place_ret_{surface}"] = sub.place.return_model
            if sub.wide is not None:
                saved[f"wide_hit_{surface}"] = sub.wide.hit_model
                saved[f"wide_ret_{surface}"] = sub.wide.return_model
            # PlaceAbilityModel (sklearn) は joblib で保存
            calibrated = None
            if sub.place_ability is not None:
                calibrated = sub.place_ability._calibrated or sub.place_ability._model
                if calibrated is not None:
                    joblib.dump(
                        calibrated,
                        models_dir / f"place_ability_{surface}.joblib",
                    )
            if sub.place_selection_gate is not None and sub.place_selection_gate.is_trained:
                sub.place_selection_gate.save(models_dir / f"place_selection_gate_{surface}.joblib")

            # --- WinSelectionGate (local) ---
            if sub.win_selection_gate is not None and sub.win_selection_gate.is_trained:
                sub.win_selection_gate.save(models_dir / f"win_selection_gate_{surface}.joblib")

            if sub.win_selection_policy is not None and sub.win_selection_policy.is_trained:
                sub.win_selection_policy.save(models_dir / f"win_selection_policy_{surface}.joblib")

            if sub.win_profit_selector is not None and sub.win_profit_selector.is_trained:
                sub.win_profit_selector.save(models_dir / f"win_profit_selector_{surface}.joblib")

            if sub.win_segment_calibrator is not None and sub.win_segment_calibrator.is_trained:
                sub.win_segment_calibrator.save(
                    models_dir / f"win_segment_calibrator_{surface}.joblib"
                )

            # Benter Combination (JSON)
            if sub.benter_combo is not None:
                sub.benter_combo.save(models_dir / f"benter_combo_{surface}.json")

            # Isotonic Calibrator (joblib)
            if sub.isotonic_calibrator is not None:
                joblib.dump(
                    sub.isotonic_calibrator,
                    models_dir / f"isotonic_place_{surface}.joblib",
                )

            # v5: Temperature Scaler (JSON)
            if sub.temperature_scaler is not None:
                sub.temperature_scaler.save(models_dir / f"temp_scale_{surface}.json")

            # Win Benter Combination (JSON)
            if sub.win_benter is not None:
                sub.win_benter.save(models_dir / f"benter_combo_win_{surface}.json")

            # Win Isotonic Calibrator (joblib)
            if sub.win_isotonic_calibrator is not None:
                joblib.dump(
                    sub.win_isotonic_calibrator,
                    models_dir / f"isotonic_win_{surface}.joblib",
                )

            # Win Temperature Scaler (JSON)
            if sub.win_temperature_scaler is not None:
                sub.win_temperature_scaler.save(models_dir / f"temp_scale_win_{surface}.json")

            # Phase 19: EV Isotonic Calibrator (joblib)
            if sub.ev_isotonic_calibrator is not None:
                joblib.dump(
                    sub.ev_isotonic_calibrator,
                    models_dir / f"ev_isotonic_{surface}.joblib",
                )

            # Phase 19: EV Odds Band Scales (JSON)
            if _valid_ev_band_scales(sub.ev_odds_band_scales):
                with open(models_dir / f"ev_odds_band_scales_{surface}.json", "w") as f:
                    json.dump(sub.ev_odds_band_scales, f, indent=2)
            elif sub.ev_odds_band_scales is not None:
                logger.warning("Skipping degenerate EV odds band scales for %s", surface)

            # INTER-03: TargetEncoder (joblib)
            if sub.target_encoder is not None:
                joblib.dump(
                    sub.target_encoder,
                    models_dir / f"target_encoder_{surface}.joblib",
                )

        saved["race_quality"] = quality_screen.model
        saved["regime_detector"] = regime_det.model

        # LightGBM モデルを model.lgb として保存; StackedEnsemble は joblib
        for name, model in saved.items():
            if model is None:
                continue
            if name.startswith("place_ability"):
                continue  # joblib で既に保存済み
            if name in ensemble_keys:
                # StackedEnsemble — joblib pickle で保存
                joblib.dump(model, models_dir / f"{name}.joblib")
            elif hasattr(model, "save_model"):
                model.save_model(str(models_dir / f"{name}.lgb"))

        # Phase 21: ConformalEVModel保存 + PFP SHA256 (per D-10)
        for surface, sub in models.items():
            if sub.conformal_ev_model is not None:
                sub.conformal_ev_model.save(models_dir, surface)

        # Phase 21: CQR model parameter SHA256 for PFP tamper detection (per D-10)
        cqr_checksums: dict[str, str] = {}
        for surface, sub in models.items():
            if sub.conformal_ev_model is not None:
                cqr_params_path = models_dir / f"cqr_params_{surface}.json"
                if cqr_params_path.is_file():
                    import hashlib

                    sha256 = hashlib.sha256(cqr_params_path.read_bytes()).hexdigest()
                    cqr_checksums[surface] = sha256
                    logger.info("CQR params SHA256 for %s: %s", surface, sha256[:16])

        # cqr_checksumsをstrategy manifestに追加 (存在する場合)
        if cqr_checksums:
            manifest_path = Path("data/strategy_manifest.json")
            if manifest_path.is_file():
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        manifest = json.load(f)
                    manifest["cqr_checksums"] = cqr_checksums
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2, ensure_ascii=False)
                    logger.info("CQR checksums written to strategy_manifest.json")
                except Exception as e:
                    logger.warning("Failed to update strategy manifest with CQR checksums: %s", e)
            else:
                logger.debug("strategy_manifest.json not found, CQR checksums logged only")

        # メタ情報
        meta = {
            "train_start": train_start,
            "train_end": train_end,
            "surfaces": list(models.keys()),
            "quality_threshold": quality_screen.threshold,
            "saved_at": pd.Timestamp.now().isoformat(),
            "use_ensemble": all(sub.use_ensemble for sub in models.values()),
        }
        with open(models_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Models saved to %s", models_dir)
        return models_dir
