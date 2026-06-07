"""FeatureBuilder — BT/PT/TrainingPipeline 共通の特徴量生成エントリポイント (D-01, D-02, D-04)。

build_for_training() と build_for_inference() を提供し、
3コピーの特徴量構築分岐 (BacktestEngine.prepare_data / .run 内部 /
TrainingPipeline._train_submodel / PaperPredictor.setup) を統一する。

13 のエンリッチメントモジュールを _train_submodel と同一の順序で実行する。
BloodlineFeatures は FeatureEngine.build_all() Group B で暗黙的に実行され、
blood_* カラムが build_all() 出力に含まれるため _enrich_features() では
明示的なステップを持たない。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from db.parquet_store import ParquetStore
from domain.types import POST_RACE_COLS
from features.feature_engine import FeatureEngine
from features.feature_manifest import FeatureBuildResult, FeatureManifest, FeatureState
from models.submodel_manager import SubModelManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers: moisture collation & aggregation (D-06)
# ---------------------------------------------------------------------------


def collate_moisture_rule(
    jra_goal: float | None,
    jra_4c: float | None,
    csv_value: float | None,
) -> str:
    """JRAライブ値と履歴CSV値を照合し、含水率集約規則を確定する (D-06)。

    優先順位: goal > 4c > mean。csv_value が JRA 値のいずれかに近い場合、
    その規則を採用する。近さ判定の閾値は 0.5%。

    Args:
        jra_goal: JRA ライブ dirt_moisture (ゴール前)。
        jra_4c: JRA ライブ dirt_moisture (4コーナー)。
        csv_value: 履歴CSV由来の dirt_moisture。

    Returns:
        採用規則 ("goal", "4c", "mean")。

    Raises:
        ValueError: 照合不能 (全規則で閾値外) の場合。
    """
    threshold = 0.5

    # 重複データ不足 → デフォルト規則 + 警告
    if csv_value is None:
        logger.warning(
            "collate_moisture_rule: csv_value is None — insufficient data, defaulting to mean"
        )
        return "mean"

    # 各規則の距離を計算
    candidates: list[tuple[float, str]] = []
    if jra_goal is not None:
        candidates.append((abs(jra_goal - csv_value), "goal"))
    if jra_4c is not None:
        candidates.append((abs(jra_4c - csv_value), "4c"))
    if jra_goal is not None and jra_4c is not None:
        mean_val = (jra_goal + jra_4c) / 2.0
        candidates.append((abs(mean_val - csv_value), "mean"))

    if not candidates:
        raise ValueError(
            f"照合不能: jra_goal={jra_goal}, jra_4c={jra_4c}, csv_value={csv_value} "
            f"— 規則候補なし"
        )

    # 最も近い規則を選択
    candidates.sort(key=lambda x: x[0])
    best_dist, best_rule = candidates[0]

    if best_dist > threshold:
        raise ValueError(
            f"照合不能: jra_goal={jra_goal}, jra_4c={jra_4c}, csv_value={csv_value} "
            f"— 最近規則({best_rule})でも距離{best_dist:.2f} > 閾値{threshold}"
        )

    return best_rule


def aggregate_dirt_moisture(
    goal: float | None,
    four_c: float | None,
    rule: str,
) -> float | None:
    """確定した規則に基づいて dirt_moisture を算出する。

    Args:
        goal: ゴール前含水率。
        four_c: 4コーナー含水率。
        rule: 集約規則 ("goal", "4c", "mean")。

    Returns:
        算出された dirt_moisture。両方 None の場合は None。
    """
    if rule == "goal":
        return goal
    elif rule == "4c":
        return four_c
    elif rule == "mean":
        if goal is not None and four_c is not None:
            return (goal + four_c) / 2.0
        elif goal is not None:
            return goal
        elif four_c is not None:
            return four_c
        return None
    else:
        logger.warning("Unknown moisture rule '%s', falling back to mean", rule)
        return aggregate_dirt_moisture(goal, four_c, "mean")


class FeatureBuilder:
    """特徴量生成の単一エントリポイント。

    Usage:
        builder = FeatureBuilder(store=parquet_store)
        result = builder.build_for_training(race_df, entry_df, odds_df)
        result = builder.build_for_inference(
            race_df, entry_df, odds_df, feature_state=state
        )
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._feat_engine = FeatureEngine()
        self._submodel_mgr = SubModelManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_for_training(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        *,
        odds_ts_df: pd.DataFrame | None = None,
        preserve_columns: list[str] | None = None,
        feature_version: str = "1.0",
    ) -> FeatureBuildResult:
        """学習用特徴量を生成。

        Args:
            race_df: レースメタデータ。
            entry_df: 出走馬データ。
            odds_df: オッズスナップショット。
            odds_ts_df: オッズ時系列データ (省略可)。
            preserve_columns: 保持するターゲット列。None 時は
                ["kakuteijyuni", "confirmed_odds"] を使用。
            feature_version: 特徴量定義バージョン。

        Returns:
            FeatureBuildResult (frame + manifest)。
        """
        if preserve_columns is None:
            preserve_columns = ["kakuteijyuni", "confirmed_odds"]
        return self._build(
            race_df,
            entry_df,
            odds_df,
            odds_ts_df=odds_ts_df,
            preserve_columns=preserve_columns,
            feature_state=None,
            feature_version=feature_version,
        )

    def build_for_inference(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        feature_state: FeatureState,
        *,
        odds_ts_df: pd.DataFrame | None = None,
        live_track_conditions: pd.DataFrame | None = None,
        feature_version: str = "1.0",
    ) -> FeatureBuildResult:
        """推論用特徴量を生成。

        POST_RACE 列が含まれていれば除去する。feature_state は必須。

        Args:
            race_df: レースメタデータ。
            entry_df: 出走馬データ。
            odds_df: オッズスナップショット。
            feature_state: 学習期間統計 (必須)。
            odds_ts_df: オッズ時系列データ (省略可)。
            feature_version: 特徴量定義バージョン。

        Returns:
            FeatureBuildResult (frame + manifest)。frame に POST_RACE 列は含まれない。

        Raises:
            ValueError: feature_state が None の場合。
        """
        if feature_state is None:
            raise ValueError(
                "feature_state is required for inference — "
                "use FeatureState.from_submodel_set() to create"
            )
        result = self._build(
            race_df,
            entry_df,
            odds_df,
            odds_ts_df=odds_ts_df,
            preserve_columns=None,
            feature_state=feature_state,
            feature_version=feature_version,
            live_track_conditions=live_track_conditions,
        )
        # POST_RACE 列の除去 (D-01)
        post_race_present = [c for c in result.frame.columns if c in POST_RACE_COLS]
        if post_race_present:
            result.frame.drop(columns=post_race_present, inplace=True)
            # manifest を再生成 (POST_RACE 除外後のカラム構成)
            new_manifest = FeatureManifest.from_dataframe(
                result.frame, feature_version
            )
            result = FeatureBuildResult(frame=result.frame, manifest=new_manifest)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        *,
        odds_ts_df: pd.DataFrame | None = None,
        preserve_columns: list[str] | None = None,
        feature_state: FeatureState | None = None,
        feature_version: str = "1.0",
        live_track_conditions: pd.DataFrame | None = None,
    ) -> FeatureBuildResult:
        """共通特徴量構築ロジック。

        Phase 1: 基礎特徴量 (FeatureEngine.build_all + distance_band)
        Phase 2: 13 エンリッチメントモジュール (_enrich_features)
        Phase 3: dtype 正規化 + FeatureManifest 生成
        """
        # Phase 1: 基礎特徴量
        feat_df = self._build_base_features(
            race_df, entry_df, odds_df,
            odds_ts_df=odds_ts_df,
            preserve_columns=preserve_columns,
        )

        # Phase 2: エンリッチメント
        feat_df = self._enrich_features(
            feat_df,
            race_df,
            entry_df,
            feature_state=feature_state,
            live_track_conditions=live_track_conditions,
        )

        # Phase 3: dtype 正規化 (object型数値列 → float64)
        for col in feat_df.columns:
            if feat_df[col].dtype == object:
                try:
                    feat_df[col] = feat_df[col].astype(float)
                except (ValueError, TypeError):
                    pass

        # FeatureManifest 生成
        manifest = FeatureManifest.from_dataframe(feat_df, feature_version)
        return FeatureBuildResult(frame=feat_df, manifest=manifest)

    def _build_base_features(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        *,
        odds_ts_df: pd.DataFrame | None = None,
        preserve_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Phase 1: FeatureEngine.build_all() + distance_band 特徴量。"""
        feat_df = self._feat_engine.build_all(
            race_df,
            entry_df,
            odds_df,
            odds_ts_df=odds_ts_df,
            store=self.store,
            preserve_columns=preserve_columns,
        )
        feat_df = self._submodel_mgr.add_distance_band_features(feat_df)
        return feat_df

    def _enrich_features(
        self,
        feat_df: pd.DataFrame,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        *,
        feature_state: FeatureState | None = None,
        live_track_conditions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Phase 2: 13 エンリッチメントモジュールを _train_submodel と同一順序で実行。

        BloodlineFeatures (blood_* カラム) は FeatureEngine.build_all() Group B で
        暗黙的に実行されるため、このメソッドでは明示的なステップを持たない。
        build_all() → _build_base_features() → このメソッドの順で呼ばれるため、
        feat_df には既に blood_* カラムが含まれている。
        """
        df = feat_df

        # (a) HorseHistoryFeatures
        from features.horse_history_features import HorseHistoryFeatures

        hist = HorseHistoryFeatures(store=self.store, n_past=5)
        race_ids = df["race_id"].unique()
        hist_df = hist.compute(race_df, entry_df, race_ids)
        df = df.merge(hist_df, on=["race_id", "umaban"], how="left")

        # (b) HorseHistoryFeatures.add_race_transforms — 直後に実行
        df = HorseHistoryFeatures.add_race_transforms(df)

        # (c) PaceAptitudeFeatures
        from features.pace_aptitude_features import PaceAptitudeFeatures

        pace_feat = PaceAptitudeFeatures(store=self.store)
        pace_df = pace_feat.compute_batch(df)
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
                c for c in [
                    "kettonum", "race_id",
                    "pace_aptitude", "front_pace_wr", "closing_pace_wr",
                    "pace_corner_stability", "pace_closing_power",
                    "pace_position_consistency",
                ] if c in pace_df.columns
            ]
            df = df.merge(pace_df[pace_merge_cols], on=["kettonum", "race_id"], how="left")
        else:
            for col in _pace_drop_cols:
                df[col] = np.nan

        # (d) CourseFeatures
        from features.course_features import CourseFeatures

        course_feat = CourseFeatures(store=self.store)
        course_df = course_feat.compute_batch(df)
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
            df["course_wr"] = np.nan
            df["course_distance_wr"] = np.nan

        # (e) SireFeatures
        from db.readers import load_horses, load_sire_stats
        from features.sire_features import SireFeatures

        sire_stats = load_sire_stats(self.store)
        if not sire_stats.empty:
            horses_df = load_horses(self.store)
            sire_feat = SireFeatures(sire_stats)
            sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
            df["sire_id"] = df["kettonum"].map(sire_map)
            bms_source_col = (
                "ketto3infohansyokunum5"
                if "ketto3infohansyokunum5" in horses_df.columns
                else "ketto3infohansyokunum3"
            )
            bms_map = horses_df.set_index("kettonum")[bms_source_col]
            df["bms_id"] = df["kettonum"].map(bms_map)
            sire_result = sire_feat.compute_batch(df)
            _sire_cols_needed = {
                "sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_prize_avg",
                "bms_wr", "bms_distance_wr", "bms_surface_wr",
                "bms_has_history", "bms_starts_log", "bms_surface_starts_log",
                "bms_distance_starts_log",
            }
            for col in _sire_cols_needed:
                if col in sire_result.columns:
                    df[col] = sire_result[col].values

        # (f) DamPedigreeFeatures
        from features.dam_pedigree_features import FEATURE_COLS as DAM_PED_FEATURE_COLS
        from features.dam_pedigree_features import DamPedigreeFeatures

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

        # (g) RecordFeatures
        from features.record_features import FEATURE_COLS as RECORD_FEATURE_COLS
        from features.record_features import RecordFeatures

        record_feat = RecordFeatures(self.store)
        record_df = record_feat.compute(df)
        _record_drop_cols = [c for c in RECORD_FEATURE_COLS if c in df.columns]
        if _record_drop_cols:
            df.drop(columns=_record_drop_cols, inplace=True)
        if not record_df.empty:
            df = df.merge(record_df, on=["race_id"], how="left")
        else:
            for col in RECORD_FEATURE_COLS:
                df[col] = np.nan

        # (h) TrackConditionFeatures
        from features.track_condition_features import (
            _compute_track_month_stats,
            _compute_track_stats,
            compute_race_condition_features,
            compute_track_condition_features,
        )

        _track_stats: dict | None = None
        _track_month_stats: dict | None = None
        if feature_state is not None:
            # 推論時: 学習期間統計を使用
            _track_stats = feature_state.track_stats
            _track_month_stats = feature_state.track_month_stats
        else:
            # 学習時: データから統計を計算
            if "turf_cushion" in df.columns and "trackcd" in df.columns:
                _track_stats = _compute_track_stats(df)
            if "trackcd" in df.columns and (
                "turf_cushion" in df.columns or "dirt_moisture" in df.columns
            ):
                _track_month_stats = _compute_track_month_stats(df)
        df = compute_track_condition_features(
            df, track_stats=_track_stats, track_month_stats=_track_month_stats
        )
        df = compute_race_condition_features(df)

        # (h.5) Live track condition override (D-07)
        if live_track_conditions is not None:
            df = self._merge_live_track_conditions(df, live_track_conditions)

        # (i) InteractionFeatures
        from features.interaction_features import compute_interaction_features

        df = compute_interaction_features(df)

        # (j) MiningFeatures
        from features.mining_features import FEATURE_COLS as MINING_FEATURE_COLS
        from features.mining_features import MiningFeatures

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

        # (k) RelativeFeatures
        from features.relative_features import compute_relative_features

        df = compute_relative_features(df)

        # (l) JockeyContextFeatures
        from features.jockey_context_features import JockeyContextFeatures

        jockey_ctx = JockeyContextFeatures(self.store)
        jockey_df = jockey_ctx.compute(entry_df)
        _jockey_merge_cols = [
            c for c in jockey_df.columns
            if c not in df.columns or c in {"race_id", "umaban"}
        ]
        if _jockey_merge_cols:
            df = df.merge(jockey_df[_jockey_merge_cols], on=["race_id", "umaban"], how="left")

        # (m) TrainerContextFeatures
        from features.trainer_context_features import TrainerContextFeatures

        trainer_ctx = TrainerContextFeatures(self.store)
        trainer_df = trainer_ctx.compute(entry_df)
        _trainer_merge_cols = [
            c for c in trainer_df.columns
            if c not in df.columns or c in {"race_id", "umaban"}
        ]
        if _trainer_merge_cols:
            df = df.merge(trainer_df[_trainer_merge_cols], on=["race_id", "umaban"], how="left")

        # (n) JockeyTrainerComboFeatures
        from features.jockey_trainer_combo import JockeyTrainerComboFeatures

        jt_combo = JockeyTrainerComboFeatures(self.store)
        jt_df = jt_combo.compute(entry_df)
        _jt_merge_cols = [
            c for c in jt_df.columns
            if c not in df.columns or c in {"race_id", "umaban"}
        ]
        if _jt_merge_cols:
            df = df.merge(jt_df[_jt_merge_cols], on=["race_id", "umaban"], how="left")

        return df

    def _merge_live_track_conditions(
        self,
        df: pd.DataFrame,
        live_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """ライブトラック条件を履歴データにマージする (D-07)。

        マージキー: race_id。ライブ値が非 NaN の場合に履歴値を上書き。
        NaN の場合は履歴値を保持。live_df のみに存在する race_id は追加しない
        (推論対象は df 側で決定)。

        Args:
            df: 履歴特徴量 DataFrame。
            live_df: ライブトラック条件 DataFrame。

        Returns:
            マージ済み DataFrame。
        """
        if live_df is None or live_df.empty:
            return df

        # マージ対象列を特定 (race_id 除外)
        merge_cols = [c for c in live_df.columns if c != "race_id" and c in df.columns]
        if not merge_cols:
            return df

        # left join on race_id (履歴側が主)
        merged = df.merge(
            live_df[["race_id"] + merge_cols],
            on="race_id",
            how="left",
            suffixes=("", "_live"),
        )

        # ライブ値で上書き (NaN は履歴値を保持)
        for col in merge_cols:
            live_col = f"{col}_live"
            if live_col in merged.columns:
                # ライブ値が非 NaN の場合のみ上書き
                mask = merged[live_col].notna()
                merged.loc[mask, col] = merged.loc[mask, live_col]
                merged.drop(columns=[live_col], inplace=True)

        return merged
