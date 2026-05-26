"""MLflow から TrainedModelsV5 をロードする"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import mlflow
import numpy as np

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


def _valid_ev_band_scales(scales: dict[str, float] | None) -> bool:
    if not scales:
        return False
    values = np.array([float(v) for v in scales.values()], dtype=float)
    return bool(np.isfinite(values).all() and not np.allclose(values, 0.0))


@dataclass
class ModelInfo:
    """ロードしたモデルのメタ情報"""

    mlflow_run_id: str
    train_start: str
    train_end: str
    loaded_at: str


class ModelLoader:
    """MLflow から TrainedModelsV5 を構築してロード"""

    def __init__(self, tracking_uri: str = "file:///mlruns") -> None:
        mlflow.set_tracking_uri(tracking_uri)

    def load(
        self, run_id: str | None = None, *, use_ensemble: bool | None = None
    ) -> tuple[TrainedModelsV5, ModelInfo]:
        """学習済みモデルを読み込み、TrainedModelsV5 を再構築。

        優先: ローカルディレクトリ (data/models/) → MLflow run artifacts
        """
        # 1. ローカルディレクトリから読み込み
        models_dir = Path("data/models")
        if models_dir.is_dir() and (models_dir / "meta.json").is_file():
            return self.load_from_dir(models_dir, use_ensemble_override=use_ensemble)

        # 2. MLflow 経由 (フォールバック)
        if run_id is None:
            run_id = self._find_latest_run()

        # MLflow API経由でパラメータを取得
        train_end = "unknown"
        train_start = "2020-01-01"
        quality_threshold = 0.0
        artifact_uri = ""

        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
            train_end = params.get("train_end", "unknown")
            train_start = params.get("train_start", "2020-01-01")
            quality_threshold = float(params.get("quality_threshold", "0.0"))
            artifact_uri = mlflow.get_artifact_uri(run_id)
        except Exception:
            # Fallback: ファイルシステムから直接読み込み
            artifact_uri, train_start, train_end, quality_threshold = self._resolve_run_from_fs(
                run_id
            )

        surfaces = ["turf", "dirt"]

        from domain.models import SubmodelSet, TrainedModelsV5
        from models.conformal_ev_model import ConformalEVModel
        from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.place_selection_gate import PlaceSelectionGateModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel
        from models.win_profit_selector import WinProfitSelector
        from models.win_segment_calibrator import WinSegmentCalibrator
        from models.win_selection_gate import WinSelectionGateModel
        from models.win_selection_policy import WinSelectionPolicy

        submodels: dict[str, SubmodelSet] = {}
        for surface in surfaces:
            # MarketModel
            market = MarketModel()
            market.model = self._load_lgbm(f"{artifact_uri}/market_{surface}")

            # AbilityModel (per-surface booster)
            ability = AbilityModel()
            ability.models = {surface: self._load_lgbm(f"{artifact_uri}/stage1_{surface}")}

            # WinTwoStageModel
            win = WinTwoStageModel()
            win.hit_model = self._load_lgbm(f"{artifact_uri}/win_hit_{surface}")
            win.return_model = self._load_lgbm(f"{artifact_uri}/win_ret_{surface}")

            # EVCorrectionModel
            ev_corr = EVCorrectionModel()
            ev_corr.p_correction_model = self._load_lgbm(f"{artifact_uri}/ev_corrector_p_{surface}")
            ev_corr.e_correction_model = self._load_lgbm(f"{artifact_uri}/ev_corrector_e_{surface}")
            ev_corr._trained = True

            # PlaceEVCorrectionModel (backward compatible with old MLflow runs)
            try:
                place_ev_corr = PlaceEVCorrectionModel()
                place_ev_corr.p_correction_model = self._load_lgbm(
                    f"{artifact_uri}/place_ev_corrector_p_{surface}"
                )
                place_ev_corr.e_correction_model = self._load_lgbm(
                    f"{artifact_uri}/place_ev_corrector_e_{surface}"
                )
                place_ev_corr._trained = True
            except Exception:
                logger.warning(
                    "PlaceEVCorrectionModel artifacts not found for %s, using passthrough", surface
                )
                place_ev_corr = PlaceEVCorrectionModel()

            # PlaceTwoStageModel (optional — may not exist when betting_target=win)
            place = None
            try:
                place = PlaceTwoStageModel()
                place.hit_model = self._load_lgbm(f"{artifact_uri}/place_hit_{surface}")
                place.return_model = self._load_lgbm(f"{artifact_uri}/place_ret_{surface}")
            except Exception:
                logger.info("Place model files not found for %s, skipping", surface)
                place = None

            place_selection_gate = None
            try:
                gate_dir = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/place_selection_gate_{surface}"
                )
            except Exception:
                try:
                    gate_dir = self._find_artifact_dir(run_id, f"place_selection_gate_{surface}")
                except Exception:
                    gate_dir = None
            if gate_dir is not None:
                gate_files = list(Path(gate_dir).glob("*.joblib"))
                if gate_files:
                    try:
                        place_selection_gate = PlaceSelectionGateModel.load(gate_files[0])
                    except Exception:
                        logger.warning("Failed to load PlaceSelectionGateModel for %s", surface)

            # --- WinSelectionGate (MLflow) ---
            win_selection_gate = None
            try:
                wsg_dir = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/win_selection_gate_{surface}"
                )
            except Exception:
                try:
                    wsg_dir = self._find_artifact_dir(run_id, f"win_selection_gate_{surface}")
                except Exception:
                    wsg_dir = None
            if wsg_dir is not None:
                wsg_files = list(Path(wsg_dir).glob("*.joblib"))
                if wsg_files:
                    try:
                        win_selection_gate = WinSelectionGateModel.load(wsg_files[0])
                    except Exception:
                        logger.warning("Failed to load WinSelectionGateModel for %s", surface)

            # --- WinSelectionPolicy (MLflow) ---
            win_selection_policy = None
            try:
                wsp_dir = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/win_selection_policy_{surface}"
                )
            except Exception:
                try:
                    wsp_dir = self._find_artifact_dir(run_id, f"win_selection_policy_{surface}")
                except Exception:
                    wsp_dir = None
            if wsp_dir is not None:
                wsp_files = list(Path(wsp_dir).glob("*.joblib"))
                if wsp_files:
                    try:
                        win_selection_policy = WinSelectionPolicy.load(wsp_files[0])
                    except Exception:
                        logger.warning("Failed to load WinSelectionPolicy for %s", surface)

            # --- WinProfitSelector (MLflow) ---
            win_profit_selector = None
            try:
                wps_dir = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/win_profit_selector_{surface}"
                )
            except Exception:
                try:
                    wps_dir = self._find_artifact_dir(run_id, f"win_profit_selector_{surface}")
                except Exception:
                    wps_dir = None
            if wps_dir is not None:
                wps_files = list(Path(wps_dir).glob("*.joblib"))
                if wps_files:
                    try:
                        win_profit_selector = WinProfitSelector.load(wps_files[0])
                    except Exception:
                        logger.warning("Failed to load WinProfitSelector for %s", surface)

            # --- WinSegmentCalibrator (MLflow) ---
            win_segment_calibrator = None
            try:
                wsc_dir = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/win_segment_calibrator_{surface}"
                )
            except Exception:
                try:
                    wsc_dir = self._find_artifact_dir(run_id, f"win_segment_calibrator_{surface}")
                except Exception:
                    wsc_dir = None
            if wsc_dir is not None:
                wsc_files = list(Path(wsc_dir).glob("*.joblib"))
                if wsc_files:
                    try:
                        win_segment_calibrator = WinSegmentCalibrator.load(wsc_files[0])
                    except Exception:
                        logger.warning("Failed to load WinSegmentCalibrator for %s", surface)

            # PlaceAbilityModel (joblib artifact)
            pa = PlaceAbilityModel()
            try:
                pa_dir = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/place_ability_{surface}"
                )
            except Exception:
                # Fallback: ファイルシステムから直接読み込み
                pa_dir = self._find_artifact_dir(run_id, f"place_ability_{surface}")
            pa_files = list(Path(pa_dir).glob("*.joblib"))
            if pa_files:
                pa._calibrated = joblib.load(pa_files[0])
            else:
                logger.warning("PlaceAbilityModel artifact not found for %s", surface)

            # WideTwoStageModel
            wide = WideTwoStageModel()
            wide.hit_model = self._load_lgbm(f"{artifact_uri}/wide_hit_{surface}")
            wide.return_model = self._load_lgbm(f"{artifact_uri}/wide_ret_{surface}")

            # Phase 21: ConformalEVModel (CQR)
            conformal_ev = None
            try:
                # Try loading CQR models (new format: per-surface files)
                q_low_path = f"{artifact_uri}/cqr_quantile_low_{surface}"
                q_high_path = f"{artifact_uri}/cqr_quantile_high_{surface}"
                cqr_params_uri = f"runs:/{run_id}/cqr_params_{surface}.json"

                import lightgbm as lgb

                obj = ConformalEVModel()
                obj.q_low_model = (
                    lgb.Booster(model_file=q_low_path)
                    if Path(q_low_path).is_file()
                    else mlflow.lightgbm.load_model(q_low_path)
                )
                obj.q_high_model = (
                    lgb.Booster(model_file=q_high_path)
                    if Path(q_high_path).is_file()
                    else mlflow.lightgbm.load_model(q_high_path)
                )
                params_path = mlflow.artifacts.download_artifacts(cqr_params_uri)
                with open(params_path, encoding="utf-8") as f:
                    cqr_data = json.load(f)
                obj.alpha = cqr_data["alpha"]
                obj._calibration_quantile_90 = cqr_data["calibration_quantile_90"]
                obj._calibration_quantile_80 = cqr_data["calibration_quantile_80"]
                obj._residual_quantile_90 = cqr_data.get("residual_quantile_90", 0.0)
                obj._residual_quantile_80 = cqr_data.get("residual_quantile_80", 0.0)
                obj.feature_cols = cqr_data.get("feature_cols")
                obj._calibrated = cqr_data.get("_calibrated", True)
                conformal_ev = obj
            except Exception as e:
                logger.warning(
                    "CQR model files not found for %s (%s), trying legacy format",
                    surface,
                    e,
                )
                # Fallback: try legacy confidence_params.json
                try:
                    conf_path = mlflow.artifacts.download_artifacts(
                        f"runs:/{run_id}/confidence_params.json"
                    )
                    with open(conf_path) as f:
                        conf_data = json.load(f)
                    legacy = ConformalEVModel()
                    legacy.alpha = conf_data["alpha"]
                    legacy._calibrated = False  # No actual CQR models; will use fallback
                    conformal_ev = legacy
                    logger.info(
                        "Loaded legacy confidence params as ConformalEVModel for %s",
                        surface,
                    )
                except Exception:
                    logger.info("ConformalEVModel not found for surface=%s, skipping", surface)

            # Benter Combination (Place)
            benter_combo = None
            try:
                bent_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/benter_combo_{surface}.json"
                )
                from models.benter_combination import BenterCombination

                benter_combo = BenterCombination.load(Path(bent_path))
            except Exception:
                pass

            # Isotonic Calibrator (Place)
            isotonic_calibrator = None
            try:
                iso_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/isotonic_place_{surface}.joblib"
                )
                isotonic_calibrator = joblib.load(iso_path)
            except Exception:
                pass

            # Temperature Scaler (Place)
            temperature_scaler = None
            try:
                temp_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/temp_scale_{surface}.json"
                )
                from models.benter_combination import TemperatureScaling

                temperature_scaler = TemperatureScaling.load(Path(temp_path))
            except Exception:
                pass

            # Win Benter Combination
            win_benter = None
            try:
                wb_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/benter_combo_win_{surface}.json"
                )
                from models.benter_combination import BenterCombination

                win_benter = BenterCombination.load(Path(wb_path))
            except Exception:
                pass

            # Win Calibrator (Beta or Isotonic)
            win_isotonic_calibrator = None
            try:
                wiso_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/isotonic_win_{surface}.joblib"
                )
                win_isotonic_calibrator = joblib.load(wiso_path)
            except Exception:
                pass

            # Win Temperature Scaler
            win_temperature_scaler = None
            try:
                wtemp_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/temp_scale_win_{surface}.json"
                )
                from models.benter_combination import TemperatureScaling

                win_temperature_scaler = TemperatureScaling.load(Path(wtemp_path))
            except Exception:
                pass

            # Phase 19: EV Isotonic Calibrator (MLflow)
            ev_isotonic_calibrator = None
            try:
                eviso_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/ev_isotonic_{surface}.joblib"
                )
                ev_isotonic_calibrator = joblib.load(eviso_path)
            except Exception:
                pass

            # Phase 19: EV Odds Band Scales (MLflow)
            ev_odds_band_scales = None
            try:
                band_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/ev_odds_band_scales_{surface}.json"
                )
                with open(band_path) as f:
                    ev_odds_band_scales = json.load(f)
                if not _valid_ev_band_scales(ev_odds_band_scales):
                    logger.warning("Ignoring degenerate EV odds band scales for %s", surface)
                    ev_odds_band_scales = None
            except Exception:
                pass

            # Wire EV Isotonic + band scales into EVCorrectionModel instance
            ev_corr.ev_isotonic_calibrator = ev_isotonic_calibrator
            ev_corr.ev_odds_band_scales = ev_odds_band_scales

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                place_ev_corrector=place_ev_corr,
                wide=wide,
                conformal_ev_model=conformal_ev,  # Phase 21: CQR model
                place_selection_gate=place_selection_gate,
                benter_combo=benter_combo,
                isotonic_calibrator=isotonic_calibrator,
                temperature_scaler=temperature_scaler,
                win_benter=win_benter,
                win_isotonic_calibrator=win_isotonic_calibrator,
                win_temperature_scaler=win_temperature_scaler,
                win_selection_gate=win_selection_gate,
                win_selection_policy=win_selection_policy,
                win_profit_selector=win_profit_selector,
                win_segment_calibrator=win_segment_calibrator,
                ev_isotonic_calibrator=ev_isotonic_calibrator,
                ev_odds_band_scales=ev_odds_band_scales,
            )
        quality = RaceQualityScreener()
        quality.model = self._load_lgbm(f"{artifact_uri}/race_quality")
        quality.threshold = quality_threshold

        # RegimeDetector
        regime = RegimeDetector()
        regime.model = self._load_lgbm(f"{artifact_uri}/regime_detector")

        models = TrainedModelsV5(
            submodels=submodels,
            quality_screener=quality,
            regime_detector=regime,
            train_period=(train_start, train_end),
        )

        info = ModelInfo(
            mlflow_run_id=run_id,
            train_start=train_start,
            train_end=train_end,
            loaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        return models, info

    def _find_latest_run(self) -> str:
        """最新の成功 run ID を取得

        MLflowのsearch_runsが失敗した場合は、ファイルシステムから直接検索。
        """
        import pandas as pd
        from mlflow.entities import ViewType

        try:
            df: pd.DataFrame = mlflow.search_runs(  # type: ignore[assignment]
                order_by=["start_time DESC"],
                max_results=1,
                filter_string="status = 'FINISHED'",
                run_view_type=ViewType.ACTIVE_ONLY,
            )
            if not df.empty:
                return str(df.iloc[0]["run_id"])
        except Exception:
            pass

        # Fallback: mlruns/ を直接スキャン
        return self._find_latest_run_from_fs()

    @staticmethod
    def _find_latest_run_from_fs() -> str:
        """mlruns/ ディレクトリから最新のrunを直接検索 (MLflowトラッキング不使用)"""

        mlruns_dir = Path("mlruns")
        if not mlruns_dir.exists():
            raise ValueError("mlruns/ directory not found")

        best_run_id: str | None = None
        best_mtime = 0.0

        for exp_dir in mlruns_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name == "models":
                    continue
                # artifacts/ または params/ があれば有効なrunとみなす
                has_artifacts = (run_dir / "artifacts").is_dir()
                has_params = (run_dir / "params").is_dir()
                if not (has_artifacts or has_params):
                    continue
                mtime = run_dir.stat().st_mtime
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_run_id = run_dir.name

        if best_run_id is None:
            raise ValueError("No MLflow runs found in mlruns/")
        return best_run_id

    @staticmethod
    def _find_run_dir(run_id: str) -> Path:
        """run_id から実験ディレクトリを検索してパスを返す"""
        mlruns_dir = Path("mlruns")
        for exp_dir in mlruns_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            run_dir = exp_dir / run_id
            if run_dir.is_dir():
                return run_dir
        raise ValueError(f"Run directory not found for {run_id}")

    @staticmethod
    def _resolve_run_from_fs(run_id: str) -> tuple[str, str, str, float]:
        """ファイルシステムからrunのパラメータを取得"""
        run_dir = ModelLoader._find_run_dir(run_id)
        artifact_uri = run_dir / "artifacts"
        train_start = "2020-01-01"
        train_end = "unknown"
        quality_threshold = 0.0

        params_dir = run_dir / "params"
        if params_dir.is_dir():
            if (params_dir / "train_start").is_file():
                train_start = (params_dir / "train_start").read_text().strip()
            if (params_dir / "train_end").is_file():
                train_end = (params_dir / "train_end").read_text().strip()
            if (params_dir / "quality_threshold").is_file():
                quality_threshold = float((params_dir / "quality_threshold").read_text().strip())

        return str(artifact_uri), train_start, train_end, quality_threshold

    @staticmethod
    def _find_artifact_dir(run_id: str, artifact_name: str) -> Path:
        """run_id からartifactディレクトリを検索"""
        run_dir = ModelLoader._find_run_dir(run_id)
        artifact_dir = run_dir / "artifacts" / artifact_name
        if artifact_dir.is_dir():
            return artifact_dir
        # ファイル名の場合 (e.g., confidence_params.json)
        artifact_file = run_dir / "artifacts" / artifact_name
        if artifact_file.is_file():
            return artifact_file.parent
        raise ValueError(f"Artifact not found: {artifact_name} in {run_dir}")

    @staticmethod
    def _load_lgbm(path: str) -> object:
        """LightGBMモデルをロード (ファイルパスまたはMLflow URI)"""
        import lightgbm as lgb

        p = Path(path)
        if p.is_file():
            # 直接ファイルパス
            return lgb.Booster(model_file=str(p))
        # MLflow URI フォールバック
        return mlflow.lightgbm.load_model(path)

    @staticmethod
    def _load_hit_model(models_dir: Path, name: str, *, use_ensemble: bool = False) -> object:
        """Load hit model from .joblib (StackedEnsemble) or .lgb (LightGBM)."""
        joblib_path = models_dir / f"{name}.joblib"
        lgb_path = models_dir / f"{name}.lgb"

        if use_ensemble:
            if joblib_path.is_file():
                logger.info("Loading StackedEnsemble: %s", joblib_path)
                return joblib.load(joblib_path)
            if lgb_path.is_file():
                logger.warning("use_ensemble=True but .joblib not found, falling back to .lgb")
                return ModelLoader._load_lgbm(str(lgb_path))
        else:
            if lgb_path.is_file():
                return ModelLoader._load_lgbm(str(lgb_path))
            if joblib_path.is_file():
                logger.info("Loading StackedEnsemble (discovered from .joblib): %s", name)
                return joblib.load(joblib_path)

        raise FileNotFoundError(f"No model file found for {name} in {models_dir}")

    def load_from_dir(
        self, models_dir: Path, *, use_ensemble_override: bool | None = None
    ) -> tuple[TrainedModelsV5, ModelInfo]:
        """指定ディレクトリから全モデルをロード。

        backtest スクリプト等で data/models-backtest/ のような
        カスタムディレクトリを指定する場合に使用する。
        """
        from domain.models import SubmodelSet, TrainedModelsV5
        from models.conformal_ev_model import ConformalEVModel
        from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.place_selection_gate import PlaceSelectionGateModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel
        from models.win_profit_selector import WinProfitSelector
        from models.win_segment_calibrator import WinSegmentCalibrator
        from models.win_selection_gate import WinSelectionGateModel
        from models.win_selection_policy import WinSelectionPolicy

        # メタ情報読み込み
        with open(models_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        train_start = meta["train_start"]
        train_end = meta["train_end"]

        # 整合性チェック: meta.json の use_ensemble と .joblib の有無が矛盾していないか
        meta_ensemble = meta.get("use_ensemble", False)
        has_joblib_hit = any(
            (models_dir / f"win_hit_{s}.joblib").is_file() for s in meta.get("surfaces", [])
        )
        if meta_ensemble and not has_joblib_hit:
            raise ValueError(
                "Model file inconsistency: meta.json says use_ensemble=true "
                "but no .joblib hit models found. Models may have been partially "
                "overwritten by a non-ensemble training. Re-run training to fix."
            )
        if not meta_ensemble and has_joblib_hit:
            raise ValueError(
                "Model file inconsistency: meta.json says use_ensemble=false "
                "but stale .joblib hit models found. Models may have been partially "
                "overwritten. Re-run training to fix."
            )
        surfaces = meta["surfaces"]
        use_ensemble = (
            use_ensemble_override
            if use_ensemble_override is not None
            else meta.get("use_ensemble", False)
        )

        submodels: dict[str, SubmodelSet] = {}
        for surface in surfaces:
            # MarketModel
            market = MarketModel()
            market.model = self._load_lgbm(str(models_dir / f"market_{surface}.lgb"))

            # AbilityModel
            ability = AbilityModel()
            ability.models = {surface: self._load_lgbm(str(models_dir / f"stage1_{surface}.lgb"))}

            # WinTwoStageModel
            win = WinTwoStageModel()
            win.hit_model = self._load_hit_model(
                models_dir, f"win_hit_{surface}", use_ensemble=use_ensemble
            )
            win.return_model = self._load_lgbm(str(models_dir / f"win_ret_{surface}.lgb"))

            # EVCorrectionModel
            ev_corr = EVCorrectionModel()
            ev_corr.p_correction_model = self._load_lgbm(
                str(models_dir / f"ev_corrector_p_{surface}.lgb")
            )
            ev_corr.e_correction_model = self._load_lgbm(
                str(models_dir / f"ev_corrector_e_{surface}.lgb")
            )
            ev_corr._trained = True

            # PlaceEVCorrectionModel (backward compatible)
            place_ev_corr_file = models_dir / f"place_ev_corrector_p_{surface}.lgb"
            if place_ev_corr_file.exists():
                place_ev_corr = PlaceEVCorrectionModel()
                place_ev_corr.p_correction_model = self._load_lgbm(
                    str(models_dir / f"place_ev_corrector_p_{surface}.lgb")
                )
                place_ev_corr.e_correction_model = self._load_lgbm(
                    str(models_dir / f"place_ev_corrector_e_{surface}.lgb")
                )
                place_ev_corr._trained = True
            else:
                place_ev_corr = PlaceEVCorrectionModel()

            # PlaceTwoStageModel (optional — may not exist when betting_target=win)
            place = None
            place_hit_file = models_dir / f"place_hit_{surface}.lgb"
            place_hit_joblib = models_dir / f"place_hit_{surface}.joblib"
            place_ret_file = models_dir / f"place_ret_{surface}.lgb"
            if place_hit_file.is_file() or place_hit_joblib.is_file():
                place = PlaceTwoStageModel()
                place.hit_model = self._load_hit_model(
                    models_dir, f"place_hit_{surface}", use_ensemble=use_ensemble
                )
                place.return_model = self._load_lgbm(str(place_ret_file))
            else:
                logger.info(
                    "Place model files not found for %s, skipping (betting_target=win?)",
                    surface,
                )

            # Place calibrator (IsotonicRegression)
            if place is not None:
                calibrator_file = models_dir / f"place_calibrator_{surface}.joblib"
                if calibrator_file.is_file():
                    try:
                        place._place_calibrator = joblib.load(calibrator_file)
                    except Exception:
                        logger.warning("Failed to load %s, skipping", calibrator_file)

            place_selection_gate = None
            gate_file = models_dir / f"place_selection_gate_{surface}.joblib"
            if gate_file.is_file():
                try:
                    place_selection_gate = PlaceSelectionGateModel.load(gate_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", gate_file)

            # --- WinSelectionGate (local) ---
            win_selection_gate = None
            wsg_file = models_dir / f"win_selection_gate_{surface}.joblib"
            if wsg_file.is_file():
                try:
                    win_selection_gate = WinSelectionGateModel.load(wsg_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", wsg_file)

            win_selection_policy = None
            wsp_file = models_dir / f"win_selection_policy_{surface}.joblib"
            if wsp_file.is_file():
                try:
                    win_selection_policy = WinSelectionPolicy.load(wsp_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", wsp_file)

            win_profit_selector = None
            wps_file = models_dir / f"win_profit_selector_{surface}.joblib"
            if wps_file.is_file():
                try:
                    win_profit_selector = WinProfitSelector.load(wps_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", wps_file)

            win_segment_calibrator = None
            wsc_file = models_dir / f"win_segment_calibrator_{surface}.joblib"
            if wsc_file.is_file():
                try:
                    win_segment_calibrator = WinSegmentCalibrator.load(wsc_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", wsc_file)

            # PlaceAbilityModel (joblib)
            pa = PlaceAbilityModel()
            pa_file = models_dir / f"place_ability_{surface}.joblib"
            if pa_file.is_file():
                try:
                    pa._calibrated = joblib.load(pa_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", pa_file)

            # WideTwoStageModel (optional — may not exist when betting_target=win)
            wide = None
            wide_hit_file = models_dir / f"wide_hit_{surface}.lgb"
            wide_hit_joblib = models_dir / f"wide_hit_{surface}.joblib"
            wide_ret_file = models_dir / f"wide_ret_{surface}.lgb"
            if wide_hit_file.is_file() or wide_hit_joblib.is_file():
                wide = WideTwoStageModel()
                wide.hit_model = self._load_lgbm(str(wide_hit_file))
                wide.return_model = self._load_lgbm(str(wide_ret_file))
            else:
                logger.info(
                    "Wide model files not found for %s, skipping (betting_target=win?)",
                    surface,
                )

            # Phase 21: ConformalEVModel (CQR per-surface files)
            conformal_ev = ConformalEVModel.load(models_dir, surface)
            # Fallback: legacy confidence_params.json
            if conformal_ev is None:
                conf_file = models_dir / "confidence_params.json"
                if conf_file.is_file():
                    try:
                        with open(conf_file) as f:
                            conf_data = json.load(f)
                        conformal_ev = ConformalEVModel()
                        conformal_ev.alpha = conf_data["alpha"]
                        conformal_ev._calibrated = False  # No actual CQR models; will use fallback
                        logger.info("Loaded legacy confidence_params.json for %s", surface)
                    except Exception:
                        logger.warning(
                            "Failed to load legacy confidence_params.json for %s",
                            surface,
                        )

            # Benter Combination (JSON)
            benter_combo = None
            benter_file = models_dir / f"benter_combo_{surface}.json"
            if benter_file.is_file():
                try:
                    from models.benter_combination import BenterCombination

                    benter_combo = BenterCombination.load(benter_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", benter_file)

            # Isotonic Calibrator (joblib)
            isotonic_calibrator = None
            iso_file = models_dir / f"isotonic_place_{surface}.joblib"
            if iso_file.is_file():
                try:
                    isotonic_calibrator = joblib.load(iso_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", iso_file)

            # v5: Temperature Scaler (JSON)
            temperature_scaler = None
            temp_file = models_dir / f"temp_scale_{surface}.json"
            if temp_file.is_file():
                try:
                    from models.benter_combination import TemperatureScaling

                    temperature_scaler = TemperatureScaling.load(temp_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", temp_file)

            # Win Benter Combination (JSON)
            win_benter = None
            win_benter_file = models_dir / f"benter_combo_win_{surface}.json"
            if win_benter_file.is_file():
                try:
                    from models.benter_combination import BenterCombination

                    win_benter = BenterCombination.load(win_benter_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", win_benter_file)

            # Win Isotonic Calibrator (joblib)
            win_isotonic_calibrator = None
            win_iso_file = models_dir / f"isotonic_win_{surface}.joblib"
            if win_iso_file.is_file():
                try:
                    win_isotonic_calibrator = joblib.load(win_iso_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", win_iso_file)

            # Win Temperature Scaler (JSON)
            win_temperature_scaler = None
            win_temp_file = models_dir / f"temp_scale_win_{surface}.json"
            if win_temp_file.is_file():
                try:
                    from models.benter_combination import TemperatureScaling

                    win_temperature_scaler = TemperatureScaling.load(win_temp_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", win_temp_file)

            # Phase 19: EV Isotonic Calibrator (joblib)
            ev_isotonic_calibrator = None
            ev_iso_file = models_dir / f"ev_isotonic_{surface}.joblib"
            if ev_iso_file.is_file():
                try:
                    ev_isotonic_calibrator = joblib.load(ev_iso_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", ev_iso_file)

            # Phase 19: EV Odds Band Scales (JSON)
            ev_odds_band_scales = None
            band_file = models_dir / f"ev_odds_band_scales_{surface}.json"
            if band_file.is_file():
                try:
                    with open(band_file) as f:
                        ev_odds_band_scales = json.load(f)
                    if not _valid_ev_band_scales(ev_odds_band_scales):
                        logger.warning("Ignoring degenerate EV odds band scales from %s", band_file)
                        ev_odds_band_scales = None
                except Exception:
                    logger.warning("Failed to load %s, skipping", band_file)

            # Wire EV Isotonic + band scales into EVCorrectionModel instance
            ev_corr.ev_isotonic_calibrator = ev_isotonic_calibrator
            ev_corr.ev_odds_band_scales = ev_odds_band_scales

            # INTER-03: TargetEncoder
            target_encoder = None
            te_path = models_dir / f"target_encoder_{surface}.joblib"
            if te_path.exists():
                target_encoder = joblib.load(te_path)

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                place_ev_corrector=place_ev_corr,
                wide=wide,
                conformal_ev_model=conformal_ev,  # Phase 21: CQR model
                place_selection_gate=place_selection_gate,
                use_ensemble=use_ensemble,
                benter_combo=benter_combo,
                isotonic_calibrator=isotonic_calibrator,
                temperature_scaler=temperature_scaler,
                win_benter=win_benter,
                win_isotonic_calibrator=win_isotonic_calibrator,
                win_temperature_scaler=win_temperature_scaler,
                win_selection_gate=win_selection_gate,
                win_selection_policy=win_selection_policy,
                win_profit_selector=win_profit_selector,
                win_segment_calibrator=win_segment_calibrator,
                ev_isotonic_calibrator=ev_isotonic_calibrator,
                ev_odds_band_scales=ev_odds_band_scales,
                target_encoder=target_encoder,
            )

        # RaceQualityScreener
        quality = RaceQualityScreener()
        quality.model = self._load_lgbm(str(models_dir / "race_quality.lgb"))
        with open(models_dir / "meta.json", encoding="utf-8") as f:
            quality.threshold = float(json.load(f).get("quality_threshold", 0.0))

        # RegimeDetector
        regime = RegimeDetector()
        regime.model = self._load_lgbm(str(models_dir / "regime_detector.lgb"))

        models = TrainedModelsV5(
            submodels=submodels,
            quality_screener=quality,
            regime_detector=regime,
            train_period=(train_start, train_end),
        )

        info = ModelInfo(
            mlflow_run_id="local",
            train_start=train_start,
            train_end=train_end,
            loaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        return models, info
