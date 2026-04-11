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

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


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
            return self._load_from_local(models_dir, use_ensemble_override=use_ensemble)

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
        artifact_uri = mlflow.get_artifact_uri(run_id)

        from domain.models import SubmodelSet, TrainedModelsV5
        from models.ev_correction_model import EVCorrectionModel
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.robust_confidence_estimator import RobustConfidenceEstimator
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel

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

            # PlaceTwoStageModel
            place = PlaceTwoStageModel()
            place.hit_model = self._load_lgbm(f"{artifact_uri}/place_hit_{surface}")
            place.return_model = self._load_lgbm(f"{artifact_uri}/place_ret_{surface}")

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

            # RobustConfidenceEstimator (JSON params)
            confidence = RobustConfidenceEstimator()
            try:
                conf_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/confidence_params.json"
                )
            except Exception:
                # Fallback: ファイルシステムから直接読み込み
                conf_dir = self._find_artifact_dir(run_id, "confidence_params.json")
                conf_path = str(conf_dir / "confidence_params.json")
                with open(conf_path) as f:
                    conf_data = json.load(f)
                confidence.alpha = conf_data["alpha"]
                confidence.rolling_window = conf_data["rolling_window"]
                confidence._win_cp_quantile = conf_data["win_cp_quantile"]
                confidence._place_cp_quantile = conf_data["place_cp_quantile"]
                confidence._win_rolling_quantile = conf_data["win_rolling_quantile"]
                confidence._place_rolling_quantile = conf_data["place_rolling_quantile"]
                confidence._calibrated = True
            except Exception:
                logger.warning("RobustConfidenceEstimator params not found, using defaults")

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                wide=wide,
                confidence=confidence,
            )

        # RaceQualityScreener
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
        import os

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
    def _load_hit_model(
        models_dir: Path, name: str, *, use_ensemble: bool = False
    ) -> object:
        """Load hit model from .joblib (StackedEnsemble) or .lgb (LightGBM)."""
        joblib_path = models_dir / f"{name}.joblib"
        lgb_path = models_dir / f"{name}.lgb"

        if use_ensemble:
            if joblib_path.is_file():
                logger.info("Loading StackedEnsemble: %s", joblib_path)
                return joblib.load(joblib_path)
            if lgb_path.is_file():
                logger.warning(
                    "use_ensemble=True but .joblib not found, falling back to .lgb"
                )
                return ModelLoader._load_lgbm(str(lgb_path))
        else:
            if lgb_path.is_file():
                return ModelLoader._load_lgbm(str(lgb_path))
            if joblib_path.is_file():
                logger.info(
                    "Loading StackedEnsemble (discovered from .joblib): %s", name
                )
                return joblib.load(joblib_path)

        raise FileNotFoundError(f"No model file found for {name} in {models_dir}")

    def _load_from_local(
        self, models_dir: Path, *, use_ensemble_override: bool | None = None
    ) -> tuple[TrainedModelsV5, ModelInfo]:
        """data/models/ から全モデルをロード"""
        from domain.models import SubmodelSet, TrainedModelsV5
        from models.ev_correction_model import EVCorrectionModel
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.robust_confidence_estimator import RobustConfidenceEstimator
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel

        # メタ情報読み込み
        with open(models_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        train_start = meta["train_start"]
        train_end = meta["train_end"]

        # 整合性チェック: meta.json の use_ensemble と .joblib の有無が矛盾していないか
        meta_ensemble = meta.get("use_ensemble", False)
        has_joblib_hit = any((models_dir / f"win_hit_{s}.joblib").is_file() for s in meta.get("surfaces", []))
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

            # PlaceTwoStageModel
            place = PlaceTwoStageModel()
            place.hit_model = self._load_hit_model(
                models_dir, f"place_hit_{surface}", use_ensemble=use_ensemble
            )
            place.return_model = self._load_lgbm(str(models_dir / f"place_ret_{surface}.lgb"))

            # PlaceAbilityModel (joblib)
            pa = PlaceAbilityModel()
            pa_file = models_dir / f"place_ability_{surface}.joblib"
            if pa_file.is_file():
                try:
                    pa._calibrated = joblib.load(pa_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", pa_file)

            # WideTwoStageModel
            wide = WideTwoStageModel()
            wide.hit_model = self._load_lgbm(str(models_dir / f"wide_hit_{surface}.lgb"))
            wide.return_model = self._load_lgbm(str(models_dir / f"wide_ret_{surface}.lgb"))

            # RobustConfidenceEstimator
            confidence = RobustConfidenceEstimator()
            conf_file = models_dir / "confidence_params.json"
            if conf_file.is_file():
                with open(conf_file) as f:
                    conf_data = json.load(f)
                confidence.alpha = conf_data["alpha"]
                confidence.rolling_window = conf_data["rolling_window"]
                confidence._win_cp_quantile = conf_data["win_cp_quantile"]
                confidence._place_cp_quantile = conf_data["place_cp_quantile"]
                confidence._win_rolling_quantile = conf_data["win_rolling_quantile"]
                confidence._place_rolling_quantile = conf_data["place_rolling_quantile"]
                confidence._calibrated = True

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                wide=wide,
                confidence=confidence,
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
