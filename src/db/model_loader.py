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

    def load(self, run_id: str | None = None) -> tuple[TrainedModelsV5, ModelInfo]:
        """MLflow から学習済みモデルを読み込み、TrainedModelsV5 を再構築。

        run_id 未指定時は最新の成功 run を使用。
        """
        if run_id is None:
            run_id = self._find_latest_run()

        run = mlflow.get_run(run_id)
        params = run.data.params
        train_end = params.get("train_end", "unknown")
        train_start = params.get("train_start", "2020-01-01")
        quality_threshold = float(params.get("quality_threshold", "0.0"))

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
            market.model = mlflow.lightgbm.load_model(f"{artifact_uri}/market_{surface}")

            # AbilityModel (per-surface booster)
            ability = AbilityModel()
            ability.models = {
                surface: mlflow.lightgbm.load_model(f"{artifact_uri}/stage1_{surface}")
            }

            # WinTwoStageModel
            win = WinTwoStageModel()
            win.hit_model = mlflow.lightgbm.load_model(f"{artifact_uri}/win_hit_{surface}")
            win.return_model = mlflow.lightgbm.load_model(f"{artifact_uri}/win_ret_{surface}")

            # EVCorrectionModel
            ev_corr = EVCorrectionModel()
            ev_corr.p_correction_model = mlflow.lightgbm.load_model(
                f"{artifact_uri}/ev_corrector_p_{surface}"
            )
            ev_corr.e_correction_model = mlflow.lightgbm.load_model(
                f"{artifact_uri}/ev_corrector_e_{surface}"
            )

            # PlaceTwoStageModel
            place = PlaceTwoStageModel()
            place.hit_model = mlflow.lightgbm.load_model(f"{artifact_uri}/place_hit_{surface}")
            place.return_model = mlflow.lightgbm.load_model(f"{artifact_uri}/place_ret_{surface}")

            # PlaceAbilityModel (joblib artifact)
            pa = PlaceAbilityModel()
            pa_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/place_ability_{surface}")
            pa_files = list(Path(pa_dir).glob("*.joblib"))
            if pa_files:
                pa._calibrated = joblib.load(pa_files[0])
            else:
                logger.warning("PlaceAbilityModel artifact not found for %s", surface)

            # WideTwoStageModel
            wide = WideTwoStageModel()
            wide.hit_model = mlflow.lightgbm.load_model(f"{artifact_uri}/wide_hit_{surface}")
            wide.return_model = mlflow.lightgbm.load_model(f"{artifact_uri}/wide_ret_{surface}")

            # RobustConfidenceEstimator (JSON params)
            confidence = RobustConfidenceEstimator()
            try:
                conf_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/confidence_params.json"
                )
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
        quality.model = mlflow.lightgbm.load_model(f"{artifact_uri}/race_quality")
        quality.threshold = quality_threshold

        # RegimeDetector
        regime = RegimeDetector()
        regime.model = mlflow.lightgbm.load_model(f"{artifact_uri}/regime_detector")

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
        """最新の成功 run ID を取得"""
        import pandas as pd
        from mlflow.entities import ViewType

        df: pd.DataFrame = mlflow.search_runs(  # type: ignore[assignment]
            order_by=["start_time DESC"],
            max_results=1,
            filter_string="status = 'FINISHED'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        if df.empty:
            raise ValueError("No successful MLflow runs found")
        return str(df.iloc[0]["run_id"])
