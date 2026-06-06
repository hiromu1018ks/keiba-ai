"""データカットオフマニフェスト (D-07).

PT 実行前に全データソースの最終日付が予測日より前であることを検証する。
モデル学習期間、統計情報、オッズバンドキャリブレーション、戦略最適化の
4つのデータソースをチェックし、未来情報の漏洩を防止する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)

# カットオフ検証対象のフィールド名と意味
_CUTOFF_FIELDS = (
    "model_train_end",
    "stats_fit_end",
    "odds_band_calibration_end",
    "strategy_optimization_end",
)


@dataclass(frozen=True)
class DataCutoffManifest:
    """データソースの最終日付を保持する不変 dataclass (D-07).

    各フィールドは YYYY-MM-DD 形式の文字列。
    prediction_date より後の日付は未来情報漏洩として検出する。
    """

    model_train_end: str
    stats_fit_end: str
    odds_band_calibration_end: str
    strategy_optimization_end: str
    prediction_date: str

    def verify(self, actual_dates: dict[str, str]) -> list[str]:
        """各データソースの実際の最終日付が prediction_date 以前か検証。

        Args:
            actual_dates: フィールド名→実際の日付文字列のマッピング。

        Returns:
            違反メッセージのリスト。空リストなら全チェック通過。
        """
        violations: list[str] = []
        try:
            pred_dt = date_type.fromisoformat(self.prediction_date)
        except ValueError:
            # prediction_date が不正フォーマットの場合は文字列比較にフォールバック
            logger.warning(
                "prediction_date %s is not valid ISO format, falling back to string comparison",
                self.prediction_date,
            )
            pred_dt = None

        for field in _CUTOFF_FIELDS:
            actual = actual_dates.get(field)
            if actual is None:
                violations.append(
                    f"{field}: actual date not provided (expected <= {self.prediction_date})"
                )
                continue
            if pred_dt is not None:
                try:
                    actual_dt = date_type.fromisoformat(actual)
                    if actual_dt > pred_dt:
                        violations.append(
                            f"{field}: {actual} > prediction_date {self.prediction_date}"
                        )
                except ValueError:
                    # actual が不正フォーマットの場合は文字列比較にフォールバック
                    logger.warning(
                        "%s date %s is not valid ISO format, using string comparison",
                        field, actual,
                    )
                    if actual > self.prediction_date:
                        violations.append(
                            f"{field}: {actual} > prediction_date {self.prediction_date}"
                        )
            else:
                # prediction_date が不正だった場合のフォールバック
                if actual > self.prediction_date:
                    violations.append(
                        f"{field}: {actual} > prediction_date {self.prediction_date}"
                    )
        return violations

    def verify_strict(self, actual_dates: dict[str, str]) -> None:
        """fail-fast 検証。違反があれば ValueError を送出。

        Args:
            actual_dates: フィールド名→実際の日付文字列のマッピング。

        Raises:
            ValueError: いずれかのデータソースが prediction_date を超過。
        """
        violations = self.verify(actual_dates)
        if violations:
            msg = "Data cutoff violations detected:\n" + "\n".join(
                f"  - {v}" for v in violations
            )
            raise ValueError(msg)

    @classmethod
    def from_config(
        cls,
        prediction_date: str,
        models: TrainedModelsV5,
        strategy_manifest_path: Path | None = None,
    ) -> DataCutoffManifest:
        """TrainedModelsV5 と戦略マニフェストから DataCutoffManifest を生成。

        Args:
            prediction_date: 予測対象日 (YYYY-MM-DD)。
            models: 学習済みモデル (train_period を参照)。
            strategy_manifest_path: 戦略マニフェストのパス (省略可)。

        Returns:
            DataCutoffManifest インスタンス。
        """
        # モデル学習期間の終了日
        model_train_end = models.train_period[1] if models.train_period else "1970-01-01"

        # 統計情報のフィット期間 (モデル学習期間と同一とする)
        stats_fit_end = model_train_end

        # オッズバンドキャリブレーション期間 (モデル学習期間と同一)
        odds_band_calibration_end = model_train_end

        # 戦略最適化期間
        strategy_optimization_end = "1970-01-01"
        if strategy_manifest_path is not None and strategy_manifest_path.exists():
            try:
                manifest_data = json.loads(
                    strategy_manifest_path.read_text(encoding="utf-8")
                )
                params = manifest_data.get("params", {})
                # 戦略マニフェストに optimization_end があれば使用
                opt_end = params.get("optimization_end")
                if opt_end is not None:
                    strategy_optimization_end = str(opt_end)
                else:
                    # 最適化日がない場合はモデル学習終了日を使用
                    strategy_optimization_end = model_train_end
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse strategy manifest: %s", e)
                strategy_optimization_end = model_train_end
        else:
            strategy_optimization_end = model_train_end

        return cls(
            model_train_end=model_train_end,
            stats_fit_end=stats_fit_end,
            odds_band_calibration_end=odds_band_calibration_end,
            strategy_optimization_end=strategy_optimization_end,
            prediction_date=prediction_date,
        )
