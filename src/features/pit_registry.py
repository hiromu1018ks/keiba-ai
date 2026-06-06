"""PIT 契約レジストリ — 特徴量モジュールの Point-in-Time 遵守を検証 (D-05)。

各モジュールの PIT 契約 (最大日付列) を登録し、推論時に
max(race_date) < prediction_date を検証する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PITContract:
    """単一モジュールの PIT 契約。"""

    module_name: str
    max_date_column: str | None  # None = PIT ソースデータが設計上安全
    description: str


class PITModuleRegistry:
    """PIT 契約レジストリ (D-05)。

    各特徴量モジュールの最大日付列を管理し、推論前に
    prediction_date より未来のデータが含まれていないことを検証する。
    """

    def __init__(self) -> None:
        self._contracts: dict[str, PITContract] = {}
        self._register_default_modules()

    def register(
        self, module_name: str, max_date_column: str | None, description: str
    ) -> None:
        """モジュールの PIT 契約を登録。

        Args:
            module_name: モジュール名。
            max_date_column: 最大日付列名。None は PIT ソースデータが設計上安全であることを示す。
            description: モジュールの説明。
        """
        self._contracts[module_name] = PITContract(
            module_name=module_name,
            max_date_column=max_date_column,
            description=description,
        )

    def verify_pit_compliance(
        self, df: pd.DataFrame, prediction_date: pd.Timestamp
    ) -> list[str]:
        """全モジュールの PIT 遵守を検証。

        Args:
            df: 特徴量 DataFrame (race_date 列を含む)。
            prediction_date: 予測日。

        Returns:
            違反メッセージのリスト。空リスト = 全モジュール遵守。
        """
        violations: list[str] = []
        for name, contract in self._contracts.items():
            if contract.max_date_column is None:
                continue
            col = contract.max_date_column
            if col not in df.columns:
                continue
            max_date = pd.to_datetime(df[col]).max()
            if pd.notna(max_date) and max_date >= prediction_date:
                violations.append(
                    f"{name}: max {col} = {max_date.date()} >= prediction_date "
                    f"{prediction_date.date()}"
                )
        return violations

    @property
    def contracts(self) -> dict[str, PITContract]:
        """登録済み契約の読み取り専用コピー。"""
        return dict(self._contracts)

    def _register_default_modules(self) -> None:
        """13 の標準エンリッチメントモジュールを事前登録。"""
        default_modules: list[tuple[str, str | None, str]] = [
            (
                "HorseHistoryFeatures",
                "race_date",
                "馬過去成績特徴量 — PITシフト済み履歴から計算",
            ),
            (
                "PaceAptitudeFeatures",
                "race_date",
                "ペース適性特徴量 — HorseHistoryFeatures に依存",
            ),
            (
                "CourseFeatures",
                "race_date",
                "コース別適性特徴量 — 過去レース統計",
            ),
            (
                "SireFeatures",
                None,
                "種牡馬産駒特徴量 — 事前計算済みキャリア統計 (PIT設計上安全)",
            ),
            (
                "DamPedigreeFeatures",
                "race_date",
                "繁殖牝馬産駒特徴量 — merge_asof で PIT 安全",
            ),
            (
                "RecordFeatures",
                None,
                "コースレコード特徴量 — 静的データ (PIT設計上安全)",
            ),
            (
                "TrackConditionFeatures",
                "race_date",
                "馬場状態トラック条件特徴量 — 学習期間統計を使用",
            ),
            (
                "InteractionFeatures",
                "race_date",
                "交互作用特徴量 — 入力列に依存",
            ),
            (
                "MiningFeatures",
                "race_date",
                "n_mining 予想特徴量 — DataKubun=3 直前予想",
            ),
            (
                "RelativeFeatures",
                "race_date",
                "レース内相対比較特徴量 — 入力列に依存",
            ),
            (
                "JockeyContextFeatures",
                "race_date",
                "騎手年度別特徴量 — SetYear < race_year",
            ),
            (
                "TrainerContextFeatures",
                "race_date",
                "調教師年度別特徴量 — SetYear < race_year",
            ),
            (
                "JockeyTrainerComboFeatures",
                "race_date",
                "騎手-調教師コンビ特徴量 — 過去実績",
            ),
        ]
        for module_name, max_date_col, desc in default_modules:
            self.register(module_name, max_date_col, desc)
