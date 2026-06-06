"""パイプライン整合性検証 (D-08).

PFPVerifier は PT 実行中のパラメータ不変性を保証する。
ParameterFreezeProtocol (モデル HP) + FeatureManifest + FeatureState +
ベッティング設定のハッシュをスナップショットし、verify() で変更を検出する。
RegimeDetector, DDController 等のランタイム状態は検証対象外 (D-08)。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5
    from features.feature_manifest import FeatureManifest, FeatureState

logger = logging.getLogger(__name__)


class PFPVerifier:
    """PT 実行中のパラメータ不変性検証 (D-08).

    ParameterFreezeProtocol でモデル HP スナップショットを取得し、
    FeatureManifest / FeatureState / betting_target / betting_mode の
    ハッシュを記録。verify() で全項目の不変性を検証する。
    ランタイム状態 (RegimeDetector, DDController) は除外。

    Usage:
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        # ... PT execution ...
        result = verifier.verify()  # {"passed": bool, "checks": {...}, ...}
    """

    def __init__(
        self,
        models: TrainedModelsV5,
        feature_manifest: FeatureManifest,
        feature_state: FeatureState,
        betting_target: str,
        betting_mode: str,
    ) -> None:
        self._pfp = ParameterFreezeProtocol(models)
        self._feature_manifest = feature_manifest
        self._feature_state = feature_state
        self._betting_target = betting_target
        self._betting_mode = betting_mode

        # 凍結状態 (freeze() で設定)
        self._frozen_pfp: bool = False
        self._frozen_manifest_hash: str | None = None
        self._frozen_state_hash: str | None = None
        self._frozen_betting_target: str | None = None
        self._frozen_betting_mode: str | None = None

    def freeze(self) -> None:
        """現在のパラメータ状態をスナップショット。"""
        self._pfp.freeze()
        self._frozen_pfp = True
        self._frozen_manifest_hash = self._feature_manifest.compute_hash()
        self._frozen_state_hash = self._feature_state.compute_hash()
        self._frozen_betting_target = self._betting_target
        self._frozen_betting_mode = self._betting_mode
        logger.info(
            "PFPVerifier: parameters frozen (manifest=%s...)",
            self._frozen_manifest_hash[:8] if self._frozen_manifest_hash else "N/A",
        )

    def verify(self) -> dict[str, Any]:
        """凍結時からパラメータが変更されていないことを検証。

        Returns:
            {
                "passed": bool,
                "checks": {
                    "model_hp": bool,
                    "feature_manifest": bool,
                    "feature_state": bool,
                    "betting_target": bool,
                    "betting_mode": bool,
                },
                "message": str,
            }
        """
        if not self._frozen_pfp:
            return {
                "passed": False,
                "checks": {},
                "message": "freeze() has not been called",
            }

        # ParameterFreezeProtocol: モデル HP 不変性
        pfp_result = self._pfp.verify()
        model_hp_ok = pfp_result["passed"]

        # FeatureManifest hash (re-compute current)
        current_manifest_hash = self._feature_manifest.compute_hash()
        manifest_ok = current_manifest_hash == self._frozen_manifest_hash

        # FeatureState hash (re-compute current)
        current_state_hash = self._feature_state.compute_hash()
        state_ok = current_state_hash == self._frozen_state_hash

        # Betting settings
        target_ok = self._betting_target == self._frozen_betting_target
        mode_ok = self._betting_mode == self._frozen_betting_mode

        checks = {
            "model_hp": model_hp_ok,
            "feature_manifest": manifest_ok,
            "feature_state": state_ok,
            "betting_target": target_ok,
            "betting_mode": mode_ok,
        }
        passed = all(checks.values())

        if passed:
            message = "PFP verification passed — all parameters unchanged"
        else:
            failed = [k for k, v in checks.items() if not v]
            message = f"PFP verification FAILED: {', '.join(failed)} changed"

        return {"passed": passed, "checks": checks, "message": message}

    def get_frozen_state(self) -> dict[str, str]:
        """凍結済みハッシュを返す (session_manifest 記録用)。

        Returns:
            全凍結ハッシュの dict。
        """
        return {
            "manifest_hash": self._frozen_manifest_hash
            if self._frozen_manifest_hash
            else self._feature_manifest.compute_hash(),
            "state_hash": self._frozen_state_hash
            if self._frozen_state_hash
            else self._feature_state.compute_hash(),
            "betting_target": self._frozen_betting_target or "",
            "betting_mode": self._frozen_betting_mode or "",
        }
