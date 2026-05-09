"""CQR (Conformalized Quantile Regression) によるEV予測区間推定.

Romano et al., 2019 "Conformalized Quantile Regression" に基づく。
LightGBM quantile regression で alpha/2 と 1-alpha/2 の分位点を学習し、
CQR非適合スコアでCP補正を適用する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.types import POST_RACE_COLS

logger = logging.getLogger(__name__)

# 学習時に除外する非特徴量列
_NON_FEATURE_COLS: set[str] = {
    # IDs / metadata
    "race_id",
    "umaban",
    "race_date",
    "surface",
    "kettonum",
    # Target (CQR predicts this)
    "actual_ev_win",
    # CQR's own outputs (circular)
    "EV_lower_win_corrected",
    "EV_upper_win_corrected",
    "conformal_confidence_score",
    # Place-related (CQR is win-only)
    "ev_place_corrected",
    "actual_ev_place",
    "EV_lower_place",
    "EV_upper_place",
    # Object dtype (LightGBM cannot handle)
    "distance_bin", "grade_code", "track_condition_code",
} | set(POST_RACE_COLS)


class ConformalEVModel:
    """CQR (Conformalized Quantile Regression) によるEV予測区間推定.

    Romano et al., 2019 "Conformalized Quantile Regression" に基づく。
    LightGBM quantile regression で分位点を学習し、CQR非適合スコアでCP補正する。
    """

    def __init__(self, alpha: float = 0.1, feature_cols: list[str] | None = None) -> None:
        """Args:
            alpha: 有意水準 (デフォルト 0.1 = 90%信頼区間)
            feature_cols: 特徴量列名。Noneの場合はtrain()で自動抽出。
        """
        self.alpha = alpha
        self.feature_cols = feature_cols
        self._calibrated = False
        # LightGBM quantile models
        self.q_low_model: lgb.Booster | None = None
        self.q_high_model: lgb.Booster | None = None
        # CQR calibration quantiles (2-alpha構成)
        self._calibration_quantile_90: float = 0.0  # alpha=0.1用
        self._calibration_quantile_80: float = 0.0  # alpha=0.2用

    def calibrate(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> None:
        """Backward compat: RobustConfidenceEstimator.calibrate() の互換shim.

        Phase 21 Plan 02で完全なCQR学習に置き換えられるまで、
        古いパイプラインコードが動作するよう残存させる。
        """
        logger.warning(
            "ConformalEVModel.calibrate() is a backward-compat shim. "
            "Full CQR training via train() will be integrated in Plan 02."
        )
        self._calibrated = True

    def predict_lower_bound(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backward compat: predict_interval()のラッパー (RobustConfidenceEstimator互換)."""
        win_result, place_result = self.predict_interval(win_df, place_df)
        win_result = win_result.drop(
            columns=["EV_upper_win_corrected", "conformal_confidence_score"],
            errors="ignore",
        )
        place_result = place_result.drop(
            columns=["EV_upper_place"],
            errors="ignore",
        )
        return win_result, place_result

    def train(
        self,
        df_calib: pd.DataFrame,
        *,
        num_threads: int = 0,
        lgb_params: dict | None = None,
        train_ratio: float = 0.8,
    ) -> None:
        """CQRモデルを学習・キャリブレーション.

        Args:
            df_calib: ev_win_calibrated, actual_ev_win, 特徴量列を含むDataFrame
            num_threads: LightGBMスレッド数
            lgb_params: LightGBM追加パラメータ
            train_ratio: 学習/キャリブレーション分割比。1.0の場合は全データを学習+キャリブレーションに使用
        """
        # 特徴量列の決定
        if self.feature_cols is None:
            exclude = _NON_FEATURE_COLS | {
                col for col in df_calib.columns
                if col.startswith("_") or col in ("distance_bin",)
            }
            self.feature_cols = [c for c in df_calib.columns if c not in exclude]

        # train_ratio=1.0の場合は全データを学習に使用し、同じデータでキャリブレーション
        if train_ratio >= 1.0:
            logger.warning(
                "train_ratio=%.1f: using same data for training and calibration. "
                "CQR intervals will be overfitted and may not achieve target coverage.",
                train_ratio,
            )
            df_train = df_calib
            df_val = df_calib
        else:
            n_total = len(df_calib)
            split_idx = int(n_total * train_ratio)
            df_train = df_calib.iloc[:split_idx]
            df_val = df_calib.iloc[split_idx:]

        # ターゲット抽出
        if "actual_ev_win" in df_train.columns:
            target_col = "actual_ev_win"
        else:
            target_col = "ev_win_calibrated"

        X_train = df_train[self.feature_cols]
        y_train = pd.to_numeric(df_train[target_col], errors="coerce")
        mask = y_train.notna()
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(y_train) < 200:
            logger.warning(
                "Insufficient samples for CQR training (%d < 200). "
                "Skipping CQR model training.",
                len(y_train),
            )
            return

        # LightGBM データセット
        train_set = lgb.Dataset(X_train, label=y_train.values)

        # デフォルトパラメータ
        default_params: dict = {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "verbose": -1,
            "num_threads": num_threads,
        }

        # q_low_model: alpha/2分位点 (例: alpha=0.1 -> quantile=0.05)
        params_low = {**default_params, "alpha": self.alpha / 2}
        if lgb_params:
            params_low.update(lgb_params)
        self.q_low_model = lgb.train(
            params_low,
            train_set,
            num_boost_round=200,
            valid_sets=[train_set],
            callbacks=[lgb.log_evaluation(0)],
        )

        # q_high_model: 1-alpha/2分位点 (例: alpha=0.1 -> quantile=0.95)
        params_high = {**default_params, "alpha": 1 - self.alpha / 2}
        if lgb_params:
            params_high.update(lgb_params)
        self.q_high_model = lgb.train(
            params_high,
            train_set,
            num_boost_round=200,
            valid_sets=[train_set],
            callbacks=[lgb.log_evaluation(0)],
        )

        # キャリブレーションセットで非適合スコア計算
        X_val = df_val[self.feature_cols]
        y_val = pd.to_numeric(df_val[target_col], errors="coerce")
        val_mask = y_val.notna()
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        if len(y_val) < 10:
            logger.warning(
                "Insufficient calibration samples (%d < 10). "
                "Skipping CQR calibration.",
                len(y_val),
            )
            return

        q_low_pred = self.q_low_model.predict(X_val)
        q_high_pred = self.q_high_model.predict(X_val)

        # CQR非適合スコア: max(q_low - y, y - q_high)
        nonconformity_scores = np.maximum(q_low_pred - y_val.values, y_val.values - q_high_pred)

        # 有限サンプル補正付き補正量子 (Romano et al., 2019)
        n = len(nonconformity_scores)
        # 90%区間用 (alpha=0.1)
        q_90_level = min((1 - 0.1) * (1 + 1 / n), 1.0)
        self._calibration_quantile_90 = float(np.quantile(nonconformity_scores, q_90_level))
        # 80%区間用 (alpha=0.2)
        q_80_level = min((1 - 0.2) * (1 + 1 / n), 1.0)
        self._calibration_quantile_80 = float(np.quantile(nonconformity_scores, q_80_level))

        self._calibrated = True
        logger.info(
            "CQR calibrated: Q_90=%.4f, Q_80=%.4f, n_calib=%d",
            self._calibration_quantile_90,
            self._calibration_quantile_80,
            n,
        )

    def predict_interval(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
        alphas: tuple[float, ...] = (0.1, 0.2),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """EVの信頼区間(上下)を複数水準で推定.

        RobustConfidenceEstimator.predict_interval()と同じシグネチャ・出力互換。

        Note: CQR is applied to win EV only. Place intervals are identity
        pass-through (EV_lower_place = EV_upper_place = place_ev), providing
        zero-width intervals with no uncertainty quantification.
        TODO: implement CQR for place EV if needed.

        Args:
            win_df: ev_win_calibrated と特徴量列を含むDataFrame
            place_df: ev_place_corrected を含むDataFrame
            alphas: 信頼水準のタプル。0.1=90%区間、0.2=80%区間

        Returns:
            (win_df, place_df) with EV_lower/upper columns and conformal_confidence_score
        """
        win_df = win_df.copy()
        place_df = place_df.copy()

        # 未キャリブレーション時のフォールバック
        if not self._calibrated or self.q_low_model is None or self.q_high_model is None:
            logger.warning("ConformalEVModel not calibrated, using fallback values")
            if "ev_win_calibrated" in win_df.columns:
                win_ev = pd.to_numeric(win_df["ev_win_calibrated"], errors="coerce").fillna(0.0)
            elif "ev_win_corrected" in win_df.columns:
                win_ev = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce").fillna(0.0)
            else:
                win_ev = pd.Series(0.0, index=win_df.index)
            win_df["EV_lower_win_corrected"] = win_ev
            win_df["EV_upper_win_corrected"] = win_ev
            win_df["conformal_confidence_score"] = 0.0

            if "ev_place_corrected" in place_df.columns:
                place_ev = pd.to_numeric(
                    place_df["ev_place_corrected"], errors="coerce"
                ).fillna(0.0)
            else:
                place_ev = pd.Series(0.0, index=place_df.index)
            place_df["EV_lower_place"] = place_ev
            place_df["EV_upper_place"] = place_ev
            return win_df, place_df

        # ベースEV取得
        if "ev_win_calibrated" in win_df.columns:
            win_ev = pd.to_numeric(win_df["ev_win_calibrated"], errors="coerce").fillna(0.0)
        elif "ev_win_corrected" in win_df.columns:
            win_ev = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce").fillna(0.0)
        else:
            win_ev = pd.Series(0.0, index=win_df.index)

        # CQR予測
        if self.feature_cols is None:
            exclude = _NON_FEATURE_COLS | {
                col for col in win_df.columns if col.startswith("_")
            }
            feature_cols = [c for c in win_df.columns if c not in exclude]
        else:
            feature_cols = self.feature_cols

        # 推論コンテキストで一部特徴量が欠落する場合（キャリブレーションBT等）、
        # 利用可能な列のみを使用し欠落列は0で埋める
        available = set(win_df.columns)
        missing = [c for c in feature_cols if c not in available]
        if missing:
            logger.warning(
                "ConformalEV: %d/%d feature_cols missing, filling with 0: %s",
                len(missing), len(feature_cols), missing[:5],
            )
            for c in missing:
                win_df[c] = 0.0

        X_win = win_df[feature_cols].copy()
        # object型列を数値に変換（学習時と推論時でdtypeが異なる場合）
        for c in X_win.columns:
            if X_win[c].dtype == object:
                X_win[c] = pd.to_numeric(X_win[c], errors="coerce").fillna(0.0)
        X_win = X_win.fillna(0.0)
        q_low = self.q_low_model.predict(X_win)
        q_high = self.q_high_model.predict(X_win)

        # Monotonicity guarantee: q_low <= q_high
        q_low = np.minimum(q_low, q_high)

        # 90%区間 (第一alpha, 通常alpha=0.1)
        lower_90 = np.maximum(q_low - self._calibration_quantile_90, 0.0)
        upper_90 = q_high + self._calibration_quantile_90

        # 出力
        win_df["EV_lower_win_corrected"] = lower_90
        win_df["EV_upper_win_corrected"] = upper_90

        # 80%区間 (第二alpha, 通常alpha=0.2) - confidence_score用
        if len(alphas) > 1:
            lower_80 = np.maximum(q_low - self._calibration_quantile_80, 0.0)
        else:
            lower_80 = lower_90

        # conformal_confidence_score: lower_80 * (1 - normalized_width)
        interval_width = pd.Series(upper_90 - lower_90, index=win_df.index).clip(lower=1e-6)
        if "race_id" in win_df.columns:
            max_width = (
                interval_width.groupby(win_df["race_id"], observed=True)
                .transform("max")
                .clip(lower=1e-6)
            )
            normalized_width = (interval_width / max_width).clip(0.0, 1.0)
        else:
            max_width = max(float(interval_width.max()), 1e-6)
            normalized_width = (interval_width / max_width).clip(0.0, 1.0)

        win_df["conformal_confidence_score"] = (lower_80 * (1.0 - normalized_width)).fillna(0.0)

        # Place側: 簡易処理 (CQR適用なし)
        if "ev_place_corrected" in place_df.columns:
            place_ev = pd.to_numeric(place_df["ev_place_corrected"], errors="coerce").fillna(0.0)
        else:
            place_ev = pd.Series(0.0, index=place_df.index)
        place_df["EV_lower_place"] = place_ev
        place_df["EV_upper_place"] = place_ev

        return win_df, place_df

    def save(self, models_dir: Path, surface: str) -> None:
        """CQRモデルをディスクに保存.

        Args:
            models_dir: 保存先ディレクトリ
            surface: サーフェス名 (turf/dirt)
        """
        if not self._calibrated or self.q_low_model is None or self.q_high_model is None:
            logger.warning("Cannot save uncalibrated ConformalEVModel for %s", surface)
            return

        models_dir.mkdir(parents=True, exist_ok=True)

        # LightGBMモデル保存
        low_path = models_dir / f"cqr_quantile_low_{surface}.lgb"
        high_path = models_dir / f"cqr_quantile_high_{surface}.lgb"
        self.q_low_model.save_model(str(low_path))
        self.q_high_model.save_model(str(high_path))

        # キャリブレーションパラメータ保存
        params_path = models_dir / f"cqr_params_{surface}.json"
        params = {
            "alpha": self.alpha,
            "calibration_quantile_90": self._calibration_quantile_90,
            "calibration_quantile_80": self._calibration_quantile_80,
            "feature_cols": self.feature_cols,
            "_calibrated": self._calibrated,
        }
        params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Saved ConformalEVModel for %s to %s", surface, models_dir)

    @classmethod
    def load(cls, models_dir: Path, surface: str) -> ConformalEVModel | None:
        """ディスクからCQRモデルを読み込み.

        Args:
            models_dir: モデルディレクトリ
            surface: サーフェス名 (turf/dirt)

        Returns:
            ConformalEVModel or None (ファイルが揃っていない場合)
        """
        low_path = models_dir / f"cqr_quantile_low_{surface}.lgb"
        high_path = models_dir / f"cqr_quantile_high_{surface}.lgb"
        params_path = models_dir / f"cqr_params_{surface}.json"

        if not (low_path.is_file() and high_path.is_file() and params_path.is_file()):
            return None

        model = cls()
        model.q_low_model = lgb.Booster(model_file=str(low_path))
        model.q_high_model = lgb.Booster(model_file=str(high_path))

        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        model.alpha = params["alpha"]
        model._calibration_quantile_90 = params["calibration_quantile_90"]
        model._calibration_quantile_80 = params["calibration_quantile_80"]
        model.feature_cols = params.get("feature_cols")
        model._calibrated = params.get("_calibrated", True)

        return model
