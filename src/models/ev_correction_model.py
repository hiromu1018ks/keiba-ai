"""EV補正モデル -- P/E分解で独立性破綻を解決 (C-5)"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _best_iteration(booster: lgb.Booster | None) -> int | None:
    if booster is None:
        return None
    if hasattr(booster, "best_iteration") and booster.best_iteration >= 0:
        return booster.best_iteration
    return None


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_probability_array(
    values: np.ndarray,
    *,
    target_sum: float,
    cap: float = 1.0,
) -> np.ndarray:
    probs = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, cap)
    if len(probs) == 0:
        return probs

    target = min(float(target_sum), cap * len(probs))
    if target <= 0:
        return np.zeros_like(probs)

    total = probs.sum()
    if total <= 0:
        return np.full_like(probs, target / len(probs))

    result = np.zeros_like(probs)
    weights = probs.copy()
    active = np.ones(len(probs), dtype=bool)
    remaining = target
    tol = 1e-9

    while active.any() and remaining > tol:
        active_idx = np.flatnonzero(active)
        active_weights = weights[active]
        active_total = float(active_weights.sum())
        if active_total <= tol:
            result[active] = remaining / len(active_idx)
            break

        scaled = active_weights * (remaining / active_total)
        over_mask = scaled > (cap + tol)
        if not over_mask.any():
            result[active] = scaled
            break

        capped_idx = active_idx[over_mask]
        result[capped_idx] = cap
        remaining -= cap * len(capped_idx)
        active[capped_idx] = False
        weights[capped_idx] = 0.0

    return np.clip(result, 0.0, cap)


def _normalize_probability_by_race(
    df: pd.DataFrame,
    source_col: str,
    *,
    target_sum: float,
) -> pd.Series:
    if source_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    if "race_id" not in df.columns:
        normalized = _normalize_probability_array(df[source_col].to_numpy(), target_sum=target_sum)
        return pd.Series(normalized, index=df.index, dtype=float)

    return (
        df.groupby("race_id", observed=True)[source_col]
        .transform(
            lambda s: pd.Series(
                _normalize_probability_array(s.to_numpy(), target_sum=target_sum),
                index=s.index,
            )
        )
        .astype(float)
    )


def _build_place_bucket_multiplier(df: pd.DataFrame, prob_col: str) -> pd.Series:
    odds = pd.to_numeric(df.get("fukuoddslow"), errors="coerce")
    popularity = pd.to_numeric(df.get("popularity_rank"), errors="coerce")
    probability = pd.to_numeric(df.get(prob_col), errors="coerce")

    odds_mult = pd.Series(1.0, index=df.index, dtype=float)
    odds_mult = odds_mult.mask(odds >= 15.0, 0.95)
    odds_mult = odds_mult.mask(odds >= 22.0, 0.85)
    odds_mult = odds_mult.mask(odds >= 30.0, 0.7)

    pop_mult = pd.Series(1.0, index=df.index, dtype=float)
    pop_mult = pop_mult.mask(popularity >= 12.0, 0.95)
    pop_mult = pop_mult.mask(popularity >= 15.0, 0.85)
    pop_mult = pop_mult.mask(popularity >= 18.0, 0.75)

    prob_mult = pd.Series(1.0, index=df.index, dtype=float)
    prob_mult = prob_mult.mask(probability < 0.10, 0.9)
    prob_mult = prob_mult.mask(probability < 0.08, 0.8)
    prob_mult = prob_mult.mask(probability < 0.06, 0.7)

    multiplier = pd.concat([odds_mult, pop_mult, prob_mult], axis=1).min(axis=1)
    return multiplier.fillna(1.0).astype(float)


class EVCorrectionModel:
    """
    2段階モデルの「独立性破綻」を補正するモデル。

    v5.4: P補正モデルとE補正モデルに分解
    v5.5: P補正に init_score = logit(p_pred) を設定 (再学習化の防止)
          E補正の weight を 1/√p に変更 (ノイズ過剰適合の防止)

    P補正: 全サンプルで binary classification (init_score付き)
    E補正: 1着馬のみで log residual を 1/√p 重み付き回帰
    最終:  EV_corrected = P_corrected × E_corrected
    """

    E_CLIP_FLOOR: float = 1.0

    def __init__(
        self,
        *,
        ev_isotonic_calibrator: IsotonicRegression | None = None,
        ev_odds_band_scales: dict[str, float] | None = None,
    ) -> None:
        self.ev_isotonic_calibrator = ev_isotonic_calibrator
        self.ev_odds_band_scales = ev_odds_band_scales
        self._trained: bool = False
        # 遅延import (循環依存回避)
        from betting.odds_band_filter import OddsBandFilter
        self._odds_band_filter_cls = OddsBandFilter

    FEATURE_COLS: list[str] = [
        # 2段階モデルの出力 (v5.5: p_win_pred を除外 → init_scoreで代替)
        "e_return_win_pred",
        # 交互作用特徴量
        "p_x_e_interaction",
        "p_minus_e_gap",
        # 市場歪み
        "signed_log_error_win",
        "abs_log_error_win",
        "market_entropy",
        "popularity_rank",
        # FLB slope (市場集中度)
        "implied_prob_hhi",
        # 市場構造 (D-06: 歪度)
        "odds_skewness",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "field_size",
        # 騎手コンテキスト (Group C, Stage2)
        "jockey_wr_overall",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        # 調教師コンテキスト (Group D, Stage2)
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        # 騎手-調教師コンビ (B4, Stage2)
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
        # 市場クロス整合性 (MCF-07)
        "rl_favorite_in_wide_top1",
        "rl_trio_overlap",
        "rl_market_consistency",
        "rl_trio_odds_ratio",
        "rl_wide_harville_ratio",
        # レースレベル集約 (RLF-01~06)
        "rl_log_odds_entropy",
        "rl_odds_dispersion",
        "rl_top3_odds_gap",
        "rl_top1_odds",
        "rl_favorite_rank_gap",
        "rl_n_horses",
        # TRF-01/02/03 + INT-01/02/03/04: Phase 36
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        "grade_x_form_trend",
        "distance_x_closing_index",
        "grade_x_blood_prize_log",
        # HLF-01/02/03: Phase 36 HaronTime L4 + LapTime pace features
        "closing_speed_ratio_avg",
        "closing_speed_ratio_zscore",
        "closing_speed_ratio_trend",
        "harontime_last3f_avg",
        "harontime_last3f_zscore",
        "harontime_last3f_trend",
        # D-02: haron_race_gap (Phase 36.1 new)
        "haron_race_gap_avg",
        "haron_race_gap_zscore",
        "haron_race_gap_trend",
        # D-03: pace_adj_finish (Phase 36.1 new)
        "pace_ratio_avg",
        "pace_early_avg",
        "pace_mid_avg",
        "pace_late_avg",
        # HLF-02: HaronTime race-rank
        "closing_speed_ratio_avg_race_rank",
        "harontime_last3f_avg_race_rank",
    ]

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量を追加"""
        df["p_x_e_interaction"] = df["p_win_pred"] * df["e_return_win_pred"]
        df["p_minus_e_gap"] = np.abs(
            np.log(df["p_win_pred"] + 1e-8) - np.log(df["e_return_win_pred"] + 1e-8)
        )
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量データフレームを準備する"""
        # Fill missing feature columns with NaN (rl_* columns may be absent in tests)
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            import logging
            logging.getLogger(__name__).debug(
                "Missing feature columns filled with NaN: %s", missing[:5],
            )
            df = df.copy()
            for c in missing:
                df[c] = float("nan")
        features = df[self.FEATURE_COLS].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P補正モデルとE補正モデルをそれぞれ学習"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        df = df.copy()
        assert "ev_win" in df.columns, (
            "ev_win が必要です。先に WinTwoStageModel.predict_ev() を実行してください"
        )

        # PITリーク防止: race_date でソートしてから時系列分割
        if "race_date" in df.columns:
            df = df.sort_values("race_date").reset_index(drop=True)

        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # ── Model P: P補正 (全サンプル・binary classification) ──
        y_p = (df["kakuteijyuni"] == 1).astype(int)
        p_pred_clipped = np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))

        # P correction: train/valid split (80/20) with init_score (時系列分割)
        n_p = len(features)
        split_p = int(n_p * 0.8)
        train_idx_p = np.arange(split_p)
        valid_idx_p = np.arange(split_p, n_p)

        train_data_p = lgb.Dataset(
            features.iloc[train_idx_p],
            label=y_p.iloc[train_idx_p],
            init_score=init_score[train_idx_p],
        )
        valid_data_p = lgb.Dataset(
            features.iloc[valid_idx_p],
            label=y_p.iloc[valid_idx_p],
            init_score=init_score[valid_idx_p],
            reference=train_data_p,
        )

        self.p_correction_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_p,
            num_boost_round=300,
            valid_sets=[valid_data_p],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        # ── Model E: E補正 (1着馬のみ・1/√p 重み付き回帰) ──
        winners = df[df["kakuteijyuni"] == 1].copy()
        # PITリーク防止: race_date でソートしてから時系列分割
        if "race_date" in winners.columns:
            winners = winners.sort_values("race_date").reset_index(drop=True)
        e_pred_clipped = np.clip(winners["e_return_win_pred"], self.E_CLIP_FLOOR, None)
        winners["log_e_correction"] = np.log(
            winners["confirmed_odds"].clip(lower=self.E_CLIP_FLOOR)
        ) - np.log(e_pred_clipped)
        winners["_e_sample_weight"] = 1.0 / np.sqrt(np.clip(winners["p_win_pred"], 0.01, None))

        features_e = self._prepare_features(winners)
        e_weight = winners["_e_sample_weight"].values

        # E correction: train/valid split (80/20) with weight (時系列分割)
        n_e = len(features_e)
        split_e = int(n_e * 0.8)
        train_idx_e = np.arange(split_e)
        valid_idx_e = np.arange(split_e, n_e)

        train_data_e = lgb.Dataset(
            features_e.iloc[train_idx_e],
            label=winners["log_e_correction"].iloc[train_idx_e],
            weight=e_weight[train_idx_e],
        )
        valid_data_e = lgb.Dataset(
            features_e.iloc[valid_idx_e],
            label=winners["log_e_correction"].iloc[valid_idx_e],
            weight=e_weight[valid_idx_e],
            reference=train_data_e,
        )

        self.e_correction_model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_e,
            num_boost_round=300,
            valid_sets=[valid_data_e],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        self._trained = True

    def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全馬のEVをP補正×E補正で補正する。
        P_corrected = sigmoid(logit(P_pred) + correction_margin) を race 内で再正規化
        E_corrected = e_return_win_pred × exp(log_e_correction)
        EV_corrected = P_corrected × E_corrected

        未学習時 (_trained=False): 元予測を補正済み列へ写像する。
        """
        if not self._trained:
            df = df.copy()
            df["p_win_corrected"] = _normalize_probability_by_race(
                df, "p_win_pred", target_sum=1.0,
            )
            df["e_return_win_corrected"] = df["e_return_win_pred"].copy()
            df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
            df["ev_win_calibrated"] = df["ev_win_corrected"].copy()
            return df
        df = df.copy()
        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # P補正の適用:
        # LightGBM binary booster の predict() は probability を返すため、
        # init_score に加算する補正量は raw_score=True の margin を使う。
        p_pred_clipped = np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))
        p_best = _best_iteration(self.p_correction_model)
        raw_margin = self.p_correction_model.predict(  # type: ignore[union-attr]
            features,
            num_iteration=p_best,
            raw_score=True,
        )
        df["_p_win_corrected_raw"] = _sigmoid(raw_margin + init_score)
        df["p_win_corrected"] = _normalize_probability_by_race(
            df,
            "_p_win_corrected_raw",
            target_sum=1.0,
        )

        # E補正の適用
        e_best = _best_iteration(self.e_correction_model)
        log_e_corr = self.e_correction_model.predict(features, num_iteration=e_best)  # type: ignore[union-attr]
        df["e_return_win_corrected"] = df["e_return_win_pred"] * np.exp(log_e_corr)

        # 最終補正EV (PxE補正)
        df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]

        # --- Phase 19: Isotonic EV Calibration (D-08) ---
        if self.ev_isotonic_calibrator is not None:
            ev_input = df["ev_win_corrected"].values.astype(float)
            valid = np.isfinite(ev_input)
            calibrated = np.copy(ev_input)
            if valid.any():
                calibrated[valid] = self.ev_isotonic_calibrator.transform(ev_input[valid])
            df["ev_win_calibrated"] = calibrated
        else:
            df["ev_win_calibrated"] = df["ev_win_corrected"].copy()

        # --- Phase 19: Odds Band Residual Scaling (D-10) ---
        if self.ev_odds_band_scales is not None:
            # ★ M3 fix: 常に発走前oddsを使用 (学習/推論一貫性)
            odds_col = "odds"
            if odds_col in df.columns:
                odds = pd.to_numeric(df[odds_col], errors="coerce").values.astype(float)
                calibrated = df["ev_win_calibrated"].values.astype(float)
                OddsBandFilter = self._odds_band_filter_cls
                for (lo, hi), band_name in zip(OddsBandFilter.BANDS, OddsBandFilter.BAND_NAMES):
                    scale = self.ev_odds_band_scales.get(band_name, 1.0)
                    if abs(scale - 1.0) < 1e-9:
                        continue
                    mask = (odds >= lo) & (odds < hi) & np.isfinite(odds)
                    calibrated[mask] *= scale
                df["ev_win_calibrated"] = calibrated

        df = df.drop(columns=["_p_win_corrected_raw"], errors="ignore")
        return df


class PlaceEVCorrectionModel:
    """
    複勝用 EV 補正モデル — P/E 分解パターンの Place 版。

    Win 版 (EVCorrectionModel) と同じ構造だが、以下が異なる:
    - P-target: kakuteijyuni <= 3 (複勝的中)
    - P-init_score: logit(p_place_pred)
    - E-target: log(fukuoddslow) - log(e_return_place_pred)
    - E-filter: placed horses only (kakuteijyuni <= 3)
    - E-weight: 1/sqrt(p_place_pred)
    - 出力列: p_place_corrected, e_return_place_corrected, ev_place_corrected
    """

    E_CLIP_FLOOR: float = 1.0

    FEATURE_COLS: list[str] = [
        # 2段階モデルの出力
        "e_return_place_pred",
        "fukuoddslow",                 # 複勝オッズ (E-correction target context)
        "p_ability_place",             # PlaceAbilityModel 出力
        # 市場歪み
        "signed_log_error_win",
        "abs_log_error_win",
        "market_entropy",
        "popularity_rank",
        # FLB slope (市場集中度)
        "implied_prob_hhi",
        # 市場構造 (D-06: 歪度)
        "odds_skewness",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "field_size",
        # 騎手コンテキスト
        "jockey_wr_overall",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        # 調教師コンテキスト
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        # 騎手-調教師コンビ
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
        # 市場クロス整合性 (MCF-07)
        "rl_favorite_in_wide_top1",
        "rl_trio_overlap",
        "rl_market_consistency",
        "rl_trio_odds_ratio",
        "rl_wide_harville_ratio",
        # レースレベル集約 (RLF-01~06)
        "rl_log_odds_entropy",
        "rl_odds_dispersion",
        "rl_top3_odds_gap",
        "rl_top1_odds",
        "rl_favorite_rank_gap",
        "rl_n_horses",
        # TRF-01/02/03 + INT-01/02/03/04: Phase 36
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        "grade_x_form_trend",
        "distance_x_closing_index",
        "grade_x_blood_prize_log",
        # HLF-01/02/03: Phase 36 HaronTime L4 + LapTime pace features
        "closing_speed_ratio_avg",
        "closing_speed_ratio_zscore",
        "closing_speed_ratio_trend",
        "harontime_last3f_avg",
        "harontime_last3f_zscore",
        "harontime_last3f_trend",
        # D-02: haron_race_gap (Phase 36.1 new)
        "haron_race_gap_avg",
        "haron_race_gap_zscore",
        "haron_race_gap_trend",
        # D-03: pace_adj_finish (Phase 36.1 new)
        "pace_ratio_avg",
        "pace_early_avg",
        "pace_mid_avg",
        "pace_late_avg",
        # HLF-02: HaronTime race-rank
        "closing_speed_ratio_avg_race_rank",
        "harontime_last3f_avg_race_rank",
    ]

    def __init__(self) -> None:
        self.p_correction_model: lgb.Booster | None = None
        self.e_correction_model: lgb.Booster | None = None
        self._trained: bool = False

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量を追加"""
        df["p_x_e_interaction_place"] = df["p_place_pred"] * df["e_return_place_pred"]
        df["p_minus_e_gap_place"] = np.abs(
            np.log(df["p_place_pred"] + 1e-8) - np.log(df["e_return_place_pred"] + 1e-8)
        )
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量データフレームを準備する"""
        # Fill missing feature columns with NaN (rl_* columns may be absent in tests)
        _all_feature = self.FEATURE_COLS + ["p_x_e_interaction_place", "p_minus_e_gap_place"]
        missing = [c for c in _all_feature if c not in df.columns]
        if missing:
            df = df.copy()
            for c in missing:
                df[c] = float("nan")
        all_cols = _all_feature
        features = df[all_cols].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P補正モデルとE補正モデルをそれぞれ学習"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        df = df.copy()
        assert "ev_place" in df.columns, (
            "ev_place が必要です。先に PlaceTwoStageModel.predict_ev() を実行してください"
        )

        # PITリーク防止: race_date でソートしてから時系列分割
        if "race_date" in df.columns:
            df = df.sort_values("race_date").reset_index(drop=True)

        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # ── Model P: P補正 (全サンプル・binary classification) ──
        y_p = (df["kakuteijyuni"] <= 3).astype(int)
        p_pred_clipped = np.clip(df["p_place_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))

        # P correction: train/valid split (80/20) with init_score (時系列分割)
        n_p = len(features)
        split_p = int(n_p * 0.8)
        train_idx_p = np.arange(split_p)
        valid_idx_p = np.arange(split_p, n_p)

        train_data_p = lgb.Dataset(
            features.iloc[train_idx_p],
            label=y_p.iloc[train_idx_p],
            init_score=init_score[train_idx_p],
        )
        valid_data_p = lgb.Dataset(
            features.iloc[valid_idx_p],
            label=y_p.iloc[valid_idx_p],
            init_score=init_score[valid_idx_p],
            reference=train_data_p,
        )

        self.p_correction_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_p,
            num_boost_round=300,
            valid_sets=[valid_data_p],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        # ── Model E: E補正 (複勝的中馬のみ・1/√p 重み付き回帰) ──
        placed = df[df["kakuteijyuni"] <= 3].copy()
        # PITリーク防止: race_date でソートしてから時系列分割
        if "race_date" in placed.columns:
            placed = placed.sort_values("race_date").reset_index(drop=True)
        e_pred_clipped = np.clip(placed["e_return_place_pred"], self.E_CLIP_FLOOR, None)
        placed["log_e_correction"] = np.log(
            placed["fukuoddslow"].clip(lower=self.E_CLIP_FLOOR)
        ) - np.log(e_pred_clipped)
        placed["_e_sample_weight"] = 1.0 / np.sqrt(np.clip(placed["p_place_pred"], 0.01, None))

        features_e = self._prepare_features(placed)
        e_weight = placed["_e_sample_weight"].values

        # E correction: train/valid split (80/20) with weight (時系列分割)
        n_e = len(features_e)
        split_e = int(n_e * 0.8)
        train_idx_e = np.arange(split_e)
        valid_idx_e = np.arange(split_e, n_e)

        train_data_e = lgb.Dataset(
            features_e.iloc[train_idx_e],
            label=placed["log_e_correction"].iloc[train_idx_e],
            weight=e_weight[train_idx_e],
        )
        valid_data_e = lgb.Dataset(
            features_e.iloc[valid_idx_e],
            label=placed["log_e_correction"].iloc[valid_idx_e],
            weight=e_weight[valid_idx_e],
            reference=train_data_e,
        )

        callbacks: list = [lgb.early_stopping(100, verbose=False)]
        if n_e < 10:
            # サンプル数不足で早期停止をスキップ
            callbacks = []

        self.e_correction_model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_e,
            num_boost_round=300,
            valid_sets=[valid_data_e],
            callbacks=callbacks,
        )

        self._trained = True

    def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全馬の複勝EVをP補正×E補正で補正する。
        P_corrected = sigmoid(logit(P_pred) + correction_margin) を race 内で再正規化
        E_corrected = e_return_place_pred × exp(log_e_correction)
        EV_corrected = P_corrected × E_corrected

        未学習時 (_trained=False): 元予測を補正済み列へ写像する。
        """
        if not self._trained:
            df = df.copy()
            df["p_place_corrected"] = _normalize_probability_by_race(
                df,
                "p_place_pred",
                target_sum=3.0,
            )
            df["place_bucket_multiplier"] = _build_place_bucket_multiplier(df, "p_place_corrected")
            df["e_return_place_corrected"] = (
                df["e_return_place_pred"] * df["place_bucket_multiplier"]
            )
            df["ev_place_corrected"] = df["p_place_corrected"] * df["e_return_place_corrected"]
            return df

        df = df.copy()
        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # P補正の適用:
        # binary booster の probability 出力ではなく、raw margin を init_score に加算する。
        p_pred_clipped = np.clip(df["p_place_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))
        p_best = _best_iteration(self.p_correction_model)
        raw_margin = self.p_correction_model.predict(  # type: ignore[union-attr]
            features,
            num_iteration=p_best,
            raw_score=True,
        )
        df["_p_place_corrected_raw"] = _sigmoid(raw_margin + init_score)
        df["p_place_corrected"] = _normalize_probability_by_race(
            df,
            "_p_place_corrected_raw",
            target_sum=3.0,
        )

        # E補正の適用
        e_best = _best_iteration(self.e_correction_model)
        log_e_corr = self.e_correction_model.predict(features, num_iteration=e_best)  # type: ignore[union-attr]
        df["e_return_place_corrected"] = df["e_return_place_pred"] * np.exp(log_e_corr)
        df["place_bucket_multiplier"] = _build_place_bucket_multiplier(df, "p_place_corrected")
        df["e_return_place_corrected"] = (
            df["e_return_place_corrected"] * df["place_bucket_multiplier"]
        )

        # 最終補正EV
        df["ev_place_corrected"] = df["p_place_corrected"] * df["e_return_place_corrected"]
        df = df.drop(columns=["_p_place_corrected_raw"], errors="ignore")
        return df
