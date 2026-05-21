"""単勝/複勝 2段階モデル -- ゼロ偏重問題の根本解決 (§2)"""

from __future__ import annotations

import logging
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import TwoStageConfig

logger = logging.getLogger(__name__)


def _train_valid_split(
    features: pd.DataFrame,
    label: pd.Series,
    valid_ratio: float = 0.2,
    seed: int = 42,  # noqa: ARG001 — kept for API compat
    sample_weight: np.ndarray | None = None,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """学習データを時系列順に train/valid に分割。

    **前提条件**: 呼び出し側は df.sort_values("race_date") で事前にソートしておくこと。
    前の80%をtrain、後の20%をvalidにする。
    時系列データでのランダム分割による look-ahead bias を防止する。
    """
    n = len(features)
    split = int(n * (1 - valid_ratio))

    train_weight = sample_weight[:split] if sample_weight is not None else None
    valid_weight = sample_weight[split:] if sample_weight is not None else None

    train_data = lgb.Dataset(
        features.iloc[:split], label=label.iloc[:split], weight=train_weight,
    )
    valid_data = lgb.Dataset(
        features.iloc[split:], label=label.iloc[split:], weight=valid_weight,
        reference=train_data,
    )
    return train_data, valid_data


class WinTwoStageModel:
    """
    単勝2段階モデル
    Stage A: P(win)              ← 2値分類
    Stage B: E(win_odds | win)   ← 的中時払戻の回帰
    EV = P(win) × E(win_odds | win)

    市場コピー防止のため p_market_pred_win は除外し、
    market_log_error (正規化差分) のみを使用する (§4)。
    """

    FEATURE_COLS: list[str] = [
        # Stage1出力
        "p_ability_win",
        # Market Model正規化差分 (v5.3: signed/abs log_error)
        "signed_log_error_win",
        "abs_log_error_win",
        # オッズ変化率
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        # ODTS-01: オッズ加速度 (2次微分)
        "odds_acceleration",
        # ODTS-02: オッズ方向一貫性
        "odds_direction_consistency",
        "popularity_change_30_10",
        # 市場歪み
        "market_entropy",
        "popularity_rank",
        "overround",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        # FLB slope (市場歪みの非対称性)
        "odds_skewness",
        # 追加改善特徴量
        "draw_ratio",
        "class_move",
        "blinker_change",
        "is_nar_transfer",
        "nar_recent_ratio",
        "track_condition_delta",
        "pace_pressure",
        "pace_scenario_fit",
        # FEAT-02: 単勝特化新特徴量 (5 from HorseHistoryFeatures + 1 late-stage)
        "distance_change",
        "surface_change",
        "class_drop_bounce",
        "win_dominance",
        "freshness_score",
        # FEAT-02: 市場確率/能力確率比 (値 > 1.0 = 過小評価, < 1.0 = 過大評価)
        "odds_to_ability_ratio",
        # TSER-02: クラス調整フォーメトリック
        "class_adj_formetric",
        # TSER-03: z-score改善トラジェクトリ
        "haron_zscore_trend",
        # PACE-02: 実績ベースのペース適性
        "actual_pace_fit",
        # ODDS-01: オッズ乖離レース内相対特徴量
        "deviation_rank",
        "deviation_zscore",
        # TRF-01/02: Phase 36 race-rank + weighted_recent_form
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
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
        # 騎手-調教師コンビ (Stage2)
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
        # n_mining予想特徴量 (DATA-04)
        "dm_time_rank",
        "dm_time_zscore",
        "dm_confidence_range",
        "dm_time_margin_to_fav",
        # 繁殖牝馬産駒特徴量 (DATA-01)
        "dam_wr",
        "breeder_strength",
        # BMS拡張特徴量 (DATA-01)
        "bms_distance_wr",
        # コースレコード特徴量 (DATA-02)
        "course_record_time",
        # レース内相対比較特徴量 (DATA-03)
        "rel_norm_finish_zscore",
        "rel_timediff_rank",
        "rel_closing_index_rank",
        "rel_popularity_rank_zscore",
        "rel_fuku_odds_zscore",
        # レース内相対比較特徴量 -- Stage2 (INTER-01)
        "rel_p_ability_win_zscore",
        "rel_p_ability_win_rank",
        "rel_odds_ability_deviation",
        # 交互作用 (12): 既存3 + 新規9 (INTER-02)
        "kyakusitu_x_distance",       # category
        "kyakusitu_x_surface",        # category
        "weight_x_distance",
        "surface_x_distance_bin",     # category
        "blood_keito_x_surface",      # category
        "grade_code_x_distance_bin",  # category
        "sire_wr_x_distance",
        "blood_surface_wr_x_condition",
        "pace_pressure_x_closing_index",
        "haron_x_distance",
        "surface_x_past_perf",
        "weight_x_class",
        # INT-01/02/03: Phase 36 交互作用特徴量
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
        # Target Encoding (INTER-03): OOF-safe expanding window
        "te_blood_keito_cd",
        "te_kisyucode",
        "te_chokyosicode",
        # 市場構造 (D-06: 市場集中度)
        "implied_prob_hhi",
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
    ]

    def __init__(self, cfg: TwoStageConfig | None = None) -> None:
        self.cfg = cfg or TwoStageConfig()

    @classmethod
    def get_filtered_feature_cols(cls, noise_features: list[str]) -> list[str]:
        """ノイズ特徴量を除外した特徴量リストを返す (クラス変数を変更しない)。

        並列学習 (ThreadPoolExecutor) 内で安全に使用できる。
        remove_noise_features() とは異なり、クラス変数を変更しないため
        スレッドセーフである。

        Args:
            noise_features: 除外する特徴量名のリスト

        Returns:
            フィルタ済み特徴量リスト (新規リスト)
        """
        return [f for f in cls.FEATURE_COLS if f not in noise_features]

    @classmethod
    def remove_noise_features(cls, noise_features: list[str]) -> None:
        """FEATURE_COLSからノイズ特徴量を除外。

        SHAP/gain分析で特定されたノイズ特徴量をFEATURE_COLSから削除する。
        validate_noise_removal()でlogloss/AUCへの影響を検証した後に呼び出すこと。

        .. warning::
            このメソッドはクラス変数を変更するため、ThreadPoolExecutor等の
            並列処理内での使用は安全ではない。並列コンテキストでは
            get_filtered_feature_cols() を使用すること。

        Args:
            noise_features: 除外する特徴量名のリスト
        """
        warnings.warn(
            "remove_noise_features() mutates class-level state and is not thread-safe. "
            "Use get_filtered_feature_cols() instead, which returns a new list. "
            "remove_noise_features() will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        before = len(cls.FEATURE_COLS)
        removed = [f for f in noise_features if f in cls.FEATURE_COLS]
        cls.FEATURE_COLS = [f for f in cls.FEATURE_COLS if f not in noise_features]
        after = len(cls.FEATURE_COLS)
        if removed:
            logger.info(
                "Removed %d noise features from FEATURE_COLS (%d -> %d): %s",
                len(removed), before, after, removed,
            )
        else:
            logger.info("No features removed (none of %s found in FEATURE_COLS)", noise_features)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # FEAT-02: 推論時にodds_to_ability_ratioが未計算ならここで計算する
        # 訓練時は_train_submodel()で既に計算済みなのでスキップされる
        if (
            "odds_to_ability_ratio" in self.FEATURE_COLS
            and "odds_to_ability_ratio" not in df.columns
        ):
            if "p_market_win_adj" in df.columns and "p_ability_win" in df.columns:
                df = df.copy()
                p_market = df["p_market_win_adj"].clip(lower=1e-6)
                p_ability = df["p_ability_win"].clip(lower=1e-6)
                df["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

        # ODDS-01: 推論時にdeviation特徴量が未計算なら計算する
        if (
            "deviation_rank" in self.FEATURE_COLS
            and "deviation_rank" not in df.columns
        ):
            from features.odds_deviation_features import compute_odds_deviation_features
            df = compute_odds_deviation_features(df)

        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        features = df[available_cols].copy()
        # Int64 (nullable int) → float64 (LightGBMが対応する型)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code",
                     "track_condition_code",
                     "kyakusitu_x_distance", "kyakusitu_x_surface",
                     "surface_x_distance_bin", "blood_keito_x_surface",
                     "grade_code_x_distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train_hit_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P(win) の学習 (全出走馬・1着=1 / 他=0)"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        features = self._prepare_features(df)
        y = (df["kakuteijyuni"] == 1).astype(int)

        train_data, valid_data = _train_valid_split(features, y)
        pos_rate = y.mean()
        neg_pos_ratio = (1 - pos_rate) / max(pos_rate, 1e-6)
        # sqrt scaling: full neg/pos ratio causes overconfidence (esp. ~11x for 8% win rate)
        # sqrt compresses the ratio while keeping positive-class boost
        scale_pos = max(1.0, neg_pos_ratio ** 0.5)

        self.hit_model = lgb.train(
            {
                "objective": "binary",
                "metric": self.cfg.hit_metric,
                "learning_rate": self.cfg.hit_lr,
                "num_leaves": self.cfg.hit_leaves,
                "scale_pos_weight": scale_pos,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data,
            num_boost_round=self.cfg.hit_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

    def train_return_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """
        E(win_odds | win) の学習 (1着馬のみ)。
        ゼロ偏重を完全に排除 -- 学習データにゼロが含まれない。
        ターゲットを log(odds) に変換し、高オッズ域の過大評価を抑制。

        Args:
            df: "confirmed_odds" 列が必須 (的中時払戻額のターゲット)
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        hit_df = df[df["kakuteijyuni"] == 1].copy()
        if len(hit_df) < self.cfg.min_hit_samples:
            raise ValueError(
                f"Stage B 学習には最低 {self.cfg.min_hit_samples} 件の"
                f"的中サンプルが必要。現在: {len(hit_df)} 件"
            )

        features = self._prepare_features(hit_df)
        y = np.log1p(hit_df["confirmed_odds"].clip(lower=1.0))

        # 低確率(高オッズ)サンプルの重みを抑制: 1/√p で高オッズ域の過大評価を防ぐ
        p_win = hit_df.get("p_win_pred")
        sample_weight = None
        if p_win is not None:
            sample_weight = (1.0 / np.sqrt(np.clip(p_win, 0.01, None))).values

        params = {
            "objective": "regression_l1",
            "metric": self.cfg.return_metric,
            "learning_rate": self.cfg.return_lr,
            "num_leaves": self.cfg.return_leaves,
            "feature_fraction": 0.7,
            "num_threads": num_threads,
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

        if len(features) < 10:
            self.return_model = lgb.train(
                params,
                lgb.Dataset(features, label=y, weight=sample_weight),
                num_boost_round=self.cfg.return_rounds,
            )
        else:
            train_data, valid_data = _train_valid_split(features, y, sample_weight=sample_weight)
            self.return_model = lgb.train(
                params,
                train_data,
                num_boost_round=self.cfg.return_rounds,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EV_win = P(win) × E(win_odds | win)
        """
        df = df.copy()
        features = self._prepare_features(df)

        hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
        ret_iter = (
            self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
        )
        df["p_win_pred"] = self.hit_model.predict(features, num_iteration=hit_iter)
        log_return = self.return_model.predict(features, num_iteration=ret_iter)
        df["e_return_win_pred"] = np.clip(np.expm1(log_return), 1.0, None)
        df["ev_win"] = df["p_win_pred"] * df["e_return_win_pred"]
        return df


class PlaceTwoStageModel:
    """
    複勝2段階モデル
    Stage A: P(place)               ← 3着以内かどうかの分類
    Stage B: E(place_odds | place)  ← 的中時払戻の回帰

    複勝は的中率が高い (約18〜35%) ため Stage B の学習データが豊富。
    return_leaves を少し増やせる (25)。
    """

    # --- Hit model (Stage A): 確率分類用 ---
    # fukuoddslow, tanodds は除外 — 二重計数 (特徴量 + edge formula) を防止。
    # 代わりに馬レベル特徴量を追加し、情報ボトルネックを解消。
    HIT_FEATURE_COLS: list[str] = [
        # Stage1 出力
        "p_ability_win",
        "p_ability_place",  # PlaceAbilityModel 出力
        # Market Model 正規化差分 (間接的市場情報)
        "signed_log_error_win",
        "abs_log_error_win",
        # --- 馬レベル特徴量 (新規) ---
        "norm_finish_logit_avg",
        "harontimel5_zscore",
        "closing_index_avg",
        "weight_zscore",
        "days_since_last_race",
        "rest_category",
        "is_debut",
        "form_trend",
        "form_consistency",
        "blood_surface_wr",
        "blood_distance_wr",
        "jockey_wr_overall",
        "trainer_wr_overall",
        "jt_combo_place_rate",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        "jt_combo_wr",
        "jt_combo_starts",
        "jt_combo_prize_log",
        "course_wr",
        "draw_ratio",
        "class_move",
        "blinker_change",
        "is_nar_transfer",
        "nar_recent_ratio",
        "track_condition_delta",
        "pace_pressure",
        "pace_scenario_fit",
        # --- 間接的市場情報 (既存) ---
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        "popularity_change_30_10",
        "market_entropy",
        "popularity_rank",
        "overround",
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        "odds_skewness",
        # --- v5: レースコンテキスト特徴量 ---
        "race_mean_fuku_odds",
        "race_std_fuku_odds",
        "odds_gap_fav12",
        "odds_popularity_gap",
        "surface_track_interaction",
        # n_mining予想特徴量 (DATA-04)
        "dm_time_rank",
        "dm_time_zscore",
        "dm_confidence_range",
        # 繁殖牝馬産駒特徴量 (DATA-01)
        "dam_wr",
        "dam_surface_wr",
        # BMS拡張特徴量 (DATA-01)
        "bms_surface_wr",
        # コースレコード特徴量 (DATA-02)
        "course_record_time",
        # レース内相対比較特徴量 (DATA-03)
        "rel_norm_finish_zscore",
        "rel_sire_quality_rank",
        "rel_haron_vs_mean",
        # オッズ相対特徴量 (INTER-01)
        "rel_popularity_rank_zscore",
        "rel_fuku_odds_zscore",
        # Stage2能力値相対特徴量 (INTER-01)
        "rel_p_ability_win_zscore",
        "rel_p_ability_win_rank",
        "rel_odds_ability_deviation",
        # 交互作用 (12): 既存3 + 新規9 (INTER-02)
        "kyakusitu_x_distance",       # category
        "kyakusitu_x_surface",        # category
        "weight_x_distance",
        "surface_x_distance_bin",     # category
        "blood_keito_x_surface",      # category
        "grade_code_x_distance_bin",  # category
        "sire_wr_x_distance",
        "blood_surface_wr_x_condition",
        "pace_pressure_x_closing_index",
        "haron_x_distance",
        "surface_x_past_perf",
        "weight_x_class",
        # TRF-01/02: Phase 36 race-rank + weighted_recent_form
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        # INT-01/02/03: Phase 36 交互作用特徴量
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
        # Target Encoding (INTER-03): OOF-safe expanding window
        "te_blood_keito_cd",
        "te_kisyucode",
        "te_chokyosicode",
        # 市場構造 (D-06: 市場集中度)
        "implied_prob_hhi",
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
    ]

    # --- Return model (Stage B): 配当回帰用 ---
    # fukuoddslow はターゲットと同じため除外 (target leakage → e_return ≈ fukuoddslow に退化)
    # 代わりに tanodds は単勝オッズとして市場規模の代理指標として使用
    RETURN_FEATURE_COLS: list[str] = [
        # Stage1 出力
        "p_ability_win",
        "p_ability_place",  # PlaceAbilityModel 出力
        # Market Model 正規化差分
        "signed_log_error_win",
        "abs_log_error_win",
        # 単勝オッズ (市場規模の代理指標、複勝オッズとは異なる情報)
        "tanodds",
        # オッズ変化率
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        "popularity_change_30_10",
        # 市場歪み
        "market_entropy",
        "popularity_rank",
        "overround",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        # FLB slope
        "odds_skewness",
        # 追加改善特徴量
        "draw_ratio",
        "class_move",
        "blinker_change",
        "is_nar_transfer",
        "nar_recent_ratio",
        "track_condition_delta",
        "pace_pressure",
        "pace_scenario_fit",
        # FEAT-02: 単勝特化新特徴量 (WinTwoStageModelと共有)
        "distance_change",
        "surface_change",
        "class_drop_bounce",
        "win_dominance",
        "freshness_score",
        "odds_to_ability_ratio",
        # Phase 5: Foundation Features
        "class_adj_formetric",
        "haron_zscore_trend",
        "pace_corner_stability",
        "pace_closing_power",
        "pace_position_consistency",
        "actual_pace_fit",
        "odds_acceleration",
        "odds_direction_consistency",
        # ODDS-01: Phase 6 deviation features
        "deviation_rank",
        "deviation_zscore",
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
        # 騎手-調教師コンビ (Stage2)
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
        # n_mining予想特徴量 (DATA-04)
        "dm_time_rank",
        "dm_time_zscore",
        "dm_confidence_range",
        "dm_time_margin_to_fav",
        # TRF-01/02: Phase 36 race-rank + weighted_recent_form
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        # 繁殖牝馬産駒特徴量 (DATA-01)
        "dam_wr",
        "breeder_strength",
        # BMS拡張特徴量 (DATA-01)
        "bms_distance_wr",
        # コースレコード特徴量 (DATA-02)
        "course_record_time",
        # レース内相対比較特徴量 (DATA-03)
        "rel_norm_finish_zscore",
        "rel_timediff_rank",
        "rel_closing_index_rank",
        # オッズ相対特徴量 (INTER-01)
        "rel_popularity_rank_zscore",
        "rel_fuku_odds_zscore",
        # Stage2能力値相対特徴量 (INTER-01)
        "rel_p_ability_win_zscore",
        "rel_p_ability_win_rank",
        "rel_odds_ability_deviation",
        # 交互作用 (12): 既存3 + 新規9 (INTER-02)
        "kyakusitu_x_distance",       # category
        "kyakusitu_x_surface",        # category
        "weight_x_distance",
        "surface_x_distance_bin",     # category
        "blood_keito_x_surface",      # category
        "grade_code_x_distance_bin",  # category
        "sire_wr_x_distance",
        "blood_surface_wr_x_condition",
        "pace_pressure_x_closing_index",
        "haron_x_distance",
        "surface_x_past_perf",
        "weight_x_class",
        # INT-01/02/03: Phase 36 交互作用特徴量
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
        # Target Encoding (INTER-03): OOF-safe expanding window
        "te_blood_keito_cd",
        "te_kisyucode",
        "te_chokyosicode",
        # 市場構造 (D-06: 市場集中度)
        "implied_prob_hhi",
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
    ]

    # 後方互換: FEATURE_COLS は return model のリストを返す (最も情報量が多いため)
    # list()で独立コピーを作成 -- RETURN_FEATURE_COLSとの共有ミュータブル参照を防ぐ
    FEATURE_COLS: list[str] = list(RETURN_FEATURE_COLS)

    def __init__(self, cfg: TwoStageConfig | None = None) -> None:
        self.cfg = cfg or TwoStageConfig()

    def _prepare_features(
        self, df: pd.DataFrame, *, use_cols: list[str] | None = None
    ) -> pd.DataFrame:
        cols = use_cols or self.FEATURE_COLS
        # FEAT-02: 推論時にodds_to_ability_ratioが未計算ならここで計算する
        # 訓練時は_train_submodel()で既に計算済みなのでスキップされる
        if (
            "odds_to_ability_ratio" in cols
            and "odds_to_ability_ratio" not in df.columns
        ):
            if "p_market_win_adj" in df.columns and "p_ability_win" in df.columns:
                df = df.copy()
                p_market = df["p_market_win_adj"].clip(lower=1e-6)
                p_ability = df["p_ability_win"].clip(lower=1e-6)
                df["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

        # ODDS-01: 推論時にdeviation特徴量が未計算なら計算する
        if (
            "deviation_rank" in cols
            and "deviation_rank" not in df.columns
        ):
            from features.odds_deviation_features import compute_odds_deviation_features
            df = compute_odds_deviation_features(df)

        # v5: 新規特徴量はテストデータに存在しない場合があるため、存在する列のみ使用
        available_cols = [c for c in cols if c in df.columns]
        features = df[available_cols].copy()
        # Int64 (nullable int) → float64 (LightGBMが対応する型)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code",
                     "track_condition_code",
                     "kyakusitu_x_distance", "kyakusitu_x_surface",
                     "surface_x_distance_bin", "blood_keito_x_surface",
                     "grade_code_x_distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train_hit_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P(place) の学習 (3着以内=1 / それ以外=0)

        Args:
            df: "fukuoddslow" 列が必須 (Benter combination 用バリデーション保存)
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
        y = (df["kakuteijyuni"] <= 3).astype(int)

        train_data, valid_data = _train_valid_split(features, y)
        self.hit_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": self.cfg.hit_lr,
                "num_leaves": self.cfg.hit_leaves,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data,
            num_boost_round=self.cfg.hit_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        # バリデーション予測を保存 (Benter combination + isotonic fitting 用)
        n = len(features)
        split = int(n * 0.8)
        hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
        self._val_p_raw = self.hit_model.predict(
            features.iloc[split:], num_iteration=hit_iter
        )
        self._val_y = y.iloc[split:].values
        self._val_fukuoddslow = df["fukuoddslow"].iloc[split:].values

    def train_return_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """E(place_odds | place) の学習 (3着以内のみ)"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        hit_df = df[df["kakuteijyuni"] <= 3].copy()

        features = self._prepare_features(hit_df, use_cols=self.RETURN_FEATURE_COLS)
        y = hit_df["fukuoddslow"]

        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": self.cfg.return_lr,
            "num_leaves": 25,  # 複勝はサンプル多めなので少し深く
            "feature_fraction": 0.7,
            "num_threads": num_threads,
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

        if len(features) < 10:
            # サンプル数が少なすぎる場合は early stopping なし
            self.return_model = lgb.train(
                params,
                lgb.Dataset(features, label=y),
                num_boost_round=self.cfg.return_rounds,
            )
        else:
            train_data, valid_data = _train_valid_split(features, y)
            self.return_model = lgb.train(
                params,
                train_data,
                num_boost_round=self.cfg.return_rounds,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """EV_place = P(place) × E(place_odds | place)"""
        df = df.copy()
        hit_features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
        ret_features = self._prepare_features(df, use_cols=self.RETURN_FEATURE_COLS)

        hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
        ret_iter = (
            self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
        )

        df["p_place_pred"] = self.hit_model.predict(hit_features, num_iteration=hit_iter)
        df["e_return_place_pred"] = self.return_model.predict(ret_features, num_iteration=ret_iter)
        df["ev_place"] = df["p_place_pred"] * df["e_return_place_pred"]
        return df
