"""Group E: 交互作用特徴量 + v5 レースコンテキスト特徴量"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 15個の交互作用特徴量名 (既存3 + INTER-02:9 + INT-01/02/03:3)
INTERACTION_COLS: list[str] = [
    # 既存 (3)
    "kyakusitu_x_distance",
    "kyakusitu_x_surface",
    "weight_x_distance",
    # 新規: カテゴリ積 (3)
    "surface_x_distance_bin",
    "blood_keito_x_surface",
    "grade_code_x_distance_bin",
    # 新規: 数値積 (6)
    "sire_wr_x_distance",
    "blood_surface_wr_x_condition",
    "pace_pressure_x_closing_index",
    "haron_x_distance",
    "surface_x_past_perf",
    "weight_x_class",
    # Phase 36 INT-01/02/03: TRF/INT交互作用 (3)
    "grade_x_form_trend",
    "distance_x_closing_index",
    "grade_x_blood_prize_log",
]


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    脚質×距離/馬場 + 体重×距離 の交互作用特徴量を追加。
    LightGBMカテゴリとして扱うため、文字列結合 → astype("category")。

    v5: レースコンテキスト特徴量 (オッズギャップ、レース荒れ指標) を追加。
    INTER-02: ドメイン知識交互作用項 (計12個) を追加。
    """
    df = df.copy()

    # 脚質×距離bin (カテゴリ積)
    # LEAK防止: kyakusitukubun_cd (過去) のみ使用。kyakusitukubun (現在=ポスト) は不可。
    if "kyakusitukubun_cd" in df.columns and "distance_bin" in df.columns:
        df["kyakusitu_x_distance"] = (
            df["kyakusitukubun_cd"].astype(str) + "_" + df["distance_bin"].astype(str)
        ).astype("category")

    # 脚質×馬場 (カテゴリ積)
    if "kyakusitukubun_cd" in df.columns and "surface" in df.columns:
        df["kyakusitu_x_surface"] = (
            df["kyakusitukubun_cd"].astype(str) + "_" + df["surface"].astype(str)
        ).astype("category")

    # 馬体重列名の解決 (weight_absolute または ba_taijyu)
    weight_col = "weight_absolute" if "weight_absolute" in df.columns else "bataijyu"

    # 馬体重×距離 (数値積)
    # NaNポリシー: いずれかがNaNなら結果もNaN (fillna(0)は使わない)
    if weight_col in df.columns and "kyori" in df.columns:
        df["weight_x_distance"] = (df[weight_col] * df["kyori"]).where(
            df[weight_col].notna() & df["kyori"].notna(),
            other=float("nan"),
        )

    # --- INTER-02: 新規ドメイン知識交互作用項 (9個) ---

    # カテゴリ積 (3個)
    # 馬場×距離bin
    if "surface" in df.columns and "distance_bin" in df.columns:
        df["surface_x_distance_bin"] = (
            df["surface"].astype(str) + "_" + df["distance_bin"].astype(str)
        ).astype("category")

    # 血統系統×馬場
    if "blood_keito_cd" in df.columns and "surface" in df.columns:
        keito = pd.to_numeric(df["blood_keito_cd"], errors="coerce")
        # blood_keito_cdが数値化できない (NaN) の場合はスキップ
        if keito.notna().any():
            df["blood_keito_x_surface"] = (
                df["blood_keito_cd"].astype(str) + "_" + df["surface"].astype(str)
            ).astype("category")

    # グレード×距離bin
    if "grade_code" in df.columns and "distance_bin" in df.columns:
        df["grade_code_x_distance_bin"] = (
            df["grade_code"].astype(str) + "_" + df["distance_bin"].astype(str)
        ).astype("category")

    # 数値積 (6個) -- .where() でNaN安全

    # 種牡馬成績×距離
    if "sire_wr" in df.columns and "kyori" in df.columns:
        df["sire_wr_x_distance"] = (df["sire_wr"] * df["kyori"]).where(
            df["sire_wr"].notna() & df["kyori"].notna(),
            other=float("nan"),
        )

    # 血統馬場勝率×馬場状態
    if "blood_surface_wr" in df.columns and "track_condition_code" in df.columns:
        df["blood_surface_wr_x_condition"] = (
            df["blood_surface_wr"] * df["track_condition_code"]
        ).where(
            df["blood_surface_wr"].notna() & df["track_condition_code"].notna(),
            other=float("nan"),
        )

    # ペース圧力×追込指数
    if "pace_pressure" in df.columns and "closing_index_avg" in df.columns:
        df["pace_pressure_x_closing_index"] = (
            df["pace_pressure"] * df["closing_index_avg"]
        ).where(
            df["pace_pressure"].notna() & df["closing_index_avg"].notna(),
            other=float("nan"),
        )

    # 末脚×距離
    if "harontimel5_avg" in df.columns and "kyori" in df.columns:
        df["haron_x_distance"] = (df["harontimel5_avg"] * df["kyori"]).where(
            df["harontimel5_avg"].notna() & df["kyori"].notna(),
            other=float("nan"),
        )

    # 馬場×過去成績 (surface_code * norm_finish_logit_avg)
    if "norm_finish_logit_avg" in df.columns and "surface" in df.columns:
        surface_code = df["surface"].map({"turf": 1, "dirt": 2}).fillna(0)
        df["surface_x_past_perf"] = (df["norm_finish_logit_avg"] * surface_code).where(
            df["norm_finish_logit_avg"].notna(),
            other=float("nan"),
        )

    # 馬体×クラス (正規化済みclass_levelを優先)
    grade_num = pd.Series(np.nan, index=df.index, dtype=float)
    if "class_level_current" in df.columns:
        grade_num = pd.to_numeric(df["class_level_current"], errors="coerce")
    elif "grade_code" in df.columns:
        grade_map = {
            "A": 8,
            "B": 7,
            "C": 6,
            "D": 5.5,
            "L": 5.5,
            "E": 5,
            "G": 5.5,
            "H": 5,
            "G1": 5,
            "G2": 4,
            "G3": 3,
            "J.G1": 5,
            "J.G2": 4,
            "J.G3": 3,
            "OP": 2,
        }
        grade_num = df["grade_code"].map(grade_map).astype(float)
    if weight_col in df.columns and grade_num.notna().any():
        df["weight_x_class"] = (df[weight_col] * grade_num).where(
            df[weight_col].notna() & grade_num.notna(),
            other=float("nan"),
        )

    # --- Phase 36: INT-01/02/03 交互作用特徴量 ---

    # INT-01: grade_x_form_trend (grade_code × form_trend, D-10)
    if "grade_code" in df.columns and "form_trend" in df.columns:
        df["grade_x_form_trend"] = (grade_num * df["form_trend"]).where(
            grade_num.notna() & df["form_trend"].notna(),
            other=float("nan"),
        )

    # INT-02: distance_x_closing_index (kyori × closing_index_avg, D-11)
    if "kyori" in df.columns and "closing_index_avg" in df.columns:
        df["distance_x_closing_index"] = (
            df["kyori"] * df["closing_index_avg"]
        ).where(
            df["kyori"].notna() & df["closing_index_avg"].notna(),
            other=float("nan"),
        )

    # INT-03: grade_x_blood_prize_log (grade_code × blood_prize_log, D-12)
    if "grade_code" in df.columns and "blood_prize_log" in df.columns:
        df["grade_x_blood_prize_log"] = (grade_num * df["blood_prize_log"]).where(
            grade_num.notna() & df["blood_prize_log"].notna(),
            other=float("nan"),
        )

    # --- v5: レースコンテキスト特徴量 ---
    _add_race_context_features(df)
    _add_pace_projection_features(df)

    return df


def _add_race_context_features(df: pd.DataFrame) -> None:
    """レースレベルのコンテキスト特徴量をインプレースで追加。

    以下の特徴量は race_id ごとに計算され、全馬に同じ値が付与される。
    PIT漏れなし: fukuoddslow は発走前オッズ (race_predictor で pre_post_odds から取得)。
    """
    if "race_id" not in df.columns:
        return

    odds_col = "fukuoddslow"

    # 1. レース平均オッズ (荒れやすさの代理指標)
    if odds_col in df.columns:
        race_mean_odds = df.groupby("race_id", observed=True)[odds_col].transform("mean")
        df["race_mean_fuku_odds"] = race_mean_odds

        # 2. レースオッズ標準偏差 (オッズ分散 → 荒れやすさ)
        race_std_odds = df.groupby("race_id", observed=True)[odds_col].transform("std")
        df["race_std_fuku_odds"] = race_std_odds.fillna(0)

        # 3. 人気1位と2位のオッズギャップ (レースの予測難易度)
        if "popularity_rank" in df.columns:
            pop1_odds = (
                df[df["popularity_rank"] == 1]
                .groupby("race_id", observed=True)[odds_col].first()
            )
            pop2_odds = (
                df[df["popularity_rank"] == 2]
                .groupby("race_id", observed=True)[odds_col].first()
            )
            odds_gap = (pop1_odds - pop2_odds).reindex(df["race_id"]).values
            df["odds_gap_fav12"] = pd.Series(odds_gap, index=df.index)

        # 4. オッズ順位と人気順位の乖離 (市場の非効率性指標)
        if "popularity_rank" in df.columns:
            odds_rank = df.groupby("race_id", observed=True)[odds_col].rank(method="min")
            df["odds_popularity_gap"] = (odds_rank - df["popularity_rank"]).abs()

    # 5. サーフェス×馬場状態交互作用 (数値)
    if "surface" in df.columns and "track_condition_code" in df.columns:
        surface_code = df["surface"].map({"turf": 1, "dirt": 2}).fillna(0)
        df["surface_track_interaction"] = surface_code * df["track_condition_code"].fillna(0)


def _add_pace_projection_features(df: pd.DataFrame) -> None:
    """履歴脚質から当日レースのペース投影特徴量を追加。"""
    if "race_id" not in df.columns or "kyakusitukubun_cd" not in df.columns:
        return

    style = pd.to_numeric(df["kyakusitukubun_cd"], errors="coerce")
    if "field_size" in df.columns:
        field_size = pd.to_numeric(df["field_size"], errors="coerce")
    else:
        field_size = pd.Series(np.nan, index=df.index, dtype=float)
    if field_size.isna().all():
        field_size = df.groupby("race_id", observed=True)["race_id"].transform("size").astype(float)

    field_size = field_size.clip(lower=1)
    front_share = (
        style.isin([1, 2]).groupby(df["race_id"], observed=True).transform("sum")
        / field_size
    )
    closer_share = (
        style.isin([3, 4]).groupby(df["race_id"], observed=True).transform("sum")
        / field_size
    )
    style_fit = style.map({1: -1.0, 2: -0.5, 3: 0.5, 4: 1.0}).fillna(0.0)

    df["pace_pressure"] = front_share.astype(float)
    df["closer_share"] = closer_share.astype(float)
    df["pace_scenario_fit"] = (style_fit * df["pace_pressure"]).astype(float)

    # PACE-02: actual_pace_fit — 実績ベースのペース適性 (D-13, D-14, D-15)
    # front_runner (脚質1=逃げ, 2=先行) → front_pace_wr
    # closer (脚質3=差し, 4=追込) → closing_pace_wr
    if "front_pace_wr" in df.columns and "closing_pace_wr" in df.columns:
        is_front_runner = style.isin([1, 2])
        is_closer = style.isin([3, 4])
        df["actual_pace_fit"] = np.where(
            is_front_runner, df["front_pace_wr"],
            np.where(is_closer, df["closing_pace_wr"], np.nan),
        )
