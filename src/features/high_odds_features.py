"""高オッズ的中パターン特徴量 — クラストラジェクトリ + フォーム改善率 + 環境変化適性

HorseHistoryFeatures の per-horse ループ内で呼び出される純粋関数群。
form_cycle_features.py のパターンに従う。

HODDS-02: クラストラジェクトリ (D-05, D-06, D-07)
  - 昇級/降級回数、ネット変化、最高クラス到達、クラス分散
  - V字回復パターン (降級→再昇級) フラグと降級期間

HODDS-03: フォーム改善率 (D-08, D-09)
  - EMAベース指数改善率 (タイムz-score + 正規化着順)
  - halflife=3 の確立パターン (horse_history_features.py と同一)

HODDS-04: 環境変化適性 (D-10, D-11)
  - 3変化(距離/サーフェス/馬場状態) x 3サブ特徴量(平均着順/勝率/経験回数)
  - 変更検出時のみ該当統計を計算、経験回数0はNaN
"""
from __future__ import annotations

import numpy as np

FEATURE_COLS: list[str] = [
    # HODDS-02: クラストラジェクトリ (D-05, D-06, D-07)
    "class_promotions",        # 直近5走の昇級回数
    "class_demotions",         # 直近5走の降級回数
    "class_net_change",        # ネット変化 (最終-最初)
    "class_max_level",         # 最高クラス到達レベル
    "class_level_std",         # クラス分散
    "v_recovery_flag",         # V字回復パターンフラグ (D-07)
    "v_recovery_duration",     # 降級期間（降級から再昇級までの走数）(D-07)
    # HODDS-03: フォーム改善率 (D-08, D-09)
    "time_improvement_rate",   # EMAベースタイム(z-score)改善率
    "position_improvement_rate", # EMAベース着順改善率
    # HODDS-04: 環境変化適性 (D-10, D-11)
    "dist_change_avg_pos",     # 距離変更時の該当距離平均着順
    "dist_change_win_rate",    # 距離変更時の該当距離勝率
    "dist_change_exp_count",   # 距離変更時の該当距離経験回数
    "surf_change_avg_pos",     # サーフェス変更時の該当サーフェス平均着順
    "surf_change_win_rate",    # サーフェス変更時の該当サーフェス勝率
    "surf_change_exp_count",   # サーフェス変更時の該当サーフェス経験回数
    "cond_change_avg_pos",     # 馬場状態変更時の該当馬場平均着順
    "cond_change_win_rate",    # 馬場状態変更時の該当馬場勝率
    "cond_change_exp_count",   # 馬場状態変更時の該当馬場経験回数
]

# クラスレベルマップ (horse_history_features.py からコピー、循環参照回避)
_CLASS_LEVEL_MAP: dict[str, float] = {
    "A": 8.0,
    "B": 7.0,
    "C": 6.0,
    "D": 5.0,
    "E": 4.0,
}


def _class_level_from_values(grade_code: object, jyoken_code: object) -> float:
    """grade_code/jyoken_codeからクラスレベルを取得。

    horse_history_features.py の同名関数と同じロジック。
    循環参照回避のためインライン化。
    """
    grade = str(grade_code).strip() if not _is_nan(grade_code) else ""
    if grade in _CLASS_LEVEL_MAP:
        return _CLASS_LEVEL_MAP[grade]
    # フォールバック: jyoken_code の数値
    val = _to_float(jyoken_code)
    return val


def _is_nan(value: object) -> bool:
    """値がNaNかどうかを判定。"""
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _to_float(value: object) -> float:
    """値をfloatに変換。失敗時はNaN。"""
    try:
        result = float(value)
        if np.isnan(result):
            return result
        return result
    except (TypeError, ValueError):
        return float("nan")


def compute_class_trajectory(
    gradecd_arr: np.ndarray,
    jyokencd1_arr: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """クラストラジェクトリ特徴量を計算。

    直近出走のgrade_code/jyoken_code配列から、
    昇級/降級回数、ネット変化、最高クラス到達、クラス分散、
    V字回復フラグと降級期間を計算する。

    Args:
        gradecd_arr: 過去N走のgrade_code配列
        jyokencd1_arr: 過去N走のjyoken_code配列 (フォールバック用)

    Returns:
        (class_promotions, class_demotions, class_net_change,
         class_max_level, class_level_std, v_recovery_flag, v_recovery_duration)
        データ不足時は全て NaN。
    """
    # 各要素のクラスレベルを計算
    levels = np.array([
        _class_level_from_values(gradecd_arr[i], jyokencd1_arr[i])
        for i in range(len(gradecd_arr))
    ])

    # NaN除外
    valid_mask = ~np.isnan(levels)
    levels_valid = levels[valid_mask]

    if len(levels_valid) < 2:
        return (
            float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
        )

    # 差分配列
    diffs = np.diff(levels_valid)

    class_promotions: float = float((diffs > 0).sum())
    class_demotions: float = float((diffs < 0).sum())
    class_net_change: float = float(levels_valid[-1] - levels_valid[0])
    class_max_level: float = float(levels_valid.max())
    class_level_std: float = float(levels_valid.std())

    # V字回復検出 (D-07)
    v_recovery_flag: float = 0.0
    v_recovery_duration: float = float("nan")

    # 最初の降級を探す
    first_demotion_idx = -1
    for i in range(len(diffs)):
        if diffs[i] < 0:
            first_demotion_idx = i
            break

    if first_demotion_idx >= 0:
        # 降級位置以降に再昇級があるか
        for j in range(first_demotion_idx + 1, len(diffs)):
            if diffs[j] > 0:
                v_recovery_flag = 1.0
                # 降級から再昇級までの走数 (インデックス差 + 1)
                v_recovery_duration = float(j - first_demotion_idx)
                break

    return (
        class_promotions,
        class_demotions,
        class_net_change,
        class_max_level,
        class_level_std,
        v_recovery_flag,
        v_recovery_duration,
    )


def compute_form_improvement_rate(
    zscore_arr: np.ndarray,
    kakuteijyuni_arr: np.ndarray,
    syussotosu_arr: np.ndarray,
    halflife: int = 3,
) -> tuple[float, float]:
    """EMAベースのフォーム改善率を計算。

    タイムz-scoreと正規化着順のEMA重み付け平均が全体平均より改善しているかを測定。
    正の値 = 直近改善。

    Args:
        zscore_arr: 過去N走のタイムz-score配列 (低いほど良いタイム)
        kakuteijyuni_arr: 過去N走の着順配列
        syussotosu_arr: 過去N走の出走頭数配列
        halflife: EMAの半減期 (default: 3)

    Returns:
        (time_improvement_rate, position_improvement_rate)
        データ不足時は両方 NaN。
    """
    # NaN除外: 全配列のいずれかがNaNのインデックスを除外
    valid_mask = (
        ~np.isnan(zscore_arr)
        & ~np.isnan(kakuteijyuni_arr)
        & ~np.isnan(syussotosu_arr)
        & (syussotosu_arr > 1)
    )

    zscore_valid = zscore_arr[valid_mask].astype(float)
    kj_valid = kakuteijyuni_arr[valid_mask].astype(float)
    ss_valid = syussotosu_arr[valid_mask].astype(float)

    if len(zscore_valid) < 2:
        return float("nan"), float("nan")

    # time_improvement_rate: z-score (低いほど良い)
    time_improvement_rate = _ema_improvement(zscore_valid, halflife, lower_is_better=True)

    # position_improvement_rate: 正規化着順 (低いほど良い)
    norm_pos = (kj_valid - 1) / np.maximum(ss_valid - 1, 1)
    position_improvement_rate = _ema_improvement(
        norm_pos, halflife, lower_is_better=True
    )

    return time_improvement_rate, position_improvement_rate


def _ema_improvement(
    values: np.ndarray,
    halflife: int,
    *,
    lower_is_better: bool = True,
) -> float:
    """EMA重み付け改善率を計算。

    EMA平均と全体平均の差を全体の標準偏差で正規化。
    lower_is_better=True: 正の値 = 直近改善 (EMAが全体より低い)

    Args:
        values: 値配列 (古い順→新しい順)
        halflife: EMA半減期
        lower_is_better: True=低い値が良い

    Returns:
        改善率。overall_std==0の場合は0.0。
    """
    n = len(values)

    # EMA重み付け (horse_history_features.py L714-723 パターン)
    decay = np.log(2) / halflife  # ≈ 0.231
    weights = (1 - decay) ** np.arange(n)
    weights = weights[::-1]  # index 0 = newest (highest weight)
    weights = weights / weights.sum()

    ema_mean = float(np.sum(values * weights))
    overall_mean = float(np.mean(values))
    overall_std = float(np.std(values))

    if overall_std < 1e-10:
        return 0.0

    if lower_is_better:
        # 正 = EMAが全体より低い = 直近改善
        return (overall_mean - ema_mean) / overall_std
    else:
        # 正 = EMAが全体より高い = 直近改善
        return (ema_mean - overall_mean) / overall_std


# ---------------------------------------------------------------------------
# 環境変化適性キー
# ---------------------------------------------------------------------------
_ENV_KEYS: list[str] = [
    "dist_change_avg_pos", "dist_change_win_rate", "dist_change_exp_count",
    "surf_change_avg_pos", "surf_change_win_rate", "surf_change_exp_count",
    "cond_change_avg_pos", "cond_change_win_rate", "cond_change_exp_count",
]


def _nan_env() -> dict[str, float]:
    """全キーNaNの結果dictを返す。"""
    return {k: float("nan") for k in _ENV_KEYS}


def _compute_change_stats(
    change_detected: bool,
    match_mask: np.ndarray,
    kakuteijyuni: np.ndarray,
    syussotosu: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """変更検出時に対象過去走の統計を計算。

    Args:
        change_detected: 変更が検出されたか
        match_mask: 対象条件に一致する過去走のブールマスク
        kakuteijyuni: 確定着順配列
        syussotosu: 出走頭数配列
        prefix: キー前置詞 ("dist_change", "surf_change", "cond_change")

    Returns:
        {prefix}_avg_pos, {prefix}_win_rate, {prefix}_exp_count
    """
    nan_result = {
        f"{prefix}_avg_pos": float("nan"),
        f"{prefix}_win_rate": float("nan"),
        f"{prefix}_exp_count": float("nan"),
    }
    if not change_detected:
        return nan_result

    matched_kj = kakuteijyuni[match_mask]
    matched_ss = syussotosu[match_mask]

    if len(matched_kj) == 0:
        return nan_result

    # avg_pos: 正規化着順 (pos-1)/(size-1) の平均。低いほど良い
    norm_pos = (matched_kj - 1) / np.maximum(matched_ss - 1, 1)
    avg_pos = float(np.nanmean(norm_pos))

    # win_rate: 1着の割合
    win_rate = float((matched_kj == 1).sum()) / float(len(matched_kj))

    # exp_count: 該当走数
    exp_count = float(len(matched_kj))

    return {
        f"{prefix}_avg_pos": avg_pos,
        f"{prefix}_win_rate": win_rate,
        f"{prefix}_exp_count": exp_count,
    }


def compute_env_adaptability(
    kakuteijyuni_arr: np.ndarray,
    syussotosu_arr: np.ndarray,
    distance_bin_arr: np.ndarray,
    surface_arr: np.ndarray,
    track_condition_arr: np.ndarray,
    current_distance_bin: str,
    current_surface: str,
    current_track_condition: float,
) -> dict[str, float]:
    """環境変化適性9特徴量を計算。

    3変化(距離/サーフェス/馬場状態) x 3サブ特徴量(平均着順/勝率/経験回数)。
    変更が検出された場合のみ該当変化の統計を計算。変更なしはNaN。

    Args:
        kakuteijyuni_arr: 過去走の確定着順配列
        syussotosu_arr: 過去走の出走頭数配列
        distance_bin_arr: 過去走の距離ビン配列 (str)
        surface_arr: 過去走のサーフェス配列 (str)
        track_condition_arr: 過去走の馬場状態コード配列 (float)
        current_distance_bin: 現在レースの距離ビン
        current_surface: 現在レースのサーフェス
        current_track_condition: 現在レースの馬場状態コード

    Returns:
        dict with 9 keys:
        dist_change_avg_pos, dist_change_win_rate, dist_change_exp_count,
        surf_change_avg_pos, surf_change_win_rate, surf_change_exp_count,
        cond_change_avg_pos, cond_change_win_rate, cond_change_exp_count
    """
    result = _nan_env()

    if len(kakuteijyuni_arr) == 0:
        return result

    kj = kakuteijyuni_arr.astype(float)
    ss = syussotosu_arr.astype(float)

    # 最後の過去走の値
    last_db = str(distance_bin_arr[-1])
    last_surf = str(surface_arr[-1])
    last_cond = _to_float(track_condition_arr[-1])

    # 距離変更検出: 最後の過去走と現在が異なる
    dist_changed = current_distance_bin != last_db
    dist_stats = _compute_change_stats(
        dist_changed,
        distance_bin_arr == current_distance_bin,
        kj, ss,
        "dist_change",
    )
    result.update(dist_stats)

    # サーフェス変更検出
    surf_changed = current_surface != last_surf
    surf_stats = _compute_change_stats(
        surf_changed,
        np.array([str(s) for s in surface_arr]) == current_surface,
        kj, ss,
        "surf_change",
    )
    result.update(surf_stats)

    # 馬場状態変更検出
    if np.isnan(last_cond) or np.isnan(current_track_condition):
        cond_changed = False
    else:
        cond_changed = current_track_condition != last_cond
    cond_match = np.array([
        (not np.isnan(_to_float(tc))) and _to_float(tc) == current_track_condition
        for tc in track_condition_arr
    ])
    cond_stats = _compute_change_stats(
        cond_changed,
        cond_match,
        kj, ss,
        "cond_change",
    )
    result.update(cond_stats)

    return result
