"""
単勝 Simple EV / 市場差分メタモデル 軽量分析スクリプト
=====================================================
既存バックテストデータのみを用いて、以下4分析を実行:

  Analysis A: Simple EV比較 (8定義)
    - p_win_pred × tanodds (Return Model疑似なし)
    - p_win_corrected × tanodds
    - p_win_final × tanodds
    - p_market_win_norm × tanodds (市場BL, sanity check)
    - ev_win (現行: p × e_return)
    - ev_win_corrected
    - ev_win_calibrated
    - p_ability_win × tanodds (Stage1のみ)

  Analysis B: 市場差分AND条件
    - edge_ratio × pred_ev の複合条件ROI

  Analysis C: メタモデル目的変数設計 (静的テーブル)

  Analysis D: 最小アブレーション案 (静的テーブル)

重要な限定事項:
  今回のSimple EVは、既存のp_win_*列にtanoddsを掛けた「疑似アブレーション」。
  Returnモデルなしで再学習したわけではない。本当にReturnなし版を作るには
  WinTwoStageModel自体の再学習が必要であり、それは次段階で別途実装。

使用データ:
  - data/backtest/bt_{2024,2025}_horse_features.parquet

出力:
  - data/analysis/win_simple_ev_meta_design.json
  - data/analysis/win_simple_ev_meta_design.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
BACKTEST_DIR = DATA_DIR / "backtest"

# ── ROI 定義 ─────────────────────────────────────────────────────
ROI_NOTE = (
    "ROIは各グループ内の全馬を単勝100円で購入した場合の仮想回収率とする。"
    "市場ベースラインA4も pred_ev = p_market_win_norm × tanodds を計算するが、"
    "AI候補とは別のsanity checkとして扱う。"
)

# ── EV定義 (Analysis A) ──────────────────────────────────────────
EV_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "A1", "label": "p_pred × odds",
        "p_col": "p_win_pred", "ev_type": "simple",
        "uses_return_model": False,
        "desc": "補正前確率 × 単勝オッズ (疑似Returnなし)",
    },
    {
        "id": "A2", "label": "p_corrected × odds",
        "p_col": "p_win_corrected", "ev_type": "simple",
        "uses_return_model": False,
        "desc": "P補正後確率 × 単勝オッズ",
    },
    {
        "id": "A3", "label": "p_final × odds",
        "p_col": "p_win_final", "ev_type": "simple",
        "uses_return_model": False,
        "desc": "MAWC後確率 × 単勝オッズ",
    },
    {
        "id": "A4", "label": "p_market × odds (BL)",
        "p_col": "p_market_win_norm", "ev_type": "simple",
        "uses_return_model": False,
        "desc": "市場正規化確率 × 単勝オッズ (sanity check: 判別力弱)",
    },
    {
        "id": "A5", "label": "ev_win (現行)",
        "p_col": "p_win_pred", "ev_type": "column", "ev_col": "ev_win",
        "uses_return_model": True,
        "desc": "p_win_pred × e_return_win_pred (現行2段階EV)",
    },
    {
        "id": "A6", "label": "ev_win_corrected",
        "p_col": "p_win_corrected", "ev_type": "column", "ev_col": "ev_win_corrected",
        "uses_return_model": True,
        "desc": "EV補正後の2段階EV",
    },
    {
        "id": "A7", "label": "ev_win_calibrated",
        "p_col": "p_win_corrected", "ev_type": "column", "ev_col": "ev_win_calibrated",
        "uses_return_model": True,
        "desc": "Isotonic校正 + オッズ帯残差スケール後",
    },
    {
        "id": "A8", "label": "p_ability × odds",
        "p_col": "p_ability_win", "ev_type": "simple",
        "uses_return_model": False,
        "desc": "Stage1 AbilityModel (ランカーsoftmax) × 単勝オッズ",
    },
]

# ── 分析パラメータ ──────────────────────────────────────────────
EV_TOPN_PCTS = [1, 5, 10]
EV_THRESHOLDS = [1.1, 1.2, 1.3]
ODDS_BANDS: list[tuple[float, float | None]] = [
    (1.0, 3.0), (3.0, 5.0), (5.0, 10.0),
    (10.0, 20.0), (20.0, 50.0), (50.0, None),
]

# ── AND条件 (Analysis B) ────────────────────────────────────────
CONDITION_FILTERS: list[dict[str, Any]] = [
    {"id": "B1", "label": "pred_ev>1.1 & edge_ratio>1.05",
     "desc": "AI確率EV>1.1 かつ AI/市場比>1.05"},
    {"id": "B2", "label": "pred_ev>1.1 & edge_ratio>1.10",
     "desc": "AI確率EV>1.1 かつ AI/市場比>1.10"},
    {"id": "B3", "label": "pred_ev>1.2 & edge_ratio>1.05",
     "desc": "AI確率EV>1.2 かつ AI/市場比>1.05"},
    {"id": "B4", "label": "pred_ev>1.2 & edge_ratio>1.10",
     "desc": "AI確率EV>1.2 かつ AI/市場比>1.10"},
    {"id": "B5", "label": "odds 3-20 & pred_ev>1.1",
     "desc": "中オッズ帯でAI確率EV>1.1"},
    {"id": "B6", "label": "odds 3-20 & pred_ev>1.2",
     "desc": "中オッズ帯でAI確率EV>1.2"},
    {"id": "B7", "label": "odds 3-20 & edge_ratio>1.10",
     "desc": "中オッズ帯でAI/市場比>1.10"},
]

# ── メタモデル目的変数 (Analysis C) ─────────────────────────────
META_TARGETS: list[dict[str, Any]] = [
    {
        "id": "T1", "name": "realized_profit",
        "formula": "win × odds - 1",
        "lightgbm_task": "回帰 (regression)",
        "noise_level": "極高",
        "pros": [
            "利益最大化に直接結びつく",
            "高オッズ的中を高く評価",
        ],
        "cons": [
            "ノイズが極めて高い (1着以外は全て負の値)",
            "高オッズ1勝で結果が大きく歪む",
            "クラス不均衡大 (1着率 ~8%)",
        ],
        "oof_metrics": ["RMSE", "MAE", "Spearman ρ vs realized", "上位10% mean profit"],
        "note": "単勝では profitable_label とほぼ同義になりやすい",
    },
    {
        "id": "T2", "name": "realized_return",
        "formula": "win × odds (1着のみ正、他は0)",
        "lightgbm_task": "回帰 (regression)",
        "noise_level": "極高",
        "pros": [
            "払戻額を直接予測",
            "高オッズ的中を高く評価",
        ],
        "cons": [
            "92%のサンプルが0",
            "1着の有無で出力が劇的に変わる",
            "目的変数の分散がオッズ依存",
        ],
        "oof_metrics": ["RMSE", "MAE", "上位10% mean return", "logloss (binarized)"],
        "note": "現行 e_return_win_pred と同じ目的変数",
    },
    {
        "id": "T3", "name": "profitable_label",
        "formula": "win × odds - 1 > 0 (1着かつオッズ>1)",
        "lightgbm_task": "二値分類 (binary)",
        "noise_level": "高",
        "pros": [
            "シンプルな二値分類",
            "評価指標が直感的 (AUC, F1)",
        ],
        "cons": [
            "単勝ではほぼ「1着になったか」と同じ",
            "オッズ情報が目的変数に含まれない (全1着が正例)",
            "クラス不均衡 (~8%正例)",
        ],
        "oof_metrics": ["AUC", "F1", "logloss", "上位10% hit rate"],
        "note": "高オッズ馬の1勝のノイズが大きい",
    },
    {
        "id": "T4", "name": "market_excess_diff",
        "formula": "win - p_market (実績 - 市場確率)",
        "lightgbm_task": "回帰 (regression)",
        "noise_level": "中",
        "pros": [
            "市場を上回る度合いを直接学習",
            "0付近に集中するためノイズが相対的に低い",
            "オッズに依存しない",
        ],
        "cons": [
            "1着以外は -p_market (負値一辺倒)",
            "市場確率の精度に依存",
            "解釈が直感的でない",
        ],
        "oof_metrics": ["RMSE", "MAE", "Spearman ρ", "上位10% mean excess"],
        "note": "AI edge を直接表現できる有力候補",
    },
    {
        "id": "T5", "name": "market_excess_ratio",
        "formula": "win / p_market (実績 / 市場確率)",
        "lightgbm_task": "回帰 (regression)",
        "noise_level": "中",
        "pros": [
            "市場に対する相対的な上振れを学習",
            "p_marketが小さい馬の的中を高く評価",
            "オッズ帯に依存しない相対評価",
        ],
        "cons": [
            "1着以外は全て0/p_market=0",
            "低人気馬の的中で極端な外れ値",
            "p_market=0での除算に注意",
        ],
        "oof_metrics": ["RMSE", "MAE (log scale)", "Spearman ρ", "上位10% mean ratio"],
        "note": "T4の比率版。極端値に注意",
    },
    {
        "id": "T6", "name": "candidate_roi_label",
        "formula": "一次条件(EV>閾値等)を満たした候補が利益になったか",
        "lightgbm_task": "二値分類 (binary)",
        "noise_level": "高",
        "pros": [
            "実運用に最も近い評価",
            "フィルタの有効性を直接判定",
            "候補に絞ることでサンプルを減らし学習を安定化",
        ],
        "cons": [
            "一次条件の設計に依存 (条件次第で結果が変わる)",
            "条件を満たすサンプルが少ない場合は学習困難",
            "循環参照のリスク (条件を学習結果で決める)",
        ],
        "oof_metrics": ["AUC", "Precision@K", "NDCG", "候補ROI"],
        "note": "メタフィルタとしての位置づけ。一次条件は固定して評価",
    },
]

# ── アブレーション案 (Analysis D) ───────────────────────────────
ABLATION_PLANS: list[dict[str, Any]] = [
    {
        "id": "A", "name": "現行パイプライン (ベースライン)",
        "ev_formula": "p_win_final (MAWC後) × tanodds → edge_win",
        "changes": "変更なし。既存データで基准値を測定。",
        "affected_files": [],
        "estimated_time": "0分 (既存データ利用)",
        "needs_oof_or_shadow": "不要",
        "rollback": "N/A",
    },
    {
        "id": "B", "name": "Return Modelなし: EV = p_win_pred × tanodds",
        "ev_formula": "p_win_pred × tanodds",
        "changes": (
            "BacktestEngine / RacePredictor で e_return_win_pred をスキップ。"
            "ev_win の代わりに p_win_pred × tanodds を使用。"
            "※ 疑似テストは今回のAnalysis Aで実施済み。"
            "本当に外すなら WinTwoStageModel の return model 出力を使わないようにする。"
        ),
        "affected_files": [
            "src/backtest/race_predictor.py (predict内EV計算)",
            "src/backtest/engine.py (bet記録列)",
            "src/betting/win_strategy.py (legacy, ev列参照)",
        ],
        "estimated_time": "1-2時間 (コード変更 + 軽量BT)",
        "needs_oof_or_shadow": "軽量OOF可 (既存モデルでpred × oddsに切り替えのみ)",
        "rollback": "ev_win 列を再使用するだけで復旧",
    },
    {
        "id": "C", "name": "Returnなし + 確率校正",
        "ev_formula": "p_win_corrected × tanodds",
        "changes": (
            "Return Modelをスキップ + EVCorrection P-correctionのみ適用。"
            "E-correction, Isotonic, MAWC はスキップまたはオプション化。"
        ),
        "affected_files": [
            "src/backtest/race_predictor.py (E-correction skip)",
            "src/models/ev_correction_model.py (P-only mode)",
            "src/models/market_aware_win_calibrator.py (skip option)",
        ],
        "estimated_time": "2-3時間",
        "needs_oof_or_shadow": "軽量OOF可",
        "rollback": "全補正を有効化して復旧",
    },
    {
        "id": "D", "name": "P補正のみ (Return + E補正なし)",
        "ev_formula": "p_win_corrected × e_return_win_pred",
        "changes": (
            "E-correction をスキップしてP-correctionのみ。"
            "Return Model自体は使用。"
            "E-correctionがreturn推定を悪化させている可能性を検証。",
        ),
        "affected_files": [
            "src/models/ev_correction_model.py (E-skip option)",
            "src/backtest/race_predictor.py",
        ],
        "estimated_time": "1-2時間",
        "needs_oof_or_shadow": "軽量OOF可",
        "rollback": "E-correctionを有効化して復旧",
    },
    {
        "id": "E", "name": "市場差分メタフィルタ追加",
        "ev_formula": "現行EV + メタフィルタ (edge_ratio等で候補を絞る)",
        "changes": (
            "新規メタモデルを追加。一次EV判定の後にフィルタを適用。"
            "特徴量: p_win_pred, p_market, edge_ratio, pred_ev, tanodds等。"
            "目的変数: market_excess_diff または candidate_roi_label。"
            "LightGBM二値分類 または Ridge回帰。",
        ),
        "affected_files": [
            "新規: src/models/market_meta_filter.py",
            "src/backtest/race_predictor.py (フィルタ統合)",
            "src/betting/win_strategy.py (フィルタ参照)",
            "src/pipelines/training_pipeline.py (メタモデル学習追加)",
        ],
        "estimated_time": "3-5時間 (学習 + Shadow比較)",
        "needs_oof_or_shadow": "Shadow比較必須 (新規モデル追加のため)",
        "rollback": "フィルタ無効化フラグで復旧",
    },
]


# ══════════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════════

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def roi_summary(
    df: pd.DataFrame,
    odds_col: str = "tanodds",
    win_col: str = "is_win",
) -> dict[str, Any]:
    """ROI・的中率などの基本統計。各馬100円買いの仮想ROI。"""
    n = len(df)
    if n == 0:
        return {"count": 0, "hit_rate": 0.0, "avg_odds": 0.0,
                "roi": 0.0, "profit": 0.0}
    hits = df[win_col].sum()
    total_payout = float((df[odds_col] * df[win_col]).sum())
    roi = safe_div(total_payout, float(n))
    return {
        "count": int(n),
        "hit_rate": round(float(hits / n), 4),
        "avg_odds": round(float(df[odds_col].mean()), 2),
        "roi": round(roi, 4),
        "profit": round(total_payout - float(n), 1),
    }


# ══════════════════════════════════════════════════════════════════
# データ読み込み
# ══════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """bt_{year}_horse_features.parquet を読み込んで結合。"""
    need_cols = [
        "race_id", "umaban", "kakuteijyuni", "surface", "tanodds",
        "confirmed_odds", "popularity_rank",
        "p_win_pred", "p_win_corrected", "p_win_final", "p_market_win_norm",
        "p_ability_win",
        "e_return_win_pred", "e_return_win_corrected",
        "ev_win", "ev_win_corrected", "ev_win_calibrated",
        "field_size", "distance_bin", "track_condition_code", "grade_code",
    ]
    frames: list[pd.DataFrame] = []
    found_cols: set[str] = set()

    for year in [2024, 2025]:
        path = BACKTEST_DIR / f"bt_{year}_horse_features.parquet"
        if not path.exists():
            print(f"  [skip] {year}: {path} not found")
            continue
        available = pd.read_parquet(path).columns.tolist()
        cols = [c for c in need_cols if c in available]
        df = pd.read_parquet(path, columns=cols)
        df["year"] = year
        df["race_id"] = df["race_id"].astype(str)
        frames.append(df)
        found_cols.update(cols)
        print(f"  {year}: {len(df)} rows, {len(cols)} cols loaded")

    if not frames:
        raise FileNotFoundError("No backtest horse_features files found")

    df = pd.concat(frames, ignore_index=True)

    # 前処理
    df = df[df["tanodds"] > 0].copy()
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)

    # p_market フォールバック
    if "p_market_win_norm" not in df.columns or df["p_market_win_norm"].isna().all():
        p_raw = 1.0 / df["tanodds"]
        p_sum = p_raw.groupby(df["race_id"]).transform("sum")
        df["p_market_win_norm"] = (p_raw / p_sum).astype(float)

    # 無効行除外
    essential = ["p_win_pred", "tanodds", "kakuteijyuni", "surface"]
    df = df.dropna(subset=essential)

    # 列の存在確認を記録
    missing = set(need_cols) - found_cols
    if missing:
        print(f"  [warn] Missing columns: {sorted(missing)}")

    print(f"  Total valid: {len(df)} rows, {len(df.columns)} cols")
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """派生列の計算。"""
    # Simple EV バリアント
    df["simple_ev_pred"] = df["p_win_pred"] * df["tanodds"]
    df["simple_ev_corrected"] = df["p_win_corrected"] * df["tanodds"]
    if "p_win_final" in df.columns:
        df["simple_ev_final"] = df["p_win_final"] * df["tanodds"]
    df["simple_ev_market"] = df["p_market_win_norm"] * df["tanodds"]
    if "p_ability_win" in df.columns:
        df["simple_ev_ability"] = df["p_ability_win"] * df["tanodds"]

    # 市場差分特徴量 (p_ai = p_win_pred)
    df["p_market"] = df["p_market_win_norm"]
    df["edge_diff_pred"] = df["p_win_pred"] - df["p_market"]
    df["edge_ratio_pred"] = (df["p_win_pred"] / df["p_market"]).clip(0.01, 100.0)
    df["pred_ev"] = df["p_win_pred"] * df["tanodds"]
    df["market_ev"] = df["p_market"] * df["tanodds"]

    # p_ai = p_win_final 版
    if "p_win_final" in df.columns:
        df["edge_ratio_final"] = (df["p_win_final"] / df["p_market"]).clip(0.01, 100.0)
        df["pred_ev_final"] = df["p_win_final"] * df["tanodds"]

    # p_win_pred == 0 の件数を記録
    zero_count = int((df["p_win_pred"] == 0).sum())
    total = len(df)
    print(f"  p_win_pred == 0: {zero_count}/{total} ({zero_count/total*100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════
# Analysis A: Simple EV比較
# ══════════════════════════════════════════════════════════════════

def get_ev_series(df: pd.DataFrame, ev_def: dict[str, Any]) -> pd.Series | None:
    """EV定義から実際のSeriesを取得。列がなければNone。"""
    if ev_def["ev_type"] == "simple":
        col = ev_def["p_col"]
        if col not in df.columns:
            return None
        return df[col] * df["tanodds"]
    elif ev_def["ev_type"] == "column":
        col = ev_def.get("ev_col", "")
        if col not in df.columns:
            return None
        return df[col]
    return None


def evaluate_ev_topn(
    df: pd.DataFrame, ev: pd.Series, pcts: list[int],
) -> list[dict[str, Any]]:
    """EV上位N%のROI評価。"""
    results: list[dict[str, Any]] = []
    for pct in pcts:
        threshold = float(ev.quantile(1 - pct / 100))
        mask = ev >= threshold
        sub = df[mask]
        s = roi_summary(sub)
        s["label"] = f"EV上位{pct}%"
        s["ev_threshold"] = round(threshold, 4)
        results.append(s)
    return results


def evaluate_ev_threshold(
    df: pd.DataFrame, ev: pd.Series, thresholds: list[float],
) -> list[dict[str, Any]]:
    """EV閾値別ROI評価。"""
    results: list[dict[str, Any]] = []
    for t in thresholds:
        mask = ev > t
        sub = df[mask]
        s = roi_summary(sub)
        s["label"] = f"EV>{t}"
        results.append(s)
    return results


def evaluate_odds_band(
    df: pd.DataFrame, ev: pd.Series,
) -> list[dict[str, Any]]:
    """オッズ帯別ROI評価。"""
    results: list[dict[str, Any]] = []
    for lo, hi in ODDS_BANDS:
        if hi is None:
            mask = df["tanodds"] >= lo
            label = f"{lo:.0f}倍+"
        else:
            mask = (df["tanodds"] >= lo) & (df["tanodds"] < hi)
            label = f"{lo:.0f}-{hi:.0f}倍"
        sub = df[mask]
        s = roi_summary(sub)
        s["label"] = label
        if len(sub) > 0:
            s["mean_ev"] = round(float(ev[mask].mean()), 4)
        results.append(s)
    return results


def evaluate_surface_breakdown(
    df: pd.DataFrame, ev: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Surface別の上位5% ROI + EV>1.2 ROI。"""
    breakdown: dict[str, dict[str, Any]] = {}
    for surface in ["turf", "dirt"]:
        sub_df = df[df["surface"] == surface]
        if len(sub_df) < 100:
            continue
        sub_ev = ev[sub_df.index]
        top5_thr = float(sub_ev.quantile(0.95))
        s_top5 = roi_summary(sub_df[sub_ev >= top5_thr])

        gt12_mask = sub_ev > 1.2
        s_gt12 = roi_summary(sub_df[gt12_mask])

        breakdown[surface] = {
            "total_count": int(len(sub_df)),
            "top5_pct_roi": s_top5,
            "ev_gt_1_2_roi": s_gt12,
        }
    return breakdown


def evaluate_ev_definition(
    df: pd.DataFrame, ev_def: dict[str, Any],
) -> dict[str, Any]:
    """1つのEV定義について全評価を実行。"""
    ev = get_ev_series(df, ev_def)
    if ev is None:
        return {"id": ev_def["id"], "label": ev_def["label"], "skipped": True,
                "reason": "required column not found"}

    # 相関 (p_col と p_market)
    p_col = ev_def["p_col"]
    corr = None
    if p_col in df.columns and "p_market_win_norm" in df.columns:
        corr = round(float(df[p_col].corr(df["p_market_win_norm"])), 4)

    result: dict[str, Any] = {
        "id": ev_def["id"],
        "label": ev_def["label"],
        "desc": ev_def["desc"],
        "uses_return_model": ev_def["uses_return_model"],
        "correlation_with_market": corr,
        "ev_stats": {
            "mean": round(float(ev.mean()), 4),
            "median": round(float(ev.median()), 4),
            "std": round(float(ev.std()), 4),
            "min": round(float(ev.min()), 4),
            "max": round(float(ev.max()), 4),
        },
        "ev_top_pct_roi": evaluate_ev_topn(df, ev, EV_TOPN_PCTS),
        "ev_threshold_roi": evaluate_ev_threshold(df, ev, EV_THRESHOLDS),
        "odds_band_roi": evaluate_odds_band(df, ev),
        "surface_breakdown": evaluate_surface_breakdown(df, ev),
    }
    return result


def run_analysis_a(df: pd.DataFrame) -> dict[str, Any]:
    """Analysis A: Simple EV比較。"""
    print("\n── Analysis A: Simple EV比較 ──")
    results: list[dict[str, Any]] = []
    for ev_def in EV_DEFINITIONS:
        r = evaluate_ev_definition(df, ev_def)
        status = "OK" if not r.get("skipped") else f"SKIP: {r.get('reason')}"
        print(f"  {ev_def['id']} {ev_def['label']}: {status}")
        results.append(r)

    # 比較サマリ: EV上位5% ROI を横断比較
    summary_rows: list[dict[str, Any]] = []
    for r in results:
        if r.get("skipped"):
            continue
        top5 = [x for x in r.get("ev_top_pct_roi", []) if "5%" in x.get("label", "")]
        gt12 = [x for x in r.get("ev_threshold_roi", []) if "1.2" in x.get("label", "")]
        summary_rows.append({
            "id": r["id"],
            "label": r["label"],
            "uses_return_model": r["uses_return_model"],
            "corr_market": r.get("correlation_with_market"),
            "top5_roi": top5[0]["roi"] if top5 else None,
            "top5_count": top5[0]["count"] if top5 else None,
            "gt12_roi": gt12[0]["roi"] if gt12 else None,
            "gt12_count": gt12[0]["count"] if gt12 else None,
        })

    return {
        "description": "Simple EV比較 (8定義)",
        "limitation": (
            "今回のSimple EVは既存p_win_*列にtanoddsを掛けた疑似アブレーション。"
            "Returnモデルなし再学習ではない。本当に外すにはWinTwoStageModel再学習が必要。"
        ),
        "ev_definitions": results,
        "comparison_summary": summary_rows,
    }


# ══════════════════════════════════════════════════════════════════
# Analysis B: 市場差分AND条件
# ══════════════════════════════════════════════════════════════════

def evaluate_condition(
    df: pd.DataFrame,
    condition: dict[str, Any],
    p_ai_col: str = "p_win_pred",
    edge_ratio_col: str = "edge_ratio_pred",
    pred_ev_col: str = "pred_ev",
) -> dict[str, Any]:
    """1つのAND条件を評価。"""
    cond_id = condition["id"]
    mask = pd.Series(True, index=df.index)

    label = condition["label"]

    # 条件分解
    if "pred_ev>1.1" in label and p_ai_col in {"p_win_pred", "p_win_final"}:
        ev_col = pred_ev_col
        mask = mask & (df[ev_col] > 1.1)
    elif "pred_ev>1.2" in label:
        ev_col = pred_ev_col
        mask = mask & (df[ev_col] > 1.2)

    if "edge_ratio>1.05" in label:
        mask = mask & (df[edge_ratio_col] > 1.05)
    elif "edge_ratio>1.10" in label:
        mask = mask & (df[edge_ratio_col] > 1.10)

    if "odds 3-20" in label:
        mask = mask & (df["tanodds"] >= 3) & (df["tanodds"] <= 20)

    sub = df[mask]
    overall = roi_summary(sub)

    # Surface別
    surface_breakdown: dict[str, dict[str, Any]] = {}
    for surface in ["turf", "dirt"]:
        ssub = sub[sub["surface"] == surface]
        if len(ssub) > 0:
            surface_breakdown[surface] = roi_summary(ssub)

    return {
        "id": cond_id,
        "label": label,
        "desc": condition["desc"],
        "p_ai_source": p_ai_col,
        "overall": overall,
        "surface_breakdown": surface_breakdown,
    }


def run_analysis_b(df: pd.DataFrame) -> dict[str, Any]:
    """Analysis B: 市場差分AND条件。"""
    print("\n── Analysis B: 市場差分AND条件 ──")

    # パターン1: p_ai = p_win_pred
    results_pred: list[dict[str, Any]] = []
    for cond in CONDITION_FILTERS:
        r = evaluate_condition(df, cond, p_ai_col="p_win_pred",
                               edge_ratio_col="edge_ratio_pred",
                               pred_ev_col="pred_ev")
        print(f"  {cond['id']} (pred): n={r['overall']['count']}, "
              f"ROI={r['overall']['roi']:.2%}")
        results_pred.append(r)

    # パターン2: p_ai = p_win_final (列があれば)
    results_final: list[dict[str, Any]] = []
    if "p_win_final" in df.columns and "edge_ratio_final" in df.columns:
        for cond in CONDITION_FILTERS:
            r = evaluate_condition(df, cond, p_ai_col="p_win_final",
                                   edge_ratio_col="edge_ratio_final",
                                   pred_ev_col="pred_ev_final")
            print(f"  {cond['id']} (final): n={r['overall']['count']}, "
                  f"ROI={r['overall']['roi']:.2%}")
            results_final.append(r)

    return {
        "description": "市場差分特徴量 AND条件評価",
        "p_ai_pred_conditions": results_pred,
        "p_ai_final_conditions": results_final,
    }


# ══════════════════════════════════════════════════════════════════
# Analysis C: メタモデル目的変数設計 (静的)
# ══════════════════════════════════════════════════════════════════

def run_analysis_c() -> dict[str, Any]:
    """Analysis C: メタモデル目的変数設計。データ不要。"""
    print("\n── Analysis C: メタモデル目的変数設計 ──")
    print(f"  {len(META_TARGETS)} 候補")
    return {
        "description": "メタモデル目的変数候補の構造化比較",
        "targets": META_TARGETS,
    }


# ══════════════════════════════════════════════════════════════════
# Analysis D: 最小アブレーション案 (静的)
# ══════════════════════════════════════════════════════════════════

def run_analysis_d() -> dict[str, Any]:
    """Analysis D: 最小アブレーション案。データ不要。"""
    print("\n── Analysis D: 最小アブレーション案 ──")
    print(f"  {len(ABLATION_PLANS)} 案")
    return {
        "description": "最小アブレーション実験計画",
        "plans": ABLATION_PLANS,
    }


# ══════════════════════════════════════════════════════════════════
# 目的変数確認 (コード上で確認した事実)
# ══════════════════════════════════════════════════════════════════

TARGET_VARIABLE_FACTS: list[dict[str, str]] = [
    {
        "model": "Stage1 AbilityModel",
        "target": "graded relevance (1着=3, 2着=2, 3着=1, 4着+=0)",
        "objective": "lambdarank (learning-to-rank)",
        "note": "二値分類ではない。softmax→p_ability_win",
    },
    {
        "model": "WinTwoStageModel (Hit)",
        "target": "kakuteijyuni == 1 (二値)",
        "objective": "binary",
        "note": "全サンプル対象。p_win_predを出力。",
    },
    {
        "model": "WinTwoStageModel (Return)",
        "target": "log1p(confirmed_odds) — 1着馬のみ",
        "objective": "regression_l1 (MAE)",
        "note": "重み=1/sqrt(p_win_pred)。e_return_win_predを出力。",
    },
    {
        "model": "EVCorrection P",
        "target": "kakuteijyuni == 1 (二値)",
        "objective": "binary",
        "note": "init_score=logit(p_win_pred)。p_win_correctedを出力。",
    },
    {
        "model": "EVCorrection E",
        "target": "log(confirmed_odds) - log(e_return_win_pred) — 1着馬のみ",
        "objective": "regression_l1 (MAE)",
        "note": "重み=1/sqrt(p_win_pred)。e_return_win_correctedを出力。",
    },
    {
        "model": "MarketModel",
        "target": "p_market_win_adj (回帰)",
        "objective": "regression_l1 (MAE)",
        "note": "log_errorのみ下流に渡す (signed/abs)。p_market_predは破棄。",
    },
    {
        "model": "TargetEncoder",
        "target": "kakuteijyuni == 1 (二値)",
        "objective": "smoothing平均 (smoothing=10)",
        "note": "te_blood_keito_cd, te_kisyucode, te_chokyosicode。",
    },
    {
        "model": "MAWC (MarketAwareWinCalibrator)",
        "target": "kakuteijyuni == 1 (二値)",
        "objective": "LogisticRegression (L2, C-selection)",
        "note": "51特徴量 (6主効果+15セグメント+30交差)。p_win_finalを出力。",
    },
]


# ══════════════════════════════════════════════════════════════════
# Markdown生成
# ══════════════════════════════════════════════════════════════════

def generate_markdown(
    results: dict[str, Any],
    df: pd.DataFrame,
) -> str:
    """分析結果をMarkdownレポートに変換。"""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def w(text: str = "") -> None:
        lines.append(text)

    # ── ヘッダー ──
    w("# 単勝 Simple EV / 市場差分メタモデル 設計診断レポート")
    w(f"生成日時: {now}")
    w(f"分析対象: bt_2024 + bt_2025 ({len(df):,} 件)")
    w(f"ROI定義: {ROI_NOTE}")
    w()

    # ── 1. 結論サマリ ──
    w("## 1. 結論サマリ")
    w()

    # Analysis A の比較サマリから最善手を判定
    comp = results["analysis_a"]["comparison_summary"]
    best_top5 = max(
        [c for c in comp if c.get("top5_roi") is not None],
        key=lambda c: c["top5_roi"] or 0,
        default=None,
    )
    best_gt12 = max(
        [c for c in comp if c.get("gt12_count") and c["gt12_count"] > 10],
        key=lambda c: c.get("gt12_roi") or 0,
        default=None,
    )

    # Analysis B の最適条件
    b_pred = results["analysis_b"]["p_ai_pred_conditions"]
    best_cond = max(
        [c for c in b_pred if c["overall"]["count"] > 50],
        key=lambda c: c["overall"]["roi"],
        default=None,
    )

    w("**結論:**")
    if best_top5:
        w(f"- **EV上位5% ROI最良**: {best_top5['id']} ({best_top5['label']}) "
          f"= ROI {best_top5['top5_roi']:.2%} (n={best_top5['top5_count']})")
    if best_gt12:
        w(f"- **EV>1.2 ROI最良**: {best_gt12['id']} ({best_gt12['label']}) "
          f"= ROI {best_gt12['gt12_roi']:.2%} (n={best_gt12['gt12_count']})")
    if best_cond:
        w(f"- **AND条件最良**: {best_cond['label']} "
          f"= ROI {best_cond['overall']['roi']:.2%} (n={best_cond['overall']['count']})")
    w()
    w("> **重要**: Simple EVは既存p_win_*列にtanoddsを掛けた疑似アブレーション。")
    w("> Returnモデルなしで再学習したわけではない。")
    w()

    # ── 初心者向け説明 ──
    w("### 初心者向けキーワード解説")
    w()
    w("- **目的変数**: モデルに「これが正解」として学ばせる値のこと。")
    w("  例: 「1着になったかどうか」(二値)、「オッズの対数」(回帰)")
    w("- **市場差分**: AIが推定した勝率と、オッズが示す市場の勝率のズレ。")
    w("  このズレが大きい馬ほど「AIは評価しているが市場は見ていない」候補。")
    w("- **メタモデル**: 勝率を予測する本体モデルではなく、")
    w("  「この候補は買ってよいか」を判定するフィルタ役のモデル。")
    w("- **Returnモデル**: 1着になった時の払戻額(オッズ)を予測する補助モデル。")
    w("  現行EVは「勝率 × 予想払戻額」で計算しているが、")
    w("  「勝率 × オッズ」のシンプル計算とどちらが良いかを比較する。")
    w()

    # ── 2. 目的変数一覧 ──
    w("## 2. コード上で確認した目的変数一覧")
    w()
    w("| モデル | 目的変数 | 学習対象 | 備考 |")
    w("|--------|---------|---------|------|")
    for t in TARGET_VARIABLE_FACTS:
        w(f"| {t['model']} | `{t['target']}` | {t['objective']} | {t['note']} |")
    w()

    # ── 3. Analysis A ──
    w("## 3. Analysis A: Simple EV比較")
    w()

    # 比較サマリ表
    w("### 3.1 比較サマリ")
    w()
    w("| ID | 定義 | Return使用 | 市場相関 | 上位5%ROI | 上位5%n | EV>1.2ROI | EV>1.2n |")
    w("|----|------|:---:|:---:|:---:|:---:|:---:|:---:|")
    for c in comp:
        ret_mark = "✓" if c["uses_return_model"] else "-"
        corr_str = f"{c['corr_market']:.3f}" if c.get("corr_market") else "-"
        t5r = f"{c['top5_roi']:.2%}" if c.get("top5_roi") else "-"
        t5n = f"{c['top5_count']}" if c.get("top5_count") else "-"
        g12r = f"{c['gt12_roi']:.2%}" if c.get("gt12_roi") else "-"
        g12n = f"{c['gt12_count']}" if c.get("gt12_count") else "-"
        w(f"| {c['id']} | {c['label']} | {ret_mark} | {corr_str} | "
          f"{t5r} | {t5n} | {g12r} | {g12n} |")
    w()

    # EV上位N% 詳細
    w("### 3.2 EV上位N% ROI 詳細")
    w()
    for r in results["analysis_a"]["ev_definitions"]:
        if r.get("skipped"):
            continue
        w(f"#### {r['id']}: {r['label']}")
        w(f"  市場相関: {r.get('correlation_with_market', '-')}")
        w(f"  EV統計: mean={r['ev_stats']['mean']:.4f}, "
          f"median={r['ev_stats']['median']:.4f}, "
          f"std={r['ev_stats']['std']:.4f}")
        w()
        w("  | 条件 | 件数 | 的中率 | 平均odds | ROI |")
        w("  |------|------|--------|----------|-----|")
        for item in r["ev_top_pct_roi"]:
            w(f"  | {item['label']} | {item['count']} | "
              f"{item['hit_rate']:.2%} | {item['avg_odds']:.1f} | "
              f"{item['roi']:.2%} |")
        for item in r["ev_threshold_roi"]:
            roi_str = f"{item['roi']:.2%}" if item["count"] > 0 else "条件なし"
            w(f"  | {item['label']} | {item['count']} | "
              f"{item['hit_rate']:.2%} | {item['avg_odds']:.1f} | "
              f"{roi_str} |")
        w()

    # オッズ帯別
    w("### 3.3 オッズ帯別 ROI (EV定義間比較)")
    w()
    w("| オッズ帯 | " + " | ".join(
        f"{c['id']}" for c in comp
    ) + " |")
    w("|----------|" + "|".join(["-----"] * len(comp)) + "|")
    for i, (lo, hi) in enumerate(ODDS_BANDS):
        label = f"{lo:.0f}-{hi:.0f}倍" if hi else f"{lo:.0f}倍+"
        vals: list[str] = []
        for r in results["analysis_a"]["ev_definitions"]:
            if r.get("skipped"):
                vals.append("-")
                continue
            band = r["odds_band_roi"][i] if i < len(r["odds_band_roi"]) else None
            if band and band["count"] > 0:
                vals.append(f"{band['roi']:.2%}(n={band['count']})")
            else:
                vals.append("-")
        w(f"| {label} | " + " | ".join(vals) + " |")
    w()

    # Surface別
    w("### 3.4 Surface別")
    w()
    for r in results["analysis_a"]["ev_definitions"]:
        if r.get("skipped"):
            continue
        sb = r.get("surface_breakdown", {})
        if not sb:
            continue
        w(f"#### {r['id']}: {r['label']}")
        w("| Surface | 総数 | 上位5%ROI | 上位5%n | EV>1.2ROI | EV>1.2n |")
        w("|---------|------|----------|---------|----------|---------|")
        for surf in ["turf", "dirt"]:
            if surf in sb:
                d = sb[surf]
                t5 = d["top5_pct_roi"]
                g12 = d["ev_gt_1_2_roi"]
                w(f"| {surf} | {d['total_count']} | "
                  f"{t5['roi']:.2%} | {t5['count']} | "
                  f"{g12['roi']:.2%} | {g12['count']} |")
        w()

    # ── 4. Analysis B ──
    w("## 4. Analysis B: 市場差分AND条件")
    w()

    w("### 4.1 p_ai = p_win_pred (補正前確率)")
    w()
    w("| 条件 | 件数 | 的中率 | 平均odds | ROI | Turf ROI | Dirt ROI |")
    w("|------|------|--------|----------|-----|----------|----------|")
    for c in results["analysis_b"]["p_ai_pred_conditions"]:
        o = c["overall"]
        turf_r = c["surface_breakdown"].get("turf", {}).get("roi", "-")
        dirt_r = c["surface_breakdown"].get("dirt", {}).get("roi", "-")
        turf_str = f"{turf_r:.2%}" if isinstance(turf_r, (int, float)) else turf_r
        dirt_str = f"{dirt_r:.2%}" if isinstance(dirt_r, (int, float)) else dirt_r
        w(f"| {c['label']} | {o['count']} | {o['hit_rate']:.2%} | "
          f"{o['avg_odds']:.1f} | {o['roi']:.2%} | {turf_str} | {dirt_str} |")
    w()

    if results["analysis_b"]["p_ai_final_conditions"]:
        w("### 4.2 p_ai = p_win_final (MAWC後確率)")
        w()
        w("| 条件 | 件数 | 的中率 | 平均odds | ROI | Turf ROI | Dirt ROI |")
        w("|------|------|--------|----------|-----|----------|----------|")
        for c in results["analysis_b"]["p_ai_final_conditions"]:
            o = c["overall"]
            turf_r = c["surface_breakdown"].get("turf", {}).get("roi", "-")
            dirt_r = c["surface_breakdown"].get("dirt", {}).get("roi", "-")
            turf_str = f"{turf_r:.2%}" if isinstance(turf_r, (int, float)) else turf_r
            dirt_str = f"{dirt_r:.2%}" if isinstance(dirt_r, (int, float)) else dirt_r
            w(f"| {c['label']} | {o['count']} | {o['hit_rate']:.2%} | "
              f"{o['avg_odds']:.1f} | {o['roi']:.2%} | {turf_str} | {dirt_str} |")
        w()

    # ── 5. Analysis C ──
    w("## 5. Analysis C: メタモデル目的変数候補")
    w()
    w("メタモデル = 勝率を予測する本体ではなく、")
    w("「買ってよい候補を残す/落とす」判定役のモデル。")
    w()

    for t in META_TARGETS:
        w(f"### {t['id']}: `{t['name']}`")
        w(f"- **数式**: `{t['formula']}`")
        w(f"- **LightGBM向き**: {t['lightgbm_task']}")
        w(f"- **ノイズレベル**: {t['noise_level']}")
        w(f"- **長所**:")
        for p in t["pros"]:
            w(f"  - {p}")
        w(f"- **短所**:")
        for c in t["cons"]:
            w(f"  - {c}")
        w(f"- **OOF評価指標**: {', '.join(t['oof_metrics'])}")
        w(f"- **注記**: {t['note']}")
        w()

    # ── 6. Analysis D ──
    w("## 6. Analysis D: 最小アブレーション案")
    w()
    w("| 案 | 内容 | 所要時間 | OOF/Shadow | 影響ファイル数 | 元に戻す方法 |")
    w("|----|------|---------|------------|--------------|-------------|")
    for p in ABLATION_PLANS:
        n_files = len(p["affected_files"])
        w(f"| {p['id']} | {p['name']} | {p['estimated_time']} | "
          f"{p['needs_oof_or_shadow']} | {n_files}ファイル | {p['rollback']} |")
    w()

    for p in ABLATION_PLANS:
        w(f"### 案{p['id']}: {p['name']}")
        w(f"- **EV計算**: `{p['ev_formula']}`")
        w(f"- **変更内容**: {p['changes']}")
        if p["affected_files"]:
            w(f"- **影響ファイル**:")
            for f in p["affected_files"]:
                w(f"  - `{f}`")
        w(f"- **所要時間**: {p['estimated_time']}")
        w(f"- **OOF/Shadow**: {p['needs_oof_or_shadow']}")
        w(f"- **元に戻す方法**: {p['rollback']}")
        w()

    # ── 7. 判断 ──
    w("## 7. すぐ実装すべきか、設計で止めるべきか")
    w()
    w("今回の分析結果に基づく推奨:")
    w()
    w("1. **まず疑似アブレーション結果を確認**: Analysis AでSimple EVが")
    w("   現行EVを上回るかどうかで方向性が変わる。")
    w("2. **Simple EVが優位なら**: 案B (Returnなし) を軽量OOFで検証。")
    w("   コード変更は小さく、リスクも低い。")
    w("3. **Simple EVでも改善しないなら**: 確率推定(p_win_pred)自体の")
    w("   市場相関が根本原因。案E (メタフィルタ) が有力。")
    w("4. **AND条件が有効なら**: pred_ev × edge_ratioの複合条件を")
    w("   既存パイプラインに追加するだけで改善する可能性がある。")
    w()
    w("**結論**: 設計で止めず、Analysis A/Bの結果を見て最小コストの")
    w("実装案を1つ選んで進めるべき。")
    w()

    # ── 8-10. メタ情報 ──
    w("## 8. 実行コマンド")
    w()
    w("```bash")
    w("python scripts/analyze_win_simple_ev_meta_design.py")
    w("```")
    w()

    w("## 9. 生成ファイル")
    w()
    w("- `data/analysis/win_simple_ev_meta_design.json`")
    w("- `data/analysis/win_simple_ev_meta_design.md`")
    w()

    w("## 10. 元に戻す方法")
    w()
    w("- 生成ファイルは新規ファイルなので削除だけで元に戻る。")
    w("- 既存モデル・データには一切影響なし。")
    w("- コミットしていない場合: ファイル削除 + `git switch main` で完全復旧。")
    w("- コミットした場合: `git revert` で復旧。")
    w()

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("単勝 Simple EV / 市場差分メタモデル 軽量分析")
    print("=" * 60)

    # データ読み込み
    print("\n[1/6] データ読み込み...")
    df = load_data()

    # 派生列計算
    print("\n[2/6] 派生列計算...")
    df = compute_derived(df)

    # Analysis A
    print("\n[3/6] Analysis A: Simple EV比較...")
    results_a = run_analysis_a(df)

    # Analysis B
    print("\n[4/6] Analysis B: 市場差分AND条件...")
    results_b = run_analysis_b(df)

    # Analysis C
    print("\n[5/6] Analysis C: メタモデル目的変数設計...")
    results_c = run_analysis_c()

    # Analysis D
    print("\n[6/6] Analysis D: 最小アブレーション案...")
    results_d = run_analysis_d()

    # 結果統合
    now = datetime.now().isoformat()
    zero_count = int((df["p_win_pred"] == 0).sum())
    total = len(df)

    results: dict[str, Any] = {
        "meta": {
            "generated_at": now,
            "total_rows": int(total),
            "years": sorted(df["year"].unique().tolist()),
            "surfaces": {s: int(n) for s, n in df["surface"].value_counts().items()},
            "p_win_pred_zero_count": zero_count,
            "p_win_pred_zero_pct": round(zero_count / total * 100, 1),
        },
        "target_variables": TARGET_VARIABLE_FACTS,
        "analysis_a": results_a,
        "analysis_b": results_b,
        "analysis_c": results_c,
        "analysis_d": results_d,
    }

    # 出力
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ANALYSIS_DIR / "win_simple_ev_meta_design.json"
    md_path = ANALYSIS_DIR / "win_simple_ev_meta_design.md"

    print(f"\nJSON出力: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"MD出力: {md_path}")
    md_content = generate_markdown(results, df)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 60)
    print("分析完了")
    print("=" * 60)

    # 簡易サマリを表示
    print("\n── 簡易サマリ ──")
    for c in results_a["comparison_summary"]:
        ret_mark = "Return" if c["uses_return_model"] else "Simple"
        corr_str = f"corr={c['corr_market']:.3f}" if c.get("corr_market") else ""
        print(f"  {c['id']} ({ret_mark:7s}): "
              f"top5% ROI={c.get('top5_roi', 0):.2%} (n={c.get('top5_count', 0):>5d})  "
              f"EV>1.2 ROI={c.get('gt12_roi', 0):.2%} (n={c.get('gt12_count', 0):>5d})  "
              f"{corr_str}")

    print()
    best_b = max(
        [c for c in results_b["p_ai_pred_conditions"] if c["overall"]["count"] > 50],
        key=lambda c: c["overall"]["roi"],
        default=None,
    )
    if best_b:
        print(f"  最良AND条件 (pred): {best_b['label']} → "
              f"ROI={best_b['overall']['roi']:.2%} (n={best_b['overall']['count']})")


if __name__ == "__main__":
    main()
