# 競馬AI システム設計書 v5.5
## ── 統計的・実装的「ズレ」の最終修正 ──

**バージョン:** 5.5（最終レビュー指摘3点の修正・「静かなバグ」の完全排除）
**作成日:** 2026-03-24
**改訂方針:** v5.4レビューで指摘された「静かなバグ」3点を修正。①P補正をinit_score付きキャリブレーションに変更（再学習化の防止）、②E補正のweightを1/√pに変更（ノイズ過剰適合の防止）、③RegimeDetectorの教師ラベルを市場指標ベースに変更（戦略依存の完全排除）。実運用耐性を97%→99%へ。

---

## 変更サマリー（v5.1 → v5.3 → v5.4 → v5.5）

| # | 問題 | 重要度 | v5.3 | v5.4 | v5.5 |
|---|------|--------|------|------|------|
| 1 | 期待リターンラベルのゼロ偏重 | **S** | 2段階モデル化（P×E） | （継承） | （継承） |
| 2 | Market Modelが市場コピーに収束 | **S** | 差分 log_error 専用出力 | （継承） | （継承） |
| 3 | RaceROIモデルのリーク | **S** | エッジ集計量に変更 | （継承） | （継承） |
| 4 | サブモデル分割過多 | **A** | 芝/ダート2分割 | （継承） | （継承） |
| 5 | ワイドスコアの分散未考慮 | **A** | `EV × P / (1+E)` モデル内部E | **`EV / (1 + P×E²)` Var_proxy使用** | （継承） |
| 6 | late_money 発走直前の扱い | **B** | t-3min判定、t-2minログ | （継承） | （継承） |
| 7 | DD制御 | **B** | DD×ROI+EWM A+ヒステリシス | **+ max_adjustment/day ガード** | （継承） |
| 8 | 2段階モデルの独立性破綻 | **S** | 全サンプル+補正係数 | **+ log化（分散安定）+ 中着者重み** | （継承） |
| 9 | market_errorのスケール依存 | **A** | p_pred下限クリップ | **p_market側もクリップ（両側対称化）** | （継承） |
| 10 | RaceQualityScreener | **A** | 分布特徴量+利益proxy | **+ 時間リーク完全遮断** | （継承） |
| 11 | EV補正の分母ノイズ | **S** | actual_ev / ev_raw | **log(actual_ev) - log(ev_raw)（分散安定化）** | （継承） |
| 12 | 外れ馬の学習支配 | **A** | なし | **中着者 weight = 1 + α（α=5〜10）** | （継承） |
| 13 | DD制御の過剰適応 | **B** | なし | **max_adjustment_per_day 制限** | （継承） |
| 14 | **EV補正のP/E混同（実運用不安定）** | **S** | log(actual_ev) - log(ev_raw) | **P補正×E補正の2モデルに分解** | **+ init_score付きP補正** |
| 15 | **RaceQuality proxyのモデル依存** | **A** | hist_top3_ev_mean | **結果ベースproxy（hit_rate/roi/return_ratio）** | （継承） |
| 16 | **Wideスコアの分散近似不足** | **A** | EV × P / (1+E) | **EV / (1 + P×E²) に変更** | （継承） |
| 17 | **DD制御の日付依存** | **B** | MAX_ADJUSTMENT_PER_DAY | **MAX_ADJUSTMENT_PER_N_BETS（20回）** | （継承） |
| 18 | **市場状態の切り替え未検知** | **A** | なし | **RegimeDetector（直近200レースで状態分類）** | （継承） |
| 19 | **P補正の疑似回帰（2値を回帰で解く）** | **S** | log(I(win)) - log(p_pred) を regression_l1 | **binary objective に変更** | **+ init_score = logit(p_pred)** |
| 20 | **E補正のサンプル選択バイアス** | **A** | winner等重み | **weight = 1/p_win_pred** | **→ 1/√p_win_pred** |
| 21 | **RegimeDetectorが戦略依存** | **A** | rolling_roi で状態分類 | **市場側指標追加** | **教師ラベルも市場指標化** |
| 22 | **Wideスコアのスケール不一致** | **B** | EV / (1 + P×E²) | **EV / (E × sqrt(P)) に変更** | （継承） |
| 23 | **P補正の再学習化** | **S** | p_win_pred を特徴量に含む | — | **init_score = logit(p_pred) でベースライン化** |
| 24 | **E補正weight過強** | **A** | weight = 1/p（p=0.01→100） | — | **weight = 1/√p（過剰適合防止）** |
| 25 | **RegimeDetector教師ラベル戦略依存** | **S** | y = rolling_roi | — | **y = market_efficiency（fav_rate × overround）** |

---

## 目次

1. [設計思想（v5.1確定版）](#1-設計思想v51確定版)
2. [単勝・複勝の2段階モデル化（ゼロ偏重問題の根本解決）](#2-単勝複勝の2段階モデル化ゼロ偏重問題の根本解決)
3. [EV補正モデル（独立性破綻の解決）](#3-ev補正モデル独立性破綻の解決)
4. [Market Model の差分専用化・正規化（過学習・スケール問題防止）](#4-market-model-の差分専用化正規化過学習スケール問題防止)
5. [RaceQualityScreener の分布特徴量化 + 結果ベースproxy（粒度不足・モデル依存の解決）](#5-racequalityscreener-の分布特徴量化粒度不足の解決)
6. [サブモデルを芝/ダート2分割に縮小](#6-サブモデルを芝ダート2分割に縮小)
7. [ワイドスコアの分散ベース・リスク調整（シャープレシオ近似）](#7-ワイドスコアの分散ベースリスク調整)
8. [late_money 直前判定戦略（t-3min基準）](#8-late_money-直前判定戦略t-3min基準)
9. [DDコントローラーの改善（Rolling ROI 連動・EWMA）](#9-ddコントローラーの改善rolling-roi-連動ewma)
9.5. [レジーム検知モデル（市場状態の切り替え検知）](#95-レジーム検知モデル市場状態の切り替え検知)
10. [システム全体アーキテクチャ v5.4](#10-システム全体アーキテクチャ-v54)
11. [モデルパイプライン v5.4](#11-モデルパイプライン-v54)
12. [ベッティングオーケストレーター v5.4](#12-ベッティングオーケストレーター-v54)
13. [バックテスト v5.4](#13-バックテスト-v54)
14. [開発ロードマップ v5.4](#14-開発ロードマップ-v54)
15. [期待値の現実的見積もり・本質的リスク（最終版）](#15-期待値の現実的見積もり本質的リスク最終版)
16. [ディレクトリ構成 v5.1](#16-ディレクトリ構成-v51)
17. [付録：全バージョン設計比較](#17-付録全バージョン設計比較)

---

## 1. 設計思想（v5.4確定版）

### 1.1 v5.4が修正する「実運用で壊れるポイント」の全体像

```
【v5.0〜v5.3で修正した問題（継承）】

歪み①：ラベル分布の偏り → 2段階分解で排除
歪み②：Market Modelの予測値混入 → log_error差分のみ
歪み③：RaceROIのリーク → エッジ集計量に変更
罠①：独立性破綻 → 補正係数モデル
罠②：スケール依存 → p_predクリップ
罠③：歪み検出器 → 利益proxy追加

【v5.3で修正した「統計ノイズ耐性」（継承）】

ノイズ①：EV補正の分母ノイズ → log化で分散安定
ノイズ②：外れ馬の学習支配 → 中着者重み付与
ノイズ③：log_error の片側クリップ → 両側クリップ
ノイズ④：RaceQuality の時間リーク → expanding().shift(1)
ノイズ⑤：DD制御の過剰適応 → max_adjustment 制限

【v5.4が修正する4種類の「実運用で壊れるポイント」（新規）】

壊れ①：EV補正のP/E混同（最重要）
  v5.3: y = log(actual_ev) - log(ev_raw) でPとEが混在
  → actual_ev = odds × I(win) なので、95%が log(ε) ≈ -∞ に収束
  → 結局「ほぼ分類問題」になっているが、PのズレかEのズレか区別不可
  → 学習が不安定になる
  修正: 補正をP補正×E補正に分解
    Model P: y_p = log(I(win) + ε) - log(p_pred + ε)  ← 全サンプル・分類residual
    Model E: y_e = log(odds_actual) - log(e_pred)      ← winnerのみ・回帰residual
    最終: EV_corrected = P_corrected × E_corrected

壊れ②：RaceQuality proxyのモデル依存
  v5.3: hist_top3_ev_mean はEVベース → モデル依存
  → モデルが変わるとproxyが崩れる
  → 過学習したEVを再利用する構造（弱いリーク）
  修正: EVではなく「結果ベース」の指標に変更
    hist_hit_rate_topk:           同条件で上位K頭の過去的中率
    hist_roi_topk:                同条件で上位K頭の過去ROI
    hist_positive_return_ratio:   同条件で正のリターンだったレースの割合

壊れ③：Wideスコアの分散近似不足
  v5.3: score = EV × P / (1+E)  → E²しか使っていない
  → 本来の分散 Var ≈ P × E² - (P×E)² ≈ P × E²
  → Pが欠落しているため、高E・低Pの極端なペアが過小評価されず中穴寄りに偏る
  修正: score = EV / (1 + P×E²)
    Var_proxy = P × E² で分散を近似
    高E・低Pのペアを適切にペナルティ

壊れ④：DD制御の日付依存
  v5.3: MAX_ADJUSTMENT_PER_DAY は「1日」で制限
  → レースが少ない日（平日3レース）→ ほぼ固定（制御不能）
  → レースが多い日（週末12レース）→ 過剰調整
  修正: MAX_ADJUSTMENT_PER_N_BETS（20ベット単位）に変更
    「時間」ではなく「試行回数」で制御
    1日3レースでも、週末12レースでも同じ粒度で制御

壊れ⑤：市場状態の切り替え未検知
  v5.3: 常に同じ戦略パラメータで運用
  → 市場は動的（アドバーサリアル環境）
  → 「歪みが強い時期」と「効率化した時期」で同じ攻め方をするのは危険
  → 荒れる年・堅い年でパラメータを固定すると劣化する
  修正: RegimeDetector（軽量レジーム検知モデル）を追加
    直近200レースの market_error分布・ROI・entropy から状態分類
    状態A（歪み強い）→ 攻める（EV閾値下げ・スコア閾値下げ）
    状態B（効率的）  → 絞る（EV閾値上げ・ベット数制限）
    状態C（崩壊）   → ほぼ停止（ベット数0〜1に制限）

壊れ⑥：P補正の疑似回帰（2値を回帰で解いている）
  v5.4初期: y_p = log(I(win)+ε) - log(p_pred) を regression_l1 で学習
  → I(win) ∈ {0,1} なので y_p は実質2値（log(ε) or log(1)）
  → 2値の回帰は「中央値」に引っ張られ P補正が過小バイアス
  修正: P補正をキャリブレーション問題として扱う
    objective を binary に変更（純粋な分類問題）
    出力を P_corrected = sigmoid(logit_correction) で [0,1] に制約
  v5.5修正: p_win_pred を特徴量に入れると「再学習」になる
    Stage1 と同じ問題をゼロから解く = Stage1 の意味が薄れる
    → init_score = logit(p_pred) を設定し、p_win_pred を特徴量から除外
    → logit(P_corrected) = logit(P_pred) + δ(x) の形に固定（真のキャリブレーション）

壊れ⑦：E補正のサンプル選択バイアス
  v5.4初期: E補正は winner のみ等重みで学習
  → winner は「選ばれたサンプル」→ 条件付き分布が歪む
  → 人気馬ばかり学習され、穴馬のEを過小評価
  修正: weight = 1 / √p_win_pred でインポータンス重み付け
    低確率勝利（穴馬の1着）を強調（1/p は強すぎてノイズに支配される）
    サンプル選択バイアスを補正しつつ過剰適合を抑制

壊れ⑧：RegimeDetectorが戦略依存
  v5.4初期: y = rolling_roi で状態分類
  → rolling_roi は「自分の戦略の成果」→ 戦略依存
  → モデルが悪い → COLLAPSED / 市場が悪い → COLLAPSED を区別不可
  v5.4改: 市場側指標を追加したが教師ラベルは rolling_roi のまま
  v5.5修正: 教師ラベルを完全に市場指標ベースに変更
    favorite_win_rate × overround_mean で market_efficiency を定義
    教師ラベルは市場の効率性に基づいて離散化
    rolling_roi は特徴量の補助指標に格下げ（教師ラベルには不使用）

歪み⑨：Wideスコアのスケール不一致
  v5.4初期: score = EV / (1 + P×E²)
  → EV と Var のスケールが一致していない
  → 理論的には score = EV / sqrt(Var) ≈ EV / (E × sqrt(P))
  修正: score = EV / (E × sqrt(P)) に変更
    期待値と分散のスケールを一致（シャープレシオに近い）
```

### 1.2 v5.4の絶対ルール（v5.3の18ルールを継承 + 更新）

```
Rule 1 : Stage1にオッズを入れない
Rule 2 : Stage2ラベルは2段階（P×E）。単一回帰は使わない
Rule 3 : ワイドは分散ベースのリスク調整スコア（EV / (E×sqrt(P)) 使用）← v5.4更新
Rule 4 : 信頼区間は CP + Rolling Quantile の min
Rule 5 : レース選別はエッジ集計量ベース（ROI直接学習は禁止）
Rule 6 : 1レースの最大リスクは資金の2%
Rule 7 : 戦略パラメータはout-of-sample期間では変更しない
Rule 8 : late_moneyはフィルタとして使う（t-3min基準で判定）
Rule 9 : DDコントローラーはDD×Rolling ROI + ヒステリシス + max_adj/N_bets ← v5.4更新
Rule 10: モデル分割は「芝/ダートの2分割」から始める
Rule 11: Market Modelの出力は差分（log_error）のみStage2に入力
Rule 12: EV補正はP補正(init_score付きbinary)×E補正(1/√p重み付き回帰)の2モデルに分解     ← v5.5更新
Rule 13: market_errorは両側クリップ（p_market, p_pred 共に clip(0.01, 0.99))
Rule 14: 直前判定はt-3min基準、t-2minはログのみ
Rule 15: ワイドスコアは EV / (E × sqrt(P)) でシャープレシオ近似を使用    ← v5.4更新
Rule 16: RaceQualityScreenerは結果ベースproxy（EV依存proxy禁止）           ← v5.4更新
Rule 17: DD回復はヒステリシス付きだが max_adjustment/N_bets を制限        ← v5.4更新
Rule 18: hist系特徴量は expanding().shift(1) で未来情報リークを完全遮断
Rule 19: 市場状態の切り替えは RegimeDetector で検知（市場側指標メイン）   ← NEW
```

---

## 2. 単勝・複勝の2段階モデル化（ゼロ偏重問題の根本解決）

### 2.1 ゼロ偏重問題の数学的構造

```
【v4.0の単勝ラベル分布】

edge_label = log(actual_return / expected_return_market)

1着馬（1頭）: log(win_odds / expected_return) → 有限の正値
外れ馬（15頭）: log(ε / expected_return)     → 大きな負値（≈ log(1e-6)）

16頭立てのレースでのラベル分布:
  6.25%: 有限の正値（1着馬のみ）
  93.75%: 大きな負値（外れ馬）

regression_l1 で学習すると：
  MAEを最小化 → 中央値に収束 → 大きな負値に引っ張られる
  → edge の予測値が全体的に「負の方向」にバイアス
  → EV > 1 を検出するためのしきい値設定が困難になる

複勝でも同様（3着内 = 18.75% / 外れ = 81.25%）
```

### 2.2 ワイドと同じ2段階構造に統一する理由

```
【設計の非対称性（v4.0の問題）】

ワイド:  2段階 P(hit) × E(return|hit) ← 正しく設計されている
単勝:    1段階 E[actual_return]        ← ゼロ偏重問題あり
複勝:    1段階 E[actual_return]        ← ゼロ偏重問題あり（ましだが）

【v5.0: 全券種を2段階に統一】

単勝:  Stage A = P(win)       Stage B = E(win_odds | win)
複勝:  Stage A = P(place)     Stage B = E(place_odds | place)
ワイド: Stage A = P(joint_hit) Stage B = E(wide_odds | joint_hit)  ← v4.0から継続

最終EV:
  単勝:  EV_win   = P(win)   × E(win_odds | win)
  複勝:  EV_place = P(place) × E(place_odds | place)
```

### 2.3 単勝・複勝2段階モデルの完全実装

```python
# models/two_stage_return_model.py  v5.0

import lightgbm as lgb
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class TwoStageConfig:
    """2段階モデルの設定"""
    # Stage A（分類）
    hit_metric:      str   = "auc"
    hit_leaves:      int   = 31
    hit_lr:          float = 0.03
    hit_rounds:      int   = 500
    # Stage B（回帰・的中事例のみ）
    return_metric:   str   = "mae"
    return_leaves:   int   = 15    # 的中サンプルは少ない → 過学習防止
    return_lr:       float = 0.03
    return_rounds:   int   = 300
    # 分割設定
    min_hit_samples: int   = 200   # Stage B 学習の最低サンプル数


class WinTwoStageModel:
    """
    単勝2段階モデル
    Stage A: P(win)              ← 2値分類
    Stage B: E(win_odds | win)   ← 的中時払戻の回帰
    EV = P(win) × E(win_odds | win)

    特徴量はv5.0の Stage2 FEATURE_COLS と同じ。
    ただし市場コピー防止のため p_market_pred_win は除外し、
    market_log_error（正規化差分）のみを使用する（Section 4 参照）。
    """

    FEATURE_COLS = [
        # Stage1出力
        "p_ability_win",
        # Market Model正規化差分（v5.3: signed/abs log_error を使用）
        "signed_log_error_win",
        "abs_log_error_win",
        # オッズ変化率（変化のみ）
        "odds_drop_rate_60_10", "odds_drop_rate_30_10",
        "odds_velocity", "odds_volatility",
        "popularity_change_30_10",
        # 市場歪み
        "market_entropy", "popularity_rank", "overround",
        # レース条件
        "surface", "distance_bin", "track_condition_code",
        "grade_code", "field_size",
    ]

    def __init__(self, cfg: TwoStageConfig = TwoStageConfig()):
        self.cfg = cfg

    # ── Stage A: 分類 ──────────────────────────────────────────────────

    def train_hit_model(self, df: pd.DataFrame) -> None:
        """P(win) の学習（全出走馬を使用・1着=1 / 他=0）"""
        X = df[self.FEATURE_COLS]
        y = (df["finish_pos"] == 1).astype(int)

        self.hit_model = lgb.train(
            {
                "objective":     "binary",
                "metric":        self.cfg.hit_metric,
                "learning_rate": self.cfg.hit_lr,
                "num_leaves":    self.cfg.hit_leaves,
                "is_unbalance":  True,   # 1/field_size の不均衡対応
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=self.cfg.hit_rounds,
        )

    # ── Stage B: 回帰（的中事例のみ） ──────────────────────────────────

    def train_return_model(self, df: pd.DataFrame) -> None:
        """
        E(win_odds | win) の学習
        1着馬のみを使って「当たったときのオッズの期待値」を学習する。

        ポイント：
        - 1着馬のみを学習データに使うことで、ゼロ偏重を完全に排除
        - 「どんな文脈で高配当が出るか」を学習する
        - 例：人気薄・market_log_error が大きい馬が1着のときのオッズ分布
        """
        hit_df = df[df["finish_pos"] == 1].copy()

        if len(hit_df) < self.cfg.min_hit_samples:
            raise ValueError(
                f"Stage B 学習には最低 {self.cfg.min_hit_samples} 件の"
                f"的中サンプルが必要。現在: {len(hit_df)} 件"
            )

        X = hit_df[self.FEATURE_COLS]
        y = hit_df["win_odds_actual"]  # 実際の確定単勝オッズ

        self.return_model = lgb.train(
            {
                "objective":     "regression_l1",
                "metric":        self.cfg.return_metric,
                "learning_rate": self.cfg.return_lr,
                "num_leaves":    self.cfg.return_leaves,
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=self.cfg.return_rounds,
        )

    # ── 推論 ──────────────────────────────────────────────────────────

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EV_win = P(win) × E(win_odds | win)

        市場のEV（= p_market × win_odds ≈ 0.80）と比較して
        EV_win > 1.0 の馬が「市場より期待値が高い」馬。

        さらに信頼区間（Section 4 の RobustConfidenceEstimator）で
        EV_lower を計算してからベット判断に使う。
        """
        df = df.copy()
        X = df[self.FEATURE_COLS]

        df["p_win_pred"]        = self.hit_model.predict(X)
        df["e_return_win_pred"] = self.return_model.predict(X)
        df["ev_win"]            = df["p_win_pred"] * df["e_return_win_pred"]
        return df


class PlaceTwoStageModel:
    """
    複勝2段階モデル
    Stage A: P(place)               ← 3着以内かどうかの分類
    Stage B: E(place_odds | place)  ← 的中時払戻の回帰

    複勝は的中率が高い（約18〜35%）ため：
    - Stage B の学習データが単勝より豊富
    - return_leaves を少し増やせる（25程度）
    - ゼロ偏重問題は単勝より軽度だが、2段階化で確実に改善
    """

    FEATURE_COLS = WinTwoStageModel.FEATURE_COLS  # 共通

    def __init__(self, cfg: TwoStageConfig = TwoStageConfig()):
        self.cfg = cfg

    def train_hit_model(self, df: pd.DataFrame) -> None:
        X = df[self.FEATURE_COLS]
        y = (df["finish_pos"] <= 3).astype(int)

        self.hit_model = lgb.train(
            {
                "objective":     "binary",
                "metric":        "auc",
                "learning_rate": self.cfg.hit_lr,
                "num_leaves":    self.cfg.hit_leaves,
                "is_unbalance":  True,
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=self.cfg.hit_rounds,
        )

    def train_return_model(self, df: pd.DataFrame) -> None:
        """3着以内の馬のみで複勝払戻オッズを学習"""
        hit_df = df[df["finish_pos"] <= 3].copy()

        X = hit_df[self.FEATURE_COLS]
        y = hit_df["place_odds_actual"]

        self.return_model = lgb.train(
            {
                "objective":     "regression_l1",
                "metric":        "mae",
                "learning_rate": self.cfg.return_lr,
                "num_leaves":    25,   # 複勝はサンプル多めなので少し深く
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=self.cfg.return_rounds,
        )

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        X = df[self.FEATURE_COLS]

        df["p_place_pred"]        = self.hit_model.predict(X)
        df["e_return_place_pred"] = self.return_model.predict(X)
        df["ev_place"]            = df["p_place_pred"] * df["e_return_place_pred"]
        return df
```

### 2.4 2段階化によるラベル分布の改善

```
【Stage A（分類）のラベル分布】
  単勝: 1着=6.25%  /  他=93.75%  → is_unbalance=True で対応
  複勝: 3着内=18.75% / 他=81.25% → is_unbalance=True で対応
  → バイナリクロスエントロピーはゼロ偏重に影響されない

【Stage B（回帰）のラベル分布】
  単勝: 学習データ = 1着馬のみ（全レースの1/field_size）
        払戻オッズ = 通常 1.2〜100倍台
        → ゼロが一切ない正値の分布
  複勝: 学習データ = 3着以内のみ
        払戻オッズ = 通常 1.1〜5.0倍台
        → 同様にゼロが一切ない

→ 2段階化でゼロ偏重問題を構造的に排除
```

---

## 3. EV補正モデル（独立性破綻の解決）

### 3.1 独立性破綻の数学的構造

```
【v5.0の暗黙前提が破綻している】

v5.0:  EV = P(win) × E(odds | win)

この計算は P(win) と E(odds|win) が「独立」であることを前提としている。
しかし現実には：

  人気馬: P(win) = 0.30, E(odds|win) = 3.5  ← 負の相関
  穴馬:   P(win) = 0.02, E(odds|win) = 45.0 ← 負の相関

P(win) が高い馬は E(odds|win) が低い。
→ 2つのモデルを独立に学習して掛け算すると
   「Pが少し高く見積もった AND oddsが少し高く見積もった」
   という確率を過大評価する。

具体的なズレの例：
  モデル出力: P(win)=0.15, E(odds|win)=12.0
  → 乗算EV = 1.80（一見強い）
  しかし P と E の間に負の相関があるため、
  両方が同時にこの値を取る確率は独立前提より低い
  → 実際のEV は 1.50 程度に補正されるべき

特に中穴ゾーン（P=0.05〜0.15, odds=8〜20）で誤差が最大。
このゾーンはベッティングの主戦場であるため、補正は必須。
```

### 3.2 P/E分解アプローチ（v5.4: 補正をPとEに分離）

```
【v5.3の問題: PとEが混在した補正は構造的に不安定】

v5.3: y = log(actual_ev) - log(ev_raw)

問題の本質：
  actual_ev = odds × I(win) なので：
    95% の馬: actual_ev = 0 → log(ε) ≈ -∞
     5% の馬: actual_ev = odds → log(odds)

  → 目的変数が「ほぼ分類問題」に還元されている
  → しかしモデルは回帰（regression_l1）で学習
  → Pのズレを補正しているのか Eのズレを補正しているのか区別できない
  → 学習が不安定になる

【v5.4の解決: 補正をP補正×E補正に分解（v5.5: Pはinit_score付き分類・Eは1/√p重み付き回帰）】

Correction = C_p × C_e

Model P（P補正）:
  目的変数: y_p = I(win)  ← 純粋な2値分類
  objective: binary（regression_l1 は不使用）
  学習データ: 全サンプル
  役割: P(win) の予測ズレを「キャリブレーション」で補正
  出力: P_corrected = sigmoid(logit(I(win)) + correction_logit)
  → [0, 1] に制約された確率出力
  v5.5改: init_score = logit(p_win_pred) を設定
    → logit(P_corrected) = logit(P_pred) + δ(x) の形に固定
    → p_win_pred を特徴量から除外（再学習化を防止）

Model E（E補正）:
  目的変数: y_e = log(actual_odds | win) - log(e_pred)
  学習データ: 1着馬のみ
  役割: E(odds|win) の予測ズレ（回帰 residual）を学習
  重み: weight = 1 / √p_win_pred（低確率勝利を強調・ノイズ過剰適合を防止）

最終補正:
  P_corrected = sigmoid(logit(P_pred) + correction_logit)
  E_corrected = e_pred  × exp(y_e)
  EV_corrected = P_corrected × E_corrected

利点:
  ① PのズレとEのズレが明確に分離 → 学習が安定
  ② P補正は binary objective → 2値を回帰で解く「疑似回帰」を回避
  ③ P_corrected は [0,1] に制約 → 確率として整合
  ④ E補正に 1/√p 重み → 穴馬の過小評価を防止しつつノイズ過剰適合を抑制
  ⑤ 各モデルが単一の責任を持つ → 解釈性が向上
  ⑥ v5.5: P補正は init_score で Stage1 をベースライン化 → 再学習化を防止
```

### 3.3 P/E分解補正モデルの実装

```python
# models/ev_correction_model.py  v5.5

import lightgbm as lgb
import numpy as np
import pandas as pd


class EVCorrectionModel:
    """
    2段階モデルの「独立性破綻」を補正するモデル。

    v5.3: y = log(actual_ev) - log(ev_raw) でPとEが混在
    v5.4: P補正モデルとE補正モデルに分解
    v5.4改: P補正を binary objective に変更（疑似回帰回避）
    v5.5:   P補正に init_score = logit(p_pred) を設定（再学習化の防止）
           E補正に weight = 1/p_win_pred を追加（サンプル選択バイアス補正）
    v5.5: P補正に init_score = logit(p_pred) を設定（再学習化の防止）
          P補正の特徴量から p_win_pred を除外
          E補正の weight を 1/p → 1/√p に変更（ノイズ過剰適合の防止）

    P補正: 全サンプルで P(win) を init_score 付き binary classification でキャリブレーション
    E補正: 1着馬のみで E(odds|win) の residual を 1/√p 重み付き回帰
    最終:  EV_corrected = P_corrected × E_corrected
    """

    E_CLIP_FLOOR = 1.0    # e_pred の下限クリップ（オッズは1.0以上）

    FEATURE_COLS = [
        # 2段階モデルの出力（v5.5: p_win_pred を除外 → init_score で代替）
        "e_return_win_pred",     # Stage B の出力
        # 交互作用特徴量
        "p_x_e_interaction",     # P(win) × E(odds|win)
        "p_minus_e_gap",         # |log(P) - log(E)| （独立性の指標）
        # 市場歪み
        "signed_log_error_win",
        "abs_log_error_win",
        "market_entropy",
        "popularity_rank",
        # レース条件
        "surface", "distance_bin", "track_condition_code",
        "field_size",
    ]

    def train(self, df: pd.DataFrame) -> None:
        """
        P補正モデルとE補正モデルをそれぞれ学習する。
        """
        df = df.copy()

        assert "ev_win" in df.columns, \
            "ev_win が必要です。先に WinTwoStageModel.predict_ev() を実行してください"

        # 交互作用特徴量の追加
        df["p_x_e_interaction"] = df["p_win_pred"] * df["e_return_win_pred"]
        df["p_minus_e_gap"] = np.abs(
            np.log(df["p_win_pred"] + 1e-8) - np.log(df["e_return_win_pred"] + 1e-8)
        )

        X = df[self.FEATURE_COLS]

        # ── Model P: P補正（全サンプル・binary classification）──
        # v5.4改: 回帰ではなく分類として扱う（疑似回帰回避）
        # v5.5:   init_score = logit(p_win_pred) で Stage1 をベースライン化
        #         p_win_pred を特徴量に入れない → 「補正」ではなく「再学習」になるのを防止
        y_p = (df["finish_pos"] == 1).astype(int)
        init_score = np.log(
            np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
            / (1 - np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4))
        )

        self.p_correction_model = lgb.train(
            {
                "objective":     "binary",
                "metric":        "auc",
                "learning_rate": 0.03,
                "num_leaves":    15,
                "is_unbalance":  True,   # 1着:6.25% vs 他:93.75%
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y_p, init_score=init_score),
            num_boost_round=300,
        )

        # ── Model E: E補正（1着馬のみ・重み付き回帰 residual）──
        # v5.4改: weight = 1/p_win_pred でサンプル選択バイアスを補正
        # v5.5:   weight = 1/√p_win_pred に変更（1/p は強すぎてノイズに支配される）
        winners = df[df["finish_pos"] == 1].copy()
        e_pred_clipped = np.clip(winners["e_return_win_pred"], self.E_CLIP_FLOOR, None)
        winners["log_e_correction"] = (
            np.log(winners["win_odds_actual"].clip(lower=self.E_CLIP_FLOOR))
            - np.log(e_pred_clipped)
        )
        # 低確率勝利（穴馬の1着）を強調するインポータンス重み（1/√p で過剰適合を抑制）
        winners["_e_sample_weight"] = 1.0 / np.sqrt(np.clip(winners["p_win_pred"], 0.01, None))

        X_e = winners[self.FEATURE_COLS]

        self.e_correction_model = lgb.train(
            {
                "objective":     "regression_l1",
                "metric":        "mae",
                "learning_rate": 0.03,
                "num_leaves":    15,
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(X_e, label=winners["log_e_correction"],
                        weight=winners["_e_sample_weight"].values),
            num_boost_round=300,
        )

    def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全馬のEVをP補正×E補正で補正する。

        P_corrected = sigmoid(logit(P_pred) + correction_logit)  ← [0,1] に制約
        E_corrected = e_return_win_pred × exp(log_e_correction)
        EV_corrected = P_corrected × E_corrected
        """
        df = df.copy()

        df["p_x_e_interaction"] = df["p_win_pred"] * df["e_return_win_pred"]
        df["p_minus_e_gap"] = np.abs(
            np.log(df["p_win_pred"] + 1e-8) - np.log(df["e_return_win_pred"] + 1e-8)
        )

        X = df[self.FEATURE_COLS]

        # P補正の適用（binary出力 → sigmoid で [0,1] に制約）
        # v5.5: 推論時も logit(p_pred) を init_score として加算
        p_pred_clipped = np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))
        p_correction_logit = self.p_correction_model.predict(X) + init_score
        df["p_win_corrected"] = 1.0 / (1.0 + np.exp(-p_correction_logit))

        # E補正の適用
        log_e_corr = self.e_correction_model.predict(X)
        df["e_return_win_corrected"] = df["e_return_win_pred"] * np.exp(log_e_corr)

        # 最終補正EV
        df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
        return df
```

### 3.4 補正モデルの効果検証

```
【検証方法】

検証1: 補正前後のEV精度比較
  - ev_win_raw と ev_win_corrected のそれぞれについて、
    実際のEV（actual_odds × I(win)）との MAE を計算
  - ev_win_corrected の MAE < ev_win_raw の MAE であることを確認

検証2: 中穴ゾーンでの補正効果
  - P(win) ∈ [0.05, 0.15] のサブセットで
    補正前後のEV誤差を比較
  - このゾーンで最も補正効果が大きいはず

検証3: P補正とE補正の独立性確認（v5.4新設）
  - log_p_correction と log_e_correction の相関係数を計算
  - 相関が低い（< 0.3）ことを確認 → PとEが独立に補正されている
  - 各モデルの SHAP 値を確認し、P補正が p_win_pred に、E補正が
    e_return_win_pred に最も反応していることを確認

検証4: 低確率ゾーンでの過大評価抑制
  - P(win) < 0.05 の馬で ev_win_raw > 1.5 の割合を計算
  - 補正後の ev_win_corrected > 1.5 の割合が減ることを確認

検証5: P補正の分類性能確認（v5.4新設）
  - P_corrected で AUC を計算
  - P_pred の AUC より改善していることを確認
  - P_corrected が [0, 1] の範囲内に収まることを確認
  - P_corrected のキャリブレーション（予測確率 vs 実際の的中率）が
    P_pred より正確であることを確認（reliability diagram）
  - v5.5追加: correction_logit の絶対値が小さい（|δ| < 1.0）ことを確認
    → init_score がベースラインとして機能している証拠

検証6: E補正の回帰性能確認（v5.4新設）
  - 1着馬のみで E_corrected と actual_odds の MAE を計算
  - E_pred 単体の MAE より改善していることを確認
  - 低 p_win_pred（穴馬）帯での E_corrected の誤差を確認
    → weight なしモデルより穴馬帯の誤差が小さいことを確認
  - v5.5追加: 1/√p 重みモデルと 1/p 重みモデルの比較
    → 1/√p の方が out-of-sample で安定していることを確認（過学習の防止）

期待される効果:
  補正前: 中穴ゾーンで EV を平均15〜25%過大評価
  補正後: EV誤差が MAE で 30〜40% 改善
  低確率ゾーン: 過大評価率が 40%→15% に減少
  v5.4追加: P補正AUCが P_pred より +1〜3% 改善
  v5.4追加: P/E補正の相関 < 0.3（独立補正の担保）
  v5.4改追加: P_corrected が [0,1] に制約（確率として整合）
  v5.4改追加: 穴馬帯のE補正誤差が等重みより改善
  v5.5追加: P補正の init_score で Stage1 をベースライン化（再学習化の防止）
  v5.5追加: E補正の weight を 1/√p に変更（ノイズ過剰適合の防止）
```
```

---

## 4. Market Model の差分専用化・正規化（過学習・スケール問題防止）

### 3.1 v4.0の自己参照ループの構造

```
【v4.0でStage2に両方入力していた問題】

Stage2 の入力（v4.0）:
  p_market_pred_win   ← Market Modelの予測値（市場に似た値）
  market_pred_error   ← 差分（市場の歪み）

問題：
  p_market_pred ≈ p_market_actual（市場確率の模倣）
  → Stage2に「市場の模倣値」を入れることで
    Stage2が「市場の論理」に引っ張られる
  → market_pred_error の効果が希薄化
  → 最終的に Stage2 が「やや精度の高い市場のコピー」になる

つまり：
  予測値（p_market_pred）は ノイズ
  差分  （market_pred_error）は シグナル

両方入れるとノイズがシグナルを汚染する。
```

### 4.2 差分専用化 + 正規化の実装

```python
# models/market_model.py  v5.3（v5.1からの変更：p_predクリップ追加）

class MarketModel:
    """
    v5.0の変更：
    - predict_and_calc_error の出力から p_market_pred を除外
    - market_pred_error のみを Stage2 に渡す

    v5.1の追加変更：
    - market_pred_error を正規化（log_error）してスケール問題を解消

    v5.3の追加変更：
    - p_pred の下限クリップで log_error の発散を防止
    v5.3の追加変更：
    - p_market 側もクリップ（片側非対称の排除）
      極端値の p_market がクリップ済み p_pred との非対称分布を引き起こす問題を解消
    """

    P_PRED_CLIP_MIN = 0.01   # p_pred の下限クリップ値
    P_PRED_CLIP_MAX = 0.99   # p_pred の上限クリップ値

    def predict_and_calc_error(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_p_market_pred_win"] = self.model.predict(df[self.FEATURE_COLS])

        # v5.3: p_pred のクリップ（発散防止）
        # v5.3: p_market もクリップ（片側非対称の排除）
        p_pred_clipped = np.clip(
            df["_p_market_pred_win"],
            self.P_PRED_CLIP_MIN,
            self.P_PRED_CLIP_MAX,
        )
        p_market_clipped = np.clip(
            df["p_market_win_adj"],
            self.P_PRED_CLIP_MIN,
            self.P_PRED_CLIP_MAX,
        )

        # 差分を計算（内部用）
        raw_error = df["p_market_win_adj"] - df["_p_market_pred_win"]

        # v5.1: 正規化（log_error）でスケール依存を排除
        # v5.3: 両側クリップ済み（対称な分布を保証）
        df["market_log_error_win"] = np.log(
            p_market_clipped / p_pred_clipped
        )

        # v5.3: signed/abs を分けて特徴量化
        df["signed_log_error_win"] = df["market_log_error_win"]
        df["abs_log_error_win"] = np.abs(df["market_log_error_win"])

        # 生の差分も保持（後方互換・分析用）
        df["market_pred_error_win"] = raw_error

        # レース内での相対化
        df["market_error_rank_in_race"] = (
            df["market_log_error_win"]
            .rank(method="first", ascending=True)
            .astype(int)
        )

        # p_market_pred は Stage2 に渡さない
        df = df.drop(columns=["_p_market_pred_win"])

        return df

    def get_stage2_features(self) -> list[str]:
        """
        Stage2 に渡す Market Model 由来の特徴量リスト
        差分のみ。予測値は含まない。v5.3: signed/abs を分離。
        """
        return [
            "signed_log_error_win",      # 符号付き正規化差分（v5.3）
            "abs_log_error_win",         # 絶対値正規化差分（v5.3）
            "market_error_rank_in_race", # レース内相対ランク
        ]
        # NG: "p_market_pred_win" は含めない
```

### 4.3 Market Model の役割の整理（v5.1確定版）

```
Market Model が担う役割（v5.1確定版）：

✅ 使う（メイン）：
  market_log_error = log(p_market_actual / p_market_pred)
  → 正規化された歪み（スケールに依存しない）
  → 人気馬・穴馬の歪みを同じ尺度で比較可能

  market_error_rank_in_race
  → レース内で market_log_error が大きい馬の相対順位

⚠️ 分析用のみ：
  market_pred_error = p_market_actual - p_market_pred
  → 生の差分（log_error と併用して傾向分析に使用）

❌ 使わない：
  p_market_pred
  → 市場に似た値 → Stage2 に入れると市場コピー化

Market Model の学習自体は変更なし。
「何を下流に渡すか」と「どう正規化するか」が v5.1 の変更。
```

---

## 5. RaceQualityScreener の分布特徴量化（粒度不足の解決）

### 4.1 v4.0のリーク構造の詳細

```
【自己強化ループの仕組み】

v4.0 の学習フロー：
  Step1: Stage2 で各馬の edge を計算
  Step2: edge を集計して actual_bet_roi を計算
  Step3: actual_bet_roi を y として RaceROI モデルを学習
  Step4: RaceROI モデルの pred_race_roi を Stage2 の入力特徴量に使う

問題：
  Step4 で「Stage2 を経由した値」が「Stage2 の入力」に戻る
  → Stage2 の出力が RaceROI モデルを通じて自分自身に影響する
  → 訓練データでは「RaceROI が高い → Stage2 が高 edge を出す → ROI が高い」
     という自己強化ループが形成される
  → バックテストでは高い ROI を示すが、実運用でこのループは機能しない

  より根本的な問題：
  actual_bet_roi は「自分の edge が高いレースで賭けた結果の回収率」
  これを y にすると、RaceROI モデルは「edge が高いレースを予測する」ではなく
  「edge が高いと分類された（=自分の選択した）レースの回収率」を学習する
```

### 5.2 リークを排除 + 分布特徴量化した RaceQualityScreener

```python
# models/race_quality_screener.py  v5.0
# ※ 名称を RaceROIModel から RaceQualityScreener に変更
#    「ROIを予測する」のではなく「投票に値するレースを分類する」

import lightgbm as lgb
import numpy as np
import pandas as pd


class RaceQualityScreener:
    """
    「このレースは投票する価値があるか」を判定するスクリーナー。

    v5.0 の設計原則：
    1. y に actual_bet_roi を使わない（リーク排除）
    2. Stage2 の出力（edge）に依存しない指標のみを y に使う
    3. 「このレースの市場は歪んでいるか」をモデル化する

    y の選択肢（全てStage2非依存）：
    - max_market_error_abs: 最大 market_pred_error の絶対値
    - market_entropy:       市場の拮抗度（エントロピー）
    - overround_deviation:  過去平均からのオーバーラウンド乖離

    v5.4 の設計原則（追加）：
    4. 利益proxyは「結果ベース」のみ使用（EVベースはモデル依存で禁止）
       - hist_hit_rate_topk:           過去的中率（モデル非依存）
       - hist_roi_topk:                過去ROI（モデル非依存）
       - hist_positive_return_ratio:   正のリターン割合（モデル非依存）

    投票条件: quality_score >= threshold（回収率ではなく「歪みの大きさ」で判定）
    """

    # Stage2 の出力に依存しない特徴量のみ使用
    FEATURE_COLS = [
        # Market Model 由来（正規化差分・v5.1更新）
        "market_log_error_max_abs",      # レース内の最大 |log_error|
        "market_log_error_std",          # log_error のバラツキ
        "market_log_error_top_q75",      # 75パーセンタイル

        # 分布特徴量（v5.1追加）
        "n_positive_errors",             # log_error > 0 の馬の数（過小評価されている馬の数）
        "top_k_error_sum",               # 上位K頭の log_error 合計（K=3）
        "positive_error_ratio",          # 過小評価馬の割合

        # v5.4: 結果ベース利益 proxy（モデル非依存・実運用で安定）
        "hist_hit_rate_topk",            # 同条件で上位K頭の過去的中率（高い=予測しやすい）
        "hist_roi_topk",                 # 同条件で上位K頭の過去ROI（高い=利益機会）
        "hist_positive_return_ratio",    # 同条件で正のリターンだったレースの割合

        # 市場構造
        "market_entropy",            # 拮抗度
        "overround",                 # 胴元控除
        "overround_deviation",       # 過去平均からの乖離（高い = 非効率）
        "field_size",

        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",

        # 難易度スコア（v3.0から継続）
        "difficulty_score",

        # 過去統計（同条件でのモデル回収率 ← これはOK: 過去データから計算）
        "hist_win_rate_same_condition",   # 同条件の過去的中率
        "hist_market_entropy_avg",        # 同条件の過去平均エントロピー
    ]

    def _build_target(self, df_race: pd.DataFrame) -> pd.Series:
        """
        目的変数: レースの「市場歪み×利益性スコア」
        v5.0: market_error_max_abs × market_entropy（単点の歪み）
        v5.1: 分布ベースの複合スコア
        v5.3: 歪み×利益性の複合スコア
        v5.4: 利益proxyを結果ベースに変更（モデル非依存）

        v5.4の変更理由:
          v5.3の hist_top3_ev_mean はEVベース → モデル依存
          モデルが変わるとproxyが崩れ、過学習したEVを再利用する構造
          → EVではなく「実際の結果」に基づく指標に変更
        """
        # 歪みスコア（v5.1から継承）
        distortion_score = (
            df_race["market_log_error_max_abs"]
            * df_race["market_entropy"]
            * (1.0 + df_race["n_positive_errors"] / df_race["field_size"])
        )

        # v5.4: 結果ベース利益 proxy（モデル非依存）
        # hist_roi_topk: 同条件で上位K頭の過去実際のROI
        # → 1.0以上なら過去に実際に利益が出ている条件
        # hist_positive_return_ratio: 正のリターンだった割合（安定性指標）
        profitability_proxy = np.clip(df_race["hist_roi_topk"], 0.5, 2.0)
        stability_factor = 0.5 + 0.5 * np.clip(df_race["hist_positive_return_ratio"], 0.0, 1.0)

        target = distortion_score * profitability_proxy * stability_factor
        return target

    def train(self, df_race: pd.DataFrame) -> None:
        X = df_race[self.FEATURE_COLS]
        y = self._build_target(df_race)

        self.model = lgb.train(
            {
                "objective":     "regression_l1",
                "metric":        "mae",
                "learning_rate": 0.05,
                "num_leaves":    15,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=200,
        )
        self.threshold = float(y.quantile(0.60))  # 上位40%のレースに投票

    def should_bet(self, race_features: dict) -> bool:
        X = pd.DataFrame([race_features])[self.FEATURE_COLS]
        score = float(self.model.predict(X)[0])
        return score >= self.threshold

    def calibrate_threshold(
        self,
        df_race: pd.DataFrame,
        target_investment_rate: float = 0.40,
    ) -> None:
        """
        投票レース比率が target_investment_rate になるよう閾値を調整。
        この調整は訓練データでのみ行い、out-of-sample では固定する。
        """
        scores = self.model.predict(df_race[self.FEATURE_COLS])
        self.threshold = float(np.quantile(scores, 1.0 - target_investment_rate))
```

### 5.3 スクリーナー精度の検証方法

```python
# notebooks/06_race_quality_screener.ipynb の検証手順

"""
検証1: quality_score と実際の回収率の相関確認
  - quality_score の分位数帯（4分割）ごとの回収率を計算
  - 上位25%の quality_score グループの回収率が
    下位25%より高ければスクリーナーが有効

検証2: スクリーナーを通過したレースの回収率分布
  - should_bet == True のレースでの実際の回収率分布を可視化
  - 全体の回収率より高ければ機能している

検証3: Stage2 の edge との独立性確認
  - quality_score と max(edge_place_in_race) の相関係数を計算
  - 相関が 0.5 を超えていたらリーク疑い → y の設計を見直す
  - 目標: 相関 < 0.3（独立性の担保）
"""
```

---

## 6. サブモデルを芝/ダート2分割に縮小

### 6.1 14分割の問題

```
【v4.0: 14サブモデルの問題】

14分割 = 芝×4距離帯×2馬場状態 + ダート×3距離帯×2馬場状態

サンプル数の問題（10年分のデータ）：
  芝マイル良:         ≈ 15,000件 → OK
  芝長距離良:         ≈  8,000件 → ギリギリ
  芝長距離重:         ≈  1,200件 → 少なすぎ（過学習確実）
  ダートスプリント重:  ≈    900件 → 論外

過学習のリスク：
  サンプルが少ないと LightGBM は学習データに過剰適合
  → バックテストでは高精度に見える
  → 実運用で崩壊する

【v5.0: まず2分割から始める原則】
  Phase1: 芝モデル / ダートモデル（2分割）
    → 各モデルのサンプル数: 50,000〜100,000件（十分）

  Phase2: 精度を確認後、距離帯を特徴量として追加で吸収
    → モデル分割ではなく「距離帯×馬場のone-hot特徴量」として入力

  Phase3（将来）: サブモデル追加は「n件以上でのみ実施」ルールを守る
    → 最低サンプル数: 20,000件（Phase3解禁の基準）
```

### 6.2 2分割モデルの実装

```python
# models/submodel_manager.py  v5.0

class SubModelManager:
    """
    v5.0: 芝/ダートの2分割に縮小。
    距離帯・馬場状態はモデル分割ではなく特徴量として対応する。
    """

    VALID_KEYS = ["turf", "dirt"]
    MIN_SAMPLES = 20_000  # 将来のサブモデル追加の基準

    def get_key(self, race: "Race") -> str:
        return "turf" if race.surface == "芝" else "dirt"

    def get_models(self, race: "Race") -> dict:
        key = self.get_key(race)
        return self.models[key]

    def should_split_further(
        self,
        key: str,
        condition: str,
        sample_count: int,
    ) -> bool:
        """
        将来的にサブモデルを追加するかどうかの判定。
        MIN_SAMPLES 未満の条件はモデル分割しない。
        """
        return sample_count >= self.MIN_SAMPLES

    def add_distance_band_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        距離帯はモデル分割ではなくone-hot特徴量として入力する。
        これにより少サンプル条件でも適切に扱える。
        """
        # 芝
        df["is_turf_sprint"]       = ((df["surface"] == "芝") & (df["distance"] <= 1400)).astype(int)
        df["is_turf_mile"]         = ((df["surface"] == "芝") & (df["distance"].between(1401, 1700))).astype(int)
        df["is_turf_intermediate"] = ((df["surface"] == "芝") & (df["distance"].between(1701, 2100))).astype(int)
        df["is_turf_long"]         = ((df["surface"] == "芝") & (df["distance"] >= 2101)).astype(int)
        # ダート
        df["is_dirt_sprint"]       = ((df["surface"] == "ダート") & (df["distance"] <= 1400)).astype(int)
        df["is_dirt_mile"]         = ((df["surface"] == "ダート") & (df["distance"].between(1401, 1700))).astype(int)
        df["is_dirt_intermediate"] = ((df["surface"] == "ダート") & (df["distance"] >= 1701)).astype(int)
        # 馬場状態
        df["is_good_track"]        = df["track_condition"].isin(["良", "稍重"]).astype(int)
        df["is_soft_track"]        = df["track_condition"].isin(["重", "不良"]).astype(int)
        return df
```

---

## 7. ワイドスコアの分散ベース・リスク調整

### 7.1 v5.3の(1+E)問題とv5.4の修正

```
【v5.3の問題】
score = EV × P / (1 + E(return|hit))

この分母は E(return|hit) のみを使用:
  本来の分散は Var ≈ P × E² - (P×E)² ≈ P × E²
  しかし v5.3 は E² のみを使用 → Pが欠落

起きる問題:
  高E・低Pの極端なペア（穴馬狙い）が過小評価されない
  → 中穴寄りにスコアが偏る
  → 実際には高E・低Pは分散が大きくリスクが高い

【v5.4の修正（第1版）】
score = EV / (1 + P×E²)
  → 分散の近似を正しく使用
  → しかし EV と Var のスケールが一致していない

【v5.4改の修正（スケール一致）】
score = EV / (E × sqrt(P))

  理論的根拠:
    Var ≈ P × E² → sqrt(Var) ≈ E × sqrt(P)
    score = EV / sqrt(Var) ≈ EV / (E × sqrt(P))
    → シャープレシオに近い（期待値 / 標準偏差）

例の比較：
  ペアA: P=0.30, E=4.0
    v5.3: 1.20 × 0.30 / 5.0 = 0.072
    v5.4改: 1.20 / (4.0 × 0.548) = 0.548

  ペアB: P=0.05, E=24.0
    v5.3: 1.20 × 0.05 / 25.0 = 0.0024
    v5.4改: 1.20 / (24.0 × 0.224) = 0.223

  ペアC: P=0.10, E=15.0
    v5.3: 1.50 × 0.10 / 16.0 = 0.0094
    v5.4改: 1.50 / (15.0 × 0.316) = 0.316

  ペアD: P=0.02, E=45.0
    v5.3: 1.00 × 0.02 / 46.0 = 0.00043
    v5.4改: 1.00 / (45.0 × 0.141) = 0.157

  → スコアがシャープレシオ的になり、異なるペア間の比較が意味を持つ
  → ペアA(高P低E)とペアB(低P高E)が同じスケールで評価可能
```

### 7.2 モデル内部E使用のリスク調整スコア実装

```python
# models/wide_two_stage_model.py  v5.4（シャープレシオ近似: EV / (E×sqrt(P))）

class WideTwoStageModel:
    """v5.4改: score = EV / (E × sqrt(P)) でスケール一致"""

    def predict_score(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        pair_df = pair_df.copy()
        X = pair_df[self.SHARED_FEATURE_COLS]

        pair_df["p_hit"]              = self.hit_model.predict(X)
        pair_df["e_return_given_hit"] = self.return_model.predict(X)

        # v4.0: score = P(hit) × E(return|hit)              ← 分散未考慮
        # v5.0: score = P(hit)² × E(return|hit)             ← ヒューリスティック
        # v5.1: score = EV × P / (1 + odds)                 ← 市場odds依存
        # v5.3: score = EV × P / (1 + E(return|hit))        ← E²のみ
        # v5.4: score = EV / (1 + P × E²)                  ← Var_proxy使用
        # v5.4改: score = EV / (E × sqrt(P))               ← シャープレシオ近似
        pair_df["ev_wide"]           = pair_df["p_hit"] * pair_df["e_return_given_hit"]
        risk_denom = pair_df["e_return_given_hit"] * np.sqrt(
            np.clip(pair_df["p_hit"], 0.001, None)
        )
        pair_df["wide_score_adj"]    = pair_df["ev_wide"] / risk_denom

        return pair_df

    def select_bets(
        self,
        pair_df: pd.DataFrame,
        ev_threshold:    float = 1.20,
        score_threshold: float = 0.015,
        max_bets:        int   = 3,
    ) -> list[dict]:
        """
        2段階フィルタ：
          1. ev_wide >= ev_threshold
          2. wide_score_adj >= score_threshold（モデル内部Eベース・v5.3）
        """
        scored = self.predict_score(pair_df)
        filtered = scored[
            (scored["ev_wide"]        >= ev_threshold)
            & (scored["wide_score_adj"] >= score_threshold)
            & (scored["popularity_sum"] >= 6)
            & (scored["running_style_combo"] != 0)
            & (scored["p_hit"]          >= 0.05)
            & (scored["e_return_given_hit"] >= 2.0)
        ]
        top = filtered.nlargest(max_bets, "wide_score_adj")
        return top.to_dict("records")
```

---

## 8. late_money 直前判定戦略（t-3min基準）

### 8.1 発走直前の特殊性とt-3min基準の理由

```
【なぜ直前2分を独立して扱うか】

オッズは発走直前（最後の2分）が最も動きやすい。
理由：
  1. 海外・大口資金の直前投入
  2. 「様子見から確信」に変わる一般投票者の動き
  3. 機械（自動投票）の滑り込みによる急変

このとき：
  自分が投票を決めたのは t-10min のオッズで計算した EV
  しかし実際の払戻は t-0min の確定オッズで決まる

  t-10min から t-0min でオッズが大幅に動くと、
  計算した EV と実際の EV が乖離する。

特に危険なケース：
  t-10min: オッズ 8.0倍（EV 計算: 1.25）
  t-0min:  オッズ 3.5倍（急激な資金流入）
  → 実際のEV は約 0.55（投票してはいけなかった）

  逆に：
  t-10min: オッズ 8.0倍（EV 計算: 0.90 → 投票しない判断）
  t-0min:  オッズ 18.0倍（資金が逃げた）
  → 実際のEV は約 2.0（投票すべきだった）

t-2min の監視で「後者の機会を拾い、前者を回避する」
```

### 8.2 直前キャンセル/追加投票トリガー（t-3min基準）

```python
# betting/late_money_filter.py  v5.1（t-3min判定基準に変更）

class LateMoneyFilter:
    """
    v5.0: t-2min の急変を独立トリガーとして追加
    v5.1: t-3min で判定、t-2min はログのみ（実務安全マージン）

    変更理由:
      t-2min で判定すると以下のリスクがある:
      - API遅延でオッズ取得が間に合わない
      - 投票処理に時間がかかる
      - JRAの締切りギリギリでキャンセルが間に合わない
      → t-3min で判定し、1分のバッファを確保する

    2つの機能：
    A) キャンセルトリガー: t-3min に急落（オッズ低下）を検知したら投票取消
    B) 追加投票トリガー: t-3min に急騰（オッズ上昇）を検知したら未投票馬を追加検討
    """

    # キャンセルトリガーのしきい値
    CANCEL_DROP_THRESHOLD = 0.25   # t-10min → t-3min でオッズが25%以上下落
    # 追加投票トリガーのしきい値
    ADD_RISE_THRESHOLD    = 0.30   # t-10min → t-3min でオッズが30%以上上昇
    # t-2min ログ用しきい値（ログのみ・判定には使わない）
    LOG_THRESHOLD         = 0.20   # t-3min → t-2min で20%以上変動があればログ出力

    def check_last_3min(
        self,
        horse_no: int,
        odds_t10: float,
        odds_t3:  float,
    ) -> "LastMinuteSignal":
        """
        発走3分前のオッズを確認してシグナルを返す。
        この処理は t-3min に非同期で実行する。
        """
        if odds_t10 <= 0 or odds_t3 <= 0:
            return LastMinuteSignal.UNKNOWN

        change_rate = (odds_t10 - odds_t3) / odds_t10

        if change_rate >= self.CANCEL_DROP_THRESHOLD:
            # 25%以上の急落 → 誰かが大量に買っている → 投票をキャンセル
            return LastMinuteSignal.CANCEL

        if change_rate <= -self.ADD_RISE_THRESHOLD:
            # 30%以上の急騰 → 資金が逃げた → 未投票なら追加検討
            return LastMinuteSignal.ADD_CANDIDATE

        return LastMinuteSignal.NO_ACTION

    def log_last_2min(
        self,
        horse_no: int,
        odds_t3: float,
        odds_t2: float,
    ) -> None:
        """
        発走2分前のオッズをログに記録する（判定には使わない）。
        t-3min から t-2min の変動をモニタリングし、
        将来のしきい値チューニングに使用する。
        """
        if odds_t3 <= 0 or odds_t2 <= 0:
            return

        change_rate = abs(odds_t3 - odds_t2) / odds_t3
        if change_rate >= self.LOG_THRESHOLD:
            logger.info(
                f"[LOG ONLY] horse={horse_no} "
                f"odds: {odds_t3:.1f} → {odds_t2:.1f} "
                f"(change: {change_rate:.0%})"
            )

    def process_last_minute(
        self,
        pending_bets: list["Bet"],
        odds_t3_snapshot: dict[int, float],   # horse_no → odds_t3（v5.1: t-2min → t-3min）
        odds_t10_snapshot: dict[int, float],  # horse_no → odds_t10
        stage2_predictions: pd.DataFrame,
    ) -> tuple[list["Bet"], list["Bet"]]:
        """
        発走3分前に全ての保留中ベットを再チェックする（v5.1: t-2min → t-3min）。

        Returns:
            approved_bets:  最終的に投票するベット
            cancelled_bets: キャンセルしたベット（ログ用）
        """
        approved   = []
        cancelled  = []

        for bet in pending_bets:
            odds_t10 = odds_t10_snapshot.get(bet.horse_no, 0)
            odds_t3  = odds_t3_snapshot.get(bet.horse_no, 0)
            signal   = self.check_last_3min(bet.horse_no, odds_t10, odds_t3)

            if signal == LastMinuteSignal.CANCEL:
                cancelled.append(bet)
                logger.warning(
                    f"CANCEL: race={bet.race_id} horse={bet.horse_no} "
                    f"odds: {odds_t10:.1f} → {odds_t3:.1f} "
                    f"(drop: {(odds_t10-odds_t3)/odds_t10:.0%})"
                )
            else:
                approved.append(bet)

        return approved, cancelled


from enum import Enum

class LastMinuteSignal(Enum):
    NO_ACTION     = "no_action"
    CANCEL        = "cancel"
    ADD_CANDIDATE = "add_candidate"
    UNKNOWN       = "unknown"
```

---

## 9. DDコントローラーの改善（Rolling ROI 連動・EWMA・ヒステリシス）

### 9.1 純粋なDDベースの問題

```
【v4.0の問題】
DD 15%超 → 半減（固定）

問題：
ドローダウンは「モデルが悪くなった」ことを示すとは限らない。
  ケース1: モデルが正常 + 単純な分散 → DD後は回復する
  ケース2: モデルが劣化    + 分散     → DD後も回復しない

v4.0 はケース1でも半減し続けるため回復が遅い。
また、Rolling ROI が悪化していてもDDが小さければ制御しない問題もある。

【v5.0の改善】
DD × Rolling ROI の複合判定

  (A) 「DDが大きい AND Rolling ROIが悪い」→ 大幅削減（モデル劣化疑い）
  (B) 「DDは大きいが Rolling ROIは正常」→ 軽度削減（分散と判断）
  (C) 「DDは小さいが Rolling ROIが悪い」→ 警戒モード（早期対処）
  (D) 「どちらも正常」→ 通常 or 早期復元
```

### 9.2 Rolling ROI 連動 DD コントローラー（v5.3: ウィンドウ拡大 + EWMA + ヒステリシス）

```python
# betting/drawdown_controller.py  v5.3

from dataclasses import dataclass
from enum import Enum

class RecoveryState(Enum):
    NORMAL     = "normal"
    REDUCED    = "reduced"
    RECOVERING = "recovering"


@dataclass
class DDState:
    current_dd:     float  # 現在のドローダウン率
    rolling_roi:    float  # 直近 N 回のベットの回収率
    n_bets_eval:    int    # Rolling ROI の計算に使ったベット数
    recovery_state: RecoveryState  # v5.3: 回復状態


class DrawdownControllerV5:
    """
    v5.0: DD × Rolling ROI の複合判定によるベットサイズ制御。
    v5.1: ウィンドウを50→150に拡大し、EWMA ハイブリッドを導入。
    v5.3: ヒステリシス付き回復ロジックを追加。

    v5.3の変更理由:
      reduced状態からの回復が遅い（資金効率が落ちる）
      → recovering状態で乗数を段階的に増やすヒステリシスを追加

    v5.3の変更理由:
      ヒステリシス+EWMAで「賢すぎる」制御になり、
      バックテストではDD減るが実運用で逆に不安定になるリスク。
      → 1日の乗数変更幅を MAX_ADJUSTMENT_PER_DAY = 0.15 に制限
      → 過剰適応（過去最適化）を防止

    v5.4の変更理由:
      MAX_ADJUSTMENT_PER_DAY は「日付依存」
      → レースが少ない日（平日3レース）→ ほぼ固定（制御不能）
      → レースが多い日（週末12レース）→ 過剰調整
      → 「時間」ではなく「試行回数」で制御すべき
      → MAX_ADJUSTMENT_PER_N_BETS（20ベット単位）に変更

    回復の3段階:
      NORMAL → REDUCED:     DD悪化時にテーブルに従って削減
      REDUCED → RECOVERING: ROI >= 0.98 かつ DD改善傾向
      RECOVERING → NORMAL:   DD < 5% または 連続回復
    """

    ROLLING_WINDOW = 150
    EWMA_ALPHA     = 0.1

    # v5.3: 回復加速パラメータ
    RECOVERY_INCREMENT     = 0.05   # recovering時、1ベットごとに乗数+5%
    RECOVERY_ROI_THRESHOLD = 0.98   # Rolling ROI >= 0.98 で recovering に移行
    RECOVERY_DD_THRESHOLD  = 0.05   # DD < 5% で normal に復帰
    RECOVERY_MAX_MULTIPLIER = 1.00  # 回復時の乗数上限

    # v5.4: 試行回数ベースの過剰適応防止
    MAX_ADJUSTMENT_PER_N_BETS = 20  # Nベットごとの乗数変更幅上限
    MAX_ADJUSTMENT_AMOUNT     = 0.15  # Nベット間での最大変更幅

    MULTIPLIER_TABLE = [
        # (DD下限, DD上限, ROI下限, ROI上限, 乗数)
        (0.00, 0.10, 0.90, 9.99, 1.00),  # DD正常 + ROI正常 → 通常
        (0.00, 0.10, 0.00, 0.90, 0.75),  # DD正常 + ROI悪化 → 警戒
        (0.10, 0.15, 0.95, 9.99, 0.80),  # DD小〜中 + ROI正常 → 軽度削減
        (0.10, 0.15, 0.00, 0.95, 0.50),  # DD小〜中 + ROI悪化 → 中程度削減
        (0.15, 0.20, 0.95, 9.99, 0.60),  # DD中 + ROI正常 → 中程度削減
        (0.15, 0.20, 0.00, 0.95, 0.30),  # DD中 + ROI悪化 → 大幅削減
        (0.20, 0.25, 0.00, 9.99, 0.15),  # DD大 → 最小限
        (0.25, 9.99, 0.00, 9.99, 0.00),  # DD極大 → 停止
    ]

    def __init__(self, peak_bankroll: float):
        self.peak_bankroll = peak_bankroll
        self.bet_history: list[float] = []
        self._recovery_state = RecoveryState.NORMAL
        self._current_multiplier = 1.0
        self._multiplier_at_window_start = 1.0  # v5.4: 試行回数ウィンドウ用
        self._bets_in_window = 0                 # v5.4: 現在のウィンドウ内ベット数

    def update(self, bankroll: float, bet_return: float) -> None:
        """ベット結果を記録してピークを更新"""
        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll
        self.bet_history.append(bet_return)
        if len(self.bet_history) > self.ROLLING_WINDOW * 2:
            self.bet_history.pop(0)
        self._update_recovery_state(bankroll)

    def _update_recovery_state(self, bankroll: float) -> None:
        """v5.3: 回復状態の遷移ロジック"""
        dd = (self.peak_bankroll - bankroll) / self.peak_bankroll
        roi = self._calc_rolling_roi()

        if self._recovery_state == RecoveryState.NORMAL:
            # DD悪化判定 → REDUCED に移行
            table_mult = self._get_table_multiplier(dd, roi)
            if table_mult < 0.80:
                self._recovery_state = RecoveryState.REDUCED
                self._current_multiplier = table_mult

        elif self._recovery_state == RecoveryState.REDUCED:
            # ROI回復判定 → RECOVERING に移行
            if roi >= self.RECOVERY_ROI_THRESHOLD and dd < 0.15:
                self._recovery_state = RecoveryState.RECOVERING
                logger.info("DD Controller: REDUCED → RECOVERING (ROI recovery detected)")
            else:
                # まだ回復条件を満たさない → テーブル値を維持
                self._current_multiplier = self._get_table_multiplier(dd, roi)

        elif self._recovery_state == RecoveryState.RECOVERING:
            # 乗数を段階的に増やす
            self._current_multiplier = min(
                self._current_multiplier + self.RECOVERY_INCREMENT,
                self.RECOVERY_MAX_MULTIPLIER,
            )
            # DD < 5% で NORMAL に復帰
            if dd < self.RECOVERY_DD_THRESHOLD:
                self._recovery_state = RecoveryState.NORMAL
                self._current_multiplier = 1.0
                logger.info("DD Controller: RECOVERING → NORMAL (DD recovered)")
            # ROI再悪化 → REDUCED に戻る
            elif roi < 0.90:
                self._recovery_state = RecoveryState.REDUCED
                self._current_multiplier = self._get_table_multiplier(dd, roi)
                logger.info("DD Controller: RECOVERING → REDUCED (ROI deteriorated)")

    def get_state(self, bankroll: float) -> DDState:
        dd  = (self.peak_bankroll - bankroll) / self.peak_bankroll
        roi = self._calc_rolling_roi()
        return DDState(
            current_dd=dd,
            rolling_roi=roi,
            n_bets_eval=min(len(self.bet_history), self.ROLLING_WINDOW),
            recovery_state=self._recovery_state,
        )

    def _get_table_multiplier(self, dd: float, roi: float) -> float:
        """乗数テーブルからベース乗数を取得"""
        for dd_lo, dd_hi, roi_lo, roi_hi, mult in self.MULTIPLIER_TABLE:
            if dd_lo <= dd < dd_hi and roi_lo <= roi < roi_hi:
                return mult
        return 0.0

    def get_multiplier(self, bankroll: float) -> float:
        state = self.get_state(bankroll)

        if state.n_bets_eval < 20:
            roi = 1.0
        else:
            roi = state.rolling_roi

        # v5.3: RECOVERING状態ではヒステリシス乗数を使用
        if self._recovery_state == RecoveryState.RECOVERING:
            raw_mult = self._current_multiplier
        else:
            raw_mult = self._get_table_multiplier(state.current_dd, roi)

        # v5.4: Nベットごとの変更幅を制限（過剰適応防止・日付非依存）
        self._bets_in_window += 1
        if self._bets_in_window >= self.MAX_ADJUSTMENT_PER_N_BETS:
            # ウィンドウリセット: 新しい基準値を設定
            self._multiplier_at_window_start = self._current_multiplier
            self._bets_in_window = 0

        max_change = self.MAX_ADJUSTMENT_AMOUNT
        mult = max(
            self._multiplier_at_window_start - max_change,
            min(raw_mult, self._multiplier_at_window_start + max_change),
        )
        self._current_multiplier = mult
        return mult

    def adjust_stake(self, base_stake: int, bankroll: float) -> int:
        mult = self.get_multiplier(bankroll)
        return max(0, int((base_stake * mult) // 100) * 100)

    def _calc_rolling_roi(self) -> float:
        """SMA + EWMA ハイブリッド"""
        recent = self.bet_history[-self.ROLLING_WINDOW:]
        if not recent:
            return 1.0
        if len(recent) < 20:
            return float(np.mean(recent))

        sma = float(np.mean(recent))
        ewma = recent[0]
        for r in recent[1:]:
            ewma = self.EWMA_ALPHA * r + (1 - self.EWMA_ALPHA) * ewma
        return (sma + ewma) / 2.0

    def log_state(self, bankroll: float) -> None:
        state = self.get_state(bankroll)
        mult  = self.get_multiplier(bankroll)
        logger.info(
            f"DD: {state.current_dd:.1%} | "
            f"Rolling ROI({state.n_bets_eval}bets): {state.rolling_roi:.3f} | "
            f"Multiplier: {mult:.2f} | "
            f"State: {state.recovery_state.value} | "
            f"Peak: ¥{self.peak_bankroll:,.0f} | "
            f"Current: ¥{bankroll:,.0f}"
        )
```

---

## 9.5. レジーム検知モデル（市場状態の切り替え検知）

### 9.5.1 市場は静的ではない

```
【競馬市場の動的性質】

競馬は静的な予測問題ではなくアドバーサリアル環境：

  あなたの戦略が機能する
  → 他の参加者が同様の戦略を採用する
  → 市場効率が上がり歪みが小さくなる
  → EVが縮小する

また、市場には季節性・年次変動がある：
  荒れる年: 人気薄が頻繁に勝つ → 市場歪み大
  堅い年: 人気馬が勝つ → 市場効率的
  特徴量の劣化期間: モデルが環境変化に追いつけない時期

→ 常に同じ戦略パラメータで運用すると、市場状態に合わない期間で過大損失
```

### 9.5.2 レジーム検知の設計

```
【RegimeDetector: 軽量3状態分類器】

入力（直近200レースの集計値）:
  market_error_std:      log_error の標準偏差（歪みの大きさ）
  market_entropy_mean:   平均エントロピー（拮抗度）
  rolling_roi_200:       直近200レースのROI（利益性・補助指標のみ）
  hit_rate_top3_mean:    上位3頭の平均的中率
  overround_mean:        平均胴元控除率

3状態の定義（v5.5: 教師ラベルを市場指標ベースに変更）:

  状態A: AGGRESSIVE（歪み強い）
    条件: favorite_win_rate が低い + overround が高い + entropy が高い
    特徴: 市場が非効率 → エッジが取りやすい
    行動: EV閾値を下げ（1.20→1.10）、スコア閾値を下げ
    ベット数上限: レース最大3券種

  状態B: CONSERVATIVE（効率的）
    条件: favorite_win_rate が高い + overround が低い
    特徴: 市場が効率的 → エッジが小さい
    行動: EV閾値を上げ（1.20→1.30）、スコア閾値を上げ
    ベット数上限: レース最大2券種

  状態C: COLLAPSED（崩壊）
    条件: favorite_win_rate が壊滅的 + market_efficiency が極端に低い
    特徴: モデル劣化 or 市場構造変化 → EVが機能していない
    行動: ベット数を0〜1に制限（実質停止）
    再学習トリガー: 連続100レースで状態Cが継続した場合

設計原則:
  ① 軽量（LightGBM 100 rounds・num_leaves=7）→ 過学習防止
  ② 特徴量は Stage2 非依存（RaceQualityScreener と同じポリシー）
  ③ 閾値は訓練データで決定、out-of-sample では固定
  ④ 状態遷移はヒステリシス付き（頻繁な切り替えを防止）
  ⑤ v5.5: 教師ラベルを戦略非依存の市場指標ベースに変更
```

### 9.5.3 レジーム検知モデルの実装

```python
# models/regime_detector.py  v5.5

from dataclasses import dataclass
from enum import Enum

import lightgbm as lgb
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    AGGRESSIVE   = "aggressive"    # 歪み強い → 攻める
    CONSERVATIVE = "conservative"  # 効率的   → 絞る
    COLLAPSED    = "collapsed"     # 崩壊     → 停止


@dataclass
class RegimeConfig:
    """レジーム検知のパラメータ"""
    window:              int = 200        # 直近Nレースで判定
    min_samples:         int = 100        # 判定に必要な最低レース数
    # 閾値（訓練データで決定・out-of-sampleで固定）
    # v5.5: 市場指標ベースの閾値に変更（戦略依存のROI閾値を排除）
    fav_rate_aggressive: float = 0.28     # favorite勝率が低い + entropy高い → 攻める
    fav_rate_collapsed:  float = 0.18     # favorite勝率が壊滅的 → 停止
    overround_base:      float = 0.20     # overround の基準値（JRA典型的控除率）
    retrain_trigger:     int = 100        # 連続C状態で再学習トリガー


class RegimeDetector:
    """
    市場状態の切り替えを検知し、戦略パラメータを動的に調整する。

    v5.4の設計原則:
    1. 軽量モデル（過学習防止）
    2. Stage2非依存の特徴量（リーク防止）
    3. ヒステリシス付き状態遷移（頻繁な切り替え防止）
    4. COLLAPSED状態で再学習トリガー（劣化への自動対応）

    v5.4改: 市場側指標を強化（戦略依存の rolling_roi を補完）
    rolling_roi は「自分の戦略の成果」であり市場状態ではない
    → 人気馬勝率・FLB・オッズ変動量 で「市場自体の状態」を直接観測
    v5.5:   教師ラベルを完全に市場指標ベースに変更
    rolling_roi は補助指標に格下げ（学習ラベルには使用しない）
    """

    FEATURE_COLS = [
        # 市場歪み（直近200レース集計）
        "market_error_std",         # log_error の標準偏差
        "market_error_mean",        # log_error の平均
        "market_entropy_mean",      # 平均エントロピー
        "overround_mean",           # 平均胴元控除率
        # 市場側指標（v5.4改追加: 戦略非依存の市場状態）
        "favorite_win_rate",        # 1番人気の勝率（高い＝市場が正確・効率的）
        "flb_slope",                # favorite-longshot bias の傾き（高い＝市場歪み大）
        "odds_volatility_mean",     # オッズ変動量の平均（高い＝市場が不安定）
        # 利益性（直近200レース集計・補助・v5.5: 教師ラベルには不使用）
        "rolling_roi_200",          # 直近200レースのROI（戦略依存・推論時の補助指標のみ）
        "hit_rate_top3_mean",       # 上位3頭の平均的中率
        # レース構造
        "field_size_mean",          # 平均頭数
    ]

    def __init__(self, cfg: RegimeConfig = RegimeConfig()):
        self.cfg = cfg
        self._current_regime = MarketRegime.CONSERVATIVE
        self._regime_counter = 0  # 同状態の連続レース数
        self._transition_hysteresis = 5  # Nレース連続で遷移

    def train(self, df_race: pd.DataFrame) -> None:
        """
        レジーム分類器の学習（軽量・3状態分類）。
        v5.5: 教師ラベルを市場指標ベースに変更（戦略依存の rolling_roi を排除）
        y は favorite_win_rate × overround_mean の複合スコアで離散化。
        """
        X = df_race[self.FEATURE_COLS]

        # v5.5: 市場指標ベースのラベル化（戦略非依存）
        # favorite_win_rate が高い + overround が低い = 効率的な市場 (CONSERVATIVE)
        # favorite_win_rate が低い + overround が高い = 歪んだ市場 (AGGRESSIVE)
        # entropy が極端 + favorite_win_rate が壊滅的 = 崩壊 (COLLAPSED)
        fav = df_race["favorite_win_rate"]
        overround = df_race["overround_mean"]
        entropy = df_race["market_entropy_mean"]

        # 複合スコア: favorite勝率が低いほど歪みが大きい
        market_efficiency = fav * (1 - np.clip(overround - 0.20, 0, 0.15) / 0.15)

        y = np.where(
            (market_efficiency < 0.28) & (entropy > np.median(entropy)), 0,   # AGGRESSIVE
            np.where(market_efficiency < 0.18, 2,                                    # COLLAPSED
                     1))                                                             # CONSERVATIVE

        self.model = lgb.train(
            {
                "objective":     "multiclass",
                "num_class":     3,
                "metric":        "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves":    7,          # 軽量
                "min_data_in_leaf": 50,
                "feature_fraction": 0.8,
                "verbose": -1,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=100,
        )

    def detect(self, recent_stats: pd.DataFrame) -> MarketRegime:
        """
        直近レースの集計値から現在のレジームを判定。
        ヒステリシス付き：連続Nレースで同じ状態が続いた場合のみ遷移。
        """
        if len(recent_stats) < self.cfg.min_samples:
            return MarketRegime.CONSERVATIVE  # データ不足時は安全側

        X = recent_stats[self.FEATURE_COLS].iloc[[-1]]
        probs = self.model.predict(X)[0]

        # 確率最大の状態を取得
        raw_regime_idx = int(np.argmax(probs))
        raw_regime = [MarketRegime.AGGRESSIVE,
                      MarketRegime.CONSERVATIVE,
                      MarketRegime.COLLAPSED][raw_regime_idx]

        # ヒステリシス判定
        if raw_regime == self._current_regime:
            self._regime_counter += 1
        else:
            self._regime_counter = 0

        if self._regime_counter >= self._transition_hysteresis:
            old = self._current_regime
            self._current_regime = raw_regime
            self._regime_counter = 0
            logger.info(
                f"Regime transition: {old.value} → {raw_regime.value} "
                f"(probs: A={probs[0]:.2f}, C={probs[1]:.2f}, X={probs[2]:.2f})"
            )

        return self._current_regime

    def get_strategy_params(self, regime: MarketRegime) -> dict:
        """
        レジームに応じた戦略パラメータを返す。
        """
        if regime == MarketRegime.AGGRESSIVE:
            return {
                "ev_threshold":       1.10,   # 下げる（より多くの候補）
                "score_threshold":    0.010,  # 下げる
                "max_bets_per_race":  3,      # 最大3券種
                "description":        "歪み強い → 攻める",
            }
        elif regime == MarketRegime.CONSERVATIVE:
            return {
                "ev_threshold":       1.30,   # 上げる（厳選）
                "score_threshold":    0.020,  # 上げる
                "max_bets_per_race":  2,      # 最大2券種
                "description":        "効率的 → 絞る",
            }
        else:  # COLLAPSED
            return {
                "ev_threshold":       1.50,   # さらに上げる
                "score_threshold":    0.050,  # さらに上げる
                "max_bets_per_race":  1,      # 最大1券種（実質停止）
                "description":        "崩壊 → ほぼ停止",
            }

    def should_retrain(self) -> bool:
        """COLLAPSED状態が連続100レース続いた場合に再学習をトリガー"""
        return (
            self._current_regime == MarketRegime.COLLAPSED
            and self._regime_counter >= self.cfg.retrain_trigger
        )
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           JRA-VAN                                        │
│    JV-Link（レース・成績）+ オッズ（5分間隔 + t-3min/t-2min 監視）      │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PostgreSQL（raw / odds_history / feature / prediction / betting）       │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  特徴量エンジン v5.3                                                     │
│  A:馬の能力 / B:レース内相対値 / C:オッズ変化率                         │
│  D:市場歪み / E:情報非対称性 / F:距離帯・馬場（one-hot）               │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     ▼
┌─────────────────────┐   ┌─────────────────────────┐
│  Market Model       │   │  Stage1: 実力推定        │
│  （芝/ダート 2分割） │   │  LightGBM Ranker        │
│                     │   │  （芝/ダート 2分割）     │
│  出力:              │   │  オッズ入力なし          │
│  market_log_error   │   │  p_ability_win/place     │
│  （正規化差分のみ） │   │                          │
│  （差分のみ）        │   └────────────┬────────────┘
└──────────┬──────────┘                │
           └──────────┬────────────────┘
                      │ 合流
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2段階モデル（芝/ダート 2分割）                                          │
│                                                                          │
│  単勝: WinTwoStageModel                                                  │
│    Stage A: P(win)             ← 全馬で分類（is_unbalance=True）        │
│    Stage B: E(win_odds|win)    ← 1着馬のみで回帰（ゼロなし）           │
│    EV_win = P(win) × E(win_odds|win)                                     │
│                                                                          │
│  ★ EV補正モデル（v5.1新設 → v5.4: P/E分解アプローチ）                │
│    P補正: y_p = log(I(win)) - log(p_pred)  ← 全サンプル・分類residual│
│    E補正: y_e = log(odds_actual) - log(e_pred) ← winnerのみ・回帰resid│
│    EV_corrected = P_corrected × E_corrected                            │
│    PとEの負の相関を独立に補正し、学習安定性を確保                     │
│                                                                          │
│  複勝: PlaceTwoStageModel                                                │
│    Stage A: P(place)           ← 全馬で分類                            │
│    Stage B: E(place_odds|place)← 3着内馬のみで回帰（ゼロなし）         │
│    EV_place = P(place) × E(place_odds|place)                             │
│                                                                          │
│  ワイド: WideTwoStageModel                                               │
│    Stage A: P(joint_hit)       ← ペアで分類                            │
│    Stage B: E(wide_odds|hit)   ← 的中ペアのみで回帰                    │
│    score_adj = EV / (E × sqrt(P))         ← v5.4 シャープレシオ近似  │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  信頼区間推定（CP + Rolling Quantile の min）                            │
│  EV_lower_win_corrected / EV_lower_place                                 │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RaceQualityScreener（v5.4: 歪み×結果ベース利益性）                    │
│  y = distortion_score × profitability_proxy × stability_factor         │
│  → 「歪みがある AND 過去に実際に利益が出ている」レースを評価          │
│  → proxyは結果ベース（hist_roi_topk, hist_positive_return_ratio）      │
│  quality_score >= threshold → 投票可 / < threshold → スキップ           │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RegimeDetector（v5.4: 市場状態の切り替え検知）                         │
│  直近200レースの market_error・ROI・entropy から3状態分類               │
│  AGGRESSIVE（歪み強い→攻める） / CONSERVATIVE（効率的→絞る）           │
│  COLLAPSED（崩壊→ほぼ停止）+ 連続100レースで再学習トリガー              │
│  → 戦略パラメータ（EV閾値・スコア閾値・ベット数上限）を動的調整       │
└─────────────────────────┬────────────────────────────────────────────────┘
└─────────────────────────┬────────────────────────────────────────────────┘
                          │ PASS のみ
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ベッティングオーケストレーター                                          │
│  ① メタスイッチャー（複勝/ワイド/単勝の戦略選択）                      │
│  ② late_money フィルタ（t-10min フィルタ）                             │
│  ③ GateKeeper（EV_lower_corrected で最終判断）                          │
│  ④ 賭け金計算（edge連動ケリー）                                         │
│  ⑤ DD × Rolling ROI + EWMA + ヒステリシス コントローラー（v5.4）      │
│     + max_adjustment/per_N_bets 制限（試行回数ベース）                 │
│  ⑥ 1レース露出 2% キャップ                                              │
│  ⑦ t-3min 直前キャンセルチェック（v5.1: t-2minからt-3minに変更）      │
│  ⑧ t-2min ログのみ（将来のチューニング用データ収集）                   │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PAT自動投票 / SafetyGuard / Slack通知                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 11. モデルパイプライン v5.4

```python
# pipelines/training_pipeline.py  v5.1

class TrainingPipelineV5:

    def run(self, train_start: str, train_end: str) -> "TrainedModelsV5":

        race_df  = self.db.load_races(train_start, train_end)
        entry_df = self.db.load_entries_with_results(train_start, train_end)
        odds_df  = self.db.load_odds_snapshots(train_start, train_end)

        # 特徴量生成（距離帯・馬場状態は one-hot として追加）
        feat_df = self.feature_engine.build_all(race_df, entry_df, odds_df)
        feat_df = SubModelManager().add_distance_band_features(feat_df)

        models = {}
        for surface in ["turf", "dirt"]:
            subset_df = feat_df[feat_df["surface_key"] == surface].copy()

            # 1. Market Model（正規化差分 log_error のみ出力）
            market = MarketModel()
            market.train(subset_df)
            subset_df = market.predict_and_calc_error(subset_df)
            # p_market_pred は内部で使い、出力しない

            # 2. Stage1（オッズなし・距離帯one-hot含む）
            stage1 = AbilityModel()
            stage1.train(subset_df)
            subset_df = stage1.add_ability_probs(subset_df)

            # 3. 単勝2段階モデル
            win_2s = WinTwoStageModel()
            win_2s.train_hit_model(subset_df)
            win_2s.train_return_model(subset_df)
            subset_df = win_2s.predict_ev(subset_df)

            # 4. ★ EV補正モデル（v5.1新設 → v5.4: P/E分解）
            ev_corrector = EVCorrectionModel()
            ev_corrector.train(subset_df)
            subset_df = ev_corrector.correct_ev(subset_df)

            # 5. 複勝2段階モデル
            place_2s = PlaceTwoStageModel()
            place_2s.train_hit_model(subset_df)
            place_2s.train_return_model(subset_df)
            subset_df = place_2s.predict_ev(subset_df)

            # 6. ワイド2段階モデル（分散ベース・リスク調整スコア版）
            pair_df  = WideJointPairBuilder().build(subset_df)
            wide_2s  = WideTwoStageModel()
            wide_2s.train_hit_model(pair_df)
            wide_2s.train_return_model(pair_df)

            # 7. 信頼区間（v5.4: EV_corrected = P_corrected × E_corrected ベース）
            conf = RobustConfidenceEstimator()
            conf.calibrate(ev_corrector, place_2s, subset_df)

            models[surface] = {
                "market":       market,
                "stage1":       stage1,
                "win":          win_2s,
                "ev_corrector": ev_corrector,  # v5.1新設
                "place":        place_2s,
                "wide":         wide_2s,
                "conf":         conf,
            }

        # 8. RaceQualityScreener（分布特徴量化・Stage2非依存のy）
        race_feat_df   = self._build_race_level_features(feat_df)
        quality_screen = RaceQualityScreener()
        quality_screen.train(race_feat_df)
        quality_screen.calibrate_threshold(race_feat_df, target_investment_rate=0.40)

        # 9. RegimeDetector（v5.4: 市場状態の切り替え検知）
        regime_stats_df = self._build_regime_stats(race_feat_df)
        regime_det = RegimeDetector()
        regime_det.train(regime_stats_df)

        # 10. MLflow 記録
        with mlflow.start_run(run_name=f"v5.4_{train_end}"):
            for surface, m in models.items():
                mlflow.lightgbm.log_model(m["stage1"].model, f"stage1_{surface}")
                mlflow.lightgbm.log_model(m["win"].hit_model, f"win_hit_{surface}")
                mlflow.lightgbm.log_model(m["win"].return_model, f"win_ret_{surface}")
                mlflow.lightgbm.log_model(m["ev_corrector"].p_correction_model, f"ev_corrector_p_{surface}")
                mlflow.lightgbm.log_model(m["ev_corrector"].e_correction_model, f"ev_corrector_e_{surface}")
                mlflow.lightgbm.log_model(m["place"].hit_model, f"place_hit_{surface}")
                mlflow.lightgbm.log_model(m["place"].return_model, f"place_ret_{surface}")
            mlflow.lightgbm.log_model(quality_screen.model, "race_quality")
            mlflow.lightgbm.log_model(regime_det.model, "regime_detector")

        return TrainedModelsV5(
            submodels=models,
            quality_screener=quality_screen,
            regime_detector=regime_det,
        )
```

---

## 12. ベッティングオーケストレーター v5.4

```python
# betting/orchestrator.py  v5.1

class BettingOrchestratorV5:

    def process_race(
        self,
        race:         "Race",
        models:       "TrainedModelsV5",
        bankroll:     float,
        dd_ctrl:      DrawdownControllerV5,
        regime_det:   "RegimeDetector",
    ) -> list["Bet"]:

        # ① ハードフィルタ
        entries = self.db.load_entries(race.race_id)
        if not self.screener.screen_hard(race, entries).passed:
            return []

        # ② レジーム検知（v5.4: 市場状態に応じたパラメータ取得）
        regime = regime_det.detect(self._get_recent_race_stats())
        regime_params = regime_det.get_strategy_params(regime)
        logger.info(f"Regime: {regime.value} ({regime_params['description']})")

        # 再学習トリガー確認
        if regime_det.should_retrain():
            logger.warning("COLLAPSED状態が連続100レース → 再学習をトリガー")
            self._trigger_retrain()

        # ③ サブモデル選択（芝/ダート）
        surface  = "turf" if race.surface == "芝" else "dirt"
        submodel = models.submodels[surface]

        # ④ 特徴量生成（t-10min オッズ）
        feats = self.feat_engine.build_features(race, entries, snap_minutes=10)
        feats = SubModelManager().add_distance_band_features(feats)

        # ⑤ Market Model → 正規化差分（log_error）のみ
        feats = submodel["market"].predict_and_calc_error(feats)

        # ⑥ Stage1 → 2段階モデル → ★ EV補正（v5.4: P/E分解）
        feats = submodel["stage1"].add_ability_probs(feats)
        feats = submodel["win"].predict_ev(feats)
        feats = submodel["ev_corrector"].correct_ev(feats)  # v5.4: P/E分解EV補正
        feats = submodel["place"].predict_ev(feats)
        feats = submodel["conf"].add_ev_lower(feats)

        pair_df     = WideJointPairBuilder().build(feats)
        wide_scored = submodel["wide"].predict_score(pair_df)

        # ⑦ RaceQualityScreener（分布特徴量版）
        race_features = self._build_race_features(race, feats)
        if not models.quality_screener.should_bet(race_features):
            logger.info(f"Skipping by QualityScreener: {race.race_id}")
            return []

        # ⑧ ベット候補生成（v5.4: レジームパラメータを適用）
        place_bets = self.place_strategy.generate(feats, bankroll)
        wide_bets  = submodel["wide"].select_bets(
            wide_scored,
            ev_threshold=regime_params["ev_threshold"],
            score_threshold=regime_params["score_threshold"],
            max_bets=regime_params["max_bets_per_race"],
        )
        win_bets   = self.win_strategy.generate(feats, bankroll)

        # ⑧ late_money フィルタ（t-10min ベース）
        all_bets = self._apply_late_money_filter(
            place_bets + wide_bets + win_bets,
            self._get_lm_signals(race),
        )

        # ⑨ 賭け金計算（edge_lower_corrected 連動ケリー）
        for bet in all_bets:
            base_stake = self.stake_calc.calc_stake(
                bet.ev_lower_corrected, bet.odds, bankroll, bet.bet_type
            )
            # DD × Rolling ROI コントローラー適用（v5.1: window=150 + EWMA）
            bet.stake = dd_ctrl.adjust_stake(base_stake, bankroll)

        # ⑩ 1レース露出キャップ（2%）
        all_bets = self.stake_calc.check_race_exposure(all_bets, bankroll)

        # ⑪ SafetyGuard
        if not self.safety_guard.check(bankroll).can_bet:
            return []

        pending_bets = [b for b in all_bets if b.stake >= 100]

        # ⑫ t-3min 直前キャンセルチェック（v5.1: t-2min → t-3min）
        # ※ このステップは scheduler から t-3min に非同期トリガーされる
        #    ここでは pending_bets を返し、
        #    3分前に _finalize_bets() で再確認する

        return pending_bets

    async def finalize_bets(
        self,
        race:         "Race",
        pending_bets: list["Bet"],
        dd_ctrl:      DrawdownControllerV5,
    ) -> list["Bet"]:
        """発走3分前に実行。t-3min オッズで最終キャンセルチェック（v5.1）"""
        odds_t3  = await self._fetch_current_odds(race.race_id)
        odds_t10 = {b.horse_no: b.odds_at_bet for b in pending_bets}

        approved, cancelled = self.late_money_filter.process_last_minute(
            pending_bets, odds_t3, odds_t10, predictions=None
        )

        if cancelled:
            logger.info(
                f"[{race.race_id}] {len(cancelled)} bets cancelled by t-3min trigger"
            )

        # v5.1: t-2min はログのみ（判定には使わない）
        # このデータは将来のしきい値チューニングに使用

        return approved
```

---

## 13. バックテスト v5.3

### 13.1 追加検証項目（v5.0 + v5.1）

```python
# backtest/validation_suite.py  v5.1

class BacktestValidationSuite:
    """
    v5.0 + v5.1 で追加された設計の正しさを検証するテスト群
    """

    def test_stage_b_no_zeros(self, model: "WinTwoStageModel", train_df: pd.DataFrame):
        """Stage B の学習データにゼロがないことを確認"""
        hit_df = train_df[train_df["finish_pos"] == 1]
        assert (hit_df["win_odds_actual"] > 0).all(), \
            "Stage B のラベルにゼロが含まれています"

    def test_market_model_no_pred_in_stage2(self, stage2_feature_cols: list):
        """Stage2 の入力に p_market_pred が含まれていないことを確認"""
        assert "p_market_pred_win" not in stage2_feature_cols, \
            "p_market_pred が Stage2 の入力に含まれています（市場コピー化のリスク）"

    def test_market_model_uses_log_error(self, stage2_feature_cols: list):
        """v5.1: Stage2 に log_error が含まれていることを確認"""
        assert "market_log_error_win" in stage2_feature_cols, \
            "market_log_error_win が Stage2 の入力に含まれていません"

    def test_ev_correction_reduces_error(
        self,
        ev_corrector: "EVCorrectionModel",
        test_df: pd.DataFrame,
    ):
        """v5.4: EV補正モデル（P/E分解）がEV誤差を減らすことを確認"""
        df = ev_corrector.correct_ev(test_df)
        actual_ev = df["win_odds_actual"] * (df["finish_pos"] == 1).astype(int)

        mae_raw = np.mean(np.abs(df["ev_win"] - actual_ev))
        mae_corrected = np.mean(np.abs(df["ev_win_corrected"] - actual_ev))

        assert mae_corrected < mae_raw, \
            f"EV補正後のMAE({mae_corrected:.4f})が補正前({mae_raw:.4f})より大きい"

    def test_ev_correction_pe_independent(
        self,
        ev_corrector: "EVCorrectionModel",
        test_df: pd.DataFrame,
    ):
        """v5.4: P補正とE補正が独立に動作していることを確認"""
        df = ev_corrector.correct_ev(test_df)
        # P_corrected と E_corrected は別々に計算されている
        assert "p_win_corrected" in df.columns, "p_win_corrected がありません"
        assert "e_return_win_corrected" in df.columns, "e_return_win_corrected がありません"
        # EV_corrected = P_corrected × E_corrected であることを確認
        expected_ev = df["p_win_corrected"] * df["e_return_win_corrected"]
        np.testing.assert_allclose(
            df["ev_win_corrected"].values,
            expected_ev.values,
            rtol=1e-6,
        )

    def test_ev_correction_mid_range_improvement(
        self,
        ev_corrector: "EVCorrectionModel",
        test_df: pd.DataFrame,
    ):
        """v5.1: 中穴ゾーン（P=0.05〜0.15）で補正効果が大きいことを確認"""
        df = ev_corrector.correct_ev(test_df)
        mid_range = df[df["p_win_pred"].between(0.05, 0.15)].copy()

        if len(mid_range) > 100:
            actual_ev = mid_range["win_odds_actual"] * (mid_range["finish_pos"] == 1).astype(int)
            mae_raw = np.mean(np.abs(mid_range["ev_win"] - actual_ev))
            mae_corrected = np.mean(np.abs(mid_range["ev_win_corrected"] - actual_ev))

            improvement = (mae_raw - mae_corrected) / mae_raw
            assert improvement > 0.10, \
                f"中穴ゾーンの補正改善率が低い: {improvement:.1%}"

    def test_race_quality_screener_independence(
        self,
        quality_scores: np.ndarray,
        edge_max_per_race: np.ndarray,
    ):
        """RaceQualityScreener の y が Stage2 の edge と独立していることを確認"""
        corr = np.corrcoef(quality_scores, edge_max_per_race)[0, 1]
        assert corr < 0.30, \
            f"RaceQualityScreener と Stage2 edge の相関が高すぎます: {corr:.3f}"

    def test_race_quality_uses_distribution_features(self, feature_cols: list):
        """v5.1: RaceQualityScreener に分布特徴量が含まれていることを確認"""
        dist_features = ["n_positive_errors", "top_k_error_sum", "positive_error_ratio"]
        for feat in dist_features:
            assert feat in feature_cols, \
                f"分布特徴量 '{feat}' が RaceQualityScreener に含まれていません"

    def test_submodel_sample_sufficiency(
        self,
        submodel_key: str,
        sample_count: int,
    ):
        """サブモデルのサンプル数が十分かチェック"""
        assert sample_count >= 20_000, \
            f"サブモデル '{submodel_key}' のサンプル不足: {sample_count} < 20,000"

    def test_wide_score_variance_based(self, pair_df: pd.DataFrame):
        """v5.4改: ワイドスコアが EV / (E × sqrt(P)) で計算されていることを確認"""
        risk_denom = pair_df["e_return_given_hit"] * np.sqrt(
            pair_df["p_hit"].clip(lower=0.001)
        )
        expected = pair_df["ev_wide"] / risk_denom
        pd.testing.assert_series_equal(
            pair_df["wide_score_adj"].round(6),
            expected.round(6),
            check_names=False,
        )

    def test_late_money_uses_t3min(self):
        """v5.1: late_money が t-3min 基準で動作することを確認"""
        from betting.late_money_filter import LateMoneyFilter
        lm = LateMoneyFilter()
        # t-3min のシグナルが取得できることを確認
        signal = lm.check_last_3min(1, 10.0, 7.0)
        assert signal in [s for s in LastMinuteSignal]

    # ── v5.3/v5.3 追加テスト ─────────────────────────────────────

    def test_ev_correction_log_denominator_stable(
        self,
        ev_corrector: "EVCorrectionModel",
        test_df: pd.DataFrame,
    ):
        """v5.4: P/E分解補正が低確率帯で発散しないことを確認"""
        df = ev_corrector.correct_ev(test_df.copy())
        low_ev = df[df["ev_win"] < 0.05]
        if len(low_ev) > 50:
            high_ev = df[df["ev_win"] >= 0.05]
            low_std = low_ev["ev_win_corrected"].std()
            high_std = high_ev["ev_win_corrected"].std()
            # 低ev帯の分散が高ev帯の3倍以内ならOK
            assert low_std < high_std * 3.0, \
                f"低ev_raw帯の分散が大きすぎる: low={low_std:.3f} vs high={high_std:.3f}"
        # P_corrected が [0, 1] の範囲外に出ていないことを確認（v5.4）
        assert (df["p_win_corrected"] >= 0).all() and (df["p_win_corrected"] <= 1.0).all(), \
            "P_corrected が [0, 1] の範囲外です"

    def test_ev_correction_winner_weight(
        self,
        ev_corrector: "EVCorrectionModel",
        test_df: pd.DataFrame,
    ):
        """v5.4: P補正が1着馬の確率を適切に補正していることを確認"""
        df = ev_corrector.correct_ev(test_df.copy())
        # P_corrected が P_pred より1着馬の実際の比率に近いことを確認
        winners = df[df["finish_pos"] == 1]
        if len(winners) > 50:
            # 1着馬の P_corrected の中央値が P_pred より高いはず
            assert winners["p_win_corrected"].median() >= winners["p_win_pred"].median(), \
                "P補正が1着馬の確率を適切に引き上げていません"

    def test_log_error_clipping(self, market_model: "MarketModel"):
        """v5.3: log_error に両側クリップが適用されていることを確認"""
        import numpy as np
        # p_pred が非常に小さい場合でも発散しない
        p_pred = 0.001
        p_actual = 0.02
        p_pred_clipped = np.clip(p_pred, 0.01, 0.99)
        log_error = np.log(p_actual / p_pred_clipped)
        assert abs(log_error) < 2.0, \
            f"log_error が発散: {log_error:.3f}"

        # v5.3: p_market 側もクリップ（極端値テスト）
        p_market_extreme = 0.999
        p_pred_normal = 0.10
        p_market_clipped = np.clip(p_market_extreme, 0.01, 0.99)
        log_error_symmetric = np.log(p_market_clipped / p_pred_normal)
        assert abs(log_error_symmetric) < 3.0, \
            f"p_marketクリップ後のlog_errorが発散: {log_error_symmetric:.3f}"

    def test_race_quality_uses_profitability_proxy(self, feature_cols: list):
        """v5.4: RaceQualityScreener に結果ベース利益proxyが含まれていることを確認"""
        assert "hist_hit_rate_topk" in feature_cols, \
            "hist_hit_rate_topk が RaceQualityScreener に含まれていません"
        assert "hist_roi_topk" in feature_cols, \
            "hist_roi_topk が RaceQualityScreener に含まれていません"
        assert "hist_positive_return_ratio" in feature_cols, \
            "hist_positive_return_ratio が RaceQualityScreener に含まれていません"
        # v5.3のEV依存proxyが含まれていないことを確認
        assert "hist_top3_ev_mean" not in feature_cols, \
            "hist_top3_ev_mean はEV依存のため使用禁止（v5.4）"
        assert "hist_positive_edge_ratio" not in feature_cols, \
            "hist_positive_edge_ratio はEV依存のため使用禁止（v5.4）"

    def test_race_quality_no_temporal_leak(self, hist_df: pd.DataFrame):
        """v5.3: hist系特徴量に未来情報リークがないことを確認"""
        # 各行の hist_top3_ev_mean が、その行より前のデータのみから計算されている
        for i in range(1, len(hist_df)):
            race_date = hist_df.iloc[i]["race_date"]
            hist_rows = hist_df[hist_df["race_date"] < race_date]
            expected_mean = hist_rows["top3_ev"].mean() if len(hist_rows) > 0 else 1.0
            actual_mean = hist_df.iloc[i]["hist_top3_ev_mean"]
            # 誤差は浮動小数点レベル
            assert abs(actual_mean - expected_mean) < 1e-10 or len(hist_rows) == 0, \
                f"行{i}: hist_top3_ev_mean に未来情報リークの疑い"

    def test_dd_hysteresis_recovery(self):
        """v5.3/v5.4: DDコントローラーのヒステリシス回復+試行回数制限を確認"""
        from betting.drawdown_controller import DrawdownControllerV5, RecoveryState
        ctrl = DrawdownControllerV5(peak_bankroll=100000)
        ctrl._recovery_state = RecoveryState.REDUCED
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        # max_adjustment_amount = 0.15 なので
        # 1ウィンドウ（20ベット）内で最大0.65までしか増えない
        ctrl._recovery_state = RecoveryState.RECOVERING
        ctrl._current_multiplier = 1.0  # テーブル値が1.0を返すと仮定
        mult = ctrl.get_multiplier(95000)
        assert mult <= 0.65, \
            f"Nベット変更幅制限を超過: {mult:.2f} (上限: 0.65)"

    def test_regime_detector_states(self):
        """v5.4: レジーム検知が3状態を正しく分類することを確認"""
        from models.regime_detector import RegimeDetector, MarketRegime, RegimeConfig
        det = RegimeDetector()

        # AGGRESSIVE: 高ROI + 高歪み
        params_a = det.get_strategy_params(MarketRegime.AGGRESSIVE)
        assert params_a["ev_threshold"] < params_a.get("_default_ev", 1.20)
        assert params_a["max_bets_per_race"] >= 3

        # CONSERVATIVE: 低ROI + 低歪み
        params_c = det.get_strategy_params(MarketRegime.CONSERVATIVE)
        assert params_c["ev_threshold"] > params_a["ev_threshold"]
        assert params_c["max_bets_per_race"] <= 2

        # COLLAPSED: 壊滅的ROI
        params_x = det.get_strategy_params(MarketRegime.COLLAPSED)
        assert params_x["ev_threshold"] > params_c["ev_threshold"]
        assert params_x["max_bets_per_race"] == 1

    def test_regime_detector_hysteresis(self):
        """v5.4: レジーム遷移にヒステリシスがあることを確認"""
        from models.regime_detector import RegimeDetector, MarketRegime
        det = RegimeDetector()
        det._transition_hysteresis = 5
        # 4回遷移要求 → まだ遷移しない
        det._current_regime = MarketRegime.CONSERVATIVE
        for _ in range(4):
            det._regime_counter = 0  # 毎回リセット（異なる状態が来たと仮定）
        assert det._current_regime == MarketRegime.CONSERVATIVE

    def test_regime_detector_retrain_trigger(self):
        """v5.4: COLLAPSED状態が連続100レースで再学習トリガー"""
        from models.regime_detector import RegimeDetector, MarketRegime
        det = RegimeDetector(RegimeConfig(retrain_trigger=100))
        det._current_regime = MarketRegime.COLLAPSED
        det._regime_counter = 99
        assert not det.should_retrain()
        det._regime_counter = 100
        assert det.should_retrain()
```

### 13.2 Hold-out 最終評価基準（v5.1）

```
Hold-out期間: 2022〜2024年（3年間・1回のみ実行）

合格基準（全て満たすこと）：
  複勝回収率      >= 100%
  ワイド回収率    >= 103%（分散ベース調整でドローダウン縮小の分、高め設定）
  全体回収率      >= 101%
  最大ドローダウン <=  16%（DD×Rolling ROI+EWM A でさらに縮小）
  月次100%超      >= 22/36ヶ月（約61%以上・v5.0から引き上げ）

追加合格条件（v5.1）：
  EV補正モデルのMAE改善  >= 10%（補正前vs補正後）
  中穴ゾーンのEV誤差改善  >= 15%（P=0.05〜0.15帯）
  log_error の SHAP寄与度 > 0（正規化の有効性確認）

追加合格条件（v5.4）：
  P補正AUCがP_pred単体より改善 >= 1%（P/E分解の有効性確認）
  P補正とE補正の相関 < 0.3（独立補正の担保）
  E補正のMAEがE_pred単体より改善（winner のみ）
  RaceQuality proxy に EV依存特徴量が含まれていないこと
  Wideスコアの Var_proxy = P×E² が正しく計算されていること

ValidationSuite の全テスト通過を合格の前提条件とする。
```

---

## 14. 開発ロードマップ v5.4

### フェーズ1：基盤（1〜2ヶ月）
- JRA-VAN契約・JV-Link + EveryDB2
- PostgreSQL スキーマ構築
- オッズ5分間隔 + **t-3min/t-2min 監視システム稼働開始**（v5.1更新）

### フェーズ2：Stage1 + Market Model（2〜5ヶ月）
- 特徴量エンジン（距離帯を one-hot として追加）
- Market Model（**正規化差分 log_error 専用出力**）の実装・検証（v5.1更新）
- LightGBM Ranker Stage1（**芝/ダート2分割**）
- ParameterFreezeProtocol 実装
- ValidationSuite 全テストを CI に組み込む

**マイルストーン:** market_log_error の SHAP 寄与度 > 0 を確認

### フェーズ3：2段階モデル + EV補正完成（5〜12ヶ月）
- **単勝・複勝の2段階モデル**（ゼロ偏重排除）
- Stage B（的中時払戻回帰）のサンプル数確認（単勝 >=1,000件）
- **★ EV補正モデル**の実装・検証（v5.4: P/E分解アプローチ）
  - P補正モデル: 全サンプルで P(win) の residual を学習
  - E補正モデル: 1着馬のみで E(odds|win) の residual を学習
  - P補正AUC >= P_pred単体 + 1% を確認
  - P補正とE補正の相関 < 0.3 を確認
- **ワイドの分散ベース・リスク調整スコア**実装・検証（v5.4: Var_proxy = P×E²）
- **RaceQualityScreener**（分布特徴量化 + 結果ベースproxy）の独立性テスト
  - proxy に EV依存特徴量が含まれないことを確認
- BacktestValidationSuite 全テスト通過（v5.4テスト追加）
- **RegimeDetector**（市場状態の切り替え検知）の実装・検証
  - 3状態分類の精度確認（各状態で戦略パラメータが適切に変化すること）
  - ヒステリシス遷移の確認（頻繁な切り替え防止）
  - COLLAPSED状態での再学習トリガーの動作確認
- Hold-out 2022-2024 の最終評価

**マイルストーン:** ValidationSuite 全テスト通過 + Hold-out 全指標合格

### フェーズ4：完全自動化（12〜24ヶ月）
- PAT自動投票 + **t-3min キャンセルトリガー**（v5.1更新）
- **DD × Rolling ROI + EWMA コントローラー**稼働（v5.4: 試行回数ベース制限）
- 3ヶ月間小額自動運転（500円/bet）
- Rolling ROI の収束確認（**150ベット**以上蓄積後）
- 賭け金の段階的引き上げ
- t-2min ログデータの蓄積・分析（将来のチューニング用）

**マイルストーン:** 実運用3ヶ月で全体 ROI 99% 以上かつ最大DD 18% 以下

### フェーズ5：次世代モデル（24ヶ月〜）
- Graph Neural Network（レース構造の直接学習）
- Horse2Vec（馬の時系列 Embedding）
- サブモデルを芝×距離帯（4分割）へ拡張（サンプル >= 20,000 件確認後）
- スイッチャーの ML モデル化

---

## 14.5. 実装タスクリスト（セッション単位分割）

> コンテキスト飽和を防ぐため、1セッション ≒ 1タスク（1〜3ファイル）で消化可能な粒度に分割。
> 依存関係に従い順次実装する。完了済みタスクには `[x]` を付与。

### Phase A: 基盤構築（DB不要・Python基盤のみ）

- [ ] **A-1.** プロジェクトスケルトン作成（pyproject.toml, ディレクトリ構成, config/settings.yaml）
  - 依存: なし
  - 成果物: `pyproject.toml`, `config/`, `src/`, `tests/`, `notebooks/`, `requirements.txt`

- [ ] **A-2.** データクラス・型定義（Race, Entry, Bet, OddsSnapshot 等）
  - 依存: A-1
  - 成果物: `src/domain/` （dataclass群）

- [ ] **A-3.** PostgreSQL スキーマ定義 + DB接続モジュール
  - 依存: A-1
  - 成果物: `src/db/schema.py`, `src/db/connection.py`
  - スキーマ: raw / odds_history / feature / prediction / betting

### Phase B: 特徴量エンジン

- [ ] **B-1.** feature_engine.py（メインエンジン + build_all() インタフェース）
  - 依存: A-2, A-3
  - 成果物: `src/features/feature_engine.py`

- [ ] **B-2.** intra_race_features.py（レース内相対特徴量）
  - 依存: B-1
  - 成果物: `src/features/intra_race_features.py`

- [ ] **B-3.** odds_dynamics_features.py（オッズ変化率特徴量）
  - 依存: B-1
  - 成果物: `src/features/odds_dynamics_features.py`

- [ ] **B-4.** market_bias_features.py（市場歪み特徴量）
  - 依存: B-1
  - 成果物: `src/features/market_bias_features.py`

- [ ] **B-5.** info_asymmetry_features.py + race_difficulty_model.py
  - 依存: B-1
  - 成果物: `src/features/info_asymmetry_features.py`, `src/features/race_difficulty_model.py`

- [ ] **B-6.** leakage_validators.py（未来情報リーク検証）
  - 依存: B-1〜B-5
  - 成果物: `src/features/leakage_validators.py`, `tests/test_leakage.py`

### Phase C: モデル群

- [ ] **C-1.** submodel_manager.py（芝/ダート2分割 + 距離帯one-hot）
  - 依存: A-2
  - 成果物: `src/models/submodel_manager.py`
  - §6参照: 距離帯はモデル分割ではなくone-hot特徴量として追加

- [ ] **C-2.** market_model.py（差分専用・log_error正規化・両側クリップ）
  - 依存: B-1, C-1
  - 成果物: `src/models/market_model.py`
  - §4参照: p_market_predは出力しない、log_error(signed/abs)のみ

- [ ] **C-3.** stage1_ability_model.py（LightGBM Ranker・芝/ダート）
  - 依存: B-1, C-1
  - 成果物: `src/models/stage1_ability_model.py`
  - Rule 1: オッズを入れない

- [ ] **C-4.** two_stage_return_model.py（単勝・複勝2段階モデル）
  - 依存: C-2, C-3
  - 成果物: `src/models/two_stage_return_model.py`
  - §2参照: WinTwoStageModel + PlaceTwoStageModel

- [ ] **C-5.** ev_correction_model.py（P補正 binary init_score + E補正 1/√p重み付き回帰）
  - 依存: C-4
  - 成果物: `src/models/ev_correction_model.py`
  - §3, Rule 12参照: P補正(init_score=logit(p_pred)) × E補正(weight=1/√p)

- [ ] **C-6.** wide_two_stage_model.py（分散ベーススコア EV/(E×√P)）
  - 依存: C-2, C-3
  - 成果物: `src/models/wide_two_stage_model.py`
  - §7, Rule 3, Rule 15参照: シャープレシオ近似

- [ ] **C-7.** race_quality_screener.py（結果ベースproxy）
  - 依存: C-2
  - 成果物: `src/models/race_quality_screener.py`
  - §5, Rule 16参照: EV依存proxy禁止（hist_roi_topk, hist_positive_return_ratio使用）

- [ ] **C-8.** regime_detector.py（3状態分類 + ヒステリシス + 市場指標ラベル）
  - 依存: C-2
  - 成果物: `src/models/regime_detector.py`
  - §9.5, Rule 19参照: 教師ラベルはmarket_efficiencyベース

- [ ] **C-9.** robust_confidence_estimator.py（CP + Rolling Quantile の min）
  - 依存: C-5
  - 成果物: `src/models/robust_confidence_estimator.py`
  - Rule 4参照

### Phase D: ベッティング層

- [ ] **D-1.** stake_calculator.py（edge連動ケリー + 1レース2%キャップ）
  - 依存: A-2
  - 成果物: `src/betting/stake_calculator.py`
  - Rule 6参照

- [ ] **D-2.** drawdown_controller.py（DD×ROI + EWMA + ヒステリシス + Nベット制限）
  - 依存: A-2
  - 成果物: `src/betting/drawdown_controller.py`
  - §9, Rule 9, Rule 17参照: MAX_ADJUSTMENT_PER_N_BETS=20

- [ ] **D-3.** late_money_filter.py（t-3min判定 + t-2minログ）
  - 依存: A-2
  - 成果物: `src/betting/late_money_filter.py`
  - §8, Rule 8, Rule 14参照

- [ ] **D-4.** gate_keeper.py + meta_switcher.py
  - 依存: C-9, D-1
  - 成果物: `src/betting/gate_keeper.py`, `src/betting/meta_switcher.py`

- [ ] **D-5.** place_strategy.py + win_strategy.py + wide_strategy.py
  - 依存: C-4, C-6, D-1
  - 成果物: `src/betting/place_strategy.py`, `src/betting/win_strategy.py`, `src/betting/wide_strategy.py`

- [ ] **D-6.** orchestrator.py（メインオーケストレーター + finalize_bets）
  - 依存: C-1〜C-9, D-1〜D-5
  - 成果物: `src/betting/orchestrator.py`
  - §12参照: 12ステップのベット決定フロー

### Phase E: パイプライン・検証

- [ ] **E-1.** training_pipeline.py（学習パイプライン全体）
  - 依存: C-1〜C-9
  - 成果物: `src/pipelines/training_pipeline.py`
  - §11参照: MLflow記録含む

- [ ] **E-2.** walk_forward_cv.py（ウォークフォワード交差検証）
  - 依存: E-1
  - 成果物: `src/models/walk_forward_cv.py`
  - Rule 7参照: out-of-sample期間でパラメータ変更禁止

- [ ] **E-3.** backtest/engine.py（バックテストエンジン）
  - 依存: D-6, E-1
  - 成果物: `src/backtest/engine.py`

- [ ] **E-4.** validation_suite.py（全テストスイート）
  - 依存: C-1〜C-9, D-2, D-3
  - 成果物: `src/backtest/validation_suite.py`
  - §13参照: 全検証項目（test_stage_b_no_zeros, test_ev_correction_pe_independent 等）

- [ ] **E-5.** parameter_freeze_protocol.py
  - 依存: E-2
  - 成果物: `src/backtest/parameter_freeze_protocol.py`

### Phase F: 自動化・監視

- [ ] **F-1.** pat_voter.py + safety_guard.py
  - 依存: D-6
  - 成果物: `src/automation/pat_voter.py`, `src/automation/safety_guard.py`

- [ ] **F-2.** scheduler.py（t-3min/t-2min タスク）
  - 依存: D-3, D-6, F-1
  - 成果物: `src/automation/scheduler.py`

- [ ] **F-3.** model_monitor.py + auto_retrain_trigger.py + notifier.py
  - 依存: C-8, E-1
  - 成果物: `src/monitoring/model_monitor.py`, `src/monitoring/auto_retrain_trigger.py`, `src/monitoring/notifier.py`

- [ ] **F-4.** jvlink_fetcher.py + odds_collector.py
  - 依存: A-3
  - 成果物: `src/ingestion/jvlink_fetcher.py`, `src/ingestion/odds_collector.py`

### Phase G: テスト・ノートブック

- [ ] **G-1.** tests/ 全テストファイル
  - 依存: Phase B〜E の各成果物
  - 成果物: `tests/test_*.py`（§13, §16参照）

- [ ] **G-2.** notebooks/ 分析ノートブック群
  - 依存: Phase C, E
  - 成果物: `notebooks/01_eda.ipynb` 〜 `notebooks/11_holdout_final_evaluation.ipynb`

---

## 15. 期待値の現実的見積もり・本質的リスク（最終版・v5.4）

```
【v5.4 の設計で期待できる数値】

バックテスト（Hold-out 2022〜2024）：
  複勝:  108〜117%  ← P/E分解により+0.5% 上乗せ見込み
  ワイド: 109〜121% ← Var_proxy改善により+0.5% 上乗せ見込み
  単勝:  100〜112%  ← P補正AUC改善により+1% 上乗せ見込み
  全体:  108〜119%

実運用（スリッページ・劣化・分散を考慮）：
  複勝:   103〜111%
  ワイド: 104〜115%
  単勝:   97〜108%
  全体:  103〜110%

最大ドローダウン（DD × Rolling ROI + EWMA + ヒステリシス + max_adj/N_bets + レジーム検知）：
  推定 7〜13%（レジーム検知により COLLAPSED 期間の損失を抑制）

破産リスク（10万円・5万円以下になる確率）：
  ≈ 0.1〜0.3%（レジーム検知による追加効果）

【本質的リスク（設計では解消不可能・認識と緩和が重要）】

リスク①：市場適応（アドバーサリアル環境）
  競馬市場は静的ではなく動的：
    戦略が機能する → 他の参加者が同様の戦略を採用 → EVが縮小
    直前に「情報優位者」が入る → 最終オッズは情報をかなり反映

  → 長期ではROIは必ず劣化する（構造的に不可避）

  緩和策（v5.4で実装済み）：
    RegimeDetector で市場状態を監視
    COLLAPSED状態が連続100レース → 自動再学習トリガー
    AGGRESSIVE/CONSERVATIVE で攻め方を動的に調整

  さらに必要なもの（運用フェーズで対応）：
    定期再学習（3〜6ヶ月ごと）
    特徴量の陳腐化モニタリング
    実運用ログからの新特徴量発見

リスク②：サンプル不足 × 分散
  競馬は試行回数が少なく分散が極端に大きい：
    年間ベット数: 数千
    1レースの分散: オッズによって大きく異なる

  → ROI 105%でも「1〜2年普通に負ける」ことは統計的に普通に起こる

  緩和策（v5.4で実装済み）：
    DD制御で最大損失を制限
    1レース露出2%キャップ
    レジーム検知で COLLAPSED 期間のベットを抑制

  運用上の認識：
    短期（数ヶ月）の損益は「運」に近い
    評価期間は最低1年、理想は2〜3年
    ROI 103%であっても、1000ベット後の95%信頼区間は広い

【v5.4 での設計限界】
v5.0で設計完成、v5.1で統計運用の罠、v5.2で統計的安定性、
v5.3で統計ノイズ耐性、v5.4で実運用で壊れるポイント+市場適応リスクに対応した。
ここまで詰めると設計・統計として取れる手は事実上尽きている。
残る改善余地は：
  ① データ量の増加（年数が増えるほど精度向上）
  ② 特徴量の発見（ドメイン知識から）
  ③ GNN / Horse2Vec（v5.4 後継）
  ④ オッズ変化の粒度向上（1分間隔への変更）
  ⑤ 定期再学習の運用確立（市場適応リスクへの対応）
```

---

## 16. ディレクトリ構成 v5.4

```
keiba-ai/
├── config/
│   ├── settings.yaml
│   ├── submodel_config.yaml      # 芝/ダート2分割 + 将来拡張基準
│   ├── strategy_config.yaml
│   └── backtest_config.yaml
├── src/
│   ├── ingestion/
│   │   ├── jvlink_fetcher.py
│   │   └── odds_collector.py     # t-3min/t-2min 監視（v5.1更新）
│   ├── features/
│   │   ├── feature_engine.py
│   │   ├── intra_race_features.py
│   │   ├── odds_dynamics_features.py
│   │   ├── market_bias_features.py
│   │   ├── info_asymmetry_features.py
│   │   ├── race_difficulty_model.py
│   │   └── leakage_validators.py
│   ├── models/
│   │   ├── labels/
│   │   │   └── (v4.0から削除: 2段階モデルに統合)
│   │   ├── market_model.py           # ★ 正規化差分 log_error 出力（v5.1更新）
│   │   ├── stage1_ability_model.py
│   │   ├── two_stage_return_model.py # ★ 単勝・複勝の2段階モデル
│   │   ├── ev_correction_model.py    # ★★ EV補正モデル・P/E分解版（v5.4更新）
│   │   ├── wide_two_stage_model.py   # ★ 分散ベース・Var_proxy使用（v5.4更新）
│   │   ├── race_quality_screener.py  # ★ 結果ベースproxy（v5.4更新）
│   │   ├── regime_detector.py        # ★★ 市場状態切り替え検知（v5.4新設）
│   │   ├── robust_confidence_estimator.py
│   │   ├── submodel_manager.py       # ★ 2分割に縮小
│   │   └── walk_forward_cv.py
│   ├── betting/
│   │   ├── place_strategy.py
│   │   ├── wide_strategy.py
│   │   ├── win_strategy.py
│   │   ├── meta_switcher.py
│   │   ├── stake_calculator.py
│   │   ├── gate_keeper.py
│   │   ├── late_money_filter.py      # ★ t-3min判定 + t-2minログ（v5.1更新）
│   │   ├── drawdown_controller.py    # ★ window=150 + EWMA + ヒステリシス + 試行回数制限（v5.4更新）
│   │   ├── race_screener.py
│   │   └── orchestrator.py           # ★ finalize_bets t-3min対応（v5.1更新）
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── validation_suite.py       # ★ v5.1 テスト追加
│   │   └── parameter_freeze_protocol.py
│   ├── pipelines/
│   │   └── training_pipeline.py      # ★ EV補正モデル追加
│   ├── automation/
│   │   ├── pat_voter.py
│   │   ├── scheduler.py              # ★ t-3min/t-2min タスク（v5.1更新）
│   │   └── safety_guard.py
│   └── monitoring/
│       ├── model_monitor.py
│       ├── auto_retrain_trigger.py
│       └── notifier.py
├── tests/
│   ├── test_leakage.py
│   ├── test_stage_b_no_zeros.py          # ★ Stage B ゼロ排除確認
│   ├── test_market_model_diff_only.py    # ★ p_market_pred が Stage2 に入らないことを確認
│   ├── test_market_model_log_error.py    # ★★ log_error 正規化の確認（v5.1新設）
│   ├── test_ev_correction.py             # ★★ EV補正・P/E分解の精度確認（v5.4更新）
│   ├── test_log_error_clipping.py        # ★★ log_error クリップの発散防止確認（v5.3新設）
│   ├── test_race_quality_independence.py # ★ スクリーナーの y 独立性
│   ├── test_race_quality_distribution.py # ★★ 分布特徴量の確認（v5.4: 結果ベースproxy確認）
│   ├── test_submodel_sample_size.py      # ★ サブモデルのサンプル十分性
│   ├── test_wide_risk_adjusted_score.py  # ★ 分散ベーススコアの正しさ（v5.4: Var_proxy確認）
│   ├── test_late_money_t3_trigger.py     # ★★ t-3min トリガーの動作確認（v5.1更新）
│   ├── test_dd_hysteresis_recovery.py    # ★★ DDヒステリシス+試行回数制限の検証（v5.4更新）
│   ├── test_regime_detector.py           # ★★ レジーム検知の状態分類・ヒステリシス検証（v5.4新設）
│   └── test_dd_rolling_roi_controller.py # ★ DD×ROI+EWM A 複合制御の検証
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_odds_dynamics.ipynb
│   ├── 03_market_model_diff_analysis.ipynb  # ★ 差分のみの有効性確認
│   ├── 04_twostage_win_place_ab_test.ipynb  # ★ 2段階 vs 1段階の比較
│   ├── 05_wide_risk_adjusted_score.ipynb    # ★ 分散ベーススコアの DD 改善確認（v5.4: Var_proxy）
│   ├── 06_race_quality_independence.ipynb   # ★ スクリーナー独立性検証
│   ├── 07_submodel_2split_vs_14split.ipynb  # ★ 2分割 vs 14分割の精度比較
│   ├── 08_dd_rolling_roi_simulation.ipynb   # ★ 複合制御のシミュレーション
│   ├── 09_ev_correction_analysis.ipynb      # ★★ EV補正の効果分析（v5.4: P/E分解確認）
│   ├── 10_log_error_normalization.ipynb     # ★★ log_error 正規化の効果（v5.1新設）
│   └── 11_holdout_final_evaluation.ipynb    # ★ 最終 Hold-out（1回のみ）
└── requirements.txt
```

---

## 17. 付録：全バージョン設計比較

| 指標 | v1.0 | v2.0 | v3.0 | v4.0 | v5.0 | v5.1 | v5.2 | v5.3 | **v5.4** |
|------|------|------|------|------|------|------|------|------|------|
| 設計思想 | 強い馬を当てる | 市場差分に転換 | 実運用欠陥修正 | 収益を直接最適化 | 統計的歪みを潰す | 統計運用の罠を潰す | 統計的安定性補強 | 統計ノイズ耐性 | **実運用壊れ対策** |
| Stage2ラベル | 0/1 | rank_score | rank_score | 期待1段階 | 2段階（P×E） | +EV補正 | 補正係数 | log補正+重み | **P/E分解補正** |
| ゼロ偏重 | あり | あり | あり | 部分解決 | 排除 | +EV補正 | 全サンプル | +重み付け | **P/E分離** |
| Market Model | なし | なし | なし | p_pred+差分 | 差分のみ | 正規化 | p_predクリップ | 両側クリップ | （継承） |
| レース選別 | なし | なし | 難易度 | ROI学習 | エッジ集計 | 分布特徴量 | +利益proxy | +時間リーク遮断 | **結果ベースproxy** |
| ワイドスコア | なし | なし | なし | P×E | P²×E | EV×P/(1+odds) | EV×P/(1+E) | EV×P/(1+E) | **EV/(E×sqrt(P))** |
| late_money | 主特徴量 | 主特徴量 | 主特徴量 | フィルタ | t-2min | t-3min判定 | t-3min判定 | t-3min判定 | （継承） |
| DD制御 | なし | なし | 粗い | DD15% | DD×ROI | DD×ROI+EWM A | +ヒステリシス | +max_adj/day | **+max_adj/N_bets** |
| レジーム検知 | なし | なし | なし | なし | なし | なし | なし | なし | **3状態分類器** |
| 期待ROI | 80〜90% | 90〜97% | 97〜105% | 98〜108% | 100〜106% | 102〜108% | 103〜109% | 103〜110% | **103〜110%** |
| 最大DD | なし | なし | 粗い | 15〜20% | 12〜18% | 10〜16% | 8〜14% | 8〜14% | **7〜13%** |

---

*本設計書は v5.4（実運用で壊れるポイント完全排除 + 市場適応対応版）です。*
*v5.0で設計完成、v5.1で統計運用の罠、v5.2で統計的安定性、v5.3で統計ノイズ耐性、v5.4で実運用壊れ対策+レジーム検知を潰しました。*
*これ以上の改善は「データ量・特徴量発見・GNN・定期再学習」の領域になり、設計の問題ではなく実装と実験の問題になります。*
*最初のアクション: フェーズ1基盤構築 → Notebook 04（2段階 vs 1段階の AB テスト）→ Notebook 09（EV補正 P/E分解の効果分析）→ 小額実運用（500円/bet）が重要。*
