# Phase 6: Odds Deviation EV - Research

**Researched:** 2026-05-03
**Domain:** オッズ乖離EV特徴量 + Conformal予測区間拡張 + パイプライン統合
**Confidence:** HIGH

## Summary

Phase 6 は3つの要件(ODDS-01, ODDS-02, ODDS-03)を1つのプランで実装する。中心となる技術課題は、(1) モデル予測確率と市場オッズの乖離を3信号(odds_to_ability_ratio, deviation_rank, deviation_zscore)として特徴量化しStage2に統合すること、(2) 既存`RobustConfidenceEstimator`を下限推定から上下区間推定に拡張し、EV区間を2段階信頼水準(80%/90%)で計算すること、(3) EV区間からconformal_confidence_scoreを合成しWinSelectionGateのスコアリングに統合することである。

コードベースの徹底調査により、全ての統合ポイントが明確に特定された。MarketModelの`predict_and_calc_error()`、FeatureEngineの`compute_market_bias()`、TrainingPipelineの`_train_submodel()`、RacePredictorの`predict()`、WinSelectionGateの`score()`が主要な変更対象である。既存パターン(groupby race_idによるランク計算、NaN安全なSeries生成、LightGBMネイティブNaN処理)が確立されており、新規コードもこれに従う。

**Primary recommendation:** 既存パターンを踏襲したstandalone関数`compute_odds_deviation_features()`を新規ファイルに配置し、TrainingPipelineとRacePredictorのAbilityModel直後に呼び出す。`RobustConfidenceEstimator`は`predict_interval()`を追加して内部で再利用し、`predict_lower_bound()`は`predict_interval()`を呼ぶ形にリファクタする。conformal_confidence_scoreはWinSelectionGateのquantile-bin scoring systemに第4次元として統合する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `odds_to_ability_ratio`(既存) + `deviation_rank`(レース内乖離順位) + `deviation_zscore`(レース内標準化乖離)の3信号。絶対乖離フラグ不要
- **D-02:** `compute_odds_deviation_features(df)`をstandalone関数として作成。MarketModelの`predict_and_calc_error()`パターンに倣う
- **D-03:** `deviation_rank`と`deviation_zscore`はレース内相対評価(`race_id`でgroupby)
- **D-04:** 3信号をWinTwoStageModelのFEATURE_COLSに追加
- **D-05:** 既存`RobustConfidenceEstimator`を拡張。`predict_lower_bound()`を`predict_interval()`に拡張し、EV区間(下限+上限)を計算
- **D-06:** EV下限(profitability filter)と区間幅(confidence filter)の両方を合成。`conformal_confidence_score`としてWinSelectionGateの`score()`に統合
- **D-07:** 90%と80%の2段階信頼水準。80%区間(高信頼)と90%区間(最低基準)で段階的信頼性評価
- **D-08:** conformal_confidence_scoreはEV下限と区間幅の合成指標。WinSelectionGateのquantile-bin scoring systemに新次元として追加
- **D-09:** 三層テスト戦略: (1) 単体テスト、(2) 統合テスト、(3) 数値的一貫性チェック
- **D-10:** 新規テストファイル`tests/test_odds_deviation.py`
- **D-11:** RobustConfidenceEstimator拡張テストは既存テストファイルに追加
- **D-12:** パイプライン統合テストは既存`tests/test_race_predictor.py`に追加
- **D-13:** 数値的一貫性チェック: 確率正規化(sum=1.0)、EV区間順序性(lower < point < upper)、NaN率検証

### Claude's Discretion
- `compute_odds_deviation_features()`の具体的な配置先ファイル(既存モジュール or 新規ユーティリティ)
- conformal_confidence_scoreの合成式の詳細(EV下限と区間幅の重み付け)
- WinSelectionGateへのconformal_confidence_score統合方法(既存quantile-binに追加 vs 新スコアリング次元)
- deviation_rank/z-scoreのNaN処理(デフォルトNaN、LightGBMネイティブ処理 — Phase 5 D-23と同一方針)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODDS-01 | p_market/p_ability比率をStage2特徴量カラムとして追加し、ROIに直結するエッジ信号をモデルに学習させる | 既存`odds_to_ability_ratio`(line 91)に`deviation_rank`と`deviation_zscore`を追加。`compute_odds_deviation_features()`関数で3信号生成。WinTwoStageModel.FEATURE_COLSに2列追加 |
| ODDS-02 | スタッキング出力がBenterGate→WinSelectionGateに正しく流れることを検証し、EV計算パイプラインの整合性を確保する | 三層テスト戦略(D-09)。RacePredictor.predict()の推論フロー(step 4a-7)の呼び出し順序検証。数値的一貫性チェックで確率正規化・EV区間順序性確認 |
| ODDS-03 | Conformal予測区間をEV区間に変換し、エッジの信頼性に基づいてベット選択を最適化する | RobustConfidenceEstimatorに`predict_interval()`追加。80%/90%の2段階信頼水準(D-07)。conformal_confidence_scoreをWinSelectionGateのquantile-bin scoringに第4次元として統合(D-08) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 乖離特徴量計算 | API / Backend (TrainingPipeline/RacePredictor) | -- | モデル出力(p_ability_win, p_market_win_adj)に依存するため、FeatureEngine(raw data)ではなくパイプライン層で計算 |
| Conformal EV区間 | API / Backend (RobustConfidenceEstimator) | -- | 既存信頼区間推定器の拡張。キャリブレーション・推論ともにバックエンド層 |
| conformal_confidence_score | API / Backend (WinSelectionGate) | -- | Gateモデルのスコアリングシステムに統合。学習済みパラメータに基づく推論 |
| パイプライン統合 | API / Backend (TrainingPipeline, RacePredictor) | -- | 学習フローと推論フローの両方に新ステップを挿入 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (既存) | groupby rank/zscore計算 | プロジェクト依存関係に既に含まれる [VERIFIED: コードベース確認] |
| pandas | (既存) | DataFrame操作、groupby transform | プロジェクト依存関係に既に含まれる [VERIFIED: コードベース確認] |
| LightGBM | (既存) | NaNネイティブ処理、Stage2モデル | deviation特徴量のNaNを自動処理 [VERIFIED: two_stage_return_model.py] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | (既存) | Conformal区間計算のstats機能 | 必要に応じて。既存コードはnumpyのみ使用 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手動conformal interval | nonconformist/crepesライブラリ | 既存RobustConfidenceEstimatorのアーキテクチャ(非適合スコア=abs(actual-predicted))が確立済み。外部ライブラリ導入は過剰 |

**Installation:** 追加インストール不要。全て既存依存関係で完結する。

## Architecture Patterns

### System Architecture Diagram

```
[Raw Data: races, entries, odds]
        |
        v
[FeatureEngine.build_all()] -- raw特徴量生成
        |
        v
[TrainingPipeline._train_submodel()] または [RacePredictor.predict()]
        |
        +---> [MarketModel.predict_and_calc_error()]  --> signed_log_error_win, abs_log_error_win
        |
        +---> [AbilityModel.train_oof() / add_ability_probs()]  --> p_ability_win
        |
        +---> [NEW: compute_odds_deviation_features()]  <-- D-02: ここに追加
        |         入力: p_market_win_adj, p_ability_win, race_id
        |         出力: deviation_rank, deviation_zscore
        |
        +---> [PlaceAbilityModel]  --> p_ability_place
        |
        +---> [WinTwoStageModel]  --> EV_win (deviation特徴量を含むFEATURE_COLSで学習/推論)
        |
        +---> ... (EV補正、Place、Benter、WinSelectionGate)
        |
        +---> [RobustConfidenceEstimator.predict_interval()]  <-- D-05: 拡張
        |         入力: ev_win_corrected, nonconformity scores
        |         出力: EV_lower_win, EV_upper_win (80%/90%の2水準)
        |         出力: conformal_confidence_score (D-08)
        |
        +---> [WinSelectionGate.score()]  <-- D-08: conformal_confidence_scoreを統合
                  quantile-bin scoring (prob, edge, odds + confidence)
```

### Recommended Project Structure
```
src/
├── features/
│   └── odds_deviation_features.py    # NEW: compute_odds_deviation_features()
├── models/
│   ├── two_stage_return_model.py     # MODIFY: FEATURE_COLSに2列追加
│   ├── robust_confidence_estimator.py # MODIFY: predict_interval()追加
│   ├── win_selection_gate.py         # MODIFY: conformal_confidence_score統合
│   └── ...
├── pipelines/
│   └── training_pipeline.py          # MODIFY: _train_submodel()に呼び出し追加
├── backtest/
│   └── race_predictor.py             # MODIFY: predict()に呼び出し追加
tests/
├── test_odds_deviation.py            # NEW: 乖離特徴量+EV区間テスト (D-10)
├── test_robust_confidence_estimator.py # MODIFY: predict_intervalテスト追加 (D-11)
└── test_race_predictor.py            # MODIFY: パイプライン統合テスト追加 (D-12)
```

### Pattern 1: Post-Model Feature Computation
**What:** モデル出力に依存する特徴量の追加パターン
**When to use:** FeatureEngine(=raw data特徴量)では計算不可能な、モデル予測値が必要な特徴量
**Example:**
```python
# 既存パターン: odds_to_ability_ratio (training_pipeline.py:430-433)
if "p_market_win_adj" in df_oof.columns and "p_ability_win" in df_oof.columns:
    p_market = df_oof["p_market_win_adj"].clip(lower=1e-6)
    p_ability = df_oof["p_ability_win"].clip(lower=1e-6)
    df_oof["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

# 新規パターン: deviation_rank, deviation_zscore
def compute_odds_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ratio = df.get("odds_to_ability_ratio", pd.Series(np.nan, index=df.index, dtype=float))
    # レース内ランク (ascending: ratio小=過大評価, 大=過小評価)
    df["deviation_rank"] = (
        ratio.groupby(df["race_id"]).rank(method="first", ascending=False)
    )
    # レース内z-score標準化
    race_mean = ratio.groupby(df["race_id"]).transform("mean")
    race_std = ratio.groupby(df["race_id"]).transform("std")
    df["deviation_zscore"] = (ratio - race_mean) / race_std.replace(0, np.nan)
    return df
```
[VERIFIED: training_pipeline.py lines 430-433, market_model.py lines 126-131]

### Pattern 2: Conformal Prediction Interval Extension
**What:** 非適合スコアに基づくEV区間の上下限計算
**When to use:** 既存の下限推定を上下区間に拡張する場合
**Example:**
```python
# 既存: predict_lower_bound() -- RobustConfidenceEstimator
# cp_lower_win = win_ev - cp_quantile_per_row
# NEW: predict_interval() は上下両方を計算
def predict_interval(self, win_df, place_df, alphas=(0.1, 0.2)):
    """alphas=(0.1, 0.2) → 90%区間と80%区間"""
    for alpha in alphas:
        cp_quantile = np.quantile(residuals, 1 - alpha)
        lower = win_ev - cp_quantile
        upper = win_ev + cp_quantile
    # conformal_confidence_score = EV下限重視 + 区間幅ペナルティ
    # score = w1 * EV_lower_80 + w2 * (1 / interval_width_90)
```
[VERIFIED: robust_confidence_estimator.py lines 96-148]

### Pattern 3: Quantile-Bin Scoring Integration
**What:** WinSelectionGateのquantile-bin scoring systemに新次元を追加
**When to use:** 新しいスコアリング信号をGateに統合する場合
**Example:**
```python
# 既存: _build_score_tables() -- prob_edges, edge_edges, odds_edgesの3次元
# D-08: conformal_confidence_scoreを第4次元として追加
confidence_edges = _quantile_edges(work["conformal_confidence_score"], self.n_bins)
work["_confidence_bin"] = _bucketize(work["conformal_confidence_score"], confidence_edges)
# combo_scores のキーが (prob_bin, edge_bin, odds_bin, confidence_bin) に拡張
```
[VERIFIED: win_selection_gate.py lines 239-325]

### Anti-Patterns to Avoid
- **FeatureEngineにdeviation計算を追加:** FeatureEngineはraw data特徴量のみ。p_ability_win, p_market_win_adjはモデル出力のためFeatureEngineでは利用不可
- **predict_lower_bound()の残存:** D-05は拡張であり、predict_lower_bound()はpredict_interval()のラッパーにリファクタすべき。並行して両メソッドを維持するとキャリブレーションデータの不整合が生じる
- **conformal_confidence_scoreをWinSelectionGate外でハードコード:** スコアの重み付けはGateのwalk-forward OOF学習で自動最適化すべき。手動ハイパーパラメータにすべきではない
- **deviation特徴量のクリッピングなし:** odds_to_ability_ratioは0.1-10.0にクリップ済み(line 433)。deviation_zscoreも適切な範囲にクリップすべき(外れ値の影響を抑制)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| レース内ランク計算 | カスタムループ | `pd.Series.groupby().rank(method="first")` | 既存パターン(market_error_rank_in_race, line 127)。ベクトル化済み |
| レース内z-score | 手動標準化ループ | `groupby().transform("mean"/"std")` | pandas groupby transformが最適化済み |
| Conformal区間 | 新しいクラス | 既存`RobustConfidenceEstimator`拡張 | キャリブレーション済みのnonconformity scores, race-condition-dependent quantilesを再利用 |
| quantile計算 | 手動パーセンタイル | `np.quantile()` | 既存パターン(line 67)。分散フリー保証 |

**Key insight:** 全ての「手作り」リスクは既存パターンの踏襲で回避できる。新しいアルゴリズムの導入はなく、既存アーキテクチャの拡張のみ。

## Common Pitfalls

### Pitfall 1: TrainingPipelineとRacePredictorでの計算タイミング不整合
**What goes wrong:** TrainingPipelineではodds_to_ability_ratio計算後にdeviation特徴量を追加するが、RacePredictorではWinTwoStageModel._prepare_features()内でodds_to_ability_ratioを遅延計算するため、deviation特徴量の計算タイミングがずれる
**Why it happens:** 推論パスと学習パスで特徴量の追加タイミングが異なる。学習はodds_to_ability_ratioが AbilityModel直後に計算されるが、推論は_prepare_features()内で遅延計算
**How to avoid:** RacePredictor.predict()ではWinTwoStageModel.predict_ev()の呼び出し前にcompute_odds_deviation_features()を呼び出す。_prepare_features()内のodds_to_ability_ratio遅延計算は維持(互換性)。deviation特徴量は_prepare_features()より前に計算されるため、FEATURE_COLSに含まれていれば自動的に使用される
**Warning signs:** テストでdeviation_rank, deviation_zscoreがNaNのままになる

### Pitfall 2: predict_lower_bound()呼び出し元の破壊的変更
**What goes wrong:** predict_lower_bound()のシグネチャや戻り値を変更すると、TrainingPipeline(placeselection_gate訓練, win_selection_gate訓練)とRacePredictor(step 7)の両方でエラー
**Why it happens:** predict_lower_bound()は3箇所(training_pipeline.py:773, 783, race_predictor.py:147)で呼び出されている
**How to avoid:** predict_lower_bound()のシグネチャは変更しない。内部でpredict_interval()を呼んで下限のみ返すラッパーとする。新規メソッドpredict_interval()は追加のみ
**Warning signs:** 既存テストの失敗、特にtest_robust_confidence_estimator.py

### Pitfall 3: WinSelectionGate scoringの次元拡張による過学習
**What goes wrong:** conformal_confidence_scoreをquantile-binに追加すると、4次元の組み合わせ爆発で各ビンのサンプル数が不足する
**Why it happens:** 既存3次元(prob, edge, odds) x 6 bins = 216組み合わせ。4次元で1296組み合わせに膨張
**How to avoid:** n_binsをconformal次元は3-4に削減。またはpair_scores(2次元組み合わせ)にconformalペアを追加し、combo_scoresは3次元のままにする。D-08の「新次元として追加」は、必ずしもcomboの全次元追加を意味しない
**Warning signs:** walk-forward foldでROIが不安定、ビンのサンプル数 < prior_weight

### Pitfall 4: PlaceTwoStageModelへの意図しない影響
**What goes wrong:** deviation特徴量をWinTwoStageModelのFEATURE_COLSに追加したが、PlaceTwoStageModelのRETURN_FEATURE_COLSにもodds_to_ability_ratioが含まれており(line 368)、deviation特徴量も必要になる可能性がある
**Why it happens:** CONTEXT.mdではWinTwoStageModelへの追加のみ決定(D-04)だが、PlaceTwoStageModelのRETURN_FEATURE_COLSにもodds_to_ability_ratioが存在
**How to avoid:** D-04はWinTwoStageModelへの追加のみを指定。PlaceTwoStageModelへは追加しない(スコープ外)。ただし_prepare_features()のavailable_colsフィルタにより、dfに列が存在すれば使用される可能性があるため、意図を明確にする
**Warning signs:** PlaceTwoStageModelのfeature importanceにdeviation特徴量が現れる

### Pitfall 5: 非適合スコアの再利用時のalpha不整合
**What goes wrong:** キャリブレーション時(alpha=0.1, 90%区間)の非適合スコアを80%区間(alpha=0.2)に再利用する際、quantileレベルが異なる
**Why it happens:** np.quantile(residuals, 1-alpha)のalphaが0.1→0.2に変わるとquantile値が変わるが、residuals自体は同じキャリブレーションデータで再利用可能
**How to avoid:** キャリブレーション時は非適合スコア(abs(actual-predicted))を保存し、predict_interval()でalphaごとにquantileを再計算する。calibrate()は変更不要
**Warning signs:** 80%区間の下限が90%区間の下限より低くなる(逆転)

## Code Examples

### compute_odds_deviation_features() -- standalone関数
```python
# src/features/odds_deviation_features.py (NEW FILE)
"""ODDS-01: モデル予測確率と市場オッズの乖離を特徴量化"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_odds_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """odds_to_ability_ratioからレース内相対特徴量を計算。

    入力前提: odds_to_ability_ratio列が既に計算済み(training_pipeline)または
    WinTwoStageModel._prepare_features()で計算される(race_predictor)。

    Args:
        df: race_id, odds_to_ability_ratio列を含むDataFrame

    Returns:
        deviation_rank, deviation_zscore列が追加されたDataFrame
    """
    df = df.copy()

    ratio = df.get("odds_to_ability_ratio")
    if ratio is None:
        df["deviation_rank"] = pd.Series(np.nan, index=df.index, dtype=float)
        df["deviation_zscore"] = pd.Series(np.nan, index=df.index, dtype=float)
        return df

    ratio = pd.to_numeric(ratio, errors="coerce")

    # レース内ランク (descending: ratio大=過小評価=高いrank)
    df["deviation_rank"] = (
        ratio.groupby(df["race_id"]).rank(method="first", ascending=False)
        .astype("Float64")  # nullable int相当 (market_error_rank_in_raceパターン)
    )

    # レース内z-score標準化
    race_mean = ratio.groupby(df["race_id"]).transform("mean")
    race_std = ratio.groupby(df["race_id"]).transform("std").replace(0, np.nan)
    df["deviation_zscore"] = ((ratio - race_mean) / race_std).clip(-5.0, 5.0)

    return df
```
[VERIFIED: market_model.py lines 126-131 のmarket_error_rank_in_raceパターンと同一構造]

### RobustConfidenceEstimator.predict_interval() -- 拡張
```python
# src/models/robust_confidence_estimator.py (MODIFY)
def predict_interval(
    self,
    win_df: pd.DataFrame,
    place_df: pd.DataFrame,
    alphas: tuple[float, ...] = (0.1, 0.2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EVの信頼区間(上下)を推定。min(CP, Rolling_Quantile)を採用 (Rule 4)。

    Args:
        alphas: 信頼水準のタプル。0.1=90%区間、0.2=80%区間

    Returns:
        win_df, place_df with EV_lower/upper columns and conformal_confidence_score
    """
    # ... 既存predict_lower_bound()のロジックをベースにupperも追加 ...
    # upper = win_ev + cp_quantile_per_row
    # conformal_confidence_score = 合成(EV下限80%, 区間幅90%)
```
[VERIFIED: robust_confidence_estimator.py lines 96-148 の構造に基づく]

### TrainingPipeline統合ポイント
```python
# src/pipelines/training_pipeline.py line 433 の直後に追加:
if "p_market_win_adj" in df_oof.columns and "p_ability_win" in df_oof.columns:
    p_market = df_oof["p_market_win_adj"].clip(lower=1e-6)
    p_ability = df_oof["p_ability_win"].clip(lower=1e-6)
    df_oof["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

# NEW: ODDS-01 deviation features
from features.odds_deviation_features import compute_odds_deviation_features
df_oof = compute_odds_deviation_features(df_oof)
```
[VERIFIED: training_pipeline.py lines 422-433]

### RacePredictor統合ポイント
```python
# src/backtest/race_predictor.py の step 4 推論チェーン内、
# AbilityModel(line 96)の直後、WinTwoStageModel(line 98)の直前に追加:

df = submodel.stage1.add_ability_probs(df)  # line 96 (existing)

# NEW: ODDS-01 deviation features (AbilityModel出力後、WinTwoStageModel前)
from features.odds_deviation_features import compute_odds_deviation_features
df = compute_odds_deviation_features(df)

df = submodel.win.predict_ev(df)  # line 98 (existing)
```
[VERIFIED: race_predictor.py lines 89-98 の推論フロー]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 単一odds_to_ability_ratio | 3信号(絶対値+順位+標準化) | Phase 6 (ODDS-01) | LightGBMに絶対+相対の両情報を提供。直交性が高く特徴量として有効 |
| EV下限のみ | EV上下区間(2段階信頼水準) | Phase 6 (ODDS-03) | ベット選択の信頼性を定量化。80%/90%で段階的評価 |
| 3次元quantile-bin (prob, edge, odds) | 4次元quantile-bin (+ confidence) | Phase 6 (ODDS-03) | conformal信頼性をスコアリングに統合 |

**Deprecated/outdated:**
- なし(既存機能は全て維持。拡張のみ)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | PlaceTwoStageModelのRETURN_FEATURE_COLSにはdeviation特徴量を追加しない(D-04はWinTwoStageModelのみ指定) | Architecture Patterns | もしPlaceにも必要なら、FEATURE_COLSにも追加が必要 |
| A2 | conformal_confidence_scoreはWinSelectionGateのみで使用(PlaceSelectionGateには統合しない) | Architecture Patterns | Place gateにも統合する場合、PlaceSelectionGateModelの変更が必要 |
| A3 | deviation_rankのascending=False(過小評価=高いrank)が最適な順序 | Code Examples | 逆順が良い場合、LightGBMが自動学習するため影響は限定的 |
| A4 | キャリブレーションデータの非適合スコアを80%区間に再利用可能 | Architecture Patterns | 分布変化がある場合、80%区間の精度が低下する可能性 |

## Open Questions

1. **deviation特徴量の配置先ファイル**
   - What we know: D-02でstandalone関数と決定。MarketModelのパターンに倣う
   - What's unclear: 既存モジュール(odds_dynamics_features.py)に追加するか、新規ファイルにするか
   - Recommendation: 新規ファイル`src/features/odds_deviation_features.py`が最適。odds_dynamics_featuresはraw data特徴量(オッズ時系列)であり、deviationはモデル出力依存特徴量のため。責務が異なる

2. **conformal_confidence_scoreの合成式の詳細**
   - What we know: D-06でEV下限と区間幅の合成と決定。D-08でquantile-binに統合
   - What's unclear: 具体的な重み付け
   - Recommendation: 初期実装は`score = EV_lower_80 * (1 - normalized_width_90)`。ただしWinSelectionGateのwalk-forward OOF学習が最適な重みを自動発見するため、初期値は大雑把でよい

3. **WinSelectionGateの次元拡張戦略**
   - What we know: D-08で新次元として追加
   - What's unclear: combo_scores(4次元)にするか、pair_scoresにconformalペアを追加するか
   - Recommendation: pair_scoresに("confidence_prob", confidence_bin, prob_bin)等のペアを追加。combo_scoresの4次元は組み合わせ爆発リスク。ただし、この判断は実装時のデータ量に依存するため、Plannerの裁量に委ねる

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全体 | ✓ | mise管理 | -- |
| numpy | deviation特徴量計算 | ✓ | 既存依存 | -- |
| pandas | groupby操作 | ✓ | 既存依存 | -- |
| LightGBM | Stage2モデル | ✓ | 既存依存 | -- |
| PostgreSQL | バックテスト実行 | ✓ | localhost:5432 | -- |
| pytest | テスト | ✓ | 既存依存 | -- |

**Missing dependencies with no fallback:**
- なし

**Missing dependencies with fallback:**
- なし

## Sources

### Primary (HIGH confidence)
- コードベース直接確認 -- 全ての統合ポイント、パターン、依存関係を検証済み
  - `src/models/two_stage_return_model.py` -- FEATURE_COLS定義, _prepare_features()
  - `src/models/market_model.py` -- predict_and_calc_error(), groupby rank パターン
  - `src/models/robust_confidence_estimator.py` -- predict_lower_bound(), calibrate()
  - `src/backtest/race_predictor.py` -- predict()推論フロー全ステップ
  - `src/pipelines/training_pipeline.py` -- _train_submodel()学習フロー全ステップ
  - `src/models/win_selection_gate.py` -- quantile-bin scoring, _build_score_tables()
  - `src/models/benter_combination.py` -- logit-space合成パターン
  - `src/models/win_benter_gate.py` -- WinBenterGate.apply()パイプライン
  - `src/features/odds_dynamics_features.py` -- compute_odds_dynamics()パターン参照
  - `src/domain/models.py` -- SubmodelSet, TrainedModelsV5データクラス定義

### Secondary (MEDIUM confidence)
- [CONTEXT.md](/.planning/phases/06-odds-deviation/06-CONTEXT.md) -- ユーザーのlocked decisionsとdiscretion areas

### Tertiary (LOW confidence)
- なし -- 全ての技術的判断はコードベース直接確認に基づく

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- 追加依存関係なし、既存パッケージのみで完結
- Architecture: HIGH -- 統合ポイントをコードベースで特定済み、既存パターンの踏襲
- Pitfalls: HIGH -- training_pipeline.py/race_predictor.pyの呼び出し順序を検証済み

**Research date:** 2026-05-03
**Valid until:** 2026-06-03 (stable codebase、大きな変更がない限り)
