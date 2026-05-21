# Phase 3: Selection Gate, Confidence & Betting - Research

**Researched:** 2026-05-02
**Domain:** Win selection gate (learned binary filter), Conformal prediction confidence, JRA takeout edge thresholds
**Confidence:** HIGH

## Summary

Phase 3 は3つの要件を実装する: (1) PlaceSelectionGateModel のパターンを完全踏襲する WinSelectionGateModel、(2) RobustConfidenceEstimator の Win CP quantile 精度向上、(3) RegimeDetector/MetaSwitcher の edge_threshold を JRA 控除率 25% に合わせて調整。

PlaceSelectionGateModel (1044行) は極めて洗練された OOF walk-forward score table 機構を持つ。3次元 binning (prob/edge/odds)、smoothed scoring (Bayesian prior)、add-second reranker、soft_pass_mask を全て含み、これを Win 向けに複製する。Win 特有の変更点は: fukuoddslow -> tanoddslow、place_selection_* -> win_selection_*、kakuteijyuni <= 3 -> kakuteijyuni == 1、place gate score 列名の変更のみ。

RobustConfidenceEstimator は既に Win/Place 両対応の CP quantile を計算済み。SELC-02 の拡張は race-condition-dependent calibration (距離/グレード等の条件別に CP quantile を細分化) が主な作業。

RegimeDetector の edge_threshold は現在 AGGRESSIVE=0.04 / CONSERVATIVE=0.05 / COLLAPSED=0.08 だが、MetaSwitcher では 0.04/0.06/0.09 と既に上方修正済み。JRA 控除率 25% を考慮すると、p差分方式 (p_model - p_market) は既に控除率込みの p_market と比較するため、edge_threshold は控除率分 (25%×p_model) を上回る必要がある。ただし D-09 の通り p_market = 1/tanoddslow は既に控除率込みであるため、edge が正であればそれは控除率を上回っていることを意味する。したがって edge_threshold の引き上げは「安全性マージン」の追加であり、現在値の +0.01~0.03 程度が妥当。

**Primary recommendation:** PlaceSelectionGateModel を Win 向けに機械的に複製し、列名と判定条件 (place top3 -> win 1着) のみ変更する。race-condition-dependent calibration は GroupBy ベースの条件別 CP quantile 計算を追加する。edge_threshold は MetaSwitcher の現在値を起点に +0.01 引き上げる。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** PlaceSelectionGate完全踏襲。OOF walk-forward score tables + smoothed scoring + add-second reranker + soft_pass_maskを全て再現。新クラスWinSelectionGateModelとして実装
- **D-02:** 入力変数はPlaceSelectionGateと同一構造（prob/edge/oddsの3次元binning）。Win特化入力の追加は行わない
- **D-03:** オッズソースはtanoddslow（最終単勝オッズ）。PlaceSelectionGateがfukuoddslowを使うのと同じパターン
- **D-04:** add-second rerankerを実装。ゲートがOOFデータから2頭目の有効性を学習するデータ駆動アプローチ
- **D-05:** 列名はwin_selection_prob, win_selection_edge, tanoddslow。PlaceSelectionGateのplace_selection_prob/place_selection_edge/fukuoddslowに対応
- **D-06:** 既存RobustConfidenceEstimatorを拡張。Win/Place両対応の既存コードを活用。CP quantileの精度向上（race-condition-dependent calibration）を行う
- **D-07:** EV下限値（EV_lower_win_corrected）をWinSelectionGateの入力edgeとして使用。3次元binning構造は維持
- **D-08:** 低信頼レースは閾値で完全除外（SELC-02要件）。WinSelectionGateのmin_prob/min_edge/max_oddsで足切り。賭け金調整ではなく除外
- **D-09:** エッジ計算はp差分方式。edge = p_model - p_market（p_market = 1/tanoddslow）。p_marketは既に控除率込み
- **D-10:** RegimeDetectorのedge_threshold設定にJRA控除率を反映。レジーム別に控除率を考慮した閾値を設定
- **D-11:** Kelly賭け金計算は既存の簡易Kelly(edge/(odds-1), cap=25%)を維持
- **D-12:** WinSelectionGateModel新クラスを作成。PlaceSelectionGateModelとは独立。SubmodelSetにwin_selection_gateフィールドを追加
- **D-13:** 学習→保存→読み込みの3点更新パターンに従い、training_pipeline.py / model_loader.py / domain/models.py を更新
- **D-14:** RacePredictor.predict()でWin予測後のBenter適用後(line 124あたり)にWinSelectionGateを適用

### Claude's Discretion
- RegimeDetectorの具体的なedge_threshold値（控除率考慮後の最適値）
- RobustConfidenceEstimatorのrace-condition-dependent calibrationの詳細実装
- WinSelectionGateのsmoothed scoreのprior_weight等ハイパーパラメータ
- add-second rerankerの閾値グリッドの範囲・粒度

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SELC-01 | PlaceSelectionGateパターンを踏襲した単勝選択ゲート(学習済みバイナリフィルター)を実装する | PlaceSelectionGateModel の完全コード解析済み。WinSelectionGateModel は列名・オッズソース・的中判定のみ変更で複製可能 |
| SELC-02 | Conformal predictionに基づく信頼性推定を実装し、低信頼度レースを除外する | RobustConfidenceEstimator の Win CP quantile 拡張ポイント特定済み。race-condition-dependent calibration の実装方針確定 |
| BETT-01 | JRA控除率25%を考慮したエッジ閾値を設定・調整する | RegimeDetector + MetaSwitcher の現在値と修正ポイント特定済み。p差分方式の理論的根拠確認済み |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| WinSelectionGate scoring | API/Backend (ML pipeline) | -- | 学習済みスコアテーブルに基づく推論はバックエンドで実行 |
| WinSelectionGate training | API/Backend (training) | -- | OOF walk-forward は学習パイプラインの一部 |
| CP confidence estimation | API/Backend (ML pipeline) | -- | RobustConfidenceEstimator 拡張、バックエンド推論 |
| Edge threshold configuration | API/Backend (betting) | -- | RegimeDetector + MetaSwitcher のパラメータ更新 |
| Win bet generation | API/Backend (betting) | -- | RacePredictor/WinStrategy での Bet リスト生成 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (installed) | 数値計算、配列操作 | 既存依存。score table 計算で使用 |
| pandas | (installed) | DataFrame操作 | 既存依存。binning/groupby/merge |
| joblib | (installed) | モデルのシリアライズ | 既存依存。PlaceSelectionGateModel.save() で使用 |
| lightgbm | (installed) | GBM モデル | 既存依存。RegimeDetector 学習済みモデル用 |
| scikit-learn | (installed) | IsotonicRegression 等 | 既存依存。Benter calibration 用 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | (installed) | optimize.minimize | Benter combination NLL 最適化 (既存パターン) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手動WinSelectionGate | PlaceSelectionGateにwin_mode追加 | 独立クラスの方が変更影響が小さい (D-12決定済み) |

**Installation:**
追加パッケージ不要。全て既存依存。

## Architecture Patterns

### System Architecture Diagram

```
TrainingPipelineV5._train_submodel()
  │
  ├─ [既存] RobustConfidenceEstimator.calibrate()
  │     └─ Win CP quantile + Rolling quantile (拡張: race-condition-dependent)
  │
  ├─ [既存] PlaceSelectionGateModel.train()
  │
  ├─ [NEW] WinSelectionGateModel.train()          ← SELC-01
  │     └─ OOF walk-forward score tables
  │     └─ add-second reranker
  │     └─ threshold grid search
  │
  └─ SubmodelSet に win_selection_gate を格納

RacePredictor.predict()
  │
  ├─ [既存] Win Benter 適用 (line 116-124)
  │     └─ WinBenterGate.apply()
  │         └─ p_win_final, edge_win を生成
  │
  ├─ [NEW] WinSelectionGate 適用                ← SELC-01, D-14
  │     └─ ensure_win_selection_columns()
  │     └─ gate_model.score()
  │     └─ win_selection_reason 設定
  │
  └─ [既存] PlaceSelectionGate 適用 (line 192-202)

RegimeDetector / MetaSwitcher
  │
  └─ [UPDATE] edge_threshold 値更新             ← BETT-01
       AGGRESSIVE:  0.04 → 0.05
       CONSERVATIVE: 0.06 → 0.07
       COLLAPSED:   0.09 → 0.10

GateKeeper.filter_bets()
  └─ edge >= edge_threshold で最終足切り
```

### Recommended Project Structure
```
src/
├── models/
│   ├── win_selection_gate.py       # [NEW] WinSelectionGateModel
│   ├── place_selection_gate.py     # [既存] 参照元
│   ├── robust_confidence_estimator.py  # [UPDATE] race-condition-dependent
│   └── regime_detector.py          # [UPDATE] edge_threshold 値
├── betting/
│   ├── meta_switcher.py            # [UPDATE] edge_threshold 値
│   ├── gate_keeper.py              # [UPDATE] デフォルト閾値
│   └── win_strategy.py             # [UPDATE] edge 設定
├── backtest/
│   └── race_predictor.py           # [UPDATE] WinSelectionGate 統合
├── pipelines/
│   └── training_pipeline.py        # [UPDATE] WinSelectionGate 学習・保存
├── db/
│   └── model_loader.py             # [UPDATE] WinSelectionGate 読み込み
└── domain/
    └── models.py                   # [UPDATE] SubmodelSet にフィールド追加
```

### Pattern 1: WinSelectionGateModel (PlaceSelectionGateModel の Win 版)
**What:** OOF walk-forward score tables + 3次元 binning による学習済みバイナリフィルター
**When to use:** RacePredictor.predict() 内で Win Benter 適用後
**Example:**
```python
# Place版との差分（変更点のみ）:
# 1. クラス名: PlaceSelectionGateModel -> WinSelectionGateModel
# 2. 列名:
#    fukuoddslow -> tanoddslow
#    place_selection_prob -> win_selection_prob
#    place_selection_edge -> win_selection_edge
#    place_gate_score -> win_gate_score
#    place_gate_pass -> win_gate_pass
#    place_gate_rank -> win_gate_rank
#    place_gate_score_gap -> win_gate_score_gap
#    place_selection_ev -> win_selection_ev
# 3. 的中判定:
#    kakuteijyuni <= 3 -> kakuteijyuni == 1
#    realized_place_roi -> realized_win_roi
# 4. EV列:
#    EV_lower_place -> EV_lower_win_corrected
#    ev_place_corrected -> ev_win_corrected
# 5. オッズの log 変換:
#    log_place_odds = log1p(fukuoddslow) -> log_win_odds = log1p(tanoddslow)
```

### Pattern 2: 3点更新パターン (学習→保存→読み込み)
**What:** 新モデル追加時の必須3ファイル更新
**When to use:** SubmodelSet に新しい Optional フィールドを追加する時
**Example:**
```python
# 1. domain/models.py - SubmodelSet にフィールド追加
@dataclass
class SubmodelSet:
    ...
    win_selection_gate: WinSelectionGateModel | None = None  # [NEW]

# 2. training_pipeline.py - _train_submodel() で学習
gate_train_df = df_oof.copy()
_, gate_win_df = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
if "EV_lower_win_corrected" in gate_win_df.columns:
    gate_train_df["EV_lower_win_corrected"] = gate_win_df["EV_lower_win_corrected"].values
gate_train_df = ensure_win_selection_columns(gate_train_df)
win_selection_gate = WinSelectionGateModel()
win_selection_gate.train(gate_train_df)

# 3. model_loader.py - _load_from_local() で読み込み
win_selection_gate = None
wsg_file = models_dir / f"win_selection_gate_{surface}.joblib"
if wsg_file.is_file():
    win_selection_gate = WinSelectionGateModel.load(wsg_file)
```

### Pattern 3: ensure_*_selection_columns / build_*_selection_ev フォールバック連鎖
**What:** 列存在確認→フォールバック生成のパターン
**When to use:** WinSelectionGate の入力列が欠損している場合
**Example:**
```python
def ensure_win_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "win_selection_ev" not in prepared.columns:
        if "EV_lower_win_corrected" in prepared.columns or "ev_win_corrected" in prepared.columns:
            prepared["win_selection_ev"] = build_win_selection_ev(prepared)
        elif "edge_win" in prepared.columns:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "edge_win") + 1.0
        else:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "ev_win")
    if "win_selection_edge" not in prepared.columns:
        prepared["win_selection_edge"] = _numeric_or_nan(prepared, "win_selection_ev") - 1.0
    if "win_selection_prob" not in prepared.columns:
        if "p_win_final" in prepared.columns:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_final")
        elif "p_win_combined" in prepared.columns:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_combined")
        else:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_corrected")
    return prepared
```

### Anti-Patterns to Avoid
- **既存PlaceSelectionGateModelの変更:** Win版は独立した新クラスとして実装。既存クラスに影響を与えないこと (D-12)
- **Win特化入力の追加:** 3次元binning構造を維持。Win特有の入力変数を追加しない (D-02)
- **p_market計算への干渉:** p_market = 1/tanoddslow は既に控除率込み。新たな補正を加えない (D-09)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| スコアテーブル機構 | 新規設計 | PlaceSelectionGateModel の _build_score_tables() をWin用に複製 | 1044行の成熟した実装。bayesian smoothing + 3層fallback (combo -> pair -> single) の再設計は高リスク |
| Walk-forward fold分割 | 独自ロジック | _build_walk_forward_folds() をそのまま使用 | min_train_races/min_fold_races/max_folds のバランスが既にチューニング済み |
| Threshold grid search | 新しいグリッド | _build_threshold_grid() をWin用に調整 (fukuoddslow -> tanoddslow) | 分位数ベース + 固定値のハイブリッドグリッドが効果的 |
| モデルシリアライズ | カスタム形式 | joblib (save/load) | PlaceSelectionGateModel と同じパターン |

**Key insight:** PlaceSelectionGateModel は「確率/エッジ/オッズの3次元binning + bayesian smoothing + walk-forward検証 + add-second reranker」という層状の設計。これを新規に設計するより、列名と判定条件だけ変えて複製する方が圧倒的に安全。

## Common Pitfalls

### Pitfall 1: Win的中判定の違い (place: top3 / win: 1着のみ)
**What goes wrong:** realized_roi の計算で kakuteijyuni <= 3 を使ってしまう
**Why it happens:** Place版のコードをそのままコピペした場合
**How to avoid:** realized_win_roi = tanoddslow if kakuteijyuni == 1 else 0.0
**Warning signs:** Win gate の global_score が極端に高くなる (place ~1.2 vs win 確率が低いので ~0.05 程度が期待値)

### Pitfall 2: tanoddslow と tanodds の混同
**What goes wrong:** WinSelectionGate で tanodds (オッズスナップショット) を使ってしまう
**Why it happens:** WinBenterGate は tanodds を使う (line 62) が、SelectionGate は最終オッズ (tanoddslow) を使うべき
**How to avoid:** WinSelectionGate のオッズソースは tanoddslow に統一 (D-03)
**Warning signs:** スコアテーブルのオッズbinが極端に広がる

### Pitfall 3: Win のサンプル数不足
**What goes wrong:** Win 1着のサンプルは Place top3 の ~1/3 しかないため、OOF fold が空になる
**Why it happens:** place_hit_rate ~37.5% vs win_hit_rate ~8-10% (JRA 18頭立てで 1/18 ≈ 5.5%)
**How to avoid:** min_fold_races を下げる、min_train_races を下げる、または _trained = False で安全にフォールバック
**Warning signs:** train() 後も is_trained == False

### Pitfall 4: edge_threshold 引き上げすぎ
**What goes wrong:** ベット数が激減して ROI 測定不能になる
**Why it happens:** Win の edge は Place より分散が大きい。高すぎる閾値はベットを完全に排除する
**How to avoid:** MetaSwitcher の現在値 (0.04/0.06/0.09) を起点に +0.01 の微小引き上げにとどめる
**Warning signs:** バックテストで 0 bets

### Pitfall 5: RacePredictor への Win 適用順序
**What goes wrong:** WinSelectionGate を Benter 適用前（ev_corrected のみ）で実行してしまう
**Why it happens:** predict() の推論チェーンの順序を誤解
**How to avoid:** D-14 で指定された通り、Win Benter 適用後 (line 124) に挿入。p_win_final と edge_win が生成された後
**Warning signs:** win_selection_prob が p_win_corrected (Benter前) を参照する

### Pitfall 6: Win edge = p_model - p_market の分散
**What goes wrong:** Win の edge 分布が Place より裾が厚い。閾値チューニングが Place のパターンと異なる
**Why it happens:** Win確率は Place確率より低く、オッズも高い。edge = p_model * odds - 1.0 は大きな正負の振れを持つ
**How to avoid:** WinSelectionGate の threshold grid search が自動的に最適閾値を見つける。手動設定に頼らない
**Warning signs:** ゲートの min_edge が 0.0 になる (全てスルー) または高すぎて全排除

## Code Examples

### WinSelectionGateModel の train() の核心変更点
```python
# Source: src/models/place_selection_gate.py (VERIFIED: codebase)

# Place版 (元コード):
prepared["realized_place_roi"] = np.where(
    prepared["kakuteijyuni"] <= 3,
    prepared["fukuoddslow"],
    0.0,
)
prepared["log_place_odds"] = np.log1p(prepared["fukuoddslow"])

# Win版 (変更):
prepared["realized_win_roi"] = np.where(
    prepared["kakuteijyuni"] == 1,
    prepared["tanoddslow"],
    0.0,
)
prepared["log_win_odds"] = np.log1p(prepared["tanoddslow"])
```

### ensure_win_selection_columns パターン
```python
# Source: src/models/place_selection_gate.py:33-54 をWin用に変換 (VERIFIED: codebase)

def build_win_selection_ev(df: pd.DataFrame) -> pd.Series:
    lower_ev = _numeric_or_nan(df, "EV_lower_win_corrected")
    corrected_ev = _numeric_or_nan(df, "ev_win_corrected")
    direct_ev = _numeric_or_nan(df, "ev_win")

    if corrected_ev.notna().any():
        selection_ev = lower_ev.where(lower_ev.notna(), corrected_ev)
        safety_floor = corrected_ev * 0.85
        return pd.concat([selection_ev, safety_floor], axis=1).max(axis=1).astype(float)
    if lower_ev.notna().any():
        return lower_ev.astype(float)
    return direct_ev.astype(float)

def ensure_win_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "win_selection_ev" not in prepared.columns:
        if "EV_lower_win_corrected" in prepared.columns or "ev_win_corrected" in prepared.columns:
            prepared["win_selection_ev"] = build_win_selection_ev(prepared)
        elif "edge_win" in prepared.columns:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "edge_win") + 1.0
        else:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "ev_win")
    if "win_selection_edge" not in prepared.columns:
        prepared["win_selection_edge"] = _numeric_or_nan(prepared, "win_selection_ev") - 1.0
    if "win_selection_prob" not in prepared.columns:
        if "p_win_final" in prepared.columns:         # Benter適用後
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_final")
        elif "p_win_combined" in prepared.columns:     # Benter適用後 (race normalization前)
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_combined")
        else:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_corrected")
    return prepared
```

### TrainingPipeline への挿入ポイント
```python
# Source: src/pipelines/training_pipeline.py:760-767 (VERIFIED: codebase)
# PlaceSelectionGate 学習の直後にWinSelectionGate学習を追加:

with TimingContext(f"{surface}/win_selection_gate"):
    wsg_train_df = df_oof.copy()
    _, wsg_win_df = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
    if "EV_lower_win_corrected" in wsg_win_df.columns:
        wsg_train_df["EV_lower_win_corrected"] = wsg_win_df["EV_lower_win_corrected"].values
    wsg_train_df = ensure_win_selection_columns(wsg_train_df)
    win_selection_gate = WinSelectionGateModel()
    win_selection_gate.train(wsg_train_df)

# SubmodelSet の戻り値に追加:
return SubmodelSet(
    ...,
    win_selection_gate=win_selection_gate,  # [NEW]
)
```

### RacePredictor.predict() への挿入ポイント
```python
# Source: src/backtest/race_predictor.py:116-134 (VERIFIED: codebase)
# D-14: Win Benter適用後 (line 124) に WinSelectionGate を適用

# --- 既存: Win Benter Combination (line 116-124) ---
if getattr(submodel, "win_benter", None) is not None:
    win_gate = WinBenterGate(...)
    df = win_gate.apply(df)

# --- [NEW] WinSelectionGate (D-14) ---
ensure_win = ensure_win_selection_columns(df)
if "win_selection_ev" not in df.columns:
    df = ensure_win
win_gate_model = getattr(submodel, "win_selection_gate", None)
win_gate_enabled = bool(
    win_gate_model is not None and getattr(win_gate_model, "is_trained", False) is True
)
if win_gate_enabled:
    df = win_gate_model.score(df)
    win_annotate = getattr(win_gate_model, "annotate_race_context", None)
    if callable(win_annotate):
        df = win_annotate(df)

# --- 既存: Place 推論 (line 126-) ---
df = submodel.place.predict_ev(df)
```

### RobustConfidenceEstimator の race-condition-dependent calibration
```python
# Source: src/models/robust_confidence_estimator.py (VERIFIED: codebase)
# 現在の Win CP quantile 計算:
self._win_cp_quantile = float(np.quantile(win_residuals.values, 1 - self.alpha))

# 拡張: 条件別 (surface, distance_bin) の CP quantile
# race_condition_key = (surface, distance_bin) の GroupBy
# 各グループで独立に CP quantile を計算
# サンプル不足時は global quantile にフォールバック
```

### RegimeDetector / MetaSwitcher の edge_threshold 更新
```python
# Source: src/models/regime_detector.py:183/211/224 (VERIFIED: codebase)
# 現在の RegimeDetector 値:
#   AGGRESSIVE:  edge_threshold = 0.04
#   CONSERVATIVE: edge_threshold = 0.05
#   COLLAPSED:   edge_threshold = 0.08

# Source: src/betting/meta_switcher.py:47/55/63 (VERIFIED: codebase)
# 現在の MetaSwitcher 値:
#   AGGRESSIVE:  edge_threshold = 0.04
#   CONSERVATIVE: edge_threshold = 0.06
#   COLLAPSED:   edge_threshold = 0.09

# BETT-01 推奨値 (JRA控除率25%考慮、Claude's Discretion):
#   AGGRESSIVE:  edge_threshold = 0.05 (+0.01)
#   CONSERVATIVE: edge_threshold = 0.07 (+0.01)
#   COLLAPSED:   edge_threshold = 0.10 (+0.01)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 固定edge閾値 0.03 | RegimeDetector連動の動的閾値 | v5.4 | レジーム別の最適化が可能に |
| 単一CP quantile | min(CP, Rolling Quantile) Rule 4 | v5.5 | 過信防止の二重安全網 |
| PlaceのみSelectionGate | WinSelectionGate 追加 | Phase 3 (本フェーズ) | Win ベットの品質フィルタリング |

**Deprecated/outdated:**
- GateKeeper.should_bet() の ev_threshold パラメータ: API互換のため残すが未使用 (line 27 "ev_threshold は無視")

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Win 1着率は ~8-10% (JRA 18頭立て) であり、OOF fold で十分なサンプルが得られる | Pitfalls | min_train_races/min_fold_races の調整が必要になる |
| A2 | tanoddslow 列は feat_df / df_oof に常に存在する | Pattern 3 | 存在しない場合、ensure_win_selection_columns のフォールバックが必要 |
| A3 | edge_threshold の +0.01 引き上げが JRA 控除率に対する適切なマージンである | BETT-01 | 大きすぎるとベット数激減、小さすぎると赤字継続 |
| A4 | RacePredictor.predict() 内で edge_win 列が Benter 適用後に生成される (WinBenterGate.apply line 82) | Pattern 5 | edge_win が未生成の場合、ensure_win_selection_columns がフォールバックする |

**All assumptions above are `[ASSUMED]` -- based on code analysis and domain knowledge, not verified by live backtest.**

## Open Questions

1. **WinSelectionGate のハイパーパラメータ調整**
   - What we know: PlaceSelectionGateModel は n_bins=6, prior_weight=24.0, min_train_races=200, min_fold_races=80
   - What's unclear: Win のサンプル密度が ~1/3 のため、min_train_races を下げる必要があるか
   - Recommendation: 初期値は Place と同じにして、train() が空フォールドを返した場合に段階的に下げる

2. **race-condition-dependent calibration の粒度**
   - What we know: RobustConfidenceEstimator は現在 global な CP quantile のみ
   - What's unclear: どの条件変数 (surface, distance_bin, grade_code) でグループ化するか
   - Recommendation: surface (芝/ダート) と distance_bin (sprint/mile/intermediate/long) の2変数 GroupBy から開始

3. **RegimeDetector と MetaSwitcher の edge_threshold 同期**
   - What we know: RegimeDetector.get_strategy_params() と MetaSwitcher._default_params() で独立に閾値を定義
   - What's unclear: どちらを正とするか。現在は MetaSwitcher の方が高い値を使っている
   - Recommendation: 両方を同じ値に更新する。RegimeDetector はバックテストの race_predictor で使用され、MetaSwitcher は本番オーケストレーターで使用される

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified -- code/config changes only, no new tools or services required)

## Sources

### Primary (HIGH confidence)
- `src/models/place_selection_gate.py` - WinSelectionGateModel の設計テンプレート完全コード (1044行)
- `src/models/robust_confidence_estimator.py` - CP quantile 計算の現状実装
- `src/models/regime_detector.py` - edge_threshold の現在値 (lines 183/211/224)
- `src/betting/meta_switcher.py` - edge_threshold の現在値 (lines 47/55/63)
- `src/backtest/race_predictor.py` - WinSelectionGate の適用ポイント (lines 116-134, 192-202)
- `src/pipelines/training_pipeline.py` - 学習パイプライン挿入ポイント (lines 760-767)
- `src/db/model_loader.py` - モデル読み込みパターン (lines 451-689)
- `src/domain/models.py` - SubmodelSet dataclass (line 229)

### Secondary (MEDIUM confidence)
- `src/betting/gate_keeper.py` - GateKeeper edge フィルタリング
- `src/betting/orchestrator.py` - BettingOrchestrator の edge_threshold 使用箇所
- `src/betting/win_strategy.py` - WinStrategy の Kelly 計算
- `src/models/win_benter_gate.py` - WinBenterGate の apply() メソッド

### Tertiary (LOW confidence)
- JRA控除率25%がp差分方式で適切に扱われるという理論的根拠: p_market = 1/tanoddslow は確率の合計が1.0を超える (overround) ため、控除率込み。p_model がこの控除率込み p_market を上回れば真のエッジがある

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全て既存コードの解析に基づく。新規ライブラリ不要
- Architecture: HIGH - PlaceSelectionGateModel の完全踏襲パターン。挿入ポイント特定済み
- Pitfalls: HIGH - Win/Place の差分 (的中判定、サンプル密度、オッズソース) を具体的に特定

**Research date:** 2026-05-02
**Valid until:** 2026-06-01 (stable - 全て既存コードベースの静的解析に基づく)
