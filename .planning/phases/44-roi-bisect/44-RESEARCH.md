# Phase 44: ROI Bisect - Research

**Researched:** 2026-05-30
**Domain:** Component isolation diagnostics for DeploymentGate FAIL root cause attribution
**Confidence:** HIGH

## Summary

Phase 44 は、Phase 43 Shadow Diagnosis の成果物を主入力とし、DeploymentGate FAIL の 4 つの直接原因（ECE悪化, bet_count低下, APR上振れ, OddsBandFilter影響）を MAWC/Ranker/OddsBandFilter/Selection logic のどのコンポーネントに帰属させるかを特定する診断フェーズである。モデル再学習は行わず、既存学習済みモデル（`data/models-backtest/{2024,2025}/`）を使用した post-hoc 分析と最小限の ablation 実行のみを行う。

**Primary recommendation:** Phase 44 の実装は (1) post-hoc コンポーネント帰属分析スクリプト、(2) MAWC/Ranker 係数分析、(3) 必要最小限の ablation 実行（ShadowComparisonFramework の VariantConfig 追加）の 3 層構造で実装する。既存の `ShadowComparisonFramework` が ablation 基盤としてそのまま使え、MAWC の `coef_` (51-dim) と Ranker の `coef_` (15-dim x 4 models) へのアクセスも検証済みである。

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Post-hoc 分析（既存成果物からの帰属） | Offline Analysis | - | Phase 41/43 成果物を読み込む純粋分析。サーバー不要 |
| MAWC/Ranker 係数抽出 | Offline Analysis | - | joblib モデルファイルの直接ロード。BacktestEngine 不要 |
| Ablation 実行（BT 再実行） | Backtest Pipeline | Offline Analysis | ShadowComparisonFramework 経由で BacktestEngine を起動 |
| 帰属結果出力 | Offline Analysis | - | JSON/MD 出力。Phase 45 が消費 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | LogisticRegression/Ridge coef_ access | MAWC と Ranker の係数抽出に必須 [VERIFIED: code inspection] |
| joblib | 1.5.3 | Model artifact loading | 既存の MAWC/Ranker モデルロード [VERIFIED: code inspection] |
| pandas | 2.3.3 | DataFrame analysis | Parquet I/O + セグメント分析 [VERIFIED: code inspection] |
| numpy | 2.4.3 | Numerical computation | 係数計算・統計 [VERIFIED: code inspection] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jinja2 | (existing) | HTML report generation | Phase 41/43 パターンに従う場合 |
| json | (stdlib) | JSON I/O | 成果物の読み書き |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom bisect script | ShadowDiagnosis extension | ShadowDiagnosis は 3-step 段階的除外に特化。Bisect は異なる分析順序（ECE→APR→bet_count→OBF）なので新クラスが適切 |

**Installation:**
```bash
# No new packages needed — all dependencies already installed
pip install -e ".[dev]"
```

## Package Legitimacy Audit

> Phase 44 does not install any new external packages. All analysis uses existing project dependencies.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*This phase requires no new package installs. All tools (scikit-learn, joblib, pandas, numpy) are existing project dependencies verified in pyproject.toml.*

## Architecture Patterns

### System Architecture Diagram

```
Phase 41/43/42 成果物 (入力)
    |
    +-- shadow_comparison_result.json  -----+
    +-- shadow_horse_diff.parquet   --------+-- Post-hoc Analysis (新クラス)
    +-- shadow_race_diff.parquet    --------+     |
    +-- shadow_diagnosis_result.json -------+     |  ECE→APR→bet_count→OBF
    +-- deployment_gate_result.json --------+     |  逐次帰属
                                             |
                                             +--> ComponentAttribution (JSON)
                                             |
data/models-backtest/{year}/                 |
    |                                        |
    +-- *_calibrator_*.joblib  -----> MAWC coef_ 分析 (51-dim)
    +-- *_ranker_*.joblib      -----> Ranker coef_ 分析 (15-dim x4)
                                             |
                                             v
                                     Ablation (必要時のみ)
                                             |
                                    ShadowComparisonFramework.run_fold()
                                    + VariantConfig 追加
                                             |
                                             v
                                     Ablation 結果 → 帰属確定
                                             |
                                             v
                                    data/backtest/shadow/bisect/
                                      +-- bisect_result.json
                                      +-- coefficient_analysis.json
                                      +-- bisect_summary.md
```

### Recommended Project Structure
```
src/backtest/
├── component_attribution.py    # 新: post-hoc コンポーネント帰属分析クラス
scripts/
├── run_component_attribution.py  # 新: CLI entry point
```

### Pattern 1: Post-hoc Component Attribution
**What:** 既存成果物（shadow_horse_diff, shadow_diagnosis_result）から MAWC/Ranker/OBF 各コンポーネントの影響を帰属する分析パターン
**When to use:** ECE/APR/bet_count の各 FAIL 原因を特定コンポーネントに紐付ける場合
**Example:**
```python
# Source: 独自設計 (CONTEXT.md D-01/D-02 に基づく)
class ComponentAttribution:
    def __init__(self, input_dir: Path):
        self.horse_diff = pd.read_parquet(input_dir / "shadow_horse_diff.parquet")
        self.race_diff = pd.read_parquet(input_dir / "shadow_race_diff.parquet")
        self.diagnosis = json.loads(
            (input_dir / "diagnosis/shadow_diagnosis_result.json").read_text()
        )

    def attribute_ece_degradation(self) -> dict:
        """D-02 Step 1: ECE悪化のコンポーネント帰属"""
        # p_win_before_mawc vs p_win_after_mawc を odds_band/popularity_band 別に比較
        ...

    def attribute_apr_deviation(self) -> dict:
        """D-02 Step 2: APR上振れのコンポーネント帰属"""
        ...

    def attribute_bet_count_loss(self) -> dict:
        """D-02 Step 3: bet_count低下のコンポーネント帰属"""
        ...
```

### Pattern 2: Coefficient Analysis
**What:** MAWC (LogisticRegression 51-dim) と Ranker (Ridge 15-dim x 4) の学習済み係数を抽出・分析するパターン
**When to use:** どの特徴量・セグメントが確率品質悪化に寄与しているかを特定する場合
**Example:**
```python
# Source: sklearn LogisticRegression.coef_ API
mawc_state = joblib.load("data/models-backtest/2024/market_aware_win_calibrator_turf.joblib")
calibrator = mawc_state["calibrator"]  # LogisticRegression
feature_names = mawc_state["feature_names"]  # 51 features

coef = calibrator.coef_[0]  # shape (51,)
# logit_market 係数 (index 1) と交互作用項の寄与を分析
market_coef = coef[feature_names.index("logit_market")]
# セグメント別交互作用項の寄与
interaction_coefs = {
    name: coef[i]
    for i, name in enumerate(feature_names)
    if name.startswith("logit_market_x_")
}
```

### Pattern 3: Ablation via ShadowComparisonFramework
**What:** VariantConfig を追加して最小限の ablation を実行するパターン
**When to use:** Post-hoc で因果分離不能な場合のみ
**Example:**
```python
# Source: src/backtest/shadow_comparison.py VariantConfig
variants = [
    VariantConfig(
        variant_name="baseline",
        model_dir=Path("data/models-backtest"),
        enable_market_aware_calibrator=False,
        enable_race_level_ranker=False,
    ),
    VariantConfig(
        variant_name="mawc_only",
        model_dir=Path("data/models-backtest"),
        enable_market_aware_calibrator=True,
        enable_race_level_ranker=False,
    ),
    VariantConfig(
        variant_name="ranker_only",
        model_dir=Path("data/models-backtest"),
        enable_market_aware_calibrator=False,
        enable_race_level_ranker=True,
    ),
]
framework = ShadowComparisonFramework(variants=variants, betting_target="win")
results = framework.run(folds=FoldDefinition.create_folds([2024, 2025]))
```

### Anti-Patterns to Avoid
- **並列分析**: ECE→APR→bet_count→OBF の逐次分析を skip して並列に回すこと。MAWC の確率補正は上流にあり、p_win_final/EV/Ranker score/候補選定/OBF 通過率に連鎖するため、確率品質を先に切り分けないと下流分析が汚染される (CONTEXT.md D-02)
- **フル N-way 比較の最初から実行**: Post-hoc で仮説を絞ってから必要最小限の ablation のみ実行すべき (CONTEXT.md D-01)
- **モデル再学習**: Phase 44 は診断のみ。既存モデル（data/models-backtest/）を使用 (CONTEXT.md D-01)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BT 再実行 | 独自 BT ループ | ShadowComparisonFramework.run_fold() | 既存の flag injection、alignment、metrics 計算が全て実装済み |
| ECE 計算 | 手動 binning | ShadowComparisonFramework._compute_ece() | Phase 41 で実装済み。10-bin equal width |
| セグメント定義 | 独自 band 定義 | ShadowDiagnosis の定数 POPULARITY_BAND_EDGES 等 | Phase 43 D-03 で定義済み |

**Key insight:** Phase 44 の新規コード量は最小限。既存の ShadowComparisonFramework + ShadowDiagnosis の拡張または新規分析クラスの追加で対応可能。

## Runtime State Inventory

> Skip: This is a greenfield analysis phase, not a rename/refactor/migration.

## Common Pitfalls

### Pitfall 1: Shadow/Variant Column Name Prefix Confusion
**What goes wrong:** `_align_horse_level()` は `{variant_name}_` prefix（例: `ridge_shadow_p_win_final`）を使用するが、`_align_race_level()` は `shadow_` prefix を使用する。分析コードでプレフィックスを間違えると KeyError
**Why it happens:** 2つのアライメントメソッドが異なる命名規則を使用
**How to avoid:** `ShadowDiagnosis._resolve_variant_col()` を再利用。horse_diff は `ridge_shadow_*`、race_diff は `shadow_*` を使用
**Warning signs:** 分析結果が空または KeyError 発生

### Pitfall 2: horse_diff の n_horses 差異
**What goes wrong:** shadow n_horses=5245 vs baseline n_horses=6662。outer join で shadow-only 馬や baseline-only 馬が存在するため、分析対象を誤るとバイアスが生じる
**Why it happens:** MAWC/Ranker の適用有無で馬数が異なるのではなく、選定結果の差分で outer join が発生
**How to avoid:** 全馬ベース（horse_diff 全体）と selected 馬ベースを明示的に分けて分析
**Warning signs:** n_horses が一致しないことを確認

### Pitfall 3: Ranker は Turf のみ学習済み
**What goes wrong:** `win_race_level_ranker_turf.joblib` のみ存在し、dirt は存在しない。dirt レースを分析しようとすると ranker 効果が測定不能
**Why it happens:** Phase 40 の学習データが turf に限定されたか、dirt は _trained=False だった
**How to avoid:** surface 別に分析を分け、dirt は ranker 効果を「N/A」として扱う
**Warning signs:** ranker coef 抽出で dirt モデルが None

### Pitfall 4: investment_score NaN の解釈
**What goes wrong:** horse_diff で `ridge_shadow_investment_score` が NaN の馬が多い。これは ranker が turf-only かつ IFF 列不足で score() が skip された可能性
**Why it happens:** _ensure_if_surface() が surface 列不在時に動作しない可能性
**How to avoid:** NaN と 0.0 を明確に区別。NaN は「ranker 未適用」を意味する
**Warning signs:** investment_score の非 NaN 率が低い

### Pitfall 5: Ablation 実行の実行時間
**What goes wrong:** ShadowComparisonFramework.run_fold() は ~41 min/year。2 variant 追加で合計 ~4h かかる可能性
**Why it happens:** BT 再実行はモデル推論 + データロードを含むため重い
**How to avoid:** Post-hoc で仮説を絞ってから必要最小限の variant のみ追加実行（D-01）
**Warning signs:** 2 variant 以上の ablation を計画している場合

### Pitfall 6: MAWC beta_market_contribution = 0.90
**What goes wrong:** MAWC の training_summary で beta_market_contribution=0.90（90%が market 由来）。これは logit(p_market) 係数が非常に大きいことを意味し、低オッズ人気馬で p_market が p_model を圧倒する可能性
**Why it happens:** L2 正則化 (C=0.03) が強いが、market signal が dominant
**How to avoid:** 係数分析で logit_market (index 1) とその交互作用項の絶対値を確認。odds_band 1-2 / popularity 1-3 での over-correction を検証
**Warning signs:** 低オッズ馬で p_win_final が p_market に極端に近い

## Code Examples

Verified patterns from actual codebase inspection:

### MAWC Coefficient Extraction (Verified)
```python
# Source: src/models/market_aware_win_calibrator.py + data/models-backtest/2024/
import joblib
mawc_state = joblib.load("data/models-backtest/2024/market_aware_win_calibrator_turf.joblib")
calibrator = mawc_state["calibrator"]        # LogisticRegression
feature_names = mawc_state["feature_names"]  # list[str], 51 items
coef = calibrator.coef_[0]                   # ndarray shape (51,)

# Key coefficient indices (verified from actual feature_names output):
# index 0: logit_model         coef=0.0435
# index 1: logit_market        coef=0.3911  ← dominant
# index 2: log_odds            coef=-0.3571
# index 6: odds_band "1-2"     coef=0.0192
# index 13: pop_1              coef=-0.0291
# index 18: top_25             coef=-0.0263
# Interaction terms (indices 21-50): logit_model_x_* and logit_market_x_*
```

### Ranker Coefficient Extraction (Verified)
```python
# Source: src/models/race_level_ranker.py + data/models-backtest/2024/
ranker_state = joblib.load("data/models-backtest/2024/win_race_level_ranker_turf.joblib")
rel_turf = ranker_state["relevance_scorer_turf"]  # Ridge, coef_ shape (15,)
val_turf = ranker_state["value_scorer_turf"]      # Ridge, coef_ shape (15,)
rel_features = ranker_state["relevance_feature_names"]  # 15 items
val_features = ranker_state["value_feature_names"]      # 15 items

# Key relevance features: if_p_win_final (0.80), if_p_ability_win (0.27), ...
# Key value features: if_ev_calibrated (0.83), if_overround (-0.74), ...
```

### Shadow Horse Diff Column Access (Verified)
```python
# Source: data/backtest/shadow/shadow_horse_diff.parquet
# Columns: ['race_id', 'umaban', 'baseline_p_win_final',
#   'baseline_investment_score', 'baseline_stake',
#   'baseline_win_market_selection_score', 'baseline_selected',
#   'ridge_shadow_p_win_final', 'ridge_shadow_investment_score',
#   'ridge_shadow_stake', 'ridge_shadow_win_market_selection_score',
#   'ridge_shadow_selected', 'kakuteijyuni', 'surface', 'tanodds',
#   'closing_win_odds', 'popularity', 'fold_year']
# Shape: (7368, 18)
```

### VariantConfig for Ablation (Verified Pattern)
```python
# Source: src/backtest/shadow_comparison.py VariantConfig
from backtest.shadow_comparison import ShadowComparisonFramework, VariantConfig, FoldDefinition

variants = [
    VariantConfig("baseline", Path("data/models-backtest"),
                  enable_market_aware_calibrator=False, enable_race_level_ranker=False),
    VariantConfig("mawc_only", Path("data/models-backtest"),
                  enable_market_aware_calibrator=True, enable_race_level_ranker=False),
]
framework = ShadowComparisonFramework(variants=variants, betting_target="win")
results = framework.run(folds=FoldDefinition.create_folds([2024, 2025]))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 手動 BT 比較 | ShadowComparisonFramework (N-way) | Phase 41 | ablation variant 追加が宣言的 |
| 3-step 段階的除外 | ECE→APR→bet_count→OBF 逐次帰属 | Phase 44 (今回) | 因果順序を尊重した分析 |
| MAWC 無し | MAWC あり (C=0.03, 51-dim) | Phase 39 | beta_market=0.90 で market dominant |
| Ranker なし | Ranker あり (shadow_only) | Phase 40 | investment_score 列追加 |

**Deprecated/outdated:**
- WinBenterGate + WinSegmentCalibrator: MAWC に統合済み (Phase 39)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MAWC の coef_ (51-dim) にアクセスしてセグメント別寄与を計算できる | Architecture | coef_ の意味解釈が間違っていれば帰属結果が不正確 |
| A2 | Ranker は turf のみ学習済みで dirt は未学習 | Common Pitfalls | dirt 分析で ranker 効果が測定不能 |
| A3 | ShadowComparisonFramework の VariantConfig 追加だけで ablation が実行できる | Architecture | 既存 BT インフラにバグがあれば ablation が不正確 |
| A4 | Post-hoc 分析で MAWC の p_win 変化を直接測定できる（baseline_p_win_final vs ridge_shadow_p_win_final の差分） | Architecture | MAWC 適用後の p_win 変化が他要因と混在している可能性 |
| A5 | MAWC training_summary.beta_market_contribution=0.90 が logit(p_market) の過大寄与を示唆している | Common Pitfalls | 必ずしも過大とは限らない（market 自体が正確な場合も） |

## Open Questions

1. **MAWC なし→MAWC ありの純粋な p_win 変化をどう測るか**
   - What we know: horse_diff に baseline_p_win_final と ridge_shadow_p_win_final があるが、shadow 側は MAWC+Ranker 両方の効果が混在
   - What's unclear: MAWC のみの純粋効果を post-hoc で分離できるか
   - Recommendation: horse_diff の馬について、baseline_p_win_corrected と ridge_shadow_p_win_final の差を「MAWC 効果 + race normalization 効果」として扱い、Ranker は selection 層の効果として分離する（D-01/D-02 の逐次分析アプローチ）

2. **OddsBandFilter の excluded_bands を baseline/shadow で直接比較する方法**
   - What we know: OBF は BacktestEngine 内部で _generate_training_bet_history から calibrate される
   - What's unclear: 各 variant の OBF excluded_bands を外部から取得する手段が無い
   - Recommendation: ablation 実行時に BacktestEngine のログまたは bet_history から band 別除外を抽出。または post-hoc で bet_history の tanodds 分布から推定

3. **investment_score NaN の実態**
   - What we know: horse_diff の ridge_shadow_investment_score に NaN が多い
   - What's unclear: Ranker が全馬に適用されたか、turf のみか、IFF 列不足で skip されたか
   - Recommendation: Phase 44 実装時に NaN 率を surface 別に確認

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全分析スクリプト | Yes | 3.11 (mise) | - |
| scikit-learn | MAWC/Ranker coef_ | Yes | 1.8.0 | - |
| joblib | model loading | Yes | 1.5.3 | - |
| pandas | DataFrame analysis | Yes | 2.3.3 | - |
| numpy | numerical computation | Yes | 2.4.3 | - |
| data/models-backtest/ | ablation models | Yes | 2023/2024/2025 | - |
| data/backtest/shadow/ | Phase 41 artifacts | Yes | verified | - |
| data/backtest/shadow/diagnosis/ | Phase 43 artifacts | Yes | verified | - |
| data/backtest/shadow/gates/ | Phase 42 artifacts | Yes | verified | - |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** none

## Sources

### Primary (HIGH confidence)
- `src/models/market_aware_win_calibrator.py` — MAWC クラス定義、coef_ アクセス、build_feature_matrix (51-dim schema) [VERIFIED: code inspection]
- `src/models/race_level_ranker.py` — Ranker クラス定義、coef_ アクセス、RELEVANCE_FEATURES/VALUE_FEATURES [VERIFIED: code inspection]
- `src/backtest/shadow_comparison.py` — ShadowComparisonFramework、VariantConfig、run_fold [VERIFIED: code inspection]
- `src/backtest/shadow_diagnosis.py` — ShadowDiagnosis クラス、セグメント定数 [VERIFIED: code inspection]
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator、GatePolicy [VERIFIED: code inspection]
- `src/backtest/race_predictor.py` — RacePredictor、MAWC/Ranker 適用箇所、shadow 診断ブロック [VERIFIED: code inspection]
- `src/betting/odds_band_filter.py` — OddsBandFilter クラス [VERIFIED: code inspection]
- `data/models-backtest/2024/` — 実際の MAWC/Ranker モデルファイル [VERIFIED: artifact inspection]
- `data/backtest/shadow/shadow_horse_diff.parquet` — 18 columns、7368 rows [VERIFIED: artifact inspection]

### Secondary (MEDIUM confidence)
- `data/backtest/shadow/diagnosis/shadow_diagnosis_result.json` — Phase 43 診断結果 [VERIFIED: artifact inspection]
- `data/backtest/shadow/gates/deployment_gate_result.json` — Gate FAIL/WARN 条件 [VERIFIED: artifact inspection]
- `data/backtest/shadow/shadow_comparison_result.json` — Phase 41 メトリクス [VERIFIED: artifact inspection]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 全て既存プロジェクト依存。新規パッケージ不要
- Architecture: HIGH — ShadowComparisonFramework と VariantConfig の ablation 対応を実装済みコードから確認
- Pitfalls: HIGH — 実際の成果物構造とモデルファイルを検証済み
- MAWC/Ranker 係数アクセス: HIGH — 実際に coef_ を抽出して数値を確認済み

**Research date:** 2026-05-30
**Valid until:** 2026-06-30 (stable — no external dependencies change expected)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 段階的ハイブリッド手法（Phase 1: Post-hoc分析 → Phase 2: Targeted ablation、最小限variant、モデル再学習なし）
- **D-02:** 逐次分析順序: ECE → APR → bet_count → OddsBandFilter（並列分析しない）
- **D-03:** MAWC + Ranker 係数分析を中心。上流木モデルは baseline/shadow で同一のため SHAP/gain は主手段にしない。全12モデル比較は範囲外
- **D-04:** OddsBandFilter は独立分析ステップではなく bet_count 分析に統合
- **D-05:** v1.7→v2.0 タグ比較は補助的歴史分析のみ。フル再学習/フル BT は主手段にしない

### Claude's Discretion
- ShadowComparisonFramework への ablation variant 追加方法
- Post-hoc 分析スクリプトの内部メソッド・データフロー設計
- MAWC/Ranker 係数の可視化方法
- テスト構造・命名（既存規約に従う）
- JSON 出力のスキーマ設計（Phase 45 が消費しやすい構造）
- Ablation 実行の具体的な RacePredictor フラグ注入方法

### Deferred Ideas (OUT OF SCOPE)
- OddsBandFilter 再学習・閾値調整（Phase 45 候補）
- MAWC segment 係数に基づく構造的修正（Phase 45 候補）
- Ranker investment_score 重み調整（Phase 45 候補）
- 全12モデル SHAP/gain 比較
- v1.7→v2.0 歴史的ビセクション（補助分析のみ）
- レジーム別分析
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BISECT-01 | DeploymentGate FAIL の4原因（bet_count低下、ECE悪化、APR上振れ、OBF影響）をコンポーネントに帰属 | Post-hoc 分析パターン (Pattern 1) + ablation パターン (Pattern 3) で実現可能。ShadowComparisonFramework の VariantConfig で MAWC only / Ranker only / MAWC+Ranker / baseline のトグルが可能。既存成果物（shadow_horse_diff.parquet の 18 columns、shadow_diagnosis_result.json の step3 セグメント）から post-hoc 帰属が可能 |
| BISECT-02 | MAWC/Ranker の係数分析で確率品質・選定悪化に寄与する特徴量・パラメータを特定 | MAWC coef_ (51-dim, LogisticRegression) と Ranker coef_ (15-dim x 4 Ridge models) へのアクセスを検証済み。feature_names と coef_ の対応が確認済み。beta_market_contribution=0.90 で logit_market が dominant。セグメント別（changed/dropped/retained races）の係数寄与分布比較が可能 |
</phase_requirements>
