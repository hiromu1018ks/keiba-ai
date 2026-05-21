# Phase 31: Race-Level Aggregation Features - Research

**Researched:** 2026-05-18
**Domain:** Feature Engineering (Race-Level Market Structure Features + Feature Promotion)
**Confidence:** HIGH

## Summary

Phase 31は、レース全体の市場構造を表す6つの新規特徴量(rl_log_odds_entropy, rl_odds_dispersion, rl_top3_odds_gap, rl_top1_odds, rl_favorite_rank_gap, rl_n_horses)を追加し、既に計算済みの2特徴量(implied_prob_hhi, odds_skewness)を全12モデルのFEATURE_COLSに昇格させる。全ての新規特徴量はtanodds(pre-race snapshot)のみに依存し、POST_RACE情報は一切使用しない。

既存の`market_bias_features.py`、`intra_race_features.py`、`race_difficulty_model.py`が確立したgroupby("race_id")パターンを踏襲する。特に`compute_flb_slope()`と`compute_market_bias()`はシャノンエントロピーやインプライド確率計算の参照実装として機能する。build_all()/build_features()パリティは、共通関数をrace_level_features.pyに実装し、両方から呼び出す形で実現する。

**Primary recommendation:** 新規モジュール`src/features/race_level_features.py`に`compute_race_level_features(df)`を実装し、build_all()のcompute_difficulty_score()直後(line ~343)で呼び出す。build_features()内でも同じ関数を呼び出す(単レースDataFrameでもgroupbyが不要な設計)。implied_prob_hhi/odds_skewnessは全12モデルのFEATURE_COLSに追加のみ。freeze_feature_manifest.pyは再実行で対応。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 新規 `src/features/race_level_features.py` を作成。既存 `market_bias_features.py` とは独立
- **D-02:** 6特徴量の定義:
  - `rl_log_odds_entropy`: インプライド確率のシャノンエントロピー `-sum(p * log(p))` where `p = 1/tanodds` normalized per race
  - `rl_odds_dispersion`: レース内tanoddsの標準偏差
  - `rl_top3_odds_gap`: 1番人気と3番人気のtanodds差 (混戦度指標)
  - `rl_top1_odds`: 1番人気のtanodds値をレース内全馬にブロードキャスト (鉄板度)
  - `rl_favorite_rank_gap`: 1番人気と2番人気の対数オッズ差 `log(odds_fav2 / odds_fav1)` (支配度)
  - `rl_n_horses`: 出走頭数 (`field_size`または`umaban`のユニーク数)
- **D-03:** 全特徴量は `tanodds` (pre-race snapshot) のみを使用。`POST_RACE_COLS` に含まれる列は一切使用しない
- **D-04:** build_features() パリティの実現方法はClaudeの判断に委ねる。推奨: 共通関数を `race_level_features.py` に実装し、build_all()とbuild_features()の両方から呼び出す
- **D-05:** 現在のbuild_features()は `_map_basic_features()` のみ呼び出し、サブモジュールをスキップしている。race-level特徴量のみパリティ対応し、他サブモジュールのパリティは今回のスコープ外
- **D-06:** implied_prob_hhi と odds_skewness を **全12モデル** のFEATURE_COLSに追加。現在未登録のモデル(AbilityModel, MarketModel, RegimeDetector等)にも追加
- **D-07:** 両特徴量は `market_bias_features.py:compute_flb_slope()` で既に計算済み。追加作業はFEATURE_COLSへの列名追加のみ
- **D-08:** `rl_favorite_rank_gap = log(odds_fav2 / odds_fav1)` (対数オッズ差) を採用

### Claude's Discretion
- build_features()へのパリティ統合の具体的な実装方法 (共通関数抽出が推奨)
- race_level_features.py の内部関数構成
- 各特徴量のエッジケース処理 (少頭数レース、オッズ欠損等)
- テストケースの具体的な設計

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RLF-01 | rl_log_odds_entropy — インプライド確率のシャノンエントロピー | market_bias_features.pyのmarket_entropy計算と同じパターン。p=1/tanodds正規化後に -sum(p*log(p)) |
| RLF-02 | rl_odds_dispersion — オッズ標準偏差 | groupby("race_id")["tanodds"].transform("std") |
| RLF-03 | rl_top3_odds_gap — 1番人気と3番人気のオッズ差 | popularity_rankまたはtanodds順位でソート後、1位と3位の差 |
| RLF-04 | rl_top1_odds — 1番人気オッズのブロードキャスト | groupby("race_id").transform("first")でtanodds最小値を全馬に割り当て |
| RLF-05 | rl_favorite_rank_gap — 1番人気と2番人気の対数オッズ差 | log(odds_fav2 / odds_fav1) per race, broadcast |
| RLF-06 | rl_n_horses — 出走頭数 | field_size列をそのままブロードキャスト、またはgroupby("race_id").size() |
| RLF-07 | build_all()とbuild_features()の両方でrace-level特徴量計算 | 共通compute関数 + 両パスからの呼び出し |
| EFP-01 | implied_prob_hhi をFEATURE_COLSに昇格 | compute_flb_slope()で既に計算済み。12モデルのFEATURE_COLSに列名追加のみ |
| EFP-02 | odds_skewness をFEATURE_COLSに昇格 | compute_flb_slope()で既に計算済み。12モデルのFEATURE_COLSに列名追加のみ |
| EFP-03 | 昇格特徴量のFEATURE_COLS manifest SHA256更新 | scripts/freeze_feature_manifest.py再実行 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Race-level feature computation | Feature Engine (src/features/) | — | 既存サブモジュールパターン(intra_race, market_bias)に従う |
| build_all() integration | FeatureEngine (src/features/feature_engine.py) | — | バッチ特徴量生成のオーケストレータ |
| build_features() parity | FeatureEngine (src/features/feature_engine.py) | — | 推論パスでの単レース特徴量生成 |
| FEATURE_COLS promotion | Model classes (src/models/) | — | 各モデルのFEATURE_COLSクラス属性更新 |
| Manifest SHA256 update | scripts/freeze_feature_manifest.py | — | 特徴量凍結manifestの再生成 |
| POST_RACE leakage test | tests/test_post_race_leakage.py | — | Layer 1/2検証の拡張 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | (existing) | DataFrame操作、groupby集計 | プロジェクト全体で使用中 |
| numpy | (existing) | 数値計算、log/sqrt | プロジェクト全体で使用中 |
| pytest | (existing) | テストフレームワーク | プロジェクト標準 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| lightgbm | (existing) | FEATURE_COLS定義の参照先 | モデルクラスのFEATURE_COLS更新時 |
| hashlib | (stdlib) | SHA256 manifest | freeze_feature_manifest.py |

**Installation:** 追加パッケージなし — 全て既存依存関係で完結

## Package Legitimacy Audit

> このフェーズは外部パッケージをインストールしないため、Package Legitimacy Gate は不要。

**No external packages to install.**

## Architecture Patterns

### System Architecture Diagram

```
build_all() (batch training path)
  │
  ├── _map_basic_features()        ← 既存
  ├── compute_intra_race_features() ← 既存
  ├── compute_odds_dynamics()       ← 既存
  ├── compute_market_bias()         ← 既存 (market_entropy, overround, p_market_win_adj)
  ├── compute_flb_slope()           ← 既存 (odds_skewness, implied_prob_hhi)
  ├── compute_difficulty_score()    ← 既存
  │
  ├── compute_race_level_features() ★ NEW ★ (RLF-01~06)
  │     │
  │     ├── _calc_log_odds_entropy()    → rl_log_odds_entropy (per-race broadcast)
  │     ├── _calc_odds_dispersion()     → rl_odds_dispersion (per-race broadcast)
  │     ├── _calc_top3_odds_gap()       → rl_top3_odds_gap   (per-race broadcast)
  │     ├── _calc_top1_odds_broadcast() → rl_top1_odds       (per-race broadcast)
  │     ├── _calc_favorite_rank_gap()   → rl_favorite_rank_gap(per-race broadcast)
  │     └── _calc_n_horses()            → rl_n_horses        (per-race broadcast)
  │
  └── SAFE-01: POST_RACE列除外 ← 既存

build_features() (inference path)
  │
  ├── _map_basic_features()        ← 既存
  │
  └── compute_race_level_features() ★ NEW ★ (RLF-07 parity)
        │
        └── 同じ関数、単レースDataFrameで動作

FEATURE_COLS Promotion (EFP-01~03)
  │
  ├── 12モデルクラス.FEATURE_COLS += ["implied_prob_hhi", "odds_skewness"]
  └── freeze_feature_manifest.py → SHA256更新
```

### Recommended Project Structure
```
src/
├── features/
│   ├── race_level_features.py     # ★ NEW: compute_race_level_features()
│   ├── feature_engine.py          # 変更: build_all/build_featuresに呼び出し追加
│   ├── market_bias_features.py    # 変更なし (implied_prob_hhi/odds_skewnessは既に計算済み)
│   └── ... (既存サブモジュール)
├── models/
│   ├── stage1_ability_model.py     # 変更: FEATURE_COLSに2列追加
│   ├── two_stage_return_model.py   # 変更: WinTwoStage/PlaceTwoStageに2列追加
│   ├── ev_correction_model.py      # 変更: EVCorrection/PlaceEVCorrectionに2列追加
│   ├── conformal_ev_model.py       # 変更: ConformalEVに2列追加
│   ├── market_model.py             # 変更: MarketModelに2列追加
│   ├── regime_detector.py          # 変更: RegimeDetectorに2列追加
│   ├── place_ability_model.py      # 変更: PlaceAbilityModelに2列追加
│   ├── race_quality_screener.py    # 変更: RaceQualityScreenerに2列追加
│   └── wide_two_stage_model.py     # 変更: WideTwoStageModelに2列追加
tests/
├── test_race_level_features.py    # ★ NEW: 単体テスト
└── test_post_race_leakage.py      # 拡張: 新特徴量のPOST_RACE検証
scripts/
└── freeze_feature_manifest.py     # 再実行のみ (コード変更不要)
```

### Pattern 1: サブモジュールパターン (既存パターンの踏襲)
**What:** 独立した`compute_*()`関数を定義し、build_all()内でTimingContext付きで呼び出す
**When to use:** 新しい特徴量カテゴリの追加時
**Example:**
```python
# src/features/race_level_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース構造特徴量を計算 (RLF-01~06)

    Args:
        df: race_id, tanodds, popularity_rank, field_size を含むDataFrame

    Returns:
        rl_* 列が追加されたDataFrame
    """
    df = df.copy()

    if "tanodds" not in df.columns:
        for col in ["rl_log_odds_entropy", "rl_odds_dispersion", "rl_top3_odds_gap",
                     "rl_top1_odds", "rl_favorite_rank_gap", "rl_n_horses"]:
            df[col] = np.nan
        return df

    tanodds = pd.to_numeric(df["tanodds"], errors="coerce")
    # ... groupby("race_id") で計算 ...
    return df
```

### Pattern 2: groupby("race_id").transform() によるブロードキャスト
**What:** レース単位の集約値を全馬にブロードキャスト
**When to use:** レース全体の統計量(entropy, std等)を各馬に行付与する場合
**Example:**
```python
# Source: market_bias_features.py lines 36-41
overround = p_raw.groupby(df["race_id"], observed=True).transform("sum") - 1.0
p_sum = p_raw.groupby(df["race_id"], observed=True).transform("sum")
df["p_market_win_adj"] = p_raw / p_sum.replace(0, np.nan)
```

### Pattern 3: groupby("race_id").apply() による複雑なレース単位計算
**What:** レース単位で複数の統計量を同時に計算
**When to use:** 1番人気/2番人気/3番人気の特定など、順位に基づく抽出が必要な場合
**Example:**
```python
# Source: market_bias_features.py lines 69-86
shapes = race_feat_df.groupby("race_id", observed=True).apply(_race_shape, include_groups=False)
result["odds_skewness"] = race_feat_df["race_id"].map(shapes.map(lambda x: x[0])).fillna(0.0)
result["implied_prob_hhi"] = race_feat_df["race_id"].map(shapes.map(lambda x: x[1])).fillna(0.0)
```

### Anti-Patterns to Avoid
- **groupby("race_id")に依存する関数をbuild_features()で直接使う:** build_features()は単一レースのDataFrame(全行が同じrace_id)を処理するが、race_id列が存在しないケースも想定すべき。共通関数はrace_id有無で分岐させるか、単レースでも動作するよう設計する
- **POST_RACE_COLSの使用:** rl_*特徴量の計算にkakuteijyuni, confirmed_odds, ninki等を使用すると漏洩テストが落ちる。tanoddsのみを使用
- **FEATURE_COLSの重複追加:** implied_prob_hhiはEVCorrectionModel, PlaceEVCorrectionModel, ConformalEVModelに既に含まれている。これらのモデルには追加不要 — 追加するとリスト内重複になる

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| シャノンエントロピー計算 | ゼロからのループ実装 | market_bias_features.pyの_entropy計算パターン + numpy | 数値安定性(0確率のlog回避)の既存実装を踏襲 |
| 人気順位計算 | カスタムランキング | _compute_popularity_rank_from_tanodds() | 既存関数。tanodds順位計算のPOST_RACE安全な実装 |
| feature manifest生成 | 手動JSON構築 | scripts/freeze_feature_manifest.py | SHA256計算、決定論的JSON生成の既存仕組み |

**Key insight:** 全ての計算パターン(entropy, groupby, broadcast)は既存コードに参照実装が存在する。新しいアルゴリズムは不要。

## Common Pitfalls

### Pitfall 1: FEATURE_COLS重複追加
**What goes wrong:** implied_prob_hhi/odds_skewnessが既にFEATURE_COLSに含まれるモデルに再度追加すると、LightGBMが重複列名でエラーを起こすか、意図しない挙動を示す
**Why it happens:** 一部のモデル(EVCorrectionModel, PlaceEVCorrectionModel, ConformalEVModel)には既に含まれている
**How to avoid:** 各モデルのFEATURE_COLSを事前に確認し、既に含まれている場合は追加しない。set()で排他判定
**Warning signs:** モデル学習時のLightGBMエラー、テスト失敗

### Pitfall 2: build_features()でのrace_id未存在
**What goes wrong:** build_features()で作成されるDataFrameはrace_idを含むが、groupby("race_id")は単一値のみで不要なオーバーヘッド。まれにrace_idが欠落するエッジケースも考慮が必要
**Why it happens:** build_features()は単一レース処理なのでrace_idは全行同じ
**How to avoid:** 共通関数内でrace_idの有無を判定し、単レースの場合はgroupbyなしで直接計算
**Warning signs:** build_features()が空の結果を返す、NaNだらけの特徴量

### Pitfall 3: 少頭数レースでのNaN/inf
**What goes wrong:** 出走頭数が2頭以下の場合、std()がNaN、log(0)が-infになる
**Why it happens:** 統計量の計算に最低サンプル数が必要
**How to avoid:** 各特徴量計算でlen(group) < 2やlen < 3のガード。fillna(0)やclipで安全な値にフォールバック
**Warning signs:** LightGBMがNaN列で警告、学習時のinf値エラー

### Pitfall 4: rl_top3_odds_gapでの3番人気不在
**What goes wrong:** 出走頭数が2頭の場合、3番人気が存在せずNaNになる
**Why it happens:** 2頭立てレースは稀だが存在する
**How to avoid:** 3番人気が存在しない場合はNaNのまま(LightGBMはNaN処理可能)または0.0にフォールバック
**Warning signs:** テストでのNaNアサーション失敗

### Pitfall 5: tanodds欠損時のフォールバック不備
**What goes wrong:** tanodds列が存在するが全てNaN/0の場合、インプライド確率が計算不能
**Why it happens:** オッズがまだ発表されていないレース
**How to avoid:** compute_flb_slope()のパターンを踏襲 — `.replace(0, np.nan).dropna()`で安全に処理
**Warning signs:** inf値の伝播、entire column NaN

## Code Examples

### Shannon Entropy Calculation (参照実装)
```python
# Source: src/features/market_bias_features.py lines 44-49
def _calc_entropy(group: pd.Series) -> float:
    p = group.values.astype(float)
    p = p[p > 0]  # log(0) を回避
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))
```

### groupby().apply() for Multi-Output Race Stats (参照実装)
```python
# Source: src/features/market_bias_features.py lines 69-86
def _race_shape(group):
    if len(group) < 2:
        return 0.0, 0.0
    odds = group["tanodds"].replace(0, np.nan).dropna().values.astype(float)
    if len(odds) < 2:
        return 0.0, 0.0
    skewness = float(pd.Series(odds).skew()) or 0.0
    inv_odds = 1.0 / odds
    total = inv_odds.sum()
    if total == 0:
        return skewness, 0.0
    p = inv_odds / total
    hhi = float(np.sum(p ** 2))
    return skewness, hhi

shapes = race_feat_df.groupby("race_id", observed=True).apply(_race_shape, include_groups=False)
result["odds_skewness"] = race_feat_df["race_id"].map(shapes.map(lambda x: x[0])).fillna(0.0)
result["implied_prob_hhi"] = race_feat_df["race_id"].map(shapes.map(lambda x: x[1])).fillna(0.0)
```

### build_all() Integration Point
```python
# Source: src/features/feature_engine.py lines 340-343
from features.race_difficulty_model import compute_difficulty_score

with TimingContext("build_all/difficulty"):
    result_df = compute_difficulty_score(result_df)

# ★ ここに race-level features の呼び出しを追加
from features.race_level_features import compute_race_level_features

with TimingContext("build_all/race_level"):
    result_df = compute_race_level_features(result_df)
```

### POST_RACE Leakage Test Pattern
```python
# Source: tests/test_post_race_leakage.py lines 76-90
def test_build_all_output_no_post_race_cols(self) -> None:
    engine = FeatureEngine(use_cache=False)
    result = engine.build_all(race_df, entry_df, odds_df)
    post_race_in_output = set(result.columns) & set(POST_RACE_COLS)
    assert not post_race_in_output, (
        f"POST_RACE_COLS found in build_all() output: {post_race_in_output}"
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| race-level特徴量なし | レース構造特徴量(entropy等) | v5.1~ | market_entropy/overroundは既にmarket_bias_features.pyで実装済み |
| implied_prob_hhi/odds_skewness未登録 | FEATURE_COLSへの昇格 | Phase 31 | 計算済み特徴量がモデルに入力されるように |

**Deprecated/outdated:**
- なし — このフェーズは全て新規追加または昇格

## Assumptions Log

> 全ての発見はコードベースの直接確認に基づく。外部パッケージ依存なし。

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | implied_prob_hhiがEVCorrectionModel, PlaceEVCorrectionModel, ConformalEVModelのFEATURE_COLSに既に含まれる | EFP-01 | 低 — コード確認済みだが、念のため各モデルで再確認が必要 |
| A2 | odds_skewnessがWinTwoStageModel, PlaceTwoStageModelのFEATURE_COLSに既に含まれる | EFP-02 | 低 — コード確認済み |
| A3 | build_features()はサブモジュールを一切呼び出さず、_map_basic_features()のみ | RLF-07 | 低 — feature_engine.py line 453-455で確認済み |
| A4 | MarketModel.FEATURE_COLSにはimplied_prob_hhi/odds_skewnessが含まれない | EFP-01 | 低 — market_model.py line 21-31で確認、7列のみ |

## Open Questions (RESOLVED)

1. **AbilityModelはオッズ特徴量を一切使用しない(Rule 1) — implied_prob_hhi/odds_skewnessの追加は矛盾しないか?**
   - What we know: AbilityModel.FEATURE_COLSのdocstringに「オッズ特徴量は一切使用しない (Rule 1)」とある。implied_prob_hhiとodds_skewnessはtanoddsから計算されるため、厳密にはオッズ由来特徴量
   - RESOLVED: CONTEXT.md D-06でユーザーが「全12モデル」への追加を明示的に決定済み。この決定に従い、AbilityModelのdocstringを「Rule 1: オッズ特徴量(implied_prob_hhi, odds_skewness)は市場構造指標として含む (D-06)」に更新する

2. **rl_n_horses と 既存 field_size の関係性**
   - What we know: field_sizeは_map_basic_features()でsyussotosuから計算済み。rl_n_horsesは同一情報の可能性
   - RESOLVED: D-02で「field_sizeまたはumabanのユニーク数」と定義済み。実装ではfield_sizeをそのまま使用し、0の場合にgroupby("race_id").size()で補完する設計を採用

## Environment Availability

> このフェーズは外部依存関係を持たない。純粋なPython/pandasコード変更のみ。

Step 2.6: SKIPPED (no external dependencies identified)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | pandas数値変換 + NaNガード |
| V6 Cryptography | no | — |

### Known Threat Patterns for Feature Engineering

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| POST_RACE情報漏洩 | Information Disclosure | POST_RACE_COLS whitelist + 3層テスト |
| Feature computation NaN/inf propagation | Denial of Service | .replace(0, np.nan).dropna() ガード |
| FEATURE_COLS重複によるLightGBMエラー | Tampering | 重複チェック追加(set判定) |

## Sources

### Primary (HIGH confidence)
- src/features/feature_engine.py — build_all() lines 198-383, build_features() lines 385-455
- src/features/market_bias_features.py — compute_flb_slope() lines 57-87, compute_market_bias() lines 15-54
- src/features/intra_race_features.py — compute_intra_race_features() lines 11-33
- src/features/race_difficulty_model.py — compute_difficulty_score() lines 28-59
- src/models/stage1_ability_model.py — FEATURE_COLS lines 28-148
- src/models/two_stage_return_model.py — WinTwoStageModel/PlaceTwoStageModel FEATURE_COLS
- src/models/ev_correction_model.py — EVCorrectionModel/PlaceEVCorrectionModel FEATURE_COLS
- src/models/conformal_ev_model.py — ConformalEVModel.FEATURE_COLS lines 81-149
- src/models/market_model.py — MarketModel.FEATURE_COLS lines 21-31
- src/models/regime_detector.py — RegimeDetector.FEATURE_COLS lines 49-62
- src/models/place_ability_model.py — PlaceAbilityModel.FEATURE_COLS lines 26-100
- src/models/race_quality_screener.py — RaceQualityScreener.FEATURE_COLS lines 23-53
- src/models/wide_two_stage_model.py — WideTwoStageModel.SHARED_FEATURE_COLS lines 44-50
- src/domain/types.py — POST_RACE_COLS lines 38-55
- tests/test_post_race_leakage.py — 3層テストアーキテクチャ
- scripts/freeze_feature_manifest.py — manifest生成スクリプト

### Secondary (MEDIUM confidence)
- なし — 全てコードベース直接確認

### Tertiary (LOW confidence)
- なし — 外部ソース参照不要

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 既存パッケージのみ、追加なし
- Architecture: HIGH — 既存サブモジュールパターンの踏襲、コードベース直接確認
- Pitfalls: HIGH — FEATURE_COLS重複・POST_RACE漏洩・少頭数エッジケースは明確

**Research date:** 2026-05-18
**Valid until:** 2026-06-17 (stable — 純粋な内部コード変更)
