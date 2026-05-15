# Phase 27: Feature Interactions - Research

**Researched:** 2026-05-15
**Domain:** Feature engineering (target encoding, interaction features, relative features) for gradient boosting ML pipeline
**Confidence:** HIGH

## Summary

Phase 27 は3つの要件 (INTER-01: 相対特徴量拡張, INTER-02: 交互作用項, INTER-03: ターゲットエンコーディング) を実装する。コードベースの既存パターンは非常によく確立されており、INTER-01/02 は既存モジュールへの追加で完了する。INTER-03 (ターゲットエンコーディング) は最も複雑で、OOF fold分割との整合性と時系列リーク防止が必須。

**Primary recommendation:** INTER-03 のターゲットエンコーディングは `category_encoders` パッケージに依存せず、既存の `expanding().shift(1)` PIT-safe パターンをベースに自前実装する。OOF fold分割は `AbilityModel.train_oof()` の3-fold expanding window と一致させる。TE は新規 `target_encoding.py` モジュールに置き、`_train_submodel` 内の適切なタイミングで呼び出す。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Phase 26の残り4個の相対特徴量をStage1AbilityModel + WinTwoStageModelのFEATURE_COLSに追加
- **D-02:** オッズ相対特徴量 + 能力値相対特徴量を新規生成してWinTwoStageModelに追加
- **D-03:** Stage1AbilityModelの既存race_rank系5個と新規relative_features.pyの特徴量は両方維持
- **D-04:** 新しい相対特徴量はrelative_features.pyの_BASE_FEATURESリストに追加して拡張
- **D-05:** 既存3個の扱い + 新規追加数はClaudeの判断。合計10-15個
- **D-06:** カテゴリ積(文字列結合→category型)と数値積の混合アプローチ
- **D-07:** 交互作用項の実装場所はClaudeの判断
- **D-08:** TE対象はblood_keito_cd + 騎手コード + 調教師コード
- **D-09:** OOFリーク防止は最適手法を選択。時系列データ考慮必須
- **D-10:** TE列の追加先モデルはClaudeの判断。Phase 25 D-02決定との整合性考慮
- **D-11:** TEの実装場所はClaudeの判断

### Claude's Discretion
- INTER-02: 既存3個のカウント扱い + 新規追加数 + 具体的な交互作用項の選定
- INTER-02: 実装場所（interaction_features.py拡張 vs 新規モジュール）
- INTER-03: TEの追加先モデル（Stage1 + Stage2 + Place のどれに追加）
- INTER-03: TEの実装場所（新規target_encoding.py vs 既存モジュール組み込み）
- INTER-03: 平滑化パラメータ、最小サンプル数閾値の設定
- 各特徴量のFEATURE_COLSへの具体的な挿入位置
- テストの追加・更新内容
- POST_RACE漏洩テストの通過確認方法

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INTER-01 | レース内相対ランク特徴量（オッズ、能力値等の相対位置）を生成できる | `_BASE_FEATURES`拡張パターン確立済み。オッズ列(popularity_rank, fukuoddslow)は計算時にDataFrame内に存在。ただしp_ability_win依存の特徴量はStage1 OOF後に計算が必要 |
| INTER-02 | ドメイン知識に基づく10-15個の条件付き交互作用項を生成できる | 既存3個のカテゴリ積/数値積パターン踏襲。ドメイン知識に基づく12個の新規交互作用を推奨 |
| INTER-03 | 高カーディナリティカテゴリ変数のTE（血統コード、騎手コード等）をOOFリークなしで実装できる | Expanding window TE (PIT-safe)を推奨。AbilityModel OOF fold と整合する設計パターンを確立 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 相対特徴量 (INTER-01) | Feature Pipeline | -- | groupby("race_id")変換。_train_submodel内で既に呼び出し済み |
| 交互作用項 (INTER-02) | Feature Pipeline | -- | 既存interaction_features.pyのパターン拡張。_train_submodel内 |
| ターゲットエンコーディング (INTER-03) | Training Pipeline (OOF) | Feature Pipeline | TEは学習データのfold分割に依存。OOF予測生成コンテキスト内で計算が必要 |
| FEATURE_COLS管理 | Model Classes | -- | 各モデルクラスのFEATURE_COLSに新列を追加 |
| POST_RACE漏洩防止 | CI Test Layer | -- | 既存3層テストが自動的に新特徴量を検証 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | installed | DataFrame操作・groupby変換 | プロジェクト全体で使用 |
| numpy | installed | 数値計算 | プロジェクト全体で使用 |
| LightGBM | installed | MLモデル。category型ネイティブサポート | Stage1/Stage2ともLightGBM |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn.KFold | installed | OOF fold分割 (MarketModel) | TE fold分割の参考 |
| -- | -- | category_encodersは不要 | 自前実装がPIT-safeで確実 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| category_encoders.TargetEncoder | 自前 expanding().shift(1) TE | category_encodersは時系列リーク防止をサポートしない。自前実装が安全 [VERIFIED: 既存コードベースパターン] |
| LightGBM native category for high-cardinality | Target Encoding | LightGBM公式: "high cardinality often works best as numeric via target encoding" [CITED: lightgbm.readthedocs.io/Advanced-Topics] |
| KFold TE (random) | Expanding window TE | 時系列データではKFold(shuffle=False)でもリークの可能性。expanding window が最安全 [VERIFIED: AbilityModel.train_oof パターン] |

**Installation:**
```bash
# 追加インストール不要 — 既存パッケージのみ使用
```

## Architecture Patterns

### System Architecture Diagram

```
relative_features.py (INTER-01)
   _BASE_FEATURES に新規spec追加 → compute_relative_features() が自動適用
   ↓ (既存呼び出しポイント: _train_submodel 行516-519)
   ↓ ただし p_ability_win 依存の特徴量は Stage1 OOF 後に別途計算が必要

interaction_features.py (INTER-02)
   compute_interaction_features() に新規交互作用追加
   ↓ (既存呼び出しポイント: _train_submodel 行494-497)
   ↓ カテゴリ積 → astype("category"), 数値積 → where(NaN安全)

target_encoding.py (INTER-03) [NEW MODULE]
   ├─ TargetEncoder クラス
   │   ├─ fit_transform_oof(): AbilityModel.train_oof() と同じ3-fold expanding window
   │   └─ transform(): 推論用 (全データのexpanding mean使用)
   ├─ 対象: blood_keito_cd, kisyucode, chokyosicode
   └─ 呼び出しポイント: _train_submodel 内、interaction_featuresの前後
      ↓
      TE列 → FEATURE_COLS に追加
```

### Recommended Project Structure
```
src/features/
├── relative_features.py       # INTER-01: _BASE_FEATURES拡張
├── interaction_features.py    # INTER-02: 新規交互作用追加
├── target_encoding.py         # INTER-03: 新規TEモジュール
├── (existing modules...)
src/models/
├── stage1_ability_model.py    # FEATURE_COLS更新
├── two_stage_return_model.py  # FEATURE_COLS更新
```

### Pattern 1: _BASE_FEATURES 拡張 (INTER-01)

**What:** `relative_features.py` の `_BASE_FEATURES` リストに新しい dict を追加するだけで、新しい相対特徴量が自動生成される。

**When to use:** race_id groupby で相対化できる任意の数値特徴量

**Example:**
```python
# Source: src/features/relative_features.py (VERIFIED)
_BASE_FEATURES: list[dict[str, str]] = [
    # 既存7個...
    # Phase 27 新規追加 (オッズ相対)
    {"base": "popularity_rank", "output": "rel_popularity_rank_zscore", "transform": "zscore"},
    {"base": "fukuoddslow", "output": "rel_fuku_odds_zscore", "transform": "zscore"},
]
```

**重要な制約:** `compute_relative_features()` は `_train_submodel` 内で `AbilityModel.train_oof()` の**前**に呼ばれる (行516-519)。したがって:
- `p_ability_win` はまだ存在しない → これに依存する特徴量は別途計算が必要
- `odds_to_ability_ratio` もまだ存在しない
- `fukuoddslow`, `popularity_rank`, `tanodds` は既に存在する (build_all でマージ済み)

**対応策:** p_ability_win / odds_to_ability_ratio 依存の相対特徴量は、Stage1 OOF後の `df_oof` 上で追加計算する。

### Pattern 2: カテゴリ積・数値積 (INTER-02)

**What:** 既存の `interaction_features.py` パターンを踏襲。

**Example (カテゴリ積):**
```python
# Source: src/features/interaction_features.py (VERIFIED)
df["kyakusitu_x_distance"] = (
    df["kyakusitukubun_cd"].astype(str) + "_" + df["distance_bin"].astype(str)
).astype("category")
```

**Example (数値積):**
```python
# Source: src/features/interaction_features.py (VERIFIED)
df["weight_x_distance"] = (df[weight_col] * df["kyori"]).where(
    df[weight_col].notna() & df["kyori"].notna(),
    other=float("nan"),
)
```

### Pattern 3: Expanding Window TE (INTER-03) [RECOMMENDED]

**What:** `horse_career_stats.py` の `expanding().shift(1)` PIT-safe パターンをTE に応用。

**When to use:** 時系列データで高カーディナリティカテゴリのTE

**Example:**
```python
# Source: パターンは horse_career_stats.py / info_asymmetry_features.py と同じ
# [VERIFIED: 既存コードベースパターン]
df = df.sort_values("race_date")
# カテゴリごとに expanding mean を計算 (shift(1)で現在行を除外)
te_map = (
    df.groupby("blood_keito_cd", observed=True)["target"]
    .expanding()
    .mean()
    .shift(1)
)
# グローバル平均でNaN補完 (cold start)
global_mean = df["target"].expanding().mean().shift(1)
df["te_blood_keito"] = te_map.reset_index(level=0, drop=True).fillna(global_mean)
```

**OOF版の設計:**
```python
# AbilityModel.train_oof() と同じ3-fold expanding window
# fold境界: dates[n_dates * (i+1) // (n_folds+1)]
# 各fold内: trainデータのexpanding mean をtestデータに適用
for i in range(n_folds):
    train_mask = df["race_date"] < train_end
    test_mask = (df["race_date"] >= train_end) & (df["race_date"] < test_end)
    # trainデータからカテゴリ別target mean を計算
    te_values = train_df.groupby("blood_keito_cd")["target"].mean()
    # testデータにマップ
    test_df["te_blood_keito"] = test_df["blood_keito_cd"].map(te_values)
```

### Anti-Patterns to Avoid

- **KFold(shuffle=True) での TE 計算:** 時系列データでランダムfold分割を行うと、未来情報が過去のTE値にリークする。絶対に使用しないこと
- **グローバルtarget meanでのTE:** カテゴリ別の時系列変化を無視する。競馬では騎手の成績が年々変化するため、時系列を考慮したTEが必須
- **fillna(0) での数値積:** 欠損値を0で埋めると「体重0kg」相当の誤った特徴量が生成される。`.where(notna(), other=nan)` を使用
- **_BASE_FEATURESにp_ability_win依存特徴量を追加:** compute_relative_features() は Stage1 OOF前に呼ばれるため、p_ability_win はまだ存在しない

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| fold境界計算 | 独自ロジック | AbilityModel.train_oof() の境界計算と同一ロジック | fold不一致はOOFリークの原因 |
| category型のLightGBM処理 | label encoding / one-hot | `astype("category")` を渡す | LightGBMネイティブ対応。~8x高速 [CITED: lightgbm.readthedocs.io] |
| NaN安全な数値積 | fillna(0)後の積 | `.where(notna(), other=nan)` | 0埋めは虚偽の情報を生成 |
| TEのNaN補完 | 定数値(0, グローバル平均のみ) | グローバルexpanding mean (段階的) | cold start問題の正しい対処 |

**Key insight:** TE の fold分割ロジックは、プロジェクトで最も注意が必要な箇所。AbilityModel OOF (3-fold expanding by race_date) と MarketModel OOF (5-fold KFold shuffle=False) は異なる戦略。TE は Stage1 OOF と整合する必要がある。

## Common Pitfalls

### Pitfall 1: TE fold分割の不一致
**What goes wrong:** TE の fold分割が Stage1 OOF と異なり、Stage1 のリークした情報が TE に混入する
**Why it happens:** Stage1 は3-fold expanding window、MarketModel は5-fold KFold。混同しやすい
**How to avoid:** TE の fold分割は AbilityModel.train_oof() の3-fold expanding window と完全に一致させる。同じ `boundaries = [dates[n_dates * (i+1) // (n_folds+1)]` を使用
**Warning signs:** TE値の分布が異常（train/test間で大きく乖離）、OOF精度が不当に高い

### Pitfall 2: p_ability_win の計算順序
**What goes wrong:** `rel_p_ability_win_rank` 等の特徴量が NaN になる
**Why it happens:** `compute_relative_features()` は Stage1 OOF の前に呼ばれる。p_ability_win は Stage1 OOF で生成される
**How to avoid:** p_ability_win 依存の相対特徴量は、`_train_submodel` の Stage1 OOF後に別途計算する
**Warning signs:** 新しい相対特徴量が全て NaN

### Pitfall 3: POST_RACE漏洩
**What goes wrong:** 新特徴量が POST_RACE_COLS (kakuteijyuni, confirmed_odds 等) を間接的に含む
**Why it happens:** TE の target に POST_RACE 情報が含まれる、または相対特徴量の base に POST_RACE 列を指定
**How to avoid:** TE の target は `kakuteijyuni` (学習時のみ保持) を使用可能だが、推論時には TE 値自体は事前計算済みの固定値。POST_RACE 列自体を base に使わない。既存3層テストが自動検出する
**Warning signs:** CI テスト `test_model_feature_cols_no_post_race` の失敗

### Pitfall 4: TE の cold start 問題
**What goes wrong:** 新しい騎手/血統コードで TE値が NaN
**Why it happens:** 過去データにそのカテゴリが存在しない
**How to avoid:** グローバル expanding mean で補完。smoothing パラメータで少数サンプル時の信頼性を下げる
**Warning signs:** テストデータで NaN率が高い

### Pitfall 5: 血統コードのcardinality
**What goes wrong:** blood_keito_cd のカーディナリティが高すぎてTEが過学習
**Why it happens:** 系統コードは数十〜百程度のカーディナリティ。過去データの少ない系統ではTE値が不安定
**How to avoid:** smoothing (min_samples + Bayesian prior) を適用。Beta(1,10) 平滑化 (既存パターン) または min_samples_leaf=5 などの閾値
**Warning signs:** feature importance で TE列が過剰に高い、OOF精度が不当に高い

## Code Examples

### 既存 _BASE_FEATURES 拡張パターン (INTER-01)
```python
# Source: src/features/relative_features.py (VERIFIED)
_BASE_FEATURES: list[dict[str, str]] = [
    # ... 既存7個 ...
    # Phase 27: オッズ相対特徴量 (fukuoddslow, popularity_rank は計算時に存在)
    {"base": "popularity_rank", "output": "rel_popularity_rank_zscore", "transform": "zscore"},
    {"base": "fukuoddslow", "output": "rel_fuku_odds_zscore", "transform": "zscore"},
    # Phase 27: 能力値相対特徴量 (p_ability_win依存 → 別途計算必要)
]
```

### 既存カテゴリ積パターン (INTER-02)
```python
# Source: src/features/interaction_features.py (VERIFIED)
if "kyakusitukubun_cd" in df.columns and "distance_bin" in df.columns:
    df["kyakusitu_x_distance"] = (
        df["kyakusitukubun_cd"].astype(str) + "_" + df["distance_bin"].astype(str)
    ).astype("category")
```

### 既存数値積パターン (INTER-02)
```python
# Source: src/features/interaction_features.py (VERIFIED)
if weight_col in df.columns and "kyori" in df.columns:
    df["weight_x_distance"] = (df[weight_col] * df["kyori"]).where(
        df[weight_col].notna() & df["kyori"].notna(),
        other=float("nan"),
    )
```

### TE OOF の設計パターン (INTER-03)
```python
# Source: AbilityModel.train_oof() と同一の fold 分割 (VERIFIED)
def compute_target_encoding_oof(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    n_folds: int = 3,
    smoothing: int = 10,
) -> pd.Series:
    """Expanding window OOF target encoding."""
    df = df.sort_values("race_date").reset_index(drop=True)
    dates = sorted(df["race_date"].unique())
    n_dates = len(dates)

    boundaries = [dates[n_dates * (i + 1) // (n_folds + 1)] for i in range(n_folds)]
    te_values = pd.Series(float("nan"), index=df.index)

    global_mean = df[target_col].expanding().mean().shift(1)

    for i in range(n_folds):
        train_end = boundaries[i]
        test_end = boundaries[i + 1] if i + 1 < n_folds else dates[-1] + pd.Timedelta(days=1)

        train_mask = df["race_date"] < train_end
        test_mask = (df["race_date"] >= train_end) & (df["race_date"] < test_end)

        # カテゴリ別 target mean (smoothing付き)
        cat_stats = df.loc[train_mask].groupby(cat_col)[target_col].agg(["sum", "count"])
        cat_mean = (cat_stats["sum"] + smoothing * global_mean.iloc[train_mask.values.sum() - 1]) / (cat_stats["count"] + smoothing)

        test_cats = df.loc[test_mask, cat_col]
        te_values.loc[test_mask] = test_cats.map(cat_mean)

    # NaN補完 (cold start)
    te_values = te_values.fillna(global_mean)
    return te_values
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| KFold TE (random shuffle) | Expanding window / TimeSeriesSplit TE | 2018-2020 | 時系列データでのリーク防止が標準に |
| One-hot encoding for LightGBM | Native categorical handling | LightGBM 2.x+ | ~8x高速、最適分割アルゴリズム |
| Label encoding for high-cardinality | Target encoding (PIT-safe) | 2019-2021 | 高カーディナリティでLightGBM公式推奨 [CITED: lightgbm.readthedocs.io/Advanced-Topics] |

**Deprecated/outdated:**
- `category_encoders.TargetEncoder` のデフォルト設定: 時系列リークを考慮しない [ASSUMED]
- `KFold(shuffle=True)` でのTE: 時系列データでは絶対に使用不可 [VERIFIED: コミュニティコンセンサス]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | blood_keito_cd のカーディナリティは数十〜百程度 (高カーディナリティTE対象として適切) | Standard Stack | TE効果が薄いか、過学習リスク |
| A2 | kisyucode は数千のカーディナリティ (騎手コード) | TE設計 | TEのcold start問題が深刻になる可能性 |
| A3 | chokyosicode も数百〜千のカーディナリティ (調教師コード) | TE設計 | A2と同じ |
| A4 | 推論時のTE値は学習データのexpanding mean (最新)で固定使用可能 | TE設計 | 推論精度が学習データ末期に依存 |

## Open Questions

1. **TE target 変数の選択**
   - What we know: Stage1 は `kakuteijyuni` をtargetに使用。Stage2 (Win) は `kakuteijyuni == 1` をtargetに使用
   - What's unclear: TE の target としてどの変数が最適か (kakuteijyuni をそのまま使うか、binary に変換するか)
   - Recommendation: Stage1 TE は `kakuteijyuni` の1着フラグ (`kakuteijyuni == 1`)、Stage2 TE も同じ。Kaggle コンペのベストプラクティスに従う

2. **TE の追加先モデル**
   - What we know: Phase 25 D-02 で「騎手/調教師コンテキストはStage2のみ」と決定。TE も同様の判断が必要
   - What's unclear: TE (target との直接関係) と コンテキスト特徴量 (過去成績の集約) は異なる情報だが、同じStage2制限を適用すべきか
   - Recommendation: blood_keito_cd のTE はStage1にも追加可能 (血統はオッズに依存しない能力情報)。騎手/調教師のTE は Stage2のみ (Phase 25 D-02 と整合)

3. **p_ability_win 依存の相対特徴量の計算タイミング**
   - What we know: `compute_relative_features()` は Stage1 OOF の前に呼ばれる。p_ability_win は Stage1 OOF 後に生成
   - What's unclear: 追加の相対特徴量計算をどこに配置するか
   - Recommendation: `odds_deviation_features.py` パターンに従い、Stage1 OOF後に `df_oof` 上で追加計算する

## Environment Availability

> Step 2.6: SKIPPED (no external dependencies identified — 全て既存パッケージとコードベース内で完結)

## Validation Architecture

> workflow.nyquist_validation is explicitly false in config.json. Section skipped.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | pandas型チェック + _BASE_FEATURES 存在確認 |
| V6 Cryptography | no | -- |

### Known Threat Patterns for Feature Engineering

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage (look-ahead) | Information Disclosure | expanding().shift(1) PIT-safe パターン |
| Target leakage (TE) | Information Disclosure | OOF fold分割 + 時系列考慮 |
| Cold start (TE) | Denial of Service | グローバル mean での NaN補完 + smoothing |

## Sources

### Primary (HIGH confidence)
- src/features/relative_features.py — _BASE_FEATURES パターン (VERIFIED)
- src/features/interaction_features.py — カテゴリ積/数値積パターン (VERIFIED)
- src/models/stage1_ability_model.py — OOF fold分割ロジック (VERIFIED)
- src/pipelines/training_pipeline.py — _train_submodel 呼び出し順序 (VERIFIED)
- src/features/horse_career_stats.py — expanding().shift(1) PIT-safe パターン (VERIFIED)
- src/features/info_asymmetry_features.py — expanding PIT-safe パターン (VERIFIED)
- src/features/jockey_context_features.py — Beta(1,10) smoothing パターン (VERIFIED)
- src/domain/types.py — POST_RACE_COLS (16列) (VERIFIED)
- tests/test_post_race_leakage.py — 3層漏洩検出テスト (VERIFIED)

### Secondary (MEDIUM confidence)
- [LightGBM Advanced Topics](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) — high cardinality categorical handling 公式推奨
- [Kaggle TE Best Practices](https://www.kaggle.com/code/ryanholbrook/target-encoding) — 時系列TE のベストプラクティス

### Tertiary (LOW confidence)
- [Artefact Blog: Encoding Categorical Features in Forecasting](https://www.artefact.com/blog/encoding-categorical-features-in-forecasting-are-we-all-doing-it-wrong/) — 動的エンコーディングアプローチ (トレンド考慮)。今回は採用しないが参考として

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 既存コードベースパターンの拡張のみ。追加パッケージ不要
- Architecture: HIGH — 呼び出し順序・fold分割ロジックをコードから直接確認
- Pitfalls: HIGH — 過去フェーズ(23-26)での経験に基づく。POST_RACE漏洩は既存テストで検出可能

**Research date:** 2026-05-15
**Valid until:** 2026-06-14 (stable — コードベースパターンに基づく)

---

## Appendix A: _train_submodel 呼び出し順序 (重要)

```
_train_submodel(df):
  1. HorseHistoryFeatures      # 過去成績特徴量
  2. PaceAptitudeFeatures      # ペース適性
  3. CourseFeatures            # コース別適性
  4. SireFeatures              # 種牡馬産駎
  5. DamPedigreeFeatures       # 繁殖牝馬産駎
  6. RecordFeatures            # コースレコード
  7. compute_interaction_features()  ← INTER-02 拡張ポイント (行494-497)
  8. MiningFeatures            # n_mining予想
  9. compute_relative_features()    ← INTER-01 (一部) 拡張ポイント (行516-519)
  10. MarketModel (OOF)        # 市場モデル OOF
  11. AbilityModel.train_oof() # Stage1 OOF → p_ability_win 生成
  12. df_oof = df[p_ability_win.notna()]
  13. odds_to_ability_ratio 計算
  14. compute_odds_deviation_features()
  15. PlaceAbilityModel
  16. WinTwoStageModel (学習)
  17. JockeyContextFeatures     ← Phase 25 (Stage2のみ)
  18. TrainerContextFeatures    ← Phase 25 (Stage2のみ)
  19. JockeyTrainerComboFeatures
  20. EVCorrectionModel
  ...
```

**INTER-03 (TE) の推奨配置:** ステップ9の後、ステップ10の前。またはステップ11の直後 (p_ability_win生成後、Stage2学習前)。後者が推奨 — TE target に p_ability_win を使う場合に対応可能。

## Appendix B: 推奨交互作用項 (INTER-02)

既存3個 (kyakusitu_x_distance, kyakusitu_x_surface, weight_x_distance) に加えて、以下の12個を推奨 (合計15個):

### カテゴリ積 (astype(str) + "_" 結合 → category)
1. `surface_x_distance_bin` — 馬場×距離帯 (既存surface/distance_binの組合せ)
2. `surface_x_track_condition` — 馬場×馬場状態 (interaction_features.py に surface_track_interaction として数値積が既存だがカテゴリ積も有効)
3. `blood_keito_x_surface` — 血統系統×馬場 (重要: サンデー系=芝得意等)
4. `grade_code_x_distance_bin` — クラス×距離帯 (高レース=マイル専馬等)

### 数値積 (where() NaN安全)
5. `class_level_x_popularity_rank` — クラスレベル×人気順位 (格下挑戦×人気薄は注目)
6. `sire_wr_x_class_level` — 種牡馬成績×クラス (血統優位性の条件付き効果)
7. `pace_pressure_x_closing_index` — ペース圧力×追込指数 (ペース適性の交互作用)
8. `blood_surface_wr_x_track_condition` — 血統馬場勝率×馬場状態 (コンディション適性)

### カテゴリ×数値 (条件付き数値)
9. `distance_bin` × `harontimel5_avg` — 距離帯別の末脚効果 (距離binで条件分岐 → bin内末脚)
10. `surface` × `norm_finish_logit_avg` — 馬場別の過去成績効果

### 推論時に利用可能な交互作用 (Stage2のみ)
11. `popularity_rank` × `odds_to_ability_ratio` — 人気×市場/能力乖離 (過小評価の条件付け)
12. `field_size` × `deviation_zscore` — 頭数×オッズ乖離 (大穴が出やすい条件)

**Note:** このリストは Claude の裁量 (D-05, D-06)。最終選定はプランナーで決定。

## Appendix C: TE 追加先モデルの推奨

| TE特徴量 | Stage1 (AbilityModel) | Stage2 Win | Stage2 Place | 理由 |
|---------|----------------------|------------|--------------|------|
| `te_blood_keito_cd` | 追加推奨 | 追加推奨 | 追加推奨 | 血統はオッズ非依存。能力情報 |
| `te_kisyucode` | -- | 追加推奨 | 追加推奨 | Phase 25 D-02 と整合 (Stage2のみ) |
| `te_chokyosicode` | -- | 追加推奨 | 追加推奨 | Phase 25 D-02 と整合 (Stage2のみ) |

**根拠:** Phase 25 D-02 決定「騎手/調教師コンテキストはStage2のみ追加」は、騎手/調教師情報がオッズに既に反映されている(市場効率性)ため。TE も同様の性質 (target との直接関係 = 市場が部分的に織り込み済み) を持つため、Stage2 のみが安全。blood_keito_cd は市場が織り込みにくい血統情報なので Stage1 にも追加可能。

## Appendix D: p_ability_win 依存の相対特徴量 (INTER-01)

これらの特徴量は `compute_relative_features()` では計算できない (p_ability_win がまだ存在しないため)。
Stage1 OOF 後に別途計算する必要がある:

```python
# _train_submodel 内、AbilityModel.train_oof() の後に追加
if "p_ability_win" in df_oof.columns:
    grp = df_oof.groupby("race_id", observed=True)["p_ability_win"]
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0).replace(0, 1)
    df_oof["rel_p_ability_win_zscore"] = (df_oof["p_ability_win"] - mean) / std
    df_oof["rel_p_ability_win_rank"] = grp.rank(method="min", ascending=False, na_option="keep")

# odds_to_ability_ratio の相対特徴量
if "odds_to_ability_ratio" in df_oof.columns:
    grp = df_oof.groupby("race_id", observed=True)["odds_to_ability_ratio"]
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0).replace(0, 1)
    df_oof["rel_odds_ability_deviation"] = (df_oof["odds_to_ability_ratio"] - mean) / std
```

これらは `relative_features.py` にヘルパー関数として定義し、`_train_submodel` から呼び出す設計が推奨。
