# Phase 23: Safety Gate - Research

**Researched:** 2026-05-11
**Domain:** Data leakage prevention + feature importance auditing in ML pipeline
**Confidence:** HIGH

## Summary

Phase 23 は、Spike調査 (data-leak-phase-20-22) で特定された5件の構造的リスク (M1-M6) を一括修正し、POST_RACE情報漏洩を特徴量パイプラインから完全に排除する。同時に、Phase 24 (Feature Audit & Pruning) で使用するfeature importance監査スクリプトを構築する。

5件のリスクは全て詳細に調査済みで、修正箇所と方法が明確。最も影響範囲が広いのはM1 (build_all()のPOST_RACE_COLS残存) とM2 (CQR blacklist→whitelist)。M3 (EV correction odds不一致) は1行変更で修正可能。M6 (popularity_rank ninki fallback) はフォールバックチェーンの短縮。

監査スクリプトは既存の `analyze_feature_importance.py` と `win_feature_analysis.py` を拡張して全モデル対応にする。LightGBMネイティブの gain importance と、sklearn `permutation_importance` の両方を実装する。sklearn 1.8.0とLightGBM 4.6.0が利用可能。

**Primary recommendation:** 修正は全て局所的 (各1-10行) で、既存テストパターン (mock-based, DB不要) に従う。3層CIテストは `test_post_race_leakage.py` に独立ファイルで追加。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Spike M1-M6の全5件の構造的リスクをPhase 23で一括修正する。ベストプラクティス追求、実装難易度問わず。
- **D-02:** build_all()のキャッシュ書き込み前にPOST_RACE_COLSをドロップする。キャッシュにはクリーンなDataFrameのみ保存。既存dropは安全ネットとして残す。
- **D-03:** CQRの特徴量抽出をブラックリスト方式から明示的FEATURE_COLS (whitelist) に変更。他のモデルと同じ設計パターンに統一。
- **D-04:** M3修正 — 学習時も発走前オッズ (tanodds) を使用。confirmed_odds→tanoddsに変更。
- **D-05:** M6修正 — popularity_rankのフォールバックチェーンからninkiを除去。
- **D-06:** 既存 `scripts/analyze_feature_importance.py` を拡張して全モデル対応にする。新規スクリプトは作成しない。
- **D-07:** 監査対象は全モデル — Stage1AbilityModel, WinTwoStageModel(hit/return), PlaceTwoStageModel, EVCorrectionModel。
- **D-08:** 出力形式はCSV + JSONの両方。
- **D-09:** 3層検証を実装 — build_all()出力 / FEATURE_COLS / predict()入力。
- **D-10:** CIテストは新規ファイル `tests/test_post_race_leakage.py` に配置。

### Claude's Discretion
- build_all()内のPOST_RACE_COLS dropの具体的な実装箇所
- キャッシュキー計算にPOST_RACE_COLS除外が影響するかどうかの判定
- CQRのFEATURE_COLSの具体的な列選定基準
- M3の具体的な修正方法 (confirmed_odds→tanoddsの変更箇所)
- M6のフォールバック先の代替 (ninkiを使わない場合)
- 監査スクリプトのCLI引数設計
- permutation重要度の計算パラメータ (n_repeats, scoring metric)
- テストのfixtureデータとモック構成
- predict()入力検証のテスト実装方法

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SAFE-01 | build_all()出口でPOST_RACE_COLSを確実にドロップするリーク修正 + CQR whitelist + EV correction odds fix + popularity_rank fallback fix | M1-M6の修正箇所を全て特定。build_all():295-300のキャッシュ書き込み前、conformal_ev_model.py:141-146のfeature_cols決定、ev_correction_model.py:370と:266のodds列、feature_engine.py:440-456のフォールバックチェーン |
| SAFE-02 | permutation重要度 + gain重要度を計算するfeature importance監査スクリプト | sklearn 1.8.0のpermutation_importance利用可能。LightGBM 4.6.0のfeature_importance(importance_type="gain")利用可能。既存win_feature_analysis.pyのパターンを拡張 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| POST_RACE列除外 | Feature Engine (MLパイプライン) | Backtest Engine (安全ネット) | 特徴量生成の出口でドロップするのが最適。バックテスト/ペーパートレードの既存dropは防御的残存 |
| CQR whitelist化 | Model層 (conformal_ev_model.py) | Training Pipeline | モデルクラス内のFEATURE_COLS定義が責任所在 |
| EV correction odds修正 | Model層 (ev_correction_model.py) | — | モデルのtrain()とcorrect_ev()でodds列を統一 |
| popularity_rank fallback修正 | Feature Engine | — | _map_basic_features()内のフォールバックロジック |
| Feature importance監査 | Scripts層 | Features層 (win_feature_analysis.py) | CLIスクリプトがエントリポイント。分析ロジックはfeatures層のモジュールを拡張 |
| CI漏洩検証テスト | Tests層 | — | pytest-based、DB不要、全mock |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | MLモデル (gain importance) | プロジェクト全体で使用中 |
| scikit-learn | 1.8.0 | permutation_importance | 既存依存関係に含まれる |
| pandas | (installed) | DataFrame操作 | プロジェクト標準 |
| numpy | (installed) | 数値計算 | プロジェクト標準 |
| pytest | (installed) | テストフレームワーク | CLAUDE.mdで指定 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | (stdlib) | テストダブル | 全テスト (DB不要パターン) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sklearn permutation_importance | ELI5 PermutationImportance | sklearnで十分。ELI5は追加依存 |
| LightGBM pred_contrib (SHAP) | shap パッケージ | 既存コードがTreeSHAP使用。shapパッケージは不要 |

**Installation:**
追加インストール不要 — 全て既存依存関係。

## Architecture Patterns

### System Architecture Diagram

```
                         ┌──────────────────────────────────────────────┐
                         │              Feature Pipeline                 │
                         │                                               │
  race_df ──┐            │  build_all()                                  │
  entry_df ─┤───────────>│    ├── merge + odds replace                   │
  odds_df ──┘            │    ├── _map_basic_features()                  │
                         │    │     └── popularity_rank (M6 fix)         │
                         │    ├── sub-modules (intra, dynamics, ...)     │
                         │    ├── ★ POST_RACE_COLS DROP (M1 fix) ◄── NEW │
                         │    └── cache write (clean data only)          │
                         │            │                                  │
                         │            ▼                                  │
                         │  ┌─────────────────┐                          │
                         │  │  FeatureEngine    │                          │
                         │  │  output (clean)   │─── POST_RACE 0列保証    │
                         │  └─────────────────┘                          │
                         └──────────┬───────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ Model Layer  │    │ Model Layer  │    │  Backtest/Paper   │
   │              │    │              │    │  Trading          │
   │ Stage1Model  │    │ WinTwoStage  │    │                   │
   │ FEATURE_COLS │    │ FEATURE_COLS │    │  POST_RACE drop   │
   │ (whitelist)  │    │ (whitelist)  │    │  (安全ネット)      │
   └──────┬───────┘    └──────┬───────┘    └───────────────────┘
          │                   │
          │    ┌──────────────┤
          ▼    ▼              ▼
   ┌──────────────────────────────────┐
   │  CQR Model (conformal_ev_model)  │
   │                                  │
   │  ★ FEATURE_COLS whitelist (M2) ◄─┤── NEW
   │  (従来: blacklist 437列)          │
   └──────────────────────────────────┘

   ┌──────────────────────────────────────┐
   │  EV Correction Model                 │
   │                                      │
   │  train(): ★ tanodds使用 (M3) ◄──── NEW
   │  correct_ev(): tanodds使用          │
   │  (従来: train=confirmed_odds)        │
   └──────────────────────────────────────┘

   ┌──────────────────────────────────────┐
   │  Audit Script (SAFE-02)              │
   │                                      │
   │  analyze_feature_importance.py       │
   │    ├── --all-models flag             │
   │    ├── gain importance (LightGBM)    │
   │    ├── permutation importance (sklearn)│
   │    └── CSV + JSON output             │
   └──────────────────────────────────────┘
```

### Recommended Project Structure
```
src/
├── domain/types.py                    # POST_RACE_COLS定義 (変更なし)
├── features/
│   ├── feature_engine.py              # ★ M1: drop追加 + M6: fallback修正
│   └── win_feature_analysis.py        # ★ 拡張: permutation importance追加
├── models/
│   ├── conformal_ev_model.py          # ★ M2: whitelist化
│   └── ev_correction_model.py         # ★ M3: odds列修正
scripts/
└── analyze_feature_importance.py      # ★ 拡張: 全モデル対応
tests/
└── test_post_race_leakage.py          # ★ 新規: 3層CIテスト
```

### Pattern 1: POST_RACE Drop at Pipeline Exit
**What:** build_all()のキャッシュ書き込み直前にPOST_RACE_COLSを一括ドロップ
**When to use:** build_all()のreturn直前 (feature_engine.py:295-300の前)
**Example:**
```python
# feature_engine.py build_all() — キャッシュ書き込み直前
# ★ M1 fix: POST_RACE列を確実に除外 (feature leakage prevention)
post_race_present = [c for c in POST_RACE_COLS if c in result_df.columns]
if post_race_present:
    logger.info("Dropping %d POST_RACE columns from build_all() output: %s",
                len(post_race_present), post_race_present)
    result_df = result_df.drop(columns=post_race_present)

# --- Feature Cache Write (PERF-03) — single write point, guaranteed ---
if self._use_cache and _cache_name is not None and not result_df.empty:
```

### Pattern 2: Whitelist-based Feature Selection
**What:** モデルクラスにFEATURE_COLS list[str]を定義し、ブラックリストからホワイトリストに変更
**When to use:** conformal_ev_model.pyの特徴量決定ロジック
**Example:**
```python
# conformal_ev_model.py — whitelist化
class ConformalEVModel:
    FEATURE_COLS: list[str] = [
        # Stage1 出力
        "p_ability_win",
        "p_ability_place",
        # Market Model
        "signed_log_error_win",
        "abs_log_error_win",
        # ... 明示的な列挙
    ]

    def train(self, df_calib, ...):
        if self.feature_cols is None:
            # ★ whitelist使用 — blacklistの自動抽出を廃止
            self.feature_cols = self.FEATURE_COLS.copy()
```

### Pattern 3: 3-Layer CI Verification
**What:** POST_RACE漏洩を3層 (build_all出力 / FEATURE_COLS / predict入力) で検証
**When to use:** test_post_race_leakage.py
**Example:**
```python
# tests/test_post_race_leakage.py
class TestPostRaceLeakage:
    def test_build_all_no_post_race_cols(self):
        """Layer 1: build_all()の出力にPOST_RACE_COLSが含まれない"""
        ...

    def test_model_feature_cols_no_post_race(self):
        """Layer 2: 全モデルのFEATURE_COLSにPOST_RACE_COLSが含まれない"""
        from domain.types import POST_RACE_COLS
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import WinTwoStageModel, PlaceTwoStageModel
        from models.ev_correction_model import EVCorrectionModel

        model_classes = [AbilityModel, WinTwoStageModel, PlaceTwoStageModel, EVCorrectionModel]
        for cls in model_classes:
            overlap = set(cls.FEATURE_COLS) & set(POST_RACE_COLS)
            assert not overlap, f"{cls.__name__}.FEATURE_COLS contains POST_RACE: {overlap}"

    def test_predict_input_no_post_race(self):
        """Layer 3: predict()にPOST_RACE_COLSが渡らない"""
        ...
```

### Anti-Patterns to Avoid
- **ブラックリスト特徴量選択:** 新しい列が追加されるたびに自動的に特徴量に混入するリスク。必ずwhitelistを使用
- **キャッシュ書き込み後のdrop:** キャッシュに汚染データが保存される。dropはキャッシュ書き込み前に行う
- **学習時と推論時で異なるオッズ列:** odds-band scalingに不整合が生じる。必ず同じoddsソースを使用

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Permutation importance | カスタムシャッフル実装 | sklearn.inspection.permutation_importance | スコアリング関数の汎用性、n_repeats統計、並列化サポート |
| SHAP values | カスタムTreeSHAP | LightGBM pred_contrib=True | 既存win_feature_analysis.pyで実績あり |
| Gain importance | 手動feature importance計算 | lgb.Booster.feature_importance(importance_type="gain") | LightGBMネイティブ、高速 |

**Key insight:** sklearn.permutation_importanceはLightGBM Boosterを直接渡せないため、sklearn互換ラッパー (predict function) 経由で使用する。既存のwin_feature_analysis.pyパターンを踏襲。

## Common Pitfalls

### Pitfall 1: キャッシュ無効化の副作用
**What goes wrong:** build_all()でPOST_RACE_COLSをdropすると、キャッシュキーが変わらずキャッシュが古いデータ (POST_RACE列あり) を返す可能性がある
**Why it happens:** キャッシュキーは入力パス+日付範囲+feature_typeに基づき、出力列数に依存しない
**How to avoid:** キャッシュキー計算には影響しない。しかし、既存キャッシュがPOST_RACE列を含むため、初回実行時にキャッシュがヒットして古いデータが返される。対策: feature_typeにバージョンを追加するか、キャッシュ強制無効化
**Warning signs:** テストでbuild_all()のモックが古いキャッシュデータを返す

### Pitfall 2: CQR whitelist選定の難しさ
**What goes wrong:** CQRは437列の特徴量を使用しており、これをwhitelistに変換する際に重要な特徴量を漏らす可能性がある
**Why it happens:** 現在のblacklist方式では「除外セット以外の全numeric」を特徴量としているため、どの列が実際に使用されているかの正確なリストがない
**How to avoid:** CQRモデルのfeature_cols (学習時にself.feature_colsに保存される) を実際の学習済みモデルから抽出し、それをwhitelistのベースにする。training_pipeline.py:903-907で計算されるfeature_colsリストを参考にする
**Warning signs:** whitelist化後のCQRモデル性能が大幅に低下

### Pitfall 3: EV correction train()のconfirmed_oddsはターゲット計算にも使用
**What goes wrong:** M3修正でconfirmed_odds→tanoddsに変更する際、E補正のlog_e_correction計算 (line 266) もconfirmed_oddsを使用しており、これも変更が必要
**Why it happens:** confirmed_oddsはE補正のターゲット (log(actual_odds) - log(predicted_odds)) としても使用されている。ターゲットは実際の払戻金額を反映すべきなので、ここはconfirmed_oddsのままにすべきか、tanoddsにすべきか判断が必要
**How to avoid:** ターゲットのconfirmed_oddsは「1着馬の実際の払戻」を表すため、これは保持。odds-band scaling (line 370) のみをtanoddsに変更する。D-04の意図は「推論時に使えないconfirmed_oddsでscalingしない」ことであり、ターゲット計算は別問題
**Warning signs:** E補正の精度低下 (実際の払戻を正しく反映しなくなる)

### Pitfall 4: PlaceEVCorrectionModelのfukuoddslow (M4)
**What goes wrong:** PlaceEVCorrectionModel.FEATURE_COLS (line 405) にfukuoddslowが含まれており、これがPlace Return Modelのターゲットと同じ
**Why it happens:** fukuoddslowは複勝オッズのスナップショットであり、学習時には利用可能だが、推論時にも利用可能 (発走前スナップショット)。M4はSpike調査で指摘されたが、今回はスコープ外
**How to avoid:** M4はPhase 23のスコープ外 (複勝モデルの修正はOut of scope)。認識のみ
**Warning signs:** 将来のフェーズでPlaceEVCorrectionModelを修正する際に注意

### Pitfall 5: permutation importanceの計算コスト
**What goes wrong:** 全モデル (Stage1 + Win2Stage + Place2Stage + EVCorrection) のpermutation importanceを計算すると、437列のCQRモデルでは非常に遅い
**Why it happens:** permutation importanceは特徴量ごとにn_repeats回のシャッフル推論を行う。437列 x n_repeats=5 = 2,185回の推論
**How to avoid:** n_repeats=5 (デフォルト10から削減)、サンプル数上限を設定 (max_samples=5000)、CQRモデルはwhitelist化後に列数が大幅に減少するので問題は自然解決
**Warning signs:** 監査スクリプトの実行時間が10分を超える

## Code Examples

### M1: build_all() POST_RACE drop (feature_engine.py:294-300)

```python
# 現在のコード (feature_engine.py:295-303):
# --- Feature Cache Write (PERF-03) --- single write point, guaranteed ---
if self._use_cache and _cache_name is not None and not result_df.empty:
    try:
        if store is not None:
            store.write(self._cache_dir, _cache_name, result_df)
            ...

# 修正後 — キャッシュ書き込みの直前にdropを挿入:
from domain.types import POST_RACE_COLS  # ファイル先頭に既に追加済み

# ★ SAFE-01: POST_RACE列を確実に除外 (leakage prevention)
post_race_present = [c for c in POST_RACE_COLS if c in result_df.columns]
if post_race_present:
    result_df = result_df.drop(columns=post_race_present)

# --- Feature Cache Write (PERF-03) --- single write point, guaranteed ---
```

### M3: EV correction odds fix (ev_correction_model.py:370)

```python
# 現在のコード (line 370):
odds_col = "confirmed_odds" if "confirmed_odds" in df.columns else "odds"

# 修正後 — 推論時のodds-band scalingに発走前オッズを使用:
odds_col = "odds"  # ★ M3 fix: 常に発走前オッズを使用 (trainと推論で一貫)

# 注意: train()のE補正ターゲット (line 266) のconfirmed_oddsは保持:
# winners["log_e_correction"] = np.log(
#     winners["confirmed_odds"].clip(lower=self.E_CLIP_FLOOR)  # これは変更しない
# ) - np.log(e_pred_clipped)
```

### M6: popularity_rank fallback fix (feature_engine.py:440-456)

```python
# 現在のコード — tanodds → tanninki → ninki (3段フォールバック)
# 修正後 — tanodds → tanninki (2段フォールバック、ninkiを除去)

if "tanninki" in df.columns:
    tanninki_values = pd.to_numeric(df["tanninki"], errors="coerce")
    usable_tanninki = (
        fallback_mask & tanninki_values.notna() & (tanninki_values > 0)
    )
    df.loc[usable_tanninki, "popularity_rank"] = tanninki_values.loc[
        usable_tanninki
    ]
    fallback_mask = fallback_mask & ~usable_tanninki
# ★ M6 fix: ninki フォールバックを除去 (POST_RACEデータ)
# if "ninki" in df.columns:  ← このブロックを削除
#     ...
if fallback_mask.any():
    logging.getLogger(__name__).warning(
        "popularity_rank missing for %d horses after tanodds/tanninki",
        int(fallback_mask.sum()),
    )
    df.loc[fallback_mask, "popularity_rank"] = float("nan")
```

### SAFE-02: Permutation Importance拡張 (win_feature_analysis.py)

```python
# 新規追加関数:
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

def compute_permutation_importance(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: int = 5000,
) -> pd.DataFrame:
    """sklearn permutation importanceをLightGBM Booster用に計算"""
    feature_names = model.feature_name()

    # sklearn互換のpredict function wrapper
    def predict_fn(X):
        return model.predict(X)

    # サブサンプリング (高速化)
    if len(features_df) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(features_df), max_samples, replace=False)
        X_sample = features_df.iloc[idx]
        y_sample = y[idx]
    else:
        X_sample = features_df
        y_sample = y

    result = permutation_importance(
        predict_fn, X_sample, y_sample,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",  # 回帰モデル用
    )

    return pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    })
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ブラックリスト特徴量選択 | ホワイトリスト (明示的FEATURE_COLS) | Phase 23 (this phase) | 新列追加時の漏洩リスクを排除 |
| confirmed_oddsでodds-band scaling | 発走前odds (tanodds)でscaling | Phase 23 (this phase) | 学習/推論のodds一貫性 |
| ninki フォールバック | tanninkiまで (POST_RACE除外) | Phase 23 (this phase) | レース後情報の混入防止 |
| SHAP only分析 | SHAP + gain + permutation | Phase 23 (this phase) | 包括的特徴量評価 |

**Deprecated/outdated:**
- `_NON_FEATURE_COLS` ブラックリスト (conformal_ev_model.py): whitelist FEATURE_COLSに置き換え
- ninki フォールバック (feature_engine.py:440-456): 削除

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | confirmed_oddsはE補正のターゲット計算には必要 (train() line 266)。odds-band scalingのみをtanoddsに変更 | M3修正 | E補正が実際の払戻を反映しなくなる |
| A2 | CQR whitelist化後も同等の性能を維持できる。feature_colsリストはtraining_pipeline.py:903-907で計算されるリストをベースにする | M2修正 | CQR性能低下 |
| A3 | ninkiフォールバック除去後、tanodds/tanninkiが利用できない馬はNaNになるが、これはLightGBMが処理可能 | M6修正 | 学習データ減少 |
| A4 | permutation importanceのscoringはneg_mean_absolute_errorで妥当 (回帰モデルと分類モデルで使い分けが必要) | SAFE-02 | 誤った重要度ランキング |

## Open Questions

1. **CQR whitelistの具体的な列リスト**
   - What we know: training_pipeline.py:903-907で除外セットベースのfeature_colsを計算している。これは約100-200列程度
   - What's unclear: どの列がCQRに本当に必要か。主モデル出力 (_MODEL_OUTPUT_COLS) は既に除外されているが、残りの列の意味的分類が未検証
   - Recommendation: training_pipeline.pyで実際に計算されるfeature_colsリストをログ出力し、それをFEATURE_COLSのベースにする

2. **permutation importanceのscoring metric**
   - What we know: Stage1 (ranking) はndcg、Win/Place (binary) はroc_auc、Return (regression) はneg_mae、EVCorrection (binary/regression mixed) は複合
   - What's unclear: 統一的なscoring metricが存在するか
   - Recommendation: モデルタイプごとにscoringを自動選択する仕様にする

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全体 | -- | -- | mise |
| lightgbm | SAFE-02 (gain importance) | Yes | 4.6.0 | -- |
| scikit-learn | SAFE-02 (permutation importance) | Yes | 1.8.0 | -- |
| pandas | 全体 | Yes | -- | -- |
| pytest | CIテスト | Yes | -- | -- |

**Missing dependencies with no fallback:**
なし — 全て既存環境で利用可能

**Missing dependencies with fallback:**
なし

## Validation Architecture

> nyquist_validation は config.json で false に設定されているため、本セクションはスキップ。

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | -- |
| V3 Session Management | No | -- |
| V4 Access Control | No | -- |
| V5 Input Validation | Yes | POST_RACE_COLS whitelist/blacklist validation |
| V6 Cryptography | No | -- |

### Known Threat Patterns for ML Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage (POST_RACE混入) | Information Disclosure | 3層検証 (build_all / FEATURE_COLS / predict入力) |
| Feature scope creep (新列自動追加) | Tampering | whitelist方式への統一 |
| Train/test data inconsistency | Repudiation | 同一odds列の使用保証 |

## Sources

### Primary (HIGH confidence)
- ソースコード直接読み込み: feature_engine.py, conformal_ev_model.py, ev_correction_model.py, types.py, race_predictor.py
- Spike調査: .planning/spikes/data-leak-phase-20-22.md — 5件の構造的リスクの詳細分析
- 既存テストパターン: tests/test_leakage.py, tests/test_oof_leakage.py, tests/test_backtest_engine.py

### Secondary (MEDIUM confidence)
- CONTEXT.md:23-CONTEXT.md — ユーザーのロック決定
- REQUIREMENTS.md — SAFE-01, SAFE-02要件定義
- ROADMAP.md — Phase 23 Success Criteria

### Tertiary (LOW confidence)
- なし — 全てソースコードベースの検証済み

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 全て既存依存、バージョン確認済み
- Architecture: HIGH — ソースコード直接読み込み、修正箇所特定済み
- Pitfalls: HIGH — Spike調査で詳細分析済み、既存テストパターン確認済み

**Research date:** 2026-05-11
**Valid until:** 2026-06-11 (stable — コードベース変更なし前提)
