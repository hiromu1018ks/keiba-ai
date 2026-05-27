# Phase 38: InvestmentFeatureFrame - Research

**Researched:** 2026-05-27
**Domain:** 投資判断用統合特徴量フレーム構築 (モデル出力+市場+OOF統合)
**Confidence:** HIGH

## Summary

Phase 38は、既存のMLパイプラインが生成するモデル出力・市場特徴量・OOF予測を統合し、投資判断に特化した90-130列の構造化特徴量フレーム(InvestmentFeatureFrame)を新規パッケージ`src/investment/`に構築するフェーズである。既存の7モデル(AbilityModel, WinTwoStageModel, PlaceTwoStageModel, EVCorrectionModel, PlaceEVCorrectionModel, MarketModel, ConformalEVModel)が246種類の特徴量を使用し、さらに31種類のモデル出力列を生成する。これらの中から投資判断に直接寄与する特徴量を選別・派生させ、9カテゴリに分類する。

データフローは2つのモード(train/infer)で異なるソース列を参照する。train modeはOOF-safe列(`p_win_oof`, `p_win_corrected` from OOF EV correction)のみを使用し、infer modeは本番列(`p_win_pred`, `ev_win_calibrated`, `p_win_final`)を使用する。出力スキーマ(列名・列順・dtype)は両モードで同一であり、これが安全性の要となる。

Phase 37で構築された`OOFHealthValidator`と`load_validated_oof()`が、train modeのOOF artifact消費時の健全性確認に直接利用できる。既存の`POST_RACE_COLS`(41列)に基づく3層漏洩テストパターン(`test_post_race_leakage.py`)を拡張して、InvestmentFeatureFrame独自の漏洩テストを実装する。

**Primary recommendation:** `src/investment/`パッケージに5モジュール(feature_frame.py, schema_registry.py, manifest.py, cache.py, leakage.py)を配置し、InvestmentFeatureSpec frozen dataclassで全特徴量のメタデータを定義。build_frame(df, mode)単一APIでデュアルモード対応。既存のParquetStoreパターンとOOFHealthValidatorのmanifestパターンを踏襲する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Phase 38はInvestmentFeatureFrame構築のみ。CAL-01~05はPhase 39、RankerはPhase 40に移行
- **D-02:** レジーム伝播/regime-dependent calibrationはv2.0全体でスコープ外
- **D-03:** ROI検証はPhase 38の成功基準ではない
- **D-04:** 軽量スモークテストのみ許可(パイプラインインターフェース破損確認用)
- **D-05:** 全9カテゴリ実装: model_prob(8-12), market_prob(6-10), model_market_gap(10-16), race_relative(12-18), odds_band(6-10), late_odds(8-12), ability_form(15-25), course_pace(10-18), uncertainty(10-16)
- **D-06:** Required-core + optional-extension設計。初期ターゲット90-130列、上限150列
- **D-07:** 信号密度重視の配分: model_market_gap, race_relative, uncertainty, ability_formに多くの容量
- **D-08:** FeatureEngine出力のパススルーではない。投資判断に特化した選別+派生のみ
- **D-09:** 各特徴量にメタデータ必須: category, source columns, train/infer source behavior, missing behavior, leakage classification, dtype, stable output name
- **D-10:** 単一API: `build_frame(df, mode=Literal["train", "infer"])` + thin convenience wrappers
- **D-11:** モード自動検出禁止。明示的mode引数必須
- **D-12:** 出力スキーマ同一性: 同じ列名・列順・dtype・feature_version・missing-indicator動作
- **D-13:** train mode: OOF-safe列(p_win_oof, p_win_final_oof)のみ使用。p_win_predは拒否
- **D-14:** infer mode: 本番列(p_win_pred, p_win_final)を使用
- **D-15:** テストでtrain/inferスキーマ同一性をアサート
- **D-16:** InvestmentFeatureSpec frozen dataclass: name, category, dtype, train_sources, infer_sources, required, default_value, missing_indicator, leakage_class, description
- **D-17:** コードが真実の情報源(YAMLはドキュメント用ミラー)
- **D-18:** Builderはモード別にソース解決: train→train_sources, infer→infer_sources
- **D-19:** required featureのsource欠損時はfail-fast。optionalはdefault_value + *_missing indicator
- **D-20:** テスト検証: 全featureにspecあり、train sourceにin-sample-only列なし、train/infer同一スキーマ、required feature fail-fast
- **D-21:** Parquetキャッシュ + sidecar manifest JSON
- **D-22:** キャッシュパス: `data/features/investment_frame/{mode}/{feature_version}_{source_artifact_hash}_{schema_hash}.parquet`
- **D-23:** キャッシュキー: mode, feature_version, source_artifact_hash, source_schema_hash, output_schema_hash, source OOF health manifest path/hash (train mode), builder_version
- **D-24:** キャッシュ読込時はsidecar manifest + output schema_hash検証
- **D-25:** メモリキャッシュは補助的のみ
- **D-26:** 決定性要件: 同一入力+同一builder_version+同一feature_version→同一出力
- **D-27:** テスト: stable row order (race_id/umaban), stable column order
- **D-28:** Phase 38成功基準: スキーマ正確性、OOF安全性、POST_RACE非混入、決定性、キャッシュ/manifest正確性
- **D-29:** VAL-01はInvestmentFeatureFrame対象にスコープ
- **D-30:** manifest要件: feature_version, schema_hash, source_artifact_hash, source OOF health manifest path, builder_version, mode, generated_at
- **D-31:** VAL-02~05 → Phase 39/40に移行
- **D-32:** VAL-06 → 廃止、v2.0 artifact manifestに置き換え
- **D-33:** 新規パッケージ `src/investment/`: __init__.py, feature_frame.py, schema_registry.py, manifest.py, cache.py, leakage.py
- **D-34:** 公開API: InvestmentFeatureFrameBuilder, InvestmentFeatureSpec, InvestmentFrameManifest, build_frame()
- **D-35:** FeatureEngineとモデルレイヤーから独立

### Claude's Discretion
- 各カテゴリの具体的な特徴量選定
- 派生特徴量の計算式
- キャッシュinvalidationの具体ロジック
- sidecar manifest JSONの完全スキーマ
- テストケースの設計詳細
- ビルダーの内部アーキテクチャ
- frame_builder.py vs feature_frame.py のファイル名
- OOF health manifest との統合インターフェース
- Phase 37のOOFHealthValidatorとの接続方法

### Deferred Ideas (OUT OF SCOPE)
- 人気帯キャリブレーション (CAL-01~05) → Phase 39
- レジーム伝播 → v2.0全体でスコープ外
- Race-Level Ranker → Phase 40
- BT 2024 ROI検証 (VAL-04) → Phase 39/40
- 芝IC b_difference確認 (VAL-02) → Phase 39/40
- 芝pop 4-12 ratio (VAL-03) → Phase 39/40
- Turf conservative ROI (VAL-05) → Phase 39/40
- v1.8 Manifest凍結 (VAL-06) → 廃止
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IFF-01 | InvestmentFeatureFrameBuilder.build_frame(df, mode) が9カテゴリ90-130列の投資特徴量を生成 | 9カテゴリの特徴量マッピング(下記Category Feature Selection参照)。既存モデル246種類FEATURE_COLS + 31種類出力列から選別+派生 |
| IFF-02 | train modeはOOF-safe列のみ、in-sample列を拒否。infer modeは本番列。同一出力スキーマ | デュアルモード設計(下記Dual Mode Design参照)。OOF列: p_win_oof, ev_win_corrected(OOF)。本番列: p_win_pred, ev_win_calibrated, p_win_final |
| IFF-03 | train/infer出力スキーマ同一性: 同一列名・列順・dtype | Schema Registry設計(下記Schema Registry参照)。InvestmentFeatureSpecで出力列名を固定 |
| IFF-04 | InvestmentFeatureSpec frozen dataclassによるスキーマレジストリ | frozen dataclass設計(下記Schema Registry参照)。9つのメタデータフィールド |
| IFF-05 | POST_RACE列非混入。漏洩テスト通過 | 3層漏洩検出パターン(下記Leakage Detection参照)。POST_RACE_COLS 41列との分離 |
| IFF-06 | Parquetキャッシュ + sidecar manifest JSON | ParquetStoreパターン(下記Caching Patterns参照)。manifestパターンはOOFHealthValidator.generate_manifest()を踏襲 |
| IFF-07 | artifact manifest: feature_version, schema_hash, source_artifact_hash等 | Manifest設計(下記Manifest Patterns参照)。OOF manifest D-10/D-11パターンを拡張 |
| VAL-01 | 3層CI漏洩テストをInvestmentFeatureFrameに適用 | 既存テストパターン(下記Leakage Detection参照)。test_post_race_leakage.pyの3層構造 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Feature frame building (9 categories) | API / Backend | - | モデル出力+市場データの統合変換ロジック |
| Schema registry (InvestmentFeatureSpec) | API / Backend | - | 型安全な特徴量定義、コードが情報源 |
| OOF artifact consumption (train mode) | API / Backend | Database / Storage | load_validated_oof()経由でParquetから読込 |
| Manifest generation | Database / Storage | - | SHA256ハッシュ、JSON永続化 |
| Parquet caching | Database / Storage | - | ParquetStoreパターン踏襲 |
| Leakage detection | API / Backend | - | POST_RACE_COLSとの分離検証 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | [ASSUMED - project dependency] | DataFrame操作 | プロジェクト全体で使用 |
| numpy | [ASSUMED - project dependency] | 数値計算 | プロジェクト全体で使用 |
| pyarrow | [ASSUMED - project dependency] | Parquet読み書き | ParquetStoreで使用済み |
| dataclasses (stdlib) | Python 3.11 | frozen dataclass | InvestmentFeatureSpec定義用 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib (stdlib) | Python 3.11 | SHA256ハッシュ | manifest schema_hash, artifact_hash |
| json (stdlib) | Python 3.11 | manifest JSON | sort_keys=True, indent=2で決定論的出力 |
| typing.Literal | Python 3.11 | mode="train"|"infer" 型制約 | build_frame()のmode引数 |

### No New External Packages Required
このフェーズは新規外部パッケージをインストールしない。全て標準ライブラリ + プロジェクト既存依存関係で完結する。

## Package Legitimacy Audit

> このフェーズは外部パッケージをインストールしないため、Package Legitimacy Gateは不要。

**Packages to install:** なし (標準ライブラリ + 既存依存関係のみ)

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────────────┐
                          │    Pipeline Consumer         │
                          │ (TrainingPipeline /           │
                          │  RacePredictor / Backtester)  │
                          └──────────┬──────────────────┘
                                     │ build_frame(df, mode=)
                                     ▼
                    ┌────────────────────────────────┐
                    │  InvestmentFeatureFrameBuilder  │
                    │  (src/investment/               │
                    │   feature_frame.py)             │
                    └──────┬──────────┬──────────────┘
                           │          │
              mode="train" │          │ mode="infer"
                           ▼          ▼
               ┌──────────────┐  ┌──────────────────┐
               │ OOF-safe     │  │ Production       │
               │ sources      │  │ sources          │
               │ p_win_oof    │  │ p_win_pred       │
               │ ev_win_corr  │  │ ev_win_calibrated│
               │ (OOF)        │  │ p_win_final      │
               └──────┬───────┘  └──────┬───────────┘
                      │                 │
                      └────────┬────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Schema Registry    │
                    │  (schema_registry)  │
                    │  InvestmentFeature  │
                    │  Spec frozen DC     │
                    └──────────┬──────────┘
                               │ source resolution
                               ▼
               ┌───────────────────────────────┐
               │  9 Category Computation       │
               │  model_prob  (8-12 cols)      │
               │  market_prob (6-10 cols)      │
               │  model_market_gap (10-16 cols)│
               │  race_relative (12-18 cols)   │
               │  odds_band   (6-10 cols)     │
               │  late_odds   (8-12 cols)     │
               │  ability_form(15-25 cols)    │
               │  course_pace (10-18 cols)    │
               │  uncertainty (10-16 cols)    │
               └──────────┬──────────────────┘
                          │ identical output schema
                          ▼
               ┌─────────────────────┐    ┌──────────────┐
               │  Output DataFrame   │───▶│ Cache Layer  │
               │  90-130 cols        │    │ (cache.py)   │
               │  same cols/dtypes   │    │ Parquet +    │
               │  for train/infer    │    │ sidecar JSON │
               └─────────────────────┘    └──────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Leakage Guard      │
               │  (leakage.py)       │
               │  POST_RACE exclusion│
               │  OOF-safe assertion │
               └─────────────────────┘
```

### Recommended Project Structure
```
src/investment/
├── __init__.py              # 公開API: InvestmentFeatureFrameBuilder, InvestmentFeatureSpec, build_frame
├── feature_frame.py         # InvestmentFeatureFrameBuilder, build_frame(), build_train_frame(), build_inference_frame()
├── schema_registry.py       # InvestmentFeatureSpec frozen dataclass, 全spec定義, FEATURE_SPECS dict
├── manifest.py              # InvestmentFrameManifest, generate_investment_manifest(), compute_investment_schema_hash()
├── cache.py                 # InvestmentFrameCache, load_or_compute(), _compute_cache_key()
└── leakage.py               # validate_no_post_race_leakage(), validate_oof_safe_sources(), validate_schema_identity()
```

### Pattern 1: Schema-Driven Dual Mode Feature Resolution
**What:** InvestmentFeatureSpecで各特徴量のtrain/inferソースを定義し、build_frame()がmodeに応じてソース列を解決する
**When to use:** 全特徴量生成において
**Example:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class InvestmentFeatureSpec:
    name: str                    # 安定出力名 (例: "if_p_win")
    category: str                # 9カテゴリの1つ
    dtype: str                   # "float64", "Int64" 等
    train_sources: tuple[str, ...]  # train mode時のソース列 (OOF-safe)
    infer_sources: tuple[str, ...]  # infer mode時のソース列 (本番)
    required: bool               # True=source欠損時fail-fast
    default_value: float | int | None  # optional時のデフォルト値
    missing_indicator: str | None     # "if_p_win_missing" 等
    leakage_class: str           # "safe", "oof_only", "post_race"
    description: str

# 例: model_prob カテゴリ
MODEL_PROB_P_WIN = InvestmentFeatureSpec(
    name="if_p_win",
    category="model_prob",
    dtype="float64",
    train_sources=("p_win_oof",),           # OOF予測 (train時のみ存在)
    infer_sources=("p_win_pred",),           # 本番予測 (infer時)
    required=True,
    default_value=None,
    missing_indicator=None,
    leakage_class="safe",
    description="モデル予測確率 (単勝)",
)

# 例: uncertainty カテゴリ (optional)
CONFORMAL_LOWER = InvestmentFeatureSpec(
    name="if_conformal_lower",
    category="uncertainty",
    dtype="float64",
    train_sources=("EV_lower_win_corrected",),  # OOF内CQR
    infer_sources=("EV_lower_win_corrected",),  # 本番CQR
    required=False,
    default_value=float("nan"),
    missing_indicator="if_conformal_lower_missing",
    leakage_class="safe",
    description="Conformal EV下限",
)
```

### Pattern 2: OOF-Safe Source Resolution
**What:** train modeではin-sample-only列(p_win_pred, ev_win_calibrated)を拒否し、OOF-safe列(p_win_oof, ev_win_corrected from OOF)のみ使用する
**When to use:** build_frame()のsource resolution段階
**Example:**
```python
def _resolve_source(self, df: pd.DataFrame, spec: InvestmentFeatureSpec, mode: str) -> pd.Series:
    """指定モードのソース列を解決してSeriesを返す。"""
    sources = spec.train_sources if mode == "train" else spec.infer_sources

    for source_col in sources:
        if source_col in df.columns:
            return df[source_col]

    # ソース列が見つからない場合
    if spec.required:
        raise ValueError(
            f"Required feature '{spec.name}': no source column found "
            f"for mode={mode}. Expected one of: {sources}"
        )
    # optional: default_value + missing indicator
    result = pd.Series(spec.default_value, index=df.index, dtype=spec.dtype)
    if spec.missing_indicator:
        # missing indicatorは別列として追加
        pass
    return result
```

### Anti-Patterns to Avoid
- **FeatureEngineパススルー:** FeatureEngine.build_all()の全出力を投資フレームに流すのはD-08違反。投資判断に特化した選別+派生のみ
- **p_win_predのtrain mode使用:** train modeでp_win_predを使用するとin-sample biasが発生。必ずp_win_oofまたはOOF由来のev_win_correctedを使用
- **モード自動検出:** df.columnsに基づくモード推測はバグの温床。明示的mode引数必須(D-11)
- **POST_RACE列の間接参照:** confirmed_oddsは訓練時E補正で使用されるが、投資フレームでは直接含めない。漏洩クラス="post_race"の列は一切使用しない
- **モデル再学習:** 投資フレーム構築はモデル出力の消費側。モデル自体の変更はスコープ外

## Category Feature Selection

以下は9カテゴリの具体的な特徴量選定案(Claude's Discretion領域)。コードベース調査に基づく。

### model_prob (8-12 cols) - モデル予測確率
| 出力名 | ソース(train) | ソース(infer) | 派生 | Required |
|--------|-------------|-------------|------|----------|
| if_p_win | p_win_oof | p_win_pred | direct | Yes |
| if_e_return | e_return_win_pred (OOF) | e_return_win_pred | direct | Yes |
| if_ev_raw | if_p_win * if_e_return | same | derived | Yes |
| if_p_win_corrected | p_win_corrected (OOF) | p_win_corrected | direct | Yes |
| if_ev_corrected | ev_win_corrected (OOF) | ev_win_corrected | direct | Yes |
| if_ev_calibrated | ev_win_corrected (OOF) | ev_win_calibrated | direct | No |
| if_p_win_final | p_win_combined (OOF) / p_win_oof | p_win_final | direct | No |
| if_edge_win | if_p_win_final * tanodds - 1.0 | same | derived | No |
| if_p_ability | p_ability_win (OOF) | p_ability_win | direct | Yes |
| if_p_ability_place | p_ability_place (OOF) | p_ability_place | direct | No |

**注:** train modeの"OOF"ソースは、generate_ev_oof_predictions()で生成されるOOF EV補正後の値(e.ev_win_corrected from OOF fold)。training_pipeline.py:893-925のフローで、OOF fold内でpredict_ev → correct_evが実行されるため、ev_win_correctedがOOF-safeに生成される。

### market_prob (6-10 cols) - 市場確率
| 出力名 | ソース(train/infer共通) | 派生 | Required |
|--------|----------------------|------|----------|
| if_implied_prob | p_market_win_adj または 1/tanodds | direct | Yes |
| if_popularity_rank | popularity_rank | direct | Yes |
| if_overround | overround | direct | Yes |
| if_market_entropy | market_entropy | direct | Yes |
| if_odds_skewness | odds_skewness | direct | No |
| if_implied_prob_hhi | implied_prob_hhi | direct | No |

### model_market_gap (10-16 cols) - モデル-市場乖離
| 出力名 | ソース | 派生 | Required |
|--------|--------|------|----------|
| if_logit_gap | if_p_win, if_implied_prob | logit(p_model) - logit(p_market) | Yes |
| if_abs_logit_gap | if_logit_gap | abs() | Yes |
| if_deviation_rank | deviation_rank | direct | No |
| if_deviation_zscore | deviation_zscore | direct | No |
| if_odds_ability_ratio | odds_to_ability_ratio | direct | No |
| if_edge_rank_in_race | if_logit_gap | race内rank(pct) | Yes |
| if_edge_zscore_in_race | if_logit_gap | race内zscore | No |
| if_signed_log_error | signed_log_error_win | direct | Yes |
| if_abs_log_error | abs_log_error_win | direct | Yes |
| if_market_error_rank | market_error_rank_in_race | direct | No |
| if_top3_gap | if_logit_gap | race内top3とのgap | No |
| if_field_ev_dispersion | if_ev_corrected | race内std | No |

### race_relative (12-18 cols) - レース内相対
| 出力名 | ソース | 派生 | Required |
|--------|--------|------|----------|
| if_p_win_race_rank | if_p_win | race内rank(pct) | Yes |
| if_ev_race_rank | if_ev_corrected | race内rank(pct) | Yes |
| if_ability_race_rank | p_ability_win | race内rank(pct) | Yes |
| if_ev_top1_gap | if_ev_corrected | top1とのgap | No |
| if_ev_top3_indicator | if_ev_race_rank | <=3で1 | No |
| if_form_trend_race_rank | form_trend_race_rank | direct | No |
| if_blood_wr_race_rank | blood_total_wr_race_rank | direct | No |
| if_closing_index_race_rank | closing_index_avg | race内rank(pct) | No |
| if_p_win_gap_to_fav | if_p_win, if_popularity_rank | 1番人気との確率gap | No |
| if_field_strength_mean | p_ability_win | race内mean | No |
| if_field_strength_std | p_ability_win | race内std | No |
| if_n_horses | field_size or rl_n_horses | direct | Yes |

### odds_band (6-10 cols) - オッズ帯
| 出力名 | ソース | 派生 | Required |
|--------|--------|------|----------|
| if_odds_band_id | tanodds | OddsBandFilter._get_band_name() | Yes |
| if_odds | tanodds or odds | direct | Yes |
| if_odds_log | if_odds | log() | No |
| if_odds_band_median_ev | if_ev_corrected, if_odds_band_id | band内median | No |
| if_odds_band_count | if_odds_band_id | band内count | No |

### late_odds (8-12 cols) - オッズ動態
| 出力名 | ソース(train/infer共通) | 派生 | Required |
|--------|----------------------|------|----------|
| if_odds_drop_60_10 | odds_drop_rate_60_10 | direct | No |
| if_odds_drop_30_10 | odds_drop_rate_30_10 | direct | No |
| if_odds_velocity | odds_velocity | direct | No |
| if_odds_volatility | odds_volatility | direct | No |
| if_odds_acceleration | odds_acceleration | direct | No |
| if_odds_direction_consistency | odds_direction_consistency | direct | No |
| if_late_money_ratio | odds_drop_rate_30_10, odds_drop_rate_60_10 | 比率 | No |
| if_popularity_change | popularity_change_30_10 | direct | No |

### ability_form (15-25 cols) - 能力・フォーム
| 出力名 | ソース | 派生 | Required |
|--------|--------|------|----------|
| if_norm_finish_avg | norm_finish_logit_avg | direct | No |
| if_haron_zscore | harontimel5_zscore | direct | No |
| if_closing_index | closing_index_avg | direct | No |
| if_form_trend | form_trend | direct | No |
| if_form_consistency | form_consistency | direct | No |
| if_blood_surface_wr | blood_surface_wr | direct | No |
| if_blood_total_wr | blood_total_wr | direct | No |
| if_sire_wr | sire_wr | direct | No |
| if_jockey_wr | jockey_wr_overall | direct | No |
| if_trainer_wr | trainer_wr_overall | direct | No |
| if_jt_combo_wr | jt_combo_wr | direct | No |
| if_class_level | class_level_current | direct | No |
| if_weighted_recent_form | weighted_recent_form_finish | direct | No |
| if_grade_x_form | grade_x_form_trend | direct | No |
| if_distance_x_closing | distance_x_closing_index | direct | No |
| if_dm_time_rank | dm_time_rank | direct | No |
| if_class_move | class_move | direct | No |

### course_pace (10-18 cols) - コース・ペース
| 出力名 | ソース | 派生 | Required |
|--------|--------|------|----------|
| if_closing_speed_ratio | closing_speed_ratio_avg | direct | No |
| if_haron_race_gap | haron_race_gap_avg | direct | No |
| if_pace_ratio | pace_ratio_avg | direct | No |
| if_surface | surface | direct (category) | Yes |
| if_distance_bin | distance_bin | direct (category) | Yes |
| if_grade_code | grade_code | direct (category) | No |
| if_track_condition | track_condition_code | direct | No |
| if_course_wr | course_wr | direct | No |
| if_pace_aptitude | pace_aptitude | direct | No |
| if_haron_zscore_trend | haron_zscore_trend | direct | No |
| if_pace_early | pace_early_avg | direct | No |
| if_pace_late | pace_late_avg | direct | No |
| if_closing_speed_race_rank | closing_speed_ratio_avg_race_rank | direct | No |

### uncertainty (10-16 cols) - 不確実性
| 出力名 | ソース(train/infer共通) | 派生 | Required |
|--------|----------------------|------|----------|
| if_conformal_lower | EV_lower_win_corrected | direct | No |
| if_conformal_upper | EV_upper_win_corrected | direct | No |
| if_conformal_width | if_conformal_upper - if_conformal_lower | derived | No |
| if_conformal_score | conformal_confidence_score | direct | No |
| if_p_x_e_interaction | p_x_e_interaction | direct | No |
| if_p_minus_e_gap | p_minus_e_gap | direct | No |
| if_ev_uncertainty_ratio | if_conformal_width / if_ev_corrected | derived | No |
| if_market_log_error | market_log_error_win | direct | No |
| if_calibration_residual | (OOF時のisotonic残差) | derived | No |
| if_odds_to_ability_dispersion | odds_to_ability_ratio | race内std | No |

## Dual Mode Design

### train mode ソース列 (OOF-safe)
以下はtraining_pipeline.pyの_train_submodel()内で生成されるOOF予測。`df_oof` DataFrame上で利用可能。

| 列名 | 生成箇所 | OOF-safe理由 |
|-------|---------|-------------|
| p_win_oof | generate_win_oof_predictions() (win_benter_gate.py:130) | KFold OOF、in-sampleなし |
| p_ability_win (OOF) | AbilityModel.train_oof() → add_ability_probs() | Expanding window OOF |
| p_win_corrected (OOF) | EVCorrectionModel.correct_ev() on OOF data | OOF EV補正 |
| ev_win_corrected (OOF) | OOF fold内 predict_ev → correct_ev | OOF EV (training_pipeline.py:1393) |
| signed_log_error_win (OOF) | MarketModel.predict_oof() | OOF Market Model |
| EV_lower_win_corrected (OOF) | ConformalEVModel on OOF data | OOF CQR |

### infer mode ソース列 (本番)
以下はRacePredictor.predict()内で生成される本番予測。

| 列名 | 生成箇所 |
|-------|---------|
| p_win_pred | WinTwoStageModel.predict_ev() (race_predictor.py:403) |
| e_return_win_pred | WinTwoStageModel.predict_ev() (race_predictor.py:405) |
| ev_win | WinTwoStageModel.predict_ev() (race_predictor.py:406) |
| p_win_corrected | EVCorrectionModel.correct_ev() (race_predictor.py:216) |
| ev_win_corrected | EVCorrectionModel.correct_ev() (race_predictor.py:216) |
| ev_win_calibrated | EVCorrectionModel.correct_ev() + Isotonic + band scales |
| p_win_final | WinBenterGate.apply() (race_predictor.py:227) |
| edge_win | WinBenterGate.apply() (race_predictor.py:82) |
| EV_lower_win_corrected | ConformalEVModel.predict_interval() (race_predictor.py:255) |
| conformal_confidence_score | ConformalEVModel.predict_interval() |
| signed_log_error_win | MarketModel.predict_and_calc_error() (race_predictor.py:167) |

### train mode で拒否すべき in-sample 列
- `p_win_pred` (in-sample予測、OOF非経由)
- `ev_win` (in-sample EV)
- `ev_win_calibrated` (Isotonic補正済みだがin-sample)
- `p_win_final` (Benter blend済みだがin-sample)
- `edge_win` (in-sample edge)

## Schema Registry

### InvestmentFeatureSpec 設計
```python
@dataclass(frozen=True)
class InvestmentFeatureSpec:
    name: str                    # 安定出力名 (prefix: "if_")
    category: str                # "model_prob", "market_prob", ...
    dtype: str                   # "float64", "Int64", "category"
    train_sources: tuple[str, ...]
    infer_sources: tuple[str, ...]
    required: bool
    default_value: float | int | None
    missing_indicator: str | None
    leakage_class: str           # "safe" | "oof_only" | "post_race"
    description: str
```

### FEATURE_SPECS: dict[str, InvestmentFeatureSpec]
全90-130 specをモジュールレベルdictで定義。カテゴリごとにgrouped access可能。

### 出力スキーマの固定
- 出力列名は常に`if_` prefix付き (入力列との衝突回避)
- 列順はcategory順 → category内でspec定義順
- dtypeはspec.dtypeに従い、build_frame()出力でastype()適用

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema hash computation | 独自ハッシュロジック | hashlib.sha256 + json.dumps(sort_keys=True) | OOFHealthValidator._compute_schema_hashes()と同一パターン (oof_health_validator.py:322-335) |
| Parquet I/O | 独自Parquet書込 | ParquetStoreパターン | pyarrow predicate pushdown対応済み |
| Manifest generation | 独自manifest構造 | OOFHealthValidator.generate_manifest()パターン拡張 | artifact_hash, schema_hash, generated_at等のD-10/D-11フィールド踏襲 |
| Leakage detection | 新しい漏洩検出ロジック | POST_RACE_COLS + 3層テストパターン | test_post_race_leakage.pyのLayer1/2/3構造 |
| OOF validation | 独自validation | OOFHealthValidator + load_validated_oof() | Phase 37で構築済みのOOF健全性確認 |

**Key insight:** このフェーズの安全性は既存パターンの再利用に依存する。OOFHealthValidatorのmanifest/hashパターン、ParquetStoreのI/Oパターン、3層漏洩テストパターンをそのまま踏襲することで、新規バグの導入を最小化する。

## Common Pitfalls

### Pitfall 1: train modeでのin-sample bias混入
**What goes wrong:** train modeでp_win_predやev_win_calibratedを使用すると、学習データにin-sample予測が混入し、過学習を引き起こす
**Why it happens:** 推論パス(RacePredictor)と学習パス(TrainingPipeline)で同じ列名(p_win_corrected等)が使われるため、混同しやすい
**How to avoid:** InvestmentFeatureSpec.train_sourcesにin-sample-only列を定義しない。leakage.pyでtrain mode出力にp_win_pred/ev_win_calibratedが含まれていないことを検証するテストを実装
**Warning signs:** 投資フレームのtrain mode出力にp_win_pred, ev_win, ev_win_calibrated, p_win_final, edge_winが含まれる場合

### Pitfall 2: train/infer スキーマ不一致
**What goes wrong:** train modeとinfer modeで異なる列名や列順が出力される
**Why it happens:** オプション列の有無や派生計算の条件分岐で列数が変動する
**How to avoid:** 全出力列をFEATURE_SPECSで固定。missing時もNaN + missing_indicatorで列は必ず存在。テストで厳密アサート(D-15)
**Warning signs:** trainとinferで列数が異なる、列順が異なる

### Pitfall 3: POST_RACE列の間接混入
**What goes wrong:** confirmed_oddsがEV補正のE補正で使われるため、その残差情報が投資フレームに間接的に漏出する
**Why it happens:** EVCorrectionModel.correct_ev()はconfirmed_oddsを使用してE補正を学習するが、推論時はoddsを使用する(race_predictor.py:216, ev_correction_model.py:431)
**How to avoid:** 投資フレームのソース列としてconfirmed_oddsを直接含めない。leakage_class="post_race"の列は使用しない
**Warning signs:** InvestmentFeatureSpecにconfirmed_odds, kakuteijyuni, time等のPOST_RACE_COLSが含まれる場合

### Pitfall 4: キャッシュ不整合
**What goes wrong:** ソースartifactが更新されたにも関わらず、古いキャッシュが使用される
**Why it happens:** キャッシュキーにsource_artifact_hashを含めていない、またはハッシュ計算が不正確
**How to avoid:** キャッシュキーにmode + feature_version + source_artifact_hash + schema_hashを含める(D-22, D-23)。sidecar manifestで検証(D-24)
**Warning signs:** 同一入力で異なる出力がキャッシュから返される

### Pitfall 5: 投資フレームをFeatureEngineに統合してしまう
**What goes wrong:** InvestmentFeatureFrameBuilderをFeatureEngine.build_all()の一部として実装する
**Why it happens:** 特徴量生成というとFeatureEngineが第一候補になる
**How to avoid:** D-35: FeatureEngineとモデルレイヤーから独立。src/investment/は独立パッケージ。FeatureEngineはモデル入力特徴量の生成のみ担当し、投資フレームはモデル出力の統合のみ担当
**Warning signs:** feature_frame.pyがFeatureEngineをimportしている場合

## Code Examples

### build_frame() 基本パターン
```python
from typing import Literal

import pandas as pd

def build_frame(
    df: pd.DataFrame,
    mode: Literal["train", "infer"],
    *,
    builder_version: str = "1.0.0",
) -> pd.DataFrame:
    """投資判断用特徴量フレームを構築。

    Args:
        df: モデル出力・市場データを含むDataFrame
        mode: "train" (OOF-safe) または "infer" (本番)
        builder_version: ビルダーバージョン (キャッシュキー)

    Returns:
        投資特徴量DataFrame (90-130列、race_id/umaban + if_*列)
    """
    if mode not in ("train", "infer"):
        raise ValueError(f"mode must be 'train' or 'infer', got '{mode}'")

    result = pd.DataFrame(index=df.index)

    # identity列
    for col in ("race_id", "umaban"):
        if col in df.columns:
            result[col] = df[col].values

    # 各specを解決
    for spec in FEATURE_SPECS.values():
        series = _resolve_source(df, spec, mode)
        # 派生特徴量の場合はseriesを計算
        if spec.name in _DERIVED_FEATURES:
            series = _DERIVED_FEATURES[spec.name](result, df)
        result[spec.name] = series.astype(spec.dtype) if spec.dtype != "category" else series

        if spec.missing_indicator:
            result[spec.missing_indicator] = series.isna().astype("int8")

    # 列順を固定 (spec定義順)
    output_cols = ["race_id", "umaban"] + [s.name for s in FEATURE_SPECS.values()]
    missing_cols = [s.missing_indicator for s in FEATURE_SPECS.values() if s.missing_indicator]
    result = result[output_cols + missing_cols]

    return result
```

### Manifest パターン (OOFHealthValidator拡張)
```python
import hashlib
import json
from datetime import datetime, timezone

def generate_investment_manifest(
    df: pd.DataFrame,
    *,
    feature_version: str,
    builder_version: str,
    mode: str,
    source_artifact_hash: str,
    source_oof_manifest_path: str | None = None,
) -> dict:
    """投資フレームartifact manifestを生成 (D-30, IFF-07)。"""
    # OOFHealthValidator._compute_schema_hashes パターンを踏襲
    cols_sorted = sorted(df.columns.tolist())
    schema_hash = hashlib.sha256(json.dumps(cols_sorted).encode()).hexdigest()

    dtype_pairs = sorted(f"{col}:{df[col].dtype}" for col in df.columns)
    schema_dtype_hash = hashlib.sha256(json.dumps(dtype_pairs).encode()).hexdigest()

    return {
        "artifact_name": "investment_feature_frame",
        "builder_version": builder_version,
        "feature_version": feature_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "row_count": len(df),
        "schema_hash": schema_hash,
        "schema_dtype_hash": schema_dtype_hash,
        "source_artifact_hash": source_artifact_hash,
        "source_oof_manifest_path": source_oof_manifest_path,
        "column_count": len(df.columns),
    }
```

### Leakage Detection パターン
```python
from domain.types import POST_RACE_COLS

def validate_no_post_race_leakage(output_columns: list[str]) -> None:
    """投資フレーム出力にPOST_RACE列が含まれていないことを検証 (IFF-05, VAL-01)。"""
    # test_post_race_leakage.py Layer 2 パターン
    overlap = set(output_columns) & set(POST_RACE_COLS)
    if overlap:
        raise ValueError(
            f"InvestmentFeatureFrame output contains POST_RACE columns: {overlap}"
        )

def validate_oof_safe_sources(
    specs: dict[str, "InvestmentFeatureSpec"],
) -> list[str]:
    """train modeの全specのtrain_sourcesにin-sample-only列がないことを検証。"""
    IN_SAMPLE_ONLY = {
        "p_win_pred", "ev_win", "ev_win_calibrated",
        "p_win_final", "edge_win",
    }
    violations = []
    for spec in specs.values():
        overlap = set(spec.train_sources) & IN_SAMPLE_ONLY
        if overlap:
            violations.append(f"{spec.name}: {overlap}")
    return violations

def validate_schema_identity(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
) -> None:
    """train/infer出力スキーマ同一性を検証 (IFF-03, D-15)。"""
    assert list(train_df.columns) == list(infer_df.columns), (
        f"Column mismatch: train has {len(train_df.columns)}, "
        f"infer has {len(infer_df.columns)}"
    )
    for col in train_df.columns:
        assert train_df[col].dtype == infer_df[col].dtype, (
            f"dtype mismatch for {col}: train={train_df[col].dtype}, "
            f"infer={infer_df[col].dtype}"
        )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| モデル出力を直接消費 | Feature Frame抽象化 | Phase 38 (this phase) | 投資判断特徴量の正式なスキーマ定義、OOF安全性の型レベル保証 |
| p_win_predをtrainで使用 | p_win_oofでOOF-safe | Phase 37 (OOFHealthValidator) | in-sample biasの排除 |
| 手動特徴量選択 | Schema Registry (InvestmentFeatureSpec) | Phase 38 | 特徴量のメタデータ管理、missing behaviorの明示 |

**Deprecated/outdated:**
- v1.8 Manifest凍結 (VAL-06): 廃止。v2.0 artifact manifest (IFF-07) に置き換え
- Regime-dependent calibration: v2.0全体でスコープ外

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | train modeでのev_win_correctedはOOF-safeである (generate_ev_oof_predictionsで生成) | Category Feature Selection / Dual Mode Design | HIGH - もしev_win_correctedがin-sampleの場合、train modeにin-sample biasが混入する |
| A2 | train modeのdf_oofにはp_win_oof列が含まれる | Category Feature Selection | MEDIUM - training_pipeline.py:130でp_win_oofが生成されるが、df_oofに残るかは実装次第 |
| A3 | training_pipeline.py:1393のev_win_correctedはOOF内のpredict_ev→correct_evで生成される | Dual Mode Design | MEDIUM - OOF EV補正チェーンの正確性に依存 |
| A4 | 全モデル出力列(p_win_pred等)はPOST_RACE_COLSに含まれない | Leakage Detection | LOW - conformal_ev_model.pyの_MODEL_OUTPUT_COLSにPOST_RACE_COLSが除外済み |
| A5 | ConformalEVModel.FEATURE_COLSにPOST_RACE列が含まれない | Leakage Detection | LOW - test_post_race_leakage.pyで検証済み |
| A6 | 派生特徴量の計算式(race-relative, uncertainty等)は具体的な設計で決定可能 | Claude's Discretion | LOW - 計算式は確定可能だが、最適な設計は実装時に検証が必要 |

## Open Questions

1. **OOF EV補正の完全な伝播**
   - What we know: generate_ev_oof_predictions()はWinTwoStage→EVCorrectionのOOFチェーンを実行する (training_pipeline.py:1361-1404)
   - What's unclear: OOF fold内でIsotonic + band scale補正も実行されるか、それともev_win_correctedのみか。もしIsotonicがOOF内で適用されない場合、train modeではev_win_calibratedは利用できず、ev_win_correctedが上限となる
   - Recommendation: training_pipeline.py:1393を確認。ev_win_correctedはcorrect_ev()の直接出力であり、Isotonicは別途fit_ev_calibration()で適用される。したがってtrain modeのキャリブレーション済みEV相当列はev_win_correctedまで

2. **Benter Gate OOF予測のtrain mode可用性**
   - What we know: generate_win_oof_predictions()でp_win_oofが生成される (win_benter_gate.py:130)
   - What's unclear: p_win_finalのOOF版(p_win_final_oof)が生成されるか。Benter combination自体は学習パイプライン内でOOF予測に対してfittingされるが、OOF予測にBenterを適用したp_win_final_oofが明示的に生成されるかは未確認
   - Recommendation: p_win_final_oofが存在しない場合、train modeではp_win_corrected(OOF)をif_p_win_finalのソースとする。これが最もOOF-safeなBenter前確率

## Environment Availability

> Step 2.6: SKIPPED (no external dependencies identified)
> このフェーズは新規外部ツール・サービス・ランタイムに依存しない。全てPython 3.11標準ライブラリ + 既存プロジェクト依存関係で完結。

## Validation Architecture

> `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json` -- this section is skipped.

## Security Domain

> `security_enforcement` is not explicitly set in config. Including abbreviated security domain.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | InvestmentFeatureSpec required/optional + dtype検証 |
| V4 Access Control | yes | train mode制限 (in-sample-only列の拒否) |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage (POST_RACE) | Information Disclosure | POST_RACE_COLS除外 + 3層テスト |
| In-sample bias | Elevation of Privilege | train mode OOF-safe source resolution + 検証テスト |

## Sources

### Primary (HIGH confidence)
- コードベース直接読込: 全7モデルのFEATURE_COLS、training_pipeline.py、race_predictor.py、oof_health_validator.py
- `src/domain/types.py` — POST_RACE_COLS 41列定義
- `tests/test_post_race_leakage.py` — 3層漏洩テストパターン
- `src/validation/oof_health_validator.py` — OOFHealthValidator, generate_manifest(), _compute_schema_hashes()
- `.planning/phases/37-ev-calibration-layers/37-VERIFICATION.md` — Phase 37完了確認

### Secondary (MEDIUM confidence)
- `src/models/win_benter_gate.py` — Benter combination + OOF予測生成パターン
- `src/pipelines/training_pipeline.py` — OOF予測保存・EV OOFチェーン・manifest生成パターン
- `src/backtest/race_predictor.py` — 推論パイプラインの完全チェーン

### Tertiary (LOW confidence)
- なし — 全てコードベース直接読込に基づく

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 新規外部パッケージなし、既存依存関係のみ
- Architecture: HIGH - 9カテゴリ設計はCONTEXT.mdで決定済み、実装パターンは既存コードから直接導出
- Pitfalls: HIGH - 既存の漏洩テスト・OOF検証インフラから直接導出
- Category feature selection: MEDIUM - 具体的な列選定はClaude's Discretion、実装時に調整が必要

**Research date:** 2026-05-27
**Valid until:** 2026-06-27 (stable - コードベース構造に依存)
