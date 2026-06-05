# Phase 49: Derived & Higher-Order Features - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Tier 3（馬個体の馬場状態適性・季節偏差）と Tier 4（ペース予測・異常値検出・レースレベル集約・既存特徴量インタラクション）の派生・高次特徴量を実装し、トラック条件特徴量の全レイヤーを完成させる。Phase 48のT1/T2交互作用特徴量を基盤として、過走履歴ベース適性・ドメイン知識スコア・race-level集約・既存特徴量拡張を追加する。

**In scope:** T3-01~04 (馬個体適性4種 + カテゴリ分類 + 季節偏差2種), T4-01 (front_bias/kickback/pace_class), T4-02 (race_condition_match 3列 + race_field_front_bias), T4-03 (anomaly/extreme flags 2種), T4-04 (5既存インタラクション), precompute parquet拡張, FEATURE_COLS 12モデル外科的ルーティング
**Out of scope:** Feature Routing Audit (Phase 50), BT ROI検証 (Phase 50), IC評価 (Phase 50), MAWC修正 (v2.4+)

</domain>

<decisions>
## Implementation Decisions

### T3: 馬個体適性のPIT-safe計算

- **D-01:** Precompute Parquet拡張アプローチ。`horse_career_stats.parquet`と同じexpanding window / shift(1)パターンで、各race_id時点より前の出走実績のみから適性率を算出。新規parquet `data/raw/horse_track_aptitude.parquet` を作成
- **D-02:** mergeキー: `race_id + kettonum`。FeatureEngine.build_all()内でleft merge。Phase 48のtrack_conditions.parquet mergeパターンに倣う
- **D-03:** 条件分類閾値は設定可能。デフォルト値:
  - ダート含水率: wet >= 12%, dry < 3% (Phase 48のdirt_moisture_high_flag/dry_flagと整合)
  - 芝クッション値: hard >= 10, soft < 8 (Phase 48のD-12ビン境界 [0,7,8,9,10,inf] と整合)
  - `config/settings.yaml`の`track_condition`セクションで上書き可能。precompute時は固定値を使用
- **D-04:** 的中定義: `kakuteijyuni <= 3` (3着以内)。取消・除外(`kakuteijyuni <= 0` or NaN)は分母から除外
- **D-05:** 適性カテゴリ分類ロジック (T3-03): 絶対閾値 + 最低出走数
  - `min_starts=3`, `hit_rate_threshold=0.3`
  - wet_rate >= threshold AND dry_rate < threshold → 湿得意
  - dry_rate >= threshold AND wet_rate < threshold → 乾得意
  - 両方 >= threshold → 万能 (balanced)
  - 両方 < threshold または分母不足 → unknown/NaN
  - 分母(wet_starts or dry_starts) < min_starts の条件は判定不可 → その条件の得意/不得意はNaN
- **D-06:** `horse_condition_versatility` = mean(wet_hit_rate, dry_hit_rate) × (1 - |wet_hit_rate - dry_hit_rate|)。成績水準とバランスの積。分母不足の条件がある場合はNaN
- **D-07:** Precompute Parquet出力スキーマ (12列):
  - キー: `race_id`, `kettonum`
  - 適性率(4): `horse_dirt_wet_hit_rate`, `horse_dirt_dry_hit_rate`, `horse_cushion_hard_hit_rate`, `horse_cushion_soft_hit_rate`
  - 出走数(4): `horse_dirt_wet_starts_count`, `horse_dirt_dry_starts_count`, `horse_cushion_hard_starts_count`, `horse_cushion_soft_starts_count`
  - 万能度(1): `horse_condition_versatility`
  - カテゴリ(1): `horse_condition_type` (string: "wet_good"/"dry_good"/"balanced"/"unknown")
  - 前走馬場(2): `prev_dirt_moisture`, `prev_turf_cushion` (T4-04 surface_condition_transition用)

### T4-01: ペース・バイアススコア

- **D-08:** ドメイン閾値ルールベースの連続スコア。過去成績回帰は不採用。当日公表値のみからPIT-safeに計算
- **D-09:** 閾値間線形補間で0〜1にマッピング:
  - `track_front_bias_score`:
    - ダート: clip((dirt_moisture - 3) / (12 - 3), 0, 1)。高含水→先行有利
    - 芝: clip((turf_cushion - 8) / (10 - 8), 0, 1)。高クッション(硬)→先行有利
    - 統一列。surface別submodelで分離済み
  - `kickback_risk_score`:
    - ダート: clip((12 - dirt_moisture) / (12 - 3), 0, 1)。低含水→蹴り返し高
    - 芝: clip((10 - turf_cushion) / (10 - 8), 0, 1)。低クッション(柔)→蹴り返し高
    - 統一列
  - NaNはNaNのまま伝播
- **D-10:** `expected_pace_class`: 3段階数値コード (slow=0, neutral=1, fast=2)。front_bias高→slow寄り、kickback高→fast寄り。NaNはNaNのまま

### T4-02: レースレベル集約

- **D-11:** `race_condition_match_score` (主代表列): 出走各馬の条件対応適性rateのmean。現在の条件に応じたrate列を選択:
  - 高含水ダート(moisture >= 12) → horse_dirt_wet_hit_rate のmean
  - 乾燥ダート(moisture < 3) → horse_dirt_dry_hit_rate のmean
  - 硬い芝(cushion >= 10) → horse_cushion_hard_hit_rate のmean
  - 柔らかい芝(cushion < 8) → horse_cushion_soft_hit_rate のmean
  - 中間域 → 両方のmean
- **D-12:** `race_condition_match_max`: 同条件のmax値 (エース適性馬の存在)
- **D-13:** `race_condition_match_ratio`: 適性rate >= hit_rate_threshold AND starts_count >= min_starts の馬数 / 有効出走馬数
- **D-14:** `race_field_front_bias` = front_runner_ratio × track_front_bias_score
  - front_runner_ratio: kyakusitukubun_cdが逃げ/先行の馬数 / 有効出走馬数
  - いずれかNaN → NaN

### T4-03: 異常値検出

- **D-15:** `cushion_anomaly_flag` / `moisture_extreme_flag`: コース平均から2σ逸脱をフラグ化。T3-04のtrack×month統計を利用:
  - |season_deviation| > 2 → 1.0 (異常)
  - |season_deviation| <= 2 → 0.0 (正常)
  - NaN条件 → NaN
  - 上下方向も保持: `cushion_anomaly_high` / `cushion_anomaly_low` (Phase 50で重要性確認後、不要なら削除)

### T4-04: 既存特徴量インタラクション

- **D-16:** 全て数値積パターン (Phase 48のdirt_moisture_x_kyakusituと同一パターン):
  - `cushion_x_distance` = turf_cushion × kyori
  - `moisture_x_weight` = dirt_moisture × bataijyu
  - `cushion_x_age` = turf_cushion × barei
  - `moisture_x_prev_kyakusitu` = dirt_moisture × prev_kyakusitu_cd
  - 片方NaN → NaN伝播
- **D-17:** `surface_condition_transition`: 前走からの馬場条件変化
  - ダート: dirt_moisture - prev_dirt_moisture
  - 芝: turf_cushion - prev_turf_cushion
  - 前走値はT3 precompute parquetの`prev_dirt_moisture` / `prev_turf_cushion`列から取得
  - 同surface前走がない、または今回値NaN → NaN

### T3-04: 季節偏差

- **D-18:** `cushion_season_deviation` / `moisture_season_deviation`: trackcd × month の学習期間統計でzscore計算
  - cushion_season_deviation = (turf_cushion - track_month_mean) / track_month_std
  - moisture_season_deviation = (dirt_moisture - track_month_mean) / track_month_std
  - 統計量は学習期間のみで算出、検証/テストへはmap適用 (Phase 48のtrack_statsパターンをmonth次元に拡張)
  - std==0 / NaN / 該当track×monthなし → NaN

### 特徴量モジュール構成

- **D-19:** T4-01/T4-03/T4-04は `compute_track_condition_features()` に追加 (行単位計算)
- **D-20:** T4-02は新規関数 `compute_race_condition_features()` に分離 (race_id groupby集約)
- **D-21:** 呼び出し順序: T3 precompute merge → compute_track_condition_features() (T1/T2/T4-01/T4-03/T4-04) → compute_race_condition_features() (T4-02)
- **D-22:** T3-04のtrack×month統計は `_compute_track_month_stats()` として `_compute_track_stats()` の拡張で実装。SubmodelSetに保存

### 外科的ルーティング (Phase 48 D-04~D-06拡張)

- **D-23:** T3馬個体適性 + T4-01バイアス/ペース + T4-04インタラクション: Phase 48と同一パターン
  - 登録: AbilityModel, WinTwoStage, PlaceTwoStage, WideTwoStage, EVCorrection, PlaceEVCorrection (各surface)
  - 除外: MarketModel, RegimeDetector
- **D-24:** T4-03異常値フラグ: MarketModelにも追加候補。Phase 50 AuditでMarketModel支配を検証
- **D-25:** T4-02 race-level集約: RaceQualityScreenerにも追加候補。Phase 50 Auditで検証
- **D-26:** 問題検出時はPhase 48同一パターン(全除外)へ戻す安全網付き

### Claude's Discretion

- テスト構成・テストケースの詳細設計 (既存パターンに従う)
- TRACK_CONDITION_COLS / TRACK_DERIVED_COLS / RACE_CONDITION_COLS の具体的な列名定義
- track_month_statsの保存形式 (dict or DataFrame、SubmodelSetへの統合)
- precomputeスクリプト `scripts/precompute_track_aptitude.py` の実装詳細
- `_compute_track_month_stats()` の実装詳細
- build_all()へのT3 parquet merge追加パターン
- ログフォーマット・進捗表示の設計
- cushion_anomaly_high/lowの閾値詳細と上下分離の要否判断

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 48 Context & Existing Implementation
- `.planning/phases/48-core-edge-features/48-CONTEXT.md` — Phase 48の全決定事項 (データマージ・外科的ルーティング・track_statsパターン)
- `src/features/track_condition_features.py` — T1/T2実装。compute_track_condition_features() + _compute_track_stats()。Phase 49でT4-01/T4-03/T4-04を追加
- `.planning/phases/47-etl-data-pipeline/47-CONTEXT.md` — Phase 47のtrack_conditions.parquet設計 (NaN処理・異常値NaN化範囲)

### Precompute Patterns
- `scripts/precompute_career_stats.py` — Precompute parquetパターン (expanding window / shift(1) / PIT-safe)。T3 precomputeの直接参考
- `src/features/track_condition_data.py` — ETL用precomputeモジュール (thin orchestrator + 変換ロジック分離パターン)
- `src/db/readers.py` (lines 310-331) — load_career_stats / load_sire_stats standalone関数パターン。T3 parquetローダー追加位置

### Feature Engine & Pipeline
- `src/features/feature_engine.py` — FeatureEngine.build_all() sequential pipeline。T3 merge追加ポイント (BloodlineFeaturesパターン)
- `src/pipelines/training_pipeline.py` — _train_submodel() 内の HorseHistoryFeatures → track_condition_features 呼び出し順序。compute_race_condition_features() の挿入位置
- `src/backtest/engine.py` — BacktestEngine.run() 内のfeature pre-computation。training_pipelineと対称な実装が必要

### Data Access & Domain
- `src/db/repository.py` — DataRepository.load_track_conditions(start, end)。T3 parquet用のload_horse_track_aptitude()追加先
- `src/domain/types.py` — POST_RACE_COLS定義。含水率/クッション値は含めない
- `src/domain/models.py` — SubmodelSet dataclass。track_month_stats保持フィールド追加先

### Feature Registration
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS
- `src/models/two_stage_return_model.py` — Win/Place/Wide TwoStage FEATURE_COLS
- `src/models/ev_correction_model.py` — EVCorrection FEATURE_COLS
- `src/features/interaction_features.py` — 純粋関数パターン + INTERACTION_COLS定数 + column existence guard

### Configuration
- `.planning/REQUIREMENTS.md` — T3-01~04, T4-01~04, REG-01~03 要件定義
- `config/settings.yaml` — track_condition設定セクション追加先 (閾値の上書き可能化)

### Domain Knowledge Context
- `src/features/horse_history_features.py` — kyakusitukubun_cd の生成、HorseHistoryFeatures出力列
- `training_pipeline.py` lines 903-904 — sire_id/bms_id mapping: horses_df.set_index("kettonum")["ketto3infohansyokunum1"]

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_compute_track_stats()`: trackcd別mean/std計算パターン。T3-04のtrack×month statsはこれを拡張
- `compute_track_condition_features()`: 行単位特徴量計算。Phase 49でT4-01/T4-03/T4-04を追加
- `expanding window + shift(1)`: horse_career_stats.parquetのPIT-safeパターン。T3 precomputeの直接テンプレート
- `frame_number`: _map_basic_features() で wakuban → frame_number に変換済み
- `kyakusitukubun_cd`: HorseHistoryFeatures出力。_train_submodel()内で利用可能
- `bataijyu`, `barei`, `kyori`: FeatureEngine.build_all()出力に既に含まれる
- `prev_kyakusitu_cd`: HorseHistoryFeatures出力の前走脚質。T4-04で使用
- `ParquetStore`: read/write/exists でParquet I/O
- `DataRepository`: 既存DIパターン。load_horse_track_aptitude()追加先

### Established Patterns
- Feature moduleパターン: `src/features/*.py` に純粋関数、定数で列名管理、column existence guard
- 遅延import: build_all() / _train_submodel() 内で `from features.xxx import compute_xxx`
- TimingContext: `with TimingContext("build_all/track_condition")` でステップ計測
- Guard clause: 空DataFrame早期リターン、列存在チェック後の計算スキップ
- NaN処理: `pd.to_numeric(errors="coerce")` + LightGBMネイティブNaN対応
- Surgical routing: 特徴量の性質に応じた登録/除外。Phase 36教訓に基づく

### Integration Points
- `FeatureEngine.build_all()`: T3 parquet mergeポイント (BloodlineFeaturesパターンに倣う)
- `_train_submodel()` / `BacktestEngine.run()`:
  1. T3 precompute merge (race_id + kettonum)
  2. HorseHistoryFeatures (kyakusitukubun_cd利用可能)
  3. `compute_track_condition_features()` (T1/T2 + T4-01/T4-03/T4-04追加)
  4. `compute_race_condition_features()` (T4-02 race-level集約。新規関数)
  5. `compute_interaction_features()` (既存)
- 12モデルの `FEATURE_COLS`: 外科的ルーティング D-23~D-26 に基づく列追加
- `SubmodelSet`: track_month_stats保持フィールド追加
- `DataRepository`: load_horse_track_aptitude()メソッド追加

### Key Data Flow
```
Precompute Pipeline:
  track_conditions.parquet + entries.parquet + races.parquet
    → precompute_track_aptitude.py
    → horse_track_aptitude.parquet (12列, PIT-safe expanding window)

FeatureEngine.build_all():
  race_df + entry_df + odds_df → merge
  → load_track_conditions() → merge on race_id (Phase 48)
  → load_horse_track_aptitude() → merge on race_id + kettonum (NEW)
  → horse_features.parquet (T3生値 + prev値含む)

_train_submodel() / BacktestEngine:
  horse_features loaded (T3 columns + moisture + cushion present)
  → HorseHistoryFeatures (kyakusitukubun_cd, prev_kyakusitu_cd available)
  → compute_track_condition_features():
      T1/T2 (Phase 48 existing)
      T4-01: front_bias, kickback_risk, pace_class (linear interpolation)
      T4-03: anomaly flags (|season_deviation| > 2)
      T4-04: 5 numeric interactions + surface_condition_transition
  → compute_race_condition_features():
      T4-02: race_condition_match_score/max/ratio (groupby race_id on T3)
             race_field_front_bias (front_runner_ratio × front_bias)
  → モデル学習/推論 (外科的ルーティング適用済みFEATURE_COLS)
```

</code_context>

<specifics>
## Specific Ideas

- 含水率 wet/dry閾値 (12%/3%) はPhase 48のdirt_moisture_high_flag/dry_flagと完全一致。既存フラグとT3適性の条件分類が統一される
- クッション値 hard/soft閾値 (10/8) はPhase 48のD-12ビン境界 [0,7,8,9,10,inf] と整合。"standard"(8-10)を中間域とし、両端をhard/softとする設計
- front_biasとkickback_riskは理論上 1-front_bias ≈ kickback_risk の関係。LightGBMが冗長性を処理可能だが、将来重要度分析で一方が低寄与なら削除検討
- T3 precomputeはhorse_career_stats.parquetと同じくレース前日までに実行可能。当日のtrack condition値は含まず、過走履歴のみから計算。prev_dirt_moisture/prev_turf_cushionも前レース時点の値
- クッション値データは2020/09開始 → 2018-2020年8月の芝レースはturf_cushionがNaN。T3/T4芝系特徴量もNaN。VLD-03(WF Fold0 NaN率検証)で確認
- cushion_anomaly_high/lowの上下分離はPhase 50重要度確認後に不要なら削除。まずは異常値フラグのみ実装し、方向性はPhase 50判断

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>

---

*Phase: 49-Derived Higher-Order Features*
*Context gathered: 2026-06-05*
