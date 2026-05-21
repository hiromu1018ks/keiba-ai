# Phase 26: EveryDB2 New Features - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning

<domain>
## Phase Boundary

EveryDB2の未活用テーブル（n_hansyoku, n_record, n_mining）から高価値特徴量を抽出し、MLモデルの入力を拡張する。加えて、レース内全馬の相対比較特徴量を新規生成する。ETL実行（3テーブルのParquet抽出）をPhase内に含める。n_miningの82列はPIT監査でPRE/POST分類し、PRE列のみを使用する。

**In scope:**
- DATA-01: n_hansyoku + n_sankuから包括的血統特徴量（種牡馬系統、母系BMS、繁殖牝馬成績等）を抽出・生成。PRE/POST分類を文書化
- DATA-02: n_recordテーブルからコース別タイム指数等の特徴量を生成
- DATA-03: レース内全馬の相対比較特徴量（相対ランク、偏差値等）を新規モジュールで生成
- DATA-04: n_miningテーブル(82列)のPRE/POST分類完了、PRE列から特徴量抽出
- ETL実行: run_etl.py --tables n_hansyoku n_record n_mining で個別テーブル抽出（Phase内スコープ）
- 既存bloodline_features.py / sire_features.pyとの統合設計
- POST_RACE漏洩テスト（Phase 23 CI）が新特徴量追加後も通過すること
- 既存テスト全通過確認

**Out of scope:**
- 特徴量の交互作用・変換（Phase 27: INTER-01/02/03）
- 最終ROI検証・特徴量凍結（Phase 28）
- モデル再学習・ハイパーパラメータ調整
- 複勝/ワイドモデルの変更
- Stage1 AbilityModelへの追加判断（Claude discretion）
- n_taisengata_mining等の他テーブル（将来フェーズ候補）

</domain>

<decisions>
## Implementation Decisions

### ETL・データ取得戦略
- **D-01:** n_hansyoku, n_record, n_miningの3テーブルを個別抽出する（フルETLは不要）。既存の`run_etl.py --tables`を使用。所要時間~1-2分。
- **D-02:** ETL実行はPhase 26のスコープ内に含める。Plan内にETLステップを定義する。但しPostgreSQL環境依存のためCI実行不可。実行自体はユーザーがローカルで行う。

### n_mining PIT監査 (DATA-04)
- **D-03:** n_miningの82列のPRE/POST分類は`docs/everyDB2/44-MINING.md`の列説明を主軸に行う。JRA-VAN公式ドキュメントに「確定後」等の記述がある列をPOSTと判定する。
- **D-04:** POST列が含まれていた場合は、その列のみ除外しPRE列から特徴量を抽出する。テーブル全体の除外は行わない。
- **D-05:** PRE/POST分類結果は文書化して出力する（Success Criteria要件）。

### 血統特徴量の設計 (DATA-01)
- **D-06:** n_hansyoku（19列）+ n_sanku（26列、繁殖成績）の両方を活用した包括的血統特徴量を実装する。繁殖牝馬の産駒成績、BMS拡張、生産牧場情報等を含む（FEATURES.md D-01完全実装）。
- **D-07:** 新規モジュール（例: `features/dam_pedigree_features.py`）を作成する。既存のbloodline_features.py / sire_features.pyは変更しない。
- **D-08:** BMS拡張（D-02: BMS distance_wr, BMS surface_wr）はsire_features.pyの拡張として実装するか、新規モジュールに含めるかはClaudeの判断に委ねる。

### 相対比較特徴量の設計 (DATA-03)
- **D-09:** FEATURES.md TS-05推奨の5-10特徴量を新規モジュール（`features/relative_features.py`）で生成する。現在のintra_race_features.py（2特徴量）はそのまま残す。
- **D-10:** 具体的な特徴量選定（どのベース特徴量を相対化するか）はClaudeの判断に委ねる。ベストプラクティスを追求する方針。
- **D-11:** 相対比較の計算手法（groupby("race_id")ランク、z-score、mean差分等）はClaudeの判断に委ねる。

### Claude's Discretion
- 各特徴量の具体的な内容と対象モデル配置（Stage1 / Stage2 / 両方）
- BMS拡張の実装場所（sire_features.py拡張 vs 新規モジュール）
- 相対比較特徴量の具体的な特徴量名・計算方法
- n_record特徴量の具体的な設計（コース別タイム指数の計算方法）
- n_mining PRE列から抽出する特徴量の選定
- 各FEATURE_COLSへの挿入位置
- テストの追加・更新内容
- POST_RACE漏洩テスト（Phase 23 CI）の通過確認方法
- ETL実行後のParquetスキーマ検証方法

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### JRA-VAN EveryDB2 テーブル定義（PIT監査・特徴量設計に必須）
- `docs/everydb2/34-HANSYOKU.md` — n_hansyokuテーブル定義（19列）。繁殖情報、種牡馬・繁殖牝馬データ
- `docs/everydb2/35-SANKU.md` — n_sankuテーブル定義（26列）。産駒成績、繁殖成績
- `docs/everydb2/36-RECORD.md` — n_recordテーブル定義（48列）。コースレコード、トラック別最速タイム
- `docs/everydb2/44-MINING.md` — n_miningテーブル定義（82列）。JRA事前計算分析データ。PIT監査の主軸
- `docs/everydb2/55-TAISENGATA_MINING.md` — n_taisengata_mining（46列）。対戦型分析（参考）
- `docs/everydb2/53-KEITO.md` — n_keitoテーブル定義。血統系統コード（既存blood_keito_cd参照）
- `docs/everydb2/52-BAMEIORIGIN.md` — n_bameioriginテーブル定義。5代血統拡張（参考）
- `docs/everydb2/CODE.md` — JRA-VANコード体系。列名解釈に必要

### 既存特徴量モジュール（統合対象・パターン参照）
- `src/features/bloodline_features.py` — BloodlineFeatures。6特徴量（blood_total_wr等）。horse_career_stats.parquet使用
- `src/features/sire_features.py` — SireFeatures。5特徴量（sire_wr等 + bms_wr）。sire_career_stats.parquet使用
- `src/features/intra_race_features.py` — 既存2特徴量（weight_diff_from_mean, odds_rank）。相対特徴量の既存パターン
- `src/features/feature_engine.py:build_all()` — 特徴量統合ポイント。compute_intra_race_features()等の呼び出し順序
- `src/features/horse_career_stats.py` — PIT-safe累積キャリア統計。bloodline_features.pyで使用パターン
- `src/features/info_asymmetry_features.py` — expanding().shift(1) PIT-safeパターンの参考

### FEATURE_COLS定義（変更対象）
- `src/models/stage1_ability_model.py:28-128` — Stage1AbilityModel.FEATURE_COLS (89特徴量)。血統・相対特徴量の追加先候補
- `src/models/two_stage_return_model.py:48-117` — WinTwoStageModel.FEATURE_COLS。Stage2追加先候補
- `src/models/two_stage_return_model.py:289-367` — PlaceTwoStageModel.HIT_FEATURE_COLS
- `src/models/two_stage_return_model.py:372-441` — PlaceTwoStageModel.RETURN_FEATURE_COLS

### ETL・データ取得
- `scripts/run_etl.py` — ETL CLI。--tables引数で個別テーブル抽出をサポート
- `config/etl_tables.yaml` — ETLテーブル設定。n_hansyoku(L278-282), n_record(L308-312), n_mining(L203-207)が定義済み
- `src/db/connection.py` — DatabaseConnection.etl_to_parquet()

### 安全性・監査（Phase 23-25の基盤）
- `src/domain/types.py:38-55` — POST_RACE_COLS定義（16列）。漏洩検出の基準
- `tests/test_post_race_leakage.py` — 3層漏洩検出CIテスト。新特徴量追加後も通過必須
- `scripts/analyze_feature_importance.py` — feature importance監査CLI。--all-models対応

### 要件・研究ドキュメント
- `.planning/REQUIREMENTS.md` — DATA-01, DATA-02, DATA-03, DATA-04の要件定義
- `.planning/ROADMAP.md` — Phase 26 Success Criteria
- `.planning/research/FEATURES.md` — 特徴量ランドスケープ研究。TS-05, D-01, D-02, D-08等の詳細分析
- `.planning/phases/25-quick-win-wire-existing/25-CONTEXT.md` — Phase 25決定（12特徴量配線パターン）
- `.planning/phases/24-feature-audit-pruning/24-CONTEXT.md` — Phase 24決定（Tier分類、監査パターン）
- `.planning/phases/23-safety-gate/23-CONTEXT.md` — Phase 23決定（漏洩防止、監査スクリプト）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **intra_race_features.py**: groupby("race_id")変換の既存パターン。weight_diff_from_meanとodds_rankの実装が相対特徴量モジュールの参考になる
- **bloodline_features.py**: Beta(1,10)平滑化パターン + horse_career_stats.parquetからのPIT-safeデータ読み込み。新血統モジュールも同じパターンを使用可能
- **sire_features.py**: sire_career_stats.parquetからの種牡馬/BMS統計読み込み。BMS ketto番号解決済み（bms_wrで実績あり）
- **Phase 23 監査スクリプト**: `scripts/analyze_feature_importance.py`。新特徴量追加後の効果測定に使用可能
- **ETL個別テーブル抽出**: `run_etl.py --tables`で既にサポート済み。n_hansyoku等もetl_tables.yamlに定義済み

### Established Patterns
- **FEATURE_COLSリスト**: モデルクラスにFEATURE_COLS list[str]を定義。Phase 25で12特徴量追加の実績あり
- **新規モジュール → build_all()統合**: feature_engine.pyのbuild_all()に新モジュールの呼び出しを追加するパターン
- **mockベーステスト**: 全テストがDB不要。FEATURE_COLS変更に伴うテスト更新はmockのcolumn list更新のみ
- **POST_RACE漏洩検出**: Phase 23の3層CIテスト。新特徴量追加時も自動的に検証される
- **コードハッシュキャッシュ無効化**: Phase 24で導入。特徴量モジュール変更時に自動キャッシュクリア

### Integration Points
- **新規dam_pedigree_features.py**: n_hansyoku + n_sankuのParquet読み込み → 特徴量計算 → build_all()統合
- **新規relative_features.py**: groupby("race_id")変換 → build_all()統合（compute_intra_race_featuresの後に追加）
- **sire_features.py拡張**: BMS distance_wr, BMS surface_wrの追加（D-08 Claude discretion）
- **n_record特徴量**: n_record Parquet読み込み → コース別タイム指数計算 → build_all()統合
- **n_mining PRE列特徴量**: PIT監査後 → PRE列のみ選択 → 特徴量計算 → build_all()統合
- **ETL実行**: `run_etl.py --mode full --tables n_hansyoku n_record n_mining --start 20140101 --end 20251231`
- **各モデルFEATURE_COLS**: 新特徴量名の追加。Phase 25パターンに従う

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- n_miningのPIT監査は`docs/everyDB2/44-MINING.md`のドキュメント照合を主軸に行う（ユーザー指定）
- n_hansyoku + n_sankuの包括的血統特徴量がDATA-01のコア。単にn_hansyokuだけでなく、繁殖牝馬の産駒成績（n_sanku）も活用する
- 相対比較特徴量はFEATURES.md TS-05推奨の5-10特徴量。新規モジュール（relative_features.py）で実装
- n_recordは静的マスターデータ（PIT-safe）。コースレコード情報からタイム指数等を生成
- FEATURES.mdのD-02（BMS拡張）は血統特徴量と一緒に実装するのが効率的

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 26-EveryDB2 New Features*
*Context gathered: 2026-05-14*
