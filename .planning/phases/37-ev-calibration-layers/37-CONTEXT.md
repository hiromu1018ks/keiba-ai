# Phase 37: OOF Health Infrastructure - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

全OOF成果物が健全性検査を通過し、下流コンポーネント(Phase 38-40のキャリブレータ・ランカー)が信頼できるOOF予測を利用できる状態にする。fail-fast validationで異常なOOF artifactがパイプラインに流入するのを防ぐ。

**In scope:**
- OOF-01: 空OOF保存をfail-fastで禁止
- OOF-02: race_id単位でtrain/valid重複検査（split metadata利用時）
- OOF-03: top1 hit rate > 35% または top1 ROI > 200% の異常検出（profile依存）
- OOF-04: OOF行数が期待行数の70%未満の場合に停止
- OOF-05: fold数 < 3 の場合に停止
- OOF-06: 同一race_idの複数fold混入検査
- OOF-07: is_oof=Trueとfold列を必須化（OOF生成時に記録）
- OOF-08: health manifest生成（artifact別JSON、XCT-08準拠）
- XCT-05: 同一入力から決定的な出力保証
- XCT-08: 全artifactにversion/schema hash/source OOF manifest path含める
- 共通OOFHealthValidatorクラスの新設
- 各OOF生成器へのfold列記録追加
- artifact profile定義（oof_predictions / win_selection_oof）

**Out of scope:**
- 新特徴量の計算
- バックテスト実行
- モデル再学習・ハイパーパラメータチューニング
- InvestmentFeatureFrame設計 (Phase 38)
- MarketAwareWinCalibrator設計 (Phase 39)
- Race-Level Ranker設計 (Phase 40)
- OOF drift detector / Reliability diagram (v2.1+)

</domain>

<decisions>
## Implementation Decisions

### Health check対象範囲とアーキテクチャ
- **D-01:** 共通`OOFHealthValidator`クラスを新設。全OOF検査(OOF-01~08)を統一的に実装
- **D-02:** 各artifactはexplicit artifact profileで検査設定を定義。profileは enabled checks, required columns, score_col, return/odds columns, fold_col, expected row source, manifest path を含む
- **D-03:** 常時有効な検査: OOF-01(empty), OOF-04(row coverage), OOF-05(min fold count), OOF-06(same race multiple fold), OOF-07(required metadata), OOF-08(manifest generation)
- **D-04:** Profile依存の検査:
  - OOF-02 (train/valid overlap): split metadataが利用可能な場合のみ有効。metadataが期待されるが存在しない場合はfail-fast
  - OOF-03 (top1 hit rate/ROI): profileがscore_colとreturn/odds columnsを定義している場合のみ有効。win_selection_oofは`win_market_selection_score`、oof_predictionsは`p_win_final_oof`または`p_win_oof`を使用。有効なのに必須列が欠損している場合はfail-fast

### fold列の追加
- **D-05:** fold列はOOF生成時に各生成器でDataFrameに記録する。後からの推測・復元は禁止
  - `AbilityModel.train_oof()` → `ability_oof_fold` 列を追加
  - `generate_ev_oof_predictions()` → `ev_oof_fold` 列を追加
  - `generate_win_oof_predictions()` → `win_oof_fold` / `win_selection_oof_fold` 列を整合的に記録
- **D-06:** `_prepare_oof_artifact()`や`OOFHealthValidator`でのfold推測は禁止。後推測は実際の学習splitと不一致の可能性があり、偽の安全性を与える
- **D-07:** レガシーartifact（fold列なし）はhealth validationでfail。明示的なone-time migrationコマンドでのみ受け入れ可能。migration時はartifactに`legacy/inferred`マークを付け、新calibrator/rankerの学習には使用不可

### Health manifest設計
- **D-08:** JSON manifest。artifact別に個別ファイルで管理:
  - `data/oof/manifests/oof_predictions.health.json`
  - `data/oof/manifests/win_selection_oof.health.json`
- **D-09:** Index file: `data/oof/manifests/index.json`。artifact_name → latest manifest path + artifact hash のマッピング
- **D-10:** Manifest内容: artifact_name, artifact_path, artifact_hash, artifact_version, schema_hash, schema_dtype_hash, row_count, expected_row_count, row_coverage_ratio, race_count, horse_count, fold_col, fold_count, fold_row_counts, fold_race_counts, fold_race_id_uniqueness, train_valid_overlap_count, same_race_multiple_fold_count, top1_score_col, top1_hit_rate, top1_roi, return_col_used, date_min, date_max, train_date_range, source_model_hash, source_code_version/git_commit, generated_at, validator_version, status, failures, warnings
- **D-11:** schema_hash計算: 2種のhashを使用
  - `schema_hash`: 列名ソート→SHA256（既存feature_manifest.jsonと同じパターン）。mismatch時はvalidation fail
  - `schema_dtype_hash`: sorted "column_name:dtype" pairs→SHA256。required列のmismatchはfail、optional列はwarning（profileでstrict指定時はfail）
  - raw Parquet metadata hashは使用しない（圧縮/エンジンmetadataが意味的変更なしに変わる可能性があるため）
- **D-12:** JSONが信頼できる唯一の情報源。Parquet metadataとMLflowは将来のoptional mirror

### Health check実行ポイント（2段階validation戦略）
- **D-13:** 生産者側（学習パイプライン内）: OOF artifact保存前に`OOFHealthValidator.validate()`を必須実行。validation fail時はartifactを書き込まない。保存後にartifact_hashを計算し、health manifestに`status=PASS`で書き込む
- **D-14:** 消費者側（バックテスト/Phase 38/39/40）: artifact-specific manifestを読み込み、status=PASS, artifact_path, artifact_hash, schema_hash, validator_version, required profile を確認。いずれかの不整合やmanifest欠損でfail-fast。artifact_hash/schema_hashが一致する場合はfull validationをスキップ可能
- **D-15:** デバッグ/CI用に`force_revalidate`オプションを提供

### Claude's Discretion
- OOFHealthValidatorの具体的なクラス構造（メソッド分割、profile class定義）
- artifact_hashの計算方法（ファイル全体のSHA256、または特定の列のみ）
- expected_row_countの計算方法（学習データ行数 × (1 - 1/n_folds) 前提）
- fold_col名の統一命名規則（ability_oof_fold / ev_oof_fold / win_oof_fold でよいか）
- 既存`_validate_win_selection_oof_health()`のOOFHealthValidatorへの移行方針
- legacy migrationコマンドのインターフェース
- OOFHealthValidatorのテストファイル配置（tests/test_oof_health_validator.py）
- training_pipeline.py内の統合箇所（既存の`_validate_win_selection_oof_health()`呼び出し2箇所の置き換え）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### OOF生成・保存（主要変更対象）
- `src/pipelines/training_pipeline.py` — `_prepare_oof_artifact()`(line 80), `_prepare_win_selection_oof_artifact()`(line 98), `_validate_win_selection_oof_health()`(line 237), `_walk_forward_race_splits()`(line 197), `generate_ev_oof_predictions()`(line 1701), OOF保存箇所(line 532-541, 543-559)。全てのOOF生成・保存・検証の拡張対象
- `src/models/stage1_ability_model.py` — `train_oof()`(line 350)。fold列(ability_oof_fold)追加の対象
- `src/models/win_benter_gate.py` — `generate_win_oof_predictions()`。win_oof_foldの整合的記録対象

### 既存キャリブレーション・検査パターン（参考）
- `src/pipelines/training_pipeline.py` — 定数定義: `WIN_SELECTION_OOF_MAX_TOP1_HIT_RATE=0.35`, `WIN_SELECTION_OOF_MAX_TOP1_ROI=2.0`, `WIN_SELECTION_OOF_MIN_GUARD_RACES=30` (lines 62-64)
- `src/pipelines/training_pipeline.py` — `_win_selection_oof_return_unit()`(line 176)。ROI計算パターンの参考
- `data/oof/oof_predictions.parquet` — メインOOF artifact（現在fold列なし）
- `data/oof/win_selection_oof.parquet` — 単勝選定OOF artifact（fold列あり）

### 要件定義
- `.planning/REQUIREMENTS.md` §OOF Health — OOF-01~08
- `.planning/REQUIREMENTS.md` §Cross-Cutting — XCT-05, XCT-08

### Prior Phase Context
- `.planning/phases/34-validation-and-manifest-update/34-CONTEXT.md` — feature manifest SHA256パターン、GPD診断パターン

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_validate_win_selection_oof_health()` (training_pipeline.py:237): top1 hit rate/ROI検証の既存実装。OOFHealthValidatorに抽出・統合可能
- `_walk_forward_race_splits()` (training_pipeline.py:197): race_id単位のexpanding walk-forward split。OOF-06(同一race_id複数fold検査)で利用
- `_prepare_oof_artifact()` (training_pipeline.py:80): is_oof, oof_artifact_version, oof_row_idの付与パターン。OOF-07の拡張基盤
- `_win_selection_oof_return_unit()` (training_pipeline.py:176): OOF ROI計算ヘルパー。OOFHealthValidatorで再利用可能
- 既存SHA256 manifestパターン: strategy_manifest.json / feature_manifest.json と同じJSON + SHA256パターン

### Established Patterns
- Fail-fast validation: ValueError raise で異常時停止（既存の_validate_win_selection_oof_healthパターン）
- Artifact versioning: `oof_artifact_version` 整数で管理。現在oof_predictions=1, win_selection_oof=4
- JSON manifest: sort_keys=True + indent=2 でdeterministic。SHA256 hashでartifact integrity確認
- OOF生成は各モデル内でwalk-forward K-fold。fold境界はrace_dateベースの等分割点
- テストは全てmock使用（DB不要）。OOFHealthValidatorのテストもmockで実装可能

### Integration Points
- `src/pipelines/training_pipeline.py` lines 532-541: oof_predictions保存直前 → OOFHealthValidator.validate()挿入
- `src/pipelines/training_pipeline.py` lines 543-559: win_selection_oof保存直前 → 既存_validate呼び出しをOOFHealthValidatorに置き換え
- `src/pipelines/training_pipeline.py` lines 1550: WinSelectionGate学習時の_validate呼び出し → OOFHealthValidator消費者側チェックに置き換え
- `src/models/stage1_ability_model.py` train_oof(): fold番号記録の追加（oof_preds Seriesと同時にfold列を設定）
- `src/pipelines/training_pipeline.py` generate_ev_oof_predictions(): fold番号記録の追加（splitsループ内でval_idxにfold番号を関連付け）
- Phase 38/39/40でのOOF読込時: manifest-first消費者側チェックの追加ポイント

</code_context>

<specifics>
## Specific Ideas

- OOFHealthValidatorの配置: `src/validation/oof_health_validator.py` を新設。training_pipeline.pyからimport
- artifact profileはdataclass or TypedDictで定義。OOFHealthValidatorProfile(protocol) + 具体profileクラス2つ
- expected_row_countの計算: 学習データ行数 × (1 - 1/n_folds) を基準とし、70%を閾値とする
- 生産者側validation flow: generate → validate(full) → save artifact → compute artifact_hash → write manifest(status=PASS)
- 消費者側validation flow: load manifest → check status/artifact_hash/schema_hash → fail or proceed
- 既存の3つの閾値定数(WIN_SELECTION_OOF_MAX_TOP1_HIT_RATE等)はartifact profileに移動し、OOFHealthValidator内部のハードコードを排除
- AbilityModel.train_oof()のfold列追加: boundaries配列のindexをfold番号として、test_maskの行にfold番号を記録するpd.Seriesを返す

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 37-OOF Health Infrastructure*
*Context gathered: 2026-05-27*
