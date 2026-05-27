# Phase 37: OOF Health Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 37-OOF Health Infrastructure
**Areas discussed:** Health check対象範囲, fold列の追加箇所, Health manifest設計, Health check実行ポイント

---

## Health check対象範囲

### Q1: Health checkの適用対象

| Option | Description | Selected |
|--------|-------------|----------|
| 両方に適用 | oof_predictions + win_selection_oof 全health check適用 | ✓ |
| oof_predictionsのみ | win_selection_oofは既存検査のまま | |
| 共通基底クラス | 共通OOFHealthValidator + artifact固有サブクラス | |

**User's choice:** 両方に適用。共通OOFHealthValidatorを作成し、空保存禁止/row/race/fold/schema/manifest検査は共通化。top1 hit rate/ROI等のartifact固有検査はprofileまたはサブクラスで分ける。Phase 38/39はoof_predictions(p_win_oof等)を、Phase 39/40はwin_selection_oof(選定/ranker学習用)を参照するため両artifactともhealth check対象。

### Q2: OOF-01~08の共通実装範囲

| Option | Description | Selected |
|--------|-------------|----------|
| OOF-01~06全て共通 | 全検査を共通validatorに実装 | |
| 最小共通セット | 空OOF禁止と行数70%のみ共通 | |
| 共通化 + 設定切替 | 全検査共通化 + 有効/無効設定ファイル | ✓ (custom) |

**User's choice:** 全OOF health checkを共通OOFHealthValidatorに実装。各artifactはexplicit profileで設定を定義。常時有効: OOF-01,04,05,06,07,08。Profile依存: OOF-02(split metadata必要時), OOF-03(score_col + return/odds列必要時)。欠損fold metadataはlegacy artifact failureとして扱い、事後推測で修復しない。

---

## fold列の追加箇所

### Q3: fold列の追加方法

| Option | Description | Selected |
|--------|-------------|----------|
| OOF生成時に記録 | AbilityModel.train_oof/generate_ev_oof_predictionsでfold番号記録 | ✓ |
| artifact準備時に推測 | _prepare_oof_artifact内で_walk_forward_race_splits再実行 | |
| 検査時に復元 | OOFHealthValidator.validate()時に復元して検査 | |

**User's choice:** OOF生成時に記録。後推測は禁止。AbilityModel.train_oof()にability_oof_fold、generate_ev_oof_predictions()にev_oof_fold、generate_win_oof_predictions()にwin_oof_foldを整合的に記録。レガシーartifact（fold列なし）はhealth validation fail。one-time migrationコマンドのみlegacy/inferredマークで受け入れ。新calibrator/ranker学習には使用不可。

---

## Health manifest設計

### Q4: manifest形式

| Option | Description | Selected |
|--------|-------------|----------|
| JSON manifest | 既存strategy_manifest.jsonパターン。data/oof/manifests/に配置 | ✓ (custom) |
| Parquetメタデータ埋め込み | 別ファイル不要だが可視性低い | |
| MLflow artifact | 既存テストmock前提でテスト追加コスト高 | |

**User's choice:** JSON manifest。artifact別個別ファイル(oof_predictions.health.json, win_selection_oof.health.json) + index.json。30+フィールドのmanifest内容を定義。OOF消費者はartifact-specific manifestを読み込み、artifact_hash/schema_hashを検証してから使用。JSONが信頼できる唯一の情報源。

### Q5: schema_hash計算方法

| Option | Description | Selected |
|--------|-------------|----------|
| 列名ソート→SHA256 | 既存feature_manifest.jsonパターン。軽量高速 | ✓ (custom) |
| 列名+dtype→SHA256 | より厳密。dtype変更でもmanifest無効 | |
| Parquet raw metadata SHA256 | 最も厳密。圧縮設定で変わる可能性 | |

**User's choice:** 2種のhashを使用。schema_hash(列名ソート→SHA256)はmismatch時fail。schema_dtype_hash(sorted "col:dtype"→SHA256)はrequired列のみfail、optional列はwarning。raw Parquet metadata hashは使用しない。

---

## Health check実行ポイント

### Q6: validation実行タイミング

| Option | Description | Selected |
|--------|-------------|----------|
| 学習直後に必須 | 保存前validate → failならsaveしない | |
| 全消費ポイントで再検査 | 学習 + BT + Phase 38/39読込時 | |
| 学習必須 + 消費時manifest確認 | 生産者full validate + 消費者はmanifest checkのみ | ✓ (custom) |

**User's choice:** 2段階validation戦略。(1)生産者側: 保存前full validate必須、fail時は保存しない、保存後artifact_hash計算→manifest(status=PASS)書き込み。(2)消費者側: manifest読込→status/artifact_hash/schema_hash/validator_version/profile確認。不整合でfail-fast。hash一致時はfull validationスキップ可能。force_revalidateオプションでデバッグ/CI対応。

---

## Claude's Discretion

- OOFHealthValidatorのクラス構造（メソッド分割、profile class定義）
- artifact_hash計算方法（ファイル全体SHA256 vs 特定列のみ）
- expected_row_countの計算方法
- fold_col名の統一命名規則
- 既存_validate_win_selection_oof_health()の移行方針
- legacy migrationコマンドのインターフェース
- テストファイル配置
- training_pipeline.py内の統合箇所

## Deferred Ideas

None — discussion stayed within phase scope
