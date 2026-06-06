# Phase 52: Shared Feature Builder & Consistency - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-06
**Phase:** 52-Shared Feature Builder & Consistency
**Areas discussed:** 抽出範囲の境界, 検証基盤の戦略, BT内部重複の解消方針

---

## 抽出範囲の境界

### Q1: build_inference_features() の開始点

| Option | Description | Selected |
|--------|-------------|----------|
| build_all()の後から | build_inference_features() は FeatureEngine.build_all() の後に呼ばれる。7ギャップのみ担当。変更範囲最小。 | |
| build_all()を含む全体 | build_inference_features() が build_all() も内部で呼び出す。単一エントリポイント。 | ✓ |

**User's choice:** build_all()を含む全体
**Notes:** Phase 52の目的は単なる7ギャップ解消ではなく「入力から完成特徴量までの一貫性保証」。後段だけ共有するとstore渡し忘れやオッズ設定差を防げない。内部をbuild_base_features()とenrich_features()に分割。RacePredictor内重複特徴量計算は撤去。

### Q2: 配置場所

| Option | Description | Selected |
|--------|-------------|----------|
| FeatureEngineクラス内 | 既存クラスにメソッド追加。import路変更なし。 | |
| 新モジュール feature_builder.py | src/features/feature_builder.py に FeatureBuilder クラス。FeatureEngineは下請け。 | ✓ |
| データ層に統合 | src/db/ に配置。既存アーキテクチャから逸脱。 | |

**User's choice:** 新モジュール feature_builder.py (名前はinference_builder.pyではなくfeature_builder.pyを推奨)
**Notes:** 学習にも使うため名前は用途を限定しない。FeatureEngineへ追加すると既に大きいクラスが肥大化。FeatureEngineは基礎特徴量生成、FeatureBuilderは全追加モジュールのオーケストレーションとmanifest生成。

### Q3: 戻り値設計

| Option | Description | Selected |
|--------|-------------|----------|
| DataFrame + FeatureManifest tuple | FeatureBuildResult dataclass。manifestでMLflow記録とPT検証が可能。 | ✓ |
| DataFrameのみ、manifest別計算 | 柔軟だが、一致性検証を呼び出し元に委ねる。 | |

**User's choice:** FeatureBuildResult dataclass (frozen)
**Notes:** 単純なtupleよりdataclass推奨。manifest hash対象: モデル入力列の名前・順序・dtype・特徴量定義バージョン。race_id/ターゲット/POST_RACE/構築日時/データ値はhash除外。構築日時をhashに含めると毎回異なるため一貫性検証に使えない。

### Q4: 学習時と推論時の計算モード切替

| Option | Description | Selected |
|--------|-------------|----------|
| mode='train'\|'inference' 引数 | 呼び出し側はモード明示のみ。 | |
| 別メソッド分離 | build_for_training()とbuild_for_inference()。型で区別。 | ✓ |
| コンテキスト注入 | TrainContext/InferenceContext注入。柔軟だが複雑。 | |

**User's choice:** 別メソッド分離
**Notes:** TargetEncoderのfitはFeatureBuilderに持たせない（責務過多）。TargetEncoder/OOF予測/モデル校正はモデル成果物として管理。FeatureState対象: track_stats, track_month_stats, 特徴量定義バージョン。推論時に状態が欠けていればfail-fast。

### Q5: PIT境界とdtype正規化

| Option | Description | Selected |
|--------|-------------|----------|
| FeatureBuilder内で自動処理 | 一律シフト+dtype正規化。 | |
| 既存モジュールに委ね、最終coerceのみ | PITは各モジュール責務、FeatureBuilderは検証層。 | ✓ |
| 既存実装を信頼、新規処理なし | テストで検出する。 | |

**User's choice:** 既存モジュールに委ね、最終coerceのみ
**Notes:** FeatureBuilderで一律シフトすると既にshift(1)済みの特徴量を二重シフトする危険。FeatureBuilderはregistryでPIT契約を管理し、推論時にmax(race_date)<prediction_dateを検証する「検証する調整層」にする。

---

## 検証基盤の戦略

### Q6: コードハッシュの計算方法

| Option | Description | Selected |
|--------|-------------|----------|
| Git commit SHA | 手軽で普遍的。uncommitted changes検知不可。 | |
| ファイル内容SHA256 | uncommitted changesも検知。ファイルリスト保守必要。 | |
| Git SHA + dirty検知 | commit SHA + dirty flag + dirty_diff_hash。両方の利点。 | ✓ |

**User's choice:** Git SHA + dirty検知
**Notes:** dirty時は対象コード差分のSHA256をdirty_diff_hashとして保存。対象: src/, scripts/run_paper_trading.py, 設定ファイル。未追跡ファイル有無も記録。通常PT runはdirty状態を拒否、開発用フラグ指定時のみ警告付き許可。

### Q7: データカットオフ検証(PLN-03)の範囲とタイミング

| Option | Description | Selected |
|--------|-------------|----------|
| 起動時一括検証 | DataCutoffManifestで一括。不整合時fail-fast。 | |
| 起動時 + 各モジュール内 | より厳密だがモジュール改修必要。 | |
| manifestバージョンのみ | データ日付は検証せず。 | |

**User's choice:** 起動時DataCutoffManifest一括検証 + FeatureBuilder実行時参照履歴データmax日付検証
**Notes:** 作成日時やFeatureManifestバージョンはデータカットオフの代用にしない。不明・欠落もfail-fast。各モジュール個別改修よりFeatureBuilderが参照データの来歴を集約して検証する方が保守しやすい。

### Q8: PFP検証(PLN-04)の実行タイミング

| Option | Description | Selected |
|--------|-------------|----------|
| 起動freeze + 終了verify | 既存パターン踏襲。 | |
| 起動freeze + レース毎verify | より厳密。オーバーヘッドあり。 | |
| 起動freezeのみ | verifyは外部ツール。 | |

**User's choice:** 起動freeze + レース予測直前verify + 終了時verify
**Notes:** 検証対象: モデルHP, FeatureState, feature manifest, strategy manifest, OddsBandFilter, betting target/mode。除外: RegimeDetector, DDController等ランタイム状態（実行中に意図的に変化する）。verify失敗時は以降予測停止→既存記録保存→非ゼロ終了。SHA256検証コストは数分間隔の競馬運用では無視可能。

### Q9: パイプライン識別情報(PLN-02)の保存先

| Option | Description | Selected |
|--------|-------------|----------|
| MLflow + ローカルJSON | MLflow tags/params + ローカルJSON。 | |
| bets.parquet内 | 各bet行に記録。列肥大化懸念。 | |
| MLflowのみ | ローカルファイルは最小限。 | |

**User's choice:** ローカルsession_manifest.jsonを正本、MLflowへ複製
**Notes:** bets.parquetにはsession_id + model_run_idのみ保存、詳細はmanifest参照。manifestはrun開始前にatomic write、終了状態・PFP検証結果・終了コードを追記。MLflow障害時もクラッシュ復旧・監査が可能。

---

## BT内部重複の解消方針

### Q10: BacktestEngineの2つの特徴量構築パスの扱い

| Option | Description | Selected |
|--------|-------------|----------|
| 両方をFeatureBuilder呼出に変更 | prepare_data()とrun()内部の両方を置換。WF互換性維持。 | ✓ |
| run()内部のみ変更 | prepare_data()はWF用に現状維持。重複残存。 | |
| prepare_data()削除・統合 | WF側もrun()経由に変更。 | |

**User's choice:** 両方をFeatureBuilder呼出に変更
**Notes:** prepare_data()はWF向け薄い互換ラッパーとして維持。run(prepared_data=None)も同じ内部ヘルパー経由。回帰テスト: 両経路で同一入力のfeature manifest hash + 主要列値が一致することを検証。旧インライン実装は完全削除、フォールバックとして残さない。

---

## Claude's Discretion

- FeatureBuilder 内部の _build() メソッドの実行順序・マージ方法
- FeatureManifest / FeatureState / FeatureBuildResult dataclass のフィールド定義
- DataCutoffManifest の具体的な検証ロジック
- FeatureBuilder PIT registry の実装形式
- session_manifest.json のスキーマ定義
- 各特徴量モジュールのFeatureBuilder統合時のエラー処理
- dtype正規化の具体的なcoerceルール
- FeatureBuilder と RacePredictor の境界線の細部

## Deferred Ideas

- Strategy manifest integration — Phase 53 (STR-01~06)
- Live data fetcher — Phase 53 (LIV-01~03)
- Regime synchronization — Phase 53 (STR-06)
- One-command run mode — Phase 54 (AUT-01~03)
- Weekly/cumulative reporting — Phase 54 (RPT-01~04)
