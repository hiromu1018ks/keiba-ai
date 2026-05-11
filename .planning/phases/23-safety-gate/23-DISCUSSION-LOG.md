# Phase 23: Safety Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 23-Safety Gate
**Areas discussed:** 漏洩修正の適用範囲, 監査スクリプトの設計, CIテストの網羅性

---

## 漏洩修正の適用範囲

### Q1: Spike M1-M6のどれをPhase 23で修正するか

| Option | Description | Selected |
|--------|-------------|----------|
| M1のみ（最小限） | build_all()の最後でPOST_RACE_COLSをドロップ。Success Criteria #1を満たす | |
| 全5件をPhase 23で修正 | M1+M2+M3+M6の構造的リスクを一括解消 | ✓ |
| M1 + M2のみ | CQR whitelist化を含む。M3/M6はPhase 24に先送り | |
| Claudeに判断を任せる | M1のみをPhase 23、M2-M6はPhase 24 | |

**User's choice:** 全5件をPhase 23で修正（ベストプラクティスを追求）
**Notes:** ユーザーは一貫して「実装難易度は問わないのでベストプラクティスを追求」と回答

### Q2: build_all()のどのタイミングでPOST_RACE_COLSをドロップするか

| Option | Description | Selected |
|--------|-------------|----------|
| キャッシュ前にdrop | キャッシュにはクリーンなDataFrameのみ保存。以後の読み出しは常に漏洩なし | ✓ |
| return直前でdrop | キャッシュデータにはPOST_RACE_COLSが残るが、API的には常に漏洩なし | |

**User's choice:** キャッシュ前にdrop（ベストプラクティスを追求）

### Q3: CQRの特徴量抽出方式

| Option | Description | Selected |
|--------|-------------|----------|
| 明示的FEATURE_COLS（whitelist） | 他モデルと同じ設計パターンに統一。将来の列追加時の漏れリスク排除 | ✓ |
| ブラックリスト強化 | 除外セットを拡張するが、whitelist化はしない | |

**User's choice:** 明示的FEATURE_COLS（ベストプラクティスを追求）

---

## 監査スクリプトの設計

### Q4: 既存スクリプト拡張か新規作成か

| Option | Description | Selected |
|--------|-------------|----------|
| 既存スクリプト拡張 | analyze_feature_importance.pyを拡張して全モデル対応 | ✓ |
| 新規スクリプト作成 | scripts/audit_feature_importance.pyを新規作成 | |

**User's choice:** 既存スクリプト拡張

### Q5: 対象モデルの範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 全モデル一括 | Stage1Ability + Win2Stage + Place2Stage + EVCorrection | ✓ |
| Stage1 + Win2Stageのみ | 最も影響の大きいモデルに集中 | |

**User's choice:** 全モデル一括（ベストプラクティスを追求）

### Q6: 出力形式

| Option | Description | Selected |
|--------|-------------|----------|
| CSV + JSONの両方 | CSV: ピボットテーブル形式、JSON: 構造化データ | ✓ |
| CSVのみ | 人間が確認しやすい | |
| JSONのみ | 自動処理に最適 | |

**User's choice:** CSV + JSONの両方

---

## CIテストの網羅性

### Q7: 検証レイヤー

| Option | Description | Selected |
|--------|-------------|----------|
| 3層検証 | build_all出力 + FEATURE_COLS + predict()入力の全レイヤー | ✓ |
| build_all()のみ | 最も効果的な検証ポイント | |
| build_all + FEATURE_COLS | predict()入力検証はmockでは困難なため除外 | |

**User's choice:** 3層検証（ベストプラクティスを追求）

### Q8: テストファイル配置

| Option | Description | Selected |
|--------|-------------|----------|
| 新規ファイル | tests/test_post_race_leakage.pyを作成 | ✓ |
| 既存ファイルに追加 | tests/test_leakage_validators.pyに追加 | |

**User's choice:** 新規ファイル

---

## Claude's Discretion

- 具体的なPOST_RACE_COLS dropの実装箇所（feature_engine.pyの行位置）
- キャッシュキー計算への影響判定
- CQRのFEATURE_COLSの具体的な列選定基準
- M3（EV correction odds）の具体的な修正方法
- M6（popularity_rank）のフォールバック先の代替
- 監査スクリプトのCLI引数設計
- permutation重要度の計算パラメータ（n_repeats, scoring等）
- テストのfixtureデータとモック構成
- predict()入力検証のテスト実装方法

## Deferred Ideas

None — discussion stayed within phase scope
