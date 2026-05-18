# Phase 32: Market Cross-Consistency Features - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 32-Market Cross-Consistency Features
**Areas discussed:** ワイドオッズ単一値導出, データマージ設計, 欠損データ戦略

---

## ワイドオッズ単一値導出

### Q1: ワイドオッズのレンジから単一値をどう導出するか

| Option | Description | Selected |
|--------|-------------|----------|
| 中間値 (Recommended) | (oddslow + oddshigh) / 2。最も偏りなし | ✓ |
| 低い方 (oddslow) | 保守的評価。バイアスあり | |
| Claude's discretion | 実装時に選択 | |

**User's choice:** 中間値 (Recommended)
**Notes:** ユーザーはベストプラクティスを追求する方針を明示

### Q2: rl_wide_harville_ratio の計算にどのワイド組合せを使用するか

| Option | Description | Selected |
|--------|-------------|----------|
| ninki=1のみ (Recommended) | 市場最効率部分とHarville乖離。シンプル | |
| 上位3組合せの平均 | より多い情報だが複雑、低人気精度低下 | |
| Claude's discretion | 実装時に選択 | |

**User's choice:** "実装難易度は問わないのでベストプラクティスを追求" (自由入力)
**Notes:** 分析の結果、ninki=1のみがベストプラクティスと判断（Harville精度が高確率組合せで最高）

### Q3: ninki=1のみで確定か

| Option | Description | Selected |
|--------|-------------|----------|
| ninki=1のみで確定 (Recommended) | Harville精度最高、信号クリーン | ✓ |
| ninki=1メイン+分散参考 | 上位3組合せ分散も記録 | |

**User's choice:** ninki=1のみで確定 (Recommended)

---

## データマージ設計

### Q1: ワイド/三連複オッズのDataFrameをどこでロード・マージするか

| Option | Description | Selected |
|--------|-------------|----------|
| build_all()内ロード (MCF-07忠実) | FeatureEngine内部でParquetStore使用 | |
| 共有ユーティリティ (Recommended) | db/readers.pyに共有関数、FeatureEngine純粋変換維持 | |
| Claude's discretion | アーキテクチャ一貫性と重複排除のバランス | ✓ |

**User's choice:** Claude's discretion
**Notes:** MCF-07の目的（重複排除）を達成しつつ、アーキテクチャの一貫性を優先

### Q2: データアクセスにどの層を使用するか

| Option | Description | Selected |
|--------|-------------|----------|
| DataRepository使用 (Phase 29) | Phase 29新規クラス。load_wide_odds()追加 | ✓ |
| readers.py使用 (既存パターン) | 既存load_wide_odds()あり | |
| Claude's discretion | 実装時に選択 | |

**User's choice:** DataRepository使用 (Phase 29)
**Notes:** 全オッズアクセスをDataRepositoryに統一する方針

---

## 欠損データ戦略

### Q1: ワイド・三連複オッズが欠損の場合、特徴量をどう扱うか

| Option | Description | Selected |
|--------|-------------|----------|
| NaNのまま (Recommended) | LightGBMネイティブ処理。シンプル | ✓ |
| フラグ列追加 | 欠損有無を明示。列数増加 | |
| Claude's discretion | 実装時に選択 | |

**User's choice:** NaNのまま (Recommended)

### Q2: 欠損データに対する監視・対応策は必要か

| Option | Description | Selected |
|--------|-------------|----------|
| ログのみ (Recommended) | カバレッジログ出力。モデル振る舞い変更なし | |
| 高欠損年度の除外 | データ品質担保だが学習データ減少 | |
| Claude's discretion | 実装時に選択 | ✓ |

**User's choice:** Claude's discretion
**Notes:** ログ出力等でカバレッジを可視化することを推奨として記録

---

## Claude's Discretion

- build_all()へのワイド/三連複データマージ統合の具体的な実装方法 (D-04)
- 欠損データの監視方法 (D-07)
- market_cross_features.py の内部関数構成
- 各特徴量のエッジケース処理
- テストケース設計
- Harville計算の数値安定性処理

## Deferred Ideas

- 上位3組合せの個別乖離評価 (ninki=1,2,3別列出力) — スコープ外。将来フェーズで三連単ベース特徴量(MCF-08)とともに検討
