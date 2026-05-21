# Phase 18: Validation & Freeze - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 18-Validation & Freeze
**Areas discussed:** バリデーション実行フロー, テスト期間と成功基準, 検証レポートとマイルストーン, 失敗時の対応

---

## バリデーション実行フロー

### 実行方法

| Option | Description | Selected |
|--------|-------------|----------|
| run_backtest.py拡張 (推奨) | --manifest引数追加でmanifest読込→SHA256検証→パラメータ注入→backtest実行を一本化。~30行変更。 | ✓ |
| 新規run_validation.py | 専用スクリプト。manifest読込→verify→backtest→結果判定→レポート生成。 | |
| 手動フロー（スクリプト不変更） | 既存スクリプトそのまま、手動settings.yaml反映。 | |

**User's choice:** run_backtest.py拡張 (推奨)
**Notes:** 最小限のコード変更で一元的なフローを実現

### manifest注入位置

| Option | Description | Selected |
|--------|-------------|----------|
| BacktestEngine.run()内で注入 (推奨) | run()の先頭でmanifest読込→SHA256検証→strategy_paramsマージ。テストもmockで容易。 | ✓ |
| スクリプトレベルで注入 | run_backtest.pyでmanifest読込、BacktestEngineコンストラクタに渡す。 | |

**User's choice:** BacktestEngine.run()内で注入 (推奨)
**Notes:** パイプライン内部で完結し、テスト容易性が高い

### 不変性検証レベル

| Option | Description | Selected |
|--------|-------------|----------|
| SHA256 + PFP二重検証 (推奨) | manifest SHA256検証 + ParameterFreezeProtocol freeze/verifyで二重保証。 | ✓ |
| SHA256のみ（最小限） | strategy manifest SHA256検証のみ。PFPは別途手動。 | |

**User's choice:** SHA256 + PFP二重検証 (推奨)
**Notes:** 最も厳密なパラメータ不変性保証

### 検証失敗時の挙動

| Option | Description | Selected |
|--------|-------------|----------|
| 即時エラー停止 (推奨) | RuntimeErrorで即座に停止。バリデーションの完全性を保証。 | ✓ |
| Warning継続 | warningログで処理続行。開発/デバッグ時向け。 | |

**User's choice:** 即時エラー停止 (推奨)
**Notes:** バリデーションフェーズとして完全性が不可欠

---

## テスト期間と成功基準

### テスト期間

| Option | Description | Selected |
|--------|-------------|----------|
| 2024単年 (推奨) | 最も厳密なOOS検証。学習(2020-2023)と完全に独立。 | |
| 2024-2025の2年 | 2024+2025の2年検証。より多くのベット数で統計的有意性を確保。 | ✓ |
| 2022-2025の4年 | Phase 17の4foldと同じ。2022-2023は学習期間と重なる。 | |

**User's choice:** 2024-2025の2年
**Notes:** より多くのデータポイントで統計的有意性を確保しつつ完全OOS

### 成功基準の解釈

| Option | Description | Selected |
|--------|-------------|----------|
| 全体ROI>100% + 合計100+ベット (推奨) | テスト期間全体の合計で判定。「年間」を「テスト期間全体」と解釈。 | ✓ |
| 年別にROI>100% + 各年100+ベット | 各年度単独で基準達成が必要。より厳しい。 | |
| 全体基準主 + 年別内訳参考 | 全体ROI>100%主基準 + 年別内訳は参考情報。 | |

**User's choice:** 全体ROI>100% + 合計100+ベット (推奨)
**Notes:** 年別内訳は参考情報として記録するが、パス/フェイル判定には使用しない

### 学習期間

| Option | Description | Selected |
|--------|-------------|----------|
| 学習2020-2024 / テスト2025のみ | データ利用効率良いがテスト年度1年のみ。 | |
| 学習2020-2023 / テスト2024-2025 (推奨) | 完全OOS。Phase 17 Optuna最適化の学習期間と整合。 | ✓ |

**User's choice:** 学習2020-2023 / テスト2024-2025 (推奨)
**Notes:** Phase 17の最適化パラメータを汎化性評価する観点でも適切

---

## 検証レポートとマイルストーン

### レポート内容

| Option | Description | Selected |
|--------|-------------|----------|
| 最小限の検証レポート (推奨) | backtest_result + PFP検証結果の組み合わせ。Phase 14-15診断は既存。 | ✓ |
| 包括的マイルストーンレポート | Phase 14-17全診断結果を統合。多くは既存データの再構成。 | |

**User's choice:** 最小限の検証レポート (推奨)
**Notes:** 既存フェーズ出力を再利用し、バリデーション固有の結果のみ新規生成

### 出力形式

| Option | Description | Selected |
|--------|-------------|----------|
| JSON形式 (推奨) | data/validation/に出力。プログラムで読み取り可能。 | ✓ |
| Markdown形式 | .planning/に出力。人間可読。 | |

**User's choice:** JSON形式 (推奨)
**Notes:** 機械可読でパースが容易

### テスト方針

| Option | Description | Selected |
|--------|-------------|----------|
| mockテスト + Human UAT (推奨) | manifest読込/SHA256/PFPの単体テスト + 実データ実行はHuman UAT。 | ✓ |
| Human UATのみ（テストなし） | テストコード追加なし。自動品質保証なし。 | |

**User's choice:** mockテスト + Human UAT (推奨)
**Notes:** 既存プロジェクトパターンに従う

---

## 失敗時の対応

### ROI<100%の場合

| Option | Description | Selected |
|--------|-------------|----------|
| 不完了+改善提案 (推奨) | フェーズ不完了とし、改善案を文書化してユーザー判断を待つ。 | ✓ |
| 結果記録して完了 | ROI<100%でも完了とし、結果を記録。Core Value未達の状態で完了宣言。 | |
| 自動再調整ループ | パラメータ微調整して再実行。過学習リスク高。 | |

**User's choice:** 不完了+改善提案 (推奨)
**Notes:** Core Valueの達成状況を透明に記録し、改善の方向性を明確にする

### 改善提案の内容

| Option | Description | Selected |
|--------|-------------|----------|
| 原因分析レポート自動生成 (推奨) | オッズバンド別/レジーム別ROI、EV診断等を自動分析。 | ✓ |
| 結果記録のみ（手動分析） | ROI結果のみ記録。原因分析は手動。 | |

**User's choice:** 原因分析レポート自動生成 (推奨)
**Notes:** ユーザーが次のアクションを判断しやすい形で情報提供

---

## Claude's Discretion

- BacktestEngine.run()内でのmanifest読込ロジックの具体的な実装
- 検証結果JSONのスキーマ設計
- 原因分析レポートの具体的な分析項目と出力形式
- --manifestと--ensemble/--years引数の組み合わせバリデーション
- テストのfixtureデータの内容
- PFP freeze/verifyのタイミング

## Deferred Ideas

None — discussion stayed within phase scope
