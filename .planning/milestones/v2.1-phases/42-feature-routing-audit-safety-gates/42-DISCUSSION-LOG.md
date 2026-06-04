# Phase 42: Feature Routing Audit & Safety Gates - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 42-Feature Routing Audit & Safety Gates
**Areas discussed:** 特徴量ルーティング監査手法, OOF健全性検証範囲, 配備ゲート条件の具体化

---

## 特徴量ルーティング監査手法

### Q1: 実装アプローチ

| Option | Description | Selected |
|--------|-------------|----------|
| ユニットテストベース | FEATURE_COLS定数とキャリブレータ特徴量の交差をテストで検出。CI自動実行。 | |
| 専用監査スクリプト | run_feature_audit.pyでFEATURE_COLSを静的スキャン。マニフェスト機能。 | |
| 両方 | ユニットテスト(CI常時保証) + 監査スクリプト(JSON/Markdown レポート)。 | ✓ |

**User's choice:** 両方実装。テストでfail-fast検出 + スクリプトで監査レポート。安全保証はテストで担保。
**Notes:** ユニットテストはMarketModel/RaceQualityScreenerのFEATURE_COLSとPhase39/40専用特徴量の交差をfail-fast検出。スクリプトはFEATURE_COLSの交差、許可リスト、禁止特徴量、対象モデル別の監査結果をJSON/Markdown出力。

### Q2: 監査スコープ

| Option | Description | Selected |
|--------|-------------|----------|
| 新規特徴量のみ | Phase 39-40キャリブレータ/ランカー特徴量のみスキャン。MarketModel + RaceQualityScreener対象。 | |
| 全モデルの全交差監査 | 全モデル(Stage1/Win/Place/Wide/EVCorrection/Regime等)の全FEATURE_COLSを相互監査。 | |

**User's choice:** Phase 39-40新規投資系特徴量を禁止特徴量セットとして定義。MarketModel/RaceQualityScreenerは必須監査(fail-fast)、他モデルは参考監査(warning/report)。
**Notes:** 低コストでFEATURE_COLSを取得できる既存モデルは参考監査対象に含める。fail-fast対象はMarketModel/RaceQualityScreenerのみ。

### Q3: 禁止特徴量セットの定義方法

| Option | Description | Selected |
|--------|-------------|----------|
| ハードコードリスト | テストファイル内にFORBIDDEN_*_FEATURESをハードコード。明示的。 | |
| モデルクラス定義から自動取得 | MarketAwareWinCalibrator.FEATURE_COLS等をソースとして使用。自動追従。 | |

**User's choice:** `src/audit/feature_routing_registry.py` に単一監査レジストリを定義。テストとスクリプトが同じレジストリを参照。実際のモデルFEATURE_COLSとの差分テストで更新漏れも検出。
**Notes:** ハードコードではなく監査レジストリを唯一の真実にする。二重安全策。

### Q4: 監査スクリプト出力フォーマット

| Option | Description | Selected |
|--------|-------------|----------|
| JSON + Markdown | JSONはCI/自動判定用。Markdownはレビュー用。 | ✓ |
| JSONのみ | マシン読み取り専用。 | |

**User's choice:** JSON + Markdown。JSON: モデル別status, forbidden_intersections, warning_intersections, checked_feature_count。Markdown: fail-fast/warning分離表示、レジストリバージョン、実行日時、FEATURE_COLS取得元。

---

## OOF健全性検証範囲

### Q5: 検証範囲

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 39-40新規コンポーネントのみ | MarketAwareWinCalibrator/RaceLevelRankerのOOF生成パスのみ。実装コスト最小。 | |
| 学習パイプライン全体E2E | 全OOFアーティファクトの健全性を一括検証。バックテストに近い統合テスト。 | |

**User's choice:** 二層構成。CI: Phase39-40新規OOF生成パスをmockベースで検証 + 軽量統合テスト。フルE2Eは手動/nightly監査コマンドとして用意。
**Notes:** CI必須は新規コンポーネントのみ。実データフルE2EはCI必須にしない。

### Q6: Anomaly定義

| Option | Description | Selected |
|--------|-------------|----------|
| 既存定義の再利用 | Phase 37 OOFHealthValidatorの既存anomaly検出ロジックをそのまま使用。 | |
| コンポーネント固有anomaly追加 | キャリブレータ/ランカー固有のanomalyをOOFHealthValidatorに追加。 | |

**User's choice:** Phase 37既存anomaly定義を共通基盤として再利用。Phase 39/40固有チェックはartifact profile/plugin的に分離。
**Notes:** MAWC profile: 確率NaN/inf, [0,1], sum-to-1.0, p_win_pred混入禁止, fold列必須。Ranker profile: score NaN/inf, 順位決定性, fold列必須。OOFHealthValidator本体に汎用性の低いロジックを直書きしない。

---

## 配備ゲート条件の具体化

### Q7: 実装レベル

| Option | Description | Selected |
|--------|-------------|----------|
| 文書化のみ | ゲート条件を文書化して手動判定。DEP-01(v2.2)で自動化。 | |
| 閾値判定を実装 | RacePredictor内で具体的な閾値チェック。DEP-01の自動デプロイ判定は実装しない。 | |
| 設定駆動ゲート | 設定ファイル(yaml/json)でゲート条件を定義。 | |

**User's choice:** 独立ゲート評価器 `src/backtest/deployment_gates.py` に `DeploymentGateEvaluator` を実装。Phase 41のshadow comparison結果を入力としてPASS/FAIL/WARNを出す。RacePredictorには閾値判定を入れない。ゲート条件はGatePolicy dataclassまたは小さなJSONで固定。
**Notes:** 自動本番切替はしない。RacePredictorはfeature flag + deployment_statusを尊重するだけ。

### Q8: GatePolicy具体条件

| Option | Description | Selected |
|--------|-------------|----------|
| 明示的閾値セット | Brier/logloss/ECE非悪化、ベット数0.95倍以上等の具体的閾値を定義。 | |
| 構造のみ、閾値は後で調整 | 閾値の構造のみ定義、数値はshadow comparison実行後に入力。 | |

**User's choice:** 明示的な非悪化閾値を定義。確率品質は原則非悪化必須（tolerance ~1e-6）。ベット数0.95倍以上。artifact SHA256一致。OOF PASS + 監査PASS必須。Selection agreementとROIはゲートに使わない。

### Q9: ゲートFAIL時の動作

| Option | Description | Selected |
|--------|-------------|----------|
| レポートのみ | PASS/FAIL/WARNを出力、自動アクションなし。v2.2で自動化拡張可能。 | ✓ |
| 自動shadow_only化 | FAIL時RacePredictorのdeployment_statusを強制shadow_onlyに。 | |

**User's choice:** レポートのみ出力。CLI非ゼロ終了コードでFAIL通知。自動変更はしない。
**Notes:** CI/手動検証で見逃さないように非ゼロ終了コード。自動shadow_only化はv2.2以降。

---

## Claude's Discretion

- 監査レジストリの具体的な特徴量リスト内容（Phase 39/40のモデル定義から正確に抽出）
- OOF artifact profileの実装方法（OOFHealthValidatorへの統合方法）
- DeploymentGateEvaluatorの内部設計
- テスト構造・命名
- GatePolicy dataclassのフィールド設計

## Deferred Ideas

- 自動デプロイ判定（DEP-01）— v2.2+で実装。Phase 42では構造のみ拡張可能にする。
- FAIL時自動shadow_only化 — v2.2+で実装。
