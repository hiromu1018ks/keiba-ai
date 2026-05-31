# Phase 46: Quality Gate Verification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-31
**Phase:** 46-quality-gate-verification
**Areas discussed:** 実行フロー設計, ROI「回復傾向」の定義, ゲートFAIL時対応, 最終成果物とマイルストーン完了証明

---

## 実行フロー設計

| Option | Description | Selected |
|--------|-------------|----------|
| 単一スクリプト全自動 | 1つのCLIでRetrain→Shadow→Gate→Audit→Report全自動。中間成果物は保存 | |
| 複数ステップ手動確認 | 各ステップ間で人間が中間成果物をレビュー | |
| 2段階: Retrain→確認→残り全自動 | Retrainのみ先に確認、残り全自動 | ✓ |

**User's choice:** 2段階設計（カスタマイズ版）。Stage 1でRetrain実行→結果確認（deployed候補なし/Favorite band guard FAILならBLOCKED）。Stage 2でFeatureRoutingAudit→OOFHealthValidator→Shadow Comparison→Shadow Diagnosis→DeploymentGateEvaluator→Final summaryを一括実行。

**Notes:** Stage 2実装はorchestration CLI + runbookの両方作成。Phase 46 CLIは既存コンポーネントをラップするだけで新しい判定ロジックを追加しない。FeatureRoutingAudit/OOFHealthValidatorの呼び出し方法はplan時にコード確認して決定。

---

## Stage 2実装形態

| Option | Description | Selected |
|--------|-------------|----------|
| 単一orchestrationスクリプト | 既存CLI/関数を順次呼び出すPythonスクリプト | |
| Runbook (手順書)のみ | 各CLIコマンドを順次実行する手順書 | |
| 両方: スクリプト + runbook | 自動実行スクリプトと再現可能な手順書の両方 | ✓ |

**User's choice:** 両方作成。`scripts/run_phase46_quality_gates.py` + `.planning/phases/46-quality-gate-verification/46-RUNBOOK.md`。

**Notes:** Phase 46 CLI内ではC値再探索、selection閾値変更、Ranker/OBF変更は一切しない。既存CLIをラップするのみ。成果物は `data/backtest/phase46_quality_gates/phase46_quality_gate_result.json`, `phase46_quality_gate_summary.md`。

---

## ROI「回復傾向」の定義

| Option | Description | Selected |
|--------|-------------|----------|
| 非劣化 (87.8%超え) | 87.8%より高いだけでOK。確率品質改善が主目的 | |
| 90%以上 | 有意な改善として認める水準 | |
| 95%以上 | v1.7水準に近い回復を要求 | |
| ROI閾値なし（品質指標のみ） | Brier/logloss/ECE + actual/predicted + bet_countのみ判定 | |

**User's choice:** 3ラベル分離判定フレームワーク（カスタム回答）。

**Notes:**
- Quality Gate: PASS/FAIL（Brier/logloss/ECE非悪化、actual/predicted非悪化、bet_count >= baseline * 0.95）
- ROI Trend: recovered（90%+）/ weak_recovery（87.8%-90%）/ not_recovered（<87.8%）
- Deployment: deployable / not_deployable / manual_review
- ROI 100%は目標だが必須PASS条件ではない。品質ゲート判定とROIは分離。
- 87.8%未満でも品質ゲート全PASSならmanual_reviewとして記録。

---

## ゲートFAIL時対応

| Option | Description | Selected |
|--------|-------------|----------|
| 記録して終了 | Phase 46内ではリトライしない。not_deployed記録 | ✓ |
| 既存候補内リトライのみ | C gridに複数deployed候補がある場合のみ次点を試す | |
| Phase 45戻り（1回限り） | Phase 45 CLI再実行でC値調整、1回限りリトライ | |

**User's choice:** 記録して終了（Phase 46内では一切リトライしない）。

**Notes:** 例外として実行系バグ（ファイルパス、manifest不整合、CLI引数ミス、保存漏れ）のみPhase 46内で修正・再実行可能。モデル修正が必要な場合はPhase 45bまたは次フェーズ。

---

## 最終成果物とマイルストーン完了証明

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 46単位の成果物 | result.json + summary.md + RUNBOOK + VERIFICATION | |
| Phase 46 + マイルストーン完了証明 | Phase 46成果物 + v2.2-MILESTONE-SUMMARY.md | ✓ |
| Phase 46 + マイルストーン + HTML | 上記に加えてHTML report | |

**User's choice:** Phase 46成果物 + v2.2マイルストーン完了証明。HTMLは新規必須成果物にしない。

**Notes:**
- Phase 46成果物: result.json, summary.md, RUNBOOK.md, VERIFICATION.md
- 検証副産物: manifest.json, retrain_summary.md, shadow_comparison_result.json, shadow_diagnosis_result.json, deployment_gate_result.json
- マイルストーン証明: .planning/v2.2-MILESTONE-SUMMARY.md（Phase 43-46 requirement traceability + 最終判定）
- 既存HTML（MAWC/Shadow/Diagnosis）へのリンクをsummaryに記載

---

## Claude's Discretion

- FeatureRoutingAudit/OOFHealthValidatorの具体的呼び出し方法
- skip/resume判定ロジックの詳細設計
- phase46_quality_gate_result.json スキーマ設計
- v2.2-MILESTONE-SUMMARY.md 構造設計

## Deferred Ideas

- Phase 45b（MAWC再調整）: Phase 46でFAIL/not_deployedの場合
- Ranker修正: v2.3+
- OddsBandFilter再学習: v2.3+
- デプロイゲート自動判定(DEP-01): v2.3+
- Optuna 19次元最適化(DEP-02): v2.3+
