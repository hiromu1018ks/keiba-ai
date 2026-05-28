# Phase 43: Shadow Diagnosis - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 43-Shadow Diagnosis
**Areas discussed:** 診断実行形態, 劣化次元の分解手法, セグメント定義, 出力フォーマット

---

## 診断実行形態

| Option | Description | Selected |
|--------|-------------|----------|
| 後処理スクリプト | Phase 41成果物を読み込みDIAG-01~03分析。再学習なし、数秒〜数十秒。 | ✓ |
| Phase 41 Framework拡張 | ShadowComparisonFrameworkにdiagnose()メソッド追加。Phase 41変更リスクあり。 | |
| 再実行付き診断 | BacktestEngine再実行で追加インストルメンテーション取得。~40分/年。 | |

**User's choice:** 後処理スクリプト
**Notes:** scripts/run_shadow_diagnosis.py + src/backtest/shadow_diagnosis.py。不足列はmissing_inputsとして明示しPhase 41拡張候補として記録。

---

## 劣化次元の分解手法

| Option | Description | Selected |
|--------|-------------|----------|
| 段階的除外アプローチ | (1)全馬確率品質→(2)selected_changed/unchanged→(3)セグメント別calibration。段階的に除外。 | ✓ |
| 寄与度分解アプローチ | ΔBrierを3次元寄与に分解。因果仮定が必要。 | |
| REQ-ID単位独立分析 | DIAG-01/02/03を独立セクション。次元間相互作用が見えにくい。 | |

**User's choice:** 段階的除外アプローチ
**Notes:** 全馬ベース確率品質→selected_changed/unchanged分離→セグメント別actual/predicted+ECE。ΔBrier/Δlogloss/ΔROIのセグメント別寄与度も診断指標として出力(因果分解とは呼ばない)。

---

## セグメント定義

| Option | Description | Selected |
|--------|-------------|----------|
| 5段階/4段階 | popularity_band [1-3,4-6,7-9,10-14,15+]、probability_rank_band [top1,2-3,4-6,7+] | ✓ |
| 3段階(シンプル) | popularity [1-3,4-9,10+]、prob_rank [top1,2-3,4+]。大穴域の分解が粗い。 | |
| odds_band流用+パーセンタイル | odds_band既存定義を流用。popularity=odds_bandと同一視。 | |

**User's choice:** 5段階/4段階
**Notes:** popularityはオッズ順位、probability_rankはレース内p_win順位。欠損/field_size不足はunknownフォールバック。odds_bandはPhase 41既存定義流用。レジーム別分解なし。

---

## 出力フォーマット

| Option | Description | Selected |
|--------|-------------|----------|
| JSON + HTML | 2ファイル。JSONは機械可読、HTMLはPhase 41パターンの段階的可視化。 | |
| JSON + HTML + Markdown | 3ファイル。Markdownはレビュー/コミット/PR用。 | ✓ |
| JSONのみ | HTML/Markdown最小限。Phase 44/45がJSON消費。 | |

**User's choice:** JSON + HTML + Markdown
**Notes:** JSONはPhase 44/45消費用+missing_inputs。HTMLは段階的可視化。Markdownは主要劣化次元・上位悪化セグメント・missing_inputs・Phase 44/45への推奨入力の要約。

---

## Claude's Discretion

- ShadowDiagnosis クラス内部設計・データフロー
- Jinja2 HTMLテンプレートのレイアウト・スタイリング
- テスト構造・命名
- JSON出力スキーマ設計
- missing_inputs 検出ロジック詳細
- popularity_band / probability_rank_band 計算のエッジケース処理

## Deferred Ideas

- Phase 41出力拡張(popularity_rank, probability_rank列追加) — missing_inputsで特定後に別フェーズで判断
- レジーム別分析 — v2.3+で検討(REQUIREMENTS.md Out of Scope)
- LightGBM LambdaRank shadow variant — Phase 41 D-09でv2.2+延期済み
- 因果分解的アプローチ — 将来検討、Phase 43では診断指標
