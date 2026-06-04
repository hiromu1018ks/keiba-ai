---
status: resolved
phase: 43-shadow-diagnosis
source: [43-VERIFICATION.md]
started: 2026-05-29T00:45:00.000Z
updated: 2026-05-29T01:00:00.000Z
---

## Current Test

[completed — synthetic data review]

## Tests

### 1. HTML レポートの visual review
expected: ShadowDiagnosis HTML レポートが3ステップ段階的分析をセクション化表示し、Delta悪化が赤色ハイライトされること。テーブルのCSS/レイアウトが崩れていないこと。
result: passed — 3ステップセクション表示、赤色ハイライト動作、5セグメントサブテーブル、Missing Inputs箇条書き、CSS正常を確認

### 2. Markdown summary の可読性確認
expected: shadow_diagnosis_summary.md が Probability Quality / Selection Pattern / Calibration Gaps / Missing Inputs セクションを含み、テーブルフォーマットが正しく表示されること。
result: passed — 5セクション構成、Markdown table正常、上位5キャリブレーションギャップ表示、Recommendations空セクション確認

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
