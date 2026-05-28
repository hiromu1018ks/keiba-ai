# Requirements: keiba-ai v2.2 ROI Recovery Analysis

**Defined:** 2026-05-28
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v2.2 Requirements

Requirements for ROI recovery via structural diagnosis and targeted fix.

### Analysis — 確率品質・選定差分分析

- [ ] **DIAG-01**: Shadow Comparisonで2024/2025固定foldのbaseline vs shadow確率品質(Brier/logloss/ECE)を比較し、劣化维度を特定する
- [ ] **DIAG-02**: RaceLevelRankerの選定パターン(baseline vs shadow)を比較し、的中/不的中レースの差分構造を明らかにする
- [ ] **DIAG-03**: actual/predicted比率をsurface、odds_band、popularity_band、probability_rank_band、selected_changed別に比較し、確率キャリブレーションの乖離箇所を特定する。レジーム別分析・レジーム別補正は対象外

### Bisect — フェーズ単位ROI劣化ビセクション

- [ ] **BISECT-01**: v1.7(Phase 34)→v2.0(Phase 38)間でartifact-level bisectを実施し、ROI劣化を引き起こしたフェーズを特定する。フェーズ別コミット/タグ/成果物が残っている範囲で比較し、再現不能なフェーズは既存成果物・git差分・OOF/BTログから原因推定に留める
- [ ] **BISECT-02**: 劣化フェーズのOOF特徴量寄与度(SHAP/gain)を比較し、ROI悪化に寄与した特徴量・パラメータを特定する

### Fix — 構造的修正

- [ ] **FIX-01**: ビセクション・診断結果に基づき、OOF/WFで説明できる構造的欠陥(特徴量ルーティング、キャリブレーション設定等)を修正する
- [ ] **FIX-02**: 修正内容が2024/2025固有係数に依存せず、汎化可能であることをOOF指標で確認する

### Quality — 品質ゲート検証

- [ ] **QUAL-01**: OOFHealthValidator PASS (修正後OOFの健全性検査)
- [ ] **QUAL-02**: FeatureRoutingAudit PASS (50+28禁止特徴量CI安全監査)
- [ ] **QUAL-03**: DeploymentGateEvaluator PASS (確率品質・ベット数維持・再現性・診断の4ゲート)
- [ ] **QUAL-04**: ROI回復傾向確認。必須条件: Brier/logloss/ECE非悪化、actual/predicted非悪化、ベット数維持。ROI 100%達成は目標だが必須条件ではない

## Out of Scope

| Feature | Reason |
|---------|--------|
| レジーム依存分析・レジーム別補正 | v2.2は構造的修正に集中。レジームロジック変更は除外 |
| ROI単独最適化 | 品質指標(Brier/logloss/ECE)とのトレードオフ不可 |
| 2024/2025固有係数チューニング | 汎化性担保 |
| ベット数削減 | 品質による自然除外のみ許容。意図的な削減は不可 |
| 新特徴量追加 | 劣化分析が先、追加はv2.3+で検討 |
| デプロイゲート自動判定(DEP-01) | v2.3+に延期 |
| Optuna 19次元最適化(DEP-02) | v2.3+に延期 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIAG-01 | — | Pending |
| DIAG-02 | — | Pending |
| DIAG-03 | — | Pending |
| BISECT-01 | — | Pending |
| BISECT-02 | — | Pending |
| FIX-01 | — | Pending |
| FIX-02 | — | Pending |
| QUAL-01 | — | Pending |
| QUAL-02 | — | Pending |
| QUAL-03 | — | Pending |
| QUAL-04 | — | Pending |

**Coverage:**
- v2.2 requirements: 11 total
- Mapped to phases: 0
- Unmapped: 11 ⚠️

---
*Requirements defined: 2026-05-28*
*Last updated: 2026-05-28 after initial definition*
