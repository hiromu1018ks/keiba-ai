---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: ROI Recovery Analysis
status: planning
last_updated: "2026-05-30T16:00:00.000Z"
---

# Roadmap: keiba-ai Win Model Improvement

## Milestones

- **v1.0 Win Model** -- Phases 1-4 (shipped 2026-05-03)
- **v1.1 ROI Advanced Model** -- Phases 5-7 (shipped 2026-05-03)
- **v1.2 Win Backtest Validation** -- Phases 8-10 (shipped 2026-05-04)
- **v1.3 Betting Strategy Optimization** -- Phases 11-13 (shipped 2026-05-05)
- **v1.4 Ensemble Filter Recalibration** -- Phases 14-18 (shipped 2026-05-07)
- **v1.5 Model Accuracy Improvement** -- Phases 19-22 (shipped 2026-05-10)
- **v1.6 Feature Engineering Overhaul** -- Phases 23-28 (shipped 2026-05-17)
- **v1.7 Market-Independent Edge Discovery** -- Phases 29-34 (shipped 2026-05-19)
- **v1.8 Turf Precision Calibration** -- Phases 35-36.1.1 (shipped 2026-05-20)
- **v2.0 Investment Pipeline Restructuring** -- Phases 37-38 (shipped 2026-05-27)
- **v2.1 MarketAware Calibration + Race-Level Ranker** -- Phases 39-42 (shipped 2026-05-28)
- **v2.2 ROI Recovery Analysis** -- Phases 43-46 (in progress)

## Phases

<details>
<summary>v1.0-v2.1 (Phases 1-42) -- All Shipped</summary>

Phases 1-42 complete across milestones v1.0 through v2.1.
See `.planning/milestones/` for archived roadmaps.

</details>

### v2.2 ROI Recovery Analysis (In Progress)

- [x] **Phase 43: Shadow Diagnosis** -- 確率品質・選定差分の全面比較 (DIAG-01~03) (completed 2026-05-29)
- [ ] **Phase 44: ROI Bisect** -- DeploymentGate FAILのコンポーネント帰属 + MAWC/Ranker係数分析 (BISECT-01~02)
- [ ] **Phase 45: Structural Fix** -- 診断結果に基づく構造的修正と汎化確認 (FIX-01~02)
- [ ] **Phase 46: Quality Gate Verification** -- 全品質ゲートPASS確認 (QUAL-01~04)

## Phase Details

### Phase 43: Shadow Diagnosis
**Goal**: baseline vs shadowの確率品質・選定パターン・キャリブレーション乖離を完全に特定し、劣化の次元を明らかにする
**Depends on**: v2.1 complete (Phase 42)
**Requirements**: DIAG-01, DIAG-02, DIAG-03
**Success Criteria** (what must be TRUE):
  1. 2024/2025固定foldのBrier/logloss/ECEがbaseline vs shadowで数値比較され、劣化维度(確率精度 vs 選定 vs キャリブレーション)が特定されている
  2. RaceLevelRankerの選定パターン差分(的中/不的中レース構造、選定変更レース一覧)がレポート化されている
  3. surface/odds_band/popularity_band/probability_rank_band/selected_changed別のactual/predicted比率乖離箇所が特定されている
**Plans**: 2 plans

Plans:
- [x] 43-01-PLAN.md -- ShadowDiagnosis クラス実装 (3ステップ段階的除外診断ロジック + テスト)
- [x] 43-02-PLAN.md -- CLI + HTML レポート + Markdown 要約 (D-04 出力フォーマット)

### Phase 44: ROI Bisect
**Goal**: DeploymentGate FAILの直接原因をコンポーネント単位(MAWC/Ranker/OBF/Selection)で帰属し、MAWC/Ranker係数分析で悪化寄与特徴量を特定する
**Depends on**: Phase 43
**Requirements**: BISECT-01, BISECT-02
**Success Criteria** (what must be TRUE):
  1. Phase 34→38間のartifact-level bisectで、ROI劣化を引き起こしたフェーズ(またはフェーズ群)が特定されている
  2. 再現不能なフェーズはgit差分・OOF/BTログ・既存成果物から原因推定が文書化されている
  3. 劣化フェーズのOOF特徴量寄与度(SHAP/gain)がbaseline vs当該フェーズで比較され、ROI悪化に寄与した特徴量・パラメータが特定されている
**Plans**: 2 plans

Plans:
- [ ] 44-01-PLAN.md -- ComponentAttribution分析エンジン (逐次帰属 + 係数分析 + 条件付き上流SHAP) + HistoricalBisect (補助的v1.7→v2.0比較)
- [ ] 44-02-PLAN.md -- CLI + JSON/MD/HTML出力 (ReportGenerator分離モジュール + Phase 45消費可能な帰属成果物)

### Phase 45: Structural Fix
**Goal**: ビセクション・診断結果に基づく最小限の構造的修正を適用し、その汎化性を確認する
**Depends on**: Phase 44
**Requirements**: FIX-01, FIX-02
**Success Criteria** (what must be TRUE):
  1. 診断・ビセクションで特定された構造的欠陥(特徴量ルーティング、キャリブレーション設定等)に対する修正が実装されている
  2. 修正がOOF/WF指標で説明可能であることが確認されている
  3. 修正内容が2024/2025固有係数に依存せず、OOF指標で汎化性が確認されている
**Plans**: TBD

### Phase 46: Quality Gate Verification
**Goal**: 全修正が安全ゲートを通過し、ROI回復傾向と品質指標非悪化が確認されている
**Depends on**: Phase 45
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04
**Success Criteria** (what must be TRUE):
  1. OOFHealthValidatorがPASSしている(修正後OOFの健全性検査)
  2. FeatureRoutingAuditがPASSしている(50+28禁止特徴量CI安全監査)
  3. DeploymentGateEvaluatorがPASSしている(確率品質・ベット数維持・再現性・診断の4ゲート)
  4. Brier/logloss/ECEがbaselineに対して非悪化、actual/predicted比率が非悪化、ベット数が維持されている
  5. ROIが87.8%(v2.0)から回復傾向を示している
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 43 -> 44 -> 45 -> 46

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-42 | v1.0-v2.1 | 85/85 | Complete | 2026-05-28 |
| 43. Shadow Diagnosis | v2.2 | 2/2 | Complete | 2026-05-29 |
| 44. ROI Bisect | v2.2 | 0/2 | Planned | - |
| 45. Structural Fix | v2.2 | 0/? | Not started | - |
| 46. Quality Gate Verification | v2.2 | 0/? | Not started | - |
