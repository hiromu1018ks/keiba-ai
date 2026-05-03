# Milestones

## v1.0 Win Model — 2026-05-03

**Status:** ✅ Shipped
**Phases:** 4 | **Plans:** 7 | **LOC:** 48,528

### Key Accomplishments

1. SHAP/gain特徴量重要度分析 + 6新特徴量(odds-to-ability比、クラス落リバウンド等)追加
2. WinBenterGate実装(基本確率×市場確率ブレンド) + Beta/Isotonicキャリブレーション比較
3. WinSelectionGate + Conformal信頼性推定 + JRA控除率25%考慮edge_threshold調整
4. WFValidationResult + 過学習検出(ROI gap/一貫性/Spearman安定性の3基準) + run_wf_validation.py CLI

### Known Deferred Items

- WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間)
- Human UAT 2項目(04-HUMAN-UAT.md)

### Archive

- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`
