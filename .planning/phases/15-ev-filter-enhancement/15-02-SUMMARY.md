---
phase: 15-ev-filter-enhancement
plan: 02
subsystem: [ml-pipeline, models]
tags: [ev-diagnostics, ece, brier-score, reliability-diagram, calibration, temporal-drift]

# Dependency graph
requires:
  - phase: 15-ev-filter-enhancement
    provides: Plan 01 — SubmodelSet動的閾値フィールド + _compute_ev_threshold() + drift_diagnostics.pyパターン
provides:
  - compute_ev_diagnostics() ECE/Brier分解/Reliability diagram/時系列ドリフト評価
  - console_summary() EV診断コンソールサマリ
  - Pipeline統合: use_ensemble=True時に自動実行
affects: [16-oddsband-recalibration, 17-optuna-optimization]

# Tech tracking
tech-stack:
  added: []
  patterns: [pipeline-integrated-diagnostics, quantile-binning-ece, murphy-brier-decomposition]

key-files:
  created:
    - src/models/ev_diagnostics.py
    - tests/test_ev_diagnostics.py
  modified:
    - src/pipelines/training_pipeline.py

key-decisions:
  - "ECE計算にquantile binning(equal-frequency)を採用 — EV予測の右裾が長いため等幅ビンでは不適切"
  - "Brier scoreをMurphy(1973)分解(reliability/resolution/uncertainty)で評価"
  - "時系列ドリフト追跡を年度別粒度で実装 — 四半期別はデータ不足リスクあり"
  - "パイプライン統合をdrift_diagnostics.pyと同じパターン(use_ensemble guard + TimingContext)で統一"

patterns-established:
  - "Pipeline-integrated diagnostics: モジュール関数+console_summaryパターンでuse_ensemble時に自動実行"
  - "Quantile binning ECE: 右裾の長い確率/EV分布に適したキャリブレーション誤差計算"

requirements-completed: [EVF-02]

# Metrics
duration: 6min
completed: 2026-05-06
---

# Phase 15 Plan 02: EV Diagnostics Module Summary

**ECE/Brier分解/Reliability diagram/時系列ドリフト追跡によるEV推定精度診断モジュールを作成し、パイプラインに統合**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-06T05:56:30Z
- **Completed:** 2026-05-06T06:02:45Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- compute_ev_diagnostics()でEV予測vs実際払戻の相関/RMSE + ECE + Brier分解 + Reliability diagram + 時系列ドリフトを計算
- _train_submodel()のuse_ensemble=TrueパスにEV診断を自動統合(data/backtest/ev_diagnostics_{surface}.jsonに出力)
- テスト11件追加(テストクラス3つ + 統合テスト6件)、全19関連テスト通過

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ev_diagnostics.py module** - `b97f1ce` (feat)
2. **Task 2: Integrate EV diagnostics into training pipeline** - `1a8c92a` (feat)

## Files Created/Modified
- `src/models/ev_diagnostics.py` - EV推定精度診断モジュール (compute_ev_diagnostics, console_summary, _compute_ece, _brier_decomposition, _reliability_diagram_data, _temporal_drift)
- `tests/test_ev_diagnostics.py` - EV診断テスト11件 (ECE/Brier/Reliability/基本診断/JSON/サンプル不足/欠損列/コンソール/時系列)
- `src/pipelines/training_pipeline.py` - drift diagnostics直後にEV診断ブロックを追加 (lines 847-858)

## Decisions Made
- ECE計算にquantile binning(equal-frequency)を採用 — EV予測は右裾が長いため等幅ビンでは不適切
- Brier scoreをMurphy(1973)のreliability/resolution/uncertaintyに分解 — 予測品質の診断性向上
- 時系列ドリフトは年度別粒度 — 四半期別はサンプル不足リスクが高い
- import別名(compute_ev_diag, ev_console_summary)でdrift_diagnostics.pyの同名関数との衝突を回避

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## Next Phase Readiness
- EV診断モジュール完成により、アンサンブルバックテスト実行時にEV推定精度が自動で可視化される
- Phase 16 (OddsBandFilter再キャリブレーション) でEV診断結果を利用可能
- Phase 17 (Optuna最適化) でBrier/ECEを評価指標として活用可能

## Self-Check: PASSED

- FOUND: src/models/ev_diagnostics.py
- FOUND: tests/test_ev_diagnostics.py
- FOUND: b97f1ce (Task 1)
- FOUND: 1a8c92a (Task 2)

---
*Phase: 15-ev-filter-enhancement*
*Completed: 2026-05-06*
