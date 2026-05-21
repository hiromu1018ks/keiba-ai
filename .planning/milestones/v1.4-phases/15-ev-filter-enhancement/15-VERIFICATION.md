---
phase: 15-ev-filter-enhancement
verified: 2026-05-06T15:25:00Z
status: human_needed
score: 7/7 must-haves verified
overrides_applied: 0
human_verification:
  - test: "run_backtest.py --ensembleを実行し、除外件数の変化を確認する"
    expected: "EV filter除外件数が従来の3,594件から大幅に減少する。ログに 'EV threshold for turf/dirt' と動的閾値の値が出力される"
    why_human: "除外件数の実際の減少はパイプラインのエンドツーエンド実行が必要であり、単体テストでは確認不可"
---

# Phase 15: EV Filter Enhancement Verification Report

**Phase Goal:** EV_lower閾値がアンサンブルOOF分布に基づく動的閾値に置き換わり、過剰除外が解消されるとともにEV推定精度が可視化されている状態になる
**Verified:** 2026-05-06T15:25:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| T1 | SubmodelSetにev_lower_threshold_turf/ev_lower_threshold_dirtフィールドが存在し、アンサンブルOOF分布から計算された値が格納される | VERIFIED | `src/domain/models.py:256-257` -- dataclass fields with default=1.0. `training_pipeline.py:875-880,901-902` -- _compute_ev_threshold()で計算しSubmodelSetに格納 |
| T2 | get_win_candidates()が固定1.0ではなくSubmodelSetから取得した動的閾値でEV_lowerフィルタリングを実行する | VERIFIED | `race_predictor.py:433-462` -- surf_keyに基づきgetattrで閾値取得。fillna(1.0)はコメント内のみ。実際は `ev_lower.fillna(threshold) >= threshold` |
| T3 | EV_lowerがNaNの場合、サーフェス別デフォルト閾値(芝0.8/ダート0.7)にフォールバックする | VERIFIED | `race_predictor.py:449-450` -- `ev_lower.fillna(threshold) >= threshold`。閾値計算のfallbackは `training_pipeline.py:876-877` (turf=0.8, dirt=0.7)。test_ev_lower_nan_uses_surface_fallback 通過 |
| T4 | TrainingPipelineがOOF positive-edge winnersから25th percentileを計算しSubmodelSetに格納する | VERIFIED | `training_pipeline.py:284-322` -- `_compute_ev_threshold()` static method。kakuteijyuni==1 AND win_selection_edge>0 でpositive-edge winnersを抽出。`ev_lower_values.quantile(0.25)` で計算。最小サンプル数30件 |
| T5 | compute_ev_diagnostics()がOOF DataFrameからECE/Brier分解/Reliability diagram/時系列ドリフトを計算してJSONを出力する | VERIFIED | `src/models/ev_diagnostics.py:160-282` -- compute_ev_diagnostics()が相関/RMSE + ECE + Brier分解 + Reliability diagram + 時系列ドリフトを計算。JSON出力はoutput_path指定時にjson.dump()で書き出し |
| T6 | run_backtest.py --ensemble実行時にEV診断が自動で実行される | VERIFIED | `training_pipeline.py:846-858` -- `if use_ensemble:` ガード内で `compute_ev_diag()` を呼び出し。TimingContextでラップ。wsg_train_dfを入力として使用 |
| T7 | コンソールにEV推定精度のサマリが表示される | VERIFIED | `training_pipeline.py:858` -- `ev_console_summary(ev_result)` を呼び出し。`ev_diagnostics.py:285-348` -- logger.infoでSamples/Correlation/RMSE/EV Bias/ECE/Brier/Temporal driftを出力 |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/domain/models.py` | SubmodelSet.ev_lower_threshold_turf/dirt fields | VERIFIED | Lines 256-257: dataclass fields with default=1.0 |
| `src/pipelines/training_pipeline.py` | _compute_ev_threshold + SubmodelSet格納 + EV診断統合 | VERIFIED | Lines 284-322: static method. Lines 875-880: 閾値計算. Lines 846-858: EV診断統合. Lines 901-902: SubmodelSet格納 |
| `src/backtest/race_predictor.py` | 動的閾値を用いたEV_lowerフィルター | VERIFIED | Lines 433-462: getattrでサーフェス別閾値取得、fillna(threshold) >= thresholdでフィルタ |
| `src/models/ev_diagnostics.py` | compute_ev_diagnostics + console_summary | VERIFIED | 349行の完全実装。_compute_ece, _brier_decomposition, _reliability_diagram_data, _temporal_drift, compute_ev_diagnostics, console_summary |
| `tests/test_ev_diagnostics.py` | EV診断テスト11件 | VERIFIED | 11テスト全通過 (ECE 2 + Brier 2 + Reliability 1 + 基本診断 6) |
| `tests/test_race_predictor.py` | 動的閾値テスト3件追加 | VERIFIED | 14テスト全通過 (既存11 + 新規3: turf/dirt動的閾値 + NaNフォールバック) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| training_pipeline.py | domain/models.py | SubmodelSet() constructor引数に閾値を渡す | WIRED | Lines 901-902: `ev_lower_threshold_turf=ev_threshold_turf, ev_lower_threshold_dirt=ev_threshold_dirt` |
| race_predictor.py | domain/models.py | submodel.ev_lower_threshold_turf/dirtから閾値を取得 | WIRED | Line 446: `getattr(_sm, "ev_lower_threshold_turf", 1.0)`. Line 448: `getattr(_sm, "ev_lower_threshold_dirt", 1.0)` |
| training_pipeline.py | ev_diagnostics.py | import compute_ev_diagnostics + console_summary | WIRED | Lines 849-850: `from models.ev_diagnostics import compute_ev_diagnostics as compute_ev_diag` + `console_summary as ev_console_summary` |
| ev_diagnostics.py | data/backtest/ev_diagnostics_{surface}.json | json.dump() output | WIRED | Lines 276-280: output_path指定時にJSON出力。パイプライン統合(Line 852)でoutput_path指定あり |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| race_predictor.py | threshold | SubmodelSet.ev_lower_threshold_turf/dirt | Yes -- _compute_ev_thresholdがOOF DataFrameから計算 | FLOWING |
| ev_diagnostics.py | result dict | df_oof (wsg_train_df) | Yes -- ev_win_corrected, confirmed_odds, kakuteijyuni, race_dateを使用 | FLOWING |
| _compute_ev_threshold | ev_lower_values | wsg_train_dfのEV_lower_win_corrected列 | Yes -- RobustConfidenceEstimator.predict_lower_bound()で計算済み | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ev_diagnostics import | `python -c "from models.ev_diagnostics import compute_ev_diagnostics, console_summary; print('OK')"` | OK: import resolved | PASS |
| SubmodelSet fields | `python -c "from domain.models import SubmodelSet; assert 'ev_lower_threshold_turf' in SubmodelSet.__dataclass_fields__"` | OK: fields exist | PASS |
| _compute_ev_threshold exists | `python -c "from pipelines.training_pipeline import TrainingPipelineV5; assert hasattr(TrainingPipelineV5, '_compute_ev_threshold')"` | OK: method exists | PASS |
| EV diagnostics tests (11) | `python -m pytest tests/test_ev_diagnostics.py -v` | 11 passed in 1.35s | PASS |
| Race predictor tests (14) | `python -m pytest tests/test_race_predictor.py::TestGetWinCandidates -v` | 14 passed in 1.27s | PASS |
| Full test suite | `python -m pytest tests/ --tb=line -q` | 1283 passed, 1 skipped, 0 failed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVF-01 | 15-01 | EV_lower閾値を固定1.0からアンサンブルOOF分布の分位点に基づく動的閾値に変更する | SATISFIED | SubmodelSet新フィールド + _compute_ev_threshold + get_win_candidates()動的閾値フィルター |
| EVF-02 | 15-02 | OOF EV推定値と実際の払戻額を比較し、EV推定精度を評価する診断機能を追加する | SATISFIED | ev_diagnostics.py (ECE/Brier分解/Reliability diagram/時系列ドリフト) + パイプライン統合 |

Orphaned requirements: None

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected in modified files |

No TODO/FIXME/PLACEHOLDER found. No stub implementations. No hardcoded empty data flows to rendering. `return []` in `_temporal_drift` is a legitimate early return when DATE_COLUMN is missing.

### Human Verification Required

### 1. EV Filter Exclusion Count Reduction

**Test:** `run_backtest.py --ensemble` を実行し、コンソールログからEV filter除外件数を確認する
**Expected:** 従来の3,594件から大幅に減少する。ログに `EV threshold for turf: X.XXXX` と動的閾値の値が表示される。各レースの除外ログに `EV_lower < X.XXXX (threshold)` と実際の閾値が表示される
**Why human:** 除外件数の実際の減少はエンドツーエンドのパイプライン実行(PostgreSQL + Parquetデータ)が必要であり、単体テストでは確認不可

### Gaps Summary

コードベースの変更は全て正しく実装・テストされており、7/7の真理は全てVERIFIED。1283テスト全通過でリグレッションなし。

唯一、ROADMAP Success Criteria 1の「除外件数が3,594件から大幅に減少する」の実際の確認には、バックテストのエンドツーエンド実行が必要。コードの変更内容(from固定1.0 to動的閾値)は減少を達成する設計であるが、実際のデータでの検証は人間による実行確認が必要。

CONTEXT.md D-06(RobustConfidenceEstimator再キャリブレーション)はPLANで明示的に実装されなかったが、RESEARCH.md A1で「Ensemble OOF residuals are already in df_oof when calibrate() is called」と検証済み。コード確認でも `df_oof.copy()` をcalibrate()に渡しているため、use_ensemble=True時はアンサンブルOOF値が既に含まれている。この判断は妥当。

---

_Verified: 2026-05-06T15:25:00Z_
_Verifier: Claude (gsd-verifier)_
