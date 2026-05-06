# Requirements: keiba-ai v1.4 Ensemble Filter Recalibration

**Defined:** 2026-05-05
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1 Requirements

### Gate Recalibration

- [ ] **GATE-01**: WinSelectionGateをアンサンブルOOF予測で再学習し、prob_edges/edge_edges/odds_edgesを再計算する
- [ ] **GATE-02**: 単一モデルとアンサンブルのOOF確率分布をks_2samp/wasserstein_distanceで比較し、ドリフトを定量化する診断機能を追加する
- [ ] **GATE-03**: use_ensembleフラグがModelLoader→RacePredictor→BacktestEngine全体で正しく伝播されていることを検証する

### EV Filter Enhancement

- [ ] **EVF-01**: EV_lower閾値を固定1.0からアンサンブルOOF分布の分位点に基づく動的閾値に変更する
- [ ] **EVF-02**: OOF EV推定値と実際の払戻額を比較し、EV推定精度を評価する診断機能を追加する

### Odds Band Rebuild

- [x] **ODDS-01**: アンサンブルモデルでtraining_bet_historyを再生成し、OddsBandFilter.calibrate()でバンド別ROIを再計算する
- [x] **ODDS-02**: strategy_optimizer.pyのルックアヘッドバイアスを修正し、training_bet_history生成にデフォルトパラメータを使用する

### Optuna Optimization

- [x] **OPT-01**: アンサンブルモデルで既存14次元Optuna最適化を実行する(フィルター再キャリブレーション完了後)
- [x] **OPT-02**: walk-forward fold数を2→4に増やし過学習リスクを軽減する
- [x] **OPT-03**: 複数seedでOptuna最適化を実行し、パラメータ安定性を検証して不安定な次元を検出する

### Validation

- [ ] **VAL-01**: アンサンブルバックテストで年間100+ベットかつROI>100%を達成することを確認する
- [ ] **VAL-02**: ParameterFreezeProtocolで最適化済みパラメータを固定し、SHA256改ざん検知を適用する

## Future Requirements

### Deferred

- **CONF-01**: Ridgeメタラーナー上にIsotonic/Temperature Scaling追加のキャリブレーションレイヤー(OOF-Inference平均シフト>0.02の場合)
- **CONF-02**: レジーム判定の発振リスクに対するヒステリシスカウンター調整
- **CONF-03**: Optuna探索空間の自動縮小(不安定次元の固定化)

## Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルのフィルター変更 | 単勝ROI最大化に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| 新規MLモデルの追加 | 既存3モデルスタッキングをそのまま使用 |
| Web UI | CLIベースで十分 |
| リアルタイムオッズ収集の改善 | 既存機能をそのまま使用 |
| LSTM/Transformer時系列モデリング | 過去5-15走では過学習リスク高 |
| betacal等の新規依存関係追加 | 既存scipy/numpy/sklearnで十分 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GATE-01 | Phase 14 | Pending |
| GATE-02 | Phase 14 | Pending |
| GATE-03 | Phase 14 | Pending |
| EVF-01 | Phase 15 | Pending |
| EVF-02 | Phase 15 | Pending |
| ODDS-01 | Phase 16 | Complete |
| ODDS-02 | Phase 16 | Complete |
| OPT-01 | Phase 17 | Complete |
| OPT-02 | Phase 17 | Complete |
| OPT-03 | Phase 17 | Complete |
| VAL-01 | Phase 18 | Pending |
| VAL-02 | Phase 18 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-05*
*Last updated: 2026-05-05 after initial definition*
