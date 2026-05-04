# Requirements: keiba-ai v1.3 Betting Strategy Optimization

**Defined:** 2026-05-04
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1.3 Requirements

Requirements for Betting Strategy Optimization milestone. Each maps to roadmap phases.

### Bet Selection (BSEL)

- [x] **BSEL-01**: バックテスト実行時、EV_lower_win_corrected >= 1.0 を満たさないベットが自動除外される
- [ ] **BSEL-02**: RegimeDetectorがCOLLAPSEDと判定したレースでベットが完全スキップされる
- [x] **BSEL-03**: オッズバンド別ROI分析に基づき、赤字バンドのベットがOddsBandFilterで除外される

### Stake Sizing (SIZE)

- [ ] **SIZE-01**: レジーム状態に応じたKelly分数が設定される (AGGRESSIVE/CONSERVATIVE/COLLAPSED別)
- [ ] **SIZE-02**: 高EV機会にEV比例乗算器 (min(ev/target_ev, max_scale)) で重点配分される

### Risk Control (RISK)

- [ ] **RISK-01**: DrawdownControllerの乗数テーブル・ローリングウィンドウ・リカバリ閾値がWIN向の中率10%に再調整される

### Validation (VAL)

- [ ] **VAL-01**: ParameterFreezeProtocolが戦略パラメータをカバーし、ルックアヘッドバイアスを防止する
- [ ] **VAL-02**: Optuna TPEで全戦略パラメータの同時最適化が実行され、最適設定が発見される

## Future Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Advanced Bet Selection

- **BSEL-04**: バックテスト分析による動的バンド閾値の自動更新
- **BSEL-05**: フィルター相互作用の定量評価ダッシュボード

### Advanced Sizing

- **SIZE-03**: バンクロール成長率最大化に基づくKelly分数自動調整

### Advanced Risk Control

- **RISK-02**: Regime別独立DD制御 (AGGRESSIVE/CONSERVATIVE/COLLAPSED別のDD乗数テーブル)
- **RISK-03**: 動的リカバリ閾値 (ROIノイズに適応する自己調整型リカバリ条件)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| LSTM/Transformerモデリング | 過去5-15走では過学習リスク高 |
| 複勝/ワイドモデルの変更 | 単勝に集中 |
| 実馬券購入機能 | ペーパートレードまで |
| Web UI | CLIベースで十分 |
| 外部Kellyライブラリ導入 | 既存StakeCalculatorで十分、JRA固有制約はカスタム実装が必要 |
| モデル再学習 | 既存3モデルスタッキングをそのまま使用 |
| 新データ源の導入 | 既存EveryDB2データで十分 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BSEL-01 | Phase 11 | Complete (Plan 01) |
| BSEL-02 | Phase 11 | Pending |
| BSEL-03 | Phase 11 | Complete (Plan 01) |
| SIZE-01 | Phase 12 | Pending |
| SIZE-02 | Phase 12 | Pending |
| RISK-01 | Phase 13 | Pending |
| VAL-01 | Phase 13 | Pending |
| VAL-02 | Phase 13 | Pending |

**Coverage:**
- v1.3 requirements: 8 total
- Mapped to phases: 8
- Unmapped: 0

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 after Plan 11-01 completion*
