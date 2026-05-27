# Requirements: keiba-ai v2.1 MarketAware Calibration + Race-Level Ranker

**Defined:** 2026-05-27
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v2.1 Requirements

### Calibration

- [ ] **CAL-01**: MarketAwareWinCalibratorがBenter型 logit(p_model) + logit(p_market) ブレンドでセグメント条件付き確率を生成する
- [ ] **CAL-02**: セグメント条件付けが人気順位/オッズバンド/確率順位をInvestmentFeatureFrame出力から取得し、キャリブレータ特徴量として統合される
- [ ] **CAL-03**: セグメント効果は独立したper-segment係数ではなく、正則化された特徴量/交互作用としてグローバルキャリブレータに統合される(スパースsegment過学習防止)
- [ ] **CAL-04**: MarketAwareWinCalibratorがWinBenterGate + WinSegmentCalibratorを置換し、二重補正を防止する
- [ ] **CAL-05**: キャリブレータ出力が正規化後に確率品質(Brier/logloss/ECE)を維持し、sum-to-1.0制約を満たす

### Race-Level Ranker

- [ ] **RNK-01**: 学習型Win relevance rankerがis_win / finishing-position関連度でレース内馬を順位付けする
- [ ] **RNK-02**: 学習型Value/mispricing rankerがキャリブレーション済みEV、model-vs-market gap、CLV診断(OOF安全)で価値誤評価を検出する
- [ ] **RNK-03**: Win ranker + Value ranker出力がinvestment_scoreに結合される
- [ ] **RNK-04**: Rankerがshadow modeで動作し、feature flagでbaseline WinSelectionGateを保持する
- [ ] **RNK-05**: One-bet-per-race baseline bet countが維持される(明示的承認なしに削減しない)

### Shadow Comparison

- [ ] **SHD-01**: Shadow比較フレームワークが2024/2025テスト期間でbaseline TrainedModelsV5 vs shadow TrainedModelsV5を実行する(固定fold検証を含む)
- [ ] **SHD-02**: 比較指標: Brier, logloss, ECE, 選択一致率, CLV, ROI, HR, DD, bet countを追跡する
- [ ] **SHD-03**: キャリブレータとランカーの変更による選択馬の差分(選択一致率)が測定・説明可能である

### Safety

- [ ] **SAF-01**: 特徴量ルーティング監査でキャリブレータ特徴量がMarketModel/RaceQualityScreenerに登録されないことを確認する
- [ ] **SAF-02**: OOFHealthValidator異常なしで全コンポーネントが動作する
- [ ] **SAF-03**: 新キャリブレータ/ランカーは確率品質ゲート + ベット数維持 + アーティファクト再現性 + diagnostics全通過までベースラインを置き換えない

## Future Requirements

### Deferred to v2.2+

- **DEP-01**: デプロイゲート自動判定(確率品質 + ベット数維持 + 再現性 + diagnostics)。選択一致率は固定閾値ではなく診断指標として扱う
- **DEP-02**: Optuna 19次元パラメータ最適化(Ranker重み込み)

## Out of Scope

| Feature | Reason |
|---------|--------|
| 2024/2025固有係数ハードコード | 汎化性能を損なう |
| ベット数削減をROI改善の主手段にする | 統計的有意性が低下する |
| 純粋なROI-label ranker | 過学習リスクが高い |
| 複勝/ワイドモデルの変更 | 単勝に集中 |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| レジーム依存キャリブレーション/伝播 | v2.1方針で却下。レジーム非依存アプローチを採用 |
| Conservative regime固有ROI調整 | レジーム特化は過学習リスクが高い |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CAL-01 | TBD | Pending |
| CAL-02 | TBD | Pending |
| CAL-03 | TBD | Pending |
| CAL-04 | TBD | Pending |
| CAL-05 | TBD | Pending |
| RNK-01 | TBD | Pending |
| RNK-02 | TBD | Pending |
| RNK-03 | TBD | Pending |
| RNK-04 | TBD | Pending |
| RNK-05 | TBD | Pending |
| SHD-01 | TBD | Pending |
| SHD-02 | TBD | Pending |
| SHD-03 | TBD | Pending |
| SAF-01 | TBD | Pending |
| SAF-02 | TBD | Pending |
| SAF-03 | TBD | Pending |

**Coverage:**
- v2.1 requirements: 16 total
- Mapped to phases: 0
- Unmapped: 16 ⚠️

---
*Requirements defined: 2026-05-27*
*Last updated: 2026-05-27 after initial definition*
