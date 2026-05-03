# Phase 6: Odds Deviation EV - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 06-Odds Deviation EV
**Areas discussed:** 乖離信号の拡張, EV区間とベット選択, パイプライン検証戦略

---

## 乖離信号の拡張

### Q1: odds_to_ability_ratioは既存。追加の乖離信号を検討するか？

| Option | Description | Selected |
|--------|-------------|----------|
| 既存のみ検証 | feature importance上位ならそれで十分 | |
| 追加信号を検討 | レース内乖離ランク、乖離z-score等を追加 | |
| 既存を再設計 | より洗練された単一指標に置き換え | |

**User's choice:** ベストプラクティスを選択 → 追加信号を検討（deviation_rank + deviation_zscore）
**Notes:** 実装難易度は問わない。品質優先の方針を確認。

### Q2: 追加の乖離信号をどこで計算するか？

| Option | Description | Selected |
|--------|-------------|----------|
| Stage2モデル内で計算 | WinTwoStageModel._prepare_features()に追記 | |
| FeatureEngineで独立計算 | 新関数をFeatureEngine.build_all()に追加 | |

**User's choice:** ベストプラクティス選択 → standalone関数 `compute_odds_deviation_features(df)` をAbilityModel後に呼び出し
**Notes:** p_ability_winに依存するためFeatureEngine.build_all()では計算不可。MarketModel.predict_and_calc_error()パターンに倣う。

### Q3: 追加する乖離信号の数は？

| Option | Description | Selected |
|--------|-------------|----------|
| 2信号追加 | deviation_rank + deviation_zscore | |
| フラグも追加 | 絶対乖離フラグ（ratio>1.5）も含める | |

**User's choice:** ベストプラクティス選択 → 2信号追加。フラグは不要
**Notes:** LightGBMは自動閾値学習するためフラグは冗長。

### Q4: deviation_rankとdeviation_zscoreの計算基準は？

| Option | Description | Selected |
|--------|-------------|----------|
| レース内相対評価 | race_idグループでrank/z-score計算 | |
| 絶対評価 | 全レース共通基準で評価 | |

**User's choice:** ベストプラクティス選択 → レース内相対評価
**Notes:** odds_to_ability_ratioが絶対値を担う。rank/z-scoreは相対値で補完。market_error_rank_in_raceと同じパターン。

---

## EV区間とベット選択

### Q5: Conformal予測区間をEV区間に変換する実装アプローチは？

| Option | Description | Selected |
|--------|-------------|----------|
| 既存クラス拡張 | RobustConfidenceEstimatorに機能追加 | |
| 新規クラス作成 | EVIntervalEstimatorを独立作成 | |

**User's choice:** ベストプラクティス選択 → 既存RobustConfidenceEstimatorを拡張
**Notes:** nonconformity score計算、CP quantile、race-condition-dependent quantileを再利用。predict_lower_bound()をpredict_interval()に拡張。

### Q6: EV区間をベット選択にどう活用するか？

| Option | Description | Selected |
|--------|-------------|----------|
| EV下限ベースの信頼性スコア | EV_lower > 1.0で高信頼判定 | |
| 区間幅ベースの不確実性スコア | 狭いほど信頼性高い | |
| 両方組み合わせ | 下限（profitability）＋幅（confidence）の合成 | |

**User's choice:** ベストプラクティス追求 → 両方組み合わせ
**Notes:** conformal_confidence_scoreとしてWinSelectionGateのscore()に統合。プロのベッティング運用に倣う。

### Q7: Conformal EV区間の信頼水準は？

| Option | Description | Selected |
|--------|-------------|----------|
| 90%のみ | 既存デフォルトと同一 | |
| 90% + 80%の2段階 | 高信頼/最低基準の段階的評価 | |

**User's choice:** ベストプラクティス追求 → 90% + 80%の2段階
**Notes:** conformal predictionの複数quantile計算は追加コストほぼゼロ。金融VaRと同様の多段階リスク評価。

---

## パイプライン検証戦略

### Q8: ODDS-02のe2e検証の深さは？

| Option | Description | Selected |
|--------|-------------|----------|
| 単体テスト＋スモーク | 各コンポーネント単体＋パイプラインスモーク | |
| フルパイプライン統合テスト | RacePredictor.predict()全出力検証 | |
| 三層すべて | 単体＋統合＋数値的一貫性チェック | |

**User's choice:** ベストプラクティス追求 → 三層すべて
**Notes:** 数値的一貫性チェックが最も価値が高い（確率正規化、EV区間順序性、NaN率）。

### Q9: テストコードの配置先は？

| Option | Description | Selected |
|--------|-------------|----------|
| 既存テストファイルに追加 | test_models.py, test_race_predictor.py等に追加 | |
| 新規テストファイル作成 | test_odds_deviation.py等を作成 | |

**User's choice:** ベストプラクティス追求 → 新規`tests/test_odds_deviation.py`で新機能集約、既存ファイルに拡張部分追加
**Notes:** ソースモジュールと1:1対応の原則。新凝集ユニットは新ファイル、既存クラス拡張は既存ファイル。

---

## Claude's Discretion

- `compute_odds_deviation_features()`の具体的な配置先ファイル（既存モジュール or 新規ユーティリティ）
- conformal_confidence_scoreの合成式の詳細（EV下限と区間幅の重み付け）
- WinSelectionGateへのconformal_confidence_score統合方法
- deviation_rank/z-scoreのNaN処理の詳細

## Deferred Ideas

None — discussion stayed within phase scope
