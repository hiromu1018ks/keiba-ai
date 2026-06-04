# Phase 45: Structural Fix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-31
**Phase:** 45-structural-fix
**Areas discussed:** 修正対象優先順位, MAWC修正手法, 汎化確認戦略

---

## 修正対象の優先順位

| Option | Description | Selected |
|--------|-------------|----------|
| MAWC単一 | MAWCのbeta_market/logit重み調整のみ。Ranker/OBF/selection閾値は一切変更なし。 | ✓ |
| MAWC + selection閾値微調 | MAWC修正 + selection gate EV_lower閾値軽微調整。 | |
| MAWC + Ranker閾値緩和 | MAWC修正 + Rankerのinvestment_score NaNフォールバック挙動調整。 | |

**User's choice:** MAWC単一
**Notes:** Phase 44の原因帰属でRankerはdormant（investment_score non-NaN=0%）、OBFは非因果、selection thresholdはMAWC EV低下の結果を受けているだけ。変更範囲最小化と因果追跡容易性を優先。selection閾値緩和はMAWC確率歪みを隠すだけ。

---

## MAWC修正手法

| Option | Description | Selected |
|--------|-------------|----------|
| MAWC再学習 (C値調整) | LogisticRegressionのC値を変更してOOF再fit。C grid [0.003-0.03]。 | ✓ |
| 係数事後clamping | 学習済みcoef_を直接調整。実装単純だがfitのバランス崩壊リスク。 | |
| RacePredictor側補正層 | MAWC呼び出し前後で補正追加。追加レイヤーの逆効果リスク。 | |

**User's choice:** MAWC再学習ベース（保守的MAWC仕様への構造変更）
**Notes:** 係数事後clamping/サンドイッチ補正は不採用（確率モデル整合性崩壊リスク + 固有チューニング化リスク）。具体的な方針として:
1. OOFデータでMAWC再fit
2. C grid [0.003, 0.005, 0.01, 0.03] で強正則化探索
3. 高リスク交互作用項（logit_model × popularity top/favorite、logit_model × low odds_band系）を削除
4. main effects（logit_model, logit_market, log_odds, popularity_rank_pct, p_win_race_rank_pct, segment one-hot）は保持
5. favorite band guard条件追加
6. LogisticRegression単一構造

### MAWC保存戦略

| Option | Description | Selected |
|--------|-------------|----------|
| 既存モデル直接置換 | 既存モデルファイルを修正版で上書き。 | |
| 新規variant保存 | 別ディレクトリに保存。前後比較可能。 | ✓ |

**User's choice:** 新規variant保存（data/models-backtest-mawc-conservative/{year}/）
**Notes:** Phase 43/44の比較根拠と再現性を保持。MAWC joblibのみ置換。manifestにメタデータ記録。Phase 46で baseline vs mawc_conservative 比較。

### C値選択基準

| Option | Description | Selected |
|--------|-------------|----------|
| 最小C選択 | 品質ゲート通過候補中最小C。汎化性最大化。 | ✓ |
| Brier最小選択 | Brier最小のCを採用。最適化バイアス。 | |
| 固定C単一試行 | C固定（例: 0.01）で1パターンのみ。 | |

**User's choice:** 最小C選択（品質ゲート通過候補中で最小C）
**Notes:** 品質ゲート条件: (1) 全体Brier/logloss/ECE非悪化、(2) 年度別非悪化、(3) odds 1-3 favorite guard（ECE/bet_count/APR）、(4) p過度圧縮チェック、(5) 複数候補→最小C、(6) 全不適格→既存維持。Brier最小・ROI最大では選ばない。

---

## 汎化確認戦略

| Option | Description | Selected |
|--------|-------------|----------|
| OOF品質確認のみ | 軽量OOF確認。Phase 46で本格検証。 | ✓ |
| Shadow Comparison再実行 | baseline vs mawc_conservative直接比較。~9h。 | |
| 軽量単年度BT | 修正版MAWCで2024のみBT。~40min。 | |

**User's choice:** OOF品質確認 + 軽量proxy確認に限定
**Notes:** Phase 45ではBrier/logloss/ECE確認 + favorite band guard + p圧縮チェック + EV>=1.0通過率 + APR非悪化確認。BT/Shadow Comparison/ROI評価はPhase 46。2層検証設計: Phase 45で方向性確認→Phase 46で安全確認。

---

## Claude's Discretion

- MAWC再学習のOOFデータ分割方法
- 削除交互作用項の正確な特定（FEATURE_COLSから）
- C grid評価の実装詳細
- conservative variantのディレクトリ構造・ファイル命名
- テスト構造・命名
- manifestスキーマ設計
- favorite band guard閾値の具体的数値

## Deferred Ideas

- Ranker修正（investment_score重み・閾値調整）— Phase 46後候補
- OddsBandFilter再学習・閾値調整 — Phase 46後候補
- Selection gate閾値調整 — MAWC修正効果で自然改善を期待
- レジーム別分析 — v2.3+
- 新特徴量追加 — v2.3+
