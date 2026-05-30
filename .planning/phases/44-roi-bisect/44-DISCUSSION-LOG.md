# Phase 44: ROI Bisect - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-30
**Phase:** 44-ROI Bisect
**Areas discussed:** コンポーネント切り分け手法, ECE/APR/bet_count切り分け順序, 特徴量寄与度分析の範囲, OddsBandFilter/training_bet_historyの扱い

---

## Phase 44 目的の再定義

ユーザーにより Phase 44 の主目的が元の ROADMAP 記載（v1.7→v2.0の歴史的ビセクション）から大きく再定義された。

**元の定義**: v1.7(Phase 34)→v2.0(Phase 38)間の artifact-level bisect で ROI 劣化フェーズを特定
**再定義後**: DeploymentGate FAIL の直接原因（bet_count低下/2025 ECE悪化/shadow APR上振れ/training_bet_history過大ROI）を MAWC/Ranker/OddsBandFilter/Selection/Calibration のどのコンポーネントに帰属させるかの切り分け

**理由**: v1.7→v2.0 タグ比較は補助的な歴史分析とし、フル再学習/フル BT 再実行は主手段にしない。

---

## コンポーネント切り分け手法

| Option | Description | Selected |
|--------|-------------|----------|
| Post-hoc分析中心 | Phase 41/43成果物の深掘り分析。p_win→p_win_market_aware変化をMAWC効果として帰属。高速だがOddsBandFilter分離困難。 | |
| Targeted ablation併用 | ShadowComparisonFrameworkにvariant追加で再実行。確定的だがBT再実行時間が必要。 | |
| 段階的ハイブリッド | まずpost-hoc → 分離不能な残りのみtargeted ablation。段階的アプローチ。 | ✓ |

**User's choice:** 段階的ハイブリッド
**Notes:** Post-hoc分析を主軸とし、ablationは最小限に限定。Variant候補: baseline / MAWC only / Ranker only / MAWC+Ranker / OddsBandFilter off/on。モデル再学習は行わず、既存モデルを使ったBT再実行のみ。フルN-way比較は最初から回さず、post-hocで仮説を絞ってから必要最小限のablationを実行。

---

## ECE悪化とbet_countの切り分け順序

| Option | Description | Selected |
|--------|-------------|----------|
| ECE→APR→bet_count | MAWC確率補正が上流。確率品質を先に確定することで下流分析がクリーンになる。 | ✓ |
| bet_count→ECE→APR | bet_count低下の影響が大きく早期確定したい。確率品質未解決で解釈が複雑に。 | |
| 並列分析（4原因同時） | 各原因を独立に分析。最も包括的だが工数大。 | |

**User's choice:** ECE→APR→bet_count の逐次分析
**Notes:** MAWC確率補正が上流にあり、p_win_final/EV/Ranker score/最終候補選定/OddsBandFilter通過率に連鎖するため、確率品質の歪みを先に切り分けないと因果判断が困難。具体順: (1) ECE: p_win before/after MAWC差分でMAWC直接効果 vs 選定後母集団効果を分離、(2) APR: 全馬 vs 選定馬でMAWC確率水準 vs Ranker選定偏りを判定、(3) bet_count: 確率/EV変化を踏まえドロップ箇所を特定、(4) OBF: 通過率比較 + 必要時のみablation。並列分析は避ける。

---

## 特徴量寄与度分析の範囲

| Option | Description | Selected |
|--------|-------------|----------|
| MAWC+Ranker係数分析中心 | LogisticRegression/Ridgeの係数を分析。上流木モデルは同じなので再分析不要。 | ✓ |
| 係数分析 + 上流モデルgain | 係数分析に加え、木モデルのgainも確認。MAWC入力特徴量支配の追跡。 | |
| 全モデルSHAP/gain比較 | 12モデル全SHAP値比較。最も包括的だが実行時間長。 | |

**User's choice:** MAWC + Ranker 係数分析中心
**Notes:** baselineとshadowは同じ上流学習済みモデルを使っており、差分は主にpost-processing層（MAWCとRanker）で発生。上流木モデルの全SHAP/gain比較は主因切り分けに遠く工数を増やすだけ。具体内容: (1) MAWC係数分析でECE/APR悪化寄与segment特定、(2) Ranker係数分析でinvestment_score重み偏り特定、(3) changed/dropped/retained races別の係数寄与分布比較、(4) 上流モデルgain/SHAPは入力特徴量異常疑い時のみ補助確認、(5) 全12モデルSHAP/gain比較はPhase44範囲外。

---

## OddsBandFilter / training_bet_history の扱い

| Option | Description | Selected |
|--------|-------------|----------|
| bet_count分析に統合 | ECE→APR分析後、bet_count分析の中でOBFを確認。独立ステップにしない。 | ✓ |
| OBF再学習 ablation 含め | OddsBandFilter再学習をablation variantに含める。フィルタ自体の問題を直接検証。 | |
| post-hocのみ（ablationなし） | 通過率変化を確認するのみ。Phase 45で調整。 | |

**User's choice:** bet_count 分析に統合
**Notes:** 比較項目: baseline/shadowのband別ROI、excluded_bands、band別候補数、filter前後通過率、bet_countへの寄与。training_bet_history ROI過大はin-sample calibrationリスクとして記録（Phase 44では修正しない）。training_bet_history ROI過大→OBF寛容→bet_count低下の因果は直接成立しにくい。Phase 44では実際のexcluded_bandsと通過率を実測で判断。OBF再学習・閾値調整はPhase 45のStructural Fix候補。

---

## 追加条件（ユーザー指定）

- Phase 44 の成功基準は ROI 改善ではなく**原因帰属の明確化**
- Phase 45 に渡すべき出力は「修正すべきコンポーネント1〜2個」と「根拠メトリクス」
- 主入力は Phase 43/43.5 で再生成済みの成果物
- v1.7→v2.0 タグ比較は補助分析（主軸ではない）

---

## Claude's Discretion

- ShadowComparisonFramework への ablation variant 追加方法（D-06 N-way design を流用）
- Post-hoc 分析スクリプトの内部メソッド・データフロー設計
- MAWC/Ranker 係数の可視化方法
- テスト構造・命名（既存規約に従う）
- JSON 出力のスキーマ設計（Phase 45 が消費しやすい構造）
- Ablation 実行の具体的な RacePredictor フラグ注入方法

## Deferred Ideas

- **OddsBandFilter 再学習・閾値調整**: Phase 45 Structural Fix 候補
- **MAWC segment 係数に基づく構造的修正**: Phase 45 候補
- **Ranker investment_score 重み調整**: Phase 45 候補
- **全12モデル SHAP/gain 比較**: Phase 44 範囲外
- **v1.7→v2.0 歴史的ビセクション**: 補助分析のみ
- **レジーム別分析**: v2.3+で検討
