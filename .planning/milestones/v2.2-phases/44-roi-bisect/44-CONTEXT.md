# Phase 44: ROI Bisect - Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

DeploymentGate FAIL の直接原因をコンポーネント単位で切り分ける診断フェーズ。Phase 43/43.5 の成果物（shadow comparison + shadow diagnosis）を主入力とし、MAWC/Ranker/OddsBandFilter/Selection logic/Calibration data のどのコンポーネントが各 FAIL 原因を引き起こしているかを特定する。

**Phase 44 は修正フェーズではない。** 原因帰属の明確化のみを行い、Phase 45 (Structural Fix) に「修正すべきコンポーネント1〜2個」と「根拠メトリクス」を渡す。

**In scope:**
- BISECT-01: DeploymentGate FAIL の4原因をコンポーネントに帰属
  1. bet_count 低下 (shadow ~2580 vs baseline ~3330, -22%)
  2. 2025 ECE 悪化 (shadow 0.0156 vs baseline 0.0093, +69%)
  3. shadow APR 上振れ (shadow 1.15 vs baseline 0.93, ideal=1.0)
  4. training_bet_history 過大 ROI / OddsBandFilter 影響
- BISECT-02: MAWC/Ranker の係数分析で確率品質・選定悪化に寄与する特徴量・パラメータを特定

**Out of scope:**
- モデル再学習・ハイパーパラメータ変更
- 構造的修正（Phase 45）
- OddsBandFilter 再学習・閾値調整（Phase 45 候補）
- v1.7→v2.0 フル再学習/フル BT 再実行（補助分析のみ）
- 全12モデル SHAP/gain 比較
- レジーム別分析
- 新特徴量追加

**成功基準は ROI 改善ではなく原因帰属の明確化。**
Phase 45 に「修正すべきコンポーネント1〜2個」と「根拠メトリクス」を渡せれば成功。

</domain>

<decisions>
## Implementation Decisions

### コンポーネント切り分け手法

- **D-01:** 段階的ハイブリッド手法を採用。
  1. **Phase 1: Post-hoc 分析** — Phase 41/43 の既存成果物（shadow_horse_diff.parquet, shadow_diagnosis_result.json 等）を深掘り。p_win→p_win_market_aware 変化を MAWC 効果、investment_score/rank/selected 変化を Ranker/selection 効果として帰属。
  2. **Phase 2: Targeted ablation** — post-hoc で因果分離不能な項目のみ。既存学習済みモデルを使った BT 再実行（モデル再学習なし）。
  3. **Ablation variants（最小限）**: baseline / MAWC only / Ranker only / MAWC+Ranker / OddsBandFilter off/on。フル N-way 比較は最初から回さず、post-hoc で仮説を絞ってから必要最小限のみ実行。
  4. モデル再学習は一切行わない。既存モデル（data/models-backtest/）を使用。

### 分析順序（逐次）

- **D-02:** ECE → APR → bet_count → OddsBandFilter の逐次分析。並列分析は行わない。
  - **理由**: MAWC の確率補正は上流にあり、p_win_final / EV / Ranker score / 最終候補選定 / OddsBandFilter 通過率に連鎖するため。確率品質の歪みを先に切り分けないと、bet_count 低下が Ranker 由来なのか MAWC で EV/edge が変化した結果なのか判断困難。
  - **具体順序**:
    1. **ECE 悪化**: odds_band/popularity_band/probability_rank_band で p_win before/after MAWC の差分と actual/predicted を確認。MAWC 直接効果か選定後母集団効果かを分ける。
    2. **APR 上振れ**: 全馬ベースと選定馬ベースを分け、MAWC の確率水準問題か Ranker の選定偏りかを判定。
    3. **bet_count 低下**: 確率/EV 変化を踏まえたうえで、Ranker / selection stack / OddsBandFilter のどこで件数が落ちているかを確認。
    4. **OddsBandFilter**: excluded_bands と通過率を baseline/shadow で比較。post-hoc で分離不能な場合のみ OBF off/on ablation を実行。

### 特徴量寄与度分析の範囲

- **D-03:** MAWC + Ranker の係数分析を中心とする。上流木モデル（ability, win_hit 等）は baseline/shadow で同一のため、全モデル SHAP/gain 比較は主手段にしない。
  1. **MAWC (LogisticRegression) 係数分析**: logit(p_model), logit(p_market), odds_band, popularity_band, probability_rank_band, 交互作用項のうち、odds_band 1-3 / popularity 10-14 / probability rank 上位などの ECE/APR 悪化に寄与している項目を特定。
  2. **Ranker (Ridge) 係数分析**: investment_score が p_win_final, EV, logit_gap, odds, uncertainty のどれに偏っているかを見る。
  3. **セグメント別係数寄与分布**: changed races / dropped races / retained races に分けて MAWC 係数寄与と Ranker 係数寄与の分布を比較。
  4. **上流木モデル gain/SHAP**: MAWC/Ranker 分析で入力特徴量そのものに異常が疑われた場合だけ補助的に確認。
  5. **全12モデル SHAP/gain 比較は Phase 44 範囲外。**

### OddsBandFilter / training_bet_history の扱い

- **D-04:** OddsBandFilter は独立した分析ステップではなく、bet_count 分析に統合する。
  - ECE→APR 分析で MAWC/Ranker 由来の確率・選定変化を切り分けた後、bet_count 分析の一部として OddsBandFilter を見る。
  - 比較項目: baseline/shadow それぞれで training_bet_history の band 別 ROI、excluded_bands、band 別候補数、filter 前後の通過率、最終 bet_count への寄与。
  - training_bet_history ROI 過大は in-sample calibration リスクとして記録。Phase 44 では修正しない。
  - training_bet_history ROI 過大 → OBF 寛容 → bet_count 低下、という因果は直接は成立しにくい。Phase 44 では実際の excluded_bands と通過率を実測で判断する。
  - OBF off/on ablation は post-hoc で分離不能な場合のみ実行。
  - OddsBandFilter 再学習・閾値調整は Phase 45 の Structural Fix 候補に回す。

### v1.7→v2.0 タグ比較の位置づけ

- **D-05:** v1.7→v2.0 git タグ比較は補助的な歴史分析として扱い、必要な場合のみ既存成果物や軽量 diff で確認する。各タグでフル再学習/フル BT を繰り返すことは Phase 44 の主手段にしない。

### Claude's Discretion

- ShadowComparisonFramework への ablation variant 追加方法（D-06 N-way design を流用）
- Post-hoc 分析スクリプトの内部メソッド・データフロー設計
- MAWC/Ranker 係数の可視化方法
- テスト構造・命名（既存規約に従う）
- JSON 出力のスキーマ設計（Phase 45 が消費しやすい構造）
- Ablation 実行の具体的な RacePredictor フラグ注入方法

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 43/43.5 成果物（主入力）
- `data/backtest/shadow/shadow_comparison_result.json` — Phase 41 baseline vs shadow メトリクス。per-fold/year, surface, odds_band, prob_rank_band 別。
- `data/backtest/shadow/shadow_horse_diff.parquet` — Phase 41 馬単位 diff。p_win, p_win_market_aware, investment_score, rank, selected 列。ECE/APR 分析の主要入力。
- `data/backtest/shadow/shadow_race_diff.parquet` — Phase 41 レース単位 diff。selected_changed, ROI, hit_rate 列。bet_count 分析の主要入力。
- `data/backtest/shadow/diagnosis/shadow_diagnosis_result.json` — Phase 43 3ステップ段階的除外診断結果。確率品質/選定差分/セグメント別キャリブレーション乖離。
- `data/backtest/shadow/gates/deployment_gate_result.json` — Phase 42 DeploymentGate 評価結果。FAIL/WARN/PASS 条件一覧。

### コンポーネント定義
- `src/models/market_aware_win_calibrator.py` — Phase 39 MAWC。LogisticRegression 51-dim segment conditioning。coef_ で係数分析可能。
- `src/models/race_level_ranker.py` — Phase 40 Ranker。Ridge relevance/value scoring。coef_ で重み分析可能。
- `src/betting/odds_band_filter.py` — OddsBandFilter。training_bet_history ベース band 別収益性計算。
- `src/betting/strategy.py` — WinStrategy。候補選定ロジック（EV_lower, selection gate）。

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor。MAWC 適用 (lines 269-277), Ranker scoring (lines 279-285), shadow 診断ブロック (lines 860-884)。enable_market_aware_calibrator / enable_race_level_ranker フラグ。
- `src/backtest/shadow_comparison.py` — ShadowComparisonFramework。D-06 N-way variant 対応。Ablation 実行の基盤。
- `src/backtest/shadow_diagnosis.py` — ShadowDiagnosis。Phase 43 診断エンジン。セグメント定数 POPULARITY_BAND_EDGES 等。
- `src/backtest/engine.py` — BacktestEngine。BacktestResult と bet_history の列定義。
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator。GatePolicy 定義。

### 既存分析基盤
- `scripts/analyze_feature_importance.py` — SHAP/gain/permutation importance 分析スクリプト。
- `scripts/run_gpd.py` — Gain per Depth 診断。
- `scripts/run_ic_eval.py` — IC 評価。

### Domain
- `src/domain/models.py` — SubmodelSet (lines 234-273), TrainedModelsV5。market_aware_win_calibrator, win_race_level_ranker フィールド。

### Requirements
- `.planning/REQUIREMENTS.md` — BISECT-01, BISECT-02 (Phase 44 requirements)。
- `.planning/ROADMAP.md` — Phase 44 success criteria (3 items)。
- `.planning/PROJECT.md` — Key Decisions (配備条件=確率品質, MAWC replaces WinBenterGate+WinSegmentCalibrator)。

### Prior Phase Context
- `.planning/phases/43-shadow-diagnosis/43-CONTEXT.md` — Phase 43 診断設計。D-02 段階的除外, D-03 セグメント定義, D-04 出力フォーマット。
- `.planning/phases/42-feature-routing-audit-safety-gates/42-CONTEXT.md` — Phase 42 GatePolicy 定義。D-11 ゲート条件と閾値。
- `.planning/phases/41-shadow-comparison-framework/41-CONTEXT.md` — Phase 41 比較基盤。D-06 N-way design, D-18 baseline definition, D-19 feature flags。

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **ShadowComparisonFramework** (`src/backtest/shadow_comparison.py`): D-06 N-way variant 対応済み。Ablation variant の追加にそのまま使える。enable_market_aware_calibrator / enable_race_level_ranker フラグで各コンポーネントを個別トグル可能。
- **ShadowDiagnosis** (`src/backtest/shadow_diagnosis.py`): Phase 43 診断エンジン。セグメント定数 (POPULARITY_BAND_EDGES, PROB_RANK_BAND_EDGES, ODDS_BAND_EDGES) を再利用。Post-hoc 分析の一部は ShadowDiagnosis の拡張または新クラスとして実装可能。
- **Phase 41 artifacts**: shadow_horse_diff.parquet に p_win, p_win_market_aware, investment_score, rank, selected が格納済み。Post-hoc 分析の主要データソース。
- **Phase 42 gate results**: deployment_gate_result.json に FAIL/WARN/PASS 条件が構造化済み。

### Established Patterns
- **JSON + Markdown + HTML 複数出力**: Phase 41/43 パターン。JSON は自動消費、MD はレビュー/コミット、HTML は人間レビュー。
- **CLI 引数パターン**: run_shadow_comparison.py / run_shadow_diagnosis.py のパターンを流用。
- **Segment 定数**: POPULARITY_BAND_EDGES [1-3, 4-6, 7-9, 10-14, 15+], PROB_RANK_BAND_EDGES [top1, 2-3, 4-6, 7+], ODDS_BAND_EDGES [1-3, 3-5, 5-10, 10-30, 30+]。

### Integration Points
- **入力**: data/backtest/shadow/ 内の Phase 41/43/42 成果物。
- **出力**: data/backtest/shadow/bisect/ ディレクトリ（推奨）。コンポーネント別帰属結果 + 係数分析 + ablation 結果。
- **消費者**: Phase 45 (Structural Fix) が bisect 結果を読み込んで修正対象を決定。

</code_context>

<specifics>
## Specific Ideas

- 分析順序 ECE→APR→bet_count→OBF は、MAWC 確率補正が上流であることを踏まえた因果順序。確率品質を先に確定することで下流分析がクリーンになる。
- MAWC 係数分析で注目すべき segments: odds_band 1-3 (ECE 3倍悪化), popularity_band 10-14 (APR 2.3倍), probability_rank_band top1 (favorite 過大補正の疑い)。
- bet_count 低下の主因仮説: (A) Ranker 候補除外 (changed races 706 vs baseline 2123) が主因, (B) OddsBandFilter 追加除外が寄与。Post-hoc で dropped レースの除外ステージを追跡して分離。
- changed races で shadow ROI (-10.1%) が baseline (-18.3%) より良好な点に注意 — Ranker は「質」を改善しつつ「量」を減らしている。Phase 45 では質を維持したまま量を回復する方策が必要。
- Phase 43 診断で odds_band 1-3 の shadow ECE=0.1444 vs baseline=0.0447。MAWC の Benter logit-blend が logit(p_market) を過度に引き上げている可能性。係数分析で logit(p_market) 重みと segment 交互作用を確認。

</specifics>

<deferred>
## Deferred Ideas

- **OddsBandFilter 再学習・閾値調整**: Phase 45 Structural Fix 候補。Phase 44 では通過率・excluded_bands の比較のみ。
- **MAWC segment 係数に基づく構造的修正**: Phase 45 候補。Phase 44 で特定した悪化寄与 segment を修正。
- **Ranker investment_score 重み調整**: Phase 45 候補。Phase 44 で特定した偏りを修正。
- **全12モデル SHAP/gain 比較**: Phase 44 範囲外。上流木モデルの特徴量分析が必要な場合は個別に補助実行。
- **v1.7→v2.0 歴史的ビセクション**: 補助分析のみ。フル再学習/フル BT は主手段にしない。
- **レジーム別分析**: REQUIREMENTS.md で明示的に除外。v2.3+で検討。

</deferred>

---

*Phase: 44-ROI Bisect*
*Context gathered: 2026-05-30*
