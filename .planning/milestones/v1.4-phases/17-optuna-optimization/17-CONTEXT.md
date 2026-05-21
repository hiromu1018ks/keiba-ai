# Phase 17: Optuna Optimization - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

アンサンブルモデルで再キャリブレーション済みのフィルター群に対してOptuna 14〜16次元最適化が実行され、fold増強(2→4)とmulti-seed安定性検証(3 seeds)を経て過学習耐性のある最適パラメータが導出されている状態になる。

**In scope:**
- OPT-01: アンサンブルモデルで既存14次元Optuna最適化を実行する(フィルター再キャリブレーション完了後)
- OPT-02: walk-forward fold数を2→4に増やし過学習リスクを軽減する
- OPT-03: 複数seedでOptuna最適化を実行し、パラメータ安定性を検証して不安定な次元を検出する
- Phase 15 D-01/D-02: EV_lowerサーフェス別閾値を15-16次元目として追加
- StrategyOptimizerの4fold対応リファクタリング
- モデルロード最適化(trial内1回)とtraining_bet_historyキャッシュ
- Multi-seed安定性検証と不安定次元の自動固定化
- 安定性レポート(JSON)の出力

**Out of scope:**
- バリデーション & パラメータ凍結 (Phase 18)
- 複勝/ワイドモデルの変更
- 新しいモデルや特徴量の追加
- ML モデルハイパーパラメータの再最適化 (OptunaTunerの対象)
- バックテストパイプラインの変更（戦略パラメータ注入のみ）

</domain>

<decisions>
## Implementation Decisions

### WF fold構成 (OPT-02)
- **D-01:** 軽量WFアプローチを採用。学習済みモデル1セットを全foldで共有し、fold毎の再学習は行わない。戦略パラメータのロバスト性を複数テスト期間で評価することが目的。
- **D-02:** 年次4fold構成: テスト期間 2022/2023/2024/2025 の4年(各fold 1年)。`_generate_folds()`をハードコードからコンストラクタ引数ベースの動的生成に変更する。2022-2023は学習期間と重なるが、戦略パラメータ評価としては有効。最終OOS検証はPhase 18で実施。
- **D-03:** fold生成ロジックの実装詳細はClaude's discretion。n_folds, train_years, test_yearsコンストラクタ引数を活用。

### 計算時間とトライアル数 (OPT-01)
- **D-04:** 100トライアル維持。Phase 13 D-11を踏襲。4fold化による計算量増加はモデルロード最適化で相殺。TPEサンプラーで16次元空間を100試行で十分に探索可能。
- **D-05:** ベストプラクティス追求: (1) モデルロードをtrial内1回に最適化 — _objective()の先頭でModelLoader.load_from_dir()を1回呼び出し、全foldで共有。(2) training_bet_historyをtrial内1回キャッシュ — 同じモデル/デフォルトパラメータで生成するため全fold共通。(3) RegimeDetector可変状態をfold間でリセット(CR-01パターン)。モデルのdeep copyは不要。
- **D-06:** MedianPrunerの設定はClaude's discretion。4fold環境での最適なpruningタイミングとパラメータは研究者・プランナーが決定。

### 探索次元 (Phase 15決定の反映)
- **D-07:** Phase 15 D-01/D-02の決定を反映: EV_lower閾値を15-16次元目として_suggest_params()に追加。サーフェス別(芝/ダート)で2次元。探索空間は合計16次元(14既存 + EV_lower芝 + EV_lowerダート)。Phase 15 CONTEXTを参照。

### Multi-seed安定性検証 (OPT-03)
- **D-08:** 3 seeds構成で実行: (1) 主実行: seed=42, 100 trials, (2) 安定性確認: seed=43, 50 trials, (3) 安定性確認: seed=44, 50 trials。追加seedは50 trialsに削減して計算コストを抑制。
- **D-09:** 安定性判定基準はClaude's discretion。CV(変動係数)ベースの閾値、stddev比、rank相関など研究者・プランナーが適切な手法を選択。
- **D-10:** 不安定次元の対応: ベストプラクティス(固定化して再実行)を採用。不安定次元をデフォルト値に固定 → 探索空間縮小 → 再最適化。REQUIREMENTS.md CONF-03「自動縮小」に相当。安定性レポート(JSON)で不安定次元を明示的に報告。

### ベット数制約
- **D-11:** 現状維持: min_bets_per_fold=1000、ハードカットオフ(ROI=-1.0ペナルティ)。Phase 13 D-09を踏襲。Phase 15-16フィルター変更後も1000件は統計的有意性の妥当な目安。

### Claude's Discretion
- `_generate_folds()`の具体的な実装(コンストラクタ引数から動的に生成)
- MedianPrunerの設定(n_startup_trials, n_warmup_steps, interval_check_steps等)
- 安定性判定の具体的な手法と閾値
- 安定性レポートのJSONスキーマ
- モデルロード最適化の具体的な実装(コピー vs リセット)
- _suggest_params()へのEV_lower 2次元追加の実装詳細
- テスト戦略(モックベース、既存パターン踏襲)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### StrategyOptimizer（主変更対象 — OPT-01/02/03）
- `src/tuning/strategy_optimizer.py` — StrategyOptimizer実装全体。_suggest_params(14次元), _generate_folds(2fold固定), _objective, optimize
- `src/tuning/strategy_optimizer.py:51-81` — _suggest_params()。EV_lower 2次元の追加ポイント(D-07)
- `src/tuning/strategy_optimizer.py:220-229` — _generate_folds()。2fold→4fold動的生成に変更(D-02)
- `src/tuning/strategy_optimizer.py:188-218` — _objective()。モデルロード最適化ポイント(D-05)
- `src/tuning/strategy_optimizer.py:99-186` — _run_single_backtest()。モデルロード最適化ポイント(D-05)

### BacktestEngine（strategy_params注入ポイント）
- `src/backtest/engine.py:414-429` — run()メソッド。strategy_params, training_bet_history受け取り
- `src/backtest/engine.py:363-385` — BacktestEngineコンストラクタ。strategy_paramsからStakeCalculator/DDController/OddsBandFilter生成

### RegimeDetector（パラメータソース）
- `src/models/regime_detector.py:185-243` — get_strategy_params()。デフォルトパラメータのソース。Phase 13 D-15外部化済み

### EV_lower動的閾値（Phase 15 D-01/D-02 — 15-16次元目の追加元）
- `src/backtest/race_predictor.py:420-480` — get_win_candidates()。EV_lowerフィルター適用箇所。動的閾値のOptuna探索範囲定義の参照
- `src/models/robust_confidence_estimator.py:96-234` — predict_interval()。EV_lower計算。閾値の探索範囲設定に参照

### ParameterFreezeProtocol（Phase 18用 — manifest自動生成）
- `src/backtest/parameter_freeze_protocol.py` — save_strategy_manifest()。Phase 17最適化結果の保存先

### デフォルトパラメータ構築（ルックアヘッド防止）
- `src/betting/default_strategy.py` — build_default_strategy_config(), build_strategy_config_from_params()。Phase 16で追加

### ドメイン型
- `src/domain/types.py:29-34` — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)

### 設定ファイル
- `config/settings.yaml` lines 39-47 — betting_strategy section。デフォルト値定義
- `config/backtest_config.yaml` — WF設定 (train_years=4, test_years=1, step_years=1)

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/16-odds-band-rebuild/16-CONTEXT.md` — Phase 16決定（ルックアヘッド修正、training_bet_history自動生成、_build_default_config）
- `.planning/phases/15-ev-filter-enhancement/15-CONTEXT.md` — Phase 15決定（EV_lower動的閾値、サーフェス別、Optuna 16次元、Conformal再キャリブレーション）
- `.planning/phases/14-gate-recalibration/14-CONTEXT.md` — Phase 14決定（ドリフト診断パターン、use_ensemble伝播）
- `.planning/phases/13-risk-calibration-parameter-optimization/13-CONTEXT.md` — Phase 13決定（StrategyOptimizer設計、14次元探索空間、ROI+ベット数制約、WF評価）

### 要件定義
- `.planning/REQUIREMENTS.md` — OPT-01, OPT-02, OPT-03の要件定義。CONF-03(自動縮小)参照
- `.planning/ROADMAP.md` — Phase 17 Success Criteria

### 既存テストパターン
- `tests/test_strategy_optimizer.py` — StrategyOptimizer既存テスト。mockベース。Phase 17テストもこのパターンに従う

### 研究
- `.planning/research/SUMMARY.md` — Phase 13/14/15研究サマリ。Optuna設計の背景

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **StrategyOptimizer** (`src/tuning/strategy_optimizer.py`): Phase 13で構築済みのOptuna最適化インフラ。_suggest_params(14次元), _objective(ROI+bet制約), optimize(TPE+MedianPruner)。4fold対応とEV_lower追加の変更ベース
- **BacktestEngine** (`src/backtest/engine.py`): strategy_params注入インターフェース完備。training_bet_history対応済み。戦略パラメータ最適化の評価バックエンド
- **build_default_strategy_config()** (`src/betting/default_strategy.py`): Phase 16 D-03で追加。ルックアヘッド防止のデフォルトパラメータ構築。training_bet_history生成に使用
- **build_strategy_config_from_params()** (`src/betting/default_strategy.py`): Optuna params → BacktestEngine injection dict変換。EV_lower追加に伴う拡張が必要

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 17テストもこのパターンに従う
- **コンストラクタ注入パターン**: パラメータはコンストラクタ引数で注入。Phase 12-13で確立
- **RegimeDetector状態リセット**: CR-01パターン。mutable stateをfold/trial間でリセット
- **JSON+コンソール出力**: Phase 14-15の診断パターン。安定性レポートもこの形式
- **パイプライン統合パターン**: use_ensemble=True時の自動診断・キャリブレーション。Phase 14-16で確立

### Integration Points
- **strategy_optimizer.py:51-81** — _suggest_params()にEV_lower 2次元を追加(D-07)
- **strategy_optimizer.py:220-229** — _generate_folds()を4fold動的生成に変更(D-02)
- **strategy_optimizer.py:188-218** — _objective()にモデルロード最適化を追加(D-05)
- **default_strategy.py** — build_strategy_config_from_params()にEV_lower閾値パラメータのマッピングを追加
- **新規機能**: multi-seed安定性検証の実行フローと安定性レポート生成

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- 軽量WFアプローチは戦略パラメータのロバスト性評価に特化。最終OOS検証はPhase 18
- モデルロード最適化は大きな実行時間短縮効果(trial内1回)
- 不安定次元の自動固定化はCONF-03の先取り実装
- Phase 15で決定したEV_lower 2次元追加をPhase 17で実装（Phase 15では閾値計算のみ実装し、Optuna次元追加はPhase 17に委ねられていた）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 17-Optuna Optimization*
*Context gathered: 2026-05-06*
