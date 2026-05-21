# Phase 13: Risk Calibration & Parameter Optimization - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

WIN向中率10%に最適化されたDD制御が動作し、ルックアヘッドバイアスを防いだ上で全戦略パラメータが最適化される。

**In scope (from ROADMAP.md):**
- RISK-01: DrawdownControllerの乗数テーブル・ローリングウィンドウ・リカバリ閾値がWIN向の中率10%に再調整される
- VAL-01: ParameterFreezeProtocolが戦略パラメータをカバーし、ルックアヘッドバイアスを防止する
- VAL-02: Optuna TPEで全戦略パラメータの同時最適化が実行され、最適設定が発見される
- 成功基準1: DD乗数がNORMAL/REDUCED/STOPを適切に遷移する (ROLLING_WINDOW 400+)
- 成功基準2: ParameterFreezeProtocolが戦略パラメータ変更を検出・警告する
- 成功基準3: 最適設定のバックテストROIがベースライン(89.0%)を上回る

**Out of scope:**
- 複勝/ワイドモデルの変更
- BettingOrchestrator/WinStrategy/PlaceStrategy (ライブパス) の変更
- 新データ源の導入
- モデル再学習
- ML モデルハイパーパラメータの再最適化 (既存OptunaTunerの対象)
- 複雑メタラーナー・LSTM/Transformer

**Plans:** TBD

</domain>

<decisions>
## Implementation Decisions

### DD制御WIN特化 (RISK-01)
- **D-01:** DrawdownControllerのコンストラクタにrolling_window, dd_thresholds, multipliers等を引数追加。Phase 12のStakeCalculatorコンストラクタ注入パターンと統一。ハードコードクラス定数は全てコンストラクタ引数に移行
- **D-02:** ROI依存を完全に除去し、DD%のみの3段階制御（NORMAL/REDUCED/STOP）に再設計。WIN的中率10%環境ではROIがノイジーすぎて信号として不適切。DD%は銀行ロール健全性の直接的な指標として信頼性が高い
- **D-03:** 状態遷移にヒステリシスバンドを追加。低的中率環境での発振（NORMAL↔REDUCEDの高速切替）を防止。各状態に最低滞在レース数を設定
- **D-04:** リカバリは段階的（STOP→REDUCED→NORMAL）。即時復帰せず、各状態間の遷移に最低滞在条件を課す
- **D-05:** WIN用とPLACE用で別々のDDControllerインスタンス。WIN的中率10%とPLACE的中率30-40%で最適パラメータが全く異なるため。BacktestEngineで2つのインスタンスを管理
- **D-06:** 具体的なDD閾値（NORMAL→REDUCED境界、REDUCED→STOP境界）、乗数、ヒステリシスバンド幅、最低滞在レース数は全てOptunaで探索

### Optuna最適化設計 (VAL-02)
- **D-07:** 全戦略パラメータを一括でOptuna最適化。TPEサンプラーがパラメータ間相互作用（Kelly分数×DD閾値等）を捉える。段階別最適化ではクロスカテゴリ効果を見逃す
- **D-08:** 探索空間は約16次元:
  - レジーム別: fractional_kelly (×3), ev_threshold (×3), edge_threshold (×3) = 9
  - DD制御: dd_threshold_1, dd_threshold_2, multiplier_REDUCED, multiplier_STOP, rolling_window = 5
  - EVスケーリング: target_ev, max_scale = 2
  - OddsBandFilter: roi_threshold = 1
  - 合計: ~17次元 (COLLAPSEDのfractional_kelly=0は固定なので実質~16)
- **D-09:** 目的関数はROI主 + ベット数制約。ROI単一最適化は過度なフィルタリング（ベット数激減→少数の的中で高ROI）を引き起こす危険がある。年間1000件以上のベット数制約で統計的有意性を担保
- **D-10:** Walk-forward枠組みで評価。既存のWalkForwardCV（walk_forward_cv.py）を拡張して戦略パラメータを各foldに注入。ルックアヘッドバイアスを構造的に防止する最も厳密なアプローチ
- **D-11:** 100トライアル + MedianPruner。TPEで16次元空間なら100回で十分な探索が可能。各トライアルはWF 2fold評価
- **D-12:** 新規ファイル `src/tuning/strategy_optimizer.py` にOptuna戦略最適化を実装。既存のOptunaTuner（ML HP用）とは独立

### パラメータ凍結・注入 (VAL-01)
- **D-13:** ParameterFreezeProtocolを戦略パラメータに拡張。JSON manifest形式で戦略パラメータを保存 + SHA256ハッシュで改ざん検知。既存のmodel pickle凍結とは独立したファイル。人間可読でdiff容易
- **D-14:** Optuna最適化完了後にJSON manifestを自動生成。テスト期間バックテスト実行前にmanifestを読み込みハッシュ照合。不一致時はWARNING
- **D-15:** RegimeDetector.get_strategy_params()の主要パラメータ（fractional_kelly, ev_threshold, edge_threshold）のみをコンストラクタ注入可能に外部化。runner-up rescue rules, rerank params等のドメイン駆動パラメータは固定値として維持
- **D-16:** MetaSwitcherの _default_params() の値をRegimeDetector.get_strategy_params()に揃える（乖離解消）。ただしMetaSwitcher自体のリファクタリングは行わない（ライブパスは変更しない Phase 11 D-13 の決定を踏襲）
- **D-17:** コンストラクタ注入でパラメータを直接渡す（Phase 12 D-12と同じ）。settings.yamlはデフォルト値の定義のみに使用し、Optuna最適化ではプログラマティックに上書き

### スコープ（Phase 11/12決定の踏襲）
- **D-18:** 変更対象はバックテストパスのみ。BettingOrchestrator, WinStrategy, PlaceStrategy（ライブパス）は変更しない
- **D-19:** 新規本番依存関係なし。optuna>=3.5は既存依存（pyproject.toml）

### Claude's Discretion
- DrawdownControllerのコンストラクタ引数の具体的なシグネチャ設計
- 3段階乗数テーブルの具体的なデータ構造（Dict vs NamedTuple vs dataclass）
- ヒステリシスバンドの実装方法（状態マシン vs バンド幅パラメータ）
- WalkForwardCVへの戦略パラメータ注入インターフェース
- strategy_optimizer.py のクラス設計（Objective関数のカプセル化）
- JSON manifestのスキーマ設計
- RegimeDetectorの主要パラメータ外部化の具体的なリファクタリング
- テスト戦略（DD制御単体テスト + Optuna統合テスト + WF評価テスト）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### DrawdownController（主変更対象 — RISK-01）
- `src/betting/drawdown_controller.py` lines 25-53 — ROLLING_WINDOW=150, MULTIPLIER_TABLE(8行), リカバリ閾値。全てコンストラクタ注入に移行
- `src/betting/drawdown_controller.py` lines 55-122 — __init__() + adjust_stake()。ROI依存部分の除去対象
- `src/betting/drawdown_controller.py` lines 124-144 — _compute_rolling_roi()。ROI計算の除去対象

### StakeCalculator（DD制御パイプラインの上流 — 参照のみ）
- `src/betting/stake_calculator.py` lines 30-40 — コンストラクタ注入済み(fractional_kelly, target_ev, max_scale)。Phase 12成果
- `src/betting/stake_calculator.py` lines 143-170 — apply_ev_scaling()。Phase 12成果

### RegimeDetector（パラメータ外部化 — VAL-01）
- `src/models/regime_detector.py` lines 185-243 — get_strategy_params()。25+ハードコードパラメータ。主要パラメータ(fractional_kelly, ev_threshold, edge_threshold)をコンストラクタ注入に外部化
- `src/models/regime_detector.py` lines 133-176 — detect()。3状態分類 + hysteresis

### MetaSwitcher（パラメータ乖離解消）
- `src/betting/meta_switcher.py` lines 41-70 — _default_params()。RegimeDetectorの値に揃える

### BacktestEngine（DD制御WIN/PLACE分離 + Optuna callback）
- `src/backtest/engine.py` lines 363-371 — StakeCalculator/DDController生成。WIN/PLACE別インスタンス管理に変更
- `src/backtest/engine.py` lines 704-715 — regime検出 + fractional_kelly注入。DDController分岐の追加ポイント
- `src/backtest/engine.py` lines 626-1028 — run() レースループ全体

### ParameterFreezeProtocol（戦略パラメータ拡張 — VAL-01）
- `src/backtest/parameter_freeze_protocol.py` lines 1-102 — 既存のmodel pickle凍結。戦略パラメータJSON manifest機能を追加

### Optuna既存インフラ（参照 — ML HPのみ）
- `src/tuning/optuna_tuner.py` lines 18-47 — SEARCH_SPACES (ML HPのみ)。戦略パラメータは新規ファイルに
- `scripts/run_tuning.py` — 既存ML HPチューニングスクリプト。参照のみ

### WalkForwardCV（Optuna評価に活用 — VAL-02）
- `src/models/walk_forward_cv.py` lines 54-189 — WalkForwardCV.generate_folds() + run()。戦略パラメータ注入ポイントを追加
- `scripts/run_wf_validation.py` — 既存WF検証スクリプト。参照パターン

### 新規ファイル
- `src/tuning/strategy_optimizer.py` (新規) — Optuna戦略パラメータ最適化クラス
- `scripts/run_strategy_optimization.py` (新規) — 戦略最適化CLIスクリプト

### ドメイン型
- `src/domain/types.py` lines 21-26 — DrawdownState enum (NORMAL, REDUCED, RECOVERING)。3段階再設計に伴う更新
- `src/domain/types.py` lines 29-34 — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)

### 設定ファイル
- `config/settings.yaml` lines 39-47 — betting_strategy section。デフォルト値定義
- `config/backtest_config.yaml` — WF設定 (train_years=4, test_years=1, step_years=1)

### 要件定義
- `.planning/REQUIREMENTS.md` — RISK-01, VAL-01, VAL-02
- `.planning/ROADMAP.md` — Phase 13 Success Criteria
- `.planning/STATE.md` — 決定済みアーキテクチャ、Look-ahead bias risk注意事項

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/12-stake-sizing-enhancement/12-CONTEXT.md` — Phase 12決定（Kelly→EV乗算→DDパイプライン、コンストラクタ注入、バックテストパスのみ変更）
- `.planning/phases/11-bet-selection-filters/11-CONTEXT.md` — Phase 11決定（フィルター適用順序、COLLAPSEDスキップ、OddsBandFilter、ベット数ガード）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **DrawdownController** (`src/betting/drawdown_controller.py`): adjust_stake() の基本構造は活用。ROLLING_WINDOW=150とMULTIPLIER_TABLE(8行)をコンストラクタ注入にリファクタリング。ROI依存部分(_compute_rolling_roi)を除去
- **StakeCalculator** (`src/betting/stake_calculator.py`): Phase 12でコンストラクタ注入済み。fractional_kelly, target_ev, max_scaleがOptunaで直接制御可能
- **RegimeDetector.get_strategy_params()** (`src/models/regime_detector.py`): 25+パラメータのハードコード。主要3パラメータをコンストラクタ引数に外部化
- **WalkForwardCV** (`src/models/walk_forward_cv.py`): generate_folds() + run() のWF検証インフラ。戦略パラメータ注入ポイントを追加するだけでOptuna評価に活用可能
- **OptunaTuner** (`src/tuning/optuna_tuner.py`): Optuna統合の参照実装。TPESampler使用パターンを踏襲
- **ParameterFreezeProtocol** (`src/backtest/parameter_freeze_protocol.py`): SHA256 hashパターンを戦略パラメータJSON manifestに応用

### Established Patterns
- **コンストラクタ注入パターン**: StakeCalculator(Phase 12)と同じパターンでDrawdownController/RegimeDetectorも注入可能に。Optuna最適化の前提
- **regime params活用パターン**: get_strategy_params() → params["ev_threshold"] 等。主要パラメータ外部化後も同じパターンでアクセス
- **除外統計ログパターン**: engine.py のレースループでカウンタ → INFO ログ出力。DD状態遷移のログにも適用
- **Kelly→EV乗算→DDパイプライン**: Phase 12 D-08/D-09。DDを最終リスクゲートとする順序は維持
- **フィルター適用順序**: Phase 11 D-09。COLLAPSEDスキップ→EV下限→OddsBandFilter。この順序はPhase 13でも変更なし

### Integration Points
- **drawdown_controller.py:25-53** — ハードコード定数 → コンストラクタ引数化 + ROI依存除去
- **drawdown_controller.py:55-122** — __init__() + adjust_stake() の3段階再設計 + ヒステリシス追加
- **engine.py:363-371** — DDController生成をWIN/PLACE別インスタンスに変更
- **engine.py:704-715** — regime検出後にDDController分岐（WIN用/PLACE用）
- **regime_detector.py:185-243** — get_strategy_params() の主要パラメータをコンストラクタ引数から取得
- **meta_switcher.py:41-70** — _default_params() の値をRegimeDetectorに揃える
- **walk_forward_cv.py:54-189** — run() に戦略パラメータ注入ポイントを追加
- **parameter_freeze_protocol.py** — 戦略パラメータJSON manifest + SHA256ハッシュ機能を追加
- **src/tuning/strategy_optimizer.py** (新規) — Optuna戦略パラメータ最適化クラス
- **scripts/run_strategy_optimization.py** (新規) — CLIスクリプト

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- ROI依存の除去はDD制御の信頼性向上に直結。WIN的中率10%環境ではROIがノイジーすぎてDD乘数の誤動作を引き起こす
- WIN/PLACE別DDControllerは統計的性質の違い（的中率10% vs 30-40%）に対応するベストプラクティス
- Walk-forward枠組みでのOptuna評価は、ルックアヘッドバイアス防止の最も厳密な方法
- JSON manifestは人間可読で、Optuna最適化の再現性と監査性を担保
- 全パラメータ一括最適化（~16次元）はTPEで100トライアルで十分に収束可能
- バックテストパスのみ変更（Phase 11 D-13の決定を踏襲）。ライブパスのMetaSwitcherは値揃えのみ

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-Risk Calibration & Parameter Optimization*
*Context gathered: 2026-05-05*
