# Phase 18: Validation & Freeze - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

アンサンブルバックテスト(学習2020-2023/テスト2024-2025)で全体ROI>100%かつ合計100+ベットを達成していることを確認し、最適化済みパラメータをSHA256改ざん検知付きで固定している状態になる。

**In scope:**
- VAL-01: アンサンブルバックテスト(--ensemble + --manifest)の結果が全体ROI>100%かつ合計100+ベットを達成していることを確認する
- VAL-02: ParameterFreezeProtocolが最適化済みパラメータをJSON manifestに固定し、SHA256ハッシュで改ざん検知が有効になっている
- run_backtest.pyへの--manifest引数追加(SHA256検証 + パラメータ注入)
- BacktestEngine.run()内でのmanifest読込 + PFP freeze/verify二重検証
- 検証結果JSON(data/validation/)の出力
- ROI<100%時の原因分析レポート自動生成
- mockベースの自動テスト

**Out of scope:**
- 新しいモデルや特徴量の追加
- フィルター閾値の再調整(別フェーズ)
- Optuna再実行(Phase 17完了済み)
- 複勝/ワイドモデルの変更
- マイルストーン完了手続き(別コマンド)

</domain>

<decisions>
## Implementation Decisions

### バリデーション実行フロー
- **D-01:** run_backtest.pyに--manifest引数を追加し、manifest読込→SHA256検証→パラメータ注入→backtest実行を一本化する。新規スクリプトは作成しない。
- **D-02:** manifestパラメータの注入はBacktestEngine.run()内で行う。run()の先頭でmanifestを読み込み、verify_strategy_manifest()でSHA256照合後にstrategy_paramsにマージしてrace loopを開始する。
- **D-03:** SHA256 + ParameterFreezeProtocol二重検証を採用。(1) verify_strategy_manifest()でmanifest SHA256照合、(2) ParameterFreezeProtocol.freeze()/verify()でOOS期間中のモデル不変性を保証。
- **D-04:** manifest検証失敗時はRuntimeErrorで即時エラー停止。SHA256不一致やPFP verify失敗の場合はバリデーションを続行しない。

### テスト期間と成功基準
- **D-05:** バリデーション対象期間: テスト2024-2025の2年。学習2020-2023に対して完全OOS。Phase 17 Optuna最適化の学習期間と整合。
- **D-06:** 成功基準の解釈: テスト期間全体のROI>100%かつ合計100+ベット。「年間」を「テスト期間全体」として解釈。年別内訳は参考情報として記録するが、パス/フェイルの判定には使用しない。

### 検証レポートとマイルストーン
- **D-07:** 最小限の検証レポートを採用。backtest_result.json + ParameterFreezeProtocol検証結果(PASS/FAIL)を組み合わせた単一レポート。Phase 14-17の診断結果は既に各フェーズで出力済みなので再構成しない。
- **D-08:** レポートはJSON形式でdata/validation/に出力。プログラムで読み取り可能。PFP検証結果、ROI、ベット数、テスト期間、年別内訳を含む。
- **D-09:** テスト方針: mockベースの自動テスト + Human UAT。manifest読込、SHA256検証、PFP freeze/verifyの単体テストを追加。実データでのバリデーション実行はHuman UAT。

### 失敗時の対応
- **D-10:** ROI<100%の場合はフェーズ不完了とし、改善提案を文書化してユーザー判断を待つ。自動再調整ループは過学習リスクが高いため組み込まない。
- **D-11:** ROI<100%時の原因分析レポートを自動生成する。オッズバンド別ROI、レジーム別ROI、EV診断の過大/過小評価、ベット数不足等の分析を含む。

### Claude's Discretion
- BacktestEngine.run()内でのmanifest読込ロジックの具体的な実装
- 検証結果JSONのスキーマ設計
- 原因分析レポートの具体的な分析項目と出力形式
- --manifestと--ensemble/--years引数の組み合わせバリデーション
- テストのfixtureデータの内容
- PFP freeze/verifyのタイミング(run()のどこでfreezeし、どこでverifyするか)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ParameterFreezeProtocol（主変更対象 — VAL-02）
- `src/backtest/parameter_freeze_protocol.py` — freeze(), verify(), frozen_period(), save_strategy_manifest(), verify_strategy_manifest(), load_and_freeze_strategy()
- `src/backtest/parameter_freeze_protocol.py:107-128` — save_strategy_manifest()。JSON保存 + SHA256返却
- `src/backtest/parameter_freeze_protocol.py:131-161` — verify_strategy_manifest()。SHA256照合 + ValueError送出

### BacktestEngine（manifest注入ポイント — VAL-01/D-02）
- `src/backtest/engine.py:414-429` — run()メソッド。manifest読込 + PFP検証の追加ポイント
- `src/backtest/engine.py:363-385` — BacktestEngineコンストラクタ。strategy_paramsから各コンポーネント生成

### run_backtest.py（CLI拡張ポイント — D-01）
- `scripts/run_backtest.py` — --manifest引数追加。--ensemble/--yearsとの組み合わせ
- `scripts/run_backtest.py:86` — --ensembleフラグ定義
- `scripts/run_backtest.py:455` — pipeline.run()呼び出し

### StrategyOptimizer（manifest生成元 — Phase 17成果物）
- `src/tuning/strategy_optimizer.py:535-537` — optimize()内のsave_strategy_manifest()呼び出し
- `src/tuning/strategy_optimizer.py:388-392` — optimize_multi_seed()内のsave_strategy_manifest()呼び出し

### 既存テストパターン
- `tests/test_backtest_engine.py` — BacktestEngine既存テスト(1198行)。mockベース
- `tests/test_parameter_freeze_protocol.py` — PFP既存テスト

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/17-optuna-optimization/17-CONTEXT.md` — Phase 17決定(16次元Optuna、4fold、multi-seed安定性、manifest自動保存)
- `.planning/phases/16-odds-band-rebuild/16-CONTEXT.md` — Phase 16決定(ルックアヘッド修正、training_bet_history自動生成)
- `.planning/phases/15-ev-filter-enhancement/15-CONTEXT.md` — Phase 15決定(EV_lower動的閾値、EV診断)
- `.planning/phases/14-gate-recalibration/14-CONTEXT.md` — Phase 14決定(ドリフト診断、use_ensemble伝播)

### 要件定義
- `.planning/REQUIREMENTS.md` — VAL-01, VAL-02の要件定義
- `.planning/ROADMAP.md` — Phase 18 Success Criteria

### ドメイン型
- `src/domain/types.py:29-34` — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **ParameterFreezeProtocol** (`parameter_freeze_protocol.py`): freeze()/verify()/frozen_period()が完備。save_strategy_manifest()/verify_strategy_manifest()でJSON manifest + SHA256管理。 VAL-02の主機能は既に実装済み
- **BacktestEngine.run()** (`engine.py:414-429`): strategy_paramsを既に受け取る。manifest読込ロジックをrun()の先頭に追加するだけでD-02実現可能
- **run_backtest.py**: --ensemble/--yearsフラグ実装済み。--manifest引数の追加は既存パターンの拡張
- **BacktestReportGenerator** (`src/backtest/report.py`): バックテスト結果のレポート生成。原因分析レポート(D-11)のベースとして再利用可能

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 18テストもこのパターンに従う
- **JSON+コンソール出力**: Phase 14-17の診断パターン。検証レポート(D-08)もこの形式
- **パイプライン統合パターン**: use_ensemble=True時の自動診断・キャリブレーション実行。Phase 14-16で確立
- **コンストラクタ注入パターン**: パラメータはコンストラクタ引数で注入

### Integration Points
- **engine.py:414-429** — BacktestEngine.run()にmanifest読込 + PFP検証を追加(D-02, D-03)
- **scripts/run_backtest.py** — --manifest引数追加 + manifestパスのエンジンへの渡し(D-01)
- **新規出力**: data/validation/ ディレクトリに検証結果JSON(D-08)
- **新規機能**: ROI<100%時の原因分析レポート自動生成(D-11)

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- SHA256 + PFP二重検証は最も厳密なパラメータ不変性保証
- テスト期間2024-2025は学習期間2020-2023に対して完全OOS
- ROI<100%の場合でも原因分析レポートを出力することで、次の改善アクションが明確になる
- Phase 17のmanifest自動保存が既に実装されているため、Phase 18は読込・検証側のみ実装すればよい

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 18-Validation & Freeze*
*Context gathered: 2026-05-06*
