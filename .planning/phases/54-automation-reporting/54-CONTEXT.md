# Phase 54: Automation & Reporting - Context

**Gathered:** 2026-06-06
**Status:** Ready for planning

<domain>
## Phase Boundary

モデル検証から精算・集計まで1コマンド(`--mode run`)で完遂し、週次/累積/target別の評価レポートでPT結果を正確に測定できること。

具体的には:
1. **1コマンドrun mode (AUT-01~03)** — `--mode run` で setup→TC取得→レース順次予測→最終発走後精算→集計→report を単日セッションで自動実行。クラッシュ復旧時は状態復元で未処理分のみ再開
2. **評価レポート拡張 (RPT-01~04)** — 週次ROI/的中率JSON、累積ベット履歴、Win/Place別集計、model identity を PaperTradingReportAggregator で統一生成

**v2.4対象は Win/Place のみ。Wide は拒否する。**

</domain>

<decisions>
## Implementation Decisions

### Run Mode Orchestration

- **D-01:** `--mode run` は Smart resume モデル。起動時に session_manifest / schedule.json / bets.parquet / race_progress.json から状態を復元し、未完了ステップを自動判定・実行。予測済みなら精算のみ、未予測なら予測→精算
- **D-02:** 単日セッション限定。`--date` 必須。各日が独立した session_id / schedule / live TC snapshot / session_manifest を持つ。日付またぎなし。深夜時点で pending 残なら保存して exit code 2。翌日 run は前日セッションを変更しない
- **D-03:** Setup を内包。schedule.json があれば検証して再利用、なければ setup(レース一覧取得)を自動実行
- **D-04:** Live track condition は最初のレース予測前に全場一括取得(1回のみ)。測定時刻・取得時刻・開催日・開催場を検証。当日 JRA 更新後の値必須。未掲載・古い・一部欠落 → 予測開始前に fail-fast。取得済み HTML と正規化データはセッション内で固定(途中更新しない)
- **D-05:** Sequential per-race 予測フロー。各レース発走 N 分前に最新オッズ・馬体重を取得→予測→記録。TC値は朝固定だが、オッズ・馬体重はレース毎に最新取得。BT の発走5分前オッズ条件と整合。最終発走時刻を過ぎてから精算リトライ開始

### Restart & Resumption

- **D-06:** Explicit progress tracking。`race_progress.json` に race_id ごとの状態を atomic write:
  - 状態: `pending → processing → predicted | no_bet | failed`
  - 記録内容: 状態 + 処理時刻 + 入力 snapshot hash + bet_id 一覧 + 失敗理由
  - 再起動時: `predicted` / `no_bet` をスキップ、`pending` / `failed` / `processing` のみ再処理
- **D-07:** 入力 snapshot 保存。`sessions/{session_id}/inputs/{race_id}.parquet` に各レースの発走前入力(特徴量・オッズ)を保存。再現・比較用途
- **D-08:** Cross-validate on resume。再起動時に race_progress / bets.parquet / 入力 snapshot の3ファイル間で整合性検証:
  - `processing` → 再処理
  - betsのみ存在 → bet_id + snapshot hash 検証 → progress 復元
  - progress=`predicted` で bets 欠損 → 不整合 → 再処理 or fail-fast
  - `no_bet` → bets 行0件を正常状態として検証
- **D-09:** Replay 機能。特徴量・モデル修正後、新しい replay セッションを作成して保存済み入力から再予測可能。元セッション・bets.parquet は変更せず、旧版との選択馬・的中率・ROI を比較
- **D-10:** 責務分離: bets.parquet = ベット記録、race_progress.json = 処理進捗、入力 snapshot = 再現・比較用。累積履歴は bets.parquet を正本とし重複コピーを生成しない

### Reporting Integration

- **D-11:** PaperTradingReportAggregator を集計の唯一実装とする(新規クラス)。新スキーマ(settlement_status/outcome/payout)の bets.parquet から日次・週次・月次・target別統計を生成。CLI JSON・HTML・将来の通知が同じ集計結果を共有し計算差異を防止
- **D-12:** PaperTradingReport は HTML レンダラーに縮小。Aggregator の集計結果を受け取って HTML を描画するのみ
- **D-13:** 出力構造:
  - `daily_summary/YYYY/YYYY-MM-DD.json` — 日次集計(既存を年サブディレクトリ化)
  - `weekly_summary/{iso_year}/W{iso_week:02d}.json` — 週次集計(ISO週、月曜開始、JST基準)
  - `target_summary/YYYY-MM-DD.json` + `target_summary/latest.json` — Win/Place別集計
  - `report.html` — 既存配置
  - 各 JSON 共通フィールド: schema_version, 集計対象期間, 生成時刻, 対象 session_id 一覧
- **D-14:** run 終了時に Aggregator で全種別を自動生成。精算リトライ終了後、pending 残りでも集計・HTML 生成を実行。集計は settled のみ ROI 対象とし、pending件数・未精算 stake・データ完全性ステータスを明示
- **D-15:** 既存 reconcile モード実行後も同じ Aggregator を呼び出し、後日 pending が解消された際にレポートを更新可能
- **D-16:** レポート生成失敗はベット・精算結果を巻き戻さず、専用非ゼロ終了コード(6)を返す

### Error Taxonomy & Exit Codes

- **D-17:** IntEnum `ExitCode` で終了コードを一元管理:
  - `0` = SUCCESS
  - `1` = GENERAL_ERROR
  - `2` = PENDING_REMAIN (精算未完了)
  - `3` = DB_FETCH_ERROR (DB接続・データ取得失敗)
  - `4` = DATA_INTEGRITY_ERROR (データ不整合・欠損)
  - `5` = MODEL_VALIDATION_ERROR (モデル検証失敗)
  - `6` = REPORT_ERROR (レポート生成失敗)
  - `130` = SIGINT (Ctrl+C)
- **D-18:** 複数障害発生時は severity 優先順位で最終コードを決定。実装時に明示的な severity 表を定義。全エラー詳細は session_manifest へ配列で保存
- **D-19:** Model identity (MLflow run ID・学習期間・manifest hash) を Aggregator が session_manifest から取得して全レポートに含める (RPT-04)

### Claude's Discretion

- RaceWatcher と run mode の統合詳細(スリープ間隔、発走時刻判定)
- 発走N分前の N の設定方法(CLI引数 vs 設定ファイル)
- race_progress.json の atomic write の具体的な実装(一時ファイル命名)
- sessions/{session_id}/ ディレクトリ構造の詳細
- Aggregator の週次集計タイミング(ISO週の切り替え境界)
- severity 優先順位表の具体的な定義
- replay セッションのCLI引数設計
- PaperTradingReport HTML テンプレートの新スキーマ対応詳細
- 入力 snapshot parquet のスキーマ(保存する列の選択)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — AUT-01~03, RPT-01~04 の要件定義。Traceability table あり
- `.planning/ROADMAP.md` §Phase 54 — Goal, Success Criteria, Requirements mapping
- `.planning/PROJECT.md` — v2.4 milestone context, Out of Scope 定義
- `.planning/STATE.md` — Phase 51/52/53 deliverables, deferred items

### Prior Phase Context (MUST read — decisions cascade)
- `.planning/phases/51-settlement-integrity-training-pipeline/51-CONTEXT.md` — bet_id, 3列状態モデル, ROI公式, ModelLoader優先度, Wide拒否
- `.planning/phases/52-shared-feature-builder-consistency/52-CONTEXT.md` — FeatureBuilder, FeatureState, DataCutoffManifest, PFPVerifier, session_manifest
- `.planning/phases/53-strategy-alignment-live-data/53-CONTEXT.md` — 戦略manifest注入, DD shadow, Regime AGGRESSIVE固定, TrackConditionFetcher, live TC merge

### PT Pipeline (must-read for integration)
- `scripts/run_paper_trading.py` — PT CLI: 5モード + parse_args() + _build_race_predictor() + _run_predict() + _run_reconcile()。`--mode run` の追加先
- `src/paper_trading/predictor.py` — PaperPredictor: RacePredictor受け取り設計
- `src/paper_trading/reconciler.py` — PaperReconciler: 精算・リトライ・bets.parquet管理
- `src/paper_trading/report.py` — PaperTradingReport: HTMLレンダラー(縮小対象)
- `src/paper_trading/watcher.py` — RaceWatcher: レース監視。run mode での再利用候補
- `src/paper_trading/config.py` — PaperTradingConfig: PT設定

### Automation Components (must-read for integration)
- `src/automation/scheduler.py` — RaceScheduler: レース日タスク統括。OrchestratorProtocol / SafetyGuardProtocol 等の Protocol 定義あり
- `src/automation/safety_guard.py` — SafetyGuard: バンクロール/損失制限(v2.5+)

### Session & Verification Infrastructure (reuse)
- `src/features/session_manifest.py` — SessionManifest: セッション識別情報・PFP検証結果・エラー記録
- `src/features/data_cutoff_manifest.py` — DataCutoffManifest: データカットオフ検証
- `src/backtest/parameter_freeze_protocol.py` — ParameterFreezeProtocol: SHA256 manifest検証

### Reporting Reference
- `src/betting/payout_maps.py` — build_win/build_place payout maps。精算結果の集計で参照
- `config/settings.yaml` — DB接続、データパス、paper_trading設定

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PaperReconciler` (reconciler.py): 冪等性(bet_id), 3列状態モデル, リトライ, atomic replace。run mode の精算フェーズでそのまま利用
- `PaperTradingReport` (report.py): HTML生成(Jinja2)。Aggregator の集計結果を受け取るRendererに縮小
- `RaceScheduler` (scheduler.py): Protocol-based DIパターン(OrchestratorProtocol, SafetyGuardProtocol等)。run mode のオーケストレーション設計参考
- `RaceWatcher` (watcher.py): レース監視機能。Sequential per-race 予測のトリガーとして再利用候補
- `SessionManifest` (session_manifest.py): セッション識別情報・PFP検証結果。run mode の状態管理で再利用
- `build_strategy_config_from_params()` (default_strategy.py): manifest→戦略設定変換。run mode でも共有
- `_build_race_predictor()` (run_paper_trading.py): 既に predict/diagnose/dry-run で共用。run mode でも同じ関数を利用

### Established Patterns
- **Protocol-based DI**: RaceScheduler の各 Protocol が DI パターンを示す。TrackConditionFetcherProtocol と同じパターン
- **Composition root at CLI**: run_paper_trading.py が composition root。run mode も同じ層に追加
- **Atomic replace for writes**: 一時ファイル経由で書込み、renameで原子性担保。race_progress.json でも適用
- **Fail-fast on validation**: OOFHealthValidator, DataCutoffManifest と同じパターン。TC取得・モデル検証でも適用
- **Session-scoped immutable artifacts**: Phase 53 のライブデータパターン。run mode の TC snapshot も同様
- **Deterministic bet_id**: SHA256(session_id|race_id|bet_type|umaban)[:32]。再実行時の重複防止

### Integration Points
- `scripts/run_paper_trading.py` main(): `args.mode == "run"` 分岐追加。parse_args() choices に "run" 追加
- `scripts/run_paper_trading.py` parse_args(): `--mode run` 追加、`--date` 必須化
- `src/paper_trading/report.py`: PaperTradingReport の入力を Aggregator 集計結果に変更
- `src/paper_trading/` 配下: 新規 PaperTradingReportAggregator クラス追加
- `data/paper_trading/` 配下: sessions/{session_id}/ ディレクトリ、race_progress.json、weekly_summary/、target_summary/
- ExitCode IntEnum: 新規 `src/paper_trading/exit_codes.py` または domain/types.py に追加

</code_context>

<specifics>
## Specific Ideas

- run mode は既存5モードとは独立した6番目のモードとして追加し、既存モードの動作を変更しない
- 朝のポーリング回避: レーススケジュールの発走時刻を基準に待機、無意味なDBポーリングを行わない
- 同一日内のTC値固定: レース毎に異なるTC値を使うと再現性とBT/PT比較が困難になるため、朝1回のスナップショットをセッション全体で使用
- 累積履歴は bets.parquet を正本とする: 別ファイルに複製すると差異の元になる
- JSON/HTML 共通集計結果: 計算差異を防ぐため Aggregator が唯一の集計エンジン
- ISO週(JST基準)で週次集計: 曜日境界が明確で年跨ぎも定義済み
- Ctrl+C は 130 を返す: Unix慣例に従い、signal handler で制御

</specifics>

<deferred>
## Deferred Ideas

- SafetyGuard 連動 — v2.5+ (SAF-01, SAF-02)
- Wide bet support — v2.5+ (WID-01, WID-02)
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+ (WRN-01)
- デプロイゲート自動判定 (DEP-01) — v2.5+
- Optuna 19次元パラメータ最適化 (DEP-02) — v2.5+
- デーモン/期間実行モード — ライブ運用の複雑性から単日セッション設計。期間実行は dry-run の責務
- RaceWatcher リアルタイム監視 — バッチ(Sequential per-race)対応を先に行う。リアルタイムは将来検討

</deferred>

---

*Phase: 54-Automation & Reporting*
*Context gathered: 2026-06-06*
