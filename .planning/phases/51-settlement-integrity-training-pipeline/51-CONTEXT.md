# Phase 51: Settlement Integrity & Training Pipeline - Context

**Gathered:** 2026-06-06
**Status:** Ready for planning

<domain>
## Phase Boundary

PT の ROI 測定が信頼できること。全ベット(的中・不的中)が正しく精算され、学習パイプラインが PT 用モデルを生成できる。

具体的には:
1. 精算整合性 — Win/Place の実払戻精算、負け含む全ベットの確定状態保存、ROI過大評価修正
2. 学習パイプライン修正 --betting-target 対応、Parquet検証、track_stats永続化、ModelLoader優先度修正

**v2.4対象は Win/Place のみ。Wide は拒否する。**

</domain>

<decisions>
## Implementation Decisions

### 精算アーキテクチャ (Reconciler Architecture)

- **D-01:** PaperReconciler クラスを精算の唯一実装とする。Phase 51で二重実装を解消し、`_run_reconcile` インラインは薄いCLIラッパー(引数構築・結果表示・終了コード制御)に縮小する。Phase 52に先送りしない。
- **D-02:** `bet_id = SHA256(session_id | race_id | bet_type | canonical_selection)[:32]`。session_id は当日run開始時に生成・永続化し、クラッシュ復旧時も再利用する。canonical_selection は馬番。時刻・stakeは含めない。
- **D-03:** 状態モデルは3列で管理:
  - `settlement_status`: pending → settled
  - `outcome`: NULL → won / lost / refunded / voided
  - `payout`: NULL → float (0.0=loss, >0=win, =stake=refunded/voided)
- **D-04:** `voided` = レース不成立等で投票全体が無効。`refunded` = 出走取消・競走除外。両方とも payout=stake、effective_stake/ROI分母から除外。同着は独立状態にせず `won` として実払戻額を記録。
- **D-05:** ROI集計公式:
  - `effective_stake` = Σ stake WHERE outcome IN (won, lost)
  - `return` = Σ payout WHERE outcome IN (won, lost)
  - `ROI` = return / effective_stake
  - `net_profit` = return - effective_stake
- **D-06:** リトライ戦略: per-race 1回取得(未到着ならpending維持) + 最終レース後60s間隔で最大10分間(絶対期限)全pending一括再取得。DB接続エラーと「接続成功・払戻未掲載」を区別して記録。期限後もpendingが残れば保存して終了コード2。再実行はpendingのみ処理。
- **D-07:** 部分精算の保存は一時ファイル経由のatomic replace。
- **D-08:** 累積 `bets.parquet` を精算状態の正本(source of truth)とする。`predictions/` は予測時点の監査記録とする。

### Win/Place精算統合 (Settlement Integration)

- **D-09:** `src/betting/payout_maps.py` に `build_win_payout_map` / `build_place_payout_map` を純粋関数として抽出。BT engine と PaperReconciler の両方が同一関数を使用。
- **D-10:** 払戻マップの出力は「100円あたりの円」ではなく**倍率**に統一。入力の文字列・数値・欠損表現を正規化。同着による複数単勝払戻に対応。
- **D-11:** 精算判定順序:
  1. 返還・不成立データを先に判定 → refunded / voided
  2. 払戻データにレースが存在 → 精算可能
  3. 対象馬が払戻マップに存在 → won
  4. 存在しない → lost
  5. 同一race_idの払戻行が複数 → 安全に統合
  6. 不正な払戻値 → lostにせずpending維持(精算エラー扱い)
- **D-12:** ヘルパーにEveryDB2アクセスやファイルI/Oを含めない。

### 学習パイプライン (Training Pipeline)

- **D-13:** `--betting-target` 別の学習スコープ:
  - `win`: 共通モデル(ability, market, regime, quality) + Win固有モデル
  - `place`: 共通モデル + Win基盤モデル + Place固有モデル (PlaceはWinに依存)
  - `wide`: v2.4では対象外として**拒否** (エラー終了)
- **D-14:** 学習targetをMLflow + `meta.json` に保存。PT起動時にモデルtargetと`--betting-target`の一致を必須検証。
- **D-15:** track_stats / track_month_stats はローカル(`data/models/`) + MLflow artifacts の両方に保存。`track_stats_{surface}.json` / `track_month_stats_{surface}.json` をモデル成果物の必須ファイルとする。SHA256を meta.json + MLflow params/tags に記録。
- **D-16:** ModelLoader優先度: PTでは `--run-id` 必須。MLflowからのみロード、暗黙ローカルフォールバックなし。ローカル利用は `--models-dir` 明示指定時のみ。`--run-id` と `--models-dir` の同時指定は禁止。「最新run自動選択」も禁止。ロード元を実行記録・レポートに保存。
- **D-17:** 必須成果物欠落時は fail-fast (非ゼロ終了)。

### ベット記録スキーマ (Bet Record Schema)

- **D-18:** `result` 列を廃止 → `payout` に完全置換。旧スキーマ(`result`列のみ)の成果物は自動変換せず明示的に拒否。
- **D-19:** `schema_version=2` 列を追加。
- **D-20:** 書き込み前整合性検証:
  - pending: outcome=NULL, payout=NULL
  - settled: outcome!=NULL, payout>=0
  - lost: payout=0
  - won: payout>0
  - refunded/voided: payout=stake
  - bet_id: 非NULLかつ一意
  - stake>0
  - schema_version=2固定

### Claude's Discretion

- payout_maps.py の内部実装詳細(正規化ロジック、統合方法)
- PaperReconciler の内部リトライループ実装
- Pre-training Parquet検証の具体的なチェック内容(NaN率閾値等)
- Feature cache dependency tracking のキャッシュキー計算方式
- atomic replace の一時ファイル命名規則

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — STL-01~05, TRN-01~05 の要件定義。Traceability table あり
- `.planning/ROADMAP.md` §Phase 51 — Goal, Success Criteria, Requirements mapping
- `.planning/PROJECT.md` — v2.4 milestone context, Out of Scope 定義

### Existing Implementation (must-read for integration)
- `src/paper_trading/reconciler.py` — 現在のPaperReconcilerクラス(拡張対象)
- `scripts/run_paper_trading.py:899-1115` — 現在の_run_reconcileインライン(ラッパー化対象)
- `scripts/run_paper_trading.py:294-684` — _run_predict (bet_id/session_id/payout列追加対象)
- `src/backtest/engine.py:211-229` — build_win_payout_map (payout_maps.pyへの移動対象)
- `src/backtest/engine.py:170-208` — build_place_payout_map (payout_maps.pyへの移動対象)
- `scripts/run_train.py` — --betting-target, track_stats保存追加対象
- `src/pipelines/training_pipeline.py:970-990` — track_stats/track_month_stats計算箇所
- `src/db/model_loader.py:39-55` — 現在の優先度ロジック(run_id指定時修正対象)

### Domain Types & Models
- `src/domain/types.py` — BetType, POST_RACE_COLS
- `src/domain/models.py` — SubmodelSet (track_statsフィールド), TrainedModelsV5, Bet

### Configuration
- `config/settings.yaml` — DB接続、データパス、feature_engine設定
- `.planning/config.json` — GSD設定

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_win_payout_map()` / `build_place_payout_map()` (engine.py): 純粋関数、payout_maps.pyへの抽出対象。既にベクトル化実装済み
- `PaperReconciler` クラス (reconciler.py): 冪等性パターン(race_id+umabanで既存チェック)が再利用可能
- `ParameterFreezeProtocol` (backtest/parameter_freeze_protocol.py): SHA256 manifest検証パターンがtrack_stats検証に適用可能
- `SubmodelSet` (domain/models.py): track_stats/track_month_stats フィールドが既に定義済み (dict | None)

### Established Patterns
- **Parquetベースデータ層**: PostgreSQLはETL専用、推論時はParquetのみ。精算もParquetが正本
- **Pure function helper抽出**: payout_maps.py は I/Oなし純粋関数。BT/PT両方から利用
- **Fail-fast on missing artifacts**: OOFHealthValidator, FeatureRoutingAudit と同じパターン
- **Atomic replace**: 一時ファイル経由で書込み、renameで原子性担保

### Integration Points
- `scripts/run_paper_trading.py` main(): `_run_reconcile` → `PaperReconciler` 呼出に変更
- `scripts/run_paper_trading.py` _run_predict(): bet_id/session_id/payout列の生成
- `scripts/run_train.py` main(): --betting-target引数追加、track_stats保存
- `src/pipelines/training_pipeline.py` _train_submodel(): track_stats保存処理
- `src/db/model_loader.py` load(): run_id指定時の優先度変更
- `src/backtest/engine.py`: build_win/build_place を payout_maps.py import に変更

</code_context>

<specifics>
## Specific Ideas

- session_id はクラッシュ復旧時に再利用可能にするため、永続化ファイルに保存
- 累積bets.parquetを精算状態の正本(source of truth)とし、predictions/は予測時点の監査記録
- place学習には依存するWin基盤モデルも含める(Stage1 → Market → Win → Placeのパイプライン順序)
- 不正な払戻値(0, NULL, negative)はlostにせずpendingを維持し、人間の目視確認を挟む

</specifics>

<deferred>
## Deferred Ideas

- Wide bet settlement — v2.5+ (WID-01, WID-02)
- SafetyGuard integration — v2.5+ (SAF-01, SAF-02)
- Shared feature builder extraction — Phase 52 (PLN-01)
- Pipeline identity recording — Phase 52 (PLN-02, PLN-03, PLN-04)
- Strategy manifest integration — Phase 53 (STR-01~06)
- Live data fetcher — Phase 53 (LIV-01~03)
- One-command run mode — Phase 54 (AUT-01~03)
- Weekly/cumulative reporting — Phase 54 (RPT-01~04)
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+ (WRN-01)

</deferred>

---

*Phase: 51-Settlement Integrity & Training Pipeline*
*Context gathered: 2026-06-06*
