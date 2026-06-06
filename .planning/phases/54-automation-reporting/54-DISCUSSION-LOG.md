# Phase 54: Automation & Reporting - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-06
**Phase:** 54-Automation & Reporting
**Areas discussed:** Run mode orchestration, Restart & resumption, Reporting integration, Error taxonomy & exit codes

---

## Run mode orchestration

### Q1: `--mode run` の実行モデル

| Option | Description | Selected |
|--------|-------------|----------|
| Smart resume | 起動時に未完了ステップを自動判定。予測済みなら精算のみ。bets.parquetのsettlement_statusで状態判定 | ✓ |
| Full pipeline (always) | 毎回 setup→predict→wait→reconcile→report を全実行。単一長時間プロセス | |
| Two-phase explicit | `--phase predict` と `--phase reconcile` で明示的に分離 | |

**User's choice:** Smart resume — 朝の予測直後から夜までのポーリングは行わない。RaceWatcherで発走前に順次予測し、全レース処理後、最終発走時刻を過ぎてから精算リトライ開始。再起動時は同じsession_id再利用。

### Q2: Live TC取得タイミング + setup内包

| Option | Description | Selected |
|--------|-------------|----------|
| Setup内包 + 朝1回取得 | run起動時にsetupも自動実行。live TCは最初のレース予測前に全場一括取得1回のみ | ✓ |
| Setup内包 + レース毎取得 | run起動時にsetupも自動実行。live TCは各レース予測直前に個別取得 | |
| Setup別 + 朝1回取得 | runはsetup済み前提。live TCは最初の予測前に1回のみ | |

**User's choice:** Setup内包 + 朝1回取得。schedule.jsonがあれば検証して再利用。当日JRA更新後の値必須。取得済みHTMLと正規化データはセッション内で固定。

### Q3: 対象期間

| Option | Description | Selected |
|--------|-------------|----------|
| Single-day only | --date必須。翌日は別プロセス起動 | ✓ |
| Single + date range | --dateで単日、--start/--endで期間対応 | |
| Daemon mode | 起動時から終了日まで連続実行 | |

**User's choice:** Single-day only。日付ごとに独立session_id。深夜時点でpending残ならexit code 2。期間実行はdry-runの責務。

### Q4: 予測フロー

| Option | Description | Selected |
|--------|-------------|----------|
| Batch predict → wait → reconcile | 全レース一括予測→最終発走時刻まで待機→精算 | |
| Sequential per-race | 各レース発走前に個別予測→最終発走後精算 | ✓ |

**User's choice:** Sequential per-race。TC値が固定でも発走5分前オッズ・オッズ時系列・馬体重はレース毎に異なるため、各レース発走N分前に最新データ取得→予測。BTの発走5分前オッズ条件と整合。

---

## Restart & resumption

### Q5: 予測済みレースの判定方法

| Option | Description | Selected |
|--------|-------------|----------|
| Bets-only (simpler) | bets.parquetのrace_idで予測済判定。should_bet=Falseは再予測 | |
| Explicit progress | session_manifestまたは別JSONに予測済race_idを記録。should_bet=Falseも記録 | ✓ |

**User's choice:** Explicit progress。race_progress.jsonにrace_idごとの状態(pending/processing/predicted/no_bet/failed)+処理時刻+入力snapshot hash+bet_id一覧+失敗理由をatomic write。再起動時はpredicted/no_betをスキップ。各レースの入力をsessions/{session_id}/inputs/{race_id}.parquetへ保存。replay機能で新セッション比較可能。bets.parquetはベット記録、race_progress.jsonは進捗、入力snapshotは再現用として責務分離。

### Q6: クラッシュ時の挙動

| Option | Description | Selected |
|--------|-------------|----------|
| Atomic write guarantees | race_progress.jsonのatomic writeで確定済みのみ反映。未記録はpending扱い | |
| Cross-validate on resume | 再起動時にprogress/bets/snapshotを相互検証し矛盾を解消 | ✓ |

**User's choice:** Cross-validate on resume。状態遷移は pending→processing→predicted/no_bet/failed。betsを先にatomic保存→progress確定。processingは再処理。betsのみ存在はbet_id+snapshot hash検証→progress復元。predictedでbets欠損は不整合→再処理or fail-fast。決定的bet_idで重複防止。

---

## Reporting integration

### Q7: 既存PaperTradingReportとの統合

| Option | Description | Selected |
|--------|-------------|----------|
| Extend existing | PaperTradingReportを更新+週次/累積メソッド追加 | |
| New aggregator class | 新規ReportAggregatorを作成。PaperTradingReportは廃止 | ✓ |
| You decide | Claudeに判断を任せる | |

**User's choice:** New aggregator class。ただしPaperTradingReportは廃止せずHTMLレンダラーとして残す。PaperTradingReportAggregatorを集計の唯一実装とし、PaperTradingReportは集計済み結果を受け取ってHTML描画するのみ。JSON/HTML共通集計で計算差異防止。

### Q8: 出力配置先

| Option | Description | Selected |
|--------|-------------|----------|
| Structured subdirs | weekly_summary/{year}/W{week}.json等のサブディレクトリ構成 | ✓ |
| Flat directory | 全てpaper_trading直下に配置 | |

**User's choice:** Structured subdirs。ISO週(月曜開始、JST基準)を使用。daily_summary/YYYY/YYYY-MM-DD.json、weekly_summary/{iso_year}/W{iso_week:02d}.json、target_summary/YYYY-MM-DD.json + latest.json。各JSONにschema_version/集計対象期間/生成時刻/session_id一覧を含める。累積履歴はbets.parquet参照のみ(複製なし)。

### Q9: レポート生成タイミング

| Option | Description | Selected |
|--------|-------------|----------|
| Auto at run end | 精算完了後にAggregatorで全種別自動生成。--mode runの最終ステップ | ✓ |
| Report as separate mode | run終了時はdailyのみ。weekly/htmlは別--mode report | |

**User's choice:** Auto at run end。精算リトライ終了後、pending残りでも集計・HTML生成実行。集計はsettledのみROI対象。pending件数・未精算stake・データ完全性ステータスを明示。レポート生成失敗はベット・精算を巻き戻さず専用終了コード。reconcileモード後も同じAggregatorを呼び出し更新可能。

---

## Error taxonomy & exit codes

### Q10: 終了コード体系

| Option | Description | Selected |
|--------|-------------|----------|
| Structured codes (0-6) | 0=成功、1=一般、2=pending、3=DB、4=データ、5=モデル、6=レポート | ✓ |
| Minimal (0/1/2) | 0=成功、1=エラー、2=pendingのみ。詳細はログ | |

**User's choice:** Structured codes (0-6)。IntEnumで一元管理。複数障害時はseverity優先順位で最終コード決定。全エラー詳細はsession_manifestへ配列で保存。Ctrl+Cは130。実装時に明示的なseverity表を定義。

---

## Claude's Discretion

- RaceWatcher と run mode の統合詳細(スリープ間隔、発走時刻判定ロジック)
- 発走N分前の N の設定方法(CLI引数 vs settings.yaml)
- race_progress.json の atomic write 実装
- sessions/{session_id}/ のディレクトリレイアウト
- ISO週のJST変換実装
- severity 優先順位表の具体的な定義
- replay セッションのCLI引数設計
- PaperTradingReport HTMLテンプレート新スキーマ対応

## Deferred Ideas

None — discussion stayed within phase scope
