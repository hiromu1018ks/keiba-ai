# Requirements: keiba-ai v2.4 Paper Trading Pipeline Integration

**Defined:** 2026-06-06
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1 Requirements

### Settlement (精算整合性)

- [ ] **STL-01**: Win/Place bet status tracking — `bets.parquet` に明示的な `status` 列 (pending/settled) を追加し、負けベットを含む全ベットの確定状態を記録する
- [ ] **STL-02**: Win actual payout settlement — `build_win_payout_map()` パターンを再利用し、単勝払戻を精算する
- [ ] **STL-03**: Place actual payout settlement — 既存複勝払戻ロジックを修正し、負けベットも `result=loss_stake` として記録する
- [ ] **STL-04**: ROI calculation fix — 的中のみ記録による ROI 過大評価を修正し、負け含む全ベットで正確な ROI を算出する
- [ ] **STL-05**: Payout retry fetch — DB 遅延時に払戻データをリトライ取得する。最終レース後に一括リトライを実行する

### Pipeline Consistency (パイプライン一貫性)

- [ ] **PLN-01**: Shared feature builder extraction — `BacktestEngine.prepare_data()` から `build_inference_features()` を抽出し、BT/PT が共通の特徴量構築関数を呼ぶことを受入条件とする。7つのギャップ (DamPedigree/Record/Mining/PaceAptitude 3列/Sire/Course) を一括解消する
- [ ] **PLN-02**: Pipeline identity recording — MLflow run ID・学習期間・コードハッシュ・feature manifest hash を PT 実行記録に保存する
- [ ] **PLN-03**: Data cutoff validation — 2026年 PT では 2025年12月31日以前のデータのみ使用。特徴量統計・OddsBandFilter 校正・HP・strategy manifest よ予測日以降の情報を含まないことを検証する
- [ ] **PLN-04**: PFP parameter immutability — PT 実行中のパラメータ不変性を ParameterFreezeProtocol で検証する

### Training Pipeline (学習パイプライン修正)

- [ ] **TRN-01**: run_train.py `--betting-target` support — `--betting-target win|place|wide` 引数を追加し、単勝 PT 用モデルを学習可能にする
- [ ] **TRN-02**: Pre-training Parquet validation — 学習開始前に必須 Parquet の日付範囲・NaN率・更新日時を検証し、不正時は非ゼロ終了する
- [ ] **TRN-03**: Feature cache dependency tracking — track_conditions.parquet / horse_track_aptitude.parquet を特徴量キャッシュの依存元に追加し、更新後も古いキャッシュを使わない
- [ ] **TRN-04**: track_stats persistence — 学習した track_stats / track_month_stats をモデル成果物に保存・復元する。PT で季節偏差等が NaN にならないようにする
- [ ] **TRN-05**: ModelLoader priority fix — `data/models/` を MLflow run ID より優先しない。意図しないローカルモデル読込を防止する

### Live Data (当日データ取得)

- [ ] **LIV-01**: JRA track condition fetcher — JRA 公式サイト (https://www.jra.go.jp/keiba/baba/kaisetsu/index.html) から開催場ごとの芝クッション値・ダート含水率を取得し、race_id へ展開する
- [ ] **LIV-02**: Live data validation — 取得値・測定時刻・取得時刻・取得元を保存する。取得失敗または値が古い場合は予測を停止し非ゼロ終了する
- [ ] **LIV-03**: Same schema as historical — 過去 CSV と当日取得値は同一スキーマ・同一集約規則で扱う。共通特徴量ビルダーへ渡す

### Strategy Alignment (戦略完全整合)

- [ ] **STR-01**: Strategy manifest integration — PT で strategy_manifest を読み込み、manifest/PFP を適用する
- [ ] **STR-02**: Betting mode/target passthrough — PT で `--betting-target win|place|wide` と `--betting-mode flat|kelly` を指定可能にする
- [ ] **STR-03**: DD control integration — DrawdownController を PT パイプラインに組み込む
- [ ] **STR-04**: OddsBandFilter integration — BT の校正済み OddsBandFilter を PT で使用する
- [ ] **STR-05**: QualityScreener integration — RaceQualityScreener を PT パイプラインに組み込む
- [ ] **STR-06**: Regime synchronization — BT/PT の regime 検出を統一する（AGGRESSIVE固定 vs 動体の決定含む）

### Automation (自動化)

- [ ] **AUT-01**: One-command run mode — `--mode run` を追加。モデル検証→予測→監視→精算→集計の全工程を1コマンドで実行する。学習は事前実行前提
- [ ] **AUT-02**: Restart resumption — 処理済みレースの再実行をスキップする冪等性を保証する。クラッシュ後の再起動で未処理レースのみ再開する
- [ ] **AUT-03**: DB failure exit codes — DB接続障害・データ欠損・モデル不整合時に非ゼロ終了コードを返す

### Reporting (評価レポート拡張)

- [ ] **RPT-01**: Weekly aggregation — 週次 ROI・的中率・ベット数の JSON 集計を出力する
- [ ] **RPT-02**: Cumulative history with losses — pending/settled/won/lost を含む累積ベット履歴を記録・出力する
- [ ] **RPT-03**: Per-target aggregation — Win/Place 別の ROI・的中率集計を出力する
- [ ] **RPT-04**: Model identity in reports — MLflow run ID・学習期間・manifest hash をレポートに含める

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Safety Guard Integration

- **SAF-01**: SafetyGuard 連動 — バンクロール/損失制限チェックを PT パイプラインに組み込む (v2.5+)
- **SAF-02**: Emergency stop — 緊急停止フラグによる全ベット即時停止 (v2.5+)

### Wide Bet Support

- **WID-01**: Wide 払戻対応 — ワイドベットの精算・ROI 計算 (複雑度高、v2.5+)
- **WID-02**: WideJointPairBuilder 統合 — PT へワイドペア生成を組み込む (v2.5+)

## Out of Scope

| Feature | Reason |
|---------|--------|
| SafetyGuard 連動 | v2.4 ではクラッシュリカバリで対応。SafetyGuard は v2.5+ で統合 |
| Wide 払戻対応 | 複雑度高。Win/Place での精正確立が先。v2.5+ で検討 |
| 実馬券購入 (IPAT API) | ペーパートレードまで。PatVoter は将来の実運用用 |
| RaceWatcher リアルタイム監視 | バッチモード対応を先に行う。リアルタイムは将来検討 |
| MLflow モデル自動デプロイ | 手動 run_train.py → PT のフローを維持 |
| PT 用マニュフェスト最適化 | BT manifest をそのまま使用。安全側調整なし |
| v2.4以前の PT レコード移行 | v2.4 開始日以降のデータのみ扱う。過去データ移行は別途検討 |
| Conservative MAWC redesign | v2.5+ |
| デプロイゲート自動判定 (DEP-01) | v2.5+ |
| Optuna 19次元パラメータ最適化 (DEP-02) | v2.5+ |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| STL-01~05 | TBD | Pending |
| PLN-01~04 | TBD | Pending |
| TRN-01~05 | TBD | Pending |
| LIV-01~03 | TBD | Pending |
| STR-01~06 | TBD | Pending |
| AUT-01~03 | TBD | Pending |
| RPT-01~04 | TBD | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 0
- Unmapped: 26 ⚠️

---
*Requirements defined: 2026-06-06*
*Last updated: 2026-06-06 after initial definition*
