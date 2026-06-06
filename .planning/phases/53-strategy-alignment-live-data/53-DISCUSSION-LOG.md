# Phase 53: Strategy Alignment & Live Data - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-06
**Phase:** 53-Strategy Alignment & Live Data
**Areas discussed:** Regime統一方針, Manifest注入アーキテクチャ, JRAライブデータ取得, OddsBandFilter適用範囲

---

## Regime統一方針 (STR-06)

| Option | Description | Selected |
|--------|-------------|----------|
| AGGRESSIVE固定を維持（推奨） | 現状維持。シンプルで安全。Turf conservativeのマイナスROI問題を回避。shadow記録のみ | ✓ |
| 両方とも動的に変更 | detect()有効化、3状態動的判定。Turf CONSERVATIVE対策必要 | |
| PTのみ動的 | PTのみdetect()有効化。「同一実装」契約に反する | |

**User's choice:** AGGRESSIVE固定を維持
**Notes:** BT/PTともベット判断はAGGRESSIVE固定。RegimeDetectorの推定結果は診断ログへshadow記録するが、閾値・Kelly・ベット停止には反映しない。動的化はTurf CONSERVATIVEを含むWF検証で有効性を確認した別マイルストーンで扱う。これで一貫性を保ちつつ、将来の動的化に必要な実績データを蓄積できる。

---

## Manifest注入アーキテクチャ (STR-01/02/03)

### Manifest注入層

| Option | Description | Selected |
|--------|-------------|----------|
| CLI層で注入（BTと同一パターン）（推奨） | run_paper_trading.pyをcomposition root。build_strategy_config_from_params()共有 | ✓ |
| PaperPredictor内で注入 | PaperPredictorクラス内でmanifest読込。CLIはpathのみ渡す | |
| 新規クラスに抽出 | StrategyConfigurator新設。BT/PT両方が利用 | |

**User's choice:** CLI層で注入
**Notes:** manifestは起動時に一度だけ検証・読み込みし、immutableな戦略設定へ変換。PaperPredictorはファイルI/Oを行わず、構築済みRacePredictorを受け取る。manifest path・SHA256・適用パラメータをsession_manifest.jsonへ保存しPFP対象。AGGRESSIVE固定のためregime overrideは記録のみ。OddsBandFilterはmanifest値だけでなく校正済み状態を明示的に注入。新規StrategyConfiguratorは不要。

### Betting Mode/Target指定

| Option | Description | Selected |
|--------|-------------|----------|
| BTと同一CLI設計（推奨） | --betting-target win|place と --betting-mode flat|kelly を追加。Wide拒否 | ✓ |
| manifestを正本・CLIは上書き | strategy_manifestでtarget/mode決定、CLIは上書き用 | |

**User's choice:** BTと同一CLI設計
**Notes:** --betting-targetと--betting-modeは必須引数とし、暗黙デフォルトを設けない。Wideは引数解析時に拒否。モデル学習target/manifest target/CLI targetが不一致ならfail-fast。mode/targetはsession_manifestとPFP対象へ含める。実行時の意図はCLIで明示する方が安全。

### DD制御のPT組み込み

| Option | Description | Selected |
|--------|-------------|----------|
| BTと同一DD制御（推奨） | kelly選択時にDDController注入。NORMAL/REDUCED/STOP 3段階 | |
| Shadow記録のみ | DD状態を計算して診断ログ記録のみ。stake調整なし | ✓ |

**User's choice:** Shadow記録のみ（ユーザー回答: カスタム）
**Notes:** PTではDDControllerによるstake縮小・新規ベット停止を適用しない。DD状態を計算して診断ログにshadow記録するだけ。ペーパートレードの観測機会を減らさず、全シグナルを評価する。実運用移行時にDD制御を有効化。PTのkellyはKelly stake変更のみ。BTとのROI比較ではDD制御なし条件のBTと比較する必要がある。

---

## JRAライブデータ取得 (LIV-01/02/03)

### スクレイピング技術

| Option | Description | Selected |
|--------|-------------|----------|
| Playwright（推奨） | 既存依存。JS対応可能。Protocol-based DIでテスト容易 | ✓ |
| requests + BeautifulSoup | 軽量。JS対応不可の場合あり | |

**User's choice:** Playwright
**Notes:** Playwrightは取得だけを担当し、HTML解析は純粋関数へ分離。取得HTMLを保存し、fixtureによるパーサーテスト、必須項目・開催日・開催場の検証、HTML構造変更検知を実装。Protocol経由でモック可能にし、取得失敗時は古い値へフォールバックせず予測を停止する。

### 含水率集約規則

| Option | Description | Selected |
|--------|-------------|----------|
| 平均値 | ゴール前+4コーナー平均。履歴CSVに最も近い | |
| ゴール前優先 | ゴール前優先、NaN時のみ4コーナー | |
| 最大値（保守的） | 複数ポイントの最大値 | |

**User's choice:** ユーザー回答: 即決しない（カスタム）
**Notes:** 平均値を即決しない。既存CSVのdirt_moistureがゴール前・4コーナー・平均のどれ由来かコード上で確認できないため、重複期間のJRA値とCSV値を照合して集約規則を確定する。ライブ取得では両地点の生値を保存し、検証済み規則からdirt_moistureを算出。照合不能なら予測を停止。推測で平均にすると学習・PT間で特徴量定義が変わる。

### FeatureBuilderへの統合

| Option | Description | Selected |
|--------|-------------|----------|
| Parquet追記（推奨） | track_conditions.parquetに追記。同一コードパス保証 | |
| in-memory merge | ライブDataFrameを別途渡し、mergeは実行時のみ | ✓ |

**User's choice:** in-memory merge（ユーザー回答: カスタム）
**Notes:** ライブ生値を履歴の正本track_conditions.parquetへ直接追記すると、不完全取得や再実行で履歴データを汚染する危険がある。ライブ生値はセッション配下へimmutableに保存し、正規化済みDataFrameをFeatureBuilder.build_for_inference()へ明示的に渡す。FeatureBuilderは対象日のライブ値を履歴Parquetより優先してマージ。取得元・測定時刻・取得時刻・raw HTML hashをsession_manifestへ記録。履歴Parquetへの反映は検証後の別ETL処理。

---

## OddsBandFilter適用範囲 (STR-04)

| Option | Description | Selected |
|--------|-------------|----------|
| winのみ（BTと同一）（推奨） | win-target時のみ作成・適用。placeでは作成しない | ✓ |
| win+place両対応 | win/placeともOddsBandFilter作成。place用バンド閾値は別途定義 | |

**User's choice:** winのみ（ユーザー回答: カスタム）
**Notes:** PTではbetting_target=winの場合のみ、BTと同じ校正済みOddsBandFilterを適用。placeでは生成・適用しない。校正データ終了日・ROI閾値・除外バンド・設定hashをモデル成果物とsession_manifestに保存し、データカットオフおよびPFP検証対象。Place対応は専用バンドとOOS検証を行う別マイルストーンで扱うべき。

---

## Claude's Discretion

- TrackConditionFetcherProtocol の具体的なメソッドシグネチャ
- HTML パーサーの DOM クエリ戦略
- FeatureBuilder のライブ値優先マージの実装詳細
- JRA/CSV照合アルゴリズムの具体的な実装
- OddsBandFilter 校正データのPT用永続化フォーマット
- DD/Regime shadow記録の診断ログフォーマット

## Deferred Ideas

- Regime動的化 — 別マイルストーン(WF検証でTurf CONSERVATIVE有効性確認後)
- Place OddsBandFilter — 別マイルストーン(専用バンド+OOS検証)
- DD制御有効化 — 実運用移行時(PT shadow記録で実績確認後)
