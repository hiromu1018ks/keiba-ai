# Phase 53: Strategy Alignment & Live Data - Context

**Gathered:** 2026-06-06
**Status:** Ready for planning

<domain>
## Phase Boundary

PT が BT で検証済みの戦略パラメータを適用して推論を実行し、当日のトラック条件データを取得して特徴量に反映できること。

具体的には:
1. **戦略完全整合 (STR-01~06)** — manifest/PFP、betting_target/mode、DD制御(shadow)、OddsBandFilter、QualityScreener、regime(AGGRESSIVE固定)をBTと同一設定契約でPTに統合
2. **当日データ取得 (LIV-01~03)** — JRA公式サイトから芝クッション値・ダート含水率をPlaywrightで取得し、FeatureBuilderに統合

**v2.4対象は Win/Place のみ。Wide は拒否する。**

</domain>

<decisions>
## Implementation Decisions

### Regime統一方針 (STR-06)

- **D-01:** BT/PTともにベット判断はAGGRESSIVE固定。RegimeDetectorの推定結果は診断ログへshadow記録するが、閾値・Kelly・ベット停止には反映しない。動的化はTurf CONSERVATIVEを含むWF検証で有効性を確認した別マイルストーンで扱う。BT/PTの両方にある`TODO: Regime動的に戻す場合はコメントアウト解除`コメントはそのまま残す。

### Manifest注入アーキテクチャ (STR-01/02/03)

- **D-02:** `run_paper_trading.py`をcomposition rootとし、既存`build_strategy_config_from_params()`をBT/PTで共有。manifestは起動時に一度だけ検証・読み込み → immutableな戦略設定へ変換。PaperPredictorはファイルI/Oを行わず、構築済みRacePredictorを受け取る。manifest path・SHA256・適用パラメータをsession_manifest.jsonに保存しPFP対象とする。AGGRESSIVE固定のためregime overrideは記録のみ(適用しない)。OddsBandFilterはmanifest値だけでなく校正済み状態を明示的に注入する。新規StrategyConfiguratorクラスは不要。
- **D-03:** `--betting-target` と `--betting-mode` は必須引数(暗黙デフォルトなし)。Wideは引数解析時に拒否する。ロードしたモデルの学習target・strategy manifestの対応target・CLI targetが一致しなければfail-fastする。mode/targetはsession_manifestとPFP対象に含める。
- **D-04:** PTではDDControllerによるstake縮小・新規ベット停止を適用しない。DD状態を計算して診断ログにshadow記録するのみ。PTのkellyはKelly計算によるstake変更のみ行い、DD補正は適用しない。BTとのROI比較ではDD制御なし条件のBTと比較する。実運用移行時にDD制御を有効化する。

### JRAライブデータ取得 (LIV-01/02/03)

- **D-05:** PlaywrightでHTML取得のみ担当。解析処理は保存HTMLを入力とする純粋関数に分離。`TrackConditionFetcherProtocol`を定義してProtocol-based DI。取得HTMLを保存しfixtureでパーサーテスト。HTML構造変更検知を実装。取得失敗時は古い値へフォールバックせず予測を停止(非ゼロ終了)。
- **D-06:** 含水率の集約規則は即決しない。重複期間のJRA値とCSV値を照合して規則を確定する。ライブ取得では両地点の生値を保存し、検証済み規則からdirt_moistureを算出する。照合不能なら予測を停止する。
- **D-07:** ライブ生値はセッション配下へimmutableに保存し、正規化済みDataFrameをFeatureBuilder.build_for_inference()へ明示的に渡す。FeatureBuilderは対象日のライブ値を履歴Parquetより優先してマージする。取得元・測定時刻・取得時刻・raw HTML hashをsession_manifestに記録する。履歴Parquetへの反映は検証後の別ETL処理とする。

### OddsBandFilter適用範囲 (STR-04)

- **D-08:** PTでも`betting_target=win`の場合のみBTと同じ校正済みOddsBandFilterを適用。placeでは生成・適用しない。校正データ終了日・ROI閾値・除外バンド・設定hashをモデル成果物とsession_manifestに保存し、データカットオフおよびPFP検証対象とする。Place OddsBandFilter対応は専用バンド定義とOOS検証を行う別マイルストーンで扱う。

### Claude's Discretion

- TrackConditionFetcherProtocol の具体的なメソッドシグネチャ
- HTML パーサーの DOM クエリ戦略とJRAサイトのHTML構造の解釈
- FeatureBuilder のライブ値優先マージの実装詳細(mergeキー、NaN取扱)
- JRA/CSV照合による集約規則確定の具体的なアルゴリズム
- OddsBandFilter 校正データのPT用永続化フォーマット
- DD shadow記録の診断ログフォーマット
- RegimeDetector shadow記録の診断ログフォーマット

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — STR-01~06, LIV-01~03 の要件定義。Traceability table あり
- `.planning/ROADMAP.md` §Phase 53 — Goal, Success Criteria, Requirements mapping
- `.planning/PROJECT.md` — v2.4 milestone context, Out of Scope 定義
- `.planning/STATE.md` — Phase 51/52 deliverables, deferred items

### Prior Phase Context (MUST read — decisions cascade)
- `.planning/phases/51-settlement-integrity-training-pipeline/51-CONTEXT.md` — bet_id, 3列状態モデル, ModelLoader優先度, Wide拒否
- `.planning/phases/52-shared-feature-builder-consistency/52-CONTEXT.md` — FeatureBuilder, FeatureState, DataCutoffManifest, PFPVerifier, session_manifest

### Strategy & Betting Components (must-read for integration)
- `src/betting/default_strategy.py` — `build_strategy_config_from_params()`: manifest → 戦略設定変換。PTでも共有
- `src/betting/drawdown_controller.py` — DDController: DD%3段階制御。PTではshadow記録のみ(D-04)
- `src/betting/odds_band_filter.py` — OddsBandFilter: オッズバンド別ROI除外。win-targetのみ(D-08)
- `src/models/regime_detector.py` — RegimeDetector: 3状態検出。detect()はshadow記録のみ(D-01)
- `src/models/race_quality_screener.py` — RaceQualityScreener: 既にBT/PT同一動作済み

### PT Pipeline (must-read for injection points)
- `src/paper_trading/predictor.py` — PaperPredictor: 構築済みRacePredictorを受け取る設計に変更(D-02)
- `scripts/run_paper_trading.py` — PT CLI: composition root。--betting-target/--betting-mode必須引数追加(D-03)
- `src/backtest/race_predictor.py` — RacePredictor: stake_calculator/dd_controller注入箇所。regime AGGRESSIVE固定箇所(1124-1131, 1265-1267)

### BT Reference (same-pattern verification)
- `src/backtest/engine.py` lines 366-440 — BacktestEngine.__init__(): manifest読込・DD/StakeCalculator/Regime注入パターン
- `src/backtest/engine.py` lines 487-514 — _calibrate_odds_band_filter(): 校正プロセス
- `scripts/run_backtest.py` lines 81-109, 202-227 — BT CLI: --betting-target/--betting-mode/--strategy-manifest

### Live Data & Scraping (must-read for implementation)
- `scripts/scrape_everydb2_manual.py` — 既存Playwrightパターン: sync_api, browser management, DOM queries, rate limiting
- `src/ingestion/odds_collector.py` — OddsFetcherProtocol + OddsCollector: Protocol-based DIパターン参考
- `src/ingestion/jvlink_fetcher.py` — JVLinkFetcher: 現在の具象実装(ParquetStore読込)

### Track Condition Data Pipeline (must-read for integration)
- `src/features/track_condition_data.py` — convert_track_conditions(): CSV→Parquet変換、aggregate_to_race_level()集約
- `src/features/track_condition_features.py` — compute_track_condition_features(): 23特徴量生成
- `src/features/feature_builder.py` lines 340-366 — FeatureBuilder内のTC特徴量統合箇所。ライブ値優先マージの注入点(D-07)
- `scripts/precompute_track_condition.py` — ETLスクリプト: CSV入力のパース・検証パターン
- `data/raw/track_conditions.parquet` — 履歴トラック条件データ(正本)

### Verification Infrastructure (reuse patterns)
- `src/backtest/parameter_freeze_protocol.py` — ParameterFreezeProtocol: SHA256 manifest検証
- `src/features/data_cutoff_manifest.py` — DataCutoffManifest: 二段階データカットオフ検証
- `src/features/feature_builder.py` — FeatureBuilder: FeatureState, FeatureBuildResult

### Configuration
- `config/settings.yaml` — DB接続、データパス、feature_engine設定
- `data/strategy_manifest.json` — Optuna最適化済み戦略パラメータ(PT注入対象)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_strategy_config_from_params()` (default_strategy.py): manifest → DDConfig/StakeCalculator/RegimeOverrides変換。PTでもそのまま利用(D-02)
- `OddsFetcherProtocol` + `OddsCollector` (odds_collector.py): Protocol-based DIパターン。TrackConditionFetcherProtocolの設計参考(D-05)
- `ParameterFreezeProtocol` (parameter_freeze_protocol.py): SHA256 manifest検証。PT strategy_params検証に再利用
- `DataCutoffManifest` (data_cutoff_manifest.py): Phase 52で実装済み。strategy_manifest日付検証は既に動作中
- `scrape_everydb2_manual.py`: Playwrightスクレイピングパターン。create_browser(), page.goto(), DOM query, rate limiting(D-05)
- `convert_track_conditions()` + `aggregate_to_race_level()` (track_condition_data.py): 既存のrace-level集約ロジック。照合検証で再利用(D-06)
- `FeatureBuilder.build_for_inference()` (feature_builder.py): ライブDataFrameの注入点。in-memory merge実装先(D-07)

### Established Patterns
- **Protocol-based DI for fetchers**: OddsFetcherProtocol → TrackConditionFetcherProtocol。テスト時モック可能
- **Composition root at CLI layer**: BTのrun_backtest.pyと同じパターンをPTでも採用(D-02)
- **Fail-fast on target mismatch**: モデル/manifest/CLI targetの3者一致検証(D-03)
- **Shadow mode for diagnostic components**: DD/Regimeは計算するがベット判断に反映しない(D-01, D-04)
- **Immutable session artifacts**: 生値・metadataをセッション配下に保存、正本は変更しない(D-07)
- **Pure function separation for parsing**: HTML取得(Playwright)と解析(純粋関数)を分離(D-05)

### Integration Points
- `scripts/run_paper_trading.py` main(): --betting-target/--betting-mode/--strategy-manifest引数追加、RacePredictor構築にstrategy_params注入
- `scripts/run_paper_trading.py` _run_predict(): PaperPredictor構築済みRacePredictor受け取りに変更
- `src/paper_trading/predictor.py` setup(): ファイルI/O削除、RacePredictorをコンストラクタ引数に
- `src/backtest/race_predictor.py` select_bets()/get_place_candidates(): regime AGGRESSIVE固定を明示化(D-01反映)
- `src/features/feature_builder.py` build_for_inference(): ライブtrack_condition DataFrameのmergeロジック追加(D-07)
- セッション配下: ライブ生値HTML + 正規化DataFrame + session_manifest拡張(D-05, D-07)

</code_context>

<specifics>
## Specific Ideas

- manifestは起動時に一度だけ検証・読み込みし、immutableな戦略設定へ変換する。レースループ内で再読込しない
- PTのkellyはKelly stake変更のみでDD補正なし。BT比較時はDD制御なし条件のBT結果と比較する必要がある
- ライブ取得値を履歴Parquetへ直接追記しない。不完全取得や再実行で正本汚染の危険がある
- 履歴CSVのdirt_moistureの由来(ゴール前/4コーナー/平均)が不明なため、集約規則は重複期間照合で確定する
- OddsBandFilterの校正済み状態はmanifest値とともに明示的に注入する。BTでのみ再校正はしない
- Place OddsBandFilterは専用バンド定義(1.0-2.0, 2.0-5.0等)とOOS検証が必要なため別マイルストーン
- RegimeDetector shadow記録は診断ログの拡張として実装。RacePredictorの戻り値にregime情報を追加

</specifics>

<deferred>
## Deferred Ideas

- Regime動的化 — Turf CONSERVATIVEのマイナスROI問題をWF検証で解決した別マイルストーンで扱う
- Place OddsBandFilter — 専用バンド定義とOOS検証が必要。別マイルストーン
- DD制御有効化 — 実運用移行時。PT shadow記録で有効性を実績確認後
- One-command run mode — Phase 54 (AUT-01~03)
- Weekly/cumulative reporting — Phase 54 (RPT-01~04)
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+ (WRN-01)
- Wide bet support — v2.5+ (WID-01, WID-02)
- SafetyGuard integration — v2.5+ (SAF-01, SAF-02)

</deferred>

---

*Phase: 53-Strategy Alignment & Live Data*
*Context gathered: 2026-06-06*
