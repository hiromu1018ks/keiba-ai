# Phase 53: Strategy Alignment & Live Data - Research

**Researched:** 2026-06-06
**Domain:** PT strategy parameter injection + JRA live track condition scraping
**Confidence:** HIGH

## Summary

Phase 53 は2つの独立した技術領域を統合する。(1) PT パイプラインに BT で検証済みの戦略パラメータ (manifest/PFP/DD/OddsBandFilter/Regime) を注入し、BT/PT の同一設定契約を確立する。(2) JRA 公式サイトから当日の芝クッション値・ダート含水率を Playwright で取得し、FeatureBuilder の推論パイプラインに反映する。

戦略注入は既存の `build_strategy_config_from_params()` と BT の `BacktestEngine.__init__()` パターンをそのまま PT の composition root (`run_paper_trading.py`) に適用する。新規クラスは不要。DD と Regime は shadow 記録のみでベット判断には反映しない。

JRA スクレイピングは既存の `scrape_everydb2_manual.py` の Playwright パターンを踏襲し、HTML 取得 (Playwright) と解析 (純粋関数) を完全分離する。JRA ページの HTML 構造を実際に確認し、クッション値 (`#cushion_num strong`) と含水率テーブル (`#turf_line` / `#dirt_line` の `.gm` / `.c4` セル) のセレクタを特定済み。

**Primary recommendation:** PT は `run_paper_trading.py` を composition root として manifest 読込→RacePredictor 構築→PaperPredictor 呼出の流れを実装。JRA データは TrackConditionFetcherProtocol で抽象化し、取得 HTML を保存して純粋関数パーサーで解析。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** BT/PTともにベット判断はAGGRESSIVE固定。RegimeDetectorの推定結果は診断ログへshadow記録するが、閾値・Kelly・ベット停止には反映しない。
- **D-02:** `run_paper_trading.py`をcomposition rootとし、既存`build_strategy_config_from_params()`をBT/PTで共有。manifestは起動時に一度だけ検証・読み込み → immutableな戦略設定へ変換。PaperPredictorはファイルI/Oを行わず、構築済みRacePredictorを受け取る。新規StrategyConfiguratorクラスは不要。
- **D-03:** `--betting-target` と `--betting-mode` は必須引数(暗黙デフォルトなし)。Wideは引数解析時に拒否する。ロードしたモデルの学習target・strategy manifestの対応target・CLI targetが一致しなければfail-fastする。
- **D-04:** PTではDDControllerによるstake縮小・新規ベット停止を適用しない。DD状態を計算して診断ログにshadow記録するのみ。PTのkellyはKelly計算によるstake変更のみ行い、DD補正は適用しない。
- **D-05:** PlaywrightでHTML取得のみ担当。解析処理は保存HTMLを入力とする純粋関数に分離。`TrackConditionFetcherProtocol`を定義してProtocol-based DI。取得HTMLを保存しfixtureでパーサーテスト。HTML構造変更検知を実装。取得失敗時は古い値へフォールバックせず予測を停止(非ゼロ終了)。
- **D-06:** 含水率の集約規則は即決しない。重複期間のJRA値とCSV値を照合して規則を確定する。ライブ取得では両地点の生値を保存し、検証済み規則からdirt_moistureを算出する。照合不能なら予測を停止する。
- **D-07:** ライブ生値はセッション配下へimmutableに保存し、正規化済みDataFrameをFeatureBuilder.build_for_inference()へ明示的に渡す。FeatureBuilderは対象日のライブ値を履歴Parquetより優先してマージする。取得元・測定時刻・取得時刻・raw HTML hashをsession_manifestに記録する。履歴Parquetへの反映は検証後の別ETL処理とする。
- **D-08:** PTでも`betting_target=win`の場合のみBTと同じ校正済みOddsBandFilterを適用。placeでは生成・適用しない。校正データ終了日・ROI閾値・除外バンド・設定hashをモデル成果物とsession_manifestに保存し、データカットオフおよびPFP検証対象とする。

### Claude's Discretion
- TrackConditionFetcherProtocol の具体的なメソッドシグネチャ
- HTML パーサーの DOM クエリ戦略とJRAサイトのHTML構造の解釈
- FeatureBuilder のライブ値優先マージの実装詳細(mergeキー、NaN取扱)
- JRA/CSV照合による集約規則確定の具体的なアルゴリズム
- OddsBandFilter 校正データのPT用永続化フォーマット
- DD shadow記録の診断ログフォーマット
- RegimeDetector shadow記録の診断ログフォーマット

### Deferred Ideas (OUT OF SCOPE)
- Regime動的化 — Turf CONSERVATIVEのマイナスROI問題をWF検証で解決した別マイルストーンで扱う
- Place OddsBandFilter — 専用バンド定義とOOS検証が必要。別マイルストーン
- DD制御有効化 — 実運用移行時。PT shadow記録で有効性を実績確認後
- One-command run mode — Phase 54 (AUT-01~03)
- Weekly/cumulative reporting — Phase 54 (RPT-01~04)
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+ (WRN-01)
- Wide bet support — v2.5+ (WID-01, WID-02)
- SafetyGuard integration — v2.5+ (SAF-01, SAF-02)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STR-01 | PT で strategy_manifest を読み込み manifest/PFP を適用 | `build_strategy_config_from_params()` + `verify_strategy_manifest()` をPT composition rootで共有 (D-02)。PFPVerifierにstrategy_paramsを含めて拡張 |
| STR-02 | PT で `--betting-target win\|place` と `--betting-mode flat\|kelly` を指定可能 | parse_args() に必須引数追加。Wide拒否 + 3者一致検証 (D-03) |
| STR-03 | DrawdownController を PT パイプラインに組み込む | DDControllerはshadow記録のみ(D-04)。RacePredictor構築時に注入するがPTループ内ではget_multiplier()をベットに反映しない |
| STR-04 | BT の校正済み OddsBandFilter を PT で使用する | win-targetのみ適用(D-08)。校正済み状態(除外バンド+閾値)をmanifest/モデル成果物から注入。新規の再校正は行わない |
| STR-05 | RaceQualityScreener を PT パイプラインに組み込む | 既にRacePredictor.should_bet()で統合済み。変更不要 |
| STR-06 | BT/PT の regime 検出を統一 | AGGRESSIVE固定(D-01)。detect()はshadow記録。TODOコメントを残す |
| LIV-01 | JRA公式サイトから芝クッション値・ダート含水率を取得 | Playwrightで取得→純粋関数パーサー(D-05)。DOM構造特定済み |
| LIV-02 | 取得値・測定時刻・取得時刻・取得元を保存。失敗時予測停止 | session_manifest拡張 + セッション配下に生HTML保存(D-05, D-07) |
| LIV-03 | 過去 CSV と当日取得値は同一スキーマ・同一集約規則 | 含水率集約規則はJRA/CSV照合で確定(D-06)。FeatureBuilderにライブ値優先マージ追加(D-07) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Strategy manifest loading & PFP verification | CLI / Composition Root | — | D-02: run_paper_trading.py がcomposition root |
| RacePredictor construction with strategy params | CLI / Composition Root | — | BacktestEngine.__init__() と同じパターン |
| DD state calculation (shadow) | RacePredictor | DiagnosticLogger | D-04: 計算するがベット判断に反映しない |
| Regime detection (shadow) | RacePredictor | DiagnosticLogger | D-01: detect()は診断ログのみ |
| OddsBandFilter injection | CLI / Composition Root | — | D-08: 校正済み状態を明示的に注入 |
| JRA HTML scraping | Ingestion Layer (Playwright) | — | D-05: TrackConditionFetcherProtocol |
| HTML parsing (pure function) | Ingestion Layer | — | D-05: 純粋関数、Playwright非依存 |
| Live value normalization & aggregation | Feature Layer | — | D-07: FeatureBuilder.build_for_inference() |
| Live/historical merge | Feature Layer | — | D-07: ライブ値優先マージ |
| Session manifest recording | CLI / Composition Root | — | D-09: session_manifest.json 拡張 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| playwright (sync_api) | 1.58.0 | JRA HTML取得 | 既存scrape_everydb2_manual.pyと同じ。pip install済み [VERIFIED: pip show] |
| lightgbm | (existing) | RegimeDetector推論 | モデル依存関係そのまま |
| pandas | (existing) | DataFrame操作 | プロジェクト標準 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib | (stdlib) | HTML構造変更検知・PFP | D-05: raw HTML hashで構造変更を検知 |
| dataclasses | (stdlib) | TrackConditionFetcherProtocol | D-05: Protocol-based DI |
| typing.Protocol | (stdlib) | Fetcher抽象化 | D-05: OddsFetcherProtocolと同じパターン |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Playwright | requests+BeautifulSoup | JRAはJS描納なしでも動作するが、既存Playwrightパターンの方が将来のJS描納変更に強い |
| Protocol-based DI | ABC | Protocolは構造的型付けでテスト時モックが容易。既存OddsFetcherProtocolと統一 |

**Installation:**
```bash
# No new packages needed — playwright 1.58.0 already installed
# Playwright browsers already installed (verified via scrape_everydb2_manual.py)
```

## Package Legitimacy Audit

> No new external packages are installed in this phase. All dependencies (playwright, lightgbm, pandas) are pre-existing project dependencies verified in prior phases.

| Package | Registry | Status |
|---------|----------|--------|
| playwright | pip | Pre-existing (1.58.0) [VERIFIED: pip show] |
| lightgbm | pip | Pre-existing |
| pandas | pip | Pre-existing |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
run_paper_trading.py (Composition Root)
  |
  +-- parse_args() → --betting-target (required) + --betting-mode (required) + --strategy-manifest
  |
  +-- [Wide rejection] → argparse choices に "wide" 含めず fail-fast
  |
  +-- _load_strategy_params() → verify_strategy_manifest() → immutable params dict
  |     |
  |     +-- build_strategy_config_from_params(params) → {dd_config, regime_overrides, ...}
  |
  +-- 3-way target match: model.betting_target == manifest.target == CLI target → fail-fast
  |
  +-- RacePredictor construction (BT pattern):
  |     if kelly:
  |       StakeCalculator(fractional_kelly, target_ev, max_scale)
  |       DDController(peak_bankroll, dd_cfg) → shadow only (D-04)
  |     RacePredictor(models, stake_calculator=..., dd_controller=..., betting_target=...)
  |
  +-- OddsBandFilter injection (win only, D-08):
  |     if betting_target == "win":
  |       load calibrated bands from model artifacts / manifest
  |       inject into RacePredictor
  |
  +-- TrackConditionFetcher (JRA live data, D-05):
  |     |
  |     +-- JRATrackConditionFetcher (Playwright) → raw HTML
  |     |
  |     +-- parse_track_condition_html(html) → {venue: {turf_cushion, dirt_moisture_goal, dirt_moisture_4c, measured_at}}
  |     |
  |     +-- Normalize to DataFrame (race_id level)
  |     |
  |     +-- Save raw HTML + normalized DF to session dir (D-07)
  |
  +-- FeatureBuilder.build_for_inference(... live_tc_df=live_df):
  |     Merge live values over historical Parquet (D-07)
  |
  +-- PaperPredictor (receives constructed RacePredictor, no file I/O)
  |     |
  |     +-- Race loop:
  |           predict() → select_bets() [AGGRESSIVE fixed regime, D-01]
  |           DD shadow: log_state() (D-04)
  |           Regime shadow: detect() → log (D-01)
  |           OddsBandFilter: filter() (win only, D-08)
  |           DiagnosticLogger: race/horse records
  |
  +-- Session manifest update (PFP verify, strategy params, live data metadata)
```

### Recommended Project Structure
```
src/
├── ingestion/
│   ├── track_condition_fetcher.py   # TrackConditionFetcherProtocol + JRA実装 + パーサー
│   ├── odds_collector.py            # 既存: OddsFetcherProtocol (参考パターン)
│   └── jvlink_fetcher.py            # 既存: 変更なし
├── paper_trading/
│   ├── predictor.py                 # 変更: RacePredictor受け取り、ファイルI/O削除
│   └── config.py                    # 変更なし
├── backtest/
│   ├── race_predictor.py            # 変更: OddsBandFilter注入ポイント追加
│   ├── engine.py                    # 参照のみ (BT注入パターン)
│   └── parameter_freeze_protocol.py # 変更なし (manifest検証再利用)
├── betting/
│   ├── default_strategy.py          # 変更なし (PTでも共有)
│   ├── drawdown_controller.py       # 変更なし
│   └── odds_band_filter.py          # 変更なし
├── features/
│   ├── feature_builder.py           # 変更: build_for_inference() にライブ値マージ追加
│   ├── track_condition_features.py  # 変更なし
│   └── track_condition_data.py      # 参照: aggregate_to_race_level() 再利用
├── features/
│   ├── session_manifest.py          # 拡張: live data metadata フィールド追加
│   └── data_cutoff_manifest.py      # 変更なし
scripts/
├── run_paper_trading.py             # 変更: 必須引数、composition root、manifest注入
└── scrape_everydb2_manual.py        # 参照のみ (Playwrightパターン)
```

### Pattern 1: Composition Root Strategy Injection
**What:** CLI スクリプトで manifest→strategy_params→RacePredictor 構築を一括管理
**When to use:** BacktestEngine.__init__() と同一パターンを PT でも適用
**Example:**
```python
# run_paper_trading.py (composition root)
from betting.default_strategy import build_strategy_config_from_params
from backtest.parameter_freeze_protocol import verify_strategy_manifest

# D-02: manifest読込→immutable変換
params = verify_strategy_manifest(manifest_path)
strategy_config = build_strategy_config_from_params(params)

# BTと同じRacePredictor構築パターン (engine.py L415-440)
if betting_mode == "kelly":
    dd_cfg = strategy_config.get("dd_config", DDConfig())
    stake_calc = StakeCalculator(
        fractional_kelly=strategy_config.get("fractional_kelly", 0.5),
        target_ev=strategy_config.get("target_ev", 1.10),
        max_scale=strategy_config.get("max_scale", 2.0),
    )
    dd_ctrl = DrawdownController(peak_bankroll=initial_bankroll, cfg=dd_cfg)
    race_predictor = RacePredictor(
        models, stake_calculator=stake_calc,
        dd_controller=dd_ctrl, betting_target=betting_target,
    )
else:
    race_predictor = RacePredictor(models, betting_target=betting_target)
```

### Pattern 2: Protocol-based Fetcher with Pure Function Parser
**What:** Playwright 取得と HTML 解析を完全分離
**When to use:** OddsFetcherProtocol と同じ DI パターン
**Example:**
```python
# src/ingestion/track_condition_fetcher.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class TrackConditionFetcherProtocol(Protocol):
    def fetch_track_conditions_html(self, venue_code: str) -> str: ...

# 純粋関数パーサー (Playwright非依存、テスト容易)
def parse_track_condition_html(html: str) -> dict[str, Any]:
    """JRA馬場情報ページHTMLからクッション値・含水率を抽出"""
    from bs4 import BeautifulSoup
    # DOM query: #cushion_num strong, #turf_line .gm, #dirt_line .gm etc.
    ...

# Playwright実装
class JRATrackConditionFetcher:
    BASE_URL = "https://www.jra.go.jp/keiba/baba/"

    def fetch_track_conditions_html(self, venue_code: str) -> str:
        # Playwright page.goto() → page.content()
        ...
```

### Pattern 3: Live Value Priority Merge in FeatureBuilder
**What:** FeatureBuilder.build_for_inference() でライブ値を履歴Parquetより優先マージ
**When to use:** D-07: FeatureBuilder にライブ DataFrame を明示的に渡す
**Example:**
```python
def build_for_inference(
    self, race_df, entry_df, odds_df, feature_state, *,
    live_track_conditions: pd.DataFrame | None = None,  # 新規引数
):
    result = self._build(...)
    # D-07: ライブ値優先マージ
    if live_track_conditions is not None and not live_track_conditions.empty:
        result.frame = self._merge_live_track_conditions(result.frame, live_track_conditions)
    return result

def _merge_live_track_conditions(self, df, live_df):
    """ライブ値を履歴より優先してマージ (D-07)"""
    # mergeキー: race_id
    # 既存の dirt_moisture/turf_cushion をライブ値で上書き
    # NaN取扱: ライブ値がNaNなら履歴値を保持
    ...
```

### Anti-Patterns to Avoid
- **PT で OddsBandFilter を再校正する** — BT の校正済み状態を注入する。PT で再校正するとデータリーク (D-08)
- **DD乗数をPTベットに反映する** — shadow記録のみ。stake調整はBTとの比較可能性を損なう (D-04)
- **Regime検出結果でベット判断を変える** — AGGRESSIVE固定。動的化は別マイルストーン (D-01)
- **ライブ取得値を履歴Parquetに直接追記する** — 不完全取得や再実行で正本汚染の危険 (D-07)
- **取得失敗時に古い値へフォールバックする** — 予測停止(非ゼロ終了)が正しい動作 (D-05)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Strategy params → config conversion | 新規StrategyConfigurator | `build_strategy_config_from_params()` | BT/PTで既に共有済み (D-02) |
| Manifest verification | 独自SHA256検証 | `verify_strategy_manifest()` | ParameterFreezeProtocolの一部 |
| Track condition aggregation | 新規集約関数 | `aggregate_to_race_level()` | track_condition_data.pyに既存 |
| Protocol-based DI pattern | 新規ABC | `typing.Protocol + @runtime_checkable` | OddsFetcherProtocolと統一 |
| Browser management | 独自Playwright管理 | `create_browser()` パターン | scrape_everydb2_manual.pyから再利用 |

## Common Pitfalls

### Pitfall 1: 3-way target mismatch silent failure
**What goes wrong:** モデルのbetting_target と manifest の target と CLI の target が一致しない状態で実行される
**Why it happens:** 各コンポーネントが独立に target を管理している
**How to avoid:** composition root で3者一致を起動時に検証。不一致時は即座に fail-fast (D-03)
**Warning signs:** モデルがplace学習でwin予測、またはその逆

### Pitfall 2: OddsBandFilter calibration data leakage
**What goes wrong:** PT で OddsBandFilter を再校正すると、予測日以降のオッズデータが校正に含まれる
**Why it happens:** BT の `_calibrate_odds_band_filter()` をそのままPTで呼ぶと、レースループ内のオッズが使われる
**How to avoid:** 校正済みOddsBandFilterの状態(除外バンド+閾値)をBT成果物からロードして注入。PT での再校正は禁止 (D-08)
**Warning signs:** OddsBandFilter除外バンドがBTと異なる

### Pitfall 3: JRA page HTML structure change
**What goes wrong:** JRA がサイトリニューアルでDOM構造を変更し、パーサーが黙って空値を返す
**Why it happens:** スクレイピングはHTML構造に強く依存する
**How to avoid:** HTML hashで構造変更を検知(D-05)。必須DOM要素が見つからない場合は例外→予測停止。パーサーテストに保存HTML fixtureを使用
**Warning signs:** パーサーが空dictを返す、または期待しない値

### Pitfall 4: DD shadow vs actual confusion
**What goes wrong:** DD乗数をPTベットに反映してしまうと、BT比較時にDD制御あり/なしの差が分からなくなる
**Why it happens:** RacePredictorにDDControllerを注入すると、select_bets()内で自動的に使われる
**How to avoid:** PT用RacePredictorではDDControllerを注入するが、select_bets()でstakeに反映しないフラグを設ける。またはPTループ内で明示的にstakeを上書き (D-04)
**Warning signs:** PT stake がBT stake と異なる

### Pitfall 5: Dirt moisture aggregation rule ambiguity
**What goes wrong:** ゴール前と4コーナーの2地点の含水率をどうdirt_moistureに集約するか不明
**Why it happens:** 履歴CSV の dirt_moisture の由来(ゴール前/4コーナー/平均)が不明
**How to avoid:** 重複期間のJRA値とCSV値を照合して規則を確定(D-06)。照合不能なら予測停止
**Warning signs:** 集約後のdirt_moistureが履歴CSVと系統的にズレる

### Pitfall 6: Multiple venue race_id mapping
**What goes wrong:** JRAページは開催場ごとに別ページ。race_idへのマッピングを間違える
**Why it happens:** 東京/京都/新潟など同日複数開催。venue_code→jyocdの対応が必要
**How to avoid:** JRAの競馬場コード(01-10)とrace_idのjyocd(9-10桁目)を対応づけ。各開催場のページを個別に取得
**Warning signs:** 芝クッション値がダートレースに適用される

## Code Examples

### JRA Track Condition HTML Parsing
```python
# Source: 実際のJRAページHTML構造 (2026-06-06 取得確認)
from bs4 import BeautifulSoup

def parse_track_condition_html(html: str) -> dict[str, dict[str, Any]]:
    """JRA馬場情報ページHTMLからクッション値・含水率を抽出する純粋関数。

    DOM構造 (確認済み):
      クッション値: <div id="cushion_num"><p><strong>9.9</strong></p></div>
      含水率テーブル:
        <tr id="turf_line"><th>芝</th><td class="gm">16.2%</td><td class="c4">15.7%</td></tr>
        <tr id="dirt_line"><th>ダート</th><td class="gm">7.7%</td><td class="c4">8.9%</td></tr>
      測定時刻: <select id="moist_list"><option selected>6月6日（土曜）5時00分</option></select>
      クッション測定時刻: <select id="cushion_list"><option selected>...</option></select>

    Returns:
        {
            "turf_cushion": float | None,
            "dirt_moisture_goal": float | None,   # ゴール前 (%)
            "dirt_moisture_4c": float | None,      # 4コーナー (%)
            "turf_moisture_goal": float | None,
            "turf_moisture_4c": float | None,
            "measured_at_moist": str,               # 含水率測定時刻
            "measured_at_cushion": str,             # クッション値測定時刻
        }
    """
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, Any] = {
        "turf_cushion": None,
        "dirt_moisture_goal": None,
        "dirt_moisture_4c": None,
        "turf_moisture_goal": None,
        "turf_moisture_4c": None,
        "measured_at_moist": "",
        "measured_at_cushion": "",
    }

    # クッション値
    cushion_el = soup.select_one("#cushion_num strong")
    if cushion_el:
        try:
            result["turf_cushion"] = float(cushion_el.get_text(strip=True))
        except ValueError:
            pass

    # 含水率 (ゴール前 / 4コーナー)
    for row_id, prefix in [("turf_line", "turf"), ("dirt_line", "dirt")]:
        row = soup.select_one(f"#{row_id}")
        if row:
            gm = row.select_one(".gm")
            c4 = row.select_one(".c4")
            if gm:
                result[f"{prefix}_moisture_goal"] = _parse_percent(gm.get_text(strip=True))
            if c4:
                result[f"{prefix}_moisture_4c"] = _parse_percent(c4.get_text(strip=True))

    # 測定時刻
    moist_select = soup.select_one("#moist_list option[selected]")
    if moist_select:
        result["measured_at_moist"] = moist_select.get_text(strip=True)

    cushion_select = soup.select_one("#cushion_list option[selected]")
    if cushion_select:
        result["measured_at_cushion"] = cushion_select.get_text(strip=True)

    return result

def _parse_percent(text: str) -> float | None:
    """'16.2%' → 16.2"""
    text = text.replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None
```

### Venue Code to race_id Mapping
```python
# JRA競馬場コード → race_id jyocd (9-10桁目)
# Source: CLAUDE.md _venue_map + EveryDB2 コード表 2001
JRA_VENUE_CODES: dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

# JRAサイトの競馬場URLセレクタ → venue code
# https://www.jra.go.jp/keiba/baba/ のタブ/リンクから取得
# 実際のHTML: 各開催場のタブにvenue codeが含まれる
```

### PT Strategy Injection (run_paper_trading.py)
```python
# Source: BT engine.py L366-440 パターンをPTに適用
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Trading")
    # ... existing args ...
    parser.add_argument(
        "--betting-target", required=True,
        choices=["win", "place"],  # Wide拒否 (D-03)
        help="ベッティング対象 (必須)",
    )
    parser.add_argument(
        "--betting-mode", required=True,
        choices=["flat", "kelly"],
        help="ベット額計算モード (必須)",
    )
    parser.add_argument(
        "--strategy-manifest", type=str, default=None,
        help="Strategy manifest JSON path",
    )
    return parser.parse_args()

def _build_race_predictor(
    models, args, strategy_config, initial_bankroll,
) -> RacePredictor:
    """BT engine.py L415-440 と同一パターン"""
    betting_target = args.betting_target
    betting_mode = args.betting_mode

    # D-03: 3者一致検証
    model_target = getattr(models, "betting_target", None)
    manifest_target = strategy_config.get("_betting_target") if strategy_config else None
    for t in [model_target, manifest_target]:
        if t is not None and t != betting_target:
            logger.error("Target mismatch: model=%s manifest=%s CLI=%s",
                         model_target, manifest_target, betting_target)
            sys.exit(1)

    if betting_mode == "kelly":
        dd_cfg = strategy_config.get("dd_config", DDConfig())
        stake_calc = StakeCalculator(
            fractional_kelly=strategy_config.get("fractional_kelly", 0.5),
            target_ev=strategy_config.get("target_ev", 1.10),
            max_scale=strategy_config.get("max_scale", 2.0),
        )
        dd_ctrl = DrawdownController(
            peak_bankroll=initial_bankroll, cfg=dd_cfg,
        )
        return RacePredictor(
            models, stake_calculator=stake_calc,
            dd_controller=dd_ctrl, betting_target=betting_target,
        )
    else:
        return RacePredictor(models, betting_target=betting_target)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PT 固定パラメータ | Strategy manifest注入 | Phase 53 | BT/PT同一設定契約 |
| Regime動的(3状態) | AGGRESSIVE固定 | Phase 53 | Turf保守的マイナス回避 |
| DD制御あり | DD shadow記録のみ | Phase 53 | BT/PT比較可能性確保 |
| 履歴CSVのみ | JRAライブ取得 | Phase 53 | 当日馬場状態のリアルタイム反映 |

**Deprecated/outdated:**
- `PaperPredictor` のファイルI/Oベース初期化: RacePredictor受け取りに変更 (D-02)
- PT でのデフォルト戦略パラメータ: manifest必須に変更 (D-02)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | JRAページのDOM構造が `#cushion_num strong` / `#turf_line .gm` / `#dirt_line .gm` である | Code Examples | 2026-06-06に実際にHTMLを取得して確認済み。JRAが構造変更した場合は構造変更検知で対処 |
| A2 | 履歴CSVのdirt_moistureはゴール前と4コーナーのどちらか(または平均)である | LIV-03 | D-06: 照合で確定するため、間違っていても照合フェーズで検出 |
| A3 | Playwrightのbrowser binaryがインストール済みである | Environment | 既存scrape_everydb2_manual.pyが動作していることから推定 |
| A4 | JRAサイトの競馬場タブは同一ページ内のDOM切り替え(別URLではない) | LIV-01 | HTML確認: 同一ページに複数開催場のデータが含まれる構造 |
| A5 | RacePredictorにOddsBandFilterを注入する方法はコンストラクタ引数またはプロパティ設定で可能 | STR-04 | BTではBacktestEngineが管理。PTではRacePredictorに直接注入する必要がある。確認必要 |

## Open Questions

1. **OddsBandFilter injection into RacePredictor**
   - What we know: BTではBacktestEngineがOddsBandFilterを管理し、select_bets()内で直接使う。RacePredictor自体にはOddsBandFilterのインスタンスがない
   - What's unclear: RacePredictorにOddsBandFilterを注入する最適な方法 (コンストラクタ引数 vs プロパティ vs select_bets()引数)
   - Recommendation: BTパターンを踏襲し、select_bets()内でOddsBandFilter.filter()を呼ぶ箇所をRacePredictorに追加する

2. **Dirt moisture aggregation rule determination**
   - What we know: JRAはゴール前・4コーナーの2地点を公表。既存CSVは単一dirt_moisture値。aggregate_to_race_level()はentry-level→race-levelの集約
   - What's unclear: CSVのdirt_moistureがゴール前/4コーナー/平均のどれか
   - Recommendation: 重複期間(例:2024年)のJRA生値とCSV値を比較して相関が最も高い規則を採用。初期実装は平均(mean)と仮定

3. **JRA venue-specific page URL structure**
   - What we know: メインURLは `https://www.jra.go.jp/keiba/baba/`。同一ページ内にタブ形式で複数開催場のデータがある
   - What's unclear: タブ切り替えがJSによるDOM表示切り替えか、別URLへの遷移か
   - Recommendation: Playwrightでタブクリック後にpage.content()を取得するアプローチで対応

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Playwright (sync_api) | JRA HTML取得 | ✓ | 1.58.0 | — |
| Chromium browser | Playwright | ✓ | (installed) | — |
| Python 3.11 | Runtime | ✓ | 3.11.15 | — |
| PostgreSQL | PT setup mode | ✓ | localhost:5432 | — |
| BeautifulSoup4 | HTML解析 (推奨) | 要確認 | — | stdlib html.parser でフォールバック (低機能) |

**Missing dependencies with no fallback:**
- なし (Playwright は既にインストール済み)

**Missing dependencies with fallback:**
- BeautifulSoup4: stdlib `html.parser` で簡易パーサーを実装可能だが、JRAの複雑なHTMLには推奨しない。pip install を推奨

## Sources

### Primary (HIGH confidence)
- ソースコード直接確認: `src/betting/default_strategy.py` — build_strategy_config_from_params() のシグネチャ・ロジック
- ソースコード直接確認: `src/backtest/engine.py` L366-440 — BTのmanifest→RacePredictor注入パターン
- ソースコード直接確認: `src/backtest/race_predictor.py` L1265-1267 — AGGRESSIVE固定箇所
- ソースコード直接確認: `src/features/feature_builder.py` L85-136, L330-366 — build_for_inference() + TrackCondition統合
- ソースコード直接確認: `scripts/run_paper_trading.py` — 現在のPT構造
- JRA公式HTML直接取得確認 (2026-06-06): `https://www.jra.go.jp/keiba/baba/` — DOM構造特定

### Secondary (MEDIUM confidence)
- WebSearch: JRAクッション値基準 (12以上=硬め、8-10=標準) [CITED: jra.go.jp/keiba/baba/kaisetsu/index.html]
- WebSearch: クッション値は芝コースのみ。ダートは含水率 [CITED: abyss-keiba.com]

### Tertiary (LOW confidence)
- なし (全てソースコード確認または公式ドキュメント確認済み)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 全て既存パッケージ。新規インストールなし
- Architecture: HIGH — BTパターンをそのまま適用。DOM構造は実際に確認済み
- Pitfalls: HIGH — 既存コードベースの制約から導出
- JRA scraping: HIGH — HTML構造を実際に取得して確認。Playwrightパターンは既存コードで実績あり

**Research date:** 2026-06-06
**Valid until:** 2026-07-06 (JRA HTML構造変更の可能性あり)
