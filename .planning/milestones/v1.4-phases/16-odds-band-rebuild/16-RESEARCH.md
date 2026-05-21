# Phase 16: Odds Band Rebuild - Research

**Researched:** 2026-05-06
**Domain:** strategy_optimizer.py look-ahead bias fix + OddsBandFilter recalibration
**Confidence:** HIGH

## Summary

Phase 16 は2つの密接に関連する課題を解決する。第一に、`strategy_optimizer.py`の`_run_single_backtest()`ステップ3がOptuna最適化対象と同じ`strategy_config`でtraining_bet_historyを生成しているルックアヘッドバイアスを修正する（ODDS-02）。第二に、アンサンブルモデルで生成されたtraining_bet_historyに基づいてOddsBandFilterを正しく再キャリブレーションするパイプライン統合を実装する（ODDS-01）。

ルックアヘッドバイアスの根本原因は明確: `_run_single_backtest()` line 155-162 で`train_engine`構築時に`strategy_config`（Optuna提案値）をそのまま渡しており、これがtraining_bet_history生成に使われる。これにより、テスト期間の最適化パラメータが学習期間のベット生成に漏洩する。修正は`_build_default_config()`メソッドを追加し、RegimeDetectorのハードコード既定値から構築したデフォルト設定をステップ3で使用することで完結する。

パイプライン統合（D-05/D-06/D-07）は、BacktestEngine.run()内でuse_ensemble=Trueの場合に自動的にtraining_bet_historyを生成する仕組みを実装する。BacktestEngineはすでに`training_bet_history`パラメータと`OddsBandFilter.calibrate()`呼び出しを受け付けるインフラ（line 419, 661-663）を持っているため、engine内部でデフォルト設定を使ったトレーニング期間バックテストを自動実行するロジックを追加すればよい。

**Primary recommendation:** _build_default_config()はRegimeDetector._get_base_params(CONSERVATIVE)の既定値をベースに、DDConfigデフォルトとStakeCalculatorデフォルトからstrategy_config dictを構築する。ステップ3の1行変更（strategy_config → default_config）でルックアヘッドバイアスが完全に除去される。

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ルックアヘッドバイアス修正 (ODDS-02) | API / Backend (strategy_optimizer.py) | -- | パラメータ生成ロジックの変更はバックエンド層 |
| OddsBandFilter再キャリブレーション (ODDS-01) | API / Backend (engine.py) | -- | BacktestEngine内部での自動training_bet_history生成 |
| デフォルトパラメータ構築 | API / Backend (strategy_optimizer.py) | API / Backend (regime_detector.py) | RegimeDetector既定値をソースとするが構築はoptimizer内 |

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** training_bet_history生成のデフォルトパラメータソースはRegimeDetector.get_strategy_params()のハードコード既定値
- **D-02:** デフォルトパラメータは16次元全てを適用（Kelly分数・EV閾値・DD制御・EVスケーリング・OddsBandFilter ROI閾値）
- **D-03:** strategy_optimizer.pyに_build_default_config()メソッドを追加（既存_build_strategy_config()と並存）
- **D-04:** _run_single_backtest()ステップ3を_build_default_config()の出力に変更、ステップ4-5はOptuna提案を使用
- **D-05:** run_backtest.py --ensemble実行時にBacktestEngine.run()内で自動的にtraining_bet_historyを生成
- **D-06:** training_bet_history生成はBacktestEngine.run()内で完結、engine.py内部で最小限のスクリプト変更
- **D-07:** training_bet_history生成時はデフォルトパラメータを使用
- **D-08:** オッズバンド境界は固定 `[1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+]` を維持
- **D-09:** ルックアヘッドバイアス修正は二段階検証（モック検証 + 除外バンド差異確認）

### Claude's Discretion
- _build_default_config()の具体的な実装（どのRegimeDetectorメソッドから値を取得するか）
- BacktestEngine.run()内でのtraining_bet_history自動生成の具体的なロジック
- テストのfixtureデータの内容
- デフォルトパラメータでDDConfigを構築する際の具体的な値

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODDS-01 | アンサンブルモデルでtraining_bet_historyを再生成し、OddsBandFilter.calibrate()でバンド別ROIを再計算する | BacktestEngine.run()内での自動生成（D-05/D-06/D-07）、既存calibrate()インフラ（line 661-663） |
| ODDS-02 | strategy_optimizer.pyのルックアヘッドバイアスを修正し、training_bet_history生成にデフォルトパラメータを使用する | _build_default_config()新規追加（D-03）、_run_single_backtest()ステップ3変更（D-04）、RegimeDetector既定値ソース（D-01） |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| optuna | installed | TPE最適化フレームワーク | Phase 13で導入済み、strategy_optimizer.pyの基盤 |
| LightGBM | installed | MLモデル | プロジェクト全体のML基盤 |
| pandas | installed | データ処理 | bet_historyやDataFrame処理 |
| pytest | installed | テストフレームワーク | 全テストmockベース、DB不要 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | モックテスト | 全テストで使用、DB不要パターン |

**Installation:** 追加インストール不要。既存依存関係のみ。

## Architecture Patterns

### System Architecture Diagram

```
Optuna Trial
    |
    v
_suggest_params() → 14次元ハイパーパラメータ
    |
    v
_build_strategy_config(params) → test_config (Optuna最適化値)
_build_default_config() → train_config (RegimeDetector既定値) [NEW]
    |
    v
_run_single_backtest()
    |
    +-- Step 1: ModelLoader.load_from_dir() → models, info
    |
    +-- Step 2: regime_overrides注入 (Optuna値)
    |
    +-- Step 3: train_engine = BacktestEngine(strategy_params=train_config)  [CHANGED]
    |           train_engine.run(train_start, train_end) → training_bet_history
    |
    +-- Step 4: test_engine = BacktestEngine(strategy_params=test_config)
    |
    +-- Step 5: test_engine.run(test_start, test_end, training_bet_history)
                |
                +-- OddsBandFilter.calibrate(training_bet_history) → バンド別ROI → 除外バンド決定
                +-- レースループ → OddsBandFilter.filter() → 候補除外
                +-- BacktestResult返却
```

### Pattern 1: _build_default_config() (NEW)
**What:** RegimeDetectorのハードコード既定値からstrategy_config dictを構築する。_build_strategy_config()と同じ出力形式だが、Optuna paramsの代わりにデフォルト値を使用。
**When to use:** _run_single_backtest()のステップ3（training_bet_history生成）で呼び出す。
**Example:**
```python
def _build_default_config(self) -> dict[str, Any]:
    """RegimeDetector既定値からデフォルトstrategy_configを構築 (ルックアヘッド防止)"""
    from betting.drawdown_controller import DDConfig
    from models.regime_detector import RegimeDetector
    from domain.types import RegimeState

    # RegimeDetectorのハードコード既定値を取得
    detector = RegimeDetector()
    conservative_params = detector._get_base_params(RegimeState.CONSERVATIVE)

    dd_config = DDConfig()  # デフォルト値: threshold_1=0.10, threshold_2=0.20等

    regime_overrides = {}
    for regime in ("aggressive", "conservative"):
        regime_state = RegimeState(regime)
        base = detector._get_base_params(regime_state)
        regime_overrides[regime] = {
            "fractional_kelly": base["fractional_kelly"],
            "ev_threshold": base["ev_threshold"],
            "edge_threshold": base["edge_threshold"],
        }

    return {
        "dd_config": dd_config,
        "regime_overrides": regime_overrides,
        "fractional_kelly": conservative_params["fractional_kelly"],
        "target_ev": 1.10,  # StakeCalculatorデフォルト
        "max_scale": 2.0,   # StakeCalculatorデフォルト
        "roi_threshold": 1.0,  # OddsBandFilterデフォルト
    }
```

### Pattern 2: BacktestEngine.run()内自動training_bet_history生成 (D-05/D-06/D-07)
**What:** use_ensemble=Trueの場合、BacktestEngine.run()内で自動的にトレーニング期間バックテストを実行しtraining_bet_historyを生成する。
**When to use:** run_backtest.py --ensemble実行時。
**Key insight:** 既存の`training_bet_history`パラメータ（line 419）は外部からの受け渡し用。D-05の要件はengine内部で自動生成すること。外部から渡された場合はそれを使用し、渡されない場合はengine内部で自動生成する設計が最適。

### Anti-Patterns to Avoid
- **RegimeDetectorインスタンスの毎回生成:** _build_default_config()内で`RegimeDetector()`をnewすると学習済みモデルを持たないインスタンスになるが、`_get_base_params()`はインスタンス状態に依存しないハードコード値を返すため問題ない。ただし、将来的に外部化されたパラメータ（override_params等）を考慮する場合は注意。
- **strategy_optimizer内でのモデル再学習:** _run_single_backtest()は学習済みモデルをロードするのみ。再学習は不可。

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| strategy_config dict構築 | カスタムパラメータハードコード | RegimeDetector._get_base_params() + DDConfig() | 既存のSingle Source of Truthを利用 |
| バンド別ROI計算 | カスタムROI計算ロジック | OddsBandFilter.calibrate() | 既存実装が堅牢でテスト済み |
| DD制御パラメータ検証 | 手動閾値チェック | DDConfig.__post_init__() | 閾値整合性の自動検証 |

## Common Pitfalls

### Pitfall 1: _build_default_config()のfractional_kellyのレジーム選択
**What goes wrong:** _build_default_config()でAGGRESSIVEレジームのfk=0.50を使うかCONSERVATIVEのfk=0.25を使うかでtraining_bet_historyの件数が大きく変わる。
**Why it happens:** レジーム別パラメータが異なるため。AGGRESSIVEは更多ベット、CONSERVATIVEはより厳選。
**How to avoid:** トップレベルの`fractional_kelly`（BacktestEngine.__init__ line 394-396で使用）はCONSERVATIVE値(0.25)を使用。regime_overridesは各レジームの既定値をそのまま使用。これが最も「最適化前」の中立な状態。
**Warning signs:** training_bet_historyの件数がtraining期間に対して異常に少ない/多い場合。

### Pitfall 2: DDConfigデフォルト値とOptuna探索範囲の不一致
**What goes wrong:** DDConfig()のデフォルト(dd_threshold_1=0.10, dd_threshold_2=0.20)がOptuna探索範囲(threshold_1: 0.05-0.20, threshold_2: 0.15-0.35)の範囲内にあることを確認する必要がある。
**Why it happens:** デフォルト値が探索範囲外の場合、最適化パラメータとデフォルトパラメータで全く異なるDD挙動になる。
**How to avoid:** DDConfig()のデフォルト値は探索範囲内にある: threshold_1=0.10 (in [0.05, 0.20]), threshold_2=0.20 (in [0.15, 0.35]). 問題なし。 [VERIFIED: src/betting/drawdown_controller.py line 21-22 vs src/tuning/strategy_optimizer.py line 68-69]

### Pitfall 3: BacktestEngine内自動生成の再帰呼び出し
**What goes wrong:** BacktestEngine.run()内でtraining_bet_history自動生成のためにBacktestEngineを再帰的にnew+runすると、その内部でもOddsBandFilter.calibrate()が呼ばれ、さらにtraining_bet_historyが必要になる可能性。
**Why it happens:** エンジン内部でエンジンを構築する再帰パターン。
**How to avoid:** training_bet_history生成用のBacktestEngineはtraining_only=Trueフラグまたはcalibrate=Falseで構築し、OddsBandFilter.calibrate()をスキップする。または、よりシンプルに: トレーニングバックテスト用エンジンはOddsBandFilterを持たない（betting_target="win"の場合でもフィルタなしでベット生成する）設計にする。CONSERVATIVEのデフォルトパラメータ自体がフィルタリングの役割を果たすため、OddsBandFilterなしでも合理的なベットが生成される。

### Pitfall 4: テストでのMockEngine構築の複雑さ
**What goes wrong:** _run_single_backtest()のテストは2回のBacktestEngine構築をモックするが、ステップ3とステップ4で異なるstrategy_configが使用されることを検証する必要がある。
**Why it happens:** 既存テスト（TestRunSingleBacktest）はMockEngine.call_count == 2を検証するが、各callのstrategy_paramsの内容まで検証していない。
**How to avoid:** MockEngine.call_args_list[0]（ステップ3）と[1]（ステップ4）のstrategy_paramsを比較するテストを追加。ステップ3はdefault_config、ステップ4はoptuna_configであることを確認。

### Pitfall 5: training_bet_history生成時のRegimeDetector未学習状態
**What goes wrong:** _build_default_config()で構築した設定をBacktestEngineに渡すと、engine内のRegimeDetectorが学習済みモデルを持っているが、default_configのregime_overridesがOptuna値ではなくデフォルト値で上書きされる。
**Why it happens:** _run_single_backtest() line 147-148で`models.regime_detector._override_params = regime_overrides`を実行するが、これはOptuna提案値のregime_overrides。
**How to avoid:** ステップ3の前にデフォルトregime_overridesで上書きし、ステップ3終了後にOptuna regime_overridesで再度上書きする。または、ステップ3用にモデルのshallow copyを作成する。よりシンプルな方法: ステップ3のtrain_engineはデフォルトregime_overridesで構築し、ステップ4の前にOptuna値で上書きする。現在のコード（line 146-148）のregime_overrides上書きをステップ3の後に移動する。

## Code Examples

### _build_default_config() -- ルックアヘッド防止 (ODDS-02)

```python
# Source: src/tuning/strategy_optimizer.py (new method)
def _build_default_config(self) -> dict[str, Any]:
    """RegimeDetector既定値からデフォルトstrategy_configを構築。

    ルックアヘッド防止: Optuna最適化値に依存せず、
    ハードコード既定値のみでtraining_bet_historyを生成する。
    """
    from betting.drawdown_controller import DDConfig
    from domain.types import RegimeState

    # 各レジームのハードコード既定値を取得
    # _get_base_params()はインスタンス状態に依存しない
    from models.regime_detector import RegimeDetector
    detector = RegimeDetector()

    conservative_params = detector._get_base_params(RegimeState.CONSERVATIVE)

    dd_config = DDConfig()  # デフォルト: window=400, t1=0.10, t2=0.20等

    regime_overrides = {}
    for regime_key in ("aggressive", "conservative"):
        state = RegimeState(regime_key)
        base = detector._get_base_params(state)
        regime_overrides[regime_key] = {
            "fractional_kelly": base["fractional_kelly"],
            "ev_threshold": base["ev_threshold"],
            "edge_threshold": base["edge_threshold"],
        }

    return {
        "dd_config": dd_config,
        "regime_overrides": regime_overrides,
        "fractional_kelly": conservative_params["fractional_kelly"],
        "target_ev": 1.10,   # StakeCalculator.__init__ デフォルト
        "max_scale": 2.0,    # StakeCalculator.__init__ デフォルト
        "roi_threshold": 1.0, # OddsBandFilter.__init__ デフォルト
    }
```

### _run_single_backtest() ステップ3変更 (ODDS-02)

```python
# Source: src/tuning/strategy_optimizer.py:118-192
# 変更箇所: line 150-168

def _run_single_backtest(self, strategy_config, test_start, test_end, trial=None, fold_idx=0):
    from backtest.engine import BacktestEngine
    from db.model_loader import ModelLoader

    # 1. 学習済みモデルをロード
    loader = ModelLoader()
    models, info = loader.load_from_dir(self.models_dir, use_ensemble_override=True)

    # 2. デフォルトconfigでtraining_bet_history生成 (ルックアヘッド防止)
    default_config = self._build_default_config()

    # 2a. デフォルトregime_overridesを一時的に注入
    default_regime_overrides = default_config.get("regime_overrides")
    if default_regime_overrides:
        models.regime_detector._override_params = default_regime_overrides

    # 3. トレーニング期間バックテスト (デフォルトパラメータで実行)
    training_bet_history = None
    try:
        train_engine = BacktestEngine(
            models=models,
            initial_bankroll=self.initial_bankroll,
            betting_mode="kelly",
            diag_prefix=f"opt_train_fold{fold_idx}",
            betting_target="win",
            strategy_params=default_config,  # [CHANGED] strategy_config -> default_config
        )
        train_result = train_engine.run(info.train_start, info.train_end)
        training_bet_history = train_result.bet_history
    except Exception as e:
        logger.warning("Fold %d: training-phase backtest failed: %s", fold_idx, e)

    # 2b. Optuna値のregime_overridesで上書き (テスト期間用)
    regime_overrides = strategy_config.get("regime_overrides")
    if regime_overrides:
        models.regime_detector._override_params = regime_overrides

    # 4-5. テスト期間バックテスト (Optuna提案値で実行) -- 変更なし
    engine = BacktestEngine(
        models=models,
        initial_bankroll=self.initial_bankroll,
        betting_mode="kelly",
        diag_prefix=f"opt_fold{fold_idx}",
        betting_target="win",
        strategy_params=strategy_config,
    )
    result = engine.run(test_start, test_end, training_bet_history=training_bet_history)
    # ...
```

### OddsBandFilter.calibrate() -- 既存インターフェース確認 (ODDS-01)

```python
# Source: src/betting/odds_band_filter.py:38-78
# 変更不要。bet_historyはlist[dict]で {"odds", "result", "stake"} キーが必要。
# BacktestEngine.run()のbet_historyはこの形式を満たす。

# line 661-663 (engine.py内):
# if self._odds_band_filter is not None and training_bet_history:
#     self._odds_band_filter.calibrate(training_bet_history)
```

### RegimeDetector._get_base_params() -- 全パラメータ確認 [VERIFIED]

```python
# Source: src/models/regime_detector.py:201-259

# AGGRESSIVE (20+パラメータ):
#   ev_threshold=1.10, edge_threshold=0.05, fractional_kelly=0.50,
#   min_place_prob=0.08, max_place_odds=18.0, wide_enabled=False,
#   score_threshold=0.010, max_bets_per_race=1, ... (+12個)

# CONSERVATIVE (9パラメータ):
#   ev_threshold=1.30, edge_threshold=0.06, fractional_kelly=0.25,
#   min_place_prob=0.09, max_place_odds=18.0, wide_enabled=False,
#   score_threshold=0.020, max_bets_per_race=1, ...

# COLLAPSED (8パラメータ):
#   ev_threshold=1.50, edge_threshold=0.09, fractional_kelly=0.00,
#   skip=True, ...

# _suggest_params()の14次元と_build_strategy_config()のマッピング:
# fk_aggressive, fk_conservative → regime_overrides[regime]["fractional_kelly"]
# ev_aggressive, ev_conservative → regime_overrides[regime]["ev_threshold"]
# edge_aggressive, edge_conservative → regime_overrides[regime]["edge_threshold"]
# dd_threshold_1, dd_threshold_2, multiplier_reduced, rolling_window, min_stay_races → DDConfig
# target_ev, max_scale → StakeCalculator
# roi_threshold → OddsBandFilter
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| training_bet_history生成にOptuna最適化値使用 | デフォルトパラメータで生成 (本Phase) | Phase 16 | ルックアヘッドバイアス完全除去 |
| 外部スクリプトでtraining_bet_history手動生成 | BacktestEngine.run()内で自動生成 | Phase 16 (D-05) | パイプライン統合、手動作業排除 |

**Deprecated/outdated:**
- なし。既存コードはPhase 13-15で確立されたパターンに基づいている。

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | RegimeDetector._get_base_params()はインスタンス状態（model等）に依存せず、ハードコード値のみを返す | Pattern 1 | LOW: コード確認済み（line 201-259）、純粋なdict返却 |
| A2 | DDConfig()のデフォルト値はOptuna探索範囲内にある | Pitfall 2 | LOW: 確認済み |
| A3 | _build_default_config()でCONSERVATIVE値をトップレベルfractional_kellyに使うのが適切 | Pattern 1 | MEDIUM: CONSERVATIVEが最も「中立」な選択だが、ユーザー確認推奨 |
| A4 | トレーニングバックテスト用のBacktestEngineはOddsBandFilterなしで十分（またはcalibrate不要） | Pitfall 3 | LOW: デフォルトパラメータ自体がフィルタリング役割を果たす |
| A5 | BacktestEngine.run()内の自動training_bet_history生成では、モデルのtrain_start/train_endをinfoから取得可能 | D-07 | LOW: ModelInfo.train_start/train_endは既に利用可能 |

**If this table is empty:** 全claimが検証済み -- ユーザー確認不要。

## Open Questions

1. **(RESOLVED) BacktestEngine.run()内自動生成のトレードオフ (D-05/D-06)**
   - What we know: BacktestEngine.run()はすでにtraining_bet_historyパラメータを受け取る。D-05/D-06はengine内部で自動生成する要件。
   - What's unclear: 自動生成のタイミング -- run()の冒頭で生成するか、__init__で準備するか。run()の冒頭が最もシンプル（データアクセスがrun内で完結）。
   - Recommendation: run()の冒頭（OddsBandFilter.calibrate()の直前、line 661の前）で、training_bet_historyがNoneの場合のみ自動生成する設計。
   - **Resolution:** Plan 02で採用。run()内のOddsBandFilter.calibrate()直前でtraining_bet_history=Noneの場合に_generate_training_bet_history()を呼び出す設計。Pitfall 3回避のため内部エンジンはbetting_target="place"で構築。

2. **(RESOLVED) _build_default_config()でのRegimeDetectorインスタンス生成コスト**
   - What we know: RegimeDetector.__init__()は軽量（dictとカウンタの初期化のみ）。train()は呼ばない。
   - What's unclear: _run_single_backtest()が100トライアル x 2fold = 200回呼ばれるため、毎回newするオーバーヘッド。
   - Recommendation: __init__で1回だけRegimeDetectorインスタンスを作成し_build_default_config()で使い回す。または_build_default_config()の結果をキャッシュする。
   - **Resolution:** 共通ユーティリティ(src/betting/default_strategy.py)に抽出。Plan 02のexecutorが実装時にキャッシュを検討可能。RegimeDetector.__init__()が軽量なため、現時点では毎回生成でも問題なし。

3. **(RESOLVED) AGGRESSIVEレジームの多数パラメータの扱い**
   - What we know: AGGRESSIVEは20+のパラメータを持つが、_build_strategy_config()が変換するのはfk, ev, edgeの3つのみ。他のパラメータ（score_threshold等）はBacktestEngineのRacePredictor内で使用される。
   - What's unclear: _build_default_config()はregime_overridesとしてfk, ev, edgeのみ返すが、これで十分か。
   - Recommendation: 十分。_get_base_params()の他のパラメータ（score_threshold等）はBacktestEngineのRacePredictor内でregime_paramsから読み込まれ、regime_overridesの3パラメータはそれらを上書きする形。デフォルトではregime_overridesが適用された後も_get_base_params()の残りパラメータは有効。
   - **Resolution:** Recommendationを採用。regime_overridesはfk, ev, edgeの3パラメータのみで十分。残りのパラメータは_get_base_params()のハードコード値が有効なため。

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified). Phase 16は純粋なコード変更のみ。既存のPython 3.11 + pytest環境で完結する。PostgreSQL不要（テストは全てmock）。

## Sources

### Primary (HIGH confidence)
- `src/tuning/strategy_optimizer.py` - _suggest_params(), _build_strategy_config(), _run_single_backtest() 全文確認
- `src/betting/odds_band_filter.py` - OddsBandFilter.calibrate(), filter() 全文確認
- `src/models/regime_detector.py` - _get_base_params(), get_strategy_params() 全文確認（3レジーム全パラメータ）
- `src/backtest/engine.py` - BacktestEngine.__init__(), run(), OddsBandFilter統合箇所確認
- `src/betting/drawdown_controller.py` - DDConfig dataclass全文確認
- `src/betting/stake_calculator.py` - StakeCalculator.__init__() デフォルト値確認
- `src/domain/types.py` - RegimeState enum確認
- `tests/test_strategy_optimizer.py` - 既存テストパターン確認
- `tests/test_odds_band_filter.py` - 既存テストパターン確認
- `config/settings.yaml` - betting_strategyデフォルト値確認

### Secondary (MEDIUM confidence)
- `.planning/phases/15-ev-filter-enhancement/15-CONTEXT.md` - Phase 15パイプライン統合パターン参照
- `.planning/phases/14-gate-recalibration/14-CONTEXT.md` - Phase 14パイプライン統合パターン参照
- `.planning/REQUIREMENTS.md` - ODDS-01, ODDS-02要件定義
- `.planning/ROADMAP.md` - Phase 16 Success Criteria

### Tertiary (LOW confidence)
- なし。全ての主要claimはソースコード直接確認済み。

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 既存プロジェクト依存関係のみ、追加不要
- Architecture: HIGH - _build_default_config()設計は既存_build_strategy_config()と同じパターン
- Pitfalls: HIGH - 全ソースコードを直接確認済み

**Research date:** 2026-05-06
**Valid until:** 30日（安定フェーズ、外部ライブラリ変更なし）

---

## Project Constraints (from CLAUDE.md)

- **Ruff**: target py311, line-length=100, rules=E/F/I/N/W
- **Mypy**: `disallow_untyped_defs = true` (全関数に型アノテーション必須)
- **テスト**: DB不要 (全て mock) -- `unittest.mock` 使用
- **コミットメッセージ**: Conventional Commits (日本語)
- **Import path**: `from db.repository import DataRepository`, `from domain.types import ...` (pythonpath = `[".", "src"]`)
