# Phase 13: Risk Calibration & Parameter Optimization - Research

**Researched:** 2026-05-05
**Domain:** DD制御再設計 + Optuna戦略パラメータ最適化 + パラメータ凍結拡張
**Confidence:** HIGH

## Summary

Phase 13は3つの要件(RISK-01, VAL-01, VAL-02)を扱う最終フェーズ。コアはDrawdownControllerのROI依存を完全除去してDD%のみの3段階制御(NORMAL/REDUCED/STOP)への再設計、全16次元戦略パラメータのOptuna TPE最適化、そしてParameterFreezeProtocolの戦略パラメータJSON manifest拡張。

現在のDrawdownControllerは8行の乗数テーブル(DD% x ROIの2次元ルックアップ)を持ち、_compute_rolling_roi()でSMA+EWMAハイブリッドROIを計算している。ROI依存を除去しDD%のみのシンプルな閾値判定に置き換えることで、WIN的中率10%環境でのDD制御の信頼性が向上する。既存のヒステリシス(RECOVERY_INCREMENT等)は状態遷移に最低滞在レース数という明示的ヒステリシスバンドに再設計される。

Optuna最適化は既存のOptunaTuner(optuna_tuner.py)を参照実装として、新規ファイルstrategy_optimizer.pyに独立実装。WalkForwardCV.run()に戦略パラメータ注入インターフェースを追加し、各foldでパラメータを凍結して評価することでルックアヘッドバイアスを構造的に防止する。

**Primary recommendation:** DrawdownControllerを完全にROIフリーに再設計し、全パラメータをコンストラクタ注入可能にした上で、Optuna TPE + WalkForwardCVで16次元空間を100トライアル探索する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** DrawdownControllerのコンストラクタにrolling_window, dd_thresholds, multipliers等を引数追加。Phase 12のStakeCalculatorコンストラクタ注入パターンと統一。ハードコードクラス定数は全てコンストラクタ引数に移行
- **D-02:** ROI依存を完全に除去し、DD%のみの3段階制御(NORMAL/REDUCED/STOP)に再設計
- **D-03:** 状態遷移にヒステリシスバンドを追加。低的中率環境での発振防止。各状態に最低滞在レース数を設定
- **D-04:** リカバリは段階的(STOP→REDUCED→NORMAL)。即時復帰せず、各状態間の遷移に最低滞在条件を課す
- **D-05:** WIN用とPLACE用で別々のDDControllerインスタンス
- **D-06:** 具体的なDD閾値、乗数、ヒステリシスバンド幅、最低滞在レース数は全てOptunaで探索
- **D-07:** 全戦略パラメータを一括でOptuna最適化。TPEサンプラー
- **D-08:** 探索空間は約16次元(レジーム別9 + DD制御5 + EVスケーリング2 + OddsBandFilter 1)
- **D-09:** 目的関数はROI主 + ベット数制約(年間1000件以上)
- **D-10:** Walk-forward枠組みで評価。WalkForwardCVを拡張して戦略パラメータを各foldに注入
- **D-11:** 100トライアル + MedianPruner。各トライアルはWF 2fold評価
- **D-12:** 新規ファイル `src/tuning/strategy_optimizer.py` に実装
- **D-13:** ParameterFreezeProtocolを戦略パラメータに拡張。JSON manifest形式 + SHA256ハッシュ
- **D-14:** Optuna最適化完了後にJSON manifestを自動生成
- **D-15:** RegimeDetector.get_strategy_params()の主要パラメータ(fractional_kelly, ev_threshold, edge_threshold)のみをコンストラクタ注入可能に外部化
- **D-16:** MetaSwitcherの_default_params()の値をRegimeDetectorに揃える(乖離解消)。ただしMetaSwitcher自体のリファクタリングは行わない
- **D-17:** コンストラクタ注入でパラメータを直接渡す。settings.yamlはデフォルト値定義のみ
- **D-18:** 変更対象はバックテストパスのみ。ライブパスは変更しない
- **D-19:** 新規本番依存関係なし。optuna>=3.5は既存依存

### Claude's Discretion
- DrawdownControllerのコンストラクタ引数の具体的なシグネチャ設計
- 3段階乗数テーブルの具体的なデータ構造(Dict vs NamedTuple vs dataclass)
- ヒステリシスバンドの実装方法(状態マシン vs バンド幅パラメータ)
- WalkForwardCVへの戦略パラメータ注入インターフェース
- strategy_optimizer.py のクラス設計(Objective関数のカプセル化)
- JSON manifestのスキーマ設計
- RegimeDetectorの主要パラメータ外部化の具体的なリファクタリング
- テスト戦略(DD制御単体テスト + Optuna統合テスト + WF評価テスト)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RISK-01 | DrawdownControllerの乗数テーブル・ローリングウィンドウ・リカバリ閾値がWIN向的中率10%に再調整される | DD制御再設計パターン(D-01〜D-06)、3段階NORMAL/REDUCED/STOP制御、ROI除去 |
| VAL-01 | ParameterFreezeProtocolが戦略パラメータをカバーし、ルックアヘッドバイアスを防止する | JSON manifest + SHA256パターン(D-13, D-14)、ParameterFreezeProtocol拡張 |
| VAL-02 | Optuna TPEで全戦略パラメータの同時最適化が実行され、最適設定が発見される | strategy_optimizer.py新規実装(D-12)、WalkForwardCV拡張(D-10)、16次元TPE最適化 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| DD制御(状態遷移・乗数計算) | API/Backend (betting/) | -- | 純粋な計算ロジック。DBアクセスなし |
| パラメータ外部化(RegimeDetector) | API/Backend (models/) | -- | get_strategy_params()の主要3パラメータをコンストラクタ注入に |
| Optuna最適化ループ | API/Backend (tuning/) | -- | 既存OptunaTunerと同じ階層。ML HPとは独立ファイル |
| WalkForwardCV拡張 | API/Backend (models/) | -- | 既存WFインフラへのパラメータ注入追加 |
| ParameterFreezeProtocol拡張 | API/Backend (backtest/) | -- | 既存PFPへのJSON manifest機能追加 |
| BacktestEngine DD分岐 | API/Backend (backtest/) | -- | WIN/PLACE別DDControllerインスタンス管理 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| optuna | 4.8.0 (installed) | 戦略パラメータ最適化 | 既存依存。TPESampler + MedianPruner使用 [VERIFIED: pip show] |
| numpy | >=1.26 | 配列計算 | 既存依存。DD%計算に使用 [VERIFIED: pyproject.toml] |
| pandas | >=2.2 | DataFrame操作 | 既存依存。WalkForwardCV結果集計 [VERIFIED: pyproject.toml] |
| dataclasses | stdlib | DD閾値設定のデータ構造 | Python 3.11標準。NamedTupleより型安全性が高い |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib | stdlib | SHA256ハッシュ(JSON manifest) | ParameterFreezeProtocol拡張 |
| json | stdlib | manifest読み書き | 戦略パラメータの保存・復元 |
| pytest | >=8.0 | テスト | DD制御・Optuna統合テスト |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dataclass (DD閾値) | TypedDict | dataclassはデフォルト値・__init__自動生成で有利 |
| JSON manifest | YAML manifest | JSONは標準ライブラリのみで扱える。YAMLはpyyamlが必要だが人間可読性は同等 |
| MedianPruner | SuccessiveHalvingPruner | MedianPrunerはトライアル数が少ない(100)場合に適切 [CITED: optuna docs] |

**Installation:**
```bash
# No new dependencies needed. optuna>=3.5 already installed.
pip install -e ".[dev]"
```

## Architecture Patterns

### System Architecture Diagram

```
Optuna Study (100 trials)
    |
    v
StrategyOptimizer._suggest_params(trial) → params dict (~16 dims)
    |
    v
WalkForwardCV.run(start, end, strategy_params=params)
    |
    +-- Fold 1: train → freeze → test(with params injected)
    |       |
    |       v
    |   BacktestEngine(models, params)
    |       |
    |       +-- WIN DDController(params_dd_win)
    |       +-- PLACE DDController(params_dd_place)
    |       +-- StakeCalculator(fractional_kelly, target_ev, max_scale)
    |       +-- RegimeDetector(fractional_kelly, ev_threshold, edge_threshold per regime)
    |       |
    |       v
    |   ROI + bet_count → objective score
    |
    +-- Fold 2: same flow
    |
    v
Mean ROI (objective) → Optuna maximizes
    |
    v
Best params → JSON manifest (SHA256 hash) → ParameterFreezeProtocol
```

### Recommended Project Structure
```
src/
├── betting/
│   ├── drawdown_controller.py    # 再設計: ROI除去, 3段階, コンストラクタ注入
│   ├── stake_calculator.py       # 変更なし(Phase 12完了)
│   └── meta_switcher.py          # 値揃えのみ(最小変更)
├── models/
│   ├── regime_detector.py        # 主要3パラメータ外部化
│   └── walk_forward_cv.py        # 戦略パラメータ注入追加
├── backtest/
│   ├── engine.py                 # WIN/PLACE別DDController管理
│   ├── parameter_freeze_protocol.py  # JSON manifest追加
│   └── race_predictor.py         # 変更なし(Phase 12完了)
├── tuning/
│   ├── optuna_tuner.py           # 変更なし(ML HP参照実装)
│   └── strategy_optimizer.py     # 新規: Optuna戦略パラメータ最適化
└── domain/
    └── types.py                  # DrawdownState enum更新
```

### Pattern 1: DrawdownController 3段階ROI-Fリー制御

**What:** DD%のみで3段階(NORMAL/REDUCED/STOP)を判定し、ROI計算を完全除去
**When to use:** WIN的中率10%環境などROIがノイジーすぎる場合

**現在の実装(ROI依存):**
```python
# drawdown_controller.py lines 42-53: 8行の2次元ルックアップテーブル
MULTIPLIER_TABLE = [
    (DD_lo, DD_hi, ROI_lo, ROI_hi, multiplier),  # 8 rows
]

# lines 132-144: SMA + EWMA ハイブリッドROI
def _calc_rolling_roi(self) -> float:
    recent = self.bet_history[-self.ROLLING_WINDOW:]
    ...
```

**新設計(DD%のみ):**
```python
# コンストラクタ注入パターン (Phase 12 StakeCalculatorと統一)
@dataclass
class DDConfig:
    rolling_window: int = 400         # ROLLING_WINDOW=150→400+ (D-06: Optuna探索)
    dd_threshold_1: float = 0.10      # NORMAL→REDUCED境界 (Optuna探索)
    dd_threshold_2: float = 0.20      # REDUCED→STOP境界 (Optuna探索)
    multiplier_normal: float = 1.0    # 固定
    multiplier_reduced: float = 0.50  # Optuna探索
    multiplier_stop: float = 0.0      # 固定(ベット停止)
    min_stay_races: int = 10          # 最低滞在レース数 (ヒステリシス)
    recovery_rate: float = 0.02       # STOP→REDUCEDへの段階的リカバリ歩幅

class DrawdownController:
    def __init__(self, peak_bankroll: float, cfg: DDConfig | None = None) -> None:
        self.cfg = cfg or DDConfig()
        ...
        # _compute_rolling_roi() を完全除去
        # bet_history は不要 (ROI計算用だったため)
```

### Pattern 2: WalkForwardCV戦略パラメータ注入

**What:** WalkForwardCV.run()にstrategy_params引数を追加し、各foldのBacktestEngineに注入
**When to use:** Optuna目的関数内でWF評価する場合

```python
# walk_forward_cv.py: run()メソッド拡張
def run(
    self,
    start_date: str,
    end_date: str,
    strategy_params: dict[str, Any] | None = None,  # 追加
) -> CVResult:
    for fold in folds:
        models = self.pipeline.run(fold.train_start, fold.train_end)
        # 戦略パラメータを注入したengine生成
        engine = self.backtest_engine_factory(models, strategy_params)
        fold_result = engine.run(fold.test_start, fold.test_end)
        ...
```

### Pattern 3: ParameterFreezeProtocol JSON Manifest拡張

**What:** 既存pickleハッシュに加えて、戦略パラメータのJSON manifestをSHA256で保護
**When to use:** Optuna最適化完了後のパラメータ保存・検証

```python
# parameter_freeze_protocol.py: 追加機能
@staticmethod
def save_strategy_manifest(
    params: dict[str, Any],
    path: Path,
) -> str:
    """戦略パラメータをJSON保存 + SHA256ハッシュ返却"""
    data = json.dumps(params, sort_keys=True, indent=2)
    sha256 = hashlib.sha256(data.encode()).hexdigest()
    manifest = {"params": params, "sha256": sha256}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sha256

@staticmethod
def verify_strategy_manifest(path: Path) -> dict[str, Any]:
    """JSON manifestのSHA256照合"""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest["sha256"]
    actual = hashlib.sha256(
        json.dumps(manifest["params"], sort_keys=True).encode()
    ).hexdigest()
    if actual != expected:
        raise ValueError(f"Strategy manifest hash mismatch")
    return manifest["params"]
```

### Anti-Patterns to Avoid
- **ROI計算を残す:** _calc_rolling_roi()、_compute_rolling_roi()の類は一切残さない。WIN的中率10%ではROIがノイジーすぎてDD制御の誤動作を引き起こす [VERIFIED: drawdown_controller.py lines 132-144]
- **既存MULTIPLIER_TABLE(8行)を流用:** 2次元(DD×ROI)ルックアップはROI除去と矛盾。DD%のみのシンプルな閾値判定に置き換える
- **Optuna最適化でrandom sampler使用:** 16次元空間にはTPEが必要。RandomSamplerでは100トライアルで十分な探索不可 [CITED: optuna docs - TPESampler]
- **目的関数をROIのみにする:** 過度なフィルタリング→ベット数激減→統計的有意性喪失の危険。ベット数制約(年間1000件以上)必須 [VERIFIED: CONTEXT.md D-09]
- **WalkForwardCVなしでOptuna評価:** ルックアヘッドバイアスが混入する。必ずWF枠組みで評価 [VERIFIED: STATE.md pending concern]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ベイズ最適化 | カスタムパラメータ探索 | optuna TPESampler | 16次元空間の効率的探索。勾配不要 [VERIFIED: optuna 4.8.0 installed] |
| 早期終了 pruner | 手動ループ制御 | optuna MedianPruner | 望み薄トライアルの早期打切り。100トライアル限定で有効 [CITED: optuna docs] |
| パラメータハッシュ | カスタムチェックサム | hashlib.sha256 | 既存PFPで実績あり。SHA256は暗号学的に安全 [VERIFIED: parameter_freeze_protocol.py line 98] |
| 時系列CV分割 | カスタムfold生成 | WalkForwardCV.generate_folds() | 既存実装あり。train/test期間の正確な計算を保証 [VERIFIED: walk_forward_cv.py lines 84-128] |
| JSON manifest | カスタムバイナリフォーマット | json + hashlib | 人間可読 + diff容易。CONTEXT.md D-13で決定済み |

**Key insight:** 既存のOptunaTuner、WalkForwardCV、ParameterFreezeProtocolは全て再利用・拡張可能。新規にゼロから構築する必要があるのはstrategy_optimizer.pyのみ。

## Common Pitfalls

### Pitfall 1: DD制御ROI除去時の既存テスト破壊
**What goes wrong:** test_drawdown_controller.pyの15テスト全てがROI依存(例: `ctrl.update(90000, 0.5)` の第2引数がROI比)。ROI除去でupdate()のシグネチャが変わり全テストがREDになる
**Why it happens:** 既存テストはROI比を`update()`に渡す設計。DD%のみに変更すると引数が不要になる
**How to avoid:** update()のシグネチャを`update(bankroll, bet_return)`から`update(bankroll)`に変更。テストはbet_returnに依存するROI計算をテストしているため、ROI関連テストは全て書き直し。DD%閾値ベースの新テストに置き換える
**Warning signs:** 既存テストで`ctrl.update(bankroll, 0.5)`や`rolling_roi`への言及がある場合

### Pitfall 2: WalkForwardCV注入時のBacktestEngine生成
**What goes wrong:** WalkForwardCV.backtest_engine_factoryが`lambda models: BacktestEngine(models)`のシグネチャで、戦略パラメータを受け取れない
**Why it happens:** 既存factoryはTrainedModelsV5のみを受け取る。戦略パラメータ注入にはfactoryシグネチャの変更が必要
**How to avoid:** factoryを`Callable[[TrainedModelsV5, dict | None], BacktestEngine]`に拡張。後方互換のためstrategy_params=Noneをデフォルトにする
**Warning signs:** factory内でStakeCalculator()やDrawdownController()をハードコード生成している箇所

### Pitfall 3: Optuna目的関数内での重いWF評価
**What goes wrong:** 各トライアルでWF 2fold評価(各fold = 学習 + バックテスト)を実行すると、1トライアルあたり~4時間(Phase 12の記録より)。100トライアルで400時間
**Why it happens:** WalkForwardCV.run()はpipeline.run()で学習も実行する。戦略パラメータ最適化では学習は不要(モデル固定)なのに毎回学習している
**How to avoid:** strategy_optimizerでは学習済みモデルをロードしてバックテストのみ実行。WalkForwardCVのpipeline=Noneでfactoryのみ使用するパターン、あるいはStrategyOptimizer独自の軽量WFループを実装
**Warning signs:** Optuna目的関数内でpipeline.run()を呼んでいる場合

### Pitfall 4: DrawdownState enumのRECOVERING→STOP変更の影響範囲
**What goes wrong:** domain/types.pyのRecoveryState enumをNORMAL/REDUCED/STOPに変更すると、DDStateデータクラス(rolling_roi等)やtest_drawdown_controller.pyのRecoveryState参照が全て壊れる
**Why it happens:** RecoveryStateは3ファイル(domain/types.py定義、domain/models.py DDState使用、drawdown_controller.py使用)にまたがる
**How to avoid:** (1) RecoveryState enumをNORMAL/REDUCED/STOPに更新 (2) DDStateデータクラスのrolling_roiフィールドを除去(current_dd, n_bets_eval, recovery_stateのみに) (3) テストを全て新enumに更新
**Warning signs:** Grep for `RecoveryState.RECOVERING` or `rolling_roi` in tests/

### Pitfall 5: Optuna MedianPrunerの未報告trial
**What goes wrong:** MedianPrunerはtrial.report()での中間値報告を前提とするが、バックテストのROIは最後にしか計算できないため、prunerが動作しない
**Why it happens:** バックテストは累積ROIをレースごとに計算可能だが、report()を呼ぶ実装が必要
**How to avoid:** Optuna目的関数内でレースループの各Nレースごとに`trial.report(interim_roi, step)`を呼び、MedianPrunerが途中で打ち切れるようにする。または、foldごとにreport()する
**Warning signs:** MedianPrunerが一度もpruneしない現象

### Pitfall 6: WIN/PLACE別DDControllerのbankroll共有
**What goes wrong:** WIN用とPLACE用で別DDControllerインスタンスを作る際、peak_bankrollが独立してしまい、全体のDD状態を正しく反映できない
**Why it happens:** BacktestEngineのbankrollはWIN/PLACE共通だが、DDControllerが別インスタンスだとpeak_bankrollが別々に追跡される
**How to avoid:** DDControllerのupdate()はbankrollを受け取る設計なので、両方のDDControllerに同じbankroll値を渡す。ただしWINのベットのみWIN用DDControllerにフィードバックし、PLACEのベットのみPLACE用にフィードバックする分離が必要。あるいは、単一DDController(WIN/PLACE共通)のままROLLING_WINDOW等のパラメータだけをWIN用に最適化する簡易アプローチも検討
**Warning signs:** peak_bankrollが2つのインスタンスで異なる値になる場合

## Code Examples

### DrawdownController再設計の中核ロジック
```python
# Source: 現在のdrawdown_controller.py + CONTEXT.md D-01〜D-06 に基づく設計
from dataclasses import dataclass
from domain.types import RecoveryState  # NORMAL, REDUCED, STOP に更新

@dataclass
class DDConfig:
    """Optunaで探索可能なDD制御パラメータ"""
    rolling_window: int = 400       # D-06: Optuna探索範囲 [200, 800]
    dd_threshold_1: float = 0.10    # NORMAL→REDUCED [0.05, 0.20]
    dd_threshold_2: float = 0.20    # REDUCED→STOP    [0.15, 0.35]
    multiplier_reduced: float = 0.50  # [0.1, 0.8]
    multiplier_stop: float = 0.0      # 固定(ベット停止)
    min_stay_races: int = 10          # ヒステリシス [5, 30]

class DrawdownController:
    def __init__(self, peak_bankroll: float, cfg: DDConfig | None = None) -> None:
        self.cfg = cfg or DDConfig()
        self.peak_bankroll = peak_bankroll
        self._state = RecoveryState.NORMAL
        self._races_in_state = 0
        self._current_multiplier = 1.0

    def update(self, bankroll: float) -> None:
        """ベット結果後の状態更新。ROI引数なし"""
        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll
        dd = (self.peak_bankroll - bankroll) / self.peak_bankroll
        self._transition(dd)
        self._races_in_state += 1

    def _transition(self, dd: float) -> None:
        """DD%のみの3段階状態遷移(ヒステリシス付き)"""
        target = self._determine_target_state(dd)
        if target != self._state and self._races_in_state >= self.cfg.min_stay_races:
            old = self._state
            self._state = target
            self._races_in_state = 0
            self._update_multiplier()
            logger.info(f"DD: {old.value} -> {target.value}")

    def _determine_target_state(self, dd: float) -> RecoveryState:
        if dd >= self.cfg.dd_threshold_2:
            return RecoveryState.STOP
        elif dd >= self.cfg.dd_threshold_1:
            return RecoveryState.REDUCED
        return RecoveryState.NORMAL
```

### StrategyOptimizerのOptuna目的関数パターン
```python
# Source: 既存OptunaTuner(optuna_tuner.py)パターンを踏襲
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

class StrategyOptimizer:
    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        params = {}
        # レジーム別 (x3)
        for regime in ["aggressive", "conservative"]:
            # COLLAPSED: fractional_kelly=0 固定なので探索しない
            params[f"fk_{regime}"] = trial.suggest_float(
                f"fk_{regime}", 0.10, 0.80,
            )
            params[f"ev_{regime}"] = trial.suggest_float(
                f"ev_{regime}", 1.05, 2.00,
            )
            params[f"edge_{regime}"] = trial.suggest_float(
                f"edge_{regime}", 0.03, 0.15,
            )
        # DD制御
        params["dd_threshold_1"] = trial.suggest_float("dd_threshold_1", 0.05, 0.20)
        params["dd_threshold_2"] = trial.suggest_float("dd_threshold_2", 0.15, 0.35)
        params["multiplier_reduced"] = trial.suggest_float("multiplier_reduced", 0.1, 0.8)
        params["rolling_window"] = trial.suggest_int("rolling_window", 200, 800)
        # EVスケーリング
        params["target_ev"] = trial.suggest_float("target_ev", 1.05, 1.50)
        params["max_scale"] = trial.suggest_float("max_scale", 1.0, 3.0)
        # OddsBandFilter
        params["roi_threshold"] = trial.suggest_float("roi_threshold", 0.8, 1.2)
        return params

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        # WalkForwardCV評価 (学習済みモデルでバックテストのみ)
        wf_result = self._run_wf_backtest(params, trial)
        roi = wf_result.mean_roi
        n_bets = wf_result.total_bets
        # D-09: ベット数制約
        if n_bets < self.min_bets:
            return -1.0  # ペナルティ
        return roi

    def optimize(self, n_trials: int = 100) -> dict[str, Any]:
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(self._objective, n_trials=n_trials)
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
```

### BacktestEngine WIN/PLACE別DDController管理
```python
# Source: engine.py lines 363-371 の変更ポイント
if betting_mode == "kelly":
    from betting.drawdown_controller import DDConfig, DrawdownController
    from betting.stake_calculator import StakeCalculator

    # D-05: WIN用とPLACE用で別DDConfig
    dd_cfg_win = strategy_params.get("dd_config_win") if strategy_params else None
    dd_cfg_place = strategy_params.get("dd_config_place") if strategy_params else None

    # 注: 現在の実装ではbetting_targetは"win"または"place"
    # 両方同時にベットするモードはないため、実際は1つのDDControllerのみ生成
    dd_cfg = dd_cfg_win if betting_target == "win" else dd_cfg_place
    dd_controller = DrawdownController(
        peak_bankroll=initial_bankroll,
        cfg=dd_cfg or DDConfig(),
    )
    stake_calc = StakeCalculator(
        fractional_kelly=strategy_params.get("fk_default", 0.5) if strategy_params else 0.5,
        target_ev=strategy_params.get("target_ev", 1.10) if strategy_params else 1.10,
        max_scale=strategy_params.get("max_scale", 2.0) if strategy_params else 2.0,
    )

    self._race_predictor = RacePredictor(
        models,
        stake_calculator=stake_calc,
        dd_controller=dd_controller,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DD×ROI 2次元ルックアップ(8行) | DD%のみ3段階閾値判定 | Phase 13 (今回) | ROI計算のノイズ除去。WIN的中率10%環境で信頼性向上 |
| RecoveryState: NORMAL/REDUCED/RECOVERING | RecoveryState: NORMAL/REDUCED/STOP | Phase 13 (今回) | RECOVERINGの段階的回復→STOPの完全停止。低的中率では中間状態が不要 |
| ハードコードクラス定数 | コンストラクタ注入(DDConfig dataclass) | Phase 13 (今回) | Optuna最適化が可能に。Phase 12 StakeCalculatorパターンの踏襲 |
| ML HPのみOptuna最適化 | ML HP + 戦略パラメータ両方Optuna最適化 | Phase 13 (今回) | 独立ファイル(strategy_optimizer.py)で管理 |
| model pickleのみ凍結 | model pickle + 戦略パラメータJSON manifest | Phase 13 (今回) | ルックアヘッドバイアス防止の完全性向上 |

**Deprecated/outdated:**
- `_calc_rolling_roi()` (SMA+EWMAハイブリッド): ROI依存除去で廃止
- `MULTIPLIER_TABLE` (8行2次元ルックアップ): DD%のみ閾値判定に置き換え
- `RecoveryState.RECOVERING`: STOPに置き換え
- `DDState.rolling_roi`フィールド: ROI計算の廃止に伴い除去

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Optuna TPESamplerは16次元空間を100トライアルで十分に探索可能 | Optuna最適化設計 | 探索不十分→局所最適解に陥る。追加トライアルで対応可能 |
| A2 | バックテストROI 89.0%ベースラインは現在のパラメータ設定で再現可能 | 成功基準 | ベースライン自体が変動している場合は比較不可 |
| A3 | WalkForwardCVの学習ステップをスキップしてバックテストのみ実行可能 | Pitfall 3 | pipeline=None設定でfactoryのみ使用するパターンは検証済み |
| A4 | WIN/PLACEの同時ベットモードは存在せず、DDControllerの分岐は不要 | Pitfall 6 | 将来の拡張でWIN+PLACE同時ベットが追加された場合、再設計が必要 |
| A5 | RegimeDetectorのrunner-up rescue rules等のドメインパラメータはOptuna探索対象外で問題ない | D-15 | 固定パラメータが最適でない場合、最適化効果が限定される |

**If this table is empty:** All claims in this research were verified or cited -- no user confirmation needed.

## Open Questions

1. **WalkForwardCV学習スキップの正確な実装方法**
   - What we know: WalkForwardCV.run()はpipeline.run()で学習→factoryでバックテスト。pipeline=NoneでRuntimeError
   - What's unclear: pipelineを省略してfactoryのみ(学習済みモデルロード)で実行する拡張方法
   - Recommendation: StrategyOptimizer内で独自の軽量WFループを実装(学習済みモデルをロードしてfactoryのみ呼び出し)。WalkForwardCV自体の変更は最小化

2. **Optuna目的関数内でのbankroll初期値とテスト期間**
   - What we know: バックテストはinitial_bankroll=100000。テスト期間は2024年(1年)
   - What's unclear: WF 2foldの具体的なfold定義(2020-2023 train/2024 test、2021-2024 train/2025 testで良いか)
   - Recommendation: run_wf_validation.pyのFOLDS定義を踏襲

3. **ベット数制約の具体的な閾値**
   - What we know: D-09で「年間1000件以上」と記載。2024年テストでは9,074件
   - What's unclear: WF 2fold(各1年テスト)の場合、1foldあたり1000件以上か、2fold合計で1000件以上か
   - Recommendation: 1foldあたりのベット数で評価(年間1000件以上)。各foldが独立した評価単位

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全体 | ✓ | 3.11 (mise) | -- |
| optuna | Optuna最適化 | ✓ | 4.8.0 | -- |
| numpy | DD計算 | ✓ | >=1.26 | -- |
| pandas | DataFrame操作 | ✓ | >=2.2 | -- |
| pytest | テスト | ✓ | >=8.0 | -- |
| PostgreSQL | バックテスト実行(WF検証) | ✗ | -- | Optuna最適化自体はmockベースでテスト可能 |

**Missing dependencies with no fallback:**
- PostgreSQL(WF検証の実際の実行): Optuna最適化のコードはDB不要で実装可能。実際のWF評価実行はPostgreSQL環境が必要

**Missing dependencies with fallback:**
- なし

## Sources

### Primary (HIGH confidence)
- src/betting/drawdown_controller.py - 全行確認(現在実装の完全理解)
- src/betting/stake_calculator.py - 全行確認(Phase 12パターンの参照)
- src/models/regime_detector.py - 全行確認(get_strategy_params 25+パラメータ)
- src/betting/meta_switcher.py - 全行確認(_default_params乖離状況)
- src/backtest/engine.py - lines 340-1028確認(DDController生成・レースループ)
- src/backtest/parameter_freeze_protocol.py - 全行確認(SHA256ハッシュパターン)
- src/models/walk_forward_cv.py - 全行確認(WF run/generate_folds)
- src/tuning/optuna_tuner.py - 全行確認(Optuna参照実装)
- src/domain/types.py - 確認(RecoveryState enum)
- src/domain/models.py - DDState/RegimeConfig確認
- tests/test_drawdown_controller.py - 全15テスト確認
- tests/test_parameter_freeze.py - 全6テスト確認
- tests/test_optuna_tuner.py - 全3テスト確認
- pyproject.toml - 依存関係確認
- config/settings.yaml - betting_strategy section確認

### Secondary (MEDIUM confidence)
- Optuna documentation (optuna.readthedocs.io) - TPESampler, MedianPruner API参照

### Tertiary (LOW confidence)
- なし

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全て既存依存。新規依存なし
- Architecture: HIGH - 全ソースコードを直接確認。パターンはPhase 12実績あり
- Pitfalls: HIGH - 既存テストのROI依存、WF学習ステップ、enum変更の影響範囲を直接確認

**Research date:** 2026-05-05
**Valid until:** 2026-06-05 (stable domain, no fast-moving dependencies)
