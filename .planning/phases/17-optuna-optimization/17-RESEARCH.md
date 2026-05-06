# Phase 17: Optuna Optimization - Research

**Researched:** 2026-05-06
**Domain:** Optuna TPE戦略パラメータ最適化 (WF fold拡張 + multi-seed安定性検証)
**Confidence:** HIGH

## Summary

Phase 17は、Phase 13-16で構築したStrategyOptimizerインフラを4fold対応に拡張し、EV_lowerサーフェス別閾値を15-16次元目として追加、3-seed安定性検証を行う最適化フェーズである。既存コードはPhase 13で14次元Optuna最適化として実装済みで、Phase 16でルックアヘッドバイアス修正済み。主な変更対象は`strategy_optimizer.py`(4fold化、モデルロード最適化、EV_lower 2次元追加、multi-seed実行)、`default_strategy.py`(EV_lower閾値マッピング)、`run_strategy_optimization.py`(multi-seed CLI拡張)。

**Primary recommendation:** 既存の`StrategyOptimizer`クラスをベースに、(1) コンストラクタ引数`n_folds`/`train_years`/`test_years`を活用した動的fold生成、(2) _objective()先頭でのモデルロード最適化、(3) EV_lower 2次元追加、(4) multi-seed実行と安定性レポート生成を実装する。全て既存パターンの拡張であり新規インフラ不要。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 軽量WFアプローチ。学習済みモデル1セットを全foldで共有、fold毎の再学習なし
- **D-02:** 年次4fold構成: テスト期間 2022/2023/2024/2025 の4年。`_generate_folds()`をハードコードからコンストラクタ引数ベースの動的生成に変更
- **D-04:** 100トライアル維持。Phase 13 D-11踏襲
- **D-05:** モデルロードをtrial内1回に最適化 + training_bet_historyをtrial内1回キャッシュ + RegimeDetector可変状態をfold間でリセット(CR-01パターン)
- **D-07:** EV_lower閾値を15-16次元目として_suggest_params()に追加。サーフェス別(芝/ダート)で2次元。合計16次元
- **D-08:** 3 seeds構成: seed=42(100trials), seed=43(50trials), seed=44(50trials)
- **D-10:** 不安定次元はデフォルト値に固定 → 探索空間縮小 → 再最適化。安定性レポート(JSON)で報告
- **D-11:** min_bets_per_fold=1000、ハードカットオフ(ROI=-1.0ペナルティ)を維持

### Claude's Discretion
- `_generate_folds()`の具体的な実装(コンストラクタ引数から動的に生成)
- MedianPrunerの設定(n_startup_trials, n_warmup_steps, interval_check_steps等)
- 安定性判定の具体的な手法と閾値
- 安定性レポートのJSONスキーマ
- モデルロード最適化の具体的な実装(コピー vs リセット)
- _suggest_params()へのEV_lower 2次元追加の実装詳細
- テスト戦略(モックベース、既存パターン踏襲)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OPT-01 | アンサンブルモデルで既存14次元Optuna最適化を実行する(フィルター再キャリブレーション完了後) | StrategyOptimizer + _suggest_params(14次元) + _objective(ROI+bet制約)が実装済み。EV_lower 2次元追加で16次元に拡張 |
| OPT-02 | walk-forward fold数を2→4に増やし過学習リスクを軽減する | _generate_folds()のハードコード→動的生成変更。コンストラクタ引数n_folds=4で対応 |
| OPT-03 | 複数seedでOptuna最適化を実行し、パラメータ安定性を検証して不安定な次元を検出する | multi-seed実行フロー + 安定性レポート(JSON) + 不安定次元の自動固定化機能を新規実装 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Optuna探索空間定義(_suggest_params) | API/Backend | — | 戦略パラメータのOptuna suggest呼び出し。Pythonロジック |
| WF fold生成(_generate_folds) | API/Backend | — | テスト期間定義の動的生成。コンストラクタ引数から計算 |
| モデルロード最適化 | API/Backend | — | ModelLoader.load_from_dir()のtrial内1回化。I/O最適化 |
| training_bet_historyキャッシュ | API/Backend | — | デフォルトパラメータでのbet history生成。BacktestEngine呼び出し |
| EV_lower閾値のOptuna注入 | API/Backend | — | Optuna params → SubmodelSet属性への動的設定 |
| 安定性検証・レポート | API/Backend | — | 3-seed結果の統計的分析計算。JSON出力 |
| RegimeDetector状態リセット | API/Backend | — | CR-01パターン。fold/trial間のmutable stateクリア |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| optuna | 4.8.0 | TPE戦略パラメータ最適化 | Phase 13から使用。TPESampler + MedianPruner [VERIFIED: pip show] |
| numpy | >=1.26 | 統計計算(mean, std, CV) | プロジェクト依存 [VERIFIED: pyproject.toml] |
| pytest | 9.0.2 | テストフレームワーク | 全テストmockベース [VERIFIED: pip show] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | >=2.2 | DataFrame操作 | テストfixture、安定性分析 |
| scipy | (transitive) | 統計検定(rank相関等) | 安定性検証の高度な分析で使用可能性 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手動CV計算 | scipy.stats.variation | scipyは追加依存不要(transitive)。CV計算はnumpy単独で十分 |

**Installation:**
```bash
# 追加インストール不要 — optuna>=3.5は既存依存
pip install -e ".[dev]"
```

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │   run_strategy_optimization.py   │
                    │   (CLI: --n-trials --seeds ...)   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      StrategyOptimizer           │
                    │                                  │
  ┌─────────────►  │  1. _suggest_params(trial)       │
  │                │     → 16 dimensions              │
  │                │                                  │
  │                │  2. _objective(trial)             │
  │                │     ├─ ModelLoader.load() (1x)   │─── trial内1回
  │                │     ├─ training_bet_history (1x) │─── trial内1回
  │                │     └─ for each fold (4x):       │
  │                │         ├─ RegimeDetector reset   │
  │  Optuna TPE    │         ├─ regime_overrides set   │
  │  Sampler       │         ├─ BacktestEngine.run()  │
  │  (16 dims)     │         └─ trial.report(roi, i)  │
  │                │                                  │
  │                │  3. _generate_folds()             │
  │                │     → [(2022), (2023), (2024),   │
  │                │        (2025)]                    │
  │                │                                  │
  └────────────────│  4. optimize()                   │
                   │     → multi-seed loop             │
                   │     → stability_report.json       │
                   └──────────────┬──────────────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
    ┌──────────▼──┐    ┌──────────▼──┐    ┌──────────▼──┐
    │  Backtest   │    │  Default    │    │  Parameter  │
    │  Engine     │    │  Strategy   │    │  Freeze     │
    │  (per fold) │    │  (params)   │    │  Protocol   │
    └─────────────┘    └─────────────┘    └─────────────┘
```

### Recommended Project Structure
```
src/tuning/
├── strategy_optimizer.py    # 主変更対象: 4fold化、モデルロード最適化、multi-seed
src/betting/
├── default_strategy.py      # EV_lower閾値マッピング追加
scripts/
├── run_strategy_optimization.py  # multi-seed CLI拡張
tests/
├── test_strategy_optimizer.py    # 新規テスト追加(4fold、EV_lower、multi-seed)
```

### Pattern 1: モデルロード最適化 (Trial内1回)
**What:** _objective()の先頭でModelLoader.load_from_dir()を1回呼び出し、全foldで共有
**When to use:** _objective()内で複数foldが同じモデルを使う場合
**Example:**
```python
def _objective(self, trial: optuna.Trial) -> float:
    params = self._suggest_params(trial)
    strategy_config = self._build_strategy_config(params)

    # D-05: モデルロード最適化 — trial内1回
    from db.model_loader import ModelLoader
    loader = ModelLoader()
    models, info = loader.load_from_dir(self.models_dir, use_ensemble_override=True)

    # D-05: training_bet_historyキャッシュ — trial内1回
    default_config = self._build_default_config()
    default_regime_overrides = default_config.get("regime_overrides")
    if default_regime_overrides:
        models.regime_detector._override_params = default_regime_overrides
    train_engine = BacktestEngine(models=models, ..., strategy_params=default_config)
    train_result = train_engine.run(info.train_start, info.train_end)
    training_bet_history = train_result.bet_history

    # 各foldでモデル共有 + RegimeDetectorリセット
    folds = self._generate_folds()
    for fold_idx, (test_start, test_end) in enumerate(folds):
        # CR-01: RegimeDetector状態リセット
        models.regime_detector._current_regime = RegimeState.CONSERVATIVE
        models.regime_detector._regime_counter = 0
        models.regime_detector._pending_regime = None
        models.regime_detector._collapsed_consecutive = 0
        # regime_overridesをOptuna値に設定
        ...
```

### Pattern 2: 動的Fold生成
**What:** コンストラクタ引数から4foldを動的生成
**When to use:** _generate_folds()のハードコード除去
**Example:**
```python
def __init__(self, ..., n_folds: int = 2, train_years: int = 4,
             test_years: int = 1, fold_start_year: int = 2024):
    self.n_folds = n_folds
    self.fold_start_year = fold_start_year
    ...

def _generate_folds(self) -> list[tuple[str, str]]:
    """D-02: 年次4fold動的生成"""
    folds = []
    for i in range(self.n_folds):
        year = self.fold_start_year + i
        folds.append((f"{year}-01-01", f"{year}-12-31"))
    return folds
```

### Pattern 3: EV_lower閾値のOptuna注入
**What:** Optuna trialで提案されたEV_lower閾値をSubmodelSet属性に設定
**When to use:** _run_single_backtest()または_objective()内でBacktestEngine呼び出し前
**Example:**
```python
# _suggest_params()に追加:
params["ev_lower_threshold_turf"] = trial.suggest_float("ev_lower_threshold_turf", 0.5, 1.5)
params["ev_lower_threshold_dirt"] = trial.suggest_float("ev_lower_threshold_dirt", 0.5, 1.5)

# モデルロード後にSubmodelSet属性に設定:
for surf_key, sm in models.submodels.items():
    if surf_key == "turf":
        sm.ev_lower_threshold_turf = params["ev_lower_threshold_turf"]
    elif surf_key == "dirt":
        sm.ev_lower_threshold_dirt = params["ev_lower_threshold_dirt"]
```

### Anti-Patterns to Avoid
- **Anti-pattern: fold毎にModelLoader.load_from_dir()を呼ぶ:** 現在の_run_single_backtest()が毎回ロードしている。trial内でモデルは不変なので1回でよい。4fold化で4倍のロードコストになる [VERIFIED: strategy_optimizer.py:119]
- **Anti-pattern: RegimeDetector状態をリセットしない:** fold 1のregime stateがfold 2にリークする。CR-01パターンで毎fold必ずリセット [VERIFIED: strategy_optimizer.py:161-165]
- **Anti-pattern: EV_lower閾値をSubmodelSetに設定せずにBacktestEngineに渡す:** get_win_candidates()はself.models.submodelsから閾値を読むため、strategy_paramsではなくSubmodelSet属性に設定する必要がある [VERIFIED: race_predictor.py:443-448]
- **Anti-pattern: training_bet_historyをfold毎に再生成:** 同じモデルとデフォルトパラメータで生成するため全fold共通。trial内1回のキャッシュで十分 [VERIFIED: CONTEXT.md D-05]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 安定性判定 | カスタム変動係数計算 | numpy.std/meanでCV | numpy演算で十分。scipy.stats不要 |
| Optuna pruning | カスタムearly stopping | MedianPruner | Optuna内蔵。n_startup_trials=5, interval_steps=1で4foldに対応 |
| JSON manifest | カスタムシリアライズ | save_strategy_manifest() | Phase 13で実装済み。SHA256改ざん検知付き |
| パラメータ→config変換 | カスタムマッピング | build_strategy_config_from_params() | Phase 16で実装済み。EV_lowerマッピングの追加のみ |

**Key insight:** 全インフラはPhase 13-16で構築済み。Phase 17は既存コンポーネントの拡張と最適化のみ。

## Common Pitfalls

### Pitfall 1: EV_lower閾値の注入経路違い
**What goes wrong:** Optuna paramsからstrategy_configにEV_lowerを含めても、BacktestEngineはそれをSubmodelSet属性から読むため反映されない
**Why it happens:** race_predictor.pyのget_win_candidates()が`self.models.submodels.get(surf_key)`から閾値を取得する設計になっているため。strategy_params dictには含まれない [VERIFIED: race_predictor.py:443-448]
**How to avoid:** EV_lower閾値はOptuna paramsから直接SubmodelSet属性に設定する。build_strategy_config_from_params()への追加は不要
**Warning signs:** EV_lower閾値がOptunaで変化しているのにベット数が変わらない

### Pitfall 2: _run_single_backtest()のモデルロード最適化でstateリーク
**What goes wrong:** モデルを1回ロードして全foldで共有すると、BacktestEngine.run()がRegimeDetector等のmutable stateを変更し、次foldに影響する
**Why it happens:** BacktestEngine.run()内でregime_detector.detect()が状態を更新する [VERIFIED: engine.py:813-816]
**How to avoid:** 各foldの開始時にCR-01パターンでRegimeDetector状態を完全リセット。4属性を初期値に戻す(_current_regime, _regime_counter, _pending_regime, _collapsed_consecutive)
**Warning signs:** fold間でROIが連続的に変化する（前foldの状態が影響）

### Pitfall 3: 4fold MedianPrunerのstep数不一致
**What goes wrong:** 2fold前提のMedianPruner設定(n_warmup_steps等)が4fold環境で不適切に動作する
**Why it happens:** fold数が2→4に増えたため、prunerのstep数(0,1,2,3)が増加。n_startup_trials=5のままでは初期5trialはpruningされず、その後のtrialでstep=0(1fold目)のROIのみでpruning判定される
**How to avoid:** MedianPrunerの設定を4fold環境に最適化。推奨: n_startup_trials=10(初期探索を十分に行う), n_warmup_steps=0, interval_steps=1, n_min_trials=1
**Warning signs:** 初期trialで過剰pruning、またはpruningが全く発生しない

### Pitfall 4: EV_lower探索範囲の不適切な設定
**What goes wrong:** EV_lower閾値の探索範囲(0.5-1.5等)が広すぎる/狭すぎる
**Why it happens:** 現在のデフォルト値(芝=0.8, ダート=0.7)からの乖離幅が不明。Phase 15のOOF分布から計算された初期値を参照する必要がある
**How to avoid:** 探索範囲は[0.5, 1.5]を推奨。現在のデフォルト値(0.7-0.8)を中心に広めに設定。TPEが適切に探索する
**Warning signs:** 最適値が範囲境界に張り付く

### Pitfall 5: multi-seed実行での不安定次元固定化の無限ループ
**What goes wrong:** 不安定次元を固定 → 再実行 → 別の次元が不安定 → 再実行...のループに陥る
**Why it happens:** 16次元のうち常にいくつかは安定性が低い可能性がある。完全な安定性は非現実的
**How to avoid:** 1回の固定化再実行のみに制限。安定性レポートで残りの不安定次元を報告し、Phase 18の最終検証で人間が判断
**Warning signs:** 再実行後も3-4次元が不安定

## Code Examples

### _suggest_params()へのEV_lower 2次元追加
```python
# Source: strategy_optimizer.py:51-81 (拡張)
def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {}

    # 既存14次元 (regime x2, DD x5, EV scaling x2, OddsBand x1)
    for regime in ("aggressive", "conservative"):
        params[f"fk_{regime}"] = trial.suggest_float(f"fk_{regime}", 0.10, 0.80)
        params[f"ev_{regime}"] = trial.suggest_float(f"ev_{regime}", 1.05, 2.00)
        params[f"edge_{regime}"] = trial.suggest_float(f"edge_{regime}", 0.03, 0.15)
    params["dd_threshold_1"] = trial.suggest_float("dd_threshold_1", 0.05, 0.20)
    params["dd_threshold_2"] = trial.suggest_float("dd_threshold_2", 0.15, 0.35)
    params["multiplier_reduced"] = trial.suggest_float("multiplier_reduced", 0.1, 0.8)
    params["rolling_window"] = trial.suggest_int("rolling_window", 200, 800)
    params["min_stay_races"] = trial.suggest_int("min_stay_races", 5, 30)
    params["target_ev"] = trial.suggest_float("target_ev", 1.05, 1.50)
    params["max_scale"] = trial.suggest_float("max_scale", 1.0, 3.0)
    params["roi_threshold"] = trial.suggest_float("roi_threshold", 0.8, 1.2)

    # D-07: EV_lower閾値 15-16次元目 (サーフェス別)
    params["ev_lower_threshold_turf"] = trial.suggest_float(
        "ev_lower_threshold_turf", 0.5, 1.5
    )
    params["ev_lower_threshold_dirt"] = trial.suggest_float(
        "ev_lower_threshold_dirt", 0.5, 1.5
    )

    return params  # 合計16次元
```

### _generate_folds()動的生成
```python
# Source: strategy_optimizer.py:220-229 (拡張)
def _generate_folds(self) -> list[tuple[str, str]]:
    """D-02: 年次4fold動的生成"""
    folds = []
    for i in range(self.n_folds):
        year = self.fold_start_year + i
        folds.append((f"{year}-01-01", f"{year}-12-31"))
    return folds
```

### Multi-seed安定性検証フロー
```python
# Source: 新規実装 — optimize()の拡張
def optimize_multi_seed(
    self,
    n_trials: int = 100,
    seeds: list[int] = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """D-08/D-09/D-10: multi-seed安定性検証"""
    if seeds is None:
        seeds = [42, 43, 44]

    seed_results: dict[int, dict] = {}
    for i, seed in enumerate(seeds):
        n = n_trials if i == 0 else n_trials // 2  # D-08: 主100 + 追加50
        result = self.optimize(n_trials=n, seed=seed)
        seed_results[seed] = result

    # 安定性分析
    stability_report = self._compute_stability_report(seed_results)

    # 不安定次元の固定化と再実行
    unstable_dims = [d for d, info in stability_report["dimensions"].items()
                     if info["is_unstable"]]
    if unstable_dims:
        fixed_report = self._optimize_with_fixed_dims(unstable_dims, n_trials, seeds)
        stability_report["reoptimization"] = fixed_report

    return stability_report
```

### 安定性レポートJSON スキーマ
```json
{
  "version": "1.0",
  "timestamp": "2026-05-06T12:00:00Z",
  "seeds": [42, 43, 44],
  "dimensions": {
    "fk_aggressive": {
      "values": [0.45, 0.47, 0.42],
      "mean": 0.447,
      "std": 0.025,
      "cv": 0.056,
      "is_unstable": false
    },
    "ev_lower_threshold_turf": {
      "values": [0.85, 0.92, 0.71],
      "mean": 0.827,
      "std": 0.106,
      "cv": 0.128,
      "is_unstable": true
    }
  },
  "best_roi_by_seed": {"42": 1.12, "43": 1.08, "44": 1.10},
  "mean_best_roi": 1.10,
  "reoptimization": null
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 2fold固定WF | 4fold動的WF | Phase 17 | 過学習リスク軽減。2022-2025の4年で評価 |
| 14次元探索 | 16次元探索 | Phase 17 D-07 | EV_lower閾値がデータ駆動で最適化される |
| 単一seed | 3-seed安定性検証 | Phase 17 D-08 | 過学習の検出と不安定次元の特定 |
| fold毎モデルロード | trial内1回ロード | Phase 17 D-05 | 実行時間短縮(約4倍) |

**Deprecated/outdated:**
- `_generate_folds()`のハードコード2fold: D-02で動的生成に置き換え
- `_run_single_backtest()`内でのモデルロード: D-05で_objective()に移動

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | EV_lower閾値の探索範囲[0.5, 1.5]が適切 | Code Examples | 範囲が狭すぎると最適値を見逃す。Phase 15のOOF閾値(芝0.8, ダート0.7)を参考に設定 |
| A2 | MedianPrunerのn_startup_trials=10が4fold環境で適切 | Pitfall 3 | 低すぎると初期trialを過剰にpruning、高すぎるとpruningが効かない |
| A3 | CV=0.20(変動係数20%)が安定/不安定の合理的な閾値 | Pattern 3 | 高すぎると不安定次元を見逃す、低すぎると過度に固定化する |
| A4 | 不安定次元の固定化は1回のみで十分 | Pitfall 5 | 複数回固定が必要な場合はPhase 18で人間が判断 |

## Open Questions

1. **EV_lower探索範囲の最適な設定**
   - What we know: 現在のデフォルト値は芝0.8、ダート0.7 (Phase 15 _compute_ev_threshold()のfallback値)
   - What's unclear: OOF分布から計算された実際の閾値がこの範囲内に収まっているか
   - Recommendation: [0.5, 1.5]を初期範囲とし、最適値が境界に張り付く場合は範囲調整

2. **安定性判定の具体的な閾値**
   - What we know: Claude's discretion (D-09)
   - What's unclear: CVベースの閾値として何%が合理的か
   - Recommendation: CV > 0.20を「不安定」とする。20%変動は実用上許容範囲の限界

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies — optuna 4.8.0, numpy, pytest全てインストール済み)

## Sources

### Primary (HIGH confidence)
- `src/tuning/strategy_optimizer.py` — 全体。現在の14次元実装、_generate_folds()ハードコード、_run_single_backtest()モデルロード
- `src/betting/default_strategy.py` — build_strategy_config_from_params()のEV_lower非対応確認
- `src/backtest/race_predictor.py:408-503` — get_win_candidates()のEV_lower閾値取得経路
- `src/domain/models.py:255-257` — SubmodelSet.ev_lower_threshold_turf/dirt属性
- `tests/test_strategy_optimizer.py` — 17テストの既存mockパターン
- `optuna 4.8.0` — MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1, n_min_trials=1) [VERIFIED: pip show + help()]

### Secondary (MEDIUM confidence)
- CONTEXT.md (Phase 13-16) — 連続する決定の履歴。全てのコードベース変更と整合していることを確認
- `config/settings.yaml` — betting_strategy設定。デフォルト値の参照元
- `config/backtest_config.yaml` — WF設定。train_years=4, test_years=1

### Tertiary (LOW confidence)
- なし — 全てのコードベース確認済み

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - optuna 4.8.0確認済み、既存インフラ活用
- Architecture: HIGH - 既存コードベースの拡張パターン、変更点を全て特定
- Pitfalls: HIGH - 5つのpitfallを実際のコードから特定、回避策を実装可能

**Research date:** 2026-05-06
**Valid until:** 2026-06-05 (stable — 既存インフラの拡張のみ)
