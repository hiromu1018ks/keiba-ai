# Phase 12: Stake Sizing Enhancement - Research

**Researched:** 2026-05-05
**Domain:** Stake sizing, Kelly criterion, regime-based parameter injection
**Confidence:** HIGH

## Summary

Phase 12 はバックテストパスの賭け金計算を最適化する。現在の `StakeCalculator` はハードコードされた `FRACTIONAL_KELLY=0.5` を使用しており、レジーム状態に関係なく同じKelly分数を適用している。本フェーズでは (1) レジーム別のKelly分数 (AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.00) をコンストラクタ注入可能にし、(2) EV比例乗算器 `scale = min(ev/target_ev, max_scale)` を `apply_ev_scaling()` として新設する。変更はバックテストパスのみ。

**Primary recommendation:** StakeCalculator のハードコード定数をコンストラクタ引数にリファクタリングし、regime_params から fractional_kelly を注入。EV乗算は `apply_ev_scaling()` メソッドとして追加し、Kelly → EV乗算 → DD のパイプライン順序を守る。

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Kelly stake calculation | API / Backend (StakeCalculator) | -- | 純粋な計算ロジック、状態なし |
| Regime-based parameter selection | API / Backend (RegimeDetector) | -- | レジーム検出結果に基づくパラメータ辞書生成 |
| EV scaling | API / Backend (StakeCalculator) | -- | Kelly計算結果の後処理 |
| Drawdown control (final gate) | API / Backend (DrawdownController) | -- | リスク管理の最終ゲート、変更なし |
| Pipeline orchestration | API / Backend (RacePredictor.select_bets) | -- | Kelly→EV→DDパイプラインの呼び出し順序管理 |
| Config defaults | Config (settings.yaml) | -- | デフォルト値の外部定義 |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** fractional_kelly をレジーム別に設定: AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.00
- **D-02:** KELLY_FRACTION_CAP=0.25 は固定。実効cap = 0.25 × fractional_kelly
- **D-03:** MIN_STAKE=100円, MAX_STAKE=10,000円 は維持
- **D-04:** RACE_EXPOSURE_CAP=0.02 (2%) は全レジーム共通固定
- **D-05:** target_ev=1.10, max_scale=2.0 で固定（レジーム非依存）
- **D-06:** 公式: `scale = min(ev / target_ev, max_scale)`、`stake = kelly_stake * scale`
- **D-07:** EV乗算器は StakeCalculator に `apply_ev_scaling()` メソッドとして追加
- **D-08:** Kelly → EV乗算 → DD の順序。DDを最終リスクゲート
- **D-09:** パイプライン: `kelly_stake = calc_stake(edge, odds, bankroll)` → `ev_scaled = apply_ev_scaling(kelly_stake, ev)` → `final_stake = dd_ctrl.adjust_stake(ev_scaled, bankroll)`
- **D-10:** コンストラクタ注入パターン。デフォルト値は settings.yaml から
- **D-11:** RegimeDetector.get_strategy_params() と MetaSwitcher._default_params() に fractional_kelly を追加
- **D-12:** Phase 13 Optuna最適化ではコンストラクタ引数で直接注入
- **D-13:** 変更対象はバックテストパスのみ。BettingOrchestrator, WinStrategy, PlaceStrategy は変更しない

### Claude's Discretion
- StakeCalculator.calc_stake() のリファクタリング（ハードコード → インスタンス変数）
- apply_ev_scaling() のシグネチャと返り値の型
- settings.yaml の betting_strategy section のスキーマ設計
- RegimeDetector.get_strategy_params() への fractional_kelly 追加方法
- テスト戦略（StakeCalculator単体 + RacePredictor統合）
- EV値の取得元（RacePredictor.select_bets() の DataFrameカラム）

### Deferred Ideas (OUT OF SCOPE)
None

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SIZE-01 | レジーム状態別にKelly分数が異なり、AGGRESSIVE > CONSERVATIVE > COLLAPSED(=0) の順で賭け金が計算される | StakeCalculator の FRACTIONAL_KELLY をコンストラクタ注入化 (D-01)。RegimeDetector.get_strategy_params() に fractional_kelly を追加 (D-11)。engine.py でレジーム検出後に fractional_kelly を StakeCalculator に注入 |
| SIZE-02 | 高EVベットの賭け金にEV比例乗算器 min(ev/target_ev, max_scale) が適用され、同一レジーム内でEVが高いほど賭け金が大きくなる | StakeCalculator.apply_ev_scaling() 新設 (D-07)。target_ev=1.10, max_scale=2.0 (D-05)。RacePredictor.select_bets() の Kelly→EV乗算→DD パイプラインに組み込み (D-08/D-09) |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11 | Runtime | mise.toml で固定済み (CLAUDE.md) |
| pytest | 9.x | Testing | プロジェクト標準、1184テストで実績あり |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyyaml | -- | settings.yaml 読み込み | config/settings.yaml の betting_strategy section |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手動settings.yaml参照 | Pydantic BaseSettings | 将来検討可能だが現在のYAML直接読み込みパターンで十分 |
| 外部Kelly最適化ライブラリ | 既存StakeCalculator拡張 | REQUIREMENTS.mdで「外部Kellyライブラリ導入」は明示的にOut of Scope |

**Installation:**
追加インストール不要。既存依存のみで完結。

## Architecture Patterns

### System Architecture Diagram

```
RacePredictor.select_bets()
        |
        v
RegimeDetector.get_strategy_params(regime)
   --> returns {fractional_kelly, ev_threshold, edge_threshold, ...}
        |
        v
StakeCalculator(fractional_kelly=regime_params["fractional_kelly"],
                target_ev=1.10, max_scale=2.0)
        |
        +---> calc_stake(edge, odds, bankroll, bet_type)
        |         |
        |         v
        |     Kelly fraction = edge / (odds - 1)
        |     kelly_fraction *= fractional_kelly   <-- レジーム別
        |     effective_cap = KELLY_FRACTION_CAP * fractional_kelly
        |     raw_stake = bankroll * min(kelly_fraction, effective_cap)
        |     --> kelly_stake (100円単位)
        |
        +---> apply_ev_scaling(kelly_stake, ev)
        |         |
        |         v
        |     scale = min(ev / target_ev, max_scale)
        |     ev_scaled_stake = kelly_stake * scale  <-- EV比例
        |
        v
DrawdownController.adjust_stake(ev_scaled_stake, bankroll)
        |
        v
    final_stake (100円単位、最終リスクゲート)
```

### Recommended Project Structure
```
src/
├── betting/
│   ├── stake_calculator.py     # [変更] コンストラクタ注入 + apply_ev_scaling()
│   ├── drawdown_controller.py  # [変更なし] 最終リスクゲート
│   └── meta_switcher.py        # [変更] _default_params() に fractional_kelly 追加
├── models/
│   └── regime_detector.py      # [変更] get_strategy_params() に fractional_kelly 追加
├── backtest/
│   ├── race_predictor.py       # [変更] select_bets() Kelly→EV→DD パイプライン
│   └── engine.py               # [変更] StakeCalculator に fractional_kelly 注入
└── config/
    └── settings.yaml           # [変更] betting_strategy section 追加
```

### Pattern 1: Constructor Injection for StakeCalculator
**What:** StakeCalculator のハードコード定数を `__init__` 引数に変更し、外部から注入可能にする
**When to use:** Phase 13 Optuna最適化でパラメータを直接制御する前提
**Example:**
```python
# 現在 (ハードコード)
class StakeCalculator:
    FRACTIONAL_KELLY: float = 0.5
    KELLY_FRACTION_CAP: float = 0.25
    ...

# 変更後 (コンストラクタ注入)
class StakeCalculator:
    def __init__(
        self,
        fractional_kelly: float = 0.5,
        kelly_fraction_cap: float = 0.25,
        target_ev: float = 1.10,
        max_scale: float = 2.0,
    ) -> None:
        self.fractional_kelly = fractional_kelly
        self.kelly_fraction_cap = kelly_fraction_cap
        self.target_ev = target_ev
        self.max_scale = max_scale
        # MIN_STAKE, MAX_STAKE, RACE_EXPOSURE_CAP, MIN_EDGE_THRESHOLD は固定のまま
```
[VERIFIED: コードベース確認 — src/betting/stake_calculator.py lines 25-30]

### Pattern 2: EV Scaling Pipeline (Kelly -> EV -> DD)
**What:** calc_stake() で Kelly stake を計算し、apply_ev_scaling() でEV比例拡大し、DDで最終キャップ
**When to use:** RacePredictor.select_bets() の kelly mode
**Example:**
```python
# RacePredictor.select_bets() 内の各候補ループ
stake = self.stake_calc.calc_stake(edge=edge_val, odds=odds_val, bankroll=bankroll, bet_type=BetType.WIN)
stake = self.stake_calc.apply_ev_scaling(stake, ev=ev_val)  # D-07
if self.dd_ctrl is not None:
    stake = self.dd_ctrl.adjust_stake(stake, bankroll)      # D-08: final gate
    stake = max(0, math.floor(stake / 100) * 100)
```
[VERIFIED: コードベース確認 — src/backtest/race_predictor.py lines 649-662 (win), 740-753 (place)]

### Pattern 3: Regime Params Extension
**What:** RegimeDetector.get_strategy_params() に fractional_kelly を追加。既存の ev_threshold, edge_threshold と同じパターン
**When to use:** レジーム別パラメータを追加する全ケース
**Example:**
```python
# RegimeDetector.get_strategy_params() の各レジームdictに追加:
def get_strategy_params(self, regime: RegimeState) -> dict[str, object]:
    if regime == RegimeState.AGGRESSIVE:
        return {
            "fractional_kelly": 0.50,   # <-- NEW: half-Kelly
            "ev_threshold": 1.10,
            "edge_threshold": 0.05,
            ...
        }
    elif regime == RegimeState.CONSERVATIVE:
        return {
            "fractional_kelly": 0.25,   # <-- NEW: quarter-Kelly
            "ev_threshold": 1.30,
            ...
        }
    else:  # COLLAPSED
        return {
            "fractional_kelly": 0.00,   # <-- NEW: no bet
            "ev_threshold": 1.50,
            ...
        }
```
[VERIFIED: コードベース確認 — src/models/regime_detector.py lines 185-240]

### Anti-Patterns to Avoid
- **ハードコード定数の残存:** calc_stake() 内で `self.FRACTIONAL_KELLY` を参照する箇所が残ると、コンストラクタ値が反映されない。全参照を `self.fractional_kelly` に統一する
- **EV乗算をDDの前に配置しない:** D-08で Kelly→EV→DD の順序が決定済み。EV拡大がDD制御をバイパスしないよう、DDを最終ゲートとする
- **COLLAPSEDでstake>0を返す:** fractional_kelly=0.00 なので calc_stake() は stake=0 を返すべきだが、effective_cap = 0.25 * 0.00 = 0.00 で自然にゼロになることをテストで確認
- **engine.py で StakeCalculator をレース毎に再生成しない:** レースループ内でレジームが変わるたびに StakeCalculator を作り直すのは非効率。レジーム変更時にパラメータを更新する方式を検討

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Kelly stake の100円丸め | 独自丸めロジック | 既存 calc_stake() の `math.floor(raw_stake / 100) * 100` | エッジケース（bankroll=0, odds<=1.0）が既にカバーされている |
| Race exposure 2%キャップ | 新しいキャップロジック | 既存 check_race_exposure() | 比例配分削減が実装済みでテスト済み |
| DD乗数テーブル | 新しい乗数計算 | 既存 DrawdownController | 3段階回復ロジックが複雑でテスト済み |

**Key insight:** 本フェーズは既存の安定したコンポーネント (StakeCalculator, DrawdownController) の振る舞いをパラメータ化して拡張するものであり、計算ロジックそのものの再設計ではない。

## Common Pitfalls

### Pitfall 1: StakeCalculator インスタンスのライフサイクル
**What goes wrong:** engine.py の `__init__` で `StakeCalculator()` を一度だけ作成し、その後レジームが変わっても fractional_kelly が更新されない
**Why it happens:** 現在の engine.py (line 369) では `StakeCalculator()` をコンストラクタで一度だけ生成。レースループ内でレジームが遷移しても同じインスタンスを使い続ける
**How to avoid:** レースループ内でレジーム遷移を検出した場合、StakeCalculator の fractional_kelly を更新するか、新しいインスタンスを RacePredictor に注入する。設計判断: `StakeCalculator` に `update_fractional_kelly()` メソッドを追加するか、レジーム変更時に再生成するか
**Warning signs:** レジームが AGGRESSIVE→CONSERVATIVE に遷移しても賭け金が変わらない

### Pitfall 2: EV値の取得元が不明確
**What goes wrong:** apply_ev_scaling() に渡す EV 値が間違ったカラムから取得される
**Why it happens:** win は `win_selection_ev`、place は `place_selection_ev` と列名が異なる
**How to avoid:** RacePredictor.select_bets() の各パスで正しい EV カラムを明示的に参照する。win: `row.get("win_selection_ev", 0)`, place: `row.get("place_selection_ev", 0)`
**Warning signs:** EV乗算が常に1.0になる（間違ったカラム=0を取得）

### Pitfall 3: EV乗算でMIN_STAKE未満になる
**What goes wrong:** EV < target_ev の場合 scale < 1.0 となり、kelly_stake が縮小されて MIN_STAKE=100 を下回る
**Why it happens:** scale = min(ev / target_ev, max_scale) で ev=1.05, target_ev=1.10 の場合 scale=0.955。kelly_stake=100 → 95.5 → 100円未満 → ベット却下
**How to avoid:** apply_ev_scaling() の結果を100円単位に切り捨てた後、MIN_STAKE未満の場合は0を返す。select_bets() の既存 `if stake < 100: continue` がこれを処理する
**Warning signs:** フィルタ通過した候補のベット数が大幅に減少

### Pitfall 4: テストのハードコード定数参照
**What goes wrong:** テストが `calc.FRACTIONAL_KELLY == 0.5` をアサートしており、インスタンス変数化後も class attribute を期待するテストが失敗する
**Why it happens:** test_stake_calculator.py line 94: `assert calc.FRACTIONAL_KELLY == 0.5`
**How to avoid:** テストを `calc.fractional_kelly == 0.5` に更新。デフォルトコンストラクタ `StakeCalculator()` で同じ値が得られることを確認
**Warning signs:** `test_calc_stake_fractional_kelly_constants` が AttributeError で失敗

### Pitfall 5: COLLAPSED fractional_kelly=0.0 のエッジケース
**What goes wrong:** fractional_kelly=0.0 のとき calc_stake() 内の kelly_fraction = edge/(odds-1) * 0.0 = 0.0 となり、raw_stake=0 で stake=0 を返す。これは正しい動作だが、DD controller に stake=0 が渡される点に注意
**Why it happens:** calc_stake() の `edge < self.MIN_EDGE_THRESHOLD` チェックは0.5%だが、fractional_kelly=0.0 の場合は raw_stake が直接0になる
**How to avoid:** calc_stake() のロジックで kelly_fraction が0以下の場合に早期リターンを追加するか、既存の `stake = max(0, ...)` に頼る
**Warning signs:** COLLAPSEDレジームで stake > 0 のベットが生成される

## Code Examples

### Example 1: StakeCalculator コンストラクタリファクタリング
```python
# src/betting/stake_calculator.py
class StakeCalculator:
    MIN_EDGE_THRESHOLD: float = 0.005
    MIN_STAKE: int = 100
    MAX_STAKE: int = 10000
    RACE_EXPOSURE_CAP: float = 0.02

    def __init__(
        self,
        fractional_kelly: float = 0.5,
        kelly_fraction_cap: float = 0.25,
        target_ev: float = 1.10,
        max_scale: float = 2.0,
    ) -> None:
        self.fractional_kelly = fractional_kelly
        self.kelly_fraction_cap = kelly_fraction_cap
        self.target_ev = target_ev
        self.max_scale = max_scale

    def calc_stake(self, edge: float, odds: float, bankroll: float, bet_type: BetType) -> float:
        # ... 既存ロジック ...
        kelly_fraction *= self.fractional_kelly  # was: self.FRACTIONAL_KELLY
        effective_cap = self.kelly_fraction_cap * self.fractional_kelly  # was: self.KELLY_FRACTION_CAP * self.FRACTIONAL_KELLY
        # ... 残りは変更なし ...

    def apply_ev_scaling(self, stake: float, ev: float) -> float:
        """EV比例乗算器 (D-06/D-07).

        scale = min(ev / target_ev, max_scale)
        EV < target_ev → scale < 1.0 → 縮小
        EV >= target_ev → scale >= 1.0 → 拡大 (max_scale まで)
        """
        if stake <= 0 or math.isnan(ev) or ev <= 0:
            return stake
        scale = min(ev / self.target_ev, self.max_scale)
        return stake * scale
```

### Example 2: RacePredictor.select_bets() のパイプライン変更 (win path)
```python
# src/backtest/race_predictor.py select_bets() win path
for _, row in candidates.iterrows():
    edge_val = float(row.get(edge_col, 0))
    odds_val = float(row.get("tanodds", 0))
    ev_val = float(row.get(ev_col, 0))  # win_selection_ev

    if self._betting_mode == "kelly" and self.stake_calc is not None:
        stake = self.stake_calc.calc_stake(
            edge=edge_val, odds=odds_val, bankroll=bankroll, bet_type=BetType.WIN,
        )
        stake = self.stake_calc.apply_ev_scaling(stake, ev=ev_val)  # D-07: EV乗算
        if self.dd_ctrl is not None:
            stake = self.dd_ctrl.adjust_stake(stake, bankroll)      # D-08: DD (final gate)
            stake = max(0, math.floor(stake / 100) * 100)
    else:
        stake = 100.0
```

### Example 3: engine.py での StakeCalculator 構築 (レースループ内)
```python
# src/backtest/engine.py run() 内のレースループ
# レジーム検出後:
regime_params = self.models.regime_detector.get_strategy_params(regime)
fractional_kelly = float(regime_params.get("fractional_kelly", 0.5))

# fractional_kelly が変わった場合のみ StakeCalculator を更新
if self._race_predictor.stake_calc is not None:
    self._race_predictor.stake_calc.fractional_kelly = fractional_kelly
```

### Example 4: settings.yaml への betting_strategy section 追加
```yaml
# config/settings.yaml に追加
betting_strategy:
  default_fractional_kelly: 0.5
  kelly_fraction_cap: 0.25
  target_ev: 1.10
  max_scale: 2.0
  regime_fractions:
    aggressive: 0.50
    conservative: 0.25
    collapsed: 0.00
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 固定 half-Kelly (0.5) | レジーム別Kelly (0.50/0.25/0.00) | Phase 12 | レジームに応じたリスク調整が可能に |
| 全ベット同額スケール | EV比例乗算器 | Phase 12 | 高EVベットに重点配分、低EVベットを縮小 |
| クラス属性ハードコード | コンストラクタ注入 | Phase 12 | Phase 13 Optuna最適化への準備 |

**Deprecated/outdated:**
- `StakeCalculator.FRACTIONAL_KELLY` (クラス属性): インスタンス変数 `self.fractional_kelly` に移行
- `StakeCalculator.KELLY_FRACTION_CAP` (クラス属性): インスタンス変数 `self.kelly_fraction_cap` に移行

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | settings.yaml の betting_strategy section を engine.py が直接読み込んでいる | Standard Stack | 低 -- 読み込み箇所が限定的 |
| A2 | RacePredictor.select_bets() でEV値は `win_selection_ev` / `place_selection_ev` カラムから取得可能 | Code Examples | 中 -- EVカラムがNaNの場合のフォールバックが必要 |
| A3 | MetaSwitcher._default_params() の変更はバックテストパスに影響しない（MetaSwitcherはライブパスで使用） | Pattern 3 | 低 -- CONTEXT.md D-13でバックテストパスのみ変更と明記 |

**If this table has entries:** 上記の `[ASSUMED]` 主張はコードベース調査に基づいているが、実行時検証が必要な項目がある。

## Open Questions

1. **StakeCalculator のライフサイクル設計**
   - What we know: engine.py で一度だけ作成される。レースループ内でレジームが変わる可能性がある
   - What's unclear: レジーム遷移時に StakeCalculator を再生成するか、fractional_kelly をインプレース更新するか
   - Recommendation: インプレース更新 (`stake_calc.fractional_kelly = new_value`) が最もシンプル。再生成は RacePredictor 内の stake_calc 参照の更新が必要で複雑になる

2. **EV < 1.0 の候補の処理**
   - What we know: ev_threshold フィルタで EV >= 1.10 等の候補のみが select_bets() に到達する
   - What's unclear: EV乗算器の scale = ev/1.10 で EV=1.05 の場合 scale=0.955 となるが、このような低EV候補はそもそもフィルタ段階で除外されているか
   - Recommendation: Phase 11のフィルタが先に動くので、EV乗算器に到達する候補は基本的に ev_threshold 以上。ただし、EV乗算器自体は安全のため ev <= 0 のエッジケースを処理すべき

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | Runtime | -- | -- | -- |
| PostgreSQL | バックテスト検証のみ | -- | -- | コード変更・テストには不要 |
| pytest | Test execution | -- | -- | -- |

**Note:** Phase 12 のコード変更とテストはPostgreSQL不要（全テストmock使用、CLAUDE.md記載）。バックテストROI検証（成功基準3）にはPostgreSQL環境が必要だが、これは手動検証。

## Key File Inventory

### 変更対象ファイル
| File | Current LOC | Change Type | Risk |
|------|------------|-------------|------|
| `src/betting/stake_calculator.py` | 123 | リファクタ + 新規メソッド | LOW |
| `src/models/regime_detector.py` | 245 | パラメータ追加 (3dictに1項目ずつ) | LOW |
| `src/betting/meta_switcher.py` | 72 | パラメータ追加 (3dictに1項目ずつ) | LOW |
| `src/backtest/race_predictor.py` | 922 | パイプライン変更 (2箇所) | MEDIUM |
| `src/backtest/engine.py` | ~1000 | StakeCalculator構築変更 | MEDIUM |
| `config/settings.yaml` | 37 | セクション追加 | LOW |

### テストファイル
| File | Current Tests | Change Type |
|------|--------------|-------------|
| `tests/test_stake_calculator.py` | 21 | 更新 (定数テスト修正) + 追加 (regime別Kelly, EV乗算) |
| `tests/test_race_predictor.py` | 35+ | 追加 (EV乗算統合テスト) |

## Sources

### Primary (HIGH confidence)
- コードベース確認 -- src/betting/stake_calculator.py (全行)
- コードベース確認 -- src/models/regime_detector.py (全行)
- コードベース確認 -- src/betting/meta_switcher.py (全行)
- コードベース確認 -- src/backtest/race_predictor.py (select_bets周辺)
- コードベース確認 -- src/backtest/engine.py (StakeCalculator構築箇所)
- コードベース確認 -- tests/test_stake_calculator.py (全21テスト)
- コードベース確認 -- tests/test_race_predictor.py (全テスト)
- コードベース確認 -- config/settings.yaml (全行)
- CONTEXT.md -- ユーザーの確定決定 (D-01 ～ D-13)
- REQUIREMENTS.md -- SIZE-01, SIZE-02 定義

### Secondary (MEDIUM confidence)
- CLAUDE.md -- プロジェクト規約 (テストはmock使用、コミット規約)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- 既存プロジェクトスタックの確認のみ
- Architecture: HIGH -- 全変更対象ファイルのコードベース確認済み
- Pitfalls: HIGH -- 既存テストコードとengine.pyの構造から推定

**Research date:** 2026-05-05
**Valid until:** 2026-06-05 (stable domain, low dependency churn)
