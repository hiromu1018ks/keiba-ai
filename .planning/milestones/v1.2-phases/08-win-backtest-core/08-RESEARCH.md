# Phase 8: Win Backtest Core - Research

**Researched:** 2026-05-04
**Domain:** Backtest settlement, win candidate selection, CLI dispatch
**Confidence:** HIGH

## Summary

Phase 8 extends the existing backtest infrastructure to support win (単勝) betting mode alongside the current place (複勝) mode. The core challenge is threefold: (1) constructing a correct win payout map from `paytansyoumaban1/paytansyopay1` data, (2) adding a win candidate selection path in RacePredictor, and (3) dispatching between win/place modes via a `--betting-target` CLI flag.

The existing codebase provides strong reference implementations: `build_payout_map()` for place payouts (engine.py:102-125), `get_place_candidates()` for place selection (race_predictor.py:408-525), and `WinSelectionGate.score()` which already computes `win_selection_ev/edge/prob` columns. The key insight is that the win path mirrors the place path almost exactly, with simpler payout logic (only 1st place pays out, vs top-3 for place).

**Primary recommendation:** Follow the existing place-mode patterns exactly -- add `build_win_payout_map()`, `get_win_candidates()`, and `final_win_odds_map` as symmetrical counterparts to their place equivalents. The `_settle_bet()` method has a confirmed bug where WIN bets incorrectly use the place payout_map; this must be fixed as part of WIN-01.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 決済は実際の払戻金(`paytansyopay1/100`)を使用
- **D-02:** `build_win_payout_map()`を新規追加。`(race_id, umaban) -> paytansyopay1/100` の辞書構築
- **D-03:** `final_win_odds_map`は`tanoddslow/100`を使用 [see Assumptions Log -- A1]
- **D-04:** `paytansyopay1`欠損時は`tanoddslow/100`にフォールバック、WARNINGログ出力
- **D-05:** `EveryDB2Queries.get_payouts()` SQLに`paytansyoumaban1, paytansyopay1`を追加
- **D-06:** 基本フィルタ: `win_selection_edge > 0` AND `tanoddslow >= 1.0`
- **D-07:** ランキング: `win_gate_score`降順ソート
- **D-08:** `win_gate_pass`はログのみ、フィルタに使用しない
- **D-09:** 1レース最大2頭候補
- **D-10:** `get_win_candidates()`をRacePredictorに追加
- **D-11:** `--betting-target`は排他型: `win|place|wide`
- **D-12:** デフォルト値は`win`
- **D-13:** ディスパッチはRacePredictor経由
- **D-14:** BacktestEngine.__init__()に`betting_target: str = "win"`パラメータを追加
- **D-15:** `run_wf_validation.py`は最小修正
- **D-16:** フォールド定義変更なし
- **D-17:** 過学習検出ロジック変更なし

### Claude's Discretion
- `build_win_payout_map()`の具体的な実装（payouts DataFrameからのマップ構築方法）
- `get_win_candidates()`の返り値の型（DataFrame or list of Bet objects）
- `select_bets()`へのwin path追加方法（既存メソッドの拡張 vs 新規メソッド）
- `BacktestEngine.run()`内のwin_payout_map/win_odds_map構築タイミング
- `_settle_bet()`のwin対応（BetType.WINの場合のwin_payout_map参照）
- ETL type rulesへの追加カラム(paytansyoumaban2/3, paytansyopay2/3)対応の要否
- MLflowログへの単勝ROI記録フォーマット

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WIN-01 | build_win_payout_map()で単勝払戻しデータ(tan_umaban/tan_pay)を読み取り、payout_mapを構築できる | payouts.parquetにpaytansyoumaban1/paytansyopay1あり (欠損0/38835行)。ETL type rulesにint+float定義済み |
| WIN-02 | final_odds_mapがtanoddslow(単勝オッズ)を使用し、単勝ベットの正しい決済を行える | odds_tanpuku.parquetのtanodds列が実際のJRA単勝オッズ (odds10変換済み)。final_win_odds_mapはfeat_dfからtanodds列で構築 |
| WIN-03 | get_win_candidates()がwin_selection_ev/edge/prob列で候補をフィルタリングし、単勝ベット候補を生成できる | WinSelectionGate.score()が既にwin_selection_ev/edge/prob + win_gate_scoreを計算。get_place_candidates()が参照実装 |
| WIN-04 | BacktestEngineにbetting_targetパラメータを追加し、単勝/複勝モードを切り替えられる(デフォルト=WIN) | BacktestEngine.__init__()にbetting_target追加。run()内でpayout_map/odds_map/candidate選択を分岐 |
| WIN-05 | Conformal信頼性スコア(conformal_confidence_score)を単勝ベット判定に組み込み、高信頼度ベットのみを生成できる | RobustConfidenceEstimator.predict_interval()が既にconformal_confidence_scoreをwin_dfに計算。WinSelectionGateの_score_frame_from_tables()もconfidence_edgesを使用 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Win payout map構築 | BacktestEngine | -- | 既存payout_mapと同じtier。データロード直後に構築 |
| Win final odds map構築 | BacktestEngine | -- | 既存final_odds_mapと同じtier。settlement用 |
| Win候補選択ロジック | RacePredictor | -- | 既存get_place_candidates()と対称。フィルタ/ランキング/候補生成 |
| Betting target dispatch | RacePredictor.select_bets() | BacktestEngine.run() | Engineがtarget指定、Predictorが呼び分け |
| Bet settlement (_settle_bet) | BacktestEngine | -- | payout_map/wide_payout_map参照と同じtier |
| CLI --betting-target | run_backtest.py / run_wf_validation.py | -- | argparseで受け取りEngineに渡す |
| Conformal confidence参照 | WinSelectionGate | RacePredictor | Gateのscore()がconfidence_edges使用。候補選択ではフィルタとして利用 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | installed | DataFrame処理 | 既存パイプライン全体で使用 |
| numpy | installed | 数値計算 | 既存パイプライン全体で使用 |
| LightGBM | installed | MLモデル | 既存推論パイプライン |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | installed | テスト | 全変更の検証 |
| unittest.mock | stdlib | モック | DB不要テスト (プロジェクト規約) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 新規get_win_candidates() | 既存get_place_candidates()を汎用化 | 対称性が損なわれる。D-10の通り独立メソッドが望ましい |

**Installation:**
```bash
# No new dependencies required -- all changes use existing stack
```

## Architecture Patterns

### System Architecture Diagram

```
run_backtest.py --betting-target win
        |
        v
BacktestEngine(betting_target="win")
   |
   |-- load payouts_df
   |-- build_win_payout_map(payouts_df)   [NEW] --> win_payout_map: (race_id,umaban) -> multiplier
   |-- build_payout_map(payouts_df)       [EXISTING] --> payout_map (place)
   |-- build_wide_payout_map(payouts_df)  [EXISTING] --> wide_payout_map
   |-- load odds snapshots
   |-- build final_win_odds_map           [NEW] --> (race_id,umaban) -> tanodds
   |-- build final_odds_map               [EXISTING] --> (race_id,umaban) -> fukuoddslow
   |
   |-- for each race:
   |      |
   |      |-- RacePredictor.predict()  [EXISTING: win_selection_ev/edge/prob already computed]
   |      |
   |      |-- IF betting_target == "win":
   |      |     get_win_candidates(result_df)    [NEW]
   |      |       -> filter: win_selection_edge > 0 AND tanodds >= 1.0
   |      |       -> sort: win_gate_score DESC
   |      |       -> head(2) max candidates
   |      |-- ELIF betting_target == "place":
   |      |     get_place_candidates(result_df)  [EXISTING]
   |      |
   |      |-- select_bets() -> Bet objects with BetType.WIN
   |      |
   |      |-- Set final_odds on Bet: final_win_odds_map lookup
   |      |
   |      |-- _settle_bet(bet)
   |            IF BetType.WIN:
   |              1. Check win_payout_map -> return stake * win_payout
   |              2. Fallback: finish_pos == 1 -> return stake * final_odds
   |              3. Else: return 0.0
```

### Recommended Project Structure
```
src/
├── backtest/
│   ├── engine.py              # build_win_payout_map() [NEW], BacktestEngine mods
│   └── race_predictor.py      # get_win_candidates() [NEW], select_bets() win path [NEW]
├── db/
│   └── everydb2_queries.py    # get_payouts() SQL update [D-05]
scripts/
├── run_backtest.py            # --betting-target arg [NEW]
└── run_wf_validation.py       # --betting-target arg [NEW, minimal]
```

### Pattern 1: Win Payout Map (mirrors build_payout_map)
**What:** Construct (race_id, umaban) -> payout_multiplier from payouts DataFrame
**When to use:** Once per backtest run, after loading payouts data
**Example:**
```python
# Source: engine.py build_payout_map() pattern (lines 102-125)
# Win version: simpler -- only 1 slot (paytansyoumaban1/paytansyopay1), no loop needed

def build_win_payout_map(payouts_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) -> odds_multiplier のマップを構築。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    """
    win_payout_map: dict[tuple[str, int], float] = {}
    if payouts_df.empty:
        return win_payout_map
    # Win payout has only 1 slot (1st place), unlike place's 5 slots
    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        umaban = row.get("paytansyoumaban1")
        pay = row.get("paytansyopay1")
        if pd.notna(umaban) and pd.notna(pay):
            try:
                key = (race_id, int(umaban))
                val = float(pay) / 100.0
                win_payout_map[key] = val
            except (ValueError, TypeError):
                continue
    return win_payout_map
```

### Pattern 2: Win Candidates Selection (mirrors get_place_candidates)
**What:** Filter and rank win bet candidates from prediction results
**When to use:** Each race during backtest, when betting_target == "win"
**Example:**
```python
# Source: race_predictor.py get_place_candidates() pattern (lines 408-525)
# Win version: uses win_selection_ev/edge/prob + tanodds + win_gate_score

def get_win_candidates(self, race_df: pd.DataFrame) -> pd.DataFrame:
    edge_col = "win_selection_edge"
    prob_col = "win_selection_prob"
    odds_col = "tanodds"  # actual JRA win odds (post /10 conversion)

    if edge_col not in race_df.columns or odds_col not in race_df.columns:
        return race_df.iloc[0:0].copy()

    selection_edge = pd.to_numeric(race_df[edge_col], errors="coerce")
    odds = pd.to_numeric(race_df[odds_col], errors="coerce")

    # D-06: Basic filter
    mask = selection_edge.fillna(0.0) > 0.0
    mask &= odds.fillna(0.0) >= 1.0

    candidates = race_df.loc[mask].copy()

    # D-07: Rank by win_gate_score DESC
    if "win_gate_score" in candidates.columns:
        candidates = candidates.sort_values(
            ["win_gate_score", edge_col],
            ascending=[False, False],
        )
    else:
        candidates = candidates.sort_values([edge_col], ascending=[False])

    # D-09: Max 2 candidates per race
    return candidates.head(2)
```

### Pattern 3: _settle_bet WIN Fix
**What:** Fix WIN settlement to use win_payout_map instead of place payout_map
**When to use:** When bet.bet_type == BetType.WIN
**Example:**
```python
# Source: engine.py _settle_bet() (lines 909-950)
# Current bug: WIN falls through to place payout_map at line 933-934

# Fix: Add WIN payout_map check before the shared payout_map block
def _settle_bet(self, bet: Bet, race_df: pd.DataFrame) -> float:
    # ... WIDE block unchanged ...

    # WIN: use win_payout_map (actual JRA win payout)
    if bet.bet_type == BetType.WIN:
        win_key = (bet.race_id, bet.umaban)
        if hasattr(self, "win_payout_map") and win_key in self.win_payout_map:
            return float(bet.stake * self.win_payout_map[win_key])
        # Fallback: finish_pos == 1 check
        horse = race_df[race_df["umaban"] == bet.umaban]
        if horse.empty:
            return 0.0
        finish_pos = int(horse.iloc[0]["kakuteijyuni"])
        if finish_pos == 1:
            settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds
            return float(bet.stake * settle_odds)
        return 0.0

    # PLACE: payout_map (existing logic)
    payout_key = (bet.race_id, bet.umaban)
    if hasattr(self, "payout_map") and payout_key in self.payout_map:
        return float(bet.stake * self.payout_map[payout_key])
    # ... rest unchanged ...
```

### Anti-Patterns to Avoid
- **Anti-pattern 1: Using place payout_map for WIN bets.** The current `_settle_bet()` at line 933-934 lets WIN bets fall through to `self.payout_map` which contains fukushou payouts. This returns incorrect amounts for WIN bets where the horse finished 1st-3rd but the place payout differs from the win payout. Must add explicit WIN branch BEFORE the shared payout_map lookup.
- **Anti-pattern 2: Confusing tanodds with tanoddslow.** The Parquet column name is `tanodds` (from odds_tanpuku.parquet via ETL odds10 rule). The WinSelectionGate code references `tanoddslow` -- these are the SAME column because the ETL names it `tanodds` but the gate's training data uses the alias `tanoddslow`. When building `final_win_odds_map`, use `tanodds` from the feature DataFrame.
- **Anti-pattern 3: Not constructing final_win_odds_map for settlement.** The current code constructs `final_odds_map` from `fukuoddslow` (line 276-281). A parallel `final_win_odds_map` from `tanodds` is needed for WIN bet settlement. Without this, WIN bets fall through to odds-based settlement which is less accurate than actual payout data.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Win payout lookup | Custom payout calculation from odds | win_payout_map from paytansyopay1/100 | JRA actual payout != odds*stake due to round-down. Actual payout data is authoritative |
| Win candidate filtering | Custom edge/prob threshold logic | WinSelectionGate.score() output (win_gate_score, win_selection_ev/edge/prob) | Already computed in predict() pipeline. Reuse existing columns |
| Conformal confidence | New confidence calculation | conformal_confidence_score from RobustConfidenceEstimator.predict_interval() | Already computed in predict() at race_predictor.py:150-153. Available in result_df |

**Key insight:** The win inference chain is already complete in the predict() pipeline. WinSelectionGate.score() computes win_selection_ev/edge/prob and win_gate_score. RobustConfidenceEstimator computes conformal_confidence_score. The missing pieces are only the settlement layer (win_payout_map) and the candidate selection layer (get_win_candidates).

## Common Pitfalls

### Pitfall 1: WIN bets using place payout_map (CONFIRMED BUG)
**What goes wrong:** `_settle_bet()` at line 931-934 has a comment "複勝/単勝: payout_map" but the shared payout_map block uses place payouts (`payfukusyoumaban/payfukusyopay`). WIN bets that hit (finish==1) get paid the place dividend instead of the win dividend.
**Why it happens:** The code was written for place-only mode. WIN branch was added later (lines 946-948) as a fallback but the primary path (lines 933-934) still uses the place map.
**How to avoid:** Add explicit WIN branch with win_payout_map lookup BEFORE the shared payout_map block. Return early for WIN type.
**Warning signs:** If WIN ROI is suspiciously similar to place ROI in backtest results, the settlement is likely wrong.

### Pitfall 2: tanodds vs tanoddslow naming confusion
**What goes wrong:** CONTEXT.md refers to "tanoddslow" but the actual Parquet column is "tanodds". If code tries to read "tanoddslow" from feat_df or entries_df, it gets NaN/key errors.
**Why it happens:** Historical naming inconsistency. ETL stores it as "tanodds" (from EveryDB2's tanoddslow field, divided by 10). WinSelectionGate training code uses "tanoddslow" as the column name because the training pipeline renames it.
**How to avoid:** Use "tanodds" when accessing Parquet/feature data. The WinSelectionGate internally handles its own column mapping. For get_win_candidates(), use whatever column name is present in the race_df passed to it.
**Warning signs:** KeyError on "tanoddslow" or all-NaN win odds.

### Pitfall 3: Missing final_win_odds_map for Bet.final_odds
**What goes wrong:** WIN bets created without proper final_odds assignment. The existing code at line 601-609 assigns final_odds from `final_odds_map` (fukuoddslow-based). For WIN bets, this assigns place odds to the win bet.
**Why it happens:** The final_odds assignment loop doesn't distinguish bet types.
**How to avoid:** Construct `final_win_odds_map` from `tanodds` in parallel with `final_odds_map`. In the Bet final_odds assignment loop, check bet.bet_type and use the appropriate map.
**Warning signs:** WIN bet final_odds values look like place odds (typically 1.0-3.0 range instead of 1.0-100+).

### Pitfall 4: paytansyoumaban1 as Int64 vs int comparison
**What goes wrong:** The ETL type rule converts paytansyoumaban1 to `Int64` (nullable integer). Comparing with Python `int` or using `int()` conversion on NaN throws errors.
**Why it happens:** pandas Int64 type has different NaN handling than regular int.
**How to avoid:** Always check `pd.notna(umaban)` before `int(umaban)`, as done in build_payout_map() pattern.
**Warning signs:** TypeError or ValueError in payout map construction.

### Pitfall 5: Conformal confidence not yet trained for win
**What goes wrong:** If RobustConfidenceEstimator is not calibrated (e.g., first run), `conformal_confidence_score` is set to 0.0 for all horses (see robust_confidence_estimator.py:121). Filtering on this score would eliminate all candidates.
**Why it happens:** The estimator requires calibration data from the training pipeline.
**How to avoid:** Do NOT use conformal_confidence_score as a hard filter in get_win_candidates(). Use it as a ranking signal or soft bonus, consistent with D-08 (gate pass is log-only, not a filter).
**Warning signs:** Zero win candidates generated despite positive win_selection_edge.

## Code Examples

### build_win_payout_map() -- Full Implementation
```python
# Verified against payouts.parquet: paytansyoumaban1=Int64, paytansyopay1=float, both 0 nulls in 38835 rows
# paytansyopay1 values: "100円あたりの円" (e.g., 240.0 = 2.4x multiplier)
# paytansyoumaban1: umaban of 1st-place finisher

def build_win_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) -> odds_multiplier のマップを構築 (単勝用)。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    単勝は1着のみ払戻しがあるため、ループは1回のみ (placeは5スロット)。
    """
    win_payout_map: dict[tuple[str, int], float] = {}
    if payouts_df.empty:
        return win_payout_map
    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        umaban = row.get("paytansyoumaban1")
        pay = row.get("paytansyopay1")
        if pd.notna(umaban) and pd.notna(pay):
            try:
                key = (race_id, int(umaban))
                val = float(pay) / 100.0
                win_payout_map[key] = val
            except (ValueError, TypeError):
                continue
    return win_payout_map
```

### final_win_odds_map Construction
```python
# Mirrors engine.py lines 276-281 (final_odds_map from fukuoddslow)
# Uses tanodds from feat_df (merged from odds_tanpuku.parquet by FeatureEngine)

final_win_odds_map: dict[tuple[str, int], float] = {}
if not feat_df.empty and "tanodds" in feat_df.columns:
    for _, r in feat_df.iterrows():
        key = (str(r["race_id"]), int(r["umaban"]))
        if pd.notna(r.get("tanodds")):
            final_win_odds_map[key] = float(r["tanodds"])
```

### _settle_bet() WIN Fix
```python
# Insert WIN branch BEFORE the shared payout_map block (line 931)
# Current code structure: WIDE check -> shared payout_map -> fallback
# New structure: WIDE check -> WIN check -> PLACE payout_map -> fallback

# WIN: win_payout_map (actual JRA win payout from paytansyopay1/100)
if bet.bet_type == BetType.WIN:
    win_key = (bet.race_id, bet.umaban)
    if hasattr(self, "win_payout_map") and win_key in self.win_payout_map:
        return float(bet.stake * self.win_payout_map[win_key])
    # D-04: Fallback to final_odds if no payout data
    logger.warning(
        "Win payout missing for %s umaban=%d, using odds fallback",
        bet.race_id, bet.umaban,
    )
    horse = race_df[race_df["umaban"] == bet.umaban]
    if horse.empty:
        return 0.0
    finish_pos = int(horse.iloc[0]["kakuteijyuni"])
    if finish_pos == 1:
        settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds
        return float(bet.stake * settle_odds)
    return 0.0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Place-only backtest | Win/Place/Wide mode dispatch | Phase 8 (v1.2) | Engine needs betting_target param |
| _settle_bet shared path for WIN+PLACE | Separate WIN/PLACE settlement | Phase 8 (v1.2) | Fix bug where WIN uses place payout |
| No win payout map | win_payout_map from paytansyopay1 | Phase 8 (v1.2) | Accurate WIN settlement |

**Deprecated/outdated:**
- `_settle_bet()` lines 931-934 treating WIN same as PLACE: must be split into separate branches

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | CONTEXT.mdの「tanoddslow」はParquet列「tanodds」と同一。odds_tanpuku.parquetに格納され、ETLのodds10ルール(÷10)で実際のJRA単勝オッズに変換済み。「tanoddslow/100」の記述は、paytansyopay1形式(100円あたり円)との混同の可能性。実際はtanodds列はすでに実際のオッズ倍率(例: 2.4) | Standard Stack, Architecture Patterns, Pitfalls | 如果「tanoddslow」实际是一个不同列名或不同数据源，final_win_odds_map構築で誤った列を参照する |
| A2 | WinSelectionGate.score()が推論時に正しくwin_gate_scoreを計算し、race_dfに含まれる。predict()パイプライン(race_predictor.py:130-143)でensure_win_selection_columns()とgate.score()が呼ばれることで保証される | WIN-03, WIN-05 | gateが未学習の場合、win_gate_scoreがNaNになりランキング不可 |
| A3 | conformal_confidence_scoreはpredict()内で計算され、result_dfに含まれる。WIN-05ではこれをsoft ranking signalとして使用するが、hard filterとしては使用しない(D-08と同様の方針) | WIN-05 | scoreが0.0の場合(未キャリブレーション)、hard filterだと全候補が除外される |

## Open Questions

1. **tanodds column availability in feat_df**
   - What we know: `tanodds` is merged from odds_tanpuku.parquet into feat_df by FeatureEngine (feature_engine.py:133). It is available in the feature DataFrame.
   - What's unclear: Whether `tanodds` is preserved through all the feature engineering transforms (merge, drop, etc.) in the race loop.
   - Recommendation: Verify that `tanodds` exists in `feat_df` after `build_all()`. The final_win_odds_map construction should use feat_df as the source, consistent with how final_odds_map uses fukuoddslow from final_odds_df (a different source). Consider using feat_df for both since tanodds is already merged there.

2. **select_bets() vs new select_win_bets()**
   - What we know: D-13 says dispatch is via RacePredictor. select_bets() currently generates Place + Wide bets.
   - What's unclear: Whether to extend select_bets() with a win branch or create a separate select_win_bets() method.
   - Recommendation: Extend select_bets() with a betting_target check -- this is simpler and follows the existing pattern. The method already takes candidates as parameter, so the dispatch happens upstream in BacktestEngine.run() when choosing between get_win_candidates() and get_place_candidates().

3. **run_wf_validation.py --betting-target integration**
   - What we know: D-15 says minimal modification. The script creates BacktestEngine at lines 174 and 188.
   - What's unclear: Whether the --betting-target flag should be passed through to BacktestEngine or if it should also affect the training pipeline.
   - Recommendation: Only pass to BacktestEngine. Training pipeline is unaffected (same models for both win and place prediction). Add argparse argument with default="win" and pass to engine constructors.

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies identified -- all changes are code-only within existing Python/pandas stack)

## Sources

### Primary (HIGH confidence)
- `src/backtest/engine.py` -- build_payout_map() (lines 102-125), _settle_bet() (lines 909-950), BacktestEngine.run() (lines 210-820)
- `src/backtest/race_predictor.py` -- get_place_candidates() (lines 408-525), select_bets() (lines 532-642), predict() (lines 51-222)
- `src/models/win_selection_gate.py` -- score() (line 982), ensure_win_selection_columns() (line 33), build_win_selection_ev() (line 19)
- `src/models/robust_confidence_estimator.py` -- predict_interval() (line 96), conformal_confidence_score computation (line 216)
- `src/db/everydb2_queries.py` -- get_payouts() SQL (lines 267-274)
- `src/db/etl.py` -- _TABLE_TYPE_RULES (lines 111-114)
- `scripts/run_backtest.py` -- CLI parser (lines 61-88), BacktestEngine construction (lines 314-319)
- `scripts/run_wf_validation.py` -- fold definitions (lines 45-58), engine construction (lines 174, 188)
- `data/raw/payouts.parquet` -- verified: paytansyoumaban1 (Int64), paytansyopay1 (float), 0 nulls in 38835 rows
- `data/odds/odds_tanpuku.parquet` -- verified: tanodds (float, actual odds post /10), fukuoddslow (float)

### Secondary (MEDIUM confidence)
- CONTEXT.md canonical_refs section -- line number references for all modification points

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, all code uses existing pandas/numpy/LightGBM
- Architecture: HIGH - patterns are direct mirrors of existing place implementations
- Pitfalls: HIGH - confirmed actual bug in _settle_bet(), verified data format in parquet files

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (stable codebase, no dependency changes expected)
