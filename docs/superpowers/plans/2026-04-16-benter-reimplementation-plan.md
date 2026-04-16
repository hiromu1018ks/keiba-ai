# Benter Two-Stage Place Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the double-counting edge calculation with a proper Benter (1994) two-stage combination and fix backtest settlement to use actual payouts.

**Architecture:** Train a fundamental LightGBM model WITHOUT direct odds features, then combine its predictions with market-implied probabilities via logit-space blending with bias term. Settlement uses `payfukusyopay` (actual JRA payouts) instead of `fukuoddslow` (pre-race odds).

**Tech Stack:** LightGBM, numpy, scipy.optimize, scikit-learn (IsotonicRegression), pandas, pytest

**Spec:** `docs/superpowers/specs/2026-04-16-benter-reimplementation-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/models/benter_combination.py` | BenterCombination class: logit-space blend + MLE fitting |
| Create | `tests/test_benter_combination.py` | Unit tests for BenterCombination |
| Modify | `src/models/two_stage_return_model.py:184-212` | Remove `fukuoddslow`, `tanodds` from HIT_FEATURE_COLS |
| Modify | `src/backtest/engine.py:105-119,152,189-195,413-418,675-698` | Settlement via payfukusyopay + payout_map |
| Modify | `src/backtest/race_predictor.py:28-42,120-129` | Benter edge calculation |
| Modify | `src/domain/models.py:239` | Replace `benter_lr` with `benter_combo` |
| Modify | `src/db/model_loader.py:472-492` | Load BenterCombination from JSON |
| Modify | `src/pipelines/training_pipeline.py:459-489,804-895` | Fit + save BenterCombination |
| Modify | `tests/test_backtest_engine.py` | Settlement tests + `benter_lr` → `benter_combo` migration |
| Modify | `tests/test_race_predictor.py` | Edge calculation tests + `benter_lr` → `benter_combo` migration |
| Modify | `tests/test_domain.py` | `benter_lr` → `benter_combo` migration in SubmodelSet tests |

---

### Task 1: BenterCombination Class

**Files:**
- Create: `src/models/benter_combination.py`
- Create: `tests/test_benter_combination.py`

- [ ] **Step 1: Write failing tests for BenterCombination**

Create `tests/test_benter_combination.py`:

```python
"""BenterCombination のテスト"""
from __future__ import annotations

import numpy as np
import pytest

from models.benter_combination import BenterCombination


class TestBenterCombine:
    """combine() メソッドのテスト"""

    def test_combine_returns_same_shape(self) -> None:
        """出力が入力と同じ形状の配列を返す"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        p_fund = np.array([0.3, 0.5, 0.7])
        p_market = np.array([0.2, 0.4, 0.6])
        result = combo.combine(p_fund, p_market)
        assert result.shape == p_fund.shape

    def test_combine_output_in_01(self) -> None:
        """出力が [0, 1] の範囲内"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        p_fund = np.array([0.01, 0.3, 0.5, 0.9, 0.99])
        p_market = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        result = combo.combine(p_fund, p_market)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_combine_equal_inputs_alpha_dominant(self) -> None:
        """alpha=1, beta=0 なら p_fund をそのまま返す（恒等写像）"""
        combo = BenterCombination(alpha=1.0, beta=0.0, gamma=0.0)
        p_fund = np.array([0.3, 0.5, 0.7])
        p_market = np.array([0.1, 0.2, 0.9])  # ignored
        result = combo.combine(p_fund, p_market)
        np.testing.assert_allclose(result, p_fund, atol=1e-6)

    def test_combine_equal_inputs_beta_dominant(self) -> None:
        """alpha=0, beta=1 なら p_market をそのまま返す"""
        combo = BenterCombination(alpha=0.0, beta=1.0, gamma=0.0)
        p_fund = np.array([0.1, 0.2, 0.9])  # ignored
        p_market = np.array([0.3, 0.5, 0.7])
        result = combo.combine(p_fund, p_market)
        np.testing.assert_allclose(result, p_market, atol=1e-6)

    def test_combine_equal_probs_returns_same(self) -> None:
        """p_fund == p_market なら結果も同じ（gamma=0）"""
        combo = BenterCombination(alpha=0.3, beta=0.7, gamma=0.0)
        p = np.array([0.3, 0.5, 0.7])
        result = combo.combine(p, p)
        np.testing.assert_allclose(result, p, atol=1e-6)

    def test_combine_extreme_values_no_nan(self) -> None:
        """極端な確率値 (0.001, 0.999) でも NaN を返さない"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        p_fund = np.array([0.001, 0.999])
        p_market = np.array([0.001, 0.999])
        result = combo.combine(p_fund, p_market)
        assert not np.any(np.isnan(result))

    def test_combine_bias_positive(self) -> None:
        """gamma > 0 は確率を引き上げる"""
        combo_zero = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        combo_pos = BenterCombination(alpha=0.5, beta=0.5, gamma=1.0)
        p = np.array([0.3, 0.5])
        m = np.array([0.3, 0.5])
        r_zero = combo_zero.combine(p, m)
        r_pos = combo_pos.combine(p, m)
        assert np.all(r_pos > r_zero)


class TestBenterFit:
    """fit() クラスメソッドのテスト"""

    def test_fit_returns_instance(self) -> None:
        """fit() が BenterCombination インスタンスを返す"""
        rng = np.random.default_rng(42)
        n = 1000
        p_fund = rng.uniform(0.1, 0.9, n)
        p_market = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < 0.3).astype(float)
        combo = BenterCombination.fit(p_fund, p_market, y)
        assert isinstance(combo, BenterCombination)

    def test_fit_alpha_beta_positive(self) -> None:
        """fit() の結果 alpha, beta は正の値"""
        rng = np.random.default_rng(42)
        n = 1000
        p_fund = rng.uniform(0.1, 0.9, n)
        p_market = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < 0.3).astype(float)
        combo = BenterCombination.fit(p_fund, p_market, y)
        assert combo.alpha > 0
        assert combo.beta > 0

    def test_fit_improves_log_likelihood(self) -> None:
        """fit() の結果が naive (p_fund) より対数尤度が良い"""
        rng = np.random.default_rng(42)
        n = 2000
        p_market_true = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < p_market_true * 0.5).astype(float)  # ~15% hit rate
        p_fund = p_market_true + rng.normal(0, 0.1, n)
        p_fund = np.clip(p_fund, 0.01, 0.99)
        p_market = p_market_true + rng.normal(0, 0.05, n)
        p_market = np.clip(p_market, 0.01, 0.99)

        combo = BenterCombination.fit(p_fund, p_market, y)
        p_combined = combo.combine(p_fund, p_market)

        # log-likelihood of combined vs fund-only
        def loglik(p, y):
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        ll_combined = loglik(p_combined, y)
        ll_fund = loglik(p_fund, y)
        assert ll_combined >= ll_fund - 1.0  # allow small tolerance


class TestBenterSerialization:
    """to_dict / from_dict のテスト"""

    def test_roundtrip(self) -> None:
        """to_dict → from_dict で同じパラメータを復元"""
        original = BenterCombination(alpha=0.35, beta=0.65, gamma=-0.05)
        d = original.to_dict()
        restored = BenterCombination.from_dict(d)
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta
        assert restored.gamma == original.gamma

    def test_dict_has_required_keys(self) -> None:
        """to_dict() が alpha, beta, gamma キーを持つ"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        d = combo.to_dict()
        assert "alpha" in d
        assert "beta" in d
        assert "gamma" in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benter_combination.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.benter_combination'`

- [ ] **Step 3: Implement BenterCombination**

Create `src/models/benter_combination.py`:

```python
"""Benter (1994) 第二段階ロジット合成レイヤー。

ファンダメンタルモデルの予測確率と市場の暗黙確率を最適な重みで合成する。
logit(p_combined) = alpha * logit(p_fundamental) + beta * logit(p_market) + gamma
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


class BenterCombination:
    """第二段階ロジット合成: ファンダメンタルモデル + 市場確率。

    Benter (1994) の多項ロジット合成を二項分類（複勝予測）に適応。
    バイアス項 gamma を含む（多項版は正規化定数で暗黙に持つ）。
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def combine(self, p_fund: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        """ロジット空間で確率を合成する。"""
        logit_combined = (
            self.alpha * self._logit(p_fund)
            + self.beta * self._logit(p_market)
            + self.gamma
        )
        return self._sigmoid(logit_combined)

    @classmethod
    def fit(
        cls, p_fund: np.ndarray, p_market: np.ndarray, y: np.ndarray
    ) -> BenterCombination:
        """最尤推定で alpha, beta, gamma を推定する。"""
        logit_f = cls._logit(p_fund)
        logit_m = cls._logit(p_market)
        y = np.asarray(y, dtype=float)

        def neg_log_likelihood(params: np.ndarray) -> float:
            alpha, beta, gamma = params
            logit_c = alpha * logit_f + beta * logit_m + gamma
            p_c = cls._sigmoid(logit_c)
            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
            return float(
                -np.sum(y * np.log(p_c) + (1 - y) * np.log(1 - p_c))
            )

        result = minimize(
            neg_log_likelihood,
            x0=[0.5, 0.5, 0.0],
            method="L-BFGS-B",
            bounds=[(0.01, 5.0), (0.01, 5.0), (-5.0, 5.0)],
        )
        return cls(
            alpha=float(result.x[0]),
            beta=float(result.x[1]),
            gamma=float(result.x[2]),
        )

    def to_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BenterCombination:
        return cls(alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BenterCombination:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benter_combination.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/benter_combination.py tests/test_benter_combination.py
git commit -m "feat: add BenterCombination class for logit-space probability blending"
```

---

### Task 2: Settlement Fix — Payout Map Builder

**Files:**
- Modify: `src/backtest/engine.py`
- Modify: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write failing test for payout map building**

Add to `tests/test_backtest_engine.py`:

```python
class TestBuildPayoutMap:
    """build_payout_map のテスト"""

    def test_basic_payout_map(self) -> None:
        """払戻データから正しい payout_map を構築する"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001", "R001", "R002"],
                "payfukusyoumaban1": [1, 3, 2],
                "payfukusyopay1": [150, 150, 200],
                "payfukusyoumaban2": [2, 5, 5],
                "payfukusyopay2": [200, 180, 150],
                "payfukusyoumaban3": [3, 7, 8],
                "payfukusyopay3": [300, 250, 100],
                "payfukusyoumaban4": [None, None, None],
                "payfukusyopay4": [None, None, None],
                "payfukusyoumaban5": [None, None, None],
                "payfukusyopay5": [None, None, None],
            }
        )
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        # R001, umaban=1 → 150/100 = 1.5
        assert payout_map[("R001", 1)] == pytest.approx(1.5)
        # R001, umaban=2 → 200/100 = 2.0
        assert payout_map[("R001", 2)] == pytest.approx(2.0)
        # R001, umaban=3 → 300/100 = 3.0
        assert payout_map[("R001", 3)] == pytest.approx(3.0)
        # R002, umaban=2 → 200/100 = 2.0
        assert payout_map[("R002", 2)] == pytest.approx(2.0)

    def test_missing_pay_columns_skipped(self) -> None:
        """payfukusyoumaban が NaN のエントリはスキップする"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001"],
                "payfukusyoumaban1": [1],
                "payfukusyopay1": [150],
                "payfukusyoumaban2": [None],
                "payfukusyopay2": [None],
                "payfukusyoumaban3": [None],
                "payfukusyopay3": [None],
                "payfukusyoumaban4": [None],
                "payfukusyopay4": [None],
                "payfukusyoumaban5": [None],
                "payfukusyopay5": [None],
            }
        )
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        assert ("R001", 1) in payout_map
        assert len(payout_map) == 1  # only 1 placed horse recorded

    def test_empty_payouts(self) -> None:
        """空の DataFrame は空の map を返す"""
        payouts = pd.DataFrame()
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        assert len(payout_map) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_engine.py::TestBuildPayoutMap -v`
Expected: FAIL — `ImportError: cannot import name 'build_payout_map'`

- [ ] **Step 3: Implement build_payout_map in engine.py**

Add this module-level function to `src/backtest/engine.py` (before the class definition):

```python
def build_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) → odds_multiplier のマップを構築。

    payfukusyopay は「100円あたりの円」なので、100で割って倍率に変換する。
    """
    payout_map: dict[tuple[str, int], float] = {}
    if payouts_df.empty:
        return payout_map
    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        for i in range(1, 6):
            umaban = row.get(f"payfukusyoumaban{i}")
            pay = row.get(f"payfukusyopay{i}")
            if pd.notna(umaban) and pd.notna(pay):
                try:
                    payout_map[(race_id, int(umaban))] = float(pay) / 100.0
                except (ValueError, TypeError):
                    continue
    return payout_map
```

Also add `import pandas as pd` at the top if not already present (it is — line 3).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_engine.py::TestBuildPayoutMap -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: add build_payout_map for actual payout settlement"
```

---

### Task 3: Settlement Fix — Engine Integration

**Files:**
- Modify: `src/backtest/engine.py` (lines 105-119, 152, 189-196, 413-418, 675-698)
- Modify: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write failing test for payout-based settlement**

Add to `tests/test_backtest_engine.py`:

```python
class TestPayoutSettlement:
    """確定配当ベースの精算テスト"""

    def test_settle_bet_uses_payout_map(self) -> None:
        """_settle_bet が payout_map を使用する"""
        from backtest.engine import BacktestEngine, build_payout_map
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R001",
            umaban=3,
            bet_type=BetType.PLACE,
            odds=2.5,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=2.5,
        )
        race_df = pd.DataFrame(
            {"umaban": [3], "kakuteijyuni": [2]}
        )
        # payout_map は実際の確定配当 (3.0倍) を返す
        payout_map = {("R001", 3): 3.0}
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = payout_map

        result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(300.0)  # 100 * 3.0

    def test_settle_bet_no_payout_entry(self) -> None:
        """payout_map にエントリがない場合 (馬が着外) は 0 を返す"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R001",
            umaban=5,
            bet_type=BetType.PLACE,
            odds=2.0,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=2.0,
        )
        race_df = pd.DataFrame(
            {"umaban": [5], "kakuteijyuni": [5]}
        )
        payout_map = {("R001", 3): 3.0}  # umaban=5 は着外
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = payout_map

        result = engine._settle_bet(bet, race_df)
        assert result == 0.0

    def test_settle_bet_fallback_to_odds(self) -> None:
        """payout_map にレースが存在しない場合は final_odds にフォールバック"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R999",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=1.8,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=1.8,
        )
        race_df = pd.DataFrame(
            {"umaban": [1], "kakuteijyuni": [1]}
        )
        payout_map = {}  # レース R999 のデータなし
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = payout_map

        result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(180.0)  # 100 * 1.8 (fallback)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_engine.py::TestPayoutSettlement -v`
Expected: FAIL — `AttributeError: 'BacktestEngine' has no attribute 'payout_map'`

- [ ] **Step 3: Modify _settle_bet to use payout_map**

In `src/backtest/engine.py`, modify `_settle_bet` (lines 675-698):

```python
    def _settle_bet(self, bet: Bet, race_df: pd.DataFrame) -> float:
        """ベットの結果を判定"""
        # 優先: payout_map（確定配当）から精算
        payout_key = (bet.race_id, bet.umaban)
        if hasattr(self, "payout_map") and payout_key in self.payout_map:
            # 確定配当がある = 馬が複勝圏内
            return float(bet.stake * self.payout_map[payout_key])

        # フォールバック: 従来の finish_pos + final_odds 方式
        horse = race_df[race_df["umaban"] == bet.umaban]
        if horse.empty:
            return 0.0

        finish_pos = int(horse.iloc[0]["kakuteijyuni"])
        settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds

        if bet.bet_type == BetType.PLACE:
            if 1 <= finish_pos <= 3:
                return float(bet.stake * settle_odds)
        elif bet.bet_type == BetType.WIN:
            if finish_pos == 1:
                return float(bet.stake * settle_odds)
        elif bet.bet_type == BetType.WIDE:
            if 1 <= finish_pos <= 3:
                pair_b = getattr(bet, "umaban_b", None)
                if pair_b is not None:
                    pair_horse = race_df[race_df["umaban"] == pair_b]
                    if not pair_horse.empty and int(pair_horse.iloc[0]["kakuteijyuni"]) <= 3:
                        return float(bet.stake * settle_odds)

        return 0.0
```

Also, in the `run()` method, add payout_map construction after line 152 (where `final_odds_df` is loaded):

```python
        # 確定配当マップを構築（精算用。実際の払戻金額を使用）
        payouts_df = load_payouts(self.store, start, end)
        self.payout_map = build_payout_map(payouts_df)
        logger.info("Loaded payout map: %d entries", len(self.payout_map))
```

Add import at top of file:
```python
from db.readers import load_payouts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: settle backtest bets with actual payouts (payfukusyopay) instead of fukuoddslow"
```

---

### Task 4: Remove Odds Features from Hit Model

**Files:**
- Modify: `src/models/two_stage_return_model.py:184-212`
- Modify: `tests/test_two_stage_return_model.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_two_stage_return_model.py`:

```python
class TestHitFeatureCols:
    """HIT_FEATURE_COLS に直接オッズ列が含まれないことを確認"""

    def test_no_direct_odds_features(self) -> None:
        """fukuoddslow, tanodds は HIT_FEATURE_COLS に含まれない"""
        from models.two_stage_return_model import PlaceTwoStageModel

        model = PlaceTwoStageModel()
        assert "fukuoddslow" not in model.HIT_FEATURE_COLS
        assert "tanodds" not in model.HIT_FEATURE_COLS

    def test_indirect_odds_features_present(self) -> None:
        """オッズ動態・市場構造特徴量は残っている"""
        from models.two_stage_return_model import PlaceTwoStageModel

        model = PlaceTwoStageModel()
        assert "odds_drop_rate_60_10" in model.HIT_FEATURE_COLS
        assert "market_entropy" in model.HIT_FEATURE_COLS
        assert "overround" in model.HIT_FEATURE_COLS
        assert "signed_log_error_win" in model.HIT_FEATURE_COLS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_two_stage_return_model.py::TestHitFeatureCols -v`
Expected: FAIL — `AssertionError: assert 'fukuoddslow' not in [...]`

- [ ] **Step 3: Remove fukuoddslow and tanodds from HIT_FEATURE_COLS**

In `src/models/two_stage_return_model.py`, modify lines 184-212:

Remove these two lines:
```python
    "fukuoddslow",  # 複勝オッズ (市場確率のベース)
    "tanodds",  # 単勝オッズ
```

The updated list should have 19 features (down from 21).

**Note:** `RETURN_FEATURE_COLS` (lines 216-244) is NOT modified. The return model predicts
expected payout given a hit, so it should still see odds information. Only the hit model
(probability of placing) is made odds-free to eliminate double-counting.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_two_stage_return_model.py::TestHitFeatureCols -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py
git commit -m "feat: remove fukuoddslow/tanodds from hit model features (eliminate double-counting)"
```

---

### Task 5: RacePredictor Benter Integration

**Files:**
- Modify: `src/backtest/race_predictor.py:28-42,120-129`
- Modify: `tests/test_race_predictor.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_race_predictor.py`:

```python
class TestBenterEdgeCalculation:
    """Benter合成によるエッジ計算のテスト"""

    def test_edge_uses_combined_probability(self) -> None:
        """edge_place が p_place_combined × odds - 1 で計算される"""
        from unittest.mock import MagicMock, patch
        from backtest.race_predictor import RacePredictor
        from models.benter_combination import BenterCombination
        from domain.models import SubmodelSet, TrainedModelsV5

        # モックセットアップ
        models = MagicMock(spec=TrainedModelsV5)
        benter = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        # SubmodelSet に benter_combo を持つモック
        sub = MagicMock(spec=SubmodelSet)
        sub.benter_combo = benter
        models.submodels = {"turf": sub}
        models.quality_screener = MagicMock()
        models.regime_detector = MagicMock()
        models.regime_detector.current_regime = MagicMock()

        predictor = RacePredictor(models)

        # predict 内でエッジ計算が行われるかを確認
        # (詳細なモック設定は実際の predict メソッドに依存)
        assert predictor.benter is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_race_predictor.py::TestBenterEdgeCalculation -v`
Expected: FAIL — `AttributeError: 'RacePredictor' has no attribute 'benter'`

- [ ] **Step 3: Modify RacePredictor to use BenterCombination**

In `src/backtest/race_predictor.py`:

**Constructor (modify lines 28-42):**

```python
    def __init__(
        self,
        models: TrainedModelsV5,
        *,
        stake_calculator: StakeCalculator | None = None,
        dd_controller: DrawdownController | None = None,
        alpha: float = 0.4,
    ) -> None:
        self.models = models
        self.stake_calc = stake_calculator
        self.dd_ctrl = dd_controller
        self._betting_mode = "kelly" if stake_calculator is not None else "flat"
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha  # kept for backwards compatibility / fallback
        # Benter合成レイヤー (最初のsurfaceから取得、benter_comboがあれば使用)
        first_sub = next(iter(models.submodels.values()), None)
        self.benter = first_sub.benter_combo if first_sub else None
```

**Edge calculation (modify lines 120-129):**

```python
        # Benter合成: ロジット空間でファンダメンタル + 市場確率を組み合わせ
        p_market = np.where(
            df["fukuoddslow"] > 0,
            1.0 / df["fukuoddslow"],
            np.nan,
        )
        if self.benter is not None:
            p_market_clipped = np.clip(p_market, 0.01, 0.99)
            df["p_place_combined"] = self.benter.combine(
                df["p_place_pred"].values, p_market_clipped
            )
        else:
            df["p_place_combined"] = df["p_place_pred"]
        df["p_market"] = p_market
        df["edge_place"] = df["p_place_combined"] * df["fukuoddslow"] - 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/race_predictor.py tests/test_race_predictor.py
git commit -m "feat: integrate BenterCombination into RacePredictor edge calculation"
```

---

### Task 6: SubmodelSet + ModelLoader Update

**Files:**
- Modify: `src/domain/models.py:239` (also remove `LogisticRegression` import in TYPE_CHECKING)
- Modify: `src/db/model_loader.py:472-492`
- Modify: `tests/test_model_loader.py`
- Modify: `tests/test_domain.py` (rename `benter_lr` tests to `benter_combo`)
- Modify: `tests/test_race_predictor.py` (update `_make_submodel_mock`)
- Modify: `tests/test_backtest_engine.py` (update all `benter_lr = None` to `benter_combo = None`)

- [ ] **Step 1: Write failing test**

Add to `tests/test_model_loader.py`:

```python
class TestBenterCombinationLoading:
    """BenterCombination のロードテスト"""

    def test_load_benter_combo_from_json(self, tmp_path) -> None:
        """benter_combo_{surface}.json から BenterCombination をロードする"""
        import json
        from db.model_loader import ModelLoader
        from models.benter_combination import BenterCombination

        # ダミー JSON ファイルを作成
        combo_data = {"alpha": 0.35, "beta": 0.65, "gamma": -0.05}
        combo_file = tmp_path / "benter_combo_turf.json"
        combo_file.write_text(json.dumps(combo_data), encoding="utf-8")

        # ローダーが正しくデシリアライズするかは model_loader の実装に依存するが、
        # BenterCombination.load() 自体は動作する
        loaded = BenterCombination.load(combo_file)
        assert loaded.alpha == 0.35
        assert loaded.beta == 0.65
        assert loaded.gamma == -0.05
```

- [ ] **Step 2: Update SubmodelSet**

In `src/domain/models.py` line 239, replace:

```python
    benter_lr: LogisticRegression | None = None
```

with:

```python
    benter_combo: BenterCombination | None = None
```

Add import at top of file:
```python
from models.benter_combination import BenterCombination
```

Remove `LogisticRegression` import if no longer used elsewhere in this file.

- [ ] **Step 3: Update ModelLoader**

In `src/db/model_loader.py` lines 472-492, replace the benter_lr loading section:

```python
            # Benter combination (logit-space blend)
            benter_combo = None
            benter_file = models_dir / f"benter_combo_{surface}.json"
            if benter_file.is_file():
                try:
                    from models.benter_combination import BenterCombination
                    benter_combo = BenterCombination.load(benter_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", benter_file)

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                place_ev_corrector=place_ev_corr,
                wide=wide,
                confidence=confidence,
                benter_combo=benter_combo,
            )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: All PASS

- [ ] **Step 5: Update test files for benter_lr → benter_combo migration**

Search and replace across test files:
- `tests/test_domain.py`: Rename `benter_lr` → `benter_combo` in test names and assertions.
  Tests `test_submodel_set_accepts_benter_lr` → `test_submodel_set_accepts_benter_combo`
  and `test_submodel_set_benter_lr_default_none` → `test_submodel_set_benter_combo_default_none`.
  Update `LogisticRegression` mock to `BenterCombination` mock.
- `tests/test_race_predictor.py`: In `_make_submodel_mock()`, change `sm.benter_lr = None`
  to `sm.benter_combo = None`.
- `tests/test_backtest_engine.py`: Replace all `submodel.benter_lr = None` (lines ~427, 589, 909, 1041)
  with `submodel.benter_combo = None`.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS (all benter_lr references updated)

- [ ] **Step 7: Commit**

```bash
git add src/domain/models.py src/db/model_loader.py tests/test_model_loader.py tests/test_domain.py tests/test_race_predictor.py tests/test_backtest_engine.py
git commit -m "feat: replace benter_lr with benter_combo (BenterCombination) in SubmodelSet"
```

---

### Task 7: Training Pipeline — Fit and Save BenterCombination

**Files:**
- Modify: `src/pipelines/training_pipeline.py:459-489,804-895`

- [ ] **Step 1: Add Benter fitting after place prediction**

In `src/pipelines/training_pipeline.py`, after line 483 (`df_oof = place_2s.predict_ev(df_oof)`), add:

```python
        # 5a. Benter合成: ファンダメンタル予測 + 市場確率 の最適重みを推定
        with TimingContext(f"{surface}/benter_fit"):
            from models.benter_combination import BenterCombination

            benter_combo = None
            if "fukuoddslow" in df_oof.columns:
                p_fund = df_oof["p_place_pred"].values
                p_market = np.where(
                    df_oof["fukuoddslow"] > 0,
                    1.0 / df_oof["fukuoddslow"].values,
                    np.nan,
                )
                y_place = (df_oof["kakuteijyuni"] <= 3).astype(float).values
                # NaN を除外
                valid = np.isfinite(p_fund) & np.isfinite(p_market)
                if valid.sum() > 1000:
                    benter_combo = BenterCombination.fit(
                        p_fund[valid], p_market[valid], y_place[valid]
                    )
                    logger.info(
                        "Benter %s: alpha=%.3f, beta=%.3f, gamma=%.3f",
                        surface,
                        benter_combo.alpha,
                        benter_combo.beta,
                        benter_combo.gamma,
                    )
                else:
                    logger.warning(
                        "Insufficient valid data for Benter fit (%d rows), skipping %s",
                        valid.sum(), surface,
                    )
            else:
                logger.warning(
                    "fukuoddslow not in df_oof columns, skipping Benter fit for %s",
                    surface,
                )

- [ ] **Step 2: Pass benter_combo to SubmodelSet**

In the SubmodelSet construction (around line 530-540 in `_train_submodel`), add `benter_combo`:

Find the `return SubmodelSet(...)` call and add `benter_combo=benter_combo` to it.

- [ ] **Step 3: Save BenterCombination in _save_models_local**

In `_save_models_local` method (around line 804-895), add saving logic:

After the existing model saving loop, add:

```python
            # BenterCombination を JSON で保存
            if sub.benter_combo is not None:
                benter_path = models_dir / f"benter_combo_{surface}.json"
                sub.benter_combo.save(benter_path)
                logger.info("Saved benter_combo for %s: %s", surface, sub.benter_combo.to_dict())
```

- [ ] **Step 4: Run existing training pipeline tests**

Run: `python -m pytest tests/ -v --tb=short -k "train"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "feat: fit BenterCombination on validation data during training pipeline"
```

---

### Task 8: End-to-End Validation

**Files:**
- No code changes — validation only

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --cov=src --cov-report=term-missing`
Expected: All PASS, no coverage regressions

- [ ] **Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

Run: `ruff format --check src/ tests/`
Expected: No errors

- [ ] **Step 3: Run type check**

Run: `mypy src/`
Expected: No errors (may need to add type annotations to new files)

- [ ] **Step 4: Run backtest (manual)**

This step requires actual data and takes ~57 minutes. Run after all tests pass:

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

**Expected results to verify:**
1. ROI should be significantly higher than 63.6% (settlement fix alone should show ~75%)
2. Benter alpha/beta values should be logged during training
3. Calibration table should show improvement in p_place_combined accuracy
4. Bet count may differ (edge calculation changed)

- [ ] **Step 5: Document results**

After backtest completes, update the spec's Expected Impact section with actual results.

---

## Dependencies

```
Task 1 (BenterCombination class)
  ↓
Task 4 (Remove odds features) ← independent of Task 1
Task 5 (RacePredictor integration) ← depends on Task 1
Task 6 (SubmodelSet + ModelLoader) ← depends on Task 1
  ↓
Task 7 (Training pipeline) ← depends on Tasks 4, 5, 6
  ↓
Task 8 (Validation) ← depends on all

Task 2 (Payout map builder) ← independent
Task 3 (Engine settlement) ← depends on Task 2
```

**Parallelizable:** Tasks 1, 2, 4 can run in parallel. Tasks 3, 5, 6 depend on 1/2.
