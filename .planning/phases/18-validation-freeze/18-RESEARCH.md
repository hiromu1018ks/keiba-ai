# Phase 18: Validation & Freeze - Research

**Researched:** 2026-05-06
**Domain:** バックテスト検証、パラメータ改ざん検知、検証レポート生成
**Confidence:** HIGH

## Summary

Phase 18は、Phase 14-17で構築したアンサンブルモデル+再キャリブレーション済みフィルター群+Optuna最適化済みパラメータの最終検証フェーズである。VAL-01(ROI>100%確認)とVAL-02(ParameterFreezeProtocol固定)の2要件を満たす。

コード調査の結果、Phase 17の`StrategyOptimizer.optimize()`が既に`save_strategy_manifest()`を自動呼び出ししており、`run_backtest.py`も既に`--strategy-manifest`引数と`_load_strategy_params()`関数を実装済みであることが判明した。したがってPhase 18の新規実装は(1)BacktestEngine.run()内でのPFP freeze/verify二重検証、(2)検証結果JSONの出力、(3)ROI<100%時の原因分析レポートの3点に集約される。

**Primary recommendation:** 既存の`--strategy-manifest`+`_load_strategy_params()`インフラを活用し、BacktestEngine.run()の先頭でmanifest読込+PFP検証を行い、run()の最後で検証結果JSONを`data/validation/`に出力する設計を採用する。ROI<100%時はbet_historyからオッズバンド別/レジーム別/EV診断別の原因分析を自動生成する。

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Manifest SHA256検証 | API/Backend (engine.py) | -- | バックテスト実行の整合性保証はエンジン層の責務 |
| PFP freeze/verify | API/Backend (engine.py) | -- | OOS期間中のモデル不変性はエンジン内で保証 |
| 検証結果JSON出力 | API/Backend (engine.py) | -- | バックテスト結果の一部として統合 |
| ROI判定+原因分析 | API/Backend (engine.py) | -- | bet_history集計はエンジンが保有 |
| CLI引数拡張 | CLI (run_backtest.py) | -- | --manifest引数は既に実装済み |
| テスト | Test層 | -- | mockベースの単体テスト |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** run_backtest.pyに--manifest引数を追加し、manifest読込->SHA256検証->パラメータ注入->backtest実行を一本化する。新規スクリプトは作成しない。
- **D-02:** manifestパラメータの注入はBacktestEngine.run()内で行う。run()の先頭でmanifestを読み込み、verify_strategy_manifest()でSHA256照合後にstrategy_paramsにマージしてrace loopを開始する。
- **D-03:** SHA256 + ParameterFreezeProtocol二重検証を採用。(1) verify_strategy_manifest()でmanifest SHA256照合、(2) ParameterFreezeProtocol.freeze()/verify()でOOS期間中のモデル不変性を保証。
- **D-04:** manifest検証失敗時はRuntimeErrorで即時エラー停止。SHA256不一致やPFP verify失敗の場合はバリデーションを続行しない。
- **D-05:** バリデーション対象期間: テスト2024-2025の2年。学習2020-2023に対して完全OOS。Phase 17 Optuna最適化の学習期間と整合。
- **D-06:** 成功基準の解釈: テスト期間全体のROI>100%かつ合計100+ベット。「年間」を「テスト期間全体」として解釈。年別内訳は参考情報として記録するが、パス/フェイルの判定には使用しない。
- **D-07:** 最小限の検証レポートを採用。backtest_result.json + ParameterFreezeProtocol検証結果(PASS/FAIL)を組み合わせた単一レポート。Phase 14-17の診断結果は既に各フェーズで出力済みなので再構成しない。
- **D-08:** レポートはJSON形式でdata/validation/に出力。プログラムで読み取り可能。PFP検証結果、ROI、ベット数、テスト期間、年別内訳を含む。
- **D-09:** テスト方針: mockベースの自動テスト + Human UAT。manifest読込、SHA256検証、PFP freeze/verifyの単体テストを追加。実データでのバリデーション実行はHuman UAT。
- **D-10:** ROI<100%の場合はフェーズ不完了とし、改善提案を文書化してユーザー判断を待つ。自動再調整ループは過学習リスクが高いため組み込まない。
- **D-11:** ROI<100%時の原因分析レポートを自動生成する。オッズバンド別ROI、レジーム別ROI、EV診断の過大/過小評価、ベット数不足等の分析を含む。

### Claude's Discretion
- BacktestEngine.run()内でのmanifest読込ロジックの具体的な実装
- 検証結果JSONのスキーマ設計
- 原因分析レポートの具体的な分析項目と出力形式
- --manifestと--ensemble/--years引数の組み合わせバリデーション
- テストのfixtureデータの内容
- PFP freeze/verifyのタイミング(run()のどこでfreezeし、どこでverifyするか)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAL-01 | アンサンブルバックテストで年間100+ベットかつROI>100%を達成することを確認する | BacktestEngine.run()がbet_history + BacktestResultを返す。bet_historyのrace_date列で年別集計可能。BacktestResult.total_roi / total_betsで判定 |
| VAL-02 | ParameterFreezeProtocolで最適化済みパラメータを固定し、SHA256改ざん検知を適用する | PFP freeze()/verify() + save_strategy_manifest()/verify_strategy_manifest()が完備。二重検証(D-03)の注入ポイントはrun()先頭 |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| (既存依存のみ) | -- | -- | 新規依存関係なし。Python標準ライブラリ(hashlib, json) + 既存srcモジュールで完結 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | 標準 | テスト用モック | 全テスト(既存パターン) |
| pytest | 既存 | テストフレームワーク | テスト実行 |
| pandas | 既存 | bet_history集計 | 原因分析レポート用 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| --strategy-manifest (既存) | 新規スクリプト作成 | D-01決定で既存拡張を_lock。新規スクリプトは保守コスト増 |

**Installation:**
```bash
# 新規依存なし — 既存環境で完結
```

## Architecture Patterns

### System Architecture Diagram

```
run_backtest.py (--ensemble --strategy-manifest PATH)
     |
     v
_load_strategy_params(path)  ──>  verify_strategy_manifest()
     |                              SHA256照合 + params返却
     v                              (ValueError on mismatch)
_build_strategy_config_from_manifest(params)
     |
     v
BacktestEngine(models, strategy_params=config)
     |
     v
engine.run(test_start, test_end)
     |
     +--> [NEW] manifest_path受け取り (コンストラクタ引数)
     +--> [NEW] PFP freeze() -- モデル不変性スナップショット
     +--> [NEW] verify_strategy_manifest() -- SHA256再検証 (D-03二重検証)
     |
     ... 既存race loop ...
     |
     +--> [NEW] PFP verify() -- モデル不変性確認
     +--> [NEW] 検証結果JSON出力 (data/validation/)
     +--> [NEW] ROI<100%時は原因分析レポートも出力
     |
     v
BacktestResult + validation_report.json
```

### Recommended Project Structure
```
src/backtest/
├── engine.py                          # [変更] run()にPFP検証+検証結果出力を追加
├── parameter_freeze_protocol.py       # [変更なし] 既存APIをそのまま使用
├── validation_report.py               # [新規] 検証結果JSON + 原因分析レポート生成
scripts/
├── run_backtest.py                    # [変更なし] --strategy-manifestは既に実装済み
tests/
├── test_backtest_validation.py        # [新規] VAL-01/VAL-02検証テスト
data/
├── validation/                        # [新規] 検証結果JSON出力先
```

### Pattern 1: BacktestEngine.run()へのPFP統合パターン
**What:** run()の先頭でPFP freeze()を実行し、run()の最後(BacktestResult返却前)でverify()を実行する
**When to use:** manifest検証付きバックテスト実行時
**Example:**
```python
# engine.py run() メソッド内 (D-02, D-03)

def run(self, test_start, test_end, training_bet_history=None):
    # --- NEW: PFP二重検証 (D-03) ---
    if self._manifest_path is not None:
        # (1) SHA256照合
        from backtest.parameter_freeze_protocol import verify_strategy_manifest
        verified_params = verify_strategy_manifest(self._manifest_path)
        logger.info("Manifest SHA256 verified OK")

        # (2) PFP freeze -- モデル不変性スナップショット
        self._pfp = ParameterFreezeProtocol(self.models)
        self._pfp.freeze()

    # ... 既存のrace loop ...

    # --- NEW: PFP verify (BacktestResult返却前) ---
    pfp_result = None
    if self._pfp is not None:
        pfp_result = self._pfp.verify()
        if not pfp_result["passed"]:
            raise RuntimeError(pfp_result["message"])  # D-04

    # --- NEW: 検証結果出力 ---
    ...
    return backtest_result
```

### Pattern 2: 検証結果JSON出力パターン
**What:** バックテスト完了後に検証結果をJSONで`data/validation/`に出力する
**When to use:** run()完了時(常時出力、manifest有無に関わらず)

**検証結果JSON schema (推奨):**
```json
{
  "validation_timestamp": "2026-05-06T14:30:00Z",
  "test_period": ["2024-01-01", "2025-12-31"],
  "train_period": ["2020-01-01", "2023-12-31"],
  "manifest": {
    "path": "data/tuning/strategy_manifest.json",
    "sha256_verified": true,
    "sha256_hash": "abc123..."
  },
  "pfp_verification": {
    "passed": true,
    "message": "Parameters unchanged (Rule 7 OK)"
  },
  "roi": {
    "total_roi": 1.05,
    "total_bets": 450,
    "total_stake": 45000,
    "total_return": 47250,
    "target_roi": 1.0,
    "target_bets": 100,
    "passed": true
  },
  "yearly_breakdown": {
    "2024": {"roi": 1.08, "bets": 230, "stake": 23000, "return": 24840},
    "2025": {"roi": 1.02, "bets": 220, "stake": 22000, "return": 22440}
  },
  "validation_result": "PASS"
}
```

### Pattern 3: ROI<100%原因分析パターン
**What:** ROI<100%時にbet_historyから自動で原因分析を生成する
**When to use:** total_roi < 1.0 の場合のみ

**分析項目 (D-11):**
```python
# 原因分析レポートの構造
{
  "cause_analysis": {
    "odds_band_roi": {        # オッズバンド別ROI
      "1.0-2.0": {"roi": 0.85, "bets": 50},
      "2.0-5.0": {"roi": 1.12, "bets": 120},
      "5.0-10.0": {"roi": 0.72, "bets": 80},
      "10.0+": {"roi": 0.45, "bets": 50}
    },
    "regime_roi": {           # レジーム別ROI
      "AGGRESSIVE": {"roi": 1.15, "bets": 40},
      "CONSERVATIVE": {"roi": 0.88, "bets": 200},
      "COLLAPSED": {"bets_skipped": 60}
    },
    "ev_diagnosis": {         # EV過大/過小評価
      "overestimated_ev": {"count": 150, "avg_ev": 1.30, "actual_roi": 0.75},
      "underestimated_ev": {"count": 50, "avg_ev": 0.90, "actual_roi": 1.25}
    },
    "bet_count_sufficiency": {
      "total": 250,
      "target": 100,
      "sufficient": true
    },
    "surface_roi": {          # 芝/ダート別
      "turf": {"roi": 1.05, "bets": 180},
      "dirt": {"roi": 0.82, "bets": 70}
    }
  }
}
```

### Anti-Patterns to Avoid
- **Anti-pattern: run()内で学習パラメータを再最適化:** D-10決定により自動再調整ループは禁止。ROI<100%時は文書化してユーザー判断を待つ
- **Anti-pattern: 新規CLIスクリプトの作成:** D-01決定によりrun_backtest.pyの拡張のみ。新規スクリプトは保守コスト増
- **Anti-pattern: PFP verify失敗時のサイレント続行:** D-04決定によりRuntimeErrorで即時停止。検証失敗を隠蔽しない

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SHA256ハッシュ計算 | hashlib直接呼び出し | verify_strategy_manifest() | sort_keys+indentの一貫性保証済み |
| パラメータ改ざん検知 | カスタム検証ロジック | ParameterFreezeProtocol.freeze()/verify() | pickleシリアライズ+SHA256で全モデル状態をカバー |
| manifest JSON生成 | 手動JSON出力 | save_strategy_manifest() | SHA256計算+JSON保存がアトミック |
| Optuna params -> engine config変換 | 手動dict変換 | build_strategy_config_from_params() | DDConfig制約(dd_threshold_2 > dd_threshold_1)等の検証済み |

**Key insight:** Phase 13-17で構築したPFP/manifest/default_strategyインフラが完備しているため、Phase 18は「グルー コード」の役割に過ぎない。新しい暗号化・検証・変換ロジックは不要。

## Runtime State Inventory

> このフェーズはrename/refactor/migrationではないため、本セクションはスキップする。

## Common Pitfalls

### Pitfall 1: manifest_paramsとstrategy_paramsの混同
**What goes wrong:** verify_strategy_manifest()が返すのはOptuna flat params (例: `{"fk_aggressive": 0.5}`) だが、BacktestEngineが期待するのはstrategy_config (例: `{"dd_config": DDConfig(...), "regime_overrides": {...}}`)
**Why it happens:** 2つのdict形式が異なる。変換に`build_strategy_config_from_params()`が必要
**How to avoid:** run_backtest.pyの`_load_strategy_params()`内で既に`_build_strategy_config_from_manifest()`を呼び出している。engine.pyには変換済みのstrategy_paramsが渡るため、engine側ではmanifest_paramsを直接使用しない
**Warning signs:** TypeError("DDConfig got unexpected keyword") または KeyError

### Pitfall 2: PFP freeze/verifyのタイミング不適切
**What goes wrong:** freeze()をrace loop内で毎回呼び出すとパフォーマンス低下。verify()をfreeze()なしで呼ぶと常にFAIL
**Why it happens:** PFPのAPI理解不足。freeze()は1回、verify()も1回、両端で挟む設計
**How to avoid:** freeze()はrun()の最初(データロード前)、verify()はrun()の最後(BacktestResult返却直前)。frozen_period()コンテキストマネージャーでも可能だが、RuntimeError制御を明示的に行うため手動呼び出しを推奨
**Warning signs:** "freeze() が呼ばれていません" エラー、または異常な実行時間

### Pitfall 3: マルチ年度でのPFP検証漏れ
**What goes wrong:** `_run_multi_year()`で年度毎にBacktestEngineを生成するため、PFP検証が年度間で独立してしまう。2年全体での不変性保証が必要
**Why it happens:** 各年度のengine.run()が独立したPFPインスタンスを持つため、年度間のモデル変更を検出できない
**How to avoid:** D-05は「テスト期間全体」でのROI判定。マルチ年度の場合でも各年度のengine.run()で独立してPFP検証を実行すればよい(各年度内での不変性保証)。2年全体での不変性は、各年度で同じキャッシュモデル(`--skip-train`)を使用することで暗黙的に保証される
**Warning signs:** 年度間でモデルが再学習されている(=--skip-trainなし)

### Pitfall 4: 原因分析でのbet_historyフィールド不在
**What goes wrong:** bet_historyのregime/surface列にアクセスしてKeyError
**Why it happens:** bet_historyのフィールドはBacktestEngine.run()内でハードコードされている。全フィールドが存在することを前提にしてはならない
**How to avoid:** 原因分析では`b.get("key", default)`パターンを使用。フィールド存在チェックを必ず行う
**Warning signs:** KeyError("regime") または KeyError("surface")

### Pitfall 5: --strategy-manifestと--ensemble/--yearsの組み合わせバリデーション不足
**What goes wrong:** --strategy-manifestのみ指定して--ensembleなしで実行すると、アンサンブルなしのバックテスト結果で検証してしまう
**Why it happens:** run_backtest.pyのvalidate_args()が--strategy-manifestと--ensembleの組み合わせを検証していない
**How to avoid:** Claude's discretion項目。validate_args()に「--strategy-manifest requires --ensemble」のチェックを追加することを推奨
**Warning signs:** 非アンサンブルモデルでのmanifest検証実行

## Code Examples

### manifest検証付きBacktestEngine統合 (D-02, D-03)
```python
# Source: src/backtest/engine.py (推奨実装パターン)
# 現状のengine.pyにはmanifest関連コードなし (grepで確認済み)

from backtest.parameter_freeze_protocol import (
    ParameterFreezeProtocol,
    verify_strategy_manifest,
)

class BacktestEngine:
    def __init__(self, ..., manifest_path: Path | None = None):
        self._manifest_path = manifest_path
        self._pfp: ParameterFreezeProtocol | None = None

    def run(self, test_start, test_end, ...):
        # D-03(1): SHA256検証
        if self._manifest_path is not None:
            verified = verify_strategy_manifest(self._manifest_path)
            logger.info("Manifest SHA256 verified: %s", self._manifest_path)

        # D-03(2): PFP freeze
        if self._manifest_path is not None:
            self._pfp = ParameterFreezeProtocol(self.models)
            self._pfp.freeze()

        # ... 既存race loop ...

        # D-03(2): PFP verify (D-04: 失敗時RuntimeError)
        pfp_result = None
        if self._pfp is not None:
            pfp_result = self._pfp.verify()
            if not pfp_result["passed"]:
                raise RuntimeError(pfp_result["message"])

        return backtest_result  # + pfp_result を出力に含める
```

### 検証結果JSONの判定ロジック (D-06)
```python
# Source: 新規モジュール (推奨)
def evaluate_validation(roi: float, total_bets: int) -> str:
    """D-06: テスト期間全体で判定"""
    if roi > 1.0 and total_bets >= 100:
        return "PASS"
    return "FAIL"
```

### 原因分析レポート生成 (D-11)
```python
# Source: 新規モジュール (推奨)
def generate_cause_analysis(bet_history: list[dict]) -> dict:
    """ROI<100%時の原因分析"""
    if not bet_history:
        return {"error": "No bet_history available"}

    # オッズバンド別ROI
    bands = {"1.0-2.0": [], "2.0-5.0": [], "5.0-10.0": [], "10.0+": []}
    for b in bet_history:
        odds = b.get("final_odds", b.get("odds", 0))
        stake = b.get("stake", 0)
        result = b.get("result", 0)
        if odds <= 2.0: band = "1.0-2.0"
        elif odds <= 5.0: band = "2.0-5.0"
        elif odds <= 10.0: band = "5.0-10.0"
        else: band = "10.0+"
        bands[band].append({"stake": stake, "result": result})

    band_roi = {}
    for name, bets in bands.items():
        total_s = sum(x["stake"] for x in bets)
        total_r = sum(x["result"] for x in bets)
        band_roi[name] = {
            "roi": total_r / total_s if total_s > 0 else 0.0,
            "bets": len(bets)
        }

    # レジーム別ROI
    regime_stats = {}
    for b in bet_history:
        regime = b.get("regime", "UNKNOWN")
        if regime not in regime_stats:
            regime_stats[regime] = {"stake": 0, "result": 0, "bets": 0}
        regime_stats[regime]["stake"] += b.get("stake", 0)
        regime_stats[regime]["result"] += b.get("result", 0)
        regime_stats[regime]["bets"] += 1

    # ... EV診断、surface別等 ...

    return {"odds_band_roi": band_roi, "regime_roi": regime_stats, ...}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ハードコードパラメータ | Optuna最適化+manifest | Phase 13-17 | パラメータがデータ駆動で決定 |
| manifest手動作成 | save_strategy_manifest()自動生成 | Phase 13 | ヒューマンエラー排除 |
| 単一seed最適化 | multi-seed安定性検証 | Phase 17 | パラメータ堅牢性向上 |
| 固定EV_lower=1.0 | 動的閾値(OOF分位点) | Phase 15 | 過剰除外の解消 |
| 2fold WF | 4fold WF | Phase 17 | 過学習耐性向上 |

**Deprecated/outdated:**
- `run_multi_year_backtest.py`: run_backtest.py --yearsに統合済み
- 固定EV_lower=1.0: Phase 15で動的閾値に置換

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Phase 17のHuman UATでOptuna最適化が実際に実行され、strategy_manifest.jsonが生成されている前提 | Standard Stack | manifest不在の場合、Phase 18のHuman UATで先にOptuna実行が必要 |
| A2 | --strategy-manifestでロードされたparamsは_build_strategy_config_from_manifest()で正しく変換され、BacktestEngineに渡される | Architecture Patterns | 変換不備の場合、engineがデフォルトパラメータで動作し検証が無意味に |
| A3 | bet_historyに"regime", "surface", "final_odds", "odds"フィールドが常に含まれる | Code Examples | 原因分析でKeyError。.get()で防御済み |

## Open Questions

1. **manifest_pathの渡し方**
   - What we know: run_backtest.pyの_load_strategy_params()は既にverify_strategy_manifest()を呼び出している。strategy_params(変換済み)がBacktestEngineに渡される。
   - What's unclear: D-02は「run()の先頭でmanifestを読み込み」としているが、現状はコンストラクタ時点でstrategy_paramsとして既に注入されている。engine.run()内でmanifest_pathを受け取ってSHA256再検証するか、コンストラクタでmanifest_pathを受け取るか。
   - Recommendation: コンストラクタに`manifest_path: Path | None = None`を追加し、run()の先頭でmanifest_pathがあればSHA256再検証+PFP freezeを実行。run_backtest.py側はmanifest_pathをそのままBacktestEngineに渡す。これによりD-02/D-03を満たす。

2. **検証結果JSONの出力タイミング**
   - What we know: D-08はdata/validation/へのJSON出力を要求
   - What's unclear: 出力をengine.run()内で行うか、run_backtest.py側で行うか
   - Recommendation: engine.run()はpfp_resultをBacktestResultに含めて返す。JSONファイル出力自体はrun_backtest.pyまたは新規validation_report.pyで行う(関心の分離)。

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies identified -- 全て既存のPython標準ライブラリ + プロジェクト内モジュールで完結)

## Validation Architecture

> nyquist_validation is explicitly false in .planning/config.json. Section skipped.

## Sources

### Primary (HIGH confidence)
- `src/backtest/engine.py` — BacktestEngine.run()の完全な実装(全1197行を確認)。manifest関連コードは現在なし [VERIFIED: codebase grep]
- `src/backtest/parameter_freeze_protocol.py` — 全187行を確認。freeze/verify/save_strategy_manifest/verify_strategy_manifest/load_and_freeze_strategyが完備 [VERIFIED: codebase read]
- `scripts/run_backtest.py` — 全734行を確認。--strategy-manifest引数、_load_strategy_params()、_build_strategy_config_from_manifest()が既に実装済み [VERIFIED: codebase read]
- `tests/test_parameter_freeze.py` — 全263行を確認。TestParameterFreezeProtocol(6テスト) + TestStrategyManifest(8テスト)が存在 [VERIFIED: codebase read]
- `tests/test_backtest_engine.py` — テストファイル存在確認 [VERIFIED: codebase glob]
- `src/betting/default_strategy.py` — build_strategy_config_from_params()がDDConfig制約処理込みで実装済み [VERIFIED: codebase read]

### Secondary (MEDIUM confidence)
- `src/tuning/strategy_optimizer.py` — optimize()/optimize_multi_seed()がsave_strategy_manifest()を自動呼び出し [VERIFIED: codebase grep]
- `.planning/phases/17-optuna-optimization/17-CONTEXT.md` — Phase 17完了、manifest自動保存の確認 [VERIFIED: read]
- `.planning/phases/17-optuna-optimization/17-02-SUMMARY.md` — multi-seed安定性検証完了 [VERIFIED: read]

### Tertiary (LOW confidence)
- なし — 全てのコードベース調査は直接確認済み

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 新規依存なし、既存モジュールの活用のみ
- Architecture: HIGH — BacktestEngine.run()の注入ポイントが明確(先頭+末尾)
- Pitfalls: HIGH — 既存テストパターンとbet_history構造を直接確認済み

**Research date:** 2026-05-06
**Valid until:** 30日 (安定ドメイン)
