---
status: resolved
trigger: "Phase 09 HTMLレポートで regime_bands が空配列。regimeテーブルがスキップされる。AI診断JSONは正常。"
created: 2026-05-04
updated: 2026-05-04
---

# Debug Session: regime-bands-empty

## Symptoms

**Expected behavior:**
- HTMLレポートに condition_stats.regime_bands に基づく regime 別テーブル（aggressive/conservative/collapsed の内訳）が表示される
- bet_history の各ベットに regime フィールドが正しく設定されている

**Actual behavior:**
- regime_breakdown が空配列
- HTMLテンプレートの `{% if condition_stats.regime_bands %}` ガードで regime テーブルがスキップされる
- AI診断JSON（highlights/monthly_trend/popularity_bands）は正常に生成されている

**Error messages:**
- エラーは発生していない（空配列として処理される）

**Timeline:**
- バックテスト実行後、HTMLレポート生成時に発生

**Reproduction:**
- `python scripts/run_backtest.py` 実行 → HTMLレポート生成 → regime_bands が空

**User context:**
- bet_history の各ベットに regime フィールドが入っていない可能性が示唆されている
- AI診断JSON側の regime データは正常（HTMLレポート側の問題の可能性）

## Current Focus

- hypothesis: CONFIRMED — bet_history の各ベットレコードに regime フィールドが設定されていない
- test: engine.py bet_history dict 構築箇所のソースコード確認
- expecting: regime_bands に aggressive/conservative/collapsed の3バンドが含まれること
- next_action: fix applied
- reasoning_checkpoint: regime は diag_logger には渡されているが bet_history dict には含まれていない。またマルチ年度モードでは betting_target が generate() に渡されていない。

## Evidence

- **2026-05-04**: `src/backtest/engine.py` lines 904-958 — bet_history.append() の dict に `"regime"` フィールドが含まれていない。`regime` 変数は lines 684-687 で検出済みでスコープ内に存在するが、dict 構築時に漏れている。
- **2026-05-04**: `src/backtest/report.py` line 256 — `_compute_regime_stats` は `b.get("regime", "unknown")` を使う。regime フィールドがない場合、全ベットが "unknown" になり、aggressive/conservative/collapsed のいずれにもマッチしないため結果が空になる。
- **2026-05-04**: `src/backtest/report.py` lines 395-408 — `_compute_condition_stats` は `betting_target == "win"` の場合のみ `regime_bands` を構築する。これは意図的なガード。
- **2026-05-04**: `src/backtest/report.py` lines 248-275 — `_compute_regime_stats` は ["aggressive", "conservative", "collapsed"] の3状態のみを出力する。
- **2026-05-04**: AI診断JSON (`save_ai_diagnostics`) は独自に `_compute_regime_stats` を呼び出すが、同じ bet_history を使うため同じく空になるはず。ユーザー報告の「AI診断JSONは正常」は、regime 以外の highlights/monthly_trend/popularity_bands を指している可能性。
- **2026-05-04**: `scripts/run_backtest.py` line 547 — マルチ年度モードで `gen.generate(all_results, all_metadata)` と betting_target を渡していない。デフォルト "place" になり、regime_bands が構築されない（第2のバグ）。

## Eliminated

- テンプレート側の問題ではない。regime_bands データが空であることが原因。
- _derive_fields は元のフィールドを保持するため、regime があれば残るはず。

## Resolution

### Root Cause

`src/backtest/engine.py` の bet_history 構築 (line 904-958) で `regime` フィールドが dict に含まれていない。これにより `_compute_regime_stats` が全ベットを "unknown" として処理し、regime_bands が空配列になる。

第2のバグ: マルチ年度モードで `betting_target` が `generate()` に渡されておらず、デフォルト "place" により regime_bands 構築がスキップされる。

### Fix Applied

1. **`src/backtest/engine.py` line 945**: bet_history.append() の dict に `"regime": str(regime)` を追加。`regime` 変数は既にスコープ内に存在。
2. **`scripts/run_backtest.py` line 547**: マルチ年度モードの `gen.generate()` 呼び出しに `betting_target=args.betting_target` を追加。

### Verification

- 1162 tests passed, 0 failures, 1 skipped
- All 32 backtest report tests pass
