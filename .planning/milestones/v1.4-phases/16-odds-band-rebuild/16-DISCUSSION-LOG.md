# Phase 16: Odds Band Rebuild - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 16-Odds Band Rebuild
**Areas discussed:** デフォルトパラメータ定義, 通常パスでのOddsBand統合, バンド境界の再検討

---

## デフォルトパラメータ定義

### デフォルトパラメータソース

| Option | Description | Selected |
|--------|-------------|----------|
| RegimeDetector既定値 | get_strategy_params()のハードコード値。Phase 13 D-15外部化済み | ✓ |
| settings.yaml | 設定ファイルから読み込み | |
| 独自定数 | strategy_optimizer.py内にDEFAULT_STRATEGY_CONFIG定数 | |

**User's choice:** ベストプラクティスを追求 → RegimeDetector既定値 (Recommended)
**Notes:** Single Source of Truthとして最も一貫性がある

### デフォルトパラメータ適用範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 全16次元デフォルト | Kelly・EV・DD制御・OddsBandFilter全て。最も厳密 | ✓ |
| OddsBand関連のみ | roi_thresholdのみデフォルト。影響範囲最小 | |
| レジーム別のみ | Kelly・EV・edgeのみデフォルト。中間アプローチ | |

**User's choice:** 全16次元デフォルト (Recommended)
**Notes:** ベット生成自体が全パラメータの影響を受けるため

### 実装方法

| Option | Description | Selected |
|--------|-------------|----------|
| 別メソッドでデフォルト構築 | _build_default_config()追加。懸念事項の分離 | ✓ |
| フラグで切り替え | use_defaults_for_trainingフラグ追加 | |

**User's choice:** ベストプラクティスを追求 → 別メソッド (Recommended)
**Notes:** 分離が明確でテストも容易

### テスト検証

| Option | Description | Selected |
|--------|-------------|----------|
| モック検証 | デフォルトパラメータ使用確認 + training/test別config検証 | ✓ |
| 出力差分検証 | デフォルトと最適化後の除外バンド差分確認 | ✓ |

**User's choice:** ベストプラクティスを追求 → 両方実施
**Notes:** モックで内部動作検証 + 出力差分で実際の効果検証の二段階

---

## 通常パスでのOddsBand統合

### パイプライン統合方式

| Option | Description | Selected |
|--------|-------------|----------|
| パイプライン統合 | run_backtest.py --ensemble時に自動実行。Phase 14-15パターン踏襲 | ✓ |
| optimizer内のみ | strategy_optimizer.py経由のみ。通常backtestはキャリブレーションなし | |
| 明示的フラグ | --calibrate-odds-band等のCLI引数追加 | |

**User's choice:** ベストプラクティスを追求 → パイプライン統合 (Recommended)
**Notes:** Phase 14-15との一貫性

### training_bet_history生成場所

| Option | Description | Selected |
|--------|-------------|----------|
| BacktestEngine内で自動生成 | run()内でtrain_start～train_endを自動実行。変更局所化 | ✓ |
| スクリプト側で2回呼び出し | run_backtest.pyで明示的に2回BacktestEngine呼び出し | |

**User's choice:** ベストプラクティスを追求 → BacktestEngine内自動生成 (Recommended)
**Notes:** スクリプト側の変更が不要

---

## バンド境界の再検討

### バンド境界の取り扱い

| Option | Description | Selected |
|--------|-------------|----------|
| 固定境界を維持 | ROI計算のみアンサンブルデータで更新。Phase 17 roi_thresholdで間接調整 | ✓ |
| データ駆動で境界決定 | 分位点ベースの境界。適合リスクあり | |
| Optunaで境界も最適化 | 14+4=18次元に増加。fold増強でリスク緩和可能 | |

**User's choice:** ベストプラクティスを追求 → 固定境界維持 (Recommended)
**Notes:** 境界変更は探索空間増大と適合リスクを伴う

---

## Claude's Discretion

- _build_default_config()の具体的な実装（RegimeDetectorからの値取得方法）
- BacktestEngine.run()内でのtraining_bet_history自動生成の具体的なロジック
- テストのfixtureデータの内容
- デフォルトパラメータでDDConfigを構築する際の具体的な値

## Deferred Ideas

None — discussion stayed within phase scope
