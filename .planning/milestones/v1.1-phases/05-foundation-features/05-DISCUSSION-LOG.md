# Phase 5: Foundation Features - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 5-Foundation Features
**Areas discussed:** TSER-01~03 (時系列重み付け), PACE-01~02 (ペースフィグア), ODTS-01~02 (オッズ変動高度化), モジュール統合方法

---

## TSER-01: 指数減衰重み付け

| Option | Description | Selected |
|--------|-------------|----------|
| 既存列を置き換え | harontimel5_avgをEMA版に置き換え | ✓ |
| 新列追加 (EMA) | 新列harontimel5_emaを追加、既存平均も残す | |

**User's choice:** ベストプラクティスを追求して → 既存列置き換え（多重共線性回避）

### Halflife

| Option | Description | Selected |
|--------|-------------|----------|
| halflife=3走 | 直近3走で重み半減。λ=0.231 | ✓ |
| halflife=2走 | 極端に直近重視 | |
| 複数halflife比較 | 3と5を比較し有効な方を残す | |

**User's choice:** ベストプラクティスを追求 → halflife=3 + ルックバック全過去走拡張

---

## TSER-02: クラス調整フォーメトリック

| Option | Description | Selected |
|--------|-------------|----------|
| form_trend加重型 | 既存form_trendをクラスレベルで加重 | |
| 新指標型 | Σ(norm_finish × class_level) / Σ(class_level) | ✓ |

**User's choice:** ベストプラクティスを追求 → 新指標型（form_trendとは独立した別次元の指標）

---

## TSER-03: z-score改善トラジェクトリ

| Option | Description | Selected |
|--------|-------------|----------|
| z-score線形回帰傾き | np.polyfit傾き。form_trendと一貫したアプローチ | ✓ |
| z-score差分型 | 直近3走 vs 以前の平均差分 | |

**User's choice:** ベストプラクティスを追求 → 線形回帰傾き（全データポイント活用）

---

## PACE-01: 総合ペースフィグア

| Option | Description | Selected |
|--------|-------------|----------|
| 複合スコア型 | 単一スコアにまとめる | |
| 複数特徴量型 | pace_corner_stability, pace_closing_power, pace_position_consistencyに分割 | ✓ |

**User's choice:** ベストプラクティスを追求 → 複数特徴量型（LightGBMが非線形組み合わせを自動学習）

---

## PACE-02: pace_scenario_fit強化

| Option | Description | Selected |
|--------|-------------|----------|
| 既存列拡張 | pace_scenario_fitを拡張して宣言+実績を統合 | |
| 新列追加 | actual_pace_fitを新規追加、pace_scenario_fitは残す | ✓ |

**User's choice:** ベストプラクティスを追求 → 新列追加（宣言脚質と実績脚質は異なる信号、両方保持が最適）

---

## ODTS-01: 2次微分(加速度)

| Option | Description | Selected |
|--------|-------------|----------|
| 3点差分型 | velocity(t-30→t-10) - velocity(t-60→t-30) | ✓ |
| 全点分散型 | 微分の分散を2次微分代替とする | |

**User's choice:** 難易度は問わないのでベストプラクティスを追求 → 3点差分型（スナップショット数がレースでバラつくため、回帰は過学習リスク）

---

## ODTS-02: 方向一貫性

| Option | Description | Selected |
|--------|-------------|----------|
| 下降率型 | オッズ下降回数 / 総変動回数 | |
| 時間加重型 | 指数減衰で直近変動を高く評価 | ✓ |

**User's choice:** 難易度は問わないのでベストプラクティスを追求 → 時間加重型（直近変動が最も予測的）

---

## モジュール統合方法

| Option | Description | Selected |
|--------|-------------|----------|
| 既存モジュールに追加 | TSER→horse_history, PACE→pace_aptitude, ODTS→odds_dynamics | ✓ |
| TSERのみ新規モジュール | time_series_features.py新規作成 | |

**User's choice:** 難易度は問わないのでベストプラクティスを追求 → 既存モジュールに追加（同じデータコンテキスト、同じパターン）

---

## NaN処理方針

| Option | Description | Selected |
|--------|-------------|----------|
| デフォルトNaN | LightGBMのネイティブNaN処理を活用 | ✓ |
| fill(0)でNaN排除 | NaN率の懸念を排除 | |

**User's choice:** 難易度は問わないのでベストプラクティスを追求 → デフォルトNaN（0埋めは「データなし」と「実際に0」を混同）

---

## Claude's Discretion

- EMAのnumpy向量化実装詳細
- class_adj_formetricのclass_level取得（history entries/racesのgrade_code/jyoken_code可用性確認）
- pace_closing_powerの上がりタイムデータソース（agi列 or harontimel3近似）
- odds_direction_consistencyの指数減衰率（halflife = スナップショット数/4程度）
- 各特徴量のNaN率50%超過時のフォールバック戦略

## Deferred Ideas

None — all discussion stayed within phase scope
