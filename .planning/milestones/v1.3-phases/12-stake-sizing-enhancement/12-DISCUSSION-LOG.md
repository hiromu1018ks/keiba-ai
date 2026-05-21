# Phase 12: Stake Sizing Enhancement - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05
**Phase:** 12-Stake Sizing Enhancement
**Areas discussed:** レジーム別Kelly分数の具体値, EV比例乗算器のパラメータ, Kelly→EV乗算→DDの適用順序, Phase 13 Optuna用パラメータ注入方法

---

## レジーム別Kelly分数の具体値

| Option | Description | Selected |
|--------|-------------|----------|
| 研究提案値 (0.50/0.25) | AGGRESSIVE=0.50, CONSERVATIVE=0.25。金融ベッティング標準値 | ✓ |
| 保守的 (0.30/0.15) | リスク抑制重視。ROI改善も控えめ | |
| 積極的 (0.75/0.30) | 高確信時に大きく。DD時の損失リスク増 | |

**User's choice:** ベストプラクティスを追求（研究提案値を採用）
**Notes:** ユーザーは専門的判断を委譲。ベストプラクティス基準で決定。

### Kelly Cap

| Option | Description | Selected |
|--------|-------------|----------|
| レジーム別cap | KELLY_FRACTION_CAP固定、fractional_kellyで実効cap自然調整 | ✓ |
| 共通cap (0.125) | 全レジーム同一cap | |

**User's choice:** ベストプラクティス（KELLY_FRACTION_CAP=0.25固定）
**Notes:** 実効cap = 0.25 × fractional_kelly でAGGRESSIVE=0.125, CONSERVATIVE=0.0625

### MIN/MAX STAKE

| Option | Description | Selected |
|--------|-------------|----------|
| 維持 (100/10,000) | JRA運用制約として固定 | ✓ |
| レジーム別MAX | CONSERVATIVEで上限を下げる | |

**User's choice:** ベストプラクティス（維持）
**Notes:** JRA最低券100円・上限設定は運用制約のため固定。

### RACE_EXPOSURE_CAP

| Option | Description | Selected |
|--------|-------------|----------|
| 共通 2% | 全レジーム共通の破滅リスク防止 | ✓ |
| レジーム別 | AGGRESSIVE=3%, CONSERVATIVE=1% | |

**User's choice:** ベストプラクティス（共通2%固定）
**Notes:** 破滅リスク防止のセーフティネットはレジーム非依存が標準。

---

## EV比例乗算器のパラメータ

| Option | Description | Selected |
|--------|-------------|----------|
| 研究提案値 (target=1.10, max=2.0) | EV≥1.10で乗算、最大2倍 | ✓ |
| 低閾値・高倍率 (target=1.05, max=2.5) | より低EVから乗算開始 | |
| 高閾値・低倍率 (target=1.20, max=1.5) | 安全重視 | |

**User's choice:** ベストプラクティス（研究提案値）
**Notes:** target_ev=1.10はAGGRESSIVEのev_thresholdと同一値で一貫性あり。max_scale=2.0は過大レバレッジ防止の標準。

---

## Kelly→EV乗算→DDの適用順序

| Option | Description | Selected |
|--------|-------------|----------|
| Kelly → EV乗算 → DD | DDが最終ゲート。EV拡大をDDが抑制 | ✓ |
| Kelly → DD → EV乗算 | DD後の縮小ステークをEV拡大 | |

**User's choice:** ベストプラクティス（Kelly → EV乗算 → DD）
**Notes:** DDを最終リスクゲートとするのが標準パターン。EV拡大がDD制御をバイパスしない。

---

## Phase 13 Optuna用パラメータ注入方法

| Option | Description | Selected |
|--------|-------------|----------|
| コンストラクタ + settings.yaml | デフォルトは設定ファイル、Optunaは引数で上書き | ✓ |
| hardcoded constants | コード修正が必要 | |
| settings.yamlのみ | 実行時設定だがOptuna注入に不向き | |

**User's choice:** ベストプラクティス（コンストラクタ注入）
**Notes:** Optuna最適化時はコンストラクタ引数で直接注入。設定ファイルはデフォルト値のみ。

---

## Claude's Discretion

以下はユーザーから「ベストプラクティスを追求」で委譲された判断:
- Kelly Cap の具体的な設計（KELLY_FRACTION_CAP固定 vs レジーム別）
- RACE_EXPOSURE_CAP の共通 vs レジーム別
- EV乗算器の target_ev/max_scale の具体値
- パイプライン適用順序（Kelly → EV → DD）
- パラメータ注入のアーキテクチャ
- settings.yaml betting_strategy section のスキーマ設計
- apply_ev_scaling() のシグネチャ設計
- テスト戦略

## Deferred Ideas

None — discussion stayed within phase scope
