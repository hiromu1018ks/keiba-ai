# Phase 36: Feature Computation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-19
**Phase:** 36-Feature Computation
**Areas discussed:** HaronTime統合ロジック, LapTimeペース特徴量設計, weighted_recent_form定義, 交互作用ベース列確認

---

## HaronTime統合ロジック

| Option | Description | Selected |
|--------|-------------|----------|
| 距離別選択 | 短距離はL3、中長距離はL4を自動選択 | ✓ |
| coalesce(L3優先) | L3があればL3、なければL4 | |
| 両方独立 | L3もL4も別々に履歴統計を計算 | |
| 既存L3 + 新規L4 | HaronTimeL3は既存に統合済み、L4のみ新規追加 | |

**User's choice:** 距離別選択 + L3/L4両履歴
**Notes:** ユーザーは「実装難易度は問わないのでベストプラクティスを追求」。距離別選択で統合列を作成しつつ、L3/L4の両方の独立した履歴統計も計算する方式に決定

### HaronTime特徴量の統計量

| Option | Description | Selected |
|--------|-------------|----------|
| 4種(avg/zscore/trend/race_rank)×3列 = 12特徴量 | 互いに直交信号を提供する4統計量 | ✓ |
| 3種×3列 = 9特徴量 | avg/zscore/trendのみ | |
| 6種×3列 = 18特徴量 | 過学習リスクあり | |

**User's choice:** 4種×3列 = 12特徴量
**Notes:** avg(絶対能力), zscore(相対能力), trend(方向性), race_rank(レース内位置)の4統計量がベストプラクティス

---

## LapTimeペース特徴量設計

### セグメント分割方式

| Option | Description | Selected |
|--------|-------------|----------|
| 等分3分割(1/3ずつ) | 距離差を吸収、ペース比が直接比較可能 | ✓ |
| 不均等分割(40/20/40) | JRA展開パターンに合致 | |
| 2分割(前半/後半) | 中盤省略でシンプル | |

**User's choice:** 等分3分割 + pace_ratio履歴
**Notes:** ユーザーは「ベストプラクティスを追求」。等分3分割が速度図システムの業界標準。pace_ratio = 後半/前半 (< 1.0 = 末脚速い)

---

## weighted_recent_form定義 (TRF-02)

### 加重方式

| Option | Description | Selected |
|--------|-------------|----------|
| EMA加重(halflife=3) | 既存harontimel5_avgと同じ方式 | ✓ |
| 線形減衰(古い程重い) | w=[1,2,3] | |
| 線形減衰(新しい程重い) | w=[3,2,1] | |

**User's choice:** EMA(halflife=3)
**Notes:** ベストプラクティス。指数減衰は時系列分析の標準、既存コードとの整合性も確保

### 評価指標

| Option | Description | Selected |
|--------|-------------|----------|
| norm_finish_logitのみ | 頭数正規化済み着順 | |
| timediffのみ | 勝馬との着差 | |
| 両方(norm_finish_logit + timediff) | 相補的信号 | ✓ |

**User's choice:** 両方
**Notes:** norm_finish_logit(位置的性能)とtimediff(時間的性能)は相補的。LightGBMが自動選択

---

## 交互作用ベース列確認 (INT-01~03)

| Interaction | Options | Selected |
|------------|---------|----------|
| INT-01 grade_x_form_trend | grade_code×form_trend / grade_num×form_trend | grade_code × form_trend |
| INT-02 distance_x_closing_index | kyori×closing_index_avg / distance_bin×closing_index_avg | kyori × closing_index_avg |
| INT-03 grade_x_blood_prize_log | grade_code×blood_prize_log / grade_code×sire_wr | grade_code × blood_prize_log |

**User's choice:** 全て推奨案に一任(「ベストプラクティスを追求。判断は任せる」)
**Notes:** grade_codeは標準グレード数値、kyoriは連続値距離(m)、blood_prize_logは情報量が多い

---

## Claude's Discretion

- 距離閾値のデフォルト値(Phase 35品質確認前の初期値): 2000m
- LapTime特徴量のexpanding_stats実装詳細
- 各特徴量のNaNハンドリング(過走0走のデフォルト値)
- テストケース設計(PIT安全性・双方パス・FEATURE_COLS完全性)
- harontime_last3fの具体的なcoalesceロジック
- LapTime列名の正規化(Phase 35 ETL出力との整合)

## Deferred Ideas

- harontime_last3fの距離閾値最終決定 — Phase 35品質確認に依存
- コーナー通過順位展開特徴量 — HLF-06 (将来フェーズ)
- ペースプロファイル分類 — HLF-07 (将来フェーズ)
- 末脚指数 — HLF-08 (将来フェーズ)
