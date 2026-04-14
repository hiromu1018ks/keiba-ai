# PIT 違反: track_condition_code リーク修正

> Date: 2026-04-14
> Status: 修正済み、バックテスト実行中

## 問題

`track_condition_code`（馬場状態: 良=1, 稍重=2, 重=3, 不良=4）が**全7モデルの入力特徴量**として使用されていた。

### なぜこれがリークか

| フェーズ | track_condition_code の値 | 入手可能性 |
|---------|--------------------------|-----------|
| 学習時 | 1~4 (確定値) | レース終了後 |
| バックテスト | 1~4 (確定値) | レース終了後 |
| Paper Trading | 0 (未確定) or 推定値 | **レース前は入手不可** |

学習データの TCC 最小値 = 1（良）。PT では TCC=0（データ欠落）。
`is_good_track = (TCC <= 1)` のロジックで、PT では悪馬場が「良」と誤認される。

### 影響を受けたモデル (7ファイル)

1. `src/models/stage1_ability_model.py` — Stage1 能力モデル
2. `src/models/place_ability_model.py` — 複勝能力モデル
3. `src/models/market_model.py` — 市場モデル
4. `src/models/two_stage_return_model.py` — 単勝2段階モデル
5. `src/models/ev_correction_model.py` — EV補正モデル
6. `src/models/wide_two_stage_model.py` — ワイド2段階モデル
7. `src/models/race_quality_screener.py` — レース品質スクリーナー

## 修正内容

全モデルの `FEATURE_COLS` / `SHARED_FEATURE_COLS` から `track_condition_code` を削除。
コメントで `# track_condition_code: PIT除外 (レース後確定情報)` を残して意図を明記。

## 除外しなかったもの

- `submodel_manager.py` の `is_good_track` / `is_soft_track` 派生計算 → FEATURE_COLS に含まれないため OK
- `wide_pair_builder.py` の TCC 参照 → データ結合用、モデル入力ではない

## バックテスト結果（TCC 修正前 vs 修正後）

修正前: ROI 136.6%, +¥214,220, maxDD 0.67% (2025テスト)
修正後: （バックテスト実行中）

## 教訓

**「答えを知っている状態」でのパフォーマンスは架空の数字。**
- BT の高 ROI は「馬場状態が既知」前提
- 実運用（PT）ではこの情報が存在しない
- 学習時も「レース前に入手可能な情報のみ」を特徴量にする必要がある
