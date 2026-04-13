# RegimeDetector 修正 + grade_code NaN バグ修正 (2026-04-13)

## 概要

RegimeDetector の学習データの分散破壊（rolling average）と grade_code NaN バグを修正。
RegimeDetector は機能するようになったが、ROI が赤字に転落。固定EV閾値の検討が必要。

## grade_code NaN バグ

### 原因

`src/db/readers.py` の `_STRING_COLUMNS` に `gradecd` が含まれておらず、
`_coerce_types()` の `pd.to_numeric(errors="coerce")` で 'A','B','C' 等のグレードコードが
全て NaN になっていた。

### 影響

- `PlaceAbilityModel` (31特徴量) と `WinTwoStageModel` (16特徴量) の `grade_code` が全て NaN
- Feature Importance で `grade_code` の gain = 0（全モデル）
- モデルがレースクラス（G1 vs 未勝利等）を一切区別できていなかった

### 修正

1. `_STRING_COLUMNS` に `gradecd`, `gradecdbefore` を追加
2. 空文字 `""` を `"X"` (未格付け) にマッピング

### コミット

未コミット（他の変更とまとめてコミット予定）

## RegimeDetector 修正

### 原因

`_build_regime_stats()` で `rolling(window=200, min_periods=50).mean()` を適用:
- `favorite_implied` の std: 0.1501 → 0.0048 (30分の1に減少)
- LightGBM が全特徴量で分割不可能 → 全 gain=0
- 常に CONSERVATIVE を予測 → EV閾値1.30固定 → ベット数113件

さらに、学習時（rolling値）と推論時（生値）で特徴量分布が異なる train/test ミスマッチも発生。

### 修正

1. **`_build_regime_stats()` rolling → raw 移行**
   - 5箇所の `.rolling(window=200, min_periods=50).mean()` を削除
   - レース毎の生の統計値をそのまま使用
   - 列名は `RegimeDetector.FEATURE_COLS` と整合するよう保持

2. **閾値調整**
   - COLLAPSED: `< 0.18` → `> 0.50`（方向を反転）
   - AGGRESSIVE: `< 0.28` AND `entropy > median`（変更なし）
   - RAW値の分布 (mean=0.25, std=0.09) に適合

3. **ハイパーパラメータ**
   - `min_data_in_leaf`: 50 → 30

4. **RegimeConfig**
   - `fav_rate_collapsed`: 0.18 → 0.50

## バックテスト結果

### 条件

- 学習期間: 2021-01-01 ~ 2024-12-31
- テスト期間: 2025-01-01 ~ 2025-12-31
- モード: ensemble, flat (固定¥100), JRAのみ

### 修正前 (PIT修正済み、RegimeDetector壊れ状態)

| 指標 | 値 |
|------|-----|
| ROI | **143.7%** |
| ベット数 | 113 |
| 利益 | +¥4,940 |
| 最大DD | 1.8% |

### 修正後 (RegimeDetector正常化 + grade_code修正)

| 指標 | 値 |
|------|-----|
| ROI | **98.8%** |
| ベット数 | **499** |
| 利益 | **-¥610** |
| 最大DD | 4.0% |

### Regime分布

| レジーム | レース数 | ベット数 | EV閾値 |
|----------|:-------:|:-------:|:------:|
| AGGRESSIVE | 2,549 (76%) | 441 | 1.10 |
| CONSERVATIVE | 787 (24%) | 58 | 1.30 |
| COLLAPSED | 0 (0%) | 0 | 1.50 |

### Feature Importance 比較

| 特徴量 | 修正前 | 修正後 |
|--------|:------:|:------:|
| entropy_rolling | 0.0% | **59.6%** |
| favorite_implied_prob_rolling | 0.0% | **20.0%** |
| overround_rolling | 0.0% | **17.9%** |
| field_size_mean | 0.0% | 2.0% |
| 他4特徴量 | 0.0% | 0.5% |
| Number of trees | 3 | **300** |

## 分析

### RegimeDetector の課題

1. **3特徴量で97.5%の gain を占める** — 実質的に entropy のルールベース分類
2. **COLLAPSED が一度も検出されない** — 閾値0.50が高すぎるか、2状態で十分
3. **AGGRESSIVE が76%を占めすぎ** — EV閾値1.10でマージナルベットを通し、ROIを押し下げ
4. **MLである必要性が薄い** — entropy高低だけで分類できる

### 前回分析の閾値スイープ（旧モデル、参考値）

| EV閾値 | ベット数 | ROI | 利益 |
|--------|:-------:|:---:|-----:|
| >= 1.50 | 59 | 181.2% | +¥4,790 |
| >= 1.30 | 147 | 186.1% | +¥12,650 |
| >= 1.10 | 399 | 164.4% | +¥25,680 |
| >= 1.05 | 530 | 163.9% | +¥33,870 |
| >= 0.88 | 1,886 | 118.7% | +¥35,200 |

※ 旧モデル（grade_code NaN、RegimeDetector壊れ）での分析。新モデルでは値が変わる。

### 次のステップ候補

1. **新モデルで閾値スイープ再実行** — grade_code + RegimeDetector修正後のEV分布を確認
2. **RegimeDetector廃止 → 固定EV閾値** — シンプル化でROI改善の可能性
3. **AGGRESSIVEのEV閾値を1.10→1.15に引き上げ** — 微調整でROI回復の可能性
4. **特徴量改善** — 真の血統特徴量（種牡馬産駒成績）、クラス変動、コース適性等

## 変更ファイル

| ファイル | 変更内容 |
|--------|----------|
| `src/db/readers.py` | `_STRING_COLUMNS` に `gradecd`, `gradecdbefore` 追加 |
| `src/features/feature_engine.py` | 空文字→"X" マッピング |
| `src/pipelines/training_pipeline.py` | `_build_regime_stats()` rolling→raw 移行 |
| `src/models/regime_detector.py` | COLLAPSED閾値反転 (0.18→0.50)、min_data_in_leaf 50→30 |
| `src/domain/models.py` | `fav_rate_collapsed` 0.18→0.50 |

## コミット

未コミット（方針決定後にまとめてコミット予定）
