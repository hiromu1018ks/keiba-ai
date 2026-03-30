# In-Sample Cascade Leakage Investigation & Fix

Date: 2026-03-30
Status: Approved

## Background

バックテスト ROI が 63.8% (旧モデル) から 143.3% (新モデル) に急上昇。
Stage1 FEATURE_COLS の 20→30列化に伴い、in-sample cascade leakage が増幅された可能性が高い。

## Root Cause Analysis

### In-Sample Cascade Leakage Chain

Stage1 (AbilityModel) を全学習データで訓練後、**同じ学習データ**を予測して `p_ability_win` を生成。
この in-sample 予測値が下流モデルの特徴量として使用される:

```
Stage1.predict(学習データ) → p_ability_win (IN-SAMPLE)
  ├→ PlaceAbilityModel.train(p_ability_win を特徴量に)
  ├→ WinTwoStageModel.train(p_ability_win を特徴量に)
  │    └→ p_win_pred, e_return_win_pred (CASCADING)
  │         └→ EVCorrectionModel.train(上記を特徴量に)
  └→ PlaceTwoStageModel.train(p_ability_win を特徴量に)
```

Stage1 の in-sample 予測は「ほぼ正解」に近い値を生成する。
下流モデルはこの過度に正確なシグナルに過学習し、EV推定が楽観的になる。
結果としてベット数が倍増 (2,967→6,134) し、ROIが143.3%に押し上げられる。

### 30個のStage1特徴量は個別にSAFE

全30特徴量を個別に検証済み:
- 26個: SAFE (searchsorted で過去データのみ使用)
- 2個: 常にNaN (Phase 2プレースホルダー)
- 2個: 要注意だが旧モデルにも存在 (ROI急騰の直接原因ではない)

### kyakusituリークは既に修正済み

コミット `5f48fd8` で修正。現在の 143.3% ROI は修正後のバックテスト結果。
従って、kyakusituリークは今回のROI急騰の原因ではない。

## Design

### Phase 1: Ablation Study (原因特定)

| # | Stage1特徴量 | 下流へのp_ability_win | 目的 |
|---|-------------|----------------------|------|
| A | 旧20列 | in-sample | ベースライン再現 (~63.8%) |
| B | 新30列 | **なし**（削除） | 特徴量追加の純粋な効果 |
| C | 新30列 | in-sample | 現状再現 (~143.3%) |
| D | 新30列 | OOF | **修正後の正しいROI** |

**最優先**: パターンBを実行し、p_ability_winリークの寄与を定量化。

結果の解釈:
- ROI < 100%: p_ability_winリークが主因
- ROI 100-120%: 特徴量有効、リークで押し上げ
- ROI > 120%: 特徴量 genuinely 有力、リークは増幅要因のみ

### Phase 2: OOF Implementation (リーク排除)

#### Approach: K-Fold Expanding Window

```
Fold 1: Stage1.train(2020)      → predict(2021) → OOF[2021]
Fold 2: Stage1.train(2020-2021) → predict(2022) → OOF[2022]
Fold 3: Stage1.train(2020-2022) → predict(2023) → OOF[2023]
↓ OOF結合 → p_ability_win_oof (2021-2023)

Final: Stage1.train(2020-2023) → 推論用モデル（バックテストで使用）
下流モデル.train(p_ability_win_oof) → OOFベースで学習
```

#### 変更ファイル

**`src/models/stage1_ability_model.py`**:
- `train_oof(df, n_folds=3)` メソッド追加
- K-fold expanding window で OOF 予測を生成
- 最終モデルを全データで学習し `self.models` に格納（推論用）
- df に OOF `p_ability_win` を追加して返す

**`src/pipelines/training_pipeline.py`**:
- `_train_submodel()` の `train()` + `add_ability_probs()` を `train_oof()` に変更
- OOF 予測を下流モデルの学習に使用

**テスト追加**:
- OOF予測に未来データが混入していないことを検証するテストケース

#### 学習時間の推定

| 項目 | 現在 | 修正後 |
|------|------|--------|
| Stage1 学習 | 1回 | 4回 (3 fold + 1 final) |
| 学習時間合計 | ~17分 | ~25-50分 |
| バックテスト | ~7分 | ~7分（変更なし） |

fold間並列化（ThreadPoolExecutor）で ~25分に短縮可能。

### Verification

#### 合否判定基準

| 条件 | 判定 |
|------|------|
| OOF ROI > 110% | 特徴量設計は有効。実運用可能 |
| OOF ROI 100-110% | 妥当。微調整で改善の余地あり |
| OOF ROI < 100% | 特徴量の再設計が必要 |

#### クリンアップ

- `p_ability_win` の in-sample パスを完全削除
- `train_oof()` をデフォルトに昇格
- OOF関連テストケースを追加
