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
  │    └→ p_ability_place は下流モデルの FEATURE_COLS に含まれない
  │       （カスケードには伝播しない。ベット評価にのみ影響）
  │
  ├→ WinTwoStageModel.train(p_ability_win を特徴量に)
  │    └→ p_win_pred, e_return_win_pred (CASCADING)
  │         └→ EVCorrectionModel.train(上記を特徴量に)
  │
  └→ PlaceTwoStageModel.train(p_ability_win を特徴量に)
       （FEATURE_COLS は WinTwoStageModel と同一）
```

Stage1 の in-sample 予測は「ほぼ正解」に近い値を生成する。
下流モデルはこの過度に正確なシグナルに過学習し、EV推定が楽観的になる。
結果としてベット数が倍増 (2,967→6,134) し、ROIが143.3%に押し上げられる。

### MarketModel の In-Sample log_error について

MarketModel も同じ in-sample パターンに従う:
```python
market.train(df)                        # 全データで学習
df = market.predict_and_calc_error(df)   # 同じデータを予測 → log_error (IN-SAMPLE)
```

`signed_log_error_win`, `abs_log_error_win` は `WinTwoStageModel` と `EVCorrectionModel`
の FEATURE_COLS に含まれる。これも in-sample リークである。

**ただし Phase 2 のスコープでは MarketModel の OOF 化は含めない。理由:**
- MarketModel は特徴量が少なく、in-sample log_error の寄与は Stage1 の p_ability_win
  に比べて小さいと推定される
- Phase 1 アブレーション（パターンB: p_ability_win削除）で Stage1 由来のリーク寄与を
  先に定量化し、MarketModel 由来の寄与が残存する場合は別途対応する

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
↓ OOF結合 → p_ability_win (OOF値で上書き)

Final: Stage1.train(2020-2023) → 推論用モデル（self.models に格納）
```

**推論時の分離**: バックテスト・本番推論では、OOF foldモデルではなく
全データで学習した最終モデル（self.models）を使用する。
`BacktestEngine` は `submodel.stage1.add_ability_probs()` を呼び出すが、
ここで使われるのは全データ学習の最終モデルであり、OOF foldモデルではない。

#### Surface分割との相互作用

現在の `_train_submodel()` は単一の `surface`（芝/ダート）に対して呼び出される。
従って、OOF fold も各 surface サブセット内で独立して生成される。

```
training_pipeline._train_submodel(df_turf):
  → df_turf 内で OOF fold 生成（surface は常に芝）
  → 全ての fold モデルと最終モデルは芝のみで学習

training_pipeline._train_submodel(df_dirt):
  → 同様にダート内で OOF fold 生成
```

Fold 1 では学習期間が最短（~1年分）になるが、芝/ダートそれぞれで
年間数千レースがあるため、データ不足にはならない。

#### 列命名規則

OOF予測は **同じ列名 `p_ability_win`** を使用する。
`train_oof()` 内で in-sample の `p_ability_win` を OOF値で上書きする。
これにより、下流モデル（PlaceAbilityModel, WinTwoStageModel,
EVCorrectionModel, PlaceTwoStageModel, WideTwoStageModel）の
FEATURE_COLS を変更する必要がない。全下流モデルが透過的にOOF値を受け取る。

in-sample パスの `add_ability_probs()` は `train_oof()` 内部でのみ使用し、
最終的に `df["p_ability_win"]` には OOF値が格納される。

#### Fold境界の決定ロジック

`n_folds` 個のほぼ等しい期間に `race_date` を分割:
- `dates = sorted(df["race_date"].unique())`
- `fold_boundaries = [dates[len(dates) * i // n_folds] for i in range(n_folds + 1)]`
- Fold i: train = `[0, fold_boundaries[i])`, predict = `[fold_boundaries[i], fold_boundaries[i+1])`

これにより任意の学習期間に対して動的に機能する。

#### PlaceAbilityModel内部分割との関係

PlaceAbilityModel は内部で80/20時系列校正分割を行うが、
OOF `p_ability_win` 導入後もこの内部分割は正しく動作する:
- 校正分割は PlaceAbilityModel 自身の学習用であり、
  Stage1 の OOF 境界とは独立して race_date ベースで分割される
- OOF `p_ability_win` は「Stage1が見ていないデータへの予測」を表すため、
  PlaceAbilityModel の視点からは legitimate な特徴量として扱える

#### 変更ファイル

**`src/models/stage1_ability_model.py`**:
- `train_oof(df, n_folds=3)` メソッド追加
- K-fold expanding window で OOF 予測を生成
- 最終モデルを全データで学習し `self.models` に格納（推論用）
- `df["p_ability_win"]` を OOF値で上書きして返す

**`src/pipelines/training_pipeline.py`**:
- `_train_submodel()` の `train()` + `add_ability_probs()` を `train_oof()` に変更
- OOF 予測を通じて全下流モデル（PlaceAbilityModel, WinTwoStageModel,
  EVCorrectionModel, PlaceTwoStageModel, WideTwoStageModel）が
  透過的に OOF `p_ability_win` を受け取る

**テスト追加**:
- OOF予測に未来データが混入していないことを検証するテストケース
- OOF値とin-sample値が異なることを確認するテスト

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

### Out of Scope (Phase 3 以降)

- MarketModel の OOF 化（Phase 1 で MarketModel 寄与が大きいと判明した場合に対応）
- 過去データスナップショットのタイムトラベル（x_UMA の日付フィルタリング）
