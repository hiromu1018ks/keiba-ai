# Phase 19: EV推定キャリブレーション - Research

**Researched:** 2026-05-07
**Domain:** EV推定キャリブレーション (Isotonic Regression + オッズバンド別補正)
**Confidence:** HIGH

## Summary

Phase 19は、既存EVCorrectionModelのPxE補正後(ev_win_corrected)に、sklearn IsotonicRegressionを適用してEV推定の体系的過大評価を是正する。OOF予測ベースで学習し、サーフェス別(芝/ダート)に独立したIsotonicモデルを構築する。

現在のEV過大評価の根本原因はPxE分解の独立性仮定にある。P補正とE補正の個別バイアスが乗算で増幅され、特に高オッズ帯(20+)で2.08倍の過大評価を生んでいる。IsotonicRegressionは非パラメトリックな単調制約付きキャリブレーションで、EV→actual_returnの体系的歪みを安全に吸収する。

**Primary recommendation:** ev_win_corrected → Isotonic → オッズバンド別残差補正の3段階パイプラインを構築し、TrainingPipelineV5._train_submodel()内でOOF EV予測を生成してIsotonicをfitする。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 既存PxE補正(ev_win_corrected)の後にIsotonicを適用する。二重補正構成。Isotonic単体置き換えではなく上乗せ。
- **D-02:** Isotonicの適用単位はサーフェス別(芝/ダート)独立モデル。オッズバンド別の独立Isotonicはサンプル不足リスクがあるため採用しない。
- **D-03:** Isotonicの学習ターゲットは EV→actual_return 直接。X=ev_win_corrected(OOF)、y=actual_return(OOF)。
- **D-04:** Isotonicの境界処理は y_min=0, out_of_bounds='clip'。
- **D-05:** OOF EV予測はTrainingPipelineV5の_train_submodel()内で生成する。
- **D-06:** IsotonicモデルはSubmodelSetの新しいフィールドとして保存する。
- **D-07:** バックテスト時は学習済みIsotonicをロードして適用するのみ。テスト期間中の再学習はデータリークになるため行わない。
- **D-08:** IsotonicキャリブレーションはEVCorrectionModel.correct_ev()の最後で適用する。ev_win_corrected → Isotonic → ev_win_calibrated。
- **D-09:** OOF EV予測生成のために_train_submodel()内でK-foldループを追加する。
- **D-10:** オッズバンド別補正はIsotonic適用後の残差に対して適用する。補正順序: PxE補正 → Isotonic → オッズバンド別補正。オッズバンド境界はPhase 16 D-08の固定値 `[1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+]` を維持。

### Claude's Discretion
- _train_submodel()内でのOOF EV生成の具体的なK-fold実装(分割数、時系列ソート)
- Isotonic適用後のオッズバンド別補正の具体的な手法(バンド別スケーリング、回帰、等)
- SubmodelSetへのIsotonicフィールドの命名規則と保存形式(.joblib)
- ModelLoaderの読み込み拡張の実装詳細
- OOF EV予測のメモリ管理(大量データの処理)
- テストのfixtureデータとモック構成
- correct_ev()へのIsotonic適用の具体的な実装(初期化判定、未学習時のフォールバック)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVC-01 | OOF予測ベースのIsotonic EVキャリブレーション。ev_win_corrected → Isotonic → ev_win_calibrated | Architecture Pattern 1 (OOF EV生成), Pattern 2 (Isotonic適用), Standard Stack (sklearn IsotonicRegression) |
| EVC-02 | オッズバンド別EV補正層。Isotonic適用後の残差をオッズバンド別に補正 | Architecture Pattern 3 (オッズバンド別補正), OddsBandFilter.BANDS定義 |
| EVC-03 | EVCorrectionModel統合。correct_ev()内でIsotonic + オッズバンド補正を適用 | Integration Points (correct_ev()拡張, SubmodelSet, ModelLoader, _save_models_local) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Isotonicモデル学習 | MLパイプライン (TrainingPipelineV5) | — | OOF EV生成+Isotonic fitは学習時のみ |
| Isotonic推論適用 | EVCorrectionModel | — | correct_ev()の最後で適用(D-08) |
| オッズバンド別補正 | EVCorrectionModel | — | Isotonic適用後の残差補正(D-10) |
| Isotonicモデル保存/読み込み | SubmodelSet / ModelLoader | — | 既存モデル管理パターンに統合(D-06) |
| EV品質診断 | ev_diagnostics | — | キャリブレーション前後のECE比較 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sklearn.isotonic.IsotonicRegression | 1.8.0 [VERIFIED: runtime] | EV→actual_returnの非パラメトリックキャリブレーション | 単調制約でEVの順序を保持、過学習リスク低、sklearn標準 |
| joblib | (bundled w/sklearn) | Isotonicモデルの永続化(.joblib) | 既存パターン(place_ability, isotonic_place_等)と統一 |
| numpy | (existing) | actual_return計算、OOF配列操作 | — |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn.model_selection.KFold | 1.8.0 | OOF EV予測生成のfold分割 | generate_win_oof_predictionsパターンと同じ |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| IsotonicRegression | Platt Scaling (LogisticRegression) | Isotonicは非パラメトリックで非線形バイアスに強い。Plattは線形前提で高オッズ帯の強い非線形歪みに対応困難 |
| IsotonicRegression | Beta Calibration | Betaは3パラメータ最適化が必要、サンプル不足で不安定。Isotonicはより堅牢 [ASSUMED] |
| 単一グローバルIsotonic | サーフェス別Isotonic (採用) | グローバルは芝/ダートのオッズ分布差を吸収できない。D-02決定済み |
| オッズバンド別Isotonic | オッズバンド別スケーリング (推奨) | バンド別Isotonicは各バンドのサンプル不足リスク。バンド別スケーリング係数の方が安定 |

**Installation:**
```bash
# 追加インストール不要 — sklearn 1.8.0 は既に依存関係に含まれる
# IsotonicRegressionはsklearn.isotonicモジュールの一部
```

## Architecture Patterns

### System Architecture Diagram

```
                    TrainingPipelineV5._train_submodel()
                               |
                ┌──────────────┼──────────────┐
                v              v               v
          AbilityModel    WinTwoStage    EVCorrection
          .train_oof()    Model.train()  Model.train()
                |              |               |
                |              v               v
                |     WinTwoStageModel    correct_ev()
                |     .predict_ev()       → ev_win_corrected
                |              |               |
                └──────────────┤               |
                               v               v
                    ┌──────────────────────────┐
                    │  NEW: OOF EV 生成ループ   │
                    │  (K-fold, D-05/D-09)     │
                    │  ┌────────────────────┐  │
                    │  │ fold model train   │  │
                    │  │ → predict_ev()     │  │
                    │  │ → correct_ev()     │  │
                    │  │ → ev_win_corrected │  │
                    │  │ + actual_return    │  │
                    │  └────────────────────┘  │
                    └──────────┬───────────────┘
                               v
                    ┌──────────────────────────┐
                    │  Isotonic fit (D-01/D-03) │
                    │  X = ev_win_corrected     │
                    │  y = actual_return        │
                    │  y_min=0, clip (D-04)     │
                    │  芝/ダート別 (D-02)       │
                    └──────────┬───────────────┘
                               v
                    ┌──────────────────────────┐
                    │  オッズバンド別残差補正   │
                    │  (D-10)                  │
                    │  バンド別median残差比で   │
                    │  スケーリング係数算出     │
                    └──────────┬───────────────┘
                               v
                    SubmodelSetに保存 (D-06)
                    → ModelLoaderで読み込み (D-07)

推論パス:
    correct_ev()
         ↓
    ev_win_corrected (既存PxE補正)
         ↓
    Isotonic.transform(ev_win_corrected)
         ↓
    オッズバンド別スケーリング
         ↓
    ev_win_calibrated
```

### Recommended Project Structure
```
src/
├── models/
│   ├── ev_correction_model.py    # (主変更) correct_ev()にIsotonic適用追加
│   └── ev_diagnostics.py         # (修正) ev_win_calibrated対応
├── pipelines/
│   └── training_pipeline.py      # (主変更) OOF EV生成+Isotonic fit
├── domain/
│   └── models.py                 # (修正) SubmodelSetに新フィールド
├── db/
│   └── model_loader.py           # (修正) Isotonic読み込み/保存対応
└── betting/
    └── odds_band_filter.py       # (参照) BANDS定義流用

tests/
├── test_ev_correction.py         # (拡張) Isotonic/オッズバンド別テスト追加
└── test_backtest_engine.py       # (参照) 統合テストパターン
```

### Pattern 1: OOF EV予測生成 (D-05, D-09)

**What:** TrainingPipelineV5._train_submodel()内で、K-foldループでOOF EV予測を生成する。

**When to use:** Isotonicモデル学習用のデータ生成。generate_win_oof_predictions()と同じK-foldパターン。

**設計参照:** `src/models/win_benter_gate.py:86-146` (generate_win_oof_predictions)

```python
# Source: src/models/win_benter_gate.py:108-146 (既存パターン)
# KFold(shuffle=False) → 時系列順序を保持
# 各foldで: fold_model学習 → predict_ev → correct_ev → ev_win_corrected取得

df_sorted = df_oof.sort_values("race_date").reset_index(drop=True)
kfold = KFold(n_splits=n_splits, shuffle=False)  # 時系列: shuffle不可
oof_ev_corrected = np.full(len(df_sorted), np.nan)
oof_actual_return = np.full(len(df_sorted), np.nan)

for train_idx, val_idx in kfold.split(df_sorted):
    # fold内で WinTwoStage学習 → predict_ev → correct_ev
    fold_win = WinTwoStageModel()
    fold_win.train_hit_model(df_sorted.iloc[train_idx], num_threads=num_threads)
    fold_win.train_return_model(df_sorted.iloc[train_idx], num_threads=num_threads)
    fold_val = fold_win.predict_ev(df_sorted.iloc[val_idx].copy())
    fold_ev_corr = EVCorrectionModel()
    fold_ev_corr.train(fold_val, num_threads=num_threads)
    fold_val = fold_ev_corr.correct_ev(fold_val)
    oof_ev_corrected[val_idx] = fold_val["ev_win_corrected"].values
    # actual_return = confirmed_odds if winner, else 0
    oof_actual_return[val_idx] = (
        fold_val["confirmed_odds"] * (fold_val["kakuteijyuni"] == 1).astype(float)
    ).values
```

### Pattern 2: Isotonicキャリブレーション適用 (D-01, D-03, D-04)

**What:** OOF EV→actual_returnのIsotonicRegression fit。

```python
# Source: [VERIFIED: sklearn 1.8.0 runtime確認]
from sklearn.isotonic import IsotonicRegression

# D-04: y_min=0 (EV非負), out_of_bounds='clip' (範囲外は最近傍)
iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
iso.fit(oof_ev_corrected_valid, oof_actual_return_valid)

# 推論時:
ev_win_calibrated = iso.transform(ev_win_corrected)  # ndarray返却
```

**重要なsklearn API詳細:**
- `fit(X, y)`: Xは1次元配列必須 (2次元不可)
- `transform(X)`: Xは1次元配列。fit済みXの範囲外はout_of_boundsで処理
- `y_min=0`: 全ての予測値が0以上に制約される。EVが負になるのを防止 [VERIFIED: runtime]
- `out_of_bounds='clip'`: 学習範囲外の入力は最近傍の学習値にクリップ [VERIFIED: runtime]
- `increasing=True` (デフォルト): 単調増加制約。EVが大きいほどactual_returnも大きいという妥当な制約
- `fit()`後の内部属性: `X_thresholds_`, `y_thresholds_` (sklearn 1.8では`f_`に名称変更)
- 欠損値(NaN)はfit/transformの入力に含められない → valid maskで除外必須

### Pattern 3: オッズバンド別残差補正 (D-10)

**What:** Isotonic適用後の残差をオッズバンド別に分析し、補正係数を算出。

**オッズバンド境界:** `src/betting/odds_band_filter.py:16-22`
```python
BANDS = [(1.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, float("inf"))]
BAND_NAMES = ["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"]
```

**推奨手法:** バンド別median残差比スケーリング

```python
# バンド別median残差比: actual / predicted の中央値
for band_name, (lo, hi) in zip(BAND_NAMES, BANDS):
    mask = (oof_odds >= lo) & (oof_odds < hi)
    if mask.sum() >= MIN_SAMPLES:
        residual_ratio = oof_actual[mask] / np.clip(oof_predicted[mask], 1e-6, None)
        band_scale[band_name] = float(np.median(residual_ratio))
    else:
        band_scale[band_name] = 1.0  # サンプル不足は補正なし

# 推論時: 補正係数を乗算
for band_name, (lo, hi) in zip(BAND_NAMES, BANDS):
    mask = (odds >= lo) & (odds < hi)
    calibrated[mask] *= band_scale[band_name]
```

**なぜmedian残差比か:**
- meanは外れ値(大穴的中)に引っ張られる
- ゼロ除算回避 (clipで下限設定)
- スケーリング係数は0.8-1.2の範囲に収まる想定
- バンド別独立Isotonicよりサンプル効率が良い (D-02決定の裏付け)

### Pattern 4: correct_ev()へのIsotonic統合 (D-08)

**What:** EVCorrectionModel.correct_ev()の最後にIsotonic適用を追加。

```python
# 既存コード (src/models/ev_correction_model.py:328-330):
df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
df = df.drop(columns=["_p_win_corrected_raw"], errors="ignore")
return df

# 変更後:
df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
df = df.drop(columns=["_p_win_corrected_raw"], errors="ignore")

# NEW: Isotonicキャリブレーション (D-08)
if self.isotonic_calibrator is not None:
    ev_input = df["ev_win_corrected"].values.astype(float)
    valid = np.isfinite(ev_input)
    calibrated = np.copy(ev_input)
    if valid.any():
        calibrated[valid] = self.isotonic_calibrator.transform(ev_input[valid])
    df["ev_win_calibrated"] = calibrated
else:
    df["ev_win_calibrated"] = df["ev_win_corrected"]  # フォールバック

# NEW: オッズバンド別補正 (D-10)
if self.odds_band_scales is not None:
    # ... バンド別スケーリング適用 ...
    pass

return df
```

### Anti-Patterns to Avoid
- **Isotonicを2次元入力でfit:** `IsotonicRegression.fit(X, y)`のXは1次元必須。DataFrame列を直接渡すとエラーになる → `.values.astype(float)`で1次元ndarrayに変換
- **OOF生成でshuffle=True:** 時系列データでシャッフルはlook-ahead biasの原因 → `KFold(shuffle=False)`必須 (generate_win_oof_predictionsと同じパターン)
- **バックテスト中のIsotonic再学習:** テスト期間データでIsotonicをfitするとデータリーク → 学習済みIsotonicをロードしてtransformのみ (D-07)
- **Isotonicの入力にNaNを含める:** `fit()`も`transform()`もNaNでエラー → valid maskで必ず除外
- **オッズバンド別独立Isotonic:** サンプル不足で過学習リスク → バンド別はスケーリング係数のみ (D-02決定)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EV→actual_returnの非線形キャリブレーション | カスタム区分線形回帰 | sklearn IsotonicRegression | PAVA (Pool Adjacent Violators Algorithm) はsklearn実装が最適化済み、境界処理も完備 |
| モデル永続化 | カスタムpickle | joblib.dump/load | 既存パターンと統一、PFP改ざん検知対象 |
| K-fold分割 | 自前foldインデックス計算 | sklearn KFold(shuffle=False) | generate_win_oof_predictionsと同じパターン |
| オッズバンド判定 | 自前if-elseチェーン | OddsBandFilter._get_band_name() | 既存境界定義と整合性保証 |

**Key insight:** sklearn IsotonicRegressionはこのユースケース(EVの体系的過大評価の是正)に理想的。非パラメトリックなので分布仮定不要、単調制約でEVの順序を保持、y_min=0でEV非負を保証。追加インストール不要で既存依存関係内で完結。

## Common Pitfalls

### Pitfall 1: OOF生成の過度な計算コスト
**What goes wrong:** _train_submodel()内でK-foldループを回すと、既存の学習パイプライン(既に~17分)に5-fold分の追加学習時間がかかる
**Why it happens:** 各foldでWinTwoStage学習(hit+return) + EVCorrection学習が必要
**How to avoid:** fold数は3-5に制限。generate_win_oof_predictionsは5-foldを採用済み。計測・ログ出力でボトルネックを可視化
**Warning signs:** _train_submodel()の実行時間が2倍以上に増加

### Pitfall 2: Isotonic過学習 (少サンプルセグメント)
**What goes wrong:** 高オッズ帯のOOFサンプルが少なく、Isotonicが局所的に不安定になる
**Why it happens:** IsotonicRegressionはバンド境界でステップ関数的な補正を学習する。サンプル不足の領域で極端な補正値が出る可能性
**How to avoid:** y_min=0制約(D-04)で下限を担保。オッズバンド別補正(D-10)はmedianベースで外れ値にロバスト。全体Isotonic + バンド別スケーリングの2段構えで安全性確保
**Warning signs:** Isotonic適用後のEV分布に極端なスパイクやギャップが発生

### Pitfall 3: EV診断列名不一致
**What goes wrong:** ev_diagnostics.pyのEV_PRED_COLUMN = "ev_win_corrected"がIsotonic適用後の列名("ev_win_calibrated")と不一致
**Why it happens:** ev_diagnostics.pyは現状ev_win_correctedのみを対象としている
**How to avoid:** Phase 19ではEV_PRED_COLUMNを変更せず、ev_win_calibratedを別列として追加。ev_diagnosticsは後続Phaseで対応可能。ただしECE評価の正確性のため、学習時に両方の列で診断を実行することを推奨
**Warning signs:** ECE計算がIsotonic適用前の値を返す

### Pitfall 4: SubmodelSetのフィールド追加時の既存テスト互換性
**What goes wrong:** SubmodelSetに新フィールド(isotonic_ev_calibrator等)を追加すると、既存のテストfixtureでSubmodelSetを構築する箇所でTypeErrorが発生
**Why it happens:** dataclassの必須フィールドが増えると、既存のコンストラクタ呼び出しが壊れる
**How to avoid:** 新フィールドはOptional(Noneデフォルト)にする。SubmodelSetの既存Optionalフィールド(| None = None)のパターンに従う
**Warning signs:** 既存テストのSubmodelSet()構築でTypeError

### Pitfall 5: correct_ev()のIsotonic未学習時フォールバック
**What goes wrong:** Isotonicが未学習(None)の状態でcorrect_ev()が呼ばれるとAttributeError
**Why it happens:** バックテストの学習前フェーズや、古いモデルを読み込んだ場合にIsotonicが存在しない
**How to avoid:** `if self.isotonic_calibrator is not None:` でガード。未学習時は`ev_win_calibrated = ev_win_corrected`でフォールバック。PlaceEVCorrectionModel._trainedパターンと同じ
**Warning signs:** モデルロード後にcorrect_ev()がAttributeErrorで失敗

## Code Examples

### IsotonicRegression基本パターン (sklearn 1.8.0)
```python
# Source: [VERIFIED: runtime動作確認]
from sklearn.isotonic import IsotonicRegression
import numpy as np

# 学習: X=ev_win_corrected(OOF), y=actual_return(OOF)
iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
valid = np.isfinite(oof_ev) & np.isfinite(oof_actual)
iso.fit(oof_ev[valid], oof_actual[valid])

# 推論: 1次元配列入力 → 1次元配列出力
calibrated = iso.transform(new_ev_values)
# 範囲外はclip: 学習X_min以下 → y_min(=0), 学習X_max以上 → y_max(学習値)
```

### OOF EV生成パターン (generate_win_oof_predictions参照)
```python
# Source: src/models/win_benter_gate.py:108-146
from sklearn.model_selection import KFold

df_sorted = df_oof.sort_values("race_date").reset_index(drop=True)
kfold = KFold(n_splits=5, shuffle=False)  # 時系列: shuffle不可
oof_ev = np.full(len(df_sorted), np.nan)
oof_actual = np.full(len(df_sorted), np.nan)

for train_idx, val_idx in kfold.split(df_sorted):
    # fold内でフルパイプライン: WinTwoStage → EVCorrection
    fold_win = WinTwoStageModel()
    fold_win.train_hit_model(df_sorted.iloc[train_idx], num_threads=nt)
    fold_win.train_return_model(df_sorted.iloc[train_idx], num_threads=nt)
    fold_val = fold_win.predict_ev(df_sorted.iloc[val_idx].copy())
    fold_ev = EVCorrectionModel()
    fold_ev.train(fold_val, num_threads=nt)
    fold_val = fold_ev.correct_ev(fold_val)
    oof_ev[val_idx] = fold_val["ev_win_corrected"].values
    oof_actual[val_idx] = (
        fold_val.get("confirmed_odds", fold_val.get("odds", 0))
        * (fold_val["kakuteijyuni"] == 1).astype(float)
    ).values

# Isotonic fit
valid = np.isfinite(oof_ev) & np.isfinite(oof_actual) & (oof_ev > 0)
iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
iso.fit(oof_ev[valid], oof_actual[valid])
```

### SubmodelSetフィールド追加パターン
```python
# Source: src/domain/models.py:229-258
@dataclass
class SubmodelSet:
    # ... 既存フィールド ...
    win_isotonic_calibrator: IsotonicRegression | None = None  # Phase 2 既存
    win_temperature_scaler: TemperatureScaling | None = None    # Phase 2 既存
    ev_lower_threshold_turf: float = 1.0
    ev_lower_threshold_dirt: float = 1.0
    # NEW Phase 19: EV Isotonic + オッズバンド別スケーリング
    ev_isotonic_calibrator: IsotonicRegression | None = None
    ev_odds_band_scales: dict[str, float] | None = None  # {"1.0-3.0": 0.95, ...}
```

### ModelLoader読み込みパターン
```python
# Source: src/db/model_loader.py:665-672 (既存win_isotonicパターン)
# Local読み込み:
ev_isotonic_calibrator = None
ev_iso_file = models_dir / f"ev_isotonic_{surface}.joblib"
if ev_iso_file.is_file():
    try:
        ev_isotonic_calibrator = joblib.load(ev_iso_file)
    except Exception:
        logger.warning("Failed to load %s, skipping", ev_iso_file)

# オッズバンド別スケーリング (JSON):
ev_odds_band_scales = None
band_file = models_dir / f"ev_odds_band_scales_{surface}.json"
if band_file.is_file():
    with open(band_file) as f:
        ev_odds_band_scales = json.load(f)
```

### _save_models_local保存パターン
```python
# Source: src/pipelines/training_pipeline.py:1322-1327 (既存パターン)
# Isotonic (joblib):
if sub.ev_isotonic_calibrator is not None:
    joblib.dump(
        sub.ev_isotonic_calibrator,
        models_dir / f"ev_isotonic_{surface}.joblib",
    )

# オッズバンド別スケーリング (JSON):
if sub.ev_odds_band_scales is not None:
    with open(models_dir / f"ev_odds_band_scales_{surface}.json", "w") as f:
        json.dump(sub.ev_odds_band_scales, f, indent=2)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| EV補正なし(raw PxE) | P補正 + E補正 (EVCorrectionModel) | v5.4 | PxE独立性バイアスの一部是正 |
| 固定EV補正 | Isotonic RegressionでEV直接キャリブレーション | Phase 19 (今回) | 非パラメトリックに体系バイアスを吸収 |
| グローバルIsotonic | サーフェス別Isotonic + オッズバンド別残差補正 | Phase 19 (今回) | セグメント別の過大評価倍率を1.0±0.2に収束 |

**Deprecated/outdated:**
- sklearn IsotonicRegressionの内部属性 `X_`, `y_` → sklearn 1.8では `X_thresholds_`, `y_thresholds_` に名称変更

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | K-fold OOF EV生成は5-foldで十分な品質を得る | Pattern 1 | 3-foldでも十分かもしれないし、5-foldでも不十分かもしれない。サンプル数次第 |
| A2 | median残差比によるオッズバンド別補正は十分な精度 | Pattern 3 | バンド内の残差分布が非対称な場合、medianが最適でない可能性 |
| A3 | 計算コスト増加は許容範囲(既存~17分 + 5-fold分) | Pitfall 1 | バックテストの~57分/年からの追加が大きい可能性 |
| A4 | Isotonicの単調増加制約(increasing=True)がEV→actual_return関係に適切 | Pattern 2 | EV低値域でactual > EVの逆転がある場合、過剰制約になる可能性 |

**If this table is empty:** 全ての主要な技術的判断はCONTEXT.mdの決定事項(D-01〜D-10)に基づき、sklearn API動作確認済み。

## Open Questions

1. **OOF fold数の最適値**
   - What we know: generate_win_oof_predictions()は5-fold。StackedEnsembleは3-fold。AbilityModel.train_oof()も3-fold。
   - What's unclear: EV Isotonic用には何foldが最適か
   - Recommendation: 5-foldで開始(generate_win_oof_predictionsと統一)。Claude's Discretionで決定

2. **correct_ev()への注入方法の詳細**
   - What we know: D-08で「correct_ev()の最後」に適用と決定
   - What's unclear: IsotonicモデルをEVCorrectionModelのコンストラクタ引数にするか、インスタンス変数にするか
   - Recommendation: コンストラクタ注入パターン(Phase 12-13で確立)に従う。`__init__`に`isotonic_calibrator`引数を追加し、`correct_ev()`内で`self.isotonic_calibrator`を参照

3. **ev_win_calibrated列のdownstream影響**
   - What we know: ev_diagnostics.pyはEV_PRED_COLUMN = "ev_win_corrected"を参照。RacePredictor.get_win_candidates()は"win_selection_edge"を使用。
   - What's unclear: ev_win_calibratedを既存のev_win_correctedの代わりに使うべきか、追加列にすべきか
   - Recommendation: Phase 19では追加列(ev_win_calibrated)として生成。downstreamのRacePredictor/BacktestEngineがev_win_correctedのままでも動作するよう互換性維持。Phase 22統合検証でev_win_calibratedへの切替を評価

## Environment Availability

Step 2.6: SKIPPED (外部依存なし — 純粋なコード変更のみ)

このフェーズはsklearn IsotonicRegression(既にインストール済みv1.8.0)と既存コードの変更のみ。外部ツール、サービス、ランタイムの追加は不要。

## Sources

### Primary (HIGH confidence)
- sklearn 1.8.0 runtime検証 — IsotonicRegression(y_min=0, out_of_bounds='clip')の動作確認
- `src/models/ev_correction_model.py` — EVCorrectionModel.correct_ev()の現状実装(全行精読)
- `src/pipelines/training_pipeline.py` — _train_submodel()(全行精読), _save_models_local()(全行精読)
- `src/models/stacked_ensemble.py` — K-fold OOF予測パターン(全行精読)
- `src/domain/models.py` — SubmodelSet dataclass(全行精読)
- `src/db/model_loader.py` — load_from_dir()(全行精読)
- `src/models/win_benter_gate.py` — generate_win_oof_predictions()(全行精読)
- `src/betting/odds_band_filter.py` — OddsBandFilter.BANDS定義(全行精読)

### Secondary (MEDIUM confidence)
- `src/models/ev_diagnostics.py` — compute_ev_diagnostics()(全行精読)
- `src/backtest/race_predictor.py:400-500` — get_win_candidates()(精読)
- `tests/test_ev_correction.py` — 既存テストパターン(全行精読)

### Tertiary (LOW confidence)
- なし — 全てのコードは直接精読済み

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — sklearn IsotonicRegression APIをruntime検証済み
- Architecture: HIGH — 全変更対象ファイルを精読、既存パターンと整合性確認済み
- Pitfalls: HIGH — 既存テストコードから潜在的互換性問題を特定済み
- OOF生成: MEDIUM — generate_win_oof_predictionsパターンを流用できるが、EV補正チェーン(Ability→WinTwoStage→EVCorrection)のfold内再学習は新規実装

**Research date:** 2026-05-07
**Valid until:** 2026-06-07 (stable — sklearn APIは安定)
