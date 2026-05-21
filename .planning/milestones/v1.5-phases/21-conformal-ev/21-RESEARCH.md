# Phase 21: Conformal EV予測区間 - Research

**Researched:** 2026-05-09
**Domain:** Conformal Prediction / CQR (Conformalized Quantile Regression) / LightGBM quantile regression
**Confidence:** HIGH

## Summary

Phase 21は、既存の`RobustConfidenceEstimator`（絶対値残差ベース）をCQR（Conformalized Quantile Regression, Romano et al. 2019）に完全置き換え、EV推定の不確実性を分布フリーに定量化する。CQRは異分散性（heteroscedasticity）を直接扱えるため、高オッズ帯と低オッズ帯で分散が異なるKEIBA-AIのEV分布に適合する。

既存パイプライン（TrainingPipelineV5._train_submodel()、RacePredictor.predict()、BacktestEngine）への統合パターンはPhase 19（Isotonic EVキャリブレーション）で確立済み。CQRモデルはサーフェス別（芝/ダート）に4つのLightGBM quantileモデル（α/2, 1-α/2の各サーフェス）を学習し、バリデーション分割（後方20%）の予測結果から非適合スコアを計算する。K-fold OOFを使わないため、実行時間への影響は数秒〜数十秒に留まる。

**Primary recommendation:** ConformalEVModel新規クラスを作成。`calibrate()`で非適合スコア計算、`predict_interval()`でEV下限/上限出力。出力列名は既存の`EV_lower_win_corrected`, `EV_upper_win_corrected`, `conformal_confidence_score`を維持し、RacePredictorのフィルター変更を不要にする。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** RobustConfidenceEstimatorを完全に置き換え、新規ConformalEVModelクラスを作成。CQR採用
- **D-02:** LightGBM quantile regressionでα/2と1-α/2の2つの分位点モデルをサーフェス別に学習。α=0.1の場合、quantile=0.05と0.95。計4つのLightGBMモデル
- **D-03:** CQRのターゲットはev_win_calibrated（Phase 19 Isotonic適用後のEV）
- **D-04:** 2-alpha構成（80%+90%）を採用。90%区間でフィルタリング、80%区間でconfidence_score計算
- **D-05:** フィルタリング閾値はEV_lower_90 < 1.0の固定閾値（JRA控除率25%後の損益分岐点）
- **D-06:** ConformalEVModelの出力列名は既存と同じ（EV_lower_win_corrected, EV_upper_win_corrected, conformal_confidence_score）
- **D-07:** TrainingPipelineV5の既存バリデーション分割（race_date後方20%）の予測結果からCQRを学習。K-fold OOFは使わない
- **D-08:** バリデーション分割での推論チェーン: AbilityModel -> WinTwoStage -> EVCorrection -> Isotonic -> ConformalEV
- **D-09:** CQR学習を_train_submodel()の学習チェーン末尾に追加。依存順序: Isotonic -> ConformalEV
- **D-10:** ConformalEVModelをSubmodelSetに追加（Phase 19パターン踏襲）。ModelLoaderが自動読み込み。PFP改ざん検知対象。.lgb形式で保存
- **D-11:** 既存EV診断（ev_diagnostics.py）を拡張してCQRカバレッジ率・区間幅の指標を追加

### Claude's Discretion
- ConformalEVModelクラスの具体的なAPI設計（calibrate/predict_intervalメソッドのシグネチャ）
- LightGBM quantile regressionのハイパーパラメータ（既存モデルと同じかCQR用に調整）
- CQR非適合スコアの計算詳細（conformity score = max(q_low - y, y - q_high)の標準実装）
- サーフェス別CQRモデルのSubmodelSetフィールド命名規則
- バリデーション分割でのactual_return計算方法（win payout map参照）
- テストのfixtureデータとモック構成
- RobustConfidenceEstimatorの削除に伴う既存テストの移行

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CONF-01 | CQRベースConformal Prediction EV区間実装。LightGBM quantile regression + CP補正 | LightGBM quantile API検証済み（objective='quantile', alphaパラメータ）。CQR非適合スコア: max(q_low - y, y - q_high)。バリデーション分割(20%)で学習。4モデル構成(2 quantiles x 2 surfaces) |
| CONF-02 | EV信頼区間下界に基づく動的フィルタリング（EV_lower_90 < 1.0で除外） | 既存race_predictor.py:432-482のフィルターロジックは変更不要（出力列名互換）。BacktestEngine.n_ev_excluded伝播も変更不要。閾値1.0固定 |
| CONF-03 | ConformalEVModelのパイプライン統合・診断レポート更新 | TrainingPipelineV5._train_submodel()への統合箇所特定済み。SubmodelSetフィールド追加。ModelLoader保存/読み込み拡張。ev_diagnostics.py拡張 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CQR分位点モデル学習 | API/Backend (TrainingPipeline) | - | 学習パイプラインでLightGBM quantileモデルを学習 |
| 非適合スコア計算 | API/Backend (ConformalEVModel) | - | バリデーション分割結果からCQR補正量子を計算 |
| EV区間予測 | API/Backend (RacePredictor) | - | 推論時にCQRモデルを適用してEV下限/上限を計算 |
| 動的フィルタリング | API/Backend (RacePredictor.get_win_candidates) | - | EV_lower < 1.0の候補を除外。既存ロジック再利用 |
| カバレッジ診断 | API/Backend (ev_diagnostics.py) | - | CQR区間のカバレッジ率・区間幅を評価 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LightGBM | (既存依存) | quantile regression for CQR | objective='quantile', alpha=0.05/0.95で分位点モデルを構築。既にプロジェクト依存関係に含まれる [VERIFIED: 既存コードで確認] |
| numpy | (既存依存) | 非適合スコアのquantile計算 | np.quantile()でQ_{1-alpha}を計算。CQRの補正量子算出 [VERIFIED: RobustConfidenceEstimatorで使用済み] |
| scikit-learn IsotonicRegression | (既存依存) | Phase 19のEVキャリブレーション | CQRの入力ev_win_calibratedの生成に使用。新規インストール不要 [VERIFIED: ev_correction_model.pyで確認] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | (既存依存) | DataFrame操作 | バリデーション分割・特徴量抽出・予測結果の結合 |
| scipy | (既存依存) | 統計検定 | カバレッジ検証の信頼区間計算（オプション） |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LightGBM quantile | sklearn GradientBoosting quantile | LightGBMの方が高速・メモリ効率良。既に依存関係にあるため統一性が高い |
| 単一alpha構成 | 2-alpha構成(80%+90%) | 単一alphaはシンプルだがconfidence_scoreの質が下がる。D-04で2-alphaが決定済み |

**Installation:**
```bash
# 追加インストール不要 — LightGBM, numpy, pandas, scikit-learnは既存依存
```

## Architecture Patterns

### System Architecture Diagram

```
TrainingPipelineV5._train_submodel()
    |
    v
[1] df_oof = AbilityModel → WinTwoStage → EVCorrection → Isotonic (Phase 19)
    |                    (学習チェーン既存部分)
    v
[2] Validation Split: df_oof の後方20% → df_cqr_calib
    |
    v
[3] CQR Quantile Model Training:
    ├── q_low_model  (α/2 = 0.05)  ← LightGBM quantile, features → ev_win_calibrated
    └── q_high_model (1-α/2 = 0.95) ← LightGBM quantile, features → ev_win_calibrated
    |
    v
[4] CQR Calibration:
    ├── df_cqr_calib に q_low, q_high を予測
    ├── E_i = max(q_low - y, y - q_high)  ← 非適合スコア
    └── Q_calib = np.quantile(E, 1-α)     ← CQR補正量子
    |
    v
[5] Save: SubmodelSet.conformal_ev_model = {q_low, q_high, Q_calib, ...}
    |
    v
RacePredictor.predict() [推論時]
    |
    v
[6] correct_ev() → ev_win_calibrated 生成
    |
    v
[7] ConformalEVModel.predict_interval():
    ├── q_low = q_low_model.predict(features)
    ├── q_high = q_high_model.predict(features)
    ├── EV_lower_90 = max(q_low - Q_calib, 0)
    ├── EV_upper_90 = q_high + Q_calib
    ├── (80%区間も同様に計算)
    └── conformal_confidence_score = EV_lower_80 × (1 - normalized_width)
    |
    v
[8] get_win_candidates(): EV_lower_win_corrected < 1.0 を除外
```

### Recommended Project Structure
```
src/models/
├── conformal_ev_model.py        # 新規: ConformalEVModelクラス
├── robust_confidence_estimator.py  # 削除対象
├── ev_correction_model.py        # 変更なし（ev_win_calibrated出力）
├── ev_diagnostics.py             # 拡張: CQRカバレッジ指標追加
src/domain/
├── models.py                     # SubmodelSet: conformal_ev_model フィールド追加
src/db/
├── model_loader.py               # CQRモデル保存/読み込み追加
src/pipelines/
├── training_pipeline.py          # _train_submodel()にCQR学習追加
src/backtest/
├── race_predictor.py             # confidence → conformal_ev_model に差し替え
tests/
├── test_conformal_ev_model.py    # 新規テスト
├── test_robust_confidence_estimator.py  # 削除（または置き換え）
```

### Pattern 1: CQR Nonconformity Score (Romano et al. 2019)
**What:** 分布フリーの予測区間を構成するCQR手法
**When to use:** 異分散性のある回帰問題の予測区間推定
**Example:**
```python
# Source: CQR (Romano et al., NeurIPS 2019) - https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf
import numpy as np
import lightgbm as lgb

# Step 1: 分位点モデル学習
q_low_model = lgb.train(
    {"objective": "quantile", "alpha": 0.05, ...},
    train_data,
)
q_high_model = lgb.train(
    {"objective": "quantile", "alpha": 0.95, ...},
    train_data,
)

# Step 2: キャリブレーションセットで非適合スコア計算
q_low_pred = q_low_model.predict(X_calib)
q_high_pred = q_high_model.predict(X_calib)
nonconformity_scores = np.maximum(q_low_pred - y_calib, y_calib - q_high_pred)

# Step 3: 補正量子を計算（有限サンプル補正付き）
n = len(nonconformity_scores)
Q_calib = np.quantile(nonconformity_scores, min((1 - alpha) * (1 + 1/n), 1.0))

# Step 4: 予測区間の生成
q_low_new = q_low_model.predict(X_new)
q_high_new = q_high_model.predict(X_new)
lower = q_low_new - Q_calib
upper = q_high_new + Q_calib
```

### Pattern 2: SubmodelSet統合パターン (Phase 19踏襲)
**What:** 新規モデルをSubmodelSetに追加する確立パターン
**When to use:** パイプラインに新しいモデルを統合する時
**Example:**
```python
# SubmodelSet (domain/models.py)
@dataclass
class SubmodelSet:
    # ... existing fields ...
    ev_isotonic_calibrator: IsotonicRegression | None = None  # Phase 19
    ev_odds_band_scales: dict[str, float] | None = None       # Phase 19
    conformal_ev_model: ConformalEVModel | None = None         # Phase 21 NEW
    confidence: RobustConfidenceEstimator  # Phase 21: 削除してconformal_ev_modelに置き換え

# ModelLoader._load_from_local() (db/model_loader.py)
conformal_ev = None
cqr_low_file = models_dir / f"cqr_quantile_low_{surface}.lgb"
cqr_high_file = models_dir / f"cqr_quantile_high_{surface}.lgb"
cqr_params_file = models_dir / f"cqr_params_{surface}.json"
if cqr_low_file.is_file() and cqr_high_file.is_file() and cqr_params_file.is_file():
    conformal_ev = ConformalEVModel()
    conformal_ev.q_low_model = self._load_lgbm(str(cqr_low_file))
    conformal_ev.q_high_model = self._load_lgbm(str(cqr_high_file))
    with open(cqr_params_file) as f:
        params = json.load(f)
    conformal_ev._calibration_quantile = params["calibration_quantile"]
    conformal_ev._calibrated = True
```

### Anti-Patterns to Avoid
- **絶対値残差ベースのCP:** 異分散性を扱えず、全EVレベルで一定幅の区間になる。高オッズ帯の分散が大きい競馬EVでは不適切。これがRobustConfidenceEstimatorの根本問題であり、EV_excluded=0の原因
- **K-fold OOFの使用:** Phase 19でrun_backtest.pyの実行時間が3時間+に増加した問題の再発。バリデーション分割(後方20%)を使用する
- **ev_win_correctedをCQR入力に使用:** Phase 19のIsotonic適用後のev_win_calibratedがCQRの正しいターゲット（D-03）
- **predict_intervalでのev_win_corrected参照:** 既存コードは`ev_win_corrected`列を参照しているが、Phase 21では`ev_win_calibrated`列が存在するため、そちらをCQRのベースEVとして使用すべき

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 分位点回帰 | カスタムquantile損失関数 | LightGBM objective='quantile' | 最適化済み。alphaパラメータで任意の分位点を指定可能 [VERIFIED: LightGBM公式ドキュメント] |
| 非適合スコアのquantile計算 | 手動ソート+線形補間 | np.quantile() | numpyの実装は正確かつ高速。有限サンプル補正は(1-alpha)*(1+1/n)で実装 |
| モデル保存/読み込み | カスタムシリアライズ | LightGBM .save_model() / Booster(model_file=) | 既存パターン（_save_models_local / _load_lgbm）と統一 |
| EV補正後の値 | ConformalEVModel内でIsotonic再適用 | ev_win_calibrated列（EVCorrectionModel既存出力） | 正しい推論チェーン。Phase 19で生成済み |

**Key insight:** CQRの核心理論はシンプル（分位点予測 + 非適合スコアのquantile補正）。複雑なのは統合ポイントの特定と出力互換性の維持。

## Common Pitfalls

### Pitfall 1: バリデーション分割でのactual_return計算エラー
**What goes wrong:** バリデーション分割（後方20%）のactual_returnを誤って計算し、CQR補正量子が不正確になる
**Why it happens:** actual_ev_win = confirmed_odds * (kakuteijyuni == 1) の計算で、confirmed_oddsが存在しない場合やNaNが含まれる場合に対応漏れ
**How to avoid:** TrainingPipelineの既存パターン（lines 854-856）をそのまま使用。actual_ev_win = confirmed_odds * (kakuteijyuni == 1).astype(int)
**Warning signs:** CQRの非適合スコアが極端に大きい/小さい。カバレッジ率が90%から大幅に乖離

### Pitfall 2: 推論時の特徴量カラム不一致
**What goes wrong:** CQRモデルの学習時特徴量と推論時特徴量が不一致でLightGBM predict()がエラー
**Why it happens:** _train_submodel()内でdf_oofに特徴量を追加しているが、推論時（RacePredictor.predict()）のDataFrameには同じ特徴量が存在しない
**How to avoid:** CQRモデルの特徴量は、correct_ev()完了時のdf_oofに存在する列のみを使用。FEATURE_COLSを明示的に定義して学習・推論で共有
**Warning signs:** 推論時のpredict()でLightGBMが特徴量不足エラーを発生

### Pitfall 3: quantileモデルのモノトonicity違反
**What goes wrong:** q_low > q_high となる予測が生じ、非適合スコアが負になる
**Why it happens:** LightGBMの2つの独立したquantileモデルは、理論上q_low <= q_highを保証しない
**How to avoid:** predict_interval()内で np.minimum(lower, ev_calibrated) のクリップ処理を追加。非適合スコアはclip(lower=0)で保護。CQRの補正量子適用後にlower = max(q_low - Q_calib, 0)で下限を0にクリップ
**Warning signs:** EV_lower_win_correctedがEV_upper_win_correctedより大きい行が存在

### Pitfall 4: RobustConfidenceEstimatorの残留参照
**What goes wrong:** import文やSubmodelSetフィールドが更新されず、ランタイムエラーが発生
**Why it happens:** RobustConfidenceEstimatorは6箇所でimport/参照されている（domain/models.py, pipelines/training_pipeline.py, db/model_loader.py, models/__init__.py, race_predictor.py内のTYPE_CHECKING, backtest_engine.py）
**How to avoid:** 置き換え時に全参照箇所をgrepで確認。SubmodelSetの`confidence`フィールドを`conformal_ev_model`にリネームし、型もConformalEVModelに変更。model_loader.py、training_pipeline.py、race_predictor.pyの3ファイルは必須更新
**Warning signs:** CI/CDでModuleNotFoundErrorまたはAttributeError

### Pitfall 5: CQR学習時のサンプル不足
**What goes wrong:** バリデーション分割（後方20%）のサンプル数が不足し、quantileモデルが不安定
**Why it happens:** サーフェス別(芝/ダート)に分割すると、各20%のデータ量が減少。特にダートはサンプルが少ない傾向
**How to avoid:** 学習前にサンプル数チェック（Phase 19パターン: `if len(df_oof) >= 500`）。最低サンプル数閾値（200程度）を設定し、不足時はフォールバック（ev_win_calibratedをそのままEV_lower/upperとする）
**Warning signs:** CQRモデル学習時のLightGBM warning "no valid splits"。カバレッジ率が80%未満

## Code Examples

### ConformalEVModel API設計案

```python
# src/models/conformal_ev_model.py
import logging
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ConformalEVModel:
    """CQR (Conformalized Quantile Regression) によるEV予測区間推定.

    Romano et al., 2019 "Conformalized Quantile Regression" に基づく。
    LightGBM quantile regression で α/2 と 1-α/2 の分位点を学習し、
    非適合スコアでCP補正を適用する。
    """

    def __init__(
        self,
        alpha: float = 0.1,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.feature_cols = feature_cols
        self._calibrated = False
        # LightGBM quantile models
        self.q_low_model: lgb.Booster | None = None
        self.q_high_model: lgb.Booster | None = None
        # CQR calibration quantile
        self._calibration_quantile_90: float = 0.0
        self._calibration_quantile_80: float = 0.0
        # Secondary alpha for confidence scoring
        self._alpha_secondary: float = 0.2  # 80% interval

    def train(
        self,
        df_calib: pd.DataFrame,
        *,
        num_threads: int = 0,
        lgb_params: dict | None = None,
    ) -> None:
        """CQRモデルを学習・キャリブレーション.

        Args:
            df_calib: ev_win_calibrated, actual_ev_win, 特徴量列を含むDataFrame
            num_threads: LightGBMスレッド数
            lgb_params: LightGBM追加パラメータ
        """
        ...  # 実装

    def predict_interval(
        self,
        df: pd.DataFrame,
        alphas: tuple[float, ...] = (0.1, 0.2),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """EVの信頼区間(上下)を複数水準で推定.

        既存RobustConfidenceEstimator.predict_interval()と同じシグネチャ・出力。

        Args:
            df: ev_win_calibrated と特徴量列を含むDataFrame
            alphas: 信頼水準のタプル (0.1=90%区間, 0.2=80%区間)

        Returns:
            (win_df, place_df) with EV_lower/upper columns and conformal_confidence_score
        """
        ...  # 実装
```

### CQR Calibration Flow in _train_submodel()

```python
# src/pipelines/training_pipeline.py (_train_submodel末尾に追加)
# D-07/D-08/D-09: CQR学習 — バリデーション分割(後方20%)を使用

conformal_ev: ConformalEVModel | None = None
if len(df_oof) >= 500 and "ev_win_calibrated" in df_oof.columns:
    with TimingContext(f"{surface}/conformal_ev"):
        # バリデーション分割: 後方20%
        n_total = len(df_oof)
        split_idx = int(n_total * 0.8)
        df_cqr_train = df_oof.iloc[:split_idx]
        df_cqr_calib = df_oof.iloc[split_idx:]

        # actual_ev_win を計算 (Phase 19パターン)
        df_cqr_train = df_cqr_train.copy()
        df_cqr_train["actual_ev_win"] = (
            df_cqr_train["confirmed_odds"]
            * (df_cqr_train["kakuteijyuni"] == 1).astype(float)
        )
        df_cqr_calib = df_cqr_calib.copy()
        df_cqr_calib["actual_ev_win"] = (
            df_cqr_calib["confirmed_odds"]
            * (df_cqr_calib["kakuteijyuni"] == 1).astype(float)
        )

        conformal_ev = ConformalEVModel(
            alpha=0.1,
            feature_cols=FEATURE_COLS,  # 学習時と同じ特徴量
        )
        conformal_ev.train(df_cqr_calib, num_threads=num_threads)
```

### EV Diagnostics拡張 (CQR Coverage Metrics)

```python
# src/models/ev_diagnostics.py に追加
# D-11: CQRカバレッジ指標

def _compute_cqr_coverage(
    df: pd.DataFrame,
    pred_col: str = "ev_win_calibrated",
    lower_col: str = "EV_lower_win_corrected",
    upper_col: str = "EV_upper_win_corrected",
    actual_col: str = "actual_ev_win",
) -> dict:
    """CQR区間のカバレッジ率と区間幅を計算."""
    lower = pd.to_numeric(df[lower_col], errors="coerce")
    upper = pd.to_numeric(df[upper_col], errors="coerce")
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    valid = lower.notna() & upper.notna() & actual.notna()

    if valid.sum() < 30:
        return {"warning": "insufficient_samples"}

    coverage = float(((actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid])).mean())
    width = (upper[valid] - lower[valid])
    return {
        "coverage_rate": coverage,
        "mean_interval_width": float(width.mean()),
        "median_interval_width": float(width.median()),
        "n_samples": int(valid.sum()),
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 絶対値残差CP (RobustConfidenceEstimator) | CQR (Conformalized Quantile Regression) | Phase 21 (2026-05) | 異分散性対応。EVレベル別に適応的な区間幅。EV_excluded=0を解消 |
| K-fold OOFキャリブレーション | バリデーション分割(後方20%) | Phase 21 (D-07) | 実行時間影響最小（数秒〜数十秒 vs 3時間+） |
| 単一alpha構成 | 2-alpha構成(80%+90%) | Phase 21 (D-04) | フィルタリング(90%)とconfidence_score(80%)の分離 |

**Deprecated/outdated:**
- `RobustConfidenceEstimator`: 絶対値残差ベース。異分散性を扱えず、EV_excluded=0の根本原因。Phase 21で完全削除
- `predict_lower_bound()`: `predict_interval()`の後方互換ラッパー。ConformalEVModelでは`predict_interval()`のみ実装すれば十分（呼び出し側は既に`predict_interval()`を使用）

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | LightGBM quantile objective='quantile'はalpha=0.05とalpha=0.95で独立したモデルとして学習可能 | Standard Stack | quantile regressionの精度に影響。ただしLightGBMの公式機能でありリスク低 |
| A2 | バリデーション分割(後方20%)のデータ量はCQR学習に十分（各サーフェス500+サンプル） | Architecture Patterns | データ不足の場合フォールバック動作が必要 |
| A3 | RacePredictor.predict()のline 169の`submodel.confidence.predict_interval(df, df)`は`submodel.conformal_ev_model.predict_interval(df, df)`に置き換え可能 | Architecture Patterns | シグネチャ互換性の前提。出力形式が同じであれば変更不要 |
| A4 | CQRの特徴量はev_win_calibrated計算時に利用可能な列に制限される | Architecture Patterns | 推論時に特徴量が不足する可能性。FEATURE_COLSを明示的に定義する必要あり |
| A5 | 2-alpha構成の80%区間は90%区間と同じモデルを使用し、異なる補正量子を計算 | Code Examples | 80%用の別モデルを学習する必要が生じる場合は実行時間増 |

## Open Questions (RESOLVED)

1. **CQR特徴量の選択**
   - What we know: correct_ev()完了後のdf_oofに含まれる全特徴量列が利用可能。LightGBMは不要特徴量を自動的に無視する傾向がある
   - What's unclear: 最適な特徴量サブセット。全特徴量（300+列）を使うか、EV関連列に絞るか
   - Recommendation: 既存のWinTwoStageモデルと同じ特徴量列をベースにする。LightGBMのfeature importanceで重要でない特徴量は自然に無視されるため、過剰な特徴量選択は不要
   - RESOLVED: Plan 21-01 Task 1で_non_feature_cols除外リストを定義し、残りを自動特徴量として採用。Plan 21-02 Task 1で_train_submodel()内にfeature_cols抽出を実装

2. **80%区間の補正量子計算方法**
   - What we know: D-04で2-alpha構成が決定。90%と80%の異なる区間が必要
   - What's unclear: 80%区間の補正量子を同じq_low/q_highモデルから計算するか、別モデルを学習するか
   - Recommendation: 同じq_low(0.05)/q_high(0.95)モデルを使用し、alpha=0.2で補正量子を再計算するアプローチを推奨。別モデルは4→8モデルに増加し、管理コストが高い。ただし、より厳密にはalpha=0.1と0.2で別々の分位点(0.1/0.9)モデルを学習すべき。トレードオフを議論する価値あり
   - RESOLVED: 同じq_low/q_highモデルでalpha=0.2の補正量子を再計算する方針を採用。Plan 21-01 Task 1で_calibration_quantile_90/80の両方をtrain()で計算する設計を実装。モデル数は4のまま管理コストを抑える


## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified - 全て既存依存パッケージで対応可能)

## Validation Architecture

> workflow.nyquist_validation is explicitly set to false in .planning/config.json — セクション省略。

## Security Domain

> セキュリティ関連の変更なし（モデル置き換えのみ）。セクション省略。

## Sources

### Primary (HIGH confidence)
- コードベース直接確認: `src/models/robust_confidence_estimator.py` (253行) — 既存CP実装の完全な構造・API・出力列名
- コードベース直接確認: `src/backtest/race_predictor.py` (952行) — predict()推論チェーン、get_win_candidates()フィルター
- コードベース直接確認: `src/pipelines/training_pipeline.py` — _train_submodel()学習チェーン、_save_models_local()保存パターン
- コードベース直接確認: `src/domain/models.py` — SubmodelSet dataclass構造
- コードベース直接確認: `src/db/model_loader.py` — load_from_dir()読み込みパターン
- コードベース直接確認: `src/models/ev_diagnostics.py` — 診断計算パターン
- コードベース直接確認: `src/models/ev_correction_model.py` — correct_ev()、ev_win_calibrated生成
- コードベース直接確認: `tests/test_robust_confidence_estimator.py` — テストパターン・fixture構造

### Secondary (MEDIUM confidence)
- [LightGBM Parameters Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — quantile objective, alpha parameter [CITED]
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html) — LGBMRegressor/Booster API [CITED]
- [Romano et al., 2019 "Conformalized Quantile Regression" (NeurIPS)](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf) — CQR理論・非適合スコア定義 [CITED]
- [CQR Official Implementation (GitHub)](https://github.com/yromano/cqr) — 非適合スコア実装の参照 [CITED]

### Tertiary (LOW confidence)
- [LightGBM Quantile Regression Tutorial (Medium)](https://medium.com/@suraj_bansal/quantile-regression-in-python-with-lightgbm-predicting-percentiles-part-1-460e6756a053) — 使用パターン参照
- [IBM Prediction Intervals Tutorial](https://developer.ibm.com/articles/prediction-intervals-explained-a-lightgbm-tutorial/) — 実装例参照

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全て既存依存パッケージ。LightGBM quantile APIは公式ドキュメントで確認済み
- Architecture: HIGH - コードベース直接確認。Phase 19統合パターンの踏襲。全統合ポイント特定済み
- Pitfalls: HIGH - コードベースgrepで全参照箇所確認。Phase 19の実行時間問題を回避する設計

**Research date:** 2026-05-09
**Valid until:** 2026-06-09 (stable domain - 30 days)
