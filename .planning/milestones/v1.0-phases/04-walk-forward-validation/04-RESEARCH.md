# Phase 4: Walk-Forward Validation - Research

**Researched:** 2026-05-03
**Domain:** Walk-forward cross-validation, overfitting detection, feature importance stability
**Confidence:** HIGH

## Summary

Phase 4 は、Phase 1-3 で実装した単勝モデル改善(特徴量分析・Benter キャリブレーション・選択ゲート)が過学習していないことを証明する検証フェーズ。既存の WalkForwardCV クラスを拡張し、各フォールドで train 期間のバックテストも実行して train-test ROI gap を計測する。2 フォールド(2024, 2025 テスト)のウォークフォワード検証を行い、ROI gap + feature importance 安定性 + プール ROI の 3 観点で自動 PASS/FAIL 判定を実装する。

**Primary recommendation:** 既存 WalkForwardCV.generate_folds() は D-02 のフォールド構成を生成できないバグがあるため、新規スクリプト run_wf_validation.py でフォールド定義をハードコードし、WalkForwardCV の run() を拡張して train 期間バックテスト + feature importance 抽出 + 過学習検出を統合する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Expanding window 方式 (train_years=4, test_years=1)
- D-02: Fold 構成: 2020-2023 -> 2024 テスト, 2021-2024 -> 2025 テスト
- D-03: 2 フォールド(2024, 2025 テスト)で実行
- D-04: WalkForwardCV 拡張 + WFValidationResult データクラス
- D-05: train 期間もバックテスト実行で train ROI 取得
- D-06: 新規スクリプト `scripts/run_wf_validation.py`
- D-07: 複合判定(ROI gap 閾値 + 両年度 ROI 一貫性 + feature importance 安定性)
- D-08: ROI gap 閾値: 20% WARNING, 30% FAIL
- D-09: Feature importance 順位相関(Spearman) rho < 0.5 で WARNING
- D-10: プール ROI(総払戻額/総投資額)を主要指標
- D-11: ベット数加重 ROI を参考指標
- D-12: JSON 形式 + MLflow 記録
- D-13: 3 基準自動 PASS/FAIL 判定
- D-14: 保存先 data/backtest/wf_validation_result.json

### Claude's Discretion
- ROI gap 閾値の初期値と調整ロジックの詳細
- WFValidationResult データクラスのフィールド設計
- Feature importance 安定性評価の具体的な計算方法
- MLflow 記録のメトリクス名とパラメータ
- PASS/FAIL 判定のロジック詳細(WARNING/FAIL の区分)
- JSON レポートのスキーマ設計

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VALI-01 | Walk-forward 交差検証で過学習を検出・防止する | WalkForwardCV 拡張パターン、generate_folds 修正、過学習検出基準(D-07/08/09)の実装方法 |
| VALI-02 | 複数年度(2024-2025)のバックテストで ROI > 100% を確認する | BacktestEngine.run() の再利用、プール ROI 計算、JSON 出力パターン |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Walk-forward フォールド生成・実行 | API / Backend | -- | WalkForwardCV がフルパイプライン(train+backtest)を管理 |
| Train 期間バックテスト | API / Backend | -- | BacktestEngine.run() を train 期間に対しても実行 |
| Feature importance 抽出 | API / Backend | -- | LightGBM Booster.feature_importance() から取得 [VERIFIED: LightGBM 4.6.0] |
| 過学習検定(ROI gap / Spearman) | API / Backend | -- | scipy.stats.spearmanr で計算 [VERIFIED: scipy 1.17.1] |
| MLflow 記録 | API / Backend | -- | mlflow.log_metrics/log_params [VERIFIED: MLflow 3.10.1] |
| 結果 JSON 出力 | API / Backend | -- | json.dumps + Path.write_text パターン |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.17.1 | spearmanr (feature importance 順位相関) | D-09 で指定。scipy.stats.spearmanr が標準 [VERIFIED: runtime] |
| numpy | 2.4.3 | ROI 計算・配列操作 | プロジェクト標準 [VERIFIED: runtime] |
| lightgbm | 4.6.0 | feature_importance() + feature_name() | feature importance 抽出用 [VERIFIED: runtime] |
| mlflow | 3.10.1 | WF 検証結果の experiment tracking | D-12 で指定。既存パイプラインと同一 [VERIFIED: runtime] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.3.3 | DataFrame 操作 | 全データ処理 [VERIFIED: runtime] |
| dataclasses | stdlib | WFValidationResult 定義 | D-04 で指定 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.spearmanr | kendalltau | Spearman が D-09 でロック済み。変更不可 |
| 手動 JSON 出力 | pydantic BaseModel | データクラスの方がプロジェクトパターンに合致 |

**Installation:**
なし。全て既存依存関係で対応可能。

## Architecture Patterns

### System Architecture Diagram

```
run_wf_validation.py (CLI entry point)
        |
        v
WalkForwardCV (拡張版)
   |
   +-- generate_folds() [FIXED: 手動フォールド定義]
   |       |
   |       v
   |   Fold 0: train 2020-2023 -> test 2024
   |   Fold 1: train 2021-2024 -> test 2025
   |
   +-- 各フォールド:
       |
       +-- (1) TrainingPipelineV5.run(train_start, train_end)
       |       -> TrainedModelsV5
       |
       +-- (2) feature_importance 抽出
       |       -> models.submodels[surface].stage1.model.feature_importance()
       |       -> models.submodels[surface].win.hit_model.feature_importance()
       |       -> top-10 リスト保存
       |
       +-- (3a) BacktestEngine.run(test_start, test_end) -> test ROI
       +-- (3b) BacktestEngine.run(train_start, train_end) -> train ROI
       |
       +-- (4) FoldResult に train_roi, test_roi, gap, importance を集約
        |
        v
WFValidationResult (データクラス)
   |
   +-- fold_results: list[FoldResult]
   +-- pool_roi: 総払戻/総投資
   +-- weighted_roi: ベット数加重 ROI
   +-- overfitting_score: 複合判定
   +-- overall_verdict: PASS/WARNING/FAIL
        |
        v
   +-- JSON 保存: data/backtest/wf_validation_result.json
   +-- MLflow 記録: wf_validation experiment
```

### Recommended Project Structure
```
src/models/walk_forward_cv.py      -- 拡張: WFValidationResult, run_validation() 追加
src/domain/models.py               -- 拡張: WFValidationResult, FoldResult データクラス
scripts/run_wf_validation.py       -- 新規: CLI エントリポイント
```

### Pattern 1: WFValidationResult データクラス (Claude's Discretion)
**What:** 各フォールド結果と過学習判定を保持するデータクラス
**When to use:** Walk-forward 検証の結果保持

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class FoldResult:
    """単一フォールドの WF 検証結果"""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_roi: float = 0.0
    test_roi: float = 0.0
    roi_gap: float = 0.0  # train_roi - test_roi
    train_bets: int = 0
    test_bets: int = 0
    train_stake: float = 0.0
    test_stake: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    max_drawdown: float = 0.0
    top_features: list[str] = field(default_factory=list)  # top-10 特徴量
    feature_ranking: dict[str, int] = field(default_factory=dict)

@dataclass
class WFValidationResult:
    """Walk-forward 検証の全体結果"""
    folds: list[FoldResult] = field(default_factory=list)
    pool_roi: float = 0.0          # 総払戻/総投資 (D-10)
    weighted_roi: float = 0.0      # ベット数加重 ROI (D-11)
    total_stake: float = 0.0
    total_return: float = 0.0
    total_bets: int = 0
    roi_gap_verdict: str = "PASS"     # PASS/WARNING/FAIL
    consistency_verdict: str = "PASS" # 両年度 ROI 一貫性
    stability_verdict: str = "PASS"   # feature importance 安定性
    overall_verdict: str = "PASS"     # 全基準の AND
    spearman_rho: float = 0.0
    roi_gap_max: float = 0.0
    git_hash: str = ""
```

### Pattern 2: Feature Importance 安定性評価 (Claude's Discretion)
**What:** 2 フォールド間の top-10 特徴量の順位相関を計算
**When to use:** D-09 過学習検出の 3 番目の基準

```python
from scipy.stats import spearmanr

def compute_feature_stability(
    rankings: list[dict[str, int]],
    top_n: int = 10,
) -> float:
    """複数フォールド間の特徴量順位相関(平均)を計算

    Args:
        rankings: 各フォールドの {feature_name: rank} 辞書リスト
        top_n: 比較対象の上位特徴量数

    Returns:
        Spearman rho の平均値。0.5 未満で WARNING
    """
    if len(rankings) < 2:
        return float("nan")

    # 全フォールドで共通する top-N 特徴量を抽出
    all_features = set()
    for r in rankings:
        top = sorted(r, key=r.get)[:top_n]
        all_features.update(top)

    if len(all_features) < 3:
        return float("nan")

    # フォールドペアごとに Spearman rho を計算
    rhos = []
    for i in range(len(rankings) - 1):
        r1 = rankings[i]
        r2 = rankings[i + 1]
        common = [f for f in all_features if f in r1 and f in r2]
        if len(common) < 3:
            continue
        ranks1 = [r1[f] for f in common]
        ranks2 = [r2[f] for f in common]
        rho, _ = spearmanr(ranks1, ranks2)
        rhos.append(rho)

    return float(np.mean(rhos)) if rhos else float("nan")
```

### Pattern 3: 過学習 PASS/FAIL 判定 (Claude's Discretion)
**What:** ROI gap + 一貫性 + 安定性の 3 基準で判定
**When to use:** D-07, D-08, D-09, D-13 の実装

```python
def judge_overfitting(
    result: WFValidationResult,
    warning_gap: float = 0.20,  # D-08
    fail_gap: float = 0.30,     # D-08
    min_rho: float = 0.5,       # D-09
) -> None:
    """3 基準の自動判定を実行し、結果を result に反映

    基準 1: ROI gap (train - test の最大値)
      - < 20% -> PASS
      - 20-30% -> WARNING
      - > 30% -> FAIL

    基準 2: 両年度 ROI 一貫性
      - 両年度 test_roi > 100% -> PASS
      - 一方のみ > 100% -> WARNING
      - 両方 < 100% -> FAIL

    基準 3: Feature importance 安定性
      - rho >= 0.5 -> PASS
      - rho < 0.5 -> WARNING (FAIL にはしない)

    全 PASS -> overall PASS
    1 つでも FAIL -> overall FAIL
    WARNING のみ -> overall WARNING
    """
    # 基準 1: ROI gap
    gaps = [f.roi_gap for f in result.folds]
    max_gap = max(gaps) if gaps else 0.0
    result.roi_gap_max = max_gap
    if max_gap > fail_gap:
        result.roi_gap_verdict = "FAIL"
    elif max_gap > warning_gap:
        result.roi_gap_verdict = "WARNING"
    else:
        result.roi_gap_verdict = "PASS"

    # 基準 2: 一貫性
    test_rois = [f.test_roi for f in result.folds]
    above_100 = sum(1 for r in test_rois if r > 1.0)
    if above_100 == len(test_rois):
        result.consistency_verdict = "PASS"
    elif above_100 > 0:
        result.consistency_verdict = "WARNING"
    else:
        result.consistency_verdict = "FAIL"

    # 基準 3: 安定性
    if not np.isnan(result.spearman_rho):
        if result.spearman_rho >= min_rho:
            result.stability_verdict = "PASS"
        else:
            result.stability_verdict = "WARNING"

    # 総合判定
    verdicts = [result.roi_gap_verdict, result.consistency_verdict, result.stability_verdict]
    if "FAIL" in verdicts:
        result.overall_verdict = "FAIL"
    elif "WARNING" in verdicts:
        result.overall_verdict = "WARNING"
    else:
        result.overall_verdict = "PASS"
```

### Anti-Patterns to Avoid
- **WalkForwardCV.generate_folds() の無変更使用:** 現在の実装は D-02 のフォールド構成を生成できない(後述「Critical Finding」参照)。フォールドをハードコードまたは generate_folds() を修正すること
- **BacktestEngine の train 期間実行で初期化漏れ:** engine.run() はレース内で自己状態(regime stats 等)を蓄積するため、train/test で別インスタンスを使うこと
- **Feature importance を 1 サブモデルのみで評価:** 芝/ダート両方の stage1 + win.hit_model から重要性を取得し、サーフェス別に比較すること

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 順位相関計算 | 自作相関関数 | scipy.stats.spearmanr | 境界ケース(n=2, tied ranks)の処理が複雑 [VERIFIED: scipy 1.17.1] |
| Feature importance 抽出 | pandas で手動計算 | lgb.Booster.feature_importance(importance_type="gain") | LightGBM ネイティブ API。feature_name() との対応が保証される [VERIFIED: LightGBM 4.6.0] |
| MLflow トラッキング | 手動ファイル I/O | mlflow.log_metrics/params | 既存パイプラインと同一パターン [VERIFIED: MLflow 3.10.1] |

**Key insight:** feature importance の安定性評価には、単に gain 値の差ではなく、**順位** の相関を使う。これは D-09 で Spearman が指定されている理由でもある。gain の絶対値は train データ量に依存して変動するが、順位は相対的な重要性を表すため、フォールド間比較に適している。

## Common Pitfalls

### Pitfall 1: WalkForwardCV.generate_folds() が D-02 のフォールドを生成できない
**What goes wrong:** generate_folds() は `current_start = test_start + step_years` で進むため、Fold 0 (test=2024) の次は Fold 1 の開始が 2025-01-01 となり、train が 2025-2028、test が 2029 になる。2025 でキャップされるため Fold 1 は生成されない。
**Why it happens:** step ロジックが「テスト開始から step 年進む」だが、D-02 は「フォールド開始から step 年進む」(スライディングウィンドウ) を想定している。
**How to avoid:** run_wf_validation.py でフォールド定義を明示的に作成する。WalkForwardCV.generate_folds() は修正せず、新規メソッドかハードコードで対応する。
**Warning signs:** generate_folds() が 1 フォールドしか返さない場合。

### Pitfall 2: BacktestEngine の自己状態蓄積による train/test 汚染
**What goes wrong:** BacktestEngine.run() は recent_stats_list にレース統計を蓄積し、RegimeDetector の状態が変化する。train 期間のバックテスト後に test 期間を実行すると、RegimeDetector の初期状態が汚染される。
**Why it happens:** BacktestEngine がステートフル(RegimeDetector へのフィードバックループ)。
**How to avoid:** train と test で別々の BacktestEngine インスタンスを作成する。各フォールドで models オブジェクトは使い回せるが、engine は新規作成する。
**Warning signs:** test 期間の ROI が engine 再利用時と非再利用時で異なる。

### Pitfall 3: Feature importance の比較で特徴量セットが異なる
**What goes wrong:** フォールドごとに学習データが異なるため、LightGBM の feature_name() が異なる可能性がある(特にカテゴリ特徴量のレベル違い)。
**Why it happens:** FeatureEngine がデータ依存で特徴量を生成する場合、フォールド間で特徴量名が微妙に変わることがある。
**How to avoid:** feature_name() の共通集合を取得し、共通特徴量のみで順位相関を計算する。共通特徴量が 3 未満の場合は SKIP とする。
**Warning signs:** spearmanr 計算時の特徴量数が極端に少ない。

### Pitfall 4: train 期間バックテストの実行時間
**What goes wrong:** D-05 で train 期間もバックテストするため、実行時間が倍になる。各フォールド ~57 分 x 2(train+test) = ~114 分/フォールド。2 フォールドで ~4 時間。
**Why it happens:** BacktestEngine.run() は特徴量生成から予測・精算まで全処理を行う。
**How to avoid:** スクリプト開始時に推定時間を表示し、途中結果を逐次 JSON に書き出す(フォールド完了ごとにセーブ)。これにより途中クラッシュでも部分結果が残る。
**Warning signs:** ユーザーが途中で中断した場合に結果が全く残らない。

### Pitfall 5: MLflow experiment name の衝突
**What goes wrong:** 既存パイプライン(v5.5_{train_end})と同一 experiment に記録すると、結果が混在する。
**Why it happens:** training_pipeline.py の MLflow run_name パターンと重複。
**How to avoid:** MLflow experiment 名を "wf_validation" に設定し、run_name は "wf_fold_{fold_idx}" 形式にする。
**Warning signs:** MLflow UI で WF 検証結果と通常学習結果が混在している。

## Code Examples

### Example 1: フォールド定義のハードコード (D-02 対応)
```python
# scripts/run_wf_validation.py で使用
FOLDS = [
    {
        "train_start": "2020-01-01",
        "train_end":   "2023-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2024-12-31",
    },
    {
        "train_start": "2021-01-01",
        "train_end":   "2024-12-31",
        "test_start":  "2025-01-01",
        "test_end":    "2025-12-31",
    },
]
```

### Example 2: Feature importance 抽出パターン
```python
# Source: src/features/win_feature_analysis.py L39-42
import lightgbm as lgb

def extract_feature_ranking(model: lgb.Booster, top_n: int = 10) -> dict[str, int]:
    """LightGBM モデルから top-N 特徴量の順位を取得"""
    feature_names = model.feature_name()  # [VERIFIED: LightGBM 4.6.0]
    gain = model.feature_importance(importance_type="gain")
    # gain 降順で順位付け
    ranking = dict(zip(feature_names, range(len(feature_names))))
    sorted_features = sorted(zip(feature_names, gain), key=lambda x: -x[1])
    top_features = [f for f, _ in sorted_features[:top_n]]
    return {f: rank for rank, (f, _) in enumerate(sorted_features[:top_n])}, top_features
```

### Example 3: プール ROI 計算 (D-10)
```python
def compute_pool_roi(fold_results: list[FoldResult]) -> float:
    """D-10: プール ROI = 総払戻額 / 総投資額"""
    total_stake = sum(f.test_stake for f in fold_results)
    total_return = sum(f.test_return for f in fold_results)
    return total_return / total_stake if total_stake > 0 else 0.0
```

### Example 4: ベット数加重 ROI (D-11)
```python
def compute_weighted_roi(fold_results: list[FoldResult]) -> float:
    """D-11: ベット数加重 ROI"""
    total_bets = sum(f.test_bets for f in fold_results)
    if total_bets == 0:
        return 0.0
    return sum(f.test_roi * f.test_bets for f in fold_results) / total_bets
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| WalkForwardCV.generate_folds() スライディング | 固定フォールド手動定義 | Phase 4 | generate_folds() の step ロジックが D-02 に合わないため手動定義へ |

**Deprecated/outdated:**
- BacktestValidationSuite.run_walk_forward_cv() の 3-window パターン: 参考にはなるが、train ROI なし・feature importance なしのため、Phase 4 の要件を満たさない。パターンは流用するが、独立実装する。

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | BacktestEngine.run() は train 期間データに対しても正常に実行できる(空結果以外) | Architecture Patterns | train 期間バックテストが空結果になる場合、train ROI が取得できず過学習検出が不可能 |
| A2 | 各フォールドで TrainingPipelineV5.run() が独立して実行できる(前フォールドの状態に依存しない) | Architecture Patterns | フォールド間でパイプライン状態が漏れる場合、再現性が失われる |
| A3 | Feature importance は芝/ダートの stage1 モデル + win.hit_model で十分評価できる | Architecture Patterns | これら以外のモデルの不安定性を見逃す可能性 |

## Open Questions

1. **BacktestEngine で train 期間バックテストが意味のある結果を返すか?**
   - What we know: BacktestEngine.run() は test_start/test_end で load_races を呼ぶ。train 期間に相当する日付を渡せば実行可能。
   - What's unclear: train 期間の特徴量生成パスでレース数が多すぎてメモリ不足になる可能性。
   - Recommendation: run_wf_validation.py の実行で確認。必要なら年単位で分割実行。

2. **ROI gap の "20% WARNING, 30% FAIL" はパーセントポイントかパーセントか?**
   - What we know: D-08 は "20%ポイント" と "30%" の両方の表記あり。
   - What's unclear: train_roi=150%, test_roi=125% の場合、gap は 25 ポイント(25%差)か、(150-125)/125=20% か。
   - Recommendation: パーセントポイント(train_roi - test_roi)で実装。金融バックテスト標準。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全体 | Y | 3.11 (mise) | -- |
| scipy | Spearman rho | Y | 1.17.1 | -- |
| lightgbm | Feature importance | Y | 4.6.0 | -- |
| mlflow | 結果記録 | Y | 3.10.1 | -- |
| numpy | ROI 計算 | Y | 2.4.3 | -- |
| pandas | DataFrame 操作 | Y | 2.3.3 | -- |
| Parquet データ (2015-2025) | バックテスト | Y | -- | -- |

**Missing dependencies with no fallback:**
なし。全依存関係が利用可能。

**Missing dependencies with fallback:**
なし。

## Security Domain

> このフェーズはバリデーション/分析のみであり、外部入力の処理や認証・認可に関わる変更を含まないため、Security Domain の詳細は省略する。既存のパイプラインコードをそのまま再利用し、新たな攻撃面は増えない。

## Sources

### Primary (HIGH confidence)
- src/models/walk_forward_cv.py -- WalkForwardCV クラス、Fold/CVResult データクラス、generate_folds()/run() メソッド
- src/backtest/engine.py -- BacktestEngine.run()、BacktestResult データクラス
- src/backtest/validation_suite.py -- BacktestValidationSuite.run_walk_forward_cv() 参考実装
- scripts/run_backtest.py -- _run_multi_year() パターン
- src/features/win_feature_analysis.py -- analyze_feature_importance()、feature importance 抽出パターン
- src/pipelines/training_pipeline.py -- TrainingPipelineV5.run()、MLflow 記録パターン
- src/domain/models.py -- SubmodelSet、TrainedModelsV5

### Secondary (MEDIUM confidence)
- runtime version verification (LightGBM 4.6.0, scipy 1.17.1, MLflow 3.10.1, numpy 2.4.3, pandas 2.3.3)
- generate_folds() 実行テストで D-02 非互換を確認

### Tertiary (LOW confidence)
なし。全てコードベース検証済み。

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全てランタイムでバージョン確認済み
- Architecture: HIGH - 既存コードベースの構造を直接確認
- Pitfalls: HIGH - generate_folds() のバグを実行して確認済み

**Research date:** 2026-05-03
**Valid until:** 2026-06-03 (stable - コア依存関係の更新なし)
