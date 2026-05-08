# Phase 20: 高オッズ的中パターン特徴量 - Research

**Researched:** 2026-05-09
**Domain:** Feature engineering for high-odds hit prediction in horse racing ML
**Confidence:** HIGH

## Summary

Phase 20は高オッズ帯(オッズ20+)の的中率を2.1%から3%+に引き上げるための新特徴量10+を設計・実装する。既存の`HorseHistoryFeatures.compute()`のper-horseループ構造に新特徴量を統合し、`AbilityModel.FEATURE_COLS`に追加する。全特徴量は軽量なpandas/numpy計算のみで構成され、学習・バックテスト時間を延ばさない。

**Primary recommendation:** 新特徴量モジュール`src/features/high_odds_features.py`を作成し、`compute_high_odds_features()`関数群を定義。`HorseHistoryFeatures.compute()`のper-horseループ内から呼び出す。`form_cycle_features.py`のパターン（純粋関数 + numpy配列入力）に従う。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** ハイブリッド分析（統計プロファイリング + SHAP）。Cohen's dで効果量順位付け + 既存LightGBMで高オッズ馬のみSHAP値計算
- **D-02:** 分析スクリプト(`scripts/`)と特徴量生成モジュール(`src/features/`)を分離
- **D-03:** 高オッズの定義は初期オッズ20+で開始。サンプル不足時は10+に拡張
- **D-04:** 分析結果は手動特徴量設計に反映。学習済みクラスタリングモデルは使用しない
- **D-05:** クラストラジェクトリは数値シーケンス分解。未勝利=0, 1勝=1, 2勝=2, OP=3, 重賞=4で数値化
- **D-06:** 直近5走を使用。既存`HorseHistoryFeatures`のn_past=5に一致
- **D-07:** V字回復パターン（降級→再昇級）をバイナリフラグ + 降級期間特徴量として追加
- **D-08:** EMAベース指数改善率。halflife=3。Phase 5で確立した標準値
- **D-09:** タイムベース(z-score改善率)と着順ベース(正規化着順改善率)の両方を計算
- **D-10:** 3変化（距離/サーフェス/馬場状態）の過去適性履歴を計算。騎手/調教師変更は含めない
- **D-11:** 各変化について3サブ特徴量（平均着順/勝率/経験回数）。3x3=9特徴量
- **D-12:** 新特徴量は`Stage1AbilityModel.FEATURE_COLS`に追加
- **D-13:** `FeatureEngine.build_all()`内で既存の`HorseHistoryFeatures`ループに統合。独立モジュールファイル(`src/features/high_odds_features.py`)を作成
- **D-14:** 欠損率10%以下を要件。LightGBMがNaN処理可能

### Claude's Discretion
- 新特徴量の具体的な命名規則（snake_case一貫性）
- `HorseHistoryFeatures.compute()`内での計算箇所（既存ループ内に統合するか独立関数にするか）
- 分析スクリプトの出力形式（JSON/Markdown/PNG等）
- クラスレベルの数値マッピングの詳細（grade_code/jyokencd1からの変換）
- サンプル不足時のフォールバック戦略（経験回数0の環境変化適性のデフォルト値）
- Feature importance分析の具体的な比較方法（ベースラインOOF AUC vs 新特徴量追加後）
- テストのfixtureデータとモック構成

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HODDS-01 | 高オッズ的中パターン分析モジュール（ハイブリッド分析: 統計プロファイリング + SHAP） | `win_feature_analysis.py`がTreeSHAP基盤（`pred_contrib=True`、外部shap不要）。統計プロファイリングはCohen's d計算の新規実装 |
| HODDS-02 | クラストラジェクトリ特徴量（数値シーケンス分解、直近5走） | `_class_level_from_values()`でgrade_code/jyokencd1→数値マッピング済み。`horse_arrs["gradecd"]`/`["jyokencd1"]`が過去走配列に含まれる |
| HODDS-03 | フォーム改善率特徴量（EMAベース指数改善率、タイム+着順） | EMA halflife=3の計算パターンは`horse_history_features.py` L714-723に確立済み。z-scoreは`harontimel5_zscore`配列が利用可能 |
| HODDS-04 | 環境変化適性特徴量（3変化x3サブ特徴量=9特徴量） | `_compute_distance_bin()`が距離bin定義済み。`track_condition_code`が馬場状態として利用可能。`surface`がサーフェスとして利用可能 |
| HODDS-05 | 新特徴量のFeatureEngine統合・モデルFEATURE_COLS更新 | `HorseHistoryFeatures.BASE_COLS`に新列名を追加。`AbilityModel.FEATURE_COLS`に新列名を追加。`_prepare_features()`がavailable_colsフィルタリングを持つため安全 |

</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 高オッズパターン分析 | scripts/ (offline) | -- | バッチ分析スクリプト。パイプライン実行時には不要 |
| クラストラジェクトリ特徴量計算 | features/horse_history_features.py (per-horse loop) | -- | 既存per-horseループ構造に統合。過去走配列に直接アクセス可能 |
| フォーム改善率特徴量計算 | features/high_odds_features.py (新規pure関数) | features/horse_history_features.py (呼び出し) | form_cycle_features.pyパターンに従い独立モジュール化 |
| 環境変化適性特徴量計算 | features/high_odds_features.py (新規pure関数) | features/horse_history_features.py (呼び出し) | 過去走配列から条件別統計を計算 |
| FEATURE_COLS更新 | models/stage1_ability_model.py | -- | 新特徴量名をFEATURE_COLSリストに追加 |
| Feature importance検証 | scripts/analyze_feature_importance.py | -- | 既存TreeSHAP基盤を活用して高オッズ馬限定分析 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.4.3 | 配列計算・EMA・シーケンス分解 | プロジェクト全体で使用 [VERIFIED: pip show] |
| pandas | 2.3.3 | DataFrame操作・グループ集計 | プロジェクト全体で使用 [VERIFIED: pip show] |
| LightGBM | 4.6.0 | モデル学習・TreeSHAP | プロジェクト標準MLフレームワーク [VERIFIED: pip show] |
| scikit-learn | 1.8.0 | AUC/log_loss評価 | モデル評価指標 [VERIFIED: pip show] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | (dev) | テスト | 全テストDB不要・mock使用 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手動Cohen's d | scipy.stats | scipy依存増だが、Cohen's dは3行で実装可能なのでnumpy直接計算が良い [ASSUMED] |
| 外部shapパッケージ | LightGBM pred_contrib | 外部パッケージ不要。4.6.0のpred_contrib=TrueでTreeSHAP完全対応 [VERIFIED: win_feature_analysis.py] |

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │  scripts/analyze_high_odds.py    │
                    │  (HODDS-01: ハイブリッド分析)     │
                    │  統計プロファイリング + SHAP分析   │
                    └──────────────┬──────────────────┘
                                   │ 分析結果 (JSON/Markdown)
                                   ▼
                    ┌─────────────────────────────────┐
                    │  src/features/high_odds_features.py │
                    │  (新規: 純粋関数群)               │
                    │  - compute_class_trajectory()     │
                    │  - compute_form_improvement()     │
                    │  - compute_env_adaptability()     │
                    └──────────────┬──────────────────┘
                                   │ 呼び出し
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  src/features/horse_history_features.py                         │
  │  HorseHistoryFeatures.compute() per-horse loop                  │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │ for each horse:                                          │   │
  │  │   ...existing features...                                │   │
  │  │   class_traj = compute_class_trajectory(arrs, ...)  [NEW] │   │
  │  │   form_imp = compute_form_improvement(arrs, ...)    [NEW] │   │
  │  │   env_adapt = compute_env_adaptability(arrs, ...)   [NEW] │   │
  │  └──────────────────────────────────────────────────────────┘   │
  │  BASE_COLS += [新特徴量名...]                                   │
  └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  src/models/stage1_ability_model.py                             │
  │  AbilityModel.FEATURE_COLS += [新特徴量名...]                    │
  │  _prepare_features() → available_cols フィルタで安全に追加      │
  └──────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
src/features/
├── high_odds_features.py    # [NEW] 高オッズ特徴量の純粋関数群
├── horse_history_features.py # [MODIFY] compute()ループに新特徴量統合
├── form_cycle_features.py   # [REFERENCE] API設計テンプレート
└── win_feature_analysis.py  # [REFERENCE] TreeSHAP分析パターン

scripts/
├── analyze_high_odds.py     # [NEW] HODDS-01ハイブリッド分析スクリプト
└── analyze_feature_importance.py  # [REFERENCE] 既存分析スクリプト

src/models/
├── stage1_ability_model.py  # [MODIFY] FEATURE_COLSに新特徴量追加
└── two_stage_return_model.py # [MODIFY] 必要に応じてHIT_FEATURE_COLS/RETURN_FEATURE_COLSに追加

tests/
├── test_high_odds_features.py   # [NEW] 新特徴量の単体テスト
├── test_horse_history_features.py # [MODIFY] 新特徴量がBASE_COLSに含まれることを検証
└── test_form_cycle_features.py  # [REFERENCE] テストパターン参考
```

### Pattern 1: 純粋関数特徴量モジュール (form_cycle_features.pyパターン)
**What:** numpy配列入力 → タプル出力の純粋関数
**When to use:** 過去走配列からスカラー特徴量を計算する場合
**Example:**
```python
# Source: src/features/form_cycle_features.py (既存パターン)
FEATURE_COLS: list[str] = ["feat_a", "feat_b"]

def compute_something(
    gradecd: np.ndarray, jyokencd1: np.ndarray, ...
) -> tuple[float, float]:
    """..."""
    if n < 2:
        return float("nan"), float("nan")
    # numpy計算
    return result_a, result_b
```

### Pattern 2: per-horseループ内統合 (horse_history_features.pyパターン)
**What:** `HorseHistoryFeatures.compute()`のitertuplesループ内で純粋関数を呼び出す
**When to use:** 既存過去走配列(`horse_arrs`)のデータを利用する特徴量
**Example:**
```python
# horse_history_features.py compute()内
# 既存パターン: form_trend等の呼び出し (L986-993)
if n_past >= 2:
    _fc_kj = horse_arrs["kakuteijyuni"][valid_mask][start:idx].astype(float)
    _fc_ss = horse_arrs["syussotosu"][valid_mask][start:idx].astype(float)
    form_trend, form_consistency, form_peak_flag = compute_form_features(_fc_kj, _fc_ss)
else:
    form_trend = float("nan")
    # ...
```

### Anti-Patterns to Avoid
- **学習済みモデルの使用:** D-04で明示的に禁止。クラスタリング等の学習済みモデルは実行時オーバーヘッドを生み、確定性を損なう
- **カテゴリ変数のパターン分類:** D-05で数値シーケンス分解を採用。LightGBMが非線形組み合わせを自動学習するため、カテゴリ変数は不要
- **form_cycle_features.pyの既存特徴量との重複:** `form_trend`(線形回帰傾き)と新しいフォーム改善率(EMA)は意図的に直交させる。両方`_compute_haron_stats`のデータを使うが、計算方法が異なる
- **Categorical castの新特徴量への適用:** 新特徴量は全て連続値（float）。`.astype("category")`は不要

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cohen's d効果量 | scipy.stats関数 | numpy直接: `d = (m1-m2) / pooled_std` | 3行のコード。scipy依存不要 [ASSUMED] |
| EMA計算 | pandas EWM | numpy直接: `weights = (1-decay)**np.arange(n)` | horse_history_features.py L714-723に確立済みパターン |
| クラスレベル数値化 | 新規マッピング | `_class_level_from_values(grade_code, jyoken_code)` | 既存関数。grade_code→A=8,B=7,...E=4。jyokencd1→数値フォールバック [VERIFIED: horse_history_features.py L59-63] |
| 距離bin計算 | 新規関数 | `_compute_distance_bin(kyori, surface)` | horse_history_features.py L73-95に既存 [VERIFIED] |
| SHAP値計算 | 外部shapパッケージ | `model.predict(features_df, pred_contrib=True)` | LightGBM 4.6.0ネイティブTreeSHAP [VERIFIED: win_feature_analysis.py L46-53] |

**Key insight:** プロジェクトには約80個の既存特徴量計算パターンが確立されている。新特徴量は既存パターンを踏襲し、独自の「賢い」実装を避ける。

## Common Pitfalls

### Pitfall 1: 高オッズサンプル不足による統計的有意性の欠如
**What goes wrong:** オッズ20+の的中は全サンプルの~0.3%。効果量Cohen's dの信頼区間が広くなり、偽陽性リスクが高い。
**Why it happens:** 高オッズ的中はレアイベント（~1000件/年）。特徴量選択がノイズに支配される。
**How to avoid:** (1) 統計プロファイリングで効果量の信頼区間も報告。(2) SHAP分析と統計分析の両方で上位に来る特徴量のみ採用。(3) サンプル不足時はオッズ10+に拡張（D-03）。
**Warning signs:** Cohen's d > 0.8だがサンプル数 < 50の特徴量は過学習リスク。

### Pitfall 2: ルックアヘッドバイアスの混入
**What goes wrong:** 新特徴量に未来の情報（当レース以降のデータ）が混入する。
**Why it happens:** `expanding_stats`のような累積統計は`searchsorted(target_date, side="left")`で正しくカットする必要がある。
**How to avoid:** (1) `compute()`内の`valid_mask` + `searchsorted(target_date_np, side="left")`パターンを厳守。(2) 新特徴量のテストで未来データ除外を明示的に検証。
**Warning signs:** テストで未来レースが含まれる特徴量値がNaNでない場合。

### Pitfall 3: HorseHistoryFeatures.compute()の肥大化によるパフォーマンス低下
**What goes wrong:** 新特徴量計算がper-horseループ内に追加され、バックテスト時間が延びる。
**Why it happens:** Phase 19.1で~41分/年のバックテスト時間を最適化したばかり。新特徴量でこれを戻してはならない。
**How to avoid:** (1) 全新特徴量をO(1)〜O(n_past)の軽量numpy計算に制限。(2) ループ内でのDataFrame作成や重いソート操作を避ける。(3) horse_arrs dict-of-ndarrayパターンを継続使用。
**Warning signs:** 新特徴量追加後のcompute()が100ms/馬を超える場合。

### Pitfall 4: 環境変化適性のサンプル不足
**What goes wrong:** 特定の距離bin/馬場状態の組み合わせで過去出走が0件の場合、NaNばかりになる。
**Why it happens:** 競走馬は特定条件に偏って出走する。例: ダート馬の芝経験なし。
**How to avoid:** (1) D-11で距離bin単位(sprint/mile/intermediate/long)・馬場広カテゴリ(良/稍重/重/不良)で集計。(2) 経験回数0の場合はNaN（LightGBMが処理）。(3) フォールバックとして「全条件の平均」は使わない（D-10で3サブ特徴量に分解）。
**Warning signs:** 環境変化適性特徴量の欠損率が30%を超える場合。

### Pitfall 5: FEATURE_COLSとBASE_COLSの不整合
**What goes wrong:** HorseHistoryFeatures.BASE_COLSに追加したがAbilityModel.FEATURE_COLSに追加忘れ。またはその逆。
**Why it happens:** 3箇所（BASE_COLS, FEATURE_COLS, results dict）の同期が必要。
**How to avoid:** (1) 新特徴量名を定数リストとして一箇所定義。(2) テストで`BASE_COLS`と`FEATURE_COLS`の整合性を検証。
**Warning signs:** `_prepare_features()`が新特徴量をavailable_colsから除外する（FEATURE_COLS未追加）。

## Code Examples

### Cohen's d効果量計算 (HODDS-01 統計プロファイリング用)
```python
# [ASSUMED] - 標準的なCohen's d計算
import numpy as np

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """2群間の効果量Cohen's dを計算。"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)
```

### クラストラジェクトリ特徴量 (HODDS-02)
```python
# [VERIFIED: _class_level_from_values() horse_history_features.py L59-63]
# 過去走のgradecd/jyokencd1配列からクラストラジェクトリを計算

def compute_class_trajectory(
    gradecd_arr: np.ndarray,
    jyokencd1_arr: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """クラストラジェクトリ特徴量を計算。

    Returns:
        (class_promotions, class_demotions, class_net_change,
         class_max_level, class_level_std, v_recovery_flag, v_recovery_duration)
    """
    # _class_level_from_values()で各走のクラスレベルを数値化
    # → 差分で昇級/降級回数、最大レベル、分散等を計算
    # → V字回復: 降級→再昇級のバイナリフラグ + 降級期間
```

### フォーム改善率特徴量 (HODDS-03)
```python
# [VERIFIED: EMAパターン horse_history_features.py L714-723]
# halflife=3のEMAで直近改善を強調

def compute_form_improvement_rate(
    zscore_arr: np.ndarray,   # harontimel5_zscoreの過去走配列
    kakuteijyuni_arr: np.ndarray,
    syussotosu_arr: np.ndarray,
    halflife: int = 3,
) -> tuple[float, float]:
    """EMAベース指数改善率を計算。

    Returns:
        (time_improvement_rate, position_improvement_rate)
    """
    # タイム: z-score配列のEMA改善率
    # 着順: 正規化着順(pos-1)/(size-1)のEMA改善率
```

### 環境変化適性特徴量 (HODDS-04)
```python
# [VERIFIED: _compute_distance_bin() horse_history_features.py L73-95]
# 3変化(距離/サーフェス/馬場) x 3サブ特徴量(平均着順/勝率/経験回数)

def compute_env_adaptability(
    kakuteijyuni_arr: np.ndarray,   # 過去走着順
    syussotosu_arr: np.ndarray,     # 過去走頭数
    distance_bin_arr: np.ndarray,   # 過去走距離bin
    surface_arr: np.ndarray,        # 過去走サーフェス
    track_condition_arr: np.ndarray,# 過去走馬場状態
    current_distance_bin: str,      # 現在レースの距離bin
    current_surface: str,           # 現在レースのサーフェス
    current_track_condition: float, # 現在レースの馬場状態
) -> dict[str, float]:
    """環境変化適性9特徴量を計算。

    Returns dict with keys:
        dist_change_avg_pos, dist_change_win_rate, dist_change_exp_count,
        surf_change_avg_pos, surf_change_win_rate, surf_change_exp_count,
        cond_change_avg_pos, cond_change_win_rate, cond_change_exp_count
    """
    # 距離変更: current_distance_bin != last_distance_binの場合、
    #   過去走のうちdistance_bin == current_distance_binの走の統計
    # サーフェス/馬場状態も同様
```

### HorseHistoryFeatures.compute()への統合パターン
```python
# horse_history_features.py compute()内、results.append()の直前

# HODDS-02: クラストラジェクトリ (D-05, D-06, D-07)
if n_past >= 2 and "gradecd" in horse_arrs and "jyokencd1" in horse_arrs:
    _ct_grade = horse_arrs["gradecd"][valid_mask][start:idx]
    _ct_jyoken = horse_arrs["jyokencd1"][valid_mask][start:idx]
    (class_promotions, class_demotions, class_net_change,
     class_max_level, class_level_std,
     v_recovery_flag, v_recovery_duration) = compute_class_trajectory(
        _ct_grade, _ct_jyoken)
else:
    class_promotions = float("nan")
    # ... (全てNaN)

# results.append()の辞書に新キーを追加
results.append({
    # ...existing features...
    "class_promotions": class_promotions,
    "class_demotions": class_demotions,
    # ...etc...
})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 外部shapパッケージ | LightGBM pred_contrib=True | LightGBM 4.x | shapインストール不要。win_feature_analysis.pyで確立済み |
| Categorical特徴量エンコーディング | 数値スカラー分解 + LightGBM非線形学習 | Phase 5 (v1.1) | ワンホット不要。Phase 5のペース3サブ特徴量分解が確立パターン |
| per-horse DataFrame操作 | dict-of-ndarray (numpy配列) | Phase 19.1 (v1.5) | キャッシュ効率向上。~100s削減 |

**Deprecated/outdated:**
- `harontimel3_avg`/`harontimel3_zscore`: `harontimel5_avg`/`harontimel5_zscore`にリネーム済み

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Cohen's dはnumpy直接計算で十分（scipy不要） | Code Examples | LOW: scipy.stats利用に変更可能 |
| A2 | grade_codeの"A"-"E"マッピングが高オッズ分析でも正確 | HODDS-02 | LOW: _CLASS_LEVEL_MAPはプロジェクト全体で確立済み |
| A3 | 環境変化適性の距離bin集計でサンプル不足を回避できる | HODDS-04 | MEDIUM: 特定条件で依然としてサンプル不足の可能性 |
| A4 | 新特徴量10+が全て欠損率10%以下を満たす | Success Criteria | MEDIUM: 初戦馬等でNaN増の可能性。LightGBMはNaN処理可能だが基準を満たすかは検証必要 |
| A5 | 過去走配列にtrack_condition_codeが含まれる（HODDS-04で必要） | HODDS-04 | LOW: horse_history_features.py L491に`"track_condition_code"`がcols_horseに含まれることを確認済み |

**Note:** A5は実際にコード検証済み。`cols_horse`リスト(L477-496)に`track_condition_code`が含まれているため、`horse_arrs["track_condition_code"]`としてアクセス可能 [VERIFIED]。

## Open Questions (RESOLVED)

1. **高オッズ的中サンプルの実際の件数**
   - What we know: 全出走の~0.3%がオッズ20+の的中 [CITED: CONTEXT.md specifics]
   - What's unclear: 年間件数、期間（2020-2024）での総件数
   - Recommendation: 分析スクリプトの最初でサンプル数を報告。D-03に従い必要に応じて10+に拡張

2. **AbilityModel.FEATURE_COLSに追加するだけで十分か**
   - What we know: D-12でAbilityModel追加が決定。HIT_FEATURE_COLS/RETURN_FEATURE_COLSは「必要に応じて追加検討」[CITED: CONTEXT.md canonical_refs]
   - What's unclear: WinTwoStageModelのHIT/RETURN FEATURE_COLSにも追加すべきか
   - Recommendation: HODDS-05の範囲はAbilityModel.FEATURE_COLSが主。WinTwoStageModelは別途Phase 22で評価

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全パイプライン | ✓ | 3.11.15 | -- |
| LightGBM | TreeSHAP, モデル学習 | ✓ | 4.6.0 | -- |
| numpy | 特徴量計算 | ✓ | 2.4.3 | -- |
| pandas | DataFrame操作 | ✓ | 2.3.3 | -- |
| scikit-learn | AUC/log_loss評価 | ✓ | 1.8.0 | -- |
| PostgreSQL | HODDS-01分析スクリプト | ✗ | -- | 分析スクリプトはParquetStore経由でデータアクセス可能。DB不要 |
| shap (外部) | -- | ✗ | -- | 不要。LightGBM pred_contrib=TrueでTreeSHAP対応済み |

**Missing dependencies with no fallback:** なし

**Missing dependencies with fallback:** なし

## Sources

### Primary (HIGH confidence)
- `src/features/horse_history_features.py` — per-horseループ構造、BASE_COLS定義、horse_arrs dict-of-ndarrayパターン [VERIFIED: ソースコード直接読取]
- `src/features/form_cycle_features.py` — 純粋関数特徴量モジュールのAPIテンプレート [VERIFIED: ソースコード直接読取]
- `src/features/feature_engine.py` — FeatureEngine.build_all()統合ポイント [VERIFIED: ソースコード直接読取]
- `src/models/stage1_ability_model.py` — FEATURE_COLS定義、_prepare_features()のavailable_colsフィルタ [VERIFIED: ソースコード直接読取]
- `src/features/win_feature_analysis.py` — TreeSHAP (pred_contrib=True)パターン [VERIFIED: ソースコード直接読取]

### Secondary (MEDIUM confidence)
- `src/features/interaction_features.py` — distance_change/surface_change既存実装の参照 [VERIFIED: ソースコード直接読取]
- `src/features/horse_career_stats.py` — 環境変化適性の計算参照 [VERIFIED: ソースコード直接読取]
- `src/db/readers.py` — 履歴データ読み込みパターン [VERIFIED: ソースコード直接読取]

### Tertiary (LOW confidence)
- なし — 全ての主張はソースコード検証済み

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全てpip showでバージョン確認済み
- Architecture: HIGH - 既存コードパターンに基づく設計。確立されたパターンの踏襲
- Pitfalls: HIGH - プロジェクト固有の落とし穴を既存コードから特定

**Research date:** 2026-05-09
**Valid until:** 2026-06-09 (stable codebase)
