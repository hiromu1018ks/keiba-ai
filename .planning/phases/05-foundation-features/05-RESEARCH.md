# Phase 5: Foundation Features - Research

**Researched:** 2026-05-03
**Domain:** 特徴量エンジニアリング (時系列特徴量・展開予測・オッズ変動)
**Confidence:** HIGH

## Summary

Phase 5 は既存の3つの特徴量モジュール (`horse_history_features.py`, `pace_aptitude_features.py`, `odds_dynamics_features.py`) に新特徴量を追加する。全モジュールがすでに `compute_batch()` または `compute()` パターンでベクトル化実装されており、新特徴量の追加はこれらの既存枠組み内で完結する。新規モジュール作成は不要 (D-22)。

**Primary recommendation:** 既存の searchsorted + numpy集計パターンに従い、各モジュールの計算ループ内に新特徴量を追加する。EMA実装には手動の指数減衰重み配列を使用し (pandas ewm は row-per-horse ループ内で非効率)、class_level は `_CLASS_LEVEL_MAP` / `_class_level_from_values()` をそのまま再利用する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `harontimel5_avg` 列を指数減衰版に**置き換え**（新列追加しない）
- **D-02:** halflife=3走、λ=ln(2)/3≈0.231
- **D-03:** ルックバックウィンドウを5走から全過去走に拡張
- **D-04:** 新指標 `class_adj_formetric` として新規追加
- **D-05:** 計算式: `Σ(norm_finish_i × class_level_i) / Σ(class_level_i)`
- **D-06:** `_compute_class_level()` または `_CLASS_LEVEL_MAP` を再利用
- **D-07:** 線形回帰の傾きとして `haron_zscore_trend` 新規追加
- **D-08:** track_condition正規化は既存 `harontimel5_zscore` と同じ方式
- **D-09:** 最低3走以上の有効z-scoreが必要、不足時はNaN
- **D-10:** 複数サブ特徴量として出力（単一スコアに圧縮しない）
- **D-11:** 出力: `pace_corner_stability`, `pace_closing_power`, `pace_position_consistency`
- **D-12:** 既存 `PaceAptitudeFeatures.compute_batch()` パターン内に追加
- **D-13:** 新列 `actual_pace_fit` を追加、既存 `pace_scenario_fit` は残す
- **D-14:** actual_pace_fit は実績ベースの front_pace_wr/closing_pace_wr を使用
- **D-15:** `interaction_features.py` の pace_scenario_fit 計算部に actual_pace_fit 生成を追加
- **D-16:** 3点差分型を採用
- **D-17:** `odds_acceleration = velocity_late - velocity_early`
- **D-18:** スナップショット不足(<3点)時はNaN
- **D-19:** 時間加重型を採用、指数減衰で重み付け
- **D-20:** `odds_direction_consistency` (0〜1)
- **D-21:** 最小スナップショット数=5点
- **D-22:** 全新特徴量を既存モジュールに追加、新規モジュールは作成しない
- **D-23:** NaN処理はデフォルトNaN、LightGBMネイティブNaN処理を活用、0埋めしない
- **D-24:** `FeatureEngine.build_all()` の既存ステップ内で各モジュールを呼び出し、新ステップ追加は不要

### Claude's Discretion
- EMA実装の詳細（numpy向量化、compute_batch内での重み配列生成）
- class_adj_formetricのclass_level取得方法（history entries/racesマージ時のgrade_code/jyoken_code可用性）
- pace_closing_powerの上がりタイムソース（entries_histのagi列 or harontimel3近似）
- odds_direction_consistencyの減衰率（halflife = スナップショット数/4程度）
- 各特徴量のNaN率が50%超の場合のフォールバック戦略

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TSER-01 | 過去走の全平均値特徴量を指数減衰重み付けに置き換え | `horse_history_features.py` lines 673-683 の `harontimel5_avg` 計算をEMA化。halflife=3, λ≈0.231 の重み配列をnumpyで生成 |
| TSER-02 | クラス調整済みフォーメトリック算出 | `_CLASS_LEVEL_MAP` + `_class_level_from_values()` を再利用。`gradecd`/`jyokencd1` は `horse_arrs` 内に既に存在 (cols_horse lines 456-457) |
| TSER-03 | z-score改善トラジェクトリ特徴量追加 | 既存 `harontimel5_zscore` の expanding_stats パターンで取得したz-score配列に `np.polyfit(x, z, 1)[0]` を適用。`form_cycle_features.py` と同一アプローチ |
| PACE-01 | コーナー位置と上がりタイムから総合ペースフィグア算出 | `pace_aptitude_features.py` の compute_batch パターンに新サブ特徴量3つを追加。jyuni1c/jyuni4c は既に hist 内に存在。上がりタイムは agi 列が無いため harontimel3 で代用 |
| PACE-02 | 実績ベースのペース適性で既存pace_scenario_fitを強化 | `interaction_features.py` `_add_pace_projection_features()` に actual_pace_fit を追加。front_pace_wr/closing_pace_wr は pace_df マージ済みの値を使用 |
| ODTS-01 | オッズ変動の2次微分(加速度)特徴量追加 | `odds_dynamics_features.py` の `compute_odds_dynamics()` 内で、既存の `_mins_before_anchor` 値を用いて3点差分を計算 |
| ODTS-02 | オッズ変動方向の一貫性特徴量追加 | 同関数内で、各スナップショットの方向 (+1/-1/0) を計算し、時間減衰重み付きで比率を算出 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| TSER-01~03 (過去走時系列) | Feature Engine (ML Pipeline) | — | `horse_history_features.py` が過去走データから計算。DBからの読み込みは既存 reader が担当 |
| PACE-01 (ペースフィグア) | Feature Engine (ML Pipeline) | — | `pace_aptitude_features.py` がコーナー位置から計算。入力は `load_history_entries/races` |
| PACE-02 (actual_pace_fit) | Feature Engine (ML Pipeline) | — | `interaction_features.py` が pace_scenario_fit と同時に計算 |
| ODTS-01~02 (オッズ変動) | Feature Engine (ML Pipeline) | — | `odds_dynamics_features.py` がオッズ時系列から計算。入力は `load_odds_time_series_range` |
| モデル FEATURE_COLS 更新 | Model Layer | — | `stage1_ability_model.py` と `two_stage_return_model.py` の FEATURE_COLS に新列名を追加 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.4.3 | ベクトル化計算、EMA重み配列生成、np.polyfit | プロジェクト全体で使用中 [VERIFIED: runtime check] |
| pandas | 2.3.3 | DataFrame操作、マージ、groupby | プロジェクト全体で使用中 [VERIFIED: runtime check] |
| lightgbm | 4.6.0 | モデル学習、ネイティブNaN処理 | プロジェクト全体で使用中 [VERIFIED: runtime check] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy (なし) | — | 線形回帰 | 不要。np.polyfit で十分 (既存 form_cycle_features.py と同じ) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 手動EMA重み配列 | pandas ewm() | ewm は Series操作のため row-per-horse ループ内では非効率。numpy配列での手動重み付けが既存パターンに適合 |
| np.polyfit | scipy.stats.linregress | linregress は p-value も返すが不要。np.polyfit は form_cycle_features.py で実績あり |

**Installation:**
```bash
# 追加インストール不要 — 全て既存依存関係
```

## Architecture Patterns

### System Architecture Diagram

```
                         Parquet Files
                              |
                    +---------+---------+
                    |                   |
              history_entries    history_races
              odds_time_series
                    |                   |
        +-----------+-----------+       |
        |           |           |       |
   TSER-01~03   PACE-01     ODTS-01~02
   horse_history pace_apt    odds_dynamics
   _features.py  itude.py   _features.py
        |           |           |
        +-----------+-----------+
                    |
            FeatureEngine.build_all()
            / _train_submodel()
                    |
            +-------+-------+
            |               |
    AbilityModel      WinTwoStageModel
    FEATURE_COLS      FEATURE_COLS
    (Stage1)          (Stage2)
```

### Recommended Project Structure
```
src/features/
├── horse_history_features.py  # TSER-01~03 追加 (既存)
├── pace_aptitude_features.py  # PACE-01 追加 (既存)
├── odds_dynamics_features.py  # ODTS-01~02 追加 (既存)
├── interaction_features.py    # PACE-02 追加 (既存)
├── feature_engine.py          # 変更不要 (D-24)
└── form_cycle_features.py     # 参照のみ (np.polyfitパターン)

src/models/
├── stage1_ability_model.py    # FEATURE_COLS に新列追加
└── two_stage_return_model.py  # FEATURE_COLS に新列追加
```

### Pattern 1: EMA (指数減衰重み付け)
**What:** 過去走データに時間減衰重みを付与し、直近の成績を強調
**When to use:** TSER-01 (harontimel5_avg の置き換え)
**Example:**
```python
# Source: 新規実装 — 既存 compute() ループ内パターンに基づく
# halflife=3, λ = ln(2)/3 ≈ 0.231
# n個の過去走に対する重み: w[i] = (1-λ)^i (i=0が最新)
halflife = 3
decay = np.log(2) / halflife  # ≈ 0.231
n = len(ht_valid)  # 全過去走
weights = np.array([(1 - decay) ** i for i in range(n)])
weights = weights[::-1]  # 古い→新しい順にソートされたデータに対応
weights = weights / weights.sum()  # 正規化
ema_avg = float(np.nansum(ht_valid * weights))
```

### Pattern 2: class_adj_formetric (クラス調整フォーメトリック)
**What:** 高クラスでの好走を高く評価する重み付き着順指標
**When to use:** TSER-02
**Example:**
```python
# Source: D-05 + _class_level_from_values() 再利用
# 各過去走の class_level を取得
# gradecd, jyokencd1 は horse_arrs 内に既に存在
grade = horse_arrs["gradecd"][valid_mask][start:idx]
jyoken = horse_arrs["jyokencd1"][valid_mask][start:idx]
class_levels = np.array([
    _class_level_from_values(g, j) for g, j in zip(grade, jyoken)
])
# norm_finish = (kakuteijyuni - 1) / (syussotosu - 1)
norm_finish = (hp_kakuteijyuni.astype(float) - 1) / np.maximum(hp_syussotosu.astype(float) - 1, 1)
valid = ~np.isnan(class_levels) & ~np.isnan(norm_finish)
if valid.any():
    cl = class_levels[valid]
    nf = norm_finish[valid]
    class_adj_formetric = float(np.sum(nf * cl) / np.sum(cl))
```

### Pattern 3: z-score改善トラジェクトリ
**What:** 過去走のz-scoreに対する線形回帰の傾き
**When to use:** TSER-03
**Example:**
```python
# Source: form_cycle_features.py と同じ np.polyfit パターン
# zscores は既存の expanding_stats パターンで計算済み
z_arr = np.array(zscores)  # 既存コード lines 710-728 で計算
if len(z_arr[~np.isnan(z_arr)]) >= 3:  # D-09: 最低3走
    valid_z = z_arr[~np.isnan(z_arr)]
    x = np.arange(len(valid_z), dtype=float)
    haron_zscore_trend = float(np.polyfit(x, valid_z, 1)[0])
else:
    haron_zscore_trend = float("nan")
```

### Pattern 4: オッズ加速度 (2次微分)
**What:** オッズ変動の2次微分でsteam moveを検出
**When to use:** ODTS-01
**Example:**
```python
# Source: D-16~17 — compute_odds_dynamics() 内
# 既存の _mins_before_anchor 値を利用して3点差分
# t-60→t-30 区間: velocity_early = (odds_30 - odds_60) / 30
# t-30→t-10 区間: velocity_late = (odds_10 - odds_30) / 20
# acceleration = velocity_late - velocity_early
# 正 = オッズ低下が加速 (steam move)
if pd.notna(odds_60) and pd.notna(odds_30) and pd.notna(odds_10):
    vel_early = (odds_30 - odds_60) / 30.0
    vel_late = (odds_10 - odds_30) / 20.0
    odds_acceleration = vel_late - vel_early
else:
    odds_acceleration = np.nan
```

### Pattern 5: オッズ方向一貫性
**What:** オッズ変動方向の時間加重一貫性
**When to use:** ODTS-02
**Example:**
```python
# Source: D-19~21 — compute_odds_dynamics() 内
# 各スナップショットの方向: +1 (上昇), -1 (低下), 0 (不変)
# 時間減衰重み: 直近ほど高く評価
# consistency = |Σ(w_i * dir_i)| / Σ(w_i)  (0〜1)
# halflife = n_snapshots / 4 程度
if n_snapshots >= 5:
    decay_rate = np.log(2) / (n_snapshots / 4)
    weights = np.array([(1 - decay_rate) ** i for i in range(n_snapshots)])
    weights = weights / weights.sum()
    directions = np.sign(odds_diffs)  # +1, -1, 0
    weighted_sum = np.sum(weights * directions)
    weight_total = np.sum(weights)
    odds_direction_consistency = float(abs(weighted_sum) / weight_total)
else:
    odds_direction_consistency = np.nan
```

### Anti-Patterns to Avoid
- **pandas ewm() in row-per-horse loop:** 非効率。numpy配列での手動重み付けを使用
- **0埋め NaN処理:** D-23で明示的に禁止。LightGBMのネイティブNaN処理を活用
- **新規モジュール作成:** D-22で明示的に禁止。既存モジュールに追加
- **単一ペーススコア圧縮:** D-10で禁止。複数サブ特徴量でLightGBMに非線形組み合わせを学習させる
- **agi列の参照:** 上がりタイムデータ (agi/agari) はコードベースに存在しない。harontimel3で代用

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| クラスレベル計算 | 新規class_level関数 | `_class_level_from_values()` (horse_history_features.py line 56) + `_compute_class_level()` (feature_engine.py line 29) | 既にgrade_code/jyoken_codeマッピングが実装済み。`_CLASS_LEVEL_MAP` も同一値 |
| 線形回帰傾き | カスタム回帰実装 | `np.polyfit(x, y, 1)[0]` | form_cycle_features.py で実績あり。scipy不要 |
| z-score計算 | 新規z-scoreパイプライン | 既存 expanding_stats + `_lookup_expanding_stats()` | 階層的フォールバック付きのPIT-safe実装が既に存在 |
| コーナー位置データ | 新規データソース | `jyuni1c`/`jyuni4c` (entries_hist内) | pace_aptitude_features.py で既に使用中 |

**Key insight:** 全ての基盤データ（コーナー位置、ハロンタイム、オッズ時系列、クラスレベル）は既存の reader/Parquet 経由で利用可能。新たなデータソースは不要。

## Common Pitfalls

### Pitfall 1: EMA置き換え時の列名互換性
**What goes wrong:** `harontimel5_avg` をEMA化する際、モデルの FEATURE_COLS 内の列名と不一致が生じる可能性
**Why it happens:** D-01で「置き換え」と決定済みだが、既存テストが旧平均値を期待している
**How to avoid:** 列名は `harontimel5_avg` のまま変更しない。計算ロジックのみEMA化する
**Warning signs:** テストで数値が変わる → テストの期待値をEMA版に更新

### Pitfall 2: class_level 取得時の NaN 処理
**What goes wrong:** `gradecd`/`jyokencd1` が未格付けレースでNaNになり、class_adj_formetric の分母が0になる
**Why it happens:** 新馬戦や条件戦では grade_code が空、jyoken_code も不正値になり得る
**How to avoid:** `_class_level_from_values()` はNaNを返す設計。TSER-02計算時は `~np.isnan(class_levels)` でフィルタしてから加重平均
**Warning signs:** class_adj_formetric のNaN率が想定以上に高い場合

### Pitfall 3: ルックバック拡張による計算量増大
**What goes wrong:** D-03で5走→全過去走に拡張したことで、EMA重み配列の長さが馬ごとに大きく異なる
**Why it happens:** 長いキャリアの馬は50走以上の過去データを持つ。重み配列生成が O(n) で各馬に適用
**How to avoid:** numpy array生成は高速。but 100走以上の馬では正規化重みの精度に注意: `weights / weights.sum()` で浮動小数点精度を確認
**Warning signs:** ベクトル化ループの実行時間が大幅に増加

### Pitfall 4: PACE-01 上がりタイムの不在
**What goes wrong:** `pace_closing_power` の設計意図は「上がりタイムの相対的位置」だが、agi/agari 列がデータに存在しない
**Why it happens:** JRA-VAN DataLab の entries テーブルには上がりタイム列が含まれない
**How to avoid:** `harontimel3` (3ハロンタイム) を代用。ハロンタイムは最終3ハロンの通過タイムであり、上がり3ハロンの近似として使用可能 [ASSUMED — 実際の harontimel3 の定義確認が必要]
**Warning signs:** harontimel3 のNaN率が高いと pace_closing_power も高NaN率に

### Pitfall 5: オッズ時系列のスナップショット不足
**What goes wrong:** ODTS-02 は最小5点のスナップショットを要求するが、多くのレースでスナップショット数が不足
**Why it happens:** jodds_tanpuku のパーティションが年/月単位で、古いデータほどスナップショット密度が低い
**How to avoid:** NaN率を計測し、50%超の場合はフォールバック検討 (Claude's Discretion内の項目)
**Warning signs:** `odds_direction_consistency` のNaN率 > 50% → 条件緩和(最小3点)または特徴量除外

### Pitfall 6: FEATURE_COLS 更新漏れ
**What goes wrong:** 新特徴量をモジュールに追加したが、モデルの FEATURE_COLS に追加し忘れる
**Why it happens:** 複数モデル (AbilityModel, WinTwoStageModel, PlaceAbilityModel) が独立した FEATURE_COLS を持つ
**How to avoid:** 実装チェックリストに FEATURE_COLS 更新を含める。`AbilityModel.FEATURE_COLS` と `WinTwoStageModel.FEATURE_COLS` の両方を確認
**Warning signs:** LightGBM 学習時に "feature name mismatch" 警告

## Code Examples

### TSER-01: EMA重み付け実装の具体例
```python
# Source: 既存 horse_history_features.py lines 673-683 を拡張
# 現在の実装 (直近5走の単純平均):
#   harontimel5_avg = float(ht_valid[-self._n_past:].mean())
#
# 変更後 (全過去走のEMA):
#   D-03: ルックバックを全過去走に拡張
halflife = 3
decay = np.log(2) / halflife  # ≈ 0.231
n = len(ht_valid)
weights = (1 - decay) ** np.arange(n)  # w[0]=最新(1.0), w[n-1]=最古
weights = weights / weights.sum()
# ht_valid は 古い→新しい 順でソート済み
harontimel5_avg = float(np.sum(ht_valid * weights))
```

### TSER-03: z-score改善トラジェクトリ
```python
# Source: form_cycle_features.py line 49 の np.polyfit パターン
# 既存の expanding_stats で計算済み zscores リスト (lines 710-728)
z_arr = np.array(zscores)
valid_z = z_arr[~np.isnan(z_arr)]
if len(valid_z) >= 3:  # D-09
    x = np.arange(len(valid_z), dtype=float)
    haron_zscore_trend = float(np.polyfit(x, valid_z, 1)[0])
    # 負の値 = z-scoreが改善傾向 (タイムが速くなっている)
else:
    haron_zscore_trend = float("nan")
```

### PACE-01: ペースフィグアサブ特徴量
```python
# Source: pace_aptitude_features.py の cum_* パターンに追加
# hist 内に既に norm_1c, norm_finish, is_front, is_closing が存在

# pace_corner_stability: 1C→4Cの位置変位の安定性
# h_jyuni1c, h_jyuni4c は hist numpy 配列として取得済み
# jyuni1c/jyuni4c は pace_aptitude の hist マージで既に利用可能
corner_disp = h_jyuni4c - h_jyuni1c  # 各走の位置変位
corner_stability = np.std(corner_disp) if len(corner_disp) >= 2 else np.nan

# pace_closing_power: 上がりタイムの相対的位置
# harontimel3 を代用 (agi 列不在)
# Note: hist に harontimel3 を含めるには追加の列マージが必要

# pace_position_consistency: 正規化着順のばらつき
position_consistency = np.std(h_norm_finish) if len(h_norm_finish) >= 2 else np.nan
```

### PACE-02: actual_pace_fit
```python
# Source: interaction_features.py _add_pace_projection_features() に追加
# front_pace_wr, closing_pace_wr は pace_df マージで既に df に存在
# actual_pace_fit = front_pace_wr if front-runner, closing_pace_wr if closer
# style は kyakusitukubun_cd から判定 (1=逃げ, 2=先行, 3=差し, 4=追込)
if "front_pace_wr" in df.columns and "closing_pace_wr" in df.columns:
    is_front_runner = style.isin([1, 2])
    is_closer = style.isin([3, 4])
    df["actual_pace_fit"] = np.where(
        is_front_runner, df["front_pace_wr"],
        np.where(is_closer, df["closing_pace_wr"], np.nan)
    )
```

### ODTS-01~02: オッズ加速度・方向一貫性
```python
# Source: odds_dynamics_features.py compute_odds_dynamics() 内に追加
# 既存の odds_10, odds_30, odds_60 (Series) をそのまま使用

# ODTS-01: 加速度
vel_early = (odds_30 - odds_60) / 30.0
vel_late = (odds_10 - odds_30) / 20.0
odds_acceleration = vel_late - vel_early
# NaN条件: 3点のうちいずれかがNaN

# ODTS-02: 方向一貫性 (groupby ベクトル化)
# ts は既にソート済み (race_id, umaban, happyotime)
ts["_odds_dir"] = np.sign(ts.groupby(["race_id", "umaban"])["tanodds"].diff())
ts["_snap_order"] = ts.groupby(["race_id", "umaban"]).cumcount(ascending=False)
# 時間減衰重み
halflife_snaps = ts.groupby(["race_id", "umaban"])["_odds_dir"].transform("count") / 4
# 各グループ内で計算...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 直近5走の単純平均 | 指数減衰重み付け (EMA) | Phase 5 (今回) | 直近の成績をより重視、情報損失なしに全データ活用 |
| 宣言脚質のみのペース適性 | 実績ベースのペース適性 | Phase 5 (今回) | 宣言と実際の走法乖離を補完 |
| オッズ変動1次微分のみ | 1次+2次微分+方向一貫性 | Phase 5 (今回) | steam moveの強さとスマートマネーの流入を検出 |

**Deprecated/outdated:**
- `harontimel5_avg` の単純平均版 → EMA版に置き換え (D-01)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `harontimel3` は最終3ハロンタイムであり、上がりタイムの近似として使用可能 | PACE-01, Pitfall 4 | pace_closing_power の意味が変わる。実際はハロンタイムの平均なら代用可能だが、別の定義なら特徴量の解釈が変わる |
| A2 | `jyuni1c`/`jyuni4c` は `pace_aptitude_features.py` の `hist` で `entries_for_merge` 経由で利用可能 | PACE-01 | hist に含まれない場合、列マージの追加が必要 |
| A3 | `odds_ts_df` の `_mins_before_anchor` は compute_odds_dynamics() 内で既に計算済み | ODTS-01 | 再計算が必要になる場合、実装が複雑化 |
| A4 | `front_pace_wr`/`closing_pace_wr` は interaction_features.py 呼び出し時に df 内に存在 | PACE-02 | マージ順序に依存。training_pipeline.py の実行順序で pace_df マージが先に行われることを確認済み |

**If this table is empty:** All claims in this research were verified or cited.

## Open Questions

1. **harontimel3 の正確な定義**
   - What we know: entries テーブルに含まれる float 列。ETLで float 型変換される
   - What's unclear: これが「最終3ハロン通過タイム」なのか「ハロンタイム3（最後から3番目）」なのか。pace_closing_power の設計に影響
   - Recommendation: harontimel3 を上がりタイムの近似として使用。NaN率を計測し、50%超の場合は pace_closing_power を簡易版（4C位置のみ）にフォールバック

2. **odds_direction_consistency のNaN率**
   - What we know: 最小5点のスナップショットが必要 (D-21)
   - What's unclear: 実際の jodds_tanpuku データで5点以上のスナップショットがあるレースの割合
   - Recommendation: 実装後にNaN率を計測。50%超なら最小3点に緩和

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全コンポーネント | ✓ | 3.11 (mise) | — |
| numpy | EMA計算、polyfit | ✓ | 2.4.3 | — |
| pandas | DataFrame操作 | ✓ | 2.3.3 | — |
| lightgbm | モデル学習 | ✓ | 4.6.0 | — |
| PostgreSQL | ETL/データ読み込み | ✓ | localhost:5432 | — |
| Parquet files | 特徴量計算の入力 | ✓ | data/raw, data/odds | — |

**Missing dependencies with no fallback:**
- なし

**Missing dependencies with fallback:**
- なし

## Sources

### Primary (HIGH confidence)
- `src/features/horse_history_features.py` — lines 262-1148 compute_batch() 本体 [VERIFIED: codebase read]
- `src/features/pace_aptitude_features.py` — lines 68-250 compute_batch() [VERIFIED: codebase read]
- `src/features/odds_dynamics_features.py` — lines 112-243 compute_odds_dynamics() [VERIFIED: codebase read]
- `src/features/interaction_features.py` — lines 87-108 _add_pace_projection_features() [VERIFIED: codebase read]
- `src/features/form_cycle_features.py` — lines 22-64 np.polyfit パターン [VERIFIED: codebase read]
- `src/features/feature_engine.py` — lines 29-48 _compute_class_level(), lines 84-196 build_all() [VERIFIED: codebase read]
- `src/models/stage1_ability_model.py` — lines 28-97 FEATURE_COLS [VERIFIED: codebase read]
- `src/models/two_stage_return_model.py` — lines 47-88 FEATURE_COLS [VERIFIED: codebase read]
- `src/pipelines/training_pipeline.py` — lines 288-378 特徴量パイプライン順序 [VERIFIED: codebase read]

### Secondary (MEDIUM confidence)
- `src/db/readers.py` — load_history_entries/races のデータ可用性 [VERIFIED: codebase read]

### Tertiary (LOW confidence)
- なし

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 全て既存依存関係、バージョン確認済み
- Architecture: HIGH — コードベースの全参照箇所を実読確認
- Pitfalls: HIGH — 実装パターンの分析に基づく

**Research date:** 2026-05-03
**Valid until:** 2026-06-03 (安定ドメイン)
