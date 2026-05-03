# Phase 5: Foundation Features - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

過去走の時系列特徴量・展開予測特徴量・オッズ変動特徴量の3系統を追加し、後続のモデル改善(Phase 6-7)がより豊かな入力から恩恵を受けられるようにする。

**In scope (from ROADMAP.md):**
- TSER-01: 過去走の全平均値特徴量を指数減衰重み付けに置き換え
- TSER-02: クラス調整済みフォーメトリック算出
- TSER-03: z-score改善トラジェクトリ特徴量追加
- PACE-01: コーナー位置と上がりタイムから総合ペースフィグア算出
- PACE-02: 実績ベースのペース適性で既存pace_scenario_fitを強化
- ODTS-01: オッズ変動の2次微分(加速度)特徴量追加
- ODTS-02: オッズ変動方向の一貫性特徴量追加

**Out of scope:**
- LightGBM/XGBoost/CatBoostスタッキング (Phase 7)
- オッズ乖離EV活用 (Phase 6)
- ベッティング戦略の変更
- 複勝/ワイドモデルの変更
- LSTM/Transformer時系列モデリング (Out of Scope per REQUIREMENTS.md)

**Plans:** 2 plans
- 05-01: Time-series and pace features (TSER-01, TSER-02, TSER-03, PACE-01, PACE-02)
- 05-02: Odds time-series features (ODTS-01, ODTS-02)

</domain>

<decisions>
## Implementation Decisions

### TSER-01: 指数減衰重み付け
- **D-01:** 既存の`harontimel5_avg`列を指数減衰版に**置き換え**（新列追加しない）。EMAは単純平均の上位互換であり、多重共線性を回避する
- **D-02:** halflife=3走を採用。λ=ln(2)/3≈0.231。3走前の重みは直近の50%、5走前は25%
- **D-03:** ルックバックウィンドウを5走から全過去走に拡張。EMAが古い走を自動的に低加重するため、情報損失なしに全データを活用

### TSER-02: クラス調整済みフォーメトリック
- **D-04:** 新指標`class_adj_formetric`として新規追加。既存`form_trend`とは独立した特徴量
- **D-05:** 計算式: `Σ(norm_finish_i × class_level_i) / Σ(class_level_i)` — 高クラスでの好走を高く評価
- **D-06:** class_levelは既存の`_compute_class_level()`関数（`feature_engine.py`）または`_CLASS_LEVEL_MAP`（`horse_history_features.py`）を再利用。grade_code/jyoken_codeから計算

### TSER-03: z-score改善トラジェクトリ
- **D-07:** 過去走のz-scoreに対する線形回帰の傾きとして計算。新列`haron_zscore_trend`
- **D-08:** track_condition正規化は既存`harontimel5_zscore`と同じ方式（distance_bin × surface階層的z-score）を使用。追加の正規化は不要
- **D-09:** 計算に最低3走以上の有効z-scoreが必要。不足時はNaN（LightGBMが自然処理）

### PACE-01: 総合ペースフィグア
- **D-10:** 複数サブ特徴量として出力（単一スコアに圧縮しない）。LightGBMが非線形組み合わせを自動学習
- **D-11:** 出力特徴量:
  - `pace_corner_stability`: 1C→4Cの位置変位の安定性（低い＝一貫した位置取り）
  - `pace_closing_power`: 上がりタイムの相対的位置（低い＝速い上がり）
  - `pace_position_consistency`: 過去走間の正規化着順のばらつき（低い＝安定）
- **D-12:** 既存`PaceAptitudeFeatures.compute_batch()`パターン内に追加。同じsearchsorted + numpy集計パターン

### PACE-02: pace_scenario_fit強化
- **D-13:** 新列`actual_pace_fit`を追加。既存`pace_scenario_fit`は宣言脚質ベースで残す
- **D-14:** actual_pace_fitは実績ベースのfront_pace_wr/closing_pace_wrを使用。実際の走法パターンを反映
- **D-15:** `interaction_features.py`の既存pace_scenario_fit計算部にactual_pace_fit生成を追加

### ODTS-01: 2次微分(加速度)
- **D-16:** 3点差分型を採用。既存のvelocity(t-60→t-30)とvelocity(t-30→t-10)の差分
- **D-17:** 出力: `odds_acceleration = velocity_late - velocity_early`。正＝オッズ低下が加速（steam move）
- **D-18:** スナップショット不足（<3点）時はNaN

### ODTS-02: 方向一貫性
- **D-19:** 時間加重型を採用。直近の変動ほど高く評価。指数減衰で重み付け
- **D-20:** 出力: `odds_direction_consistency`。重み付き方向比率（0〜1、1＝全て同一方向）
- **D-21:** 最小スナップショット数=5点で足切り。不足時はNaN

### モジュール統合とNaN処理
- **D-22:** 全新特徴量を既存モジュールに追加。新規モジュールは作成しない:
  - TSER系 → `horse_history_features.py`
  - PACE系 → `pace_aptitude_features.py`
  - ODTS系 → `odds_dynamics_features.py`
- **D-23:** NaN処理はデフォルトNaN。LightGBMのネイティブNaN処理を活用。0埋めしない
- **D-24:** `FeatureEngine.build_all()`の既存ステップ内で各モジュールを呼び出し。新ステップ追加は不要（既存呼び出しの拡張のみ）

### Claude's Discretion
- EMA実装の詳細（numpy向量化、compute_batch内での重み配列生成）
- class_adj_formetricのclass_level取得方法（history entries/racesマージ時のgrade_code/jyoken_code可用性）
- pace_closing_powerの上がりタイムソース（entries_histのagi列 or harontimel3近似）
- odds_direction_consistencyの減衰率（halflife = スナップショット数/4程度）
- 各特徴量のNaN率が50%超の場合のフォールバック戦略

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### TSER系特徴量の参照コード
- `src/features/horse_history_features.py` — HorseHistoryFeatures.compute_batch()。TSER-01~03の実装対象。lines 262-1148にcompute_batch本体。BASE_COLS（line 265）に新列追加が必要
- `src/features/form_cycle_features.py` — compute_form_features()。form_trendの線形回帰実装。TSER-03と同じパターン
- `src/features/feature_engine.py` lines 84-182 — FeatureEngine.build_all()。特徴量パイプライン統合ポイント

### PACE系特徴量の参照コード
- `src/features/pace_aptitude_features.py` — PaceAptitudeFeatures.compute_batch()。PACE-01の実装対象。searchsorted + numpy集計パターン
- `src/features/interaction_features.py` line 107 — pace_scenario_fit計算。PACE-02のactual_pace_fit追加ポイント

### ODTS系特徴量の参照コード
- `src/features/odds_dynamics_features.py` — compute_odds_dynamics()。ODTS-01~02の実装対象。_build_snapshot_datetimes(), _build_post_time_map()ヘルパー

### モデル特徴量カラム参照
- `src/models/stage1_ability_model.py` line 96 — AbilityModel FEATURE_COLS（pace_scenario_fitを含む）
- `src/models/two_stage_return_model.py` lines 79, 289, 351 — WinTwoStageModel FEATURE_COLS
- `src/domain/models.py` — SubmodelSet dataclass

### データソース
- `src/db/readers.py` — load_history_entries(), load_history_races(), load_odds_time_series_range()
- `data/odds/jodds_tanpuku/` — オッズ時系列Parquet（year/monthパーティション）

### REQUIREMENTS.md
- `.planning/REQUIREMENTS.md` — TSER-01~03, PACE-01~02, ODTS-01~02の要件定義

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **HorseHistoryFeatures.compute_batch()** (`horse_history_features.py`): searchsorted + numpy集計の完全ベクトル化パターン。horse_arrs辞書による列アクセス、valid_mask、expanding_statsの仕組みをそのまま活用。TSER-01~03の追加はこのループ内に組み込む
- **PaceAptitudeFeatures.compute_batch()** (`pace_aptitude_features.py`): 同じsearchsortedパターン。cum_front_count等の累積和アプローチを参考。PACE-01の新特徴量も同パターンで追加
- **compute_odds_dynamics()** (`odds_dynamics_features.py`): _build_snapshot_datetimes(), _build_post_time_map()ヘルパーはそのまま再利用。ODTS-01~02はこの関数内に追加
- **_compute_class_level()** (`feature_engine.py` lines 29-48): grade_code/jyoken_code → class_level変換。TSER-02のclass_level取得に再利用
- **compute_form_features()** (`form_cycle_features.py`): np.polyfitを使った線形回帰傾き計算。TSER-03と同じアプローチ

### Established Patterns
- **FeatureEngine.build_all()ステップパターン**: TimingContextでラップしたfrom-import + 関数呼び出し。新ステップは不要（既存呼び出しの拡張のみ）
- **BASE_COLS拡張**: HorseHistoryFeatures.BASE_COLSに新列名を追加。モデルのFEATURE_COLSにも追加が必要
- **compute_batch → merge パターン**: PaceAptitudeFeatures.compute_batch()がkettonum + race_idをキーにDataFrameを返す → FeatureEngine.build_all()でmerge
- **NaN-safe Series構築**: `pd.Series(np.nan, index=df.index, dtype=float)` でデフォルトNaN Series生成

### Integration Points
- **FeatureEngine.build_all() line 155-161**: compute_intra_race_features → compute_odds_dynamics の間。PACE-02のactual_pace_fitはinteraction_featuresの後にcompute可能
- **HorseHistoryFeatures.compute_batch() lines 673-748**: harontimel5_avg/harontimel5_zscore/harontime_late_trend計算部。TSER-01のEMA化とTSER-03のz-score trend追加ポイント
- **PaceAptitudeFeatures.compute_batch() lines 138-250**: ペース特徴量計算ループ。PACE-01の新特徴量追加ポイント
- **compute_odds_dynamics() line 112-**: オッズ変動計算。ODTS-01~02の追加ポイント
- **モデルFEATURE_COLS更新**: AbilityModel, WinTwoStageModel等のFEATURE_COLSに新列名を追加（Phase 7のスタッキング時にも必要）

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「難易度は問わない」方針。品質優先で実装する
- 指数減衰重み付けのhalflife=3は金融時系列解析の標準的な選択。過去走5-15走という限られたデータ点数に適合
- クラス調整フォーメトリックは高クラス好走を高く評価する設計。重賞勝利と未勝利戦勝利の価値を区別
- ペース特徴量は複数サブ特徴量に分割。単一スコアよりLightGBMの学習効率が良い
- オッズ方向一貫性は時間加重型。直近の変動パターンが最も予測的（steam moveはレース直前に加速）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 5-Foundation Features*
*Context gathered: 2026-05-03*
