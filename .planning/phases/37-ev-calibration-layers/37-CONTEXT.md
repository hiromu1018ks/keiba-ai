# Phase 37: EV Calibration Layers - Context

**Gathered:** 2026-05-20
**Status:** Ready for planning

<domain>
## Phase Boundary

人気帯キャリブレーションとレジーム×サーフェスEV補正でEV精度を改善し、特に芝の中穴（人気4-12）のキャリブレーションratioを0.527から改善する。

**In scope:**
- CAL-01: 人気帯キャリブレーション (1-3, 4-6, 7-9, 10-12, 13+) のOOF residual ratio スケーリングを ev_correction_model.py に追加
- CAL-02: 人気帯キャリブレーションに拡張ウィンドウOOF計算を適用し、ルックアヘッドバイアスを防止
- CAL-03: EVCorrectionModel.FEATURE_COLS に regime_state, surface_x_popularity, market_entropy_x_surface を追加
- CAL-04: regime_state を RacePredictor → EVCorrectionModel 間で伝播させる仕組みを実装
- CAL-05: レジーム-EVフィードバックループの強制遷移テストを実装
- Pop-bandスケール係数はsurface別（turf/dirt）に計算（5バンド×2サーフェス = 10係数）
- Win + Place 両方に適用
- 正味のEV補正パイプライン: P×E → Isotonic → Odds-band → Pop-band

**Out of scope:**
- 新特徴量の計算 (Phase 36)
- バックテスト実行 (Phase 38)
- モデル再学習・ハイパーパラメータチューニング
- ワイドモデルへの適用 (スコープ外)
- OddsBandFilterの再構築

</domain>

<decisions>
## Implementation Decisions

### 人気帯キャリブレーション層の設計 (CAL-01)
- **D-01:** Pop-bandキャリブレーションは Isotonic + Odds-band の **後**（最も外側の層）に適用。既存層の最終残差をターゲットにする。パイプライン: `P×E → Isotonic → Odds-band → Pop-band`
- **D-02:** スケーリング係数は **Median residual ratio**（actual/calibratedの中央値）。既存Odds-bandスケーリングと同一方式。heavy-tail分布に対して頑健
- **D-03:** 適用範囲は **Win + Place 両方**。PlaceEVCorrectionModelにも別のpop_band_scalesを計算
- **D-04:** 5バンド固定境界 (1-3, 4-6, 7-9, 10-12, 13+)。スケール係数は **surface別**（turf/dirt）。5バンド×2サーフェス = 10係数。バンド境界は芝ダート共通、係数のみ別計算
- **D-05:** PlaceEVCorrectionModelの既存 `_build_place_bucket_multiplier()` (ハードコードpenalty for pop>=12/15/18) は残差ベースpop-band scalingと共存させるか置き換えるかは実装時に判断

### 拡張ウィンドウOOF計算 (CAL-02)
- **D-06:** **Expanding window 5-fold OOF** を適用。時系列ソート済みデータで各foldは「それ以前の全データ」で学習、「次のセグメント」で予測。ルックアヘッド完全防止
- **D-07:** Pop-band scalesの計算は既存 `fit_ev_calibration()` 内に統合。同一OOFサイクル内でOdds-band scales + Pop-band scalesを同時計算。学習時間の追加なし
- **D-08:** Pop-band scalesの格納形式: `ev_pop_band_scales: dict[str, dict[str, float]]`（key1=surface "turf"/"dirt", key2=band_name "1-3"/"4-6"/"7-9"/"10-12"/"13+"）
- **D-09:** 最小サンプル閾値: 各band×surfaceでN件未満（Nは実装時決定、例: 30件）の場合、累積推定を使用（分散安定化）

### レジーム伝播アーキテクチャ (CAL-03/04)
- **D-10:** regime_stateは **事前計算 → DataFrame列** としてpredict()に渡す。BacktestEngine.run()でRegimeDetector.detect()をpredict()の前に呼び、結果をdf["regime_state"]として追加。predict()は純粋関数のまま
- **D-11:** regime_stateのエンコーディングは **Ordinal** (0=AGGRESSIVE, 1=CONSERVATIVE, 2=COLLAPSED)。自然な順序（リスク昇順）。RegimeState.valueが既にint
- **D-12:** interaction featuresは **乗算相互作用**:
  - `surface_x_popularity = surface * popularity_rank`
  - `market_entropy_x_surface = market_entropy * surface`
  - 計算場所: RacePredictor.predict()内でEV補正前に列を生成（interaction_features.pyではなくEV補正専用）
  - 追加先: **EVCorrectionModel.FEATURE_COLSのみ**（全12モデルではなく）

### フィードバックループテスト (CAL-05)
- **D-13:** **Regime independence test** を実装。同一レースに対してregime_state=0/1/2でcorrect_ev()を呼び、EV出力の相対変動率を検証
- **D-14:** 検証基準: `max_diff(regime_states) / median_ev < 5%`。LightGBMがregime_stateに過度に依存していないことを確認

### Claude's Discretion
- PlaceEVCorrectionModelの_build_place_bucket_multiplier()とpop-band scalingの共存/置き換え判断
- Expanding window foldの具体的な分割ポイント（時系列の等分割 or 日付ベース）
- 最小サンプル閾値Nの具体的な値
- Pop-band scale係数の下限・上限クリッピング（外れ値防止）
- テストケースの設計（regime independence testの具体的なテストデータ）
- 初回200レースのregime未定義時のregime_stateデフォルト値
- fit_ev_calibration()内でのexpanding window OOFの具体的な実装

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### EV補正モデル（主要変更対象）
- `src/models/ev_correction_model.py` — EVCorrectionModel / PlaceEVCorrectionModel。FEATURE_COLS(72列)、correct_ev()、_build_place_bucket_multiplier()。Pop-band scaling追加とregime_state列追加の主要変更対象
- `src/pipelines/training_pipeline.py` — fit_ev_calibration()（5-fold OOF isotonic + odds-band calibration）。Pop-band scales同時計算の拡張対象。lines 671-697

### レジーム検出・伝播
- `src/models/regime_detector.py` — RegimeDetector。detect()、get_strategy_params()、FEATURE_COLS(51列)。regime_stateのordinal値ソース
- `src/backtest/race_predictor.py` — predict()。EV補正フロー(line 161)。regime_state列追加 + interaction feature生成の実装場所
- `src/backtest/engine.py` — BacktestEngine.run()。RegimeDetector.detect()の呼び出し箇所。regime_stateのdf列追加実装場所

### 既存キャリブレーションパターン（参考）
- `src/betting/odds_band_filter.py` — OddsBandFilter.BANDS（4バンド閾値）。Pop-band閾値定義の参考

### テスト
- `tests/test_ev_correction_model.py` — 既存テスト。Pop-band scalingのテスト追加先
- `tests/test_post_race_leakage.py` — 3層CI漏洩検出テスト

### 要件定義
- `.planning/REQUIREMENTS.md` §CAL — CAL-01~05

### Prior Phase Context
- `.planning/phases/36-feature-computation/36-CONTEXT.md` — TRF/INT/HLF特徴量（23列）のEVCorrectionModel.FEATURE_COLS登録済み
- `.planning/phases/34-validation-and-manifest-update/34-CONTEXT.md` — fit_ev_calibration()パターン、Odds-band residual scaling

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fit_ev_calibration()` (training_pipeline.py lines 671-697): 5-fold OOF → IsotonicRegression → odds-band residual scales。Pop-band scales計算の拡張基盤。既存OOF予測を再利用可能
- `EVCorrectionModel.correct_ev()` (ev_correction_model.py lines 361-437): P補正 → E補正 → Isotonic → Odds-band のチェーン。Pop-bandステップを最後に追加するだけ
- `OddsBandFilter.BANDS`: 4バンド閾値 `[1.0-3.0), [3.0-10.0), [10.0-30.0), [30.0+inf)`。Pop-band閾値定義のパターン参照
- `RegimeState` enum (regime_detector.py): AGGRESSIVE=0, CONSERVATIVE=1, COLLAPSED=2。ordinal値としてそのままDataFrame列に使用可能
- `BacktestEngine.run()`: regime_state検出ループが既に存在。predict()の前にdf列として追加するだけ

### Established Patterns
- キャリブレーション層追加: コンストラクタに新しいcalibrator dictを追加 → correct_ev()に適用ステップを追加 → training_pipelineでOOF計算 → インジェクト
- OOF residual ratio: `actual / predicted` をバンド別にgroupby → `.median()` → dict化
- Feature列追加: FEATURE_COLS class-level listに追加 → `[c for c in FEATURE_COLS if c in df.columns]` で安全選択
- Regime伝播: detect() → DataFrame列追加 → predict()内でFEATURE_COLSに含まれる → LightGBMが自動利用

### Integration Points
- `src/models/ev_correction_model.py::EVCorrectionModel.__init__()`: ev_pop_band_scales引数の追加
- `src/models/ev_correction_model.py::EVCorrectionModel.correct_ev()`: Pop-band適用ステップの追加（Isotonic + Odds-bandの後）
- `src/pipelines/training_pipeline.py::fit_ev_calibration()`: Pop-band scales同時計算の追加
- `src/backtest/engine.py::run()`: regime_state列のdf追加（RegimeDetector.detect()直後）
- `src/backtest/race_predictor.py::predict()`: regime_state列 + interaction features(surface_x_popularity, market_entropy_x_surface)の生成
- 12モデルのFEATURE_COLS: EVCorrectionModel + PlaceEVCorrectionModelのみにregime_state + 2 interaction列を追加

</code_context>

<specifics>
## Specific Ideas

- Pop-band scale係数の計算はOdds-bandと同じmedian residual ratioパターン。`actual_ev / calibrated_ev` を人気帯×サーフェス別にgroupby→median
- ev_pop_band_scalesの構造: `{"turf": {"1-3": 1.02, "4-6": 0.95, ...}, "dirt": {"1-3": 1.01, "4-6": 0.98, ...}}`。値1.0 = 補正なし
- regime_stateはpredict()の入力DataFrameに列として追加される。BacktestEngine側で「現在のregime」を全馬に同じ値で設定。PaperPredictorでも同様
- 初回200レース（regime未定義）のデフォルト値はCONSERVATIVE(1)が安全（リスク回避的）
- Expanding window OOF: 時系列順に5分割。Fold1=train[0:20%]→test[20%:30%]、Fold2=train[0:30%]→test[30%:40%]...Fold4=train[0:70%]→test[70%:80%]、Fold5=train[0:80%]→test[80%:100%]
- Feedback loop testはunittest mockでregime_stateを0/1/2に変更してcorrect_ev()を呼び、EV出力の相対変動が5%以内であることを検証

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 37-EV Calibration Layers*
*Context gathered: 2026-05-20*
