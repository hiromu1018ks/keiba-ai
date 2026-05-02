# Phase 2: Win Benter Combination & Calibration - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

単勝予測に市場確率（単勝オッズ由来）をブレンドするWinBenterGateを実装し、Beta/Isotonicキャリブレーションを比較評価し、レース単位正規化（ΣP=1.0）を適用して、EV推定精度を飛躍的に向上させる。

**In scope (from ROADMAP.md):**
- BENT-01: 単勝予測にBenter組み合わせを実装
- BENT-02: Beta calibrationとIsotonic calibrationの比較評価
- BENT-03: レース単位正規化（P合計=1.0）

**Out of scope:**
- Place/Wide Benterの変更
- ベッティング戦略の変更（Phase 3）
- 選択ゲートの実装（Phase 3）
- Walk-forward検証（Phase 4）

</domain>

<decisions>
## Implementation Decisions

### Benter入力の設計
- **D-01:** fundamental確率は2Stage EV補正後（WinTwoStageModel.predict_ev() → EVCorrection後）を使用する
- **D-02:** 市場確率ソースは `tanoddslow`（最終単勝オッズ）。レース前オッズなので情報リークなし
- **D-03:** 前処理は `p_market = 1/tanoddslow` のまま。JRA控除率はBenterのβパラメータが吸収する
- **D-04:** Benter学習データは専用OOF（out-of-fold）予測で生成する。`use_ensemble`に依存しない独立したKFold CV方式。ベストプラクティス追求

### キャリブレーション手法選択
- **D-05:** Beta calibrationとIsotonic calibrationの両方を実装し、比較評価する（BENT-02要件）
- **D-06:** パイプライン構成: `raw_p → Benter → {Beta|Isotonic} → TempScale(オプション)`。TempScaleは追加改善がある場合のみ適用
- **D-07:** 比較評価指標は Brier Score + ECE (Expected Calibration Error) で定量比較。信頼性ダイアグラムは可視化用
- **D-08:** Beta calibration（3パラメータ）がIsotonicより過学習しにくく推奨。PlaceでのIsotonic失敗は自由度過多が原因

### レース正規化の設計
- **D-09:** Benter学習は馬単位で行い、レース正規化は後処理として独立適用する。ベストプラクティス
- **D-10:** 正規化方式は単純正規化 `P_normalized = P_i / Σ(P_j)` を標準とする。Benter最適化内への制約追加は複雑化が大きく不採用

### Placeパターンからの逸脱
- **D-11:** WinBenterGate新クラスを作成。既存BenterCombinationを内部で利用しつつ、Win固有の前処理（tanoddslow読み込み）と後処理（レース正規化）を統合。Placeコードに影響なし
- **D-12:** SubmodelSetに `win_benter`, `win_isotonic_calibrator`, `win_temperature_scaler` フィールドを追加。Placeと並列構造
- **D-13:** Win Benterの最適化パラメータ（α/β/γ）はグリッドサーチで最適な初期値を探索する

### Claude's Discretion
- キャリブレーションパイプラインの詳細実装（各ステップの有効/無効判定ロジック）
- グリッドサーチの範囲・粒度の設定
- TempScale適用の閾値（どの程度の改善があれば適用するか）
- 信頼性ダイアグラムの出力形式とバケット数

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Benter・キャリブレーション関連コード
- `src/models/benter_combination.py` — 既存BenterCombination + TemperatureScalingクラス。WinBenterGateの内部コンポーネント
- `src/backtest/race_predictor.py` — 予測パイプライン。Benter/Calibの適用ポイント。lines 124-156がPlace Benter適用部
- `src/domain/models.py` — SubmodelSet dataclass (line 228)。win_*フィールド追加対象
- `src/pipelines/training_pipeline.py` — 学習パイプライン。Benter学習はlines 528-565。Win Benter学習の追加ポイント

### 学習データ・保存・読み込み
- `src/db/model_loader.py` — モデル読み込み。lines 502-529がBenter/Isotonic/TempScale読み込み。win_*フィールドの読み込み追加が必要
- `src/pipelines/training_pipeline.py` lines 998-1012 — モデル保存。win_*フィールドの保存追加が必要

### 既存のキャリブレーション失敗の教訓
- `src/backtest/race_predictor.py` lines 143-148 — Isotonic無効化のコメント（v5.6）。平均確率0.224 vs 真値0.375の過剰押し下げ問題

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **BenterCombination** (`src/models/benter_combination.py`): fit/combine/to_dict/from_dict/save/load APIをそのままWinBenterGate内部で利用可能
- **TemperatureScaling** (`src/models/benter_combination.py`): 同ファイルに既存実装あり。Win用にも再利用
- **IsotonicRegression** (`sklearn.isotonic`): 学習パイプラインで既にimport済み。Win用に追加fit可能
- **OOF予測生成パターン**: TrainingPipelineV5のKFoldパターンを流用

### Established Patterns
- **SubmodelSet拡張パターン**: Optionalフィールド追加（`benter_combo: BenterCombination | None = None`）の既存パターンに従う
- **学習→保存→読み込みの3点更新**: 新モデル追加時は training_pipeline.py / model_loader.py / domain/models.py の3ファイル更新が必須
- **RacePredictor適用パターン**: predict()内でsubmodelからcomponentを取得→適用→結果をDataFrameに書き戻す

### Integration Points
- **TrainingPipelineV5._train_submodel()**: Win Benter学習の挿入ポイント。Place Benter学習(lines 528-565)の後にWin Benterを追加
- **RacePredictor.predict()**: Win予測後のBenter適用ポイント。EV correction後(line 113あたり)にWinBenterGateを適用
- **BacktestEngine.run()**: バックテスト時のtanoddslow列アクセス。既にDataFrame内に存在する列
- **SubmodelSet**: win_*フィールドの追加。domain/models.py:228のdataclass

</code_context>

<specifics>
## Specific Ideas

- ユーザーは「実装難易度は問わない、ベストプラクティスを追求」という方針。品質優先で実装する
- グリッドサーチによるα/β/γ初期値探索は実行時間増を許容する
- 信頼性ダイアグラムによるキャリブレーション品質の可視化確認がSuccess Criteriaに含まれる

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 2-Win Benter Combination & Calibration*
*Context gathered: 2026-05-02*
