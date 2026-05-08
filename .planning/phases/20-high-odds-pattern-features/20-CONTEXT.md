# Phase 20: 高オッズ的中パターン特徴量 - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

高オッズ帯(20+)の的中率を2.1%から3%+に引き上げるための新特徴量（HODDS-01~05）を設計・実装し、AbilityModelとWinTwoStageModelに統合する。

**In scope:**
- HODDS-01: 高オッズ的中パターン分析モジュール（ハイブリッド分析: 統計プロファイリング + SHAP）
- HODDS-02: クラストラジェクトリ特徴量（数値シーケンス分解、直近5走）
- HODDS-03: フォーム改善率特徴量（EMAベース指数改善率、タイム+着順両方）
- HODDS-04: 環境変化適性特徴量（3変化×3サブ特徴量=9特徴量）
- HODDS-05: 新特徴量のFeatureEngine統合・モデルFEATURE_COLS更新
- Feature importance分析による効果検証

**Out of scope:**
- モデル再学習・ハイパーパラメータ再最適化 (Phase 22で検証)
- 複勝/ワイドモデルへの特徴量追加
- ベッティング戦略の変更
- Conformal EV予測区間 (Phase 21)
- 学習済みクラスタリングモデルの推論パイプライン組み込み

</domain>

<decisions>
## Implementation Decisions

### 高オッズパターン分析手法 (HODDS-01)
- **D-01:** ハイブリッド分析を採用。統計プロファイリング（高オッズ的中馬 vs 非的中馬の特徴量分布差、効果量Cohen's dで順位付け）+ SHAP分析（既存LightGBMモデルで高オッズ馬のみフィルタしてSHAP値を計算）。両方の結果を統合して特徴量設計に反映する。
- **D-02:** 分析スクリプトと特徴量生成モジュールを分離。分析は `scripts/` に、特徴量生成は `src/features/` に配置。分析結果に基づいて特徴量設計を独立して反復可能にする。
- **D-03:** 高オッズの定義は初期オッズ20+で分析を開始。サンプル不足が判明した場合はオッズ10+に拡張して分析範囲を調整。
- **D-04:** 分析結果は手動特徴量設計に反映する。クラスタリングモデル等の学習済みモデルは使用しない。実行時オーバーヘッドなし、確定的特徴量、高い解釈性を確保。

### クラストラジェクトリ設計 (HODDS-02)
- **D-05:** 数値シーケンス分解を採用。直近5走のクラス変遷を数値化（未勝利=0, 1勝=1, 2勝=2, OP=3, 重賞=4）し、昇級回数・降級回数・ネット変化・最高クラス到達・クラス分散のスカラー値に分解。LightGBMが非線形組み合わせを自動学習する。パターン分類（カテゴリ変数）は採用しない。
- **D-06:** 直近5走を使用。既存HorseHistoryFeaturesのn_past=5に一致。計算コストと情報量のバランスが最適。
- **D-07:** V字回復パターン（降級→再昇級）をバイナリフラグ特徴量として追加。降級からの再昇級は高オッズ的中の強力シグナル。既存class_drop_bounce（直近1走のみ）と異なり、複数走にわたるバウンスを捉える。降級期間（降級から再昇級までの走数）も併せて特徴量化。

### フォーム改善率の測定 (HODDS-03)
- **D-08:** EMAベース指数改善率を採用。halflife=3のEMA重み付けで直近の改善を強調。線形回帰のform_trendやharon_zscore_trendとは直交する非線形回復パターンを捉える。
- **D-09:** タイムベースと着順ベースの両方を計算。タイムはクラス調整済みz-score（harontimel5_zscore）のEMA改善率。着順は正規化着順（pos-1)/(size-1）のEMA改善率。計2特徴量。2期間差分（直近 vs 以前）は採用しない。

### 環境変化適性の範囲 (HODDS-04)
- **D-10:** 既存3変化（距離変更・サーフェス変更・馬場状態変更）の過去適性履歴を計算。騎手変更・調教師変更は含めない（新しい騎手ごとにサンプル不足になりやすいため）。
- **D-11:** 各変化について3サブ特徴量に分解: 「同変更時の平均着順」「同変更時の勝率」「同変更の経験回数」。3変化×3特徴量=9特徴量。距離はbin単位（sprint/mile/intermediate/long）、馬場は広カテゴリ（良/稍重/重/不良）で集計してサンプル不足を補う。

### モデル統合 (HODDS-05)
- **D-12:** 新特徴量はStage1AbilityModel.FEATURE_COLSに追加。AbilityModelはP(hit)のベースモデルであり、高オッズ的中パターンの改善に最も直接的に寄与する。
- **D-13:** FeatureEngine.build_all()内で既存のHorseHistoryFeaturesループに新特徴量の計算を統合する。独立した特徴量モジュールファイル（src/features/high_odds_features.py等）を作成し、FeatureEngineから呼び出す。
- **D-14:** 新特徴量は欠損率10%以下を要件とする（ROADMAP Success Criteria）。過去走が不足する初戦馬等ではNaNとなり、LightGBMがNaN処理可能なため問題なし。

### Claude's Discretion
- 新特徴量の具体的な命名規則（snake_case一貫性）
- HorseHistoryFeatures.compute()内での計算箇所（既存ループ内に統合するか独立関数にするか）
- 分析スクリプトの出力形式（JSON/Markdown/PNG等）
- クラスレベルの数値マッピングの詳細（grade_code/jyokencd1からの変換）
- サンプル不足時のフォールバック戦略（経験回数0の環境変化適性のデフォルト値）
- Feature importance分析の具体的な比較方法（ベースラインOOF AUC vs 新特徴量追加後）
- テストのfixtureデータとモック構成

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### モデルFEATURE_COLS（主変更対象 — HODDS-05）
- `src/models/stage1_ability_model.py:28-107` — Stage1AbilityModel.FEATURE_COLS（~80列）。新特徴量の追加先
- `src/models/two_stage_return_model.py:289-404` — WinTwoStageModel HIT_FEATURE_COLS / RETURN_FEATURE_COLS。必要に応じて追加検討

### 特徴量計算エンジン（統合ポイント — HODDS-05）
- `src/features/feature_engine.py:118` — FeatureEngineクラス。build_all()内で新特徴量モジュールを呼び出し
- `src/features/horse_history_features.py:265-450` — HorseHistoryFeaturesクラス。per-horseループ内での特徴量計算の統合先
- `src/features/form_cycle_features.py` — フォームサイクル特徴量の既存パターン参考。compute_form_features()のAPI設計テンプレート

### 既存環境変化特徴量（HODDS-04の関連コード）
- `src/features/horse_history_features.py:73` — `_compute_distance_bin()`。距離bin定義
- `src/features/horse_career_stats.py` — 馬のキャリア統計。環境変化適性の計算に参照
- `src/features/interaction_features.py` — 交互作用特徴量。既存distance_change/surface_change/class_moveの生成

### 分析スクリプト（HODDS-01の参考）
- `scripts/analyze_feature_importance.py` — 既存のSHAP/gain特徴量重要度分析スクリプト。高オッズ分析の拡張ベース

### データレイヤー
- `src/db/parquet_store.py` — ParquetStore。特徴量データの読み書き
- `src/db/repository.py` — DataRepository。MLパイプラインのデータアクセス窓口

### ドメイン型
- `src/domain/types.py` — Surface, BetType等の型定義
- `src/domain/models.py` — SubmodelSet等のデータクラス

### 設定ファイル
- `config/settings.yaml` — feature_engine設定

### テスト
- `tests/test_feature_engine.py` — FeatureEngine既存テスト
- `tests/test_horse_history_features.py` — HorseHistoryFeatures既存テスト
- `tests/test_form_cycle_features.py` — FormCycleFeatures既存テスト

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **HorseHistoryFeatures** (`src/features/horse_history_features.py`): per-horseループ構造が確立。新特徴量はこのループ内に統合可能。compute()メソッドが戻り値としてDataFrame["race_id", "umaban"] + BASE_COLSを返すパターン
- **FormCycleFeatures** (`src/features/form_cycle_features.py`): 過去走配列から特徴量を計算する関数型API。`compute_form_features(kakuteijyuni, syussotosu) -> tuple[float, float, float]`のパターンを踏襲
- **FeatureEngine** (`src/features/feature_engine.py`): build_all()内で各特徴量モジュールを順次呼び出すオーケストレーション。新モジュールを追加するだけで統合完了
- **Stage1AbilityModel** (`src/models/stage1_ability_model.py`): _prepare_features()がavailable_colsフィルタリングを持つ。新特徴量が欠損していても安全にスキップ可能
- **InteractionFeatures** (`src/features/interaction_features.py`): class_move/distance_change/surface_change生成の既存実装。HODDS-04の入力データ参照

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。新特徴量テストもこのパターンに従う
- **FEATURE_COLSリスト**: モデルクラスにFEATURE_COLS list[str]を定義。利用可能列のみ使用（available_cols フィルタ）
- **時系列順の維持**: 過去走はrace_date降順（idx=0が最新）。新特徴量もこの順序を前提とする
- **Categorical cast**: surface, distance_bin, grade_codeは.astype("category")でLightGBMに渡す
- **サブ特徴量分解**: LightGBMが非線形組み合わせを自動学習するため、スカラー値に分解する設計が好まれる（Phase 5のペース3サブ特徴量分解が確立したパターン）

### Integration Points
- **feature_engine.py:build_all()** — 新特徴量モジュールの呼び出し追加ポイント
- **horse_history_features.py:compute()** — per-horseループ内に新特徴量計算を統合
- **stage1_ability_model.py:FEATURE_COLS** — 新特徴量名の追加
- **feature_engine.py:build_features()** — 推論パス（単レース特徴量計算）。build_all()と対になる実装

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- 学習時間・バックテスト時間を延ばさないことが明示的な制約。新特徴量は軽量なpandas/numpy計算のみ
- 高オッズ的中パターンの分析は限定的なサンプル（~0.3%）で行うため、統計的有意性の確認が重要
- V字回復（降級→再昇級）パターンは高オッズ的中の強力シグナルという仮説。分析で検証が必要
- EMA halflife=3はPhase 5で確立した標準値。フォーム改善率もこれに従う
- クラスレベルの数値化はgrade_code/jyokencd1から既存の_class_level_from_values()で取得可能

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 20-高オッズ的中パターン特徴量*
*Context gathered: 2026-05-09*
