# Phase 10: Pipeline Performance - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

バックテスト・学習パイプラインの実行時間が短縮され、ボトルネックが定量測定可能になる。

**In scope (from ROADMAP.md):**
- PERF-01: build_payout_map()/build_wide_payout_map()のiterrows()をベクトル化pandas操作に置き換えられる
- PERF-02: レースごとのDataFrameフィルタリングをgroupby辞書の前処理に置き換え、O(n_races * n_rows)→O(1)ルックアップにできる
- PERF-03: HorseHistoryFeatures等の履歴特徴量をParquetキャッシュし、バックテスト再実行時に再計算をスキップできる
- PERF-04: pyinstrumentによるプロファイリングを統合し、ボトルネックの定量測定ができる

**Out of scope:**
- アルゴリズムの変更（モデル精度の改善）
- 複勝/ワイドモデルの変更
- 新データ源の導入
- マルチプロセス/分散処理の導入

**Plans:** 2 plans
- 10-01: Vectorize payout maps + groupby dict lookups (PERF-01, PERF-02)
- 10-02: Feature cache + pyinstrument profiling (PERF-03, PERF-04)

</domain>

<decisions>
## Implementation Decisions

### ベクトル化の範囲と手法 (PERF-01)
- **D-01:** engine.py の iterrows() 7箇所すべてをベクトル化する。部分的な残置はしない
- **D-02:** wide payout map の _parse_kumi() ベクトル化は正規表現ベースの pandas vectorized string 操作で実装。str.len() で文字列長を分類し、条件付きで分割する pandas-idiomatic なアプローチ
- **D-03:** top3 抽出(3箇所)も nsmallest() ベースのベクトル化に置き換え。レース内3行のみだが統一性のため
- **D-04:** build_payout_map() は melt + groupby でベクトル化。payfukusyoumaban1-5/payfukusyopay1-5 を縦持ちにして一括処理
- **D-05:** build_win_payout_map() は最もシンプル。単一列の map 構築なので直接 Series → dict 変換
- **D-06:** final_odds_map / final_win_odds_map も set_index + to_dict() でベクトル化

### groupby辞書戦略 (PERF-02)
- **D-07:** feat_df + hist/jockey/trainer/jt の5つのDataFrameすべてをgroupby辞書に変換。O(1)ルックアップで統一
- **D-08:** ヘルパー関数 `build_race_groups()` を作成し、groupby辞書構築をカプセル化。辞書構築 + 空グループログ + メモリ使用量ログを統合
- **D-09:** メモリ安全性: pandas≥2.0 の groupby は view を返すため、実質的なメモリ増加は元の1.1〜1.2倍程度。バックテストデータ(~38,000行)であればオーバーフローリスクなし

### 特徴量キャッシュ (PERF-03)
- **D-10:** キャッシュ対象は事前計算される全6種の特徴量:
  - HorseHistoryFeatures
  - JockeyContextFeatures
  - TrainerContextFeatures
  - JockeyTrainerComboFeatures
  - SireFeatures
  - PaceAptitudeFeatures + CourseFeatures
- **D-11:** キャッシュ保存場所は `data/features/cache/` 専用ディレクトリ。既存の特徴量ファイルと明確に分離
- **D-12:** キャッシュ無効化はハイブリッド方式: タイムスタンプで高速チェック → 変更ありならコンテンツハッシュで検証。ベストプラクティス
- **D-13:** キャッシュキー: 入力Parquetファイルのパス + 日付範囲 + 特徴量種別 をハッシュ化したもの

### pyinstrument統合 (PERF-04)
- **D-14:** `--profile` CLIフラグで起動。run_backtest.py と run_wf_validation.py の両方に統合
- **D-15:** 共通プロファイリングユーティリティを `src/utils/profiling.py` に抽出。両スクリプトから利用
- **D-16:** 出力形式はHTML + テキストの両方。HTMLは `data/profiles/` に保存、テキストはstdoutに出力
- **D-17:** pyinstrumentのオーバーヘッドは5%未満。普段の実行(--profile未指定)には影響なし

### Claude's Discretion
- 正規表現の具体的なパターン設計(kumi文字列の3パターン分解)
- build_race_groups() のシグネチャと返り値の型
- キャッシュ無効化ハッシュの具体的な計算方法(MD5, SHA256等)
- キャッシュファイルの命名規則
- pyinstrumentユーティリティのAPI設計(context manager vs decorator)
- HTMLレポートのテンプレート(デフォルトpyinstrument HTMLで十分)
- ベクトル化後のビルドアップリケーションテストの範囲

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### バックテストエンジン（主変更対象）
- `src/backtest/engine.py` lines 102-209 — build_payout_map()/build_win_payout_map()/build_wide_payout_map()。ベクトル化の対象
- `src/backtest/engine.py` lines 320-352 — final_odds_map/final_win_odds_map構築。ベクトル化の対象
- `src/backtest/engine.py` lines 385-468 — 特徴量事前計算。キャッシュ統合ポイント
- `src/backtest/engine.py` lines 476-530 — レースループ。groupby辞書化の対象(feat_df + hist/jockey/trainer/jt)
- `src/backtest/engine.py` lines 503-519 — top3抽出 iterrows()。nsmallest() ベクトル化の対象

### 特徴量モジュール（キャッシュ対象）
- `src/features/horse_history_features.py` lines 262-320 — HorseHistoryFeatures クラス。インメモリキャッシュのみ。Parquetキャッシュ追加ポイント
- `src/features/jockey_context_features.py` lines 30-37 — JockeyContextFeatures。_stats_cache
- `src/features/trainer_context_features.py` lines 30-37 — TrainerContextFeatures。_stats_cache
- `src/features/jockey_trainer_combo.py` lines 31-42 — JockeyTrainerComboFeatures。_cache
- `src/features/sire_features.py` — SireFeatures。compute_batch()
- `src/features/pace_aptitude_features.py` — PaceAptitudeFeatures。compute_batch()
- `src/features/course_features.py` — CourseFeatures。compute_batch()
- `src/features/feature_engine.py` — FeatureEngine。build_all()。キャッシュ制御の統合ポイント

### データアクセス層
- `src/db/parquet_store.py` — ParquetStore。キャッシュ読み書きに使用
- `src/db/readers.py` — DataReaders。特徴量キャッシュの読み込みポイント

### CLIスクリプト（プロファイリング統合）
- `scripts/run_backtest.py` — --profile フラグ追加ポイント
- `scripts/run_wf_validation.py` — --profile フラグ追加ポイント

### ユーティリティ（新規作成）
- `src/utils/profiling.py` (新規) — 共通プロファイリングユーティリティ
- `src/utils/timing.py` — 既存 TimingContext。参考パターン

### 要件定義
- `.planning/REQUIREMENTS.md` — PERF-01, PERF-02, PERF-03, PERF-04
- `.planning/ROADMAP.md` — Phase 10 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **TimingContext** (`src/utils/timing.py`): 既存のコンテキストマネージャー。pyinstrumentユーティリティの設計パターンとして参照可能
- **ParquetStore** (`src/db/parquet_store.py`): ParquetファイルI/O。特徴量キャッシュの読み書きにそのまま利用可能
- **FeatureEngine** (`src/features/feature_engine.py`): 全特徴量モジュールのオーケストレーター。キャッシュ制御の統合先
- **DataRepository** (`src/db/repository.py`): MLパイプラインのデータアクセス窓口。キャッシュファイルパスの管理に利用

### Established Patterns
- **Parquetファイルパターン**: `data/{category}/{name}.parquet` 形式。`data/features/cache/` も同じ規約に従う
- **CLIフラグパターン**: argparse に add_argument()。既存 --betting-target, --betting-mode が参考
- **コンテキストマネージャーパターン**: TimingContext が with ブロックでタイミング測定。pyinstrument も同パターンで統合
- **インメモリキャッシュパターン**: HorseHistoryFeatures._entries_cache, JockeyContextFeatures._stats_cache 等。`if self._cache is None:` の遅延初期化パターン

### Integration Points
- **engine.py:341-343** — feat_engine.build_all()。特徴量キャッシュの判定ポイント。キャッシュ有効ならビルドをスキップ
- **engine.py:385-468** — 特徴量事前計算ブロック全体。キャッシュ判定 + キャッシュ読み込み/書き込みの統合ポイント
- **engine.py:476** — レースループの race_ids イテレーション。groupby辞書への置き換えポイント
- **scripts/run_backtest.py** — CLI引数定義。--profile フラグ追加
- **scripts/run_wf_validation.py** — CLI引数定義。--profile フラグ追加

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・完全性を優先
- メモリ安全性は重要な考慮事項。groupby辞書化でメモリオーバーフローが起きないよう、データサイズに応じたログ出力を実装
- pyinstrument は run_backtest.py だけでなく run_wf_validation.py にも統合。WF検証は複数フォールド実行で時間がかかるため
- 全6種特徴量キャッシュは包括的だが、ハイブリッド無効化(タイムスタンプ + ハッシュ)で安全性を確保
- ヘルパー関数 build_race_groups() で groupby 辞書構築をカプセル化し、再利用性と保守性を確保

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-Pipeline Performance*
*Context gathered: 2026-05-04*
