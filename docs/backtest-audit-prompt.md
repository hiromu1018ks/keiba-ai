# バックテスト監査: PIT・リーク・現実性の徹底調査

あなたは競馬AI予測システムのバックテスト品質監査員です。
バックテストに Point-in-Time (PIT) 違反、データリーク、非現実的な前提がないか、
体系的に調査・修正してください。

## 進行管理

1. `docs/backtest-audit.md` を読み、既に完了した項目を確認
2. **未完了の最初の項目**から再開する
3. `docs/backtest-audit.md` が存在しない場合は、本ファイル末尾のテンプレートで作成

## 監査の方針

### PIT (Point-in-Time) とは
「時点 T での予測に使うデータは、すべて時点 T より前に利用可能だったものでなければならない」
競馬では: レース発走前までに知り得る情報のみを使用。レース結果、確定オッズ、確定人気等は使用不可。

### 各項目の進め方
1. **調査**: 該当コードを読み、データフローを追跡。「この情報は発走前に知り得たか？」を問う
2. **判定**: ✅ OK / ⚠️ 要確認 / 🐛 リーク発見 / ℹ️ メモ
3. **修正**: リークが確認された場合、最小限の変更で修正
4. **検証**: `python -m pytest tests/ -v` でテスト確認

### 修正の原則
- 最小限の変更で PIT 整合性を保証する
- 修正前後でロジックの差分を明確にする
- テストが通ることを確認してから次へ進む
- **backtest の再実行は最後に一括で行う** (各修正ごとには実行しない)

## 監査チェックリスト

### A. 特徴量 PIT 監査 (個別モジュール)

#### A1: HorseHistoryFeatures — 過去レース履歴の PIT
- `src/features/horse_history_features.py`
- `searchsorted(target_date, side='left')` が正しく「厳密な過去のみ」を保証しているか
- `compute()` 内の全特徴量で、未来レースが混入しないか確認
- 5年ルックバックが正しく機能しているか

#### A2: HorseCareerStats — 累積統計の PIT
- `src/features/horse_career_stats.py`
- `shift(1).fillna(0).cumsum()` が確実に現在行を除外しているか
- 出力 parquet の内容をサンプル確認

#### A3: SireFeatures — 種牡馬統計の PIT ⚠️ 要注意
- `src/features/sire_features.py`
- **既知の疑念**: `compute_batch()` 内の `subset.iloc[idx_arr[valid]].iloc[0]` が
  全行に同じ値を返す可能性 (race_date 無視)
- `compute_single()` との動作比較で正しさを検証
- BMS (母父) ルックアップにも同様の問題がないか

#### A4: BloodlineFeatures — 血統特徴量の PIT
- `src/features/bloodline_features.py`
- `horse_career_stats.parquet` の読み込みで PIT 整合性が保たれているか
- `drop_duplicates(keep='first')` が正しい累積値を保持しているか

#### A5: JockeyContextFeatures — 騎手統計の PIT
- `src/features/jockey_context_features.py`
- `setyear < race_year` フィルタが正しいか
- 年をまたぐ境界ケース (12月→1月) の処理

#### A6: TrainerContextFeatures — 調教師統計の PIT
- `src/features/trainer_context_features.py`
- A5 と同様の確認

#### A7: JockeyTrainerComboFeatures — 騎手×調教師の PIT
- `src/features/jockey_trainer_combo.py`
- `searchsorted(target_date, side='left')` の正しさ

#### A8: PaceAptitudeFeatures — ペース適性の PIT
- `src/features/pace_aptitude_features.py`
- `searchsorted(target_dates, side='right')` で累積値を正しく取得しているか

#### A9: CourseFeatures — コース特徴量の PIT
- `src/features/course_features.py`
- A8 と同様の確認

#### A10: OddsDynamicsFeatures — オッズ動態の PIT ⚠️ 要注意
- `src/features/odds_dynamics_features.py`
- **既知の疑念**: `compute_roi_ema()` の `ewm().mean()` が未来データを取り込む可能性
- `compute_rolling_volatility()` の `rolling()` の window 方向
- `tail(60)` が発走後データを含んでいないか

#### A11: MarketBiasFeatures — 市場バイアスの PIT
- `src/features/market_bias_features.py`
- `tanodds` (スナップショット) のみを使用しているか

#### A12: InfoAsymmetryFeatures — 情報非対称性の PIT
- `src/features/info_asymmetry_features.py`
- `expanding().mean().shift(1)` が正しく現在行を除外しているか

#### A13: FormCycleFeatures — フォームサイクルの PIT
- `src/features/form_cycle_features.py`
- **既知の疑念**: `norm[:2]` が「最新2走」ではなく「最古2走」を取得している可能性
- 配列のソート順序 (race_date昇順) とインデックスの対応

#### A14: IntraRaceFeatures — レース内特徴量の PIT
- `src/features/intra_race_features.py`
- 同一レース内の計算のみで PIT 問題がないか

#### A15: InteractionFeatures — 交互作用特徴量の PIT
- `src/features/interaction_features.py`
- `kyakusitukubun_cd` が確定後のデータではないか確認

#### A16: FeatureEngine 全体 — 特徴量統合の PIT
- `src/features/feature_engine.py`
- `build_all()` 内の odds 列の置換 (confirmed → pre-post) が正しいか
- `popularity_rank` が `tanninki` (事前) のみを使用しているか
- POST_RACE_COLS の除外が完全か

### B. データパイプライン PIT 監査

#### B1: DataRepository — 日付フィルタの正しさ
- `src/db/readers.py`
- `_date_filters()` の境界条件 (>= と <=)
- `load_history_entries()` が `datetime.now()` を使う問題
  (searchsorted で保護されているが、より安全な設計にできるか)

#### B2: ParquetStore — データ読み込みの安全性
- `src/db/parquet_store.py`
- 述語プッシュダウンが正しく動作しているか

#### B3: OddsExtractor — オッズ抽出タイミング
- `src/db/odds_extractor.py`
- `extract_pre_post_odds()` の発走5分前カットオフが現実的か
- データが欠落していた場合のフォールバック先が PIT 安全か

### C. 学習パイプライン PIT 監査

#### C1: 訓練/テスト分割 — 時間厳守
- `src/pipelines/training_pipeline.py`
- 学習データがテスト期間を含んでいないか
- サブモデル (芝/ダート) 分割時にテスト情報が混入しないか

#### C2: OOF 予測 — 時間順序の維持
- `src/models/walk_forward_cv.py`
- Expanding window が正しく過去のみを使用しているか
- `KFold(shuffle=False)` が時間順を保持しているか

#### C3: MarketModel — 時間ベース分割
- `src/models/market_model.py`
- 80/20 分割が時間順である前提の検証

#### C4: TwoStageModel — init_score / weight の PIT
- `src/models/two_stage_return_model.py`
- `init_score=logit(p_pred)` にリークがないか
- `weight=1/√p` がテスト情報を含んでいないか

#### C5: StackedEnsemble — メタモデルの PIT
- `src/pipelines/training_pipeline.py` (ensemble 部分)
- 80/20 分割がデータのソート順に依存している脆弱性

### D. バックテスト実行 PIT 監査

#### D1: BacktestEngine — 特徴量計算の PIT
- `src/backtest/engine.py`
- テスト期間全体を一括で特徴量計算しているが、
  各レースの特徴量が未来レースの情報を含んでいないか
- HorseHistoryFeatures 等の searchsorted に依存している部分の信頼性

#### D2: RacePredictor — 予測時のデータクリーンアップ
- `src/backtest/race_predictor.py`
- POST_RACE_COLS の除外が予測前に確実に行われているか
- EV 計算にリークがないか

#### D3: ベット判定 — 発走前情報のみ使用
- `src/backtest/race_predictor.py` (select_bets)
- `fukuoddslow` (発走前オッズ) のみを使用しているか
- 閾値判定に確定情報が混入していないか

#### D4: 精算 — 確定オッズの使用
- `src/backtest/engine.py` (_settle_bet)
- 精算に確定オッズを使っているか (正しい動作)
- 複勝オッズの精算計算が正しいか

### E. 現実性監査

#### E1: オッズスナップショットの現実性
- 発走5分前のオッズが、実際に投票可能なオッズか
- オッズは刻々変わる — スナップショットとの乖離リスク

#### E2: 最小賭け金額の制約
- 100円単位の制約が正しく反映されているか

#### E3: 複数同レースベット
- 同一レースの複数馬にベットした場合のリスク評価
- 実際の投票では同時に複数馬へ投票可能か

#### E4: データ欠損の影響
- オッズ時系列が欠損しているレースの扱い
- フォールバック先が PIT 安全かつ現実的か

#### E5: 払戻の実現可能性
- 高配当の実現可能性 (JPRA 経由の投票制限等)
- 払戻計算の精度 (小数点の扱い)

## 進行ルール

1. **1イテレーション = 1〜3項目の完了** (深く調査する)
2. 全項目完了後に `docs/backtest-audit.md` を最終更新
3. 最後にバックテストを再実行し、修正前後のROI比較を行う:
   ```bash
   python scripts/run_backtest.py \
     --years 2023 2024 2025 \
     --train-window 4 \
     --ensemble \
     --report
   ```
4. 結果を `docs/backtest-audit.md` の「最終結果」セクションに記録

## 完了条件

全チェックリスト項目 (A1〜A16, B1〜B3, C1〜C5, D1〜D4, E1〜E5) が
✅ または ℹ️ になったら以下を出力:

```
<promise>AUDIT COMPLETE</promise>
```

---

## テンプレート: docs/backtest-audit.md

```markdown
# バックテスト PIT 監査レポート

監査開始日: YYYY-MM-DD
監査完了日: (進行中)

## 進捗サマリー

| カテゴリ | 完了 | 全項目 | ステータス |
|----------|------|--------|-----------|
| A. 特徴量 | 0/16 | 16 | 🔲 未着手 |
| B. データ  | 0/3  | 3  | 🔲 未着手 |
| C. 学習   | 0/5  | 5  | 🔲 未着手 |
| D. BT実行 | 0/4  | 4  | 🔲 未着手 |
| E. 現実性 | 0/5  | 5  | 🔲 未着手 |

## 詳細

### A. 特徴量 PIT 監査

#### A1: HorseHistoryFeatures
- **ステータス**: 🔲 未着手
- **調査内容**:
- **判定**:
- **修正内容**: (あれば)

(以下、各項目について同様のフォーマット)

## 発見事項サマリー

| ID | 重要度 | 項目 | 内容 | 修正状況 |
|----|--------|------|------|---------|

## 最終結果

### 修正前バックテスト (ベースライン)
- 実行日:
- コマンド:
- 結果:

### 修正後バックテスト
- 実行日:
- コマンド:
- 結果:

### 比較
- ROI変化:
- ベット数変化:
- 備考:
```
