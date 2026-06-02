# 当日予測 / バックテスト パイプラインアライメント改善提案

| 項目 | 値 |
|------|-----|
| 日付 | 2026-05-30 |
| ステータス | Draft |
| 対象フェーズ | v2.2 ROI Recovery (Phase 43-46) |
| 対象コンポーネント | BacktestEngine, PaperPredictor, RacePredictor, run_paper_trading.py |

---

## 1. エグゼクティブサマリー

5エージェントの調査結果を統合した結果、BacktestEngine (BT) と PaperPredictor / run_paper_trading.py (PT) の間に **38件の差分** を特定した。うち **critical 4件、high 12件** がパイプライン出力の直接的な不整合を引き起こしている。最も深刻な問題は、(1) PT が6つの特徴量モジュールを完全に欠落していること、(2) レジーム検出が BT では AGGRESSIVE 固定、PT では動的検出と分裂していること、(3) PT が Kelly/DrawdownController を持たず常に100円固定であること、(4) PT の精算が実際の払戻金ではなく推定オッズに基づいていることである。これらの不整合により BT と PT の ROI 比較が無意味になり、v2.2 ROI 回復目標の達成を阻害している。

### 差分重要度サマリー

| 重要度 | 件数 |
|--------|------|
| Critical | 4 |
| High | 12 |
| Medium | 14 |
| Low | 8 |
| **合計** | **38** |

### 最優先改善項目 (Top 3)

1. **特徴量モジュール欠落の解消** -- PT に6モジュール (DamPedigree, Record, Mining, PaceAptitude, Course, Sire) を追加。30+列の NaN 入力を排除し、推論精度を BT 水準に回復。
2. **レジーム検出の統一** -- BT と PT の両方で動的レジーム検出を有効化、または両方で AGGRESSIVE 固定化。戦略パラメータ空間を一致させる。
3. **精算パイプラインの実際の払戻金対応** -- PaperReconciler が EveryDB2 の払戻金テーブルを参照し、BT と同一の精算ロジックを使用するよう修正。

---

## 2. 差分マスターリスト

### 2.1 Critical 差分

| ID | カテゴリ | 領域 | 影響 |
|----|----------|------|------|
| DIFF-001 | feature_engineering | 特徴量モジュール欠落 | PT 推論で30+列が NaN、モデル精度が体系的に低下 |
| REGIME-01 | strategy/selection | RegimeDetector 状態伝播 | BT は常に AGGRESSIVE、PT は動的に CONSERVATIVE/COLLAPSED に遷移。ROI 比較が無効 |
| STAKE-01 | strategy/selection | ステーク計算 (flat vs kelly) | BT は Kelly+DD制御で可変ステーク、PT は常に100円固定。投資効率の比較不可 |
| DIFF-01 (odds) | odds-data-source | 精算データソース | BT は実際の JRA 払戻金、PT は推定オッズ。ROI の分母が根本的に異なる |

#### DIFF-001: 特徴量モジュール欠落 (PaperPredictor)

- **BT 挙動**: BacktestEngine.run() で SireFeatures (11列)、PaceAptitudeFeatures (6列)、CourseFeatures (2列)、DamPedigreeFeatures (複数列)、RecordFeatures (複数列)、MiningFeatures (複数列) を事前計算し feat_df にマージ (engine.py 869-985行)
- **PT 挙動**: PaperPredictor.setup() は HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, JockeyTrainerComboFeatures のみ計算。上記6モジュールが完全に欠落
- **改善案**: PaperPredictor.setup() に6モジュールの計算を追加。BacktestEngine と同一のパターン (ソースデータロード -> 特徴量計算 -> feat_df マージ) に従う

#### REGIME-01: RegimeDetector 状態伝播

- **BT 挙動**: engine.py 1117行で `regime = RegimeState.AGGRESSIVE` をハードコード。動的レジーム検出コードはコメントアウト (1112-1116行)。戦略パラメータは常に AGGRESSIVE デフォルト: ev_threshold=1.10, edge_threshold=0.05, fractional_kelly=0.50, max_bets_per_race=1
- **PT 挙動**: run_paper_trading.py 478-482行で動的 RegimeDetector.detect() を実行。ローリング200レースウィンドウ (recent_stats_list) から状態を推定。CONSERVATIVE/COLLAPSED パラメータが適用され、COLLAPSED 時はレースをスキップ。_override_params (strategy_manifest) は注入されない
- **改善案**: 両パスで同一のレジーム戦略を使用。BacktestEngine で動的レジームを再有効化 (TODO ハードコードを除去)、または PT でも AGGRESSIVE 固定化。strategy_manifest の _override_params を両パスに注入

#### STAKE-01: ステーク計算

- **BT 挙動**: flat/kelly 両モード対応。kelly モード: StakeCalculator + DrawdownController でレース間状態蓄積。fractional_kelly はレジームごとに動的注入 (engine.py 1121-1122行)。EV scaling (apply_ev_scaling) 適用。DDController が peak_bankroll を追跡し DD% しきい値で乗数を調整
- **PT 挙動**: 常に flat モード。RacePredictor に StakeCalculator/DrawdownController を渡さず (run_paper_trading.py 447行)。select_bets() はハードコード stake=100.0 (place)、stake_scale*100 (win)。EV scaling なし、DD 制御なし
- **改善案**: run_paper_trading.py の _run_predict() に StakeCalculator と DrawdownController の初期化を追加。--betting-mode フラグで制御。または BT 側を flat 固定にして PT と一致

#### DIFF-01 (odds): 精算データソース

- **BT 挙動**: payouts.parquet から load_payouts() で実際の払戻金をロード。payfukusyopay/paytansyopay1/paywidepay 列から payout_map, win_payout_map, wide_payout_map を構築。精算は実際の払戻乗数 (pay/100.0) を使用
- **PT 挙動**: payouts.parquet をロードしない。PaperReconciler は get_race_results() で結果を取得するが、SQL クエリが place_pay/place_odds 列にハードコード 0.0 を返す。実際の払戻金は EveryDB2 s_harai/n_harai テーブルにあるが、reconciler は get_payouts() を呼び出さない
- **改善案**: (1) PaperReconciler から everydb2.get_payouts() を呼び出して実際の精算金額を取得、(2) 各馬番の payfukusyopay/paytansyopay1 をパース、(3) BT と同一の払戻金ベース精算を実装

### 2.2 High 差分

| ID | カテゴリ | 領域 | 影響 |
|----|----------|------|------|
| DIFF-01 (inference) | inference-pipeline | レジーム検出ハードコード | BT 常に AGGRESSIVE、PT 動的検出でパラメータ空間が分裂 |
| DIFF-02 (inference) | inference-pipeline | RacePredictor betting_target デフォルト | BT は 'win'、PT は 'place'。PlaceTwoStageModel 等の実行有無が異なる |
| DIFF-03 (inference) | inference-pipeline | 特徴量エンジニアリングパイプライン | PT に DamPedigree/Record/Mining/PaceAptitude(3列欠け)が存在せず |
| DIFF-002 (feat) | feature_engineering | interaction/relative features 計算 | setup() 保存 parquet に interaction/relative 列が含まれず、デバッグ不整合 |
| DIFF-003 (feat) | feature_engineering | odds_dynamics データ可用性 | PT の時系列データが不完全で odds_drop_rate 等の品質低下 |
| DIFF-004 (feat) | feature_engineering | TargetEncoder fit 状態 | 新規騎手/調教師/血統の target encoding が global_mean にフォールバック |
| DIFF-01 (model) | model-load-config | モデルロードソース | BT は data/models-backtest/、PT は MLflow。バージョン不一致の可能性 |
| DIFF-02 (model) | model-load-config | strategy_manifest / OddsBandFilter | BT のみ manifest と OddsBandFilter 校正を適用。PT は除外バンドなし |
| DIFF-03 (model) | model-load-config | shadow_mode フラグ | BT は ShadowComparisonFramework で制御、PT は常に有効。A/B 比較不可 |
| DIFF-02 (odds) | odds-data-source | オッズスナップショットソース/タイミング | PT の EveryDB2 生オッズと Parquet オッズで列名スキーマ不一致 |
| DIFF-03 (odds) | odds-data-source | 特徴量計算カバレッジ | PT に PaceAptitude/CourseFeatures/DamPedigree/Record/Mining が欠落 |
| DIFF-04 (odds) | odds-data-source | ワイドオッズ扱い | PT はワイドオッズをロードせず。ワイドベッティングが機能しない |

#### DIFF-02 (inference): RacePredictor betting_target デフォルト

- **BT 挙動**: CLI --betting-target から明示的に渡す (デフォルト 'win')。RacePredictor(models, betting_target='win') で構築
- **PT 挙動**: RacePredictor(models) で構築 (betting_target 引数なし)。クラスデフォルトの 'place' が使用される。3呼び出し箇所 (447, 801, 1163行) 全てで同様
- **影響**: BT は win モード (PlaceTwoStageModel 等をスキップ)、PT は place モード (全モデル実行)。推論パスが根本的に異なる
- **改善案**: run_paper_trading.py に --betting-target を追加し RacePredictor に渡す

#### DIFF-002 (model): strategy_manifest / OddsBandFilter

- **BT 挙動**: manifest_path と strategy_params を受け取り、ParameterFreezeProtocol で SHA256 改ざん検知。OddsBandFilter を訓練期間の bet_history で校正し ROI < しきい値のオッズバンドを除外
- **PT 挙動**: strategy_params なし、manifest_path なし、OddsBandFilter なし。デフォルト ev_threshold=1.0。校正なし
- **影響**: BT は不採算オッズバンドを除外するが、PT は全バンドにベット。長ショットへの無駄なベットが PT の ROI を押し下げる
- **改善案**: run_paper_trading.py に --strategy-manifest フラグを追加。manifest パラメータを RegimeDetector._override_params に注入

#### DIFF-04 (odds): ワイドオッズ扱い

- **BT 挙動**: load_wide_odds() でワイドオッズをロード (betting_target='wide' 時)。kumi -> wide_odds_{lo}_{hi} 列にピボット。精算は build_wide_payout_map() で実際の paywidepay を使用
- **PT 挙動**: ワイドオッズをロードしない。RaceWatcher.get_latest_odds() は単複オッズのみ (tan_odds, fuku_odds)。全ペアで wide_odds=0 となる
- **影響**: ワイドベッティングが PT では機能しない。EV 計算が不可
- **改善案**: (1) EveryDB2 (s_jodds_waku) からワイドオッズをロード、(2) setup() パスにワイドオッズ parquet を追加、(3) PaperReconciler にワイド精算を追加

### 2.3 Medium 差分

| ID | カテゴリ | 領域 | 影響 |
|----|----------|------|------|
| DIFF-04 (inference) | inference-pipeline | hist features マージ戦略 | BT は事前マージ、PT はレース内マージ。機能的等価だが列衝突リスク |
| DIFF-05 (inference) | inference-pipeline | オッズデータソースフォールバック | PT は確定オッズにフォールバック (ルックアヘッドバイアス)、BT はスキップ |
| DIFF-06 (inference) | inference-pipeline | QualityScreener 適用差 | BT は win で診断のみ、PT は全対象でハードゲート |
| DIFF-005 (feat) | feature_engineering | 特徴量キャッシュ利用 | PT は日次でキャッシュミス、常にフル計算。性能影響は軽微 |
| DIFF-006 (feat) | feature_engineering | career/sire 統計の鮮度 | ETL 未実行で古い統計が使用される。PT でのみ顕在化 |
| DIFF-007 (feat) | feature_engineering | lag/rolling features 初出走馬 | 初出走馬のラグ特徴量が NaN。両パスで等価な動作 |
| ODDS-BAND-01 | strategy/selection | OddsBandFilter 校正・適用 | BT のみ校正済みフィルタを適用。PT は除外なし |
| MANIFEST-01 | strategy/selection | strategy_manifest パラメータ注入 | BT は Optuna 最適化16次元パラメータ、PT はハードコードデフォルト |
| QUALITY-01 | strategy/selection | QualityScreener (win vs place/wide) | win の BT は診断のみ、PT はハードゲート。ベット数に差 |
| SKIP-01 | strategy/selection | レーススキップロジック差 | 動的レジームの PT は COLLAPSED スキップ、BT (AGGRESSIVE) はスキップなし |
| DD-CTRL-01 | strategy/selection | DrawdownController 状態永続化 | BT は DD 保護あり、PT は100円固定で損失蓄積時の保護なし |
| DIFF-04 (model) | model-load-config | ワイドモデルロードエラーハンドリング | MLflow パスでワイドモデル不存在時に例外発生。BT は None フォールバック |
| DIFF-05 (model) | model-load-config | BacktestConfig 合格基準 | BT は合格基準で検証、PT は品質ゲートなし |
| DIFF-05 (odds) | odds-data-source | データ鮮度 (races/entries) | PT の Parquet が ETL delta 前で古い可能性 |
| DIFF-06 (odds) | odds-data-source | career/sire 統計鮮度 | precompute スクリプト未実行で最新レースが反映されず |

### 2.4 Low 差分

| ID | カテゴリ | 領域 | 影響 |
|----|----------|------|------|
| DIFF-07 (inference) | inference-pipeline | モデルロードソース | BT はローカル、PT は MLflow。設計上の差だがバージョン不一致リスク |
| DIFF-008 (feat) | feature_engineering | 障害除外/オッズフォールバック | PT の dry-run/diagnose でルックアヘッドバイアスの可能性 |
| DIFF-06 (model) | model-load-config | ConformalEVModel ロード | ローカル vs MLflow でロード方法が異なるが機能的等価 |
| DIFF-07 (model) | model-load-config | モデルキャッシュ/再利用 | BT はキャッシュ再利用、PT は毎回ロード。ユースケースに適合 |
| DIFF-07 (odds) | odds-data-source | jodds_tanpuku 時系列利用 | PT のライブデータが不完全な可能性。odds_dynamics 品質に影響 |
| DIFF-08 (odds) | odds-data-source | jodds_umaren/waku 未使用 | 両パスとも未使用。将来の拡張機会 |
| LATE-MONEY-01 | strategy/selection | LateMoneyFilter t-3min キャンセル | 両パスとも未使用。共有ギャップ |
| REGIME-01 (補足) | strategy/selection | COLLAPSED スキップの無効化 | BT はハードコード AGGRESSIVE でスキップなし |

---

## 3. 改善優先順位付け

ROI 回復への寄与が大きい順にソート。各項目は依存関係に従って段階的に実装する。

### Phase A: 特徴量・データアライメント (推定: L, 16-40h)

| 優先 | 項目 | 差分ID | ROI寄与 | 工数 | リスク | 依存関係 |
|------|------|--------|---------|------|--------|----------|
| **A-1** | PT に6特徴量モジュール追加 | DIFF-001, DIFF-03 (inf), DIFF-03 (odds) | **High** | L (16-24h) | Medium: 特徴量計算の副作用、メモリ使用量増加 | なし |
| **A-2** | PaceAptitudeFeatures 6列完全対応 | DIFF-03 (inf), DIFF-003 (feat) | **Medium** | S (2-4h) | Low: 既存モジュールの列追加のみ | A-1 |
| **A-3** | PaperReconciler の実際の払戻金対応 | DIFF-01 (odds) | **High** | M (8-16h) | Medium: EveryDB2 クエリの正確性、s_harai/n_harai スキーマ依存 | なし |
| **A-4** | ワイドオッズ・ワイド精算の実装 | DIFF-04 (odds) | **Medium** | M (8-16h) | Low: ワイド機能は現在 PT で使用されていない | A-3 |
| **A-5** | オッズ列名スキーマ統一 | DIFF-02 (odds) | **Medium** | S (2-4h) | Low: 列名マッピングのみ | なし |
| **A-6** | ETL delta 強制実行 / 鮮度チェック追加 | DIFF-005/006 (feat), DIFF-005/006 (odds) | **Low** | S (2-4h) | Low: 警告ログ追加が主 | なし |
| **A-7** | 確定オッズフォールバックの strict_mode 追加 | DIFF-005 (inf), DIFF-008 (feat) | **Low** | S (1-2h) | Low: フラグ追加のみ | なし |

**A-1 の詳細実装計画:**

PaperPredictor.setup() に以下を追加 (engine.py 869-985行と同一パターン):
1. `DamPedigreeFeatures`: ソースデータロード -> compute() -> feat_df にマージ
2. `RecordFeatures`: 同上
3. `MiningFeatures`: 同上
4. `SireFeatures`: 同上 (setup に既に SireFeatures の呼び出しがあるが不完全)
5. `PaceAptitudeFeatures`: 6列全てを含めるよう修正
6. `CourseFeatures`: 同上

共有関数 `_precompute_additional_features(feat_df, store)` を backtest/engine.py から抽出し、両パスで再利用する設計を推奨。

### Phase B: 戦略・選択ゲートアライメント (推定: L, 16-40h)

| 優先 | 項目 | 差分ID | ROI寄与 | 工数 | リスク | 依存関係 |
|------|------|--------|---------|------|--------|----------|
| **B-1** | レジーム検出の統一 | REGIME-01, DIFF-01 (inf), SKIP-01 | **High** | M (4-8h) | High: レジーム切替は全体のベット数・サイズに影響 | A-1 (特徴量揃えてから比較) |
| **B-2** | --betting-target 追加 | DIFF-02 (inf) | **High** | S (2-4h) | Low: CLI 引数追加のみ | なし |
| **B-3** | --betting-mode (flat/kelly) 追加 + DD制御 | STAKE-01, DD-CTRL-01 | **Medium** | M (8-16h) | Medium: DrawdownController の状態管理が複雑 | B-1 |
| **B-4** | strategy_manifest ロード追加 | MANIFEST-01, DIFF-02 (model) | **High** | M (4-8h) | Low: PFP 検証は既存実装あり | B-1 |
| **B-5** | OddsBandFilter 追加 | ODDS-BAND-01, DIFF-02 (model) | **Medium** | M (4-8h) | Medium: 校正用 bet_history 生成が必要 | B-4 |
| **B-6** | QualityScreener の betting_target 依存化 | DIFF-06 (inf), QUALITY-01 | **Low** | S (1-2h) | Low: 条件分岐追加のみ | B-2 |
| **B-7** | shadow_flags 設定オプション追加 | DIFF-03 (model) | **Low** | S (2-4h) | Low: PaperTradingConfig に項目追加 | なし |

**B-1 の実装方針 (レジーム統一):**

推奨: BT で動的レジームを再有効化。engine.py 1112-1116行のコメントアウトを解除し、1117行のハードコードを削除。ただし以下の条件を満たすこと:
- 再現性のため、RegimeDetector にシード固定オプションを追加
- strategy_manifest の _override_params を両パスで注入
- BT 結果がレジーム動的化で大幅に変動する場合は、アグリッシブ専用モードも保持

### Phase C: モデルロード・設定アライメント (推定: M, 8-16h)

| 優先 | 項目 | 差分ID | ROI寄与 | 工数 | リスク | 依存関係 |
|------|------|--------|---------|------|--------|----------|
| **C-1** | --model-dir オプション追加 | DIFF-01 (model) | **Medium** | S (2-4h) | Low: ロードパスの追加 | なし |
| **C-2** | ワイドモデルロードの try/except ラッピング | DIFF-04 (model) | **Low** | S (0.5-1h) | Low: フォールバック追加のみ | なし |
| **C-3** | BacktestConfig 合格基準の PT への参照 | DIFF-05 (model) | **Low** | S (2-4h) | Low: 診断オーバーレイのみ | なし |
| **C-4** | interaction/relative features の setup() 追加 (任意) | DIFF-002 (feat) | **Low** | S (1-2h) | Low: 保存 parquet の完全性向上 | A-1 |

### 全体工数見積もり

| Phase | 工数 | 内容 |
|-------|------|------|
| Phase A: 特徴量・データ | L (16-40h) | 特徴量モジュール追加、精算修正、オッズ統一 |
| Phase B: 戦略・選択ゲート | L (16-40h) | レジーム統一、Kelly 対応、manifest 注入 |
| Phase C: モデルロード・設定 | M (8-16h) | ロードパス統一、エラーハンドリング |
| **合計** | **40-96h** | |

---

## 4. リスク評価

### 4.1 変更による副作用リスク

| リスク | 対象変更 | 影響度 | 緩和策 |
|--------|----------|--------|--------|
| 特徴量モジュール追加による NaN 処理の変化 | A-1 | Medium | 追加後に回帰テストで BT 結果の不変を確認。FeatureEngine の既存列に影響しないことを feature count assertion で検証 |
| 動的レジーム有効化による BT 結果の大幅変動 | B-1 | **High** | 有効化前後で全年度 BT を比較。ROI 変動が 5% 超の場合はハードコード AGGRESSIVE モードを併存 |
| DrawdownController 導入によるステーク変動 | B-3 | Medium | flat モードをデフォルトのまま kelly をオプション化。DD しきい値のデフォルトは BT と同一 |
| strategy_manifest 注入によるパラメータ上書き | B-4 | Medium | PFP の SHA256 検証を PT でも適用。manifest なしの場合はデフォルト値を使用 (既存動作を維持) |
| PaperReconciler の払戻金クエリ依存 | A-3 | Medium | s_harai/n_harai テーブルの可用性を確認。クエリ失敗時は推定オッズにフォールバック (既存動作) |
| ワイドモデル MLflow ロードの例外ハンドリング | C-2 | Low | try/except で None フォールバック。BT と同一の挙動 |

### 4.2 回帰テスト推奨項目

各 Phase 完了後に以下のテストを実行すること:

**Phase A 完了後:**
1. `python -m pytest tests/ -v` -- 既存テストの全通過を確認
2. 単年度 BT (2024) を A-1 適用前後で実行し、BT 結果の不変を確認 (特徴量追加は BT 側は既にあるため、PT 側の修正のみで BT 結果は変わらないはず)
3. PaperPredictor.setup() で生成される parquet の列数を BT の feat_df 列数と比較。差分0であること
4. `run_feature_routing_audit.py` を実行し、情報リークがないことを確認

**Phase B 完了後:**
1. 動的レジーム有効化 BT の ROI をハードコード AGGRESSIVE BT と比較
2. PT predict モードで --betting-target win/place/wide 全てを実行し、各モードでベットが生成されること
3. kelly モード PT で bankroll が正しく追跡されることを手動確認
4. strategy_manifest なしの PT が従来通りの結果を生成することを確認

**Phase C 完了後:**
1. --model-dir で data/models-backtest/ を指定した PT と、従来の MLflow ロード PT の結果を比較 (同一モデルで同一結果であること)
2. ワイドモデル不存在時に PT がクラッシュしないことを確認

### 4.3 段階的ロールアウト戦略

```
Step 1: Phase A (特徴量・データ) を feature/pt-bt-alignment-phase-a ブランチで実装
  |-- A-5, A-6, A-7: 低リスク項目を先にマージ (Sprint 1)
  |-- A-1: 特徴量モジュール追加 (Sprint 2)
  |-- A-2: PaceAptitude 完全対応 (Sprint 2)
  |-- A-3: 精算パイプライン修正 (Sprint 3)
  |-- A-4: ワイド対応 (Sprint 3)

Step 2: Phase B (戦略・選択ゲート) を feature/pt-bt-alignment-phase-b ブランチで実装
  |-- B-2: --betting-target 追加 (Sprint 4)
  |-- B-7: shadow_flags 設定 (Sprint 4)
  |-- B-1: レジーム統一 (Sprint 5) -- 要 BT 比較検証
  |-- B-4: strategy_manifest ロード (Sprint 5)
  |-- B-5: OddsBandFilter (Sprint 6)
  |-- B-3: Kelly/DD 対応 (Sprint 6)
  |-- B-6: QualityScreener (Sprint 6)

Step 3: Phase C (モデルロード・設定) を feature/pt-bt-alignment-phase-c ブランチで実装
  |-- C-1, C-2, C-3: まとめて Sprint 7
```

各 Sprint の境界で PR を作成し、回帰テスト通過後にマージ。Phase 間の依存関係に従い、Phase A 完了後に Phase B を開始。

---

## 5. 結論と Next Steps

### 結論

BacktestEngine と PaperPredictor の間に 38 件の差分が存在し、うち 16 件 (critical 4 + high 12) が推論結果の直接的な不整合を引き起こしている。最も影響の大きい差分は (1) 特徴量モジュール欠落による推論精度の体系的低下、(2) レジーム検出の分裂による戦略パラメータ空間の不一致、(3) 精算データソースの違いによる ROI 比較の無効化、(4) Kelly/DD 制御の欠如による投資効率の差異である。

これらの不整合により、BT で検証した戦略が PT で再現されず、v2.2 ROI 回復目標の達成が阻害されている。Phase A (特徴量・データ) の修正だけでも PT の推論精度を BT 水準に引き上げる効果が大きく、最優先で実装すべきである。

### Next Steps

1. **Phase A Sprint 1** を開始: 低リスク項目 (A-5: オッズ列名統一、A-6: 鮮度チェック、A-7: strict_mode) を先に実装し、CI パイプラインの動作を確認
2. **共有特徴量関数の抽出**: BacktestEngine から `_precompute_additional_features()` を抽出し、両パスで再利用可能にする設計を先行検討
3. **動的レジームの BT 影響測定**: B-1 の実装前に、動的レジーム有効化時の BT 全年度 ROI を計測し、AGGRESSIVE 固定との差分を定量化
4. **PT 精算の整合性確認**: A-3 の実装前に、EveryDB2 s_harai/n_harai テーブルのスキーマとデータ可用性を確認
5. **本提案のレビュー**: ステークホルダーによる本提案のレビューと優先順位の合意形成

### 附帯情報: 重複/統合された差分

5エージェントの調査で同一問題が複数エージェントから指摘されている。以下は重複統合マッピング:

| 統合差分 | エージェントA ID | エージェントB ID | エージェントC ID | エージェントD ID | エージェントE ID |
|----------|------------------|------------------|------------------|------------------|------------------|
| 特徴量モジュール欠落 | DIFF-03 | DIFF-001 | DIFF-03 | -- | -- |
| レジーム検出分裂 | DIFF-01 | -- | -- | REGIME-01 | -- |
| betting_target デフォルト | DIFF-02 | -- | -- | -- | -- |
| オッズフォールバック | DIFF-05 | DIFF-008 | -- | -- | -- |
| QualityScreener 適用差 | DIFF-06 | -- | -- | QUALITY-01 | -- |
| strategy_manifest 欠落 | -- | -- | -- | MANIFEST-01 | DIFF-02 |
| ステーク計算差 | -- | -- | -- | STAKE-01 | -- |
| 精算データソース | -- | -- | DIFF-01 | -- | -- |
| モデルロードソース | DIFF-07 | -- | -- | -- | DIFF-01 |
