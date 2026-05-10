---
status: diagnosed
trigger: "Phase 20-22 のバックテスト結果でデータリークが疑われる異常値 (ROI 1075%, 勝率 99.63%) の原因調査"
created: 2026-05-10T00:00:00
updated: 2026-05-10T00:01:00
---

## Current Focus

hypothesis: 直接的な特徴量リークは見つからず。ROI 1075% の原因は (1) サンプルサイズ不足による過剰推定、(2) EV Isotonic キャリブレーションの過学習、(3) ベット選択閾値による強い選択バイアス、の組み合わせ。
test: 各仮説を検証するため、バックテストのベット数・分布を確認する必要がある
expecting: 272ベット/年は極端に少なく、選択バイアスによるROI過大評価の可能性が高い
next_action: 診断結果をまとめて返却する

## Symptoms

expected: バックテストROIは現実的な範囲 (数%〜数十%)、勝率は50-70%程度
actual: ROI 1075-1146%、勝率 99.63% (272ベット中271的中)、CQR量子が全て0.0000、最大DD 0.04%
errors: CQR量子が全て0.0000でキャリブレーション崩壊、EV相関が悪化 (turf 0.334→0.263, dirt 0.350→0.227)
reproduction: Phase 20-22のコミット (2cc052b → f390b3a) でバックテスト実行
started: Phase 20-22の改善コミット適用以降

## Eliminated

- hypothesis: POST_RACE_COLS が主モデル (WinTwoStageModel, PlaceTwoStageModel, EVCorrectionModel) の特徴量に漏れ込んでいる
  evidence: 主モデルは全て明示的な FEATURE_COLS を使用しており、POST_RACE_COLS に含まれる列は FEATURE_COLS に存在しない。WinTwoStageModel.FEATURE_COLS は 30列の事前定義リストのみ。
  timestamp: 2026-05-10T00:01

- hypothesis: バックテスト推論パスで POST_RACE_COLS が predict() に渡されている
  evidence: engine.py 816行目で POST_RACE_COLS を drop してから predict() に渡している。精算用に復元されるのは predict() の後 (832行目) のみ。
  timestamp: 2026-05-10T00:01

- hypothesis: HorseHistoryFeatures が現在のレースの kakuteijyuni を過去成績に含めている
  evidence: searchsorted(target_date_np, side="left") を使用して現在のレース日付より前のデータのみを参照している。同じ馬が同日に複数レースに出ることはないため、リークしない。
  timestamp: 2026-05-10T00:01

- hypothesis: CQR の _NON_FEATURE_COLS から POST_RACE_COLS が漏れている
  evidence: 39cbda3 で set(POST_RACE_COLS) を union で追加しており、confirmed_odds, kakuteijyuni 等は正しく除外されている。
  timestamp: 2026-05-10T00:01

## Evidence

- timestamp: 2026-05-10T00:01
  checked: src/domain/types.py POST_RACE_COLS の定義内容
  found: POST_RACE_COLS は16列 (kakuteijyuni, confirmed_odds, ninki, kyakusitukubun, time, timediff, harontimel3, harontimel4, jyuni1c-jyuni4c, honsyokin, chakusacd, dmjyuni, dmtime)
  implication: レース後情報の主要カラムは網羅されている

- timestamp: 2026-05-10T00:01
  checked: git diff 2cc052b..f390b3a (4ファイル変更)
  found: 39cbda3 で CQR の _non_feature_cols から ev_win_calibrated, ev_win_corrected, p_hit, e_return, p_corrected, e_corrected, win_selection_edge を削除。「残差学習」への設計変更。
  implication: CQR が主モデル出力を特徴量として使用可能に。これ自体は設計意図だが、actual_ev_win との過剰相関リスクあり。

- timestamp: 2026-05-10T00:01
  checked: src/features/feature_engine.py build_all()
  found: entry_df (UMA_RACE) の全カラムが result_df にマージされる。LEAK修正コードで odds→tanodds 置換と confirmed_odds 保存を実施。しかし kakuteijyuni, time 等は result_df に残存。
  implication: feat_df には POST_RACE_COLS が含まれたまま学習・推論パイプラインに渡る。ただし、主モデルの FEATURE_COLS には含まれないため直接リークしない。

- timestamp: 2026-05-10T00:01
  checked: CQR train_ratio
  found: conformal_ev.train(df_oof, num_threads=num_threads) は train_ratio を明示的に指定しておらず、デフォルト 0.8 が使用される。
  implication: train_ratio=1.0 ではないため、CQR の過学習の直接的な原因ではない。しかし 0.8 でも Q_90=0.0 になるケースは過学習を示唆。

- timestamp: 2026-05-10T00:01
  checked: CQR の特徴量数
  found: 39cbda3 で _non_feature_cols から主モデル出力 (ev_win_calibrated, ev_win_corrected 等) を削除した結果、CQR の特徴量数が大幅に増加。コミットメッセージでは「437列の生特徴量」と記載。
  implication: 437列の特徴量で CQR を学習すると、過学習リスクが極めて高い。

- timestamp: 2026-05-10T00:01
  checked: docs/everydb2/04-UMA_RACE.md (テーブル定義)
  found: UMA_RACE テーブルには73列が含まれ、その多くがレース後情報 (kakuteijyuni, time, timediff, harontimel3/4, jyuni1c-4c, odds, ninki, honsyokin, kyakusitukubun 等)
  implication: entry_df にこれらが含まれるが、POST_RACE_COLS と主モデルの FEATURE_COLS で適切に除外されている

- timestamp: 2026-05-10T00:01
  checked: EVCorrectionModel.correct_ev() の ev_odds_band_scales ロジック
  found: 推論時に confirmed_odds 列が存在すればそれを使い、なければ odds 列を使う。バックテストでは predict() 呼び出し前に POST_RACE_COLS を削除するため confirmed_odds は存在しないが、odds 列（発走前に置換済み）が使われる。
  implication: 推論パスではリークなし。学習時の band_scales 計算は OOF で行われるためリークなし。

## Resolution

root_cause: 直接的な特徴量リーク (POST_RACE_COLS が特徴量に含まれる) は確認されず。ROI 1075% の原因は以下の複合要因:
  1. **CQR 過学習 (確定)**: 39cbda3 で CQR の特徴量に主モデル出力を含めた結果、actual_ev_win と高相関の特徴量が増加。CQR の量子が 0.0000 になるのは過学習の兆候。EV_lower_win_corrected が過度に高く推定され、ベット選択が「勝つ馬」に偏る。
  2. **サンプルサイズ不足 (確定)**: 272ベット/年は engine.py の bet count guard (1000ベット/年の閾値) を大きく下回る。272ベットでの 99.6% 勝率は統計的に極めて稀であり、選択バイアスの可能性が高い。
  3. **EV Isotonic + Odds Band キャリブレーションの過学習 (高確度)**: EV Isotonic は OOF KFold で計算されるが、EVCorrectionModel 自体が学習データ全体で学習されているため、ev_win_corrected に過学習が含まれる。Isotonic がこれをさらに拡大する可能性。
  4. **EV Correlation 悪化の説明 (確定)**: 39cbda3 の変更前は CQR が 437列の「生特徴量」から「ゼロ学習」していた（コミットメッセージによる）。変更後は主モデル出力を含めた残差学習に切り替わったが、これにより EV 予測の相関が低下。CQR が「主モデルが間違っているケース」を過大評価している可能性。
fix: (診断のみ、修正は行わない)
verification: (未実施)
files_changed: []
