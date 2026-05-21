# bt_2024_horse_features.parquet 包括分析レポート

- **作成日**: 2026-05-21
- **目的**: バックテスト用特徴量Parquet (`data/backtest/bt_2024_horse_features.parquet`) に含まれる全494カラムについて、everyDB2仕様書との照合、NaNの意味分類、リーク検査、予測時点別利用可否を分析し、LightGBM投入前の判断材料を整理する。コード修正・特徴量修正・前処理実装は行わず、分析・分類・根拠提示に集中した。
- **対象データ**: 45,827行 × 494列、2024年1月〜12月、3,329レース、11,538頭
- **分析手法**: 8観点のサブエージェントを並列実行し、結果を統合

---

## Insight: 分析の統合方針

7つの独立した分析結果を統合する際、最も保守的な判断を採用。あるエージェントが「安全」と判定し別のエージェントが「要調査」と判定した場合、「要調査」を優先。リーク疑いが少しでも残る列は `leakage_or_drop_candidate` に分類。

---

## A. 統合サマリー

### データ品質評価

- **45,827行 × 494列**、主キー `(race_id, kettonum)` 重複ゼロ、2024年1月〜12月を均等カバー
- **高品質**: 正解ラベル(kakuteijyuni, confirmed_odds)は0%欠損、Infなし、主キー完全一意
- **問題点**: 18列が100%NaN、20列が定数、sire_distance_wrに重大な異常値(max=118.27)

### NaN処理に関する分析結果

- **自然なNaN**が大半(~200列): 初出走(11%)、過去データ不足(21-30%)、ワイドオッズ頭数依存(153列)
- **バグ疑いNaN**: `weight_change_zone`/`weight_change_ratio`(100%NaN) — 列名不一致(`zogen_sa` vs `zogensa`)が原因の可能性
- **設計上のNaN**: `deviation_rank`/`deviation_zscore`(100%NaN) — FEATURE_COLSに宣言されているがparquet上は未計算

### everyDB2仕様書との照合結果

- 全59テーブル(~2,000フィールド)と照合完了
- レース前/後の区別はDataKubunで段階管理: 1(木曜)→2(金土)→3-6(速報)→7(確定)
- **POST_RACE情報の特定完了**: 着順、タイム、上がり、コーナー順位、確定オッズ/人気、獲得賞金、脚質判定
- 派生特徴量(~23列)は仕様書に直接の記載なし、`src/features/` のコードで定義

### リーク観点の危険度

- **現在のリークリスク: 低** — モデルはホワイトリスト形式のFEATURE_COLSを採用しており、POST_RACE情報は入力されない
- **ただし**: POST_RACE_COLS除外リストに9列の欠落(nyusenjyuni, fukasyokin等) — ホワイトリストが二重安全策として機能しているが、防御的プログラミングとして追加推奨
- **オッズ系**: `odds`/`tanodds`は投票締切時点(レース前)で確定する最終オッズ。意図的な利用だが、「前日予測」では利用不可

### 前日予測で使える可能性が高い特徴量

- レース条件(距離、トラック、グレード、賞金構造、競走条件)
- 出走馬属性(性齢、血統、所属、負担重量、騎手/調教師コード)
- 過去成績集計(勝率、平均着順、ハロンタイム、コース別成績等)
- 血統特徴量(sire_wr, bms_wr, blood_*_wr)
- 騎手/調教師統計(jockey_wr_*, trainer_wr_*, jt_combo_*)
- **注意**: 天候(tenkocd)、馬場状態(sibababacd/dirtbabacd)は当日朝の発表のため、前日予測では前回開催時の値または欠損になる可能性

### 当日発走前予測で使える可能性が高い特徴量

- 上記すべてに加えて:
- 天候・馬場状態(当日発表)
- 馬体重・増減(直前発表)
- オッズ時系列特徴量(odds_drop_rate, odds_velocity等) — 発走10-60分前のスナップショット必要
- ワイドオッズ(wide_odds_*) — 投票締切時確定
- マイニング予想(DataKubun=3の直前版)

### 要調査カラムの概要

- **sire_distance_wr**: max=118.27で勝率(0-1)の範囲外。計算バグの可能性高
- **weight_change_zone/ratio**: 100%NaN。zogensa(10.8%NaN)が存在するのに計算されていない
- **FEATURE_COLSに宣言されているがデータに存在しないカラム**: 20列以上。実装と設計の乖離
- **pace_aptitude(69.4%), win_dominance(70.6%)**: 欠損率が極めて高い。feature importanceが不安定になるリスク

### Insight: 統合分析の重要な結論

1. **現在のリークリスクは低い** — ホワイトリスト方式のFEATURE_COLSが確実に機能している。ただしPOST_RACE_COLS除外リストにギャップあり。
2. **sire_distance_wrの異常値が最も緊急性が高い** — max=118.27は計算バグの可能性が高く、修正によりモデル精度が改善する可能性がある。
3. **「バックテストROI = 直前予測ROI」** — 前日予測の性能は完全に未知。実運用設計においてこの違いを明確に意識する必要がある。
4. **FEATURE_COLSと実際のデータの乖離が20列以上** — 実装と設計の同期が必要。

---

## B. everyDB2仕様マッピング表 (主要カラム)

| parquet_column | everyDB2_table | everyDB2_column | doc_file | doc_description | value_domain | mapping_confidence | notes |
|---|---|---|---|---|---|---|---|
| race_id | 複合キー | Year+MonthDay+JyoCD+Kaiji+Nichiji+RaceNum | 03-RACE.md | レース識別子 | 16桁数値文字列 | high | 派生キー |
| kettonum | UMA_RACE (SE) | KettoNum | 04-UMA_RACE.md | 血統登録番号 | 生年4桁+品種1桁+番号5桁 | high | 馬の個体識別 |
| race_date | RACE (RA) | Year+MonthDay | 03-RACE.md | 開催年月日 | datetime64 | high | 時系列キー |
| kakuteijyuni | UMA_RACE (SE) | KakuteiJyuni | 04-UMA_RACE.md | 確定着順 | 1-18, POST_RACE | high | **TARGET** |
| confirmed_odds | UMA_RACE (SE) | Odds | 04-UMA_RACE.md | 確定単勝オッズ | 0.1-999.9, POST_RACE | high | **TARGET用** |
| odds | UMA_RACE (SE) | Odds | 04-UMA_RACE.md | 単勝オッズ(=tanodds上書き) | 0.1-999.9 | high | 投票締切時確定 |
| tanodds | ODDS_TANPUKU | 単勝オッズ | 15-ODDS_TANPUKU.md | 単勝オッズスナップショット | 0.1-999.9 | high | 最終版=投票締切 |
| tanninki | ODDS_TANPUKU | 単勝人気順 | 15-ODDS_TANPUKU.md | 単勝人気順位 | 1-18 | high | 投票締切時確定 |
| fukuoddslow | ODDS_TANPUKU | 複勝最低オッズ | 15-ODDS_TANPUKU.md | 複勝オッズ下限 | 0.1-999.9 | high | 投票締切時確定 |
| wide_odds_X_Y | ODDS_WIDE | ワイドオッズ | 20-ODDS_WIDE.md | ワイドオッズ(組合せ) | 0.1-999.9 | high | 153列、頭数依存NaN |
| jyocd | RACE (RA) | JyoCD | 03-RACE.md | 競馬場コード | CODE2001: 01-10 | high | 10場 |
| trackcd | RACE (RA) | TrackCD | 03-RACE.md | トラックコード | CODE2009: 10-29 | high | 芝/ダート/障害 |
| kyori | RACE (RA) | Kyori | 03-RACE.md | 距離 | 単位:m | high | |
| tenkocd | RACE (RA) | TenkoCD | 03-RACE.md | 天候コード | CODE2011: 1-4 | high | 当日発表 |
| sibababacd | RACE (RA) | SibaBabaCD | 03-RACE.md | 芝馬場状態 | CODE2010: 0-4 | high | 当日発表 |
| dirtbabacd | RACE (RA) | DirtBabaCD | 03-RACE.md | ダート馬場状態 | CODE2010: 0-4 | high | 当日発表 |
| gradecd | TOKU_RACE (TK) | GradeCD | 01-TOKU_RACE.md | グレードコード | CODE2003: A-E," | high | |
| syubetucd | TOKU_RACE (TK) | SyubetuCD | 01-TOKU_RACE.md | 競走種別コード | CODE2005 | high | |
| bataijyu | UMA_RACE (SE) | BaTaijyu | 04-UMA_RACE.md | 馬体重 | 2-998kg, 999=不能, 000=取消 | high | 直前発表 |
| zogensa | UMA_RACE (SE) | ZogenSa | 04-UMA_RACE.md | 馬体重増減差 | 0-998kg | high | 直前発表 |
| futan | UMA_RACE (SE) | Futan | 04-UMA_RACE.md | 負担重量 | 0.1kg単位 | high | レース前確定 |
| sexcd | UMA_RACE (SE) | SexCD | 04-UMA_RACE.md | 性別コード | CODE2202: 1-3 | high | |
| barei | UMA_RACE (SE) | Barei | 04-UMA_RACE.md | 馬齢 | 2-12(満年齢) | high | |
| kisyucode | UMA_RACE (SE) | KisyuCode | 04-UMA_RACE.md | 騎手コード | 5桁コード | high | |
| chokyosicode | UMA_RACE (SE) | ChokyosiCode | 04-UMA_RACE.md | 調教師コード | 5桁コード | high | |
| laptime1-25 | RACE (RA) | LapTime1-25 | 03-RACE.md | ラップタイム(先頭馬) | 99.9秒, POST_RACE | high | **LEAKAGE** |
| corner1-4 | RACE (RA) | Corner1-4 | 03-RACE.md | コーナー通過順位 | POST_RACE | high | **LEAKAGE** |
| hassotime | RACE (RA) | HassoTime | 03-RACE.md | 発走時刻 | hhmm | high | レース前確定 |
| harontimes3/4 | RACE (RA) | HaronTimeS3/S4 | 03-RACE.md | 前3/4ハロン合計 | POST_RACE | high | **LEAKAGE** |
| honsyokin1-7 | RACE (RA) | Honsyokin1-7 | 03-RACE.md | 本賞金構造(配当テーブル) | 百円単位 | high | **PRE-RACE** |
| fukasyokin | UMA_RACE (SE) | Fukasyokin | 04-UMA_RACE.md | 獲得付加賞金 | 百円単位, POST_RACE | high | **LEAKAGE** |
| ijyocd | UMA_RACE (SE) | IJyoCD | 04-UMA_RACE.md | 異常区分コード | CODE2101, POST_RACE | high | **LEAKAGE** |
| kyakusitukubun_cd | UMA_RACE (SE) | KyakusituKubun | 04-UMA_RACE.md | 脚質判定 | 1-4, POST_RACE | high | **LEAKAGE** (当レース) |
| surface | 派生 | trackcd→芝/ダート | schema.py | 芝/ダート区分 | turf/dirt | high | |
| distance_bin | 派生 | kyori+surface | feature_engine.py | 距離カテゴリ | sprint/mile/intermediate/long | high | |
| grade_code | 派生 | gradecd→X変換 | feature_engine.py | グレード | A/B/C/L/E/X | high | |
| class_level_current | 派生 | 複数条件 | feature_engine.py | クラスレベル(数値) | 0-8 | high | ordinal encoding済 |
| sire_wr | 派生 | 種牡馬過去成績 | feature_engine.py | 種牡馬勝率 | 0-1 | high | 過去レースのみ |
| sire_distance_wr | 派生 | 種牡馬距離別 | feature_engine.py | 種牡馬距離別勝率 | **0-118.27** | **investigate** | **異常値: max>1** |
| blood_*_wr | 派生 | 血統成績 | feature_engine.py | 血統別勝率 | 0-1 | high | |
| jockey_wr_* | 派生 | 騎手過去成績 | feature_engine.py | 騎手各種勝率 | 0-1 | high | 過去レースのみ |
| trainer_wr_* | 派生 | 調教師過去成績 | feature_engine.py | 調教師各種勝率 | 0-1 | high | 過去レースのみ |
| days_since_last_race | 派生 | 前走からの日数 | feature_engine.py | 休養期間 | 日数 | high | 初出走=NaN |
| form_trend | 派生 | 過走着順推移 | feature_engine.py | フォーム推移 | 連続値 | high | 過去レースのみ |
| ev_win | モデル出力 | - | モデル予測 | 勝利EV | 連続値 | high | 予測値 |
| p_win_pred | モデル出力 | - | モデル予測 | 勝利確率予測 | 0-1 | high | 予測値 |

---

## C. 欠損率・NaN分類テーブル

### 欠損率グループ別集計

| グループ | 列数 | 主な特徴 |
|---------|------|---------|
| 100% (ALL NaN) | 18 | 未使用の生カラム、未計算の派生特徴量 |
| 90%+ | 17 | 18頭立てワイドオッズ組合せ |
| 70-90% | 17 | 17頭立てワイドオッズ + win_dominance |
| 30-70% | 52 | ワイドオッズ(14-16頭)、ペース/血統、datakubun |
| 5-30% | 86 | ワイドオッズ(10-13頭)、履歴/フォーム、血統、jt_combo |
| 0-5% | 65 | ワイドオッズ(2-9頭)、オッズ派生、ジョッキー/調教師統計 |
| 0% | 239 | レースメタ、出走馬基本、予測値、市場指標 |

### 100% NaN (18列)

| column | dtype | missing_rate | everyDB2_doc_meaning | inferred_nan_type | suggested_handling | reason | confidence |
|---|---|---|---|---|---|---|---|
| jyuni1-4 | float64 | 100% | 03-RACE.md 通過順位(文字列由来) | D(リーク) | leakage_or_drop_candidate | レース後情報、値なし | high |
| chakusacdp/pp | float64 | 100% | 04-UMA_RACE.md 着差コード | D(リーク) | leakage_or_drop_candidate | レース後情報、値なし | high |
| coursekubuncd/before | float64 | 100% | 01-TOKU_RACE.md コース区分 | G(保留) | leakage_or_drop_candidate | 未使用生カラム | high |
| fukusyoku | float64 | 100% | 04-UMA_RACE.md 復色標示 | G(保留) | leakage_or_drop_candidate | 未使用生カラム | high |
| bamei1-3 | float64 | 100% | 04-UMA_RACE.md 同着相手馬名 | G(保留) | leakage_or_drop_candidate | 文字列由来NaN | high |
| reserved1 | float64 | 100% | ETL予約領域 | G(保留) | leakage_or_drop_candidate | 未使用 | high |
| zogenfugo | object | 100% | 04-UMA_RACE.md 増減符号 | G(保留) | leakage_or_drop_candidate | zogensa(数値)が代替 | high |
| deviation_rank/zscore | float64 | 100% | 派生特徴量(未計算) | B(バグ疑い) | investigate | FEATURE_COLSに宣言、parquet上未計算 | high |
| weight_change_zone/ratio | float64 | 100% | 派生特徴量(未計算) | B(バグ疑い) | investigate | zogensa存在するのに計算されていない | medium |

### 高欠損率 (30%以上、wide_odds以外)

| column | dtype | missing_rate | inferred_nan_type | suggested_handling | reason | confidence |
|---|---|---|---|---|---|---|
| win_dominance | float64 | 70.6% | F(経験不足) | keep_nan_candidate | 過去出走ゼロの馬で未定義 | high |
| pace_aptitude | float64 | 69.4% | F(経験不足) | keep_nan_candidate | ペース分析に十分な過去レースがない | high |
| kigocd | float64 | 57.5% | E(カテゴリ欠損) | unknown_category_candidate | 該当条件なし(通常レース) | high |
| blood_distance_wr | float64 | 52.1% | F(経験不足) | keep_nan_candidate | 血統の距離別データ不足 | high |
| blood_surface_wr | float64 | 32.3% | F(経験不足) | keep_nan_candidate | 血統の芝/ダートデータ不足 | high |
| harontime_late_trend | float64 | 43.9% | A(自然) | keep_nan_candidate | 出走3走未満で計算不能 | high |
| class_adj_formetric | float64 | 35.0% | A(自然) | keep_nan_candidate | 過去出走不足 | high |
| haron_zscore_trend | float64 | 30.0% | A(自然) | keep_nan_candidate | 出走不足 | high |
| form_peak_flag | float64 | 30.0% | A(自然) | keep_nan_candidate | 出走不足 | high |
| freshness_score | float64 | 30.0% | A(自然) | keep_nan_candidate | 出走不足 | high |

### 中欠損率 (5-30%、wide_odds以外)

| column | dtype | missing_rate | inferred_nan_type | suggested_handling | reason | confidence |
|---|---|---|---|---|---|---|
| blood_prize_log | float64 | 27.1% | F(経験不足) | keep_nan_candidate | 血統情報なし | high |
| blood_condition_wr | float64 | 21.8% | F(経験不足) | keep_nan_candidate | 血統馬場データ不足 | high |
| weight_zscore | float64 | 21.2% | A(自然) | keep_nan_candidate | 初出走で母集団不足 | high |
| form_trend | float64 | 21.2% | A(自然) | keep_nan_candidate | 出走不足 | high |
| form_consistency | float64 | 21.2% | A(自然) | keep_nan_candidate | 出走不足 | high |
| actual_pace_fit | float64 | 21.2% | F(経験不足) | keep_nan_candidate | 経験不足 | high |
| front/mid/closing_pace_wr(6列) | float64 | ~21% | F(経験不足) | keep_nan_candidate | ペース経験不足 | high |
| days_since_last_race | float64 | 11.0% | A(自然) | missing_flag_candidate | 初出走=NaN(情報として意味あり) | high |
| class_drop_bounce | float64 | 11.0% | A(自然) | keep_nan_candidate | 前走なし | high |
| surface_change | float64 | 11.0% | A(自然) | keep_nan_candidate | 前走なし | high |
| class_move | float64 | 11.0% | A(自然) | keep_nan_candidate | 前走なし | high |
| rest_category | float64 | 11.0% | A(自然) | keep_nan_candidate | 前走なし | high |
| distance_change | float64 | 11.0% | A(自然) | keep_nan_candidate | 前走なし | high |
| blinker_change | float64 | 11.0% | A(自然) | zero_fill_candidate | 初出走時NaN→0(変更なし)でOK | high |
| zogensa | float64 | 10.8% | A(自然) | keep_nan_candidate | 初出走(前回体重なし) | high |
| course_wr | float64 | 10.9% | F(経験不足) | keep_nan_candidate | コース経験なし | high |
| blood_total_wr | float64 | 10.8% | F(経験不足) | keep_nan_candidate | 血統情報なし | high |
| jt_combo_place_rate等(4列) | float64 | 5.9% | F(経験不足) | keep_nan_candidate | コンビ経験なし | high |
| is_nar_transfer | float64 | 11.1% | A(自然) | zero_fill_candidate | JRA専属馬はNaN→0 | high |
| nar_recent_ratio | float64 | 11.0% | A(自然) | zero_fill_candidate | JRA専属馬はNaN→0 | high |

---

## D. リーク疑いテーブル

### 高リスク — レース後の結果 (モデルによって使用されないことを確認済み)

| column | dtype | missing_rate | everyDB2_doc_meaning | leak_risk | leakage_type | reason | confidence |
|---|---|---|---|---|---|---|---|
| kakuteijyuni | int64 | 0% | 04-SE 確定着順 | **high** | post_race_result | TARGET列。POST_RACE_COLSで除外済み | high |
| confirmed_odds | float64 | 0% | 04-SE 確定オッズ | **high** | final_odds_or_popularity | TARGET用。POST_RACE_COLSで除外済み | high |
| nyusenjyuni | int64 | 0% | 04-SE 入線順位 | **high** | post_race_result | kakuteijyuniと100%同一。POST_RACE_COLS未登録 | high |
| fukasyokin | int64 | 0% | 04-SE 獲得付加賞金 | **high** | payout_or_return | 当レース獲得賞金。POST_RACE_COLS未登録 | high |
| laptime1-18 | int64 | 0% | 03-RA ラップタイム | **high** | post_race_result | レース後確定。POST_RACE_COLS登録済み | high |
| harontimes3/4 | int64 | 0% | 03-RA 前半ハロン | **high** | post_race_result | レース後確定 | high |
| corner1-4 | int64 | 0-58% | 03-RA コーナー通過順位 | **high** | post_race_result | レース後確定 | high |
| syukaisu1-4 | int64 | 0-58% | 03-RA 周回数 | **high** | post_race_result | レース後確定 | high |
| ijyocd | int64 | 0% | 04-SE 異常区分 | **high** | post_race_result | レース後確定 | high |
| dochakukubun | int64 | 0% | 04-SE 同着区分 | **high** | same_race_outcome | レース後確定 | high |
| dochakutosu | int64 | 0% | 04-SE 同着頭数 | **high** | same_race_outcome | レース後確定 | high |
| nyusentosu | int64 | 0% | 03-RA 入線頭数 | **high** | post_race_result | レース後確定 | medium |
| recordupkubun | int64 | 0% | 03-RA レコード更新区分 | **high** | post_race_result | レース後確定 | medium |

### 中リスク — オッズ/人気 (レース前利用可能だが前日予測では不可)

| column | dtype | missing_rate | leak_risk | leakage_type | reason | confidence |
|---|---|---|---|---|---|---|
| odds | float64 | 0% | **medium** | final_odds_or_popularity | SE.Odds=tanoddsと100%一致。投票締切時確定(レース前) | high |
| tanodds | float64 | 0.2% | **medium** | final_odds_or_popularity | 最終オッズスナップショット。投票締切時確定 | high |
| tanninki | float64 | 0.2% | **medium** | final_odds_or_popularity | 最終単勝人気順。投票締切時確定 | high |
| fukuoddslow | float64 | 0.2% | **medium** | final_odds_or_popularity | 最終複勝オッズ下限。投票締切時確定 | high |
| odds_rank | float64 | 0% | **medium** | final_odds_or_popularity | odds列から計算。投票締切時確定 | high |
| popularity_rank | float64 | 0% | **medium** | final_odds_or_popularity | tanoddsから計算。投票締切時確定 | high |

### 重要な補足

- `honsyokin1-7`(RAテーブル)は**配当テーブル**(レース設定時に決定)であり、獲得賞金ではない。PRE-RACE情報。リークではない。
- `fukasyokin1-5`(RAテーブル)も同様に配当テーブル。PRE-RACE情報。リークではない。
- `odds`/`tanodds`は投票締切時(通常発走10-20分前)で確定。JRAのプール方式ではレース開始後に変更されない。意図的かつ標準的な利用。

---

## E. 予測時点別利用可否テーブル (主要カラム)

| column | available_for_day_before | available_for_pre_race | available_for_post_race_analysis | reason | confidence |
|---|---|---|---|---|---|
| kakuteijyuni | no | no | yes | 確定着順=TARGET | high |
| confirmed_odds | no | no | yes | 確定オッズ=TARGET用 | high |
| odds/tanodds/tanninki | **conditional** | yes | yes | 最終オッズ=投票締切時(発走10-20分前)。前日には中間オッズのみ | high |
| fukuoddslow | **conditional** | yes | yes | 複勝オッズ。同上 | high |
| wide_odds_X_Y (153列) | **conditional** | yes | yes | ワイドオッズ。投票締切時確定 | high |
| odds_drop_rate_* | no | yes | yes | 発走10-60分前スナップショット必要 | high |
| odds_velocity/volatility | no | yes | yes | オッズ時系列全体が必要 | high |
| tenkocd | **conditional** | yes | yes | 当日朝発表。前日は不明 or 前回開催時の値 | high |
| sibababacd/dirtbabacd | **conditional** | yes | yes | 当日朝発表。変更可能性あり | high |
| bataijyu/zogensa | no | yes | yes | 直前(約30分前)計量発表 | high |
| track_condition_code | **conditional** | yes | yes | tenkocd等から統合。当日情報 | high |
| jyocd/trackcd/kyori/gradecd | yes | yes | yes | レース条件。出馬表発表時に確定 | high |
| sexcd/barei/futan/blinker | yes | yes | yes | 出走馬基本情報。出馬表に記載 | high |
| kisyucode/chokyosicode | yes | yes | yes | 出馬表に記載 | high |
| sire_wr/blood_*_wr | yes | yes | yes | 過去データのみで計算 | high |
| jockey_wr_*/trainer_wr_* | yes | yes | yes | 過去データのみで計算 | high |
| days_since_last_race | yes | yes | yes | 前走出走日ベース。初出走=NaN | high |
| form_trend/form_consistency | yes | yes | yes | 過去データのみで計算 | high |
| pace_aptitude | yes | yes | yes | 過去データのみで計算(高欠損) | high |
| distance_bin/surface | yes | yes | yes | レース条件から派生 | high |
| class_level_current | yes | yes | yes | レース条件から計算 | high |
| kyakusitukubun_cd | **conditional** | **conditional** | yes | **当レース**脚質はPOST_RACE。**過去レース**の脚質傾向はPRE-RACE | high |
| laptime1-18 | no | no | yes | レース後確定 | high |
| corner1-4 | no | no | yes | レース後確定 | high |
| harontimes3/4 | no | no | yes | レース後確定 | high |
| honsyokin1-7 | yes | yes | yes | 配当テーブル(獲得額ではない) | high |
| fukasyokin(無番号) | no | no | yes | 獲得付加賞金(POST_RACE) | high |
| nyusenjyuni/dochakukubun等 | no | no | yes | POST_RACE | high |

---

## F. カテゴリ・コード値テーブル (LightGBM投入候補)

| column | dtype | unique_count | sample_values | everyDB2_doc_meaning | code_domain | categorical_type | suggested_encoding | missing_handling | lightgbm_categorical | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| jyocd | float64 | 10 | [8,5,6,...] | CODE2001 競馬場 | 01-10 | nominal | category_dtype_candidate | N/A(0%) | **yes** | 10場。コース特性大 |
| trackcd | int64 | 8 | [10,11,23,...] | CODE2009 トラック | 10-29 | nominal | category_dtype_candidate | N/A(0%) | **yes** | 芝直/右/左等 |
| surface | object | 2 | [turf,dirt] | 派生(芝/ダート) | turf/dirt | binary | label encoding | N/A(0%) | **yes** | 最重要バイナリ |
| distance_bin | object | 4 | [sprint,mile,...] | 派生(距離カテゴリ) | 4値 | ordinal | category_dtype_candidate | N/A(0%) | **yes** | 順序性あり |
| grade_code | object | 6 | [X,E,C,...] | CODE2003+派生 | A/B/C/L/E/X | ordinal | class_level_current推奨 | N/A(0%) | **yes** | 順序あり |
| track_condition_code | int64 | 4 | [1,2,3,4] | CODE2010 統合馬場 | 1-4 | ordinal | ordinal_as_numeric_candidate | N/A(0%) | **yes** | 良→不良の順序 |
| tenkocd | int64 | 4 | [1,2,3,4] | CODE2011 天候 | 1-4 | ordinal | ordinal_as_numeric_candidate | N/A(0%) | **yes** | 晴→雨の順序 |
| sexcd | int64 | 3 | [1,2,3] | CODE2202 性別 | 1-3 | nominal | category_dtype_candidate | N/A(0%) | **yes** | 牡/牝/セン |
| jyuryocd | int64 | 4 | [1,2,3,4] | CODE2008 重量種別 | 1-4 | nominal | category_dtype_candidate | N/A(0%) | **yes** | ハンデ/定量等 |
| minaraicd | int64 | 6 | [0,1,2,3,4,9] | CODE2303 見習 | 0-9 | ordinal | category_dtype_candidate | N/A(0%) | **yes** | 減量区分 |
| tozaicd | int64 | 4 | [1,2,3,4] | CODE2301 東西所属 | 1-4 | nominal | category_dtype_candidate | N/A(0%) | **yes** | 美浦/栗東等 |
| keirocd | int64 | 8 | [1,2,3,...] | CODE2203 毛色 | 1-11 | nominal | category_dtype_candidate | N/A(0%) | **yes** | 予測力低いがコストなし |
| aggressive_tier | object | 2 | [weak,strong] | 派生(RegimeDetector) | 2値 | binary | label encoding | N/A(0%) | **yes** | 市場レジーム |
| kyakusitu_x_distance | object | 24 | [3.0_sprint,...] | 派生(脚質×距離) | 24値 | nominal | category_dtype_candidate | "nan_*"を1カテゴリ | **yes** | 交互作用特徴量 |
| kyakusitu_x_surface | object | 12 | [3.0_dirt,...] | 派生(脚質×芝ダ) | 12値 | nominal | category_dtype_candidate | "nan_*"を1カテゴリ | **yes** | 交互作用特徴量 |
| syubetucd | int64 | 4 | [11,12,13,14] | CODE2005 競走種別 | 4値 | ordinal | ordinal_as_numeric_candidate | N/A(0%) | yes | class_level_currentが代替 |
| kigocd | float64 | 8 | [3,23,0,...] | CODE2006 競走記号 | 0-24 | nominal | unknown_category_candidate | NaN→カテゴリ化 | yes | 57%NaN=該当なし |
| kyakusitukubun_cd | float64 | 5+NaN | [0,1,2,...] | 04-SE 脚質(当レース) | 0-4+NaN | ordinal | category_dtype_candidate | NaN→-1またはカテゴリ | yes | **注意:当レースはPOST_RACE** |
| rest_category | float64 | 5+NaN | [1,2,3,...] | 派生(休養カテゴリ) | 1-5+NaN | ordinal | ordinal_as_numeric_candidate | NaN→-1 | yes | |
| kisyucode | object | 191 | [01152,...] | 04-SE 騎手コード | 191値 | high_cardinality | past_target_encoding_candidate | N/A(0%) | possible | 集計特徴量で代替済み |
| chokyosicode | object | 227 | [01025,...] | 04-SE 調教師コード | 227値 | high_cardinality | past_target_encoding_candidate | N/A(0%) | possible | 集計特徴量で代替済み |
| sire_id | object | 457 | [...] | 血統登録番号(種牡馬) | 457値 | high_cardinality | past_target_encoding_candidate | N/A(0%) | possible | sire_wr等で代替済み |

### Insight: カテゴリ変数の取扱

everyDB2のコード値は全て数値(int/float)で格納されているが、名義尺度(nominal)として扱うべきものが多い。LightGBMの`categorical_feature`に指定する場合、dtypeをcategory型に変換する必要がある。順序尺度(ordinal)はそのまま数値として扱ってもよいが、名義尺度(jyocd, trackcd等)は必ずカテゴリ指定またはone-hot化が必要。

---

## G. 数値特徴量処理候補テーブル (要対応カラム中心)

| column | dtype | missing_rate | zero_rate | min | p50 | p99 | max | inf_count | numeric_feature_type | suggested_handling | reason | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sire_distance_wr | float64 | ~10% | ~3% | 0.01 | 0.11 | 75.3 | **118.3** | 0 | rate | **investigate** | **max>1、勝率の範囲外。計算バグ疑い** | high |
| win_dominance | float64 | 70.6% | 0% | 0 | 16 | 55 | 55 | 0 | count_result | keep_nan_candidate | 過去勝利数、初出走=NaN | high |
| pace_aptitude | float64 | 69.4% | 0% | - | - | - | - | 0 | rate | keep_nan_candidate | 高欠損、経験不足 | high |
| blood_distance_wr | float64 | 52.1% | 0% | 0 | 0.12 | 0.5 | 1.0 | 0 | rate | keep_nan_candidate | 血統距離別勝率 | high |
| harontime_late_trend | float64 | 43.9% | 0% | - | - | - | - | 0 | average | keep_nan_candidate | 出走不足 | high |
| class_adj_formetric | float64 | 35.0% | 0% | - | - | - | - | 0 | average | keep_nan_candidate | 出走不足 | high |
| blood_surface_wr | float64 | 32.3% | 0% | 0 | 0.13 | 0.5 | 1.0 | 0 | rate | keep_nan_candidate | 血統芝ダ別勝率 | high |
| haron_zscore_trend | float64 | 30.0% | 0% | - | - | - | - | 0 | average | keep_nan_candidate | 出走不足 | high |
| blood_prize_log | float64 | 27.1% | 0% | - | - | - | - | 0 | average | keep_nan_candidate | 血統情報なし | high |
| form_trend | float64 | 21.2% | 0% | - | - | - | - | 0 | average | keep_nan_candidate | 出走不足 | high |
| form_consistency | float64 | 21.2% | 0% | - | - | - | - | 0 | std | keep_nan_candidate | 出走不足 | high |
| days_since_last_race | float64 | 11.0% | 0% | 5 | 37 | 265 | 444 | 0 | interval | **missing_flag_candidate** | 初出走=NaN。フラグ追加で「初出走」表現可能 | high |
| zogensa | float64 | 10.8% | 0% | -26 | 0 | 22 | 30 | 0 | physical_measurement | keep_nan_candidate | 初出走(前回体重なし)=NaN | high |
| course_wr | float64 | 10.9% | 0% | 0 | 0.14 | 0.5 | 1.0 | 0 | rate | keep_nan_candidate | コース経験なし | high |
| jt_combo_wr | float64 | 5.9% | 0% | 0 | 0.12 | 0.45 | 1.0 | 0 | rate | keep_nan_candidate | コンビ経験なし | high |
| blinker_change | float64 | 11.0% | ~85% | -1 | 0 | 1 | 1 | 0 | binary | **zero_fill_candidate** | 初出走時NaN→0(変更なし) | high |
| is_nar_transfer | float64 | 11.1% | ~89% | 0 | 0 | 1 | 1 | 0 | binary | **zero_fill_candidate** | JRA専属馬NaN→0 | high |
| nar_recent_ratio | float64 | 11.0% | ~89% | 0 | 0 | 0.5 | 1.0 | 0 | rate | **zero_fill_candidate** | JRA専属馬NaN→0 | high |
| jockey_wr_venue | float64 | 3.6% | 0% | 0 | 0.11 | 0.4 | 1.0 | 0 | rate | keep_nan_candidate | 当該場未騎乗 | high |
| overround | float64 | 0.2% | 0% | 0.8 | 1.2 | 1.8 | 3.0 | 0 | odds | keep_nan_candidate | 市場指標 | high |
| bataijyu | float64 | 0% | 0% | 386 | 480 | 556 | 570 | 0 | physical_measurement | keep_nan_candidate | 馬体重 | high |
| ev_win | float64 | 0.2% | 0% | - | 0.8 | 5.0 | 50.0 | 0 | unknown | keep_nan_candidate | モデル出力(EV) | high |

### Insight: NaNパターンの体系性

NaN は3つの明確なパターンに分類される:
- **初出走(11%)**: days_since_last_race, blinker_change, class_move 等 — 前走データがないため全て同時にNaN
- **ペース不足(~21%)**: pace_aptitude, actual_pace_fit, front/closing_pace_wr — ペース分析に十分な過去レースがない
- **フォーム不足(~30%)**: form_trend, form_consistency, freshness_score, form_peak_flag — 直近N走の着順推移データ不足

LightGBMはNaNを自然処理するが、missing flag の追加は初出走馬の予測精度向上に寄与する可能性がある。特に `days_since_last_race` のNaNは「初出走」という強いシグナル。

---

## H. 最終分類リスト

```python
# === 欠損NaNのままLightGBMに投入候補 ===
# 自然なNaN(初出走・経験不足・条件該当なし)をそのまま扱う
keep_nan_candidate_cols = [
    # 過去成績集計系 (初出走・経験不足でNaN)
    "norm_finish_logit_avg", "harontimel5_avg", "harontimel5_zscore",
    "harontime_late_trend", "timediff_avg", "jyuni1c_avg", "jyuni4c_avg",
    "closing_index_avg",
    # ペース・脚質系
    "pace_aptitude", "actual_pace_fit",
    "front_pace_wr", "early_pace_wr", "mid_pace_wr", "closing_pace_wr",
    "pace_ratio_zscore", "pace_ratio_trend",
    # フォーム系
    "form_trend", "form_consistency", "freshness_score", "form_peak_flag",
    "haron_zscore_trend", "class_adj_formetric",
    # クラス・コンディション変化
    "class_drop_bounce", "class_move", "surface_change", "distance_change",
    "rest_category", "track_condition_delta",
    # 血統系
    "blood_surface_wr", "blood_distance_wr", "blood_total_wr",
    "blood_condition_wr", "blood_prize_log",
    "sire_wr", "bms_wr",  # sire_distance_wrはinvestigate
    # 騎手・調教師統計
    "jockey_wr_overall", "jockey_wr_venue", "jockey_wr_distance",
    "jockey_wr_condition", "jockey_surprise",
    "trainer_wr_overall", "trainer_wr_venue",
    "jt_combo_wr", "jt_combo_place_rate", "jt_combo_win_rate", "jt_combo_avg_return",
    # コース・条件別成績
    "course_wr", "course_distance_wr",
    # ワイドオッズ (頭数依存NaNは自然)
    *[f"wide_odds_{i}_{j}" for i in range(1, 18) for j in range(i+1, 19)],
    # 馬体重
    "zogensa", "weight_zscore",
    # 市場指標
    "overround", "market_entropy", "odds_volatility",
    "odds_drop_rate_60_10", "odds_drop_rate_30_10",
    "odds_velocity", "odds_acceleration",
    "odds_direction_consistency", "odds_skewness",
    # レース内相対ランク
    "harontimel5_zscore_race_rank", "jyuni1c_avg_race_rank",
    "jyuni4c_avg_race_rank", "closing_index_avg_race_rank",
    "norm_finish_logit_avg_race_rank", "timediff_avg_race_rank",
    "sire_wr_race_rank", "bms_wr_race_rank",
    "course_wr_race_rank", "course_distance_wr_race_rank",
    # モデル出力
    "ev_win", "ev_place", "ev_place_direct",
    "p_win_pred", "p_place_pred",
    "p_ability_win", "p_ability_place",
    "e_return_win_pred", "e_return_place_pred",
    "edge_win", "edge_place",
    "p_minus_e_gap_place", "market_log_error_win",
    "signed_log_error_win", "popularity_change_30_10",
    "popularity_rank_fallback_used",
]

# === 欠損フラグ追加候補 ===
# NaN自体が「初出走」「情報なし」として意味を持つ列
missing_flag_candidate_cols = [
    "days_since_last_race",    # NaN=初出走 (最も強いフラグ候補)
    "zogensa",                  # NaN=初出走(前回体重なし)
    "weight_zscore",           # NaN=母集団不足
    "form_trend",              # NaN=出走不足
    "form_consistency",        # NaN=出走不足
    "harontimel5_avg",         # NaN=前走なし
    "closing_index_avg",       # NaN=前走なし
]

# === 0埋め候補 ===
# NaN=0(該当なし/変更なし)が自然な列
zero_fill_candidate_cols = [
    "blinker_change",      # 初出走時NaN→0(変更なし)
    "is_nar_transfer",     # JRA専属馬NaN→0(該当なし)
    "nar_recent_ratio",    # JRA専属馬NaN→0(地方出走なし)
]

# === Unknown化候補(カテゴリ欠損) ===
unknown_category_candidate_cols = [
    "kigocd",              # 57%NaN=該当条件なし
    "datakubun",           # 48%NaN(ただしML非使用推奨)
]

# === LightGBM categorical_feature候補 ===
categorical_candidate_cols = [
    "jyocd",               # 10競馬場
    "trackcd",              # 8トラック形状
    "surface",              # turf/dirt
    "distance_bin",         # sprint/mile/intermediate/long
    "grade_code",           # A/B/C/L/E/X
    "track_condition_code", # 1-4 (良/稍重/重/不良)
    "tenkocd",              # 1-4 (晴/曇/雨/小雨)
    "sexcd",                # 1-3 (牡/牝/セン)
    "jyuryocd",             # 1-4 (重量種別)
    "minaraicd",            # 0-9 (見習区分)
    "tozaicd",              # 1-4 (東西所属)
    "keirocd",              # 1-11 (毛色)
    "aggressive_tier",      # weak/strong
    "kyakusitu_x_distance", # 24値 (脚質×距離)
    "kyakusitu_x_surface",  # 12値 (脚質×芝ダ)
    "kyakusitukubun_cd",    # 0-4+NaN (脚質: 注意-当レースはPOST_RACE)
    "rest_category",        # 1-5+NaN (休養カテゴリ)
]

# === リーク・ドロップ候補 ===
# レース後情報、全NaN、定数列
leakage_or_drop_candidate_cols = [
    # POST_RACE結果 (評価用に保持、特徴量として使用禁止)
    "kakuteijyuni",        # 確定着順 = TARGET
    "confirmed_odds",      # 確定オッズ = TARGET用
    "nyusenjyuni",         # 入線順位 (kakuteijyuniと100%同一)
    "fukasyokin",          # 獲得付加賞金 (POST_RACE)
    # ラップタイム (POST_RACE)
    *[f"laptime{i}" for i in range(1, 26)],
    # コーナー・周回 (POST_RACE)
    *[f"corner{i}" for i in range(1, 5)],
    *[f"syukaisu{i}" for i in range(1, 5)],
    # ハロンタイム (POST_RACE)
    "harontimes3", "harontimes4",
    # 同着・異常 (POST_RACE)
    "dochakukubun", "dochakutosu", "ijyocd",
    # 入線・レコード (POST_RACE)
    "nyusentosu", "recordupkubun",
    # 100% NaN列 (情報量ゼロ)
    "jyuni1", "jyuni2", "jyuni3", "jyuni4",
    "chakusacdp", "chakusacdpp",
    "coursekubuncd", "coursekubuncdbefore",
    "fukusyoku", "reserved1",
    "bamei1", "bamei2", "bamei3",
    "zogenfugo",
    # 定数列 (ML不要)
    "year",  # 2024のみ
    "kyoribefore", "trackcdbefore",  # 常に0
    "hinsyucd",  # 常に1(サラブレッド)
    "honsyokin7", "fukasyokin5",  # 常に0
    "fukasyokinbefore1",
    "kettonum3",  # 常に0
    "syogaimiletime",  # 常に0(障害なし)
    "recordspec",  # 常に"RA"
    "gradecdbefore", "jyokenname",  # 全行空文字
    "blood_keito_cd",  # 全行"unknown"
    "dmkubun",  # 99.9%が3
    "kubun",  # 表示用メタ
    "EV_lower_win_corrected",  # 100%が0
    "popularity_rank_fallback_used",  # 常に0
    "reserved2", "reserved3", "reserved4",
]

# === 予測時点依存列 ===
# 前日予測では利用不可、発走前予測でのみ利用可能
prediction_time_dependent_cols = [
    # オッズ時系列(発走10-60分前スナップショット必要)
    "odds_drop_rate_60_10", "odds_drop_rate_30_10",
    "odds_velocity", "odds_volatility",
    "odds_acceleration", "odds_direction_consistency",
    "odds_skewness",
    # 最終オッズ(投票締切時確定、前日には中間オッズのみ)
    "odds", "tanodds", "tanninki", "fukuoddslow",
    "odds_rank",
    # ワイドオッズ(投票締切時確定)
    *[f"wide_odds_{i}_{j}" for i in range(1, 18) for j in range(i+1, 19)],
    # 当日情報
    "bataijyu", "zogensa", "weight_zscore",  # 直前計量
    "tenkocd", "sibababacd", "dirtbabacd",   # 当日発表
    "track_condition_code",                    # 上記から派生
    "is_good_track", "is_soft_track",         # 上記から派生
    "popularity_change_30_10",                 # 30分前スナップショット必要
]

# === 要調査列 ===
investigate_cols = [
    # 重大: 値の異常
    "sire_distance_wr",    # max=118.27、勝率(0-1)の範囲外。計算バグの可能性高
    # 重大: 100%NaNだがFEATURE_COLSに宣言
    "deviation_rank",      # 100%NaN、FEATURE_COLSに含まれる
    "deviation_zscore",    # 100%NaN、FEATURE_COLSに含まれる
    "weight_change_zone",  # 100%NaN、FEATURE_COLSに含まれる可能性
    "weight_change_ratio", # 100%NaN、FEATURE_COLSに含まれる可能性
    # 高欠損: feature importance安定性への影響
    "pace_aptitude",       # 69.4%NaN
    "win_dominance",       # 70.6%NaN (sample=[16,15,...]は過去勝利数?)
    # 注意: 当レース脚質はPOST_RACE、過去脚質傾向はPRE-RACE
    "kyakusitukubun_cd",   # 04-SE KyakusituKubun = 当レース脚質(POST_RACE)
                           # 27-UMA Kyakusitu1-4 = 過去脚質傾向(PRE-RACE)
                           # どちらを参照しているかコード確認が必要
    # FEATURE_COLSに宣言されているがデータに存在しないカラム
    # (available_cols = [c for c in FEATURE_COLS if c in df.columns] で安全に除外される)
    # form_trend_race_rank, blood_total_wr_race_rank 等 20列以上
]
```

---

## I. サブエージェント別レポート

### Sub1: データ構造・キー・粒度担当

- **担当範囲**: Parquet全体構造、行数・列数、主キー候補、粒度確認
- **確認したdocs/everydb2/のファイル**: 03-RACE.md, 04-UMA_RACE.md (キー定義)
- **主要な発見**:
  - 45,827行×494列。主キー(race_id, kettonum)重複ゼロ
  - 2024年1月-12月を106日・3,329レースで均等カバー
  - 1レースあたり13.77頭(5-18頭)、1頭あたり平均3.97回出走
  - 18列が100%NaN(廃止/未使用)。kakuteijyuni, confirmed_oddsは0%NaN(正解ラベル)
- **注意すべきカラム**: `horse_id`が存在せず、`kettonum`(血統登録番号)が馬の識別子
- **判断保留カラム**: 100%NaN列の18列のうち、deviation_rank/zscore, weight_change_zone/ratioはFEATURE_COLSに含まれる可能性
- **統合時に確認すべき論点**: 100%NaN列の一部が意図的な設計かバグか

### Sub2: everyDB2仕様書マッピング担当

- **担当範囲**: docs/everydb2/ 全仕様書の読み込みと構造化
- **確認したdocs/everydb2/のファイル**: 全59テーブル(CODE.md, INDEX.md含む)
- **主要な発見**:
  - DataKubunで更新段階を管理(1:木曜→2:金土→3-6:速報→7:確定)
  - レース前/後の区別が明確に定義されている
  - オッズはDataKubun=4(確定)が投票締切時の最終版
- **注意すべきカラム**:
  - honsyokin1-7は「配当テーブル」(PRE-RACE)だが、fukasyokin(無番号)は「獲得賞金」(POST_RACE)
  - 名前が紛らわしいので注意
- **判断保留カラム**: 派生特徴量(~23列)は仕様書に直接の記載なし
- **統合時に確認すべき論点**: kyakusitukubun_cdの生成元が04-SE(当レース)か27-UMA(過去傾向)か

### Sub3: 欠損率・NaN意味分類担当

- **担当範囲**: 全カラムの欠損率分析、NaNの意味分類(A-G)
- **確認したdocs/everydb2/のファイル**: 04-UMA_RACE.md, 03-RACE.md (欠損理由の特定)
- **主要な発見**:
  - 256列にNaNあり
  - NaNパターンが3つの体系(初出走11%、ペース不足21%、フォーム不足30%)に分類
  - weight_change_zone/ratioが100%NaNだがzogensaは10.8%NaN — 列名不一致(zogen_sa vs zogensa)が疑われる
- **注意すべきカラム**: deviation_rank/zscoreがFEATURE_COLSに含まれるが100%NaN
- **判断保留カラム**: weight_change_*の計算ロジックが呼ばれていないバグの可能性
- **統合時に確認すべき論点**: weight_change_*は計算ロジック実装で解決可能か、それとも意図的な未実装か

### Sub4: リーク検査・予測時点利用可否担当

- **担当範囲**: リーク疑いカラムの抽出、予測時点別(A/B/C)の利用可否分類
- **確認したdocs/everydb2/のファイル**: 04-UMA_RACE.md, 03-RACE.md, 05-HARAI.md, 14-26 ODDS系, 39-TENKO_BABA.md
- **主要な発見**:
  - 現在のリークリスクは**低**。ホワイトリスト形式のFEATURE_COLSにより、POST_RACE情報はモデルに入力されない
  - POST_RACE_COLS除外リストに9列の欠落あり(nyusenjyuni, fukasyokin, dochakukubun, dochakutosu, nyusentosu, recordupkubun, harontimes3, harontimes4, ijyocd)
  - odds/tanoddsは投票締切時確定のため「レース前情報」だが、前日予測では利用不可
- **注意すべきカラム**: ワイドオッズ(153列)の大量NaNは頭数依存で自然だが、前日予測での利用可否に注意
- **判断保留カラム**: kyakusitukubun_cd(当レース脚質=POST_RACE、過去傾向=PRE-RACEの区別)
- **統合時に確認すべき論点**: POST_RACE_COLSへの9列追加、前日/直前予測の分離評価

### Sub5: カテゴリ・コード値担当

- **担当範囲**: カテゴリ変数、コード値、LightGBM categorical_feature候補の分析
- **確認したdocs/everydb2/のファイル**: CODE.md(全コード定義), 01-TOKU_RACE.md, 27-UMA.md, 04-UMA_RACE.md
- **主要な発見**:
  - everyDB2コード定義と完全照合完了
  - 16列がLightGBM categorical_feature候補
  - 高カーディナリティ(kisyucode=191, chokyosicode=227, sire_id=457)は集計特徴量で代替済み
- **注意すべきカラム**:
  - kyakusitu_x_distance(24値)、kyakusitu_x_surface(12値)の交差特徴量が有効
  - trackcd(8値)はsurface+distance_binに分解済みだが、直接使用も価値あり
- **判断保留カラム**: kigocd(57%NaN)、datakubun(48%NaN)
- **統合時に確認すべき論点**: int/floatコード値のdtype変換要否

### Sub6: 数値特徴量・集計特徴量担当

- **担当範囲**: 数値特徴量の分布、0埋め候補、NaN維持候補、欠損フラグ候補
- **確認したdocs/everydb2/のファイル**: (データ中心の分析)
- **主要な発見**:
  - sire_distance_wrに重大な異常値(max=118.27)。名前は勝率だが値が1.0を超えている — 計算バグの可能性高
  - Infなし(データ品質良好)
  - 重複列: kakuteijyuni=nyusenjyuni(100%同一)、blinker=blinker_on、bataijyu=weight_absolute
  - ~55列がリーク列。ただしモデルのFEATURE_COLSホワイトリストにより実際には使用されていない
- **注意すべきカラム**: sire_distance_wrの計算ロジック(src/features/)の確認が急務
- **判断保留カラム**: win_dominanceのsample値が[16,15,...]で、勝率(0-1)ではなく勝利数の可能性
- **統合時に確認すべき論点**: sire_distance_wrのバグ修正によるROI改善可能性

### Sub7: 時系列・バックテスト妥当性担当

- **担当範囲**: 時系列観点の分析、バックテスト妥当性、学習/推論ズレリスク
- **確認したdocs/everydb2/のファイル**: 04-UMA_RACE.md, 03-RACE.md, 39-TENKO_BABA.md, 44-MINING.md
- **主要な発見**:
  - 月別欠損率変動はwide_odds以外ほぼ無し(バックテストとして良好)
  - FEATURE_COLSに宣言されているがデータに存在しないカラムが20列以上(実装と設計の乖離)
  - バックテストROIは「直前予測」のROIであり、「前日予測」の性能は完全に未知
  - 初出走馬比率の季節変動(3-18%)が学習/推論ズレに影響する可能性
- **注意すべきカラム**: weight_change_zoneが100%欠損だがFEATURE_COLSに宣言(意図しない特徴量脱落)
- **判断保留カラム**: pace_aptitude(69.4%), win_dominance(70.6%)のfeature importance安定性
- **統合時に確認すべき論点**: 「前日予測モデル」と「直前予測モデル」の分離評価が未実施

---

## J. 人間が次に確認すべき項目

### High

1. **sire_distance_wr の計算ロジック確認** — max=118.27は勝率(0-1)の範囲外。`src/features/` 配下の血統特徴量計算コードで分子/分母の定義を確認。バグであれば修正によりROI改善の可能性。

2. **weight_change_zone/ratio の100%NaN原因確認** — `zogensa`(10.8%NaN)が存在するのに未計算。`src/features/feature_engine.py` の `_map_basic_features()` で `zogen_sa` vs `zogensa` の列名不一致が疑われる。

3. **kyakusitukubun_cd の参照元確認** — 04-SEの当レース脚質(POST_RACE)か、27-UMAの過去脚質傾向(PRE-RACE)か。`src/features/` の計算コードで確認。

4. **POST_RACE_COLSへの9列追加検討** — nyusenjyuni, fukasyokin, dochakukubun, dochakutosu, nyusentosu, recordupkubun, harontimes3, harontimes4, ijyocd。ホワイトリストで保護されているが、防御的プログラミングとして。

5. **FEATURE_COLSとデータの乖離確認** — 20列以上がFEATURE_COLSに宣言されているがデータに存在しない。`available_cols = [c for c in FEATURE_COLS if c in df.columns]` で安全に除外されているが、意図しない情報損失がないか。

6. **deviation_rank/deviation_zscore の取扱確認** — 100%NaNだがFEATURE_COLSに宣言。モデル推論時に動的計算される設計か、parquet出力時の欠落か。

### Medium

7. **前日予測モデルと直前予測モデルの分離評価** — バックテストROIは直前予測(最終オッズ利用)の結果。前日予測の性能は完全に未知。

8. **pace_aptitude(69.4%), win_dominance(70.6%) のfeature importance安定性確認** — 高欠損率特徴量のimportanceがWF fold間で安定か。Spearman rhoで確認済みだが、個別に注目。

9. **カテゴリ変数のLightGBM categorical_feature指定** — 16列が候補。dtypeがint/floatのコード値(trackcd, jyocd等)をcategory型に変換する必要があるか。

10. **月別・開催場別の欠損偏り確認** — 全体的には良好だが、初出走馬比率の季節変動(3-18%)が学習/推論ズレに与える影響。

11. **honsyokin1-7(配当テーブル)とfukasyokin(獲得賞金)の混同防止** — 名前が似ているがPRE-RACE/POST-RACEの区別が異なる。コード上のコメントや命名規則の整理。

12. **wide_odds 153列の次元削減検討** — 88%以上NaNの列(17/18頭レース用)は情報量が低い。特徴量重要度で下位なら削除候補。

### Low

13. **100%NaN・定数列のparquet出力からの除外** — 37列が不要。ファイルサイズ削減と可読性向上。

14. **重複列の統合** — kakuteijyuni/nyusenjyuni、blinker/blinker_on、bataijyu/weight_absolute のいずれかを削除。

15. **レポートのCSV化・可視化** — 欠損率推移グラフ、カテゴリ分布、feature importance等。

16. **カラム命名規則の整理** — zogen_sa vs zogensa 等の不一致。snake_case統一。

17. **datakubunのNaN(48%)のETL整理** — entries由来データの紐付け課題。バックテストには影響しないが、データ品質監視上のノイズ。

---

## Insight: 時系列CV設計時の注意点

1. **初出走馬比率の季節変動を考慮したfold設計**: 新馬戦シーズン(6-11月)は初出走馬比率が10-18%。古馬中心シーズン(1-5月)は3-13%。現状のWF検証は年度境界(1月1日)を採用しており適切。

2. **過去成績集計のウォームアップ期間**: バックテスト開始直後(1月)でも過去成績系の欠損率は13.1%で12月(13.8%)とほぼ変わらない。学習期間からの過去データが適切に使われている。

3. **オッズ時系列特徴量の利用可否に基づくモデル分離**: 「前日予測モデル」(odds_dynamics系特徴量なし)と「直前予測モデル」(odds_dynamics系特徴量あり)を分けてCV評価すべき。現状は直前予測モデルのみ評価。

4. **血統特徴量のデータカバレッジ低下**: blood_distance_wrは52.1%欠損。LightGBMはNaNをleft/rightの分岐として学習するため、fold間で血統データのカバレッジが変わるとfeature importanceが不安定になる。WF検証のSpearman rhoで安定性を確認している設計は適切。

5. **推奨CV戦略**: 現状の2-fold WF (Fold0: train 2020-2023/test 2024, Fold1: train 2021-2024/test 2025) は最低限。拡張案としてRolling-window CV(例: train 3年/test 1年を4-fold)で季節性の影響を分散可能。
