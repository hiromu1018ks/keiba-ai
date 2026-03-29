# 特徴量エンジニアリング設計書 — ROI >100% 達成

**日付:** 2026-03-29
**対象:** Stage1 AbilityModel / PlaceAbilityModel
**現状:** ROI 66.3% (2024テスト / 2,967 bets / 296,700円投資 / 196,780円払戻)
**目標:** ROI >100% (特徴量改善による予測精度向上)

---

## 1. 現状の特徴量一覧

### Stage1 AbilityModel (20 features → 3 dead = 実質17)

| # | 特徴量 | カテゴリ | 状態 | 備考 |
|---|--------|----------|------|------|
| 1 | surface | レース条件 | OK | カテゴリ |
| 2 | distance_bin | レース条件 | OK | カテゴリ |
| 3 | track_condition_code | レース条件 | OK | 数値 |
| 4 | grade_code | レース条件 | OK | カテゴリ |
| 5 | field_size | レース条件 | OK | 数値 |
| 6 | weight_diff_from_mean | レース条件 | OK | 数値 |
| 7 | difficulty_score | レース条件 | OK | 数値 |
| 8 | norm_finish_logit_avg | 過去成績 | OK | logit変換済 |
| 9 | jockey_surprise | 過去成績 | OK | Beta平滑化 |
| 10 | haron_time_zscore_avg | 過去成績 | **DEAD** | 常にNaN |
| 11 | norm_finish_logit_avg_race_z | レース内z | OK | |
| 12 | jockey_surprise_race_z | レース内z | OK | |
| 13 | haron_time_zscore_avg_race_z | レース内z | **DEAD** | 常にNaN |
| 14 | norm_finish_logit_avg_race_pct | レース内pct | OK | |
| 15 | jockey_surprise_race_pct | レース内pct | OK | |
| 16 | haron_time_zscore_avg_race_pct | レース内pct | **DEAD** | 常にNaN |
| 17 | jockey_cond_wr | 騎手 | OK | k=25平滑化 |
| 18 | jockey_cond_wr_race_z | 騎手 | OK | |
| 19 | jockey_cond_wr_race_pct | 騎手 | OK | |
| 20 | weight_absolute | 馬体 | OK | |

### PlaceAbilityModel (17 features)

AbilityModel から haron_time 系3列を除外した同一構成。

### MarketModel (7 features)

`e_return_win_pred`, `signed_log_error_win`, `abs_log_error_win`, `market_entropy`, `popularity_rank`, `p_win_pred_ratio`, `odds_rank`

### EVCorrectionModel (11 features)

`e_return_win_pred`, `p_x_e_interaction`, `p_minus_e_gap`, `signed_log_error_win`, `abs_log_error_win`, `market_entropy`, `popularity_rank`, `surface`, `distance_bin`, `track_condition_code`, `field_size`

---

## 2. 改善方針

### 2.1 コードレビュー指摘への対応

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| R1 | データリーク: shift(1)なし | 全expanding統計に明示的shift(1) | **S** |
| R2 | スパース統計の分散爆発 | Beta/additive平滑化を全カテゴリ統計に適用 | **S** |
| R3 | 「好調」定義が弱い | 正規化着順ベースに統一 | **A** |
| R4 | 冗長特徴量 | race_z/race_pctをrank_pctに統一 | **B** |
| R5 | 未正規化着順 | norm_finish_logitを全面採用 | **A** |
| R6 | 騎手→Stage2限定 | Stage1の騎手特徴量を削除しStage2に移動 | **A** |
| R7 | ETL拡張が必要な特徴量 | 新規ETL処理を追加 | **S** |

### 2.2 Stage分離の再設計

**Stage1 (純粋能力):** 馬そのものの能力のみ。騎手・調教師コンテキストを含めない。
→ AbilityModel, PlaceAbilityModel

**Stage2 (コンテキスト補正):** 騎手・調教師・市場情報による補正。
→ MarketModel, EVCorrectionModel (既存), + 新規: JockeyContextFeatures, TrainerContextFeatures

### 2.3 設計決定事項

- **血統スコア**: 静的 (Pattern A) — 馬の生涯不変値。x_UMAの`Ketto3InfoHansyokuNum1-14`から産駒成績を集計
- **dm_time_pred**: 保留 — JRA既存モデル出力の為、含まれる情報が不明。リークリスク高
- **騎手データ**: 年次集約 (SetYear < race_year) — x_KISYU_SEISEKIの年度単位データを使用。日付レベル分解は過剰
- **脚質**: x_UMA_RACEの`KyakusituKubun` (公式分類: 1=逃/2=先/3=差/4=追) を直接使用

---

## 3. 新規特徴量グループ (7 groups)

### Group A: 過走成績タイムシリーズ (修正)

既存の`HorseHistoryFeatures`を拡張。リーク防止と正規化を強化。

| 特徴量名 | 説明 | 計算方法 | ソース |
|----------|------|----------|--------|
| norm_finish_logit_avg | 既存（変更なし） | logit(1 - (pos-1)/(size-1))の3走平均 | x_UMA_RACE.finish_pos, races.field_size |
| haron_time_l3_avg | **新規** (dead feature置換) | 上り3Fタイムの3走平均 (NaN許容) | x_UMA_RACE.HaronTimeL3 |
| haron_time_l3_zscore | **新規** | 上り3Fの距離別z-score 3走平均 | HaronTimeL3 + 距離層別平均/std |
| time_diff_avg | **新規** | 勝馬差タイム3走平均 (秒) | x_UMA_RACE.TimeDIFN |
| corner_1c_avg | **新規** | 1コーナー通過順位の3走平均 (前サイド指標) | x_UMA_RACE.Jyuni1c |
| corner_4c_avg | **新規** | 4コーナー通過順位の3走平均 | x_UMA_RACE.Jyuni4c |
| closing_index_avg | **新規** | 追い込み指数 = (4C rank - finish rank) の3走平均。正=末脚 | Jyuni4c - finish_pos |
| kyakusitu_cd | **新規** | 公式脚質コード (1=逃/2=先/3=差/4=追) | x_UMA_RACE.KyakusituKubun |

**リーク防止:** 全て `race_date < target_date` で厳格フィルタ + `.tail(3)` で直近3走のみ。
**haron_time修正:** `HorseHistoryFeatures.compute()` 内のハードコードNaN (`haron_time_zscore_avg = float("nan")`) を実際の計算に置換。`HaronTimeL3`列がentries_histに存在する場合は利用、無い場合はNaNを返す（既存のfallbackロジックは維持）。

### Group B: 血統・産駒成績 (新規)

x_UMAの集計済み統計を利用。静的特徴量（馬ごとに1回計算、キャッシュ可能）。

| 特徴量名 | 説明 | 計算方法 | ソース |
|----------|------|----------|--------|
| blood_surface_wr | 馬場別勝率（平滑化） | Beta(1,10)平滑化: (wins+1)/(total+11) | x_UMA.Ba1-6Chakukaisu1-6 |
| blood_distance_wr | 距離別勝率（平滑化） | 同上 | x_UMA.Kyori1-6Chakukaisu1-6 |
| blood_condition_wr | 馬場状態別勝率 | 同上 | x_UMA.Jyotai1-12Chakukaisu1-6 |
| blood_total_wr | 総合成績勝率 | Beta平滑化 | x_UMA.ChuoChakukaisu1-6 |
| blood_prize_log | 累計賞金 (log変換) | log(1 + RuikeiHonsyoHeichi) | x_UMA.RuikeiHonsyoHeichi |
| blood_keito_cd | 系統コード (カテゴリ) | サンデーサイレンス系 etc. | x_KEITO.KeitoId (join via x_HANSYOKU) |

**注意:** x_UMAの成績は「その馬自身の全成績」の累計。最新race_date時点での累計値を使うが、静的近似（全期間累計）でも十分。理由: Stage1の目的は「馬の生来の適性」を捉えることであり、過走成績と二重計上を避けるため、血統系は静的で良い。

**平滑化の考え方:**
- α=1, β=10 のBeta事前分布
- 5戦10勝と50戦100勝を同じ勝率にしないため
- smoothed_wr = (wins + 1) / (total + 11)
- total=0 の場合は NaN

### Group C: 騎手コンテキスト (新規 → Stage2のみ)

Stage1から騎手特徴量を削除し、Stage2に移動。

| 特徴量名 | 説明 | 計算方法 | ソース |
|----------|------|----------|--------|
| jockey_wr_overall | 騎手全体勝率 (年度別) | SetYear < race_year の最新年を使用 | x_KISYU_SEISEKI.HeichiChakukaisu |
| jockey_wr_distance | 騎手距離別勝率 | 同上 (距離カテゴリ別) | x_KISYU_SEISEKI.Kyori1-6Chakukaisu |
| jockey_wr_venue | 騎手競馬場別勝率 | 同上 (場別) | x_KISYU_SEISEKI.Jyo1-20Chakukaisu |
| jockey_prize_log | 騎手賞金 (log変換) | log(1 + HonSyokinHeichi) | x_KISYU_SEISEKI.HonSyokinHeichi |
| jockey_surprise | 既存 (維持) | Beta平滑化勝率サプライズ | 過去成績から計算 |
| jockey_cond_wr | 既存 (Stage2に移動) | 階層平滑化条件勝率 | 過去成績から計算 |

**Stage1からの削除:** `jockey_surprise`, `jockey_cond_wr`, `jockey_surprise_race_z`, `jockey_surprise_race_pct`, `jockey_cond_wr_race_z`, `jockey_cond_wr_race_pct` をAbilityModel/PlaceAbilityModelのFEATURE_COLSから除外。

**Stage2への追加:** EVCorrectionModel または新規 JockeyContextFeatures モデルに上記を追加。

### Group D: 調教師コンテキスト (新規 → Stage2のみ)

x_CHOKYO_SEISEKIから騎手と同様の特徴量を生成。

| 特徴量名 | 説明 | 計算方法 | ソース |
|----------|------|----------|--------|
| trainer_wr_overall | 調教師全体勝率 | SetYear < race_year の最新年 | x_CHOKYO_SEISEKI.HeichiChakukaisu |
| trainer_wr_distance | 調教師距離別勝率 | 同上 | x_CHOKYO_SEISEKI.Kyori1-6Chakukaisu |
| trainer_wr_venue | 調教師場別勝率 | 同上 | x_CHOKYO_SEISEKI.Jyo1-20Chakukaisu |
| trainer_prize_log | 調教師賞金 (log) | log(1 + HonSyokinHeichi) | x_CHOKYO_SEISEKI.HonSyokinHeichi |

### Group E: 交互作用特徴量 (新規 → Stage1)

LightGBMが自動検出するが、明示的に与えることで少ないツリーで表現可能。

| 特徴量名 | 説明 | 計算方法 |
|----------|------|----------|
| kyakusitu_x_distance | 脚質×距離bin | 逃げ×短距離 など (カテゴリ積) |
| kyakusitu_x_surface | 脚質×馬場 | 逃げ×芝 など (カテゴリ積) |
| weight_x_distance | 馬体重×距離 | 数値の積 (距離が長いほど体重の影響増) |

### Group F: レース内正規化 (修正)

既存の race_z / race_pct を rank_pct に統一し、特徴量数を削減。

**変更前 (現在):**
- `*_race_z`: レース内z-score
- `*_race_pct`: レース内percentile rank

**変更後:**
- `*_race_rank`: レース内percentile rank のみ (z-scoreは削除)

理由: z-scoreとrank_pctは高い相関があり冗長。rank_pctの方が外れ値に強く、LightGBMの分岐に適している。

### Group G: ペース適性 (新規 → Stage1, 条件付き)

上り3Fタイムが使える場合のみ。HaronTimeL3が取得できない場合はNaN。

| 特徴量名 | 説明 | 計算方法 |
|----------|------|----------|
| pace_aptitude | 前傾/後傾ペースでの着順差 | 前傾レース着順 - 後傾レース着順の平均 |

**ペース判定:** レースの上り3Fタイムが距離別中央値より速ければ前傾、遅ければ後傾。データが不足している初期段階ではNaN。

---

## 4. Stage1 特徴量 (AbilityModel / PlaceAbilityModel) — 改定版

### レース条件 (7) — 変更なし
1. surface (cat)
2. distance_bin (cat)
3. track_condition_code (num)
4. grade_code (cat)
5. field_size (num)
6. weight_diff_from_mean (num)
7. difficulty_score (num)

### 過走成績 (8) — 6追加・3削除
8. norm_finish_logit_avg — 既存
9. haron_time_l3_avg — **新規** (haron_time_zscore_avg置換)
10. haron_time_l3_zscore — **新規** (haron_time_zscore_avg_race_z置換)
11. time_diff_avg — **新規**
12. corner_1c_avg — **新規**
13. corner_4c_avg — **新規**
14. closing_index_avg — **新規**
15. kyakusitu_cd — **新規** (cat)

### 血統 (6) — 全て新規
16. blood_surface_wr
17. blood_distance_wr
18. blood_condition_wr
19. blood_total_wr
20. blood_prize_log
21. blood_keito_cd (cat)

### 交互作用 (3) — 全て新規
22. kyakusitu_x_distance (cat)
23. kyakusitu_x_surface (cat)
24. weight_x_distance (num)

### レース内正規化 (5) — rank_pctに統一
25. norm_finish_logit_avg_race_rank
26. haron_time_l3_avg_race_rank
27. time_diff_avg_race_rank
28. corner_1c_avg_race_rank
29. closing_index_avg_race_rank

### 馬体 (1) — 変更なし
30. weight_absolute

**合計: 30特徴量** (既存17 + 新規16 - 削除3)

**PlaceAbilityModel:** 同じ30特徴量からStage1出力(`p_ability_win`)を追加して31特徴量。

---

## 5. Stage2 特徴量 (EVCorrectionModel) — 改定版

既存のEVCorrectionModel特徴量に騎手・調教師コンテキストを追加。

### 既存 (11) — 変更なし
1-11. (既存のまま)

### 騎手コンテキスト (6) — Stage1から移動+新規
12. jockey_surprise
13. jockey_cond_wr
14. jockey_wr_overall — **新規**
15. jockey_wr_distance — **新規**
16. jockey_wr_venue — **新規**
17. jockey_prize_log — **新規**

### 調教師コンテキスト (4) — 全て新規
18. trainer_wr_overall
19. trainer_wr_distance
20. trainer_wr_venue
21. trainer_prize_log

**合計: 21特徴量** (既存11 + 新規10)

---

## 6. ETL拡張

### 6.1 必要な新規Parquetファイル

| ファイル | ソーステーブル | 主要カラム | 用途 |
|----------|--------------|-----------|------|
| `data/raw/horses.parquet` | x_UMA | KettoNum, Ketto3Info*, Ba*, Kyori*, Jyotai*, ChuoChakukaisu*, RuikeiHonsyo*, Kyakusitu* | 血統・産駒成績 |
| `data/raw/jockey_stats.parquet` | x_KISYU_SEISEKI | SetYear, KisyuCode, Heichi*, Jyo*, Kyori*, HonSyokin* | 騎手成績 |
| `data/raw/trainer_stats.parquet` | x_CHOKYO_SEISEKI | SetYear, ChokyosiCode, Heichi*, Jyo*, Kyori*, HonSyokin* | 調教師成績 |

### 6.2 ETLスクリプト拡張

`scripts/run_etl.py` に以下を追加:

1. `x_UMA` → `data/raw/horses.parquet`
   - WHERE: `KettoNum IS NOT NULL`
   - カラム: KettoNum, Ketto3InfoHansyokuNum1-14, Ba1-6Chakukaisu1-6, Kyori1-6Chakukaisu1-6, Jyotai1-12Chakukaisu1-6, ChuoChakukaisu1-6, RuikeiHonsyoHeichi, Kyakusitu1-4

2. `x_KISYU_SEISEKI` → `data/raw/jockey_stats.parquet`
   - WHERE: `SetYear IS NOT NULL`
   - カラム: SetYear, KisyuCode, HeichiChakukaisu1-6, Jyo1-20Chakukaisu1-6, Kyori1-6Chakukaisu1-6, HonSyokinHeichi

3. `x_CHOKYO_SEISEKI` → `data/raw/trainer_stats.parquet`
   - WHERE: `SetYear IS NOT NULL`
   - カラム: SetYear, ChokyosiCode, HeichiChakukaisu1-6, Jyo1-20Chakukaisu1-6, Kyori1-6Chakukaisu1-6, HonSyokinHeichi

### 6.3 x_UMA_RACEの追加カラム

`entries.parquet` のETLで以下を追加取得:

- `HaronTimeL3` (上り3Fタイム) — 既にETL済みのはず、要確認
- `TimeDIFN` (勝馬差タイム) — 新規取得
- `Jyuni1c` (1コーナー通過順位) — 新規取得
- `Jyuni4c` (4コーナー通過順位) — 新規取得
- `KyakusituKubun` (公式脚質コード) — 新規取得

### 6.4 DataRepository拡張

```python
class DataRepository:
    def load_horses(self) -> pd.DataFrame: ...       # horses.parquet
    def load_jockey_stats(self) -> pd.DataFrame: ...  # jockey_stats.parquet
    def load_trainer_stats(self) -> pd.DataFrame: ... # trainer_stats.parquet
```

---

## 7. 新規ファイル構成

```
src/features/
  horse_history_features.py   — 修正 (haron_time計算実装, 新特徴量追加)
  bloodline_features.py       — 新規 (Group B: 血統・産駒成績)
  jockey_context_features.py  — 新規 (Group C: 騎手コンテキスト → Stage2)
  trainer_context_features.py — 新規 (Group D: 調教師コンテキスト → Stage2)
  interaction_features.py     — 新規 (Group E: 交互作用)
  feature_engine.py           — 修正 (新特徴量グループの統合)
  intra_race_features.py      — 修正 (rank_pct統一)
```

---

## 8. リスクと対策

| リスク | 対策 |
|--------|------|
| x_UMA成績が「直近累計」でない可能性 | 最初のETLでKettoNum単位で成績分布を確認。全期間累計なら静的扱いでOK |
| HaronTimeL3がentries_histに含まれない | 既存のNaN fallbackロジックを維持。Phase 2で対応 |
| x_KISYU_SEISEKIのSetYear粒度 | 年度集約で設計。日付レベル分解はYAGNI |
| 血統特徴量の多重共線性 | LightGBMは相関特徴量に強い。不要ならfeature_importanceで確認後に削除 |
| Stage1/2分割によるStage1精度低下 | 騎手をStage2に移すことでStage1の「純粋能力」評価がより安定する見込み |

---

## 9. 実装順序

1. **ETL拡張** — x_UMA, x_KISYU_SEISEKI, x_CHOKYO_SEISEKI のParquet化 + entries追加カラム
2. **DataRepository拡張** — 新Parquetファイルのローダー追加
3. **Group A: 過走成績修正** — haron_time計算実装 + 新特徴量追加
4. **Group B: 血統特徴量** — 新規ファイル作成
5. **Group F: レース内正規化統一** — rank_pctへの移行
6. **Group E: 交互作用特徴量** — 新規ファイル作成
7. **Stage1 FEATURE_COLS更新** — AbilityModel, PlaceAbilityModel
8. **Group C/D: 騎手・調教師コンテキスト** — Stage2移動 + 新規特徴量
9. **Stage2 FEATURE_COLS更新** — EVCorrectionModel
10. **バックテスト検証** — 2024テストでROI比較
