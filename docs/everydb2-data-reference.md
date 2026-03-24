# EveryDB2 データリファレンス

**目的:** 設計書 v5.5 の実装に必要なデータと EveryDB2 (JRA-VAN DataLab) の対応関係を整理する。
**対象DB:** PostgreSQL (EveryDB2構築済み前提)
**作成日:** 2026-03-24

---

## 1. EveryDB2 テーブル一覧と用途

| EveryDB2テーブル | 設計書での用途 | 優先度 |
|---|---|---|
| `RACE` | レース条件（場所・距離・コース・馬場状態・天候・クラス等） | **必須** |
| `UMA_RACE` | 出走馬ごとの結果（着順・タイム・オッズ・人気・体重・騎手・調教師等） | **必須** |
| `UMA` | 馬のマスタ情報（血統・能力統計・脚質傾向・獲得賞金等） | **必須** |
| `HARAI` | 払戻情報（単勝・複勝・ワイド等のオッズと払戻金） | **必須** |
| `ODDS_TANPUKU` | 単勝・複勝オッズ | **必須** |
| `ODDS_WIDE` | ワイドオッズ（組番ごとのオッズ） | **必須** |
| `JODDS_TANPUKU` | 時系列単勝・複勝オッズ（Late Money Filter用） | **必須** |
| `KISYU_SEISEKI` | 騎手成績（場所別・距離帯別） | **推奨** |
| `CODE` | コードマスタ（全コード体系のデコード） | **必須** |
| `RECORD` | コースレコード・G1レコード | 補助 |
| `COURSE` | コース情報（坂・直線等） | 補助 |

---

## 2. 共通キー構造

EveryDB2 の主テーブルは以下の5カラムでレースを一意に識別する:

| キーカラム | 例 | 説明 |
|---|---|---|
| `Year` | `2024` | 年 |
| `MonthDay` | `0324` | 月日 (MMDD) |
| `JyoCD` | `05` | 場所コード (01=Sapporo〜10=Kokura) |
| `Kaiji` | `03` | 回次 (01〜06) |
| `Nichiji` | `02` | 日次 (01〜12) |
| `RaceNum` | `08` | レース番号 (01〜12) |

**複合主キー:** `(Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum)`

---

## 3. 各テーブル詳細とフィールドマッピング

### 3.1 RACE（レース情報・110列）

**テーブル名:** `n_race`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `Year` | - | 主キー（年） | そのまま |
| `MonthDay` | - | 主キー（月日） | そのまま |
| `JyoCD` | - | 場所コード | CODE 2001 でデコード |
| `Kaiji` | - | 回次 | そのまま |
| `Nichiji` | - | 日次 | そのまま |
| `RaceNum` | - | レース番号 | そのまま |
| `TrackCD` | `surface` | **芝/ダート判定** | 10-22=芝, 23-29=ダート |
| `Kyori` | `distance` | 距離(m) | そのまま |
| `TenkoCD` | - | 天候 | CODE 2011 でデコード |
| `SibaBabaCD` | `track_condition` (芝) | 芝馬場状態 | CODE 2010 でデコード |
| `DirtBabaCD` | `track_condition` (ダート) | ダート馬場状態 | CODE 2010 でデコード |
| `SyubetuCD` | `race_type` | レース種別 | CODE 2005 でデコード |
| `JyokenCD` | `class_code` | 条件コード（クラス） | CODE 2007 でデコード |
| `GradeCD` | `grade_code` | グレード | CODE 2003: A=G1, B=G2, C=G3, D=重賞, E=特別, _=一般 |
| `Honsyokin` | - | 本賞金 | そのまま |
| `Fukasyokin` | - | 付加賞金 | そのまま |
| `TorokuTosu` | `field_size` | 頭数 | そのまま |
| `LapTime1`〜`LapTime25` | - | ラップタイム | 上位着順のラップ分析に使用 |
| `HaronTimeS3` | - | 3ハロンタイム(秒) | ペース分析に使用 |
| `HaronTimeS4` | - | 4ハロンタイム(秒) | ペース分析に使用 |
| `HaronTimeL3` | - | 後3ハロンタイム(秒) | 上がり3ハロン |
| `HaronTimeL4` | - | 後4ハロンタイム(秒) | 上がり4ハロン |

**surface（芝/ダート）の判定ロジック:**
```
TrackCD 10-22 → 芝 (turf)
TrackCD 23-29 → ダート (dirt)
TrackCD 51-59 → 障害 (steeple) → 対象外
```

**distance_bin の生成:**
```
芝:  sprint(<=1400), mile(1401-1700), intermediate(1701-2100), long(>=2101)
ダート: sprint(<=1400), mile(1401-1700), intermediate(>=1701)
```

**track_condition_code の生成:**
```
CODE 2010: 1=良, 2=稍重, 3=重/不良, 4=不良
→ is_good_track: 良 or 稍重
→ is_soft_track: 重 or 不良
```

---

### 3.2 UMA_RACE（出走馬ごとのレース結果・73列）

**テーブル名:** `n_uma_race`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `KakuteiJyuni` | `finish_pos` | **確定着順** | そのまま (1=1着, 0=取消等) |
| `Time` | - | タイム | そのまま |
| `HaronTimeL3` | - | 上がり3ハロン | 上がり分析 |
| `HaronTimeL4` | - | 上がり4ハロン | 上がり分析 |
| `Odds` | `win_odds_actual` | **確定単勝オッズ** | そのまま |
| `Ninki` | `popularity_rank` | **人気順位** | そのまま |
| `BaTaijyu` | - | 馬体重 | そのまま |
| `ZogenFugo` | - | 体重増減符号 | 1=増, 2=減, 3=不变 |
| `ZogenSa` | - | 体重増減幅 | そのまま |
| `KettoNum` | - | 血統番号 → UMAマスタ結合キー | そのまま |
| `KisyuCode` | - | 騎手コード → KISYU_SEISEKI結合 | そのまま |
| `ChokyosiCode` | - | 調教師コード | そのまま |
| `KyakusituKubun` | `running_style` | **脚質** | 1=逃げ, 2=先行, 3=差し, 4=追込 |
| `Honsyokin` | - | 本賞金（当該レース） | そのまま |
| `Fukasyokin` | - | 付加賞金 | そのまま |
| `Jyuni1c`〜`Jyuni4c` | - | コーナー通過順位 | ペース・位置取り分析 |
| `IJyoCD` | - | 異常コード | CODE 2101: 1=取消, 2=騎手変更等 |

**I(win) / I(place) ラベル生成:**
```
I(win)   = (KakuteiJyuni == 1)  → Stage A (単勝) の y
I(place) = (KakuteiJyuni <= 3)  → Stage A (複勝) の y
```

**p_market_win（市場確率）の計算:**
```
p_market_win = 1.0 / Odds
※ ODDS_TANPUKU の TanOdds を使用することも可
※ p_market_win_adj = p_market_win / overround で正規化
```

---

### 3.3 HARAI（払戻情報・199列）

**テーブル名:** `n_harai`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `PayTansyoUmaban1` | - | 単勝当該馬番 | そのまま |
| `PayTansyoPay1` | - | 単勝払戻金 | そのまま |
| `PayFukusyoUmaban1`〜`5` | - | 複勝当該馬番(最大5頭) | そのまま |
| `PayFukusyoPay1`〜`5` | `place_odds_actual` | **複勝払戻金** | 着順対応で取得 |
| `PayWideKumi1`〜`7` | - | ワイド組番 | そのまま |
| `PayWidePay1`〜`7` | `wide_odds_actual` | **ワイド払戻金** | ペア対応で取得 |
| `FuseirituFlag1`〜`9` | - | 不成立フラグ | 0=成立, 1=不成立 |
| `TokubaraiFlag1`〜`9` | - | 特払フラグ | 0=通常, 1=特払 |
| `HenkanFlag1`〜`9` | - | 返還フラグ | 0=通常, 1=返還あり |

**複勝オッズの取得:**
```
HARAI の PayFukusyoUmaban1-5 と PayFukusyoPay1-5 を使って
各着順馬の複勝払戻金を取得
※ 着順と馬番の対応は UMA_RACE の KakuteiJyuni で確認
```

**ワイドオッズの取得:**
```
HARAI の PayWideKumi1-7 と PayWidePay1-7 を使って
各組番のワイド払戻金を取得
※ Kumi は "馬番-馬番" の形式 (例: "3-7")
```

---

### 3.4 ODDS_TANPUKU（単勝・複勝オッズ・13列）

**テーブル名:** `n_odds_tanpuku`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `Umaban` | - | 馬番 | そのまま |
| `TanOdds` | `win_odds` | 単勝オッズ | 999.9=max, 特殊値除外 |
| `FukuOdds` | `place_odds` | 複勝オッズ | 999.9=max, 特殊値除外 |

**特殊値の扱い:**
```
"9999" / "0000" / "----" / "****" → 欠損扱い (NULL)
999.9 → 最大オッズ（人気薄・オッズ未確定の場合）
```

**overround（胴元控除率）の計算:**
```
overround = sum(1.0 / TanOdds) for all horses
通常 0.20〜0.25 (JRAは約22%)
market_entropy = -sum(p_market * log(p_market))
```

---

### 3.5 ODDS_WIDE（ワイドオッズ・11列）

**テーブル名:** `n_odds_wide`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `Kumi` | - | 組番 (例: "3-7") | 馬番1-馬番2に分割 |
| `OddsLow` | `wide_odds_low` | ワイドオッズ(下位) | そのまま |
| `OddsHigh` | `wide_odds_high` | ワイドオッズ(上位) | そのまま |

**WideTwoStageModel 用データ:**
```
各馬ペアの P(joint_hit) を学習するため、
全組番のオッズデータが必要
※ 全組み合わせ (C(18,2)=153) のオッズが取得可能
```

---

### 3.6 JODDS_TANPUKU（時系列単勝・複勝オッズ・14列）

**テーブル名:** `n_jodds_tanpuku`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `HappyoTime` | - | 発表時刻 (MMDDHHmm) | **Late Money Filterの時間軸キー** |
| `Umaban` | - | 馬番 | そのまま |
| `TanOdds` | `odds_t{time}` | 時点tの単勝オッズ | そのまま |
| `FukuOdds` | `fuku_odds_t{time}` | 時点tの複勝オッズ | そのまま |

**Late Money Filter (§8) での使い方:**
```
HappyoTime から発走時刻を計算:
  - odds_t10: 発走10分前のオッズ
  - odds_t3:  発走3分前のオッズ  ← 判定基準
  - odds_t2:  発走2分前のオッズ  ← ログのみ

キャンセル判定:
  change_rate = (odds_t10 - odds_t3) / odds_t10
  if change_rate >= 0.25: CANCEL（25%以上急落）
  if change_rate <= -0.30: ADD_CANDIDATE（30%以上急騰）
```

**オッズ変化率特徴量 (odds_dynamics_features.py):**
```
odds_drop_rate_60_10:  60分前→10分前の変化率
odds_drop_rate_30_10:  30分前→10分前の変化率
odds_velocity:         直近10分間の変化速度
odds_volatility:       オッズ変動のボラティリティ
popularity_change_30_10: 人気順位の変化
```

---

### 3.7 UMA（馬マスタ・227列）

**テーブル名:** `n_uma`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `KettoNum` | - | 血統番号（主キー） | そのまま |
| `HinsyuCD` | - | 品種 | CODE 2201: 1=サラブレッド |
| `SexCD` | - | 性別 | CODE 2202: 1=牡, 2=牝, 3=騸 |
| `UmaKigoCD` | - | 馬記号 | CODE 2204: (抽)(市)(外)[地]等 |
| `TozaiCD` | - | 所属 | CODE 2301: 1=関東(美浦), 2=関西(栗東) |
| `Kyakusitu1`〜`Kyakusitu4` | `running_style_tendency` | 脚質傾向(%) | 1=逃げ, 2=先行, 3=差し, 4=追込 |
| `TotalSyokin` | - | 総獲得賞金 | そのまま |
| `Blood*` (14系統) | - | 3代血統 | 各系統のKettoNum |

**芝/ダート別能力統計（UMAに事前計算済み）:**
```
芝成績: StartCount_Turf, WinCount_Turf, PlaceRate_Turf 等
ダート成績: StartCount_Dirt, WinCount_Dirt, PlaceRate_Dirt 等
距離帯別: StartCount_Sprint/Mile/Intermediate/Long
馬場状態別: StartCount_Good/Soft 等
方向別: StartCount_Left/Right/Straight
```

**Stage1 (AbilityModel) 用特徴量として活用可能:**
```
Rule 1: Stage1にオッズを入れない → UMAの能力統計はOK
芝/ダート別の勝率・連対率・複勝率を特徴量として使用
```

---

### 3.8 KISYU_SEISEKI（騎手成績・176列）

**テーブル名:** `n_kisyu_seiseki`

| EveryDB2フィールド | 設計書での特徴量名 | 用途 | 変換 |
|---|---|---|---|
| `KisyuCode` | - | 騎手コード（主キー） | そのまま |
| `SetYear` | - | 対象年 | そのまま |
| `JyoCD*` (10場所) | `jockey_{track}_win_rate` | 場所別勝率 | JyoCDごとの勝率・連対率 |
| `DistBand*` | `jockey_{dist}_win_rate` | 距離帯別勝率 | 距離帯ごとの勝率・連対率 |
| `TotalWinRate` | `jockey_overall_win_rate` | 総合勝率 | そのまま |

**特徴量エンジンでの活用:**
```
騎手×場所の勝率: jockey_win_rate_at_track
騎手×距離帯の勝率: jockey_win_rate_at_distance
→ 騎手の得意不得意を定量化
```

---

### 3.9 CODE（コードマスタ）

**テーブル名:** `n_code`

### 主要コード体系

| コードID | 名称 | 値 |
|---|---|---|
| 2001 | JyoCD (場所) | 01=札幌, 02=函館, 03=福島, 04=新潟, 05=東京, 06=中山, 07=中京, 08=京都, 09=阪神, 10=小倉 |
| 2003 | GradeCD (グレード) | A=G1, B=G2, C=G3, D=重賞(格付なし), E=特別, _=一般 |
| 2005 | SyubetuCD (種別) | 11=2歳, 12=3歳, 13=3歳以上, 14=4歳以上 (21-24=アラブ系) |
| 2007 | JyokenCD (条件) | 001-100=賞金条件, 701=新馬, 702=未出走, 703=未勝利, 999=オープン |
| 2009 | TrackCD (トラック) | 10=芝直, 11-16=芝左/右, 23-29=ダート, 51-59=障害 |
| 2010 | BabaCD (馬場状態) | 1=良, 2=稍重, 3=重(不良), 4=不良 |
| 2011 | TenkoCD (天候) | 1=晴, 2=曇, 3=雨, 4=霧, 5=雪, 6=小雨 |
| 2101 | IJyoCD (異常) | 1=取消, 2=騎手変更, 3=進上取消, 4=落馬, 5=失格, 7=失格着順 |
| 2201 | HinsyuCD (品種) | 1=サラブレッド, 2=サラ系, 5=アングロアラブ |
| 2202 | SexCD (性別) | 1=牡, 2=牝, 3=騸 |
| 2204 | UmaKigoCD (馬記号) | (抽)=抽選, (市)=市場, (外)=海外産, [地]=地方 |
| 2301 | TozaiCD (東西) | 1=関東(美浦), 2=関西(栗東) |

---

### 3.10 RECORD（コース/G1レコード・48列）

**テーブル名:** `n_record`

| EveryDB2フィールド | 用途 |
|---|---|
| `RecInfoKubun` | 1=コースレコード, 2=G1レコード |
| `RecordTime` | レコードタイム |

**用途:** レース難易度スコア (difficulty_score) の計算に使用可能

---

### 3.11 COURSE（コース情報・8列）

**テーブル名:** `n_course`

| EveryDB2フィールド | 用途 |
|---|---|
| `JyoCD` | 場所コード |
| `Kyori` | 距離 |
| `TrackCD` | トラック種別 |
| `KaishuDate` | 開催日 |

**用途:** コース特性（坂・直線長等）の分析に使用。設計書では直接的な特徴量としては未定義だが、レース難易度モデルの将来拡張で活用可能。

---

## 4. 設計書の特徴量カテゴリとEveryDB2対応

### 4.1 特徴量カテゴリ A: 馬の能力（Stage1用・オッズなし）

| 特徴量 | ソーステーブル | ソースフィールド | 変換 |
|---|---|---|---|
| 馬の芝/ダート別勝率 | UMA | WinCount_Turf/Dirt, StartCount_Turf/Dirt | 勝率 = WinCount / StartCount |
| 馬の芝/ダート別連対率 | UMA | PlaceRate_Turf/Dirt | そのまま |
| 馬の距離帯別成績 | UMA | StartCount_Sprint/Mile/Intermediate/Long | 勝率計算 |
| 馬の馬場状態別成績 | UMA | StartCount_Good/Soft | 勝率計算 |
| 脚質傾向 | UMA | Kyakusitu1〜4 | 比率として使用 |
| 過去のタイム傾向 | UMA_RACE | Time, HaronTimeL3 | 平均・標準偏差 |
| 過去の着順傾向 | UMA_RACE | KakuteiJyuni | 平均着順・連対率 |
| 騎手×場所の勝率 | KISYU_SEISEKI | JyoCD別WinRate | そのまま |
| 騎手×距離帯の勝率 | KISYU_SEISEKI | DistBand別WinRate | そのまま |
| 馬体重・変化 | UMA_RACE | BaTaijyu, ZogenFugo, ZogenSa | 体重/体重変化 |

### 4.2 特徴量カテゴリ B: レース内相対値

| 特徴量 | ソーステーブル | ソースフィールド | 変換 |
|---|---|---|---|
| 人気順位 | UMA_RACE | Ninki | そのまま |
| オッズ内ランク | ODDS_TANPUKU | TanOdds | レース内順位 |
| 馬体重内ランク | UMA_RACE | BaTaijyu | レース内順位 |
| 脚質分布 | UMA_RACE | KyakusituKubun | レース内脚質カウント |

### 4.3 特徴量カテゴリ C: オッズ変化率

| 特徴量 | ソーステーブル | ソースフィールド | 変換 |
|---|---|---|---|
| odds_drop_rate_60_10 | JODDS_TANPUKU | TanOdds (60分前 vs 10分前) | 変化率 |
| odds_drop_rate_30_10 | JODDS_TANPUKU | TanOdds (30分前 vs 10分前) | 変化率 |
| odds_velocity | JODDS_TANPUKU | TanOdds (直近数点) | 速度計算 |
| odds_volatility | JODDS_TANPUKU | TanOdds (全時系列) | 標準偏差 |
| popularity_change_30_10 | JODDS_TANPUKU | Ninki相当（オッズ順位の変化） | 人気変化 |

### 4.4 特徴量カテゴリ D: 市場歪み

| 特徴量 | ソーステーブル | ソースフィールド | 変換 |
|---|---|---|---|
| p_market_win | ODDS_TANPUKU | TanOdds | 1.0 / TanOdds |
| p_market_win_adj | ODDS_TANPUKU | TanOdds | p_market / overround |
| market_entropy | ODDS_TANPUKU | TanOdds (全頭) | -sum(p * log(p)) |
| overround | ODDS_TANPUKU | TanOdds (全頭) | sum(1/Odds) |
| popularity_rank | UMA_RACE | Ninki | そのまま |
| market_log_error | Market Model出力 | - | log(p_market / p_pred) |

### 4.5 特徴量カテゴリ E: 情報非対称性

| 特徴量 | ソーステーブル | ソースフィールド | 変換 |
|---|---|---|---|
| running_style_combo | UMA_RACE | KyakusituKubun | ペアの脚質組み合わせ |
| 情報非対称性スコア | KISYU_SEISEKI + UMA | 騎手成績 + 馬の成績 | 複合指標 |

### 4.6 特徴量カテゴリ F: 距離帯・馬場（one-hot）

| 特徴量 | ソーステーブル | ソースフィールド | 変換 |
|---|---|---|---|
| is_turf_sprint | RACE | TrackCD, Kyori | surface=芝 AND distance<=1400 |
| is_turf_mile | RACE | TrackCD, Kyori | surface=芝 AND 1401<=distance<=1700 |
| is_turf_intermediate | RACE | TrackCD, Kyori | surface=芝 AND 1701<=distance<=2100 |
| is_turf_long | RACE | TrackCD, Kyori | surface=芝 AND distance>=2101 |
| is_dirt_sprint | RACE | TrackCD, Kyori | surface=ダート AND distance<=1400 |
| is_dirt_mile | RACE | TrackCD, Kyori | surface=ダート AND 1401<=distance<=1700 |
| is_dirt_intermediate | RACE | TrackCD, Kyori | surface=ダート AND distance>=1701 |
| is_good_track | RACE | SibaBabaCD/DirtBabaCD | BabaCD=1 or 2 |
| is_soft_track | RACE | SibaBabaCD/DirtBabaCD | BabaCD=3 or 4 |

---

## 5. 学習ラベルの生成

### 5.1 単勝 2段階モデル (WinTwoStageModel)

| ステージ | ラベル | ソース | SQL例 |
|---|---|---|---|
| Stage A: P(win) | `I(win)` = 着順==1 | UMA_RACE.KakuteiJyuni | `(KakuteiJyuni = 1)::int` |
| Stage B: E(odds\|win) | 確定単勝オッズ | UMA_RACE.Odds (where KakuteiJyuni=1) | `Odds WHERE KakuteiJyuni = 1` |

### 5.2 複勝 2段階モデル (PlaceTwoStageModel)

| ステージ | ラベル | ソース | SQL例 |
|---|---|---|---|
| Stage A: P(place) | `I(place)` = 着順<=3 | UMA_RACE.KakuteiJyuni | `(KakuteiJyuni <= 3)::int` |
| Stage B: E(odds\|place) | 確定複勝オッズ | HARAI.PayFukusyoPay (着順対応) | `PayFukusyoPay* WHERE 対応馬番` |

### 5.3 ワイド 2段階モデル (WideTwoStageModel)

| ステージ | ラベル | ソース | SQL例 |
|---|---|---|---|
| Stage A: P(joint_hit) | `I(joint_hit)` = 両方3着内 | UMA_RACE.KakuteiJyuni (ペア) | ペアの両方が着順<=3 |
| Stage B: E(odds\|hit) | ワイドオッズ | HARAI.PayWidePay (組番対応) | `PayWidePay* WHERE 対応組番` |

### 5.4 EV補正モデル (EVCorrectionModel)

| モデル | ラベル | ソース | 変換 |
|---|---|---|---|
| P補正 | `I(win)` | UMA_RACE.KakuteiJyuni | binary classification (init_score=logit(p_pred)) |
| E補正 | `log(actual_odds) - log(e_pred)` | UMA_RACE.Odds (winnerのみ) | 回帰 (weight=1/√p_win_pred) |

### 5.5 RaceQualityScreener

| ラベル | ソース | 計算方法 |
|---|---|---|
| distortion_score | Market Model出力 | market_log_error_max_abs × market_entropy × (1 + n_positive/field_size) |
| profitability_proxy | HARAI + ODDS_TANPUKU | hist_roi_topk: 過去の実際ROI |
| stability_factor | HARAI + ODDS_TANPUKU | hist_positive_return_ratio: 正リターン割合 |

### 5.6 RegimeDetector

| ラベル | ソース | 計算方法 |
|---|---|---|
| market_efficiency | UMA_RACE + ODDS_TANPUKU | favorite_win_rate × (1 - clip(overround-0.20, 0, 0.15)/0.15) |
| 3状態クラス | market_efficiency + entropy | 離散化: AGGRESSIVE/CONSERVATIVE/COLLAPSED |

---

## 6. データギャップ分析

### 6.1 利用可能なデータ

| データ | EveryDB2対応 | 備考 |
|---|---|---|
| レース条件（場所・距離・馬場・天候・クラス） | RACE | 完全対応 |
| 出走馬の着順・タイム・オッズ・人気 | UMA_RACE | 完全対応 |
| 単勝・複勝オッズ | ODDS_TANPUKU | 完全対応 |
| ワイドオッズ | ODDS_WIDE | 完全対応 |
| 時系列オッズ | JODDS_TANPUKU | 完全対応（Late Money Filterに必須） |
| 払戻金（単勝・複勝・ワイド） | HARAI | 完全対応 |
| 馬マスタ（血統・成績・脚質） | UMA | 完全対応（227列・非常に詳細） |
| 騎手成績 | KISYU_SEISEKI | 完全対応（場所別・距離帯別） |
| コード体系 | CODE | 完全対応 |
| ラップタイム | RACE | 完全対応（LapTime1-25） |
| 上がりハロン | RACE + UMA_RACE | 完全対応 |

### 6.2 注意事項

| 項目 | 説明 | 対応方針 |
|---|---|---|
| 障害レースの除外 | TrackCD 51-59 は障害レース | WHERE句でフィルタ |
| 取消・出走除外の処理 | IJyoCD 1-7 は異常扱い | ラベル生成時に除外 |
| オッズ特殊値 | "9999"/"0000"/"----"/"****" | NULL/欠損として扱う |
| 複勝払戻の馬番対応 | HARAI は馬番順ではない | PayFukusyoUmaban で着順と馬番を対応させる |
| 時系列オッズの時間解像度 | HappyoTime = MMDDHHmm | 発走時刻との差分で t-Nmin を計算 |

### 6.3 設計書で使用だがEveryDB2に直接ないデータ

| データ | 説明 | 対応 |
|---|---|---|
| JV-Linkリアルタイムデータ | 実運用時のリアルタイムオッズ | EveryDB2は過去データ。実運用はJV-Link APIで別途取得 |
| PAT投票API | 投票実行 | EveryDB2外のシステム |
| 調教師成績 | 調教師の能力指標 | EveryDB2には調教師テーブルなし（※JRA-VAN DataLabにはある可能性あり） |
| 馬主情報 | 馬主の成績指標 | EveryDB2には馬主テーブルなし |

---

## 7. SQLクエリ例（学習データ取得）

### 7.1 メイン学習データの結合

```sql
-- レース情報 + 出走馬結果 + 馬マスタ + 単勝オッズ + 払戻 の結合
SELECT
    r.Year, r.MonthDay, r.JyoCD, r.Kaiji, r.Nichiji, r.RaceNum,
    r.TrackCD,
    CASE WHEN r.TrackCD BETWEEN 10 AND 22 THEN 'turf'
         WHEN r.TrackCD BETWEEN 23 AND 29 THEN 'dirt'
         ELSE 'exclude' END AS surface,
    r.Kyori AS distance,
    r.TenkoCD,
    CASE WHEN r.TrackCD <= 22 THEN r.SibaBabaCD ELSE r.DirtBabaCD END AS baba_cd,
    r.SyubetuCD, r.JyokenCD, r.GradeCD,
    r.TorokuTosu AS field_size,
    u.KakuteiJyuni AS finish_pos,
    u.Time AS finish_time,
    u.HaronTimeL3,
    u.Ninki AS popularity_rank,
    u.Odds AS win_odds_actual,
    u.BaTaijyu,
    u.ZogenFugo, u.ZogenSa,
    u.KettoNum,
    u.KisyuCode, u.ChokyosiCode,
    u.KyakusituKubun AS running_style,
    m.Kyakusitu1, m.Kyakusitu2, m.Kyakusitu3, m.Kyakusitu4,
    o.TanOdds, o.FukuOdds
FROM n_race r
JOIN n_uma_race u
    ON r.Year = u.Year
    AND r.MonthDay = u.MonthDay
    AND r.JyoCD = u.JyoCD
    AND r.Kaiji = u.Kaiji
    AND r.Nichiji = u.Nichiji
    AND r.RaceNum = u.RaceNum
LEFT JOIN n_uma m ON u.KettoNum = m.KettoNum
LEFT JOIN n_odds_tanpuku o
    ON r.Year = o.Year
    AND r.MonthDay = o.MonthDay
    AND r.JyoCD = o.JyoCD
    AND r.Kaiji = o.Kaiji
    AND r.Nichiji = o.Nichiji
    AND r.RaceNum = o.RaceNum
    AND u.Umaban = o.Umaban
WHERE r.TrackCD NOT BETWEEN 51 AND 59  -- 障害除外
    AND u.KakuteiJyuni > 0;             -- 取消・除外を除外
```

### 7.2 時系列オッズの取得（Late Money Filter用）

```sql
-- 特定レースの時系列オッズ取得
SELECT
    j.HappyoTime,
    j.Umaban,
    j.TanOdds,
    j.FukuOdds
FROM n_jodds_tanpuku j
WHERE j.Year = {year}
    AND j.MonthDay = {monthday}
    AND j.JyoCD = {jyocd}
    AND j.Kaiji = {kaiji}
    AND j.Nichiji = {nichiji}
    AND j.RaceNum = {racenum}
ORDER BY j.HappyoTime;
```

### 7.3 ワイド学習データのペア構築

```sql
-- ワイド2段階モデル用: 全馬ペア×的中フラグ×オッズ
WITH horses AS (
    SELECT Umaban, KakuteiJyuni
    FROM n_uma_race
    WHERE Year = {year} AND MonthDay = {monthday}
        AND JyoCD = {jyocd} AND Kaiji = {kaiji}
        AND Nichiji = {nichiji} AND RaceNum = {racenum}
        AND KakuteiJyuni > 0
),
pairs AS (
    SELECT
        h1.Umaban AS umaban1,
        h2.Umaban AS umaban2,
        CASE WHEN h1.KakuteiJyuni <= 3 AND h2.KakuteiJyuni <= 3
             THEN 1 ELSE 0 END AS joint_hit
    FROM horses h1
    CROSS JOIN horses h2
    WHERE h1.Umaban < h2.Umaban
)
SELECT
    p.umaban1, p.umaban2, p.joint_hit,
    w.OddsLow AS wide_odds_low,
    w.OddsHigh AS wide_odds_high,
    (w.OddsLow + w.OddsHigh) / 2.0 AS wide_odds_mid
FROM pairs p
LEFT JOIN n_odds_wide w
    ON w.Year = {year} AND w.MonthDay = {monthday}
        AND w.JyoCD = {jyocd} AND w.Kaiji = {kaiji}
        AND w.Nichiji = {nichiji} AND w.RaceNum = {racenum}
        AND w.Kumi = CONCAT(LPAD(p.umaban1,2,'0'), '-', LPAD(p.umaban2,2,'0'));
```

---

## 8. データ取得の優先順位（実装開始前に準備すべきこと）

### Step 1: 基礎データの確認（Phase A と並行）

1. EveryDB2 で PostgreSQL データベースが構築済みか確認
2. 各テーブルのカラム名・型を実際のDBと照合（EveryDB2バージョンによる差異）
3. 対象期間のデータ件数を確認（最低5年分、理想10年分）

### Step 2: 学習データのエクスポート（Phase B 開始前）

1. §7.1 のメイン結合クエリで学習用CSVをエクスポート
2. §7.2 の時系列オッズで Late Money Filter 用データをエクスポート
3. §7.3 のワイドペアクエリで WideTwoStageModel 用データをエクスポート

### Step 3: 特徴量エンジン開発時（Phase B）

1. 騎手成績テーブル (KISYU_SEISEKI) の結合
2. 過去統計 (hist_*系) の計算に必要な履歴データの準備
3. CODE テーブルによるコードデコード

### Step 4: 実運用データ取得（Phase F）

1. JV-Link API からのリアルタイムオッズ取得
2. PAT投票APIの連携（実運用時のみ）
