# 12-3-28. 騎手マスタ（KISYU）

**テーブル:** `KS`

**RecordSpec:** `KS`

**フィールド数:** 67

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |
|---:|:---:|---|---|---|---:|---|---|
| 1 |  | レコード種別ID | `RecordSpec` | varchar | 2 |  | KS をセットレコードフォーマットを特定する |
| 2 |  | データ区分 | `DataKubun` | varchar | 1 | 0 | 1:新規登録 2:更新 0:該当レコード削除(提供ミスなどの理由による) |
| 3 |  | データ作成年月日 | `MakeDate` | varchar | 8 | 0 | 西暦4桁＋月日各2桁 yyyymmdd 形式 |
| 4 | PK | 騎手コード | `KisyuCode` | varchar | 5 | 0 |  |
| 5 |  | 騎手抹消区分 | `DelKubun` | varchar | 1 | 0 | 0:現役 1:抹消 |
| 6 |  | 騎手免許交付年月日 | `IssueDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 7 |  | 騎手免許抹消年月日 | `DelDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 8 |  | 生年月日 | `BirthDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 9 |  | 騎手名 | `KisyuName` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 10 |  | 予備 | `reserved` | varchar | 34 | Ｓ |  |
| 11 |  | 騎手名半角ｶﾅ | `KisyuNameKana` | varchar | 30 | sp | 半角30文字 姓15文字＋名15文字 外国人の場合は連続30文字 |
| 12 |  | 騎手名略称 | `KisyuRyakusyo` | varchar | 8 | Ｓ | 全角4文字 |
| 13 |  | 騎手名欧字 | `KisyuNameEng` | varchar | 80 | sp | 半角80文字 姓＋半角空白1文字＋名 フルネームで記載 |
| 14 |  | 性別区分 | `SexCD` | varchar | 1 | 0 | 1:男性 2:女性 |
| 15 |  | 騎乗資格コード | `SikakuCD` | varchar | 1 | 0 | <コード表 2302.騎乗資格コード>参照 [コード表2302.騎乗資格コード&gt;](CODE.md#2302) |
| 16 |  | 騎手見習コード | `MinaraiCD` | varchar | 1 | 0 | <コード表 2303.騎手見習コード>参照 [コード表2303.騎手見習コード&gt;](CODE.md#2303) |
| 17 |  | 騎手東西所属コード | `TozaiCD` | varchar | 1 | 0 | <コード表 2301.東西所属コード>参照 [コード表2301.東西所属コード&gt;](CODE.md#2301) |
| 18 |  | 招待地域名 | `Syotai` | varchar | 20 | Ｓ | 全角10文字 |
| 19 |  | 所属調教師コード | `ChokyosiCode` | varchar | 5 | 0 | 騎手の所属厩舎の調教師コード、フリー騎手の場合はALL0を設定 |
| 20 |  | 所属調教師名略称 | `ChokyosiRyakusyo` | varchar | 8 | Ｓ | 全角4文字 |
| 21 |  | 初騎乗1_年月日場回日R | `HatuKiJyo1Hatukijyoid` | varchar | 16 | 0 | 平地初騎乗 レース詳細のキー情報 |
| 22 |  | 初騎乗1_出走頭数 | `HatuKiJyo1SyussoTosu` | varchar | 2 | 0 | 平地初騎乗 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 23 |  | 初騎乗1_血統登録番号 | `HatuKiJyo1KettoNum` | varchar | 10 | 0 | 平地初騎乗 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 24 |  | 初騎乗1_馬名 | `HatuKiJyo1Bamei` | varchar | 36 | Ｓ | 平地初騎乗 全角18文字 |
| 25 |  | 初騎乗1_確定着順 | `HatuKiJyo1KakuteiJyuni` | varchar | 2 | 0 | 平地初騎乗 |
| 26 |  | 初騎乗1_異常区分コード | `HatuKiJyo1IJyoCD` | varchar | 1 | 0 | 平地初騎乗 <コード表 2101.異常区分コード>参照 [コード表2101.異常区分コード&gt;](CODE.md#2101) |
| 27 |  | 初騎乗2_年月日場回日R | `HatuKiJyo2Hatukijyoid` | varchar | 16 | 0 | 障害初騎乗 レース詳細のキー情報 |
| 28 |  | 初騎乗2_出走頭数 | `HatuKiJyo2SyussoTosu` | varchar | 2 | 0 | 障害初騎乗 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 29 |  | 初騎乗2_血統登録番号 | `HatuKiJyo2KettoNum` | varchar | 10 | 0 | 障害初騎乗 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 30 |  | 初騎乗2_馬名 | `HatuKiJyo2Bamei` | varchar | 36 | Ｓ | 障害初騎乗 全角18文字 |
| 31 |  | 初騎乗2_確定着順 | `HatuKiJyo2KakuteiJyuni` | varchar | 2 | 0 | 障害初騎乗 |
| 32 |  | 初騎乗2_異常区分コード | `HatuKiJyo2IJyoCD` | varchar | 1 | 0 | 障害初騎乗 <コード表 2101.異常区分コード>参照 [コード表2101.異常区分コード&gt;](CODE.md#2101) |
| 33 |  | 初勝利1_年月日場回日R | `HatuSyori1Hatusyoriid` | varchar | 16 | 0 | 平地初勝利 レース詳細のキー情報 |
| 34 |  | 初勝利1_出走頭数 | `HatuSyori1SyussoTosu` | varchar | 2 | 0 | 平地初勝利 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 35 |  | 初勝利1_血統登録番号 | `HatuSyori1KettoNum` | varchar | 10 | 0 | 平地初勝利 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 36 |  | 初勝利1_馬名 | `HatuSyori1Bamei` | varchar | 36 | Ｓ | 平地初勝利 全角18文字 |
| 37 |  | 初勝利2_年月日場回日R | `HatuSyori2Hatusyoriid` | varchar | 16 | 0 | 障害初勝利 レース詳細のキー情報 |
| 38 |  | 初勝利2_出走頭数 | `HatuSyori2SyussoTosu` | varchar | 2 | 0 | 障害初勝利 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 39 |  | 初勝利2_血統登録番号 | `HatuSyori2KettoNum` | varchar | 10 | 0 | 障害初勝利 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 40 |  | 初勝利2_馬名 | `HatuSyori2Bamei` | varchar | 36 | Ｓ | 障害初勝利 全角18文字 |
| 41 |  | 最近重賞勝利1_年月日場回日R | `SaikinJyusyo1SaikinJyusyoid` | varchar | 16 | 0 | レース詳細のキー情報 |
| 42 |  | 最近重賞勝利1_競走名本題 | `SaikinJyusyo1Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 43 |  | 最近重賞勝利1_競走名略称10字 | `SaikinJyusyo1Ryakusyo10` | varchar | 20 | Ｓ | 全角10文字 |
| 44 |  | 最近重賞勝利1_競走名略称6字 | `SaikinJyusyo1Ryakusyo6` | varchar | 12 | Ｓ | 全角6文字 |
| 45 |  | 最近重賞勝利1_競走名略称3字 | `SaikinJyusyo1Ryakusyo3` | varchar | 6 | Ｓ | 全角3文字 |
| 46 |  | 最近重賞勝利1_グレードコード | `SaikinJyusyo1GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 47 |  | 最近重賞勝利1_出走頭数 | `SaikinJyusyo1SyussoTosu` | varchar | 2 | 0 | 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 48 |  | 最近重賞勝利1_血統登録番号 | `SaikinJyusyo1KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 49 |  | 最近重賞勝利1_馬名 | `SaikinJyusyo1Bamei` | varchar | 36 | Ｓ | 全角18文字 |
| 50 |  | 最近重賞勝利2_年月日場回日R | `SaikinJyusyo2SaikinJyusyoid` | varchar | 16 | 0 | レース詳細のキー情報 |
| 51 |  | 最近重賞勝利2_競走名本題 | `SaikinJyusyo2Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 52 |  | 最近重賞勝利2_競走名略称10字 | `SaikinJyusyo2Ryakusyo10` | varchar | 20 | Ｓ | 全角10文字 |
| 53 |  | 最近重賞勝利2_競走名略称6字 | `SaikinJyusyo2Ryakusyo6` | varchar | 12 | Ｓ | 全角6文字 |
| 54 |  | 最近重賞勝利2_競走名略称3字 | `SaikinJyusyo2Ryakusyo3` | varchar | 6 | Ｓ | 全角3文字 |
| 55 |  | 最近重賞勝利2_グレードコード | `SaikinJyusyo2GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 56 |  | 最近重賞勝利2_出走頭数 | `SaikinJyusyo2SyussoTosu` | varchar | 2 | 0 | 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 57 |  | 最近重賞勝利2_血統登録番号 | `SaikinJyusyo2KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 58 |  | 最近重賞勝利2_馬名 | `SaikinJyusyo2Bamei` | varchar | 36 | Ｓ | 全角18文字 |
| 59 |  | 最近重賞勝利3_年月日場回日R | `SaikinJyusyo3SaikinJyusyoid` | varchar | 16 | 0 | レース詳細のキー情報 |
| 60 |  | 最近重賞勝利3_競走名本題 | `SaikinJyusyo3Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 61 |  | 最近重賞勝利3_競走名略称10字 | `SaikinJyusyo3Ryakusyo10` | varchar | 20 | Ｓ | 全角10文字 |
| 62 |  | 最近重賞勝利3_競走名略称6字 | `SaikinJyusyo3Ryakusyo6` | varchar | 12 | Ｓ | 全角6文字 |
| 63 |  | 最近重賞勝利3_競走名略称3字 | `SaikinJyusyo3Ryakusyo3` | varchar | 6 | Ｓ | 全角3文字 |
| 64 |  | 最近重賞勝利3_グレードコード | `SaikinJyusyo3GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 65 |  | 最近重賞勝利3_出走頭数 | `SaikinJyusyo3SyussoTosu` | varchar | 2 | 0 | 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 66 |  | 最近重賞勝利3_血統登録番号 | `SaikinJyusyo3KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 67 |  | 最近重賞勝利3_馬名 | `SaikinJyusyo3Bamei` | varchar | 36 | Ｓ | 全角18文字 |
