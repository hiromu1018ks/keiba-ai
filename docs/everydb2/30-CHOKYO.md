# 12-3-30. 調教師マスタ（CHOKYO）

**テーブル:** `CH`

**RecordSpec:** `CH`

**フィールド数:** 42

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |
|---:|:---:|---|---|---|---:|---|---|
| 1 |  | レコード種別ID | `RecordSpec` | varchar | 2 |  | CH をセットレコードフォーマットを特定する |
| 2 |  | データ区分 | `DataKubun` | varchar | 1 | 0 | 1:新規登録 2:更新 0:該当レコード削除(提供ミスなどの理由による) |
| 3 |  | データ作成年月日 | `MakeDate` | varchar | 8 | 0 | 西暦4桁＋月日各2桁 yyyymmdd 形式 |
| 4 | PK | 調教師コード | `ChokyosiCode` | varchar | 5 | 0 |  |
| 5 |  | 調教師抹消区分 | `DelKubun` | varchar | 1 | 0 | 0:現役 1:抹消 |
| 6 |  | 調教師免許交付年月日 | `IssueDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 7 |  | 調教師免許抹消年月日 | `DelDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 8 |  | 生年月日 | `BirthDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 9 |  | 調教師名 | `ChokyosiName` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 10 |  | 調教師名半角ｶﾅ | `ChokyosiNameKana` | varchar | 30 | sp | 半角30文字 姓15文字＋名15文字 外国人の場合は連続30文字 |
| 11 |  | 調教師名略称 | `ChokyosiRyakusyo` | varchar | 8 | Ｓ | 全角4文字 |
| 12 |  | 調教師名欧字 | `ChokyosiNameEng` | varchar | 80 | sp | 半角80文字 姓＋半角空白1文字＋名 フルネームで記載 |
| 13 |  | 性別区分 | `SexCD` | varchar | 1 | 0 | 1:男性 2:女性 |
| 14 |  | 調教師東西所属コード | `TozaiCD` | varchar | 1 | 0 | <コード表 2301.東西所属コード>参照 [コード表2301.東西所属コード&gt;](CODE.md#2301) |
| 15 |  | 招待地域名 | `Syotai` | varchar | 20 | Ｓ | 全角10文字 |
| 16 |  | 最近重賞勝利1_年月日場回日R | `SaikinJyusyo1SaikinJyusyoid` | varchar | 16 | 0 | レース詳細のキー情報 |
| 17 |  | 最近重賞勝利1_競走名本題 | `SaikinJyusyo1Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 18 |  | 最近重賞勝利1_競走名略称10字 | `SaikinJyusyo1Ryakusyo10` | varchar | 20 | Ｓ | 全角10文字 |
| 19 |  | 最近重賞勝利1_競走名略称6字 | `SaikinJyusyo1Ryakusyo6` | varchar | 12 | Ｓ | 全角6文字 |
| 20 |  | 最近重賞勝利1_競走名略称3字 | `SaikinJyusyo1Ryakusyo3` | varchar | 6 | Ｓ | 全角3文字 |
| 21 |  | 最近重賞勝利1_グレードコード | `SaikinJyusyo1GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 22 |  | 最近重賞勝利1_出走頭数 | `SaikinJyusyo1SyussoTosu` | varchar | 2 | 0 | 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 23 |  | 最近重賞勝利1_血統登録番号 | `SaikinJyusyo1KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 24 |  | 最近重賞勝利1_馬名 | `SaikinJyusyo1Bamei` | varchar | 36 | Ｓ | 全角18文字 |
| 25 |  | 最近重賞勝利2_年月日場回日R | `SaikinJyusyo2SaikinJyusyoid` | varchar | 16 | 0 | レース詳細のキー情報 |
| 26 |  | 最近重賞勝利2_競走名本題 | `SaikinJyusyo2Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 27 |  | 最近重賞勝利2_競走名略称10字 | `SaikinJyusyo2Ryakusyo10` | varchar | 20 | Ｓ | 全角10文字 |
| 28 |  | 最近重賞勝利2_競走名略称6字 | `SaikinJyusyo2Ryakusyo6` | varchar | 12 | Ｓ | 全角6文字 |
| 29 |  | 最近重賞勝利2_競走名略称3字 | `SaikinJyusyo2Ryakusyo3` | varchar | 6 | Ｓ | 全角3文字 |
| 30 |  | 最近重賞勝利2_グレードコード | `SaikinJyusyo2GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 31 |  | 最近重賞勝利2_出走頭数 | `SaikinJyusyo2SyussoTosu` | varchar | 2 | 0 | 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 32 |  | 最近重賞勝利2_血統登録番号 | `SaikinJyusyo2KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 33 |  | 最近重賞勝利2_馬名 | `SaikinJyusyo2Bamei` | varchar | 36 | Ｓ | 全角18文字 |
| 34 |  | 最近重賞勝利3_年月日場回日R | `SaikinJyusyo3SaikinJyusyoid` | varchar | 16 | 0 | レース詳細のキー情報 |
| 35 |  | 最近重賞勝利3_競走名本題 | `SaikinJyusyo3Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 36 |  | 最近重賞勝利3_競走名略称10字 | `SaikinJyusyo3Ryakusyo10` | varchar | 20 | Ｓ | 全角10文字 |
| 37 |  | 最近重賞勝利3_競走名略称6字 | `SaikinJyusyo3Ryakusyo6` | varchar | 12 | Ｓ | 全角6文字 |
| 38 |  | 最近重賞勝利3_競走名略称3字 | `SaikinJyusyo3Ryakusyo3` | varchar | 6 | Ｓ | 全角3文字 |
| 39 |  | 最近重賞勝利3_グレードコード | `SaikinJyusyo3GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 40 |  | 最近重賞勝利3_出走頭数 | `SaikinJyusyo3SyussoTosu` | varchar | 2 | 0 | 登録頭数から出走取消と競走除外･発走除外を除いた頭数 |
| 41 |  | 最近重賞勝利3_血統登録番号 | `SaikinJyusyo3KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 42 |  | 最近重賞勝利3_馬名 | `SaikinJyusyo3Bamei` | varchar | 36 | Ｓ | 全角18文字 |
