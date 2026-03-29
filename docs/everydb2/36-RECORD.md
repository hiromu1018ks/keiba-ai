# 12-3-36. レコードマスタ（RECORD）

**テーブル:** `RC`

**RecordSpec:** `RC`

**フィールド数:** 48

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |
|---:|:---:|---|---|---|---:|---|---|
| 1 |  | レコード種別ID | `RecordSpec` | varchar | 2 |  | RC をセットレコードフォーマットを特定する |
| 2 |  | データ区分 | `DataKubun` | varchar | 1 | 0 | 1:初期値 0:該当レコード削除(提供ミスなどの理由による) |
| 3 |  | データ作成年月日 | `MakeDate` | varchar | 8 | 0 | 西暦4桁＋月日各2桁 yyyymmdd 形式 |
| 4 | PK | レコード識別区分 | `RecInfoKubun` | varchar | 1 | 0 | 1:コースレコード 2:ＧⅠレコード |
| 5 | PK | 開催年 | `Year` | varchar | 4 | 0 | 該当レース施行年 西暦4桁 yyyy形式 |
| 6 | PK | 開催月日 | `MonthDay` | varchar | 4 | 0 | 該当レース施行月日 各2桁 mmdd形式 |
| 7 | PK | 競馬場コード | `JyoCD` | varchar | 2 | 0 | 該当レース施行競馬場 <コード表 2001.競馬場コード>参照 [コード表2001.競馬場コード&gt;](CODE.md#2001) |
| 8 | PK | 開催回[第N回] | `Kaiji` | varchar | 2 | 0 | 該当レース施行回 その競馬場でその年の何回目の開催かを示す |
| 9 | PK | 開催日目[N日目] | `Nichiji` | varchar | 2 | 0 | 該当レース施行日目 そのレース施行回で何日目の開催かを示す |
| 10 | PK | レース番号 | `RaceNum` | varchar | 2 | 0 | 該当レース番号 |
| 11 | PK | 特別競走番号 | `TokuNum` | varchar | 4 | 0 | ＧⅠレコードのみのキー |
| 12 |  | 競走名本題 | `Hondai` | varchar | 60 | Ｓ | 全角30文字 |
| 13 |  | グレードコード | `GradeCD` | varchar | 1 | sp | <コード表 2003.グレードコード>参照 [コード表2003.グレードコード&gt;](CODE.md#2003) |
| 14 | PK | 競走種別コード | `SyubetuCD` | varchar | 2 | 0 | <コード表 2005.競走種別コード>参照 [コード表2005.競走種別コード&gt;](CODE.md#2005) |
| 15 | PK | 距離 | `Kyori` | varchar | 4 | 0 | 単位:メートル |
| 16 | PK | トラックコード | `TrackCD` | varchar | 2 | 0 | <コード表 2009.トラックコード>参照 [コード表2009.トラックコード&gt;](CODE.md#2009) |
| 17 |  | レコード区分 | `RecKubun` | varchar | 1 | 0 | 1:基準タイム 2:レコードタイム 3:参考タイム 4:備考タイム |
| 18 |  | レコードタイム | `RecTime` | varchar | 4 | 0 | 9分99秒9 |
| 19 |  | 天候コード | `TenkoCD` | varchar | 1 | 0 | <コード表 2011.天候コード>参照 [コード表2011.天候コード&gt;](CODE.md#2011) |
| 20 |  | 芝馬場状態コード | `SibaBabaCD` | varchar | 1 | 0 | <コード表 2010.馬場状態コード>参照 [コード表2010.馬場状態コード&gt;](CODE.md#2010) |
| 21 |  | ダート馬場状態コード | `DirtBabaCD` | varchar | 1 | 0 | <コード表 2010.馬場状態コード>参照 [コード表2010.馬場状態コード&gt;](CODE.md#2010) |
| 22 |  | 血統登録番号1 | `RecUmaKettoNum1` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁＋数字5桁 |
| 23 |  | 馬名1 | `RecUmaBamei1` | varchar | 36 | Ｓ | 全角18文字 |
| 24 |  | 馬記号コード1 | `RecUmaUmaKigoCD1` | varchar | 2 | 0 | <コード表 2204.馬記号コード>参照 [コード表2204.馬記号コード&gt;](CODE.md#2204) |
| 25 |  | 性別コード1 | `RecUmaSexCD1` | varchar | 1 | 0 | <コード表 2202.性別コード>参照 [コード表2202.性別コード&gt;](CODE.md#2202) |
| 26 |  | 調教師コード1 | `RecUmaChokyosiCode1` | varchar | 5 | 0 |  |
| 27 |  | 調教師名1 | `RecUmaChokyosiName1` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 28 |  | 負担重量1 | `RecUmaFutan1` | varchar | 3 | 0 | 単位:0.1kg |
| 29 |  | 騎手コード1 | `RecUmaKisyuCode1` | varchar | 5 | 0 |  |
| 30 |  | 騎手名1 | `RecUmaKisyuName1` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 31 |  | 血統登録番号2 | `RecUmaKettoNum2` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁＋数字5桁 |
| 32 |  | 馬名2 | `RecUmaBamei2` | varchar | 36 | Ｓ | 全角18文字 |
| 33 |  | 馬記号コード2 | `RecUmaUmaKigoCD2` | varchar | 2 | 0 | <コード表 2204.馬記号コード>参照 [コード表2204.馬記号コード&gt;](CODE.md#2204) |
| 34 |  | 性別コード2 | `RecUmaSexCD2` | varchar | 1 | 0 | <コード表 2202.性別コード>参照 [コード表2202.性別コード&gt;](CODE.md#2202) |
| 35 |  | 調教師コード2 | `RecUmaChokyosiCode2` | varchar | 5 | 0 |  |
| 36 |  | 調教師名2 | `RecUmaChokyosiName2` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 37 |  | 負担重量2 | `RecUmaFutan2` | varchar | 3 | 0 | 単位:0.1kg |
| 38 |  | 騎手コード2 | `RecUmaKisyuCode2` | varchar | 5 | 0 |  |
| 39 |  | 騎手名2 | `RecUmaKisyuName2` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 40 |  | 血統登録番号3 | `RecUmaKettoNum3` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁＋数字5桁 |
| 41 |  | 馬名3 | `RecUmaBamei3` | varchar | 36 | Ｓ | 全角18文字 |
| 42 |  | 馬記号コード3 | `RecUmaUmaKigoCD3` | varchar | 2 | 0 | <コード表 2204.馬記号コード>参照 [コード表2204.馬記号コード&gt;](CODE.md#2204) |
| 43 |  | 性別コード3 | `RecUmaSexCD3` | varchar | 1 | 0 | <コード表 2202.性別コード>参照 [コード表2202.性別コード&gt;](CODE.md#2202) |
| 44 |  | 調教師コード3 | `RecUmaChokyosiCode3` | varchar | 5 | 0 |  |
| 45 |  | 調教師名3 | `RecUmaChokyosiName3` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
| 46 |  | 負担重量3 | `RecUmaFutan3` | varchar | 3 | 0 | 単位:0.1kg |
| 47 |  | 騎手コード3 | `RecUmaKisyuCode3` | varchar | 5 | 0 |  |
| 48 |  | 騎手名3 | `RecUmaKisyuName3` | varchar | 34 | Ｓ | 全角17文字 姓＋全角空白1文字＋名 外国人の場合は連続17文字 |
