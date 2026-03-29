# 12-3-42. 発走時刻変更（HASSOU_JIKOKU_CHANGE）

**テーブル:** `TC`

**RecordSpec:** `TC`

**フィールド数:** 14

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |
|---:|:---:|---|---|---|---:|---|---|
| 1 |  | レコード種別ID | `RecordSpec` | varchar | 2 |  | TC をセットレコードフォーマットを特定する |
| 2 |  | データ区分 | `DataKubun` | varchar | 1 | 0 | 1:初期値 |
| 3 |  | データ作成年月日 | `MakeDate` | varchar | 8 | 0 | 西暦4桁＋月日各2桁 yyyymmdd 形式 |
| 4 | PK | 開催年 | `Year` | varchar | 4 | 0 | 該当レース施行年 西暦4桁 yyyy形式 |
| 5 | PK | 開催月日 | `MonthDay` | varchar | 4 | 0 | 該当レース施行月日 各2桁 mmdd形式 |
| 6 | PK | 競馬場コード | `JyoCD` | varchar | 2 | 0 | 該当レース施行競馬場 <コード表 2001.競馬場コード>参照 [コード表2001.競馬場コード&gt;](CODE.md#2001) |
| 7 | PK | 開催回[第N回] | `Kaiji` | varchar | 2 | 0 | 該当レース施行回 その競馬場でその年の何回目の開催かを示す |
| 8 | PK | 開催日目[N日目] | `Nichiji` | varchar | 2 | 0 | 該当レース施行日目 そのレース施行回で何日目の開催かを示す |
| 9 | PK | レース番号 | `RaceNum` | varchar | 2 | 0 | 該当レース番号 |
| 10 | PK | 発表月日時分 | `HappyoTime` | varchar | 8 | 0 | 月日時分各2桁 |
| 11 |  | 変更後情報_時 | `AtoJi` | varchar | 2 | 0 | 時 hh形式 |
| 12 |  | 変更後情報_分 | `AtoFun` | varchar | 2 | 0 | 分 mm形式 |
| 13 |  | 変更前情報_時 | `MaeJi` | varchar | 2 | 0 | 時 hh形式 |
| 14 |  | 変更前情報_分 | `MaeFun` | varchar | 2 | 0 | 分 mm形式 |
