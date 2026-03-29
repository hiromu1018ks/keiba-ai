# 12-3-53. 系統情報（KEITO）

**テーブル:** `BT`

**RecordSpec:** `BT`

**フィールド数:** 7

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |
|---:|:---:|---|---|---|---:|---|---|
| 1 |  | レコード種別ID | `RecordSpec` | varchar | 2 | sp | BTをセットレコードフォーマットを特定する |
| 2 |  | データ区分 | `DataKubun` | varchar | 1 | 0 | 1:新規登録 2:更新 0:該当レコード削除(提供ミスなどの理由による) |
| 3 |  | データ作成年月日 | `MakeDate` | varchar | 8 | 0 | 西暦4桁＋月日各2桁 yyyymmdd 形式 |
| 4 | PK | 繁殖登録番号 | `HansyokuNum` | varchar | 10 | 0 | ‐ |
| 5 |  | 系統ID | `KeitoId` | varchar | 30 | sp | 2桁ごとに系譜を表現するID。詳しくはJV-DATA仕様書を参照 |
| 6 |  | 系統名 | `KeitoName` | varchar | 36 | Ｓ | サンデーサイレンス系など、その系統の名称 |
| 7 |  | 系統説明 | `KeitoEx` | varchar | 6800 | sp | テキスト文 |
