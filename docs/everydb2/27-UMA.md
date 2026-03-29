# 12-3-27. 競走馬マスタ（UMA）

**テーブル:** `UM`

**RecordSpec:** `UM`

**フィールド数:** 227

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明 |
|---:|:---:|---|---|---|---:|---|---|
| 1 |  | レコード種別ID | `RecordSpec` | varchar | 2 |  | UM をセットレコードフォーマットを特定する |
| 2 |  | データ区分 | `DataKubun` | varchar | 1 | 0 | 1:新規馬名登録 2:馬名変更 3:再登録(抹消後の再登録) 4:その他更新 9:抹消 0:該当レコード削除(提供ミスなどの理由による) |
| 3 |  | データ作成年月日 | `MakeDate` | varchar | 8 | 0 | 西暦4桁＋月日各2桁 yyyymmdd 形式 |
| 4 | PK | 血統登録番号 | `KettoNum` | varchar | 10 | 0 | 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 5 |  | 競走馬抹消区分 | `DelKubun` | varchar | 1 | 0 | 0:現役 1:抹消 |
| 6 |  | 競走馬登録年月日 | `RegDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 7 |  | 競走馬抹消年月日 | `DelDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 8 |  | 生年月日 | `BirthDate` | varchar | 8 | 0 | 年4桁(西暦)＋月日各2桁 yyyymmdd 形式 |
| 9 |  | 馬名 | `Bamei` | varchar | 36 | Ｓ | 全角18文字 |
| 10 |  | 馬名半角ｶﾅ | `BameiKana` | varchar | 36 | sp | 半角36文字 |
| 11 |  | 馬名欧字 | `BameiEng` | varchar | 60 | sp | 半角60文字 |
| 12 |  | JRA施設在きゅうフラグ | `ZaikyuFlag` | varchar | 1 | sp | 0:JRA施設に在きゅうしていない。 1:JRA施設の在きゅうしている。 JRA施設とは競馬場およびトレセンなどを指す。 (平成18年6月6日以降設定) |
| 13 |  | 予備 | `Reserved` | varchar | 19 | sp | 予備 |
| 14 |  | 馬記号コード | `UmaKigoCD` | varchar | 2 | 0 | <コード表 2204.馬記号コード>参照 [コード表2204.馬記号コード&gt;](CODE.md#2204) |
| 15 |  | 性別コード | `SexCD` | varchar | 1 | 0 | <コード表 2202.性別コード>参照 [コード表2202.性別コード&gt;](CODE.md#2202) |
| 16 |  | 品種コード | `HinsyuCD` | varchar | 1 | 0 | <コード表 2201.品種コード>参照 [コード表2201.品種コード&gt;](CODE.md#2201) |
| 17 |  | 毛色コード | `KeiroCD` | varchar | 2 | 0 | <コード表 2203.毛色コード>参照 [コード表2203.毛色コード&gt;](CODE.md#2203) |
| 18 |  | 3代血統情報_繁殖登録番号1 | `Ketto3InfoHansyokuNum1` | varchar | 10 | 0 | (父馬)繁殖馬マスタにリンク |
| 19 |  | 3代血統情報_馬名1 | `Ketto3InfoBamei1` | varchar | 36 | Ｓ sp | (父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 20 |  | 3代血統情報_繁殖登録番号2 | `Ketto3InfoHansyokuNum2` | varchar | 10 | 0 | (母馬)繁殖馬マスタにリンク |
| 21 |  | 3代血統情報_馬名2 | `Ketto3InfoBamei2` | varchar | 36 | Ｓ sp | (母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 22 |  | 3代血統情報_繁殖登録番号3 | `Ketto3InfoHansyokuNum3` | varchar | 10 | 0 | (父父馬)繁殖馬マスタにリンク |
| 23 |  | 3代血統情報_馬名3 | `Ketto3InfoBamei3` | varchar | 36 | Ｓ sp | (父父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 24 |  | 3代血統情報_繁殖登録番号4 | `Ketto3InfoHansyokuNum4` | varchar | 10 | 0 | (父母馬)繁殖馬マスタにリンク |
| 25 |  | 3代血統情報_馬名4 | `Ketto3InfoBamei4` | varchar | 36 | Ｓ sp | (父母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 26 |  | 3代血統情報_繁殖登録番号5 | `Ketto3InfoHansyokuNum5` | varchar | 10 | 0 | (母父馬)繁殖馬マスタにリンク |
| 27 |  | 3代血統情報_馬名5 | `Ketto3InfoBamei5` | varchar | 36 | Ｓ sp | (母父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 28 |  | 3代血統情報_繁殖登録番号6 | `Ketto3InfoHansyokuNum6` | varchar | 10 | 0 | (母母馬)繁殖馬マスタにリンク |
| 29 |  | 3代血統情報_馬名6 | `Ketto3InfoBamei6` | varchar | 36 | Ｓ sp | (母母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 30 |  | 3代血統情報_繁殖登録番号7 | `Ketto3InfoHansyokuNum7` | varchar | 10 | 0 | (父父父馬)繁殖馬マスタにリンク |
| 31 |  | 3代血統情報_馬名7 | `Ketto3InfoBamei7` | varchar | 36 | Ｓ sp | (父父父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 32 |  | 3代血統情報_繁殖登録番号8 | `Ketto3InfoHansyokuNum8` | varchar | 10 | 0 | (父父母馬)繁殖馬マスタにリンク |
| 33 |  | 3代血統情報_馬名8 | `Ketto3InfoBamei8` | varchar | 36 | Ｓ sp | (父父母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 34 |  | 3代血統情報_繁殖登録番号9 | `Ketto3InfoHansyokuNum9` | varchar | 10 | 0 | (父母父馬)繁殖馬マスタにリンク |
| 35 |  | 3代血統情報_馬名9 | `Ketto3InfoBamei9` | varchar | 36 | Ｓ sp | (父母父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 36 |  | 3代血統情報_繁殖登録番号10 | `Ketto3InfoHansyokuNum10` | varchar | 10 | 0 | (父母母馬)繁殖馬マスタにリンク |
| 37 |  | 3代血統情報_馬名10 | `Ketto3InfoBamei10` | varchar | 36 | Ｓ sp | (父母母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 38 |  | 3代血統情報_繁殖登録番号11 | `Ketto3InfoHansyokuNum11` | varchar | 10 | 0 | (母父父馬)繁殖馬マスタにリンク |
| 39 |  | 3代血統情報_馬名11 | `Ketto3InfoBamei11` | varchar | 36 | Ｓ sp | (母父父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 40 |  | 3代血統情報_繁殖登録番号12 | `Ketto3InfoHansyokuNum12` | varchar | 10 | 0 | (母父母馬)繁殖馬マスタにリンク |
| 41 |  | 3代血統情報_馬名12 | `Ketto3InfoBamei12` | varchar | 36 | Ｓ sp | (母父母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 42 |  | 3代血統情報_繁殖登録番号13 | `Ketto3InfoHansyokuNum13` | varchar | 10 | 0 | (母母父馬)繁殖馬マスタにリンク |
| 43 |  | 3代血統情報_馬名13 | `Ketto3InfoBamei13` | varchar | 36 | Ｓ sp | (母母父馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 44 |  | 3代血統情報_繁殖登録番号14 | `Ketto3InfoHansyokuNum14` | varchar | 10 | 0 | (母母母馬)繁殖馬マスタにリンク |
| 45 |  | 3代血統情報_馬名14 | `Ketto3InfoBamei14` | varchar | 36 | Ｓ sp | (母母母馬)全角18文字 ～ 半角36文字 （全角と半角が混在） 外国の繁殖馬の場合は、16.繁殖馬マスタの10.馬名欧字の頭36バイトを設定。 |
| 46 |  | 東西所属コード | `TozaiCD` | varchar | 1 | 0 | <コード表 2301.東西所属コード>参照 [コード表2301.東西所属コード&gt;](CODE.md#2301) |
| 47 |  | 調教師コード | `ChokyosiCode` | varchar | 5 | 0 | 調教師マスタへリンク |
| 48 |  | 調教師名略称 | `ChokyosiRyakusyo` | varchar | 8 | Ｓ | 全角4文字 |
| 49 |  | 招待地域名 | `Syotai` | varchar | 20 | Ｓ | 全角10文字 |
| 50 |  | 生産者コード | `BreederCode` | varchar | 8 | 0 | 生産者マスタへリンク |
| 51 |  | 生産者名 | `BreederName` | varchar | 72 | Ｓ sp | 全角35文字 ～ 半角70文字 （全角と半角が混在） 株式会社、有限会社などの法人格を示す文字列が頭もしくは末尾にある場合にそれを削除したものを設定 また、外国生産者の場合は、１４.生産者マスタの8.生産者名欧字の頭70バイトを設定。 |
| 52 |  | 産地名 | `SanchiName` | varchar | 20 | Ｓ sp | 全角10文字 または 半角20文字 (設定値が英数の場合は半角で設定） |
| 53 |  | 馬主コード | `BanusiCode` | varchar | 6 | 0 | 馬主マスタへリンク |
| 54 |  | 馬主名 | `BanusiName` | varchar | 64 | Ｓ sp | 全角32文字 ～ 半角64文字 （全角と半角が混在） 株式会社、有限会社などの法人格を示す文字列が頭もしくは末尾にある場合にそれを削除したものを設定 また、外国馬主の場合は、１５.馬主マスタの8.馬主名欧字の頭64バイトを設定。 |
| 55 |  | 平地本賞金累計 | `RuikeiHonsyoHeiti` | varchar | 9 | 0 | 単位：百円 （中央の平地本賞金の合計） |
| 56 |  | 障害本賞金累計 | `RuikeiHonsyoSyogai` | varchar | 9 | 0 | 単位：百円 （中央の障害本賞金の合計） |
| 57 |  | 平地付加賞金累計 | `RuikeiFukaHeichi` | varchar | 9 | 0 | 単位：百円 （中央の平地付加賞金の合計） |
| 58 |  | 障害付加賞金累計 | `RuikeiFukaSyogai` | varchar | 9 | 0 | 単位：百円 （中央の障害付加賞金の合計） |
| 59 |  | 平地収得賞金累計 | `RuikeiSyutokuHeichi` | varchar | 9 | 0 | 単位：百円 （中央＋中央以外の平地累積収得賞金） 4歳夏季競馬以降は4歳春季競馬までに獲得した収得賞金について2分の1としたものを設定する。 |
| 60 |  | 障害収得賞金累計 | `RuikeiSyutokuSyogai` | varchar | 9 | 0 | 単位：百円 （中央＋中央以外の障害累積収得賞金） |
| 61 |  | 総合着回数1 | `SogoChakukaisu1` | varchar | 3 | 0 | 1着の回数（中央＋地方＋海外) |
| 62 |  | 総合着回数2 | `SogoChakukaisu2` | varchar | 3 | 0 | 2着の回数（中央＋地方＋海外) |
| 63 |  | 総合着回数3 | `SogoChakukaisu3` | varchar | 3 | 0 | 3着の回数（中央＋地方＋海外) |
| 64 |  | 総合着回数4 | `SogoChakukaisu4` | varchar | 3 | 0 | 4着の回数（中央＋地方＋海外) |
| 65 |  | 総合着回数5 | `SogoChakukaisu5` | varchar | 3 | 0 | 5着の回数（中央＋地方＋海外) |
| 66 |  | 総合着回数6 | `SogoChakukaisu6` | varchar | 3 | 0 | 着外の回数（中央＋地方＋海外) |
| 67 |  | 中央合計着回数1 | `ChuoChakukaisu1` | varchar | 3 | 0 | 1着の回数（中央のみ) |
| 68 |  | 中央合計着回数2 | `ChuoChakukaisu2` | varchar | 3 | 0 | 2着の回数（中央のみ) |
| 69 |  | 中央合計着回数3 | `ChuoChakukaisu3` | varchar | 3 | 0 | 3着の回数（中央のみ) |
| 70 |  | 中央合計着回数4 | `ChuoChakukaisu4` | varchar | 3 | 0 | 4着の回数（中央のみ) |
| 71 |  | 中央合計着回数5 | `ChuoChakukaisu5` | varchar | 3 | 0 | 5着の回数（中央のみ) |
| 72 |  | 中央合計着回数6 | `ChuoChakukaisu6` | varchar | 3 | 0 | 着外の回数（中央のみ) |
| 73 |  | 芝直・着回数1 | `Ba1Chakukaisu1` | varchar | 3 | 0 | 芝・直線コースでの1着の回数（中央のみ) |
| 74 |  | 芝直・着回数2 | `Ba1Chakukaisu2` | varchar | 3 | 0 | 芝・直線コースでの2着の回数（中央のみ) |
| 75 |  | 芝直・着回数3 | `Ba1Chakukaisu3` | varchar | 3 | 0 | 芝・直線コースでの3着の回数（中央のみ) |
| 76 |  | 芝直・着回数4 | `Ba1Chakukaisu4` | varchar | 3 | 0 | 芝・直線コースでの4着の回数（中央のみ) |
| 77 |  | 芝直・着回数5 | `Ba1Chakukaisu5` | varchar | 3 | 0 | 芝・直線コースでの5着の回数（中央のみ) |
| 78 |  | 芝直・着回数6 | `Ba1Chakukaisu6` | varchar | 3 | 0 | 芝・直線コースでの着外の回数（中央のみ) |
| 79 |  | 芝右・着回数1 | `Ba2Chakukaisu1` | varchar | 3 | 0 | 芝・右回りコースでの1着の回数（中央のみ) |
| 80 |  | 芝右・着回数2 | `Ba2Chakukaisu2` | varchar | 3 | 0 | 芝・右回りコースでの2着の回数（中央のみ) |
| 81 |  | 芝右・着回数3 | `Ba2Chakukaisu3` | varchar | 3 | 0 | 芝・右回りコースでの3着の回数（中央のみ) |
| 82 |  | 芝右・着回数4 | `Ba2Chakukaisu4` | varchar | 3 | 0 | 芝・右回りコースでの4着の回数（中央のみ) |
| 83 |  | 芝右・着回数5 | `Ba2Chakukaisu5` | varchar | 3 | 0 | 芝・右回りコースでの5着の回数（中央のみ) |
| 84 |  | 芝右・着回数6 | `Ba2Chakukaisu6` | varchar | 3 | 0 | 芝・右回りコースでの着外の回数（中央のみ) |
| 85 |  | 芝左・着回数1 | `Ba3Chakukaisu1` | varchar | 3 | 0 | 芝・左回りコースでの1着の回数（中央のみ) |
| 86 |  | 芝左・着回数2 | `Ba3Chakukaisu2` | varchar | 3 | 0 | 芝・左回りコースでの2着の回数（中央のみ) |
| 87 |  | 芝左・着回数3 | `Ba3Chakukaisu3` | varchar | 3 | 0 | 芝・左回りコースでの3着の回数（中央のみ) |
| 88 |  | 芝左・着回数4 | `Ba3Chakukaisu4` | varchar | 3 | 0 | 芝・左回りコースでの4着の回数（中央のみ) |
| 89 |  | 芝左・着回数5 | `Ba3Chakukaisu5` | varchar | 3 | 0 | 芝・左回りコースでの5着の回数（中央のみ) |
| 90 |  | 芝左・着回数6 | `Ba3Chakukaisu6` | varchar | 3 | 0 | 芝・左回りコースでの着外の回数（中央のみ) |
| 91 |  | ダ直・着回数1 | `Ba4Chakukaisu1` | varchar | 3 | 0 | ダート・直線コースでの1着の回数（中央のみ) |
| 92 |  | ダ直・着回数2 | `Ba4Chakukaisu2` | varchar | 3 | 0 | ダート・直線コースでの2着の回数（中央のみ) |
| 93 |  | ダ直・着回数3 | `Ba4Chakukaisu3` | varchar | 3 | 0 | ダート・直線コースでの3着の回数（中央のみ) |
| 94 |  | ダ直・着回数4 | `Ba4Chakukaisu4` | varchar | 3 | 0 | ダート・直線コースでの4着の回数（中央のみ) |
| 95 |  | ダ直・着回数5 | `Ba4Chakukaisu5` | varchar | 3 | 0 | ダート・直線コースでの5着の回数（中央のみ) |
| 96 |  | ダ直・着回数6 | `Ba4Chakukaisu6` | varchar | 3 | 0 | ダート・直線コースでの着外の回数（中央のみ) |
| 97 |  | ダ右・着回数1 | `Ba5Chakukaisu1` | varchar | 3 | 0 | ダート・右回りコースでの1着の回数（中央のみ) |
| 98 |  | ダ右・着回数2 | `Ba5Chakukaisu2` | varchar | 3 | 0 | ダート・右回りコースでの2着の回数（中央のみ) |
| 99 |  | ダ右・着回数3 | `Ba5Chakukaisu3` | varchar | 3 | 0 | ダート・右回りコースでの3着の回数（中央のみ) |
| 100 |  | ダ右・着回数4 | `Ba5Chakukaisu4` | varchar | 3 | 0 | ダート・右回りコースでの4着の回数（中央のみ) |
| 101 |  | ダ右・着回数5 | `Ba5Chakukaisu5` | varchar | 3 | 0 | ダート・右回りコースでの5着の回数（中央のみ) |
| 102 |  | ダ右・着回数6 | `Ba5Chakukaisu6` | varchar | 3 | 0 | ダート・右回りコースでの着外の回数（中央のみ) |
| 103 |  | ダ左・着回数1 | `Ba6Chakukaisu1` | varchar | 3 | 0 | ダート・左回りコースでの1着の回数（中央のみ) |
| 104 |  | ダ左・着回数2 | `Ba6Chakukaisu2` | varchar | 3 | 0 | ダート・左回りコースでの2着の回数（中央のみ) |
| 105 |  | ダ左・着回数3 | `Ba6Chakukaisu3` | varchar | 3 | 0 | ダート・左回りコースでの3着の回数（中央のみ) |
| 106 |  | ダ左・着回数4 | `Ba6Chakukaisu4` | varchar | 3 | 0 | ダート・左回りコースでの4着の回数（中央のみ) |
| 107 |  | ダ左・着回数5 | `Ba6Chakukaisu5` | varchar | 3 | 0 | ダート・左回りコースでの5着の回数（中央のみ) |
| 108 |  | ダ左・着回数6 | `Ba6Chakukaisu6` | varchar | 3 | 0 | ダート・左回りコースでの着外の回数（中央のみ) |
| 109 |  | 障害・着回数1 | `Ba7Chakukaisu1` | varchar | 3 | 0 | 障害レースでの1着の回数（中央のみ) |
| 110 |  | 障害・着回数2 | `Ba7Chakukaisu2` | varchar | 3 | 0 | 障害レースでの2着の回数（中央のみ) |
| 111 |  | 障害・着回数3 | `Ba7Chakukaisu3` | varchar | 3 | 0 | 障害レースでの3着の回数（中央のみ) |
| 112 |  | 障害・着回数4 | `Ba7Chakukaisu4` | varchar | 3 | 0 | 障害レースでの4着の回数（中央のみ) |
| 113 |  | 障害・着回数5 | `Ba7Chakukaisu5` | varchar | 3 | 0 | 障害レースでの5着の回数（中央のみ) |
| 114 |  | 障害・着回数6 | `Ba7Chakukaisu6` | varchar | 3 | 0 | 障害レースでの着外の回数（中央のみ) |
| 115 |  | 芝良・着回数1 | `Jyotai1Chakukaisu1` | varchar | 3 | 0 | 芝・良馬場での1着の回数（中央のみ) |
| 116 |  | 芝良・着回数2 | `Jyotai1Chakukaisu2` | varchar | 3 | 0 | 芝・良馬場での2着の回数（中央のみ) |
| 117 |  | 芝良・着回数3 | `Jyotai1Chakukaisu3` | varchar | 3 | 0 | 芝・良馬場での3着の回数（中央のみ) |
| 118 |  | 芝良・着回数4 | `Jyotai1Chakukaisu4` | varchar | 3 | 0 | 芝・良馬場での4着の回数（中央のみ) |
| 119 |  | 芝良・着回数5 | `Jyotai1Chakukaisu5` | varchar | 3 | 0 | 芝・良馬場での5着の回数（中央のみ) |
| 120 |  | 芝良・着回数6 | `Jyotai1Chakukaisu6` | varchar | 3 | 0 | 芝・良馬場での着外の回数（中央のみ) |
| 121 |  | 芝稍・着回数1 | `Jyotai2Chakukaisu1` | varchar | 3 | 0 | 芝・稍重馬場での1着の回数（中央のみ) |
| 122 |  | 芝稍・着回数2 | `Jyotai2Chakukaisu2` | varchar | 3 | 0 | 芝・稍重馬場での2着の回数（中央のみ) |
| 123 |  | 芝稍・着回数3 | `Jyotai2Chakukaisu3` | varchar | 3 | 0 | 芝・稍重馬場での3着の回数（中央のみ) |
| 124 |  | 芝稍・着回数4 | `Jyotai2Chakukaisu4` | varchar | 3 | 0 | 芝・稍重馬場での4着の回数（中央のみ) |
| 125 |  | 芝稍・着回数5 | `Jyotai2Chakukaisu5` | varchar | 3 | 0 | 芝・稍重馬場での5着の回数（中央のみ) |
| 126 |  | 芝稍・着回数6 | `Jyotai2Chakukaisu6` | varchar | 3 | 0 | 芝・稍重馬場での着外の回数（中央のみ) |
| 127 |  | 芝重・着回数1 | `Jyotai3Chakukaisu1` | varchar | 3 | 0 | 芝・重馬場での1着の回数（中央のみ) |
| 128 |  | 芝重・着回数2 | `Jyotai3Chakukaisu2` | varchar | 3 | 0 | 芝・重馬場での2着の回数（中央のみ) |
| 129 |  | 芝重・着回数3 | `Jyotai3Chakukaisu3` | varchar | 3 | 0 | 芝・重馬場での3着の回数（中央のみ) |
| 130 |  | 芝重・着回数4 | `Jyotai3Chakukaisu4` | varchar | 3 | 0 | 芝・重馬場での4着の回数（中央のみ) |
| 131 |  | 芝重・着回数5 | `Jyotai3Chakukaisu5` | varchar | 3 | 0 | 芝・重馬場での5着の回数（中央のみ) |
| 132 |  | 芝重・着回数6 | `Jyotai3Chakukaisu6` | varchar | 3 | 0 | 芝・重馬場での着外の回数（中央のみ) |
| 133 |  | 芝不・着回数1 | `Jyotai4Chakukaisu1` | varchar | 3 | 0 | 芝・不良馬場での1着の回数（中央のみ) |
| 134 |  | 芝不・着回数2 | `Jyotai4Chakukaisu2` | varchar | 3 | 0 | 芝・不良馬場での2着の回数（中央のみ) |
| 135 |  | 芝不・着回数3 | `Jyotai4Chakukaisu3` | varchar | 3 | 0 | 芝・不良馬場での3着の回数（中央のみ) |
| 136 |  | 芝不・着回数4 | `Jyotai4Chakukaisu4` | varchar | 3 | 0 | 芝・不良馬場での4着の回数（中央のみ) |
| 137 |  | 芝不・着回数5 | `Jyotai4Chakukaisu5` | varchar | 3 | 0 | 芝・不良馬場での5着の回数（中央のみ) |
| 138 |  | 芝不・着回数6 | `Jyotai4Chakukaisu6` | varchar | 3 | 0 | 芝・不良馬場での着外の回数（中央のみ) |
| 139 |  | ダ良・着回数1 | `Jyotai5Chakukaisu1` | varchar | 3 | 0 | ダート・良馬場での1着の回数（中央のみ) |
| 140 |  | ダ良・着回数2 | `Jyotai5Chakukaisu2` | varchar | 3 | 0 | ダート・良馬場での2着の回数（中央のみ) |
| 141 |  | ダ良・着回数3 | `Jyotai5Chakukaisu3` | varchar | 3 | 0 | ダート・良馬場での3着の回数（中央のみ) |
| 142 |  | ダ良・着回数4 | `Jyotai5Chakukaisu4` | varchar | 3 | 0 | ダート・良馬場での4着の回数（中央のみ) |
| 143 |  | ダ良・着回数5 | `Jyotai5Chakukaisu5` | varchar | 3 | 0 | ダート・良馬場での5着の回数（中央のみ) |
| 144 |  | ダ良・着回数6 | `Jyotai5Chakukaisu6` | varchar | 3 | 0 | ダート・良馬場での着外の回数（中央のみ) |
| 145 |  | ダ稍・着回数1 | `Jyotai6Chakukaisu1` | varchar | 3 | 0 | ダート・稍重馬場での1着の回数（中央のみ) |
| 146 |  | ダ稍・着回数2 | `Jyotai6Chakukaisu2` | varchar | 3 | 0 | ダート・稍重馬場での2着の回数（中央のみ) |
| 147 |  | ダ稍・着回数3 | `Jyotai6Chakukaisu3` | varchar | 3 | 0 | ダート・稍重馬場での3着の回数（中央のみ) |
| 148 |  | ダ稍・着回数4 | `Jyotai6Chakukaisu4` | varchar | 3 | 0 | ダート・稍重馬場での4着の回数（中央のみ) |
| 149 |  | ダ稍・着回数5 | `Jyotai6Chakukaisu5` | varchar | 3 | 0 | ダート・稍重馬場での5着の回数（中央のみ) |
| 150 |  | ダ稍・着回数6 | `Jyotai6Chakukaisu6` | varchar | 3 | 0 | ダート・稍重馬場での着外の回数（中央のみ) |
| 151 |  | ダ重・着回数1 | `Jyotai7Chakukaisu1` | varchar | 3 | 0 | ダート・重馬場での1着の回数（中央のみ) |
| 152 |  | ダ重・着回数2 | `Jyotai7Chakukaisu2` | varchar | 3 | 0 | ダート・重馬場での2着の回数（中央のみ) |
| 153 |  | ダ重・着回数3 | `Jyotai7Chakukaisu3` | varchar | 3 | 0 | ダート・重馬場での3着の回数（中央のみ) |
| 154 |  | ダ重・着回数4 | `Jyotai7Chakukaisu4` | varchar | 3 | 0 | ダート・重馬場での4着の回数（中央のみ) |
| 155 |  | ダ重・着回数5 | `Jyotai7Chakukaisu5` | varchar | 3 | 0 | ダート・重馬場での5着の回数（中央のみ) |
| 156 |  | ダ重・着回数6 | `Jyotai7Chakukaisu6` | varchar | 3 | 0 | ダート・重馬場での着外の回数（中央のみ) |
| 157 |  | ダ不・着回数1 | `Jyotai8Chakukaisu1` | varchar | 3 | 0 | ダート・不良馬場での1着の回数（中央のみ) |
| 158 |  | ダ不・着回数2 | `Jyotai8Chakukaisu2` | varchar | 3 | 0 | ダート・不良馬場での2着の回数（中央のみ) |
| 159 |  | ダ不・着回数3 | `Jyotai8Chakukaisu3` | varchar | 3 | 0 | ダート・不良馬場での3着の回数（中央のみ) |
| 160 |  | ダ不・着回数4 | `Jyotai8Chakukaisu4` | varchar | 3 | 0 | ダート・不良馬場での4着の回数（中央のみ) |
| 161 |  | ダ不・着回数5 | `Jyotai8Chakukaisu5` | varchar | 3 | 0 | ダート・不良馬場での5着の回数（中央のみ) |
| 162 |  | ダ不・着回数6 | `Jyotai8Chakukaisu6` | varchar | 3 | 0 | ダート・不良馬場での着外の回数（中央のみ) |
| 163 |  | 障良・着回数1 | `Jyotai9Chakukaisu1` | varchar | 3 | 0 | 障害レース・良馬場での1着の回数（中央のみ) |
| 164 |  | 障良・着回数2 | `Jyotai9Chakukaisu2` | varchar | 3 | 0 | 障害レース・良馬場での2着の回数（中央のみ) |
| 165 |  | 障良・着回数3 | `Jyotai9Chakukaisu3` | varchar | 3 | 0 | 障害レース・良馬場での3着の回数（中央のみ) |
| 166 |  | 障良・着回数4 | `Jyotai9Chakukaisu4` | varchar | 3 | 0 | 障害レース・良馬場での4着の回数（中央のみ) |
| 167 |  | 障良・着回数5 | `Jyotai9Chakukaisu5` | varchar | 3 | 0 | 障害レース・良馬場での5着の回数（中央のみ) |
| 168 |  | 障良・着回数6 | `Jyotai9Chakukaisu6` | varchar | 3 | 0 | 障害レース・良馬場での着外の回数（中央のみ) |
| 169 |  | 障稍・着回数1 | `Jyotai10Chakukaisu1` | varchar | 3 | 0 | 障害レース・稍重馬場での1着の回数（中央のみ) |
| 170 |  | 障稍・着回数2 | `Jyotai10Chakukaisu2` | varchar | 3 | 0 | 障害レース・稍重馬場での2着の回数（中央のみ) |
| 171 |  | 障稍・着回数3 | `Jyotai10Chakukaisu3` | varchar | 3 | 0 | 障害レース・稍重馬場での3着の回数（中央のみ) |
| 172 |  | 障稍・着回数4 | `Jyotai10Chakukaisu4` | varchar | 3 | 0 | 障害レース・稍重馬場での4着の回数（中央のみ) |
| 173 |  | 障稍・着回数5 | `Jyotai10Chakukaisu5` | varchar | 3 | 0 | 障害レース・稍重馬場での5着の回数（中央のみ) |
| 174 |  | 障稍・着回数6 | `Jyotai10Chakukaisu6` | varchar | 3 | 0 | 障害レース・稍重馬場での着外の回数（中央のみ) |
| 175 |  | 障重・着回数1 | `Jyotai11Chakukaisu1` | varchar | 3 | 0 | 障害レース・重馬場での1着の回数（中央のみ) |
| 176 |  | 障重・着回数2 | `Jyotai11Chakukaisu2` | varchar | 3 | 0 | 障害レース・重馬場での2着の回数（中央のみ) |
| 177 |  | 障重・着回数3 | `Jyotai11Chakukaisu3` | varchar | 3 | 0 | 障害レース・重馬場での3着の回数（中央のみ) |
| 178 |  | 障重・着回数4 | `Jyotai11Chakukaisu4` | varchar | 3 | 0 | 障害レース・重馬場での4着の回数（中央のみ) |
| 179 |  | 障重・着回数5 | `Jyotai11Chakukaisu5` | varchar | 3 | 0 | 障害レース・重馬場での5着の回数（中央のみ) |
| 180 |  | 障重・着回数6 | `Jyotai11Chakukaisu6` | varchar | 3 | 0 | 障害レース・重馬場での着外の回数（中央のみ) |
| 181 |  | 障不・着回数1 | `Jyotai12Chakukaisu1` | varchar | 3 | 0 | 障害レース・不良馬場での1着の回数（中央のみ) |
| 182 |  | 障不・着回数2 | `Jyotai12Chakukaisu2` | varchar | 3 | 0 | 障害レース・不良馬場での2着の回数（中央のみ) |
| 183 |  | 障不・着回数3 | `Jyotai12Chakukaisu3` | varchar | 3 | 0 | 障害レース・不良馬場での3着の回数（中央のみ) |
| 184 |  | 障不・着回数4 | `Jyotai12Chakukaisu4` | varchar | 3 | 0 | 障害レース・不良馬場での4着の回数（中央のみ) |
| 185 |  | 障不・着回数5 | `Jyotai12Chakukaisu5` | varchar | 3 | 0 | 障害レース・不良馬場での5着の回数（中央のみ) |
| 186 |  | 障不・着回数6 | `Jyotai12Chakukaisu6` | varchar | 3 | 0 | 障害レース・不良馬場での着外の回数（中央のみ) |
| 187 |  | 芝16下・着回数1 | `Kyori1Chakukaisu1` | varchar | 3 | 0 | 芝･1600M以下での1着の回数（中央のみ) |
| 188 |  | 芝16下・着回数2 | `Kyori1Chakukaisu2` | varchar | 3 | 0 | 芝･1600M以下での2着の回数（中央のみ) |
| 189 |  | 芝16下・着回数3 | `Kyori1Chakukaisu3` | varchar | 3 | 0 | 芝･1600M以下での3着の回数（中央のみ) |
| 190 |  | 芝16下・着回数4 | `Kyori1Chakukaisu4` | varchar | 3 | 0 | 芝･1600M以下での4着の回数（中央のみ) |
| 191 |  | 芝16下・着回数5 | `Kyori1Chakukaisu5` | varchar | 3 | 0 | 芝･1600M以下での5着の回数（中央のみ) |
| 192 |  | 芝16下・着回数6 | `Kyori1Chakukaisu6` | varchar | 3 | 0 | 芝･1600M以下での着外の回数（中央のみ) |
| 193 |  | 芝22下・着回数1 | `Kyori2Chakukaisu1` | varchar | 3 | 0 | 芝･1601Ｍ以上2200M以下での1着の回数（中央のみ) |
| 194 |  | 芝22下・着回数2 | `Kyori2Chakukaisu2` | varchar | 3 | 0 | 芝･1601Ｍ以上2200M以下での2着の回数（中央のみ) |
| 195 |  | 芝22下・着回数3 | `Kyori2Chakukaisu3` | varchar | 3 | 0 | 芝･1601Ｍ以上2200M以下での3着の回数（中央のみ) |
| 196 |  | 芝22下・着回数4 | `Kyori2Chakukaisu4` | varchar | 3 | 0 | 芝･1601Ｍ以上2200M以下での4着の回数（中央のみ) |
| 197 |  | 芝22下・着回数5 | `Kyori2Chakukaisu5` | varchar | 3 | 0 | 芝･1601Ｍ以上2200M以下での5着の回数（中央のみ) |
| 198 |  | 芝22下・着回数6 | `Kyori2Chakukaisu6` | varchar | 3 | 0 | 芝･1601Ｍ以上2200M以下での着外の回数（中央のみ) |
| 199 |  | 芝22超・着回数1 | `Kyori3Chakukaisu1` | varchar | 3 | 0 | 芝･2201M以上での1着の回数（中央のみ) |
| 200 |  | 芝22超・着回数2 | `Kyori3Chakukaisu2` | varchar | 3 | 0 | 芝･2201M以上での2着の回数（中央のみ) |
| 201 |  | 芝22超・着回数3 | `Kyori3Chakukaisu3` | varchar | 3 | 0 | 芝･2201M以上での3着の回数（中央のみ) |
| 202 |  | 芝22超・着回数4 | `Kyori3Chakukaisu4` | varchar | 3 | 0 | 芝･2201M以上での4着の回数（中央のみ) |
| 203 |  | 芝22超・着回数5 | `Kyori3Chakukaisu5` | varchar | 3 | 0 | 芝･2201M以上での5着の回数（中央のみ) |
| 204 |  | 芝22超・着回数6 | `Kyori3Chakukaisu6` | varchar | 3 | 0 | 芝･2201M以上での着外の回数（中央のみ) |
| 205 |  | ダ16下・着回数1 | `Kyori4Chakukaisu1` | varchar | 3 | 0 | ダート･1600M以下での1着の回数（中央のみ) |
| 206 |  | ダ16下・着回数2 | `Kyori4Chakukaisu2` | varchar | 3 | 0 | ダート･1600M以下での2着の回数（中央のみ) |
| 207 |  | ダ16下・着回数3 | `Kyori4Chakukaisu3` | varchar | 3 | 0 | ダート･1600M以下での3着の回数（中央のみ) |
| 208 |  | ダ16下・着回数4 | `Kyori4Chakukaisu4` | varchar | 3 | 0 | ダート･1600M以下での4着の回数（中央のみ) |
| 209 |  | ダ16下・着回数5 | `Kyori4Chakukaisu5` | varchar | 3 | 0 | ダート･1600M以下での5着の回数（中央のみ) |
| 210 |  | ダ16下・着回数6 | `Kyori4Chakukaisu6` | varchar | 3 | 0 | ダート･1600M以下での着外の回数（中央のみ) |
| 211 |  | ダ22下・着回数1 | `Kyori5Chakukaisu1` | varchar | 3 | 0 | ダート･1601Ｍ以上2200M以下での1着の回数（中央のみ) |
| 212 |  | ダ22下・着回数2 | `Kyori5Chakukaisu2` | varchar | 3 | 0 | ダート･1601Ｍ以上2200M以下での2着の回数（中央のみ) |
| 213 |  | ダ22下・着回数3 | `Kyori5Chakukaisu3` | varchar | 3 | 0 | ダート･1601Ｍ以上2200M以下での3着の回数（中央のみ) |
| 214 |  | ダ22下・着回数4 | `Kyori5Chakukaisu4` | varchar | 3 | 0 | ダート･1601Ｍ以上2200M以下での4着の回数（中央のみ) |
| 215 |  | ダ22下・着回数5 | `Kyori5Chakukaisu5` | varchar | 3 | 0 | ダート･1601Ｍ以上2200M以下での5着の回数（中央のみ) |
| 216 |  | ダ22下・着回数6 | `Kyori5Chakukaisu6` | varchar | 3 | 0 | ダート･1601Ｍ以上2200M以下での着外の回数（中央のみ) |
| 217 |  | ダ22超・着回数1 | `Kyori6Chakukaisu1` | varchar | 3 | 0 | ダート･2201M以上での1着の回数（中央のみ) |
| 218 |  | ダ22超・着回数2 | `Kyori6Chakukaisu2` | varchar | 3 | 0 | ダート･2201M以上での2着の回数（中央のみ) |
| 219 |  | ダ22超・着回数3 | `Kyori6Chakukaisu3` | varchar | 3 | 0 | ダート･2201M以上での3着の回数（中央のみ) |
| 220 |  | ダ22超・着回数4 | `Kyori6Chakukaisu4` | varchar | 3 | 0 | ダート･2201M以上での4着の回数（中央のみ) |
| 221 |  | ダ22超・着回数5 | `Kyori6Chakukaisu5` | varchar | 3 | 0 | ダート･2201M以上での5着の回数（中央のみ) |
| 222 |  | ダ22超・着回数6 | `Kyori6Chakukaisu6` | varchar | 3 | 0 | ダート･2201M以上での着外の回数（中央のみ) |
| 223 |  | 脚質傾向1 | `Kyakusitu1` | varchar | 3 | 0 | 逃げ回数、先行回数、差し回数、追込回数を設定 過去出走レースの脚質を判定しカウントしたもの(中央レースのみ) |
| 224 |  | 脚質傾向2 | `Kyakusitu2` | varchar | 3 | 0 | 逃げ回数、先行回数、差し回数、追込回数を設定 過去出走レースの脚質を判定しカウントしたもの(中央レースのみ) |
| 225 |  | 脚質傾向3 | `Kyakusitu3` | varchar | 3 | 0 | 逃げ回数、先行回数、差し回数、追込回数を設定 過去出走レースの脚質を判定しカウントしたもの(中央レースのみ) |
| 226 |  | 脚質傾向4 | `Kyakusitu4` | varchar | 3 | 0 | 逃げ回数、先行回数、差し回数、追込回数を設定 過去出走レースの脚質を判定しカウントしたもの(中央レースのみ) |
| 227 |  | 登録レース数 | `RaceCount` | varchar | 3 | 0 | JRA-VANに登録されている成績レース数 |
