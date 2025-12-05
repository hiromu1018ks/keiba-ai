# 特徴量分析レポート

## 1. 現在実装されている特徴量 (Current Features)

現在 `experiments/v2_feature_expansion/src/feature_engineering.py` で実装されている主な特徴量は以下の通りです。

*   **基本情報**:
    *   `time_seconds` (タイム秒換算), `distance_category` (距離区分), `weight_bin` (馬体重区分), `running_style` (脚質), `jockey_trainer_pair` (騎手・調教師の組み合わせ)
*   **エンコーディング**:
    *   Target Encoding: `jockey_id`, `trainer_id`, `jockey_trainer_pair`
    *   Label Encoding: `weather`, `condition`, `surface` 等
*   **集計・履歴特徴量 (Lag Features)**:
    *   **近走成績**: 前走着順 (`prev_rank`), 3/5/10走の平均着順・賞金 (`rank_5races` 等)
    *   **統計量**: 過去5走の着順・賞金の標準偏差・最大・最小 (`rank_5_std` 等)
    *   **間隔**: 前走からの日数 (`interval`), 季節性 (`month_sin/cos`)
    *   **適性 (Suitability)**:
        *   距離別 (`avg_rank_Sprint` 等)
        *   馬場別 (`avg_rank_turf/dirt`)
        *   競馬場別 (`avg_rank_Tokyo` 等) - *Expanding meanで実装*
*   **騎手・調教師の直近成績**:
    *   過去100走の勝率・連対率・複勝率 (`jockey_win_rate_100` 等)
*   **相対評価 (Z-Score)**:
    *   レース内での偏差値: 馬体重, 年齢, オッズ, 近走成績(`ewma_rank_5`), 間隔, 上がり補正(margin)

## 2. 足りない特徴量・追加提案 (Missing Features)

### A. 現在のデータから作成可能な特徴量 (Immediately Implementable)

現在の `results.csv` に含まれるデータから、追加で作成可能な特徴量です。

1.  **コース・条件別の騎手/調教師成績 (Course/Distance Aptitude)**
    *   現在は「全レース」での過去100走成績ですが、「**現在の競馬場（例：東京）**」や「**現在の距離区分（例：マイル）**」に限定した成績の方が、予測精度への寄与が高い可能性があります。
    *   例: `jockey_course_win_rate` (騎手の当該コース勝率), `trainer_distance_win_rate` (調教師の当該距離勝率)

2.  **馬 × 騎手の相性 (Horse-Jockey Synergy)**
    *   `jockey_trainer_pair` はありますが、**馬と騎手のコンビ**での成績（過去にこのコンビで何勝しているか、平均着順はどうか）が不足しています。乗り替わりの影響を測るのに重要です。

3.  **斤量 (Impost) 関連**
    *   `impost` (斤量) そのものや、**馬体重に対する斤量の比率** (`impost_ratio = impost / horse_weight`)。小柄な馬にとっての重い斤量は不利になる傾向があります。

4.  **ローテーション詳細 (Rotation Details)**
    *   **叩き2戦目**: 長期休養明け (`interval` > 90日など) の次走はパフォーマンスが上がることが多いと言われます。「休養明け2戦目フラグ」など。
    *   **連闘**: `interval` < 7日 の場合のフラグ。

5.  **レースレベル・頭数 (Race Context)**
    *   `n_horses` (頭数): 少頭数と多頭数ではレース展開が異なります。
    *   **相手関係**: 出走馬の過去の平均賞金や平均着順の平均値（レースのレベル）。「今回は相手が弱い/強い」を数値化します。

6.  **枠順の有利不利 (Bracket Bias)**
    *   コースごとの枠順別成績（内枠有利・外枠不利など）をTarget Encodingした特徴量。現在は `bracket` そのものはありますが、コース特性と絡めた特徴量が必要です。

### B. 追加データが必要な特徴量 (Requires New Data)

現在の `results.csv` には含まれておらず、スクレイピング等の改修が必要な項目です。

1.  **血統情報 (Pedigree)** - **重要**
    *   **父 (Sire)**, **母 (Dam)**, **母父 (Broodmare Sire)**
    *   特に芝・ダート替わりや、距離短縮・延長の際に血統傾向は非常に強力なファクターになります。

2.  **上がり3ハロン (Agari 3F)**
    *   レースのペース配分や、馬の瞬発力を測るために不可欠です。現在は `passing_order` (通過順) はありますが、終いの脚（上がりタイム）がありません。

3.  **詳細な通過順・ペース (Detailed Pace)**
    *   前半3ハロン（テン3F）のタイムがあれば、ハイペース・スローペースの判定ができ、展開予想（逃げ馬有利か差し馬有利か）が可能になります。

## 3. 推奨される次のステップ

まずは **A. 現在のデータから作成可能な特徴量** のうち、特に効果が見込めそうな以下の3点の実装を推奨します。

1.  **コース・距離別** の騎手/調教師成績
2.  **馬 × 騎手** の相性特徴量
3.  **斤量比率** (`impost / horse_weight`)
