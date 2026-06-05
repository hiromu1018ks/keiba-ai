# Phase 49: Derived & Higher-Order Features - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 49-Derived Higher-Order Features
**Areas discussed:** 馬個体適性のPIT-safe計算, ペース/バイアススコアの算出方法, レースレベル集約戦略, 外科的ルーティングの拡張判定, T3-04季節偏差, T4-04既存インタラクション

---

## 馬個体適性のPIT-safe計算

### Q1: 過走履歴からの適性をどうPIT-safeに計算するか

| Option | Description | Selected |
|--------|-------------|----------|
| Precompute Parquet拡張 | horse_career_stats.parquetのexpanding windowパターンに倣い事前計算。テスト容易性高 | ✓ |
| Inline計算 (_train_submodel内) | groupby + cumulative count/sumでリアルタイム計算。新規parquet不要 | |
| Target Encoding風 OOF構造 | OOF fold境界を利用。最も厳密だが実装複雑 | |

**User's choice:** Precompute Parquet拡張。expanding window / shift(1)パターン、race_id + kettonumでmerge
**Notes:** テスト容易性と既存パターンとの整合性を重視

### Q2: 高含水/低含水・硬/柔の条件分類閾値

| Option | Description | Selected |
|--------|-------------|----------|
| 固定閾値 | 含水率≥10%を高含水。シンプルで解釈性高い | |
| 分位数閾値 | データ分布の分位数で定義。fold間変動に注意 | |
| 設定可能閾値 | 固定閾値をデフォルトとしconfigで上書き可能 | ✓ |

**User's choice:** 設定可能閾値。含水率 wet>=12%, dry<3%、クッション hard>=10, soft<8。>=10/<10の二分は「標準域」を広く含みすぎるため不採用
**Notes:** Phase 48の既存フラグ(JRA公式基準/実データ分布)と整合。config/settings.yamlで上書き可能

### Q3: T3-03の適性カテゴリ分類ロジック

| Option | Description | Selected |
|--------|-------------|----------|
| 率差分ベース | wet_rate > dry_rate で分類。シンプルだが少数出走馬で不安定 | |
| 絶対閾値＋最低出走数 | hit_rate>=thresholdかつmin_starts以上で判定 | ✓ |
| カテゴリ不要・連続値のみ | hit_rate列のみ。LightGBMに分類を任せる | |

**User's choice:** 絶対閾値＋最低出走数。min_starts=3, hit_rate_threshold=0.3。分母不足の条件は判定しない
**Notes:** 閾値はconfigで上書き可能

### Q4: 適性率の的中の定義

| Option | Description | Selected |
|--------|-------------|----------|
| 着順3着以内 | 複勝国際基準。サンプル数が多い | ✓ |
| 着順上位半分 | 出走頭数依存で基準が変動 | |
| 着順1着のみ | 最も厳しいがサンプル不足リスク | |

**User's choice:** kakuteijyuni <= 3。取消・除外は分母から除外。勝利適性ではなく好走適性を測る

### Q5: horse_condition_versatilityの定義

| Option | Description | Selected |
|--------|-------------|----------|
| 率差の逆数 | 1 - |wet - dry|。シンプル | |
| 経験多様性 | 出走条件の広さ。成績は反映しない | |
| 不要・rate列のみ | カテゴリ分類だけで十分 | |

**User's choice:** mean(wet_rate, dry_rate) × (1 - |wet_rate - dry_rate|)。成績水準とバランスの積。分母不足はNaN

### Q6: Precompute Parquetの出力スキーマ

| Option | Description | Selected |
|--------|-------------|----------|
| rate+count+category全部出力 | hit_rate + starts_count + versatility + condition_type。10+2列 | ✓ |
| rateのみ最小構成 | hit_rate列のみ。parquet容量削減 | |

**User's choice:** rate+count+category全部 + prev_dirt_moisture/prev_turf_cushion追加(12列)。unknownは明示的文字列

---

## ペース/バイアススコアの算出方法

### Q1: T4-01の算出方法

| Option | Description | Selected |
|--------|-------------|----------|
| ドメイン閾値ルール | 物理メカニズムに基づく閾値。解釈性高 | ✓ |
| 連続値正規化 | moisture / max_moisture。間接的 | |
| データ駆動回帰 | 過去成績から学習。PIT-safe複雑 | |

**User's choice:** ドメイン閾値ルールベースの連続スコア。0/0.5/1の粗い段階値ではなく、固定閾値に基づく連続スコア

### Q2: 連続スコアの変換方式

| Option | Description | Selected |
|--------|-------------|----------|
| 閾値間線形補間 | clip((moisture-3)/(12-3),0,1)。シンプル | ✓ |
| Sigmoid平滑化 | 滑らかなS字カーブ。ハイパラ必要 | |
| zone別固定スコア | pd.cut + 固定値。段階的 | |

**User's choice:** 閾値間線形補間。dirt_front_bias=clip((moisture-3)/(12-3),0,1), kickback=clip((12-moisture)/(12-3),0,1)。芝はcushion基準8-10範囲

### Q3: expected_pace_classの出力形式

| Option | Description | Selected |
|--------|-------------|----------|
| 3段階数値コード | slow=0, neutral=1, fast=2。LightGBM順序学習可能 | ✓ |
| category型文字列 | slow/neutral/fast。人間可読 | |
| ペースクラス不要 | front_bias/kickbackで十分 | |

**User's choice:** 3段階数値コード。NaNはNaNのまま

### Q4: 芝とダートのスコア統合

| Option | Description | Selected |
|--------|-------------|----------|
| 統一列 | 同一列でsurface別submodelで分離。Phase 48パターン | ✓ |
| surface別列 | dirt_front_bias / turf_front_bias。解釈性高い | |

**User's choice:** 統一列。ダートはmoisture由来、芝はcushion由来。他surfaceはNaN

---

## レースレベル集約戦略

### Q1: race_condition_match_scoreの集約方法

| Option | Description | Selected |
|--------|-------------|----------|
| max集約 | エース適性馬の存在を強調 | |
| mean集約 | レース全体の平均的条件合致度 | |
| 高適性馬の割合 | threshold超えの馬数/出走頭数 | |
| 複合 (max+mean+割合) | 情報量最大だが列数増 | ✓ |

**User's choice:** 複合集約。mean(主代表) + max + ratio。条件に応じて対応するrate列を選択

### Q2: race_field_front_biasの計算方法

| Option | Description | Selected |
|--------|-------------|----------|
| 先行馬密度×bias | front_runner_ratio × track_front_bias_score | ✓ |
| T4-01と同一 | 冗長 | |
| 先行馬密度のみ | トラック条件を考慮しない | |

**User's choice:** front_runner_ratio × track_front_bias_score。逃げ/先行馬の比率と馬場バイアスの積

### Q3: T4-02 race-level特徴量の計算タイミング

| Option | Description | Selected |
|--------|-------------|----------|
| compute内で完結 | 全てcompute_track_condition_features()内 | |
| T4-01とT4-02で関数分離 | 行単位とgroupby集約で責務分離 | ✓ |
| build_all()内別ステップ | 他T4との依存管理が複雑 | |

**User's choice:** 関数分離。compute_track_condition_features() → compute_race_condition_features()。呼び出し順: T3 merge → compute_track_condition_features → compute_race_condition_features

---

## 外科的ルーティングの拡張判定

### Q1: T3/T4特徴量の外科的ルーティング

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 48同一パターン | MarketModel/RaceQualityScreener/RegimeDetector全除外 | |
| 特徴量別精密ルーティング | T4-03→MarketModel候補、T4-02→RaceQualityScreener候補 | ✓ |
| 全モデル一律登録 | Phase 36の失敗パターン | |

**User's choice:** 特徴量別精密ルーティング。T3+T4-01+T4-04はPhase 48同一。T4-03異常値はMarketModel追加候補。T4-02集約はRaceQualityScreener追加候補。Phase 50 Auditで検証、問題あればPhase 48パターンへ戻す

---

## T3-04 季節偏差

### Q1: 季節偏差の計算方法

| Option | Description | Selected |
|--------|-------------|----------|
| track×month zscore | Phase 48 track_statsパターンのmonth拡張 | ✓ |
| track×month 差分のみ | std正規化なし。スケール不統一 | |
| track年間偏差 | 季節性を捉えられない | |

**User's choice:** trackcd × month の学習期間zscore。Phase 48のtrack_statsパターンをmonth次元に拡張

---

## T4-04 既存インタラクション

### Q1: インタラクションの算出方式

| Option | Description | Selected |
|--------|-------------|----------|
| 数値積パターン | Phase 48と同一パターン。シンプル | ✓ |
| カテゴリ積パターン | ビン化+積。解釈性高いが境界設計が必要 | |
| 混合パターン | 特徴量ごとに柔軟選択 | |

**User's choice:** 数値積。cushion×distance, moisture×weight, cushion×age, moisture×prev_kyakusitu。ビン化は後続重要度確認後に検討

### Q2: surface_condition_transitionの定義

| Option | Description | Selected |
|--------|-------------|----------|
| 前走からの馬場変化 | current - prev。T3 precomputeで前走値保持 | ✓ |
| 前回開催からの変化 | 開催周期データが必要で複雑 | |
| 今回見送り | 他4つで十分 | |

**User's choice:** 前走からの馬場変化。T3 precomputeにprev_dirt_moisture/prev_turf_cushionを追加。同surface前走なし→NaN

---

## Claude's Discretion

- テスト構成・テストケースの詳細設計 (既存パターンに従う)
- 各COLS定数の具体的な列名定義
- track_month_statsの保存形式
- precomputeスクリプトの実装詳細
- build_all()へのT3 parquet merge追加パターン
- ログフォーマット・進捗表示
- cushion_anomaly_high/lowの上下分離要否判断

## Deferred Ideas

None — discussion stayed within phase scope
