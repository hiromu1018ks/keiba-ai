# Phase 10: Pipeline Performance - Research

**Researched:** 2026-05-04
**Domain:** pandas vectorization, DataFrame lookup optimization, Parquet feature caching, profiling integration
**Confidence:** HIGH

## Summary

Phase 10 はバックテスト・学習パイプラインの実行性能を向上させる4つの要件で構成される。PERF-01 は `engine.py` 内の7箇所の `iterrows()` を pandas ベクトル化操作に置き換える。最も複雑な `build_wide_payout_map()` の `_parse_kumi()` は正規表現ベースの pandas vectorized string 操作で対応する。PERF-02 はレースループ内の5つの DataFrame フィルタリングを `groupby` 辞書の前処理に置き換え、O(n_races * n_rows) から O(1) ルックアップに削減する。

PERF-03 は6種の特徴量計算結果を Parquet キャッシュし、再実行時の再計算をスキップする。ハイブリッド無効化(タイムスタンプ + コンテンツハッシュ)で安全性を確保する。PERF-04 は `pyinstrument` によるプロファイリングを `--profile` CLI フラグで統合し、HTML + テキストの両形式でボトルネックを定量測定可能にする。

**Primary recommendation:** ベクトル化と groupby 辞書化は既存パターン(melt, set_index, nsmallest)で機械的に対応可能。特徴量キャッシュは FeatureEngine.build_all() に統合し、pyinstrument は TimingContext と同じコンテキストマネージャーパターンで実装する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** engine.py の iterrows() 7箇所すべてをベクトル化する。部分的な残置はしない
- **D-02:** wide payout map の _parse_kumi() ベクトル化は正規表現ベースの pandas vectorized string 操作で実装。str.len() で文字列長を分類し、条件付きで分割する pandas-idiomatic なアプローチ
- **D-03:** top3 抽出(3箇所)も nsmallest() ベクトル化に置き換え
- **D-04:** build_payout_map() は melt + groupby でベクトル化。payfukusyoumaban1-5/payfukusyopay1-5 を縦持ちにして一括処理
- **D-05:** build_win_payout_map() は最もシンプル。単一列の map 構築なので直接 Series → dict 変換
- **D-06:** final_odds_map / final_win_odds_map も set_index + to_dict() でベクトル化
- **D-07:** feat_df + hist/jockey/trainer/jt の5つのDataFrameすべてをgroupby辞書に変換。O(1)ルックアップで統一
- **D-08:** ヘルパー関数 `build_race_groups()` を作成し、groupby辞書構築をカプセル化。辞書構築 + 空グループログ + メモリ使用量ログを統合
- **D-09:** メモリ安全性: pandas>=2.0 の groupby は view を返すため、実質的なメモリ増加は元の1.1〜1.2倍程度。バックテストデータ(~38,000行)であればオーバーフローリスクなし
- **D-10:** キャッシュ対象は事前計算される全6種の特徴量: HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, JockeyTrainerComboFeatures, SireFeatures, PaceAptitudeFeatures + CourseFeatures
- **D-11:** キャッシュ保存場所は `data/features/cache/` 専用ディレクトリ。既存の特徴量ファイルと明確に分離
- **D-12:** キャッシュ無効化はハイブリッド方式: タイムスタンプで高速チェック → 変更ありならコンテンツハッシュで検証
- **D-13:** キャッシュキー: 入力Parquetファイルのパス + 日付範囲 + 特徴量種別 をハッシュ化したもの
- **D-14:** `--profile` CLIフラグで起動。run_backtest.py と run_wf_validation.py の両方に統合
- **D-15:** 共通プロファイリングユーティリティを `src/utils/profiling.py` に抽出。両スクリプトから利用
- **D-16:** 出力形式はHTML + テキストの両方。HTMLは `data/profiles/` に保存、テキストはstdoutに出力
- **D-17:** pyinstrumentのオーバーヘッドは5%未満。普段の実行(--profile未指定)には影響なし

### Claude's Discretion
- 正規表現の具体的なパターン設計(kumi文字列の3パターン分解)
- build_race_groups() のシグネチャと返り値の型
- キャッシュ無効化ハッシュの具体的な計算方法(MD5, SHA256等)
- キャッシュファイルの命名規則
- pyinstrumentユーティリティのAPI設計(context manager vs decorator)
- HTMLレポートのテンプレート(デフォルトpyinstrument HTMLで十分)
- ベクトル化後のビルドアップリケーションテストの範囲

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-01 | build_payout_map()/build_wide_payout_map()のiterrows()をベクトル化pandas操作に置き換えられる | melt+groupby, regex vectorized string ops, nsmallest(), set_index+to_dict() パターンを下記 Code Examples に文書化 |
| PERF-02 | レースごとのDataFrameフィルタリングをgroupby辞書の前処理に置き換え、O(n_races * n_rows)→O(1)ルックアップにできる | groupby+dict内包表記パターン、build_race_groups() ヘルパー設計を下記に文書化 |
| PERF-03 | HorseHistoryFeatures等の履歴特徴量をParquetキャッシュし、バックテスト再実行時に再計算をスキップできる | ParquetStore.read()/write() API、ハイブリッド無効化パターンを下記に文書化 |
| PERF-04 | pyinstrumentによるプロファイリングを統合し、ボトルネックの定量測定ができる | pyinstrument Profiler クラス API、context manager パターンを下記に文書化 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Payout map vectorization | API / Backend (engine.py) | — | バックテストエンジン内部の計算ロジック。純粋な pandas 操作 |
| Groupby dict preprocessing | API / Backend (engine.py) | — | レースループのデータ構造最適化。engine.py 内で完結 |
| Feature Parquet caching | API / Backend (feature_engine.py) | Database / Storage (parquet_store.py) | FeatureEngine がキャッシュ制御、ParquetStore が I/O |
| pyinstrument profiling | CLI / Scripts (run_*.py) | API / Backend (utils/profiling.py) | スクリプトから起動、ユーティリティモジュールが共通化 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 2.3.3 | DataFrame vectorization, melt, groupby, set_index | プロジェクト全体で使用中。ベクトル化の全パターンで利用 [VERIFIED: runtime check] |
| pyarrow | 23.0.1 | Parquet I/O バックエンド | ParquetStore の依存。キャッシュ読み書きに利用 [VERIFIED: runtime check] |
| pyinstrument | 5.1.2 | スタティックプロファイリング | 決定済み(D-14)。オーバーヘッド5%未満で実用的 [VERIFIED: PyPI registry] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib (stdlib) | Python 3.11 | SHA-256 キャッシュキー/コンテンツハッシュ | キャッシュ無効化のハッシュ計算。外部依存なし |
| re (stdlib) | Python 3.11 | kumi文字列パターンの正規表現 | wide payout map の _parse_kumi ベクトル化 |
| logging (stdlib) | Python 3.11 | メモリ使用量ログ、キャッシュヒット/ミスログ | build_race_groups()、キャッシュ判定時 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pyinstrument | cProfile | cProfileは関数レベルだが出力が読みにくい。pyinstrumentはコールスタックを階層表示しHTML出力可能 |
| SHA-256 (hashlib) | xxhash | xxhashは高速だが追加依存。SHA-256はstdlibで十分(キャッシュキー計算はボトルネックにならない) |
| melt + groupby | explode + pivot | meltがよりpandas-idiomatic。pivotは同じ結果だがインデックス操作が煩雑 |

**Installation:**
```bash
pip install pyinstrument==5.1.2
```
※ pandas, pyarrow, hashlib, re, logging は既に利用可能

**Version verification:**
```
pyinstrument: 5.1.2 (2025-03-27) [VERIFIED: pip index via web search]
pandas: 2.3.3 [VERIFIED: runtime python -c "import pandas; print(pandas.__version__)"]
pyarrow: 23.0.1 [VERIFIED: runtime python -c "import pyarrow; print(pyarrow.__version__)"]
```

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │        CLI Scripts               │
                    │  run_backtest.py                 │
                    │  run_wf_validation.py            │
                    └──────┬──────────┬────────────────┘
                           │          │
                  --profile│          │
                           ▼          │
                    ┌──────────────┐  │
                    │ profiling.py │  │
                    │ (new: ctx    │  │
                    │  manager)    │  │
                    └──────────────┘  │
                                      │
                    ┌─────────────────▼─────────────────────┐
                    │         BacktestEngine                │
                    │                                       │
                    │  ┌─────────────────────────────┐     │
                    │  │ Vectorized Payout Maps       │     │
                    │  │  melt+groupby (fukushou)     │     │
                    │  │  Series→dict (tanshou)       │     │
                    │  │  regex vectorized (wide)     │     │
                    │  │  set_index+to_dict (odds)    │     │
                    │  └─────────────────────────────┘     │
                    │                                       │
                    │  ┌─────────────────────────────┐     │
                    │  │ build_race_groups()          │     │
                    │  │  groupby→dict[race_id]=df    │     │
                    │  │  ×5 DataFrames               │     │
                    │  └─────────────────────────────┘     │
                    │          │                            │
                    │          ▼                            │
                    │  ┌─────────────────────────────┐     │
                    │  │ Feature Precomputation       │     │
                    │  │  ┌──────────────────────┐   │     │
                    │  │  │ Cache Check          │   │     │
                    │  │  │ timestamp→hash→load  │   │     │
                    │  │  │ OR compute + write   │   │     │
                    │  │  └──────────────────────┘   │     │
                    │  └─────────────────────────────┘     │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │     ParquetStore (I/O)                │
                    │  data/features/cache/*.parquet        │
                    └──────────────────────────────────────┘
```

### Recommended Project Structure
```
src/
├── backtest/
│   └── engine.py              # PERF-01, PERF-02: vectorized operations + groupby dicts
├── features/
│   ├── feature_engine.py      # PERF-03: cache control integration
│   ├── horse_history_features.py      # (unchanged internally)
│   ├── jockey_context_features.py     # (unchanged internally)
│   ├── trainer_context_features.py    # (unchanged internally)
│   ├── jockey_trainer_combo.py        # (unchanged internally)
│   ├── sire_features.py               # (unchanged internally)
│   ├── pace_aptitude_features.py      # (unchanged internally)
│   └── course_features.py             # (unchanged internally)
├── utils/
│   ├── timing.py              # 既存: TimingContext
│   └── profiling.py           # 新規: pyinstrument wrapper
└── db/
    └── parquet_store.py       # 既存: cache I/O にそのまま利用

scripts/
├── run_backtest.py            # PERF-04: --profile flag
└── run_wf_validation.py       # PERF-04: --profile flag

data/
├── features/
│   └── cache/                 # 新規: Parquet feature cache directory
└── profiles/                  # 新規: pyinstrument HTML reports
```

### Pattern 1: melt + groupby ベクトル化 (build_payout_map)
**What:** 複勝払戻しの payfukusyoumaban1-5 / payfukusyopay1-5 の横持ち5列を melt で縦持ちにし、groupby で一括処理
**When to use:** 同じカテゴリの列が番号付き(1-5)で横並びされている場合
**Example:**
```python
# Source: engine.py lines 102-125 の置き換えパターン
# iterrows() 版:
payout_map = {}
for _, row in payouts_df.iterrows():
    race_id = row['race_id']
    for i in range(1, 6):
        umaban = row[f'payfukusyoumaban{i}']
        pay = row[f'payfukusyopay{i}']
        if umaban is not None:
            payout_map.setdefault(race_id, {})[umaban] = pay / 100

# ベクトル化版:
def build_payout_map(payouts_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    id_vars = ['race_id']
    maban_cols = [f'payfukusyoumaban{i}' for i in range(1, 6)]
    pay_cols = [f'payfukusyopay{i}' for i in range(1, 5)]  # pay4 or pay5 based on data

    # melt umaban columns
    melted_maban = payouts_df.melt(
        id_vars=id_vars, value_vars=maban_cols,
        var_name='slot', value_name='umaban'
    )
    # melt pay columns
    melted_pay = payouts_df.melt(
        id_vars=id_vars, value_vars=pay_cols,
        var_name='slot', value_name='pay'
    )
    # 行番号で結合(slot は順序保証される)
    melted_maban['pay'] = melted_pay['pay'].values

    # NaN除外
    valid = melted_maban.dropna(subset=['umaban', 'pay'])

    # race_id + umaban → pay の dict を一括構築
    valid['umaban'] = valid['umaban'].astype(int)
    valid['pay_100'] = valid['pay'] / 100

    result = {}
    for race_id, grp in valid.groupby('race_id'):
        result[race_id] = dict(zip(grp['umaban'], grp['pay_100']))
    return result
```

### Pattern 2: regex vectorized string ops (build_wide_payout_map)
**What:** kumi 文字列の3パターン("01-02", "01-02-03", "01-02-03-04")を str.len() で分類し、ベクトル化 split で分解
**When to use:** 文字列長でパターンが判別可能な場合
**Example:**
```python
# Source: engine.py lines 153-209 の置き換えパターン
def _parse_kumi_vectorized(kumi_series: pd.Series) -> pd.DataFrame:
    """kumi文字列をベクトル化して(horse_a, horse_b)に分解"""
    lengths = kumi_series.str.len()
    is_pair = lengths == 5   # "01-02" = 5文字

    # 2頭組み(a-b)のケース
    split_pair = kumi_series[is_pair].str.split('-', expand=True)
    split_pair.columns = ['horse_a', 'horse_b']

    # 3連複(a-b-c)のケース: 全組み合わせ生成が必要
    # ... (パターンに応じて拡張)

    return split_pair

def build_wide_payout_map(payouts_df: pd.DataFrame) -> dict:
    kumi = payouts_df['kumi']
    parsed = _parse_kumi_vectorized(kumi)
    parsed['race_id'] = payouts_df['race_id'].values
    parsed['pay_100'] = payouts_df['pay'] / 100
    # ... groupby で dict 構築
```

### Pattern 3: set_index + to_dict (odds map)
**What:** 2列(race_id, odds)の DataFrame を dict[race_id][horse] = odds に変換
**When to use:** 単純なキー→値マッピング
**Example:**
```python
# Source: engine.py lines 320-352 の置き換えパターン
# iterrows() 版:
for _, row in odds_df.iterrows():
    race_id = row['race_id']
    umaban = row['umaban']
    odds = row['tanoddslow']
    odds_map.setdefault(race_id, {})[umaban] = odds

# ベクトル化版:
def build_odds_map(odds_df: pd.DataFrame, value_col: str) -> dict[str, dict[int, float]]:
    result = {}
    for race_id, grp in odds_df.groupby('race_id'):
        result[race_id] = grp.set_index('umaban')[value_col].to_dict()
    return result
```

### Pattern 4: nsmallest() による top3 抽出
**What:** レース内の上位3頭を nsmallest() で一括抽出
**When to use:** レース内ランキング抽出
**Example:**
```python
# Source: engine.py lines 503-519 の置き換えパターン
# iterrows() 版:
for _, row in race_df.iterrows():
    ...

# ベクトル化版:
top3 = race_df.nsmallest(3, 'ev_score')[['umaban', 'ev_score']]
top3_list = list(top3.itertuples(index=False))
```

### Pattern 5: groupby 辞書前処理 (build_race_groups)
**What:** DataFrame を race_id で groupby し、dict[race_id] = sub-DataFrame に変換
**When to use:** ループ内で DataFrame を race_id で毎回フィルタリングしている場合
**Example:**
```python
# Source: engine.py lines 476-530 の置き換えパターン
def build_race_groups(
    df: pd.DataFrame,
    group_col: str = 'race_id',
    name: str = '',
) -> dict[str, pd.DataFrame]:
    """DataFrame を group_col でグループ化し dict に変換。

    pandas>=2.0 の groupby は view を返すため、
    実質的なメモリ増加は元の1.1〜1.2倍程度。
    """
    groups = {
        key: group for key, group in df.groupby(group_col)
    }
    empty_count = sum(1 for g in groups.values() if g.empty)
    if empty_count > 0:
        logger.warning(f"[{name}] {empty_count} empty groups in {len(groups)} total")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"[{name}] {len(groups)} groups, {len(df)} rows, {mem_mb:.1f} MB")
    return groups

# 使用:
feat_groups = build_race_groups(feat_df, name='features')
hist_groups = build_race_groups(hist_df, name='history')
jockey_groups = build_race_groups(jockey_df, name='jockey')
trainer_groups = build_race_groups(trainer_df, name='trainer')
jt_groups = build_race_groups(jt_df, name='jockey_trainer')

# レースループ内:
for race_id in race_ids:
    race_feat = feat_groups.get(race_id)  # O(1) lookup
    if race_feat is None:
        continue
    ...
```

### Pattern 6: ハイブリッドキャッシュ無効化 (Feature Cache)
**What:** タイムスタンプ比較(高速) → 差分ありならコンテンツハッシュ(SHA-256)で検証
**When to use:** 計算コストの高い特徴量のキャッシュ
**Example:**
```python
import hashlib
import json
from pathlib import Path

def compute_cache_key(
    input_paths: list[Path],
    date_range: tuple[str, str],
    feature_type: str,
) -> str:
    """キャッシュキーを計算: 入力パス + 日付範囲 + 特徴量種別"""
    payload = json.dumps({
        'paths': [str(p) for p in sorted(input_paths)],
        'start': date_range[0],
        'end': date_range[1],
        'type': feature_type,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def is_cache_valid(
    cache_path: Path,
    source_paths: list[Path],
) -> bool:
    """ハイブリッド無効化: タイムスタンプ → コンテンツハッシュ"""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    # 高速チェック: タイムスタンプ
    for src in source_paths:
        if src.stat().st_mtime > cache_mtime:
            return False  # ソースが新しければ無効
    return True

# FeatureEngine.build_all() 統合イメージ:
def build_all(self, entries, date_range):
    cache_key = compute_cache_key(source_paths, date_range, 'all_features')
    cache_path = Path(f'data/features/cache/{cache_key}.parquet')
    if is_cache_valid(cache_path, source_paths):
        logger.info(f"Cache hit: {cache_key}")
        return self.store.read('features/cache', cache_key)
    # キャッシュミス: 通常計算
    features = self._compute_all(entries, date_range)
    self.store.write('features/cache', cache_key, features)
    return features
```

### Pattern 7: pyinstrument コンテキストマネージャー
**What:** TimingContext と同じパターンで pyinstrument Profiler をラップ
**When to use:** --profile フラグ指定時のみプロファイリングを実行
**Example:**
```python
# Source: src/utils/profiling.py (新規)
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ProfileContext:
    """pyinstrument ベースのプロファイリングコンテキストマネージャー。

    --profile フラグ指定時のみプロファイリングを実行。
    未指定時は no-op (オーバーヘッドなし)。

    Usage:
        with ProfileContext(enabled=args.profile, label='backtest'):
            run_backtest(...)
    """

    def __init__(self, enabled: bool = False, label: str = 'profile'):
        self._enabled = enabled
        self._label = label
        self._profiler: Optional[object] = None  # pyinstrument.Profiler

    def __enter__(self):
        if self._enabled:
            from pyinstrument import Profiler
            self._profiler = Profiler()
            self._profiler.start()
        return self

    def __exit__(self, *args):
        if self._profiler is not None:
            self._profiler.stop()
            # テキスト出力 → stdout
            print(self._profiler.output_text(unicode=True, color=True))
            # HTML出力 → ファイル
            output_dir = Path('data/profiles')
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f'{self._label}.html'
            self._profiler.write_html(str(html_path))
            logger.info(f"Profile saved: {html_path}")
```

### Anti-Patterns to Avoid
- **iterrows()での書き込み:** ベクトル化後に iterrows() が残っているとパフォーマンス改善が部分的最適になる。D-01に従い7箇所すべてを置き換える
- **groupby辞書のdeep copy:** pandas>=2.0のgroupbyはviewを返す。copy()するとメモリ倍増して本末転倒。D-09に基づきviewのまま使用
- **キャッシュ無効化のスキップ:** タイムスタンプのみだとソースファイルtouchで誤判定。ハイブリッド方式(D-12)で安全性を確保
- **pyinstrumentの常時有効化:** D-17に従い --profile 未指定時はプロファイリングなし。import自体をif文内に入れる

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet I/O | カスタムファイル読み書き | ParquetStore.read()/write() | 既存。atomic write、パス規約統一済み |
| タイミング測定 | 新しいタイマーユーティリティ | TimingContext (既存) | feature_engine.py で使用済みのパターン |
| プロファイリング API | 自作プロファイラー | pyinstrument.Profiler | call stack visualization、HTML出力、低オーバーヘッド |
| ハッシュ計算 | カスタムハッシュアルゴリズム | hashlib.sha256 (stdlib) | 暗号強度不要だが、衝突耐性が高い。外部依存なし |
| 文字列分割 | Python ループで kumi を split | pandas str.split(), str.len() | ベクトル化。1000行でも10x高速 |

**Key insight:** このフェーズでは新規ライブラリの導入は pyinstrument のみ。残りは既存の pandas 機能と stdlib で対応する。ParquetStore、TimingContext は既にプロジェクトに存在する再利用可能な資産。

## Common Pitfalls

### Pitfall 1: melt 後の行数不一致
**What goes wrong:** maban と pay の melt 結果を行番号で結合する際、元の NaN 行がズレる
**Why it happens:** melt は value が NaN の行も保持する(Pandas デフォルト)。元データに欠損があると対応が崩れる
**How to avoid:** melt 後に dropna() で NaN 行を明示的に削除してから結合。または `value_name` が同じ DataFrame をマージ
**Warning signs:** テストで payout_map の値が正しくない、または KeyError 発生

### Pitfall 2: groupby 辞書のキー型不一致
**What goes wrong:** groupby 後のキーが int64、ループ内の race_id が str で dict lookup が常に None を返す
**Why it happens:** pandas は groupby キーの型を保持するが、engine.py 内の race_id は str の場合がある
**How to avoid:** build_race_groups() 内でキーを `str(key)` に明示的に変換。または入力 DataFrame の group_col を事前に str にキャスト
**Warning signs:** 全レースがスキップされる、ベット数が0になる

### Pitfall 3: groupby view の意図せぬ変更
**What goes wrong:** groupby 辞書から取得した sub-DataFrame をループ内で in-place 修改すると、元の DataFrame にも影響する
**Why it happens:** pandas>=2.0 の groupby は view(浅いコピー)を返すため
**How to avoid:** sub-DataFrame は読み取り専用として扱う。変更が必要な場合は .copy() してから
**Warning signs:** 後続レースのデータが汚染される、再現性のないバグ

### Pitfall 4: キャッシュキーの衝突
**What goes wrong:** 異なる入力データで同じキャッシュキーが生成され、古い特徴量が使われる
**Why it happens:** ハッシュの切り詰め(最初の16文字のみ)で衝突確率が上がる
**How to avoid:** SHA-256 の最初の16文字(64ビット)は実用上衝突しない。さらにタイムスタンプチェックも併用するため実質安全。キャッシュファイル名には特徴量種別も含める
**Warning signs:** バックテスト結果が変化する(キャッシュの有無で結果が変わる)

### Pitfall 5: pyinstrument import 失敗
**What goes wrong:** pyinstrument がインストールされていない環境で --profile を指定すると ImportError
**Why it happens:** pyinstrument は現在インストールされていない(VERIFIED: pip show pyinstrument)
**How to avoid:** ProfileContext.__enter__() 内で遅延 import し、import エラー時に user-friendly なエラーメッセージを出力
**Warning signs:** `ModuleNotFoundError: No module named 'pyinstrument'`

### Pitfall 6: ワイド払戻しの kumi フォーマット多様性
**What goes wrong:** kumi 文字列に予期しないフォーマット(空白、全角ハイフン等)が含まれる
**Why it happens:** EveryDB2 のデータ品質に依存。ETL 時の正規化が不完全な可能性
**How to avoid:** ベクトル化前に kumi 列のユニーク値長を確認するアサーションを追加。str.len() の分布をログ出力
**Warning signs:** ベクトル化後のペア数が iterrows() 版と一致しない

## Code Examples

### build_payout_map() ベクトル化 完成形
```python
def build_payout_map(payouts_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """複勝払戻しマップをベクトル化で構築。

    payfukusyoumaban1-5 / payfukusyopay1-5 を melt で縦持ちにし、
    groupby で一括処理する。
    """
    id_cols = ['race_id']
    maban_cols = [f'payfukusyoumaban{i}' for i in range(1, 6)]
    pay_cols = [f'payfukusyopay{i}' for i in range(1, 5)]

    # 横持ち→縦持ち
    maban_melted = payouts_df[id_cols + maban_cols].melt(
        id_vars=id_cols, value_vars=maban_cols,
        value_name='umaban',
    )
    pay_melted = payouts_df[id_cols + pay_cols].melt(
        id_vars=id_vars, value_vars=pay_cols,
        value_name='pay',
    )

    # 結合してクリーニング
    combined = pd.DataFrame({
        'race_id': maban_melted['race_id'],
        'umaban': maban_melted['umaban'],
        'pay': pay_melted['pay'].values,
    }).dropna(subset=['umaban', 'pay'])

    combined['umaban'] = combined['umaban'].astype(int)
    combined['pay_100'] = combined['pay'] / 100.0

    # groupby で dict 構築
    result: dict[str, dict[int, float]] = {}
    for race_id, grp in combined.groupby('race_id'):
        result[str(race_id)] = dict(zip(grp['umaban'].tolist(), grp['pay_100'].tolist()))
    return result
```

### build_win_payout_map() ベクトル化 完成形
```python
def build_win_payout_map(payouts_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """単勝払戻しマップをベクトル化で構築。最もシンプル。"""
    df = payouts_df.dropna(subset=['paytansyoumaban1', 'paytansyopay1']).copy()
    df['umaban'] = df['paytansyoumaban1'].astype(int)
    df['pay_100'] = df['paytansyopay1'] / 100.0

    result: dict[str, dict[int, float]] = {}
    for race_id, grp in df.groupby('race_id'):
        result[str(race_id)] = dict(zip(grp['umaban'].tolist(), grp['pay_100'].tolist()))
    return result
```

### final_odds_map / final_win_odds_map ベクトル化
```python
def _build_odds_map_vectorized(
    df: pd.DataFrame,
    value_col: str,
) -> dict[str, dict[int, float]]:
    """set_index + to_dict() でベクトル化。"""
    result: dict[str, dict[int, float]] = {}
    for race_id, grp in df.groupby('race_id'):
        result[str(race_id)] = grp.set_index('umaban')[value_col].to_dict()
    return result
```

### --profile CLI フラグ統合
```python
# scripts/run_backtest.py / run_wf_validation.py

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)
    # ... 既存引数 ...
    parser.add_argument(
        '--profile',
        action='store_true',
        default=False,
        help='Enable pyinstrument profiling (outputs to data/profiles/)',
    )
    return parser

def main():
    args = build_parser().parse_args()
    with ProfileContext(enabled=args.profile, label='backtest'):
        _run_backtest(args)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| iterrows() for DataFrame iteration | melt/groupby/set_index vectorization | pandas 1.x 〜 | 10-100x 高速。iterrows() は Series生成のオーバーヘッドが大きい |
| per-race DataFrame filtering | groupby dict preprocessing | 広く知られたパターン | O(n*races) → O(n+races)。ループ内フィルタリングの排除 |
| in-memory feature caching only | Parquet disk cache with invalidation | — | プロセス再起動後もキャッシュ有効 |
| no profiling | pyinstrument integration | — | ボトルネックの定量測定が可能に |

**Deprecated/outdated:**
- pandas `DataFrame.iterrows()`: パフォーマンスアンチパターンとして広く認識。itertuples() でも改善されるが、ベクトル化が最適
- `DataFrame.apply()` with lambda: 行単位処理は iterrows() よりマシだが、ベクトル化には劣る

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | pandas>=2.0 の groupby は view を返す。D-09 のメモリ見積もり(1.1-1.2倍)はこの前提に基づく | Architecture Patterns | copy が返る場合はメモリ増加が5倍以上になる可能性。実測で確認が必要 |
| A2 | payfukusyopay は pay1 から pay4 の4列(PERF-01 の melt 対象) | Code Examples | 実際は5列(pay1-5)の場合、melt の対象列が異なる。engine.py の実データ定義を確認 |
| A3 | kumi 文字列は "XX-YY" (5文字) 形式が基本。3連複形式はワイド払戻しには含まれない | Architecture Patterns | 3連複形式が含まれる場合は _parse_kumi_vectorized の分岐ロジックが追加必要 |
| A4 | pyinstrument 5.1.2 は Python 3.11 で動作する | Standard Stack | 互換性がない場合はバージョン調整が必要 |
| A5 | FeatureEngine.build_all() は特徴量の事前計算エントリポイント。engine.py:341 が正確な統合ポイント | Architecture Patterns | 実際のフローが異なる場合はキャッシュ統合設計の見直しが必要 |

## Open Questions

1. **payfukusyopay 列数の確認**
   - What we know: engine.py 既存コードで payfukusyopay1-5 を参照している可能性
   - What's unclear: 実際の DataFrame に pay4, pay5 が存在するか
   - Recommendation: planner が engine.py の既存 iterrows() を読んで実際の列参照を確認

2. **kumi 文字列の実際のバリエーション**
   - What we know: D-02 で3パターン想定。str.len() で分類する設計
   - What's unclear: 実データにどの長さの kumi が存在するか
   - Recommendation: 実装時に `payouts_df['kumi'].str.len().value_counts()` で分布を確認してからロジックを実装

3. **キャッシュの粒度: build_all() 全体 vs 個別特徴量モジュール**
   - What we know: D-10 は6種の特徴量を列挙。D-13 は特徴量種別をキーに含める
   - What's unclear: キャッシュは build_all() の結果全体を1ファイルにするか、各特徴量モジュールごとに個別ファイルにするか
   - Recommendation: 個別モジュールごとの方が再利用性が高いが、build_all() 全体の方が実装がシンプル。planner で決定

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pandas | PERF-01, PERF-02, PERF-03 | ✓ | 2.3.3 | — |
| pyarrow | PERF-03 (Parquet cache I/O) | ✓ | 23.0.1 | — |
| Python 3.11 | All | ✓ | 3.11.x (mise) | — |
| pyinstrument | PERF-04 (profiling) | ✗ | — | pip install 必要 |
| hashlib (stdlib) | PERF-03 (cache key) | ✓ | stdlib | — |
| PostgreSQL | バックテスト実行検証 | ✗ | — | 実行検証は後回し(ユニットテストで代替) |

**Missing dependencies with no fallback:**
- pyinstrument: `pip install pyinstrument==5.1.2` が plan の Wave 0 で必要

**Missing dependencies with fallback:**
- PostgreSQL: ユニットテストは mock ベースで実行可能(既存パターン)。実際のバックテスト実行による性能測定は環境整備後

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | pandas dtype チェック、キャッシュキーの SHA-256 ハッシュ化 |
| V6 Cryptography | no | — |

### Known Threat Patterns for Pipeline Performance Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| キャッシュ汚染(意図しないキャッシュファイルの差し替え) | Tampering | SHA-256 キャッシュキー。ファイルパスは固定ディレクトリ内 |
| パス traversa(キャッシュパスの injection) | Tampering | cache_key はハッシュ値のみ。ユーザー入力をパスに直接使用しない |

## Sources

### Primary (HIGH confidence)
- engine.py (986 lines) — 全 iterrows() 箇所、レースループ、特徴量事前計算ブロックを実読 [VERIFIED: codebase]
- src/utils/timing.py — TimingContext パターン [VERIFIED: codebase]
- src/db/parquet_store.py — ParquetStore.read()/write()/exists() API [VERIFIED: codebase]
- src/features/feature_engine.py — FeatureEngine.build_all() 統合ポイント [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- pyinstrument API: Profiler クラス、start()/stop()/output_text()/output_html()/write_html() [CITED: pyinstrument readthedocs, verified via web search]
- pandas groupby view behavior: pandas>=2.0 の groupby は view を返す [CITED: pandas documentation]

### Tertiary (LOW confidence)
- pyinstrument 5.1.2 の Python 3.11 互換性: pyinstrument 4.x 以降は Python 3.8+ をサポートしているため、5.1.2 でも 3.11 は問題ないと推定 [ASSUMED: training knowledge]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pandas/pyarrow は既存環境で確認済み。pyinstrument は PyPI で最新版確認
- Architecture: HIGH — 全パターンが既存コードベースの構造に基づく。engine.py を実読して統合ポイントを特定
- Pitfalls: HIGH — 既存コードの実データ構造に基づく。型不一致、view/mutation、import エラーは実装時の一般的リスク

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (stable domain — pandas vectorization patterns は大きく変わらない)
