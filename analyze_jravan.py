"""
JRA-VANデータの分析スクリプト
回収率向上につながる項目を特定する
"""

import pandas as pd
import numpy as np

# データ読み込み
print("データを読み込み中...")
df = pd.read_csv('data/jra_van/race/2015-2025.csv', encoding='cp932', low_memory=False)
print(f"総レコード数: {len(df):,}")
print(f"カラム数: {len(df.columns)}")
print()

# ========================================
# 1. カラム別の基本情報
# ========================================
print("=" * 60)
print("1. カラム別の基本情報")
print("=" * 60)

col_info = []
for col in df.columns:
    missing_rate = df[col].isna().sum() / len(df) * 100
    unique = df[col].nunique()
    dtype = str(df[col].dtype)
    col_info.append({
        'カラム': col,
        '欠損率(%)': round(missing_rate, 1),
        'ユニーク数': unique,
        '型': dtype
    })

col_df = pd.DataFrame(col_info)
print(col_df.to_string(index=False))
print()

# ========================================
# 2. 回収率向上に寄与しうる特殊カラムの特定
# ========================================
print("=" * 60)
print("2. 回収率向上に寄与しうる特殊カラム")
print("=" * 60)

# JRA-VAN独自と思われる項目
special_columns = [
    'PCI', 'RPCI', 'PCI3', '補正タイム', 'タイムS', '基準タイム(秒)',
    '馬印1', 'レース印1', 'レース印2', 'レース印3',
    '脚質', '上り3F順位', 'Ave-3F', '-3F差',
    'レースコメント', '結果コメント', 'KOL関係者コメント', 'KOL次走へのメモ',
    '父タイプ', '母父タイプ', '馬コメント',
    'ブリンカー', '増減'
]

for col in special_columns:
    if col in df.columns:
        print(f"\n■ {col}")
        print(f"  欠損率: {df[col].isna().sum() / len(df) * 100:.1f}%")
        print(f"  ユニーク数: {df[col].nunique()}")
        if df[col].dtype in ['int64', 'float64']:
            print(f"  統計: 平均={df[col].mean():.2f}, 標準偏差={df[col].std():.2f}")
        else:
            # カテゴリカルの場合、上位5件を表示
            top_vals = df[col].value_counts().head(5)
            print(f"  上位5値: {top_vals.to_dict()}")

# ========================================
# 3. 着順との相関分析（数値カラム）
# ========================================
print()
print("=" * 60)
print("3. 着順との相関分析（数値カラム）")
print("=" * 60)

# 確定着順を数値化（1着から分析）
df_valid = df[df['確定着順'].notna() & (df['確定着順'] > 0)].copy()
df_valid['確定着順'] = df_valid['確定着順'].astype(float)

# 1着フラグ
df_valid['win'] = (df_valid['確定着順'] == 1).astype(int)
# 3着以内フラグ
df_valid['top3'] = (df_valid['確定着順'] <= 3).astype(int)

# 数値カラムとの相関
numeric_cols = df_valid.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['確定着順', 'win', 'top3', '入線着順', '人気']]

correlations = []
for col in numeric_cols:
    if df_valid[col].notna().sum() > 10000:  # 十分なデータがある場合のみ
        corr_rank = df_valid['確定着順'].corr(df_valid[col])
        corr_win = df_valid['win'].corr(df_valid[col])
        correlations.append({
            'カラム': col,
            '着順との相関': round(corr_rank, 3),
            '1着との相関': round(corr_win, 3)
        })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('1着との相関', ascending=False, key=abs)
print(corr_df.to_string(index=False))

# ========================================
# 4. 人気別・PCI別の回収率分析
# ========================================
print()
print("=" * 60)
print("4. 人気別・PCI別の分析")
print("=" * 60)

# オッズ×的中からの簡易回収率
df_valid['単勝オッズ'] = pd.to_numeric(df_valid['単勝オッズ'], errors='coerce')

# 人気別
print("\n■ 人気別 勝率・平均オッズ:")
pop_group = df_valid.groupby('人気').agg({
    'win': ['sum', 'count', 'mean'],
    '単勝オッズ': 'mean'
}).head(10)
pop_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
pop_group['簡易回収率'] = pop_group['勝率'] * pop_group['平均オッズ'] * 100
print(pop_group)

# PCI分析（存在する場合）
if 'PCI' in df_valid.columns:
    print("\n■ PCI別 勝率分析:")
    df_valid['PCI_num'] = pd.to_numeric(df_valid['PCI'], errors='coerce')
    df_valid['PCI_bin'] = pd.cut(df_valid['PCI_num'], bins=[-np.inf, 40, 45, 50, 55, 60, np.inf], 
                                  labels=['~40', '40-45', '45-50', '50-55', '55-60', '60~'])
    pci_group = df_valid.groupby('PCI_bin').agg({
        'win': ['sum', 'count', 'mean'],
        '単勝オッズ': 'mean'
    })
    pci_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
    pci_group['簡易回収率'] = pci_group['勝率'] * pci_group['平均オッズ'] * 100
    print(pci_group)

# 補正タイム分析
if '補正タイム' in df_valid.columns:
    print("\n■ 補正タイム別 勝率分析:")
    df_valid['補正タイム_num'] = pd.to_numeric(df_valid['補正タイム'], errors='coerce')
    df_valid['補正タイム_bin'] = pd.cut(df_valid['補正タイム_num'], bins=[-np.inf, 90, 95, 100, 105, 110, np.inf], 
                                  labels=['~90', '90-95', '95-100', '100-105', '105-110', '110~'])
    time_group = df_valid.groupby('補正タイム_bin').agg({
        'win': ['sum', 'count', 'mean'],
        '単勝オッズ': 'mean'
    })
    time_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
    time_group['簡易回収率'] = time_group['勝率'] * time_group['平均オッズ'] * 100
    print(time_group)

# ========================================
# 5. 脚質・ブリンカー分析
# ========================================
print()
print("=" * 60)
print("5. 脚質・ブリンカー分析")
print("=" * 60)

if '脚質' in df_valid.columns:
    print("\n■ 脚質別 勝率分析:")
    style_group = df_valid.groupby('脚質').agg({
        'win': ['sum', 'count', 'mean'],
        '単勝オッズ': 'mean'
    })
    style_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
    style_group['簡易回収率'] = style_group['勝率'] * style_group['平均オッズ'] * 100
    print(style_group)

if 'ブリンカー' in df_valid.columns:
    print("\n■ ブリンカー別 勝率分析:")
    blinker_group = df_valid.groupby('ブリンカー').agg({
        'win': ['sum', 'count', 'mean'],
        '単勝オッズ': 'mean'
    })
    blinker_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
    blinker_group['簡易回収率'] = blinker_group['勝率'] * blinker_group['平均オッズ'] * 100
    print(blinker_group)

# ========================================
# 6. 馬体重増減分析
# ========================================
print()
print("=" * 60)
print("6. 馬体重増減分析")
print("=" * 60)

if '増減' in df_valid.columns:
    print("\n■ 馬体重増減別 勝率分析:")
    df_valid['増減_num'] = pd.to_numeric(df_valid['増減'], errors='coerce')
    df_valid['増減_bin'] = pd.cut(df_valid['増減_num'], bins=[-np.inf, -20, -10, -4, 0, 4, 10, 20, np.inf], 
                                  labels=['大幅減(-20~)', '-20~-10', '-10~-4', '-4~0', '0~4', '4~10', '10~20', '大幅増(20~)'])
    weight_group = df_valid.groupby('増減_bin').agg({
        'win': ['sum', 'count', 'mean'],
        '単勝オッズ': 'mean'
    })
    weight_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
    weight_group['簡易回収率'] = weight_group['勝率'] * weight_group['平均オッズ'] * 100
    print(weight_group)

# ========================================
# 7. 印（◎○▲△など）分析
# ========================================
print()
print("=" * 60)
print("7. 印（◎○▲△など）分析")
print("=" * 60)

mark_cols = ['馬印1', 'レース印1', 'レース印2', 'レース印3']
for mark_col in mark_cols:
    if mark_col in df_valid.columns and df_valid[mark_col].notna().sum() > 1000:
        print(f"\n■ {mark_col}別 勝率分析:")
        mark_group = df_valid.groupby(mark_col).agg({
            'win': ['sum', 'count', 'mean'],
            '単勝オッズ': 'mean'
        })
        mark_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
        mark_group['簡易回収率'] = mark_group['勝率'] * mark_group['平均オッズ'] * 100
        mark_group = mark_group.sort_values('勝率', ascending=False).head(10)
        print(mark_group)

# ========================================
# 8. 上り3F分析
# ========================================
print()
print("=" * 60)
print("8. 上り3F関連分析")
print("=" * 60)

if 'Ave-3F' in df_valid.columns:
    print("\n■ Ave-3F (平均上り3F) 分析:")
    df_valid['Ave-3F_num'] = pd.to_numeric(df_valid['Ave-3F'], errors='coerce')
    print(f"  平均: {df_valid['Ave-3F_num'].mean():.2f}")
    print(f"  標準偏差: {df_valid['Ave-3F_num'].std():.2f}")

if '-3F差' in df_valid.columns:
    print("\n■ -3F差 分析:")
    df_valid['-3F差_num'] = pd.to_numeric(df_valid['-3F差'], errors='coerce')
    df_valid['-3F差_bin'] = pd.cut(df_valid['-3F差_num'], bins=[-np.inf, -1.0, -0.5, 0, 0.5, 1.0, np.inf], 
                                  labels=['大幅速(-1.0~)', '-1.0~-0.5', '-0.5~0', '0~0.5', '0.5~1.0', '遅(1.0~)'])
    f3_group = df_valid.groupby('-3F差_bin').agg({
        'win': ['sum', 'count', 'mean'],
        '単勝オッズ': 'mean'
    })
    f3_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
    f3_group['簡易回収率'] = f3_group['勝率'] * f3_group['平均オッズ'] * 100
    print(f3_group)

# ========================================
# 9. 父タイプ・母父タイプ分析
# ========================================
print()
print("=" * 60)
print("9. 血統タイプ分析")
print("=" * 60)

for type_col in ['父タイプ', '母父タイプ']:
    if type_col in df_valid.columns and df_valid[type_col].notna().sum() > 1000:
        print(f"\n■ {type_col}別 勝率分析:")
        type_group = df_valid.groupby(type_col).agg({
            'win': ['sum', 'count', 'mean'],
            '単勝オッズ': 'mean'
        })
        type_group.columns = ['勝利数', 'レース数', '勝率', '平均オッズ']
        type_group['簡易回収率'] = type_group['勝率'] * type_group['平均オッズ'] * 100
        type_group = type_group.sort_values('簡易回収率', ascending=False).head(10)
        print(type_group)

# ========================================
# 10. 推奨特徴量のまとめ
# ========================================
print()
print("=" * 60)
print("10. 回収率向上に寄与する推奨特徴量")
print("=" * 60)

print("""
【強く推奨する特徴量】
1. PCI (ペースチェンジ指数)
   - レースのペースを数値化した指標
   - 着順・勝率と高い相関を持つ可能性

2. 補正タイム
   - 馬場状態や距離を考慮した標準化タイム
   - 能力比較に有効

3. 上り3F関連 (上り3F順位, Ave-3F, -3F差)
   - 終盤の脚力を示す重要指標
   - 勝敗に直結する項目

4. 脚質
   - 逃げ・先行・差し・追込のスタイル
   - コース/距離/馬場状態との組み合わせで有効

5. 馬印1, レース印1 (予想印)
   - 専門家の評価を数値化できる可能性

【追加検討する特徴量】
6. 馬体重増減
   - 急激な増減は調子のサイン

7. ブリンカー
   - 集中力向上による成績変化

8. 父タイプ / 母父タイプ
   - 血統傾向の分類

9. 基準タイム(秒)
   - レースの基準となるタイム（馬場/距離考慮）

10. 通過順位（1角〜4角）
    - 位置取りとレース展開の分析

【注意が必要な特徴量】
- レースコメント / 結果コメント / 馬コメント
  → テキストデータのため、NLP処理が必要
""")

print("\n分析完了!")
