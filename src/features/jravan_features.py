"""
JRA-VAN固有の特徴量を抽出・マージするモジュール

リーク防止のため:
- PCI系、上り3F関連、脚質、通過順位 → 前走データとして使用
- 父タイプ、母父タイプ → そのまま使用（不変の属性）
- 馬体重増減 → そのまま使用（レース前に公開）
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JraVanFeatures:
    """JRA-VAN固有の特徴量を抽出するクラス"""
    
    def __init__(self, jravan_path='data/jra_van/race/2015-2025.csv'):
        self.jravan_path = jravan_path
        self.jravan_df = None
        self._loaded = False
        
        # JRA-VANの場所コードマッピング（netkeiba形式への変換）
        self.place_code_map = {
            '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
            '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10'
        }
        
        # 脚質のエンコーディング
        self.running_style_map = {
            '逃げ': 1, 'ﾏｸﾘ': 1,  # 逃げ系
            '先行': 2,
            '差し': 3,
            '追込': 4,
            '中団': 3,  # 差しに近い
            '後方': 4   # 追込に近い
        }
    
    def load_data(self):
        """JRA-VANデータを読み込む"""
        if self._loaded:
            return self.jravan_df
            
        try:
            logger.info(f"Loading JRA-VAN data from {self.jravan_path}...")
            self.jravan_df = pd.read_csv(
                self.jravan_path, 
                encoding='cp932', 
                low_memory=False
            )
            logger.info(f"Loaded {len(self.jravan_df):,} rows from JRA-VAN data")
            
            # 基本的な前処理
            self._preprocess()
            self._loaded = True
            
            return self.jravan_df
            
        except Exception as e:
            logger.error(f"Failed to load JRA-VAN data: {e}")
            return None
    
    def _preprocess(self):
        """JRA-VANデータの前処理"""
        df = self.jravan_df
        
        # 日付カラム作成
        df['date_jv'] = pd.to_datetime(
            df['年'].astype(str).str.zfill(2).apply(lambda x: '20' + x if int(x) < 50 else '19' + x) + 
            df['月'].astype(str).str.zfill(2) + 
            df['日'].astype(str).str.zfill(2),
            format='%Y%m%d',
            errors='coerce'
        )
        
        # 場所コード変換
        df['place_code_jv'] = df['場所'].map(self.place_code_map)
        
        # 血統登録番号を文字列に統一
        df['horse_id_jv'] = df['血統登録番号'].astype(str)
        
        # 数値カラムの変換
        numeric_cols = ['PCI', 'RPCI', 'PCI3', '上り3F順位', 'Ave-3F', '-3F差', '増減',
                        '通過順1角', '通過順2角', '通過順3角', '通過順4角']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 脚質の数値エンコード
        if '脚質' in df.columns:
            df['running_style_encoded'] = df['脚質'].map(self.running_style_map).fillna(0)
        
        self.jravan_df = df
    
    def _create_horse_history(self):
        """馬ごとの履歴データを作成（前走データ抽出用）"""
        df = self.jravan_df.copy()
        
        # 日付とレース番号でソート
        df = df.sort_values(['horse_id_jv', 'date_jv', 'レース番号'])
        
        # 前走データを作成（shift(1)で1つ前のレースを取得）
        prev_cols = {
            'PCI': 'prev_pci',
            'RPCI': 'prev_rpci',
            'PCI3': 'prev_pci3',
            '上り3F順位': 'prev_agari3f_rank',
            'Ave-3F': 'prev_ave3f',
            '-3F差': 'prev_3f_diff',
            'running_style_encoded': 'prev_running_style_jv',
            '通過順1角': 'prev_corner1',
            '通過順2角': 'prev_corner2',
            '通過順3角': 'prev_corner3',
            '通過順4角': 'prev_corner4',
            '確定着順': 'prev_rank_jv'
        }
        
        for orig_col, new_col in prev_cols.items():
            if orig_col in df.columns:
                df[new_col] = df.groupby('horse_id_jv')[orig_col].shift(1)
        
        return df
    
    def extract_features_for_merge(self):
        """マージ用の特徴量を抽出"""
        if not self._loaded:
            self.load_data()
        
        if self.jravan_df is None:
            return None
        
        # 前走データを含むデータフレームを作成
        df_with_history = self._create_horse_history()
        
        # マージに必要なカラムを選択
        merge_cols = [
            'horse_id_jv', 'date_jv', 'place_code_jv', 'レース番号',
            # 前走特徴量
            'prev_pci', 'prev_rpci', 'prev_pci3',
            'prev_agari3f_rank', 'prev_ave3f', 'prev_3f_diff',
            'prev_running_style_jv',
            'prev_corner1', 'prev_corner2', 'prev_corner3', 'prev_corner4',
            'prev_rank_jv',
            # 静的特徴量
            '父タイプ', '母父タイプ',
            # 当日特徴量
            '増減'
        ]
        
        # 存在するカラムのみ選択
        available_cols = [c for c in merge_cols if c in df_with_history.columns]
        result = df_with_history[available_cols].copy()
        
        # カラム名をリネーム
        result = result.rename(columns={
            '父タイプ': 'sire_type',
            '母父タイプ': 'bms_type',
            '増減': 'weight_change_jv'
        })
        
        return result
    
    def merge_with_scraped_data(self, df_scraped):
        """
        スクレイピングデータにJRA-VAN特徴量をマージ
        
        Args:
            df_scraped: スクレイピングで取得したデータフレーム
                       必須カラム: horse_id, date_dt (または date), race_id
        
        Returns:
            JRA-VAN特徴量がマージされたデータフレーム
        """
        if not self._loaded:
            self.load_data()
        
        if self.jravan_df is None:
            logger.warning("JRA-VAN data not available. Returning original data.")
            return df_scraped
        
        df = df_scraped.copy()
        
        # JRA-VAN特徴量を抽出
        jv_features = self.extract_features_for_merge()
        if jv_features is None:
            return df
        
        # === マージキーの準備（スクレイピングデータ側） ===
        
        # horse_idを文字列に変換（浮動小数点の.0を除去）
        if 'horse_id' in df.columns:
            # horse_idがfloatの場合、intに変換してから文字列化（.0を除去）
            df['horse_id_str'] = df['horse_id'].apply(
                lambda x: str(int(x)) if pd.notna(x) and not isinstance(x, str) else str(x).replace('.0', '')
            )
        else:
            logger.warning("horse_id column not found in scraped data")
            return df
        
        # 日付の準備（複数フォーマット対応）
        if 'date_dt' not in df.columns or df['date_dt'].isna().all():
            if 'date' in df.columns:
                # 日本語フォーマット: 2018年12月15日
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
                
                # パースできなかった場合、他のフォーマットを試す
                if df['date_dt'].isna().all():
                    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
        
        if 'date_dt' not in df.columns or df['date_dt'].isna().all():
            logger.warning("Failed to parse date in scraped data. Trying alternative merge strategy.")
            # 日付なしでのマージを試みる
            return self._merge_by_horse_id_only(df, jv_features)
        
        # race_idからレース番号と場所コードを抽出
        if 'race_id' in df.columns:
            race_id_str = df['race_id'].astype(str)
            df['race_no_scraped'] = race_id_str.str[-2:].astype(int)
            df['place_code_scraped'] = race_id_str.str[4:6]
            # 年も抽出（race_idの先頭4桁）
            df['year_scraped'] = race_id_str.str[:4].astype(int)
        else:
            logger.warning("race_id column not found")
            return self._merge_by_horse_id_only(df, jv_features)
        
        # === JRA-VAN側の準備 ===
        jv_features = jv_features.copy()
        jv_features['date_jv'] = pd.to_datetime(jv_features['date_jv'])
        jv_features['year_jv'] = jv_features['date_jv'].dt.year
        
        # デバッグ: キーの値を確認
        logger.info(f"Scraped data - horse_id sample: {df['horse_id_str'].head(3).tolist()}")
        logger.info(f"JRA-VAN data - horse_id sample: {jv_features['horse_id_jv'].head(3).tolist()}")
        logger.info(f"Scraped data - date_dt sample: {df['date_dt'].head(3).tolist()}")
        logger.info(f"JRA-VAN data - date_jv sample: {jv_features['date_jv'].head(3).tolist()}")
        
        # === マージ実行 ===
        # 試行1: 馬ID + 日付 + 場所 + レース番号（最も厳密）
        df_merged = df.merge(
            jv_features,
            left_on=['horse_id_str', 'date_dt', 'place_code_scraped', 'race_no_scraped'],
            right_on=['horse_id_jv', 'date_jv', 'place_code_jv', 'レース番号'],
            how='left',
            suffixes=('', '_jv_dup')
        )
        
        merged_count = df_merged['prev_pci'].notna().sum()
        total_count = len(df_merged)
        
        # マージ率が低い場合、馬ID + 年 + 場所 + レース番号でリトライ
        if merged_count / total_count < 0.1:
            logger.info("Low merge rate with date. Trying year-based merge...")
            
            df_merged = df.merge(
                jv_features,
                left_on=['horse_id_str', 'year_scraped', 'place_code_scraped', 'race_no_scraped'],
                right_on=['horse_id_jv', 'year_jv', 'place_code_jv', 'レース番号'],
                how='left',
                suffixes=('', '_jv_dup')
            )
            merged_count = df_merged['prev_pci'].notna().sum()
        
        # 一時カラムを削除
        temp_cols = ['horse_id_str', 'race_no_scraped', 'place_code_scraped', 'year_scraped',
                     'horse_id_jv', 'date_jv', 'place_code_jv', 'レース番号', 'year_jv']
        for col in temp_cols:
            if col in df_merged.columns:
                df_merged.drop(columns=[col], inplace=True, errors='ignore')
        
        logger.info(f"JRA-VAN features merged: {merged_count:,}/{total_count:,} ({merged_count/total_count*100:.1f}%)")
        
        return df_merged
    
    def _merge_by_horse_id_only(self, df, jv_features):
        """馬IDのみでマージ（最新のJRA-VANデータを使用）"""
        logger.info("Falling back to horse_id-only merge (using latest JRA-VAN data per horse)")
        
        # 各馬の最新データを取得
        jv_latest = jv_features.sort_values('date_jv').groupby('horse_id_jv').last().reset_index()
        
        df_merged = df.merge(
            jv_latest[['horse_id_jv', 'prev_pci', 'prev_rpci', 'prev_pci3',
                       'prev_agari3f_rank', 'prev_ave3f', 'prev_3f_diff',
                       'prev_running_style_jv', 'prev_corner1', 'prev_corner2', 
                       'prev_corner3', 'prev_corner4', 'prev_rank_jv',
                       'sire_type', 'bms_type', 'weight_change_jv']],
            left_on='horse_id_str',
            right_on='horse_id_jv',
            how='left'
        )
        
        # 一時カラム削除
        if 'horse_id_str' in df_merged.columns:
            df_merged.drop(columns=['horse_id_str'], inplace=True)
        if 'horse_id_jv' in df_merged.columns:
            df_merged.drop(columns=['horse_id_jv'], inplace=True)
        
        merged_count = df_merged['prev_pci'].notna().sum()
        logger.info(f"Horse-ID only merge: {merged_count:,}/{len(df_merged):,} ({merged_count/len(df_merged)*100:.1f}%)")
        
        return df_merged
    
    def merge_all_features(self, df):
        """
        全ての特徴量をマージ（FeatureEngineerから呼び出される）
        
        Args:
            df: 特徴量生成途中のデータフレーム
        
        Returns:
            JRA-VAN特徴量が追加されたデータフレーム
        """
        return self.merge_with_scraped_data(df)


if __name__ == "__main__":
    # テスト用コード
    jv = JraVanFeatures()
    jv.load_data()
    
    print("=== JRA-VAN Data Info ===")
    print(f"Total rows: {len(jv.jravan_df):,}")
    print(f"Date range: {jv.jravan_df['date_jv'].min()} - {jv.jravan_df['date_jv'].max()}")
    
    # 特徴量抽出テスト
    features = jv.extract_features_for_merge()
    print(f"\n=== Extracted Features ===")
    print(f"Columns: {features.columns.tolist()}")
    print(f"Sample data:")
    print(features.head())
    
    # 欠損率の確認
    print(f"\n=== Missing Rate ===")
    for col in features.columns:
        missing_rate = features[col].isna().sum() / len(features) * 100
        print(f"{col}: {missing_rate:.1f}%")
