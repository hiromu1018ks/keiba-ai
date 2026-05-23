"""欠損率分析・NaN意味分類スクリプト (読み取り専用)"""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/backtest/bt_2024_horse_features.parquet')
n = len(df)
stats = pd.read_csv('tmp/missing_stats.csv')

wide_cols = [c for c in df.columns if c.startswith('wide_odds_')]
full_nan_cols = [c for c in df.columns if df[c].isna().all()]


def classify_nan(col, dtype, missing_rate, sample_vals, unique_count):
    reasons = []
    nan_type = 'G'
    handling = 'investigate'
    confidence = 'low'
    doc_meaning = ''

    # === wide_odds ===
    if col.startswith('wide_odds_'):
        parts = col.replace('wide_odds_', '').split('_')
        i, j = int(parts[0]), int(parts[1])
        doc_meaning = f'ワイドオッズ 馬番{i}-{j}'
        if missing_rate >= 99.99:
            nan_type = 'A'
            handling = 'leakage_or_drop_candidate'
            confidence = 'high'
            reasons.append('100% NaN - 常に出走頭数不足. 削除推奨')
        else:
            nan_type = 'A'
            handling = 'keep_nan_candidate'
            confidence = 'high'
            reasons.append('レース頭数に応じて自然にNaN (出走頭数<max(i,j))')
        return nan_type, handling, confidence, doc_meaning, '; '.join(reasons)

    # === 100% NaN ===
    if missing_rate >= 99.99:
        doc_map_100 = {
            'coursekubuncd': 'コース区分コード (RA, 障害用)',
            'coursekubuncdbefore': 'コース区分変更前 (RA)',
            'jyuni1': '1コーナー通過順位 (RA, レース後)',
            'jyuni2': '2コーナー通過順位 (RA, レース後)',
            'jyuni3': '3コーナー通過順位 (RA, レース後)',
            'jyuni4': '4コーナー通過順位 (RA, レース後)',
            'fukusyoku': '服色標示 (SE)',
            'reserved1': '予備1 (SE)',
            'zogenfugo': '増減符号 (SE, zogensaで代替済み)',
            'chakusacdp': '+着差コード (SE, レース後)',
            'chakusacdpp': '++着差コード (SE, レース後)',
            'bamei1': '相手馬1名 (SE, 同着)',
            'bamei2': '相手馬2名 (SE, 同着)',
            'bamei3': '相手馬3名 (SE, 同着)',
            'dam_wr': '母父勝率 (未実装/データ不十分)',
            'dam_surface_wr': '母父芝ダ別勝率 (未実装)',
            'dam_prize_log': '母父賞金log (未実装)',
            'breeder_strength': '生産者強度 (未実装)',
            'course_record_time': 'コースレコードタイム (未実装)',
            'dm_time_rank': 'マイニング順位 (未実装)',
            'dm_time_zscore': 'マイニングzscore (未実装)',
            'dm_confidence_range': 'マイニング信頼区間 (未実装)',
            'dm_time_margin_to_fav': 'マイニング人気差 (未実装)',
        }
        doc_meaning = doc_map_100.get(col, '100% NaN 未使用/未実装')
        if col in ['jyuni1','jyuni2','jyuni3','jyuni4','chakusacdp','chakusacdpp']:
            nan_type = 'D'
            handling = 'leakage_or_drop_candidate'
            confidence = 'high'
            reasons.append('レース後情報. 実害ないが削除推奨')
        elif col in ['dam_wr','dam_surface_wr','dam_prize_log','breeder_strength',
                      'course_record_time','dm_time_rank','dm_time_zscore',
                      'dm_confidence_range','dm_time_margin_to_fav']:
            nan_type = 'B'
            handling = 'leakage_or_drop_candidate'
            confidence = 'high'
            reasons.append('未実装特徴量. 削除推奨')
        else:
            nan_type = 'G'
            handling = 'leakage_or_drop_candidate'
            confidence = 'high'
            reasons.append('100% NaN. 削除推奨')
        return nan_type, handling, confidence, doc_meaning, '; '.join(reasons)

    # === 0% (完全データ) ===
    if missing_rate == 0:
        all_0pct = {
            'recordspec': ('レコード種別ID', 'N', 'keep'), 'makedate': ('データ作成年月日', 'N', 'keep'),
            'year': ('開催年', 'N', 'keep'), 'monthday': ('開催月日', 'N', 'keep'),
            'jyocd': ('競馬場コード', 'N', 'keep'), 'kaiji': ('開催回', 'N', 'keep'),
            'nichiji': ('開催日目', 'N', 'keep'), 'racenum': ('レース番号', 'N', 'keep'),
            'youbicd': ('曜日コード', 'N', 'keep'), 'tokunum': ('特別レース番号', 'N', 'keep'),
            'hondai': ('レース名(漢字)', 'E', 'unknown_category_candidate'),
            'fukudai': ('副題', 'E', 'unknown_category_candidate'),
            'kakko': ('括弧題名', 'E', 'unknown_category_candidate'),
            'hondaieng': ('レース名(英字)', 'E', 'unknown_category_candidate'),
            'fukudaieng': ('副題(英字)', 'E', 'unknown_category_candidate'),
            'kakkoeng': ('括弧題名(英字)', 'E', 'unknown_category_candidate'),
            'ryakusyo10': ('レース名略称10字', 'E', 'unknown_category_candidate'),
            'ryakusyo6': ('レース名略称6字', 'E', 'unknown_category_candidate'),
            'ryakusyo3': ('レース名略称3字', 'E', 'unknown_category_candidate'),
            'kubun': ('レース区分', 'N', 'keep'), 'nkai': ('回数', 'N', 'keep'),
            'gradecd': ('グレードコード', 'N', 'keep'),
            'gradecdbefore': ('グレード変更前', 'N', 'keep'),
            'syubetucd': ('種別コード(芝/ダ/障)', 'N', 'keep'),
            'jyuryocd': ('重量コード', 'N', 'keep'),
            'jyokencd1': ('条件コード1', 'N', 'keep'), 'jyokencd2': ('条件コード2', 'N', 'keep'),
            'jyokencd3': ('条件コード3', 'N', 'keep'), 'jyokencd4': ('条件コード4', 'N', 'keep'),
            'jyokencd5': ('条件コード5', 'N', 'keep'), 'jyokenname': ('条件名', 'E', 'unknown_category_candidate'),
            'kyori': ('距離(m)', 'N', 'keep'), 'kyoribefore': ('距離変更前', 'N', 'keep'),
            'trackcd': ('トラックコード', 'N', 'keep'), 'trackcdbefore': ('トラック変更前', 'N', 'keep'),
            'honsyokin1': ('本賞金1着', 'N', 'keep'), 'honsyokin2': ('本賞金2着', 'N', 'keep'),
            'honsyokin3': ('本賞金3着', 'N', 'keep'), 'honsyokin4': ('本賞金4着', 'N', 'keep'),
            'honsyokin5': ('本賞金5着', 'N', 'keep'), 'honsyokin6': ('本賞金6着', 'N', 'keep'),
            'honsyokin7': ('本賞金7着', 'N', 'keep'),
            'honsyokinbefore1': ('本賞金変更前1着', 'N', 'keep'),
            'honsyokinbefore2': ('本賞金変更前2着', 'N', 'keep'),
            'honsyokinbefore3': ('本賞金変更前3着', 'N', 'keep'),
            'honsyokinbefore4': ('本賞金変更前4着', 'N', 'keep'),
            'honsyokinbefore5': ('本賞金変更前5着', 'N', 'keep'),
            'fukasyokin1': ('付加賞金1着', 'N', 'keep'),
            'fukasyokin2': ('付加賞金2着', 'N', 'keep'),
            'fukasyokin3': ('付加賞金3着', 'N', 'keep'),
            'fukasyokin4': ('付加賞金4着', 'N', 'keep'),
            'fukasyokin5': ('付加賞金5着', 'N', 'keep'),
            'fukasyokinbefore1': ('付加賞金変更前1着', 'N', 'keep'),
            'fukasyokinbefore2': ('付加賞金変更前2着', 'N', 'keep'),
            'fukasyokinbefore3': ('付加賞金変更前3着', 'N', 'keep'),
            'hassotime': ('発走時刻', 'N', 'keep'),
            'hassotimebefore': ('発走時刻変更前', 'N', 'keep'),
            'torokutosu': ('登録頭数', 'N', 'keep'), 'syussotosu': ('出走頭数', 'N', 'keep'),
            'tenkocd': ('天候コード', 'N', 'keep'),
            'sibababacd': ('芝馬場状態コード', 'N', 'keep'),
            'dirtbabacd': ('ダート馬場状態コード', 'N', 'keep'),
            'syogaimiletime': ('障害マイルタイム', 'N', 'keep'),
            'corner1': ('コーナー数1', 'N', 'keep'), 'syukaisu1': ('周回数1', 'N', 'keep'),
            'corner2': ('コーナー数2', 'N', 'keep'), 'syukaisu2': ('周回数2', 'N', 'keep'),
            'corner3': ('コーナー数3', 'N', 'keep'), 'syukaisu3': ('周回数3', 'N', 'keep'),
            'corner4': ('コーナー数4', 'N', 'keep'), 'syukaisu4': ('周回数4', 'N', 'keep'),
            'race_date': ('レース日付', 'N', 'keep'), 'race_id': ('レースID', 'N', 'keep'),
            'surface': ('芝/ダート (computed)', 'N', 'keep'),
            'track_condition_code': ('馬場状態コード (computed)', 'N', 'keep'),
            'wakuban': ('枠番', 'N', 'keep'), 'umaban': ('馬番', 'N', 'keep'),
            'kettonum': ('血統登録番号', 'N', 'keep'),
            'bamei': ('馬名', 'E', 'unknown_category_candidate'),
            'umakigocd': ('馬記号コード', 'N', 'keep'), 'sexcd': ('性別コード', 'N', 'keep'),
            'hinsyucd': ('品種コード', 'N', 'keep'), 'keirocd': ('毛色コード', 'N', 'keep'),
            'barei': ('馬齢', 'N', 'keep'), 'tozaicd': ('東西所属コード', 'N', 'keep'),
            'chokyosicode': ('調教師コード', 'N', 'keep'),
            'chokyosiryakusyo': ('調教師名略称', 'E', 'unknown_category_candidate'),
            'banusicode': ('馬主コード', 'N', 'keep'),
            'banusiname': ('馬主名', 'E', 'unknown_category_candidate'),
            'futan': ('負担重量', 'N', 'keep'), 'futanbefore': ('変更前負担重量', 'N', 'keep'),
            'blinker': ('ブリンカー区分', 'N', 'keep'), 'reserved2': ('予備2 (SE)', 'N', 'keep'),
            'kisyucode': ('騎手コード', 'N', 'keep'),
            'kisyucodebefore': ('変更前騎手コード', 'N', 'keep'),
            'kisyuryakusyo': ('騎手名略称', 'E', 'unknown_category_candidate'),
            'kisyuryakusyobefore': ('変更前騎手名略称', 'E', 'unknown_category_candidate'),
            'minaraicd': ('騎手見習コード', 'N', 'keep'),
            'minaraicdbefore': ('変更前騎手見習コード', 'N', 'keep'),
            'bataijyu': ('馬体重(kg)', 'N', 'keep'),
            'odds': ('単勝オッズ (tanodds上書き済み)', 'N', 'keep'),
            'reserved3': ('予備3 (SE)', 'N', 'keep'), 'reserved4': ('予備4 (SE)', 'N', 'keep'),
            'kettonum1': ('血統登録番号(父)', 'N', 'keep'),
            'kettonum2': ('血統登録番号(母父)', 'N', 'keep'),
            'kettonum3': ('血統登録番号(母母父)', 'N', 'keep'),
            'dmkubun': ('マイニング区分', 'N', 'keep'),
            'dmgosap': ('マイニング誤差+', 'N', 'keep'),
            'dmgosam': ('マイニング誤差-', 'N', 'keep'),
            'tanodds': ('単勝オッズ(スナップショット)', 'N', 'keep'),
            'fukuoddslow': ('複勝オッズ下限', 'N', 'keep'),
            'tanninki': ('単勝人気順', 'N', 'keep'),
            'distance_bin': ('距離帯カテゴリ', 'N', 'keep'),
            'grade_code': ('グレードコード', 'N', 'keep'),
            'effective_jyokencd': ('有効条件コード', 'N', 'keep'),
            'class_level_current': ('現在クラスレベル', 'N', 'keep'),
            'class_level_source_flag': ('クラスレベルソース', 'N', 'keep'),
            'class_bucket': ('クラスバケット名', 'N', 'keep'),
            'class_regime_after_202406': ('2024/6以降制度フラグ', 'N', 'keep'),
            'field_size': ('フィールドサイズ', 'N', 'keep'),
            'popularity_rank_fallback_used': ('人気順フォールバック使用', 'N', 'keep'),
            'popularity_rank': ('人気順', 'N', 'keep'),
            'draw_ratio': ('枠順比率', 'N', 'keep'),
            'frame_number': ('枠番(数値)', 'N', 'keep'),
            'blinker_on': ('ブリンカー装着フラグ', 'N', 'keep'),
            'weight_change_known': ('体重変化既知フラグ', 'N', 'keep'),
            'weight_change_missing_flag': ('体重変化不明フラグ', 'N', 'keep'),
            'weight_diff_from_mean': ('平均体重との差分', 'N', 'keep'),
            'odds_rank': ('オッズ順位', 'N', 'keep'),
            'odds_drop_rate_60_10': ('オッズ下落率(60->10分)', 'N', 'keep'),
            'odds_drop_rate_30_10': ('オッズ下落率(30->10分)', 'N', 'keep'),
            'odds_velocity': ('オッズ変化速度', 'N', 'keep'),
            'popularity_change_30_10': ('人気変化(30->10分)', 'N', 'keep'),
            'odds_acceleration': ('オッズ加速度', 'N', 'keep'),
            'odds_direction_consistency': ('オッズ方向一貫性', 'N', 'keep'),
            'overround': ('オーバーラウンド', 'N', 'keep'),
            'p_market_win_adj': ('市場勝率調整値', 'N', 'keep'),
            'market_entropy': ('市場エントロピー', 'N', 'keep'),
            'odds_skewness': ('オッズ歪度', 'N', 'keep'),
            'implied_prob_hhi': ('暗示確率HHI', 'N', 'keep'),
            'difficulty_score': ('レース難易度スコア', 'N', 'keep'),
            'rl_log_odds_entropy': ('RL対数オッズエントロピー', 'N', 'keep'),
            'rl_odds_dispersion': ('RLオッズ分散', 'N', 'keep'),
            'rl_top1_odds': ('RL1位オッズ', 'N', 'keep'),
            'rl_top3_odds_gap': ('RL上位3頭オッズギャップ', 'N', 'keep'),
            'rl_favorite_rank_gap': ('RL人気順位ギャップ', 'N', 'keep'),
            'rl_n_horses': ('RL出走頭数', 'N', 'keep'),
            'rl_favorite_in_wide_top1': ('RL1番人気ワイド1位フラグ', 'N', 'keep'),
            'rl_trio_overlap': ('RL三連複重複数', 'N', 'keep'),
            'rl_market_consistency': ('RL市場一貫性', 'N', 'keep'),
            'blood_keito_cd': ('血統系統コード', 'N', 'keep'),
            'is_turf_sprint': ('芝スプリントフラグ', 'N', 'keep'),
            'is_turf_mile': ('芝マイルフラグ', 'N', 'keep'),
            'is_turf_intermediate': ('芝中距離フラグ', 'N', 'keep'),
            'is_turf_long': ('芝長距離フラグ', 'N', 'keep'),
            'is_dirt_sprint': ('ダートスプリントフラグ', 'N', 'keep'),
            'is_dirt_mile': ('ダートマイルフラグ', 'N', 'keep'),
            'is_dirt_intermediate': ('ダート中距離フラグ', 'N', 'keep'),
            'is_good_track': ('良馬場フラグ', 'N', 'keep'),
            'is_soft_track': ('重馬場フラグ', 'N', 'keep'),
            'sire_id': ('種牡馬ID', 'N', 'keep'),
            'bms_id': ('母父ID', 'N', 'keep'),
            'bms_wr': ('母父勝率', 'N', 'keep'),
            'sire_distance_wr': ('種牡馬距離別勝率', 'N', 'keep'),
            'sire_wr': ('種牡馬勝率', 'N', 'keep'),
            'sire_surface_wr': ('種牡馬芝ダ別勝率', 'N', 'keep'),
            'bms_surface_starts_log': ('母父芝ダ出走数log', 'N', 'keep'),
            'bms_surface_wr': ('母父芝ダ別勝率', 'N', 'keep'),
            'sire_prize_avg': ('種牡馬平均賞金log', 'N', 'keep'),
            'bms_starts_log': ('母父出走数log', 'N', 'keep'),
            'bms_distance_starts_log': ('母父距離別出走数log', 'N', 'keep'),
            'bms_has_history': ('母父履歴ありフラグ', 'N', 'keep'),
            'bms_distance_wr': ('母父距離別勝率', 'N', 'keep'),
            'weight_absolute': ('馬体重(絶対値)', 'N', 'keep'),
            'is_debut': ('初出走フラグ', 'N', 'keep'),
            'blinker_change': ('ブリンカー変更フラグ', 'N', 'keep'),
            'kyakusitu_x_distance': ('脚質x距離帯交差', 'N', 'keep'),
            'kyakusitu_x_surface': ('脚質x芝ダ交差', 'N', 'keep'),
            'weight_x_distance': ('体重x距離交差', 'N', 'keep'),
            'surface_x_distance_bin': ('芝ダx距離帯交差', 'N', 'keep'),
            'grade_code_x_distance_bin': ('グレードx距離帯交差', 'N', 'keep'),
            'sire_wr_x_distance': ('種牡馬勝率x距離交差', 'N', 'keep'),
            'weight_x_class': ('体重xクラス交差', 'N', 'keep'),
            'race_mean_fuku_odds': ('レース平均複勝オッズ', 'N', 'keep'),
            'race_std_fuku_odds': ('レース複勝オッズ標準偏差', 'N', 'keep'),
            'odds_gap_fav12': ('1-2番人気オッズギャップ', 'N', 'keep'),
            'odds_popularity_gap': ('オッズ人気ギャップ', 'N', 'keep'),
            'surface_track_interaction': ('芝ダx馬場状態交差', 'N', 'keep'),
            'pace_pressure': ('ペース圧力', 'N', 'keep'),
            'closer_share': ('差し馬シェア', 'N', 'keep'),
            'pace_scenario_fit': ('ペースシナリオ適合度', 'N', 'keep'),
            'rel_sire_quality_rank': ('相対種牡馬品質順位', 'N', 'keep'),
            'rel_bms_quality_rank': ('相対母父品質順位', 'N', 'keep'),
            'rel_bms_surface_quality_rank': ('相対母父芝ダ品質順位', 'N', 'keep'),
            'rel_fuku_odds_zscore': ('相対複勝オッズzscore', 'N', 'keep'),
            'rel_popularity_rank_zscore': ('相対人気順位zscore', 'N', 'keep'),
            'market_log_error_win': ('市場対数誤差(単勝)', 'N', 'keep'),
            'signed_log_error_win': ('符号付き対数誤差', 'N', 'keep'),
            'abs_log_error_win': ('絶対対数誤差', 'N', 'keep'),
            'market_pred_error_win': ('市場予測誤差', 'N', 'keep'),
            'market_error_rank_in_race': ('市場誤差レース内順位', 'N', 'keep'),
            'p_ability_win': ('能力勝率(Stage1)', 'N', 'keep'),
            'p_ability_place_raw': ('能力複勝率(raw)', 'N', 'keep'),
            'p_ability_place': ('能力複勝率(調整)', 'N', 'keep'),
            'odds_to_ability_ratio': ('オッズ/能力比', 'N', 'keep'),
            'rel_p_ability_win_zscore': ('相対能力勝率zscore', 'N', 'keep'),
            'rel_p_ability_win_rank': ('相対能力勝率順位', 'N', 'keep'),
            'rel_odds_ability_deviation': ('相対オッズ能力偏差', 'N', 'keep'),
            'te_blood_keito_cd': ('TE:血統系統コード', 'N', 'keep'),
            'te_kisyucode': ('TE:騎手コード', 'N', 'keep'),
            'te_chokyosicode': ('TE:調教師コード', 'N', 'keep'),
            'deviation_rank': ('偏差順位', 'N', 'keep'),
            'deviation_zscore': ('偏差zscore', 'N', 'keep'),
            'p_win_pred': ('予測勝率(P補正)', 'N', 'keep'),
            'e_return_win_pred': ('予測払戻(E補正)', 'N', 'keep'),
            'ev_win': ('期待値(単勝)', 'N', 'keep'),
            'jockey_wr_overall': ('騎手全体勝率', 'N', 'keep'),
            'jockey_prize_log': ('騎手賞金log', 'N', 'keep'),
            'trainer_wr_overall': ('調教師全体勝率', 'N', 'keep'),
            'trainer_wr_distance': ('調教師距離別勝率', 'N', 'keep'),
            'trainer_prize_log': ('調教師賞金log', 'N', 'keep'),
            'p_x_e_interaction': ('PxE交互作用', 'N', 'keep'),
            'p_minus_e_gap': ('P-Eギャップ', 'N', 'keep'),
            'p_win_corrected': ('補正勝率', 'N', 'keep'),
            'e_return_win_corrected': ('補正払戻', 'N', 'keep'),
            'ev_win_corrected': ('補正期待値', 'N', 'keep'),
            'ev_win_calibrated': ('較正期待値', 'N', 'keep'),
            'p_win_combined': ('統合勝率', 'N', 'keep'),
            'p_win_final': ('最終勝率', 'N', 'keep'),
            'edge_win': ('エッジ(単勝)', 'N', 'keep'),
            'win_selection_ev': ('投票判定EV', 'N', 'keep'),
            'win_selection_edge': ('投票判定エッジ', 'N', 'keep'),
            'win_selection_prob': ('投票判定確率', 'N', 'keep'),
            'win_gate_score': ('ゲートスコア', 'N', 'keep'),
            'win_gate_pass': ('ゲート通過フラグ', 'N', 'keep'),
            'win_gate_rank': ('ゲート順位', 'N', 'keep'),
            'win_gate_score_gap': ('ゲートスコアギャップ', 'N', 'keep'),
            'runner_up_gate_score': ('2着候補ゲートスコア', 'N', 'keep'),
            'runner_up_gate_score_gap': ('2着候補スコアギャップ', 'N', 'keep'),
            'runner_up_win_selection_prob': ('2着候補投票確率', 'N', 'keep'),
            'runner_up_win_selection_edge': ('2着候補投票エッジ', 'N', 'keep'),
            'runner_up_tanodds': ('2着候補単勝オッズ', 'N', 'keep'),
            'market_condition_score': ('市場状態スコア', 'N', 'keep'),
            'aggressive_strength': ('攻撃的強度', 'N', 'keep'),
            'aggressive_tier': ('攻撃的ティア', 'N', 'keep'),
            'ev_place_corrected': ('複勝補正期待値', 'N', 'keep'),
            'EV_lower_win_corrected': ('EV下限(補正)', 'N', 'keep'),
            'EV_upper_win_corrected': ('EV上限(補正)', 'N', 'keep'),
            'conformal_confidence_score': ('Conformal信頼度', 'N', 'keep'),
            'EV_lower_place': ('複勝EV下限', 'N', 'keep'),
            'kakuteijyuni': ('確定着順 (POST_RACE)', 'D', 'leakage_or_drop_candidate'),
            'confirmed_odds': ('確定オッズ (POST_RACE)', 'D', 'leakage_or_drop_candidate'),
            'win_selection_ev_raw': ('生EV', 'N', 'keep'),
            'win_selection_edge_raw': ('生エッジ', 'N', 'keep'),
            'win_selection_ev_tail_calibrated': ('テール較正EV', 'N', 'keep'),
            'selected_rank_by_p_win_final': ('p_win_final順位', 'N', 'keep'),
            'selected_rank_by_win_selection_ev': ('win_selection_ev順位', 'N', 'keep'),
            'filter_pass_flags': ('フィルタ通過フラグ', 'N', 'keep'),
            'candidate_count_before_filter': ('フィルタ前候補数', 'N', 'keep'),
            'candidate_count_after_filter': ('フィルタ後候補数', 'N', 'keep'),
            'is_actual_bet': ('実際投票フラグ', 'N', 'keep'),
        }
        if col in all_0pct:
            doc_meaning, nan_type, hand_code = all_0pct[col]
            if hand_code == 'keep':
                handling = 'keep_nan_candidate'
            elif hand_code == 'unknown_category_candidate':
                handling = 'unknown_category_candidate'
            elif hand_code == 'leakage_or_drop_candidate':
                handling = 'leakage_or_drop_candidate'
            confidence = 'high'
            if nan_type == 'D':
                reasons.append('POST_RACE列 - 学習ターゲット/評価用. 特徴量投入不可')
            elif nan_type == 'E':
                reasons.append('カテゴリ文字列 - 欠損なし')
            else:
                reasons.append('欠損なし - 完全データ')
            return nan_type, handling, confidence, doc_meaning, '; '.join(reasons)

        # 不明0%
        doc_meaning = '欠損なし (詳細不明)'
        nan_type = 'N'
        handling = 'keep_nan_candidate'
        confidence = 'medium'
        reasons.append('欠損なし')
        return nan_type, handling, confidence, doc_meaning, '; '.join(reasons)

    # === 0-5% ===
    if missing_rate <= 5:
        s05 = {
            'odds_volatility': ('オッズボラティリティ', 'C', 'keep_nan_candidate', 'high', '直前オッズ時系列が必要. 欠損=時系列なし'),
            'rl_trio_odds_ratio': ('三連複オッズ比率', 'A', 'keep_nan_candidate', 'high', 'trio oddsなし'),
            'rl_wide_harville_ratio': ('ワイドHarville比率', 'A', 'keep_nan_candidate', 'high', 'wide odds計算不可'),
            'jockey_surprise': ('騎手サプライズ指標', 'F', 'keep_nan_candidate', 'high', '騎手出走数不足'),
            'jockey_cond_wr': ('騎手条件別勝率', 'F', 'keep_nan_candidate', 'high', '該当条件出走実績なし'),
            'is_nar_transfer': ('地方転籍フラグ', 'A', 'keep_nan_candidate', 'high', '中央のみ出走=0'),
            'nar_recent_ratio': ('地方出走比率', 'A', 'keep_nan_candidate', 'high', '中央のみ出走=0'),
            'jockey_wr_distance': ('騎手距離別勝率', 'F', 'keep_nan_candidate', 'high', '距離別出走実績不足'),
            'jockey_wr_venue': ('騎手場別勝率', 'F', 'keep_nan_candidate', 'high', '場別出走実績不足'),
            'trainer_wr_venue': ('調教師場別勝率', 'F', 'keep_nan_candidate', 'high', '場別出走実績不足'),
        }
        if col in s05:
            doc_meaning, nan_type, handling, confidence, reason = s05[col]
            return nan_type, handling, confidence, doc_meaning, reason
        doc_meaning = f'低欠損 ({missing_rate:.1f}%)'
        nan_type = 'A'
        handling = 'keep_nan_candidate'
        confidence = 'medium'
        reasons.append('低欠損')
        return nan_type, handling, confidence, doc_meaning, '; '.join(reasons)

    # === 5-30% ===
    if missing_rate <= 30:
        s530 = {
            'zogensa': ('増減差', 'A', 'keep_nan_candidate', 'high', '初出走=zogensa 999→NaN'),
            'weight_change_abs_capped': ('体重変化絶対値(cap)', 'A', 'keep_nan_candidate', 'high', 'zogensa依存. 初出走=NaN'),
            'weight_change_zone': ('体重変化ゾーン', 'A', 'missing_flag_candidate', 'high', '初出走=NaN. NaN自体が初出走信号'),
            'weight_change_ratio': ('体重変化率', 'A', 'keep_nan_candidate', 'high', '初出走=NaN'),
            'blood_condition_wr': ('血統条件別勝率', 'F', 'keep_nan_candidate', 'high', '条件別血統出走実績なし'),
            'blood_total_wr': ('血統全体勝率', 'F', 'keep_nan_candidate', 'high', '血統出走実績なし'),
            'blood_prize_log': ('血統賞金log', 'F', 'keep_nan_candidate', 'high', '血統出走実績なし'),
            'front_pace_wr': ('逃げ/先行ペース勝率', 'F', 'keep_nan_candidate', 'high', 'ペース出走経験なし'),
            'closing_pace_wr': ('差し/追込ペース勝率', 'F', 'keep_nan_candidate', 'high', 'ペース出走経験なし'),
            'pace_corner_stability': ('コーナー順位安定性', 'A', 'keep_nan_candidate', 'high', '過去コーナーデータなし'),
            'pace_closing_power': ('末脚パワー', 'A', 'keep_nan_candidate', 'high', '過去ハロンタイムなし'),
            'pace_position_consistency': ('位置取り一貫性', 'A', 'keep_nan_candidate', 'high', '過去コーナーデータなし'),
            'course_wr': ('コース別勝率', 'F', 'keep_nan_candidate', 'high', '該当コース経験なし'),
            'course_distance_wr': ('コース距離別勝率', 'F', 'keep_nan_candidate', 'high', '該当コース距離経験なし'),
            'norm_finish_logit_avg': ('平均着順logit', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'harontimel5_avg': ('後5ハロン平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'harontimel5_zscore': ('後5ハロンzscore', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'timediff_avg': ('平均タイム差', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'jyuni1c_avg': ('平均1コーナー順位', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'jyuni4c_avg': ('平均4コーナー順位', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'closing_index_avg': ('平均クロージング指数', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'kyakusitukubun_cd': ('脚質コード', 'A', 'missing_flag_candidate', 'high', '過去出走なし→脚質判定不能. NaN=初出走信号'),
            'weight_zscore': ('体重zscore', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'days_since_last_race': ('前走からの日数', 'A', 'keep_nan_candidate', 'high', '初出走'),
            'rest_category': ('休養カテゴリ', 'A', 'keep_nan_candidate', 'high', '初出走'),
            'form_trend': ('調子トレンド', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'form_consistency': ('調子一貫性', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'class_move': ('クラス変動', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'track_condition_delta': ('馬場状態変化', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'distance_change': ('距離変化フラグ', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'surface_change': ('芝ダ変更フラグ', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'class_drop_bounce': ('クラス降級バウンス', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'class_adj_formetric': ('クラス調整フォーム', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'class_promotions': ('クラス昇級回数', 'F', 'zero_fill_candidate', 'high', '過去出走なし→0が自然'),
            'class_demotions': ('クラス降級回数', 'F', 'zero_fill_candidate', 'high', '過去出走なし→0が自然'),
            'class_net_change': ('クラス純変動', 'F', 'zero_fill_candidate', 'high', '過去出走なし→0が自然'),
            'class_max_level': ('過去最高クラス', 'F', 'keep_nan_candidate', 'high', '過去出走なし. NaN=初出走信号'),
            'class_level_std': ('クラスレベル標準偏差', 'F', 'zero_fill_candidate', 'high', '過去出走なし→0が自然'),
            'v_recovery_flag': ('V字回復フラグ', 'F', 'zero_fill_candidate', 'high', '過去出走なし→0が自然'),
            'time_improvement_rate': ('タイム改善率', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'position_improvement_rate': ('順位改善率', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'weighted_recent_form_finish': ('加重最近着順', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'weighted_recent_form_time': ('加重最近タイム', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'closing_speed_ratio_avg': ('平均上がり比率', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'closing_speed_ratio_zscore': ('上がり比率zscore', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'closing_speed_ratio_trend': ('上がり比率トレンド', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'harontime_last3f_avg': ('後3ハロン平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'harontime_last3f_zscore': ('後3ハロンzscore', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'harontime_last3f_trend': ('後3ハロントレンド', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'pace_ratio_avg': ('ペース比平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'pace_ratio_zscore': ('ペース比zscore', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'pace_ratio_trend': ('ペース比トレンド', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'pace_early_avg': ('前半ペース平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'pace_mid_avg': ('中盤ペース平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'pace_late_avg': ('後半ペース平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'haron_race_gap_avg': ('ハロンレースギャップ平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'haron_race_gap_zscore': ('ハロンレースギャップzscore', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'haron_race_gap_trend': ('ハロンレースギャップトレンド', 'A', 'keep_nan_candidate', 'high', '過去出走不足'),
            'pace_adj_finish_avg': ('ペース調整着順平均', 'A', 'keep_nan_candidate', 'high', '過去出走なし'),
            'actual_pace_fit': ('実際のペース適合度', 'F', 'keep_nan_candidate', 'high', '過去ペース経験なし'),
            'jt_combo_wr': ('騎手調教師コンビ勝率', 'F', 'keep_nan_candidate', 'high', 'コンビ実績なし'),
            'jt_combo_place_rate': ('コンビ複勝率', 'F', 'keep_nan_candidate', 'high', 'コンビ実績なし'),
            'jt_combo_starts': ('コンビ出走数', 'F', 'zero_fill_candidate', 'high', 'コンビ出走なし→0'),
            'jt_combo_prize_log': ('コンビ賞金log', 'F', 'keep_nan_candidate', 'high', 'コンビ実績なし'),
            'excluded_reason': ('除外理由', 'N', 'keep_nan_candidate', 'high', '投票対象外=NaN, 対象=理由文字列'),
            'haron_x_distance': ('ハロンx距離交差', 'A', 'keep_nan_candidate', 'high', '元特徴量NaNに連動'),
            'surface_x_past_perf': ('芝ダx過去成績交差', 'A', 'keep_nan_candidate', 'high', '元特徴量NaNに連動'),
            'grade_x_form_trend': ('グレードx調子交差', 'A', 'keep_nan_candidate', 'high', '元特徴量NaNに連動'),
            'distance_x_closing_index': ('距離xクロージング交差', 'A', 'keep_nan_candidate', 'high', '元特徴量NaNに連動'),
            'grade_x_blood_prize_log': ('グレードx血統賞金交差', 'F', 'keep_nan_candidate', 'high', '元特徴量NaNに連動'),
        }
        if col.endswith('_race_rank'):
            base = col.replace('_race_rank', '')
            return 'A', 'keep_nan_candidate', 'high', f'{base}のレース内相対順位', '元特徴量のNaNに連動'
        if col.startswith('rel_') and col not in s530:
            return 'A', 'keep_nan_candidate', 'high', f'相対特徴量 ({col})', '元特徴量のNaNに連動'
        if col in s530:
            doc_meaning, nan_type, handling, confidence, reason = s530[col]
            return nan_type, handling, confidence, doc_meaning, reason
        doc_meaning = f'軽度欠損 ({missing_rate:.1f}%)'
        return 'A', 'keep_nan_candidate', 'medium', doc_meaning, '過去出走なし等の自然欠損'

    # === 30-70% ===
    if missing_rate <= 70:
        s3070 = {
            'datakubun': ('データ区分 (SE)', 'B', 'investigate', 'medium', '48%欠損. 7=確定成績のみ? 要確認'),
            'kigocd': ('馬記号コード (SE)', 'A', 'unknown_category_candidate', 'high', '57%欠損. 多くの馬に記号なし→NaN自然'),
            'blood_surface_wr': ('血統芝ダ別勝率', 'F', 'keep_nan_candidate', 'high', '芝ダ別血統出走実績なし'),
            'blood_distance_wr': ('血統距離別勝率', 'F', 'keep_nan_candidate', 'high', '距離別血統出走実績なし'),
            'pace_aptitude': ('ペース適性スコア', 'F', 'keep_nan_candidate', 'high', '69%欠損. ペース経験不足'),
            'harontime_late_trend': ('後半ハロントレンド', 'A', 'keep_nan_candidate', 'high', '過去出走不足で回帰不可'),
            'blood_surface_wr_race_rank': ('血統芝ダ勝率順位', 'F', 'keep_nan_candidate', 'high', 'blood_surface_wr NaN連動'),
            'blood_surface_wr_x_condition': ('血統芝ダ勝率x馬場交差', 'F', 'keep_nan_candidate', 'high', 'blood_surface_wr NaN連動'),
            'win_dominance': ('単勝支配力', 'A', 'keep_nan_candidate', 'high', '71%欠損. 計算条件不足'),
        }
        if col in s3070:
            doc_meaning, nan_type, handling, confidence, reason = s3070[col]
            return nan_type, handling, confidence, doc_meaning, reason
        return 'A', 'keep_nan_candidate', 'medium', f'中度欠損 ({missing_rate:.1f}%)', '経験不足による自然欠損'

    # === 70-90% ===
    if missing_rate <= 90:
        s7090 = {
            'v_recovery_duration': ('V字回復期間', 'F', 'keep_nan_candidate', 'high', '90%欠損. 回復経験なし'),
            'dist_change_avg_pos': ('距離変更時平均着順', 'F', 'keep_nan_candidate', 'high', '87%欠損. 距離変更経験なし'),
            'dist_change_win_rate': ('距離変更時勝率', 'F', 'keep_nan_candidate', 'high', '87%欠損. 距離変更経験なし'),
            'dist_change_exp_count': ('距離変更経験回数', 'F', 'zero_fill_candidate', 'high', '87%欠損. 0埋め自然'),
            'cond_change_avg_pos': ('条件変更時平均着順', 'F', 'keep_nan_candidate', 'high', '79%欠損. 条件変更経験なし'),
            'cond_change_win_rate': ('条件変更時勝率', 'F', 'keep_nan_candidate', 'high', '79%欠損. 条件変更経験なし'),
            'cond_change_exp_count': ('条件変更経験回数', 'F', 'zero_fill_candidate', 'high', '79%欠損. 0埋め自然'),
        }
        if col in s7090:
            doc_meaning, nan_type, handling, confidence, reason = s7090[col]
            return nan_type, handling, confidence, doc_meaning, reason
        return 'A', 'keep_nan_candidate', 'medium', f'重度欠損 ({missing_rate:.1f}%)', '経験不足による自然欠損'

    # === 90%以上 (100%除く) ===
    s90p = {
        'surf_change_avg_pos': ('芝ダ変更時平均着順', 'F', 'keep_nan_candidate', 'high', '95%欠損. 芝ダ変更経験なし'),
        'surf_change_win_rate': ('芝ダ変更時勝率', 'F', 'keep_nan_candidate', 'high', '95%欠損. 芝ダ変更経験なし'),
        'surf_change_exp_count': ('芝ダ変更経験回数', 'F', 'zero_fill_candidate', 'high', '95%欠損. 0埋め自然'),
        'stake': ('掛け金', 'D', 'leakage_or_drop_candidate', 'high', '95%欠損. 投票結果(レース後). BT専用'),
        'result': ('投票結果(払戻)', 'D', 'leakage_or_drop_candidate', 'high', '95%欠損. 払戻金(レース後). BT専用'),
        'final_odds': ('最終オッズ', 'D', 'leakage_or_drop_candidate', 'high', '95%欠損. 確定オッズ(レース後). BT専用'),
    }
    if col in s90p:
        doc_meaning, nan_type, handling, confidence, reason = s90p[col]
        return nan_type, handling, confidence, doc_meaning, reason
    return 'A', 'keep_nan_candidate', 'low', f'ほぼ欠損 ({missing_rate:.1f}%)', '欠損率高すぎ. 除去検討'


# 実行
results = []
for _, row in stats.iterrows():
    col = row['column']
    dtype = row['dtype']
    miss_count = int(row['missing_count'])
    miss_rate = row['missing_rate']
    unique_count = int(row['unique_count'])
    samples = [row['sample_1'], row['sample_2'], row['sample_3']]
    samples = [str(x) for x in samples if str(x) != 'nan']
    sample_str = ', '.join(samples[:3]) if samples else '(all NaN)'

    nan_type, handling, confidence, doc_meaning, reason = classify_nan(
        col, dtype, miss_rate, samples, unique_count
    )

    results.append({
        'column': col,
        'dtype': dtype,
        'missing_count': miss_count,
        'missing_rate': miss_rate,
        'unique_count': unique_count,
        'sample_values': sample_str,
        'everyDB2_doc_meaning': doc_meaning,
        'inferred_nan_type': nan_type,
        'suggested_handling': handling,
        'reason': reason,
        'confidence': confidence,
    })

result_df = pd.DataFrame(results)
result_df.to_csv('tmp/nan_classification_full.csv', index=False, encoding='utf-8-sig')
print(f'Total: {len(result_df)} columns classified')

# NaNタイプ集計
print('\n=== NaNタイプ別集計 ===')
nt_labels = {'N': 'N: 欠損なし', 'A': 'A: 自然NaN(初出走等)', 'B': 'B: 取得ミス疑い',
             'C': 'C: 予測時点依存', 'D': 'D: リーク/レース後', 'E': 'E: カテゴリ欠損',
             'F': 'F: 経験回数/母数0', 'G': 'G: 判断保留'}
for nt, label in nt_labels.items():
    subset = result_df[result_df['inferred_nan_type'] == nt]
    if len(subset) > 0:
        print(f'  {label}: {len(subset)}列')

print('\n=== 推奨ハンドリング別集計 ===')
for h in ['keep_nan_candidate', 'missing_flag_candidate', 'zero_fill_candidate',
          'unknown_category_candidate', 'leakage_or_drop_candidate', 'investigate']:
    subset = result_df[result_df['suggested_handling'] == h]
    print(f'  {h}: {len(subset)}列')

# wide_odds 特別確認
print('\n=== wide_odds 検証 ===')
wide_result = result_df[result_df['column'].str.startswith('wide_odds_')]
print(f'  wide_odds総数: {len(wide_result)}')
print(f'  NaNタイプA: {(wide_result["inferred_nan_type"] == "A").sum()}')
print(f'  欠損率範囲: {wide_result["missing_rate"].min():.1f}% - {wide_result["missing_rate"].max():.1f}%')

# 100% NaN 削除推奨リスト
full_nan = result_df[result_df['missing_rate'] >= 99.99]
print(f'\n=== 100% NaN 削除推奨: {len(full_nan)}列 ===')
for _, r in full_nan.iterrows():
    print(f'  {r["column"]}: {r["everyDB2_doc_meaning"]}')

# リーク警告
leak = result_df[result_df['inferred_nan_type'] == 'D']
print(f'\n=== リーク/レース後情報: {len(leak)}列 ===')
for _, r in leak.iterrows():
    print(f'  {r["column"]} (miss={r["missing_rate"]:.1f}%): {r["everyDB2_doc_meaning"]}')
