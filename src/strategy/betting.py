import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BettingStrategy:
    def __init__(self, ev_threshold=1.5):
        self.ev_threshold = ev_threshold

    def calculate_expectation(self, win_prob, odds):
        """
        Calculates Expected Value (EV).
        EV = (Win Probability * Odds) - 1 (cost is included in odds, so just Prob * Odds is return ratio)
        Actually, let's define EV as Return Ratio: Prob * Odds.
        If > 1.0, it's profitable in theory.
        """
        if odds is None:
            return 0.0
        return win_prob * odds

    def decide_bet(self, df_preds, odds_data):
        """
        Decides which horses to bet on.
        
        Args:
            df_preds (pd.DataFrame): DataFrame with 'horse_num' and 'pred_prob' (predicted win probability).
            odds_data (dict): Dictionary mapping horse_num to current odds.
            
        Returns:
            list: List of dictionaries [{'horse_num': 1, 'type': 'tansho', 'amount': 100}]
        """
        bets = []
        
        for _, row in df_preds.iterrows():
            horse_num = int(row['horse_num'])
            prob = row['pred_prob']
            
            current_odds = odds_data.get(horse_num)
            if current_odds is None:
                continue
                
            ev = self.calculate_expectation(prob, current_odds)
            
            if ev > self.ev_threshold:
                logger.info(f"Bet Signal: Horse {horse_num}, Prob {prob:.4f}, Odds {current_odds}, EV {ev:.4f}")
                bets.append({
                    'horse_num': horse_num,
                    'type': 'tansho', # Currently only supporting Single Win
                    'amount': 100 # Fixed amount for now
                })
        
        return bets
