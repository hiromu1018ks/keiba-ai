import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ScenarioBettingStrategy:
    """
    Simulates various betting strategies based on configuration.
    """
    def __init__(self, config):
        self.config = config
        self.bet_type = config.get('bet_type', 'win') # only 'win' supported for now
        self.method = config.get('method', 'fixed_amount')
        self.base_amount = config.get('amount', 100)
        self.threshold_ev = config.get('threshold_ev', 1.0)
        self.threshold_prob_min = config.get('threshold_prob_min', 0.0)
        self.threshold_prob_max = config.get('threshold_prob_max', 1.0)
        self.threshold_odds_min = config.get('threshold_odds_min', 1.0)
        self.threshold_odds_max = config.get('threshold_odds_max', 1000.0)

    def calculate_expectation(self, win_prob, odds):
        if odds is None:
            return 0.0
        return win_prob * odds

    def decide_bet(self, df_preds, odds_data):
        bets = []
        
        for _, row in df_preds.iterrows():
            horse_num = int(row['horse_num'])
            prob = row['pred_prob']
            
            current_odds = odds_data.get(horse_num)
            if current_odds is None:
                continue
                
            # Filters
            if not (self.threshold_prob_min <= prob <= self.threshold_prob_max):
                continue
            if not (self.threshold_odds_min <= current_odds <= self.threshold_odds_max):
                continue
                
            ev = self.calculate_expectation(prob, current_odds)
            
            if ev > self.threshold_ev:
                amount = self._calculate_amount(prob, current_odds, ev)
                
                if amount > 0:
                    bets.append({
                        'horse_num': horse_num,
                        'type': 'tansho',
                        'amount': amount,
                        'odds': current_odds,
                        'prob': prob,
                        'ev': ev
                    })
        
        return bets

    def _calculate_amount(self, prob, odds, ev):
        if self.method == 'fixed_amount':
            return self.base_amount
        elif self.method == 'ev_weighted':
            # Example: Base amount * EV (higher EV -> higher bet)
            # Scale simple: amount * (EV - 1) * 10 or something? 
            # Let's keep it simple: amount * EV
            return int(self.base_amount * ev)
        elif self.method == 'proportional':
            # Simplified proportional: not tracking total bankroll here deeply, 
            # but acting as if we bet % of 'base unit'.
            # Real Kelly Criterion would be: f* = (bp - q) / b
            # b = odds - 1
            # p = prob
            # q = 1 - p
            # f = ( (odds-1)*prob - (1-prob) ) / (odds-1)
            #   = ( prob*odds - prob - 1 + prob ) / (odds-1)
            #   = ( prob*odds - 1 ) / (odds-1)
            #   = ( EV - 1 ) / (odds-1)
            b = odds - 1
            if b <= 0: return 0
            kelly_fraction = (prob * odds - 1) / b
            
            # Use fractional Kelly (e.g. Quarter Kelly) to be safe, or just use raw kelly as requested config
            # Here we treat base_amount as "Bankroll" for the sake of calculation if method is 'proportional'
            # But usually 'amount' in config implies a bet unit.
            # Let's interpret 'amount' as Max Bet Cap or Unit?
            # User requirement was vague, let's implement a 'Kelly-like' scaled by base_amount
            # If base_amount is 10000, and kelly is 0.1, bet 1000.
            
            if kelly_fraction <= 0: return 0
            
            # Cap at 50% kelly for safety in this simple sim
            kelly_fraction = max(0, min(kelly_fraction, 0.5)) 
            
            bet_val = int(self.base_amount * kelly_fraction)
            return bet_val if bet_val >= 100 else 0 # Minimum bet 100
            
        return self.base_amount
