import numpy as np
import pandas as pd
from scipy.optimize import linprog
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PortfolioOptimizer:
    def __init__(self, budget=10000, risk_aversion=0.5):
        self.budget = budget
        self.risk_aversion = risk_aversion # Not fully used in simple linear optimization but good for future

    def harville_formula(self, win_probs, bet_type='uren'):
        """
        Estimates probabilities for complex bets using Harville's Formula.
        
        Args:
            win_probs (dict): Dictionary mapping horse_num to win probability.
            bet_type (str): 'uren' (Quinella), 'utan' (Exacta), 'sanfuku' (Trio), 'santan' (Trifecta).
            
        Returns:
            dict: Mapping of combination tuple to probability.
        """
        horses = list(win_probs.keys())
        n = len(horses)
        probs = {}
        
        if bet_type == 'utan': # Exacta: 1st -> 2nd
            for h1 in horses:
                p1 = win_probs[h1]
                for h2 in horses:
                    if h1 == h2: continue
                    p2_given_not_1 = win_probs[h2] / (1.0 - p1 + 1e-9)
                    probs[(h1, h2)] = p1 * p2_given_not_1
                    
        elif bet_type == 'uren': # Quinella: 1st-2nd (order doesn't matter)
            # Uren is sum of Exacta(A, B) and Exacta(B, A)
            exacta_probs = self.harville_formula(win_probs, 'utan')
            for (h1, h2), p in exacta_probs.items():
                combo = tuple(sorted((h1, h2)))
                probs[combo] = probs.get(combo, 0) + p
                
        elif bet_type == 'santan': # Trifecta: 1st -> 2nd -> 3rd
            for h1 in horses:
                p1 = win_probs[h1]
                for h2 in horses:
                    if h1 == h2: continue
                    p2_given_not_1 = win_probs[h2] / (1.0 - p1 + 1e-9)
                    for h3 in horses:
                        if h3 in (h1, h2): continue
                        p3_given_not_1_2 = win_probs[h3] / (1.0 - p1 - win_probs[h2] + 1e-9)
                        probs[(h1, h2, h3)] = p1 * p2_given_not_1 * p3_given_not_1_2

        elif bet_type == 'sanfuku': # Trio: 1st-2nd-3rd (order doesn't matter)
            trifecta_probs = self.harville_formula(win_probs, 'santan')
            for (h1, h2, h3), p in trifecta_probs.items():
                combo = tuple(sorted((h1, h2, h3)))
                probs[combo] = probs.get(combo, 0) + p
                
        return probs

    def optimize_bets(self, candidates):
        """
        Optimizes bet allocation using Linear Programming.
        
        Args:
            candidates (list): List of dicts with keys:
                'type': bet type
                'combo': combination tuple
                'prob': estimated probability
                'odds': odds (estimated or actual)
                'ev': expected value (prob * odds)
                
        Returns:
            list: List of bets with 'amount' populated.
        """
        # Filter candidates with EV > 1.0 (or threshold)
        viable_candidates = [c for c in candidates if c['ev'] > 1.5]
        
        if not viable_candidates:
            return []
            
        n_bets = len(viable_candidates)
        
        # Objective: Maximize Total Expected Return
        # scipy.optimize.linprog minimizes, so we use negative EV as coefficients
        # We want to maximize sum(amount_i * (EV_i - 1)) -> Net Profit
        # Or just maximize sum(amount_i * EV_i) -> Total Return
        # Let's maximize Net Profit: sum(amount_i * (prob_i * odds_i - 1))
        
        # Coefficients for minimization (negative net expected profit per unit)
        c = [-1 * (item['ev'] - 1) for item in viable_candidates]
        
        # Constraints
        # 1. Total Budget: sum(amount_i) <= Budget
        A_ub = [[1] * n_bets]
        b_ub = [self.budget]
        
        # 2. Individual Bounds: 0 <= amount_i <= Kelly_Limit
        # Kelly Criterion: f* = (bp - q) / b
        # b = odds - 1
        # p = prob
        # q = 1 - p
        bounds = []
        for item in viable_candidates:
            b = item['odds'] - 1
            p = item['prob']
            q = 1 - p
            if b <= 0:
                kelly_f = 0
            else:
                kelly_f = (b * p - q) / b
            
            # Fractional Kelly (Quarter Kelly) for safety
            kelly_f = max(0, kelly_f * 0.25)
            
            max_bet = self.budget * kelly_f
            bounds.append((0, max_bet))
            
        # Solve
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            logger.warning(f"Optimization failed: {res.message}")
            return []
            
        # Extract results
        optimized_bets = []
        for i, amount in enumerate(res.x):
            # Round to nearest 100 yen (JRA unit)
            amount_100 = int(round(amount / 100) * 100)
            if amount_100 > 0:
                bet = viable_candidates[i].copy()
                bet['amount'] = amount_100
                optimized_bets.append(bet)
                
        return optimized_bets
