import pandas as pd
import os
import sys
from src.strategy.scenario_betting import ScenarioBettingStrategy
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_predictions(path='simulation_predictions.csv'):
    if not os.path.exists(path):
        logger.error(f"Prediction file not found: {path}")
        return None
    logger.info(f"Loading predictions from {path}...")
    df = pd.read_csv(path)
    # Ensure necessary columns
    required = ['race_id', 'horse_num', 'pred_prob', 'odds', 'rank', 'date']
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            # If odds is missing (might be log_odds?), we need to be careful.
            # But simulate.py saved 'odds' specifically.
            return None
    return df

def run_scenarios():
    df = load_predictions()
    if df is None:
        return

    # Define Scenarios
    scenarios = [
        {
            'name': 'Standard (EV>1.2)',
            'config': {
                'method': 'fixed_amount',
                'amount': 100,
                'threshold_ev': 1.2
            }
        },
        {
            'name': 'Conservative (EV>1.5)',
            'config': {
                'method': 'fixed_amount',
                'amount': 100,
                'threshold_ev': 1.5
            }
        },
        {
            'name': 'Confident (Prob>0.3)',
            'config': {
                'method': 'fixed_amount',
                'amount': 100,
                'threshold_ev': 1.0, # Just positive EV
                'threshold_prob_min': 0.3
            }
        },
        {
            'name': 'Longshot (Odds>20)',
            'config': {
                'method': 'fixed_amount',
                'amount': 100,
                'threshold_ev': 1.0,
                'threshold_odds_min': 20.0,
                'threshold_odds_max': 100.0
            }
        },
        {
            'name': 'Aggressive (EV Weighted)',
            'config': {
                'method': 'ev_weighted',
                'amount': 100, # Base unit
                'threshold_ev': 1.2
            }
        },
        {
            'name': 'Scientific (Kelly Criterion)',
            'config': {
                'method': 'proportional',
                'amount': 100000, # Virtual Bankroll for proportion calculation
                'threshold_ev': 1.0
            }
        },
        {
            'name': 'Ultra High EV (>3.0)',
            'config': {
                'method': 'fixed_amount',
                'amount': 100,
                'threshold_ev': 3.0
            }
        }
    ]

    results = []
    
    # Process simulations
    logger.info("Starting Scenario Simulations...")
    
    # Sort by date for consistent simulation (though we aren't doing bankroll carry-over seriously yet)
    # Sort by date for consistent simulation
    try:
        df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
    except Exception as e:
        logger.warning(f"Date parsing with Japanese format failed, trying default: {e}")
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
        
    df = df.sort_values(['date_dt', 'race_id'])
    
    for scenario in scenarios:
        name = scenario['name']
        config = scenario['config']
        strategy = ScenarioBettingStrategy(config)
        
        logger.info(f"Running Scenario: {name}")
        
        total_return = 0
        total_bet = 0
        hit_count = 0
        bet_count = 0
        
        # Group by race to simulate betting per race
        for race_id, race_data in df.groupby('race_id'):
            odds_map = dict(zip(race_data['horse_num'], race_data['odds']))
            
            # Predict & Bet
            bets = strategy.decide_bet(race_data, odds_map)
            
            for bet in bets:
                amount = bet['amount']
                horse_num = bet['horse_num']
                
                total_bet += amount
                bet_count += 1
                
                # Check Result
                result_row = race_data[race_data['horse_num'] == horse_num].iloc[0]
                if result_row['rank'] == 1:
                    hit_count += 1
                    total_return += amount * result_row['odds']

        recovery = (total_return / total_bet * 100) if total_bet > 0 else 0
        hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0
        profit = total_return - total_bet
        
        results.append({
            'Scenario': name,
            'Total Bet': total_bet,
            'Total Return': int(total_return),
            'Profit': int(profit),
            'Recovery (%)': round(recovery, 1),
            'Hit Rate (%)': round(hit_rate, 2),
            'Bets Placed': bet_count
        })

    # Output Results
    results_df = pd.DataFrame(results)

    # Save to file FIRST to ensure data is safe
    output_csv = 'simulation_scenarios_result.csv'
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Saved results to {output_csv}")

    # Output Results
    print("\n" + "="*50)
    print(" SCENARIO SIMULATION RESULTS")
    print("="*50 + "\n")
    try:
        print(results_df.to_markdown(index=False))
    except ImportError:
        print(results_df.to_string(index=False))
    except Exception as e:
        print(f"Error printing table: {e}")
        print(results_df)

if __name__ == "__main__":
    run_scenarios()
