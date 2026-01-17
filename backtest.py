import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
import dataloading
from trading_env import TradingEnv

def calculate_period_stats(df_log, start_date, end_date, initial_capital=10000):
    """Hjælpefunktion til at beregne stats for en specifik periode"""
    mask = (df_log.index >= start_date) & (df_log.index <= end_date)
    period_data = df_log.loc[mask]
    
    if period_data.empty:
        return None

    start_balance = period_data.iloc[0]['Balance']
    end_balance = period_data.iloc[-1]['Balance']
    agent_return = ((end_balance - start_balance) / start_balance) * 100
    
    start_price = period_data.iloc[0]['Price']
    end_price = period_data.iloc[-1]['Price']
    market_return = ((end_price - start_price) / start_price) * 100
    
    return {
        'Agent': agent_return,
        'Market': market_return,
        'Diff': agent_return - market_return
    }

def run_backtest():
    print("--- Starter Dybdegående Backtest (2024) ---")

    # 1. Data Setup (RETTET)
    # Vi bruger nu den nye 'get_full_dataset' funktion
    df_full = dataloading.get_full_dataset()
    
    # Definer Valideringsperioden (samme som i main.py)
    VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
    
    # Snit rådata til 2024 (Validering)
    df_raw_val = df_full[(df_full.index >= VAL_START) & (df_full.index < TEST_START)].copy()
    
    # Indlæs de processerede features (skal være genereret af main.py først)
    try:
        df_features_val = pd.read_csv('data/processed_val.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("FEJL: Kunne ikke finde 'data/processed_val.csv'. Kør main.py først!")
        return

    # Sørg for at indeks matcher præcist (Intersection)
    common_index = df_raw_val.index.intersection(df_features_val.index)
    df_raw_aligned = df_raw_val.loc[common_index]
    df_features_aligned = df_features_val.loc[common_index]
    
    print(f"Backtest Data: {len(df_features_aligned)} candles")

    # 2. Miljø & Model
    # Vi bruger de alignede data
    env = TradingEnv(df_features_aligned, df_raw_aligned, spread=0.0002)
    model_path = "models/ppo_novo_agent_optimized"
    
    try:
        print(f"Indlæser model fra: {model_path}")
        model = RecurrentPPO.load(model_path)
    except Exception as e:
        print(f"Kunne ikke finde model: {e}")
        print("Husk at køre 'trade.py' først for at træne modellen!")
        return

    # 3. Kør Simulering
    obs, _ = env.reset()
    
    # LSTM States håndtering
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    
    done = False
    history = []
    
    print("Genererer handelsdata...")
    while not done:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_starts = terminated or truncated
        
        # Gem timestamp og data
        # Vi bruger env.current_step - 1, fordi step lige er blevet talt op
        current_time = df_raw_aligned.index[env.current_step - 1] 
        
        history.append({
            'Date': current_time,
            'Balance': info['balance'],
            'Price': info['price'],
            'Position': info['position']
        })
        
        if terminated or truncated:
            done = True

    # Konverter til DataFrame
    df_res = pd.DataFrame(history)
    df_res.set_index('Date', inplace=True)
    
    # 4. KVARTALSVIS ANALYSE
    print("\n" + "="*65)
    print(f"{'PERIODE':<15} | {'AGENT %':<10} | {'MARKED %':<10} | {'ALPHA (Diff)':<10}")
    print("-" * 65)
    
    quarters = [
        ('Q1 2024', '2024-01-01', '2024-03-31'),
        ('Q2 2024', '2024-04-01', '2024-06-30'),
        ('Q3 2024', '2024-07-01', '2024-09-30'),
        ('Q4 2024', '2024-10-01', '2024-12-31'),
        ('TOTAL 2024', '2024-01-01', '2024-12-31')
    ]
    
    for name, start, end in quarters:
        stats = calculate_period_stats(df_res, start, end)
        if stats:
            print(f"{name:<15} | {stats['Agent']:>9.2f}% | {stats['Market']:>9.2f}% | {stats['Diff']:>9.2f}%")
    print("="*65)

    # 5. Plotting
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Øverste graf: Værdier (Rebased til 100)
        agent_norm = (df_res['Balance'] / df_res['Balance'].iloc[0]) * 100
        market_norm = (df_res['Price'] / df_res['Price'].iloc[0]) * 100
        
        ax1.plot(agent_norm, label='AI Agent (Rebased)', color='blue')
        ax1.plot(market_norm, label='Buy & Hold (Rebased)', color='gray', linestyle='--')
        ax1.set_title('Performance Sammenligning (2024)')
        ax1.set_ylabel('Værdi (Indeks 100)')
        ax1.legend()
        ax1.grid(True)
        
        # Nederste graf: Alpha
        alpha_curve = agent_norm - market_norm
        
        ax2.fill_between(alpha_curve.index, alpha_curve, 0, where=(alpha_curve >= 0), color='green', alpha=0.3, label='Outperformance')
        ax2.fill_between(alpha_curve.index, alpha_curve, 0, where=(alpha_curve < 0), color='red', alpha=0.3, label='Underperformance')
        
        ax2.plot(alpha_curve, color='black', linewidth=1)
        ax2.set_title('Relative Performance (Alpha)')
        ax2.set_ylabel('Agent minus Market (%)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('backtest_detailed_2024.png')
        print("\nDetaljeret graf gemt som 'backtest_detailed_2024.png'")
        plt.show()
        
    except Exception as e:
        print(f"Kunne ikke plotte: {e}")

if __name__ == "__main__":
    run_backtest()