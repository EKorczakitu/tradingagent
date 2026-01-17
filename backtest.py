import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
import dataloading
from trading_env import TradingEnv
from collections import Counter

def calculate_period_stats(df_log, start_date, end_date):
    mask = (df_log.index >= start_date) & (df_log.index <= end_date)
    period_data = df_log.loc[mask]
    if period_data.empty: return None

    start_bal = period_data.iloc[0]['Balance']
    end_bal = period_data.iloc[-1]['Balance']
    agent_ret = ((end_bal - start_bal) / start_bal) * 100
    
    start_pr = period_data.iloc[0]['Price']
    end_pr = period_data.iloc[-1]['Price']
    mkt_ret = ((end_pr - start_pr) / start_pr) * 100
    
    return {'Agent': agent_ret, 'Market': mkt_ret, 'Diff': agent_ret - mkt_ret}

def run_discrete_ensemble_backtest():
    print("--- Starter Discrete Ensemble Backtest (Voting - 5 Modeller) ---")

    # 1. Data Setup
    df_full = dataloading.get_full_dataset()
    VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
    df_raw = df_full[(df_full.index >= VAL_START) & (df_full.index < TEST_START)].copy()
    
    try:
        df_feat = pd.read_csv('data/processed_val.csv', index_col=0, parse_dates=True)
    except:
        print("Fejl: Kunne ikke finde processed_val.csv")
        return

    common_index = df_raw.index.intersection(df_feat.index)
    df_raw = df_raw.loc[common_index]
    df_feat = df_feat.loc[common_index]
    
    # 2. Indlæs Modeller (Nu 5 stk)
    models = []
    n_models = 5
    print(f"Indlæser {n_models} modeller...")
    
    for i in range(n_models):
        path = f"models/ppo_discrete_ensemble_{i}"
        try:
            # Vi loader på CPU for ikke at løbe tør for VRAM under backtest
            models.append(RecurrentPPO.load(path, device='cpu'))
            print(f" -> Loaded {path}")
        except Exception as e:
            print(f"FEJL: Kunne ikke loade {path}. Har du trænet alle 5? ({e})")
            return

    # 3. Miljø
    env = TradingEnv(df_feat, df_raw, spread=0.0002)
    obs, _ = env.reset()
    
    # Hver agent har sin egen hjerne (LSTM states)
    lstm_states = [None] * n_models
    episode_starts = np.ones((1,), dtype=bool)
    
    done = False
    history = []
    
    print("Kører Voting...")
    while not done:
        votes = []
        # Hent bud fra alle 5 modeller
        for i, model in enumerate(models):
            action, new_state = model.predict(obs, state=lstm_states[i], episode_start=episode_starts, deterministic=True)
            lstm_states[i] = new_state
            votes.append(int(action.item()))
            
        # --- NY VOTING LOGIK (Tilpasset 5 modeller) ---
        c = Counter(votes)
        most_common = c.most_common() # Returnerer liste af tuples, f.eks. [(1, 3), (0, 2)]
        
        winner_action = most_common[0][0]
        
        # Tie-breaker / Safety First:
        # Hvis der er flere end 1 valgmulighed, og de to øverste har lige mange stemmer
        # (F.eks. 2 stemmer på 'Long' og 2 stemmer på 'Short') -> Så vælg 'Hold' (0)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            winner_action = 0 
            
        obs, reward, terminated, truncated, info = env.step(winner_action)
        
        episode_starts = terminated or truncated
        current_time = df_raw.index[env.current_step - 1]
        
        history.append({
            'Date': current_time,
            'Balance': info['balance'],
            'Price': info['price'],
            'Position': info['position'],
            # Hvor enige er de? (Antal stemmer på vinderen / total antal modeller)
            'Vote_Agreement': most_common[0][1] / n_models 
        })
        
        if terminated or truncated:
            done = True

    # 5. Resultater & Plotting
    df_res = pd.DataFrame(history)
    df_res.set_index('Date', inplace=True)
    
    print("\n" + "="*65)
    print(f"{'PERIODE':<15} | {'AGENT %':<10} | {'MARKED %':<10} | {'ALPHA':<10}")
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

    # Plot
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        agent_norm = (df_res['Balance'] / df_res['Balance'].iloc[0]) * 100
        market_norm = (df_res['Price'] / df_res['Price'].iloc[0]) * 100
        
        ax1.plot(agent_norm, label='Ensemble Agent (Voting)', color='blue')
        ax1.plot(market_norm, label='Market', color='gray', linestyle='--')
        ax1.set_title('Ensemble Performance (5 Models)')
        ax1.legend()
        ax1.grid(True)
        
        alpha = agent_norm - market_norm
        ax2.fill_between(alpha.index, alpha, 0, where=(alpha>=0), color='green', alpha=0.3)
        ax2.fill_between(alpha.index, alpha, 0, where=(alpha<0), color='red', alpha=0.3)
        ax2.plot(alpha, color='black', lw=1)
        ax2.set_title('Alpha (Agent - Market)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_ensemble.png')
        print("\nGemt: backtest_ensemble.png")
        plt.show()
        
    except Exception as e:
        print(f"Plot fejl: {e}")

if __name__ == "__main__":
    run_discrete_ensemble_backtest()