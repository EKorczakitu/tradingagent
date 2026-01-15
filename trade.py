import pandas as pd
import os
import torch.nn as nn
import numpy as np
import ast

# --- IMPORTS ----
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

import dataloading
from trading_env import TradingEnv

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- NEW: Monitoring Callback ---
class HedgeFundCallback(BaseCallback):
    """
    Custom callback to print 'Real Trading Metrics' (Sharpe/Sortino) 
    to the console during training.
    """
    def __init__(self, check_freq: int, verbose=1):
        super(HedgeFundCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.returns_buffer = []

    def _on_step(self) -> bool:
        # Collect 'net_return' from all parallel environments
        # 'infos' is a list of dictionaries from the vectorized env
        for info in self.locals['infos']:
            if 'net_return' in info:
                self.returns_buffer.append(info['net_return'])

        if self.n_calls % self.check_freq == 0 and len(self.returns_buffer) > 0:
            # Calculate metrics on the recent history
            returns = np.array(self.returns_buffer)
            
            # Annualized Sharpe (assuming hourly data -> 252 days * 8 hours = ~2000 hours)
            # Adjust '2000' if you trade 24/7 crypto (use 8760)
            ANNUAL_FACTOR = np.sqrt(2000) 
            
            mean_ret = np.mean(returns)
            std_dev = np.std(returns)
            
            if std_dev > 0.000001:
                sharpe = (mean_ret / std_dev) * ANNUAL_FACTOR
            else:
                sharpe = 0.0
            
            # Sortino (Downside Risk only)
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 0.000001
            sortino = (mean_ret / downside_std) * ANNUAL_FACTOR

            print(f"\n--- HEDGE FUND REPORT (Step {self.num_timesteps}) ---")
            print(f"Est. Sharpe Ratio:  {sharpe:.2f}")
            print(f"Est. Sortino Ratio: {sortino:.2f}")
            print(f"Mean Step Return:   {mean_ret*100:.4f}%")
            print("---------------------------------------------")
            
            # Reset buffer for next batch
            self.returns_buffer = []
            
        return True

def load_and_slice_data():
    print("--- Indlæser Data (Walk-Forward Setup) ---")
    
    # --- CHANGE 2: Use the robust 'get_full_dataset' logic ---
    df_full = dataloading.get_full_dataset()
    
    # Define the Training Cutoff (Same as main.py)
    VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
    
    # Slice only the Training data (Before 2024)
    df_raw_train = df_full[df_full.index < VAL_START].copy()
    
    # Load features (ensure they align)
    try:
        df_features_train = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("CRITICAL ERROR: 'data/processed_train.csv' not found. Run main.py first!")
        exit()
        
    # Align timestamps exactly
    # We take the intersection of indices to be safe
    common_index = df_raw_train.index.intersection(df_features_train.index)
    
    df_raw_aligned = df_raw_train.loc[common_index]
    df_features_aligned = df_features_train.loc[common_index]
    
    print(f"Training Data: {len(df_features_aligned)} candles ({df_features_aligned.index[0]} -> {df_features_aligned.index[-1]})")
    
    return df_features_aligned, df_raw_aligned


def get_optimal_params():
    """Loads parameters from best_hyperparams.txt or uses defaults"""
    default_params = {
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 128,
        'ent_coef': 0.001,
        'gamma': 0.99,
        'lstm_hidden': 256 # Default LSTM size
    }
    
    if os.path.exists("best_hyperparams.txt"):
        print("--- Loading Optimized Hyperparameters from tune.py ---")
        try:
            with open("best_hyperparams.txt", "r") as f:
                content = f.read()
                best_params = ast.literal_eval(content) # Safely parse string to dict
                
            # Merge with defaults (overwrites defaults with loaded values)
            default_params.update(best_params)
            
            print(f"Loaded Params: {default_params}")
        except Exception as e:
            print(f"Error loading hyperparams: {e}. Using defaults.")
    else:
        print("--- No optimization file found. Using Default Parameters. ---")
        
    return default_params

def make_env(rank, df_features, df_raw, spread=0.0002):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        return env
    return _init

def train():
    df_features, df_raw = load_and_slice_data()
    
    # 1. PARALLEL ENVIRONMENTS
    num_cpu = 8 
    print(f"--- Creating {num_cpu} Parallel Environments ---")
    env = SubprocVecEnv([make_env(i, df_features, df_raw) for i in range(num_cpu)])
    env = VecMonitor(env) 

    # 2. LOAD PARAMS
    params = get_optimal_params()

    # 3. MODEL SETUP
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1, 
        device="cuda", 
        tensorboard_log="./logs/",
        
        # Inject the loaded parameters
        learning_rate=params['learning_rate'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        ent_coef=params['ent_coef'],
        gamma=params['gamma'],
        
        policy_kwargs=dict(
            enable_critic_lstm=True, 
            lstm_hidden_size=params.get('lstm_hidden', 256), # Handle the specific LSTM param
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.Tanh
        )
    )
    
    print("--- Starting Institutional Training ---")
    hedge_fund_monitor = HedgeFundCallback(check_freq=10000)
    
    # 2 Million Steps is a good baseline
    model.learn(total_timesteps=2_000_000, callback=hedge_fund_monitor, progress_bar=True)
    
    print("--- Saving Model ---")
    model.save("models/ppo_novo_agent_optimized")
    print("Model saved to models/ppo_novo_agent_optimized.zip")

if __name__ == "__main__":
    train()