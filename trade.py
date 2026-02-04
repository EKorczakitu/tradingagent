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

# --- Monitoring Callback (Uændret) ---
class HedgeFundCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(HedgeFundCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.returns_buffer = []

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'net_return' in info:
                self.returns_buffer.append(info['net_return'])

        if self.n_calls % self.check_freq == 0 and len(self.returns_buffer) > 0:
            returns = np.array(self.returns_buffer)
            ANNUAL_FACTOR = np.sqrt(2000) 
            mean_ret = np.mean(returns)
            std_dev = np.std(returns)
            sharpe = (mean_ret / std_dev) * ANNUAL_FACTOR if std_dev > 1e-6 else 0.0
            
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-6
            sortino = (mean_ret / downside_std) * ANNUAL_FACTOR

            print(f"\n--- HEDGE FUND REPORT (Step {self.num_timesteps}) ---")
            print(f"Est. Sharpe Ratio:  {sharpe:.2f}")
            print(f"Est. Sortino Ratio: {sortino:.2f}")
            print(f"Mean Step Return:   {mean_ret*100:.4f}%")
            print("---------------------------------------------")
            self.returns_buffer = []
        return True

def load_and_slice_data():
    print("--- Indlæser Data (Walk-Forward Setup) ---")
    df_full = dataloading.get_full_dataset()
    VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
    df_raw_train = df_full[df_full.index < VAL_START].copy()
    
    try:
        df_features_train = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("CRITICAL ERROR: 'data/processed_train.csv' not found. Run main.py first!")
        exit()
        
    common_index = df_raw_train.index.intersection(df_features_train.index)
    df_raw_aligned = df_raw_train.loc[common_index]
    df_features_aligned = df_features_train.loc[common_index]
    
    return df_features_aligned, df_raw_aligned

def get_optimal_params():
    default_params = {
        'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 128,
        'ent_coef': 0.001, 'gamma': 0.99, 'lstm_hidden': 256
    }
    if os.path.exists("best_hyperparams.txt"):
        try:
            with open("best_hyperparams.txt", "r") as f:
                best_params = ast.literal_eval(f.read())
            default_params.update(best_params)
            print(f"Loaded Params: {default_params}")
        except Exception:
            pass
    return default_params

def make_env(rank, df_features, df_raw, spread=0.0002):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        return env
    return _init

# ... (Imports og helper functions som før) ...

def train_ensemble(n_models=50):
    print(f"--- Starter Discrete Ensemble Træning ({n_models} Agenter) ---")
    
    df_features, df_raw = load_and_slice_data()
    params = get_optimal_params()

    for i in range(n_models):
        seed = 42 + i 
        print(f"\nTRÆNER AGENT {i+1}/{n_models} (Seed {seed})")
        set_random_seed(seed)
        
        env = SubprocVecEnv([make_env(k, df_features, df_raw) for k in range(8)])
        env = VecMonitor(env)
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            verbose=1, 
            device="cuda", 
            tensorboard_log=f"./logs/ensemble_discrete_{seed}/",
            
            # Dine GODE parametre:
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            ent_coef=params['ent_coef'],
            gamma=params['gamma'],
            
            # VIGTIGT: Ingen gSDE her!
            policy_kwargs=dict(
                enable_critic_lstm=True, 
                lstm_hidden_size=params.get('lstm_hidden', 256),
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                activation_fn=nn.Tanh
            )
        )
        
        # 2 mio steps gav gode resultater sidst
        model.learn(total_timesteps=2_000_000, progress_bar=True)
        model.save(f"models/ppo_discrete_ensemble_{i}")
        env.close()

if __name__ == "__main__":
    train_ensemble(n_models=50)