import pandas as pd
import os
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

import dataloading
from trading_env import TradingEnv

# Opret mapper
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_and_align_data():
    print("--- Indlæser Data ---")
    df_raw_train, df_raw_val, _ = dataloading.get_novo_nordisk_split()
    df_features_train = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
    
    # Alignment
    df_raw_aligned = df_raw_train.loc[df_features_train.index]
    return df_features_train, df_raw_aligned

# Wrapper function for multiprocessing
def make_env(rank, df_features, df_raw, spread=0.0002):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        return env
    return _init

def train():
    df_features, df_raw = load_and_align_data()
    
    # 1. PARALLEL ENVIRONMENTS (Faster Data Collection)
    # i7-8700 has 12 threads. We use 8 to leave room for the OS/GPU drivers.
    num_cpu = 8 
    
    print(f"--- Creating {num_cpu} Parallel Environments ---")
    # SubprocVecEnv creates separate processes for each env
    env = SubprocVecEnv([make_env(i, df_features, df_raw) for i in range(num_cpu)])
    env = VecMonitor(env) # Helper to track rewards across processes

    # 2. OPTIMIZED PPO SETTINGS FOR GPU
    # Larger batch size = better GPU usage
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", # Force GPU
        tensorboard_log="./logs/",
        clip_range=0.1,
        learning_rate=1.4033409789977257e-05,
        n_steps=2048,
        batch_size=2048,
        ent_coef=7.664396737253413e-06,
        gamma=0.9814531835264428,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256,256]), activation_fn=nn.Tanh)
    )
    
    print("--- Starting High-Speed Training ---")
    model.learn(total_timesteps=5_000_000, progress_bar=True)
    
    model.save("models/ppo_novo_agent_optimized")

if __name__ == "__main__":
    # Windows requires this protection for multiprocessing
    train()