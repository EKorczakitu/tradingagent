import optuna
from optuna.trial import TrialState
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging

# Mute Optuna's standard logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import local modules
import dataloading
from trading_env import TradingEnv

def optimize_agent(trial):
    # 1. Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.01, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64,128,256])
    
    # Network Architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_type == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_type == "large":
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    # 2. Environment (Train)
    # Note: We assume TradingEnv defaults to reward_scale=100.0 here, which is good for training
    env_train = DummyVecEnv([lambda: Monitor(TradingEnv(DF_FEATURES_TRAIN, DF_RAW_TRAIN_ALIGNED))])
    
    # 3. Model
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=0,
        device="cpu",
        clip_range=0.1,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        policy_kwargs=dict(net_arch=net_arch, activation_fn=nn.Tanh)
    )
    
    # 4. Train (Short run for evaluation)
    try:
        model.learn(total_timesteps=250_000)
    except Exception as e:
        print(f"Training crashed: {e}")
        return -1000 # Penalty for crashing

    # 5. Evaluate on VALIDATION
    # We use a fresh environment for validation
    env_val = TradingEnv(DF_FEATURES_VAL, DF_RAW_VAL_ALIGNED, spread=0.0002)
    obs, _ = env_val.reset()
    done = False
    
    # FIXED: List to store actual financial returns (not RL rewards)
    val_returns = [] 
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_val.step(action)
        
        # FIXED: Extract the actual 'net_return' from info for correct Sharpe calculation
        # This ignores the reward scaling/shaping and looks at raw PnL
        actual_return = info.get('net_return', 0)
        val_returns.append(actual_return)
        
        if terminated or truncated:
            done = True
            
    # FIXED: Calculate Sharpe Ratio on the list we just filled
    if not val_returns:
        return -1000

    returns_array = np.array(val_returns)
    std_dev = np.std(returns_array)
    mean_return = np.mean(returns_array)
    
    # Avoid division by zero if flat-line
    if std_dev == 0:
        return -1000
    
    # Annualized Sharpe (assuming hourly data: sqrt(252 trading days * 8 hours) ~ sqrt(2000))
    # But for raw comparison, simple mean/std is fine.
    sharpe = mean_return / std_dev
    
    return sharpe

if __name__ == "__main__":
    print("--- Starter Optuna Tuning ---")
    
    print("Indlæser data...")
    df_raw_train, df_raw_val, _ = dataloading.get_novo_nordisk_split()
    
    try:
        DF_FEATURES_TRAIN = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
        DF_FEATURES_VAL = pd.read_csv('data/processed_val.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Fejl: Kør main.py først.")
        exit()

    # Align Data
    DF_RAW_TRAIN_ALIGNED = df_raw_train.loc[DF_FEATURES_TRAIN.index]
    DF_RAW_VAL_ALIGNED = df_raw_val.loc[DF_FEATURES_VAL.index]
    
    print("Data klar. Starter optimering...")

    study = optuna.create_study(direction="maximize")
    
    N_TRIALS = 25
    with tqdm(total=N_TRIALS, desc="Optimizing Hyperparams") as pbar:
        for _ in range(N_TRIALS):
            study.optimize(optimize_agent, n_trials=1)
            pbar.update(1)
            pbar.set_postfix({"Best Sharpe": f"{study.best_value:.4f}"})
    
    print("\n" + "="*40)
    print("BEDSTE RESULTAT (SHARPE):")
    print(f"Value: {study.best_value}")
    print("Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)
    
    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))