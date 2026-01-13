import optuna
from optuna.trial import TrialState
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm  # Progress bar
import logging

# Sluk for Optunas egen støjende logning, så vi kan se vores tqdm bar
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Importér vores moduler
import dataloading
from trading_env import TradingEnv

def optimize_agent(trial):
    # 1. Hyperparametre
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    # Netværksarkitektur
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_type == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_type == "large":
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    # 2. Miljøet (Train)
    env_train = DummyVecEnv([lambda: Monitor(TradingEnv(DF_FEATURES_TRAIN, DF_RAW_TRAIN_ALIGNED))])
    
    # 3. Modellen
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=0,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=0.99,
        policy_kwargs=dict(net_arch=net_arch, activation_fn=nn.Tanh)
    )
    
    # 4. Træning (Hurtig evaluering: 30.000 steps er nok til at se tendensen)
    # Vi bruger ikke Pruning callback her for simpelhedens skyld i koden, 
    # men vi straffer dårlige modeller hårdt.
    try:
        model.learn(total_timesteps=30000)
    except Exception as e:
        return -1000 # Crash straf

    # 5. Evaluer på VALIDATION (Hele 2024)
    env_val = TradingEnv(DF_FEATURES_VAL, DF_RAW_VAL_ALIGNED, spread=0.0002)
    obs, _ = env_val.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_val.step(action)
        total_reward += reward
        if terminated or truncated:
            done = True
            
    return total_reward

if __name__ == "__main__":
    print("--- Starter Optuna Tuning ---")
    
    # Indlæs data globalt
    print("Indlæser data...")
    df_raw_train, df_raw_val, _ = dataloading.get_novo_nordisk_split()
    
    try:
        DF_FEATURES_TRAIN = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
        DF_FEATURES_VAL = pd.read_csv('data/processed_val.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Fejl: Kør main.py først.")
        exit()

    # Align
    DF_RAW_TRAIN_ALIGNED = df_raw_train.loc[DF_FEATURES_TRAIN.index]
    DF_RAW_VAL_ALIGNED = df_raw_val.loc[DF_FEATURES_VAL.index]
    
    print("Data klar. Starter optimering...")

    # Opret studie
    study = optuna.create_study(direction="maximize")
    
    # Kør trials med TQDM progress bar
    N_TRIALS = 30
    with tqdm(total=N_TRIALS, desc="Optimizing Hyperparams") as pbar:
        for _ in range(N_TRIALS):
            study.optimize(optimize_agent, n_trials=1)
            pbar.update(1)
            # Opdater pbar med bedste resultat indtil videre
            pbar.set_postfix({"Best Reward": f"{study.best_value:.2f}"})
    
    print("\n" + "="*40)
    print("BEDSTE RESULTAT:")
    print(f"Value: {study.best_value}")
    print("Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)
    
    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))