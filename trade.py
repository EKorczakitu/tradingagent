import pandas as pd
import os
import torch.nn as nn
import numpy as np
import ast
import multiprocessing
import shutil

# --- IMPORTS ----
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

# Import schedule helper
from typing import Callable

import dataloading
from trading_env import TradingEnv

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- Linear Schedule Helper ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

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
            # Justerer Sharpe faktor til time-level (hvis data er 1H, er faktor sqrt(24*252) ~ sqrt(6000))
            # Hvis du bruger data med crypto 24/7 er det ca sqrt(365*24) = 93.
            # Vi holder din faktor for konsistens, men overvej at justere den.
            ANNUAL_FACTOR = np.sqrt(2000) 
            mean_ret = np.mean(returns)
            std_dev = np.std(returns)
            sharpe = (mean_ret / std_dev) * ANNUAL_FACTOR if std_dev > 1e-6 else 0.0
            
            print(f"\n--- HEDGE FUND REPORT (Step {self.num_timesteps}) ---")
            print(f"Est. Sharpe Ratio:  {sharpe:.2f}")
            print(f"Mean Step Return:   {mean_ret*100:.4f}%")
            print("---------------------------------------------")
            self.returns_buffer = []
        return True

def load_and_slice_data():
    print("--- Indlæser Data (Walk-Forward Setup) ---")
    df_full = dataloading.get_full_dataset()
    VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
    
    # Train og Val (til eval callback)
    df_raw_train = df_full[df_full.index < VAL_START].copy()
    
    # Vi har brug for et valideringssæt til vores EvalCallback, så agenten måles på nye data
    # Vi tager 2024 som validering under træningen
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
    df_raw_val = df_full[(df_full.index >= VAL_START) & (df_full.index < TEST_START)].copy()
    
    try:
        # Load både Train og Val features
        df_features_train = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
        df_features_val = pd.read_csv('data/processed_val.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("CRITICAL ERROR: Processed CSVs not found. Run main.py first!")
        exit()
        
    # Align Train
    common_train = df_raw_train.index.intersection(df_features_train.index)
    df_raw_train = df_raw_train.loc[common_train]
    df_features_train = df_features_train.loc[common_train]

    # Align Val
    common_val = df_raw_val.index.intersection(df_features_val.index)
    df_raw_val = df_raw_val.loc[common_val]
    df_features_val = df_features_val.loc[common_val]
    
    return df_features_train, df_raw_train, df_features_val, df_raw_val

def get_optimal_params():
    default_params = {
        'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 128,
        'ent_coef': 0.001, 'gamma': 0.99, 'lstm_hidden': 256,
        'net_arch': 'small'
    }
    if os.path.exists("best_hyperparams.txt"):
        try:
            with open("best_hyperparams.txt", "r") as f:
                content = f.read()
                if content:
                    best_params = ast.literal_eval(content)
                    default_params.update(best_params)
                    print(f"Loaded Optimized Params: {best_params}")
        except Exception:
            pass
    return default_params

def get_net_arch_dict(arch_type):
    if arch_type == 'medium': return dict(pi=[128, 128], vf=[128, 128])
    elif arch_type == 'large': return dict(pi=[256, 256], vf=[256, 256])
    else: return dict(pi=[64, 64], vf=[64, 64])

def make_env(rank, df_features, df_raw, spread=0.0002):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        return env
    return _init

def train_ensemble(n_models=50): # Opdateret default til 50
    print(f"--- Starter HPC Ensemble Træning ({n_models} Agenter) ---")
    
    # Load data (Inkl. validation set nu)
    df_feat_train, df_raw_train, df_feat_val, df_raw_val = load_and_slice_data()
    params = get_optimal_params()
    
    current_net_arch = get_net_arch_dict(params.get('net_arch', 'small'))
    
    # HPC Scaling
    num_cpus = multiprocessing.cpu_count()
    n_envs = min(32, max(8, num_cpus - 2)) 
    print(f"CPU cores: {num_cpus}, bruger {n_envs} envs pr. agent.")

    # Opgraderet Learning Rate med Decay
    initial_lr = params['learning_rate']
    lr_schedule = linear_schedule(initial_lr)

    for i in range(n_models):
        seed = 42 + i 
        print(f"\n============================================")
        print(f"TRÆNER AGENT {i+1}/{n_models} (Seed {seed})")
        print(f"============================================")
        set_random_seed(seed)
        
        # 1. Training Envs (Parallel)
        env = SubprocVecEnv([make_env(k, df_feat_train, df_raw_train) for k in range(n_envs)])
        env = VecMonitor(env)
        
        # 2. Validation Env (Single - til callback)
        # Vi bruger val-data til at måle om modellen faktisk lærer noget generelt
        eval_env = SubprocVecEnv([make_env(0, df_feat_val, df_raw_val)])
        eval_env = VecMonitor(eval_env)
        
        # 3. Setup Callback til at gemme BEDSTE model
        best_model_path = f"models/ensemble_seed_{i}/"
        os.makedirs(best_model_path, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=f"logs/ensemble_{seed}/",
            eval_freq=max(50000 // n_envs, 1000), # Evaluer hver ~50k steps
            deterministic=True,
            render=False,
            verbose=0
        )
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            verbose=1, 
            device="cuda", 
            tensorboard_log=f"./logs/ensemble_{seed}/",
            
            learning_rate=lr_schedule, # <-- Bruger decay nu
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            ent_coef=params['ent_coef'],
            gamma=params['gamma'],
            
            policy_kwargs=dict(
                enable_critic_lstm=True, 
                lstm_hidden_size=params.get('lstm_hidden', 256),
                net_arch=current_net_arch,
                activation_fn=nn.Tanh
            )
        )
        
        try:
            # Opgraderet til 8 mio steps for sikkerheds skyld
            # EvalCallback sørger for at vi ikke mister den bedste model, selvom vi træner for længe
            model.learn(total_timesteps=8_000_000, callback=eval_callback, progress_bar=True)
            
            # Efter træning: Hent den bedste model tilbage og gem den som den endelige fil
            # EvalCallback gemmer som 'best_model.zip'. Vi omdøber den til vores standard.
            if os.path.exists(f"{best_model_path}/best_model.zip"):
                print(f"Fandt bedre checkpoint! Oerstatter final model.")
                shutil.copy(f"{best_model_path}/best_model.zip", f"models/ppo_ensemble_seed_{i}.zip")
            else:
                print("Advarsel: Ingen bedre model fundet undervejs, gemmer sidste state.")
                model.save(f"models/ppo_ensemble_seed_{i}")
                
        except Exception as e:
            print(f"FEJL under træning af agent {i}: {e}")
            continue # Fortsæt til næste agent i stedet for at crashe alt
            
        finally:
            env.close()
            eval_env.close()

if __name__ == "__main__":
    train_ensemble(n_models=50)