import optuna
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

# --- Library Imports ---
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv # Tilføjet
import multiprocessing

# --- Local Imports ---
from trading_env import TradingEnv

def run_tuning(train_feat, val_feat, train_prices, val_prices):
    print("\n--- Starting Optuna Tuning (HPC Mode) ---")
    print(f"Tuning Input Data -> Train: {train_feat.shape}, Val: {val_feat.shape}")
    
    def objective(trial):
        # --- 1. Suggest Hyperparameters ---
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.995)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 1.0)
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
        
        if net_arch_type == "small":
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        elif net_arch_type == "medium":
            net_arch = dict(pi=[128, 128], vf=[128, 128])
        elif net_arch_type == "large":
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden", [128, 256, 512])
        
        if batch_size > n_steps:
            batch_size = n_steps

        # --- 2. Setup Environments (Nu 100% DummyVecEnv med Monitor) ---
        train_env = DummyVecEnv([lambda: Monitor(TradingEnv(train_feat, train_prices))])
        val_env = DummyVecEnv([lambda: Monitor(TradingEnv(val_feat, val_prices))])

        # --- 3. Define Model ---
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=dict(
                enable_critic_lstm=True,
                lstm_hidden_size=lstm_hidden_size,
                net_arch=net_arch,
                activation_fn=nn.Tanh
            ),
            verbose=0,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # --- 4. Train with Early Stopping ---
        eval_callback = EvalCallback(
            val_env, 
            best_model_save_path=None,
            log_path=None, 
            eval_freq=50000,
            n_eval_episodes=1,
            deterministic=True, 
            render=False
        )
        
        try:
            model.learn(total_timesteps=500_000, callback=eval_callback) # Hurtig test
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1000 
        finally:
            train_env.close()
            
        # --- 5. Evaluate Performance ---
        mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=1)
        trial.set_user_attr("net_arch", net_arch_type)

        # --- NYT: TVING GPU TIL AT RYDDE OP ---
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return mean_reward

    print("--- Starting Optuna Study ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # KUN 2 TRIALS TIL TEST
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    print("\n--- Tuning Complete ---")
    print("Best Params:", study.best_params)
    
    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    # Test block (hvis man kører tune.py alene, skal man bruge dummy data eller loade selv)
    print("Run main.py to execute the full pipeline including tuning.")
