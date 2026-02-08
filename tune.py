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

# --- Local Imports ---
from trading_env import TradingEnv

def run_tuning(train_feat, val_feat, train_prices, val_prices):
    """
    Kører Optuna tuning på de dataframes, der leveres fra main.py pipeline.
    """
    print("\n--- Starting Optuna Tuning (HPC Mode) ---")
    print(f"Tuning Input Data -> Train: {train_feat.shape}, Val: {val_feat.shape}")
    
    def objective(trial):
        """
        Optuna Objective Function (Closure der har adgang til dataframes)
        """
        
        # --- 1. Suggest Hyperparameters ---
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.995)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 1.0)
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        
        # --- HPC SCALING: Architecture Search ---
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large", "xlarge"])
        
        if net_arch_type == "small":
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        elif net_arch_type == "medium":
            net_arch = dict(pi=[128, 128], vf=[128, 128])
        elif net_arch_type == "large":
            net_arch = dict(pi=[256, 256], vf=[256, 256])
        elif net_arch_type == "xlarge":
            net_arch = dict(pi=[512, 512, 512], vf=[512,512,512])

        # LSTM Specifics
        n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden", [128, 256, 512, 1024])
        
        # Constraint: Batch size must be a factor of n_steps (or smaller)
        if batch_size > n_steps:
            batch_size = n_steps

        # --- 2. Setup Environments ---
        # Vi bruger dataframes sendt fra main.py
        train_env = DummyVecEnv([lambda: TradingEnv(train_feat, train_prices)])
        val_env = DummyVecEnv([lambda: Monitor(TradingEnv(val_feat, val_prices))])
        
        # --- 3. Define Model (RecurrentPPO) ---
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
        eval_freq = max(25000, n_steps)
        
        eval_callback = EvalCallback(
            val_env, 
            best_model_save_path=None,
            log_path=None, 
            eval_freq=eval_freq,
            deterministic=True, 
            render=False
        )
        
        try:
            # Vi kører lidt færre steps pr. trial i tuning for at spare tid, eller du kan beholde 500k
            model.learn(total_timesteps=500000, callback=eval_callback)
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return -1000 
            
        # --- 5. Evaluate Performance ---
        mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=5)
        
        # Gem net_arch typen så vi kan bruge den senere
        trial.set_user_attr("net_arch", net_arch_type)
        
        return mean_reward

    print("--- Starting Optuna Study ---")
    
    # Create study to MAXIMIZE reward
    study = optuna.create_study(direction="maximize")
    
    # Kør 500 trials (HPC mode). Juster n_trials ned, hvis du vil teste hurtigere.
    print("Running 500 trials. This is a brute-force optimization.")
    study.optimize(objective, n_trials=500, show_progress_bar=True)
    
    print("\n--- Tuning Complete ---")
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)
    
    # Save best params to file so trade.py can read them automatically
    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))
    
    print("Saved best_hyperparams.txt")

if __name__ == "__main__":
    # Test block (hvis man kører tune.py alene, skal man bruge dummy data eller loade selv)
    print("Run main.py to execute the full pipeline including tuning.")