import optuna
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

# --- Library Imports ---
from stable_baselines3 import PPO # Ændret fra RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # Tilføjet VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- Local Imports ---
from trading_env import TradingEnv
# Vi importerer din nye Custom Transformer fra trade.py
from trade import TimeSeriesTransformerExtractor 

def run_tuning(train_feat, val_feat, train_prices, val_prices):
    print("\n--- Starting Optuna Tuning (Transformer HPC Mode) ---")
    print(f"Tuning Input Data -> Train: {train_feat.shape}, Val: {val_feat.shape}")
    
    def objective(trial):
        # --- 1. Suggest Hyperparameters ---
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.995)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 1.0)
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        
        # Reduceret net_arch størrelser for at spare hukommelse med Transformeren
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
        if net_arch_type == "small":
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        elif net_arch_type == "medium":
            net_arch = dict(pi=[128, 128], vf=[128, 128])

        n_steps = trial.suggest_categorical("n_steps", [2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [512, 1024])
        
        # --- NYT: TST Hyperparameter ---
        # Hvor mange fortidige timer skal Attention-mekanismen kigge på ad gangen?
        window_size = trial.suggest_categorical("window_size", [10, 20, 30])
        
        if batch_size > n_steps:
            batch_size = n_steps

        # --- 2. Setup Environments med VecFrameStack ---
        train_env = DummyVecEnv([lambda: Monitor(TradingEnv(train_feat, train_prices))])
        train_env = VecFrameStack(train_env, n_stack=window_size)
        
        val_env = DummyVecEnv([lambda: Monitor(TradingEnv(val_feat, val_prices))])
        val_env = VecFrameStack(val_env, n_stack=window_size)

        # --- 3. Define Model (PPO med Transformer Extractor) ---
        policy_kwargs = dict(
            features_extractor_class=TimeSeriesTransformerExtractor,
            features_extractor_kwargs=dict(
                window_size=window_size,
                features_dim=128,
                n_heads=4,
                n_layers=2,
                d_model=64
            ),
            net_arch=net_arch,
            activation_fn=nn.Tanh
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
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
            # Optuna behøver ikke køre 1 mio. steps pr. trial. 300k er nok til at finde retningen.
            model.learn(total_timesteps=300_000, callback=eval_callback) 
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1000 
        finally:
            train_env.close()
            
        # --- 5. Evaluate Performance ---
        mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=1)
        trial.set_user_attr("net_arch", net_arch_type)

        # --- Tving GPU til at rydde op ---
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return mean_reward

    print("--- Starting Optuna Study ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    print("\n--- Tuning Complete ---")
    print("Best Params:", study.best_params)
    
    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    print("Run main.py to execute the full pipeline including tuning.")