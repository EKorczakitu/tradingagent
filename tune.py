import optuna
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

# --- Library Imports ---
from optuna.pruners import MedianPruner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack # Tilføjet VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- Local Imports ---
from trading_env import TradingEnv
# Vi importerer din nye Custom Transformer fra trade.py
import trade

def generate_purged_cv_splits(n_samples, n_splits=3, purge_size=20):
    # Combinatorial Purged Cross-Validation (CPCV) logic (Simplified for HPC)
    # Deler data op i tids-blokke, og efterlader en "purge" margin mellem train og val
    fold_size = n_samples // n_splits
    splits = []
    for i in range(n_splits - 1):
        train_end = (i + 1) * fold_size - purge_size
        val_start = (i + 1) * fold_size + purge_size
        val_end = (i + 2) * fold_size
        splits.append(( (0, train_end), (val_start, val_end) ))
    return splits

def run_tuning(train_feat, val_feat, train_prices, val_prices, n_trials=20, total_timesteps=100_000):
    print("\n--- Starting Optuna Tuning (Transformer HPC Mode) ---")
    print(f"Tuning Input Data -> Train: {train_feat.shape}, Val: {val_feat.shape}")
    
    cv_splits = generate_purged_cv_splits(len(train_feat), n_splits=3, purge_size=24) # 24 timers purge

    def objective(trial):
        # --- 1. Suggest Hyperparameters ---
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999])
        
        # SAC har ikke ent_coef på samme måde som PPO, men auto er default. Vi tuner initial value af auto ent_coef eller sætter den fast.
        # Vi tuner i stedet tau (polyak averaging) for target network
        tau = trial.suggest_categorical("tau", [0.005, 0.01, 0.02, 0.05])
        
        net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
        window_size = trial.suggest_categorical("window_size", [10, 20, 30])
        
        params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gamma': gamma,
            'tau': tau,
            'net_arch': net_arch,
            'window_size': window_size
        }

        # --- 2. Setup Environments med VecFrameStack ---
        # I stedet for et enkelt train/val split, udfører vi Purged Cross-Validation
        cv_rewards = []
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            train_f = train_feat.iloc[train_idx[0]:train_idx[1]]
            train_p = train_prices.iloc[train_idx[0]:train_idx[1]]
            
            val_f = train_feat.iloc[val_idx[0]:val_idx[1]]
            val_p = train_prices.iloc[val_idx[0]:val_idx[1]]

            # Parallelize tuning environments
            def make_train_env(i):
                return lambda: Monitor(TradingEnv(train_f, train_p))
            
            def make_val_env(i):
                return lambda: Monitor(TradingEnv(val_f, val_p))

            train_env = SubprocVecEnv([make_train_env(i) for i in range(8)], start_method="spawn")
            train_env = VecFrameStack(train_env, n_stack=window_size)

            val_env = SubprocVecEnv([make_val_env(i) for i in range(4)], start_method="spawn")
            val_env = VecFrameStack(val_env, n_stack=window_size)

            # --- 3. Define Model (SAC med Transformer Extractor) ---
            policy_kwargs = dict(
                features_extractor_class=trade.PatchTSTExtractor,
                features_extractor_kwargs=dict(
                    window_size=params['window_size'],
                    features_dim=128,
                    patch_len=5,
                    n_heads=4,
                    n_layers=2
                ),
                net_arch=trade.get_net_arch(params['net_arch']),
                activation_fn=torch.nn.GELU
            )

            model = SAC(
                "MlpPolicy",
                train_env,
                verbose=0,
                learning_rate=params['learning_rate'],
                buffer_size=50000,
                learning_starts=1000,
                batch_size=params['batch_size'],
                tau=params['tau'],
                gamma=params['gamma'],
                policy_kwargs=policy_kwargs,
                seed=42
            )
            
            # Kortere total_timesteps pr. fold for at spare tid
            try:
                model.learn(total_timesteps=total_timesteps) 
            except Exception as e:
                print(f"Trial fold {fold} failed: {e}")
                return -1000 
            finally:
                train_env.close()
                
            # Evaluate current fold
            mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=1)
            cv_rewards.append(mean_reward)
            val_env.close()

            # Oprydning
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        trial.set_user_attr("net_arch", net_arch)
        
        # Returner gennemsnittet af CPCV reward
        return float(np.mean(cv_rewards))

    print("--- Starting Optuna Study (SQLite-backed for SLURM resume) ---")
    os.makedirs("tuning_meta", exist_ok=True)
    storage = "sqlite:///tuning_meta/optuna_study.db"
    study = optuna.create_study(
        study_name="quantforge_tuning",
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )
    
    # Resume: spring afsluttede trials over
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if remaining > 0:
        print(f"  Fundet {completed} færdige trials. Kører {remaining} mere...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True, n_jobs=1)
    else:
        print(f"  Alle {n_trials} trials allerede fuldført. Springer tuning over.")
    
    print("\n--- Tuning Complete ---")
    print("Best Params:", study.best_params)
    
    with open("best_hyperparams.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    print("Run main.py to execute the full pipeline including tuning.")