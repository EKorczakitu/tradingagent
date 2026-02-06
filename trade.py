import os
import multiprocessing
import ast
import torch.nn as nn
from typing import Callable
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from trading_env import TradingEnv

# Settings
LOG_DIR = "logs/PPO_Agent"
MODEL_DIR = "models/PPO_Agent"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate decay"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_optimal_params():
    """Henter hyperparametre eller bruger defaults"""
    default_params = {
        'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 256,
        'ent_coef': 0.001, 'gamma': 0.99, 'lstm_hidden': 512,
        'net_arch': 'medium' # 'small', 'medium', 'large'
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

def get_net_arch(arch_type):
    if arch_type == 'medium': return dict(pi=[128, 128], vf=[128, 128])
    elif arch_type == 'large': return dict(pi=[256, 256], vf=[256, 256])
    return dict(pi=[64, 64], vf=[64, 64])

def make_env(rank, df_features, df_raw, spread=0.0002):
    """Factory function for multiprocessing"""
    def _init():
        return TradingEnv(df_features, df_raw, spread=spread)
    return _init

def train_agent(train_df, val_df, raw_prices_train, raw_prices_val):
    """
    Hovedfunktion kaldt af main.py.
    Træner agenten og returnerer den bedste model.
    """
    params = get_optimal_params()
    net_arch = get_net_arch(params.get('net_arch', 'small'))
    
    # Setup HPC Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    n_envs = min(32, max(4, num_cpus - 1)) # Brug næsten alle kerner
    print(f"Training on {n_envs} parallel environments...")

    # Opret Training Environment
    env = SubprocVecEnv([make_env(i, train_df, raw_prices_train) for i in range(n_envs)])
    env = VecMonitor(env, filename=f"{LOG_DIR}/monitor_train")

    # Opret Validation Environment
    eval_env = SubprocVecEnv([make_env(0, val_df, raw_prices_val)])
    eval_env = VecMonitor(eval_env, filename=f"{LOG_DIR}/monitor_val")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(20000 // n_envs, 1000),
        deterministic=True,
        render=False,
        verbose=1
    )

    # Definer Modellen (Recurrent PPO for tidsserier)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(params['learning_rate']),
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        ent_coef=params['ent_coef'],
        gamma=params['gamma'],
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            enable_critic_lstm=True,
            lstm_hidden_size=params.get('lstm_hidden', 256),
            net_arch=net_arch,
            activation_fn=nn.Tanh
        )
    )

    print("--- Starter Træning ---")
    try:
        # Kør 5 millioner steps (juster efter behov på HPC)
        model.learn(total_timesteps=20_000_000, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("Træning stoppet manuelt. Gemmer status...")
    
    # Indlæs den bedste model fra validering
    best_model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        print("Indlæser bedste model fra validering...")
        model = RecurrentPPO.load(best_model_path, env=env)
    
    env.close()
    eval_env.close()
    
    return model