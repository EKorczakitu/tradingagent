import os
import multiprocessing
import ast
import torch.nn as nn
from typing import Callable
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed # Vigtig
from trading_env import TradingEnv

# Settings
LOG_DIR = "logs/PPO_Agent"
MODEL_DIR = "models/PPO_Agent"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_optimal_params():
    default_params = {
        'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 256,
        'ent_coef': 0.001, 'gamma': 0.99, 'lstm_hidden': 512,
        'net_arch': 'medium'
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
    elif arch_type == 'xlarge': return dict(pi=[512, 512, 512], vf=[512,512,512]) # Tilføjet xlarge support
    return dict(pi=[64, 64], vf=[64, 64])

def make_env(rank, df_features, df_raw, spread=0.0002, seed=0):
    """Factory function for multiprocessing med Seed"""
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        env.reset(seed=seed + rank) # Seeding af selve environment
        return env
    return _init

def train_agent(train_df, val_df, raw_prices_train, raw_prices_val, seed=None):
    """
    Nu med 'seed' argument for at skabe diversitet i ensemblet.
    """
    # 1. Set global random seed hvis angivet
    if seed is not None:
        print(f"--- Setting Random Seed to {seed} ---")
        set_random_seed(seed)
    
    params = get_optimal_params()
    net_arch = get_net_arch(params.get('net_arch', 'small'))
    
    # Setup HPC Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    # Vi bruger lidt færre environments pr. model når vi kører ensemble for ikke at dræbe RAM
    n_envs = min(8, max(2, num_cpus - 2)) 
    
    # Opret Training Environment med seed offsets
    env = SubprocVecEnv([make_env(i, train_df, raw_prices_train, seed=(seed or 0)) for i in range(n_envs)])
    env = VecMonitor(env, filename=f"{LOG_DIR}/monitor_train_{seed}")

    # Opret Validation Environment
    eval_env = SubprocVecEnv([make_env(0, val_df, raw_prices_val, seed=(seed or 0))])
    eval_env = VecMonitor(eval_env, filename=f"{LOG_DIR}/monitor_val_{seed}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(20000 // n_envs, 1000),
        deterministic=True,
        render=False,
        verbose=0
    )

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
        seed=seed, # VIGTIGT: Sender seed til PPO
        policy_kwargs=dict(
            enable_critic_lstm=True,
            lstm_hidden_size=params.get('lstm_hidden', 256),
            net_arch=net_arch,
            activation_fn=nn.Tanh
        )
    )

    try:
        # Juster timesteps efter behov. 5M er standard, men til ensemble kan man evt. køre lidt mindre pr. model
        model.learn(total_timesteps=5_000_000, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        pass
    
    # Indlæs bedste model for dette seed
    best_model_path = os.path.join(f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        model = RecurrentPPO.load(best_model_path, env=env)
    
    env.close()
    eval_env.close()
    
    return model