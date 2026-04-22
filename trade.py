import os
import multiprocessing
import ast
import torch
import torch.nn as nn
import math
from typing import Callable

from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from trading_env import TradingEnv

LOG_DIR = "logs/PPO_Agent"
MODEL_DIR = "models/PPO_Agent"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class AdvancedQuantTransformer(BaseFeaturesExtractor):
    """
    State-of-the-Art Transformer Extractor.
    Korrekt Causal Masking (Look-Ahead Bias protection) og Regime-Aware Positional Encoding.
    """
    def __init__(self, observation_space, window_size=20, features_dim=256, n_heads=8, n_layers=3):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.n_features = observation_space.shape[0] // window_size
        self.d_model = 128 

        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # SOTA: Learnable Positional Encoding for Regime Adaptation
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, self.d_model) * (1 / math.sqrt(self.d_model)))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=0.2, # Øget dropout for regularisering i finansiel tidsserie
            activation="gelu", 
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, features_dim),
            nn.LayerNorm(features_dim)
        )

        # Pre-compute causal mask (lower triangular)
        # PyTorch expects True for masked (ignored) positions
        mask = torch.triu(torch.ones(window_size, window_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.window_size, self.n_features)
        
        x = self.input_proj(x) + self.pos_embedding
        
        # Apply strict causal masking
        x = self.transformer(x, is_causal=True, mask=self.causal_mask)
        
        # Udtrækker konklusionen for det nuværende timestep
        return self.output_proj(x[:, -1, :])

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_optimal_params():
    default_params = {
        'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 256,
        'ent_coef': 0.005, 'gamma': 0.99, 'net_arch': 'medium',
        'window_size': 20
    }
    # ... (Samme læsning af best_hyperparams.txt som før)
    return default_params

def get_net_arch(arch_type):
    if arch_type == 'medium': return dict(pi=[128, 128], vf=[128, 128])
    elif arch_type == 'large': return dict(pi=[256, 256], vf=[256, 256])
    return dict(pi=[64, 64], vf=[64, 64])

def make_env(rank, df_features, df_raw, spread=0.001, seed=0):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_agent(train_df, val_df, raw_prices_train, raw_prices_val, seed=None):
    if seed is not None:
        set_random_seed(seed)
    
    params = get_optimal_params()
    net_arch = get_net_arch(params.get('net_arch', 'small'))
    window_size = params.get('window_size', 20)
    
    env = DummyVecEnv([make_env(0, train_df, raw_prices_train, seed=(seed or 0))])
    env = VecFrameStack(env, n_stack=window_size)
    env = VecMonitor(env, filename=f"{LOG_DIR}/monitor_train_{seed}")

    eval_env = DummyVecEnv([make_env(0, val_df, raw_prices_val, seed=(seed or 0))])
    eval_env = VecFrameStack(eval_env, n_stack=window_size)
    eval_env = VecMonitor(eval_env, filename=f"{LOG_DIR}/monitor_val_{seed}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR,
        log_path=LOG_DIR, eval_freq=10000, n_eval_episodes=1, deterministic=True, verbose=0
    )

    # CRITICAL FIX: Nu peger vi på AdvancedQuantTransformer
    policy_kwargs = dict(
        features_extractor_class=AdvancedQuantTransformer,
        features_extractor_kwargs=dict(
            window_size=window_size,
            features_dim=128,
            n_heads=8,
            n_layers=3
        ),
        net_arch=net_arch,
        activation_fn=nn.GELU # SOTA activation for RL/Transformers
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(params['learning_rate']),
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        ent_coef=params['ent_coef'],
        gamma=params['gamma'],
        policy_kwargs=policy_kwargs,
        seed=seed
    )

    model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
    
    best_model_path = os.path.join(f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        model = PPO.load(best_model_path, env=env)
    
    env.close()
    eval_env.close()
    
    return model