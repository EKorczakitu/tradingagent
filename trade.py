import os
import multiprocessing
import ast
import torch
import torch.nn as nn
import math
from typing import Callable

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from trading_env import TradingEnv

LOG_DIR = "logs/SAC_Agent"
MODEL_DIR = "models/SAC_Agent"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class PatchTSTExtractor(BaseFeaturesExtractor):
    """
    SOTA PatchTST Feature Extractor.
    Deler tidsrækken op i 'patches' frem for punkt-til-punkt attention for bedre lokal semantik.
    """
    def __init__(self, observation_space, window_size=20, features_dim=128, patch_len=5, n_heads=4, n_layers=2, d_model=64):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.n_features = observation_space.shape[0] // window_size
        self.patch_len = patch_len
        self.n_patches = window_size // patch_len
        self.d_model = d_model 

        # Projicér hver patch til d_model dimensioner
        self.patch_proj = nn.Linear(self.patch_len * self.n_features, self.d_model)
        
        # Learnable Positional Encoding for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * (1 / math.sqrt(self.d_model)))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=0.2,
            activation="gelu", 
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Saml alle patches til sidst for at trække features
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * self.d_model, features_dim),
            nn.LayerNorm(features_dim)
        )

        mask = nn.Transformer.generate_square_subsequent_mask(self.n_patches)
        self.register_buffer("causal_mask", mask)

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.window_size, self.n_features)
        
        # Patching: (batch, n_patches, patch_len, features)
        x = x.view(batch_size, self.n_patches, self.patch_len, self.n_features)
        x = x.view(batch_size, self.n_patches, -1) # Flatten hver patch internt
        
        x = self.patch_proj(x) + self.pos_embedding
        
        x = self.transformer(x, mask=self.causal_mask, is_causal=True)
        
        return self.output_proj(x)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_optimal_params():
    default_params = {
        'learning_rate': 3e-4, 'batch_size': 256,
        'ent_coef': 'auto', 'gamma': 0.99, 'net_arch': 'medium',
        'window_size': 20
    }
    # Hvis hyperparams fil eksisterer fra tuning, så læs dem
    if os.path.exists('best_hyperparams.txt'):
        try:
            with open('best_hyperparams.txt', 'r') as f:
                content = f.read()
                tuned_params = ast.literal_eval(content)
                default_params.update(tuned_params)
        except Exception as e:
            print(f"Kunne ikke læse best_hyperparams.txt: {e}")
    return default_params

def get_net_arch(arch_type):
    # SAC bruger 'qf' (Q-funktion) frem for 'vf' (Value-funktion)
    if arch_type == 'medium': return dict(pi=[128, 128], qf=[128, 128])
    elif arch_type == 'large': return dict(pi=[256, 256], qf=[256, 256])
    return dict(pi=[64, 64], qf=[64, 64])

def make_env(rank, df_features, df_raw, spread=0.001, seed=0):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_agent(train_df, val_df, raw_prices_train, raw_prices_val, seed=None, total_timesteps=15_000_000, gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    if seed is not None:
        set_random_seed(seed)
    
    params = get_optimal_params()
    net_arch = get_net_arch(params.get('net_arch', 'small'))
    window_size = params.get('window_size', 20)
    
    n_envs = 16
    
    # Parallel CPU-dataopsamling (massiv acceleration)
    env = SubprocVecEnv([make_env(i, train_df, raw_prices_train, seed=(seed or 0)) for i in range(n_envs)], start_method="spawn")
    env = VecFrameStack(env, n_stack=window_size)

    # Eval env kan køre med 1 eller flere, vi kører med 4 for hurtigere evaluation
    eval_env = SubprocVecEnv([make_env(i, val_df, raw_prices_val, seed=(seed or 0)) for i in range(4)], start_method="spawn")
    eval_env = VecFrameStack(eval_env, n_stack=window_size)
    eval_env = VecMonitor(eval_env, filename=f"{LOG_DIR}/monitor_val_{seed}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR,
        log_path=LOG_DIR, eval_freq=10000, n_eval_episodes=1, deterministic=True, verbose=0
    )

    # CRITICAL FIX: Nu peger vi på PatchTSTExtractor
    policy_kwargs = dict(
        features_extractor_class=PatchTSTExtractor,
        features_extractor_kwargs=dict(
            window_size=window_size,
            features_dim=128,
            patch_len=5,
            n_heads=4,
            n_layers=2
        ),
        net_arch=net_arch,
        activation_fn=nn.GELU # SOTA activation for RL/Transformers
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=params['learning_rate'],
        buffer_size=100000,
        learning_starts=1000,
        batch_size=params['batch_size'],
        ent_coef=params.get('ent_coef', 'auto'),
        gamma=params['gamma'],
        policy_kwargs=policy_kwargs,
        seed=seed
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    
    best_model_path = os.path.join(f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        model = SAC.load(best_model_path, env=env)
    
    env.close()
    eval_env.close()
    
    return model