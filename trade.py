import os
import multiprocessing
import ast
import torch
import torch.nn as nn
from typing import Callable

# --- ÆNDRET: Vi dropper RecurrentPPO og bruger standard PPO ---
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from trading_env import TradingEnv

# Settings
LOG_DIR = "logs/PPO_Agent"
MODEL_DIR = "models/PPO_Agent"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- NYT: CUSTOM TIME-SERIES TRANSFORMER ---
class TimeSeriesTransformerExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor der bygger en Transformer Encoder.
    Agenten lærer at bruge Multi-Head Attention på historiske markedsstrukturer.
    """
    def __init__(self, observation_space, window_size=20, features_dim=128, n_heads=4, n_layers=2, d_model=64):
        # SB3 forventer vi kalder super() med den endelige feature_dim (som sendes videre til MLP'en)
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        
        # VecFrameStack flader automatisk data ud til et 1D array. 
        # Vi udregner det oprindelige antal features pr. timestep:
        self.n_features = observation_space.shape[0] // window_size
        
        # 1. Input Projection: Skalerer features op til Transformerens d_model dimension
        self.input_proj = nn.Linear(self.n_features, d_model)
        
        # 2. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 2, 
            dropout=0.1, 
            batch_first=True # Vigtigt: Sørger for formen (batch_size, seq_len, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Output Projection: Skalerer ned til den forventede features_dim
        self.output_proj = nn.Linear(d_model, features_dim)

    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # 1. Genscab tidssekvensen (fra 1D til 2D tidsrække per batch)
        x = observations.view(batch_size, self.window_size, self.n_features)
        
        # 2. Transformer logik
        x = self.input_proj(x)
        x = self.transformer(x)
        
        # 3. Vi tager "konklusionen" fra det allersidste timestep i vinduet 
        # (svarende til nutiden, men med opmærksomhed på hele fortiden)
        last_step_features = x[:, -1, :]
        
        return self.output_proj(last_step_features)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_optimal_params():
    default_params = {
        'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 256,
        'ent_coef': 0.001, 'gamma': 0.99, 'net_arch': 'medium',
        # Vi tilføjer window_size til TST
        'window_size': 20
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

def make_env(rank, df_features, df_raw, spread=0.001, seed=0):
    def _init():
        env = TradingEnv(df_features, df_raw, spread=spread)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_agent(train_df, val_df, raw_prices_train, raw_prices_val, seed=None):
    if seed is not None:
        print(f"--- Setting Random Seed to {seed} ---")
        set_random_seed(seed)
    
    params = get_optimal_params()
    net_arch = get_net_arch(params.get('net_arch', 'small'))
    window_size = params.get('window_size', 20)
    
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = multiprocessing.cpu_count()
    
    print(f"SLURM allocated {num_cpus} CPUs for this job.")

    # --- NYT: VecFrameStack IMPLEMENTERING ---
    # Vi wrapper vores miljø med FrameStack, så den udspytter 'window_size' steps ad gangen
    env = DummyVecEnv([make_env(0, train_df, raw_prices_train, seed=(seed or 0))])
    env = VecFrameStack(env, n_stack=window_size)
    env = VecMonitor(env, filename=f"{LOG_DIR}/monitor_train_{seed}")

    eval_env = DummyVecEnv([make_env(0, val_df, raw_prices_val, seed=(seed or 0))])
    eval_env = VecFrameStack(eval_env, n_stack=window_size)
    eval_env = VecMonitor(eval_env, filename=f"{LOG_DIR}/monitor_val_{seed}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000, 
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=0
    )

    # Policy kwargs peger nu på din Custom Transformer Extractor
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

    # Brug standard PPO i stedet for RecurrentPPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(params['learning_rate']),
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        ent_coef=params['ent_coef'],
        gamma=params['gamma'],
        tensorboard_log=None,
        seed=seed,
        policy_kwargs=policy_kwargs
    )

    try:
        model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        pass
    
    best_model_path = os.path.join(f"{MODEL_DIR}/seed_{seed}" if seed is not None else MODEL_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        model = PPO.load(best_model_path, env=env)
    
    env.close()
    eval_env.close()
    
    return model