import pandas as pd
import os
import torch.nn as nn # Vigtig for at kunne bruge Tanh aktivering
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Importér vores egne moduler
import dataloading
from trading_env import TradingEnv

# Opret mapper
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_and_align_data():
    print("--- Indlæser Data ---")
    df_raw_train, df_raw_val, _ = dataloading.get_novo_nordisk_split()
    df_features_train = pd.read_csv('data/processed_train.csv', index_col=0, parse_dates=True)
    
    # Alignment
    df_raw_aligned = df_raw_train.loc[df_features_train.index]
    return df_features_train, df_raw_aligned

def train():
    # 1. Gør data klar
    df_features, df_raw = load_and_align_data()
    
    # 2. Opret Miljøet
    print("--- Starter Miljø ---")
    env = DummyVecEnv([lambda: Monitor(TradingEnv(df_features, df_raw))])
    
    # 3. Initialiser Agenten (Med OPTUNA VINDER PARAMETRE)
    print("--- Initialiserer PPO Agent med Optimerede Hyperparametre ---")
    
    # Her indsætter vi tallene fra dit resultat:
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
        
        # --- DINE VINDER TAL ---
        learning_rate=5.04e-5,        # Optuna: 5.0405...e-05
        ent_coef=0.0021,              # Optuna: 0.00211...
        batch_size=128,               # Optuna: 128
        n_steps=2048,                 # Optuna: 2048
        
        # Oversættelse af "net_arch: large":
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]), 
            activation_fn=nn.Tanh # Vi beholder Tanh, det er standard for finans
        ),
        
        gamma=0.99
    )
    
    # 4. Start Træning (Lang tid nu!)
    # Vi giver den 5.000.000 steps nu, da vi ved, at "hjernen" er indstillet korrekt.
    TRAIN_STEPS = 5_000_000 
    print(f"--- Starter Endelig Træning ({TRAIN_STEPS} steps) ---")
    
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
    
    # 5. Gem Modellen
    model_path = "models/ppo_novo_agent_optimized"
    model.save(model_path)
    print(f"--- Træning Færdig. Model gemt som '{model_path}.zip' ---")

if __name__ == "__main__":
    train()