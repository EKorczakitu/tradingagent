import os
import pandas as pd
import numpy as np

import dataloading
import features
import trading_env
import backtest
import trade
import tune
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import SAC
import main
import trade # For get_optimal_params

def run_test():
    print("--- 1. STARTING LOCAL TEST PIPELINE ---")
    
    # Load and subset data to be fast
    df_full = dataloading.get_full_dataset()
    print(f"Original dataset shape: {df_full.shape}")
    df_full = df_full.iloc[-2000:].copy() # Kun de seneste 2000 rækker for testen
    print(f"Test dataset shape: {df_full.shape}")
    
    # Generate features
    df_features_full = features.generate_alpha_pool(df_full)
    
    # Dynamiske tids-splits baseret på dataens længde
    TEST_START_DATE = df_features_full.index[-400]
    VAL_START_DATE  = df_features_full.index[-800]
    
    mask_train = df_features_full.index < VAL_START_DATE
    mask_val   = (df_features_full.index >= VAL_START_DATE) & (df_features_full.index < TEST_START_DATE)
    mask_test  = df_features_full.index >= TEST_START_DATE
    
    X_train = df_features_full[mask_train].copy()
    X_val   = df_features_full[mask_val].copy()
    X_test  = df_features_full[mask_test].copy()
    
    prices_full = df_full.copy()

    print("\n--- 2. DATA CLEANING ---")
    
    X_train_scaled = X_train.dropna().copy()
    X_val_scaled  = X_val.dropna().copy()
    X_test_scaled = X_test.dropna().copy()
    
    print("\n--- 3. FEATURE SELECTION ---")
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train_scaled)
    selected_cols = X_train_scaled.columns[selector.get_support()].tolist()
    
    train_final = X_train_scaled[selected_cols].copy()
    val_final  = X_val_scaled.loc[X_val_scaled.index.intersection(X_val_scaled.index), selected_cols]
    test_final = X_test_scaled.loc[X_test_scaled.index.intersection(X_test_scaled.index), selected_cols]
    
    def align_prices(features_df, raw_df):
        common_idx = features_df.index.intersection(raw_df.index)
        return raw_df.loc[common_idx]

    prices_train_aligned = align_prices(train_final, prices_full)
    prices_val_aligned   = align_prices(val_final, prices_full)
    prices_test_aligned  = align_prices(test_final, prices_full)

    print("\n--- 4. FAST HYPERPARAMETER TUNING ---")
    tune.run_tuning(
        train_feat=train_final,
        val_feat=val_final,
        train_prices=prices_train_aligned,
        val_prices=prices_val_aligned,
        n_trials=2,             # Minimal tuning
        total_timesteps=2000    # Hurtig eval
    )
    
    print("\n--- 5. TRAINING SMALL ENSEMBLE ---")
    MODEL_SAVE_PATH = "models/test_ensemble"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    ensemble_models = []
    
    # Train 2 models synchronously for test purpose
    for i in range(2):
        seed = 42 + i
        print(f"\nTraining Model {i+1}...")
        model = trade.train_agent(
            train_df=train_final, 
            val_df=val_final, 
            raw_prices_train=prices_train_aligned,
            raw_prices_val=prices_val_aligned,
            seed=seed,
            total_timesteps=5000 # Kun 5000 timesteps til at validere pipelines loops
        )
        save_path = os.path.join(MODEL_SAVE_PATH, f"test_model_seed_{seed}.zip")
        model.save(save_path)
        
        # Load directly
        params = trade.get_optimal_params()
        window_size = params.get('window_size', 20)
        dummy_env = DummyVecEnv([lambda: trading_env.TradingEnv(val_final, prices_val_aligned)])
        dummy_env = VecFrameStack(dummy_env, n_stack=window_size)
        loaded_model = SAC.load(save_path, env=dummy_env)
        ensemble_models.append(loaded_model)

    ensemble_agent = main.EnsembleModel(ensemble_models)
    
    print("\n--- 6. BACKTESTING ---")
    env_test_base = DummyVecEnv([lambda: trading_env.TradingEnv(test_final, prices_test_aligned)])
    env_test = VecFrameStack(env_test_base, n_stack=20) # Sørger for at observationer er batched og timet
    backtest.run_backtest_engine(env_test, ensemble_agent, title="Test Run - Out of Sample")
    
    print("\n--- PIPELINE EXECUTED SUCCESSFULLY ---")

if __name__ == "__main__":
    run_test()
