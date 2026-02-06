import pandas as pd
import numpy as np
import os
import dataloading
import features
import feature_selection
import trading_env
import backtest
import trade

# Settings
MODEL_SAVE_PATH = "models/ppo_agent"
TEST_START_DATE = "2025-01-01"
VAL_START_DATE  = "2024-01-01"

def run_pipeline():
    print("\n--- 1. STARTING PIPELINE (HPC MODE - ALIGNMENT FIX) ---")

    # --- TRIN 1: LOAD DATA & GENERATE FEATURES (samlet for at undgå warm-up tab) ---
    print("Loading data and generating features...")
    df_full = dataloading.get_full_dataset()
    
    # Generer features på hele sættet FØR split
    # Dette sikrer at 1. januar 2024 har valid RSI/MACD baseret på dec 2023 data
    df_features_full = features.generate_alpha_pool(df_full)
    
    # --- TRIN 2: SPLIT DATA ---
    print("Splitting data...")
    # Vi bruger .loc for at sikre tids-indeks overholdes
    mask_train = df_features_full.index < VAL_START_DATE
    mask_val   = (df_features_full.index >= VAL_START_DATE) & (df_features_full.index < TEST_START_DATE)
    mask_test  = df_features_full.index >= TEST_START_DATE
    
    X_train = df_features_full[mask_train].copy()
    X_val   = df_features_full[mask_val].copy()
    X_test  = df_features_full[mask_test].copy()
    
    # Vi gemmer også rå priser til Environments, men venter med alignment
    prices_full = df_full.copy()

    # --- TRIN 3: NORMALISERING & CLEANING ---
    print("\n--- 3. NORMALIZING ---")
    # Fit scaler KUN på training data for at undgå data leakage
    X_train_scaled, scaler = features.normalize_features(X_train)
    
    # Helper der håndterer NaNs og alignment
    def process_split(df_feat, scl):
        numeric = df_feat.select_dtypes(include=['float32', 'float64']).columns
        # Clip outliers
        df_feat[numeric] = df_feat[numeric].clip(upper=1e9, lower=-1e9)
        
        # VIGTIGT: Drop NaNs der opstod under feature gen
        df_clean = df_feat.dropna()
        
        # Transform
        data_scaled = scl.transform(df_clean)
        return pd.DataFrame(data_scaled, columns=df_clean.columns, index=df_clean.index)

    X_val_scaled  = process_split(X_val, scaler)
    X_test_scaled = process_split(X_test, scaler)
    
    print(f"Cleaned shapes -> Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # --- TRIN 4: FEATURE SELECTION ---
    print("\n--- 4. FEATURE SELECTION (PERMUTATION) ---")
    # Bemærk: Feature selection kan fjerne yderligere rækker (f.eks. sidste række pga. shift)
    train_final, dropped_cols = feature_selection.feature_selection_funnel(
        X_train_scaled, 
        method='permutation', 
        top_k_features=50
    )
    
    selected_cols = train_final.columns.tolist()
    
    # Anvend selection på Val og Test
    # Vi bruger .loc for at sikre vi kun tager de rækker der overlevede cleaningen i trin 3
    val_final  = X_val_scaled.loc[X_val_scaled.index.intersection(X_val_scaled.index), selected_cols]
    test_final = X_test_scaled.loc[X_test_scaled.index.intersection(X_test_scaled.index), selected_cols]
    
    # --- TRIN 4.5: CRITICAL DATA ALIGNMENT ---
    # Dette er tricket! Vi skal sikre at 'prices' har PRÆCIS samme index som 'features'
    # Ellers crasher TradingEnv fordi arrays har forskellig længde.
    
    def align_prices(features_df, raw_df):
        # Find fælles index
        common_idx = features_df.index.intersection(raw_df.index)
        return raw_df.loc[common_idx]

    prices_train_aligned = align_prices(train_final, prices_full)
    prices_val_aligned   = align_prices(val_final, prices_full)
    prices_test_aligned  = align_prices(test_final, prices_full)
    
    # Check for sikkerheds skyld
    assert len(train_final) == len(prices_train_aligned), "CRITICAL: Train Features/Prices length mismatch!"
    assert len(val_final) == len(prices_val_aligned), "CRITICAL: Val Features/Prices length mismatch!"
    
    print(f"Selected {len(selected_cols)} features.")
    print(f"Final dataset sizes (aligned): Train: {len(train_final)}, Val: {len(val_final)}")

    # --- TRIN 5: TRÆNING (AI) ---
    print("\n--- 5. TRAINING AGENT ---")
    
    model = trade.train_agent(
        train_df=train_final, 
        val_df=val_final, 
        raw_prices_train=prices_train_aligned, # Nu aligned!
        raw_prices_val=prices_val_aligned      # Nu aligned!
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"Model gemt til {MODEL_SAVE_PATH}")

    # --- TRIN 6: BACKTEST ---
    print("\n--- 6. BACKTESTING ---")
    
    # Opret Environments med ALIGNED data
    env_val = trading_env.TradingEnv(val_final, prices_val_aligned)
    env_test = trading_env.TradingEnv(test_final, prices_test_aligned)
    
    print("\n>>> VALIDATION SET RESULTS:")
    backtest.run_backtest_engine(env_val, model, title="Validation 2024")
    
    print("\n>>> TEST SET RESULTS (OUT-OF-SAMPLE):")
    backtest.run_backtest_engine(env_test, model, title="Test 2025")

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()