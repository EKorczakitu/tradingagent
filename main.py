import pandas as pd
import os
import dataloading
import features
import feature_selection
import trading_env
import backtest
import trade  # Din fil til træning (skal indeholde train_model funktion)

# Settings
MODEL_SAVE_PATH = "models/ppo_agent"
TEST_START_DATE = "2025-01-01"
VAL_START_DATE  = "2024-01-01"

def run_pipeline():
    print("\n--- 1. STARTING PIPELINE (HPC MODE) ---")

    # --- TRIN 1: LOAD DATA ---
    print("Loading data...")
    df_full = dataloading.get_full_dataset()
    
    # Split Data
    df_train = df_full[df_full.index < VAL_START_DATE].copy()
    df_val   = df_full[(df_full.index >= VAL_START_DATE) & (df_full.index < TEST_START_DATE)].copy()
    df_test  = df_full[df_full.index >= TEST_START_DATE].copy()
    
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # --- TRIN 2: FEATURE ENGINEERING ---
    print("\n--- 2. GENERATING ALPHA POOL ---")
    # Bruger din eksisterende logik (forenklet her for overblik)
    # Husk warm-up logikken fra din originale main.py hvis nødvendigt, 
    # men her kalder vi bare direkte for demonstration.
    train_processed = features.generate_alpha_pool(df_train)
    val_processed   = features.generate_alpha_pool(df_val)
    test_processed  = features.generate_alpha_pool(df_test)
    
    # --- TRIN 3: NORMALISERING ---
    print("\n--- 3. NORMALIZING ---")
    train_scaled, scaler = features.normalize_features(train_processed)
    
    # Helper til at bruge train-scaler på val/test
    def apply_scaler(df, scl):
        numeric = df.select_dtypes(include=['float32', 'float64']).columns
        df[numeric] = df[numeric].clip(upper=1e9, lower=-1e9) # Safety clip
        df.dropna(inplace=True)
        # Note: Dette er en forenkling. I din fulde kode, brug din 'apply_transform' funktion
        data = scl.transform(df)
        return pd.DataFrame(data, columns=df.columns, index=df.index)

    val_scaled  = apply_scaler(val_processed, scaler)
    test_scaled = apply_scaler(test_processed, scaler)

    # --- TRIN 4: FEATURE SELECTION ---
    print("\n--- 4. FEATURE SELECTION (PERMUTATION) ---")
    # Vi bruger nu Permutation Importance som aftalt
    train_final, dropped_cols = feature_selection.feature_selection_funnel(
        train_scaled, 
        method='permutation', 
        top_k_features=50
    )
    
    selected_cols = train_final.columns.tolist()
    val_final  = val_scaled[selected_cols]
    test_final = test_scaled[selected_cols]
    
    print(f"Selected {len(selected_cols)} features.")

    # --- TRIN 5: TRÆNING (AI) ---
    print("\n--- 5. TRAINING AGENT ---")
    
    # Her antager vi at trade.py har en funktion 'train_agent'
    # som tager dataframes og returnerer en trænet model.
    # Du skal måske justere navnet her afhængig af din trade.py
    model = trade.train_agent(
        train_df=train_final, 
        val_df=val_final, 
        raw_prices_train=df_train, # Rå priser til environment
        raw_prices_val=df_val
    )
    
    # Gem modellen
    model.save(MODEL_SAVE_PATH)
    print(f"Model gemt til {MODEL_SAVE_PATH}")

    # --- TRIN 6: BACKTEST (VALIDERING & TEST) ---
    print("\n--- 6. BACKTESTING ---")
    
    # Opret Environments til test
    env_val = trading_env.TradingEnv(val_final, df_val)
    env_test = trading_env.TradingEnv(test_final, df_test)
    
    print("\n>>> VALIDATION SET RESULTS:")
    backtest.run_backtest_engine(env_val, model, title="Validation 2024")
    
    print("\n>>> TEST SET RESULTS (OUT-OF-SAMPLE):")
    backtest.run_backtest_engine(env_test, model, title="Test 2025")

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()