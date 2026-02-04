import pandas as pd
import dataloading
import features
import feature_selection

def run_pipeline():
    print("--- Starter Pipeline (HPC / Walk-Forward Setup) ---")

    # 1. LOAD EVERYTHING
    df_full = dataloading.get_full_dataset()
    
    # 2. DEFINER SPLIT (Regime Awareness)
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
    VAL_START  = pd.Timestamp("2024-01-01", tz="UTC")
    
    # Train slutter hvor Val starter
    df_train = df_full[df_full.index < VAL_START].copy()
    df_val   = df_full[(df_full.index >= VAL_START) & (df_full.index < TEST_START)].copy()
    df_test  = df_full[df_full.index >= TEST_START].copy()

    print(f"Train Range: {df_train.index[0]} -> {df_train.index[-1]} ({len(df_train)} rows)")
    print(f"Val Range:   {df_val.index[0]}   -> {df_val.index[-1]}   ({len(df_val)} rows)")
    print(f"Test Range:  {df_test.index[0]}  -> {df_test.index[-1]}  ({len(df_test)} rows)")

    # 2. GENERER FEATURES (Med "Warm-Up" Buffer)
    print("\n--- TRIN 2: Genererer Alpha Pool med Warm-Up ---")
    
    # CRITICAL CHANGE: Øget til 300 for at støtte SMA200 og Volatility Long (100)
    # Hvis denne er for lav, mister vi data i starten af Val/Test pga. NaNs
    WARMUP_ROWS = 300 
    
    # --- Train Processing ---
    # Train starter fra scratch 
    train_processed = features.generate_alpha_pool(df_train)
    
    # --- Validation Processing (Fix for "Cold Start") ---
    if len(df_train) >= WARMUP_ROWS:
        warmup_data = df_train.iloc[-WARMUP_ROWS:].copy()
        val_with_warmup = pd.concat([warmup_data, df_val])
    else:
        val_with_warmup = df_val 
        
    val_processed_full = features.generate_alpha_pool(val_with_warmup)
    
    # Fjern warm-up rækkerne igen 
    val_start_time = df_val.index[0]
    val_processed = val_processed_full[val_processed_full.index >= val_start_time].copy()
    
    # --- Test Processing (Fix for "Cold Start") ---
    if len(df_val) >= WARMUP_ROWS:
        warmup_data_test = df_val.iloc[-WARMUP_ROWS:].copy()
        test_with_warmup = pd.concat([warmup_data_test, df_test])
    else:
        test_with_warmup = df_test
        
    test_processed_full = features.generate_alpha_pool(test_with_warmup)
    
    test_start_time = df_test.index[0]
    test_processed = test_processed_full[test_processed_full.index >= test_start_time].copy()

    print(f"Processed Train: {train_processed.shape}")
    print(f"Processed Val:   {val_processed.shape} (Ingen tab i starten!)")
    print(f"Processed Test:  {test_processed.shape} (Ingen tab i starten!)")

    # 3. NORMALISERING (Vigtigt: Fit på Train, Transform på alle)
    print("\n--- TRIN 3: Normalisering (Med Outlier Clipping) ---")
    
    # Fit scaler på Train
    train_scaled, scaler = features.normalize_features(train_processed)
    
    # Helper til at transformere Val/Test med Train's parametre
    def apply_transform(df_in, scaler_obj, clip_lower, clip_upper):
        df = df_in.copy()
        # Sikr at inf håndteres før vi kigger på typer
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        # Apply clipping limits from train
        df[numeric_cols] = df[numeric_cols].clip(lower=clip_lower, upper=clip_upper, axis=1)
        df.dropna(inplace=True)
        
        scaled_array = scaler_obj.transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

    # Hent clipping grænser fra Train
    numeric_cols = train_processed.select_dtypes(include=[float, int]).columns
    train_lower = train_processed[numeric_cols].quantile(0.001)
    train_upper = train_processed[numeric_cols].quantile(0.999)

    val_scaled = apply_transform(val_processed, scaler, train_lower, train_upper)
    test_scaled = apply_transform(test_processed, scaler, train_lower, train_upper)
    
    print("Normalisering færdig.")

    # 4. FEATURE SELECTION (The Funnel - HPC MODE)
    print("\n--- TRIN 4: Feature Selection (Funnel) ---")
    
    # HER SKER MAGIEN: Vi bruger 'rfe' (Recursive Feature Elimination)
    # og beder om 50 features i stedet for 20.
    train_final, dropped_log = feature_selection.feature_selection_funnel(
        train_scaled, 
        method='rfe',        # <--- Ændret til RFE (HPC krævende men bedre)
        top_k_features=50    # <--- Ændret til 50 features
    )

    selected_columns = train_final.columns.tolist()
    print(f"\nValgte features ({len(selected_columns)}): {selected_columns}")

    # Filtrer Validation og Test så de matcher
    val_final = val_scaled[selected_columns]
    test_final = test_scaled[selected_columns]

    print("\n--- Pipeline Færdig ---")
    print(f"Endelige dimensioner til AI-modellen:")
    print(f"Train: {train_final.shape}")
    print(f"Val:   {val_final.shape}")
    print(f"Test:  {test_final.shape}")
    
    # Gem til CSV
    train_final.to_csv('data/processed_train.csv')
    val_final.to_csv('data/processed_val.csv')
    test_final.to_csv('data/processed_test.csv')
    
    return train_final, val_final, test_final

if __name__ == "__main__":
    run_pipeline()