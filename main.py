import pandas as pd
import dataloading
import features
import feature_selection

def run_pipeline():
    print("--- Starter Pipeline (Walk-Forward Setup) ---")

    # 1. LOAD EVERYTHING
    df_full = dataloading.get_full_dataset()
    
    # 2. DEFINER SPLIT (Regime Awareness)
    # Vi vil teste på 2024 (Val) og 2025 (Test)
    # Men vi vil træne på data LIGE INDEN 2024 for at fange det nyeste regime.
    
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
    VAL_START  = pd.Timestamp("2024-01-01", tz="UTC")
    
    # Train slutter hvor Val starter
    df_train = df_full[df_full.index < VAL_START].copy()
    df_val   = df_full[(df_full.index >= VAL_START) & (df_full.index < TEST_START)].copy()
    df_test  = df_full[df_full.index >= TEST_START].copy()

    # OPTIONAL: "Windowing" - Klip starten af Train af, hvis den er for gammel (f.eks. kun behold 4 år)
    # Dette hjælper hvis markedet i 2016 er irrelevant for 2024.
    # train_start_cutoff = VAL_START - pd.DateOffset(years=4)
    # df_train = df_train[df_train.index >= train_start_cutoff]

    print(f"Train Range: {df_train.index[0]} -> {df_train.index[-1]} ({len(df_train)} rows)")
    print(f"Val Range:   {df_val.index[0]}   -> {df_val.index[-1]}   ({len(df_val)} rows)")
    print(f"Test Range:  {df_test.index[0]}  -> {df_test.index[-1]}  ({len(df_test)} rows)")

    # 2. GENERER FEATURES (Med "Warm-Up" Buffer)
    print("\n--- TRIN 2: Genererer Alpha Pool med Warm-Up ---")
    
    # Hvor meget historik skal indikatorer bruge? (MACD bruger 26+9, så 100 er rigeligt sikkert)
    WARMUP_ROWS = 100 
    
    # --- Train Processing ---
    # Train starter fra scratch (accepterer koldt start tab i begyndelsen af 2016)
    train_processed = features.generate_alpha_pool(df_train)
    
    # --- Validation Processing (Fix for "Cold Start") ---
    # Tag slutningen af Train og sæt foran Val
    if len(df_train) >= WARMUP_ROWS:
        warmup_data = df_train.iloc[-WARMUP_ROWS:].copy()
        val_with_warmup = pd.concat([warmup_data, df_val])
    else:
        val_with_warmup = df_val # Fallback hvis train er for lille
        
    val_processed_full = features.generate_alpha_pool(val_with_warmup)
    
    # Fjern warm-up rækkerne igen (så vi kun har Val data tilbage)
    # Vi finder indekset hvor Val faktisk starter
    val_start_time = df_val.index[0]
    val_processed = val_processed_full[val_processed_full.index >= val_start_time].copy()
    
    # --- Test Processing (Fix for "Cold Start") ---
    # Tag slutningen af Val og sæt foran Test
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
    
    # Transform Val og Test med samme scaler
    # Note: Vi bruger scaler.transform direkte. normalize_features funktionen bruges kun til at fitte.
    # Vi skal dog stadig håndtere inf/nan og clipping på val/test manuelt eller via en helper
    # For at gøre det rent, genbruger vi logikken fra normalize_features men uden fit
    
    def apply_transform(df_in, scaler_obj, clip_lower, clip_upper):
        df = df_in.copy()
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        # Apply clipping limits from train
        df[numeric_cols] = df[numeric_cols].clip(lower=clip_lower, upper=clip_upper, axis=1)
        df.dropna(inplace=True)
        
        scaled_array = scaler_obj.transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

    # Hent clipping grænser fra Train (vi genberegner dem her for at have dem til apply_transform)
    # (Ideelt set skulle normalize_features returnere disse, men vi gør det her for at holde features.py simpel)
    numeric_cols = train_processed.select_dtypes(include=[float, int]).columns
    train_lower = train_processed[numeric_cols].quantile(0.001)
    train_upper = train_processed[numeric_cols].quantile(0.999)

    val_scaled = apply_transform(val_processed, scaler, train_lower, train_upper)
    test_scaled = apply_transform(test_processed, scaler, train_lower, train_upper)
    
    print("Normalisering færdig.")

    # 4. FEATURE SELECTION (The Funnel)
    print("\n--- TRIN 4: Feature Selection (Funnel) ---")
    
    train_final, dropped_log = feature_selection.feature_selection_funnel(
        train_scaled, 
        method='xgboost', 
        top_k_features=20
    )

    selected_columns = train_final.columns.tolist()
    print(f"\nValgte features ({len(selected_columns)}): {selected_columns}")

    # Filtrer Validation og Test
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