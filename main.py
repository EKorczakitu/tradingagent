import pandas as pd
import dataloading
import features
import feature_selection

def run_pipeline():
    print("--- Starter Pipeline ---")

    # 1. INDLÆS DATA
    print("\n--- TRIN 1: Indlæser Data ---")
    df_train, df_val, df_test = dataloading.get_novo_nordisk_split()
    
    print(f"Train size: {df_train.shape}")
    print(f"Val size:   {df_val.shape}")
    print(f"Test size:  {df_test.shape}")

    # 2. GENERER FEATURES (Alpha Pool)
    # Vi skal gøre dette separat for hvert datasæt for at sikre, at features 
    # som rolling windows er korrekte (selvom man mister de første par rækker i hvert sæt).
    print("\n--- TRIN 2: Genererer Alpha Pool ---")
    train_processed = features.generate_alpha_pool(df_train)
    val_processed   = features.generate_alpha_pool(df_val)
    test_processed  = features.generate_alpha_pool(df_test)

    # 3. NORMALISERING (Vigtigt: Fit på Train, Transform på alle)
    print("\n--- TRIN 3: Normalisering ---")
    # Vi bruger funktionen fra features.py, men vi skal ændre lidt i logikken, fordi vi nu har 3 sæt. 
    # Jeg antager her, at du har normalize_features i features.py.
    # Vi kalder den for at få scaleren baseret på Train.
    
    train_scaled, scaler = features.normalize_features(train_processed)
    
    # Nu bruger vi den SAMME scaler til val og test manuelt for at undgå data leakage
    # Vi skal håndtere NaN/Inf først som i din funktion
    val_processed.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    val_processed.dropna(inplace=True)
    test_processed.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    test_processed.dropna(inplace=True)

    # Transform (brug scaler.transform, som returnerer numpy array, så vi laver det til DF igen)
    val_scaled_array = scaler.transform(val_processed)
    val_scaled = pd.DataFrame(val_scaled_array, columns=val_processed.columns, index=val_processed.index)

    test_scaled_array = scaler.transform(test_processed)
    test_scaled = pd.DataFrame(test_scaled_array, columns=test_processed.columns, index=test_processed.index)
    
    print("Normalisering færdig baseret på Træningssættets statistik.")

    # 4. FEATURE SELECTION (The Funnel)
    print("\n--- TRIN 4: Feature Selection (Funnel) ---")
    # Vi kører feature selection på Træningssættet for at finde de bedste kolonner
    train_final, dropped_log = feature_selection.feature_selection_funnel(
        train_scaled, 
        method='xgboost', 
        top_k_features=20
    )

    # Hvilke kolonner blev valgt?
    selected_columns = train_final.columns.tolist()
    print(f"\nValgte features ({len(selected_columns)}): {selected_columns}")

    # Filtrer Validation og Test sæt så de har PRÆCIS de samme kolonner
    val_final = val_scaled[selected_columns]
    test_final = test_scaled[selected_columns]

    print("\n--- Pipeline Færdig ---")
    print(f"Endelige dimensioner til AI-modellen:")
    print(f"Train: {train_final.shape}")
    print(f"Val:   {val_final.shape}")
    print(f"Test:  {test_final.shape}")
    
    # Gem de processerede data til CSV, så vi ikke skal køre det hele igen
    train_final.to_csv('data/processed_train.csv')
    val_final.to_csv('data/processed_val.csv')
    test_final.to_csv('data/processed_test.csv')

    return train_final, val_final, test_final

if __name__ == "__main__":
    run_pipeline()