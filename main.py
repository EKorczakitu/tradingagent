import pandas as pd
import numpy as np
import os
import dataloading
import features
import feature_selection
import trading_env
import backtest
import trade
import tune 

# Settings
MODEL_SAVE_PATH = "models/ppo_ensemble" # Mappe til at gemme alle modeller
TEST_START_DATE = "2025-01-01"
VAL_START_DATE  = "2024-01-01"

# --- ENSEMBLE CLASS ---
class EnsembleModel:
    """
    En wrapper klasse der indeholder en liste af modeller.
    Når man kalder .predict(), kører den alle modeller og tager gennemsnittet (Soft Voting).
    Håndterer også LSTM states for alle modellerne.
    """
    def __init__(self, models):
        self.models = models
        print(f"Ensemble initialized with {len(models)} models.")

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # state parameteren er her en liste af states (én for hver model)
        # Hvis state er None (første step), laver vi en liste af Nones
        if state is None:
            state = [None] * len(self.models)
        
        all_actions = []
        new_states = []
        
        # Iterer gennem hver model og dens tilhørende state
        for i, model in enumerate(self.models):
            # Hent model-specifik state
            model_state = state[i]
            
            # Predict
            action, next_state = model.predict(obs, state=model_state, episode_start=episode_start, deterministic=deterministic)
            
            all_actions.append(action)
            new_states.append(next_state)
            
        # SOFT VOTING: Gennemsnit af actions på tværs af modeller (axis 0)
        avg_action = np.mean(all_actions, axis=0)
        
        return avg_action, new_states

    def save(self, path):
        # Vi gemmer ikke selve ensemble objektet, men vi antager at modellerne er gemt individuelt
        pass

def run_pipeline():
    print("\n--- 1. STARTING PIPELINE (HPC MODE - ENSEMBLE 9 MODELS) ---")

    # --- TRIN 1: LOAD DATA & GENERATE FEATURES ---
    print("Loading data and generating features...")
    df_full = dataloading.get_full_dataset()
    
    # Generer features på hele sættet FØR split
    df_features_full = features.generate_alpha_pool(df_full)
    
    # --- TRIN 2: SPLIT DATA ---
    print("Splitting data...")
    mask_train = df_features_full.index < VAL_START_DATE
    mask_val   = (df_features_full.index >= VAL_START_DATE) & (df_features_full.index < TEST_START_DATE)
    mask_test  = df_features_full.index >= TEST_START_DATE
    
    X_train = df_features_full[mask_train].copy()
    X_val   = df_features_full[mask_val].copy()
    X_test  = df_features_full[mask_test].copy()
    
    prices_full = df_full.copy()

    # --- TRIN 3: NORMALISERING & CLEANING ---
    print("\n--- 3. NORMALIZING ---")
    X_train_scaled, scaler = features.normalize_features(X_train)
    
    def process_split(df_feat, scl):
        numeric = df_feat.select_dtypes(include=['float32', 'float64']).columns
        df_feat[numeric] = df_feat[numeric].clip(upper=1e9, lower=-1e9)
        df_clean = df_feat.dropna()
        data_scaled = scl.transform(df_clean)
        return pd.DataFrame(data_scaled, columns=df_clean.columns, index=df_clean.index)

    X_val_scaled  = process_split(X_val, scaler)
    X_test_scaled = process_split(X_test, scaler)
    
    print(f"Cleaned shapes -> Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # --- TRIN 4: FEATURE SELECTION ---
    print("\n--- 4. FEATURE SELECTION (PERMUTATION) ---")
    train_final, dropped_cols = feature_selection.feature_selection_funnel(
        X_train_scaled, 
        method='permutation', 
        top_k_features=50
    )
    
    selected_cols = train_final.columns.tolist()
    
    val_final  = X_val_scaled.loc[X_val_scaled.index.intersection(X_val_scaled.index), selected_cols]
    test_final = X_test_scaled.loc[X_test_scaled.index.intersection(X_test_scaled.index), selected_cols]
    
    # --- TRIN 4.5: CRITICAL DATA ALIGNMENT ---
    def align_prices(features_df, raw_df):
        common_idx = features_df.index.intersection(raw_df.index)
        return raw_df.loc[common_idx]

    prices_train_aligned = align_prices(train_final, prices_full)
    prices_val_aligned   = align_prices(val_final, prices_full)
    prices_test_aligned  = align_prices(test_final, prices_full)
    
    assert len(train_final) == len(prices_train_aligned), "CRITICAL: Train Features/Prices length mismatch!"
    assert len(val_final) == len(prices_val_aligned), "CRITICAL: Val Features/Prices length mismatch!"
    
    print(f"Selected {len(selected_cols)} features.")

    # --- TRIN 4.8: HYPERPARAMETER TUNING ---
    print("\n--- 4.8. RUNNING HYPERPARAMETER TUNING (ONCE) ---")
    # Vi tuner kun én gang for at finde de generelle bedste parametre
    tune.run_tuning(
        train_feat=train_final,
        val_feat=val_final,
        train_prices=prices_train_aligned,
        val_prices=prices_val_aligned
    )

    # --- TRIN 5: ENSEMBLE TRAINING (9 MODELS) ---
    print("\n--- 5. TRAINING ENSEMBLE (9 RANDOM SEEDS) ---")
    
    ensemble_models = []
    n_models = 9
    
    # Opret mappe til ensemble modeller hvis den ikke findes
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    for i in range(n_models):
        seed = 42 + i # Forskellige seeds: 42, 43, 44...
        print(f"\n>>> Training Model {i+1}/{n_models} (Seed: {seed}) <<<")
        
        model = trade.train_agent(
            train_df=train_final, 
            val_df=val_final, 
            raw_prices_train=prices_train_aligned,
            raw_prices_val=prices_val_aligned,
            seed=seed # Sender seed videre til trade.py
        )
        
        # Gem den individuelle model
        save_path = os.path.join(MODEL_SAVE_PATH, f"model_seed_{seed}.zip")
        model.save(save_path)
        ensemble_models.append(model)
        print(f"Model {i+1} saved to {save_path}")

    # Opret Ensemble Objektet
    ensemble_agent = EnsembleModel(ensemble_models)

    # --- TRIN 6: BACKTEST ---
    print("\n--- 6. BACKTESTING ENSEMBLE ---")
    
    env_val = trading_env.TradingEnv(val_final, prices_val_aligned)
    env_test = trading_env.TradingEnv(test_final, prices_test_aligned)
    
    print("\n>>> VALIDATION SET RESULTS (ENSEMBLE):")
    backtest.run_backtest_engine(env_val, ensemble_agent, title="Validation Ensemble 2024")
    
    print("\n>>> TEST SET RESULTS (OUT-OF-SAMPLE ENSEMBLE):")
    backtest.run_backtest_engine(env_test, ensemble_agent, title="Test Ensemble 2025")

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()