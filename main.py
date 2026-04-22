import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import dataloading
import features
import feature_selection
import trading_env
import backtest
import trade
import tune 
import shutil
import sys
import concurrent.futures
import multiprocessing

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

# --- CUSTOM TRANSFORMER TIL AT FORHINDRE DATA LÆKAGE ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class TrainQuantileWinsorizer(BaseEstimator, TransformerMixin):
    """
    Klipper outliers baseret på fraktiler udregnet UDELUKKENDE fra træningsdata.
    Dette forhindrer distribution shift og fremtids-lækage.
    """
    def __init__(self, lower_q=0.001, upper_q=0.999):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        
    def fit(self, X, y=None):
        # Gemmer de specifikke værdi-grænser for hver kolonne fra træningssættet
        self.lower_bounds_ = X.quantile(self.lower_q)
        self.upper_bounds_ = X.quantile(self.upper_q)
        return self
        
    def transform(self, X, y=None):
        # Bruger de GEMTE grænser til at klippe nye (val/test) data
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

# --- HARDCORE SIKKERHED MOD TMP FEJL ---
# Tjek om vi kører på HPC (hvis TMPDIR er sat)
if "TMPDIR" in os.environ:
    if os.environ["TMPDIR"].startswith("/home"):
        print(f"CRITICAL WARNING: TMPDIR is set to {os.environ['TMPDIR']} (Network Drive).")
        print("This WILL cause crashes. Please fix run_ensemble.sh to use /tmp/...")
        local_tmp = f"/tmp/{os.environ.get('USER', 'user')}/fallback_job"
        os.makedirs(local_tmp, exist_ok=True)
        os.environ["TMPDIR"] = local_tmp
        os.environ["JOBLIB_TEMP_FOLDER"] = local_tmp
        print(f"FORCED SWITCH to local disk: {local_tmp}")

print(f"Running with TMPDIR: {os.environ.get('TMPDIR', 'Not Set')}")

MODEL_SAVE_PATH = "models/ppo_ensemble" 
TEST_START_DATE = "2025-01-01"
VAL_START_DATE  = "2024-01-01"

def train_and_save_model(i, seed, df_t, df_v, prices_t, prices_v, model_save_path):
    import torch
    torch.set_num_threads(6) 
    
    print(f"--> Starter Model {i+1} (Seed: {seed}) på sin egen proces med 6 tråde...")
    model = trade.train_agent(
        train_df=df_t, 
        val_df=df_v, 
        raw_prices_train=prices_t,
        raw_prices_val=prices_v,
        seed=seed
    )
    save_path = os.path.join(model_save_path, f"model_seed_{seed}.zip")
    model.save(save_path)
    print(f"<-- Model {i+1} (Seed: {seed}) er FÆRDIG og gemt!")
    return save_path

class EnsembleModel:
    """
    Sharpe-vægtet Continuous Ensemble (Meta-Learner).
    Tager et vægtet gennemsnit af modellernes allokerings-outputs baseret på deres rullende Sharpe Ratio.
    """
    def __init__(self, models, window_size=50):
        self.models = models
        self.n_models = len(models)
        self.window_size = window_size
        
        # Buffer til at gemme hypotetiske afkast for hver model
        self.model_returns = [[] for _ in range(self.n_models)]
        self.last_target_positions = [0.0] * self.n_models
        
        print(f"Continuous Soft Voting Ensemble initialized med {self.n_models} modeller (Vindue: {window_size}).")

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        import numpy as np
        from scipy.special import softmax
        
        if state is None:
            state = [None] * self.n_models
        
        new_states = []
        
        # 1. Udregn dynamiske vægte baseret på Rullende Sharpe Ratio
        weights = np.ones(self.n_models) / self.n_models  # Default: ligelig fordeling
        
        if len(self.model_returns[0]) > 10: 
            sharpes = []
            for i in range(self.n_models):
                rets = np.array(self.model_returns[i])
                mean_ret = np.mean(rets)
                std_ret = np.std(rets) + 1e-9 # Undgå division med nul
                
                sharpe = mean_ret / std_ret
                sharpes.append(sharpe)
            
            temperature = 1.0 
            weights = softmax(np.array(sharpes) / temperature)

        # 2. Indhent float-forudsigelser fra hver model og vægt dem
        weighted_action = 0.0
        
        for i, model in enumerate(self.models):
            model_state = state[i]
            
            # Action her er nu et numpy array, f.eks. [0.45]
            action, next_state = model.predict(obs, state=model_state, episode_start=episode_start, deterministic=deterministic)
            new_states.append(next_state)
            
            action_val = float(action[0])
            
            # Gem modellens position til PnL track-record
            self.last_target_positions[i] = action_val
            
            # Vægt allokeringen
            weighted_action += weights[i] * action_val

        # 3. Formatér endelig handling til det format miljøet forventer (Numpy array, shape=(1,))
        final_action = np.array([np.clip(weighted_action, -1.0, 1.0)], dtype=np.float32)
        
        return final_action, new_states

    def update_performance(self, market_return):
        """
        Beregner hvad hver model VILLE have tjent med deres float-positioner.
        """
        for i in range(self.n_models):
            # Float-allokering * market return
            hypothetical_return = self.last_target_positions[i] * market_return
            self.model_returns[i].append(hypothetical_return)
            
            if len(self.model_returns[i]) > self.window_size:
                self.model_returns[i].pop(0)

    def save(self, path):
        pass

    def update_performance(self, market_return):
        """
        Beregner hvad hver model VILLE have tjent, og opdaterer deres track-record.
        """
        for i in range(self.n_models):
            # Hypotetisk afkast: (Modellens valg) * (Markedets bevægelse)
            hypothetical_return = self.last_target_positions[i] * market_return
            self.model_returns[i].append(hypothetical_return)
            
            # Fjern det ældste afkast, så vi kun kigger på de seneste 'window_size' steps
            if len(self.model_returns[i]) > self.window_size:
                self.model_returns[i].pop(0)

    def save(self, path):
        pass

def run_pipeline():
    print("\n--- 1. STARTING PIPELINE (HPC MODE - ENSEMBLE 9 MODELS) ---")

    print("Loading data and generating features...")
    df_full = dataloading.get_full_dataset()
    
    df_features_full = features.generate_alpha_pool(df_full)
    
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
    
    # Fjern NaNs fra træningssættet inden vi lærer fordelingerne
    X_train_clean = X_train.dropna().copy()
    
    # Opret pipelinen: Først klipper vi (Winsorizer), derefter skalerer vi
    prep_pipeline = Pipeline([
        ('winsorizer', TrainQuantileWinsorizer(lower_q=0.001, upper_q=0.999)),
        ('scaler', StandardScaler())
    ])
    
    # Vi kalder FIT KUN PÅ TRÆNINGSDATA. Dette fastlåser fraktiler og z-score gennemsnit.
    X_train_scaled_array = prep_pipeline.fit_transform(X_train_clean)
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train_clean.columns, index=X_train_clean.index)
    
    def process_split(df_feat, fitted_pipeline):
        # Her har vi fjernet det hardcodede .clip() og lader i stedet pipelinen 
        # udføre magien med de grænser, den lærte fra X_train.
        df_clean = df_feat.dropna().copy()
        data_scaled_array = fitted_pipeline.transform(df_clean)
        return pd.DataFrame(data_scaled_array, columns=df_clean.columns, index=df_clean.index)

    # Transformér Validation og Test med den trænede pipeline
    X_val_scaled  = process_split(X_val, prep_pipeline)
    X_test_scaled = process_split(X_test, prep_pipeline)
    
    print(f"Cleaned shapes -> Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # --- TRIN 4: FEATURE SELECTION (VARIANCE THRESHOLD) ---
    # ... (Resten af din pipeline fra TRIN 4 og frem forbliver uændret) ...
    
    print("\n--- 4. FEATURE SELECTION (KEEPING CONTEXT) ---")
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
    
    assert len(train_final) == len(prices_train_aligned), "CRITICAL: Train Features/Prices length mismatch!"
    assert len(val_final) == len(prices_val_aligned), "CRITICAL: Val Features/Prices length mismatch!"
    
    print(f"Selected {len(selected_cols)} features (dropped {len(X_train_scaled.columns) - len(selected_cols)} flat features).")

    print("\n--- 4.8. RUNNING HYPERPARAMETER TUNING (ONCE) ---")
    tune.run_tuning(
        train_feat=train_final,
        val_feat=val_final,
        train_prices=prices_train_aligned,
        val_prices=prices_val_aligned
    )

    print("\n--- 5. TRAINING ENSEMBLE (WITH BLOCK BOOTSTRAPPING) ---")
    
    ensemble_models = []
    n_models = 5
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    mp_context = multiprocessing.get_context('spawn')

    def apply_block_bootstrap(df_feat, df_prices, seed, drop_fraction=0.15):
        """ Dropper en tilfældig sammenhængende tidsblok for at gennemtvinge dekorrelation. """
        np.random.seed(seed)
        n_rows = len(df_feat)
        drop_size = int(n_rows * drop_fraction)
        
        # Undgå at droppe data helt i starten eller slutningen for stabilitet
        start_idx = np.random.randint(int(n_rows * 0.1), int(n_rows * 0.9) - drop_size)
        
        # Behold alt uden for blokken
        mask = np.ones(n_rows, dtype=bool)
        mask[start_idx : start_idx + drop_size] = False
        
        return df_feat.iloc[mask].copy(), df_prices.iloc[mask].copy()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_models, mp_context=mp_context) as executor:
        futures = []
        for i in range(n_models):
            seed = 42 + i
            
            # Sub-sample træningsdata per agent for at undgå Ensemble Collapse
            train_feat_bagged, prices_train_bagged = apply_block_bootstrap(train_final, prices_train_aligned, seed)
            
            futures.append(executor.submit(
                train_and_save_model, 
                i, seed, train_feat_bagged, val_final, prices_train_bagged, prices_val_aligned, MODEL_SAVE_PATH
            ))

        for future in concurrent.futures.as_completed(futures):
            future.result()  

    print("\nAlle modeller er færdigtrænet! Indlæser dem nu til backtest...")

    dummy_env = DummyVecEnv([lambda: trading_env.TradingEnv(val_final, prices_val_aligned)])

    for i in range(n_models):
        seed = 42 + i
        save_path = os.path.join(MODEL_SAVE_PATH, f"model_seed_{seed}.zip")
        loaded_model = RecurrentPPO.load(save_path, env=dummy_env)
        ensemble_models.append(loaded_model)

    ensemble_agent = EnsembleModel(ensemble_models)

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