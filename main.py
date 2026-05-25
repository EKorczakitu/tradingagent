import os
import random
import numpy as np
import pandas as pd
import torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import SAC

# --- CUSTOM TRANSFORMER TIL AT FORHINDRE DATA LÆKAGE ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# TrainQuantileWinsorizer er fjernet, da vi nu bruger dynamisk rullende z-score i features.py

# --- HPC SIKKERHED OG OPTIMERING ---
# Tillad TensorFloat-32 for massiv matrix speedup på Ampere A100 GPU'er
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Tjek om vi kører på HPC (hvis TMPDIR er sat)
try:
    tmpdir = os.environ.get('TMPDIR')
    if tmpdir:
        print(f"Running with TMPDIR: {tmpdir}")
    else:
        print("Running with TMPDIR: Not Set")
except Exception as e:
    pass

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

def train_and_save_model(i, seed, df_t, df_v, prices_t, prices_v, model_save_path, gpu_id=0):
    import torch
    import gc
    torch.set_num_threads(1) 
    
    final_model_path = os.path.join(model_save_path, f"model_seed_{seed}.zip")
    checkpoint_dir = os.path.join(model_save_path, f"checkpoints_seed_{seed}")
    
    # Skip hvis modellen allerede er færdigtrænet fra et tidligere job
    if os.path.exists(final_model_path):
        print(f"<-- Model {i+1} (Seed: {seed}) allerede færdigtrænet. Springer over.")
        return final_model_path
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # CheckpointCallback: gem modeltilstand periodisk for at overleve SLURM timeouts
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=checkpoint_dir,
        name_prefix="sac_checkpoint"
    )
    
    # Find seneste checkpoint til resume efter afbrudt job
    resume_path = None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")] if os.path.exists(checkpoint_dir) else []
    if checkpoints:
        checkpoints.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))
        resume_path = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"--> [RESUME] Model {i+1} (Seed: {seed}) genoptages fra: {resume_path}")
    else:
        print(f"--> [START] Starter Model {i+1} (Seed: {seed}) på GPU {gpu_id}...")
    
    model = trade.train_agent(
        train_df=df_t, 
        val_df=df_v, 
        raw_prices_train=prices_t,
        raw_prices_val=prices_v,
        seed=seed,
        total_timesteps=15_000_000,
        gpu_id=gpu_id,
        callback=checkpoint_callback,
        resume_path=resume_path
    )
    model.save(final_model_path)
    
    # Ryd op i midlertidige checkpoints efter succesfuld træning
    try:
        shutil.rmtree(checkpoint_dir)
    except Exception:
        pass
    
    # --- MEMORY CLEANUP FOR HPC/SLURM ---
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print(f"<-- Model {i+1} (Seed: {seed}) er FÆRDIG og gemt!")
    return final_model_path

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
    print("\n--- 1. STARTING PIPELINE (HPC MODE - ENSEMBLE 21 MODELS) ---")

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
    print("\n--- 3. DATA CLEANING (SCALING ER NU INDLEJRET I FEATURES.PY) ---")
    
    X_train_scaled = X_train.dropna().copy()
    X_val_scaled  = X_val.dropna().copy()
    X_test_scaled = X_test.dropna().copy()
    
    print(f"Cleaned shapes -> Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # --- TRIN 4: FEATURE SELECTION (VARIANCE THRESHOLD) ---
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

    print("\n--- 4.8. HYPERPARAMETER TUNING ---")
    if os.path.exists("best_hyperparams.txt") and os.path.getsize("best_hyperparams.txt") > 10:
        print("  best_hyperparams.txt fundet. Springer tuning over og bruger eksisterende parametre.")
    else:
        print("  Ingen eksisterende parametre fundet. Starter tuning...")
        print("  Forventet Tuning Tid: ~3-5 timer på GPU (50 trials x 2 folds x 200k steps)")
        tune.run_tuning(
            train_feat=train_final,
            val_feat=val_final,
            train_prices=prices_train_aligned,
            val_prices=prices_val_aligned,
            n_trials=50,
            total_timesteps=250_000 
        )

    print("\n--- 5. TRAINING ENSEMBLE (WITH BLOCK BOOTSTRAPPING & CHECKPOINTING) ---")
    print("Kører 21 modeller sekventielt med periodisk checkpointing for SLURM-resume.")
    
    ensemble_models = []
    n_models = 21 # Ujævnt tal for at bryde 'ties' og maksimere ensemble styrke
    model_seeds = [42 + i for i in range(n_models)]
    
    num_gpus = max(1, torch.cuda.device_count())
    print(f"Opdaget {num_gpus} fysiske GPU'er til rådighed på noden.")
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp_context) as executor:
        futures = []
        # Opretter processer med maksimal load balanceret over GPU'er
        for i, seed in enumerate(model_seeds):
            gpu_assigned = i % num_gpus
            train_feat_bagged, prices_train_bagged = apply_block_bootstrap(train_final, prices_train_aligned, seed)
            
            futures.append(executor.submit(
                train_and_save_model, 
                i, seed, train_feat_bagged, val_final, prices_train_bagged, prices_val_aligned, MODEL_SAVE_PATH, gpu_assigned
            ))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[CRITICAL] En underproces fejlede: {e}")

    print("\nAlle modeller er færdigtrænet! Indlæser dem nu til backtest...")

    params = trade.get_optimal_params()
    window_size = params.get('window_size', 20)
    dummy_env = DummyVecEnv([lambda: trading_env.TradingEnv(val_final, prices_val_aligned)])
    dummy_env = VecFrameStack(dummy_env, n_stack=window_size)

    for i in range(n_models):
        seed = 42 + i
        save_path = os.path.join(MODEL_SAVE_PATH, f"model_seed_{seed}.zip")
        try:
            model = SAC.load(save_path, env=dummy_env)
            ensemble_models.append(model)
        except Exception as e:
            print(f"Failed to load model {seed}: {e}")

    ensemble_agent = EnsembleModel(ensemble_models)

    print("\n--- 6. BACKTESTING ENSEMBLE ---")
    
    params = trade.get_optimal_params()
    window_size = params.get('window_size', 20)
    
    env_val = DummyVecEnv([lambda: trading_env.TradingEnv(val_final, prices_val_aligned)])
    env_val = VecFrameStack(env_val, n_stack=window_size)
    
    env_test = DummyVecEnv([lambda: trading_env.TradingEnv(test_final, prices_test_aligned)])
    env_test = VecFrameStack(env_test, n_stack=window_size)
    
    print("\n>>> VALIDATION SET RESULTS (ENSEMBLE):")
    backtest.run_backtest_engine(env_val, ensemble_agent, title="Validation Ensemble 2024")
    
    print("\n>>> TEST SET RESULTS (OUT-OF-SAMPLE ENSEMBLE):")
    backtest.run_backtest_engine(env_test, ensemble_agent, title="Test Ensemble 2025")

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()