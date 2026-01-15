from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def feature_selection_funnel(input_df, method='xgboost', n_components_pca=0.95, top_k_features=20):
    df = input_df.copy()
    dropped_log = {} 
    
    # 1. CRASH FIX: Save 'Close' for later target calculation
    if 'Close' in input_df.columns:
        close_price_reference = input_df['Close'].copy()
    else:
        close_price_reference = df.iloc[:, 0].copy() 

    print(f"Startede med {len(df.columns)} features.")

    # --- TRIN 1: Stationarity Filter (ADF Test) ---
    non_stationary_cols = []
    
    # FEATURES WE REFUSE TO DELETE (The "Gold" List)
    # We force these to stay because we know they are valuable, 
    # even if they fail the strict ADF math.
    WHITELIST = ['frac_diff_close', 'vol_regime', 'rsi_14']
    
    print("\n--- Kører ADF Test (Stationarity) ---")
    
    for col in df.columns:
        if col in WHITELIST:
            print(f"  [KEEP] {col} er whitelistet (skipper test)")
            continue
            
        try:
            # CRITICAL FIX: Drop NaNs/Infs before passing to ADF
            # This prevents the "Silent Killer" crash
            clean_series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_series) < 20: # Skip if empty
                non_stationary_cols.append(col)
                continue

            result = adfuller(clean_series.values)
            p_value = result[1]
            
            # Print status for key features so we can debug
            if 'frac' in col or 'close' in col.lower():
                print(f"  [TEST] {col}: p-value={p_value:.4f}")

            if p_value > 0.05:
                non_stationary_cols.append(col)
                
        except Exception as e:
            print(f"  [ERROR] Kunne ikke teste {col}: {e}")
            non_stationary_cols.append(col)

    if non_stationary_cols:
        print(f"Fjerner {len(non_stationary_cols)} features: {non_stationary_cols}")
        df.drop(columns=non_stationary_cols, inplace=True)
        dropped_log['non_stationary'] = non_stationary_cols
    else:
        print("Alle features bestod.")

    # --- TRIN 2: Correlation Filter ---
    print("\n--- Kører Correlation Filter ---")
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Don't drop whitelisted features even if correlated
    to_drop = [c for c in to_drop if c not in WHITELIST]
    
    if to_drop:
        print(f"Fjerner {len(to_drop)} dubletter: {to_drop}")
        df.drop(columns=to_drop, inplace=True)
        dropped_log['high_correlation'] = to_drop

    # --- TRIN 3: XGBoost Selection ---
    final_df = None
    
    if method == 'xgboost':
        print("\n--- Kører XGBoost Selection ---")
        LOOK_AHEAD = 6
        
        # Use the saved reference price
        target = (close_price_reference.shift(-LOOK_AHEAD) > close_price_reference).astype(int)

        X = df.iloc[:-LOOK_AHEAD]
        y = target.iloc[:-LOOK_AHEAD]
        
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X, y)
        
        feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Ensure Whitelisted features are included if they have >0 importance
        # Or just take top K
        top_features = feature_imp.head(top_k_features).index.tolist()
        
        # FORCE ADD WHITELIST if missing (Optional, but recommended)
        for w in WHITELIST:
            if w in df.columns and w not in top_features:
                print(f"  -> Redder {w} ind i Top Features (Force Add)")
                top_features.append(w)
        
        final_df = df[top_features]
        
        print(f"XGBoost Top Features:")
        print(feature_imp.head(10))
        
    return final_df, dropped_log