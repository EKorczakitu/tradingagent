from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def feature_selection_funnel(input_df, method='rfe', n_components_pca=0.95, top_k_features=50):
    """
    Avanceret Feature Selection Funnel til HPC.
    
    Methods:
      - 'rfe': Recursive Feature Elimination (Langsom, men meget præcis).
      - 'permutation': Permutation Importance (Robust mod overfitting).
      - 'xgboost': Standard feature importance (Hurtig baseline).
    """
    df = input_df.copy()
    dropped_log = {} 
    
    # 1. CRASH FIX: Save 'Close' for later target calculation
    if 'Close' in input_df.columns:
        close_price_reference = input_df['Close'].copy()
    else:
        # Fallback hvis Close er blevet scaleret væk eller omdøbt
        close_price_reference = df.iloc[:, 0].copy() 

    print(f"Startede med {len(df.columns)} features.")

    # --- TRIN 1: Stationarity Filter (ADF Test) ---
    non_stationary_cols = []
    
    # FEATURES WE REFUSE TO DELETE (The "Gold" List)
    WHITELIST = ['frac_diff_close', 'vol_regime', 'rsi_14', 'z_score', 'linreg_slope']
    
    print("\n--- Kører ADF Test (Stationarity) ---")
    
    for col in df.columns:
        if col in WHITELIST:
            print(f"  [KEEP] {col} er whitelistet (skipper test)")
            continue
            
        try:
            # CRITICAL FIX: Drop NaNs/Infs before passing to ADF
            clean_series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_series) < 20: # Skip if empty
                non_stationary_cols.append(col)
                continue

            result = adfuller(clean_series.values)
            p_value = result[1]
            
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

    # --- TRIN 3: Model-Based Selection (HPC POWER) ---
    final_df = None
    
    # Forbered data til model
    LOOK_AHEAD = 6
    target = (close_price_reference.shift(-LOOK_AHEAD) > close_price_reference).astype(int)
    
    # Juster X og y længder
    X = df.iloc[:-LOOK_AHEAD]
    y = target.iloc[:-LOOK_AHEAD]
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        eval_metric='logloss',
        tree_method='hist', # Hurtigere på store datasæt
        device='cuda' if method != 'rfe' else 'cpu' # RFE kan drille med GPU contexts i loops
    )

    selected_features = []

    if method == 'rfe':
        print(f"\n--- Kører Recursive Feature Elimination (RFE) til {top_k_features} features ---")
        print("Dette kan tage tid, men giver det bedste resultat...")
        
        # RFE træner modellen, fjerner dårligste feature, og gentager.
        rfe = RFE(estimator=model, n_features_to_select=top_k_features, step=1)
        rfe.fit(X, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        print(f"RFE valgte {len(selected_features)} features.")
        
    elif method == 'permutation':
        print(f"\n--- Kører Permutation Importance ---")
        
        # Fit modellen først
        model.fit(X, y)
        
        # Kør permutation (kræver valideringsdata for at være ægte, men vi bruger training her for selection)
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        
        # Sorter features baseret på 'importances_mean'
        perm_sorted_idx = result.importances_mean.argsort()[::-1]
        
        # Vælg top K
        top_indices = perm_sorted_idx[:top_k_features]
        selected_features = X.columns[top_indices].tolist()
        
        print("Top 5 Permutation Features:")
        for i in top_indices[:5]:
            print(f"{X.columns[i]}: {result.importances_mean[i]:.4f}")

    else: # method == 'xgboost' (Legacy fast mode)
        print("\n--- Kører Standard XGBoost Selection ---")
        model.fit(X, y)
        feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        selected_features = feature_imp.head(top_k_features).index.tolist()
        
        print("XGBoost Top Features:")
        print(feature_imp.head(10))

    # --- FINAL: Whitelist Check & Construction ---
    
    # FORCE ADD WHITELIST hvis de mangler (Sikkerhedsnet)
    for w in WHITELIST:
        if w in df.columns and w not in selected_features:
            print(f"  -> Redder {w} ind i Top Features (Force Add)")
            selected_features.append(w)
    
    # Fjern eventuelle dubletter i listen
    selected_features = list(set(selected_features))
    
    final_df = df[selected_features]
    
    return final_df, dropped_log