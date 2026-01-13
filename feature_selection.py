from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

def feature_selection_funnel(input_df, method='xgboost', n_components_pca=0.95, top_k_features=20):
    """
    Phase 2: The Funnel.
    1. Stationarity Filter (ADF)
    2. Correlation Filter
    3. Dimensionality Reduction (PCA eller XGBoost Feature Importance)
    """
    df = input_df.copy()
    dropped_log = {} # Log til at se, hvad vi smider væk
    
    print(f"Startede med {len(df.columns)} features.")

    # --- TRIN 1: Stationarity Filter (ADF Test) ---
    non_stationary_cols = []
    print("Kører ADF Test...")
    
    for col in df.columns:
        try:
            # ADF test
            result = adfuller(df[col].values)
            p_value = result[1]
            
            # Hvis p-value > 0.05, kan vi ikke afvise null-hypotesen (data er ikke-stationær)
            if p_value > 0.05:
                non_stationary_cols.append(col)
        except Exception as e:
            # Hvis testen fejler (f.eks. konstant værdi), smid den væk
            non_stationary_cols.append(col)

    if non_stationary_cols:
        print(f"Fjerner {len(non_stationary_cols)} ikke-stationære features: {non_stationary_cols}")
        df.drop(columns=non_stationary_cols, inplace=True)
        dropped_log['non_stationary'] = non_stationary_cols
    else:
        print("Alle features bestod ADF testen (er stationære).")

    # --- TRIN 2: Correlation Filter ---
    print("Kører Correlation Filter...")
    # Beregn korrelationsmatrix (absolut værdi)
    corr_matrix = df.corr().abs()
    
    # Vælg øvre trekant af korrelationsmatricen
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features med korrelation > 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if to_drop:
        print(f"Fjerner {len(to_drop)} højt korrelerede features (dubletter): {to_drop}")
        df.drop(columns=to_drop, inplace=True)
        dropped_log['high_correlation'] = to_drop
    else:
        print("Ingen features overskred korrelationsgrænsen.")

    # --- TRIN 3: Dimensionality Reduction ---
    
    final_df = None
    
    if method == 'pca':
        print(f"Kører PCA (beholder {n_components_pca*100}% varians)...")
        pca = PCA(n_components=n_components_pca)
        principal_components = pca.fit_transform(df)
        
        # Lav kolonnenavne til PC'erne
        pc_columns = [f'PC_{i+1}' for i in range(principal_components.shape[1])]
        final_df = pd.DataFrame(data=principal_components, columns=pc_columns, index=df.index)
        
        print(f"PCA reducerede features fra {len(df.columns)} til {len(pc_columns)} komponenter.")
        
    elif method == 'xgboost':
        print("Kører XGBoost Feature Selection...")
        # Vi skal lave et simpelt 'target' for at træne XGBoost.
        # Target: 1 hvis næste Close er højere end nuværende, ellers 0 (Next Bar Up/Down)
        # Vi bruger df_processed (før normalisering) til at lave target, da vi har brug for priserne, 
        # men her antager vi at input_df allerede er features. 
        # Vi laver et 'dummy' target baseret på log_ret hvis det findes, ellers kan vi ikke køre superviseret feature selection.
        
        if 'log_ret' in df.columns:
            target = (df['log_ret'].shift(-1) > 0).astype(int)
        else:
            # Fallback: Hvis log_ret blev fjernet eller omdøbt, brug den første kolonne som proxy eller stop
            print("Advarsel: 'log_ret' mangler til target generering. Bruger første feature til simpel demo (dette bør rettes til rigtig target).")
            target = (df.iloc[:, 0].shift(-1) > 0).astype(int)

        # Fjern sidste række pga shift
        X = df.iloc[:-1]
        y = target.iloc[:-1]
        
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X, y)
        
        # Hent feature importance
        importance = model.feature_importances_
        feature_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
        
        # Vælg top K features
        top_features = feature_imp.head(top_k_features).index.tolist()
        final_df = df[top_features]
        
        print(f"XGBoost valgte top {top_k_features} features:")
        print(feature_imp.head(top_k_features))
        
    return final_df, dropped_log
