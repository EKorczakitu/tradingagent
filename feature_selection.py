import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

def feature_selection_funnel(df_train, method='permutation', top_k_features=50):
    """
    Vælger de bedste features baseret på metoden.
    
    Args:
        df_train: Træningsdata (features + target implicit i logikken hvis nødvendigt)
                  Bemærk: Her antager vi df_train kun er features, eller vi skal generere et target.
                  I TradingAgent setup genererer vi ofte et midlertidigt target (f.eks. næste bars retning)
                  for at måle feature importance.
        method: 'rfe' eller 'permutation'
        top_k_features: Antal features vi vil beholde.
    """
    
    # 1. Opret et midlertidigt Target (Y) for at måle importance
    # Vi vil forudsige om prisen stiger eller falder i næste bar
    # (Dette matcher hvad PPO agenten ofte lærer implicit)
    X = df_train.copy()
    
    # Antag at 'log_ret' findes, ellers brug Close diff. 
    # Vi shifter -1 for at forudsige NÆSTE step.
    if 'log_ret' in X.columns:
        y_raw = X['log_ret'].shift(-1)
    else:
        # Fallback
        y_raw = X['Close'].pct_change().shift(-1)
        
    y = (y_raw > 0).astype(int) # 1 hvis op, 0 hvis ned
    
    # Fjern rækker med NaNs (pga. shift)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    # Sørg for at index matcher
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    print(f"Feature Selection running on {len(X)} rows with method: {method.upper()}")
    
    selected_features = []
    
    if method == 'rfe':
        # Den gamle metode (Hurtig, men ofte for lineær)
        model = LogisticRegression(solver='liblinear', penalty='l1', C=0.1)
        rfe = RFE(estimator=model, n_features_to_select=top_k_features, step=0.1)
        rfe.fit(X, y)
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask].tolist()
        
    elif method == 'permutation':
        # Den NYE metode (Bedre til AI/Non-lineær)
        # Vi bruger Random Forest da den fanger ikke-lineære sammenhænge (som PPO gør)
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            n_jobs=-1, 
            random_state=42
        )
        model.fit(X, y)
        
        # Beregn Permutation Importance
        # n_repeats=5 er et godt kompromis mellem fart og præcision
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        
        # Lav en DataFrame med scores
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': r.importances_mean
        }).sort_values(by='Importance', ascending=False)
        
        # Vælg toppen
        selected_features = importance_df.head(top_k_features)['Feature'].tolist()
        
        print("\nTop 5 Vigtigste Features (Permutation):")
        print(importance_df.head(5).to_string(index=False))

    else:
        # Fallback: Brug alle
        selected_features = X.columns.tolist()

    # Returner kun de valgte kolonner
    df_selected = df_train[selected_features].copy()
    dropped_log = [c for c in df_train.columns if c not in selected_features]
    
    return df_selected, dropped_log