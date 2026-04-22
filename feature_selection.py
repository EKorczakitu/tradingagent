import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

def feature_selection_funnel(df_train, method='permutation', top_k_features=50, purge_gap=100):
    """
    Vælger de bedste features baseret på metoden.
    Inkluderer Purged Time-Series Cross-Validation for finansiel robusthed.
    
    Args:
        df_train: Træningsdata (features)
        method: 'rfe' eller 'permutation'
        top_k_features: Antal features vi vil beholde.
        purge_gap: Antal steps der skal "purges" mellem train og test (typisk din max lookback periode).
    """
    
    # 1. Opret et midlertidigt Target (Y) for at måle importance
    X = df_train.copy()
    
    if 'log_ret' in X.columns:
        y_raw = X['log_ret'].shift(-1)
    else:
        y_raw = X['Close'].pct_change().shift(-1)
        
    y = (y_raw > 0).astype(int) 
    
    # Fjern rækker med NaNs
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    # Sørg for at index matcher
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    print(f"Feature Selection running on {len(X)} rows with method: {method.upper()}")
    
    selected_features = []
    
    if method == 'rfe':
        model = LogisticRegression(solver='liblinear', penalty='l1', C=0.1)
        rfe = RFE(estimator=model, n_features_to_select=top_k_features, step=0.1)
        rfe.fit(X, y)
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask].tolist()
        
    elif method == 'permutation':
        # --- NYT: PURGED TIME-SERIES CROSS VALIDATION ---
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=purge_gap)
        
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            n_jobs=-1, 
            random_state=42
        )
        
        # Array til at samle gennemsnitlig importance på tværs af folds
        feature_importances = np.zeros(X.shape[1])
        
        print(f"\nStarter Purged Time-Series CV (n_splits={n_splits}, gap={purge_gap})...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Træn KUN på det rensede, fortidige data
            model.fit(X_train_fold, y_train_fold)
            
            # Mål VIGTIGHEDEN på det rene fremtidige test-sæt
            # Vi bruger færre repeats for hastighed, da vi kompenserer via n_splits
            r = permutation_importance(
                model, X_test_fold, y_test_fold, 
                n_repeats=3, random_state=42, n_jobs=-1
            )
            
            feature_importances += r.importances_mean
            
        # Divider med antallet af folds for at få gennemsnittet
        feature_importances /= float(n_splits)
        
        # Lav en DataFrame med scores
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        # Vælg toppen
        selected_features = importance_df.head(top_k_features)['Feature'].tolist()
        
        print("\nTop 5 Vigtigste Features (Purged Permutation):")
        print(importance_df.head(5).to_string(index=False))

    else:
        # Fallback: Brug alle
        selected_features = X.columns.tolist()

    # Returner kun de valgte kolonner
    df_selected = df_train[selected_features].copy()
    dropped_log = [c for c in df_train.columns if c not in selected_features]
    
    return df_selected, dropped_log