import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

# --- OPTIMERET MATEMATIK (HPC SPEEDUP) ---

def get_weights_ffd(d, thres):
    w, k = [1.], 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-3):
    if len(series) == 0: return series
    series = series.dropna()
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    output = []
    for i in range(width, len(series)):
        val = np.dot(w.T, series.iloc[i-width:i+1])[0]
        output.append(val)
    return pd.Series(output, index=series.index[width:])

# --- HOVEDFUNKTIONER ---

def generate_alpha_pool(input_df):
    """
    Genererer Alpha Pool med Polars for ekstrem performance.
    Inkluderer Rullende Z-score standardisering.
    """
    df = input_df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except:
            pass

    dt_index = df.index
    
    # Fractional differentiation (Kræver pandas autoregressiv logik)
    frac_diff = frac_diff_ffd(df['Close'], d=0.8, thres=1e-3)
    
    # Forbered tids-data for Polars
    df['datetime'] = df.index
    pldf = pl.from_pandas(df)
    
    # ADVARSEL (Data-lækage risiko): Polars bevarer kausalitet, men sørg for at splitte tidsrækker korrekt.
    q = (
        pldf.lazy()
        .with_columns([
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_ret"),
            pl.col("Close").diff().alias("delta"),
        ])
        .with_columns([
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).rolling_mean(window_size=14).alias("gain_14"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).rolling_mean(window_size=14).alias("loss_14"),
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).rolling_mean(window_size=7).alias("gain_7"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).rolling_mean(window_size=7).alias("loss_7"),
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).rolling_mean(window_size=21).alias("gain_21"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).rolling_mean(window_size=21).alias("loss_21"),
            
            (pl.col("Close").pct_change(n=10) * 100).alias("roc_10"),
            pl.col("Low").rolling_min(window_size=14).alias("low_14"),
            pl.col("High").rolling_max(window_size=14).alias("high_14"),
            
            pl.col("Close").ewm_mean(span=12, adjust=False).alias("exp1"),
            pl.col("Close").ewm_mean(span=26, adjust=False).alias("exp2"),
            
            pl.col("Close").rolling_mean(window_size=20).alias("bb_sma"),
            pl.col("Close").rolling_std(window_size=20).alias("bb_std"),
            
            pl.col("Close").rolling_mean(window_size=50).alias("sma_50"),
            pl.col("Close").rolling_mean(window_size=200).alias("sma_200"),
            
            (2 * np.pi * pl.col("datetime").dt.hour() / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col("datetime").dt.weekday() / 7).sin().alias("day_sin"),
        ])
        .with_columns([
            (100 - (100 / (1 + (pl.col("gain_14") / pl.col("loss_14"))))).alias("rsi_14"),
            (100 - (100 / (1 + (pl.col("gain_7") / pl.col("loss_7"))))).alias("rsi_7"),
            (100 - (100 / (1 + (pl.col("gain_21") / pl.col("loss_21"))))).alias("rsi_21"),
            
            (100 * (pl.col("Close") - pl.col("low_14")) / (pl.col("high_14") - pl.col("low_14"))).alias("stoch_k"),
            (pl.col("exp1") - pl.col("exp2")).alias("macd_line"),
            
            (4 * pl.col("bb_std") / pl.col("bb_sma")).alias("bb_width"),
            pl.col("log_ret").rolling_std(window_size=20).alias("hist_vol_20"),
            
            ((pl.col("Close") / pl.col("sma_50")) - 1).alias("dist_to_sma50"),
            ((pl.col("Close") / pl.col("sma_200")) - 1).alias("dist_to_sma200"),
            ((pl.col("sma_50") / pl.col("sma_200")) - 1).alias("sma_cross_spread"),
            
            pl.col("Close").rolling_max(window_size=14).alias("roll_max"),
            (pl.col("Close").rolling_std(window_size=10) / pl.col("Close").rolling_std(window_size=100)).alias("vol_ratio"),
            
            pl.col("log_ret").shift(1).alias("log_ret_lag1"),
            pl.col("log_ret").shift(2).alias("log_ret_lag2"),
            pl.col("log_ret").shift(3).alias("log_ret_lag3"),
        ])
        .with_columns([
            pl.col("macd_line").ewm_mean(span=9, adjust=False).alias("macd_signal"),
            ((pl.col("Close") / pl.col("roll_max")) - 1).alias("rolling_drawdown"),
        ])
        .with_columns([
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist"),
        ])
    )
    
    result_df = q.collect().to_pandas()
    result_df.index = dt_index
    
    result_df['frac_diff_close'] = frac_diff
    
    # Komplekse features der er nemmest i pandas
    tr = np.maximum(
        result_df['High'] - result_df['Low'], 
        np.maximum(
            abs(result_df['High'] - result_df['Close'].shift(1)), 
            abs(result_df['Low'] - result_df['Close'].shift(1))
        )
    )
    result_df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    result_df['atr_normalized'] = result_df['atr'] / result_df['Close']
    
    result_df['obv_roc'] = (np.sign(result_df['Close'].diff()) * result_df['Volume']).fillna(0).cumsum().pct_change()
    vwap_tp = (result_df['High'] + result_df['Low'] + result_df['Close']) / 3
    pv_cum = (vwap_tp * result_df['Volume']).rolling(24).sum()
    v_cum = result_df['Volume'].rolling(24).sum().replace(0, np.nan)
    result_df['dist_to_vwap'] = (result_df['Close'] / (pv_cum / v_cum)) - 1
    
    change = result_df['Close'].diff(10).abs()
    volatility = result_df['Close'].diff().abs().rolling(10).sum()
    result_df['efficiency_ratio'] = change / volatility.replace(0, np.nan)
    
    hl_ratio = result_df['High'] / result_df['Low'].replace(0, np.nan)
    result_df['parkinson_vol'] = np.sqrt(0.361 * np.log(hl_ratio)**2)
    
    roll_mean = result_df['Close'].rolling(20).mean()
    roll_std = result_df['Close'].rolling(20).std()
    result_df['z_score'] = (result_df['Close'] - roll_mean) / roll_std.replace(0, np.nan)

    # Fjern midlertidige kolonner
    drop_cols = ['datetime', 'delta', 'gain_14', 'loss_14', 'gain_7', 'loss_7', 'gain_21', 'loss_21', 'low_14', 'high_14', 'exp1', 'exp2', 'bb_sma', 'bb_std', 'sma_50', 'sma_200', 'roll_max']
    result_df.drop(columns=[c for c in drop_cols if c in result_df.columns], inplace=True)
    
    # --- SOTA: ROLLING Z-SCORE SCALING PÅ ALLE FEATURES ---
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_scale = [c for c in numeric_cols if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    roll_window = 100
    rolling_means = result_df[features_to_scale].rolling(window=roll_window, min_periods=30).mean()
    rolling_stds = result_df[features_to_scale].rolling(window=roll_window, min_periods=30).std()
    
    # Standardiserer features i forhold til de seneste 100 tids-skridt for at modvirke distribution shift
    result_df[features_to_scale] = (result_df[features_to_scale] - rolling_means) / (rolling_stds + 1e-9)

    # Winsorize internt (ABSOLUT CLIP FOR AT UNDGÅ DATA LEAKAGE)
    # Da vi allerede har z-scoret features til lokalt mean=0, std=1,
    # kan vi robust klippe ved ekstreme standardafvigelser (f.eks. +/- 10) 
    # uden at kigge fremad i datasættet med globale quantiles.
    result_df[features_to_scale] = result_df[features_to_scale].clip(lower=-10.0, upper=10.0, axis=1)

    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df.dropna(inplace=True)
    
    return result_df

def normalize_features(input_df):
    """
    Udgået pga. rullende indlejret z-score
    """
    return input_df, None