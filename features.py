import numpy as np
import pandas as pd
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
    # Hurtig exit hvis serien er tom
    if len(series) == 0: return series
    series = series.dropna()
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    output = []
    # Note: For ekstrem speedup kunne dette gøres med np.convolve, 
    # men loopen her er ok for fractional diff da den kun køres én gang per kolonne.
    for i in range(width, len(series)):
        val = np.dot(w.T, series.iloc[i-width:i+1])[0]
        output.append(val)
    return pd.Series(output, index=series.index[width:])

def calculate_slope_fast(series, window=14):
    """
    Lynhurtig vektoriseret beregning af hældning (slope).
    Erstatter den langsomme scipy.linregress.
    Formel: (N * Sum(xy) - Sum(x)Sum(y)) / (N * Sum(xx) - Sum(x)^2)
    """
    y = series.values
    n = window
    x = np.arange(n)
    
    # Præ-kalkuler konstante summer for x
    sum_x = np.sum(x)
    sum_xx = np.sum(x**2)
    denom = n * sum_xx - sum_x**2
    
    # Rullende summer for y og xy
    # Vi bruger pandas rolling til at styre vinduet, men numpy til beregningen
    # rolling_sum_y er nem. rolling_sum_xy kræver lidt snilde.
    
    # Den hurtigste metode i pandas context uden at skrive ren numpy stride tricks:
    def linear_slope(y_window):
        return (n * np.sum(x * y_window) - sum_x * np.sum(y_window)) / denom

    # 'raw=True' er nøglen til performance her - sender numpy arrays i stedet for Series
    return series.rolling(window=window).apply(linear_slope, raw=True)

# --- HOVEDFUNKTIONER ---

def generate_alpha_pool(input_df):
    """
    Genererer Alpha Pool med optimeret kode.
    """
    df = input_df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except:
            pass

    # --- KATEGORI 1: Returns & Stationarity ---
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['frac_diff_close'] = frac_diff_ffd(df['Close'], d=0.8, thres=1e-3) 

    # --- KATEGORI 2: Momentum ---
    # RSI (Vectorized)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # RSI varianter
    for p in [7, 21]:
        g = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        l = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        df[f'rsi_{p}'] = 100 - (100 / (1 + (g/l)))

    # ROC
    df['roc_10'] = df['Close'].pct_change(10) * 100

    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14).replace(0, np.nan)

    # --- KATEGORI 3: Trend ---
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = exp1 - exp2
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # ADX / ATR
    df['tr'] = np.maximum(
        df['High'] - df['Low'], 
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)), 
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['atr_normalized'] = df['atr'] / df['Close']

    # --- KATEGORI 4: Volatility & Bollinger ---
    bb_sma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_width'] = (4 * bb_std) / bb_sma.replace(0, np.nan) # (Upper-Lower)/Mid = 4*std/mean
    df['hist_vol_20'] = df['log_ret'].rolling(20).std()

    # --- KATEGORI 5: Volume ---
    # OBV ROC
    df['obv_roc'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum().pct_change()
    
    # VWAP Distance
    vwap_tp = (df['High'] + df['Low'] + df['Close']) / 3
    pv_cum = (vwap_tp * df['Volume']).rolling(24).sum()
    v_cum = df['Volume'].rolling(24).sum().replace(0, np.nan)
    df['dist_to_vwap'] = (df['Close'] / (pv_cum / v_cum)) - 1

    # --- KATEGORI 6: Long Term Trend ---
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    df['dist_to_sma50'] = (df['Close'] / sma_50) - 1
    df['dist_to_sma200'] = (df['Close'] / sma_200) - 1
    df['sma_cross_spread'] = (sma_50 / sma_200) - 1

    # Efficiency Ratio
    change = df['Close'].diff(10).abs()
    volatility = df['Close'].diff().abs().rolling(10).sum()
    df['efficiency_ratio'] = change / volatility.replace(0, np.nan)

    # --- KATEGORI 7: Context ---
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    
    # Parkinson Vol
    hl_ratio = df['High'] / df['Low'].replace(0, np.nan)
    df['parkinson_vol'] = np.sqrt(0.361 * np.log(hl_ratio)**2)
    
    # Lagged Returns
    for i in [1, 2, 3]:
        df[f'log_ret_lag{i}'] = df['log_ret'].shift(i)

    # Volatility Regimes
    roll_max = df['Close'].rolling(14).max()
    df['rolling_drawdown'] = (df['Close'] / roll_max) - 1
    df['vol_ratio'] = df['Close'].rolling(10).std() / df['Close'].rolling(100).std()

    # --- KATEGORI 8: Statistical (OPTIMERET) ---
    
    # 1. Slope (Nu lynhurtig)
    df['linreg_slope'] = calculate_slope_fast(df['Close'], window=14)
    
    # 2. Z-Score
    roll_mean = df['Close'].rolling(20).mean()
    roll_std = df['Close'].rolling(20).std()
    df['z_score'] = (df['Close'] - roll_mean) / roll_std.replace(0, np.nan)

    # Rensning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def normalize_features(input_df):
    """
    Standard Z-score normalization with clipping
    """
    df = input_df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Winsorization (Clip outliers)
    lower = df[numeric_cols].quantile(0.001)
    upper = df[numeric_cols].quantile(0.999)
    df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)
    
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    df_scaled_array = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns, index=df.index)
    
    return df_scaled, scaler