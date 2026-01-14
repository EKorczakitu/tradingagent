import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_alpha_pool(input_df):
    """
    Genererer en "Alpha Pool" af features til DRL baseret på OHLCV data.
    Indeholder sikkerhed mod division-med-nul og håndterer inaktive perioder.
    """
    df = input_df.copy()
    
    # Sørg for at indeks er datetime med UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except:
            pass

    # --- KATEGORI 1: Returns & Stationarity ---
    
    # Log Returns (Baseline feature)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Fractional Differentiation (Udkommenteret som før)
    # df['frac_diff_close'] = frac_diff_ffd(df['Close'], d=0.4, thres=1e-3) 

    # --- KATEGORI 2: Momentum ---
    
    # RSI
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi_7'] = calculate_rsi(df['Close'], 7)
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    df['rsi_21'] = calculate_rsi(df['Close'], 21)

    # ROC
    n = 10
    df['roc_10'] = ((df['Close'] - df['Close'].shift(n)) / df['Close'].shift(n)) * 100

    # Stochastic Oscillator
    stoch_k_period = 14
    lowest_low = df['Low'].rolling(window=stoch_k_period).min()
    highest_high = df['High'].rolling(window=stoch_k_period).max()
    
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan) 
    df['stoch_k'] = 100 * ((df['Close'] - lowest_low) / denominator)
    
    # --- KATEGORI 3: Trend ---
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = exp1 - exp2
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci_period = 20
    sma_tp = tp.rolling(window=cci_period).mean()
    mean_dev = tp.rolling(window=cci_period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    
    mean_dev = mean_dev.replace(0, np.nan)
    df['cci'] = (tp - sma_tp) / (0.015 * mean_dev)

    # ADX
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    adx_period = 14
    df['atr'] = df['tr'].ewm(alpha=1/adx_period, adjust=False).mean()
    
    atr_safe = df['atr'].replace(0, np.nan)
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / atr_safe)
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / atr_safe)
    
    sum_di = df['plus_di'] + df['minus_di']
    sum_di = sum_di.replace(0, np.nan)
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / sum_di
    df['adx'] = df['dx'].ewm(alpha=1/adx_period, adjust=False).mean()

    # Oprydning
    df.drop(columns=['tr1', 'tr2', 'tr3', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'dx'], inplace=True)

    # --- KATEGORI 4: Volatility ---
    
    df['atr_normalized'] = df['atr'] / df['Close']

    # Bollinger Band Width
    bb_period = 20
    bb_sma = df['Close'].rolling(window=bb_period).mean()
    bb_std = df['Close'].rolling(window=bb_period).std()
    upper_band = bb_sma + (2 * bb_std)
    lower_band = bb_sma - (2 * bb_std)
    
    bb_sma_safe = bb_sma.replace(0, np.nan)
    df['bb_width'] = (upper_band - lower_band) / bb_sma_safe

    # Historic Volatility
    df['hist_vol_20'] = df['log_ret'].rolling(window=20).std()

    # --- KATEGORI 5: Volume & Microstructure ---
    
    # OBV
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Rolling VWAP (24 Timer) - Omdøbt for klarhed
    vwap_window = 24 
    vwap_tp = (df['High'] + df['Low'] + df['Close']) / 3
    
    v_cum = df['Volume'].rolling(window=vwap_window).sum()
    pv_cum = (vwap_tp * df['Volume']).rolling(window=vwap_window).sum()
    
    v_cum = v_cum.replace(0, np.nan)
    
    df['rolling_vwap_24h'] = pv_cum / v_cum
    
    # Distance to VWAP
    df['dist_to_vwap'] = (df['Close'] - df['rolling_vwap_24h']) / df['rolling_vwap_24h']
    
    # --- KATEGORI 6: Others ---
    
    # Efficiency Ratio (ER)
    # 1.0 = Perfect Trend (Price went straight up)
    # 0.0 = Pure Noise (Price bounced but went nowhere)
    er_period = 10
    change = df['Close'].diff(er_period).abs()
    volatility = df['Close'].diff().abs().rolling(window=er_period).sum()
    
    # Safe division
    volatility = volatility.replace(0, np.nan)
    df['efficiency_ratio'] = change / volatility

    # --- RENSNING ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

def normalize_features(input_df):
    """
    Standardiserer features (Z-score normalization).
    Bruger 'Clipping' for at håndtere outliers i stedet for at slette rækker.
    """
    df = input_df.copy()
    
    # Håndter uendelige værdier ved at erstatte med NaN først
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 1. Clipping (Winsorization) - Fix for "Inf/Outlier" feedback
    # Vi sætter en grænse ved 0.1% og 99.9% fraktilen for at fjerne ekstreme outliers
    # der kan ødelægge træningen, uden at slette data.
    # Vi beregner quantiles kun på numeriske kolonner
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Clip værdierne
    lower = df[numeric_cols].quantile(0.001)
    upper = df[numeric_cols].quantile(0.999)
    df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)
    
    # 2. Fjern eventuelle resterende NaNs (burde være få efter clipping)
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    
    # Fit og transform
    df_scaled_array = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns, index=df.index)
    
    return df_scaled, scaler

# Hjælpefunktioner til FracDiff (uændret)
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
    series = series.dropna()
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    output = []
    for i in range(width, len(series)):
        val = np.dot(w.T, series.iloc[i-width:i+1])[0]
        output.append(val)
    return pd.Series(output, index=series.index[width:])