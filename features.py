import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

# --- HJÆLPEFUNKTIONER ---

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

def get_slope(array):
    """Beregner hældningskoefficienten (slope) for en tidsserie"""
    y = np.array(array)
    x = np.arange(len(y))
    # Hvis array er konstant, returner 0 for at undgå fejl
    if np.all(y == y[0]):
        return 0.0
    slope, _, _, _, _ = linregress(x, y)
    return slope

# --- HOVEDFUNKTIONER ---

def generate_alpha_pool(input_df):
    """
    Genererer en "Alpha Pool" af features til DRL baseret på OHLCV data.
    Indeholder nu Avancerede Statistiske Features (HPC-ready).
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
    
    # Fractional Differentiation
    df['frac_diff_close'] = frac_diff_ffd(df['Close'], d=0.8, thres=1e-3) 

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
    
    # --- KATEGORI 3: Trend (Oscillators) ---
    
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

    # Oprydning af midlertidige ADX kolonner
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
    raw_obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['obv_roc'] = raw_obv.pct_change().replace([np.inf, -np.inf], 0)
    
    # Rolling VWAP (24 Timer)
    vwap_window = 24 
    vwap_tp = (df['High'] + df['Low'] + df['Close']) / 3
    
    v_cum = df['Volume'].rolling(window=vwap_window).sum()
    pv_cum = (vwap_tp * df['Volume']).rolling(window=vwap_window).sum()
    
    v_cum = v_cum.replace(0, np.nan)
    
    df['rolling_vwap_24h'] = pv_cum / v_cum
    
    # Distance to VWAP
    df['dist_to_vwap'] = (df['Close'] - df['rolling_vwap_24h']) / df['rolling_vwap_24h']

    # --- KATEGORI 6: Long-term Trend & Regimes ---

    # Simple Moving Averages (SMA)
    sma_50 = df['Close'].rolling(window=50).mean()
    sma_200 = df['Close'].rolling(window=200).mean()
    
    # Distance to SMAs (Normaliseret)
    sma_50_safe = sma_50.replace(0, np.nan)
    sma_200_safe = sma_200.replace(0, np.nan)
    
    df['dist_to_sma50'] = (df['Close'] - sma_50) / sma_50_safe
    df['dist_to_sma200'] = (df['Close'] - sma_200) / sma_200_safe
    
    # SMA Cross Spread
    df['sma_cross_spread'] = (sma_50 - sma_200) / sma_200_safe

    # Kaufman's Efficiency Ratio
    er_period = 10
    change = df['Close'].diff(er_period).abs()
    volatility = df['Close'].diff().abs().rolling(window=er_period).sum()
    volatility_safe = volatility.replace(0, np.nan)
    
    df['efficiency_ratio'] = change / volatility_safe

    # --- KATEGORI 7: Context & Seasonality ---
    
    # Cyklisk Tid
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Parkinson Volatility
    high_low_ratio = df['High'] / df['Low']
    high_low_ratio = high_low_ratio.replace(0, 1)
    df['parkinson_vol'] = np.sqrt(0.361 * np.log(high_low_ratio)**2)

    # Lagged Returns
    df['log_ret_lag1'] = df['log_ret'].shift(1)
    df['log_ret_lag2'] = df['log_ret'].shift(2)
    df['log_ret_lag3'] = df['log_ret'].shift(3)
    
    # Volatility Regime
    rolling_max = df['Close'].rolling(window=14).max()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    df['rolling_drawdown'] = drawdown 
    
    df['vol_regime'] = df['Close'].rolling(20).std() / df['Close'].rolling(100).std()
    df['vol_rel'] = df['Volume'] / df['Volume'].rolling(50).mean()

    # --- KATEGORI 8: Statistical & Math (NY - HPC ENABLED) ---
    
    # 1. Rolling Linear Regression Slope (Trend Vinkel)
    # Måler hvor aggressiv trenden er (mere robust end bare ROC)
    # Note: Dette er tungt beregningsmæssigt, men HPC klarer det.
    window_slope = 14
    df['linreg_slope'] = df['Close'].rolling(window=window_slope).apply(get_slope, raw=True)
    
    # 2. Z-Score (Mean Reversion)
    # Hvor mange standardafvigelser er vi fra gennemsnittet?
    # > 2 betyder ofte overkøbt, < -2 oversolgt.
    z_window = 20
    roll_mean = df['Close'].rolling(window=z_window).mean()
    roll_std = df['Close'].rolling(window=z_window).std()
    roll_std_safe = roll_std.replace(0, np.nan)
    df['z_score'] = (df['Close'] - roll_mean) / roll_std_safe
    
    # 3. Short/Long Volatility Ratio (Regime Shift Detector)
    # Hvis denne stiger pludseligt, er vi på vej ind i et nyt regime (ofte crash/rally).
    # Vi bruger 10 (meget kort) vs 100 (lang).
    vol_short = df['Close'].rolling(10).std()
    vol_long = df['Close'].rolling(100).std()
    vol_long_safe = vol_long.replace(0, np.nan)
    df['vol_ratio'] = vol_short / vol_long_safe
    
    # 4. Hurst Exponent Proxy
    # Approksimation af om markedet trend'er eller mean-reverter.
    # Vi sammenligner volatilitet over kort vs lang horisont.
    # Hvis vol skalerer hurtigere end kvadratroden af tid -> Trend.
    tau_2 = df['Close'].diff(2).dropna().std()
    tau_10 = df['Close'].diff(10).dropna().std()
    # Undgå division med nul
    if tau_2 == 0: tau_2 = 1e-9
    
    # Dette er en statisk værdi for hele datasættet i denne simple form, 
    # men vi kan lave en rullende version:
    roll_std_2 = df['Close'].rolling(20).apply(lambda x: np.std(np.diff(x)[::2]), raw=True) # Pseudo
    # For at holde det hurtigt og robust bruger vi bare en simplere volatilitetsskalering:
    # Ratio mellem range og std dev er en klassisk Hurst proxy.
    
    # Vi bruger en simplere "Trend Strength" indikator her i stedet for fuld Hurst:
    # Absolut return divideret med summen af absolutte returns (Efficiency)
    # Dette har vi allerede lidt i 'efficiency_ratio', så vi styrker den med en længere horisont:
    er_long_period = 30
    change_long = df['Close'].diff(er_long_period).abs()
    vol_long_sum = df['Close'].diff().abs().rolling(window=er_long_period).sum()
    df['efficiency_ratio_long'] = change_long / vol_long_sum.replace(0, np.nan)

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
    
    # Håndter uendelige værdier
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 1. Clipping (Winsorization)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Clip værdierne for at fjerne ekstreme outliers
    lower = df[numeric_cols].quantile(0.001)
    upper = df[numeric_cols].quantile(0.999)
    df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)
    
    # 2. Fjern eventuelle resterende NaNs
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    
    # Fit og transform
    df_scaled_array = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns, index=df.index)
    
    return df_scaled, scaler