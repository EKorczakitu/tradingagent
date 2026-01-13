import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_alpha_pool(input_df):
    """
    Genererer en "Alpha Pool" af features til DRL baseret på OHLCV data.
    """
    # 1. Lav en kopi for at undgå at ændre originalen
    df = input_df.copy()
    
    # Sørg for at indeks er datetime hvis muligt, ellers ignoreres dette
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass

    # --- KATEGORI 1: Returns & Stationarity ---
    
    # Log Returns (ln(P_t / P_{t-1}))
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Fractional Differentiation (En simpel implementering af FFDimplementation)
    # Bemærk: Dette er en tung beregning. Her bruges en differencing faktor (d) på 0.5 som eksempel.
    # For en DRL model er Log Returns ofte tilstrækkeligt som baseline, men her er logikken:
    def get_weights_ffd(d, thres):
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(series, d, thres=1e-5):
        # Håndterer manglende data før beregning
        series = series.dropna()
        w = get_weights_ffd(d, thres)
        width = len(w) - 1
        output = []
        # Loop for at påføre vægtene (kan være langsomt på meget store dataframes uden optimering)
        # Her bruger vi en rullende metode for klarhedens skyld
        for i in range(width, len(series)):
            val = np.dot(w.T, series.iloc[i-width:i+1])[0]
            output.append(val)
        return pd.Series(output, index=series.index[width:])

    # Vi anvender FracDiff på Close prisen (d=0.4 er ofte et godt startpunkt for finansiel tidsserie)
    df['frac_diff_close'] = frac_diff_ffd(df['Close'], d=0.4) 

    # --- KATEGORI 2: Momentum ---
    
    # RSI (Relative Strength Index) - Funktion
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi_7'] = calculate_rsi(df['Close'], 7)
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    df['rsi_21'] = calculate_rsi(df['Close'], 21)

    # ROC (Rate of Change)
    n = 10
    df['roc_10'] = ((df['Close'] - df['Close'].shift(n)) / df['Close'].shift(n)) * 100

    # Stochastic Oscillator
    stoch_k_period = 14
    lowest_low = df['Low'].rolling(window=stoch_k_period).min()
    highest_high = df['High'].rolling(window=stoch_k_period).max()
    df['stoch_k'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # --- KATEGORI 3: Trend ---
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = exp1 - exp2
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci_period = 20
    sma_tp = tp.rolling(window=cci_period).mean()
    mean_dev = tp.rolling(window=cci_period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['cci'] = (tp - sma_tp) / (0.015 * mean_dev)

    # ADX (Average Directional Index) - Lidt mere kompleks
    # Først True Range
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Directional Movement
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smooth (Wilder's Smoothing er standard for ADX, her bruger vi EWM for approksimation)
    adx_period = 14
    df['atr'] = df['tr'].ewm(alpha=1/adx_period, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=1/adx_period, adjust=False).mean()

    # Oprydning af midlertidige ADX kolonner
    df.drop(columns=['tr1', 'tr2', 'tr3', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'dx'], inplace=True)

    # --- KATEGORI 4: Volatility ---
    
    # ATR Normalized (ATR / Price)
    # Vi har allerede beregnet 'atr' i ADX sektionen
    df['atr_normalized'] = df['atr'] / df['Close']

    # Bollinger Band Width
    bb_period = 20
    bb_sma = df['Close'].rolling(window=bb_period).mean()
    bb_std = df['Close'].rolling(window=bb_period).std()
    upper_band = bb_sma + (2 * bb_std)
    lower_band = bb_sma - (2 * bb_std)
    df['bb_width'] = (upper_band - lower_band) / bb_sma

    # Historic Volatility (Rolling Std Dev af Log Returns)
    df['hist_vol_20'] = df['log_ret'].rolling(window=20).std()

    # --- KATEGORI 5: Volume & Microstructure ---
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # VWAP (Approximation) = (Typical Price * Volume).cumsum() / Volume.cumsum()
    # Bemærk: VWAP nulstilles normalt dagligt (intraday), men her laver vi en rullende eller kumulativ version.
    # For DRL er en rullende VWAP (f.eks. over 1 dag eller uge) ofte bedre end uendelig kumulativ.
    # Her laver vi en simpel kumulativ approksimation:
    vwap_tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['vwap'] = (vwap_tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # For at gøre VWAP stationær (model-venlig), beregner vi afstanden til VWAP:
    df['dist_to_vwap'] = (df['Close'] - df['vwap']) / df['vwap']

    # Bid-Ask Spread (hvis tilgængeligt)
    if 'Ask' in df.columns and 'Bid' in df.columns:
        df['spread'] = df['Ask'] - df['Bid']
        # Normalize spread (Spread / Close) for at gøre det sammenligneligt over tid
        df['spread_normalized'] = df['spread'] / df['Close']
    
    # --- RENSNING ---
    # Fjern de første rækker, der indeholder NaN pga. rolling windows (f.eks. MACD 26+9 perioder)
    df.dropna(inplace=True)

    return df

def normalize_features(input_df):
    """
    Standardiserer features (Z-score normalization).
    Gemmer scaler-objektet, så vi kan bruge det på ny data senere (vigtigt til live trading).
    """
    df = input_df.copy()
    
    # Vi vil kun normalisere features, ikke target (hvis du har det) eller datoer.
    # Antager at alt i df nu er numeriske features fra Phase 1.
    
    # Håndter uendelige værdier (kan opstå ved division med 0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    
    # Fit og transform
    df_scaled_array = scaler.fit_transform(df)
    
    # Skab en ny dataframe med de skalerede data og behold kolonnenavne og index
    df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns, index=df.index)
    
    return df_scaled, scaler



df_processed = generate_alpha_pool(df)
df_normalized, scaler_model = normalize_features(df_processed)
