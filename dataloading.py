import pandas as pd
import os

DATA_DIR = 'data'

def load_data(filename):
    """Indlæser CSV og parser datoer korrekt"""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Advarsel: Filen {file_path} mangler.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        # Parse 'Local time' format: 08.11.2016 00:00:00.000 GMT+0100
        df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc=True)
        df.set_index('Local time', inplace=True)
        df.sort_index(inplace=True)
        df.index.name = 'Date'
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns: df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"Fejl ved indlæsning af {filename}: {e}")
        return None

def get_full_dataset():
    """
    Combines ALL history (2016-2025) into one DataFrame.
    This allows the Pipeline to do 'Walk-Forward' splitting.
    """
    files = [
        "NOVOB.DKDKK_Candlestick_1_Hour_BID_08.11.2016-31.12.2023.csv",
        "NOVOB.DKDKK_Candlestick_1_Hour_BID_01.01.2024-31.12.2024.csv",
        "NOVOB.DKDKK_Candlestick_1_Hour_BID_01.01.2025-31.12.2025.csv"
    ]
    
    dfs = []
    for f in files:
        data = load_data(f)
        if data is not None:
            dfs.append(data)
    
    if not dfs:
        raise ValueError("Ingen data fundet!")
        
    full_df = pd.concat(dfs)
    full_df.sort_index(inplace=True)
    
    # Fjern dubletter hvis filerne overlapper
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    print(f"Total History: {full_df.index[0]} -> {full_df.index[-1]}")
    return full_df