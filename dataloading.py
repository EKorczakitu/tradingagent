import pandas as pd
import os

# Angiv stien til mappen, hvor dine csv-filer ligger
DATA_DIR = 'data'

def load_data(filename):
    """
    Indlæser en CSV fil med formatet: 
    Local time, Open, High, Low, Close, Volume
    Datoformat: 08.11.2016 00:00:00.000 GMT+0100
    """
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Kunne ikke finde filen: {file_path}")
    
    print(f"Indlæser {filename}...")
    
    try:
        # Indlæs csv
        df = pd.read_csv(file_path)
        
        # Rens kolonnenavne for eventuelle mellemrum
        df.columns = df.columns.str.strip()

        # Parse 'Local time' med det specifikke format
        # Format forklaring:
        # %d.%m.%Y    = 08.11.2016
        # %H:%M:%S.%f = 00:00:00.000
        # GMT%z       = GMT+0100 (Håndterer tidszonen)
        df['Local time'] = pd.to_datetime(
            df['Local time'], 
            format='%d.%m.%Y %H:%M:%S.%f GMT%z',
            utc=True
        )
        
        # Sæt index og sorter
        df.set_index('Local time', inplace=True)
        df.sort_index(inplace=True)
        df.index.name = 'Date' # Omdøb index til 'Date' for standardisering

        # Sørg for at numeriske kolonner er floats
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    except Exception as e:
        print(f"Fejl ved indlæsning af {filename}: {e}")
        return None

def get_novo_nordisk_split():
    """
    Hjælpefunktion der returnerer Train, Validation og Test sæt.
    """
    file_train = "NOVOB.DKDKK_Candlestick_1_Hour_BID_08.11.2016-31.12.2023.csv"
    file_val   = "NOVOB.DKDKK_Candlestick_1_Hour_BID_01.01.2024-31.12.2024.csv"
    file_test  = "NOVOB.DKDKK_Candlestick_1_Hour_BID_01.01.2025-31.12.2025.csv"

    print("--- Henter Train Data ---")
    df_train = load_data(file_train)
    
    print("--- Henter Validation Data ---")
    df_val = load_data(file_val)
    
    print("--- Henter Test Data ---")
    df_test = load_data(file_test)

    return df_train, df_val, df_test