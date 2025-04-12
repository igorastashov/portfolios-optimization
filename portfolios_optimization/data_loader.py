import pandas as pd
import numpy as np
import os
import glob

# Data paths
DATA_DIR = "data"
MODELS_PREDICTION_DIR = os.path.join(DATA_DIR, "models_predictions")
MODEL_PREFIX = "model_"

def load_price_data():
    """
    Load and process asset price data from CSV files
    
    Returns:
        pd.DataFrame: DataFrame with price data for all assets
    """
    # Read the combined data file
    file_path = os.path.join(DATA_DIR, "data_compare_eda.csv")
    
    if os.path.exists(file_path):
        # Read the data with dates as index
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        
        # List of tickers
        tickers = ["BNBUSDT", "BTCUSDT", "CAKEUSDT", "ETHUSDT",
                  "LTCUSDT", "SOLUSDT", "STRKUSDT", "TONUSDT",
                  "USDCUSDT", "XRPUSDT", "PEPEUSDT",
                  "HBARUSDT", "APTUSDT", "LDOUSDT", "JUPUSDT"]
        
        # Rename columns to match tickers
        df.columns = tickers
        
        # Fill missing values and resample to hourly data
        df = df.fillna(method='ffill').resample('h').ffill()
        
        return df
    else:
        # If the combined file doesn't exist, create a placeholder DataFrame
        print(f"Warning: {file_path} not found. Creating placeholder data.")
        return pd.DataFrame()

def load_return_data():
    """
    Load and combine return series data from different models
    
    Returns:
        pd.DataFrame: DataFrame with return series for all models
    """
    result_df = pd.DataFrame()

    # Find all model return series files
    file_pattern = os.path.join(MODELS_PREDICTION_DIR, f"{MODEL_PREFIX}*_return_series.csv")
    filepaths = [f for f in glob.glob(file_pattern) if os.path.isfile(f)]

    if not filepaths:
        print("No model return series files found.")
        return result_df

    for filepath in filepaths:
        try:
            # Extract model name from filename
            filename = os.path.basename(filepath)
            model_name = filename[len(MODEL_PREFIX):-len("_return_series.csv")].upper()

            # Read data from file
            df_tmp = pd.read_csv(filepath)

            # Determine which column to use as date
            if 'date' in df_tmp.columns:
                date_column = 'date'
            elif 'Close time' in df_tmp.columns:
                date_column = 'Close time'
            elif df_tmp.columns[0].lower() in ['date', 'time', 'timestamp']:
                date_column = df_tmp.columns[0]
            else:
                # If no date column found, assume the first column is the date
                date_column = df_tmp.columns[0]

            # Determine which column contains the return values
            if 'daily_return' in df_tmp.columns:
                value_column = 'daily_return'
            elif len(df_tmp.columns) > 1:
                value_column = df_tmp.columns[1]
            else:
                print(f"File {filepath} does not contain necessary return data. Skipping.")
                continue

            # Convert date column to datetime and set as index
            df_tmp[date_column] = pd.to_datetime(df_tmp[date_column])
            df_tmp.set_index(date_column, inplace=True)

            # Rename the return column to the model name
            df_tmp = df_tmp[[value_column]].rename(columns={value_column: model_name})

            # Add data to the result DataFrame
            if result_df.empty:
                result_df = df_tmp
            else:
                result_df = result_df.join(df_tmp, how='outer')

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    # Fill missing values and format index
    result_df = result_df.fillna(0)
    result_df.index = pd.to_datetime(result_df.index).strftime('%Y-%m-%d')
    
    return result_df

def load_model_actions():
    """
    Load portfolio allocation data for different models
    
    Returns:
        dict: Dictionary with model names as keys and allocation DataFrames as values
    """
    model_actions = {}

    # Find all model action files
    file_pattern = os.path.join(MODELS_PREDICTION_DIR, f"{MODEL_PREFIX}*_actions.csv")
    filepaths = [f for f in glob.glob(file_pattern) if os.path.isfile(f)]

    if not filepaths:
        print("No model action files found.")
        return model_actions

    for filepath in filepaths:
        try:
            # Extract model name from filename
            filename = os.path.basename(filepath)
            model_name = filename[len(MODEL_PREFIX):-len("_actions.csv")].lower()

            # Read data from file
            actions_df = pd.read_csv(filepath)

            # Ensure 'date' column exists
            if 'date' not in actions_df.columns:
                print(f"File {filepath} does not contain a 'date' column. Skipping.")
                continue

            # Convert date column to datetime and set as index
            actions_df['date'] = pd.to_datetime(actions_df['date'])
            actions_df.set_index('date', inplace=True)

            # Add to dictionary
            model_actions[model_name] = actions_df

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    return model_actions 