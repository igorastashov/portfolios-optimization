import pandas as pd
import numpy as np
import os
import glob
import streamlit as st # For secrets
from binance import Client # For API calls
from datetime import datetime, timedelta
from tqdm import tqdm # For progress tracking (optional, might use st.progress)
import time
import plotly.graph_objects as go
import plotly.express as px
import io # Add io import

# Data paths
DATA_DIR = "data"
MODELS_PREDICTION_DIR = os.path.join(DATA_DIR, "models_predictions")
MODEL_PREFIX = "model_"

# --- Binance Client Initialization --- #
@st.cache_resource # Cache the client object
def get_binance_client():
    """Initializes and returns the Binance client using secrets."""
    try:
        api_key = st.secrets["binance"]["api_key"]
        api_secret = st.secrets["binance"]["api_secret"]
        client = Client(api_key, api_secret)
        client.ping() # Check connection
        print("Binance Client Initialized Successfully.")
        return client
    except KeyError:
        st.error("Binance API keys not found in Streamlit secrets ([binance].api_key, [binance].api_secret).")
        return None
    except Exception as e:
        st.error(f"Failed to connect to Binance API: {e}")
        return None

def load_price_data():
    """
    Load and process asset price data from the combined CSV file.
    Assumes the CSV has dates as the first column (index) and tickers as column headers.

    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex and price data for assets.
    """
    file_path = os.path.join(DATA_DIR, "data_compare_eda.csv")

    if os.path.exists(file_path):
        try:
            # Read the data with dates as index
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)

            # Sort by index (date)
            df.sort_index(inplace=True)

            # Remove duplicate index entries (if any), keeping the first
            df = df[~df.index.duplicated(keep='first')]

            # Optional: Clean stablecoin price (Uncomment and adapt if needed)
            # if 'USDCUSDT' in df.columns:
            #     close_orig = df['USDCUSDT'].copy()
            #     df['USDCUSDT'] = np.where((df['USDCUSDT'] > 1.02) | (df['USDCUSDT'] < 0.98), 1.0, df['USDCUSDT'])
            #     # If other OHLC columns exist and need cleaning:
            #     # mask_changed = (df['USDCUSDT'] == 1.0) & (close_orig != 1.0)
            #     # df.loc[mask_changed, ['relevant_open', 'relevant_high', 'relevant_low']] = 1.0

            # Resample to hourly frequency and forward fill (if needed)
            try:
                df = df.resample('h').ffill()
                print(f"Data resampled to hourly frequency. Shape: {df.shape}")
            except TypeError as e:
                print(f"Warning: Could not resample data to hourly. Check index type. Error: {e}")
                # Proceed without resampling if it fails

            return df

        except KeyError as e:
             print(f"Error reading {file_path}. Missing expected column/index? KeyError: {e}")
             return pd.DataFrame()
        except Exception as e:
             print(f"Error processing {file_path}: {e}")
             return pd.DataFrame()
    else:
        print(f"Warning: {file_path} not found. Cannot load price data.")
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

# --- NEW Data Update Functions --- #

def get_last_date_from_csv(file_path, buffer_size=8192):
    """
    Gets the last timestamp from the first column of a CSV file by reading the end of the file.
    Assumes the timestamp is in milliseconds.
    """
    try:
        with open(file_path, 'rb') as f:
            # Go near the end of the file
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            if file_size == 0:
                print(f"File is empty: {file_path}")
                return None

            # Seek back a buffer size (or from the start if file is smaller)
            seek_pos = max(0, file_size - buffer_size)
            f.seek(seek_pos)

            # Read the end chunk
            end_chunk = f.read().decode('utf-8', errors='ignore')

            # Find the last newline character to get the last full line
            last_newline_pos = end_chunk.rfind('\n', 0, -1) # Exclude potential trailing newline
            if last_newline_pos == -1 and seek_pos == 0: # No newline found, might be single line file
                 last_line = end_chunk
            elif last_newline_pos != -1:
                 last_line = end_chunk[last_newline_pos + 1:].strip()
            else: # Newline not found in buffer, means last line is incomplete? Unlikely but possible.
                 # Try reading a larger chunk or assume the buffer contains the start?
                 # For simplicity, let's try the last non-empty line found in the buffer
                 lines = end_chunk.strip().split('\n')
                 last_line = lines[-1] if lines else ""

            if not last_line or last_line.isspace():
                 print(f"Could not find a valid last line in the buffer for {file_path}")
                 # Maybe the header is the only line? Check if file_size implies only header
                 # This part can be complex, returning None for now.
                 return None

            # Assume timestamp is the first column
            try:
                last_timestamp_str = last_line.split(',')[0]
            except IndexError:
                 print(f"Could not split last line '{last_line}' by comma in {file_path}")
                 return None

            try:
                # --- ROBUST PARSING LOGIC --- 
                try:
                    # First, try parsing as millisecond timestamp
                    last_timestamp_ms = int(float(last_timestamp_str))
                    last_date = pd.Timestamp(last_timestamp_ms, unit='ms', tz='UTC')
                except ValueError:
                    # If it fails, try parsing as datetime string
                    last_date = pd.to_datetime(last_timestamp_str)

                # Ensure output is naive UTC representation
                if last_date.tzinfo is not None:
                    last_date = last_date.tz_convert('UTC').tz_localize(None)
                else:
                    # If already naive, assume it represents UTC
                    pass
                    
                print(f"Found last date {last_date} in {file_path}")
                return last_date
                # --- END ROBUST PARSING LOGIC --- 

            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                print(f"Error parsing timestamp value '{last_timestamp_str}' from last line '{last_line}' in {file_path}: {e}")
                return None

    except FileNotFoundError:
        print(f"File not found for last date check: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading last date from {file_path} using tail method: {e}")
        return None

def update_single_asset_data(client, symbol, interval, file_path):
    """Fetches new data for a single asset and appends it to its CSV file."""
    last_date = get_last_date_from_csv(file_path)
    if last_date is None:
        # If no last date, perhaps start fresh? For now, assume we need a base file.
        print(f"Skipping {symbol}: Could not determine last date from {file_path}. Base file might be missing or invalid.")
        return False, f"Missing/invalid base file for {symbol}"

    # Add a small delta (e.g., 1 millisecond) to start after the last record
    start_dt = last_date + timedelta(milliseconds=1)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S") # Binance client needs string
    end_str = None # Fetch up to current time

    print(f"  Fetching new data for {symbol} from {start_str}...")
    try:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str
        )

        if not klines:
            print(f"  No new data found for {symbol}.")
            return True, f"No new data for {symbol}"

        # Convert to DataFrame
        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        new_data = pd.DataFrame(klines, columns=columns)

        # --- Convert timestamp before saving --- 
        # Convert Open time from ms string/int to datetime objects
        new_data['Open time'] = pd.to_datetime(new_data['Open time'], unit='ms')
        # Also convert Close time for consistency if needed later
        new_data['Close time'] = pd.to_datetime(new_data['Close time'], unit='ms')
        # --- End Conversion --- 

        # Important: Don't drop first row blindly, check timestamps
        # Compare the 'Open time' of the first new record with the last saved time
        # Use the converted datetime for comparison
        first_new_time = new_data.iloc[0]['Open time'] 

        # Make sure last_date is timezone-naive for comparison if first_new_time is naive
        if first_new_time.tzinfo is None and last_date.tzinfo is not None:
            last_date_compare = last_date.tz_localize(None)
        elif first_new_time.tzinfo is not None and last_date.tzinfo is None:
            # This case is less likely if last_date logic is correct, but handle anyway
            # Assuming last_date represents UTC
            last_date_compare = last_date.tz_localize('UTC').tz_convert(first_new_time.tzinfo)
        else:
            last_date_compare = last_date

        if first_new_time <= last_date_compare:
             # Filter based on datetime comparison
             new_data = new_data[new_data['Open time'] > last_date_compare]

        if new_data.empty:
            print(f"  No new, non-overlapping data found for {symbol}.")
            return True, f"No new data for {symbol} (after overlap check)"

        # Append new data without header
        # pd.to_csv will now write the datetime objects in standard string format
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            # Specify float_format to avoid potential scientific notation for prices/volumes
            new_data.to_csv(f, header=False, index=False, float_format='%.8f') 

        print(f"  Successfully appended {len(new_data)} new rows for {symbol}.")
        return True, f"Appended {len(new_data)} rows for {symbol}"

    except Exception as e:
        print(f"  ERROR fetching/appending data for {symbol}: {e}")
        return False, f"Error for {symbol}: {e}"

def update_all_asset_data(progress_bar=None):
    """Updates all *_hourly_data.csv files in the DATA_DIR."""
    client = get_binance_client()
    if client is None:
        return False, "Failed to initialize Binance client."

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_hourly_data.csv")]
    if not csv_files:
        return False, f"No *_hourly_data.csv files found in {DATA_DIR}"

    num_files = len(csv_files)
    results = {"success": [], "failed": [], "no_new": []}
    interval = Client.KLINE_INTERVAL_1HOUR

    for i, file_name in enumerate(csv_files):
        symbol = file_name.replace("_hourly_data.csv", "")
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"Processing {i+1}/{num_files}: {symbol}")
        success, message = update_single_asset_data(client, symbol, interval, file_path)

        if success:
            if "Appended" in message:
                results["success"].append(symbol)
            else:
                results["no_new"].append(symbol)
        else:
            results["failed"].append(f"{symbol}: {message}")

        if progress_bar:
            progress_bar.progress((i + 1) / num_files, text=f"Updating {symbol}...")
        time.sleep(0.2) # Small delay to avoid hitting API rate limits too hard

    summary_message = f"Update complete. Success: {len(results['success'])}, No new data: {len(results['no_new'])}, Failed: {len(results['failed'])}."
    if results['failed']:
        summary_message += f" Failures: {'; '.join(results['failed'])}"
        return False, summary_message
    else:
        return True, summary_message

# --- NEW Function to Create Combined Data --- #

def create_combined_data(output_filename="data_compare_eda.csv"):
    """Reads individual asset CSVs and creates a combined file with close prices."""
    print(f"Creating/updating combined file: {output_filename}")
    combined_data = pd.DataFrame()
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_hourly_data.csv")]

    if not csv_files:
        print(f"No source CSV files found in {DATA_DIR} to create combined data.")
        return False, "No source CSVs found."

    asset_list = [] # Keep track of successfully processed assets
    for file_name in tqdm(csv_files, desc="Combining data", unit="file"):
        symbol = file_name.replace("_hourly_data.csv", "")
        file_path = os.path.join(DATA_DIR, file_name)
        try:
            # Read 'Open time' as string initially to handle mixed formats
            df = pd.read_csv(file_path, dtype={'Open time': str})

            # --- ROBUST DATE PARSING for combined data --- 
            # Try parsing as milliseconds first
            parsed_ms = pd.to_datetime(df['Open time'], unit='ms', errors='coerce')
            # Try parsing remaining as strings
            parsed_str = pd.to_datetime(df.loc[parsed_ms.isna(), 'Open time'], errors='coerce')
            # Combine results
            df['date'] = parsed_ms.fillna(parsed_str)
            # Drop rows where parsing failed completely
            df.dropna(subset=['date'], inplace=True)
            # --- END ROBUST DATE PARSING --- 

            # Keep only date and Close price
            df = df[['date', 'Close']].copy()
            df.rename(columns={'Close': symbol}, inplace=True)
            df.set_index('date', inplace=True)

            if combined_data.empty:
                combined_data = df
            else:
                combined_data = combined_data.join(df, how='outer')
            asset_list.append(symbol)

        except Exception as e:
            print(f"Error processing {file_name} for combining: {e}")

    if not combined_data.empty:
        print("Filling NaNs in combined data...")
        combined_data.sort_index(inplace=True)
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        # Forward fill is generally preferred for price data
        combined_data.ffill(inplace=True)
        # Optional: Backward fill for any remaining NaNs at the beginning
        combined_data.bfill(inplace=True)

        output_path = os.path.join(DATA_DIR, output_filename)
        try:
            combined_data.to_csv(output_path)
            print(f"Combined data saved to {output_path}. Assets included: {len(asset_list)}")
            return True, f"Combined data saved. Assets: {len(asset_list)}"
        except Exception as e:
            print(f"Error saving combined data to {output_path}: {e}")
            return False, f"Error saving combined file: {e}"
    else:
        print("Failed to create combined data - no valid individual files processed.")
        return False, "Failed to create combined data."

# --- NEW Plotting Functions (Plotly) --- #

def generate_normalized_plot(combined_df, days=180, stablecoin='USDCUSDT'): # Default to 180 days
    """
    Generates a Plotly chart of normalized prices for the last N days or all time.

    Args:
        combined_df (pd.DataFrame): DataFrame with DatetimeIndex and asset prices as columns.
        days (int | None): Number of past days to display. If None, display all available data.
        stablecoin (str): The ticker of the stablecoin to exclude from normalization.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object. Returns empty Figure on error.
    """
    fig = go.Figure()
    fig.update_layout(template="plotly_dark") # Apply dark theme

    if combined_df.empty or not isinstance(combined_df.index, pd.DatetimeIndex):
        print("Error generating normalized plot: Input DataFrame is empty or has wrong index type.")
        return fig

    try:
        # Select data based on 'days' parameter
        if days is not None:
            # Calculate records needed (days * 24 hours for hourly data)
            num_records = days * 24
            recent_data = combined_df.tail(num_records)
            plot_title = f"Нормализованные цены активов (последние {days} дней)"
            if recent_data.empty:
                print(f"Error generating normalized plot: No data found for the last {days} days.")
                return fig
        else:
            recent_data = combined_df # Use all data
            plot_title = "Нормализованные цены активов (все время)"
            if recent_data.empty:
                 print(f"Error generating normalized plot: DataFrame is empty.")
                 return fig

        # Exclude stablecoin if present
        assets_to_normalize = recent_data.columns.tolist()
        if stablecoin in assets_to_normalize:
            assets_to_normalize.remove(stablecoin)

        if not assets_to_normalize:
             print(f"Error generating normalized plot: No non-stablecoin assets found.")
             return fig

        # Normalize the selected data using mean and std dev of the selected period
        # Avoid division by zero if std dev is zero for any asset in the period
        means = recent_data[assets_to_normalize].mean()
        stds = recent_data[assets_to_normalize].std()
        stds[stds == 0] = 1 # Replace 0 std dev with 1 to avoid division by zero

        normalized_data = (recent_data[assets_to_normalize] - means) / stds

        # Add traces for each asset
        for column in normalized_data.columns:
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[column],
                mode='lines',
                name=column,
                hovertemplate = f"<b>{column}</b><br>Date: %{{x|%Y-%m-%d %H:%M}}<br>Normalized Price: %{{y:.3f}}<extra></extra>"
            ))

        # Update layout
        fig.update_layout(
            title=plot_title, # Use dynamic title
            title_x=0.5,
            xaxis_title='Дата',
            yaxis_title='Нормализованная цена (Std Dev)',
            hovermode='x unified',
            legend_title_text='Активы',
            height=500,
            margin=dict(l=50, r=30, t=60, b=50)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=True, zerolinewidth=2, zerolinecolor='grey')

        return fig

    except Exception as e:
        print(f"Error during normalized plot generation: {e}")
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Ошибка при генерации графика нормализованных цен")
        return fig

def generate_correlation_heatmap(combined_df, start_date="2023-01-01", stablecoin='USDCUSDT'):
    """
    Generates a Plotly heatmap of weekly return correlations.

    Args:
        combined_df (pd.DataFrame): DataFrame with DatetimeIndex and asset prices as columns.
        start_date (str): Start date for correlation calculation ('YYYY-MM-DD').
        stablecoin (str): Stablecoin ticker to exclude.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object. Returns empty Figure on error.
    """
    fig = go.Figure()
    fig.update_layout(template="plotly_dark")

    if combined_df.empty or not isinstance(combined_df.index, pd.DatetimeIndex):
        print("Error generating correlation heatmap: Input DataFrame is empty or has wrong index type.")
        return fig

    try:
        # Filter data from start_date
        filtered_data = combined_df.loc[start_date:]
        if filtered_data.empty:
            print(f"Error generating correlation heatmap: No data found after {start_date}.")
            return fig

        # Exclude stablecoin if present
        assets_to_correlate = filtered_data.columns.tolist()
        if stablecoin in assets_to_correlate:
            assets_to_correlate.remove(stablecoin)

        if len(assets_to_correlate) < 2:
             print(f"Error generating correlation heatmap: Need at least 2 non-stablecoin assets.")
             return fig

        # Resample to weekly, calculate pct_change
        weekly_data = filtered_data[assets_to_correlate].resample('W').last() # Use last price of week
        weekly_pct_change = weekly_data.pct_change().dropna(how='all')

        if weekly_pct_change.empty or len(weekly_pct_change) < 2:
            print("Error generating correlation heatmap: Not enough weekly data points after pct_change.")
            return fig

        # Calculate correlation matrix
        corr_matrix = weekly_pct_change.corr()

        # Create heatmap using Plotly Express
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f", # Display correlation values on heatmap
            aspect="equal",
            color_continuous_scale='RdBu_r', # Red-Blue diverging scale, reversed
            zmin=-1, zmax=1 # Fix scale from -1 to 1
        )

        # Update layout for clarity
        fig_heatmap.update_layout(
            title="Матрица корреляций недельных доходностей",
            title_x=0.5,
            height=600, # Adjust as needed
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_tickangle=-45,
            margin=dict(l=50, r=30, t=60, b=100) # Increase bottom margin for labels
        )
        fig_heatmap.update_coloraxes(colorbar_title='Корреляция')

        return fig_heatmap

    except Exception as e:
        print(f"Error during correlation heatmap generation: {e}")
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Ошибка при генерации матрицы корреляций")
        return fig

def generate_single_asset_plot(combined_df, asset_ticker, resolution='h'):
    """
    Generates a Plotly line chart for a single asset's Close price at a specified resolution.

    Args:
        combined_df (pd.DataFrame): DataFrame with DatetimeIndex and asset prices.
        asset_ticker (str): The ticker of the asset to plot.
        resolution (str): Pandas frequency string ('h', '4h', 'D', 'W', 'MS').

    Returns:
        plotly.graph_objects.Figure: Plotly figure object. Returns empty Figure on error.
    """
    fig = go.Figure()
    fig.update_layout(template="plotly_dark")

    if combined_df.empty or not isinstance(combined_df.index, pd.DatetimeIndex):
        print("Error generating single asset plot: Input DataFrame is empty or has wrong index type.")
        return fig
    if asset_ticker not in combined_df.columns:
        print(f"Error generating single asset plot: Asset '{asset_ticker}' not found in DataFrame.")
        return fig

    try:
        # Select the asset data
        asset_data = combined_df[[asset_ticker]].copy()
        asset_data.dropna(inplace=True) # Drop NaNs for this asset

        if asset_data.empty:
             print(f"Error generating single asset plot: No data available for '{asset_ticker}'.")
             return fig

        # Resample the data to the specified resolution
        # For line chart, we just need the last closing price in the interval
        resampled_data = asset_data.resample(resolution).last()
        resampled_data.dropna(inplace=True) # Drop intervals with no data

        if resampled_data.empty:
            print(f"Error generating single asset plot: No data available for '{asset_ticker}' after resampling to '{resolution}'.")
            return fig

        # Create the line chart
        fig.add_trace(go.Scatter(
            x=resampled_data.index,
            y=resampled_data[asset_ticker],
            mode='lines',
            name=asset_ticker,
            line=dict(color='cyan'), # Example color
            hovertemplate = f"<b>{asset_ticker}</b><br>Дата: %{{x|%Y-%m-%d %H:%M}}<br>Цена: %{{y:,.4f}}<extra></extra>" # Adjust precision as needed
        ))

        # Update layout
        fig.update_layout(
            title=f"График цены {asset_ticker} ({resolution})",
            title_x=0.5,
            xaxis_title='Дата',
            yaxis_title='Цена (USDT)',
            hovermode='x unified',
            height=500,
            margin=dict(l=50, r=30, t=60, b=50)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        return fig

    except Exception as e:
        print(f"Error during single asset plot generation for {asset_ticker} ({resolution}): {e}")
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"Ошибка при генерации графика {asset_ticker}")
        return fig

# --- Placeholder for Plotting Functions (to be added in next step) --- #
# def generate_normalized_plot(...):
#     pass
# def generate_correlation_heatmap(...):
#     pass 