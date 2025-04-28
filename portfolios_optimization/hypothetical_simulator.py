import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback

# Assuming these modules contain necessary functions (adjust imports as needed)
from .data_loader import load_price_data # Or load specific asset data
from .portfolio_optimizer import optimize_markowitz_portfolio # Assuming this works with asset list
from .portfolio_analysis import calculate_metrics # For final metrics if needed

# <<< Add DRL related imports >>>
from portfolio_analyzer import (
    preprocess_asset_data as drl_preprocess_asset_data, # Rename to avoid conflict if any
    FeatureEngineer as DRLFeatureEngineer,
    INDICATORS as DRL_INDICATORS,
    STABLECOIN_ASSET as DRL_STABLECOIN, # Ensure consistency
    softmax_normalization # <<< Ensure softmax_normalization is imported >>>
)
from stable_baselines3 import A2C, PPO, SAC, DDPG
# <<< End DRL Imports >>>

# Constants (potentially import from a central place)
STABLECOIN_ASSET = 'USDCUSDT'
# Define the assets DRL models were trained on (MUST MATCH TRAINING)
DRL_TRAINING_ASSETS_HYPO = ['APTUSDT', 'CAKEUSDT', 'HBARUSDT', 'JUPUSDT', 'PEPEUSDT', 'STRKUSDT', 'USDCUSDT']
# <<< Add Oracle strategy name >>>
ORACLE_STRATEGY_NAME = "Oracle (Perfect Foresight)"

# --- Preprocessing Function (Adapted from portfolio_rebalance.py for hourly data) ---
def preprocess_asset_data_hourly(filepath, asset_name, stablecoin_name):
    """Loads and processes HOURLY data for one asset.
       Expects columns like 'Open time', 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    try:
        # Read CSV, explicitly parsing 'Open time' as datetime (ms or string)
        try:
            # Read Open time as string first to handle mixed formats (ms/ISO)
            df = pd.read_csv(filepath, dtype={'Open time': str})
            
            # --- Robust Date Parsing --- 
            time_col = 'Open time' # Assume default
            if time_col not in df.columns:
                 # Find first column containing 'time' case-insensitive
                 time_col = next((c for c in df.columns if 'time' in c.lower()), None)
                 if time_col is None: raise ValueError("Timestamp column not found.")
            
            # Try parsing as milliseconds first, then as standard datetime strings
            parsed_ms = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
            parsed_str = pd.to_datetime(df.loc[parsed_ms.isna(), time_col], errors='coerce')
            df['date'] = parsed_ms.fillna(parsed_str)
            df.dropna(subset=['date'], inplace=True)
            # --- End Robust Date Parsing --- 
            
        except Exception as parse_e:
            print(f"  Error reading/parsing date for {asset_name}: {parse_e}")
            return pd.DataFrame()
        
        # --- Column Renaming and Validation (Simplified for known hourly format) ---
        # Expected columns (lowercase after standardization below)
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.strip().str.lower()
        rename_map = {'open time': 'date'} # Date already handled
        missing_cols = []
        for col in expected_cols:
            if col not in df.columns:
                missing_cols.append(col)
            else:
                 rename_map[col] = col # Keep original name if present

        if missing_cols:
            print(f"  Error: Missing expected columns in {asset_name}: {missing_cols}.")
            return pd.DataFrame()
            
        # Select only needed cols (date is already generated)
        df = df[list(expected_cols) + ['date']]
        # Rename not needed as we use standard lowercase names now

        # --- Data Cleaning --- 
        df = df.sort_values(by='date')
        df = df.drop_duplicates(subset=['date'], keep='first')

        # Convert numeric columns
        for col in expected_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected_cols, inplace=True)

        if df.empty: return pd.DataFrame()

        # Clean stablecoin price (if applicable)
        if asset_name == stablecoin_name:
            close_orig = df['close'].copy()
            df['close'] = np.where((df['close'] > 1.02) | (df['close'] < 0.98), 1.0, df['close'])
            mask_changed = (df['close'] == 1.0) & (close_orig != 1.0)
            df.loc[mask_changed, ['open', 'high', 'low']] = 1.0

        df['tic'] = asset_name
        # Set datetime index for easier time-based operations
        df = df.set_index('date')
        df.sort_index(inplace=True)
        return df[['tic', 'open', 'high', 'low', 'close', 'volume']]

    except FileNotFoundError:
        print(f"  Warning: HOURLY data file not found for {asset_name}: {filepath}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ERROR processing HOURLY {asset_name}: {e}")
        traceback.print_exc()
        return pd.DataFrame()
# --- End Preprocessing Function ---

# --- Helper Function for DRL State Calculation ---
def get_drl_state(current_sim_day: pd.Timestamp,
                    drl_processed_data_hourly: pd.DataFrame, # <<< Expect HOURLY data
                    drl_training_assets: list[str],
                    drl_indicators: list[str],
                    covariance_lookback_hours: int = 24 
                   ):
    """Calculates the DRL state (Covariance Matrix + Indicators) for a given day 
       using HOURLY DRL data.
    """
    # Target time: Last hour of the PREVIOUS day
    target_time = (current_sim_day - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
    
    if drl_processed_data_hourly.empty or not isinstance(drl_processed_data_hourly.index, pd.DatetimeIndex):
        print(f"Error (get_drl_state): HOURLY DRL processed data is empty or has wrong index type.")
        return None

    # --- 1. Extract Indicator Values for the target time ---
    try:
        # Find the latest data point at or before the target time
        state_data_time_index = drl_processed_data_hourly.index[drl_processed_data_hourly.index <= target_time]
        if state_data_time_index.empty:
            print(f"Error (get_drl_state): No HOURLY DRL data found at or before target time {target_time} for state.")
            return None
        actual_state_time = state_data_time_index.max()
        last_processed_data_hour = drl_processed_data_hourly[drl_processed_data_hourly.index == actual_state_time]

        # Ensure data is sorted by the canonical asset list order for consistency
        last_processed_data_hour = last_processed_data_hour.reset_index().set_index('tic')
        last_processed_data_hour = last_processed_data_hour.reindex(drl_training_assets).reset_index()

        indicator_values = last_processed_data_hour[drl_indicators].fillna(0).values.T
        if indicator_values.shape[1] != len(drl_training_assets):
            print(f"Error (get_drl_state): Indicator shape mismatch @ {actual_state_time}.")
            return None

    except Exception as ind_e:
        print(f"Error (get_drl_state): Failed to extract indicators for {actual_state_time}: {ind_e}")
        return None

    # --- 2. Calculate Covariance Matrix using HOURLY data ---
    cov_matrix = np.zeros((len(drl_training_assets), len(drl_training_assets))) 
    try:
        cov_end_time = actual_state_time # End covariance period at the time indicators were taken
        cov_start_time = cov_end_time - timedelta(hours=covariance_lookback_hours) # Look back N hours

        cov_data_input = drl_processed_data_hourly[
            (drl_processed_data_hourly.index >= cov_start_time) & 
            (drl_processed_data_hourly.index <= cov_end_time)
        ]

        if len(cov_data_input.index.unique()) < len(drl_training_assets): 
            print(f"Warning (get_drl_state): Not enough unique timestamps ({len(cov_data_input.index.unique())}) for HOURLY covariance calc ending {cov_end_time}. Using zero matrix.")
        else:
            price_lookback = cov_data_input.pivot_table(index=cov_data_input.index, columns='tic', values='close')
            price_lookback = price_lookback.reindex(columns=drl_training_assets).ffill().bfill()
            
            # Use HOURLY returns
            return_lookback = price_lookback.pct_change().dropna(how='all') 
            
            if len(return_lookback) >= len(drl_training_assets):
                 return_lookback_valid = return_lookback.dropna(axis=1, how='all')
                 if not return_lookback_valid.empty and return_lookback_valid.shape[1] > 1:
                     cov_matrix_calc = return_lookback_valid.cov()
                     cov_matrix = cov_matrix_calc.reindex(index=drl_training_assets, columns=drl_training_assets).fillna(0).values
                 else:
                     print(f"Warning (get_drl_state): Not enough valid HOURLY return series for covariance calc ending {cov_end_time}. Using zero matrix.")
            else:
                 print(f"Warning (get_drl_state): Not enough HOURLY return data points ({len(return_lookback)}) for covariance calc ending {cov_end_time}. Using zero matrix.")

    except Exception as cov_e:
        print(f"Error (get_drl_state): Failed to calculate HOURLY covariance for {cov_end_time}: {cov_e}. Using zero matrix.")

    # --- 3. Construct State --- 
    try:
        state = np.append(cov_matrix, indicator_values, axis=0) 
        expected_shape = (len(drl_training_assets) + len(drl_indicators), len(drl_training_assets))
        if state.shape != expected_shape:
            print(f"Error (get_drl_state): Final state shape mismatch. Got {state.shape}, expected {expected_shape}.")
            return None
        return state
    except Exception as combine_e:
         print(f"Error (get_drl_state): Failed to combine cov matrix and indicators: {combine_e}")
         return None
# --- End Helper Function ---

def run_hypothetical_analysis(
    # user_transactions_list: list[dict], # REMOVED
    initial_investment: float, # ADDED
    hypothetical_assets: list[str],
    start_date_str: str,
    end_date_str: str,
    data_path: str = "data", # Assuming combined data is used
    bank_apr: float = 0.2,
    commission_rate: float = 0.001,
    rebalance_interval_days: int = 20,
    drl_rebalance_interval_days: int = 20, # Added
    drl_models_dir: str = "notebooks/trained_models", # Added
):
    """
    Simulates portfolio performance for a HYPOTHETICAL set of assets starting 
    with an INITIAL INVESTMENT and NO further cash flows.

    Args:
        initial_investment (float): The starting amount in USD.
        hypothetical_assets (list[str]): List of asset tickers to simulate.
        start_date_str (str): Simulation start date ('YYYY-MM-DD').
        end_date_str (str): Simulation end date ('YYYY-MM-DD').
        data_path (str): Path to the directory containing asset price data (e.g., hourly CSVs or a combined file).
        bank_apr (float): Annual percentage rate for the bank deposit comparison.
        commission_rate (float): Commission fee per trade/rebalance.
        rebalance_interval_days (int): How often to rebalance EqualWeight/Markowitz/Oracle.
        drl_rebalance_interval_days (int): How often to rebalance DRL strategies.
        drl_models_dir (str): Path to the directory containing trained DRL models.

    Returns:
        tuple: (pd.DataFrame results_summary, go.Figure plot) or (None, None) on error.
    """
    print("--- Starting Hypothetical Analysis (Initial Investment Mode) ---")
    results = {} # To store daily portfolio values for each strategy

    # 1. Preprocess Transactions - REMOVED
    # print("Preprocessing user transactions...")
    # daily_cash_flows = preprocess_transactions(user_transactions_list) ...
    sim_start_date = pd.to_datetime(start_date_str)
    sim_end_date = pd.to_datetime(end_date_str)
    print(f"Simulation period: {sim_start_date.date()} to {sim_end_date.date()}")

    # 2. Load Price Data (Adjusted date range slightly)
    print(f"Loading price data for hypothetical assets: {hypothetical_assets}")
    assets_to_load = list(set(hypothetical_assets + [STABLECOIN_ASSET]))
    try:
        all_price_data = load_price_data() # Assumes this loads combined DAILY data now
        if all_price_data.empty:
            raise ValueError("Combined price data could not be loaded.")
        missing_assets = [a for a in assets_to_load if a not in all_price_data.columns]
        if missing_assets:
             raise ValueError(f"Price data missing for: {missing_assets}")
             
        # Load data from one day before sim_start (for Markowitz/DRL lookback)
        # up to sim_end_date. Ensure index is datetime.
        load_start_date = sim_start_date - timedelta(days=366) # Need ~1 year lookback
        price_data_full_hist = all_price_data[assets_to_load].loc[load_start_date:sim_end_date].copy()
        if not isinstance(price_data_full_hist.index, pd.DatetimeIndex):
             price_data_full_hist.index = pd.to_datetime(price_data_full_hist.index)

        # --- Ensure data is daily --- 
        # Resample to daily frequency using the end of day, forward fill then back fill
        price_data_daily = price_data_full_hist.resample('D').last()
        price_data_daily.ffill(inplace=True)
        price_data_daily.bfill(inplace=True) # Fill gaps after resampling
        # Normalize index to remove time component, ensuring clean daily keys
        price_data_daily.index = price_data_daily.index.normalize()

        # Use the daily data for the simulation period + lookback
        price_data = price_data_daily # Rename for clarity in the rest of the function

        # Filter price_data to only the necessary range for the simulation + 1 day lookback
        sim_period_start_for_data = (sim_start_date - timedelta(days=1)).normalize() # Ensure start is normalized
        sim_end_date_norm = sim_end_date.normalize()
        price_data = price_data.loc[sim_period_start_for_data:sim_end_date_norm]

        # Adjust sim_start_date to the first available date in the daily index if needed
        sim_start_date_norm = sim_start_date.normalize()
        if sim_start_date_norm not in price_data.index:
            available_dates_after_start = price_data.index[price_data.index >= sim_start_date_norm]
            if not available_dates_after_start.empty:
                sim_start_date_norm = available_dates_after_start[0]
                # Update the original sim_start_date as well if needed elsewhere, though normalized is used now
                sim_start_date = sim_start_date_norm 
                print(f"Warning: Original start date data missing/adjusted. Effective simulation start date: {sim_start_date_norm.date()}")
            else:
                raise ValueError(f"No price data available at or after simulation start date: {start_date_str}")
        # Use the normalized start date from now on
        sim_start_date = sim_start_date_norm
            
        risky_assets = [a for a in hypothetical_assets if a != STABLECOIN_ASSET]
        print(f"Using Risky Assets: {risky_assets}")
        
    except Exception as e:
        print(f"Error loading/processing price data: {e}")
        traceback.print_exc()
        return None, None

    # --- DRL Setup Section --- Now needs to handle HOURLY data ---
    print("--- Setting up DRL (using HOURLY data) --- ")
    # Determine compatible assets for DRL
    drl_training_risky = set(DRL_TRAINING_ASSETS_HYPO) - {STABLECOIN_ASSET}
    user_hypothetical_risky = set(hypothetical_assets) - {STABLECOIN_ASSET}
    compatible_drl_risky_assets = list(user_hypothetical_risky.intersection(drl_training_risky))
    
    drl_models_to_run = []
    drl_models_available = {"A2C": A2C, "PPO": PPO, "SAC": SAC, "DDPG": DDPG}
    drl_agent = {}
    drl_fe = None
    # <<< RENAME: This will now hold HOURLY processed data >>>
    drl_processed_data_hourly = pd.DataFrame() 

    if not compatible_drl_risky_assets:
        print("User-selected risky assets have no overlap with DRL training assets. Skipping DRL strategies.")
        run_drl = False
    else:
        print(f"DRL simulation will run on compatible risky assets: {compatible_drl_risky_assets}")
        run_drl = True
        drl_models_to_run = list(drl_models_available.keys())
        
        # --- Load and preprocess HOURLY data for ALL DRL training assets --- 
        print("Loading and preprocessing HOURLY data for ALL DRL training assets...")
        all_drl_hourly_frames = []
        missing_drl_hourly_assets = []
        for asset in DRL_TRAINING_ASSETS_HYPO:
            # Assuming hourly files are named like asset_hourly_data.csv
            filepath = os.path.join(data_path, f"{asset}_hourly_data.csv") 
            try:
                # <<< Use the new HOURLY preprocessing function >>>
                df_asset_hourly = preprocess_asset_data_hourly(filepath, asset, STABLECOIN_ASSET)
                if not df_asset_hourly.empty:
                    all_drl_hourly_frames.append(df_asset_hourly)
                else:
                    print(f"Warning: HOURLY data processing failed for DRL training asset: {asset}")
                    missing_drl_hourly_assets.append(asset)
            except Exception as data_err:
                 print(f"Warning: Failed DRL HOURLY data prep for {asset}: {data_err}")
                 missing_drl_hourly_assets.append(asset)
        
        if not all_drl_hourly_frames or len(missing_drl_hourly_assets) > 0:
            print(f"Error: Could not load/process HOURLY data for all DRL training assets (Missing: {missing_drl_hourly_assets}). Disabling DRL.")
            run_drl = False
            drl_models_to_run = []
        else:
            # Combine all hourly dataframes
            full_drl_hourly_data = pd.concat(all_drl_hourly_frames)
            full_drl_hourly_data.sort_index(inplace=True) # Sort by hourly datetime index
            
            # Filter HOURLY data for the relevant range (simulation + lookbacks)
            # Need lookback for features (depends on INDICATORS) + covariance
            feature_lookback_days = 5 # Example, adjust based on longest indicator period
            cov_lookback_hours = 24
            drl_data_start_date = sim_start_date - timedelta(days=feature_lookback_days) # Start earlier for features
            # Ensure we load data up to the simulation end date
            full_drl_hourly_data = full_drl_hourly_data[
                (full_drl_hourly_data.index >= drl_data_start_date) & 
                (full_drl_hourly_data.index <= sim_end_date.replace(hour=23, minute=59, second=59))
            ].copy() 
            
            if full_drl_hourly_data.empty:
                print("Error: No HOURLY DRL training asset data found within the required date range. Disabling DRL.")
                run_drl = False
                drl_models_to_run = []
            else:
                # --- Feature Engineering on HOURLY data --- 
                try:
                    print("Running FeatureEngineer on HOURLY DRL data...")
                    drl_fe = DRLFeatureEngineer(use_technical_indicator=True, tech_indicator_list=DRL_INDICATORS)
                    # FeatureEngineer needs 'date' and 'tic' columns, reset index temporarily
                    drl_processed_hourly_temp = drl_fe.preprocess_data(full_drl_hourly_data.reset_index()) 
                    # Drop NaNs introduced by indicators
                    drl_processed_hourly_temp.dropna(subset=DRL_INDICATORS, inplace=True)
                    print(f"FeatureEngineer completed. Shape before cov: {drl_processed_hourly_temp.shape}")
                    if drl_processed_hourly_temp.empty:
                         raise ValueError("DRL processed data is empty after FeatureEngineer & dropna.")
                    # <<< NOTE: Covariance is calculated dynamically in get_drl_state >>>
                    # <<< We need to keep the hourly data with indicators >>>
                    # Set index back to datetime for get_drl_state
                    drl_processed_data_hourly = drl_processed_hourly_temp.set_index('date')
                    drl_processed_data_hourly.sort_index(inplace=True)
                    
                except Exception as fe_e:
                     print(f"Error during DRL Feature Engineering on HOURLY data: {fe_e}. Disabling DRL.")
                     traceback.print_exc()
                     run_drl = False
                     drl_models_to_run = []

        # Load DRL models if preprocessing succeeded
        if run_drl:
            print("Loading DRL models...")
            for model_name in list(drl_models_available.keys()): # Use keys() to avoid modifying during iteration
                try:
                    model_path = os.path.join(drl_models_dir, f"trained_{model_name.lower()}.zip")
                    if not os.path.exists(model_path):
                         print(f"Warning: Model file not found for {model_name} at {model_path}. Skipping model.")
                         drl_models_to_run.remove(model_name)
                         continue
                    model_class = drl_models_available[model_name]
                    drl_agent[model_name] = model_class.load(model_path)
                    print(f"Loaded DRL model: {model_name}")
                except Exception as load_e:
                    print(f"Error loading DRL model {model_name}: {load_e}. Skipping model.")
                    if model_name in drl_models_to_run: drl_models_to_run.remove(model_name)
            # Check if any models were loaded
            if not drl_models_to_run:
                 print("No DRL models successfully loaded. Disabling DRL strategies.")
                 run_drl = False
                 
    # --- End DRL Setup Section ---

    # 4. Initialize Strategy Portfolios & Initial Investment
    print("Initializing strategy portfolios with initial investment...")
    # <<< Define the final bank column name ONCE >>>
    bank_col_name = f"Bank Deposit ({bank_apr*100:.1f}% APR)"
    # <<< Modify the strategies list to include the final bank name >>>
    base_strategies = ["Buy & Hold", "Equal Weight", "Markowitz", "Bank Deposit", ORACLE_STRATEGY_NAME]
    strategies = [(bank_col_name if s == "Bank Deposit" else s) for s in base_strategies] + drl_models_to_run
    
    # <<< Initialize dictionaries using the potentially renamed strategy names >>>
    portfolio_values = {strategy: {} for strategy in strategies}
    portfolio_holdings = {strategy: {} for strategy in strategies} 
    initial_holdings = {strategy: {asset: 0.0 for asset in assets_to_load} for strategy in strategies}
    initial_weights = {strategy: {} for strategy in strategies} # Store initial risky weights

    # Get prices on the effective simulation start date
    try:
        start_prices = price_data.loc[sim_start_date]
        # Handle potential NaNs in start prices (use previous day if available)
        if start_prices.isnull().any():
             prev_day_prices = price_data.loc[sim_start_date - timedelta(days=1)]
             start_prices = start_prices.fillna(prev_day_prices)
             if start_prices.isnull().any(): # Still NaN? Critical error
                 raise ValueError(f"Cannot determine valid starting prices for all assets on {sim_start_date.date()}")
    except Exception as e:
        print(f"Critical Error: Cannot get valid starting prices. {e}")
        traceback.print_exc()
        return None, None


    # --- Allocate Initial Investment --- 
    # <<< Use the correct strategy names (already in the strategies list) >>>
    for strategy in strategies: # Loop through potentially renamed strategy names
        current_holdings_init = {asset: 0.0 for asset in assets_to_load} # Local var for initial calc

        # <<< Check against the potentially renamed bank column name >>>
        if strategy == bank_col_name:
             # Use the stablecoin price on the start date
             stable_price = start_prices.get(STABLECOIN_ASSET, 1.0) 
             if stable_price is None or pd.isna(stable_price) or stable_price < 1e-9:
                 print(f"Warning: Cannot get valid start price for {STABLECOIN_ASSET}. Bank Deposit might be inaccurate.")
                 stable_price = 1.0
             current_holdings_init[STABLECOIN_ASSET] = initial_investment / stable_price
             # <<< Use the correct strategy key >>>
             portfolio_values[strategy][sim_start_date] = initial_investment
             
        else:
             # Calculate target weights for risky assets at the start
             target_weights = {} # Weights for risky assets
             
             # Default to Equal Weight if no risky assets selected
             if not risky_assets:
                 target_weights = {} # All goes to stablecoin
             
             elif strategy == "Equal Weight" or strategy == "Buy & Hold" or strategy == ORACLE_STRATEGY_NAME: # B&H and Oracle also start EW
                  num_risky = len(risky_assets)
                  equal_weight = 1.0 / num_risky
                  target_weights = {asset: equal_weight for asset in risky_assets}
                  if strategy == "Buy & Hold": 
                      initial_weights[strategy] = target_weights.copy() # Store for B&H P&L calc later if needed
             
             elif strategy == "Markowitz":
                  if risky_assets:
                      # <<< Use sim_start_date for INITIAL calculation >>>
                      hist_data_end = sim_start_date - timedelta(days=1)
                      # Use the full history loaded earlier
                      hist_prices = price_data_full_hist.loc[:hist_data_end, risky_assets]
                      hist_returns = hist_prices.pct_change().dropna()
                      if len(hist_returns) >= len(risky_assets) * 2: 
                          try:
                              optimal_weights_df, _, _ = optimize_markowitz_portfolio(hist_returns)
                              if optimal_weights_df is not None: 
                                  target_weights = optimal_weights_df['Optimal Weight'].to_dict()
                              else: target_weights = None # Opt failed
                          except Exception as opt_e:
                               # <<< Use sim_start_date in warning message >>>
                               print(f"Warning ({strategy} Initial): Markowitz optimization failed: {opt_e}. Skipping initial calc.")
                               target_weights = None
                      else:
                           # <<< Use sim_start_date in warning message >>>
                           print(f"Warning ({strategy} Initial): Not enough historical data ({len(hist_returns)} days before {sim_start_date.date()}) for Markowitz. Skipping initial calc.")
                           target_weights = None
                      # Fallback to Equal Weight if Markowitz failed or None
                      if target_weights is None: # Check if it's still None
                          # <<< Use sim_start_date in warning message >>>
                          print(f"Warning ({strategy} Initial): Falling back to Equal Weight for initial allocation.")
                          equal_weight = 1.0 / len(risky_assets)
                          target_weights = {asset: equal_weight for asset in risky_assets}
                  else: # No risky assets
                       target_weights = {}
             
             elif strategy in drl_models_to_run:
                  # Initial DRL: Requires state calculation at sim_start_date - timedelta(days=1)
                  # Placeholder: Fallback to Equal Weight among COMPATIBLE DRL assets
                  compatible_risky_for_init = [a for a in risky_assets if a in compatible_drl_risky_assets]
                  if compatible_risky_for_init:
                      print(f"Warning (Initial {strategy}): Using Equal Weight for initial allocation among compatible DRL assets ({compatible_risky_for_init}).")
                      equal_weight = 1.0 / len(compatible_risky_for_init)
                      target_weights = {asset: equal_weight for asset in compatible_risky_for_init}
                  else:
                      print(f"Warning (Initial {strategy}): No compatible DRL risky assets selected. Starting with 100% {STABLECOIN_ASSET}.")
                      target_weights = {} # All goes to stablecoin


             # Apply initial weights
             investment_in_risky = 0
             for asset in risky_assets:
                 asset_value = initial_investment * target_weights.get(asset, 0.0)
                 asset_start_price = start_prices.get(asset)
                 if asset_start_price is None or pd.isna(asset_start_price) or asset_start_price < 1e-9:
                      print(f"Warning: Cannot get valid start price for {asset}. Initial allocation for {strategy} might be inaccurate.")
                      current_holdings_init[asset] = 0.0
                 else:
                      current_holdings_init[asset] = asset_value / asset_start_price
                      investment_in_risky += asset_value
                      
             # Remainder in stablecoin
             stablecoin_value = initial_investment - investment_in_risky
             stable_price = start_prices.get(STABLECOIN_ASSET, 1.0)
             if stable_price is None or pd.isna(stable_price) or stable_price < 1e-9:
                  print(f"Warning: Cannot get valid start price for {STABLECOIN_ASSET}. Initial allocation for {strategy} might be inaccurate.")
                  current_holdings_init[STABLECOIN_ASSET] = 0.0
             else:
                  current_holdings_init[STABLECOIN_ASSET] = max(0, stablecoin_value / stable_price) # Ensure non-negative

             # Verify initial value ~ initial_investment (within tolerance)
             calculated_initial_value = sum(current_holdings_init[asset] * start_prices.get(asset, 0) for asset in assets_to_load)
             if not np.isclose(calculated_initial_value, initial_investment, rtol=1e-3):
                  print(f"Warning ({strategy}): Initial calculated value ${calculated_initial_value:.2f} differs significantly from investment ${initial_investment:.2f}.")

             initial_holdings[strategy] = current_holdings_init
             portfolio_values[strategy][sim_start_date] = initial_investment # Start exactly at the investment amount

    # 5. Simulation Loop (No Cash Flows)
    print("Running daily simulation (no external cash flows)...")
    # <<< Use correct strategy keys for initialization >>>
    last_rebalance_day = {strategy: sim_start_date for strategy in strategies} 
    full_holdings_history = {strategy: {sim_start_date: initial_holdings[strategy].copy()} for strategy in strategies}

    # Use the actual dates from the *daily* loaded price_data index within the sim range
    sim_date_range = price_data.index[(price_data.index > sim_start_date) & (price_data.index <= sim_end_date_norm)]

    for current_day in sim_date_range:
        # current_day is now guaranteed to be a normalized daily timestamp
        day_str = current_day.strftime('%Y-%m-%d') 
        # Calculate previous day (also normalized)
        previous_day = (current_day - timedelta(days=1)).normalize()
        
        # The while loop to find previous_day might be redundant now, but keep for robustness against gaps
        while previous_day not in price_data.index and previous_day >= sim_start_date:
             previous_day -= timedelta(days=1)
             previous_day = previous_day.normalize() # Ensure it stays normalized
             
        if previous_day < sim_start_date: 
             print(f"Critical Error: Cannot find previous day in index for {day_str}") 
             continue 

        # Get today's prices
        try:
            current_prices = price_data.loc[current_day]
             # Handle potential NaNs by forward filling from the valid previous day's prices
            if current_prices.isnull().any():
                prev_valid_prices = price_data.loc[previous_day]
                current_prices = current_prices.fillna(prev_valid_prices)
                if current_prices.isnull().any(): # Still NaN? Problem!
                    print(f"Warning: Cannot determine valid prices for all assets on {day_str} even after fill. Skipping day.")
                    # Propagate previous day's value? Or stop? Let's propagate for now.
                    for strategy in strategies:
                         portfolio_values[strategy][current_day] = portfolio_values[strategy][previous_day]
                         full_holdings_history[strategy][current_day] = full_holdings_history[strategy][previous_day].copy()
                    continue
        except KeyError:
             # This should not happen if sim_date_range uses index, but as fallback:
            print(f"Warning: Price data key error for {day_str}. Using previous day's prices.")
            current_prices = price_data.loc[previous_day] 

        # --- Process each strategy --- 
        for strategy in strategies:
            # <<< Initialize needs_rebalance for each strategy >>>
            needs_rebalance = False
            
            # Get holdings from the actual previous day in history
            if strategy == "Buy & Hold":
                 # For B&H, always use the initial holdings structure
                 prev_holdings = initial_holdings[strategy].copy() 
                 # Ensure these initial holdings are recorded for the *current* day too
                 current_holdings = prev_holdings.copy()
            else:
                 # For other strategies, use the history
                 prev_holdings = full_holdings_history[strategy][previous_day].copy()
                 current_holdings = prev_holdings.copy() # Start with previous holdings for this day
            
            # A. Calculate current value based on price changes ONLY
            current_value_before_rebalance = sum(prev_holdings.get(asset, 0) * current_prices.get(asset, 0) 
                                                 for asset in assets_to_load)
            
            final_day_value = current_value_before_rebalance # Default value if no rebalance/interest

            # B. Handle Bank Deposit Interest
            # <<< Check against the potentially renamed bank column name >>>
            if strategy == bank_col_name: 
                 # <<< Use the strategy name directly as the key >>>
                 prev_bank_value = portfolio_values[strategy].get(previous_day, 0) 
                 
                 # Ensure previous value is valid before calculating interest
                 if pd.notna(prev_bank_value) and prev_bank_value > 0:
                      daily_rate = (1 + bank_apr)**(1/365.25) - 1 # Use 365.25 for accuracy
                      bank_interest = prev_bank_value * daily_rate
                      final_day_value = prev_bank_value + bank_interest 
                 else: # If previous value is invalid or zero, do not add interest, keep previous value
                      final_day_value = prev_bank_value # Keep the previous value (which might be 0)
                 
                 # Update holdings (assuming all in stablecoin)
                 stable_price = current_prices.get(STABLECOIN_ASSET, 1.0)
                 if stable_price < 1e-9: stable_price = 1.0
                 current_holdings = {asset: 0.0 for asset in assets_to_load}
                 # Only assign holdings if value > 0
                 if final_day_value > 0:
                    current_holdings[STABLECOIN_ASSET] = final_day_value / stable_price
                 needs_rebalance = False # Bank deposit doesn't rebalance based on interval
            # C. Check if Rebalance is Needed (for other strategies)
            elif strategy != "Buy & Hold": # Moved the condition here, Bank Deposit handled above
                needs_rebalance = False
                rebalance_interval = drl_rebalance_interval_days if strategy in drl_models_to_run else rebalance_interval_days
                if strategy != "Buy & Hold": # Buy & Hold never rebalances
                    if (current_day - last_rebalance_day[strategy]).days >= rebalance_interval:
                         # Only rebalance if portfolio has value and interval is met
                         if current_value_before_rebalance > 1: 
                             needs_rebalance = True
                             last_rebalance_day[strategy] = current_day

            # D. Perform Rebalancing if needed
            if needs_rebalance:
                target_weights = None # Reset
                if strategy == "Equal Weight": 
                    if risky_assets: # Check if there are risky assets
                        num_risky = len(risky_assets)
                        equal_weight = 1.0 / num_risky
                        target_weights = {asset: equal_weight for asset in risky_assets}
                    else: 
                        target_weights = {} # All in stablecoin
                elif strategy == "Markowitz": 
                    if risky_assets:
                        hist_data_end = current_day - timedelta(days=1)
                        # Use full history available up to the day before
                        hist_prices = price_data_full_hist.loc[:hist_data_end, risky_assets]
                        hist_returns = hist_prices.pct_change().dropna()
                        if len(hist_returns) >= len(risky_assets) * 2: 
                            try:
                                optimal_weights_df, _, _ = optimize_markowitz_portfolio(hist_returns)
                                if optimal_weights_df is not None: 
                                    target_weights = optimal_weights_df['Optimal Weight'].to_dict()
                                else: target_weights = None # Opt failed
                            except Exception as opt_e:
                                 # Use day_str here, which IS defined in the loop
                                 print(f"Warning ({strategy} {day_str}): Markowitz optimization failed: {opt_e}. Skipping rebalance.")
                                 target_weights = None
                        else:
                             print(f"Warning ({strategy} {day_str}): Not enough historical data ({len(hist_returns)} days) for Markowitz. Skipping rebalance.")
                             target_weights = None
                        # Fallback to Equal Weight if Markowitz failed or None
                        if target_weights is None: # Check if it's still None
                            print(f"Warning ({strategy} {day_str}): Falling back to Equal Weight.")
                            equal_weight = 1.0 / len(risky_assets)
                            target_weights = {asset: equal_weight for asset in risky_assets}
                    else: # No risky assets
                         target_weights = {}
                elif strategy == ORACLE_STRATEGY_NAME:
                    target_weights = None # Reset
                    if not risky_assets and STABLECOIN_ASSET in hypothetical_assets:
                         # Only stablecoin selected, stay 100% stablecoin
                         target_weights = {STABLECOIN_ASSET: 1.0}
                    elif risky_assets or STABLECOIN_ASSET in hypothetical_assets:
                        next_rebalance_target_date = current_day + timedelta(days=rebalance_interval)
                        potential_next_dates = price_data.index[
                            (price_data.index > current_day) & 
                            (price_data.index <= min(next_rebalance_target_date, sim_end_date_norm))
                        ]
                        if not potential_next_dates.empty:
                            next_rebalance_date_actual = potential_next_dates[-1]
                            try:
                                # Get prices for current and next date
                                prices_now = current_prices # Already fetched for the current day
                                prices_next = price_data.loc[next_rebalance_date_actual]

                                # Identify assets valid for calculation (have price now and next)
                                assets_to_consider = list(set(hypothetical_assets) | {STABLECOIN_ASSET})
                                valid_assets = []
                                for asset in assets_to_consider:
                                    price_now_val = prices_now.get(asset)
                                    price_next_val = prices_next.get(asset)
                                    if pd.notna(price_now_val) and price_now_val > 1e-9 and pd.notna(price_next_val):
                                         valid_assets.append(asset)
                                
                                if not valid_assets:
                                     print(f"Warning ({strategy} {day_str}): No valid assets found with prices for lookahead. Skipping rebalance.")
                                     target_weights = None
                                else:
                                     # Calculate returns ONLY for valid assets
                                     returns = (prices_next.loc[valid_assets] / prices_now.loc[valid_assets]) - 1
                                     # Handle potential NaNs/Infs arising from division if validation missed something
                                     returns = returns.replace([np.inf, -np.inf], np.nan).fillna(-1.0) # Treat NaN return as loss

                                     # Ensure stablecoin return is 0 if it's valid
                                     if STABLECOIN_ASSET in valid_assets:
                                         returns[STABLECOIN_ASSET] = 0.0
                                     
                                     # Find best asset among VALID assets
                                     best_asset = returns.idxmax()
                                     best_return = returns.max()
                                     
                                     # Decide target: best asset if return >= 0, else stablecoin (if valid), else skip
                                     chosen_asset = None
                                     if best_return >= 0:
                                          chosen_asset = best_asset
                                     elif STABLECOIN_ASSET in valid_assets:
                                          chosen_asset = STABLECOIN_ASSET
                                          
                                     if chosen_asset:
                                          target_weights = {asset: 1.0 if asset == chosen_asset else 0.0 for asset in valid_assets}
                                          # Normalize just in case (should be 1.0 already)
                                          w_sum = sum(target_weights.values())
                                          if w_sum > 1e-9: target_weights = {a: w/w_sum for a,w in target_weights.items()}
                                     else:
                                          print(f"Warning ({strategy} {day_str}): Could not determine optimal asset (all returns < 0, no valid stablecoin?). Skipping rebalance.")
                                          target_weights = None

                            except Exception as oracle_e:
                                print(f"Warning ({strategy} {day_str}): Error calculating returns/best asset: {oracle_e}. Skipping rebalance.")
                                target_weights = None
                        else:
                             print(f"Warning ({strategy} {day_str}): Cannot find future price data for lookahead. Skipping rebalance.")
                             target_weights = None
                    # else: No assets selected for simulation, target_weights remains None
                
                # --- DRL Strategy Logic (Keep active) ---
                elif strategy in drl_models_to_run:
                    # Initial DRL: Requires state calculation at sim_start_date - timedelta(days=1)
                    # Placeholder: Fallback to Equal Weight among COMPATIBLE DRL assets
                    compatible_risky_for_init = [a for a in risky_assets if a in compatible_drl_risky_assets]
                    if compatible_risky_for_init:
                        print(f"Warning (Initial {strategy}): Using Equal Weight for initial allocation among compatible DRL assets ({compatible_risky_for_init}).")
                        equal_weight = 1.0 / len(compatible_risky_for_init)
                        target_weights = {asset: equal_weight for asset in compatible_risky_for_init}
                    else:
                        print(f"Warning (Initial {strategy}): No compatible DRL risky assets selected. Starting with 100% {STABLECOIN_ASSET}.")
                        target_weights = {} # All goes to stablecoin

                    # Calculate the state for the DRL model
                    # Use the globally available drl_processed_data_hourly (assuming it's indexed by date)
                    # Ensure drl_processed_data_hourly is accessible here (passed as arg or global? Let's assume passed/accessible)
                    # Note: covariance_lookback_hours might need adjustment based on DRL data frequency (e.g., use 24 for daily)
                    current_state = get_drl_state(current_day, 
                                                  drl_processed_data_hourly, # Needs access to this DataFrame
                                                  DRL_TRAINING_ASSETS_HYPO, 
                                                  DRL_INDICATORS, 
                                                  covariance_lookback_hours=24) # Adjust lookback as needed

                    if current_state is not None:
                        try:
                            model = drl_agent[strategy]
                            raw_actions, _ = model.predict(current_state, deterministic=True)
                            # Normalize actions to get weights for ALL DRL training assets
                            norm_weights_all_drl = softmax_normalization(raw_actions)
                            target_weights_all_drl = dict(zip(DRL_TRAINING_ASSETS_HYPO, norm_weights_all_drl))

                            # Filter weights for assets relevant to *this* simulation
                            # Relevant = assets in hypothetical_assets + STABLECOIN_ASSET
                            user_assets_in_sim = set(hypothetical_assets).union({STABLECOIN_ASSET})
                            
                            # Keep only weights for assets the model predicted AND are in user's sim set
                            filtered_weights = {
                                asset: weight 
                                for asset, weight in target_weights_all_drl.items() 
                                if asset in user_assets_in_sim
                            }
                            
                            # Renormalize the filtered weights to sum to 1
                            total_filtered_weight = sum(filtered_weights.values())
                            if total_filtered_weight > 1e-9:
                                target_weights = {asset: weight / total_filtered_weight 
                                                  for asset, weight in filtered_weights.items()}
                            else:
                                # Fallback if filtered weights sum to zero or are empty
                                print(f"Warning ({strategy} {day_str}): Filtered DRL weights sum to zero. Falling back.")
                                # Fallback logic: Equal weight among user's risky assets? Or 100% stablecoin?
                                # Let's use equal weight among hypothetical risky assets as a fallback
                                if risky_assets: 
                                     equal_weight = 1.0 / len(risky_assets)
                                     target_weights = {asset: equal_weight for asset in risky_assets}
                                else: # If no risky assets, 100% stablecoin
                                     target_weights = {STABLECOIN_ASSET: 1.0}
                                
                        except Exception as pred_e:
                            print(f"Error ({strategy} {day_str}): DRL prediction/processing failed: {pred_e}. Skipping rebalance.")
                            target_weights = None # Skip rebalance on prediction error
                    else:
                        print(f"Error ({strategy} {day_str}): Failed to calculate DRL state. Skipping rebalance.")
                        target_weights = None # Skip rebalance if state calculation failed

                # If target_weights is None after all this, rebalance is skipped in the next block

                # --- Apply Rebalancing & Commission (if target weights determined) ---
                if target_weights is not None: 
                    portfolio_value_before_rebalance_fees = current_value_before_rebalance
                    new_holdings = {asset: 0.0 for asset in assets_to_load}
                    commission_cost = 0
                    value_after_commission = 0 # Track value used for allocation after fees

                    # Calculate target values based on *risky* weights first
                    target_risky_value = 0
                    for asset in risky_assets:
                        target_asset_value = portfolio_value_before_rebalance_fees * target_weights.get(asset, 0.0)
                        target_risky_value += target_asset_value

                    # Calculate required stablecoin value
                    target_stablecoin_value = portfolio_value_before_rebalance_fees * target_weights.get(STABLECOIN_ASSET, 0.0)
                    # If stablecoin wasn't explicitly in target_weights (e.g., EW, Markowitz only gave risky), calculate remainder
                    if STABLECOIN_ASSET not in target_weights:
                         target_stablecoin_value = portfolio_value_before_rebalance_fees - target_risky_value
                    
                    # Calculate total target value (should be close to original value)
                    total_target_value_check = target_risky_value + target_stablecoin_value
                    if not np.isclose(total_target_value_check, portfolio_value_before_rebalance_fees):
                         print(f"Debug ({strategy} {day_str}): Target value check failed. Target={total_target_value_check:.2f}, Before={portfolio_value_before_rebalance_fees:.2f}")
                         # Adjust target stablecoin to ensure sum matches?
                         target_stablecoin_value = portfolio_value_before_rebalance_fees - target_risky_value

                    # Calculate turnover and commission
                    turnover = 0
                    temp_target_values = {asset: portfolio_value_before_rebalance_fees * target_weights.get(asset, 0.0) for asset in risky_assets}
                    temp_target_values[STABLECOIN_ASSET] = target_stablecoin_value

                    for asset in assets_to_load:
                         current_asset_value = current_holdings.get(asset, 0.0) * current_prices.get(asset, 0)
                         target_asset_value_final = temp_target_values.get(asset, 0.0)
                         turnover += abs(target_asset_value_final - current_asset_value)
                    
                    # Commission is on the total turnover value / 2 (since each trade involves two sides) - OR apply to each asset trade?
                    # Let's apply based on change in asset value, sum of absolute changes / 2 * rate?
                    # Simplified: Apply commission to the total value changed calculated above (turnover / 2)
                    # Alternative: Apply to each transaction abs(target_value - current_value) * rate?
                    # Let's stick to the simpler model: total value moved * commission rate
                    commission_cost = (turnover / 2.0) * commission_rate # Apply to half the turnover

                    # Value available for allocation *after* commission
                    value_after_commission = portfolio_value_before_rebalance_fees - commission_cost

                    # Allocate the value_after_commission based on target_weights
                    final_value_check = 0
                    for asset in assets_to_load:
                         weight = target_weights.get(asset, 0.0)
                         # Recalculate stablecoin weight if it wasn't explicit
                         if asset == STABLECOIN_ASSET and asset not in target_weights:
                              risky_weight_sum = sum(target_weights.get(r_asset, 0.0) for r_asset in risky_assets)
                              weight = max(0, 1.0 - risky_weight_sum) # Ensure non-negative

                         target_asset_value_post_fee = value_after_commission * weight
                         asset_price = current_prices.get(asset, 0)
                         if asset_price > 1e-9:
                              new_holdings[asset] = target_asset_value_post_fee / asset_price
                         else:
                              new_holdings[asset] = 0.0
                         final_value_check += target_asset_value_post_fee

                    # Update portfolio state
                    final_day_value = value_after_commission # Value after rebalance & fees
                    current_holdings = new_holdings # Update holdings for next day
                    
                    # Debug print for rebalance
                    # print(f"Rebalanced {strategy} on {day_str}. Value: {final_day_value:.2f}, Commission: {commission_cost:.2f}")
                    
                else: # Rebalance target weights failed to calculate (Markowitz error, Oracle error etc.)
                    print(f"Skipped rebalance for {strategy} on {day_str} due to weight calculation failure.")
                    final_day_value = current_value_before_rebalance # Keep value as is

            # E. Handle Buy & Hold (value changes, no rebalance)
            elif strategy == "Buy & Hold":
                 # Value calculated at the start of the loop using initial holdings
                 final_day_value = current_value_before_rebalance 
                 # current_holdings already set to initial_holdings at the start of the loop for B&H
                 pass # No further action needed here for B&H

            # F. Store results for the day
            # <<< Use the strategy name directly as the key (it's already correct) >>>
            portfolio_values[strategy][current_day] = final_day_value
            # Ensure B&H stores its constant initial holdings
            if strategy == "Buy & Hold":
                 full_holdings_history[strategy][current_day] = initial_holdings[strategy].copy()
            else:
                 full_holdings_history[strategy][current_day] = current_holdings # Store final holdings for next day

    # 6. Process Results
    print("Processing simulation results...")
    try:
        results_df = pd.DataFrame(portfolio_values)
        results_df.index.name = 'Date'
        # Ensure index is DatetimeIndex
        if not isinstance(results_df.index, pd.DatetimeIndex):
            results_df.index = pd.to_datetime(results_df.index)
        results_df.sort_index(inplace=True)

        # Calculate Summary Metrics 
        summary_metrics = {}
        # Use ffill first on results_df to handle potential intermediate NaNs before pct_change
        results_df_filled = results_df.ffill()
        returns_df = results_df_filled.pct_change() # Use simple returns for consistency with other calcs if needed

        final_values = results_df.iloc[-1] # Use original results for final value
        # Ensure initial value is valid and > 0 for return calculation
        initial_values = results_df.iloc[0].replace(0, np.nan) # Replace 0 with NaN temporarily
        total_return = (final_values / initial_values) - 1
        total_return = total_return.fillna(-1) # If initial was NaN (or 0), return is -100%

        # Calculate annualized return (geometric mean)
        years = (results_df.index[-1] - results_df.index[0]).days / 365.25
        annualized_return = pd.Series(np.nan, index=final_values.index)
        if years > 1e-6: # Avoid division by zero if duration is effectively zero
             # Calculate based on valid initial and final values
             valid_return_calc = (final_values > 1e-9) & (initial_values.notna()) & (initial_values > 1e-9)
             # Use np.power for safe exponentiation, handle base <= 0
             base = final_values[valid_return_calc] / initial_values[valid_return_calc]
             # Set return to 0 if base is non-positive
             base[base <= 0] = 1 # Treat non-positive growth as 0 return for calculation
             annualized_return_calc = np.power(base, (1/years)) - 1
             # Create a series with NaNs and fill with calculated values
             annualized_return.loc[valid_return_calc] = annualized_return_calc
             # Fill remaining NaNs (where calculation wasn't valid) with 0 or appropriate value
             annualized_return.fillna(0, inplace=True) # Default to 0 if calculation failed
        else:
             annualized_return = pd.Series(0.0, index=final_values.index) # Zero duration = zero annualized return
        
        # Годовая волатильность (по дневным лог. доходностям)
        daily_values = results_df.resample('D').last().ffill().dropna(how='all') # Resample to daily
        annualized_volatility = pd.Series(np.nan, index=final_values.index) # Initialize with NaNs

        if len(daily_values) > 1:
            # Calculate log returns, handle potential errors
            log_returns_df = np.log(daily_values.astype(float) / daily_values.shift(1).astype(float))
            # Replace inf/-inf with NaN
            log_returns_df = log_returns_df.replace([np.inf, -np.inf], np.nan)

            # Calculate std dev only where there are enough finite points
            for col in log_returns_df.columns:
                 valid_returns = log_returns_df[col].dropna()
                 if len(valid_returns) > 1:
                      std_dev_daily = valid_returns.std()
                      # Use 252 trading days for annualization, consistent with portfolio_rebalance.py
                      annualized_volatility.loc[col] = std_dev_daily * np.sqrt(252)
        # Fill any remaining NaNs (e.g., for constant value series) with 0
        annualized_volatility.fillna(0.0, inplace=True)

        # Коэффициент Шарпа (используем annual_risk_free_rate = bank_apr)
        sharpe_ratio = pd.Series(np.nan, index=final_values.index)
        # Calculate where both return and volatility are valid
        valid_sharpe_calc = annualized_return.notna() & annualized_volatility.notna()
        
        # Case 1: Volatility is near zero
        zero_vol_mask = valid_sharpe_calc & (annualized_volatility < 1e-9)
        sharpe_ratio.loc[zero_vol_mask & (annualized_return > bank_apr)] = np.inf
        sharpe_ratio.loc[zero_vol_mask & (annualized_return < bank_apr)] = -np.inf
        sharpe_ratio.loc[zero_vol_mask & np.isclose(annualized_return, bank_apr)] = 0.0

        # Case 2: Volatility is positive
        pos_vol_mask = valid_sharpe_calc & (annualized_volatility >= 1e-9)
        sharpe_ratio.loc[pos_vol_mask] = (annualized_return.loc[pos_vol_mask] - bank_apr) / annualized_volatility.loc[pos_vol_mask]

        # Fill remaining NaNs with 0
        sharpe_ratio.fillna(0.0, inplace=True)

        summary_metrics['Final Value ($)'] = final_values
        summary_metrics['Total Return (%)'] = total_return * 100
        summary_metrics['Annualized Return (%)'] = annualized_return * 100
        summary_metrics['Annualized Volatility (%)'] = annualized_volatility * 100
        summary_metrics['Sharpe Ratio'] = sharpe_ratio

        metrics_summary_df = pd.DataFrame(summary_metrics).round(2)
        # Format columns using .map() and string.format()
        if 'Final Value ($)' in metrics_summary_df.columns:
             metrics_summary_df['Final Value ($)'] = metrics_summary_df['Final Value ($)'].map("{:,.2f}".format)
        for col in ['Total Return (%)', 'Annualized Return (%)', 'Annualized Volatility (%)']:
             if col in metrics_summary_df.columns:
                  metrics_summary_df[col] = metrics_summary_df[col].map("{:.2f}%".format)
        if 'Sharpe Ratio' in metrics_summary_df.columns:
             metrics_summary_df['Sharpe Ratio'] = metrics_summary_df['Sharpe Ratio'].map("{:.2f}".format)

    except Exception as result_e:
        print(f"Error processing results: {result_e}")
        traceback.print_exc()
        return None, None

    # 7. Create Plot
    print("Generating plot...")
    fig = go.Figure()
    fig.update_layout(
         title='Исследование: Динамика стоимости гипотетических портфелей',
         xaxis_title='Дата',
         yaxis_title='Стоимость портфеля (USD)',
         template='plotly_dark',
         hovermode='x unified'
    )
    # Use the (potentially renamed) strategies list for plotting
    plot_strategies = results_df.columns # Get actual columns after potential rename
    for strategy in plot_strategies:
         fig.add_trace(go.Scatter(
              x=results_df.index, 
              y=results_df[strategy], 
              mode='lines', 
              name=strategy,
              hovertemplate = f'<b>{strategy}</b><br>Дата: %{{x|%Y-%m-%d}}<br>Стоимость: $ %{{y:,.2f}}<extra></extra>'
         ))

    print("--- Hypothetical Analysis (Initial Investment) Finished ---")
    return metrics_summary_df, fig

# Example Usage Block (commented out)
# if __name__ == '__main__':
#     # Define assets for hypothetical test
#     assets_hypo = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] # Example
#     start = '2023-01-01'
#     end = '2023-12-31'
#     initial = 10000
# 
#     # Assume load_price_data() works and loads daily data into combined_data.csv or similar
#     # Assume DRL models exist in notebooks/trained_models
#     
#     print(f"Running example: Initial ${initial}, Assets: {assets_hypo}, Period: {start} to {end}")
# 
#     metrics, fig = run_hypothetical_analysis(
#         initial_investment=initial,
#         hypothetical_assets=assets_hypo,
#         start_date_str=start,
#         end_date_str=end,
#         data_path="data", # Adjust if needed
#         bank_apr=0.05, # 5% APR example
#         commission_rate=0.001, # 0.1% commission
#         rebalance_interval_days=30,
#         drl_rebalance_interval_days=30,
#         drl_models_dir="notebooks/trained_models" # Adjust if needed
#     )
# 
#     if fig and metrics is not None:
#         print("--- Results Summary ---")
#         print(metrics)
#         # To display the plot locally:
#         # import plotly.io as pio
#         # pio.show(fig) 
#         print("Plot generated successfully (use pio.show(fig) to display).")
#     else:
#         print("Analysis failed or returned no results.") 