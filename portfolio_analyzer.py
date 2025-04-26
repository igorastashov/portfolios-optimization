# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import OrderedDict
import warnings
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, SAC, DDPG
import json
import traceback # Added for error handling
import ta # <<< Import ta library

# Define INDICATORS globally or pass as argument if needed
# <<< Rename dx_30 to adx and use ADXIndicator >>>
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "adx", "close_30_sma", "close_60_sma"]
STABLECOIN_ASSET = 'USDCUSDT' # Define globally for now

# --- FeatureEngineer Class (Simplified version or import) ---
# (Assuming FinRL is not reliably installed in the Streamlit env)
class FeatureEngineer:
    def __init__(self, use_technical_indicator=True, tech_indicator_list=INDICATORS, use_turbulence=False, user_defined_feature=False):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        print("Basic FeatureEngineer initialized.")

    def preprocess_data(self, df):
        """Adds technical indicators to the dataframe.
        :param df: (df) pandas dataframe with columns 'date', 'tic', 'close', 'high', 'low', 'open', 'volume'
        :return: (df) pandas dataframe
        """
        # Ensure required columns exist for TA calculations
        required_ta_cols = ['date', 'tic', 'close', 'high', 'low', 'open', 'volume']
        missing_cols = [col for col in required_ta_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"FeatureEngineer expects the following columns for TA: {required_ta_cols}. Missing: {missing_cols}")

        if self.use_technical_indicator:
            print(f"Adding {len(self.tech_indicator_list)} technical indicators...")
            df = df.copy()
            grouped_df = df.groupby('tic', group_keys=False)

            def apply_ta_indicators(group):
                # SMA
                if 'close_30_sma' in self.tech_indicator_list:
                    group['close_30_sma'] = ta.trend.sma_indicator(group['close'], window=30, fillna=True)
                if 'close_60_sma' in self.tech_indicator_list:
                    group['close_60_sma'] = ta.trend.sma_indicator(group['close'], window=60, fillna=True)
                # MACD
                if 'macd' in self.tech_indicator_list:
                    group['macd'] = ta.trend.macd(group['close'], fillna=True)
                # RSI
                if 'rsi_30' in self.tech_indicator_list:
                    group['rsi_30'] = ta.momentum.rsi(group['close'], window=30, fillna=True)
                # CCI
                if 'cci_30' in self.tech_indicator_list:
                    group['cci_30'] = ta.trend.cci(group['high'], group['low'], group['close'], window=30, fillna=True)

                # Bollinger Bands
                if 'boll_ub' in self.tech_indicator_list:
                    group['boll_ub'] = ta.volatility.bollinger_hband(group['close'], window=20, window_dev=2, fillna=True)
                if 'boll_lb' in self.tech_indicator_list:
                    group['boll_lb'] = ta.volatility.bollinger_lband(group['close'], window=20, window_dev=2, fillna=True)
                
                # Average Directional Index (ADX) - formerly dx_30
                if 'adx' in self.tech_indicator_list:
                    # Use the ADXIndicator class
                    adx_indicator = ta.trend.ADXIndicator(high=group['high'], low=group['low'], close=group['close'], window=14, fillna=True) # Default window 14 for ADX, adjust if needed
                    # Assign the ADX value
                    group['adx'] = adx_indicator.adx()
                    # If you need +DI or -DI:
                    # group['adx_pos'] = adx_indicator.adx_pos()
                    # group['adx_neg'] = adx_indicator.adx_neg()

                return group

            df = grouped_df.apply(apply_ta_indicators)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            print("Technical indicators added.")

        # Re-enable dropna as it was likely present during training
        df.dropna(inplace=True)

        return df


# --- StockPortfolioEnv Class (Copied and slightly adapted) ---
class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                df,
                stock_dim,
                hmax, # Unused
                initial_amount,
                transaction_cost_pct, # Unused in this env version
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None, # Unused
                lookback=24,
                day = 0):
        self.day = day
        self.lookback=lookback
        self.df = df # Expects preprocessed df with integer index 'day_index' and 'cov_list'
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.Box(low = 0, high = 1, shape = (self.action_space,))
        obs_shape = (self.stock_dim + len(self.tech_indicator_list), self.stock_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        if not pd.api.types.is_integer_dtype(self.df.index):
             raise ValueError("StockPortfolioEnv expects an integer index (days).")
        if not self.df.index.is_monotonic_increasing:
             print("Warning: StockPortfolioEnv DataFrame index is not sorted.")
             # Attempt to sort if possible
             try:
                 self.df = self.df.sort_index()
             except Exception as e:
                 raise ValueError(f"StockPortfolioEnv DataFrame index is not sorted and sorting failed: {e}")


        try:
            # Using .iloc because index is integer days starting from 0
            self.data = self.df.loc[self.day,:]
            if isinstance(self.data, pd.DataFrame) and self.data.empty: # Check if DataFrame is empty
                raise IndexError(f"No data found for day index {self.day}")
            elif isinstance(self.data, pd.Series) and self.data.empty: # Check if Series is empty
                 raise IndexError(f"No data found for day index {self.day}")
        except KeyError:
             max_idx = self.df.index.max() if not self.df.empty else 'N/A'
             raise KeyError(f"Day index {self.day} not found in DataFrame index. Max index: {max_idx}")
        except IndexError as e:
             raise IndexError(f"Error accessing data for day index {self.day}: {e}")


        # Handle potential single-row DataFrame for a given day
        current_day_data = self.df.loc[self.df.index == self.day]
        self.covs = current_day_data['cov_list'].iloc[0] # Expects cov_list column, get from first row

        # Construct state: append indicators as rows below cov matrix
        indicator_values = np.array([current_day_data[tech].values.tolist() for tech in self.tech_indicator_list]).T # Transpose needed

        if indicator_values.shape[0] != self.stock_dim: # Check rows now due to transpose
             raise ValueError(f"Indicator rows ({indicator_values.shape[0]}) != stock_dim ({self.stock_dim}) on day {self.day}")

        # Append indicators below covariance matrix
        self.state =  np.append(np.array(self.covs), indicator_values.T, axis=0) # Transpose back for stacking


        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        # Store normalized weights (initial state: equal weights)
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        # Store actual dates
        self.date_memory = [current_day_data['date'].iloc[0]] # Get date from first row


    def step(self, actions):
        max_day_index = len(self.df.index.unique()) -1
        self.terminal = self.day >= max_day_index # Use max_day_index

        if self.terminal:
            return self.state, 0, self.terminal, False, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory_data = self.df.loc[self.df.index == self.day] # Data from previous step

            self.day += 1
            try:
                 current_day_data = self.df.loc[self.df.index == self.day]
                 if current_day_data.empty: raise IndexError(f"No data for day index {self.day}")
            except (KeyError, IndexError):
                  print(f"Error: Day index {self.day} not found in DRL DataFrame index during step. Terminal.")
                  self.terminal = True
                  # Return previous state? Or a dummy state? Let's return previous state.
                  return self.state, 0, self.terminal, False, {}

            self.covs = current_day_data['cov_list'].iloc[0]
            # Construct state for next step
            indicator_values = np.array([current_day_data[tech].values.tolist() for tech in self.tech_indicator_list]).T

            if indicator_values.shape[0] != self.stock_dim:
                 raise ValueError(f"Indicator rows ({indicator_values.shape[0]}) != stock_dim ({self.stock_dim}) in step for day {self.day}")

            self.state = np.append(np.array(self.covs), indicator_values.T, axis=0)

            # Calculate portfolio return using previous weights and current price changes
            current_prices = current_day_data.close.values
            last_prices = last_day_memory_data.close.values

            individual_returns = np.divide(current_prices - last_prices, last_prices,
                                        out=np.zeros_like(current_prices, dtype=float),
                                        where=last_prices!=0)

            # Use weights from the *previous* step (before appending new ones)
            portfolio_return = sum(individual_returns * self.actions_memory[-1]) # Use previous weights

            # Update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            # Reward could be Sharpe, return, or something else. Keeping simple return.
            self.reward = portfolio_return * self.reward_scaling

            self.portfolio_value = new_portfolio_value

            # Save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(current_day_data['date'].iloc[0])
            self.asset_memory.append(new_portfolio_value)

            return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None):
        if seed is not None:
            self._seed(seed)

        self.day = 0
        try:
             current_day_data = self.df.loc[self.df.index == self.day]
             if current_day_data.empty: raise IndexError(f"No data for day index {self.day}")
        except (KeyError, IndexError):
             max_idx = self.df.index.max() if not self.df.empty else 'N/A'
             raise KeyError(f"Day index {self.day} (reset) not found in DataFrame index. Max index: {max_idx}")

        self.covs = current_day_data['cov_list'].iloc[0]
        indicator_values = np.array([current_day_data[tech].values.tolist() for tech in self.tech_indicator_list]).T

        if indicator_values.shape[0] != self.stock_dim:
             raise ValueError(f"Indicator rows ({indicator_values.shape[0]}) != stock_dim ({self.stock_dim}) in reset")

        self.state = np.append(np.array(self.covs), indicator_values.T, axis=0)

        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim] * self.stock_dim]
        self.date_memory = [current_day_data['date'].iloc[0]]
        self.terminal = False

        return self.state, {}

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        actions = np.array(actions)
        stable_actions = actions - np.max(actions)
        numerator = np.exp(stable_actions)
        denominator = np.sum(numerator)
        if denominator == 0 or not np.isfinite(denominator) or denominator < 1e-9: # Added threshold
            print("Warning: Softmax denominator issue. Using uniform weights.")
            return np.ones_like(actions) / self.stock_dim
        softmax_output = numerator / denominator
        return softmax_output / np.sum(softmax_output)

    def save_asset_memory(self):
        min_len = min(len(self.date_memory), len(self.portfolio_return_memory))
        df_account_value = pd.DataFrame({
             'date': self.date_memory[:min_len],
             'daily_return': self.portfolio_return_memory[:min_len]
         })
        return df_account_value

    def save_action_memory(self):
        min_len = min(len(self.date_memory), len(self.actions_memory))
        date_list = self.date_memory[:min_len]
        action_list = self.actions_memory[:min_len]
        df_actions = pd.DataFrame(action_list)

        # Try to get ticker names from the original df used to init the env
        if hasattr(self, 'df') and not self.df.empty and 'tic' in self.df.columns:
             tic_names = self.df['tic'].unique()
             if len(tic_names) == df_actions.shape[1]:
                 df_actions.columns = tic_names
             else:
                  print(f"Warning: Action columns ({df_actions.shape[1]}) != unique tics ({len(tic_names)}). Using generic names.")
                  df_actions.columns = [f'action_{i}' for i in range(df_actions.shape[1])]
        else:
             df_actions.columns = [f'action_{i}' for i in range(df_actions.shape[1])]

        df_actions['date'] = date_list
        df_actions = df_actions.set_index('date')
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# --- DRLAgent Class (Simplified - only prediction needed) ---
class DRLAgent:
    @staticmethod
    def DRL_prediction(model, environment):
        test_env = environment
        try:
             obs, _ = test_env.reset()
        except Exception as e:
             print(f"Error during test_env.reset(): {e}")
             return pd.DataFrame(columns=['date', 'daily_return']), pd.DataFrame()

        terminated = False
        truncated = False
        i = 0
        max_steps = len(environment.df.index.unique()) # Limit steps to avoid infinite loop
        while not (terminated or truncated) and i < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            try:
                 obs, rewards, terminated, truncated, info = test_env.step(action)
            except Exception as e:
                 print(f"Error during test_env.step() at step {i}: {e}")
                 traceback.print_exc()
                 break
            i += 1
        if i >= max_steps:
            print(f"Warning: DRL_prediction reached max_steps ({max_steps}).")


        df_daily_return = test_env.save_asset_memory()
        df_actions = test_env.save_action_memory()
        df_daily_return['date'] = pd.to_datetime(df_daily_return['date'])
        return df_daily_return, df_actions

# --- Helper Function: Preprocess Asset Data ---
def preprocess_asset_data(filepath, asset_name, stablecoin_name):
    try:
        try:
            df = pd.read_csv(filepath)
            date_col = None
            for col in df.columns:
                if col.strip().lower() == 'open time':
                    date_col = col; break
                elif col.strip().lower() == 'date':
                    date_col = col; break
            if date_col is None:
                print(f"  Error: Date column ('Open time' or 'date') not found in {asset_name}.")
                return pd.DataFrame()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception as parse_e:
            print(f"  Error reading CSV or parsing date for {asset_name}: {parse_e}")
            return pd.DataFrame()

        required_cols_map_keys = [date_col, 'Open', 'High', 'Low', 'Close', 'Volume']
        required_cols_map_vals = ['date', 'open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.strip().str.lower() # Standardize incoming column names to lower

        rename_actual = {}
        missing_cols = []
        current_cols_lower = df.columns.tolist()

        for req_val, req_key in zip(required_cols_map_vals, required_cols_map_keys):
             # Check if the standardized lowercase value exists
             if req_val in current_cols_lower:
                 rename_actual[req_val] = req_val # Already correct
             # Check if the original key (lowercased) exists
             elif req_key.lower() in current_cols_lower:
                  rename_actual[req_key.lower()] = req_val # Need to rename
             else:
                 missing_cols.append(req_val) # Truly missing

        if missing_cols:
            print(f"  Error: Missing required columns in {asset_name} after check: {missing_cols}. Expected: {required_cols_map_vals}")
            return pd.DataFrame()

        # Select and rename columns based on the final map
        df = df[list(rename_actual.keys())].copy()
        df.rename(columns=rename_actual, inplace=True)

        df.dropna(subset=['date'], inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        if df.empty: return pd.DataFrame()

        df = df.sort_values(by='date')
        df = df.drop_duplicates(subset=['date'], keep='first')

        if asset_name == stablecoin_name:
            close_orig = df['close'].copy()
            df['close'] = np.where((df['close'] > 1.02) | (df['close'] < 0.98), 1.0, df['close'])
            mask_changed = (df['close'] == 1.0) & (close_orig != 1.0)
            df.loc[mask_changed, ['open', 'high', 'low']] = 1.0

        df['tic'] = asset_name
        return df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]

    except FileNotFoundError:
        print(f"  WARNING: File not found for {asset_name}: {filepath}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ERROR processing {asset_name}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# --- Main Analysis Function ---
def run_portfolio_analysis(
    transactions_df: pd.DataFrame, # User's transactions
    start_date_str: str,           # Start date for analysis 'YYYY-MM-DD'
    end_date_str: str,             # End date for analysis 'YYYY-MM-DD'
    data_path: str,                # Path to historical data CSVs
    drl_models_dir: str,           # Path to trained DRL models directory
    bank_apr: float = 0.20,
    commission_rate: float = 0.001,
    rebalance_interval_days: int = 20,
    drl_rebalance_interval_days: int = 20,
    # Optional: Allow passing pre-loaded models? For now, load inside.
    ):
    """
    Runs the portfolio comparison analysis based on user transactions and settings.

    Args:
        transactions_df: DataFrame with user transactions.
                         Expected columns: 'Дата_Транзакции', 'Актив', 'Тип', 'Количество', 'Общая стоимость'.
        start_date_str: Analysis start date string ('YYYY-MM-DD').
        end_date_str: Analysis end date string ('YYYY-MM-DD').
        data_path: Path to directory containing historical asset data CSV files (e.g., 'BTCUSDT_hourly_data.csv').
        drl_models_dir: Path to the directory containing trained DRL model .zip files.
        bank_apr: Annual percentage rate for the Bank Deposit strategy.
        commission_rate: Commission rate for Rebalance and Perfect Foresight strategies.
        rebalance_interval_days: Rebalancing interval for non-DRL strategies (days).
        drl_rebalance_interval_days: Rebalancing interval for DRL strategies (days).

    Returns:
        tuple: (results_df_display, fig)
               - results_df_display: DataFrame with formatted performance metrics.
               - fig: Matplotlib figure object for the comparison plot.
               Returns (None, None) if an error occurs during analysis.
    """
    try:
        # --- Setup & Initial Data Validation ---
        print("--- Starting Portfolio Analysis ")
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Validate and parse dates
        try:
            today = pd.Timestamp(end_date_str)
            start_date_history = pd.Timestamp(start_date_str)
            # Calculate days_history based on input dates
            days_history = (today - start_date_history).days
            if days_history <= 0:
                 raise ValueError("End date must be after start date.")
            print(f"Analysis Period: {start_date_history.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')} ({days_history} days)")
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return None, None

        portfolio_df = transactions_df.copy()

        # --- Validate Input Transactions DataFrame ---
        required_cols = ['Дата_Транзакции', 'Актив', 'Тип', 'Количество', 'Общая стоимость']
        if not all(col in portfolio_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in portfolio_df.columns]
            print(f"ERROR: Input transactions_df is missing required columns: {missing}. Expected: {required_cols}")
            return None, None

        # Ensure correct data types
        try:
            portfolio_df['Дата_Транзакции'] = pd.to_datetime(portfolio_df['Дата_Транзакции'], errors='coerce')
            portfolio_df.dropna(subset=['Дата_Транзакции'], inplace=True) # Drop invalid dates
            # Quantity and Общая стоимость are handled later during price lookup/calculation
        except Exception as e:
            print(f"Error converting transaction dates: {e}")
            return None, None

        portfolio_df = portfolio_df.sort_values(by='Дата_Транзакции').reset_index(drop=True)

        valid_types = ['Покупка', 'Продажа']
        if not portfolio_df['Тип'].isin(valid_types).all():
            invalid = portfolio_df[~portfolio_df['Тип'].isin(valid_types)]['Тип'].unique()
            print(f"ERROR: Invalid transaction types found: {invalid}. Allowed: {valid_types}")
            return None, None

        # --- Determine Assets and Load Historical Data ---
        ALL_PORTFOLIO_ASSETS = portfolio_df['Актив'].unique().tolist()
        # Define DRL assets (should match models) - consider making this an argument?
        DRL_TRAINING_ASSETS = ['APTUSDT', 'CAKEUSDT', 'HBARUSDT', 'JUPUSDT', 'PEPEUSDT', 'STRKUSDT', 'USDCUSDT']
        ALL_REQUIRED_ASSETS_FOR_LOADING = list(set(ALL_PORTFOLIO_ASSETS + DRL_TRAINING_ASSETS + [STABLECOIN_ASSET]))

        print(f"\n--- Loading and Preprocessing Data ({len(ALL_REQUIRED_ASSETS_FOR_LOADING)} assets) ---")
        all_data_frames = []
        missing_essential_assets = []
        for asset in ALL_REQUIRED_ASSETS_FOR_LOADING:
            filename = f"{asset}_hourly_data.csv" # Assuming standard naming
            filepath = os.path.join(data_path, filename)
            print(f"  Loading: {asset} from {filename}")
            df_processed = preprocess_asset_data(filepath, asset, STABLECOIN_ASSET)
            if not df_processed.empty:
                all_data_frames.append(df_processed)
            else:
                if asset in ALL_PORTFOLIO_ASSETS or asset in DRL_TRAINING_ASSETS or asset == STABLECOIN_ASSET:
                     print(f"  -> CRITICAL WARNING: Failed to load data for {asset}. Analysis might be inaccurate or fail.")
                     missing_essential_assets.append(asset)

        if not all_data_frames:
            print("ERROR: No historical data loaded for analysis!")
            return None, None
        if any(asset in missing_essential_assets for asset in ALL_PORTFOLIO_ASSETS):
            print("ERROR: Missing data for one or more assets from the transaction portfolio. Cannot proceed.")
            return None, None
        # Allow proceeding if only non-essential DRL assets are missing, but DRL might fail later

        raw_historical_data = pd.concat(all_data_frames, ignore_index=True)

        # --- Prepare Historical Data ---
        print("Setting 'date' as index and sorting...")
        if 'date' not in raw_historical_data.columns:
            print("ERROR: 'date' column missing after concatenation.")
            return None, None
        raw_historical_data['date'] = pd.to_datetime(raw_historical_data['date'], errors='coerce')
        raw_historical_data.dropna(subset=['date'], inplace=True)
        raw_historical_data = raw_historical_data.set_index('date')
        raw_historical_data.sort_index(inplace=True)
        print("Raw data indexing complete.")

        print("\n--- Filtering Historical Data for Analysis Period ---")
        # Use the exact start/end times from input strings
        historical_data_filtered = raw_historical_data[(raw_historical_data.index >= start_date_history) & (raw_historical_data.index <= today)].copy()

        if historical_data_filtered.empty:
            print(f"ERROR: No historical data found in the specified date range: {start_date_history} - {today}!")
            return None, None
        print(f"Filtered {len(historical_data_filtered)} records (all assets).")

        historical_data_filtered = historical_data_filtered.reset_index() # Get 'date' back

        # --- Create Pivot Table ---
        print("Creating pivot table for prices...")
        historical_prices_pivot = historical_data_filtered.set_index('date').pivot_table(index='date', columns='tic', values='close')
        historical_prices_pivot.columns = [f'{col}_Price' for col in historical_prices_pivot.columns]
        print(f"Pivot table created: {historical_prices_pivot.shape}")

        print("Filling NaNs in price pivot table...")
        if historical_prices_pivot.isnull().values.any():
            historical_prices_pivot = historical_prices_pivot.ffill().bfill()
            if historical_prices_pivot.isnull().values.any():
                cols_with_nan = historical_prices_pivot.columns[historical_prices_pivot.isnull().any()].tolist()
                print(f"  Warning: NaNs remain in pivot table after fill: {cols_with_nan}. Strategies using these might be affected.")
            else:
                print("  NaNs filled.")
        else:
            print("  No NaNs found in pivot table.")

        stablecoin_price_col = f'{STABLECOIN_ASSET}_Price'
        has_stablecoin_data = stablecoin_price_col in historical_prices_pivot.columns and not historical_prices_pivot[stablecoin_price_col].isnull().all()

        # --- Process Transactions (Lookup Prices, Calculate Quantities) ---
        print("\nProcessing transactions (finding prices, calculating quantities)...")
        rows_to_drop = []
        portfolio_df['Actual_Transaction_Time_Index'] = pd.NaT
        portfolio_df['Transaction_Price_Actual'] = np.nan

        for index, row in portfolio_df.iterrows():
            asset = row['Актив']
            transaction_date = row['Дата_Транзакции']
            price_col = f'{asset}_Price'
            trans_type = row['Тип']

            if price_col not in historical_prices_pivot.columns:
                print(f"  Warning: No price data for {asset} ({price_col}) in pivot table. Transaction ID {row.get('ID', index)} will be dropped.")
                rows_to_drop.append(index)
                continue

            relevant_times = historical_prices_pivot.index[historical_prices_pivot.index >= transaction_date]
            if not relevant_times.empty:
                actual_transaction_time_index = relevant_times[0]
                if actual_transaction_time_index in historical_prices_pivot.index:
                     transaction_price = historical_prices_pivot.loc[actual_transaction_time_index, price_col]
                     if pd.notna(transaction_price) and transaction_price > 0:
                         portfolio_df.loc[index, 'Transaction_Price_Actual'] = transaction_price
                         portfolio_df.loc[index, 'Actual_Transaction_Time_Index'] = actual_transaction_time_index
                         if trans_type == 'Покупка' and pd.isna(row['Количество']):
                             if pd.notna(row['Общая стоимость']) and row['Общая стоимость'] > 0:
                                 portfolio_df.loc[index, 'Количество'] = row['Общая стоимость'] / transaction_price
                             else:
                                 print(f"  Warning: Cannot calculate Quantity for Buy {asset} (ID {row.get('ID', index)}): Общая стоимость missing or invalid. Transaction dropped.")
                                 rows_to_drop.append(index)
                         elif trans_type == 'Продажа' and pd.isna(row['Количество']):
                              print(f"  Warning: Quantity missing for Sell {asset} (ID {row.get('ID', index)}). Transaction dropped.")
                              rows_to_drop.append(index)
                     else:
                         print(f"  Warning: Invalid price ({transaction_price}) for {asset} (ID {row.get('ID', index)}) at {actual_transaction_time_index}. Transaction dropped.")
                         rows_to_drop.append(index)
                else:
                     print(f"  Logic Error: Time {actual_transaction_time_index} not found in index for {asset}. Transaction dropped.")
                     rows_to_drop.append(index)
            else:
                print(f"  Warning: No historical data after transaction date {transaction_date} for {asset} (ID {row.get('ID', index)}). Transaction dropped.")
                rows_to_drop.append(index)

        if rows_to_drop:
            print(f"  Dropping {len(rows_to_drop)} transactions due to errors/missing data...")
            portfolio_df.drop(rows_to_drop, inplace=True)
            portfolio_df.reset_index(drop=True, inplace=True)

        portfolio_df.dropna(subset=['Количество'], inplace=True)
        portfolio_df = portfolio_df[portfolio_df['Количество'] > 0]
        portfolio_df.reset_index(drop=True, inplace=True)

        print(f"Transaction processing complete. Valid transactions: {len(portfolio_df)}. ")
        if len(portfolio_df) == 0:
            print("\nERROR: No valid transactions remain after price/quantity check!")
            return None, None

        first_investment_time = portfolio_df[portfolio_df['Тип'] == 'Покупка']['Actual_Transaction_Time_Index'].min()
        if pd.isna(first_investment_time):
            print("ERROR: Cannot determine first purchase time to start simulation!")
            return None, None
        print(f"First actual investment (purchase) considered at: {first_investment_time}")

        # --- DRL Preprocessing & Prediction ---
        print("\n--- Preparing Data and Running DRL Model Predictions ---")
        # 1. Filter data for DRL training assets
        drl_assets_to_load = list(set(DRL_TRAINING_ASSETS + [STABLECOIN_ASSET])) # Ensure stablecoin is included
        drl_input_df = historical_data_filtered[historical_data_filtered['tic'].isin(drl_assets_to_load)].copy()
        drl_input_df = drl_input_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        print(f"Data for FeatureEngineer (DRL assets): {drl_input_df.shape}")
        if drl_input_df.empty:
            print("ERROR: No historical data found for DRL training assets.")
            return None, None

        # 2. Run FeatureEngineer
        fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS)
        try:
            print("Running FeatureEngineer (DRL assets)...")
            drl_processed_df = fe.preprocess_data(drl_input_df)
            print(f"FeatureEngineer complete. Data shape: {drl_processed_df.shape}")
            drl_processed_df = drl_processed_df.dropna() # Drop NaNs from indicators
            print(f"Shape after dropna: {drl_processed_df.shape}")
            if drl_processed_df.empty: raise ValueError("DataFrame empty after FeatureEngineer and dropna.")
        except Exception as e:
            print(f"ERROR during FeatureEngineer: {e}")
            traceback.print_exc()
            return None, None

        # 3. Add Covariance Matrix
        print("Adding covariance matrix (DRL assets)...")
        drl_processed_df = drl_processed_df.sort_values(['date', 'tic'])
        cov_list = []
        lookback = 24
        unique_dates = drl_processed_df['date'].unique()
        num_unique_dates = len(unique_dates)

        if num_unique_dates <= lookback:
             print(f"ERROR: Insufficient unique dates ({num_unique_dates}) for lookback ({lookback}) for DRL covariance.")
             return None, None

        date_to_int_map = {date: i for i, date in enumerate(unique_dates)}
        drl_processed_df['day_index'] = drl_processed_df['date'].map(date_to_int_map)

        # Use DRL_TRAINING_ASSETS for cov matrix dimension consistency
        drl_stock_dim_for_cov = len(DRL_TRAINING_ASSETS)
        print(f"Using {drl_stock_dim_for_cov} assets for covariance matrix shape: {DRL_TRAINING_ASSETS}")

        for i in range(lookback, num_unique_dates):
            data_lookback = drl_processed_df[(drl_processed_df['day_index'] >= i - lookback) & (drl_processed_df['day_index'] <= i - 1)]
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            # Ensure all DRL training assets are columns before pct_change
            price_lookback = price_lookback.reindex(columns=DRL_TRAINING_ASSETS, fill_value=np.nan)
            return_lookback = price_lookback.pct_change().dropna(how='all') # Drop rows where all returns are NaN

            # Calculate cov only if we have enough data points
            if len(return_lookback) >= drl_stock_dim_for_cov: # Use the defined dimension
                 # Drop assets (columns) that are all NaN before calculating cov
                 return_lookback_valid = return_lookback.dropna(axis=1, how='all')
                 if not return_lookback_valid.empty and return_lookback_valid.shape[1] > 1: # Need at least 2 assets
                     covs = return_lookback_valid.cov().reindex(index=DRL_TRAINING_ASSETS, columns=DRL_TRAINING_ASSETS).fillna(0).values
                     cov_list.append(covs)
                 else: # Not enough valid assets/data for cov calculation
                      cov_list.append(np.full((drl_stock_dim_for_cov, drl_stock_dim_for_cov), 0.0)) # Use zeros or NaNs? Using zeros.
            else:
                 cov_list.append(np.full((drl_stock_dim_for_cov, drl_stock_dim_for_cov), 0.0)) # Use zeros if not enough history

        cov_dates = unique_dates[lookback:]
        if len(cov_dates) != len(cov_list):
            print(f"Warning: Mismatch between cov_dates ({len(cov_dates)}) and cov_list ({len(cov_list)}). Trimming...")
            min_len = min(len(cov_dates), len(cov_list))
            cov_dates = cov_dates[:min_len]
            cov_list = cov_list[:min_len]

        if not cov_list:
             print("ERROR: Covariance list is empty after calculation.")
             return None, None

        df_cov = pd.DataFrame({'date': cov_dates, 'cov_list': cov_list})
        drl_processed_df = drl_processed_df.merge(df_cov, on='date', how='left')
        drl_processed_df['cov_list'] = drl_processed_df['cov_list'].ffill() # Forward fill cov matrix
        drl_processed_df = drl_processed_df.dropna(subset=['cov_list']) # Drop rows where cov matrix couldn't be calculated initially

        # Create final integer index
        if drl_processed_df.empty:
            print("ERROR: DRL DataFrame is empty after covariance processing.")
            return None, None
        drl_processed_df = drl_processed_df.sort_values(by='date')
        drl_processed_df['day_index'] = pd.factorize(drl_processed_df['date'])[0]
        drl_final_env_df = drl_processed_df.set_index('day_index')
        drl_final_env_df = drl_final_env_df.sort_index()

        if drl_final_env_df.empty:
            print("ERROR: DRL DataFrame for environment is empty after final processing.")
            return None, None
        print(f"Final data for DRL environment prepared: {drl_final_env_df.shape}")

        # 4. Define DRL Environment Kwargs
        stock_dimension_drl = len(DRL_TRAINING_ASSETS) # Dimension based on training assets
        state_space_drl = stock_dimension_drl

        env_kwargs_drl_pred = {
            "hmax": 100, "initial_amount": 1000000, "transaction_cost_pct": 0.0,
            "state_space": state_space_drl, "stock_dim": stock_dimension_drl,
            "tech_indicator_list": INDICATORS, "action_space": stock_dimension_drl,
            "reward_scaling": 1e-4, "lookback": lookback
        }

        # 5. Load DRL Models and Get Predictions
        print("\n--- Loading and Predicting with DRL Models ---")
        # Adjust model paths based on drl_models_dir parameter
        drl_model_paths = {
            name: os.path.join(drl_models_dir, f"trained_{name.lower()}.zip")
            for name in ["DDPG", "A2C", "PPO", "SAC"]
        }
        loaded_models = {}
        df_actions_all_models = {}
        trade_env_drl = None # Initialize env variable

        for model_name, model_path in drl_model_paths.items():
            print(f"\nProcessing model: {model_name} from {model_path}...")
            if not os.path.exists(model_path):
                print(f"  ERROR: Model file not found: {model_path}")
                continue
            try:
                model_class = globals()[model_name] # Get class by name
                model = model_class.load(model_path)
                print(f"  Model {model_name} loaded successfully.")
                loaded_models[model_name] = model

                if trade_env_drl is None: # Create env only once
                     print(f"  Creating DRL environment...")
                     trade_env_drl = StockPortfolioEnv(df=drl_final_env_df, **env_kwargs_drl_pred)
                     print("  DRL prediction environment created.")

                print(f"  Running DRL prediction for {model_name}...")
                _, df_actions = DRLAgent.DRL_prediction(model=model, environment=trade_env_drl)
                print(f"  Prediction {model_name} complete. Got {len(df_actions)} actions.")
                if not df_actions.empty:
                    df_actions_all_models[model_name] = df_actions
                else:
                    print(f"  Warning: Prediction {model_name} returned no actions.")
            except Exception as e:
                print(f"  ERROR during loading/prediction for model {model_name}: {e}")
                traceback.print_exc()

        if not df_actions_all_models:
            print("\nERROR: Failed to get predictions from any DRL model. Cannot proceed.")
            return None, None

        # --- Step 3: Simulate Strategy Performance ---
        print("\n--- Calculating Strategy Performance (Based on Transactions) ---")
        historical_data_final = historical_prices_pivot.copy()

        # Determine simulation time range
        sim_start_time = first_investment_time
        # Ensure end time covers analysis range AND last transaction
        sim_end_time_requested = today
        last_actual_transaction_time = portfolio_df['Actual_Transaction_Time_Index'].max()
        sim_end_time_effective = max(sim_end_time_requested, last_actual_transaction_time) if not pd.isna(last_actual_transaction_time) else sim_end_time_requested

        sim_data_index = historical_data_final.index[(historical_data_final.index >= sim_start_time) & (historical_data_final.index <= sim_end_time_effective)]
        if sim_data_index.empty:
            print(f"ERROR: No historical data found in simulation range {sim_start_time} - {sim_end_time_effective}. Cannot proceed.")
            return None, None
        sim_data = historical_data_final.loc[sim_data_index].copy()
        print(f"Simulation period: {sim_data.index.min()} to {sim_data.index.max()}")

        transactions_sim = portfolio_df.set_index('Actual_Transaction_Time_Index').sort_index()

        # Initialize strategies
        strategies = {
            'Actual_Buy_Hold': {'value': 0.0, 'holdings': {}},
            'Rebalance_To_Equal': {'value': 0.0, 'holdings': {}, 'last_rebalance': pd.NaT, 'commission_paid': 0.0},
            'Stablecoin_Only': {'value': 0.0},
            'Bank_Deposit': {'value': 0.0},
            'Perfect_Foresight': {'value': 0.0, 'holdings': {}, 'last_rebalance': pd.NaT, 'commission_paid': 0.0},
        }
        for model_name in df_actions_all_models.keys():
            strategies[f'DRL_{model_name}'] = {'value': 0.0, 'holdings': {}, 'last_weights': {}, 'last_rebalance': pd.NaT}

        # Add result columns to sim_data
        sim_data['Total_Invested'] = 0.0
        sim_data['Total_Withdrawn'] = 0.0 # Note: This tracks withdrawals from *crypto* strategies
        for s_name in strategies.keys():
            sim_data[f'Value_{s_name}'] = 0.0
            if 'holdings' in strategies[s_name]: sim_data[f'Holdings_{s_name}'] = None
        if 'commission_paid' in strategies['Rebalance_To_Equal']: sim_data['Commission_S2'] = 0.0
        if 'commission_paid' in strategies['Perfect_Foresight']: sim_data['Commission_PF'] = 0.0

        total_invested_overall = 0.0
        total_withdrawn_overall = 0.0

        # --- Simulation Loop ---
        print("Running simulation loop...")
        for i, current_time in enumerate(sim_data.index):
            previous_time = sim_data.index[i-1] if i > 0 else None

            # 1. Update values based on price changes
            if previous_time:
                for s_name, state in strategies.items():
                    if 'holdings' in state and state['value'] > 1e-9:
                        current_portfolio_value = 0.0
                        valid_holdings = { a: q for a, q in state['holdings'].items() if q > 1e-9 and f'{a}_Price' in sim_data.columns }
                        for asset, quantity in valid_holdings.items():
                            price_col = f'{asset}_Price'
                            current_price = sim_data.loc[current_time, price_col]
                            if pd.notna(current_price):
                                current_portfolio_value += quantity * current_price
                            else:
                                last_price = sim_data.loc[previous_time, price_col]
                                if pd.notna(last_price): current_portfolio_value += quantity * last_price
                        state['value'] = current_portfolio_value
                    elif s_name == 'Bank_Deposit' and state['value'] > 0:
                        hourly_rate = (1 + bank_apr)**(1 / (365.25 * 24)) - 1 if bank_apr > 0 else 0
                        state['value'] *= (1 + hourly_rate)

            # 2. Process Transactions
            investment_change_this_step = 0.0
            withdrawal_this_step = 0.0
            is_transaction_time_now = current_time in transactions_sim.index # Define for rebalance triggers

            if is_transaction_time_now:
                todays_transactions = transactions_sim.loc[[current_time]]
                for _, trans in todays_transactions.iterrows():
                    asset, quantity, price, trans_type = trans['Актив'], trans['Количество'], trans['Transaction_Price_Actual'], trans['Тип']
                    if pd.isna(price) or price <= 0: continue

                    if trans_type == 'Покупка':
                        cost = quantity * price
                        total_invested_overall += cost
                        investment_change_this_step += cost
                        for s_name, state in strategies.items():
                            if 'holdings' in state:
                                state['value'] += cost
                                state['holdings'][asset] = state['holdings'].get(asset, 0) + quantity
                            elif s_name in ['Stablecoin_Only', 'Bank_Deposit']:
                                state['value'] += cost # Add cost to cash strategies
                    elif trans_type == 'Продажа':
                        proceeds = quantity * price
                        can_process_sale_globally = False
                        for s_name, state in strategies.items():
                            if 'holdings' in state:
                                asset_holding = state['holdings'].get(asset, 0)
                                if asset_holding >= quantity - 1e-9:
                                    can_process_sale_globally = True
                                    state['holdings'][asset] -= quantity
                                    if state['holdings'][asset] < 1e-9: del state['holdings'][asset]
                                    state['value'] -= proceeds
                                    if state['value'] < 0: state['value'] = 0
                                elif asset_holding > 1e-9:
                                     print(f"  Warning ({current_time}): Insufficient {asset} ({asset_holding:.4f}) for sale {quantity:.4f} in {s_name}.")
                        if can_process_sale_globally:
                            total_withdrawn_overall += proceeds
                            withdrawal_this_step += proceeds
                            # No update to Stablecoin/Bank on sale
                        else:
                            print(f"  Warning ({current_time}): Sale of {quantity:.4f} {asset} not possible in any crypto strategy.")

            # 3. Rebalancing Logic

            # Strategy 2: Rebalance_To_Equal
            s2_state = strategies['Rebalance_To_Equal']
            perform_rebalance_s2 = False
            is_first_investment_step_s2 = (investment_change_this_step > 0 and (s2_state['value'] - investment_change_this_step) <= 1e-9)
            if not pd.isna(s2_state['last_rebalance']):
                if current_time - s2_state['last_rebalance'] >= pd.Timedelta(days=rebalance_interval_days): perform_rebalance_s2 = True
            elif is_first_investment_step_s2 and s2_state['value'] > 1e-9:
                 s2_state['last_rebalance'] = current_time; perform_rebalance_s2 = True
            if perform_rebalance_s2 and s2_state['value'] > 1e-6:
                 current_holdings_s2 = { a: q for a, q in s2_state['holdings'].items() if q > 1e-9 and f'{a}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{a}_Price']) }
                 num_assets_s2 = len(current_holdings_s2)
                 if num_assets_s2 > 0:
                     value_before_commission = s2_state['value']
                     commission = value_before_commission * commission_rate
                     s2_state['commission_paid'] += commission
                     value_to_rebalance = value_before_commission - commission
                     if value_to_rebalance > 1e-9:
                         target_value_per_asset = value_to_rebalance / num_assets_s2
                         new_holdings_s2 = {}
                         recalculated_value = 0
                         for asset in current_holdings_s2.keys():
                             price_col = f'{asset}_Price'
                             current_price = sim_data.loc[current_time, price_col]
                             if current_price > 1e-9:
                                 qty = target_value_per_asset / current_price
                                 new_holdings_s2[asset] = qty
                                 recalculated_value += qty * current_price
                         s2_state['holdings'] = new_holdings_s2
                         s2_state['value'] = recalculated_value
                     else: s2_state['holdings'] = {}; s2_state['value'] = 0
                     s2_state['last_rebalance'] = current_time
                 else: s2_state['last_rebalance'] = current_time # Shift date if nothing to rebalance


            # DRL Strategies (Loop)
            for model_name in df_actions_all_models.keys():
                 s_name = f'DRL_{model_name}'
                 s_drl_state = strategies[s_name]
                 df_actions = df_actions_all_models[model_name]
                 perform_rebalance_drl = False
                 is_first_investment_step_drl = (investment_change_this_step > 0 and (s_drl_state['value'] - investment_change_this_step) <= 1e-9)
                 if not pd.isna(s_drl_state['last_rebalance']):
                     if current_time - s_drl_state['last_rebalance'] >= pd.Timedelta(days=drl_rebalance_interval_days): perform_rebalance_drl = True
                 elif is_first_investment_step_drl and s_drl_state['value'] > 1e-9:
                      s_drl_state['last_rebalance'] = current_time; perform_rebalance_drl = True

                 # Get target weights from DRL prediction if available
                 target_drl_weights = {}
                 if current_time in df_actions.index and s_drl_state['value'] > 1e-9:
                      model_weights_all_assets = df_actions.loc[current_time]
                      allocatable_drl_assets = { a for a in DRL_TRAINING_ASSETS if f'{a}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{a}_Price']) }
                      relevant_weights = { asset: model_weights_all_assets.get(asset, 0.0) for asset in allocatable_drl_assets }
                      relevant_weights = { a: w for a, w in relevant_weights.items() if pd.notna(w) and w > 1e-9 }
                      total_relevant_weight = sum(relevant_weights.values())
                      if total_relevant_weight > 1e-6: target_drl_weights = {a: w / total_relevant_weight for a, w in relevant_weights.items()}
                 if not target_drl_weights: target_drl_weights = s_drl_state.get('last_weights', {})
                 # Fallback if still no weights (e.g., very first step)
                 if not target_drl_weights:
                     current_portfolio_assets_drl = {a for a, q in s_drl_state['holdings'].items() if q > 1e-9}
                     if has_stablecoin_data and STABLECOIN_ASSET in DRL_TRAINING_ASSETS: target_drl_weights = {STABLECOIN_ASSET: 1.0}
                     elif current_portfolio_assets_drl: target_drl_weights = {asset: 1.0/len(current_portfolio_assets_drl) for asset in current_portfolio_assets_drl}


                 # Apply DRL Rebalance
                 if (perform_rebalance_drl or is_transaction_time_now) and s_drl_state['value'] > 1e-9:
                     value_to_rebalance_drl = s_drl_state['value']
                     current_holdings_drl = {a: q for a, q in s_drl_state['holdings'].items() if q > 1e-9}
                     current_held_assets = set(current_holdings_drl.keys())
                     eligible_assets_for_rebalance = current_held_assets.copy()
                     stablecoin_price_col_local = f'{STABLECOIN_ASSET}_Price' # Use local var name
                     if has_stablecoin_data and stablecoin_price_col_local in sim_data.columns and pd.notna(sim_data.loc[current_time, stablecoin_price_col_local]):
                          eligible_assets_for_rebalance.add(STABLECOIN_ASSET)
                     eligible_assets_for_rebalance = { asset for asset in eligible_assets_for_rebalance if f'{asset}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{asset}_Price']) }

                     final_weights_to_apply = {}
                     if not eligible_assets_for_rebalance:
                          print(f"  Warning (DRL @ {current_time}, {s_name}): No eligible assets for rebalance. Skipping.")
                          s_drl_state['last_rebalance'] = current_time
                     elif not target_drl_weights:
                          print(f"  Warning (DRL @ {current_time}, {s_name}): No target weights from model/fallback. Using equal weights for eligible.")
                          num_eligible = len(eligible_assets_for_rebalance); final_weights_to_apply = {a: 1.0/num_eligible for a in eligible_assets_for_rebalance}
                     else:
                          filtered_weights = {a: w for a, w in target_drl_weights.items() if a in eligible_assets_for_rebalance}
                          if not filtered_weights or sum(filtered_weights.values()) < 1e-9:
                               print(f"  Warning (DRL @ {current_time}, {s_name}): Filtered weights invalid/zero. Using equal weights for eligible.")
                               num_eligible = len(eligible_assets_for_rebalance); final_weights_to_apply = {a: 1.0/num_eligible for a in eligible_assets_for_rebalance}
                          else:
                               total_filtered_weight = sum(filtered_weights.values()); final_weights_to_apply = {a: w / total_filtered_weight for a, w in filtered_weights.items()}

                     if final_weights_to_apply:
                          new_holdings_drl = {}
                          recalculated_value_drl = 0
                          for asset, weight in final_weights_to_apply.items():
                              price_col = f'{asset}_Price'
                              current_price = sim_data.loc[current_time, price_col]
                              if current_price > 1e-9:
                                   qty = (value_to_rebalance_drl * weight) / current_price
                                   if qty > 1e-9: new_holdings_drl[asset] = qty; recalculated_value_drl += qty * current_price
                          s_drl_state['holdings'] = new_holdings_drl
                          s_drl_state['value'] = recalculated_value_drl
                          s_drl_state['last_rebalance'] = current_time
                          s_drl_state['last_weights'] = final_weights_to_apply

                 # Update actual weights for next step (if no rebalance)
                 current_value_drl = s_drl_state['value']
                 actual_weights = {}
                 if current_value_drl > 1e-9:
                     holdings_drl = s_drl_state['holdings']
                     for asset, quantity in holdings_drl.items():
                          if quantity > 1e-9:
                              price_col = f'{asset}_Price'
                              if price_col in sim_data.columns:
                                  current_price = sim_data.loc[current_time, price_col]
                                  if pd.notna(current_price) and current_price > 0: actual_weights[asset] = (quantity * current_price) / current_value_drl
                 s_drl_state['last_weights'] = actual_weights if actual_weights else s_drl_state.get('last_weights', {})


            # Strategy: Perfect_Foresight
            s_pf_state = strategies['Perfect_Foresight']
            perform_rebalance_pf = False
            is_first_investment_step_pf = (investment_change_this_step > 0 and (s_pf_state['value'] - investment_change_this_step) <= 1e-9)
            if not pd.isna(s_pf_state['last_rebalance']):
                 if current_time - s_pf_state['last_rebalance'] >= pd.Timedelta(days=rebalance_interval_days): perform_rebalance_pf = True
            elif is_first_investment_step_pf and s_pf_state['value'] > 1e-9:
                 s_pf_state['last_rebalance'] = current_time; perform_rebalance_pf = True

            if (perform_rebalance_pf or is_transaction_time_now) and s_pf_state['value'] > 1e-6:
                 lookahead_end_time = min(current_time + pd.Timedelta(days=rebalance_interval_days), sim_data.index.max())
                 lookahead_indices = sim_data.index.get_indexer([lookahead_end_time], method='nearest')
                 actual_lookahead_index = sim_data.index[lookahead_indices[0]] if lookahead_indices[0] != -1 else current_time

                 best_future_asset = None
                 max_future_return = -np.inf
                 eligible_assets_pf = set(s_pf_state['holdings'].keys())
                 if has_stablecoin_data: eligible_assets_pf.add(STABLECOIN_ASSET)
                 eligible_assets_pf = { a for a in eligible_assets_pf if f'{a}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{a}_Price']) }
                 risky_assets_pf = {a for a in eligible_assets_pf if a != STABLECOIN_ASSET}

                 if actual_lookahead_index > current_time and risky_assets_pf:
                     future_returns = {}
                     for asset in risky_assets_pf:
                         price_col = f'{asset}_Price'
                         price_now = sim_data.loc[current_time, price_col]
                         price_future = sim_data.loc[actual_lookahead_index, price_col] if actual_lookahead_index in sim_data.index else np.nan
                         if pd.notna(price_now) and price_now > 1e-9 and pd.notna(price_future): future_returns[asset] = price_future / price_now
                     if future_returns:
                          best_future_asset_risky = max(future_returns, key=future_returns.get)
                          if future_returns[best_future_asset_risky] > 1.0: best_future_asset = best_future_asset_risky

                 target_asset_pf = None
                 if best_future_asset is not None: target_asset_pf = best_future_asset
                 elif has_stablecoin_data and STABLECOIN_ASSET in eligible_assets_pf: target_asset_pf = STABLECOIN_ASSET
                 elif risky_assets_pf: # Fallback to best risky even if return < 1
                      best_return_asset = max(future_returns, key=future_returns.get) if future_returns else None
                      if best_return_asset : target_asset_pf = best_return_asset
                      else: target_asset_pf = list(risky_assets_pf)[0] if risky_assets_pf else None # Absolute fallback
                 # else: No eligible assets at all

                 if target_asset_pf:
                     value_before_commission_pf = s_pf_state['value']
                     commission_pf = value_before_commission_pf * commission_rate
                     s_pf_state['commission_paid'] += commission_pf
                     value_to_rebalance_pf = value_before_commission_pf - commission_pf
                     if value_to_rebalance_pf > 1e-9:
                         price_col_target = f'{target_asset_pf}_Price'
                         current_price_target = sim_data.loc[current_time, price_col_target]
                         if pd.notna(current_price_target) and current_price_target > 1e-9:
                             new_quantity_pf = value_to_rebalance_pf / current_price_target
                             s_pf_state['holdings'] = {target_asset_pf: new_quantity_pf}
                             s_pf_state['value'] = new_quantity_pf * current_price_target
                         else: s_pf_state['holdings'] = {}; s_pf_state['value'] = 0; print(f" Warning (PF @ {current_time}): Target price invalid.")
                     else: s_pf_state['holdings'] = {}; s_pf_state['value'] = 0
                     s_pf_state['last_rebalance'] = current_time
                 else:
                      print(f" Warning (PF @ {current_time}): No target asset found. Rebalance skipped.")
                      s_pf_state['last_rebalance'] = current_time # Still update time


            # 4. Record Results for the Step
            sim_data.at[current_time, 'Total_Invested'] = total_invested_overall
            sim_data.at[current_time, 'Total_Withdrawn'] = total_withdrawn_overall
            for s_name, state in strategies.items():
                sim_data.at[current_time, f'Value_{s_name}'] = state['value']
                if 'holdings' in state:
                    try: sim_data.at[current_time, f'Holdings_{s_name}'] = json.dumps(state['holdings'].copy())
                    except TypeError: sim_data.at[current_time, f'Holdings_{s_name}'] = "{}"
                if s_name == 'Rebalance_To_Equal': sim_data.at[current_time, 'Commission_S2'] = state['commission_paid']
                elif s_name == 'Perfect_Foresight': sim_data.at[current_time, 'Commission_PF'] = state['commission_paid']
        # --- End Simulation Loop ---
        print("Simulation loop finished.")

        # --- Step 4: Calculate Performance Metrics ---
        print("\n--- Calculating Performance Metrics ---")
        strategy_cols_map = {
            'Actual Buy & Hold': 'Value_Actual_Buy_Hold', 'Rebalance to Equal': 'Value_Rebalance_To_Equal',
            'Perfect Foresight': 'Value_Perfect_Foresight', 'Stablecoin Only': 'Value_Stablecoin_Only',
            'Bank Deposit': 'Value_Bank_Deposit',
        }
        for model_name in df_actions_all_models.keys(): strategy_cols_map[f'DRL {model_name}'] = f'Value_DRL_{model_name}'

        metrics_data = sim_data.copy() # Use sim_data which has the simulation results

        results = {}
        if not metrics_data.empty:
             final_total_invested = metrics_data['Total_Invested'].iloc[-1]
             final_total_withdrawn = metrics_data['Total_Withdrawn'].iloc[-1]
             start_date = metrics_data.index.min(); end_date = metrics_data.index.max()
             duration_days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
             duration_years = duration_days / 365.25
             print(f"Metrics Period: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')} ({duration_years:.2f} years)")
             print(f"Total Invested: ${final_total_invested:,.2f}")
             print(f"Total Withdrawn (from crypto): ${final_total_withdrawn:,.2f}")
             trading_days_per_year = 252

             for name, col in strategy_cols_map.items():
                 if col not in metrics_data.columns or metrics_data[col].isnull().all(): continue
                 hourly_values = metrics_data[col].ffill().replace([np.inf, -np.inf], np.nan).dropna()
                 if hourly_values.empty or len(hourly_values) <= 1: continue

                 final_value = hourly_values.iloc[-1]
                 # Adjusted Net Profit for cash strategies (relative to total invested)
                 if name in ['Stablecoin Only', 'Bank Deposit']:
                      net_profit = final_value - final_total_invested
                 else: # For crypto strategies
                      net_profit = final_value + final_total_withdrawn - final_total_invested

                 total_return = (net_profit / final_total_invested) if final_total_invested > 1e-9 else 0
                 annualized_return = np.nan
                 if duration_years > (1 / 365.25):
                      if final_total_invested > 1e-9:
                           try: annualized_return = (1 + total_return)**(1 / duration_years) - 1
                           except (ValueError, OverflowError): annualized_return = np.nan
                      else: annualized_return = 0.0 if net_profit <= 0 else np.inf

                 daily_values = hourly_values.resample('D').last().ffill().dropna()
                 annualized_volatility = np.nan
                 if len(daily_values) > 1:
                      daily_log_returns = np.log(daily_values / daily_values.shift(1))[1:] # Skip first NaN
                      daily_log_returns = daily_log_returns[np.isfinite(daily_log_returns)]
                      if len(daily_log_returns) > 1:
                           std_dev_daily_log = daily_log_returns.std()
                           if name in ['Stablecoin Only', 'Bank Deposit'] or std_dev_daily_log < 1e-9: annualized_volatility = 0.0
                           else: annualized_volatility = std_dev_daily_log * np.sqrt(trading_days_per_year)

                 rolling_max = hourly_values.cummax()
                 drawdown = pd.Series(np.nan, index=hourly_values.index)
                 valid_rolling_max = rolling_max[rolling_max > 1e-9]
                 if not valid_rolling_max.empty: drawdown.loc[valid_rolling_max.index] = (hourly_values.loc[valid_rolling_max.index] - valid_rolling_max) / valid_rolling_max
                 max_drawdown = drawdown.fillna(0).min()

                 sharpe_ratio = np.nan
                 if pd.notna(annualized_return) and pd.notna(annualized_volatility):
                     if annualized_volatility > 1e-9: sharpe_ratio = (annualized_return - bank_apr) / annualized_volatility
                     elif annualized_return > bank_apr: sharpe_ratio = np.inf
                     elif annualized_return < bank_apr: sharpe_ratio = -np.inf
                     else: sharpe_ratio = 0.0

                 results[name] = {
                     'Final Value': final_value, 'Net Profit': net_profit,
                     'Total Return (%)': total_return * 100, 'Annualized Return (%)': annualized_return * 100 if pd.notna(annualized_return) else np.nan,
                     'Annualized Volatility (%)': annualized_volatility * 100 if pd.notna(annualized_volatility) else np.nan,
                     'Max Drawdown (%)': max_drawdown * 100, 'Sharpe Ratio': sharpe_ratio
                 }

             results_df = pd.DataFrame(results).T
             def format_value(value, format_str):
                 if pd.isna(value): return 'N/A'
                 if np.isinf(value): return 'inf' if value > 0 else '-inf'
                 try: return format_str.format(value)
                 except (ValueError, TypeError): return str(value)
             results_df_display = results_df.copy()
             for col, fmt in {'Final Value': '${:,.2f}', 'Net Profit': '${:,.2f}', 'Total Return (%)': '{:.2f}%', 'Annualized Return (%)': '{:.2f}%', 'Annualized Volatility (%)': '{:.2f}%', 'Max Drawdown (%)': '{:.2f}%', 'Sharpe Ratio': '{:.3f}'}.items():
                  if col in results_df_display.columns: results_df_display[col] = results_df_display[col].apply(lambda x: format_value(x, fmt))
             print("\nMetrics Calculation Complete:")
             print(results_df_display)
        else:
             print("No metrics data available.")
             results_df_display = pd.DataFrame() # Return empty dataframe

        # --- Step 5: Visualization ---
        print("\n--- Generating Visualization ---")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(17, 10))

        plot_order_base = {
            'Value_Perfect_Foresight': {'label': f'Perfect Foresight (Comm: {commission_rate*100:.2f}%)', 'color': 'cyan', 'style': '--', 'lw': 2.0, 'z': 11},
            'Value_Bank_Deposit': {'label': f'Bank Deposit ({bank_apr*100:.0f}% APR)', 'color': 'purple', 'style': '-', 'lw': 1.5, 'z': 6},
            'Value_Stablecoin_Only': {'label': f'Stablecoin Only ({STABLECOIN_ASSET})', 'color': 'green', 'style': '-', 'lw': 1.5, 'z': 7},
            'Value_Rebalance_To_Equal': {'label': f'Rebalance to Equal (Comm: {commission_rate*100:.2f}%)', 'color': 'blue', 'style': '--', 'lw': 2.5, 'z': 9},
            'Value_Actual_Buy_Hold': {'label': 'Actual Buy & Hold', 'color': 'black', 'style': '-', 'lw': 2.5, 'z': 10},
        }
        drl_plot_settings = {
            "DDPG": {"color": "red", "style": "-.", "lw": 2.0, "z": 8}, "A2C": {"color": "orange", "style": "-.", "lw": 1.8, "z": 8},
            "PPO": {"color": "magenta", "style": "-.", "lw": 1.8, "z": 8}, "SAC": {"color": "brown", "style": "-.", "lw": 1.8, "z": 8},
        }
        plot_order = plot_order_base.copy()
        for model_name in df_actions_all_models.keys():
             col_name = f"Value_DRL_{model_name}"; strategy_label_name = f"DRL {model_name}"
             settings = drl_plot_settings.get(model_name, {})
             plot_order[col_name] = { "label": strategy_label_name, "color": settings.get("color", "grey"), "style": settings.get("style", "-."), "lw": settings.get("lw", 1.8), "z": settings.get("z", 8) }

        print("  Plotting strategy performance...")
        for col, settings in plot_order.items():
            if col in sim_data.columns: # Use sim_data for plotting
                plot_data = sim_data[col].ffill()
                if not plot_data.isnull().all() and not (plot_data == 0).all():
                     ax.plot(plot_data.index, plot_data, label=settings['label'], color=settings['color'], linewidth=settings['lw'], linestyle=settings['style'], zorder=settings['z'])
                else: print(f"    Skipping plot for {col}: All values NaN or 0.")
            else: print(f"    Skipping plot for {col}: Column not found.")

        print("  Plotting transaction markers...")
        plotted_marker_labels = set()
        asset_color_map = {"BTCUSDT": "orange", "BNBUSDT": "gold", "LTCUSDT": "silver", "HBARUSDT": "cyan", STABLECOIN_ASSET: "lightgreen"}
        default_marker_color = "grey"
        for index, row in portfolio_df.iterrows(): # Use original portfolio_df
            plot_time = row['Actual_Transaction_Time_Index']
            if pd.isna(plot_time) or plot_time not in sim_data.index: continue
            value_at_transaction = sim_data.loc[plot_time, 'Value_Actual_Buy_Hold'] # Plot against B&H
            if pd.isna(value_at_transaction): continue

            asset = row['Актив']; trans_type = row['Тип']
            color = asset_color_map.get(asset, default_marker_color)
            marker = "^" if trans_type == "Покупка" else "v"
            label_key = f"{trans_type} ({asset})"
            current_label = None
            if label_key not in plotted_marker_labels: current_label = label_key; plotted_marker_labels.add(label_key)
            ax.scatter(plot_time, value_at_transaction, color=color, marker=marker, s=80, zorder=15, label=current_label, edgecolors='black', alpha=0.8)

        print("  Formatting plot...")
        ax.set_title(f"Сравнение стратегий ({days_history} дней до {today.strftime('%Y-%m-%d')}) - С учетом транзакций", fontsize=16)
        ax.set_xlabel('Дата', fontsize=12); ax.set_ylabel('Стоимость портфеля (USDT) - Лог. шкала', fontsize=12)
        ax.grid(True, linestyle=':', linewidth=0.6); plt.xticks(rotation=30, ha='right'); ax.set_yscale('log'); ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray'); ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter()); ax.yaxis.get_major_formatter().set_scientific(False); ax.yaxis.get_major_formatter().set_useOffset(False)

        try:
            plot_cols = [col for col, settings in plot_order.items() if col in sim_data]
            all_vals = sim_data[plot_cols][sim_data[plot_cols] > 1e-6] # Ignore zero/near-zero for limits
            if not all_vals.empty:
                min_val = all_vals.min().min(); max_val = all_vals.max().max()
                if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val: ax.set_ylim(bottom=min_val * 0.8, top=max_val * 1.2)
                elif pd.notna(max_val): ax.set_ylim(top=max_val * 1.2)
        except Exception as e: print(f"Warning: Could not set Y-axis limits: {e}")

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        strategy_labels = {lbl: hnd for lbl, hnd in by_label.items() if not any(ttype in lbl for ttype in valid_types)}
        transaction_labels = {lbl: hnd for lbl, hnd in by_label.items() if any(ttype in lbl for ttype in valid_types)}
        def sort_key_transaction(label): type_order = 0 if 'Покупка' in label else 1; asset_name = label[label.find('(')+1:label.find(')')]; return (type_order, asset_name)
        sorted_transaction_keys = sorted(transaction_labels.keys(), key=sort_key_transaction)
        final_legend_order = OrderedDict()
        for col, settings in plot_order.items():
             label = settings.get('label')
             if label in strategy_labels: final_legend_order[label] = strategy_labels[label]
        for key in sorted_transaction_keys: final_legend_order[key] = transaction_labels[key]
        ax.legend(final_legend_order.values(), final_legend_order.keys(), loc='upper left', bbox_to_anchor=(1.03, 1), fontsize=9, ncol=1, borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        # Do not save or show plot here, return the figure object
        print("\n--- Portfolio Analysis Finished Successfully ---")
        return results_df_display, fig

    except Exception as e:
        print("\n--- ERROR during Portfolio Analysis --- ")
        traceback.print_exc()
        # Ensure plot is closed if it was created before error
        if 'fig' in locals() and fig:
             plt.close(fig)
        return None, None # Return None on error

# Example usage (if run as standalone script for testing)
if __name__ == '__main__':
    print("Running portfolio_analyzer as standalone script for testing...")
    # --- Define Sample Data for Testing ---
    test_portfolio_data = {
        "ID": [3, 2, 1, 0, 4],
        "Дата_Транзакции": ["2025-01-12T14:29:48.000", "2025-02-09T14:21:24.000", "2025-03-05T14:21:17.000", "2025-04-01T14:21:01.000", "2025-04-10T10:00:00.000"],
        "Актив": ["HBARUSDT", "LTCUSDT", "BTCUSDT", "BNBUSDT", "LTCUSDT"],
        "Тип": ['Покупка', 'Покупка', 'Покупка', 'Покупка', 'Продажа'],
        "Количество": [np.nan, np.nan, np.nan, np.nan, 2.0],
        "Общая стоимость": [1000.00, 500.00, 1000.00, 1000.00, np.nan]
    }
    test_transactions = pd.DataFrame(test_portfolio_data)
    test_start_date = '2025-01-10'
    test_end_date = '2025-04-19'
    # IMPORTANT: Adjust these paths according to your project structure
    # Assuming data and models are relative to the project root where streamlit runs from
    test_data_path = '../data' # Adjust relative path if needed
    test_models_dir = 'notebooks/trained_models' # Adjust relative path if needed

    if not os.path.exists(test_data_path):
         print(f"ERROR: Test data path not found: {test_data_path}")
    elif not os.path.exists(test_models_dir):
         print(f"ERROR: Test models directory not found: {test_models_dir}")
    else:
         results_df, fig_obj = run_portfolio_analysis(
              transactions_df=test_transactions,
              start_date_str=test_start_date,
              end_date_str=test_end_date,
              data_path=test_data_path,
              drl_models_dir=test_models_dir,
              bank_apr=0.20,
              commission_rate=0.001,
              rebalance_interval_days=20,
              drl_rebalance_interval_days=20
         )

         if results_df is not None and fig_obj is not None:
              print("\n--- Standalone Test Results --- ")
              print(results_df)
              # To view the plot when running standalone:
              plt.show()
         else:
              print("\n--- Standalone Test Failed --- ")