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

# --- FinRL/DRL Components (Copy or Import) ---
# Placeholder for INDICATORS (copy from finrl.config or define)
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

# --- FeatureEngineer Class (Simplified version or import) ---
try:
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    print("Using FeatureEngineer from finrl library.")
except ImportError:
    print("finrl not found. Using basic FeatureEngineer placeholder.")
    print("WARNING: Technical indicator calculation might be missing or incorrect.")
    class FeatureEngineer:
        def __init__(self, use_technical_indicator=True, tech_indicator_list=INDICATORS, use_turbulence=False, user_defined_feature=False):
            self.use_technical_indicator = use_technical_indicator
            self.tech_indicator_list = tech_indicator_list
            print("Basic FeatureEngineer initialized.")
        def preprocess_data(self, df):
            print("WARNING: FeatureEngineer.preprocess_data needs real implementation using 'ta' library!")
            if self.use_technical_indicator:
                print("Adding dummy technical indicators.")
                for indicator in self.tech_indicator_list:
                    if indicator not in df.columns:
                        if indicator == 'close_30_sma':
                             df[indicator] = df.groupby('tic')['close'].transform(lambda x: x.rolling(30, min_periods=30).mean())
                        elif indicator == 'close_60_sma':
                             df[indicator] = df.groupby('tic')['close'].transform(lambda x: x.rolling(60, min_periods=60).mean())
                        else:
                             df[indicator] = np.nan
            return df

# --- StockPortfolioEnv Class (Copied from 3_finrl_02_tests.ipynb) ---
class StockPortfolioEnv(gym.Env):
    # ... (Paste the ENTIRE class definition as in the previous response) ...
    metadata = {'render.modes': ['human']}
    # ... (Full class definition) ...
    def __init__(self,
                df,
                stock_dim,
                hmax, # hmax seems unused in this env, but keeping for consistency
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None, # seems unused
                lookback=24, # Default lookback used in cov matrix calc in notebook
                day = 0):
        self.day = day
        self.lookback=lookback # Used for state construction if needed, not directly here
        self.df = df # Expects preprocessed df with cov_list
        self.stock_dim = stock_dim
        self.hmax = hmax # Unused
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct # Used in reward if implemented differently, here only affects final perf metrics externally
        self.reward_scaling = reward_scaling # Used in reward calculation
        self.state_space = state_space # Dimension of price/indicator part? Here it's stock_dim
        self.action_space = action_space # Dimension of output weights
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))

        # Observation space: Covariance matrix + technical indicators
        # Shape: (stock_dim + num_indicators, stock_dim) -> Check this logic
        # Original notebook seems to append indicators as rows: (stock_dim + num_indicators, stock_dim)
        # The state shape should align with what the model expects!
        # Let's assume the notebook structure: cov matrix (stock_dim, stock_dim)
        # and indicators (num_indicators, stock_dim) -> needs verification
        # If state is just cov + indicators per stock: (stock_dim, stock_dim + num_indicators)?
        # Let's follow the notebook: (stock_dim + num_indicators, stock_dim)
        obs_shape = (self.stock_dim + len(self.tech_indicator_list), self.stock_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        # load data from a pandas dataframe
        # Ensure index is integer days 0, 1, ...
        if not pd.api.types.is_integer_dtype(self.df.index):
             # This should not happen if preprocessing is correct
             raise ValueError("StockPortfolioEnv expects an integer index (days).")

        try:
            self.data = self.df.loc[self.day,:] # Assumes df index is integer days 0, 1, ...
            if self.data.empty:
                 raise IndexError(f"No data found for day {self.day}")
        except KeyError:
             raise KeyError(f"Day {self.day} not found in DataFrame index. Max index: {self.df.index.max()}")


        self.covs = self.data['cov_list'].values[0] # Expects cov_list column
        # Construct state: append indicators as rows below cov matrix
        # Ensure data has one row per stock for the current day
        # Need to handle case where data might be a Series if only one stock
        if isinstance(self.data, pd.Series):
             # If only one stock, reshape indicator data appropriately
             indicator_values = np.array([self.data[tech] for tech in self.tech_indicator_list]).reshape(-1, 1) # Shape (num_ind, 1)
        else:
             indicator_values = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])

        # Check dimensions. covs=(stock_dim, stock_dim), indicator_values=(num_ind, stock_dim)
        # State shape needs to be (stock_dim + num_ind, stock_dim)
        # Pad covs or indicators if dimensions mismatch (should align with model)
        if indicator_values.shape[1] != self.stock_dim:
             # This indicates an issue with data preparation or stock_dim mismatch
             raise ValueError(f"Indicator columns ({indicator_values.shape[1]}) != stock_dim ({self.stock_dim})")

        # Append indicators below covariance matrix
        self.state =  np.append(np.array(self.covs), indicator_values, axis=0)


        self.terminal = False
        # self.turbulence_threshold = turbulence_threshold # Unused

        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        # Store normalized weights
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        # Store actual dates
        try:
             self.date_memory=[self.data['date'].unique()[0]] # Expects date column
        except AttributeError: # Handle Series case
              self.date_memory=[self.data['date']]


    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # Return dummy state, reward, terminated, truncated, info
            return self.state, 0, self.terminal, False, {} # Return 0 reward at terminal step

        else:
            # Normalize actions (weights)
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data # Data from previous step

            # Load next state
            self.day += 1
            try:
                 self.data = self.df.loc[self.day, :] # Assumes integer index
                 if self.data.empty: raise IndexError(f"No data for day {self.day}")
            except KeyError:
                  print(f"Error: Day {self.day} not found in DRL DataFrame index during step. Terminal.")
                  self.terminal = True
                  return self.state, 0, self.terminal, False, {}

            self.covs = self.data['cov_list'].values[0]
            # Construct state for next step
            if isinstance(self.data, pd.Series):
                 indicator_values = np.array([self.data[tech] for tech in self.tech_indicator_list]).reshape(-1, 1)
            else:
                 indicator_values = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])

            if indicator_values.shape[1] != self.stock_dim:
                 raise ValueError(f"Indicator columns ({indicator_values.shape[1]}) != stock_dim ({self.stock_dim}) in step")

            self.state = np.append(np.array(self.covs), indicator_values, axis=0)

            # Calculate portfolio return using previous weights and current price changes
            # Assumes 'close' column exists
            # Handle Series vs DataFrame for price access
            if isinstance(self.data, pd.Series):
                 current_prices = np.array([self.data['close']])
                 last_prices = np.array([last_day_memory['close']])
            else:
                 current_prices = self.data.close.values
                 last_prices = last_day_memory.close.values

            # Element-wise return calculation, handling potential division by zero
            individual_returns = np.divide(current_prices - last_prices, last_prices,
                                        out=np.zeros_like(current_prices, dtype=float),
                                        where=last_prices!=0)

            portfolio_return = sum(individual_returns * weights)

            # Update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.reward = portfolio_return * self.reward_scaling # Reward as return * scaling

            self.portfolio_value = new_portfolio_value


            # Save into memory
            self.portfolio_return_memory.append(portfolio_return) # Store raw return
            # Store date
            current_date = self.data['date'].unique()[0] if not isinstance(self.data, pd.Series) else self.data['date']
            self.date_memory.append(current_date)
            self.asset_memory.append(new_portfolio_value) # Store value


            # Return observation, reward, terminated, truncated, and info
            return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None): # Removed options for Gymnasium compatibility
        if seed is not None:
            self._seed(seed)

        self.day = 0
        try:
             self.data = self.df.loc[self.day,:]
             if self.data.empty: raise IndexError(f"No data for day {self.day}")
        except KeyError:
             raise KeyError(f"Day {self.day} (reset) not found in DataFrame index. Max index: {self.df.index.max()}")

        self.covs = self.data['cov_list'].values[0]
        if isinstance(self.data, pd.Series):
             indicator_values = np.array([self.data[tech] for tech in self.tech_indicator_list]).reshape(-1, 1)
             self.date_memory = [self.data['date']]
        else:
             indicator_values = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])
             self.date_memory = [self.data['date'].unique()[0]] # Get date from first day's data

        if indicator_values.shape[1] != self.stock_dim:
             raise ValueError(f"Indicator columns ({indicator_values.shape[1]}) != stock_dim ({self.stock_dim}) in reset")

        self.state = np.append(np.array(self.covs), indicator_values, axis=0)

        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim] * self.stock_dim]
        # self.date_memory initialized above based on Series/DataFrame
        self.terminal = False

        # Gymnasium expects obs and info dictionary
        return self.state, {}

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        # Ensure actions is numpy array for np.exp
        actions = np.array(actions)
        # Subtract max for numerical stability before exp
        stable_actions = actions - np.max(actions)
        numerator = np.exp(stable_actions)
        denominator = np.sum(numerator)

        if denominator == 0 or not np.isfinite(denominator):
            # Fallback to uniform weights if exp results in issues
            print("Warning: Softmax denominator issue. Using uniform weights.")
            return np.ones_like(actions) / self.stock_dim
        softmax_output = numerator / denominator
        # Ensure output sums to 1 (handle potential minor floating point errors)
        return softmax_output / np.sum(softmax_output)


    def save_asset_memory(self):
        # Returns dataframe with 'date' and 'daily_return'
        min_len = min(len(self.date_memory), len(self.portfolio_return_memory))
        # Align lengths, starting from the *first* return/date
        df_account_value = pd.DataFrame({
             'date': self.date_memory[:min_len],
             'daily_return': self.portfolio_return_memory[:min_len]
         })
        return df_account_value

    def save_action_memory(self):
        # Returns dataframe with 'date' index and action columns (weights)
        min_len = min(len(self.date_memory), len(self.actions_memory))
        # Align lengths
        date_list = self.date_memory[:min_len]
        action_list = self.actions_memory[:min_len]

        df_actions = pd.DataFrame(action_list)

        # Get ticker names consistently
        if not self.df.empty:
             # Get unique tics preserving order from the initial DataFrame passed to env
             # Assuming 'tic' column exists and df covers all tics
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
             # Return empty dataframes if reset fails
             return pd.DataFrame(columns=['date', 'daily_return']), pd.DataFrame()

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            try:
                 obs, rewards, terminated, truncated, info = test_env.step(action)
            except Exception as e:
                 print(f"Error during test_env.step(): {e}")
                 # Stop prediction loop on error
                 break

        df_daily_return = test_env.save_asset_memory()
        df_actions = test_env.save_action_memory()
        df_daily_return['date'] = pd.to_datetime(df_daily_return['date'])
        return df_daily_return, df_actions

# --- End of Copied Classes ---

# Подавляем несущественные предупреждения
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Настройки ---
days_history = 180
today = pd.Timestamp('2025-04-19')
# today = None
rebalance_interval_days = 20 # Интервал для Стратегии 2
drl_rebalance_interval_days = 20 # <<<=== НОВЫЙ ПАРАМЕТР: Интервал для DRL (в днях)
STABLECOIN_ASSET = 'USDCUSDT' # Правильное имя для файла USDCUSDT_hourly_data.csv
commission_rate = 0.001
bank_apr = 0.20
data_path = 'D:\\__projects__\\diploma\\portfolios-optimization\\data\\'
# drl_model_path = './trained_models/trained_ddpg.zip' # Неправильный путь
drl_model_path = 'notebooks/trained_models/trained_ddpg.zip' # Правильный путь от корня проекта

# --- Пути к моделям DRL --- 
drl_models_dir = "notebooks/trained_models" # Папка с моделями
drl_model_paths = {
    "DDPG": os.path.join(drl_models_dir, "trained_ddpg.zip"),
    "A2C": os.path.join(drl_models_dir, "trained_a2c.zip"),
    "PPO": os.path.join(drl_models_dir, "trained_ppo.zip"),
    "SAC": os.path.join(drl_models_dir, "trained_sac.zip"),
    # "SAC_RiskAware": os.path.join(drl_models_dir, "trained_sac_risk_aware_7assets.zip") # Пример добавления другой модели
}

# --- Шаг 1: Загрузка данных о портфеле ---
# --- REVERTED to original portfolio ---
print("--- Начальные Условия (Оригинальный портфель) ---")
portfolio_data = {
    "ID": [3, 2, 1, 0],
    "Дата": ["2025-01-12T14:29:48.000", "2025-02-09T14:21:24.000", "2025-03-05T14:21:17.000", "2025-04-01T14:21:01.000"],
    "Актив": ["HBARUSDT", "LTCUSDT", "BTCUSDT", "BNBUSDT"],
    "Общая стоимость": [1000.00, 500.00, 1000.00, 1000.00]
}
portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df['Дата'] = pd.to_datetime(portfolio_df['Дата'])
portfolio_df = portfolio_df.sort_values(by='Дата').reset_index(drop=True)

ORIGINAL_PORTFOLIO_ASSETS = portfolio_df['Актив'].unique().tolist()
# Assets the DRL model was trained on and outputs weights for
DRL_TRAINING_ASSETS = ['APTUSDT', 'CAKEUSDT', 'HBARUSDT', 'JUPUSDT', 'PEPEUSDT', 'STRKUSDT', 'USDCUSDT']
# Combine all assets needed for data loading
ALL_REQUIRED_ASSETS_FOR_LOADING = list(set(ORIGINAL_PORTFOLIO_ASSETS + DRL_TRAINING_ASSETS + [STABLECOIN_ASSET]))

print("Начальный портфель:")
print(portfolio_df[['ID', 'Дата', 'Актив', 'Общая стоимость']])
print(f"Комиссия за ребалансировку (Стратегия 2): {commission_rate*100:.3f}%")
print(f"Годовая ставка банка (Стратегия 4): {bank_apr*100:.2f}%")
print(f"DRL Модель: {drl_model_path}")
print(f"Активы DRL модели (для предсказания): {DRL_TRAINING_ASSETS}")


# --- Шаг 2: Загрузка и Предобработка Данных ---
# --- Load data for ALL assets (Original + DRL Training) ---
asset_files = {asset: f'{asset}_hourly_data.csv' for asset in ALL_REQUIRED_ASSETS_FOR_LOADING}
print(f"\n--- Загрузка и Предобработка Данных ({len(ALL_REQUIRED_ASSETS_FOR_LOADING)} активов) ---")

def preprocess_asset_data(filepath, asset_name, stablecoin_name):
    """Загружает и обрабатывает данные одного актива (OHLCV + tic + date column)."""
    try:
        # Read CSV, handling potential parsing errors and date parsing
        try:
            # Try reading with explicit date parsing first
            df = pd.read_csv(filepath)
            # Find the date column ('Open time' or 'date') case-insensitively
            date_col = None
            for col in df.columns:
                if col.strip().lower() == 'open time':
                    date_col = col
                    break
                elif col.strip().lower() == 'date':
                    date_col = col
                    break
            if date_col is None:
                print(f"  Ошибка: Не найден столбец даты ('Open time' или 'date') в {asset_name}.")
                return pd.DataFrame()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception as parse_e:
                 print(f"  Ошибка чтения CSV или парсинга даты {asset_name}: {parse_e}")
                 return pd.DataFrame()


        # --- Column Renaming and Validation ---
        required_cols_map = {date_col: 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.columns = df.columns.str.strip().str.lower() # Standardize incoming column names to lower
        # Map required std names back to potential original lower names
        reverse_map_lower = {v: k for k, v in {'opentime': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}.items()}

        rename_actual = {}
        missing_cols = []

        for std_name in required_cols_map.values(): # std_name is 'date', 'open', etc.
            original_lower = reverse_map_lower.get(std_name) # e.g., 'opentime' for 'date'
            if std_name in df.columns:
                 rename_actual[std_name] = std_name # Already has 'date', 'open', etc.
            elif original_lower in df.columns:
                 rename_actual[original_lower] = std_name # Has 'opentime', needs rename to 'date'
            else:
                 # Check original case from required_cols_map keys as last resort
                 original_case_key = [k for k, v in required_cols_map.items() if v == std_name]
                 if original_case_key and original_case_key[0].lower() in df.columns:
                     rename_actual[original_case_key[0].lower()] = std_name
                 else:
                    missing_cols.append(std_name) # Truly missing


        if missing_cols:
            print(f"  Ошибка: В {asset_name} отсутствуют столбцы после проверки: {missing_cols}.")
            return pd.DataFrame()

        # Select and rename columns based on the final map
        df = df[list(rename_actual.keys())].copy()
        df.rename(columns=rename_actual, inplace=True)


        # --- Data Cleaning ---
        # 'date' column should already be datetime from parsing above
        df.dropna(subset=['date'], inplace=True)

        # Convert numeric columns, coercing errors
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

        if df.empty: return pd.DataFrame()

        # Sort by date (important!)
        df = df.sort_values(by='date')

        # Handle duplicate timestamps (keep first) - duplicates might arise from different sources or errors
        # It's better to handle duplicates based on the date column *before* setting it as index if needed later
        df = df.drop_duplicates(subset=['date'], keep='first')

        # Clean stablecoin price
        if asset_name == stablecoin_name:
            close_orig = df['close'].copy()
            df['close'] = np.where((df['close'] > 1.02) | (df['close'] < 0.98), 1.0, df['close'])
            mask_changed = (df['close'] == 1.0) & (close_orig != 1.0)
            df.loc[mask_changed, ['open', 'high', 'low']] = 1.0

        df['tic'] = asset_name
        # Return with 'date' as a COLUMN
        return df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]

    except FileNotFoundError:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: Файл не найден для {asset_name}: {filepath}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ОШИБКА при обработке {asset_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Load data for all required assets
all_data_frames = []
missing_essential_assets = []
for asset in ALL_REQUIRED_ASSETS_FOR_LOADING:
    filename = asset_files.get(asset, f"{asset}_hourly_data.csv") # Get filename or default
    filepath = os.path.join(data_path, filename)
    print(f"  Загрузка: {asset} из {filename}")
    df_processed = preprocess_asset_data(filepath, asset, STABLECOIN_ASSET)
    if not df_processed.empty:
        all_data_frames.append(df_processed)
    else:
        # Check if the missing asset is crucial
        if asset in ORIGINAL_PORTFOLIO_ASSETS or asset in DRL_TRAINING_ASSETS:
             print(f"  -> КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить данные для {asset}. Стратегии, использующие его, могут быть неточными.")
             missing_essential_assets.append(asset)
        elif asset == STABLECOIN_ASSET:
             print(f"  -> ПРЕДУПРЕЖДЕНИЕ: Данные для {STABLECOIN_ASSET} не загружены.")


if not all_data_frames: print("ОШИБКА: Нет данных для анализа!"); exit()
# Check if critical portfolio assets are missing
if any(asset in missing_essential_assets for asset in ORIGINAL_PORTFOLIO_ASSETS):
     print("ОШИБКА: Отсутствуют данные для одного или нескольких активов из начального портфеля. Невозможно продолжить.")
     exit()
# Check if critical DRL assets are missing (needed for prediction)
if any(asset in missing_essential_assets for asset in DRL_TRAINING_ASSETS):
     print("ОШИБКА: Отсутствуют данные для одного или нескольких активов, необходимых для DRL предсказания. Невозможно продолжить.")
     exit()


raw_historical_data = pd.concat(all_data_frames, ignore_index=True) # Use ignore_index=True

# --- FIX: Set 'date' column as DatetimeIndex and sort ---
print("Установка 'date' как индекса и сортировка...")
if 'date' not in raw_historical_data.columns:
    print("ОШИБКА: Столбец 'date' отсутствует после конкатенации.")
    exit()

# Ensure 'date' is datetime type before setting index
raw_historical_data['date'] = pd.to_datetime(raw_historical_data['date'], errors='coerce')
raw_historical_data.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed

raw_historical_data = raw_historical_data.set_index('date') # Set the datetime index
raw_historical_data.sort_index(inplace=True) # Sort by the new DatetimeIndex
# --- End of FIX ---

print("Объединение и индексация сырых данных завершены.")
# Optional: Add verification print statement here if needed
# print(f"Тип индекса raw_historical_data: {type(raw_historical_data.index)}")

# --- Filter Date Range ---
print("\n--- Подготовка Исторических Данных ---")
if today is None:
    # Check the index type *after* setting it
    if isinstance(raw_historical_data.index, pd.DatetimeIndex):
         today = raw_historical_data.index.max()
    else:
         # This error check should ideally not be needed if the above fix works
         print(f"Ошибка: Индекс не DatetimeIndex после установки. Тип: {type(raw_historical_data.index)}"); exit()
elif not isinstance(today, pd.Timestamp): today = pd.Timestamp(today)
start_date_history = today - pd.Timedelta(days=days_history)
print(f"Анализ с {start_date_history.strftime('%Y-%m-%d %H:%M')} по {today.strftime('%Y-%m-%d %H:%M')}")

# --- This line should now work because the index is DatetimeIndex ---
historical_data_filtered = raw_historical_data[(raw_historical_data.index >= start_date_history) & (raw_historical_data.index <= today)].copy()
# --- End of previously problematic line ---

if historical_data_filtered.empty: print(f"ОШИБКА: Нет данных в диапазоне!"); exit()
print(f"Отфильтровано {len(historical_data_filtered)} записей (все активы).")

# --- Reset index to get 'date' column back for subsequent steps (needed for DRL preprocessing) ---
historical_data_filtered = historical_data_filtered.reset_index()

# --- Create Pivot Table for Close Prices (needed for non-DRL strategies and lookups) ---
# Pivot table creation will set 'date' back as index for the pivot table
historical_prices_pivot = historical_data_filtered.set_index('date').pivot_table(index='date', columns='tic', values='close')
# Add '_Price' suffix for consistency with original logic accessing prices
historical_prices_pivot.columns = [f'{col}_Price' for col in historical_prices_pivot.columns]
print(f"Создана сводная таблица цен: {historical_prices_pivot.shape}")

# Fill NaNs in the pivoted table (strategies expect complete data)
print("Заполнение пропусков (NaN) в сводной таблице цен...")
if historical_prices_pivot.isnull().values.any():
    # Use ffill first to carry forward last known price, then bfill for start
    historical_prices_pivot = historical_prices_pivot.ffill().bfill()
    if historical_prices_pivot.isnull().values.any():
        print("  Предупреждение: Остались NaN в сводной таблице после ffill/bfill.")
        # Identify columns with remaining NaNs
        cols_with_nan = historical_prices_pivot.columns[historical_prices_pivot.isnull().any()].tolist()
        print(f"  Колонки с NaN: {cols_with_nan}")
        # Consider dropping these assets or handling differently if critical
        # For now, proceed, but strategies might fail if they access these NaNs
    else:
        print("  Пропуски в сводной таблице заполнены.")
else:
    print("  Пропусков в сводной таблице не обнаружено.")

# --- Check Stablecoin Data Availability ---
stablecoin_price_col = f'{STABLECOIN_ASSET}_Price'
has_stablecoin_data = stablecoin_price_col in historical_prices_pivot.columns and not historical_prices_pivot[stablecoin_price_col].isnull().all()


# --- Поиск Цен Покупки (Use Pivot Table) ---
print("\nПоиск цен на момент покупки...")
rows_to_drop = []
portfolio_df['Actual_Purchase_Time_Index'] = pd.NaT
portfolio_df['Purchase_Price_Actual'] = np.nan

for index, row in portfolio_df.iterrows():
    asset = row['Актив']
    purchase_date = row['Дата']
    price_col = f'{asset}_Price'

    if price_col not in historical_prices_pivot.columns:
        print(f"  Предупреждение: Нет данных цен для {asset} ({price_col}) в сводной таблице. Покупка ID {row.get('ID', index)} будет удалена.")
        rows_to_drop.append(index)
        continue

    # Find the first available timestamp in the index >= purchase_date
    relevant_times = historical_prices_pivot.index[historical_prices_pivot.index >= purchase_date]

    if not relevant_times.empty:
        actual_purchase_time_index = relevant_times[0]
        # Check if the found time actually exists in the index (should always be true here)
        if actual_purchase_time_index in historical_prices_pivot.index:
             purchase_price = historical_prices_pivot.loc[actual_purchase_time_index, price_col]
             if pd.notna(purchase_price) and purchase_price > 0:
                 portfolio_df.loc[index, 'Purchase_Price_Actual'] = purchase_price
                 portfolio_df.loc[index, 'Actual_Purchase_Time_Index'] = actual_purchase_time_index
             else:
                 print(f"  Предупреждение: Не найдена валидная цена (>0) для {asset} (ID {row.get('ID', index)}) в {actual_purchase_time_index}. Покупка будет удалена.")
                 rows_to_drop.append(index)
        else:
             # This case should theoretically not happen if relevant_times is not empty
             print(f"  Логическая ошибка: Время {actual_purchase_time_index} не найдено в индексе для {asset}. Покупка будет удалена.")
             rows_to_drop.append(index)
    else:
        print(f"  Предупреждение: Нет данных в истории после даты покупки {purchase_date} для {asset} (ID {row.get('ID', index)}). Покупка будет удалена.")
        rows_to_drop.append(index)

if rows_to_drop:
    print(f"  Удаление {len(rows_to_drop)} покупок из-за отсутствия цен...")
    portfolio_df.drop(rows_to_drop, inplace=True)
    portfolio_df.reset_index(drop=True, inplace=True)

print(f"Поиск цен завершен. Учтено покупок: {len(portfolio_df)}.")
if len(portfolio_df) == 0: print("\nОШИБКА: Нет покупок для анализа после проверки цен!"); exit()

first_investment_time = portfolio_df['Actual_Purchase_Time_Index'].min()
if pd.isna(first_investment_time): print("ОШИБКА: Не удалось определить время первой покупки!"); exit()
print(f"Первая фактическая инвестиция учтена в: {first_investment_time}")


# --- DRL Preprocessing & Prediction ---
print("\n--- Подготовка данных и запуск DRL Модели (на активах для обучения) ---")
# 1. Filter data for DRL training assets - uses historical_data_filtered where date is a column
drl_input_df = historical_data_filtered[historical_data_filtered['tic'].isin(DRL_TRAINING_ASSETS)].copy()
# Sort by date, then tic for FeatureEngineer consistency
drl_input_df = drl_input_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
print(f"Данные для FeatureEngineer (DRL активы): {drl_input_df.shape}")

# 2. Run FeatureEngineer (Now expects 'date' column)
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS)
try:
    print("Запуск FeatureEngineer (DRL активы)...")
    # Pass the dataframe with 'date' as a column
    drl_processed_df = fe.preprocess_data(drl_input_df) # <-- This should now work
    print(f"FeatureEngineer завершен. Форма данных: {drl_processed_df.shape}")
    # Drop rows with NaNs introduced by indicators/rolling windows
    drl_processed_df = drl_processed_df.dropna()
    print(f"Форма данных после dropna: {drl_processed_df.shape}")
    if drl_processed_df.empty: raise ValueError("DataFrame пуст после FeatureEngineer и dropna.")
except Exception as e:
    print(f"ОШИБКА при запуске FeatureEngineer: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback
    exit()

# 3. Add Covariance Matrix
print("Добавление матрицы ковариации (DRL активы)...")
# Sort again just in case FE changed order (unlikely but safe)
drl_processed_df = drl_processed_df.sort_values(['date', 'tic'])

cov_list = []; lookback = 24
# Use unique dates from the 'date' column
unique_dates = drl_processed_df['date'].unique()
num_unique_dates = len(unique_dates)

if num_unique_dates <= lookback:
     print(f"ОШИБКА: Недостаточно данных ({num_unique_dates} уник. дат) для lookback ({lookback}) для DRL.")
     exit()

# We need to map dates to integer indices for the lookback loop logic
date_to_int_map = {date: i for i, date in enumerate(unique_dates)}
drl_processed_df['day_index'] = drl_processed_df['date'].map(date_to_int_map)

# Iterate using integer indices
for i in range(lookback, num_unique_dates):
    # Select data for the lookback window using integer index range
    # Window ends at day i-1
    data_lookback = drl_processed_df[(drl_processed_df['day_index'] >= i - lookback) &
                                     (drl_processed_df['day_index'] <= i - 1)].copy()
    # Pivot requires datetime index for time series operations
    # Set 'date' as index temporarily for pivoting
    price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
    return_lookback = price_lookback.pct_change().dropna()
    if len(return_lookback) >= len(DRL_TRAINING_ASSETS):
        # Reindex to ensure consistent asset order if needed before cov()
        # covs = return_lookback.reindex(columns=DRL_TRAINING_ASSETS).cov().values
        covs = return_lookback.cov().values # Assuming pivot preserves order or cov handles it
        cov_list.append(covs)
    else:
         cov_list.append(np.full((len(DRL_TRAINING_ASSETS), len(DRL_TRAINING_ASSETS)), np.nan))

# Align cov_list with dates (dates corresponding to index i)
cov_dates = unique_dates[lookback:]
if len(cov_dates) != len(cov_list):
    print(f"Предупреждение: Несоответствие длины дат ({len(cov_dates)}) и ковариаций ({len(cov_list)}) для DRL. Выравнивание...")
    min_len = min(len(cov_dates), len(cov_list))
    cov_dates = cov_dates[:min_len]
    cov_list = cov_list[:min_len]

df_cov = pd.DataFrame({'date': cov_dates, 'cov_list': cov_list})

# Merge back using the 'date' column
drl_processed_df = drl_processed_df.merge(df_cov, on='date', how='left')
drl_processed_df['cov_list'] = drl_processed_df['cov_list'].ffill()
drl_processed_df = drl_processed_df.dropna(subset=['cov_list'])

# Create the final integer index ('day_index') needed by StockPortfolioEnv
# Use factorize on the final 'date' column after merging and dropping NaNs
drl_processed_df = drl_processed_df.sort_values(by='date') # Ensure sorted by date
drl_processed_df['day_index'] = pd.factorize(drl_processed_df['date'])[0]
drl_final_env_df = drl_processed_df.set_index('day_index') # Set the integer index
drl_final_env_df = drl_final_env_df.sort_index() # Ensure sorted by integer index

if drl_final_env_df.empty:
    print("ОШИБКА: DataFrame для среды DRL пуст после обработки ковариации."); exit()
print(f"Финальные данные для среды DRL готовы: {drl_final_env_df.shape}")

# 4. Define DRL Environment Kwargs for Prediction
stock_dimension_drl = len(DRL_TRAINING_ASSETS)
state_space_drl = stock_dimension_drl

# Use a dummy initial amount for prediction env, real amount used later
env_kwargs_drl_pred = {
    "hmax": 100, "initial_amount": 1000000, "transaction_cost_pct": 0.0, # Cost not needed for weights
    "state_space": state_space_drl, "stock_dim": stock_dimension_drl,
    "tech_indicator_list": INDICATORS, "action_space": stock_dimension_drl,
    "reward_scaling": 1e-4, "lookback": lookback
}

# 5. Load Models and Run Predictions
print("\n--- Загрузка и предсказание DRL моделей ---")
loaded_models = {}
df_actions_all_models = {}

for model_name, model_path in drl_model_paths.items():
    print(f"\nЗагрузка модели: {model_name} из {model_path}...")
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл модели не найден: {model_path}")
        continue # Пропускаем эту модель
    try:
        # Определяем класс модели по имени
        if model_name == "DDPG": model_class = DDPG
        elif model_name == "A2C": model_class = A2C
        elif model_name == "PPO": model_class = PPO
        elif model_name == "SAC": model_class = SAC
        # Добавить другие классы при необходимости
        # elif model_name == "SAC_RiskAware": model_class = SAC
        else: raise ValueError(f"Неизвестный тип модели: {model_name}")

        model = model_class.load(model_path)
        print(f"Модель {model_name} успешно загружена.")
        loaded_models[model_name] = model

        print(f"Создание среды DRL для {model_name}...")
        # Среда создается один раз, если все модели используют одинаковые данные/параметры
        if 'trade_env_drl' not in locals(): # Создаем только один раз
             trade_env_drl = StockPortfolioEnv(df=drl_final_env_df, **env_kwargs_drl_pred)
             print("Среда для предсказаний DRL создана.")

        print(f"Запуск предсказания DRL для {model_name} (получение весов)...")
        _, df_actions = DRLAgent.DRL_prediction(model=model, environment=trade_env_drl)
        print(f"Предсказание {model_name} завершено. Получено {len(df_actions)} записей.")
        if not df_actions.empty:
            df_actions_all_models[model_name] = df_actions
        else:
            print(f"Предупреждение: Предсказание {model_name} не вернуло действий.")

    except Exception as e:
        print(f"ОШИБКА во время загрузки/предсказания модели {model_name}: {e}")
        import traceback
        traceback.print_exc()

# Проверка, что хотя бы одна модель загрузилась и предсказала
if not df_actions_all_models:
    print("\nОШИБКА: Не удалось получить предсказания ни от одной DRL модели. Выход.")
    exit()

# Пример: доступ к весам для DDPG:
# df_actions_ddpg = df_actions_all_models.get("DDPG", pd.DataFrame())


# --- Шаг 3: Расчет Стоимости Стратегий ---
print("\n--- Расчет Стоимости Стратегий (Оригинальный портфель) ---")
# Use the pivoted price data for calculations
historical_data_final = historical_prices_pivot.copy()

# --- Strategy 1: Buy & Hold (Original Portfolio) ---
print("  Расчет Стратегии 1: Buy & Hold...")
historical_data_final['Total_Value_Relative'] = 0.0
total_relative_value = pd.Series(0.0, index=historical_data_final.index)
for _, purchase_row in portfolio_df.iterrows():
    asset = purchase_row['Актив']; price_col = f'{asset}_Price'
    if price_col not in historical_data_final.columns or pd.isna(purchase_row['Purchase_Price_Actual']) or pd.isna(purchase_row['Actual_Purchase_Time_Index']): continue
    initial_investment = purchase_row['Общая стоимость']; purchase_price = purchase_row['Purchase_Price_Actual']; purchase_time_index = purchase_row['Actual_Purchase_Time_Index']
    if purchase_price <= 0: continue
    price_ratio = historical_data_final[price_col] / purchase_price; investment_value = initial_investment * price_ratio
    investment_value.loc[investment_value.index < purchase_time_index] = 0
    total_relative_value = total_relative_value.add(investment_value, fill_value=0)
historical_data_final['Total_Value_Relative'] = total_relative_value

# --- Strategy 3: Stablecoin Only ---
print(f"  Расчет Стратегии 3: {STABLECOIN_ASSET} Only...")
historical_data_final['Total_Value_Stablecoin'] = np.nan
investments_by_time = portfolio_df.groupby('Actual_Purchase_Time_Index')['Общая стоимость'].sum().to_dict()
investment_series = pd.Series(investments_by_time)
aligned_investments = investment_series.reindex(historical_data_final.index, fill_value=0)
cumulative_investments = aligned_investments.cumsum()
historical_data_final['Total_Value_Stablecoin'] = cumulative_investments
historical_data_final.loc[historical_data_final.index < first_investment_time, 'Total_Value_Stablecoin'] = np.nan
if has_stablecoin_data and stablecoin_price_col in historical_data_final.columns:
    mask_invested = historical_data_final['Total_Value_Stablecoin'].notna() & (historical_data_final['Total_Value_Stablecoin'] > 0)
    if mask_invested.any():
        first_valid_time_invested = historical_data_final.loc[mask_invested].index.min()
        first_usdc_price = historical_data_final.loc[first_valid_time_invested, stablecoin_price_col]
        if pd.notna(first_usdc_price) and first_usdc_price > 0:
             usdc_price_ratio = historical_data_final[stablecoin_price_col] / first_usdc_price
             historical_data_final.loc[mask_invested, 'Total_Value_Stablecoin'] *= usdc_price_ratio.loc[mask_invested]

# --- Strategies 2 (Rebalance), 4 (Bank), 5 (DRL) - Simulation Loop ---
print(f"  Расчет Стратегий 2, 4, 5 (Цикл)...")
sim_data_index = historical_data_final.index[historical_data_final.index >= first_investment_time]
sim_data = historical_data_final.loc[sim_data_index].copy()

# Initialize columns
sim_data['Total_Value_Perfect'] = np.nan
sim_data['Held_Asset_Perfect'] = ''
sim_data['Total_Value_Bank'] = np.nan
# sim_data['Total_Value_DRL_DDPG'] = np.nan # Заменено словарем ниже

# Инициализация колонок для всех DRL стратегий
for model_name in df_actions_all_models.keys():
    sim_data[f'Total_Value_{model_name}'] = np.nan

# State variables for Strategy 2
current_perfect_value = 0.0
# last_rebalance_time = pd.NaT # Old name
last_rebalance_time_s2 = pd.NaT # Уточнено имя переменной
held_asset_perfect_col = None
total_commission_paid = 0.0

# State variables for Strategy 4
current_bank_value = 0.0
hourly_rate = (1 + bank_apr)**(1 / (365.25 * 24)) - 1 if bank_apr > 0 else 0.0

# State variables for Strategy 5 (All DRL Models)
# Используем словари для хранения состояния каждой DRL модели
current_drl_values = {name: 0.0 for name in df_actions_all_models.keys()}
last_drl_weights_all = {name: {} for name in df_actions_all_models.keys()}
last_drl_rebalance_time = pd.NaT # Одно время ребалансировки для всех DRL

first_step_simulation = True

# --- Main Simulation Loop ---
for current_time in sim_data.index:
    investment_added_this_step = 0.0
    assets_in_portfolio_now = portfolio_df[portfolio_df['Actual_Purchase_Time_Index'] <= current_time]['Актив'].unique().tolist()
    asset_price_cols_now = [f"{asset}_Price" for asset in assets_in_portfolio_now]

    # Check for new investments
    if current_time in investments_by_time:
        added_value = investments_by_time[current_time]
        investment_added_this_step = added_value
        # Add to all strategies
        current_perfect_value += added_value
        current_bank_value += added_value
        # current_drl_value += added_value # Старое
        for model_name in current_drl_values:
            current_drl_values[model_name] += added_value

        if pd.isna(last_rebalance_time_s2): last_rebalance_time_s2 = current_time
        if pd.isna(last_drl_rebalance_time): last_drl_rebalance_time = current_time

        trigger_rebalance_s2 = True
        trigger_rebalance_drl = True
    else:
        trigger_rebalance_s2 = False
        trigger_rebalance_drl = False

    # --- Update Values Based on Price Changes from Previous Step ---
    if not first_step_simulation:
        current_loc = sim_data.index.get_loc(current_time)
        previous_time = sim_data.index[current_loc - 1]

        # Update Strategy 2 (Perfect Rebalance)
        if held_asset_perfect_col and current_perfect_value > 1e-9:
            if held_asset_perfect_col in sim_data.columns:
                current_price = sim_data.loc[current_time, held_asset_perfect_col]
                previous_price = sim_data.loc[previous_time, held_asset_perfect_col]
                if pd.notna(previous_price) and previous_price > 0 and pd.notna(current_price):
                    price_ratio_step = current_price / previous_price
                    if 0.01 < price_ratio_step < 100: current_perfect_value *= price_ratio_step

        # Update Strategy 4 (Bank)
        if current_bank_value > 0 and hourly_rate > 0:
              current_bank_value *= (1 + hourly_rate)

        # Update Strategy 5 (All DRL Models)
        for model_name, last_weights in last_drl_weights_all.items():
            current_value = current_drl_values[model_name]
            if last_weights and current_value > 1e-9:
                 portfolio_return_drl = 0.0
                 valid_weights_sum = 0.0
                 for asset, weight in last_weights.items():
                     price_col = f"{asset}_Price"
                     if price_col in sim_data.columns:
                         current_price = sim_data.loc[current_time, price_col]
                         previous_price = sim_data.loc[previous_time, price_col]
                         if pd.notna(previous_price) and previous_price > 0 and pd.notna(current_price):
                             individual_return = (current_price / previous_price) - 1
                             portfolio_return_drl += weight * individual_return
                             valid_weights_sum += weight

                 if abs(valid_weights_sum - 1.0) < 1e-6:
                      current_drl_values[model_name] *= (1 + portfolio_return_drl)


    # --- Rebalance/Decision Making for Current Step ---

    # Strategy 2: Perfect Rebalance Check & Logic
    perform_rebalance_check_s2 = False
    if not pd.isna(last_rebalance_time_s2):
        time_since_last_rebalance_s2 = current_time - last_rebalance_time_s2
        if trigger_rebalance_s2 or (time_since_last_rebalance_s2 >= pd.Timedelta(days=rebalance_interval_days)):
            perform_rebalance_check_s2 = True

    if perform_rebalance_check_s2 and current_perfect_value > 1e-6:
         eligible_alt_price_cols = [col for col in asset_price_cols_now if col != stablecoin_price_col]
         best_alt_for_future = None; max_future_return = -np.inf
         if eligible_alt_price_cols:
             lookahead_end_time = min(current_time + pd.Timedelta(days=rebalance_interval_days), sim_data.index.max())
             actual_lookahead_index = sim_data.index.asof(lookahead_end_time)
             future_returns = {}
             if actual_lookahead_index > current_time:
                 for asset_col in eligible_alt_price_cols:
                     price_now = sim_data.loc[current_time, asset_col]
                     price_future = sim_data.loc[actual_lookahead_index, asset_col]
                     if pd.notna(price_now) and price_now > 0 and pd.notna(price_future): future_returns[asset_col] = price_future / price_now
             if future_returns:
                 best_alt_for_future = max(future_returns, key=future_returns.get)
                 max_future_return = future_returns[best_alt_for_future]

         new_held_asset_col_s2 = None
         if best_alt_for_future is not None and max_future_return > 1.0: new_held_asset_col_s2 = best_alt_for_future
         elif has_stablecoin_data: new_held_asset_col_s2 = stablecoin_price_col
         elif best_alt_for_future is not None: new_held_asset_col_s2 = best_alt_for_future
         elif held_asset_perfect_col: new_held_asset_col_s2 = held_asset_perfect_col

         if new_held_asset_col_s2 is not None and held_asset_perfect_col is not None and new_held_asset_col_s2 != held_asset_perfect_col:
             commission_cost = current_perfect_value * commission_rate
             current_perfect_value -= commission_cost
             total_commission_paid += commission_cost
         if new_held_asset_col_s2: held_asset_perfect_col = new_held_asset_col_s2
         last_rebalance_time_s2 = current_time # Обновляем время S2

    # Strategy 5: DRL Weight Rebalance Check and Application (for ALL models)
    perform_rebalance_check_drl = False
    if not pd.isna(last_drl_rebalance_time):
        time_since_last_rebalance_drl = current_time - last_drl_rebalance_time
        if first_step_simulation or trigger_rebalance_drl or \
           (time_since_last_rebalance_drl >= pd.Timedelta(days=drl_rebalance_interval_days)):
            perform_rebalance_check_drl = True

    if perform_rebalance_check_drl:
        # Определяем доступные активы для DRL (включая стейблкоин)
        drl_allocation_assets = assets_in_portfolio_now.copy()
        if has_stablecoin_data and STABLECOIN_ASSET in DRL_TRAINING_ASSETS:
            if STABLECOIN_ASSET not in drl_allocation_assets:
                drl_allocation_assets.append(STABLECOIN_ASSET)

        # Обновляем веса для КАЖДОЙ DRL модели
        for model_name, df_actions in df_actions_all_models.items():
            current_drl_weights_calc = {} # Временные веса для текущей модели
            if current_time in df_actions.index:
                 drl_target_weights_all = df_actions.loc[current_time]
                 filtered_weights = {}
                 for asset_ticker in drl_allocation_assets:
                     if asset_ticker in drl_target_weights_all.index:
                          weight = drl_target_weights_all[asset_ticker]
                          filtered_weights[asset_ticker] = weight if pd.notna(weight) else 0.0
                     else:
                         filtered_weights[asset_ticker] = 0.0

                 total_filtered_weight = sum(filtered_weights.values())
                 if total_filtered_weight > 1e-6:
                     current_drl_weights_calc = {asset: weight / total_filtered_weight
                                                 for asset, weight in filtered_weights.items()}
                 else: # Fallback
                     if has_stablecoin_data and STABLECOIN_ASSET in drl_allocation_assets:
                          current_drl_weights_calc = {asset: (1.0 if asset == STABLECOIN_ASSET else 0.0) for asset in drl_allocation_assets}
                     elif drl_allocation_assets:
                          num_assets = len(drl_allocation_assets)
                          current_drl_weights_calc = {asset: 1.0 / num_assets for asset in drl_allocation_assets}

            else: # Fallback if no DRL prediction for this time
                if has_stablecoin_data and STABLECOIN_ASSET in drl_allocation_assets:
                     current_drl_weights_calc = {asset: (1.0 if asset == STABLECOIN_ASSET else 0.0) for asset in drl_allocation_assets}
                elif drl_allocation_assets:
                     num_assets = len(drl_allocation_assets)
                     current_drl_weights_calc = {asset: 1.0 / num_assets for asset in drl_allocation_assets}

            # --- Обновляем `last_drl_weights` для ТЕКУЩЕЙ модели --- 
            last_drl_weights_all[model_name] = current_drl_weights_calc
            # --- Конец логики для одной DRL модели --- 

        last_drl_rebalance_time = current_time # Обновляем общее время ребалансировки DRL

    # Инициализация весов DRL на первом шаге (для всех моделей)
    if first_step_simulation:
        fallback_weights = {}
        if has_stablecoin_data and STABLECOIN_ASSET in assets_in_portfolio_now:
            fallback_weights = {asset: (1.0 if asset == STABLECOIN_ASSET else 0.0) for asset in drl_allocation_assets}
        elif assets_in_portfolio_now:
             num_assets = len(assets_in_portfolio_now)
             fallback_weights = {asset: 1.0 / num_assets for asset in assets_in_portfolio_now}
        # Применяем fallback к тем моделям, чьи веса еще не инициализированы
        for model_name in last_drl_weights_all:
            if not last_drl_weights_all[model_name]: # Если словарь пуст
                last_drl_weights_all[model_name] = fallback_weights.copy()


    # --- Record results for the current time step ---
    # Strategy 2: Perfect Rebalance
    sim_data.loc[current_time, 'Held_Asset_Perfect'] = held_asset_perfect_col.replace("_Price", "") if held_asset_perfect_col else 'N/A'
    sim_data.loc[current_time, 'Total_Value_Perfect'] = current_perfect_value
    # Strategy 4: Bank
    sim_data.loc[current_time, 'Total_Value_Bank'] = current_bank_value
    # Strategy 5: DRL (All Models)
    # sim_data.loc[current_time, 'Total_Value_DRL_DDPG'] = current_drl_value # Old
    for model_name, current_value in current_drl_values.items():
        sim_data.loc[current_time, f'Total_Value_{model_name}'] = current_value

    # Record Investment Added
    sim_data.loc[current_time, 'Investment_Added'] = investment_added_this_step

    first_step_simulation = False # Mark first step as done

# --- End of Simulation Loop ---

# Post-simulation calculations (e.g., performance metrics) can be added here
print("Simulation finished.")
print(f"Total commission paid (Strategy 2): {total_commission_paid}")

# Join simulation results back to the main dataframe
historical_data_final = historical_data_final.join(sim_data[['Total_Value_Perfect', 'Held_Asset_Perfect', 'Total_Value_Bank'] + [f'Total_Value_{model_name}' for model_name in df_actions_all_models.keys()]])


# --- Расчет Траекторий Отдельных Покупок (Original Portfolio Buy & Hold) ---
print("  Расчет траекторий отдельных покупок (Buy & Hold)...")
individual_purchase_cols = []
for index, purchase_row in portfolio_df.iterrows():
    asset = purchase_row['Актив']; purchase_id = purchase_row.get('ID', index); price_col = f'{asset}_Price'
    if price_col not in historical_data_final.columns or pd.isna(purchase_row['Purchase_Price_Actual']) or pd.isna(purchase_row['Actual_Purchase_Time_Index']): continue
    initial_investment = purchase_row['Общая стоимость']; purchase_price = purchase_row['Purchase_Price_Actual']; purchase_time_index = purchase_row['Actual_Purchase_Time_Index']
    if purchase_price <= 0: continue
    col_name = f"Value_{asset}_ID{purchase_id}"
    individual_purchase_cols.append(col_name)
    historical_data_final[col_name] = np.nan
    time_slice = historical_data_final.index >= purchase_time_index
    current_prices = historical_data_final.loc[time_slice, price_col]
    values = initial_investment * (current_prices / purchase_price)
    historical_data_final.loc[time_slice, col_name] = values
print("Расчет траекторий B&H завершен.")


# --- Шаг 4: Расчет Метрик Эффективности ---
print("\n--- Расчет Метрик Эффективности ---")
strategy_cols = {
    'Buy & Hold': 'Total_Value_Relative',
    'Perfect Rebalance': 'Total_Value_Perfect',
    'Stablecoin Only': 'Total_Value_Stablecoin',
    'Bank Deposit': 'Total_Value_Bank',
}
# Добавляем DRL стратегии динамически
for model_name in df_actions_all_models.keys():
    strategy_cols[f'DRL {model_name}'] = f'Total_Value_{model_name}'

# --- (Rest of the metrics calculation code remains the same as previous response) ---
metrics_data = historical_data_final.loc[historical_data_final.index >= first_investment_time].copy()
results = {}
if not metrics_data.empty:
    start_date = metrics_data.index.min(); end_date = metrics_data.index.max();
    duration_days = (end_date - start_date).total_seconds() / (24 * 60 * 60); duration_years = duration_days / 365.25;
    print(f"Период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')} ({duration_years:.2f} лет)");
    total_investment_in_period = portfolio_df[portfolio_df['Actual_Purchase_Time_Index'] >= first_investment_time]['Общая стоимость'].sum();
    print(f"Инвестировано за период: ${total_investment_in_period:,.2f}");

    annual_risk_free_rate = bank_apr
    trading_days_per_year = 252

    print(f"Годовая безрисковая ставка (для Sharpe): {annual_risk_free_rate*100:.2f}%")

    for name, col in strategy_cols.items():
        if col not in metrics_data.columns or metrics_data[col].isnull().all():
             print(f"  Пропуск метрик для {name}: нет данных.")
             continue

        hourly_values = metrics_data[col].ffill().replace([np.inf, -np.inf], np.nan).dropna();
        if hourly_values.empty or len(hourly_values) <= 1:
             print(f"  Пропуск метрик для {name}: недостаточно данных после очистки.")
             continue

        final_value = hourly_values.iloc[-1]
        start_value = hourly_values.iloc[0]
        total_return = (final_value / total_investment_in_period) - 1 if total_investment_in_period > 0 else 0

        annualized_return = np.nan
        if duration_years > (1 / 365.25) and start_value > 1e-9:
            base_cagr = final_value / start_value
            if base_cagr > 0:
                try: annualized_return = (base_cagr)**(1 / duration_years) - 1
                except ValueError: annualized_return = np.nan

        daily_values = hourly_values.resample('D').last().ffill().dropna()
        annualized_volatility = np.nan
        if len(daily_values) > 1:
            daily_log_returns = np.log(daily_values / daily_values.shift(1))
            daily_log_returns = daily_log_returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(daily_log_returns) > 1:
                std_dev_daily_log = daily_log_returns.std()
                if name in ['Stablecoin Only', 'Bank Deposit']: annualized_volatility = 0.0
                elif std_dev_daily_log < 1e-9: annualized_volatility = 0.0
                else: annualized_volatility = std_dev_daily_log * np.sqrt(trading_days_per_year)

        rolling_max = hourly_values.cummax()
        drawdown = (hourly_values - rolling_max) / rolling_max
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        sharpe_ratio = np.nan
        if pd.notna(annualized_return) and pd.notna(annualized_volatility):
            if annualized_volatility > 1e-9: sharpe_ratio = (annualized_return - annual_risk_free_rate) / annualized_volatility
            elif annualized_return > annual_risk_free_rate: sharpe_ratio = np.inf
            elif annualized_return < annual_risk_free_rate: sharpe_ratio = -np.inf
            else: sharpe_ratio = 0.0

        results[name] = {
            'Final Value': final_value, 'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100 if pd.notna(annualized_return) else np.nan,
            'Annualized Volatility (%)': annualized_volatility * 100 if pd.notna(annualized_volatility) else np.nan,
            'Max Drawdown (%)': max_drawdown * 100, 'Sharpe Ratio': sharpe_ratio }

    results_df = pd.DataFrame(results).T

    def format_value(value, format_str):
        if pd.isna(value): return 'N/A'
        if np.isinf(value): return 'inf' if value > 0 else '-inf'
        try: return format_str.format(value)
        except (ValueError, TypeError): return str(value)

    results_df['Final Value'] = results_df['Final Value'].apply(lambda x: format_value(x, '${:,.2f}'))
    results_df['Total Return (%)'] = results_df['Total Return (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
    results_df['Annualized Return (%)'] = results_df['Annualized Return (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
    results_df['Annualized Volatility (%)'] = results_df['Annualized Volatility (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
    results_df['Max Drawdown (%)'] = results_df['Max Drawdown (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
    results_df['Sharpe Ratio'] = results_df['Sharpe Ratio'].apply(lambda x: format_value(x, '{:.3f}'))

else:
    print("Нет данных для расчета метрик.")


# --- Шаг 5: Визуализация ---
print("\n--- Визуализация ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(17, 10))

# Base plot order (non-DRL)
plot_order_base = {
    'Total_Value_Bank': {'label': f'Стратегия 4: Банк ({bank_apr*100:.0f}% APR)', 'color': 'purple', 'style': '-', 'lw': 1.5, 'z': 6},
    'Total_Value_Stablecoin': {'label': f'Стратегия 3: {STABLECOIN_ASSET} Only', 'color': 'green', 'style': '-', 'lw': 1.5, 'z': 7},
    'Total_Value_Perfect': {'label': f'Стр. 2: Ребаланс (Ком: {commission_rate*100:.2f}%, Парк: {STABLECOIN_ASSET if has_stablecoin_data else "Нет"})', 'color': 'blue', 'style': '--', 'lw': 2.5, 'z': 9},
    'Total_Value_Relative': {'label': 'Стр. 1: Buy & Hold (портфель)', 'color': 'black', 'style': '-', 'lw': 2.5, 'z': 10},
}

# Colors and styles for DRL models
drl_plot_settings = {
    "DDPG": {'color': 'red', 'style': '-.', 'lw': 2.0, 'z': 8},
    "A2C": {'color': 'orange', 'style': '-.', 'lw': 1.8, 'z': 8},
    "PPO": {'color': 'magenta', 'style': '-.', 'lw': 1.8, 'z': 8},
    "SAC": {'color': 'brown', 'style': '-.', 'lw': 1.8, 'z': 8},
}

# Combine base and DRL settings
plot_order = plot_order_base.copy()
next_z_order = 8 # Starting z-order for DRL models
for i, model_name in enumerate(df_actions_all_models.keys()):
    col_name = f'Total_Value_{model_name}'
    settings = drl_plot_settings.get(model_name, {})
    plot_order[col_name] = {
        # 'label': f'Стратегия 5: DRL {model_name}', # Старый формат
        'label': f'DRL {model_name}', # <<<=== Новый формат метки
        'color': settings.get('color', plt.cm.viridis(i / len(df_actions_all_models))), # Fallback color
        'style': settings.get('style', '-.'),
        'lw': settings.get('lw', 1.8),
        'z': settings.get('z', next_z_order)
    }
    # Optional: increment z-order if you want DRL lines clearly layered
    # next_z_order += 0.1


# --- Debug: Check data and plot order before plotting --- 
print("\n--- Debug: Checking data and plot order before plotting ---")
print("Columns available in historical_data_final:", historical_data_final.columns.tolist())
print("Plot order dictionary keys:", list(plot_order.keys()))
# print("Plot order dictionary full:") # Optional: uncomment for full details
# import json 
# print(json.dumps(plot_order, indent=2))
print("-----------------------------------------------------------\n")
# --- End Debug --- 


# --- (Plotting code remains the same as previous response, using historical_data_final and original asset_color_map) ---
# Plot main strategies
print("\n--- Plotting Main Strategies ---") # Add header for clarity
for col, settings in plot_order.items():
    if col in historical_data_final.columns:
        if not historical_data_final[col].isnull().all():
             print(f"  Plotting: {col} with label: \"{settings['label']}\"") # Debug print
             ax.plot(historical_data_final.index, historical_data_final[col], label=settings['label'], color=settings['color'], linewidth=settings['lw'], linestyle=settings['style'], zorder=settings['z'])
        else:
             print(f"  Skipping plot for {col}: All NaN values.") # Debug print for NaN columns
    else:
         # This case should not happen based on previous debug output, but good to keep
         print(f"  Skipping plot for {col}: Column not found in historical_data_final.")
print("--- Finished Plotting Main Strategies ---\n") # Add footer

# Original asset color map
asset_color_map = {'BTCUSDT': 'orange', 'BNBUSDT': 'gold', 'LTCUSDT': 'silver', 'HBARUSDT': 'cyan', STABLECOIN_ASSET: 'lightgreen'}
default_colors = plt.cm.tab10(np.linspace(0, 1, 10)); color_idx = 0
plotted_individual_labels = set()

# Plot individual B&H lines
for i, (index, purchase_row) in enumerate(portfolio_df.iterrows()):
    asset = purchase_row['Актив']; purchase_id = purchase_row.get('ID', index)
    col_name = f"Value_{asset}_ID{purchase_id}"; purchase_date_str = purchase_row['Дата'].strftime('%Y-%m-%d')
    if col_name in historical_data_final.columns and not historical_data_final[col_name].isnull().all():
        plot_color = asset_color_map.get(asset, default_colors[color_idx % len(default_colors)]); color_idx += 1
        label = f'{asset} (B&H)'
        current_label = label if label not in plotted_individual_labels else None
        if current_label: plotted_individual_labels.add(label)
        ax.plot(historical_data_final.index, historical_data_final[col_name],
                label=current_label, color=plot_color, linewidth=1.0, linestyle=':', alpha=0.7, zorder=5)

# Plot purchase markers
plotted_marker_assets = set()
full_relative_values = historical_data_final['Total_Value_Relative'].reindex(historical_data_final.index).ffill()
for index, row in portfolio_df.iterrows():
    plot_time = row['Actual_Purchase_Time_Index']; asset = row['Актив']
    if pd.isna(plot_time) or plot_time not in full_relative_values.index: continue
    value_at_purchase_s1 = full_relative_values.loc[plot_time]
    value_at_purchase_s1 = value_at_purchase_s1 if pd.notna(value_at_purchase_s1) else 0
    label_key = f'Покупка ({asset})'; current_label = None
    if asset not in plotted_marker_assets: current_label = label_key; plotted_marker_assets.add(asset)
    marker_color = asset_color_map.get(asset, 'red')
    ax.scatter(plot_time, value_at_purchase_s1, color=marker_color, s=60, zorder=15, label=current_label, marker='o', edgecolors='black')

# Chart formatting
ax.set_title(f'Сравнение стратегий ({days_history} дней до {today.strftime("%Y-%m-%d")}) - Ориг. Портфель', fontsize=16)
ax.set_xlabel('Дата', fontsize=12); ax.set_ylabel('Стоимость портфеля (USDT) - Лог. шкала', fontsize=12)
ax.grid(True, linestyle=':', linewidth=0.6); plt.xticks(rotation=30, ha='right'); ax.set_yscale('log'); ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray'); ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter()); ax.yaxis.get_major_formatter().set_scientific(False); ax.yaxis.get_major_formatter().set_useOffset(False)
min_val = historical_data_final[[c for c in plot_order.keys() if c in historical_data_final]].min().min()
max_val = historical_data_final[[c for c in plot_order.keys() if c in historical_data_final]].max().max()
if pd.notna(min_val) and min_val > 0 and pd.notna(max_val) and max_val > min_val: ax.set_ylim(bottom=min_val * 0.8, top=max_val * 1.2)

# Legend Handling - Revised Logic
handles, labels = ax.get_legend_handles_labels()
label_handle_map = dict(zip(labels, handles))
by_label = OrderedDict()

# Add strategy lines first, maintaining order from plot_order
for col, settings in plot_order.items():
    label = settings.get('label')
    if label in label_handle_map:
        by_label[label] = label_handle_map[label]

# Add marker labels, sorted
marker_labels = {lbl: hnd for lbl, hnd in label_handle_map.items() if "Покупка" in lbl}
marker_sorted_keys = sorted(marker_labels.keys(), key=lambda x: x[x.find("(")+1:x.find(")")])
for key in marker_sorted_keys:
    by_label[key] = marker_labels[key]

# Add individual B&H asset labels, sorted
individual_asset_labels = {lbl: hnd for lbl, hnd in label_handle_map.items() if "(B&H)" in lbl}
asset_sorted_keys = sorted(individual_asset_labels.keys())
for key in asset_sorted_keys:
    by_label[key] = individual_asset_labels[key]

ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.03, 1), fontsize=9, ncol=1, borderaxespad=0.)

# Adjust layout to prevent legend cutoff
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust the right boundary to make space for legend

# Save the plot to a file
plt.savefig('portfolio_comparison_plot.png', bbox_inches='tight', dpi=300) # Added dpi for better resolution

plt.show() # Optional: Keep show() if you want to see it interactively too

print("--- Визуализация завершена и сохранена в portfolio_comparison_plot.png ---")
