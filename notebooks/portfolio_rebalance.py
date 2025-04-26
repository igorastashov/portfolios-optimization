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
print("--- Начальные Условия (Портфель Транзакций) ---")
# --- ПРИМЕР СТРУКТУРЫ: Замените реальными данными ---
# Необходимо добавить столбец 'Тип' ('Покупка'/'Продажа') и 'Количество'.
# 'Общая стоимость' используется для Покупок. Для Продаж выручка будет рассчитана.
portfolio_data = {
    "ID": [3, 2, 1, 0, 4], # Добавлен ID для примера продажи
    "Дата_Транзакции": ["2025-01-12T14:29:48.000", "2025-02-09T14:21:24.000", "2025-03-05T14:21:17.000", "2025-04-01T14:21:01.000", "2025-04-10T10:00:00.000"], # Переименовано, добавлена дата продажи
    "Актив": ["HBARUSDT", "LTCUSDT", "BTCUSDT", "BNBUSDT", "LTCUSDT"], # Актив для продажи
    "Тип": ['Покупка', 'Покупка', 'Покупка', 'Покупка', 'Продажа'], # Новый столбец
    "Количество": [np.nan, np.nan, np.nan, np.nan, 2.0], # Новый столбец, NaN для покупок (будет рассчитано), указано для продажи
    "Общая стоимость": [1000.00, 500.00, 1000.00, 1000.00, np.nan] # NaN для продажи
}
# --- Конец Примера ---

portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df['Дата_Транзакции'] = pd.to_datetime(portfolio_df['Дата_Транзакции'])
portfolio_df = portfolio_df.sort_values(by='Дата_Транзакции').reset_index(drop=True)

# Проверка наличия необходимых столбцов
required_cols = ['Дата_Транзакции', 'Актив', 'Тип', 'Количество', 'Общая стоимость']
if not all(col in portfolio_df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in portfolio_df.columns]
    print(f"ОШИБКА: В данных портфеля отсутствуют необходимые столбцы: {missing}. Ожидаются: {required_cols}")
    exit()

# Валидация типов транзакций
valid_types = ['Покупка', 'Продажа']
if not portfolio_df['Тип'].isin(valid_types).all():
    invalid = portfolio_df[~portfolio_df['Тип'].isin(valid_types)]['Тип'].unique()
    print(f"ОШИБКА: Найдены неверные типы транзакций: {invalid}. Допустимые типы: {valid_types}")
    exit()

# Определение активов для загрузки
ALL_PORTFOLIO_ASSETS = portfolio_df['Актив'].unique().tolist()
DRL_TRAINING_ASSETS = ['APTUSDT', 'CAKEUSDT', 'HBARUSDT', 'JUPUSDT', 'PEPEUSDT', 'STRKUSDT', 'USDCUSDT'] # Оставляем как есть
ALL_REQUIRED_ASSETS_FOR_LOADING = list(set(ALL_PORTFOLIO_ASSETS + DRL_TRAINING_ASSETS + [STABLECOIN_ASSET]))

print("Портфель транзакций (начальные данные):")
print(portfolio_df)
print(f"Комиссия за ребалансировку (Стратегия 2): {commission_rate*100:.3f}%")
print(f"Годовая ставка банка (Стратегия 4): {bank_apr*100:.2f}%")
# print(f"DRL Модель: {drl_model_path}") # DRL модель теперь загружается позже
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
        if asset in ALL_PORTFOLIO_ASSETS or asset in DRL_TRAINING_ASSETS:
             print(f"  -> КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить данные для {asset}. Стратегии, использующие его, могут быть неточными.")
             missing_essential_assets.append(asset)
        elif asset == STABLECOIN_ASSET:
             print(f"  -> ПРЕДУПРЕЖДЕНИЕ: Данные для {STABLECOIN_ASSET} не загружены.")


if not all_data_frames: print("ОШИБКА: Нет данных для анализа!"); exit()
# Check if critical portfolio assets are missing
if any(asset in missing_essential_assets for asset in ALL_PORTFOLIO_ASSETS):
     print("ОШИБКА: Отсутствуют данные для одного или нескольких активов из портфеля транзакций. Невозможно продолжить.")
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


# --- Поиск Цен Транзакций и Расчет Количества ---
print("\nПоиск цен на момент транзакций и расчет количества...")
rows_to_drop = []
portfolio_df['Actual_Transaction_Time_Index'] = pd.NaT # Переименовано
portfolio_df['Transaction_Price_Actual'] = np.nan # Переименовано

for index, row in portfolio_df.iterrows():
    asset = row['Актив']
    transaction_date = row['Дата_Транзакции']
    price_col = f'{asset}_Price'
    trans_type = row['Тип']

    if price_col not in historical_prices_pivot.columns:
        print(f"  Предупреждение: Нет данных цен для {asset} ({price_col}) в сводной таблице. Транзакция ID {row.get('ID', index)} будет удалена.")
        rows_to_drop.append(index)
        continue

    # Find the first available timestamp in the index >= transaction_date
    relevant_times = historical_prices_pivot.index[historical_prices_pivot.index >= transaction_date]

    if not relevant_times.empty:
        actual_transaction_time_index = relevant_times[0]
        # Check if the found time actually exists in the index
        if actual_transaction_time_index in historical_prices_pivot.index:
             transaction_price = historical_prices_pivot.loc[actual_transaction_time_index, price_col]
             if pd.notna(transaction_price) and transaction_price > 0:
                 portfolio_df.loc[index, 'Transaction_Price_Actual'] = transaction_price
                 portfolio_df.loc[index, 'Actual_Transaction_Time_Index'] = actual_transaction_time_index

                 # Рассчитать количество для покупок, если не указано
                 if trans_type == 'Покупка' and pd.isna(row['Количество']):
                     if pd.notna(row['Общая стоимость']) and row['Общая стоимость'] > 0:
                         portfolio_df.loc[index, 'Количество'] = row['Общая стоимость'] / transaction_price
                     else:
                         print(f"  Предупреждение: Невозможно рассчитать Количество для Покупки {asset} (ID {row.get('ID', index)}): не указана Общая стоимость > 0. Транзакция будет удалена.")
                         rows_to_drop.append(index)
                 # Проверить наличие количества для продаж
                 elif trans_type == 'Продажа' and pd.isna(row['Количество']):
                      print(f"  Предупреждение: Не указано Количество для Продажи {asset} (ID {row.get('ID', index)}). Транзакция будет удалена.")
                      rows_to_drop.append(index)

             else:
                 print(f"  Предупреждение: Не найдена валидная цена (>0) для {asset} (ID {row.get('ID', index)}) в {actual_transaction_time_index}. Транзакция будет удалена.")
                 rows_to_drop.append(index)
        else:
             print(f"  Логическая ошибка: Время {actual_transaction_time_index} не найдено в индексе для {asset}. Транзакция будет удалена.")
             rows_to_drop.append(index)
    else:
        print(f"  Предупреждение: Нет данных в истории после даты транзакции {transaction_date} для {asset} (ID {row.get('ID', index)}). Транзакция будет удалена.")
        rows_to_drop.append(index)

if rows_to_drop:
    print(f"  Удаление {len(rows_to_drop)} транзакций из-за ошибок/отсутствия данных...")
    portfolio_df.drop(rows_to_drop, inplace=True)
    portfolio_df.reset_index(drop=True, inplace=True)

# Дополнительная проверка: у всех транзакций должно быть валидное количество
portfolio_df.dropna(subset=['Количество'], inplace=True) # Удаляем строки, где количество все еще NaN
portfolio_df = portfolio_df[portfolio_df['Количество'] > 0] # Удаляем строки с нулевым или отрицательным количеством
portfolio_df.reset_index(drop=True, inplace=True)


print(f"Обработка транзакций завершена. Учтено транзакций: {len(portfolio_df)}.")
if len(portfolio_df) == 0: print("\nОШИБКА: Нет транзакций для анализа после проверки цен и количества!"); exit()

# Определяем время первой *инвестиции* (покупки) для начала симуляции
first_investment_time = portfolio_df[portfolio_df['Тип'] == 'Покупка']['Actual_Transaction_Time_Index'].min()
if pd.isna(first_investment_time): print("ОШИБКА: Не удалось определить время первой покупки для начала симуляции!"); exit()
print(f"Первая фактическая инвестиция (покупка) учтена в: {first_investment_time}")


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


# --- Шаг 3: Расчет Стоимости Стратегий --- (Переписанный цикл)
print("\n--- Расчет Стоимости Стратегий (На основе транзакций) ---")

# Use the pivoted price data for calculations
historical_data_final = historical_prices_pivot.copy()

# Определяем временной диапазон для симуляции
sim_start_time = first_investment_time
sim_end_time = today # Конец периода - последняя доступная дата в истории
# +++ FIX: Ensure sim_data_index covers the last transaction date if it's after first_investment_time +++
last_transaction_time = portfolio_df['Actual_Transaction_Time_Index'].max()
sim_end_time_actual = max(sim_end_time, last_transaction_time) if not pd.isna(last_transaction_time) else sim_end_time
# --- End FIX ---
sim_data_index = historical_data_final.index[(historical_data_final.index >= sim_start_time) & (historical_data_final.index <= sim_end_time_actual)]
if sim_data_index.empty:
    print(f"ОШИБКА: Не найдены исторические данные в диапазоне симуляции {sim_start_time} - {sim_end_time_actual}. Выход.")
    exit()
sim_data = historical_data_final.loc[sim_data_index].copy()

# Подготовка DataFrame для транзакций (индекс - время транзакции)
transactions_sim = portfolio_df.set_index('Actual_Transaction_Time_Index').sort_index()

# Инициализация состояний для всех стратегий
strategies = {
    # Стратегия 1: Фактический Buy & Hold (с учетом продаж)
    'Actual_Buy_Hold': {'value': 0.0, 'holdings': {}},
    # Стратегия 2: Ребалансировка к равным долям ТЕКУЩИХ активов
    'Rebalance_To_Equal': {'value': 0.0, 'holdings': {}, 'last_rebalance': pd.NaT, 'commission_paid': 0.0},
    # Стратегия 3: Только стейблкоин (No holdings needed, just tracks value)
    'Stablecoin_Only': {'value': 0.0}, # <<< MODIFIED: Removed holdings
    # Стратегия 4: Банковский депозит (No holdings needed, just tracks value)
    'Bank_Deposit': {'value': 0.0}, # <<< MODIFIED: Removed holdings
      # +++ Новая стратегия: Идеальное предвидение +++
    'Perfect_Foresight': {'value': 0.0, 'holdings': {}, 'last_rebalance': pd.NaT, 'commission_paid': 0.0},
}
# Добавляем DRL стратегии
for model_name in df_actions_all_models.keys():
    strategies[f'DRL_{model_name}'] = {'value': 0.0, 'holdings': {}, 'last_weights': {}, 'last_rebalance': pd.NaT}

# Добавление колонок для результатов в sim_data
sim_data['Total_Invested'] = 0.0
sim_data['Total_Withdrawn'] = 0.0
for s_name in strategies.keys():
    sim_data[f'Value_{s_name}'] = 0.0
    if 'holdings' in strategies[s_name]:
        sim_data[f'Holdings_{s_name}'] = None # Будем хранить словари как объекты
if 'commission_paid' in strategies['Rebalance_To_Equal']:
    sim_data['Commission_S2'] = 0.0
    # +++ Добавляем колонку для комиссии Perfect_Foresight +++
    sim_data['Commission_PF'] = 0.0

# Переменные для отслеживания общих вводов/выводов
total_invested_overall = 0.0
total_withdrawn_overall = 0.0
last_recorded_invested = 0.0
last_recorded_withdrawn = 0.0

# Основной цикл симуляции
print("Запуск симуляционного цикла...")
for i, current_time in enumerate(sim_data.index):
    previous_time = sim_data.index[i-1] if i > 0 else None

    # 1. Обновление стоимости на основе изменения цен с предыдущего шага
    if previous_time:
        for s_name, state in strategies.items():
            # --- REMOVED the Stablecoin_Only block that was here ---
            # --- It should only change based on transactions ---
            if 'holdings' in state and state['value'] > 1e-9: # For strategies holding crypto
                current_portfolio_value = 0.0
                # Отфильтровываем холдинги с количеством > 0 и наличием цены
                valid_holdings = {
                    a: q for a, q in state['holdings'].items()
                    if q > 1e-9 and f'{a}_Price' in sim_data.columns
                }
                for asset, quantity in valid_holdings.items():
                    price_col = f'{asset}_Price'
                    current_price = sim_data.loc[current_time, price_col]
                    if pd.notna(current_price):
                         current_portfolio_value += quantity * current_price
                    else: # Если цена внезапно стала NaN, используем последнюю известную
                         last_price = sim_data.loc[previous_time, price_col]
                         if pd.notna(last_price):
                              current_portfolio_value += quantity * last_price
                         # Иначе стоимость этой части холдинга обнуляется

                state['value'] = current_portfolio_value
            elif s_name == 'Bank_Deposit' and state['value'] > 0:
                hourly_rate = (1 + bank_apr)**(1 / (365.25 * 24)) - 1 if bank_apr > 0 else 0
                state['value'] *= (1 + hourly_rate)

    # 2. Обработка транзакций (покупки/продажи) в текущий момент времени
    investment_change_this_step = 0.0
    withdrawal_this_step = 0.0
    # Проверяем, есть ли транзакции для текущего времени
    if current_time in transactions_sim.index:
        # Получаем все транзакции для данного часа (может быть несколько)
        todays_transactions = transactions_sim.loc[[current_time]] # Use [[]] to get DataFrame
        for _, trans in todays_transactions.iterrows():
            asset = trans['Актив']; quantity = trans['Количество']; price = trans['Transaction_Price_Actual']; trans_type = trans['Тип']
            # Убедимся, что цена валидна
            if pd.isna(price) or price <= 0:
                print(f"  ПРЕДУПРЕЖДЕНИЕ ({current_time}): Пропуск транзакции ID {trans.get('ID', 'N/A')} из-за невалидной цены ({price}).")
                continue

            if trans_type == 'Покупка':
                cost = quantity * price
                total_invested_overall += cost
                investment_change_this_step += cost
                # Обновляем каждую стратегию
                for s_name, state in strategies.items():
                    if 'holdings' in state: # Стратегии, держащие крипту
                         state['value'] += cost # Увеличиваем стоимость на сумму покупки
                         state['holdings'][asset] = state['holdings'].get(asset, 0) + quantity
                    elif s_name in ['Stablecoin_Only', 'Bank_Deposit']: # Стратегии - кэш прокси
                         state['value'] += cost # Добавляем стоимость покупки (капитал аллоцирован сюда)
                         # if state['value'] < 0: state['value'] = 0 # Предотвращаем отрицательный баланс (this check might be redundant now)

            elif trans_type == 'Продажа':
                proceeds = quantity * price
                can_process_sale_globally = False # Флаг, что продажа вообще возможна хотя бы в одной крипто-стратегии

                # Сначала обрабатываем продажу для крипто-стратегий
                for s_name, state in strategies.items():
                    if 'holdings' in state: # Только для стратегий с холдингами
                         asset_holding = state['holdings'].get(asset, 0)
                         # Продаем только если есть что продавать в этой стратегии
                         if asset_holding >= quantity - 1e-9: # Сравнение с учетом погрешности
                             can_process_sale_globally = True # Продажа возможна хотя бы в одной стратегии
                             state['holdings'][asset] -= quantity
                             if state['holdings'][asset] < 1e-9:
                                 del state['holdings'][asset] # Удаляем актив, если его не осталось
                             # Уменьшаем стоимость стратегии НА СУММУ ВЫВОДА
                             state['value'] -= proceeds
                             if state['value'] < 0: state['value'] = 0 # Стоимость не может быть < 0
                         elif asset_holding > 1e-9: # Если актива недостаточно
                              print(f"  ПРЕДУПРЕЖДЕНИЕ ({current_time}): Недостаточно {asset} ({asset_holding:.4f}) для продажи {quantity:.4f} в стратегии {s_name}. Продажа для этой стратегии пропущена.")
                         # Если актива вообще нет, ничего не делаем для этой стратегии

                # Если продажа была возможна хотя бы для одной крипто-стратегии,
                # то деньги поступили на счета кэш-прокси стратегий
                if can_process_sale_globally:
                    total_withdrawn_overall += proceeds
                    withdrawal_this_step += proceeds
                    # Обновляем кэш-стратегии
                    for s_name, state in strategies.items():
                         if s_name in ['Stablecoin_Only', 'Bank_Deposit']:
                              state['value'] += proceeds # Деньги пришли
                else:
                    print(f"  ПРЕДУПРЕЖДЕНИЕ ({current_time}): Продажа {quantity:.4f} {asset} невозможна ни в одной крипто-стратегии. Вывод средств не выполнен, кэш-стратегии не обновлены.")

    # 3. Логика ребалансировки для Стратегий 2, 5 и DRL (без изменений здесь)

    # Стратегия 2: Rebalance_To_Equal
    s2_state = strategies['Rebalance_To_Equal']
    perform_rebalance_s2 = False
    # Определяем, была ли это первая инвестиция в эту стратегию на этом шаге
    is_first_investment_step_s2 = (investment_change_this_step > 0 and (s2_state['value'] - investment_change_this_step) <= 1e-9)

    if not pd.isna(s2_state['last_rebalance']):
        if current_time - s2_state['last_rebalance'] >= pd.Timedelta(days=rebalance_interval_days):
            perform_rebalance_s2 = True
    elif is_first_investment_step_s2 and s2_state['value'] > 1e-9: # Ребаланс при первой инвестиции, если есть стоимость
        s2_state['last_rebalance'] = current_time
        perform_rebalance_s2 = True

    if perform_rebalance_s2 and s2_state['value'] > 1e-6:
        # Активы, которые есть в портфеле S2 и для которых есть цена
        current_holdings_s2 = {
            a: q for a, q in s2_state['holdings'].items()
            if q > 1e-9 and f'{a}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{a}_Price'])
        }
        num_assets_s2 = len(current_holdings_s2)

        if num_assets_s2 > 0:
            value_before_commission = s2_state['value']
            commission = value_before_commission * commission_rate
            s2_state['commission_paid'] += commission # Накапливаем комиссию
            value_to_rebalance = value_before_commission - commission

            if value_to_rebalance > 1e-9:
                target_value_per_asset = value_to_rebalance / num_assets_s2
                new_holdings_s2 = {}
                recalculated_value = 0
                for asset in current_holdings_s2.keys(): # Ребалансируем только между текущими активами с ценой
                    price_col = f'{asset}_Price'
                    current_price = sim_data.loc[current_time, price_col]
                    if current_price > 1e-9: # Цена должна быть > 0
                        qty = target_value_per_asset / current_price
                        new_holdings_s2[asset] = qty
                        recalculated_value += qty * current_price
                    # Если current_price <= 0, актив пропускается

                s2_state['holdings'] = new_holdings_s2
                s2_state['value'] = recalculated_value # Обновляем стоимость на основе реальных количеств
                s2_state['last_rebalance'] = current_time
                # sim_data.loc[current_time, 'Commission_S2'] = commission # Записываем комиссию ШАГА (можно убрать, если нужна общая)
            else: # Если после комиссии не осталось денег
                s2_state['holdings'] = {}
                s2_state['value'] = 0
                s2_state['last_rebalance'] = current_time
                # sim_data.loc[current_time, 'Commission_S2'] = commission
        else: # Если нет активов для ребалансировки
             s2_state['last_rebalance'] = current_time # Сдвигаем дату, чтобы не пытаться ребалансировать пустоту

    # Стратегия 5: DRL (для каждой модели)
    for model_name in df_actions_all_models.keys():
        s_name = f'DRL_{model_name}'
        s_drl_state = strategies[s_name]
        df_actions = df_actions_all_models[model_name]

        perform_rebalance_drl = False
        # Определяем, была ли первая инвестиция в эту DRL стратегию
        is_first_investment_step_drl = (investment_change_this_step > 0 and (s_drl_state['value'] - investment_change_this_step) <= 1e-9)

        if not pd.isna(s_drl_state['last_rebalance']):
            if current_time - s_drl_state['last_rebalance'] >= pd.Timedelta(days=drl_rebalance_interval_days):
                 perform_rebalance_drl = True
        elif is_first_investment_step_drl and s_drl_state['value'] > 1e-9:
             s_drl_state['last_rebalance'] = current_time
             perform_rebalance_drl = True

        # Получаем целевые веса DRL на ТЕКУЩИЙ момент (если есть предсказание)
        target_drl_weights = {} # Веса для активов, в которые МОЖНО ребалансироваться
        if current_time in df_actions.index and s_drl_state['value'] > 1e-9:
             model_weights_all_assets = df_actions.loc[current_time]
             # Активы, на которые DRL МОЖЕТ дать вес (из DRL_TRAINING_ASSETS) И для которых есть цена
             allocatable_drl_assets = {
                 a for a in DRL_TRAINING_ASSETS
                 if f'{a}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{a}_Price'])
             }

             relevant_weights = {}
             for asset in allocatable_drl_assets:
                 if asset in model_weights_all_assets.index:
                      weight = model_weights_all_assets[asset]
                      relevant_weights[asset] = weight if pd.notna(weight) and weight > 1e-9 else 0.0 # Убираем NaN и <=0 веса
                 else:
                      relevant_weights[asset] = 0.0

             # Отфильтровываем нулевые веса
             relevant_weights = {a: w for a, w in relevant_weights.items() if w > 1e-9}
             total_relevant_weight = sum(relevant_weights.values())

             if total_relevant_weight > 1e-6:
                 # Нормализуем веса
                 target_drl_weights = {asset: w / total_relevant_weight for asset, w in relevant_weights.items()}
             # else: Fallback будет ниже, если target_drl_weights остался пустым
        # else: # Если нет предсказания DRL на эту дату или стоимость портфеля = 0, Fallback ниже

        # Если целевые веса не определились (нет предсказания, нулевые веса), используем старые или fallback
        if not target_drl_weights:
            target_drl_weights = s_drl_state.get('last_weights', {})
            # Дополнительный fallback, если и старых весов нет
            if not target_drl_weights:
                 current_portfolio_assets_drl = {a for a, q in s_drl_state['holdings'].items() if q > 1e-9}
                 if has_stablecoin_data and STABLECOIN_ASSET in DRL_TRAINING_ASSETS:
                     target_drl_weights = {STABLECOIN_ASSET: 1.0} # По умолчанию в стейблкоин
                 elif current_portfolio_assets_drl: # Или поровну между текущими
                     target_drl_weights = {asset: 1.0/len(current_portfolio_assets_drl) for asset in current_portfolio_assets_drl}


        # Применяем ребалансировку DRL, если нужно и есть целевые веса
        if perform_rebalance_drl and target_drl_weights and s_drl_state['value'] > 1e-9:
            value_to_rebalance_drl = s_drl_state['value'] # Без комиссии для DRL
            new_holdings_drl = {}
            recalculated_value_drl = 0
            assets_in_target = set(target_drl_weights.keys())

            for asset, weight in target_drl_weights.items():
                price_col = f'{asset}_Price'
                # Цена должна быть доступна (проверяли при формировании allocatable_drl_assets)
                current_price = sim_data.loc[current_time, price_col]
                if current_price > 1e-9:
                     qty = (value_to_rebalance_drl * weight) / current_price
                     new_holdings_drl[asset] = qty
                     recalculated_value_drl += qty * current_price
                # else: Актив уже был отфильтрован ранее

            s_drl_state['holdings'] = new_holdings_drl
            s_drl_state['value'] = recalculated_value_drl
            s_drl_state['last_rebalance'] = current_time

        # Сохраняем фактические веса портфеля после ребалансировки (или старые)
        # для использования на следующем шаге, если не будет ребалансировки
        current_value_drl = s_drl_state['value']
        actual_weights = {}
        if current_value_drl > 1e-9:
            holdings_drl = s_drl_state['holdings']
            for asset, quantity in holdings_drl.items():
                 if quantity > 1e-9:
                     price_col = f'{asset}_Price'
                     if price_col in sim_data.columns:
                         current_price = sim_data.loc[current_time, price_col]
                         if pd.notna(current_price) and current_price > 0:
                            actual_weights[asset] = (quantity * current_price) / current_value_drl
        # Сохраняем актуальные веса, если они есть, иначе оставляем целевые (или старые)
        s_drl_state['last_weights'] = actual_weights if actual_weights else target_drl_weights

    # --- Стратегия: Perfect_Foresight ---
    s_pf_state = strategies['Perfect_Foresight']
    perform_rebalance_pf = False
    is_first_investment_step_pf = (investment_change_this_step > 0 and (s_pf_state['value'] - investment_change_this_step) <= 1e-9)

    # <<< MODIFIED: Add transaction time as a trigger >>>
    is_transaction_time_now = current_time in transactions_sim.index

    if not pd.isna(s_pf_state['last_rebalance']):
        if current_time - s_pf_state['last_rebalance'] >= pd.Timedelta(days=rebalance_interval_days): # Используем тот же интервал
            perform_rebalance_pf = True
    elif is_first_investment_step_pf and s_pf_state['value'] > 1e-9:
        s_pf_state['last_rebalance'] = current_time
        perform_rebalance_pf = True

    # <<< MODIFIED: Include transaction time in the trigger condition >>>
    if (perform_rebalance_pf or is_transaction_time_now) and s_pf_state['value'] > 1e-6:
        # Определяем конец периода для предвидения
        lookahead_end_time = min(current_time + pd.Timedelta(days=rebalance_interval_days), sim_data.index.max())
        # Находим ближайший индекс в sim_data к lookahead_end_time
        # Используем get_indexer с методом 'nearest' для надежности
        lookahead_indices = sim_data.index.get_indexer([lookahead_end_time], method='nearest')
        # Проверяем, что индекс валиден
        if lookahead_indices[0] != -1:
            actual_lookahead_index = sim_data.index[lookahead_indices[0]]
        else: # Не удалось найти ближайший индекс (маловероятно)
            actual_lookahead_index = current_time # Остаемся на текущем

        best_future_asset = None
        max_future_return = -np.inf
        best_future_asset_risky = None # Для хранения лучшего рискового отдельно

        # Активы, доступные для инвестирования (те, что есть в холдингах + стейблкоин, если есть)
        eligible_assets_pf = set(s_pf_state['holdings'].keys())
        if has_stablecoin_data:
            eligible_assets_pf.add(STABLECOIN_ASSET)

        # Убираем активы без цены на текущий момент
        eligible_assets_pf = {
            a for a in eligible_assets_pf
            if f'{a}_Price' in sim_data.columns and pd.notna(sim_data.loc[current_time, f'{a}_Price'])
        }

        # Исключаем стейблкоин из поиска лучшего *рискового* актива
        risky_assets_pf = {a for a in eligible_assets_pf if a != STABLECOIN_ASSET}

        if actual_lookahead_index > current_time and risky_assets_pf:
            future_returns = {}
            for asset in risky_assets_pf:
                price_col = f'{asset}_Price'
                price_now = sim_data.loc[current_time, price_col]
                # Убедимся, что цена на дату предвидения тоже есть
                if actual_lookahead_index in sim_data.index:
                    price_future = sim_data.loc[actual_lookahead_index, price_col]
                else: # Если lookahead индекс вышел за пределы sim_data
                    price_future = np.nan

                # Рассчитываем доходность, только если обе цены валидны
                if pd.notna(price_now) and price_now > 1e-9 and pd.notna(price_future):
                    future_returns[asset] = price_future / price_now

            if future_returns: # Если удалось рассчитать доходности
                 best_future_asset_risky = max(future_returns, key=future_returns.get)
                 max_future_return = future_returns[best_future_asset_risky]
                 # Выбираем лучший рисковый актив, если его доходность > 1.0 (или другого порога)
                 if max_future_return > 1.0:
                      best_future_asset = best_future_asset_risky

        # Определяем целевой актив для ребалансировки
        target_asset_pf = None
        if best_future_asset is not None: # Если нашли лучший рисковый актив с доходом > 1
             target_asset_pf = best_future_asset
        elif has_stablecoin_data and STABLECOIN_ASSET in eligible_assets_pf: # Иначе выбираем стейблкоин, если он доступен
             target_asset_pf = STABLECOIN_ASSET
        elif risky_assets_pf: # Если стейблкоина нет, берем лучший из доступных рисковых (даже если доход < 1)
              if best_future_asset_risky: # Если был найден лучший рисковый
                   target_asset_pf = best_future_asset_risky
              else: # Если вообще не удалось рассчитать доходности рисковых
                   # Проверяем, есть ли вообще рисковые активы
                   if risky_assets_pf:
                        target_asset_pf = list(risky_assets_pf)[0] # Просто берем первый доступный
        # Если нет ни стейблкоина, ни рисковых активов, target_asset_pf останется None

        # Применяем ребалансировку
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
                    s_pf_state['holdings'] = {target_asset_pf: new_quantity_pf} # Все в один актив
                    s_pf_state['value'] = new_quantity_pf * current_price_target # Обновляем стоимость
                else:
                    # Не удалось получить цену целевого актива, обнуляем холдинги
                    print(f" ПРЕДУПРЕЖДЕНИЕ (PF Rebalance @ {current_time}): Цена для целевого актива {target_asset_pf} недоступна. Холдинги обнулены.")
                    s_pf_state['holdings'] = {}
                    s_pf_state['value'] = 0
            else: # Стоимость <= 0 после комиссии
                s_pf_state['holdings'] = {}
                s_pf_state['value'] = 0

            s_pf_state['last_rebalance'] = current_time
            # sim_data.loc[current_time, 'Commission_PF'] = commission_pf # Записываем комиссию ШАГА (убрано, т.к. пишем накопленную ниже)
        else: # Не удалось выбрать целевой актив
             print(f" ПРЕДУПРЕЖДЕНИЕ (PF Rebalance @ {current_time}): Не удалось выбрать целевой актив. Ребалансировка пропущена.")
             s_pf_state['last_rebalance'] = current_time # Сдвигаем дату, чтобы не пытаться ребалансировать пустоту

    # 4. Запись результатов текущего шага
    # Используем .at для предотвращения SettingWithCopyWarning при записи словарей
    sim_data.at[current_time, 'Total_Invested'] = total_invested_overall
    sim_data.at[current_time, 'Total_Withdrawn'] = total_withdrawn_overall
    last_recorded_invested = total_invested_overall
    last_recorded_withdrawn = total_withdrawn_overall

    for s_name, state in strategies.items():
        sim_data.at[current_time, f'Value_{s_name}'] = state['value']
        if 'holdings' in state:
            # Преобразуем holdings в строку json для сохранения, чтобы избежать проблем с типами
            try:
                holdings_str = json.dumps(state['holdings'].copy())
                sim_data.at[current_time, f'Holdings_{s_name}'] = holdings_str
            except TypeError:
                 sim_data.at[current_time, f'Holdings_{s_name}'] = "{}" # Пустой словарь в виде строки
        if s_name == 'Rebalance_To_Equal':
             sim_data.at[current_time, 'Commission_S2'] = state['commission_paid'] # Записываем накопленную комиссию
        # +++ Записываем комиссию Perfect_Foresight +++
        elif s_name == 'Perfect_Foresight':
             sim_data.at[current_time, 'Commission_PF'] = state['commission_paid']

# --- Конец Симуляционного Цикла ---
print("Симуляционный цикл завершен.")

# Расчет итоговой комиссии для S2
total_commission_paid_s2 = strategies['Rebalance_To_Equal']['commission_paid']
print(f"Итоговая комиссия (Стратегия 2): {total_commission_paid_s2:.2f}")

# Присоединяем результаты симуляции к основному DataFrame
# Выбираем только нужные колонки для присоединения
sim_cols_to_join = sim_data.filter(regex='^Value_|^Holdings_|^Total_Invested|^Total_Withdrawn|^Commission_S2')
historical_data_final = historical_data_final.join(sim_cols_to_join)

# --- Расчет Траекторий Отдельных Транзакций (для визуализации, опционально) ---
# Этот блок больше не релевантен в старом виде, так как Actual_Buy_Hold теперь считает общую стоимость
print("Расчет траекторий отдельных транзакций B&H - БЛОК ПРОПУЩЕН (используйте Value_Actual_Buy_Hold)")
# Убираем расчет старых колонок Value_ASSET_ID
# individual_purchase_cols = [] # Очищаем список


# --- Шаг 4: Расчет Метрик Эффективности --- (Переписанный код)
print("\n--- Расчет Метрик Эффективности ---")

# Переименование колонок стратегий для удобства в метриках
strategy_cols_map = {
    'Actual Buy & Hold': 'Value_Actual_Buy_Hold',
    'Rebalance to Equal': 'Value_Rebalance_To_Equal',
    'Perfect Foresight': 'Value_Perfect_Foresight', # +++ Добавлено +++
    'Stablecoin Only': 'Value_Stablecoin_Only',
    'Bank Deposit': 'Value_Bank_Deposit',
}
for model_name in df_actions_all_models.keys():
    strategy_cols_map[f'DRL {model_name}'] = f'Value_DRL_{model_name}'

# Отфильтровываем данные для периода симуляции
metrics_data = historical_data_final.loc[sim_data_index].copy()

# Проверяем наличие необходимых колонок
required_metrics_cols = ['Total_Invested', 'Total_Withdrawn'] + list(strategy_cols_map.values())
missing_cols_metrics = [col for col in required_metrics_cols if col not in metrics_data.columns]
if missing_cols_metrics:
    print(f"ОШИБКА: Отсутствуют необходимые колонки для расчета метрик: {missing_cols_metrics}")
    exit()

results = {}
if not metrics_data.empty:
    start_date = metrics_data.index.min()
    end_date = metrics_data.index.max()
    duration_days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
    duration_years = duration_days / 365.25
    print(f"Период анализа метрик: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')} ({duration_years:.2f} лет)")

    # Получаем итоговые инвестиции и выводы
    final_total_invested = metrics_data['Total_Invested'].iloc[-1] if not metrics_data['Total_Invested'].empty else 0
    final_total_withdrawn = metrics_data['Total_Withdrawn'].iloc[-1] if not metrics_data['Total_Withdrawn'].empty else 0
    print(f"Всего инвестировано за период: ${final_total_invested:,.2f}")
    print(f"Всего выведено за период: ${final_total_withdrawn:,.2f}")

    annual_risk_free_rate = bank_apr # Используем ту же ставку для Шарпа
    trading_days_per_year = 252 # Для аннуализации волатильности
    print(f"Годовая безрисковая ставка (для Sharpe): {annual_risk_free_rate*100:.2f}%")

    for name, col in strategy_cols_map.items():
        if col not in metrics_data.columns or metrics_data[col].isnull().all():
             print(f"  Пропуск метрик для {name}: нет данных в колонке {col}.")
             continue

        hourly_values = metrics_data[col].ffill().replace([np.inf, -np.inf], np.nan).dropna()
        if hourly_values.empty or len(hourly_values) <= 1:
             print(f"  Пропуск метрик для {name}: недостаточно данных после очистки ({len(hourly_values)} точек).")
             continue

        final_value = hourly_values.iloc[-1]
        # Чистая прибыль = Конечная стоимость + Выведено - Инвестировано
        net_profit = final_value + final_total_withdrawn - final_total_invested

        # Общая доходность (Simple Return)
        total_return = (net_profit / final_total_invested) if final_total_invested > 1e-9 else 0

        # Годовая доходность (Simple Annualized)
        # (Может быть неточной при значительных вводах/выводах)
        annualized_return = np.nan
        if duration_years > (1 / 365.25): # Хотя бы 1 день
            if final_total_invested > 1e-9:
                 # Простая аннуализация общей доходности
                 try:
                     # Используем (1 + total_return) как базу для аннуализации
                     annualized_return = (1 + total_return)**(1 / duration_years) - 1
                 except (ValueError, OverflowError):
                     annualized_return = np.nan # Ошибка, если base < 0
            elif net_profit > 0: # Если не инвестировали, но получили прибыль (странно)
                annualized_return = np.inf
            else:
                 annualized_return = 0.0 # Если не инвестировали и нет прибыли

        # Годовая волатильность (по дневным лог. доходностям)
        daily_values = hourly_values.resample('D').last().ffill().dropna()
        annualized_volatility = np.nan
        if len(daily_values) > 1:
            # log(P_t / P_{t-1})
            daily_log_returns = np.log(daily_values / daily_values.shift(1))
            # Исключаем бесконечности и NaN, возникающие при нулевых значениях
            daily_log_returns = daily_log_returns[np.isfinite(daily_log_returns)]
            if len(daily_log_returns) > 1:
                std_dev_daily_log = daily_log_returns.std()
                # Для безрисковых стратегий волатильность = 0
                if name in ['Stablecoin Only', 'Bank Deposit']:
                    annualized_volatility = 0.0
                elif std_dev_daily_log < 1e-9:
                    annualized_volatility = 0.0
                else:
                    annualized_volatility = std_dev_daily_log * np.sqrt(trading_days_per_year)

        # Максимальная просадка (Max Drawdown)
        rolling_max = hourly_values.cummax()
        # Рассчитываем просадку только там, где rolling_max > 0
        drawdown = pd.Series(np.nan, index=hourly_values.index)
        valid_rolling_max = rolling_max[rolling_max > 1e-9]
        if not valid_rolling_max.empty:
             drawdown.loc[valid_rolling_max.index] = (hourly_values.loc[valid_rolling_max.index] - valid_rolling_max) / valid_rolling_max
        # Заменяем NaN (где rolling_max был 0) нулями
        drawdown = drawdown.fillna(0)
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        # Коэффициент Шарпа
        sharpe_ratio = np.nan
        if pd.notna(annualized_return) and pd.notna(annualized_volatility):
            if annualized_volatility > 1e-9:
                sharpe_ratio = (annualized_return - annual_risk_free_rate) / annualized_volatility
            elif annualized_return > annual_risk_free_rate:
                sharpe_ratio = np.inf
            elif annualized_return < annual_risk_free_rate:
                sharpe_ratio = -np.inf
            else:
                sharpe_ratio = 0.0

        results[name] = {
            'Final Value': final_value,
            'Net Profit': net_profit, # Добавлено для информации
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100 if pd.notna(annualized_return) else np.nan,
            'Annualized Volatility (%)': annualized_volatility * 100 if pd.notna(annualized_volatility) else np.nan,
            'Max Drawdown (%)': max_drawdown * 100,
            'Sharpe Ratio': sharpe_ratio
        }

    if results:
        results_df = pd.DataFrame(results).T

        # Форматирование для вывода
        def format_value(value, format_str):
            if pd.isna(value): return 'N/A'
            if np.isinf(value): return 'inf' if value > 0 else '-inf'
            try: return format_str.format(value)
            except (ValueError, TypeError): return str(value)

        results_df_display = results_df.copy()
        results_df_display['Final Value'] = results_df_display['Final Value'].apply(lambda x: format_value(x, '${:,.2f}'))
        results_df_display['Net Profit'] = results_df_display['Net Profit'].apply(lambda x: format_value(x, '${:,.2f}'))
        results_df_display['Total Return (%)'] = results_df_display['Total Return (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
        results_df_display['Annualized Return (%)'] = results_df_display['Annualized Return (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
        results_df_display['Annualized Volatility (%)'] = results_df_display['Annualized Volatility (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
        results_df_display['Max Drawdown (%)'] = results_df_display['Max Drawdown (%)'].apply(lambda x: format_value(x, '{:.2f}%'))
        results_df_display['Sharpe Ratio'] = results_df_display['Sharpe Ratio'].apply(lambda x: format_value(x, '{:.3f}'))

        print("\nСводная таблица метрик:")
        print(results_df_display)
    else:
        print("Нет данных для расчета метрик.")
else:
    print("Нет данных для расчета метрик (metrics_data пуст).")


# --- Шаг 5: Визуализация --- (Адаптированный код)
print("\n--- Визуализация ---")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import OrderedDict

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(17, 10))

# Используем карту имен стратегий и колонок из блока метрик
plot_order_base = {
    # +++ Добавляем Perfect Foresight +++
    'Value_Perfect_Foresight': {'label': f'Perfect Foresight (Comm: {commission_rate*100:.2f}%)', 'color': 'cyan', 'style': '--', 'lw': 2.0, 'z': 11}, # Выше всех
    'Value_Bank_Deposit': {'label': f'Bank Deposit ({bank_apr*100:.0f}% APR)', 'color': 'purple', 'style': '-', 'lw': 1.5, 'z': 6},
    'Value_Stablecoin_Only': {'label': f'Stablecoin Only ({STABLECOIN_ASSET})', 'color': 'green', 'style': '-', 'lw': 1.5, 'z': 7},
    'Value_Rebalance_To_Equal': {'label': f'Rebalance to Equal (Comm: {commission_rate*100:.2f}%)', 'color': 'blue', 'style': '--', 'lw': 2.5, 'z': 9},
    'Value_Actual_Buy_Hold': {'label': 'Actual Buy & Hold', 'color': 'black', 'style': '-', 'lw': 2.5, 'z': 10},
}

# Цвета и стили для DRL моделей (остаются без изменений)
drl_plot_settings = {
    "DDPG": {"color": "red", "style": "-.", "lw": 2.0, "z": 8},
    "A2C": {"color": "orange", "style": "-.", "lw": 1.8, "z": 8},
    "PPO": {"color": "magenta", "style": "-.", "lw": 1.8, "z": 8},
    "SAC": {"color": "brown", "style": "-.", "lw": 1.8, "z": 8},
}

# Комбинируем настройки базовых и DRL стратегий
plot_order = plot_order_base.copy()
next_z_order = 8 # Starting z-order for DRL models
for model_name in df_actions_all_models.keys():
    col_name = f"Value_DRL_{model_name}" # Правильное имя колонки
    # Находим имя стратегии для метки
    strategy_label_name = f"DRL {model_name}"
    settings = drl_plot_settings.get(model_name, {})
    plot_order[col_name] = {
        "label": strategy_label_name, # Используем короткое имя DRL {model_name}
        "color": settings.get("color", plt.cm.viridis(i / len(df_actions_all_models) if len(df_actions_all_models)>0 else 0)), # Fallback color
        "style": settings.get("style", "-."),
        "lw": settings.get("lw", 1.8),
        "z": settings.get("z", next_z_order)
    }
    # next_z_order += 0.1 # Опционально для слоев

# --- Отрисовка основных стратегий ---
print("\n--- Отрисовка основных стратегий ---")
for col, settings in plot_order.items():
    if col in historical_data_final.columns:
        # Убираем fillna(0) - пусть рисует с пропусками, если они есть
        plot_data = historical_data_final[col].ffill() # Только ffill
        if not plot_data.isnull().all() and not (plot_data == 0).all():
             print(f"  Рисуем: {col} с меткой: \"{settings['label']}\"")
             ax.plot(plot_data.index, plot_data, label=settings['label'], color=settings['color'], linewidth=settings['lw'], linestyle=settings['style'], zorder=settings['z'])
        else:
             print(f"  Пропуск отрисовки {col}: Все значения NaN или 0.")
    else:
         print(f"  Пропуск отрисовки {col}: Колонка не найдена в historical_data_final.")
print("--- Завершена отрисовка основных стратегий ---")

# --- Удаление отрисовки отдельных линий B&H ---
# Код, который рисовал линии Value_ASSET_ID..., удален.

# --- Отрисовка маркеров транзакций --- 
print("--- Отрисовка маркеров транзакций ---")
# Используем исходный portfolio_df с рассчитанными Actual_Transaction_Time_Index и Transaction_Price_Actual
plotted_marker_labels = set() # Отслеживаем уникальные метки для легенды
asset_color_map = {"BTCUSDT": "orange", "BNBUSDT": "gold", "LTCUSDT": "silver", "HBARUSDT": "cyan", STABLECOIN_ASSET: "lightgreen"}
default_marker_color = "grey"

for index, row in portfolio_df.iterrows():
    plot_time = row['Actual_Transaction_Time_Index']
    asset = row['Актив']
    trans_type = row['Тип']
    # Находим значение стратегии Actual_Buy_Hold в момент транзакции для Y-координаты
    value_at_transaction = np.nan
    if plot_time in historical_data_final.index:
        value_at_transaction = historical_data_final.loc[plot_time, 'Value_Actual_Buy_Hold']

    # Пропускаем, если нет времени или значения
    if pd.isna(plot_time) or pd.isna(value_at_transaction):
        continue

    # Определяем цвет и маркер
    color = asset_color_map.get(asset, default_marker_color)
    marker = "^" if trans_type == "Покупка" else "v" # Треугольник вверх для покупки, вниз для продажи
    label_key = f"{trans_type} ({asset})"
    current_label = None
    if label_key not in plotted_marker_labels:
        current_label = label_key
        plotted_marker_labels.add(label_key)

    ax.scatter(plot_time, value_at_transaction,
               color=color,
               marker=marker,
               s=80, # Увеличим размер маркеров
               zorder=15,
               label=current_label,
               edgecolors='black',
               alpha=0.8)
print("--- Завершена отрисовка маркеров транзакций ---")

# --- Форматирование графика --- (В основном без изменений)
ax.set_title(f"Сравнение стратегий ({days_history} дней до {today.strftime('%Y-%m-%d')}) - С учетом транзакций", fontsize=16) # Обновлен заголовок
ax.set_xlabel('Дата', fontsize=12); ax.set_ylabel('Стоимость портфеля (USDT) - Лог. шкала', fontsize=12)
ax.grid(True, linestyle=':', linewidth=0.6); plt.xticks(rotation=30, ha='right'); ax.set_yscale('log'); ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray'); ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter()); ax.yaxis.get_major_formatter().set_scientific(False); ax.yaxis.get_major_formatter().set_useOffset(False)

# Настройка пределов Y-оси (осторожно с логарифмической шкалой)
try:
    plot_cols = [col for col, settings in plot_order.items() if col in historical_data_final]
    min_val = historical_data_final[plot_cols][historical_data_final[plot_cols] > 1e-6].min().min() # Игнорируем околонулевые значения для min
    max_val = historical_data_final[plot_cols].max().max()
    if pd.notna(min_val) and min_val > 0 and pd.notna(max_val) and max_val > min_val:
        ax.set_ylim(bottom=min_val * 0.8, top=max_val * 1.2)
    elif pd.notna(max_val):
         ax.set_ylim(top=max_val * 1.2) # Если только максимум валиден
except Exception as e:
    print(f"Предупреждение: Не удалось установить пределы Y-оси: {e}")

# --- Легенда --- (Обновленная логика)
handles, labels = ax.get_legend_handles_labels()
# Удаляем дубликаты, сохраняя порядок
by_label = OrderedDict(zip(labels, handles))

# Разделяем метки на стратегии и транзакции
strategy_labels = {lbl: hnd for lbl, hnd in by_label.items() if not ('Покупка' in lbl or 'Продажа' in lbl)}
transaction_labels = {lbl: hnd for lbl, hnd in by_label.items() if ('Покупка' in lbl or 'Продажа' in lbl)}
# --- Corrected way to filter transaction labels based on actual transaction types --- 
transaction_labels = {lbl: hnd for lbl, hnd in by_label.items() if any(ttype in lbl for ttype in valid_types)} # valid_types=['Покупка', 'Продажа']

# Сортируем метки транзакций (сначала Покупки, потом Продажи, затем по активу)
def sort_key_transaction(label):
    type_order = 0 if 'Покупка' in label else 1
    asset_name = label[label.find('(')+1:label.find(')')]
    return (type_order, asset_name)

sorted_transaction_keys = sorted(transaction_labels.keys(), key=sort_key_transaction)

# Формируем финальный порядок легенды: стратегии, затем транзакции
final_legend_order = OrderedDict()
# Добавляем стратегии в порядке их отрисовки (из plot_order)
for col, settings in plot_order.items():
    label = settings.get('label')
    if label in strategy_labels:
        final_legend_order[label] = strategy_labels[label]

# Добавляем транзакции в отсортированном порядке
for key in sorted_transaction_keys:
    final_legend_order[key] = transaction_labels[key]

ax.legend(final_legend_order.values(), final_legend_order.keys(), loc='upper left', bbox_to_anchor=(1.03, 1), fontsize=9, ncol=1, borderaxespad=0.)

# Adjust layout to prevent legend cutoff
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust the right boundary to make space for legend

# Save the plot to a file
plt.savefig('portfolio_comparison_plot.png', bbox_inches='tight', dpi=300) # Added dpi for better resolution

plt.show() # Optional: Keep show() if you want to see it interactively too

print("\n--- Визуализация завершена и сохранена в portfolio_comparison_plot.png ---")
