# Placeholder for DRL Data Preparation Script
import argparse
import pandas as pd
import numpy as np
from clearml import Task, Dataset
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import ta # For Technical Analysis
from ta.utils import dropna # Utility to remove NaNs created by TA indicators

log = logging.getLogger(__name__)

# (Helper functions for loading data, feature engineering specific to DRL will be added here)
# For DRL, data typically needs to be in a format that includes multiple tickers 
# and features for each ticker over time to construct the state space.

# Функция для загрузки данных из ClearML Dataset (пример)
# В реальном сценарии эта функция должна быть более надежной
# и соответствовать структуре ваших датасетов
def load_single_asset_data_from_clearml(asset_ticker: str, cfg_data_source: DictConfig, task_project_name: str) -> pd.DataFrame:
    log.info(f"Attempting to load data for {asset_ticker} from ClearML Dataset...")
    try:
        dataset_name_formatted = cfg_data_source.get("name_template", "market_data_{ticker}").format(ticker=asset_ticker)
        dataset_project = cfg_data_source.get("project", task_project_name)
        
        dataset = Dataset.get(dataset_name=dataset_name_formatted, dataset_project=dataset_project)
        if not dataset:
            log.warning(f"Dataset {dataset_name_formatted} for {asset_ticker} not found in project {dataset_project}.")
            return pd.DataFrame()
            
        dataset_path = dataset.get_local_copy()
        files = dataset.list_files()
        if not files:
            log.warning(f"Dataset {dataset_name_formatted} for {asset_ticker} contains no files.")
            return pd.DataFrame()
        
        data_file_path = None
        for f_name in files:
            if f_name.endswith('.parquet'):
                data_file_path = os.path.join(dataset_path, f_name)
                break
            elif f_name.endswith('.csv'):
                data_file_path = os.path.join(dataset_path, f_name)
        
        if not data_file_path:
            log.warning(f"No .parquet or .csv file found in dataset {dataset_name_formatted} for {asset_ticker}.")
            return pd.DataFrame()
        
        log.info(f"Reading file {data_file_path} for {asset_ticker}")
        if data_file_path.endswith('.parquet'):
            df = pd.read_parquet(data_file_path)
        else:
            df = pd.read_csv(data_file_path)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            # Attempt to find and set a datetime index
            potential_date_cols = ['date', 'timestamp', 'Date', 'Timestamp', 'Time', 'datetime', 'Datetime']
            found_date_col = False
            for col_name in df.columns:
                if col_name in potential_date_cols:
                    try:
                        pd.to_datetime(df[col_name].iloc[:5], errors='raise') 
                        df[col_name] = pd.to_datetime(df[col_name])
                        df = df.set_index(col_name)
                        log.info(f"Automatically set '{col_name}' as DatetimeIndex for {asset_ticker}.")
                        found_date_col = True
                        break
                    except Exception:
                        pass # Try next potential date column
            if not found_date_col:
                 log.warning(f"Could not auto-set DatetimeIndex for {asset_ticker}. Ensure data has a proper time index.")
        
        df = df.sort_index()
        log.info(f"Data for {asset_ticker} loaded. Shape: {df.shape}")
        return df

    except Exception as e:
        log.error(f"Error loading data for {asset_ticker} from ClearML: {e}", exc_info=True)
        return pd.DataFrame()

def add_technical_indicators_to_multi_asset_df(df: pd.DataFrame, tic_col_name: str = 'tic', ti_config: DictConfig = None) -> pd.DataFrame:
    """Добавляет технические индикаторы к DataFrame с данными по нескольким активам."""
    if not ti_config or not ti_config.get("enabled", False):
        log.info("Technical indicators for DRL are disabled in config.")
        return df

    all_tics_data = []
    for tic_value in df[tic_col_name].unique():
        tic_df = df[df[tic_col_name] == tic_value].copy()
        log.info(f"Calculating TIs for {tic_value}...")
        
        column_mapping = {col: col.lower() for col in tic_df.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']}
        tic_df_renamed = tic_df.rename(columns=column_mapping)
        
        required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
        for col_lower in required_cols_lower:
            if col_lower not in tic_df_renamed.columns:
                # If OHLCV are missing, attempt to use 'close' if available, else fill with NaN to be handled later.
                # This assumes 'close' is the most critical price. For DRL, robust handling is key.
                if 'close' in tic_df_renamed.columns:
                    tic_df_renamed[col_lower] = tic_df_renamed['close']
                    log.warning(f"Column '{col_lower}' missing for {tic_value}, filled with 'close' values.")
                else:
                    tic_df_renamed[col_lower] = np.nan 
                    log.warning(f"Column '{col_lower}' and 'close' missing for {tic_value}. '{col_lower}' filled with NaN.")
        
        # Drop rows where essential price data (after potential filling) is still NaN before TI calculation
        # TIs usually need at least 'close'. Depending on indicators, 'high', 'low', 'open' are also needed.
        tic_df_processed = dropna(tic_df_renamed, subset=['close']) 

        if tic_df_processed.empty:
            log.warning(f"DataFrame for {tic_value} is empty after dropna on 'close' before TI calculation. Appending original (unmodified) data for this ticker.")
            all_tics_data.append(tic_df) # Append original segment for this tic if processing fails
            continue
        
        # RSI
        if ti_config.rsi.enabled and 'close' in tic_df_processed.columns and not tic_df_processed['close'].isnull().all():
            try: tic_df_processed['rsi'] = ta.momentum.RSIIndicator(close=tic_df_processed["close"], window=ti_config.rsi.window).rsi()
            except Exception as e_ti: log.error(f"Error RSI for {tic_value}: {e_ti}")
        # MACD
        if ti_config.macd.enabled and 'close' in tic_df_processed.columns and not tic_df_processed['close'].isnull().all():
            try: 
                macd_indicator = ta.trend.MACD(close=tic_df_processed["close"], 
                                               window_slow=ti_config.macd.window_slow,
                                               window_fast=ti_config.macd.window_fast,
                                               window_sign=ti_config.macd.window_sign)
                tic_df_processed['macd'] = macd_indicator.macd()
                tic_df_processed['macd_signal'] = macd_indicator.macd_signal()
            except Exception as e_ti: log.error(f"Error MACD for {tic_value}: {e_ti}")
        # Bollinger Bands
        if ti_config.bollinger.enabled and 'close' in tic_df_processed.columns and not tic_df_processed['close'].isnull().all():
            try:
                bollinger_indicator = ta.volatility.BollingerBands(close=tic_df_processed["close"], 
                                                              window=ti_config.bollinger.window,
                                                              window_dev=ti_config.bollinger.window_dev)
                tic_df_processed['bb_mavg'] = bollinger_indicator.bollinger_mavg()
                tic_df_processed['bb_hband'] = bollinger_indicator.bollinger_hband()
                tic_df_processed['bb_lband'] = bollinger_indicator.bollinger_lband()
            except Exception as e_ti: log.error(f"Error Bollinger Bands for {tic_value}: {e_ti}")

        if 'close' in tic_df_processed.columns and not tic_df_processed['close'].isnull().all():
             try: tic_df_processed['change_pct'] = tic_df_processed['close'].pct_change()
             except Exception as e_chg: log.error(f"Error calculating 'change_pct' for {tic_value}: {e_chg}")

        # Preserve original columns that were not lowercased
        original_cols_to_restore = {v: k for k, v in column_mapping.items() if k != v} 
        tic_df_restored = tic_df_processed.rename(columns=original_cols_to_restore)

        # Merge TI columns back to original tic_df to keep all original columns
        # and ensure only newly added TI columns are considered for ffill/bfill
        ti_cols_added = [col for col in tic_df_restored.columns if col not in tic_df.columns and col in ['rsi', 'macd', 'macd_signal', 'bb_mavg', 'bb_hband', 'bb_lband', 'change_pct']]
        final_tic_df = tic_df.copy() # Start with the original data for this tic
        for ti_col in ti_cols_added:
            if ti_col in tic_df_restored:
                final_tic_df[ti_col] = tic_df_restored[ti_col]

        numeric_ti_cols = final_tic_df[ti_cols_added].select_dtypes(include=np.number).columns
        final_tic_df[numeric_ti_cols] = final_tic_df[numeric_ti_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        all_tics_data.append(final_tic_df)

    if not all_tics_data:
        log.warning("No TIs calculated for any ticker or data segments were empty.")
        return df 
    
    result_df = pd.concat(all_tics_data)
    log.info("Technical indicators for DRL added.")
    return result_df

@hydra.main(config_path="../../configs", config_name="drl_config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio_id', type=str, default=cfg.data_preparation.get('default_portfolio_id', 'ALL_ASSETS'))
    args, unknown = parser.parse_known_args()

    portfolio_id = args.portfolio_id
    task_name_parametrized = f"{cfg.data_preparation.task_name_prefix}_{portfolio_id}"

    task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [portfolio_id, "data_preparation", "drl"],
        output_uri=True
    )
    task.connect(OmegaConf.to_container(cfg, resolve=True), name='Hydra_Configuration')
    task.connect_configuration(vars(args), name='Argparse_Configuration') # Log argparse arguments

    log.info(f"Starting DRL data preparation for portfolio: {portfolio_id}...")
    dp_cfg = cfg.data_preparation

    log.info("Loading source data for DRL...")
    asset_tickers_to_load = list(dp_cfg.get("asset_tickers", []))
    if not asset_tickers_to_load:
        log.error("Asset tickers list in drl_config is empty. Aborting.")
        task.close(); raise ValueError("asset_tickers list is empty in drl_config.")

    all_asset_data_list = []
    current_task_project_name = task.get_project_name() # Get project name from current task for dataset loading

    for ticker in asset_tickers_to_load:
        df_asset = load_single_asset_data_from_clearml(ticker, dp_cfg.input_market_data, current_task_project_name)
        if df_asset.empty:
            log.warning(f"Data for ticker {ticker} not loaded from ClearML. This asset will be excluded.")
            # Optionally, one could implement a fallback to dummy data here if strictly needed for dev/testing,
            # but for production, it's better to fail or exclude if data is missing.
            continue # Skip this asset if data is missing
        
        df_asset['tic'] = ticker 
        all_asset_data_list.append(df_asset)

    if not all_asset_data_list:
        log.error("No data loaded for any asset. Aborting.")
        task.close(); raise ValueError("No asset data loaded for DRL.")

    processed_df = pd.concat(all_asset_data_list)
    if not isinstance(processed_df.index, pd.DatetimeIndex):
        log.info("Combined DataFrame index is not DatetimeIndex, attempting conversion...")
        processed_df.index = pd.to_datetime(processed_df.index, errors='coerce')
        if processed_df.index.isnull().any():
            log.warning("Null values in index after to_datetime conversion. Dropping rows with NaT index.")
            processed_df = processed_df[processed_df.index.notnull()]
    
    # Handle potential duplicate indices from concat, common if assets have overlapping time periods but slightly different times
    # For DRL, a consistent timeline across assets is important. Resampling or careful selection might be needed.
    # Here, we just keep the first occurrence if exact duplicates exist.
    if processed_df.index.duplicated().any():
        log.warning(f"Duplicate indices found ({processed_df.index.duplicated().sum()} instances). Keeping first.")
        processed_df = processed_df[~processed_df.index.duplicated(keep='first')]

    processed_df = processed_df.sort_index() 
    log.info(f"Data for {len(all_asset_data_list)} assets combined. Total shape: {processed_df.shape}")

    log.info("Adding technical indicators for DRL...")
    # Ensure 'tic' column exists before passing to TI function
    if 'tic' not in processed_df.columns:
        log.error("'tic' column is missing in the combined dataframe before TI calculation.")
        # This should not happen if load_single_asset_data_from_clearml adds it.
        # Attempt to infer it if only one asset was loaded, otherwise error out.
        if len(asset_tickers_to_load) == 1 and not processed_df.empty:
            processed_df['tic'] = asset_tickers_to_load[0]
            log.info(f"Added 'tic' column with value '{asset_tickers_to_load[0]}' as only one asset was loaded.")
        else:
            task.close(); raise ValueError("'tic' column missing for multi-asset TI calculation.")

    processed_df = add_technical_indicators_to_multi_asset_df(processed_df, tic_col_name='tic', ti_config=cfg.drl_specific_features.technical_indicators)
    
    # FinRL format: sort by date, then by tic. Reset index to have 'date' as a column.
    if isinstance(processed_df.index, pd.DatetimeIndex) and processed_df.index.name == 'date':
         processed_df = processed_df.reset_index() # Ensure 'date' is a column
    elif 'date' not in processed_df.columns and 'index' in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df['index']):
        processed_df = processed_df.rename(columns={'index': 'date'})
    elif 'date' not in processed_df.columns:
        log.error("Column 'date' is missing or not a datetime type for sorting. DRL data might be malformed.")
        # Attempt to use index if it's datetime
        if isinstance(processed_df.index, pd.DatetimeIndex):
            processed_df['date'] = processed_df.index
            log.info("Used DataFrame index as 'date' column.")
        else:
            task.close(); raise ValueError ("Missing 'date' column for DRL processing.")

    processed_df = processed_df.sort_values(by=['date', 'tic'])

    key_ohlcv_cols = [col.lower() for col in ['open', 'high', 'low', 'close', 'volume'] if col.lower() in processed_df.columns]
    if processed_df[key_ohlcv_cols].isnull().values.any():
        log.warning("NaNs found in key OHLCV columns after processing. Applying ffill/bfill/0 globally.")
        numeric_cols_all = processed_df.select_dtypes(include=np.number).columns
        processed_df[numeric_cols_all] = processed_df[numeric_cols_all].fillna(method='ffill').fillna(method='bfill').fillna(0)

    if processed_df.empty:
        log.error("DRL data is empty after processing. Aborting.")
        task.close(); raise ValueError("DRL data became empty after processing.")
    
    log.info(f"Final DRL data processing complete. Shape: {processed_df.shape}. Columns: {processed_df.columns.tolist()}")
    task.get_logger().report_table(title=f"DRL Processed Data Head for {portfolio_id}", series="Sample_Data", table_plot=processed_df.head(20))

    train_start_date_str = str(dp_cfg.split.train_start_date)
    train_end_date_str = str(dp_cfg.split.train_end_date)
    trade_start_date_str = str(dp_cfg.split.trade_start_date)
    trade_end_date_str = str(dp_cfg.split.trade_end_date)

    log.info(f"Splitting data: Train ({train_start_date_str} - {train_end_date_str}), Trade ({trade_start_date_str} - {trade_end_date_str})")
    
    if 'date' not in processed_df.columns:
        log.error("Column 'date' missing for data splitting. Aborting.")
        task.close(); raise ValueError("Column 'date' is missing for data splitting.")
    try:
        processed_df['date'] = pd.to_datetime(processed_df['date'])
    except Exception as e_date_conv:
        log.error(f"Failed to convert 'date' column to datetime: {e_date_conv}")
        task.close(); raise ValueError("Date column conversion to datetime failed.")

    train_data = processed_df[(processed_df['date'] >= pd.to_datetime(train_start_date_str)) & (processed_df['date'] < pd.to_datetime(train_end_date_str))]
    trade_data = processed_df[(processed_df['date'] >= pd.to_datetime(trade_start_date_str)) & (processed_df['date'] <= pd.to_datetime(trade_end_date_str))]

    if train_data.empty:
        log.warning(f"Training data is empty after split for range {train_start_date_str} - {train_end_date_str}. Check date ranges and data.")
    if trade_data.empty:
        log.warning(f"Trading data is empty after split for range {trade_start_date_str} - {trade_end_date_str}. Check date ranges and data.")

    log.info(f"Train data shape: {train_data.shape}, Trade data shape: {trade_data.shape}")

    train_artifact_name = f"drl_train_data_{portfolio_id}.parquet"
    trade_artifact_name = f"drl_trade_data_{portfolio_id}.parquet"

    if not train_data.empty:
        train_data.to_parquet(train_artifact_name)
        task.upload_artifact(name="drl_train_data", artifact_object=train_artifact_name)
        log.info(f"Uploaded DRL train data artifact: {train_artifact_name}")
    else:
        log.warning("DRL Train data is empty, skipping artifact upload.")

    if not trade_data.empty:
        trade_data.to_parquet(trade_artifact_name)
        task.upload_artifact(name="drl_trade_data", artifact_object=trade_artifact_name)
        log.info(f"Uploaded DRL trade data artifact: {trade_artifact_name}")
    else:
        log.warning("DRL Trade data is empty, skipping artifact upload.")

    task.close()
    log.info(f"DRL data preparation task for portfolio {portfolio_id} completed.")

if __name__ == '__main__':
    main() 