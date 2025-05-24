# CatBoost data preparation script integrating logic from research notebook
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset, Artifact
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import ta # For Technical Analysis
from ta.utils import dropna as ta_dropna # Utility to remove NaNs created by TA indicators
import json
import os

# Настройка логирования
log = logging.getLogger(__name__)


def generate_lags(df: pd.DataFrame, column_name: str, lags: list) -> pd.DataFrame:
    """Генерирует лаговые признаки для указанной колонки."""
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{column_name}_lag_{lag}h'] = df_copy[column_name].shift(lag)
    log.info(f"Lags generated for {column_name} with lags: {lags}")
    return df_copy

def generate_technical_indicators(df: pd.DataFrame, ti_config: DictConfig) -> pd.DataFrame:
    """Генерирует технические индикаторы на основе конфигурации."""
    df_copy = df.copy()
    column_mapping = {col: col.lower() for col in df_copy.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']}
    df_copy = df_copy.rename(columns=column_mapping)
    required_ohlc_cols = ['open', 'high', 'low', 'close']
    has_ohlc = all(c in df_copy.columns for c in required_ohlc_cols)
    has_volume = 'volume' in df_copy.columns
    has_close = 'close' in df_copy.columns

    if ti_config.rsi.enabled and has_close:
        df_copy['RSI_14'] = ta.momentum.RSIIndicator(close=df_copy["close"], window=ti_config.rsi.window).rsi()
    if ti_config.macd.enabled and has_close:
        macd_indicator = ta.trend.MACD(close=df_copy["close"], window_slow=ti_config.macd.window_slow, window_fast=ti_config.macd.window_fast, window_sign=ti_config.macd.window_sign)
        df_copy['MACD_12_26_9'] = macd_indicator.macd()
        df_copy['MACD_signal_12_26_9'] = macd_indicator.macd_signal()
        df_copy['MACD_hist_12_26_9'] = macd_indicator.macd_diff()
    if ti_config.bollinger.enabled and has_close:
        bollinger_indicator = ta.volatility.BollingerBands(close=df_copy["close"], window=ti_config.bollinger.window, window_dev=ti_config.bollinger.window_dev)
        df_copy['bollinger_mavg'] = bollinger_indicator.bollinger_mavg()
        df_copy['bollinger_hband'] = bollinger_indicator.bollinger_hband()
        df_copy['bollinger_lband'] = bollinger_indicator.bollinger_lband()
    if ti_config.stoch.enabled and has_ohlc:
        stoch_indicator = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=ti_config.stoch.window, smooth_window=ti_config.stoch.smooth_window)
        df_copy['STOCH_k'] = stoch_indicator.stoch()
        df_copy['STOCH_d'] = stoch_indicator.stoch_signal()
    if ti_config.atr.enabled and has_ohlc:
        df_copy['ATR_14'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=ti_config.atr.window).average_true_range()
    if ti_config.obv.enabled and has_close and has_volume:
        df_copy['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df_copy['close'], volume=df_copy['volume']).on_balance_volume()
    
    reverse_column_mapping = {v: k for k, v in column_mapping.items()}
    df_copy = df_copy.rename(columns=reverse_column_mapping)
    log.info("Technical indicators generated.")
    return df_copy

def generate_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Гeneрирует календарные признаки из DateTimeIndex."""
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        log.warning("DataFrame index is not DatetimeIndex. Attempting to convert.")
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            log.error(f"Failed to convert index to DatetimeIndex: {e}. Calendar features might be incorrect or missing.")
            return df_copy

    # Raw features (can be used for direct catboost cat_features or for mean encoding later)
    df_copy['hour_raw'] = df_copy.index.hour
    df_copy['dayofweek_raw'] = df_copy.index.dayofweek
    
    # Features as named in notebook that are used for mean encoding or direct use
    df_copy['hour'] = df_copy.index.hour 
    df_copy['dayofweek'] = df_copy.index.dayofweek 
    df_copy['dayofmonth'] = df_copy.index.day
    df_copy['month'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year # Added from notebook implicitly
    log.info("Calendar features generated.")
    return df_copy

def code_mean(df_to_encode: pd.DataFrame, cat_feature_col: str, real_target_col: str, global_mean: float) -> dict:
    """Рассчитывает средние значения таргета для категориального признака."""
    return df_to_encode.groupby(cat_feature_col)[real_target_col].mean().fillna(global_mean).to_dict()

def apply_mean_encoding(df_to_transform: pd.DataFrame, cat_feature_col: str, new_encoded_col_name: str, encoding_map: dict, global_mean: float) -> pd.DataFrame:
    """Применяет mean encoding к датафрейму используя предоставленную карту."""
    df_copy = df_to_transform.copy()
    df_copy[new_encoded_col_name] = df_copy[cat_feature_col].map(encoding_map).fillna(global_mean)
    return df_copy

def load_clearml_dataset_to_pandas(dataset_project: str, dataset_name: str, dataset_id: str) -> pd.DataFrame:
    try:
        dataset = None
        if dataset_id:
            dataset = Dataset.get(dataset_id=dataset_id)
        elif dataset_name and dataset_project:
            dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)
        else:
            log.error("Dataset ID or Project/Name not provided.")
            return pd.DataFrame()
        if not dataset: 
            log.error(f"Could not find ClearML dataset (Proj: {dataset_project}, Name: {dataset_name}, ID: {dataset_id}).")
            return pd.DataFrame()
        log.info(f"ClearML Dataset '{dataset.name}' (ID: {dataset.id}) found.")
        dataset_path = dataset.get_local_copy()
        data_file = next((f for f in dataset.list_files() if f.endswith(('.parquet', '.csv'))), None)
        if not data_file:
            log.error(f"No .parquet or .csv file found in dataset {dataset.id}")
            return pd.DataFrame()
        file_path = os.path.join(dataset_path, data_file)
        log.info(f"Reading data from: {file_path}")
        if data_file.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else: # .csv
            df = pd.read_csv(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                potential_date_cols = ['date', 'timestamp', 'Date', 'Timestamp', 'Time', 'datetime', 'Datetime']
                for col_name in df.columns:
                    if col_name in potential_date_cols:
                        try:
                            pd.to_datetime(df[col_name].iloc[:5], errors='raise') 
                            df[col_name] = pd.to_datetime(df[col_name])
                            df = df.set_index(col_name)
                            log.info(f"Automatically set '{col_name}' as DatetimeIndex for CSV.")
                            break
                        except Exception as e_idx:
                            log.warning(f"Could not auto-parse/set '{col_name}' as DatetimeIndex: {e_idx}.")
        if not isinstance(df.index, pd.DatetimeIndex):
             log.warning("Index is not DatetimeIndex after loading.")
        return df
    except Exception as e:
        log.error(f"Error loading ClearML dataset (Proj: {dataset_project}, Name: {dataset_name}, ID: {dataset_id}): {e}", exc_info=True)
        return pd.DataFrame()

@hydra.main(config_path="../../configs", config_name="catboost_config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_asset_ticker', type=str, default=cfg.data_preparation.target_asset_ticker)
    args, unknown = parser.parse_known_args()
    target_asset_ticker = args.target_asset_ticker
    task = Task.init(
        project_name=cfg.project_name,
        task_name=f"{cfg.data_preparation.task_name_prefix}_{target_asset_ticker}",
        tags=list(cfg.global_tags) + [target_asset_ticker, "data_preparation"],
        output_uri=True
    )
    current_task_params = OmegaConf.to_container(cfg, resolve=True)
    current_task_params['data_preparation']['target_asset_ticker'] = target_asset_ticker
    task.connect(current_task_params, name='Effective_Hydra_Configuration')
    log.info(f"Starting data preparation for asset: {target_asset_ticker}")
    dp_cfg = cfg.data_preparation
    
    log.info("Loading market data...")
    df_market = load_clearml_dataset_to_pandas(
        dataset_project=dp_cfg.input_market_data.project,
        dataset_name=dp_cfg.input_market_data.name,
        dataset_id=dp_cfg.input_market_data.id
    )
    if df_market.empty:
        log.error(f"Market data for {target_asset_ticker} is empty. Exiting."); task.close(); return
    df_processed = df_market.copy().sort_index()
    target_col = dp_cfg.target_col_name
    if target_col not in df_processed.columns:
        log.error(f"Target column '{target_col}' not found. Available: {df_processed.columns.tolist()}. Exiting."); task.close(); return

    if dp_cfg.input_news_data.enabled:
        log.info("Loading news data...")
        df_news = load_clearml_dataset_to_pandas(
            dataset_project=dp_cfg.input_news_data.project,
            dataset_name=dp_cfg.input_news_data.name,
            dataset_id=dp_cfg.input_news_data.id
        )
        if not df_news.empty:
            log.info(f"Merging news data with market data (market: {len(df_processed)}, news: {len(df_news)})")
            if not isinstance(df_news.index, pd.DatetimeIndex):
                log.warning("News data index is not DatetimeIndex. Attempting to convert.")
                try: df_news.index = pd.to_datetime(df_news.index)
                except: log.error("Failed to convert news index. Merge may fail."); df_news = pd.DataFrame()
            if not df_news.empty:
                df_news = df_news.sort_index()
                news_cols_to_use = [col for col in dp_cfg.news_cols_to_merge if col in df_news.columns]
                if not news_cols_to_use:
                     log.warning("Specified news_cols_to_merge not found in news data. Skipping news merge.")
                else:
                    log.info(f"Using news columns for merge: {news_cols_to_use}")
                    df_processed = pd.merge_asof(df_processed, df_news[news_cols_to_use], 
                                                   left_index=True, right_index=True, direction='backward')
                    for nc in news_cols_to_use: # Fill NaNs from merge_asof
                        if nc in df_processed.columns: df_processed[nc] = df_processed[nc].fillna(0) 
        else: log.warning("News data is empty or not loaded. Skipping merge.")
    else: log.info("News data loading is disabled.")

    log.info("Generating target, lags, calendar, and technical features...")
    df_processed['y_model_future'] = df_processed[target_col].shift(-dp_cfg.target_horizon_hours)
    df_processed['y_pred_naive_source'] = df_processed[target_col] 
    df_processed = generate_lags(df_processed, target_col, list(dp_cfg.lags_config))
    df_processed = generate_calendar_features(df_processed)
    df_processed = generate_technical_indicators(df_processed, dp_cfg.technical_indicators_config)
    
    max_lag = max(dp_cfg.lags_config) if dp_cfg.lags_config else 0
    essential_na_check_cols = ['y_model_future']
    if max_lag > 0 : essential_na_check_cols.append(f'{target_col}_lag_{max_lag}h')
    initial_len = len(df_processed)
    df_processed.dropna(subset=essential_na_check_cols, inplace=True)
    log.info(f"Dropped {initial_len - len(df_processed)} rows due to NaNs from target shift/max lag.")
    if df_processed.empty:
        log.error("DataFrame empty after essential NaN handling. Cannot proceed."); task.close(); return

    log.info(f"Splitting data {'shuffled' if dp_cfg.shuffle_data_before_split else 'sequentially'}...")
    df_processed = df_processed.sort_index()
    y = df_processed['y_model_future']
    X_full_features = df_processed.drop(columns=['y_model_future'])
    
    split_random_state = cfg.train_model.catboost_model_params.get('random_seed', 42) if dp_cfg.shuffle_data_before_split else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full_features, y, 
        test_size=(dp_cfg.validation_size_ratio + dp_cfg.test_size_ratio),
        shuffle=dp_cfg.shuffle_data_before_split,
        random_state=split_random_state
    )
    relative_test_size = 0.0
    if (dp_cfg.validation_size_ratio + dp_cfg.test_size_ratio) > 1e-6: # Avoid division by zero
        relative_test_size = dp_cfg.test_size_ratio / (dp_cfg.validation_size_ratio + dp_cfg.test_size_ratio)
    
    if relative_test_size > 0 and len(X_temp) > 1:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=relative_test_size,
            shuffle=dp_cfg.shuffle_data_before_split,
            random_state=split_random_state
        )
    elif len(X_temp) <=1 and relative_test_size > 0:
         log.warning(f"Not enough temp_df data ({len(X_temp)}) to split to val/test. Assigning all to validation.")
         X_val, y_val = X_temp, y_temp
         X_test, y_test = pd.DataFrame(columns=X_temp.columns), pd.Series(dtype=y_temp.dtype, name=y_temp.name)
    else: # No test set needed or not enough data for it
        X_val, y_val = X_temp, y_temp
        X_test, y_test = pd.DataFrame(columns=X_temp.columns), pd.Series(dtype=y_temp.dtype, name=y_temp.name)
    log.info(f"Data split: Train ({len(X_train)}), Validation ({len(X_val)}), Test ({len(X_test)})")

    log.info("Applying Mean Encoding based on training set...")
    mean_encoding_maps = {}
    global_mean_target_train = y_train.mean()
    categorical_features_for_model = [] 

    for feat_cat_config_name in dp_cfg.features_for_mean_encoding:
        feature_to_encode_on = feat_cat_config_name
        if feat_cat_config_name == 'hour' and not dp_cfg.use_mean_encoding_for_hour:
            if 'hour_raw' in X_train.columns and 'hour_raw' not in categorical_features_for_model:
                categorical_features_for_model.append('hour_raw')
            log.info(f"Skipping mean encoding for 'hour', using 'hour_raw' as categorical.")
            continue
        
        if feature_to_encode_on not in X_train.columns:
            log.warning(f"Feature '{feature_to_encode_on}' for mean encoding not found in X_train. Skipping.")
            raw_version = f"{feature_to_encode_on}_raw" # Check if raw version should be added as categorical
            if raw_version in X_train.columns and raw_version not in categorical_features_for_model:
                categorical_features_for_model.append(raw_version)
            continue

        log.info(f"Mean encoding for feature: {feature_to_encode_on}")
        temp_train_for_encoding = X_train[[feature_to_encode_on]].copy()
        temp_train_for_encoding['__target_for_encoding__'] = y_train
        current_map = code_mean(temp_train_for_encoding, feature_to_encode_on, '__target_for_encoding__', global_mean_target_train)
        mean_encoding_maps[feature_to_encode_on] = current_map
        
        encoded_col_name = f'{feature_to_encode_on}_avg_target'
        X_train = apply_mean_encoding(X_train, feature_to_encode_on, encoded_col_name, current_map, global_mean_target_train)
        X_val = apply_mean_encoding(X_val, feature_to_encode_on, encoded_col_name, current_map, global_mean_target_train)
        if not X_test.empty: X_test = apply_mean_encoding(X_test, feature_to_encode_on, encoded_col_name, current_map, global_mean_target_train)
        
        X_train.drop(feature_to_encode_on, axis=1, inplace=True, errors='ignore')
        X_val.drop(feature_to_encode_on, axis=1, inplace=True, errors='ignore')
        if not X_test.empty: X_test.drop(feature_to_encode_on, axis=1, inplace=True, errors='ignore')
        log.info(f"Applied mean encoding for {feature_to_encode_on}, created {encoded_col_name}, original dropped.")

    for raw_cal_feat in ['hour_raw', 'dayofweek_raw']:
        base_cal_feat = raw_cal_feat.replace('_raw','')
        is_base_mean_encoded = (base_cal_feat in dp_cfg.features_for_mean_encoding and 
                                not (base_cal_feat == 'hour' and not dp_cfg.use_mean_encoding_for_hour))
        if not is_base_mean_encoded: # If base was not mean encoded, use the _raw as categorical
            if raw_cal_feat in X_train.columns and raw_cal_feat not in categorical_features_for_model:
                categorical_features_for_model.append(raw_cal_feat)
                log.info(f"Added '{raw_cal_feat}' to categorical features as base '{base_cal_feat}' was not mean encoded.")

    log.info(f"Final categorical features for model: {categorical_features_for_model}")
    cat_features_artifact_name = f"{target_asset_ticker}_{dp_cfg.categorical_features_list_artifact_name}"
    with open(cat_features_artifact_name, 'w') as f_cat_list:
        json.dump(categorical_features_for_model, f_cat_list, indent=4)
    task.upload_artifact(name="categorical_features_list", artifact_object=cat_features_artifact_name)
    log.info(f"Categorical features list uploaded: {cat_features_artifact_name}")
    
    mean_map_artifact_name = f"{target_asset_ticker}_{dp_cfg.mean_encoding_map_artifact_name}"
    with open(mean_map_artifact_name, 'w') as f_map:
        json.dump(mean_encoding_maps, f_map, indent=4)
    task.upload_artifact(name="mean_encoding_maps", artifact_object=mean_map_artifact_name)
    log.info(f"Mean encoding maps uploaded: {mean_map_artifact_name}")

    log.info("Finalizing feature sets by dropping unnecessary columns...")
    cols_to_drop_runtime = [col.replace("TARGET_REPLACE", target_col) if col == "TARGET_REPLACE" else col for col in dp_cfg.cols_to_drop_from_X]
    if 'y_pred_naive_source' not in cols_to_drop_runtime: 
        cols_to_drop_runtime.append('y_pred_naive_source')

    for feat_raw in ['hour_raw', 'dayofweek_raw']:
        feat_base = feat_raw.replace('_raw', '')
        was_base_mean_encoded = (feat_base in dp_cfg.features_for_mean_encoding and 
                                 not (feat_base == 'hour' and not dp_cfg.use_mean_encoding_for_hour))
        if was_base_mean_encoded and feat_raw in X_train.columns and feat_raw not in categorical_features_for_model:
            if feat_raw not in cols_to_drop_runtime:
                cols_to_drop_runtime.append(feat_raw)
                log.info(f"Adding '{feat_raw}' to drop list as base '{feat_base}' was mean encoded and raw not used as categorical.")
    
    X_train = X_train.drop(columns=[col for col in cols_to_drop_runtime if col in X_train.columns], errors='ignore')
    X_val = X_val.drop(columns=[col for col in cols_to_drop_runtime if col in X_val.columns], errors='ignore')
    if not X_test.empty:
        X_test = X_test.drop(columns=[col for col in cols_to_drop_runtime if col in X_test.columns], errors='ignore')
    
    final_train_cols = X_train.columns.tolist() # Ensure consistent column order and presence
    X_val = X_val.reindex(columns=final_train_cols)
    if not X_test.empty: X_test = X_test.reindex(columns=final_train_cols)
    
    # Final NaN fill after all ops (e.g. from reindex or residual from TIs)
    X_train = X_train.ffill().bfill().fillna(0)
    X_val = X_val.ffill().bfill().fillna(0)
    if not X_test.empty: X_test = X_test.ffill().bfill().fillna(0)
    log.info(f"Final shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    if not X_train.empty: log.info(f"Final X_train columns: {X_train.columns.tolist()}")

    log.info("Saving processed datasets as Parquet artifacts...")
    def save_and_upload_df(df_to_save: pd.DataFrame, artifact_name_suffix: str, base_artifact_name: str):
        is_test_artifact_type = (base_artifact_name == "test_data" and 
                                 artifact_name_suffix in ["features", "target", "naive_baseline"])
        if df_to_save.empty and not is_test_artifact_type:
             log.warning(f"DataFrame for {target_asset_ticker}_{base_artifact_name}_{artifact_name_suffix} is empty. Skipping.")
             return
        
        if isinstance(df_to_save, pd.Series): 
             df_to_save = df_to_save.to_frame(name=df_to_save.name or target_col if artifact_name_suffix == "target" else 'y_pred_naive_source')
        
        if df_to_save.empty and is_test_artifact_type: 
            log.info(f"Test artifact {target_asset_ticker}_{base_artifact_name}_{artifact_name_suffix} is empty. Creating empty parquet.")
            if df_to_save.columns.empty and not df_to_save.index.empty: 
                df_to_save = pd.DataFrame(index=df_to_save.index, columns=['data']) # Preserve index if any
            elif df_to_save.columns.empty and df_to_save.index.empty:
                 df_to_save = pd.DataFrame(columns=['data']) 

        filename = f"{target_asset_ticker}_{base_artifact_name}_{artifact_name_suffix}.parquet"
        try:
            df_to_save.to_parquet(filename)
            task.upload_artifact(name=f"{base_artifact_name}_{artifact_name_suffix}", artifact_object=filename)
            log.info(f"Uploaded: {filename} as {base_artifact_name}_{artifact_name_suffix}")
        except Exception as e_parq:
            log.error(f"Failed to save/upload parquet {filename}: {e_parq}", exc_info=True)
            try: # Fallback to CSV
                csv_filename = filename.replace('.parquet', '.csv')
                df_to_save.to_csv(csv_filename, index=(isinstance(df_to_save.index, pd.MultiIndex) or df_to_save.index.name is not None))
                task.upload_artifact(name=f"{base_artifact_name}_{artifact_name_suffix}_csv_fallback", artifact_object=csv_filename)
                log.info(f"Uploaded CSV fallback: {csv_filename}")
            except Exception as e_csv:
                log.error(f"Failed to save CSV fallback {csv_filename}: {e_csv}")

    save_and_upload_df(X_train, "features", "train_data")
    save_and_upload_df(y_train, "target", "train_data")
    save_and_upload_df(X_val, "features", "validation_data")
    save_and_upload_df(y_val, "target", "validation_data")
    
    y_pred_naive_test_set = pd.Series(dtype=float, name='y_pred_naive_source') # Ensure name for saving
    if 'y_pred_naive_source' in X_full_features.columns and not y_test.empty: 
        y_pred_naive_test_set = X_full_features.loc[y_test.index, 'y_pred_naive_source']
    elif not y_test.empty: 
        log.warning("'y_pred_naive_source' not in X_full_features. Naive baseline might be empty.")

    save_and_upload_df(X_test, "features", "test_data") 
    save_and_upload_df(y_test, "target", "test_data")   
    save_and_upload_df(y_pred_naive_test_set, "naive_baseline", "test_data")
    
    log.info(f"Data preparation for {target_asset_ticker} completed.")
    task.close()

if __name__ == '__main__':
    main() 