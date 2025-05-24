import argparse
import pandas as pd
import argparse
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import os

from clearml import Task, Dataset, Logger
import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import ccxt
except ImportError:
    ccxt = None

log = logging.getLogger(__name__)

def get_binance_client(api_key: str = None, secret_key: str = None) -> ccxt.binance | None:
    """Initializes and returns a Binance client using ccxt."""
    if not ccxt:
        log.error("CCXT library is not installed. Cannot fetch Binance data. pip install ccxt")
        return None
    try:
        exchange_params = {
            'rateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        }
        if api_key and secret_key:
            exchange_params['apiKey'] = api_key
            exchange_params['secret'] = secret_key
            log.info("Initializing Binance client with API key.")
        else:
            log.info("Initializing Binance client without API key (public data only).")
        
        exchange = ccxt.binance(exchange_params)
        return exchange
    except Exception as e:
        log.error(f"Failed to initialize CCXT Binance client: {e}", exc_info=True)
        return None

def fetch_ohlcv_data(client: ccxt.binance, symbol: str, timeframe: str, since_timestamp: int = None, limit: int = 1000) -> pd.DataFrame:
    """Fetches OHLCV data for a single symbol and timeframe."""
    try:
        log.info(f"Fetching OHLCV for {symbol}, timeframe {timeframe}, since {since_timestamp}, limit {limit}")
        ohlcv = client.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
        if not ohlcv:
            log.warning(f"No data returned for {symbol} with timeframe {timeframe} and since {since_timestamp}.")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        log.info(f"Fetched {len(df)} records for {symbol}.")
        return df
    except ccxt.NetworkError as e:
        log.error(f"CCXT NetworkError fetching {symbol}: {e}")
        time.sleep(client.rateLimit / 1000 * 2)
    except ccxt.ExchangeError as e:
        log.error(f"CCXT ExchangeError fetching {symbol}: {e}")
    except Exception as e:
        log.error(f"Generic error fetching OHLCV for {symbol}: {e}", exc_info=True)
    return pd.DataFrame()

@hydra.main(config_path="../../configs", config_name="data_fetch_config", version_base=None)
def main(cfg: DictConfig) -> None:
    current_cml_task = Task.init(
        project_name=cfg.project_name,
        task_name=cfg.data_sources.binance.task_name,
        tags=["data_fetch", "market_data", "binance"] + list(cfg.tags)
    )
    current_cml_task.connect(OmegaConf.to_container(cfg, resolve=True), name='Effective_Hydra_Configuration')
    logger = current_cml_task.get_logger()

    log.info("Starting Binance market data fetching process...")
    bin_cfg = cfg.data_sources.binance

    if not ccxt:
        log.error("CCXT library not available. Halting Binance data fetch.")
        current_cml_task.close(); return

    client = get_binance_client(api_key=bin_cfg.get("api_key"), secret_key=bin_cfg.get("secret_key"))
    if not client:
        log.error("Failed to initialize Binance client. Aborting.")
        current_cml_task.close(); return

    symbols_to_fetch = list(bin_cfg.tickers)
    timeframes_to_fetch = [bin_cfg.interval] if isinstance(bin_cfg.interval, str) else list(bin_cfg.interval)
    limit_per_call = bin_cfg.get("limit_per_call", 1000)
    default_lookback_days = bin_cfg.get("default_lookback_days_if_new", 7) 

    for symbol_slash_sep in symbols_to_fetch:
        symbol_ccxt_format = symbol_slash_sep.replace('/', '') # e.g., BTC/USDT -> BTCUSDT for Binance API via CCXT
        log.info(f"Processing symbol: {symbol_ccxt_format} (original: {symbol_slash_sep})")

        for timeframe in timeframes_to_fetch:
            log.info(f"Fetching data for {symbol_ccxt_format}, timeframe {timeframe}")
            all_data_for_symbol_timeframe = pd.DataFrame()

            dataset_name = bin_cfg.get("output_dataset_name_template", "market_data_{ticker}_{timeframe}").format(ticker=symbol_ccxt_format, timeframe=timeframe)
            dataset_project = cfg.project_name
            since_ms = None

            try:
                latest_dataset = Dataset.get(
                    dataset_project=dataset_project,
                    dataset_name=dataset_name,
                    only_completed=True,
                )
                if latest_dataset:
                    log.info(f"Found existing dataset '{dataset_name}' (ID: {latest_dataset.id}). Attempting incremental fetch.")
                    last_ts_from_meta = latest_dataset.get_metadata().get("latest_data_timestamp_ms")
                    if last_ts_from_meta:
                        since_ms = int(last_ts_from_meta) + 1
                        log.info(f"Incremental fetch: last timestamp from metadata is {last_ts_from_meta}, fetching from {since_ms}")
                    else:
                        since_ms = int((datetime.utcnow() - timedelta(days=default_lookback_days)).timestamp() * 1000)
                        log.warning(f"No last timestamp in dataset metadata. Fetching last {default_lookback_days} days for {symbol_ccxt_format} {timeframe}.")
            except Exception:
                log.info(f"No existing dataset '{dataset_name}' found for {symbol_ccxt_format} {timeframe}. Performing initial historical fetch.")
                if bin_cfg.get("days_to_fetch_historical"):
                     since_ms = int((datetime.utcnow() - timedelta(days=bin_cfg.days_to_fetch_historical)).timestamp() * 1000)
                     log.info(f"Initial fetch: {bin_cfg.days_to_fetch_historical} days for {symbol_ccxt_format} {timeframe}, from {since_ms}.")
                else:
                    since_ms = int((datetime.utcnow() - timedelta(days=default_lookback_days)).timestamp() * 1000)
                    log.warning(f"Initial fetch: days_to_fetch_historical not set. Fetching last {default_lookback_days} days for {symbol_ccxt_format} {timeframe}.")
            
            current_since_timestamp = since_ms
            max_fetch_iterations = bin_cfg.get("max_fetch_iterations_per_symbol", 100)

            for i in range(max_fetch_iterations):
                log.info(f"Fetch iteration {i+1}/{max_fetch_iterations} for {symbol_ccxt_format} {timeframe}, since {current_since_timestamp}")
                df_batch = fetch_ohlcv_data(client, symbol_ccxt_format, timeframe, since_timestamp=current_since_timestamp, limit=limit_per_call)
                df_batch = fetch_ohlcv_data(client, symbol_ccxt_format, timeframe, since_timestamp=current_since_timestamp, limit=limit_per_call)
                
                if df_batch.empty:
                    log.info(f"No more data returned for {symbol_ccxt_format} {timeframe} since {current_since_timestamp}. Ending pagination.")
                    break
                
                all_data_for_symbol_timeframe = pd.concat([all_data_for_symbol_timeframe, df_batch], ignore_index=True)
                all_data_for_symbol_timeframe.drop_duplicates(subset=['timestamp'], inplace=True)
                
                last_fetched_timestamp_ms = df_batch['timestamp'].iloc[-1].timestamp() * 1000
                current_since_timestamp = int(last_fetched_timestamp_ms) + 1
                
                if current_since_timestamp > (datetime.utcnow().timestamp() * 1000 - client.parse_timeframe(timeframe) * 1000):
                    log.info(f"Fetched data up to near current time for {symbol_ccxt_format} {timeframe}.")
                    break
                time.sleep(bin_cfg.get("delay_between_calls_sec", 1))
            
            if not all_data_for_symbol_timeframe.empty:
                all_data_for_symbol_timeframe = all_data_for_symbol_timeframe.sort_values(by='timestamp').reset_index(drop=True)
                log.info(f"Total OHLCV records fetched for {symbol_ccxt_format} {timeframe}: {len(all_data_for_symbol_timeframe)}")
                
                parquet_file = f"binance_ohlcv_{symbol_ccxt_format.replace('/','_')}_{timeframe}.parquet"
                all_data_for_symbol_timeframe.to_parquet(parquet_file, index=False)
                current_cml_task.upload_artifact(name=f"ohlcv_parquet_{symbol_ccxt_format}_{timeframe}", artifact_object=parquet_file)
                log.info(f"Data for {symbol_ccxt_format} {timeframe} saved to {parquet_file} and uploaded as artifact.")

                try:
                    parent_dataset_id = latest_dataset.id if 'latest_dataset' in locals() and latest_dataset and latest_dataset.id else None
                    output_dataset = Dataset.create(
                        dataset_project=dataset_project,
                        dataset_name=dataset_name,
                        parent_datasets=[parent_dataset_id] if parent_dataset_id else None 
                    )
                    output_dataset.add_files(parquet_file)
                    new_latest_ts_ms = all_data_for_symbol_timeframe['timestamp'].iloc[-1].timestamp() * 1000
                    output_dataset.set_metadata({
                        "source": "Binance", 
                        "symbol": symbol_ccxt_format,
                        "timeframe": timeframe,
                        "total_records": len(all_data_for_symbol_timeframe),
                        "data_range_start": all_data_for_symbol_timeframe['timestamp'].min().isoformat(),
                        "data_range_end": all_data_for_symbol_timeframe['timestamp'].max().isoformat(),
                        "latest_data_timestamp_ms": int(new_latest_ts_ms) # Store for incremental logic
                    })
                    output_dataset.upload(output_url=bin_cfg.get("clearml_dataset_storage_url"))
                    output_dataset.finalize()
                    log.info(f"ClearML Dataset '{dataset_name}' (ID: {output_dataset.id}) created/updated for {symbol_ccxt_format} {timeframe}.")
                    logger.report_text(f"Dataset ID for {symbol_ccxt_format} {timeframe}: {output_dataset.id}", level=logging.INFO)
                except Exception as e_ds:
                    log.error(f"Error creating/uploading ClearML Dataset for {symbol_ccxt_format} {timeframe}: {e_ds}", exc_info=True)
            else:
                log.info(f"No data fetched for {symbol_ccxt_format} {timeframe}. Dataset will not be created/updated.")
            
            if 'latest_dataset' in locals(): del latest_dataset

    current_cml_task.close()
    log.info("Binance market data fetching task finished.")

if __name__ == '__main__':
    main() 