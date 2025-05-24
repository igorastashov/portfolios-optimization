import argparse
import pandas as pd
import time
import requests
import logging
from datetime import datetime, timedelta

from clearml import Task, Dataset, Logger
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

MAX_CALLS_PER_MINUTE_DEFAULT = 5
MAX_CALLS_PER_DAY_DEFAULT = 500

def fetch_news_batch(api_key: str, tickers: list[str], time_from: str = None, time_to: str = None, limit: int = 200):
    """Fetches a single batch of news from Alpha Vantage."""
    tickers_str = ",".join(tickers)
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "tickers": tickers_str,
        "limit": str(limit),
        "sort": "EARLIEST"
    }
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "Information" in data and "Rate Limit" in data["Information"]:
            log.warning(f"Alpha Vantage API rate limit likely hit: {data['Information']}")
            return None, True

        if "feed" not in data or not data["feed"]:
            log.info("No news feed found in response for the given parameters.")
            return pd.DataFrame(), False
        
        df_batch = pd.DataFrame(data["feed"])
        df_batch['date_published'] = pd.to_datetime(df_batch['time_published'], format='%Y%m%dT%H%M%S')
        df_batch.drop(columns=['time_published'], inplace=True)
        log.info(f"Fetched batch of {len(df_batch)} news items.")
        return df_batch, False
        
    except requests.exceptions.RequestException as e:
        log.error(f"API request failed: {e}")
        return pd.DataFrame(), False
    except Exception as e:
        log.error(f"Error processing news batch: {e}", exc_info=True)
        return pd.DataFrame(), False

@hydra.main(config_path="../../configs", config_name="data_fetch_config", version_base=None)
def main(cfg: DictConfig) -> None:
    task = Task.init(
        project_name=cfg.project_name,
        task_name=cfg.data_sources.alphavantage_news.task_name,
        tags=["data_fetch", "news", "alphavantage"] + list(cfg.tags) 
    )
    task.connect(OmegaConf.to_container(cfg, resolve=True), name='Effective_Hydra_Configuration')
    logger = task.get_logger()

    log.info("Starting Alpha Vantage news fetching process...")
    av_cfg = cfg.data_sources.alphavantage_news

    api_key = av_cfg.api_key
    if not api_key:
        log.error("Alpha Vantage API key is not configured. Aborting.")
        task.close(); return

    tickers_of_interest = list(av_cfg.tickers_of_interest)
    if not tickers_of_interest:
        log.warning("No tickers specified for Alpha Vantage news. Process might yield broad results or fail.")

    api_calls_this_minute = 0
    api_calls_this_day = 0
    minute_start_time = time.time()
    call_delay_seconds = av_cfg.get("call_delay_seconds", 15)

    all_news_df = pd.DataFrame()
    
    dataset_name = av_cfg.output_dataset_name
    dataset_project = cfg.project_name
    time_from_str = None
    initial_fetch_mode = "config_start_date"

    try:
        latest_dataset = Dataset.get(
            dataset_project=dataset_project,
            dataset_name=dataset_name,
            only_completed=True,
        )
        if latest_dataset:
            log.info(f"Found existing dataset '{dataset_name}' (ID: {latest_dataset.id}). Will try to fetch incrementally.")
            log.warning("Incremental fetching logic from existing dataset not fully implemented. Defaulting to config start_date/lookback.")
            log.warning("Incremental fetching logic from existing dataset not fully implemented. Defaulting to config start_date/lookback.")
            initial_fetch_mode = "config_start_date"
    except Exception:
        log.info(f"No existing dataset '{dataset_name}' found or error accessing it. Will perform initial fetch.")
        initial_fetch_mode = "config_start_date"

    if initial_fetch_mode == "config_start_date":
        if av_cfg.get("start_date"): # YYYY-MM-DD
            dt_from = datetime.strptime(av_cfg.start_date, '%Y-%m-%d')
            time_from_str = dt_from.strftime('%Y%m%dT%H%M')
            log.info(f"Initial fetch from configured start_date: {av_cfg.start_date} (API format: {time_from_str})")
        elif av_cfg.get("lookback_days_initial"):
            dt_from = datetime.utcnow() - timedelta(days=av_cfg.lookback_days_initial)
            time_from_str = dt_from.strftime('%Y%m%dT%H%M')
            log.info(f"Initial fetch for the last {av_cfg.lookback_days_initial} days. From: {time_from_str}")
        else:
            log.warning("Neither start_date nor lookback_days_initial configured for initial fetch. Fetching all available (up to API limit per call).")
    
    current_time_from = time_from_str
    max_iterations = av_cfg.get("max_pagination_iterations", 50)
    
    for i in range(max_iterations):
        log.info(f"Fetching attempt {i+1}/{max_iterations}. Current time_from: {current_time_from}")
        
        if api_calls_this_minute >= av_cfg.get("max_calls_per_minute", MAX_CALLS_PER_MINUTE_DEFAULT):
            sleep_duration = max(0, 60 - (time.time() - minute_start_time))
            log.info(f"Minute rate limit hit. Sleeping for {sleep_duration:.2f} seconds.")
            time.sleep(sleep_duration)
            api_calls_this_minute = 0
            minute_start_time = time.time()
        
        if api_calls_this_day >= av_cfg.get("max_calls_per_day", MAX_CALLS_PER_DAY_DEFAULT):
            log.error("Daily API call limit reached. Stopping fetch.")
            break

        df_batch, rate_limit_hit = fetch_news_batch(api_key, tickers_of_interest, time_from=current_time_from, limit=av_cfg.get("limit_per_call", 200))
        api_calls_this_minute += 1
        api_calls_this_day += 1

        if rate_limit_hit:
            log.warning("Rate limit hit during fetch_news_batch. Waiting and will retry if appropriate, or stop.")
            time.sleep(call_delay_seconds * 5)
            continue

        if df_batch is None or df_batch.empty:
            log.info("No more news items returned in this batch or error occurred. Ending pagination.")
            break 

        all_news_df = pd.concat([all_news_df, df_batch], ignore_index=True)
        all_news_df.drop_duplicates(subset=['url', 'title', 'date_published'], inplace=True)
        all_news_df.sort_values(by='date_published', ascending=True, inplace=True)
        
        last_article_time = df_batch['date_published'].max()
        current_time_from = (last_article_time + timedelta(seconds=1)).strftime('%Y%m%dT%H%M%S')
        current_time_from = (last_article_time + timedelta(minutes=1)).strftime('%Y%m%dT%H%M') 

        logger.report_scalar(title="API Calls", series="Per Minute", value=api_calls_this_minute, iteration=i)
        logger.report_scalar(title="API Calls", series="Per Day", value=api_calls_this_day, iteration=i)
        logger.report_scalar(title="News Items", series="Fetched Total", value=len(all_news_df), iteration=i)

        log.info(f"Pausing for {call_delay_seconds} seconds before next API call...")
        time.sleep(call_delay_seconds)

    if not all_news_df.empty:
        log.info(f"Total news items fetched and deduplicated: {len(all_news_df)}")
        all_news_df = all_news_df.sort_values(by='date_published', ascending=False)
        
        parquet_file = "alphavantage_news.parquet"
        all_news_df.to_parquet(parquet_file, index=False)
        task.upload_artifact(name="alphavantage_news_parquet", artifact_object=parquet_file)
        log.info(f"News data saved to {parquet_file} and uploaded as artifact.")

        try:
            dataset = Dataset.create(
                dataset_project=dataset_project,
                dataset_name=dataset_name,
                parent_datasets=[latest_dataset.id] if 'latest_dataset' in locals() and latest_dataset else None
            )
            dataset.add_files(parquet_file)
            dataset.set_metadata({
                "source": "Alpha Vantage", 
                "tickers_fetched": tickers_of_interest,
                "total_articles": len(all_news_df),
                "fetch_range_start": all_news_df['date_published'].min().isoformat() if not all_news_df.empty else None,
                "fetch_range_end": all_news_df['date_published'].max().isoformat() if not all_news_df.empty else None,
                "last_api_call_params": {"time_from": current_time_from}
            })
            dataset.upload(output_url=av_cfg.get("clearml_dataset_storage_url"))
            dataset.finalize()
            log.info(f"ClearML Dataset '{dataset_name}' (ID: {dataset.id}) created/updated and finalized.")
            logger.report_text(f"Dataset ID: {dataset.id}", level=logging.INFO)
        except Exception as e_ds:
            log.error(f"Error creating/uploading ClearML Dataset: {e_ds}", exc_info=True)
    else:
        log.info("No news data fetched. Dataset will not be created/updated.")

    task.close()
    log.info("Alpha Vantage news fetching task finished.")

if __name__ == '__main__':
    main() 