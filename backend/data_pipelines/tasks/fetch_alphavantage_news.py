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

MAX_CALLS_PER_MINUTE_DEFAULT = 5 # Alpha Vantage free tier limit
MAX_CALLS_PER_DAY_DEFAULT = 500 # Alpha Vantage free tier limit (example)

def fetch_news_batch(api_key: str, tickers: list[str], time_from: str = None, time_to: str = None, limit: int = 200):
    """Fetches a single batch of news from Alpha Vantage."""
    # Alpha Vantage API requires tickers to be a comma-separated string
    tickers_str = ",".join(tickers)
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "tickers": tickers_str,
        "limit": str(limit), # API expects string for limit
        "sort": "EARLIEST" # To fetch chronologically for pagination
    }
    if time_from:
        params["time_from"] = time_from # Format YYYYMMDDTHHMM
    if time_to:
        params["time_to"] = time_to

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        if "Information" in data and "Rate Limit" in data["Information"]:
            log.warning(f"Alpha Vantage API rate limit likely hit: {data['Information']}")
            return None, True # Indicates rate limit hit

        if "feed" not in data or not data["feed"]:
            log.info("No news feed found in response for the given parameters.")
            return pd.DataFrame(), False # Empty DataFrame, no rate limit issue
        
        df_batch = pd.DataFrame(data["feed"])
        # Convert time_published to datetime and rename
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
        # Depending on API behavior, this might fetch general news or require specific tickers.
        # For now, we proceed, but this should be clarified based on API docs for 'NEWS_SENTIMENT' without tickers.

    # Pagination and rate limiting parameters
    # Use API limits from config, with defaults for safety
    api_calls_this_minute = 0
    api_calls_this_day = 0
    minute_start_time = time.time()
    # daily_limit = av_cfg.get("max_calls_per_day", MAX_CALLS_PER_DAY_DEFAULT)
    # minutely_limit = av_cfg.get("max_calls_per_minute", MAX_CALLS_PER_MINUTE_DEFAULT)
    # call_delay_seconds = 60 / minutely_limit if minutely_limit > 0 else 12 # e.g., 12s for 5 calls/min
    call_delay_seconds = av_cfg.get("call_delay_seconds", 15) # Default to 15s pause (4 calls/min)

    all_news_df = pd.DataFrame()
    
    # Determine time_from for the initial fetch
    # If a ClearML dataset exists, fetch news since the last entry. Otherwise, from config start_date.
    dataset_name = av_cfg.output_dataset_name
    dataset_project = cfg.project_name # Or a specific dataset project from av_cfg
    time_from_str = None
    initial_fetch_mode = "config_start_date"

    try:
        latest_dataset = Dataset.get(
            dataset_project=dataset_project,
            dataset_name=dataset_name,
            only_completed=True,
            # auto_prefetch=False # Don't download files yet
        )
        if latest_dataset:
            log.info(f"Found existing dataset '{dataset_name}' (ID: {latest_dataset.id}). Will try to fetch incrementally.")
            # To fetch incrementally, we need the timestamp of the latest news item in the dataset.
            # This requires downloading a sample or metadata if ClearML dataset stores it.
            # For simplicity here, we will assume the dataset has a file like 'latest_news_timestamp.txt' or similar
            # or we parse the latest Parquet file if it's small enough.
            # Placeholder: This logic needs to be robust.
            # For now, if dataset exists, we fall back to config date to avoid complexity.
            # A more robust way is to store the last fetched timestamp as a task parameter or artifact.
            log.warning("Incremental fetching logic from existing dataset not fully implemented. Defaulting to config start_date/lookback.")
            initial_fetch_mode = "config_start_date" # Re-affirm or change based on actual incremental logic capability
    except Exception:
        log.info(f"No existing dataset '{dataset_name}' found or error accessing it. Will perform initial fetch.")
        initial_fetch_mode = "config_start_date"

    if initial_fetch_mode == "config_start_date":
        if av_cfg.get("start_date"): # YYYY-MM-DD
            dt_from = datetime.strptime(av_cfg.start_date, '%Y-%m-%d')
            time_from_str = dt_from.strftime('%Y%m%dT%H%M')
            log.info(f"Initial fetch from configured start_date: {av_cfg.start_date} (API format: {time_from_str})")
        elif av_cfg.get("lookback_days_initial"): # Number of days to look back for the very first fetch
            dt_from = datetime.utcnow() - timedelta(days=av_cfg.lookback_days_initial)
            time_from_str = dt_from.strftime('%Y%m%dT%H%M')
            log.info(f"Initial fetch for the last {av_cfg.lookback_days_initial} days. From: {time_from_str}")
        else:
            log.warning("Neither start_date nor lookback_days_initial configured for initial fetch. Fetching all available (up to API limit per call).")
            # time_from_str will be None, fetching latest news
    
    # For iterative fetching (pagination based on time)
    current_time_from = time_from_str
    max_iterations = av_cfg.get("max_pagination_iterations", 50) # Safety break for pagination loop
    
    for i in range(max_iterations):
        log.info(f"Fetching attempt {i+1}/{max_iterations}. Current time_from: {current_time_from}")
        
        # Basic rate limiting (simplified)
        if api_calls_this_minute >= av_cfg.get("max_calls_per_minute", MAX_CALLS_PER_MINUTE_DEFAULT):
            sleep_duration = max(0, 60 - (time.time() - minute_start_time))
            log.info(f"Minute rate limit hit. Sleeping for {sleep_duration:.2f} seconds.")
            time.sleep(sleep_duration)
            api_calls_this_minute = 0
            minute_start_time = time.time()
        
        if api_calls_this_day >= av_cfg.get("max_calls_per_day", MAX_CALLS_PER_DAY_DEFAULT):
            log.error("Daily API call limit reached. Stopping fetch.")
            break

        # Fetch batch of news
        df_batch, rate_limit_hit = fetch_news_batch(api_key, tickers_of_interest, time_from=current_time_from, limit=av_cfg.get("limit_per_call", 200))
        api_calls_this_minute += 1
        api_calls_this_day += 1

        if rate_limit_hit:
            log.warning("Rate limit hit during fetch_news_batch. Waiting and will retry if appropriate, or stop.")
            # Implement more sophisticated retry or stop logic here based on your needs
            time.sleep(call_delay_seconds * 5) # Longer sleep after a rate limit hint
            continue # Retry the same time_from after a longer pause

        if df_batch is None or df_batch.empty:
            log.info("No more news items returned in this batch or error occurred. Ending pagination.")
            break 

        all_news_df = pd.concat([all_news_df, df_batch], ignore_index=True)
        all_news_df.drop_duplicates(subset=['url', 'title', 'date_published'], inplace=True) # Deduplicate
        all_news_df.sort_values(by='date_published', ascending=True, inplace=True)
        
        # Prepare for next iteration: set time_from to the timestamp of the last received article + 1 second
        last_article_time = df_batch['date_published'].max()
        current_time_from = (last_article_time + timedelta(seconds=1)).strftime('%Y%m%dT%H%M%S') # Note: API uses HHMM for time_from, but needs HHMMSS for exactness
        # The API docs state time_from format: YYYYMMDDTHHMM. Using HHMMSS might be more precise for pagination.
        # Let's stick to YYYYMMDDTHHMM for time_from based on typical API behavior for ranges.
        current_time_from = (last_article_time + timedelta(minutes=1)).strftime('%Y%m%dT%H%M') 

        logger.report_scalar(title="API Calls", series="Per Minute", value=api_calls_this_minute, iteration=i)
        logger.report_scalar(title="API Calls", series="Per Day", value=api_calls_this_day, iteration=i)
        logger.report_scalar(title="News Items", series="Fetched Total", value=len(all_news_df), iteration=i)

        log.info(f"Pausing for {call_delay_seconds} seconds before next API call...")
        time.sleep(call_delay_seconds)

    if not all_news_df.empty:
        log.info(f"Total news items fetched and deduplicated: {len(all_news_df)}")
        all_news_df = all_news_df.sort_values(by='date_published', ascending=False)
        
        # Save as Parquet artifact
        parquet_file = "alphavantage_news.parquet"
        all_news_df.to_parquet(parquet_file, index=False)
        task.upload_artifact(name="alphavantage_news_parquet", artifact_object=parquet_file)
        log.info(f"News data saved to {parquet_file} and uploaded as artifact.")

        # Create or update ClearML Dataset
        try:
            dataset = Dataset.create(
                dataset_project=dataset_project,
                dataset_name=dataset_name,
                parent_datasets=[latest_dataset.id] if 'latest_dataset' in locals() and latest_dataset else None
            )
            dataset.add_files(parquet_file)
            # Store metadata about the fetch
            dataset.set_metadata({
                "source": "Alpha Vantage", 
                "tickers_fetched": tickers_of_interest,
                "total_articles": len(all_news_df),
                "fetch_range_start": all_news_df['date_published'].min().isoformat() if not all_news_df.empty else None,
                "fetch_range_end": all_news_df['date_published'].max().isoformat() if not all_news_df.empty else None,
                "last_api_call_params": {"time_from": current_time_from} # Last attempted time_from
            })
            dataset.upload(output_url=av_cfg.get("clearml_dataset_storage_url")) # Optional: custom storage URL
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