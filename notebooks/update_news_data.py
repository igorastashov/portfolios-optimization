import os
import re
import time
import json
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

# --- Configuration ---
BASE_DATA_DIR = 'news_data'
API_KEY = 'RA2WC9ADPHNM289Z'  # Replace with your actual key if needed
STATE_FILE = os.path.join(BASE_DATA_DIR, 'update_status.json')
# Alpha Vantage free tier limit is often around 5 calls/minute
API_SLEEP_TIME = 15  # Seconds between API calls
FILENAME_PATTERN = re.compile(r"news_data_(\d{8}T\d{4})_to_(\d{8}T\d{4})\.csv")
DATE_FORMAT_API = "%Y%m%dT%H%M"
DATE_FORMAT_FILENAME = "%Y%m%dT%H%M"
DATE_FORMAT_STATE = "%Y-%m-%dT%H:%M:%S" # ISO 8601 subset

# --- State Management ---

def load_update_state(filepath):
    """Loads the update state from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                state_str_keys = json.load(f)
            # Convert ISO strings back to datetime objects
            state = {
                ticker: datetime.strptime(dt_str, DATE_FORMAT_STATE).replace(tzinfo=timezone.utc)
                for ticker, dt_str in state_str_keys.items()
            }
            print(f"Loaded update state from {filepath}")
            return state
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Error loading state file {filepath}: {e}. Starting fresh.")
            return {}
    else:
        print("No state file found. Starting fresh.")
        return {}

def save_update_state(filepath, state):
    """Saves the update state to a JSON file."""
    # Convert datetime objects to ISO strings for JSON serialization
    state_str_keys = {
        ticker: dt.strftime(DATE_FORMAT_STATE)
        for ticker, dt in state.items()
    }
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state_str_keys, f, indent=4)
        print(f"Saved update state to {filepath}")
    except IOError as e:
        print(f"Error saving state file {filepath}: {e}")


# --- File and Date Utilities ---

def get_asset_folders(base_dir):
    """Gets a list of asset tickers (subdirectories) in the base data directory."""
    try:
        return sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    except FileNotFoundError:
        print(f"Error: Base data directory not found: {base_dir}")
        return []

def get_last_update_date_from_files(asset_dir):
    """Finds the latest end timestamp from news CSV files in the asset directory."""
    latest_date = None
    try:
        for filename in os.listdir(asset_dir):
            match = FILENAME_PATTERN.match(filename)
            if match:
                end_date_str = match.group(2)
                try:
                    current_date = datetime.strptime(end_date_str, DATE_FORMAT_FILENAME)
                    # Make timezone-aware (assume UTC as API likely uses it)
                    current_date = current_date.replace(tzinfo=timezone.utc)
                    if latest_date is None or current_date > latest_date:
                        latest_date = current_date
                except ValueError:
                    print(f"Warning: Could not parse date from filename: {filename}")
    except FileNotFoundError:
        print(f"Warning: Asset directory not found: {asset_dir}")
    return latest_date

# --- Data Fetching ---

def fetch_news_since(ticker, start_datetime, api_key):
    """Fetches news data iteratively starting from start_datetime."""
    all_news_data = []
    current_start_dt = start_datetime
    print(f"[{ticker}] Fetching news since {current_start_dt.strftime(DATE_FORMAT_API)}...")

    while True:
        start_date_str = current_start_dt.strftime(DATE_FORMAT_API)
        # Include end time to potentially limit scope per call, though API behavior might vary
        # Let's fetch up to now() - though API might return less if limit is hit
        end_date_str = datetime.now(timezone.utc).strftime(DATE_FORMAT_API)

        url = (f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
               f'&tickers=CRYPTO:{ticker.upper()}'
               f'&apikey={api_key}'
               f'&time_from={start_date_str}'
            #    f'&time_to={end_date_str}' # Optional: Add end time if API supports well
               f'&limit=1000' # Max limit per request
               f'&sort=EARLIEST')

        print(f"[{ticker}] Calling API: ...&time_from={start_date_str}...")

        try:
            r = requests.get(url)
            r.raise_for_status() # Raise HTTPError for bad responses (4XX, 5XX)
            data = r.json()

            # Check for API limit messages or errors
            if "Information" in data and "rate limit" in data["Information"].lower():
                 print(f"[{ticker}] Error: API rate limit hit. Message: {data['Information']}")
                 return (pd.concat(all_news_data) if all_news_data else pd.DataFrame(), "RATE_LIMIT_ERROR")
            if "Error Message" in data:
                 print(f"[{ticker}] Error: API error message. Message: {data['Error Message']}")
                 return (pd.concat(all_news_data) if all_news_data else pd.DataFrame(), "API_ERROR")

            if 'feed' not in data:
                print(f'[{ticker}] Warning: No "feed" key in the response for start={start_date_str}. Assuming no more data.')
                break # Exit loop if feed key is missing

            feed_data = data['feed']

            if not feed_data:
                print(f'[{ticker}] Feed is empty for start={start_date_str}. Assuming no more data.')
                break # Exit loop if feed is empty

            # Process batch
            batch_df = pd.DataFrame(feed_data)
            # Convert time_published and handle potential parsing errors
            try:
                 batch_df['date_published'] = pd.to_datetime(batch_df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')
                 batch_df.dropna(subset=['date_published'], inplace=True) # Drop rows where conversion failed
                 batch_df['date_published'] = batch_df['date_published'].dt.tz_localize(timezone.utc) # Assume UTC
            except Exception as e:
                 print(f"[{ticker}] Error converting time_published: {e}. Skipping batch.")
                 continue # Skip this batch if major parsing error

            if batch_df.empty:
                 print(f'[{ticker}] No valid date_published entries found in batch for start={start_date_str}.')
                 continue # Or break, depending on desired behavior


            batch_df.drop('time_published', axis=1, inplace=True, errors='ignore')
            batch_df.sort_values(by='date_published', inplace=True)

            # --- Crucial Check: Avoid Infinite Loops ---
            # If the first item in the new batch is not later than the start time
            # requested, it means we're getting overlapping or old data. Stop.
            first_item_time = batch_df['date_published'].iloc[0]
            if first_item_time < current_start_dt:
                 print(f"[{ticker}] Warning: Received data older ({first_item_time}) than requested start ({current_start_dt}). Stopping fetch for this asset to prevent loops.")
                 # Keep only data strictly after the requested start time
                 batch_df = batch_df[batch_df['date_published'] >= current_start_dt]
                 if not batch_df.empty:
                     all_news_data.append(batch_df)
                 break # Stop fetching for this asset

            all_news_data.append(batch_df)

            # Prepare for next iteration
            last_time_in_batch = batch_df['date_published'].iloc[-1]
            current_start_dt = last_time_in_batch + timedelta(seconds=1) # Start next fetch 1s after the last received item

            # Check if data is recent enough
            if last_time_in_batch >= datetime.now(timezone.utc) - timedelta(minutes=5):
                print(f"[{ticker}] Fetched data up to {last_time_in_batch}. Considered up-to-date.")
                break

            print(f"[{ticker}] Batch fetched. Last entry: {last_time_in_batch}. Next fetch starts after this time. Sleeping {API_SLEEP_TIME}s...")
            time.sleep(API_SLEEP_TIME) # Respect API limits

        except requests.exceptions.RequestException as e:
            print(f"[{ticker}] Error: Network or HTTP error during API call: {e}")
            # Decide if this is fatal (stop all) or per-asset (stop current)
            # For now, stop current asset fetch, maybe retry later
            return (pd.concat(all_news_data) if all_news_data else pd.DataFrame(), "NETWORK_ERROR")
        except Exception as e:
            print(f"[{ticker}] Error: Unexpected error during fetch: {e}")
            return (pd.concat(all_news_data) if all_news_data else pd.DataFrame(), "UNEXPECTED_ERROR")


    if not all_news_data:
        print(f"[{ticker}] No new data found.")
        return pd.DataFrame(), "NO_NEW_DATA"
    else:
        result_df = pd.concat(all_news_data).drop_duplicates(subset=['url']).sort_values(by='date_published').reset_index(drop=True)
        print(f"[{ticker}] Successfully fetched {len(result_df)} new entries.")
        return result_df, "SUCCESS"


# --- Main Execution ---

def main():
    print("--- Starting News Data Update Script ---")
    script_start_time = datetime.now(timezone.utc)
    state = load_update_state(STATE_FILE)
    assets = get_asset_folders(BASE_DATA_DIR)

    if not assets:
        print("No asset folders found. Exiting.")
        return

    rate_limit_hit = False
    updated_state = state.copy() # Work on a copy

    # Simple prioritization: Check assets not in state first, then oldest state entry
    assets_to_process = sorted(
        assets,
        key=lambda x: updated_state.get(x, datetime.min.replace(tzinfo=timezone.utc)) # Assets not in state get earliest date
    )

    print(f"Found assets: {assets}")
    print(f"Processing order based on state: {assets_to_process}")

    for ticker in assets_to_process:
        if rate_limit_hit:
            print(f"Rate limit was hit. Skipping remaining assets: {ticker} and onwards for this run.")
            break

        print(f"\n--- Processing Asset: {ticker} ---")
        asset_dir = os.path.join(BASE_DATA_DIR, ticker)
        os.makedirs(asset_dir, exist_ok=True) # Ensure directory exists

        # Determine start time for this asset
        start_datetime_for_update = None
        if ticker in updated_state:
            start_datetime_for_update = updated_state[ticker]
            print(f"[{ticker}] Found start time in state: {start_datetime_for_update}")
        else:
            last_file_date = get_last_update_date_from_files(asset_dir)
            if last_file_date:
                start_datetime_for_update = last_file_date + timedelta(seconds=1)
                print(f"[{ticker}] Last file date: {last_file_date}. Setting start time from file: {start_datetime_for_update}")
            else:
                # Default: Fetch last 7 days if no files and no state? Or skip?
                # Let's skip for now, assuming initial files must exist or state must be seeded.
                print(f"[{ticker}] Warning: No state and no data files found. Cannot determine start time. Skipping.")
                continue # Skip this asset

        # Check if the start time is in the future (e.g., clock issues, bad state)
        if start_datetime_for_update > script_start_time:
             print(f"[{ticker}] Warning: Start time {start_datetime_for_update} is in the future. Resetting to now to avoid issues.")
             start_datetime_for_update = script_start_time


        # Fetch data
        fetched_df, status_code = fetch_news_since(ticker, start_datetime_for_update, API_KEY)

        # Process result
        if status_code == "RATE_LIMIT_ERROR":
            rate_limit_hit = True
            # State for this ticker is NOT updated, it will retry from same start time next run
            print(f"[{ticker}] Halting further processing in this run due to rate limit.")
            # Save partially fetched data if any? For now, discard if rate limit hit during fetch.
            if not fetched_df.empty:
                print(f"[{ticker}] Discarding {len(fetched_df)} partially fetched entries due to rate limit.")

        elif status_code in ["API_ERROR", "NETWORK_ERROR", "UNEXPECTED_ERROR"]:
            # State for this ticker is NOT updated, it will retry from same start time next run
            print(f"[{ticker}] Skipping save due to error: {status_code}")
            # Save partially fetched data? For now, discard.
            if not fetched_df.empty:
                 print(f"[{ticker}] Discarding {len(fetched_df)} partially fetched entries due to error.")

        elif status_code == "NO_NEW_DATA":
            print(f"[{ticker}] No new data found or fetched.")
            # Update state to now, indicating it was checked and is up-to-date
            updated_state[ticker] = script_start_time # Mark as checked up to script start time

        elif status_code == "SUCCESS":
            if not fetched_df.empty:
                min_date = fetched_df['date_published'].min()
                max_date = fetched_df['date_published'].max()

                # Ensure we don't save data older than requested start time (safety check)
                fetched_df = fetched_df[fetched_df['date_published'] >= start_datetime_for_update]

                if fetched_df.empty:
                    print(f"[{ticker}] Success reported, but all fetched data was before requested start time. No data saved.")
                    updated_state[ticker] = script_start_time # Mark as checked
                else:
                    min_date = fetched_df['date_published'].min() # Recalculate after filtering
                    max_date = fetched_df['date_published'].max()

                    start_str = min_date.strftime(DATE_FORMAT_FILENAME)
                    end_str = max_date.strftime(DATE_FORMAT_FILENAME)
                    filename = f"news_data_{start_str}_to_{end_str}.csv"
                    save_path = os.path.join(asset_dir, filename)

                    try:
                        fetched_df.to_csv(save_path, index=False)
                        print(f"[{ticker}] Successfully saved {len(fetched_df)} new entries to {filename}")
                        # Update state to start *after* the last saved item
                        updated_state[ticker] = max_date + timedelta(seconds=1)
                    except IOError as e:
                        print(f"[{ticker}] Error saving file {save_path}: {e}")
                        # Don't update state if save failed
            else:
                 print(f"[{ticker}] Success reported, but DataFrame is empty. No data saved.")
                 updated_state[ticker] = script_start_time # Mark as checked

    # Save the final state after processing all possible assets
    save_update_state(STATE_FILE, updated_state)
    print("\n--- News Data Update Script Finished ---")

if __name__ == "__main__":
    # Change directory to the script's location to ensure relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")
    main() 