import pandas as pd
import os
import glob

def process_asset_news(asset_name, base_dir="news_data"):
    """
    Combines news CSV files for a given asset, handles incremental updates
    by reading existing combined data, removes duplicates, and saves the result.

    Args:
        asset_name (str): The name of the asset (e.g., 'btc', 'eth').
        base_dir (str): The base directory containing asset subdirectories.
    """
    asset_dir = os.path.join(base_dir, asset_name)
    output_file = os.path.join(asset_dir, f"combined_{asset_name}_news.csv")

    if not os.path.isdir(asset_dir):
        print(f"Directory not found for asset '{asset_name}': {asset_dir}")
        return

    # --- Find source CSV files ---
    csv_pattern = os.path.join(asset_dir, "news_data_*.csv")
    source_csv_files = glob.glob(csv_pattern)

    # --- Load existing combined data if it exists ---
    existing_df = None
    if os.path.exists(output_file):
        try:
            print(f"Found existing combined file for '{asset_name}': {output_file}. Reading...")
            existing_df = pd.read_csv(output_file, low_memory=False)
            print(f"  Read {len(existing_df)} rows from existing file.")
        except Exception as e:
            print(f"Error reading existing combined file {output_file}: {e}. Will proceed using only source files.")
            existing_df = None # Ensure it's None if read fails
    else:
        print(f"No existing combined file found for '{asset_name}'. Creating new one.")


    if not source_csv_files:
        if existing_df is not None:
             print(f"No new source CSV files found for asset '{asset_name}'. Existing combined file '{output_file}' remains unchanged.")
        else:
             print(f"No source CSV files found and no existing combined file for asset '{asset_name}' in {asset_dir}.")
        return # Nothing to do if no source files and no existing file

    print(f"Processing {len(source_csv_files)} source CSV files for asset '{asset_name}'...")

    # --- Read source CSV files ---
    source_dataframes = []
    for f in source_csv_files:
        try:
            # Specify low_memory=False if getting DtypeWarning, adjust as needed
            df = pd.read_csv(f, low_memory=False)
            source_dataframes.append(df)
            print(f"  Read {f} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading source file {f}: {e}")
            # Decide how to handle errors: skip file, stop process, etc.
            # continue # Skip this file on error

    if not source_dataframes and existing_df is None:
        print(f"No source dataframes were successfully read and no existing data for asset '{asset_name}'.")
        return

    # --- Combine existing data (if any) with new source data ---
    all_dataframes = []
    if existing_df is not None:
        all_dataframes.append(existing_df)

    all_dataframes.extend(source_dataframes) # Add the newly read source dataframes

    if not all_dataframes:
         print(f"No data to process for asset '{asset_name}'.")
         return

    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total rows before deduplication for '{asset_name}': {len(combined_df)}")

    # --- Remove duplicate rows ---
    initial_rows = len(combined_df)
    # Keep='first' is the default, ensuring older entries are preferred if duplicates exist
    deduplicated_df = combined_df.drop_duplicates(keep='first')
    final_rows = len(deduplicated_df)
    duplicates_removed = initial_rows - final_rows
    print(f"Removed {duplicates_removed} duplicate rows for asset '{asset_name}'. Final rows: {final_rows}")

    # --- Save the combined and deduplicated dataframe ---
    try:
        deduplicated_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully saved combined and deduplicated data for '{asset_name}' to {output_file}")
    except Exception as e:
        print(f"Error saving combined file for asset '{asset_name}': {e}")

if __name__ == "__main__":
    # Ensure the script looks for news_data relative to its own location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    news_data_base_dir = os.path.join(script_dir, "news_data")
    # news_data_base_dir = "news_data"  # Use this if running script from notebooks/ manually

    assets_to_process = ["btc", "cake", "eth", "hbar", "ltc", "sol",  "usdt", "xrp"] # Add other assets as needed

    print(f"Base directory for news data: {news_data_base_dir}")

    for asset in assets_to_process:
        process_asset_news(asset, news_data_base_dir)
        print("-" * 30)

    print("Processing complete.") 