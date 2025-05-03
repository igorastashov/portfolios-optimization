'''
Standalone script to update market data from Binance.

This script performs the following actions:
1. Loads Binance API keys from environment variables or a .env file.
2. Calls `update_all_asset_data` to refresh individual asset CSV files.
3. Calls `create_combined_data` to generate the aggregated market data file.

Setup:
- Ensure the `python-dotenv` package is installed (`pip install python-dotenv`).
- Create a `.env` file in the project root directory (the parent directory of `notebooks`).
- Add your Binance API keys to the `.env` file:
  ```
  BINANCE_API_KEY=your_actual_api_key
  BINANCE_API_SECRET=your_actual_api_secret
  ```
- Alternatively, set the keys as environment variables in your system.

Usage:
Run this script from the project root directory or the `notebooks` directory:
  `python notebooks/update_market_data.py`
'''

import os
import sys

# --- Attempt to load environment variables from .env file ---
try:
    from dotenv import load_dotenv
    # Assume the script is run from the 'notebooks' dir or the root.
    # Go up one level from this script's directory to find the root.
    project_root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(project_root_env, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from: {dotenv_path}")
    else:
        # Check if .env exists in current dir (if run from root)
        dotenv_path_alt = os.path.join(os.getcwd(), '.env')
        if os.path.exists(dotenv_path_alt):
            load_dotenv(dotenv_path=dotenv_path_alt)
            print(f"Loaded environment variables from: {dotenv_path_alt}")
        else:
             print(f"Info: .env file not found at {dotenv_path} or {dotenv_path_alt}. Relying on system environment variables.")
except ImportError:
    print("Warning: python-dotenv not installed. Relying solely on system environment variables.")
    print("Install with: pip install python-dotenv")
# -----------------------------------------------------------

# --- Determine project root and add to Python path ---------
# Assuming the script is in 'notebooks' directory, one level down from root
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
# -----------------------------------------------------------

# --- Import necessary functions from the application ------
try:
    from portfolios_optimization.data_loader import update_all_asset_data, create_combined_data
except ImportError as e:
    print(f"ERROR: Could not import required functions: {e}")
    print("Please ensure:")
    print("  1. You are running this script from the project root directory OR the 'notebooks' directory.")
    print("  2. The 'portfolios_optimization' package is correctly located relative to the script or project root.")
    print(f"   (Project root determined as: {project_root_path})")
    sys.exit(1)
# -----------------------------------------------------------

def main():
    """
    Main function to execute the data update process.
    """
    # --- Check for API Keys ---
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("\n" + "="*60)
        print(" ERROR: Binance API keys (BINANCE_API_KEY, BINANCE_API_SECRET) ")
        print("        not found in environment variables or .env file.      ")
        print(" Please refer to the script's header documentation for setup. ")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("Binance API keys found. Proceeding with update.")
        # Optional: Mask keys partially if printing
        # print(f"  API Key: {api_key[:5]}...{api_key[-4:]}")

    # --- Step 1: Update individual asset CSV files ---
    print("\n--- [Step 1/2] Starting Market Data Update ---")
    print("Calling `update_all_asset_data`...")
    update_successful = False
    message = "Update process did not complete successfully."
    try:
        # We assume `update_all_asset_data` reads keys from the environment
        # and handles its own progress indication/logging.
        # The function in the Streamlit app doesn't explicitly return status,
        # so we rely on the absence of exceptions as a sign of basic success.
        update_all_asset_data()
        print("`update_all_asset_data` finished without critical errors.")
        update_successful = True
        message = "Asset data update process completed (check console output from the function for details)." # Adjusted message

    except Exception as e:
        print(f"ERROR during `update_all_asset_data`: {e}")
        import traceback
        traceback.print_exc()
        update_successful = False
        message = f"Asset data update failed: {e}"

    print(f"Update Status: {message}")

    # --- Step 2: Create combined data file (only if update seemed okay) ---
    if update_successful:
        print("\n--- [Step 2/2] Creating Combined Data File ---")
        print("Calling `create_combined_data`...")
        try:
            success_combine, message_combine = create_combined_data()
            if success_combine:
                print(f"SUCCESS: {message_combine}")
            else:
                # The function itself might print errors, but we add one here too.
                print(f"ERROR: `create_combined_data` reported an issue: {message_combine}")
        except Exception as e:
            print(f"ERROR during `create_combined_data`: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- [Step 2/2] Skipping combined data file creation due to previous errors. ---")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main() 

'''
poetry run python notebooks/update_market_data.py
'''