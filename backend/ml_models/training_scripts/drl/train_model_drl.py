# Placeholder for DRL Model Training Script
import argparse
import pandas as pd
from clearml import Task, Artifact, OutputModel
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import numpy as np # Added for state_space calculation if needed

# DRL specific imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# FinRL environment for portfolio management. Ensure FinRL is installed.
# It might be necessary to adjust imports based on the exact FinRL version and structure.
try:
    from finrl.meta.env_portfolio import StockPortfolioEnv
except ImportError:
    log = logging.getLogger(__name__) # temp logger for import error
    log.error("FinRL StockPortfolioEnv not found. Make sure FinRL is correctly installed.")
    # You might need to fall back to a simpler Gym/Gymnasium env or ensure FinRL is in PYTHONPATH
    StockPortfolioEnv = None # Placeholder if import fails

# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader # Not used here
# from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split # Not used here

log = logging.getLogger(__name__)

# Function to load data from ClearML task artifact (similar to catboost scripts)
def load_drl_data_from_task(task_id: str, artifact_name: str) -> pd.DataFrame:
    """Загружает DataFrame из Parquet артефакта DRL данных задачи ClearML."""
    try:
        artifact = Artifact.get(task_id=task_id, name=artifact_name)
        if not artifact:
            log.error(f"DRL data artifact '{artifact_name}' from task {task_id} not found.")
            return pd.DataFrame()
            
        artifact_path = artifact.get_local_copy()
        if not artifact_path or not os.path.exists(artifact_path) or os.path.getsize(artifact_path) == 0:
            log.error(f"Local copy of DRL artifact '{artifact_name}' from task {task_id} (path: {artifact_path}) is invalid.")
            return pd.DataFrame()
        log.info(f"Loading DRL data from artifact: {artifact_path}")
        df = pd.read_parquet(artifact_path)
        # Ensure the 'date' column is in datetime format if it exists, as FinRL envs often expect it
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log.error(f"Error loading DRL artifact '{artifact_name}' from task {task_id}: {e}", exc_info=True)
        return pd.DataFrame()

@hydra.main(config_path="../../configs", config_name="drl_config", version_base=None)
def main(cfg: DictConfig) -> None:
    if StockPortfolioEnv is None:
        log.error("StockPortfolioEnv from FinRL is not available. Ensure FinRL is correctly installed. Training cannot proceed.")
        raise ImportError("StockPortfolioEnv (FinRL) is required but was not found.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio_id', type=str, required=True, help="Portfolio ID for DRL model training")
    parser.add_argument('--data_prep_task_id', type=str, required=True, help="ClearML task ID of DRL data preparation step")
    args, unknown = parser.parse_known_args()

    portfolio_id = args.portfolio_id
    data_prep_task_id = args.data_prep_task_id
    task_name_parametrized = f"{cfg.train_model.task_name_prefix}_{portfolio_id}"

    task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [portfolio_id, "training", "drl"],
        output_uri=True
    )
    effective_config_dict = OmegaConf.to_container(cfg, resolve=True)
    effective_config_dict['train_model']['portfolio_id'] = portfolio_id 
    effective_config_dict['train_model']['data_prep_task_id'] = data_prep_task_id
    task.connect(effective_config_dict, name='Effective_Hydra_Plus_Args_Configuration')

    log.info(f"Starting DRL model training for portfolio: {portfolio_id}")
    log.info(f"DRL data preparation task ID: {data_prep_task_id}")
    tm_cfg = cfg.train_model 
    env_cfg = cfg.env 
    agent_cfg = cfg.agent

    log.info("Loading prepared DRL training data...")
    train_artifact_name = "drl_train_data" 
    train_df = load_drl_data_from_task(data_prep_task_id, train_artifact_name)

    if train_df.empty:
        log.error("Failed to load DRL training data. Aborting training.")
        task.close(); raise ValueError("Failed to load DRL train data.")

    log.info(f"DRL training data loaded. Shape: {train_df.shape}.")
    # StockPortfolioEnv expects specific column names like 'date', 'tic', 'close', and technical indicators.
    # Ensure data_preparation_drl.py produces these correctly.
    # Required columns for StockPortfolioEnv typically: date, tic, close, open, high, low, volume + tech_indicator_list
    # We assume 'close' is the primary price column for StockPortfolioEnv state representation
    # and other OHLCV are present as produced by data_preparation_drl.py

    log.info("Setting up DRL environment...")
    if 'tic' not in train_df.columns:
        log.error("'tic' column missing in training data. Cannot determine stock_dimension.")
        task.close(); raise ValueError("Missing 'tic' column for DRL environment.")
    stock_dimension = len(train_df.tic.unique())
    
    tech_indicator_list = list(env_cfg.tech_indicator_list)
    log.info(f"Using technical indicators for environment: {tech_indicator_list}")
    for ti in tech_indicator_list:
        if ti not in train_df.columns:
            log.error(f"Technical indicator '{ti}' (from config) not found in training data columns.")
            task.close(); raise ValueError(f"Missing technical indicator '{ti}' in training data.")

    # State space for StockPortfolioEnv: 1 (cash) + stock_dim (shares held) + stock_dim (close price of each stock) + num_tech_indicators * stock_dim
    # This is a common formulation for FinRL's StockPortfolioEnv. Adjust if your env is different.
    state_space = 1 + stock_dimension + stock_dimension + len(tech_indicator_list) * stock_dimension
    log.info(f"Stock Dimension: {stock_dimension}, Calculated State Space: {state_space}")

    env_kwargs = {
        "df": train_df,
        "stock_dim": stock_dimension,
        "hmax": env_cfg.hmax,
        "initial_amount": env_cfg.initial_amount,
        "buy_cost_pct": env_cfg.buy_cost_pct,
        "sell_cost_pct": env_cfg.sell_cost_pct,
        "state_space": state_space,
        "action_space": stock_dimension, # Action space is typically number of stocks for portfolio weights/amounts
        "tech_indicator_list": tech_indicator_list,
        "reward_scaling": env_cfg.reward_scaling,
        "print_verbosity": env_cfg.print_verbosity,
        # FinRL envs often look for a 'date' column for time steps, ensure it's present and correct.
        # It also expects specific column names for OHLCV, often lowercase.
    }
    try:
        env_train_instance = StockPortfolioEnv(**env_kwargs)
        env_train = DummyVecEnv([lambda: env_train_instance]) # Wrap in DummyVecEnv for SB3
        log.info(f"DRL environment '{env_cfg.name}' created and wrapped in DummyVecEnv.")
    except Exception as e_env:
        log.error(f"Error creating DRL environment: {e_env}", exc_info=True)
        task.close(); raise

    log.info("Training DRL agent...")
    agent_params = OmegaConf.to_container(agent_cfg.params, resolve=True)
    # Add seed from agent_params if it exists to the task for reproducibility
    if 'seed' in agent_params:
        task.connect_configuration({"agent_seed": agent_params['seed']}, name="Agent_Seed")

    try:
        if agent_cfg.name.upper() == "PPO":
            model = PPO(
                policy=tm_cfg.policy_name, 
                env=env_train, 
                **agent_params
            )
        else:
            log.error(f"DRL Agent '{agent_cfg.name}' is not supported. Only PPO is currently implemented.")
            task.close(); raise NotImplementedError(f"DRL Agent {agent_cfg.name} not implemented.")
        
        log.info(f"Agent {agent_cfg.name} initialized with parameters: {agent_params}")
        
        # TODO: Consider adding a ClearML callback for SB3 for live metric logging if available/needed
        # from stable_baselines3.common.callbacks import BaseCallback
        # class ClearMLReportCallback(BaseCallback): ...
        # callback_clearml = ClearMLReportCallback() if cfg.training.get("clearml_callback", False) else None
        
        model.learn(
            total_timesteps=tm_cfg.total_timesteps,
            log_interval=tm_cfg.log_interval,
            # callback=callback_clearml 
        )
        log.info(f"DRL agent training completed after {tm_cfg.total_timesteps} timesteps.")

    except Exception as e_train:
        log.error(f"Error during DRL agent training: {e_train}", exc_info=True)
        task.close(); raise

    log.info("Saving trained DRL model...")
    clearml_model_name = f"{agent_cfg.model_registry_name_prefix}_{agent_cfg.name.lower()}_{portfolio_id}"
    model_local_path_zip = f"{clearml_model_name}.zip" # SB3 models often save as .zip
    
    try:
        model.save(model_local_path_zip)
        log.info(f"DRL model saved locally: {model_local_path_zip}")
    except Exception as e_save:
        log.error(f"Error saving DRL model locally: {e_save}", exc_info=True)
        task.close(); raise

    # Загрузка модели в ClearML
    output_model_metadata = {
        "agent_name": agent_cfg.name,
        "stock_dimension": stock_dimension,
        "state_space": state_space,
        "tech_indicator_list": tech_indicator_list,
        "env_config": OmegaConf.to_container(env_cfg, resolve=True),
        "agent_hyperparameters": agent_params,
        "training_total_timesteps": tm_cfg.total_timesteps,
        "data_preparation_task_id": data_prep_task_id,
        "portfolio_id": portfolio_id
    }

    output_model_clearml = OutputModel(
        task=task,
        framework=str(agent_cfg.library), 
        name=clearml_model_name,
    )
    for meta_key, meta_val in output_model_metadata.items():
        output_model_clearml.set_metadata(meta_key, meta_val)

    try:
        output_model_clearml.update_weights(weights_filename=model_local_path_zip)
        output_model_clearml.set_tags(list(cfg.global_tags) + [portfolio_id, "drl_agent", agent_cfg.name.lower()])
        log.info(f"DRL model '{clearml_model_name}' and weights uploaded as OutputModel to ClearML.")
    except Exception as e_upload:
        log.error(f"Error uploading DRL model/weights to ClearML: {e_upload}", exc_info=True)
        # Не прерываем задачу, если только загрузка не удалась, модель все еще сохранена локально

    task.close() # Close the ClearML task
    log.info(f"DRL training task for portfolio {portfolio_id} completed.")

if __name__ == '__main__':
    # Example call:
    # python backend/ml_models/training_scripts/drl/train_model_drl.py --portfolio_id=CRYPTO_PORTFOLIO --data_prep_task_id=SOME_CLEARML_TASK_ID
    main() 