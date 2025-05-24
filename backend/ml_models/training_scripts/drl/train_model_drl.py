import argparse
import pandas as pd
from clearml import Task, Artifact, OutputModel
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from finrl.meta.env_portfolio import StockPortfolioEnv
except ImportError:
    log = logging.getLogger(__name__)
    log.error("FinRL StockPortfolioEnv not found. Make sure FinRL is correctly installed.")
    StockPortfolioEnv = None

log = logging.getLogger(__name__)

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
        "action_space": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "reward_scaling": env_cfg.reward_scaling,
        "print_verbosity": env_cfg.print_verbosity,
    }
    try:
        env_train_instance = StockPortfolioEnv(**env_kwargs)
        env_train = DummyVecEnv([lambda: env_train_instance])
        log.info(f"DRL environment '{env_cfg.name}' created and wrapped in DummyVecEnv.")
    except Exception as e_env:
        log.error(f"Error creating DRL environment: {e_env}", exc_info=True)
        task.close(); raise

    log.info("Training DRL agent...")
    agent_params = OmegaConf.to_container(agent_cfg.params, resolve=True)
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
        
        model.learn(
            total_timesteps=tm_cfg.total_timesteps,
            log_interval=tm_cfg.log_interval,
        )
        log.info(f"DRL agent training completed after {tm_cfg.total_timesteps} timesteps.")

    except Exception as e_train:
        log.error(f"Error during DRL agent training: {e_train}", exc_info=True)
        task.close(); raise

    log.info("Saving trained DRL model...")
    clearml_model_name = f"{agent_cfg.model_registry_name_prefix}_{agent_cfg.name.lower()}_{portfolio_id}"
    model_local_path_zip = f"{clearml_model_name}.zip"
    
    try:
        model.save(model_local_path_zip)
        log.info(f"DRL model saved locally: {model_local_path_zip}")
    except Exception as e_save:
        log.error(f"Error saving DRL model locally: {e_save}", exc_info=True)
        task.close(); raise

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

    task.close()
    log.info(f"DRL training task for portfolio {portfolio_id} completed.")

if __name__ == '__main__':
    main() 