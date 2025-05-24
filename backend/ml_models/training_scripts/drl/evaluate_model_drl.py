import argparse
import pandas as pd
import numpy as np
from clearml import Task, Artifact, Model, Logger as ClearMLLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import json
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from finrl.meta.env_portfolio import StockPortfolioEnv
    from finrl.plot import backtest_stats, get_daily_return
except ImportError:
    log = logging.getLogger(__name__)
    log.error("FinRL components (StockPortfolioEnv, plot utils) not found. Make sure FinRL is correctly installed.")
    StockPortfolioEnv = None
    backtest_stats = None
    get_daily_return = None

log = logging.getLogger(__name__)

def load_drl_data_from_task(task_id: str, artifact_name: str) -> pd.DataFrame:
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

def load_drl_model_from_clearml_task(train_task_id: str, model_clearml_name: str, project_name: str, agent_name_from_cfg: str = "PPO"):
    """Загружает обученную DRL модель (OutputModel) из задачи обучения ClearML."""
    try:
        log.info(f"Loading DRL OutputModel '{model_clearml_name}' from training task {train_task_id} in project '{project_name}'.")
        models = Model.query_models(task_ids=[train_task_id], model_name=model_clearml_name, project_name=project_name, max_results=1)
        
        if not models:
            log.error(f"DRL OutputModel '{model_clearml_name}' not found for task {train_task_id} in project '{project_name}'.")
            return None

        model_object = models[0]
        model_local_path = model_object.get_local_copy()
        log.info(f"DRL Model '{model_object.name}' (ID: {model_object.id}) downloaded from: {model_local_path}")
        
        if agent_name_from_cfg.upper() == "PPO":
            loaded_model = PPO.load(model_local_path)
            log.info(f"PPO model loaded successfully from {model_local_path}.")
            return loaded_model
        elif agent_name_from_cfg.upper() == "A2C":
            from stable_baselines3 import A2C
            loaded_model = A2C.load(model_local_path)
            return loaded_model
        else: 
            log.error(f"Unsupported DRL agent type from config: {agent_name_from_cfg}. Cannot load model.")
            return None
        
    except Exception as e:
        log.error(f"Error loading DRL model '{model_clearml_name}' from task {train_task_id}: {e}", exc_info=True)
        return None

@hydra.main(config_path="../../configs", config_name="drl_config", version_base=None)
def main(cfg: DictConfig) -> None:
    if StockPortfolioEnv is None or backtest_stats is None or get_daily_return is None:
        log.error("Core FinRL components (StockPortfolioEnv, plot utils) are not available. Evaluation cannot proceed.")
        raise ImportError("Required FinRL components are not found.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio_id', type=str, required=True, help="Portfolio ID for DRL model evaluation")
    parser.add_argument('--data_prep_task_id', type=str, required=True, help="ClearML task ID of DRL data preparation (for trade data)")
    parser.add_argument('--train_task_id', type=str, required=True, help="ClearML task ID of DRL model training")
    args, unknown = parser.parse_known_args()

    portfolio_id = args.portfolio_id
    data_prep_task_id = args.data_prep_task_id
    train_task_id = args.train_task_id

    task_name_parametrized = f"{cfg.evaluation.task_name_prefix}_{portfolio_id}"
    current_task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [portfolio_id, "evaluation", "drl"],
        output_uri=True
    )
    effective_config_dict = OmegaConf.to_container(cfg, resolve=True)
    effective_config_dict['evaluation']['portfolio_id'] = portfolio_id
    effective_config_dict['evaluation']['data_prep_task_id'] = data_prep_task_id
    effective_config_dict['evaluation']['train_task_id'] = train_task_id
    current_task.connect(effective_config_dict, name='Effective_Hydra_Plus_Args_Configuration')

    log.info(f"Starting DRL model evaluation for portfolio: {portfolio_id}...")
    eval_cfg = cfg.evaluation
    env_cfg = cfg.env
    agent_cfg = cfg.agent

    log.info("Loading DRL trade data...")
    trade_artifact_name = "drl_trade_data" 
    trade_df = load_drl_data_from_task(data_prep_task_id, trade_artifact_name)

    if trade_df.empty:
        log.error("Failed to load DRL trade data. Aborting evaluation.")
        current_task.close(); raise ValueError("Failed to load DRL trade data.")
    log.info(f"DRL trade data loaded. Shape: {trade_df.shape}.")

    log.info("Loading trained DRL model...")
    model_clearml_name = f"{agent_cfg.model_registry_name_prefix}_{agent_cfg.name.lower()}_{portfolio_id}"
    trained_drl_agent = load_drl_model_from_clearml_task(
        train_task_id, 
        model_clearml_name, 
        cfg.project_name, 
        agent_cfg.name
    )

    if trained_drl_agent is None:
        log.error("Failed to load trained DRL model. Aborting evaluation.")
        current_task.close(); raise ValueError("Failed to load trained DRL model.")

    log.info("Evaluating DRL model on trade data (backtesting)...")
    stock_dimension_eval = len(trade_df.tic.unique())
    tech_indicator_list_eval = list(env_cfg.tech_indicator_list)
    state_space_eval = 1 + stock_dimension_eval * (2 + len(tech_indicator_list_eval))

    env_kwargs_eval = {
        "df": trade_df,
        "stock_dim": stock_dimension_eval,
        "hmax": env_cfg.hmax,
        "initial_amount": env_cfg.initial_amount,
        "buy_cost_pct": env_cfg.buy_cost_pct,
        "sell_cost_pct": env_cfg.sell_cost_pct,
        "state_space": state_space_eval,
        "action_space": stock_dimension_eval,
        "tech_indicator_list": tech_indicator_list_eval,
        "reward_scaling": env_cfg.reward_scaling,
        "print_verbosity": eval_cfg.get("finrl_env_print_verbosity", 0),
        "initial": True, 
        "previous_state": [],
        "previous_portfolio_value": [], 
        "model_name": agent_cfg.name, 
        "mode": "trade"
    }
    try:
        env_eval_instance = StockPortfolioEnv(**env_kwargs_eval)
        env_eval = DummyVecEnv([lambda: env_eval_instance])
        log.info(f"DRL evaluation environment ('{env_cfg.name}') created.")
    except Exception as e_env_eval:
        log.error(f"Error creating DRL evaluation environment: {e_env_eval}", exc_info=True)
        current_task.close(); raise

    account_values = [env_cfg.initial_amount]
    obs, _ = env_eval.reset()
    terminated = False
    truncated = False 
    total_steps = 0
    max_episode_steps = len(trade_df.date.unique()) 

    log.info(f"Starting backtesting loop for {max_episode_steps} steps...")
    for step_num in range(max_episode_steps):
        action, _ = trained_drl_agent.predict(obs, deterministic=eval_cfg.get("deterministic_actions", True))
        obs, reward, terminated, truncated, info = env_eval.step(action)
        
        if info and isinstance(info, list) and len(info) > 0 and 'portfolio_value' in info[0]:
            account_values.append(info[0]['portfolio_value'])
        else:
            if account_values:
                account_values.append(account_values[-1]) 
            else:
                account_values.append(env_cfg.initial_amount)
            log.debug(f"Portfolio value not in info dict at step {step_num}. Info: {info}")

        if terminated or truncated:
            log.info(f"Backtesting episode ended at step {total_steps + 1} due to termination/truncation.")
            break
        total_steps += 1
    log.info(f"Backtesting loop completed. Total steps: {total_steps}. Portfolio values recorded: {len(account_values)}")

    unique_trade_dates = pd.to_datetime(trade_df.date.unique())
    if len(account_values) > len(unique_trade_dates):
        account_values = account_values[:len(unique_trade_dates)]
    elif len(account_values) < len(unique_trade_dates):
        padding_needed = len(unique_trade_dates) - len(account_values)
        account_values.extend([account_values[-1]] * padding_needed)
        log.warning(f"Account values ({len(account_values)}) shorter than unique dates ({len(unique_trade_dates)}). Padded with last value.")

    portfolio_value_df = pd.DataFrame({'date': unique_trade_dates, 'account_value': account_values})
    portfolio_value_df.set_index('date', inplace=True)

    log.info("Calculating DRL performance metrics...")
    metrics = {}
    try:
        if not portfolio_value_df.empty and 'account_value' in portfolio_value_df.columns:
            perf_stats_series = backtest_stats(account_value=portfolio_value_df['account_value'], value_col_name = 'account_value')
            if perf_stats_series is not None:
                metrics = perf_stats_series.to_dict()
                log.info(f"DRL Performance Metrics for {portfolio_id}: {metrics}")
                clearml_logger = current_task.get_logger()
                for metric_name, value in metrics.items():
                    if pd.notna(value):
                        clearml_logger.report_scalar(title="DRL Evaluation Metrics", series=metric_name, value=value, iteration=0)
            else:
                log.warning("backtest_stats returned None, no metrics calculated.")
        else:
            log.warning("Portfolio value DataFrame is empty or missing 'account_value' column. Cannot calculate metrics.")
    except Exception as e_metrics:
        log.error(f"Error calculating DRL performance metrics: {e_metrics}", exc_info=True)
        metrics["metrics_calculation_error"] = str(e_metrics)

    if eval_cfg.get("generate_plots", True) and not portfolio_value_df.empty:
        log.info("Generating DRL portfolio value plot...")
        try:
            plt.figure(figsize=tuple(eval_cfg.get("plot_figsize", [12, 6])))
            portfolio_value_df['account_value'].plot(title=f'DRL Agent Portfolio Value Over Time - {portfolio_id}')
            plt.xlabel(str(eval_cfg.get("plot_xlabel", 'Date')))
            plt.ylabel(str(eval_cfg.get("plot_ylabel", 'Portfolio Value')))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            current_task.get_logger().report_matplotlib_figure(
                title=f"DRL Portfolio Value Plot - {portfolio_id}", 
                series="Portfolio Performance", 
                figure=plt,
                iteration=0,
                report_image=True
            )
            log.info(f"Plot 'DRL Portfolio Value' for {portfolio_id} logged to ClearML.")
        except Exception as e_plot:
            log.error(f"Error generating or logging DRL plot: {e_plot}", exc_info=True)
        finally:
            plt.close()

    log.info("Saving DRL evaluation metrics...")
    metrics_to_save = {k: (None if pd.isna(v) else v) for k, v in metrics.items()} # Handle NaN for JSON
    metrics_filename = f"drl_evaluation_metrics_{portfolio_id}.json"
    with open(metrics_filename, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    current_task.upload_artifact(name=f"{portfolio_id}_drl_evaluation_metrics", artifact_object=metrics_filename)
    log.info(f"DRL evaluation metrics saved as artifact: {metrics_filename}")
    
    current_task.close()
    log.info(f"DRL evaluation task for portfolio {portfolio_id} completed.")

if __name__ == '__main__':
    main() 