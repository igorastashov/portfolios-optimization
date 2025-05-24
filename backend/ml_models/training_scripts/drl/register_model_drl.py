import argparse
import json
import os
from clearml import Task, Model, Artifact
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pandas as pd

log = logging.getLogger(__name__)

def load_drl_evaluation_metrics(eval_task_id: str, artifact_name: str) -> dict:
    try:
        artifact = Artifact.get(task_id=eval_task_id, name=artifact_name)
        if not artifact:
            log.error(f"DRL evaluation metrics artifact '{artifact_name}' from task {eval_task_id} not found.")
            return {}
        artifact_path = artifact.get_local_copy()
        if not artifact_path or not os.path.exists(artifact_path) or os.path.getsize(artifact_path) == 0:
            log.error(f"Local copy of DRL metrics artifact '{artifact_name}' from task {eval_task_id} (path: {artifact_path}) is invalid.")
            return {}
        with open(artifact_path, 'r') as f:
            metrics = json.load(f)
        log.info(f"DRL evaluation metrics loaded from artifact: {artifact_path}, content: {metrics}")
        return metrics
    except Exception as e:
        log.error(f"Error loading DRL metrics from artifact '{artifact_name}' task {eval_task_id}: {e}", exc_info=True)
        return {}

def get_drl_input_model_from_task(train_task_id: str, model_clearml_name: str, project_name: str) -> Model | None:
    try:
        log.info(f"Fetching DRL OutputModel '{model_clearml_name}' from training task {train_task_id} in project {project_name}")
        models = Model.query_models(task_ids=[train_task_id], model_name=model_clearml_name, project_name=project_name, max_results=1)
        if not models:
            log.error(f"DRL OutputModel '{model_clearml_name}' not found for task {train_task_id} in project {project_name}.")
            return None
        input_model = models[0]
        log.info(f"Found DRL model (OutputModel) for registration: ID {input_model.id}, name '{input_model.name}'")
        return input_model
    except Exception as e:
        log.error(f"Error getting DRL OutputModel '{model_clearml_name}' from task {train_task_id}: {e}", exc_info=True)
        return None

@hydra.main(config_path="../../configs", config_name="drl_config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio_id', type=str, required=True, help="Portfolio ID for DRL model registration")
    parser.add_argument('--train_task_id', type=str, required=True, help="ClearML task ID of DRL model training")
    parser.add_argument('--evaluation_task_id', type=str, required=True, help="ClearML task ID of DRL model evaluation")
    args, unknown = parser.parse_known_args()

    portfolio_id = args.portfolio_id
    train_task_id = args.train_task_id
    evaluation_task_id = args.evaluation_task_id

    task_name_parametrized = f"{cfg.registration.task_name_prefix}_{portfolio_id}"
    current_task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [portfolio_id, "registration", "drl"],
        output_uri=True
    )
    effective_config_dict = OmegaConf.to_container(cfg, resolve=True)
    effective_config_dict['registration']['portfolio_id'] = portfolio_id
    effective_config_dict['registration']['train_task_id'] = train_task_id
    effective_config_dict['registration']['evaluation_task_id'] = evaluation_task_id
    current_task.connect(effective_config_dict, name='Effective_Hydra_Plus_Args_Configuration')

    log.info(f"Starting DRL model registration for portfolio: {portfolio_id}...")
    reg_cfg = cfg.registration
    agent_cfg = cfg.agent 

    log.info("Loading DRL evaluation metrics...")
    metrics_artifact_name = f"{portfolio_id}_drl_evaluation_metrics" 
    metrics = load_drl_evaluation_metrics(evaluation_task_id, metrics_artifact_name)
    if not metrics or metrics.get("metrics_calculation_error"):
        log.error(f"DRL evaluation metrics not loaded or contain error ({metrics.get('metrics_calculation_error')}). Model registration cannot proceed.")
        current_task.close(); raise ValueError("DRL evaluation metrics are invalid or not found.")

    log.info("Fetching DRL OutputModel from training task...")
    model_clearml_name = f"{agent_cfg.model_registry_name_prefix}_{agent_cfg.name.lower()}_{portfolio_id}"
    input_model_to_register = get_drl_input_model_from_task(train_task_id, model_clearml_name, cfg.project_name)
    if not input_model_to_register:
        log.error("Failed to retrieve DRL OutputModel for registration. Aborting.")
        current_task.close(); raise ValueError("Could not retrieve DRL OutputModel.")

    log.info("Deciding on DRL model registration based on metrics...")
    metric_to_evaluate = reg_cfg.metric_to_evaluate
    metric_threshold = float(reg_cfg.metric_threshold) 
    comparison_operator = reg_cfg.get("comparison_operator", "greater_than").lower()

    should_register = False
    current_metric_value = metrics.get(metric_to_evaluate)

    if current_metric_value is not None and pd.notna(current_metric_value):
        try:
            current_metric_value_float = float(current_metric_value)
            log.info(f"DRL registration evaluation metric: {metric_to_evaluate} = {current_metric_value_float}, Threshold: {metric_threshold}, Operator: {comparison_operator}")
            if comparison_operator == "greater_than":
                should_register = current_metric_value_float > metric_threshold
            elif comparison_operator == "less_than":
                should_register = current_metric_value_float < metric_threshold
            else:
                log.error(f"Unknown comparison operator: {comparison_operator}. Allowed: 'less_than', 'greater_than'.")

            if should_register:
                log.info("DRL model meets registration criteria.")
            else:
                log.warning(f"DRL model does NOT meet criteria: {metric_to_evaluate} ({current_metric_value_float}) vs threshold ({metric_threshold}) with operator '{comparison_operator}'.")
        except ValueError:
            log.warning(f"Metric value '{current_metric_value}' for '{metric_to_evaluate}' cannot be converted to float. Model will not be registered.")
    else:
        log.warning(f"Metric '{metric_to_evaluate}' not found or is NaN in DRL evaluation results ({metrics.keys()}). Model will not be registered.")
    
    current_task.get_logger().report_scalar(title="DRL Registration Decision", series="ShouldRegister", value=1 if should_register else 0, iteration=0)

    if should_register:
        log.info("Registering DRL model in ClearML Model Registry...")
        try:
            registered_model_name = reg_cfg.get("registered_model_name_prefix", "DRL_PortfolioRebalancer") + f"_{portfolio_id}_{agent_cfg.name.lower()}"
            
            cloned_model_for_registry = Model.clone(
                source_model=input_model_to_register.id, 
                new_model_name=registered_model_name, 
                project_name=cfg.project_name, # Ensure it's registered to the correct project
                comment=f"Registered DRL model for {portfolio_id}. Source train task: {train_task_id}, eval task: {evaluation_task_id}. Metric {metric_to_evaluate}: {current_metric_value}."
            )
            log.info(f"DRL OutputModel ID {input_model_to_register.id} cloned for registry as new model ID {cloned_model_for_registry.id} under name '{registered_model_name}'.")
            
            cloned_model_for_registry.set_metadata("evaluation_metrics_artifact_name", metrics_artifact_name)
            cloned_model_for_registry.set_metadata("evaluation_metrics_content", metrics)
            cloned_model_for_registry.set_metadata("source_training_task_id", train_task_id)
            cloned_model_for_registry.set_metadata("source_evaluation_task_id", evaluation_task_id)
            cloned_model_for_registry.set_metadata("portfolio_id", portfolio_id)
            cloned_model_for_registry.set_metadata("drl_agent_name", agent_cfg.name.lower())
            cloned_model_for_registry.set_metadata("registration_metric_evaluated", metric_to_evaluate)
            cloned_model_for_registry.set_metadata("registration_metric_value", current_metric_value)
            cloned_model_for_registry.set_metadata("registration_metric_threshold", metric_threshold)
            
            model_tags = list(cfg.global_tags) + \
                         [portfolio_id, "registered", "drl_agent", agent_cfg.name.lower(), reg_cfg.get("default_stage", "Production")]
            cloned_model_for_registry.set_tags(model_tags)
            
            cloned_model_for_registry.publish()
            log.info(f"DRL model ID {cloned_model_for_registry.id} ('{cloned_model_for_registry.name}') published to ClearML Model Registry.")
            current_task.set_parameter("registered_drl_model_id", cloned_model_for_registry.id)
            current_task.set_parameter("registered_drl_model_name", cloned_model_for_registry.name)

        except Exception as e_reg:
            log.error(f"Error during DRL model registration: {e_reg}", exc_info=True)
            current_task.set_parameter("registered_drl_model_id", None)
            current_task.set_parameter("registered_drl_model_name", None)
    else:
        log.info("DRL model will not be registered as it does not meet the criteria.")
        current_task.set_parameter("registered_drl_model_id", None)
        current_task.set_parameter("registered_drl_model_name", None)

    current_task.close()
    log.info(f"DRL model registration task for portfolio {portfolio_id} completed.")

if __name__ == '__main__':
    main() 