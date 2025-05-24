import argparse
import json
import os
from clearml import Task, Model, Artifact
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pandas as pd

log = logging.getLogger(__name__)


def load_evaluation_metrics(eval_task_id: str, artifact_name: str) -> dict:
    """Loads evaluation metrics from a JSON artifact of a ClearML task."""
    try:
        artifact_obj = Artifact.get(task_id=eval_task_id, name=artifact_name)
        if not artifact_obj:
            log.error(f"Metrics artifact '{artifact_name}' from task {eval_task_id} not found.")
            return {}
        artifact_path = artifact_obj.get_local_copy()
        if not artifact_path or not os.path.exists(artifact_path) or os.path.getsize(artifact_path) == 0:
            log.error(f"Metrics artifact file '{artifact_name}' (path: {artifact_path}) is empty or does not exist.")
            return {}
        with open(artifact_path, 'r') as f:
            metrics = json.load(f)
        log.info(f"Metrics loaded from artifact: {artifact_path}, content: {metrics}")
        return metrics
    except Exception as e:
        log.error(f"Error loading metrics from artifact '{artifact_name}' task {eval_task_id}: {e}", exc_info=True)
        return {}


def get_output_model_object(train_task_id: str, model_clearml_name: str, project_name: str) -> Model | None:
    """Gets the OutputModel object from the training task by its name in ClearML."""
    try:
        models = Model.query_models(
            task_ids=[train_task_id], 
            model_name=model_clearml_name, 
            project_name=project_name
        )
        if not models:
            log.error(f"OutputModel '{model_clearml_name}' not found for task {train_task_id} in project {project_name}.")
            return None
        
        output_model = models[0]
        log.info(f"Found OutputModel: ID {output_model.id}, name '{output_model.name}'")
        return output_model
    except Exception as e:
        log.error(f"Error getting OutputModel '{model_clearml_name}' from task {train_task_id}: {e}", exc_info=True)
        return None

@hydra.main(config_path="../../configs", config_name="catboost_config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_asset_ticker', type=str, required=True, help="Asset ticker")
    parser.add_argument('--train_task_id', type=str, required=True, help="ClearML ID of the training task")
    parser.add_argument('--evaluation_task_id', type=str, required=True, help="ClearML ID of the evaluation task")
    args, unknown = parser.parse_known_args()

    target_asset_ticker = args.target_asset_ticker
    train_task_id = args.train_task_id
    evaluation_task_id = args.evaluation_task_id

    task_name_parametrized = f"{cfg.register_model.task_name_prefix}_{target_asset_ticker}"
    task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [target_asset_ticker, "registration", "catboost"],
        output_uri=True
    )
    effective_config = OmegaConf.to_container(cfg, resolve=True)
    effective_config['register_model']['target_asset_ticker'] = target_asset_ticker
    effective_config['register_model']['train_task_id'] = train_task_id
    effective_config['register_model']['evaluation_task_id'] = evaluation_task_id
    task.connect(effective_config, name='Effective_Hydra_Plus_Args_Configuration')

    log.info(f"Starting model registration for {target_asset_ticker}...")
    reg_cfg = cfg.register_model
    tm_cfg = cfg.train_model

    log.info("Loading evaluation metrics...")
    metrics_artifact_name = f"{target_asset_ticker}_evaluation_metrics"
    metrics = load_evaluation_metrics(evaluation_task_id, metrics_artifact_name)

    if not metrics:
        log.error("Evaluation metrics not loaded. Model registration cannot proceed."); task.close(); return

    log.info("Getting OutputModel from training task...")
    output_model_name_from_training = f"{tm_cfg.model_registry_name_prefix}_{target_asset_ticker}"
    clearml_project_name = cfg.project_name 
    
    model_to_register = get_output_model_object(train_task_id, output_model_name_from_training, clearml_project_name)

    if not model_to_register:
        log.error("Failed to get OutputModel for registration. Aborting."); task.close(); return

    log.info("Deciding on model registration...")
    metric_key = reg_cfg.registration_threshold_metric
    threshold_value = reg_cfg.registration_threshold_value
    operator = reg_cfg.comparison_operator.lower()
    should_register = False

    if metric_key in metrics:
        current_metric_value = metrics[metric_key]
        if pd.isna(current_metric_value):
            log.warning(f"Metric '{metric_key}' value is NaN. Model will not be registered.")
        else:
            log.info(f"Evaluation metric: {metric_key} = {current_metric_value}, Threshold: {threshold_value}, Operator: {operator}")
            if operator == "less_than":
                should_register = current_metric_value < threshold_value
            elif operator == "greater_than":
                should_register = current_metric_value > threshold_value
            else:
                log.error(f"Unknown comparison operator: {operator}. Allowed: 'less_than', 'greater_than'.")

            if should_register:
                log.info("Model meets registration criteria.")
            else:
                log.warning(f"Model does NOT meet criteria. {metric_key} ({current_metric_value}) {operator} {threshold_value} = False")
    else:
        log.warning(f"Metric '{metric_key}' not found in evaluation results ({metrics.keys()}). Model will not be registered.")

    task.get_logger().report_scalar(
        title="Registration Decision", series="ShouldRegister", value=1 if should_register else 0, iteration=0
    )

    if should_register:
        log.info(f"Publishing model ID {model_to_register.id} ({model_to_register.name}) in ClearML...")
        try:
            model_to_register.publish()
            current_tags = model_to_register.tags or []
            new_tags = list(set(current_tags + ["registered", reg_cfg.get("default_stage", "Production")]))
            model_to_register.set_tags(new_tags)
            model_to_register.set_metadata_item(key="registration_evaluation_metrics", value=json.dumps(metrics))
            model_to_register.set_metadata_item(key="registration_decision_metric", value=f"{metric_key}: {metrics.get(metric_key)}")
            model_to_register.set_metadata_item(key="registration_decision_threshold", value=f"{operator} {threshold_value}")
            model_to_register.update(suppress_warnings=True)

            log.info(f"Model ID {model_to_register.id} ({model_to_register.name}) published and updated.")
            task.set_parameter("registered_published_model_id", model_to_register.id)
            task.set_parameter("registered_published_model_name", model_to_register.name)
        except Exception as e_pub:
            log.error(f"Error publishing or updating model: {e_pub}", exc_info=True)
    else:
        log.info("Model will not be published/registered based on metrics.")
        task.set_parameter("registered_published_model_id", None)

    task.close()
    log.info(f"Registration task for {target_asset_ticker} completed.")

if __name__ == '__main__':
    main() 