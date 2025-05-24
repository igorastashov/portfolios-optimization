from clearml import PipelineController, Task
import hydra
from clearml import PipelineController, Task
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

log = logging.getLogger(__name__)
SCRIPT_BASE_PATH = "backend/ml_models/training_scripts/catboost/"


@hydra.main(config_path="../../configs", config_name="catboost_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Defining CatBoost training pipeline...")

    pipeline_project = cfg.project_name
    pipeline_name = cfg.pipeline_name
    pipeline_version = cfg.get("catboost_pipeline_version", "1.0.0") 
    default_queue = cfg.queue_name
    
    target_asset_tickers = list(cfg.pipeline_params.target_asset_tickers)
    if not target_asset_tickers:
        log.error("target_asset_tickers list is empty. Pipeline will not be created.")
        return

    log.info(f"Pipeline will be defined for assets: {target_asset_tickers}")

    pipe = PipelineController(
        project=pipeline_project,
        name=pipeline_name,
        version=pipeline_version,
        add_pipeline_tags=True,
        target_project=pipeline_project 
    )

    pipe.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name='Pipeline_Effective_Configuration')
    pipe.set_default_execution_queue(default_queue)

    for ticker in target_asset_tickers:
        log.info(f"Defining pipeline steps for ticker: {ticker}")
        
        data_prep_script_path = os.path.join(SCRIPT_BASE_PATH, "data_preparation.py")
        data_prep_task_id = pipe.add_step(
            name=f"catboost_data_prep_{ticker}",
            script_path=data_prep_script_path,
            parameters={"target_asset_ticker": ticker}
        )
        log.info(f"Data preparation step for {ticker} added.")

        train_model_script_path = os.path.join(SCRIPT_BASE_PATH, "train_model.py")
        train_task_id = pipe.add_step(
            name=f"catboost_train_{ticker}",
            parents=[data_prep_task_id],
            script_path=train_model_script_path,
            parameters={
                "target_asset_ticker": ticker,
                "data_prep_task_id": f"${{{data_prep_task_id}.id}}"
            }
        )
        log.info(f"Model training step for {ticker} added.")

        eval_model_script_path = os.path.join(SCRIPT_BASE_PATH, "evaluate_model.py")
        eval_task_id = pipe.add_step(
            name=f"catboost_evaluate_{ticker}",
            parents=[train_task_id],
            script_path=eval_model_script_path,
            parameters={
                "target_asset_ticker": ticker,
                "data_prep_task_id": f"${{{data_prep_task_id}.id}}",
                "train_task_id": f"${{{train_task_id}.id}}"
            }
        )
        log.info(f"Model evaluation step for {ticker} added.")

        register_model_script_path = os.path.join(SCRIPT_BASE_PATH, "register_model.py")
        pipe.add_step(
            name=f"catboost_register_{ticker}",
            parents=[eval_task_id],
            script_path=register_model_script_path,
            parameters={
                "target_asset_ticker": ticker,
                "train_task_id": f"${{{train_task_id}.id}}",
                "evaluation_task_id": f"${{{eval_task_id}.id}}"
            }
        )
        log.info(f"Model registration step for {ticker} added.")

    if cfg.pipeline_params.get("run_pipeline_locally_after_definition", False):
        log.info(f"Starting pipeline '{pipeline_name}' v{pipeline_version} locally in queue '{default_queue}'...")
        pipe.start(queue_name=default_queue)
        log.info("Pipeline sent for execution.")
    else:
        log.info(f"Pipeline '{pipeline_name}' v{pipeline_version} defined. Execute via ClearML UI or CLI.")

    pipeline_controller_task_id = pipe.get_id()
    log.info(f"ClearML Pipeline Controller Task ID: {pipeline_controller_task_id}")
    print(f"CLEARML_PIPELINE_TASK_ID:{pipeline_controller_task_id}")

    current_defining_task = Task.current_task()
    if current_defining_task:
        log.info(f"Pipeline definition script running as ClearML Task ID: {current_defining_task.id}")
        if current_defining_task.id != pipeline_controller_task_id:
             current_defining_task.set_parameter("PipelineControllerTaskID", pipeline_controller_task_id)
             current_defining_task.set_parameter("GeneratedPipelineName", pipeline_name)
             current_defining_task.set_parameter("GeneratedPipelineVersion", pipeline_version)

if __name__ == '__main__':
    main() 