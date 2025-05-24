# Placeholder for DRL Training Pipeline Orchestrator 
from clearml import PipelineController, Task
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

log = logging.getLogger(__name__)

# Path to DRL training scripts (relative to project root or agent's working dir)
SCRIPT_BASE_PATH = "backend/ml_models/training_scripts/drl/"

@hydra.main(config_path="../../configs", config_name="drl_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Defining DRL training pipeline...")

    pipeline_project = cfg.project_name
    pipeline_name = cfg.drl_pipeline_name
    pipeline_version = cfg.drl_pipeline_version
    default_queue = cfg.default_execution_queue

    pipe = PipelineController(
        project=pipeline_project,
        name=pipeline_name,
        version=pipeline_version,
        add_pipeline_tags=True,
        target_project=pipeline_project 
    )
    pipe.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name='Pipeline_Effective_Configuration')
    pipe.set_default_execution_queue(default_queue)

    portfolio_id_for_pipeline = cfg.pipeline_params.get("portfolio_id", "DEFAULT_DRL_PORTFOLIO")
    log.info(f"Defining DRL pipeline steps for portfolio: {portfolio_id_for_pipeline}")

    data_prep_drl_script_path = os.path.join(SCRIPT_BASE_PATH, "data_preparation_drl.py")
    data_prep_drl_task_id = pipe.add_step(
        name=f"drl_data_prep_{portfolio_id_for_pipeline}",
        script_path=data_prep_drl_script_path,
        parameters={"portfolio_id": portfolio_id_for_pipeline}
    )
    log.info(f"DRL data preparation step for {portfolio_id_for_pipeline} added.")

    train_model_drl_script_path = os.path.join(SCRIPT_BASE_PATH, "train_model_drl.py")
    train_drl_task_id = pipe.add_step(
        name=f"drl_train_model_{portfolio_id_for_pipeline}",
        parents=[data_prep_drl_task_id],
        script_path=train_model_drl_script_path,
        parameters={
            "portfolio_id": portfolio_id_for_pipeline,
            "data_prep_task_id": f"${{{data_prep_drl_task_id}.id}}"
        }
    )
    log.info(f"DRL model training step for {portfolio_id_for_pipeline} added.")

    eval_model_drl_script_path = os.path.join(SCRIPT_BASE_PATH, "evaluate_model_drl.py")
    eval_drl_task_id = pipe.add_step(
        name=f"drl_evaluate_model_{portfolio_id_for_pipeline}",
        parents=[train_drl_task_id], 
        script_path=eval_model_drl_script_path,
        parameters={
            "portfolio_id": portfolio_id_for_pipeline,
            "data_prep_task_id": f"${{{data_prep_drl_task_id}.id}}", 
            "train_task_id": f"${{{train_drl_task_id}.id}}"
        }
    )
    log.info(f"DRL model evaluation step for {portfolio_id_for_pipeline} added.")

    register_model_drl_script_path = os.path.join(SCRIPT_BASE_PATH, "register_model_drl.py")
    pipe.add_step(
        name=f"drl_register_model_{portfolio_id_for_pipeline}",
        parents=[eval_drl_task_id],
        script_path=register_model_drl_script_path,
        parameters={
            "portfolio_id": portfolio_id_for_pipeline,
            "train_task_id": f"${{{train_drl_task_id}.id}}",
            "evaluation_task_id": f"${{{eval_drl_task_id}.id}}"
        }
    )
    log.info(f"DRL model registration step for {portfolio_id_for_pipeline} added.")

    if cfg.run_pipeline_locally_after_definition:
        log.info(f"Starting DRL pipeline '{pipeline_name}' v{pipeline_version} locally in queue '{default_queue}'...")
        pipe.start(queue_name=default_queue)
        log.info("DRL pipeline sent for execution.")
    else:
        log.info(f"DRL pipeline '{pipeline_name}' v{pipeline_version} defined. Execute via ClearML UI or CLI.")

    pipeline_controller_task_id = pipe.get_id()
    log.info(f"ClearML DRL Pipeline Controller Task ID: {pipeline_controller_task_id}")
    print(f"CLEARML_PIPELINE_TASK_ID:{pipeline_controller_task_id}")

    current_defining_task = Task.current_task()
    if current_defining_task:
        log.info(f"DRL pipeline definition script running as ClearML Task ID: {current_defining_task.id}")
        if current_defining_task.id != pipeline_controller_task_id:
             current_defining_task.set_parameter("PipelineControllerTaskID", pipeline_controller_task_id)
             current_defining_task.set_parameter("GeneratedPipelineName", pipeline_name)
             current_defining_task.set_parameter("GeneratedPipelineVersion", pipeline_version)

if __name__ == '__main__':
    # python backend/ml_models/clearml_pipelines/drl_training_pipeline.py pipeline_params.portfolio_id=MY_CRYPTO
    main() 