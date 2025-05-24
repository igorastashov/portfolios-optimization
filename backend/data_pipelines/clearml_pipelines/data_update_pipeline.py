# Placeholder for data update pipeline orchestrator
from clearml import PipelineController, Task
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

log = logging.getLogger(__name__)

# Пути к скриптам относительно корня проекта (предполагается, что пайплайн запускается из корня)
# или относительно рабочей директории агента ClearML, если он клонирует репозиторий.
# Если `backend` - это папка в корне репозитория:
SCRIPT_BASE_PATH = "backend/data_pipelines/tasks/"

# Предполагаем, что пути к скриптам даны относительно корня проекта
BINANCE_SCRIPT_PATH = "backend/data_pipelines/tasks/fetch_binance_data.py"
NEWS_SCRIPT_PATH = "backend/data_pipelines/tasks/fetch_alphavantage_news.py" # Пример, если есть
# NEWS_SCRIPT_PATH = "backend/data_pipelines/tasks/fetch_finnhub_news.py"

@hydra.main(config_path="../../configs", config_name="data_fetch_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Defining data update pipeline...")

    pipeline_project = cfg.project_name
    pipeline_name = cfg.data_update_pipeline_name
    pipeline_version = cfg.data_update_pipeline_version
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

    executed_steps_ids = []

    if cfg.data_sources.binance.get("enabled", False):
        binance_script_path = os.path.join(SCRIPT_BASE_PATH, "fetch_binance_data.py")
        # Parameters for the script are typically handled by its own Hydra config, 
        # but can be overridden here if script supports argparse for Hydra overrides.
        binance_step_id = pipe.add_step(
            name="fetch_binance_market_data_step",
            script_path=binance_script_path
            # No parameters needed if fetch_binance_data.py uses its own section of data_fetch_config.yaml
        )
        log.info(f"Binance data fetching step added.")
        executed_steps_ids.append(binance_step_id)
    else:
        log.info("Binance data fetching is disabled in config.")

    if cfg.data_sources.alphavantage_news.get("enabled", False):
        alphavantage_script_path = os.path.join(SCRIPT_BASE_PATH, "fetch_alphavantage_news.py")
        alphavantage_step_id = pipe.add_step(
            name="fetch_alphavantage_news_step",
            script_path=alphavantage_script_path
            # No parameters needed if script uses its own section of data_fetch_config.yaml
        )
        log.info(f"Alpha Vantage news fetching step added.")
        executed_steps_ids.append(alphavantage_step_id)
    else:
        log.info("Alpha Vantage news fetching is disabled in config.")

    if not executed_steps_ids:
        log.warning("No data fetching steps were enabled. Pipeline will be empty.")

    if cfg.run_pipeline_locally_after_definition:
        log.info(f"Starting data update pipeline '{pipeline_name}' v{pipeline_version} locally in queue '{default_queue}'...")
        pipe.start(queue_name=default_queue)
        log.info("Data update pipeline sent for execution.")
    else:
        log.info(f"Data update pipeline '{pipeline_name}' v{pipeline_version} defined. Execute via ClearML UI or CLI.")

    pipeline_controller_task_id = pipe.get_id()
    log.info(f"ClearML Data Update Pipeline Controller Task ID: {pipeline_controller_task_id}")
    print(f"CLEARML_PIPELINE_TASK_ID:{pipeline_controller_task_id}")

    current_defining_task = Task.current_task()
    if current_defining_task:
        log.info(f"Data update pipeline definition script running as ClearML Task ID: {current_defining_task.id}")
        if current_defining_task.id != pipeline_controller_task_id:
             current_defining_task.set_parameter("PipelineControllerTaskID", pipeline_controller_task_id)
             current_defining_task.set_parameter("GeneratedPipelineName", pipeline_name)
             current_defining_task.set_parameter("GeneratedPipelineVersion", pipeline_version)

if __name__ == '__main__':
    # Пример запуска:
    # python backend/data_pipelines/clearml_pipelines/data_update_pipeline.py
    #
    # Для запуска с выполнением (если агент настроен):
    # python backend/data_pipelines/clearml_pipelines/data_update_pipeline.py run_pipeline_locally_after_definition=true
    #
    # Также убедитесь, что есть data_fetch_config.yaml в ../../configs
    main() 