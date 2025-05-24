# Placeholder for CatBoost training pipeline orchestrator 

from clearml import PipelineController, Task
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

# Настройка логирования
log = logging.getLogger(__name__)

# Абсолютный путь к директории со скриптами обучения. 
# Это необходимо, чтобы ClearML мог найти скрипты при запуске пайплайна.
# Путь должен быть относительно МЕСТА ЗАПУСКА этого скрипта `catboost_training_pipeline.py`
# или абсолютным.
# Если pipeline-скрипт лежит в backend/ml_models/clearml_pipelines,
# а скрипты шагов в backend/ml_models/training_scripts/catboost,
# то относительный путь будет ../training_scripts/catboost/

# Более надежный способ - использовать абсолютные пути или пути относительно корня проекта,
# если скрипт запускается из определенного места.
# Для примера, предполагаем, что этот скрипт запускается из `backend/ml_models/clearml_pipelines`
# или что `PYTHONPATH` настроен так, что `ml_models.training_scripts...` доступно.

# Определим путь к скриптам более гибко:
# Это текущая директория (где лежит этот pipeline-скрипт)
# PIPELINE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Директория `ml_models` (на два уровня выше)
# ML_MODELS_DIR = os.path.dirname(os.path.dirname(PIPELINE_FILE_DIR))
# Путь к скриптам CatBoost
# CATBOOST_SCRIPTS_DIR = os.path.join(ML_MODELS_DIR, "training_scripts", "catboost")

# Упрощенный вариант: предполагаем, что скрипты находятся в ../training_scripts/catboost
# относительно этого файла. ClearML агент должен будет найти их по этому пути.
# При запуске через `clearml-pipeline execute` или из UI, агент будет использовать этот путь.
# Важно, чтобы эти пути были корректными в КОНТЕКСТЕ ВЫПОЛНЕНИЯ агента ClearML.

# Самый простой вариант для ClearML - указать путь к скрипту относительно корня репозитория,
# когда агент клонирует репозиторий. Например, "ml_models/training_scripts/catboost/data_preparation.py".
# Либо, если запускаем задачу из локального кода, то путь должен быть доступен.

# Давайте использовать пути относительно `backend/` как базовой директории при указании скриптов.
# То есть, если агент ClearML клонирует репозиторий и `backend` - это поддиректория,
# то пути будут `ml_models/training_scripts/catboost/data_preparation.py` и т.д.,
# если рабочая директория агента установлена в `backend`.
# Или полный путь от корня репозитория: `backend/ml_models/training_scripts/catboost/data_preparation.py`

# Предположим, что ClearML агент будет запускать шаги из корня проекта,
# и мы указываем полный путь к скриптам от корня.
# Если проект структурирован так, что `backend` - это папка в корне:
SCRIPT_BASE_PATH = "backend/ml_models/training_scripts/catboost/"


@hydra.main(config_path="../../configs", config_name="catboost_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Defining CatBoost training pipeline...")
    # log.info(f"Hydra Config: {OmegaConf.to_yaml(cfg)}") # Optional: for debugging

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
    # Optionally set a default docker image if all steps use the same and it's not in clearml.conf
    # default_docker_image = cfg.get("base_docker_image", "python:3.10-slim")
    # if default_docker_image: pipe.set_default_docker_image(default_docker_image)

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
    print(f"CLEARML_PIPELINE_TASK_ID:{pipeline_controller_task_id}") # For Celery worker

    current_defining_task = Task.current_task()
    if current_defining_task:
        log.info(f"Pipeline definition script running as ClearML Task ID: {current_defining_task.id}")
        if current_defining_task.id != pipeline_controller_task_id:
             current_defining_task.set_parameter("PipelineControllerTaskID", pipeline_controller_task_id)
             current_defining_task.set_parameter("GeneratedPipelineName", pipeline_name)
             current_defining_task.set_parameter("GeneratedPipelineVersion", pipeline_version)

if __name__ == '__main__':
    # Для запуска этого скрипта:
    # 1. Убедитесь, что ClearML сконфигурирован (clearml-init).
    # 2. Убедитесь, что есть конфигурация `catboost_config.yaml` в `../../configs`.
    # 3. Запустите: `python backend/ml_models/clearml_pipelines/catboost_training_pipeline.py`
    #    (путь может зависеть от вашей текущей директории)
    #
    # По умолчанию, он только определит пайплайн. 
    # Чтобы запустить его локально (если агент настроен), установите 
    # `run_pipeline_locally_after_definition: true` в `catboost_config.yaml` или передайте как параметр Hydra:
    # `python ... catboost_training_pipeline.py run_pipeline_locally_after_definition=true`
    main() 