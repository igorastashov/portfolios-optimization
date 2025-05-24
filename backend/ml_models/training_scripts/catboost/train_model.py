# Placeholder for CatBoost model training script 

import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool # cb_metrics not used
from clearml import Task, Artifact, OutputModel
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import json 

# Настройка логирования
log = logging.getLogger(__name__)

def load_parquet_artifact(data_prep_task_id: str, artifact_name: str) -> pd.DataFrame:
    """Загружает DataFrame из Parquet артефакта задачи ClearML."""
    try:
        artifact = Artifact.get(task_id=data_prep_task_id, name=artifact_name)
        if not artifact:
            log.error(f"Artifact '{artifact_name}' from task {data_prep_task_id} not found.")
            return pd.DataFrame()
        artifact_path = artifact.get_local_copy()
        if not artifact_path or not os.path.exists(artifact_path) or os.path.getsize(artifact_path) == 0:
             log.error(f"Artifact file '{artifact_name}' (path: {artifact_path}) is empty or does not exist.")
             return pd.DataFrame()
        log.info(f"Loading data from artifact: {artifact_path}")
        return pd.read_parquet(artifact_path)
    except Exception as e:
        log.error(f"Error loading artifact '{artifact_name}' from task {data_prep_task_id}: {e}", exc_info=True)
        return pd.DataFrame()

def load_json_artifact(data_prep_task_id: str, artifact_name: str) -> list | dict:
    """Загружает JSON из артефакта задачи ClearML."""
    try:
        artifact = Artifact.get(task_id=data_prep_task_id, name=artifact_name)
        if not artifact:
            log.error(f"JSON artifact '{artifact_name}' from task {data_prep_task_id} not found.")
            return [] 
        artifact_path = artifact.get_local_copy()
        if not artifact_path or not os.path.exists(artifact_path) or os.path.getsize(artifact_path) == 0:
             log.error(f"JSON artifact file '{artifact_name}' (path: {artifact_path}) is empty or does not exist.")
             return []
        log.info(f"Loading JSON from artifact: {artifact_path}")
        with open(artifact_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error loading JSON artifact '{artifact_name}' from task {data_prep_task_id}: {e}", exc_info=True)
        return []

@hydra.main(config_path="../../configs", config_name="catboost_config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_asset_ticker', type=str, required=True, help="Asset ticker for model training")
    parser.add_argument('--data_prep_task_id', type=str, required=True, help="ClearML task ID for data preparation")
    args, unknown = parser.parse_known_args()
    
    target_asset_ticker = args.target_asset_ticker
    data_prep_task_id = args.data_prep_task_id
    
    # task_name uses train_model prefix from config
    task_name_parametrized = f"{cfg.train_model.task_name_prefix}_{target_asset_ticker}"

    task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [target_asset_ticker, "training", "catboost"],
        output_uri=True 
    )
    
    effective_config = OmegaConf.to_container(cfg, resolve=True)
    effective_config['train_model']['target_asset_ticker'] = target_asset_ticker 
    effective_config['train_model']['data_prep_task_id'] = data_prep_task_id
    task.connect(effective_config, name='Effective_Hydra_Plus_Args_Configuration')

    log.info(f"Starting model training for {target_asset_ticker}.")
    log.info(f"Data preparation task ID: {data_prep_task_id}")
    tm_cfg = cfg.train_model 
    dp_cfg = cfg.data_preparation

    log.info("Loading prepared data...")
    X_train = load_parquet_artifact(data_prep_task_id, "train_data_features")
    y_train_df = load_parquet_artifact(data_prep_task_id, "train_data_target")
    X_val = load_parquet_artifact(data_prep_task_id, "validation_data_features")
    y_val_df = load_parquet_artifact(data_prep_task_id, "validation_data_target")
    cat_features = load_json_artifact(data_prep_task_id, "categorical_features_list")

    if X_train.empty or y_train_df.empty or X_val.empty or y_val_df.empty:
        log.error("Failed to load training or validation data (X or y). Aborting task.")
        task.close(); raise ValueError("Failed to load train/validation X or y data.")

    target_column_name_in_y_artifacts = dp_cfg.target_col_name
    if target_column_name_in_y_artifacts not in y_train_df.columns:
        log.warning(f"Target column '{target_column_name_in_y_artifacts}' not found in y_train_df. Columns: {y_train_df.columns.tolist()}. Using first column.")
        if not y_train_df.columns.empty: target_column_name_in_y_artifacts = y_train_df.columns[0]
        else: log.error("y_train_df has no columns."); task.close(); raise ValueError("y_train_df has no columns.")
            
    y_train = y_train_df[target_column_name_in_y_artifacts]
    # Ensure consistent target column name for validation set
    val_target_col_name = target_column_name_in_y_artifacts
    if val_target_col_name not in y_val_df.columns:
        log.warning(f"Target column '{val_target_col_name}' not found in y_val_df. Columns: {y_val_df.columns.tolist()}. Using first column.")
        if not y_val_df.columns.empty: val_target_col_name = y_val_df.columns[0]
        else: log.error("y_val_df has no columns."); task.close(); raise ValueError("y_val_df has no columns.")
    y_val = y_val_df[val_target_col_name]

    log.info(f"Categorical features loaded: {cat_features}")
    valid_cat_features = [cf for cf in cat_features if cf in X_train.columns]
    if len(valid_cat_features) != len(cat_features):
        log.warning(f"Not all categorical features found in X_train. Original: {cat_features}, Found: {valid_cat_features}")
    
    log.info(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
    log.info(f"Using categorical features (string names): {valid_cat_features}")
    
    log.info("Training CatBoost model...")
    model_params_dict = OmegaConf.to_container(tm_cfg.catboost_model_params, resolve=True)
    if 'cat_features' in model_params_dict: del model_params_dict['cat_features'] 
    verbose_level = model_params_dict.pop('verbose', tm_cfg.get('catboost_verbose', 100))

    model = CatBoostRegressor(**model_params_dict)
    train_pool = Pool(data=X_train, label=y_train, cat_features=valid_cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=valid_cat_features)

    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=verbose_level 
    )

    log.info("Model training completed.")
    best_iteration = -1
    best_score_dict = {}
    try:
        best_iteration = model.get_best_iteration() if hasattr(model, 'get_best_iteration') else model.tree_count_ - 1
        best_score_dict = model.get_best_score() if hasattr(model, 'get_best_score') else {}
    except Exception as e_best_score:
        log.warning(f"Could not get best_score/best_iteration from model: {e_best_score}")

    log.info(f"Best iteration: {best_iteration}")
    log.info(f"Best validation score (from get_best_score()): {best_score_dict}")

    if best_score_dict and 'validation' in best_score_dict:
        for metric_name_key, metric_value in best_score_dict['validation'].items():
            task.get_logger().report_scalar(
                title="Best Validation Score", 
                series=str(metric_name_key), 
                value=metric_value, 
                iteration=best_iteration
            )
            task.set_parameter(f"best_val_{metric_name_key}", metric_value)
            
    log.info("Saving trained model...")
    # Use model_registry_name_prefix for the OutputModel name in ClearML
    clearml_model_name = f"{tm_cfg.model_registry_name_prefix}_{target_asset_ticker}"
    model_local_path_cbm = f"{clearml_model_name}.cbm" 

    model.save_model(model_local_path_cbm)
    log.info(f"Model saved locally: {model_local_path_cbm}")
    
    output_model_clearml = OutputModel(
        task=task, 
        framework="CatBoost", 
        name=clearml_model_name 
    )
    output_model_clearml.update_weights(weights_filename=model_local_path_cbm)
    output_model_clearml.set_metadata("model_config_params", model_params_dict)
    output_model_clearml.set_metadata("data_preparation_task_id", data_prep_task_id)
    output_model_clearml.set_metadata("target_asset_ticker", target_asset_ticker)
    if best_iteration != -1: output_model_clearml.set_metadata("best_iteration_catboost", best_iteration)
    if best_score_dict: output_model_clearml.set_metadata("best_validation_scores_catboost", best_score_dict)
    output_model_clearml.set_tags(list(cfg.global_tags) + [target_asset_ticker, "catboost_regressor", "price_prediction"])

    log.info(f"Model '{clearml_model_name}' registered as OutputModel in ClearML.")
    task.close()
    log.info(f"Training task for {target_asset_ticker} completed.")

if __name__ == '__main__':
    # Для локального запуска и отладки
    # Убедитесь, что у вас есть:
    # 1. Конфигурация catboost_config.yaml в ../../configs
    # 2. Артефакты от предыдущей задачи (data_preparation) доступны.
    #    Для локального теста, можно создать dummy артефакты или мокнуть load_data_from_task_artifact.
    #
    # Пример вызова (потребует реальный data_prep_task_id):
    # python train_model.py --target_asset_ticker=BTCUSDT --data_prep_task_id=your_task_id_here
    #
    # Или так, если параметры Hydra передаются через CLI (но argparse для task_id и ticker обязателен)
    # python train_model.py --target_asset_ticker=BTCUSDT --data_prep_task_id=xxx model_params.iterations=100
    
    # ClearML PipelineController будет вызывать этот скрипт примерно так:
    # clearml-task --project PortfolioOptimization/PricePrediction \
    #              --name train_catboost_model_BTCUSDT \
    #              --script backend/ml_models/training_scripts/catboost/train_model.py \
    #              --args target_asset_ticker=BTCUSDT data_prep_task_id=${data_prep_task.id} \
    #              --docker python:3.10-slim \
    #              --queue default
    # Параметр data_prep_task_id=${data_prep_task.id} будет автоматически подставлен ClearML Pipelines
    # при указании зависимости от предыдущего шага.

    # Этот вызов main() здесь для того, чтобы Hydra отработала при запуске скрипта напрямую.
    # Если передавать аргументы командной строки, они будут обработаны argparse.
    main() 