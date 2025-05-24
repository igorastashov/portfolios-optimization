import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from clearml import Task, Artifact, Model, Logger
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import json
import matplotlib.pyplot as plt

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

def load_catboost_model_from_task(train_task_id: str, model_clearml_name: str) -> CatBoostRegressor | None:
    """Загружает модель CatBoost из OutputModel задачи обучения по ее имени в ClearML."""
    try:
        log.info(f"Loading OutputModel '{model_clearml_name}' from training task {train_task_id}")
        current_task = Task.get_current_task()
        project_name = current_task.project_name if current_task else None
        
        models = Model.query_models(
            project_name=project_name, 
            model_name=model_clearml_name, 
            task_ids=[train_task_id],
        )
        
        if not models:
            log.error(f"OutputModel '{model_clearml_name}' not found for task {train_task_id} in project {project_name}.")
            return None
        
        model_object = models[0]
        log.info(f"Found model '{model_clearml_name}' (ID: {model_object.id}). Downloading...")
        model_local_path = model_object.get_local_copy()
        
        model = CatBoostRegressor()
        model.load_model(model_local_path)
        log.info(f"Model '{model_clearml_name}' loaded successfully from {model_local_path}.")
        return model
        
    except Exception as e:
        log.error(f"Error loading model '{model_clearml_name}' from task {train_task_id}: {e}", exc_info=True)
        return None

@hydra.main(config_path="../../configs", config_name="catboost_config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_asset_ticker', type=str, required=True, help="Asset ticker")
    parser.add_argument('--data_prep_task_id', type=str, required=True, help="ClearML task ID for data preparation")
    parser.add_argument('--train_task_id', type=str, required=True, help="ClearML task ID for model training")
    args, unknown = parser.parse_known_args()

    target_asset_ticker = args.target_asset_ticker
    data_prep_task_id = args.data_prep_task_id
    train_task_id = args.train_task_id

    task_name_parametrized = f"{cfg.evaluate_model.task_name_prefix}_{target_asset_ticker}"
    task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name_parametrized,
        tags=list(cfg.global_tags) + [target_asset_ticker, "evaluation", "catboost"],
        output_uri=True
    )
    effective_config = OmegaConf.to_container(cfg, resolve=True)
    effective_config['evaluate_model']['target_asset_ticker'] = target_asset_ticker
    effective_config['evaluate_model']['data_prep_task_id'] = data_prep_task_id
    effective_config['evaluate_model']['train_task_id'] = train_task_id
    task.connect(effective_config, name='Effective_Hydra_Plus_Args_Configuration')

    log.info(f"Starting model evaluation for {target_asset_ticker}...")
    dp_cfg = cfg.data_preparation
    tm_cfg = cfg.train_model
    eval_cfg = cfg.evaluate_model

    log.info("Loading test data...")
    X_test = load_parquet_artifact(data_prep_task_id, "test_data_features")
    y_test_df = load_parquet_artifact(data_prep_task_id, "test_data_target")
    y_naive_pred_df = load_parquet_artifact(data_prep_task_id, "test_data_naive_baseline")

    if X_test.empty or y_test_df.empty:
        log.error("Failed to load X_test or y_test_df. Aborting task.")
        task.close(); raise ValueError("Failed to load test X or y data.")
    
    target_column_name_in_y_artifacts = dp_cfg.target_col_name
    if target_column_name_in_y_artifacts not in y_test_df.columns:
        log.warning(f"Target column '{target_column_name_in_y_artifacts}' not found in y_test_df. Using first column.")
        target_column_name_in_y_artifacts = y_test_df.columns[0] if not y_test_df.columns.empty else None
    if not target_column_name_in_y_artifacts:
        log.error("y_test_df has no columns."); task.close(); raise ValueError("y_test_df has no columns.")
    y_test = y_test_df[target_column_name_in_y_artifacts]
    
    log.info(f"Test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
    if not y_naive_pred_df.empty:
        log.info(f"Naive baseline predictions loaded: {y_naive_pred_df.shape}")

    log.info("Loading trained CatBoost model...")
    model_clearml_name = f"{tm_cfg.model_registry_name_prefix}_{target_asset_ticker}" 
    model = load_catboost_model_from_task(train_task_id, model_clearml_name)

    if model is None:
        log.error("Failed to load model. Aborting task.")
        task.close(); raise ValueError("Failed to load trained model.")

    log.info("Evaluating model on test data...")
    y_pred = model.predict(X_test)

    metrics = {}
    y_test_np = y_test.squeeze().values if isinstance(y_test, pd.DataFrame) else y_test.values 
    
    if len(y_test_np) == 0 or len(y_pred) == 0 or len(y_test_np) != len(y_pred):
        log.error(f"y_test ({len(y_test_np)}) and y_pred ({len(y_pred)}) sizes mismatch or empty. Cannot calculate metrics.")
    else:
        try:
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_test_np, y_pred))
            metrics["MAE"] = mean_absolute_error(y_test_np, y_pred)
            mask_zero_true = y_test_np != 0
            if np.any(mask_zero_true):
                 metrics["MAPE"] = mean_absolute_percentage_error(y_test_np[mask_zero_true], y_pred[mask_zero_true])
            else:
                 metrics["MAPE"] = np.nan
            metrics["R2_score"] = r2_score(y_test_np, y_pred)   
        except Exception as e_metrics:
            log.error(f"Error calculating main metrics: {e_metrics}", exc_info=True)

    log.info(f"Test set metrics for {target_asset_ticker}: {metrics}")
    clearml_logger = task.get_logger()
    for metric_name, value in metrics.items():
        if pd.notna(value):
            clearml_logger.report_scalar(title="Test Set Metrics", series=metric_name, value=value, iteration=0)

    y_naive_final_predictions = None
    if eval_cfg.get("compare_with_naive_model", False) and not y_naive_pred_df.empty:
        log.info("Comparing with naive model...")
        naive_pred_col_name = 'y_pred_naive_source'
        if naive_pred_col_name not in y_naive_pred_df.columns:
            log.warning(f"Column '{naive_pred_col_name}' not found in y_naive_pred_df. Columns: {y_naive_pred_df.columns.tolist()}")
        else:
            y_naive_final_predictions = y_naive_pred_df[naive_pred_col_name].values
            if len(y_test_np) == len(y_naive_final_predictions):
                try:
                    naive_rmse = np.sqrt(mean_squared_error(y_test_np, y_naive_final_predictions))
                    naive_mae = mean_absolute_error(y_test_np, y_naive_final_predictions)
                    mask_zero_true_naive = y_test_np != 0
                    naive_mape = np.nan
                    if np.any(mask_zero_true_naive):
                        naive_mape = mean_absolute_percentage_error(y_test_np[mask_zero_true_naive], y_naive_final_predictions[mask_zero_true_naive])
                    
                    log.info(f"Naive model: RMSE={naive_rmse:.4f}, MAE={naive_mae:.4f}, MAPE={naive_mape:.4f}")
                    metrics["Naive_RMSE"] = naive_rmse
                    metrics["Naive_MAE"] = naive_mae
                    metrics["Naive_MAPE"] = naive_mape
                    if pd.notna(naive_rmse): clearml_logger.report_scalar(title="Naive Model Test Metrics", series="RMSE", value=naive_rmse, iteration=0)
                    if pd.notna(naive_mae): clearml_logger.report_scalar(title="Naive Model Test Metrics", series="MAE", value=naive_mae, iteration=0)
                    if pd.notna(naive_mape): clearml_logger.report_scalar(title="Naive Model Test Metrics", series="MAPE", value=naive_mape, iteration=0)
                except Exception as e_naive_metrics:
                    log.error(f"Error calculating naive model metrics: {e_naive_metrics}", exc_info=True)
            else:
                log.warning(f"y_test ({len(y_test_np)}) and y_naive_pred ({len(y_naive_final_predictions)}) sizes mismatch. Skipping naive comparison.")
                y_naive_final_predictions = None
    elif eval_cfg.get("compare_with_naive_model", False) and y_naive_pred_df.empty:
        log.warning("Compare with naive model enabled, but y_naive_pred_df is empty.")

    if eval_cfg.get("generate_plots", True) and len(y_test_np) > 0:
        log.info("Generating and logging plots...")
        try:
            plt.figure(figsize=tuple(eval_cfg.get("plot_figsize", [12, 6])))
            
            plot_x_axis = X_test.index if isinstance(X_test.index, pd.DatetimeIndex) else np.arange(len(y_test_np))
            
            plt.plot(plot_x_axis, y_test_np, label='Actual', color='blue', alpha=0.9, linewidth=1.5)
            plt.plot(plot_x_axis, y_pred, label=f'Predicted ({target_asset_ticker})', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            
            if y_naive_final_predictions is not None and len(y_naive_final_predictions) == len(plot_x_axis):
                plt.plot(plot_x_axis, y_naive_final_predictions, label='Naive Baseline', color='green', linestyle=':', alpha=0.6, linewidth=1)
            
            plt.title(f'Actual vs Predicted Prices for {target_asset_ticker}', fontsize=16)
            plt.xlabel(str(eval_cfg.get("plot_xlabel", 'Time Steps (Test Set)')), fontsize=12)
            plt.ylabel(str(eval_cfg.get("plot_ylabel", 'Price')), fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            clearml_logger.report_matplotlib_figure(
                title=f"Actual vs Predicted Plot - {target_asset_ticker}", 
                series="Price Forecast Evaluation", 
                figure=plt,
                iteration=0,
                report_image=True
            )
            log.info(f"Plot 'Actual vs Predicted' for {target_asset_ticker} logged to ClearML.")
        except Exception as e_plot:
            log.error(f"Error generating or logging plot: {e_plot}", exc_info=True)
        finally:
            plt.close()
    elif len(y_test_np) == 0: 
        log.warning("Skipping plot generation as y_test_np is empty.")

    log.info("Saving evaluation results (metrics)...")
    metrics_to_save = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
    metrics_filename = f"evaluation_metrics_{target_asset_ticker}.json"
    with open(metrics_filename, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    task.upload_artifact(name=f"{target_asset_ticker}_evaluation_metrics", artifact_object=metrics_filename)
    log.info(f"Evaluation results saved as artifact: {metrics_filename}")
    task.close()
    log.info(f"Evaluation task for {target_asset_ticker} completed.")

if __name__ == '__main__':
    main() 