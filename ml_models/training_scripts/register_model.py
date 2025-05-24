import argparse
import logging
import time
from clearml import Task, Model, InputModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def register_model(model_path: str, model_name: str, project_name: str, evaluation_report_path: str, metrics: dict):
    """
    Регистрация модели в системе реестра моделей.
    """
    logging.info(f"Начало регистрации модели: {model_name} в проекте: {project_name}")
    
    validate_inputs(model_path, model_name, project_name, evaluation_report_path, metrics)
    
    registration_result = perform_model_registration(
        model_path=model_path,
        model_name=model_name,
        project_name=project_name,
        metrics=metrics
    )
    
    if not registration_result:
        raise RuntimeError("Ошибка при регистрации модели.")
    
    attach_artifacts(registration_result["model_id"], evaluation_report_path)
    
    logging.info(f"Модель '{model_name}' успешно зарегистрирована с ID: {registration_result['model_id']}.")
    return registration_result["model_id"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a trained model in ClearML Model Registry.")
    parser.add_argument("--model_path", type=str, default="./models/price_predictor.cbm", help="Path to the trained model file (artifact from training)")
    parser.add_argument("--model_name", type=str, default="PricePredictorCatBoost", help="Name for the model in the registry")
    parser.add_argument("--project_name", type=str, default="Production Models", help="ClearML project for model registry")
    parser.add_argument("--evaluation_report_path", type=str, default="./reports/evaluation_report.txt", help="Path to evaluation report (artifact from evaluation)")
    parser.add_argument("--metrics_json", type=str, default='{"mse": 0.123, "r2": 0.789}', help='JSON string of evaluation metrics')

    args = parser.parse_args()

    import json
    metrics = json.loads(args.metrics_json)

    logging.info(f"Запуск скрипта register_model с параметрами: {args}")
    register_model(args.model_path, args.model_name, args.project_name, args.evaluation_report_path, metrics)
    logging.info("Скрипт register_model завершил работу.") 