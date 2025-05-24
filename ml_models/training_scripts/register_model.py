import argparse
import logging
import time
# from clearml import Task, Model, InputModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def register_model(model_path: str, model_name: str, project_name: str, evaluation_report_path: str, metrics: dict):
    """
    Заглушка для этапа регистрации модели в ClearML Model Registry.
    """
    logging.info(f"Начало регистрации модели. Модель: {model_path}, Имя: {model_name}, Проект: {project_name}")
    logging.info(f"Отчет об оценке: {evaluation_report_path}, Метрики: {metrics}")

    # Имитация регистрации модели
    # task = Task.current_task()
    # 
    # # Получаем InputModel из предыдущего шага (если модель была выходом той задачи)
    # # или просто регистрируем по пути, если модель была создана в текущей задаче.
    # # input_model = InputModel.get(task_id=task.get_task_id(task_name="Model Training")) 
    # # model_uri = input_model.get_weights() # или model_path, если передается напрямую

    # registered_model = Model.create(
    #     project_name=project_name,
    #     model_name=model_name,
    #     tags=["price_prediction", "catboost"],
    #     comment=f"Registered from pipeline. Evaluation metrics: {metrics}"
    # )
    # registered_model.update_weights(weights_filename=model_path) # Загружаем артефакт модели
    # registered_model.upload_artifact(name="evaluation_summary", artifact_object=evaluation_report_path)
    # registered_model.set_metadata("evaluation_metrics", metrics)
    # registered_model.publish() # Публикуем модель, делая ее доступной
    
    time.sleep(2)
    model_id = "mock_model_registry_id_abc123"
    logging.info(f"Модель '{model_name}' зарегистрирована в проекте '{project_name}' с ID: {model_id} (имитация).")
    
    return model_id

if __name__ == "__main__":
    # task = Task.init(project_name="Model Training Pipelines", task_name="Model Registration", task_type=Task.TaskTypes.deploy)
    parser = argparse.ArgumentParser(description="Register a trained model in ClearML Model Registry.")
    parser.add_argument("--model_path", type=str, default="./models/price_predictor.cbm", help="Path to the trained model file (artifact from training)")
    parser.add_argument("--model_name", type=str, default="PricePredictorCatBoost", help="Name for the model in the registry")
    parser.add_argument("--project_name", type=str, default="Production Models", help="ClearML project for model registry")
    parser.add_argument("--evaluation_report_path", type=str, default="./reports/evaluation_report.txt", help="Path to evaluation report (artifact from evaluation)")
    parser.add_argument("--metrics_json", type=str, default='{"mse": 0.123, "r2": 0.789}', help='JSON string of evaluation metrics')

    args = parser.parse_args()
    # import json
    # metrics = json.loads(args.metrics_json)

    # clearml_task_params = task.get_parameters_as_dict()
    # args.model_path = clearml_task_params.get('Args/model_path', args.model_path)
    # args.evaluation_report_path = clearml_task_params.get('Args/evaluation_report_path', args.evaluation_report_path)
    # metrics_from_clearml = clearml_task_params.get('Args/metrics', {})
    # if metrics_from_clearml: metrics = metrics_from_clearml
    
    import json # для metrics
    metrics = json.loads(args.metrics_json)

    logging.info(f"Запуск скрипта register_model с параметрами: {args}")
    register_model(args.model_path, args.model_name, args.project_name, args.evaluation_report_path, metrics)
    logging.info("Скрипт register_model завершил работу.") 