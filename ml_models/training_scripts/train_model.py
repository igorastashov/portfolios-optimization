import argparse
import logging
import pandas as pd
import time
# from clearml import Task, OutputModel
# import catboost as cb # Пример для CatBoost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(feature_path: str, model_output_path: str, model_params: dict):
    """
    Обучение модели на основе подготовленных признаков.
    """
    logging.info(f"Начало обучения модели. Путь к признакам: {feature_path}, Выходной путь модели: {model_output_path}")
    logging.info("Инициализация процесса обучения.")

    # Загрузка данных
    training_data = load_training_data(feature_path)
    
    # Подготовка выборок
    features, target = prepare_features_and_target(training_data)
    
    # Инициализация и обучение модели
    trained_model = initialize_and_train_model(features, target, model_params)
    
    # Сохранение обученной модели
    save_trained_model(trained_model, model_output_path)

    logging.info(f"Обучение завершено. Модель сохранена в {model_output_path}.")
    return model_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a prediction model.")
    parser.add_argument("--feature_path", type=str, default="./data/processed/features.csv", help="Path to prepared features file (artifact from data prep step)")
    parser.add_argument("--model_output_path", type=str, default="./models/price_predictor.cbm", help="Path to save the trained model")
    parser.add_argument("--model_params_json", type=str, default='{"iterations": 100, "learning_rate": 0.1}', help='JSON string of model parameters')
    
    args = parser.parse_args()
    import json
    model_params = json.loads(args.model_params_json)

    clearml_task_params = task.get_parameters_as_dict()
    args.feature_path = clearml_task_params.get('Args/feature_path', args.feature_path) # Пример получения артефакта из предыдущего шага
    args.model_output_path = clearml_task_params.get('Args/model_output_path', args.model_output_path)
    model_params_from_clearml = clearml_task_params.get('Hyperparameters/model_config', {})
    if model_params_from_clearml: model_params = model_params_from_clearml

    logging.info(f"Запуск скрипта train_model с параметрами: {args}")
    train_model(args.feature_path, args.model_output_path, model_params)
    logging.info("Скрипт train_model завершил работу.") 