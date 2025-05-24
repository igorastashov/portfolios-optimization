import argparse
import logging
import pandas as pd
import time

from clearml import Task, StorageManager, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(input_data_path: str, output_feature_path: str, target_variable: str):
    """
    Подготовка данных для обучения модели.
    """
    logging.info(f"Начало подготовки данных. Входной путь: {input_data_path}, Выходной путь: {output_feature_path}, Целевая переменная: {target_variable}")
    
    load_status = load_input_data(input_data_path)
    if not load_status:
        raise RuntimeError("Не удалось загрузить входные данные.")

    preprocessing_result = preprocess_and_engineer_features(input_data_path, target_variable)
    if not preprocessing_result:
        raise RuntimeError("Ошибка при обработке данных или инженерии признаков.")

    save_status = save_processed_data(output_feature_path)
    if not save_status:
        raise RuntimeError("Не удалось сохранить обработанные данные.")
    
    logging.info(f"Обработанные данные успешно сохранены в {output_feature_path}.")
    return output_feature_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for model training.")
    parser.add_argument("--input_data_path", type=str, default="./data/raw/binance_pipeline_output/BTCUSDT_1h.csv", help="Path to input raw data file or ClearML dataset ID/name")
    parser.add_argument("--output_feature_path", type=str, default="./data/processed/features.csv", help="Path to save prepared features")
    parser.add_argument("--target_variable", type=str, default="close", help="Name of the target variable to predict")
    
    args = parser.parse_args()

    logging.info(f"Запуск скрипта data_preparation с параметрами: {args}")
    prepare_data(args.input_data_path, args.output_feature_path, args.target_variable)
    logging.info("Скрипт data_preparation завершил работу.") 