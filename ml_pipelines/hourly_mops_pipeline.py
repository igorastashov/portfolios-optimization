import os
import tempfile
import shutil
from datetime import datetime, timedelta

from clearml import PipelineController, Task, Dataset, Logger
from clearml.enums import TaskTypes

PROJECT_NAME = "PortfolioOptimization"
PIPELINE_NAME = "Hourly_MOps_Pipeline"
PIPELINE_VERSION = "0.1.0"
BASE_DOCKER_IMAGE = "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
PIPELINE_REQUIREMENTS_FILE = get_requirements_file_path()

ASSET_SYMBOLS = fetch_asset_symbols()
DRL_HYPERPARAMS = load_drl_hyperparameters()

# ASSET_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
# DRL_HYPERPARAMS = {
#     "learning_rate": 0.0003, 
#     "n_steps": 2048, 
#     "batch_size": 64,
#     "n_epochs": 10,
#     "gamma": 0.99,
#     "gae_lambda": 0.95,
#     "clip_range": 0.2,
#     "ent_coef": 0.0,
#     "vf_coef": 0.5,
#     "max_grad_norm": 0.5,
#     "total_timesteps": 1000,
#     "env_params": {
#         "initial_amount": 100000,
#         "transaction_cost_pct": 0.001,
#         "lookback_window_size": 50 
#     },
#     "min_data_length_for_env": 60
# }


# OUTPUT_URL_DATASETS = "s3://ws42-2:/clearml/datasets" 
# OUTPUT_URL_MODELS ="s3://ws42-2/clearml/models" 

OUTPUT_URL_DATASETS = get_dataset_output_url()
OUTPUT_URL_MODELS = get_model_output_url()


@PipelineController.component(
    return_values=['market_data_dataset_id'],
    task_type=TaskTypes.data_processing,
    # cache=True,
    # docker=BASE_DOCKER_IMAGE, 
    requirements=PIPELINE_REQUIREMENTS_FILE,
    # execution_queue="default",
)
def task_update_market_data(
    asset_symbols: list,
    dataset_project: str,
    dataset_name: str,
    output_url: str = None, 
):
    import os
    import tempfile
    import shutil
    import pandas as pd
    from clearml import Task, Dataset, Logger
    from datetime import datetime, timedelta


    current_task = Task.current_task()
    logger = current_task.get_logger()


    logger.report_text(f"Запуск обновления рыночных данных для активов: {asset_symbols}")

    temp_data_dir = tempfile.mkdtemp(prefix="market_data_")
    logger.report_text(f"Временная директория для данных: {temp_data_dir}")

    processed_files_paths = []
    
    for symbol in asset_symbols:
        try:
            logger.report_text(f"Получение рыночных данных для {symbol}...")
            from portfolios_optimization.data_loader import get_binance_client, update_single_asset_market_data
            client = get_binance_client()
            market_df = update_single_asset_market_data(client, symbol, interval='1h', lookback_days=90)

            if market_df is None or market_df.empty:
                logger.report_warning(f"Нет данных для {symbol} или получены пустые данные.")
                continue
            
            file_path = os.path.join(temp_data_dir, f"{symbol.replace('/', '_')}_market_data.csv")
            market_df.to_csv(file_path)
            processed_files_paths.append(file_path)
            
            logger.report_text(f"Данные для {symbol} успешно обработаны и сохранены в {file_path}")

        except Exception as e:
            logger.report_error(f"Ошибка при получении/обработке рыночных данных для {symbol}: {str(e)}")
    
    if not processed_files_paths:
        error_message = "Не было обработано ни одного файла с рыночными данными. Создание датасета прервано."
        logger.report_error(error_message)
        raise ValueError(error_message)

    try:
        dataset = Dataset.create(
            dataset_project=dataset_project,
            dataset_name=dataset_name,
        )
    except Exception as e:
        logger.report_error(f"Критическая ошибка при создании датасета {dataset_name}: {str(e)}")
        raise

    dataset.add_files(temp_data_dir) 
    processed_asset_names = [os.path.basename(p).split('_market_data.csv')[0] for p in processed_files_paths]
    dataset.set_tags(["market_data", "hourly_update"] + [f"asset:{name}" for name in processed_asset_names])
    dataset.set_metadata({"asset_count": len(processed_files_paths), "timestamp": datetime.now().isoformat()})

    logger.report_text("Загрузка содержимого датасета...")
    if output_url: 
        dataset.upload(output_url=output_url)
    else: 
        dataset.upload()
        
    dataset.finalize()
    logger.report_text(f"Обновление рыночных данных завершено. ID датасета: {dataset.id}")
    
    try:
        shutil.rmtree(temp_data_dir)
        logger.report_text(f"Временная директория {temp_data_dir} успешно удалена.")
    except Exception as e:
        logger.report_warning(f"Не удалось удалить временную директорию {temp_data_dir}: {str(e)}")

    return dataset.id


@PipelineController.component(
    return_values=['news_data_dataset_id'],
    task_type=TaskTypes.data_processing,
    # cache=True,
    # docker=BASE_DOCKER_IMAGE,
    requirements=PIPELINE_REQUIREMENTS_FILE,
    # execution_queue="default",
)
def task_update_news_data(
    asset_symbols: list,
    dataset_project: str,
    dataset_name: str,
    output_url: str = None,
):
    import os
    import tempfile
    import shutil
    import pandas as pd 
    from clearml import Task, Dataset, Logger
    from datetime import datetime, timedelta # Добавлен timedelta

    current_task = Task.current_task()
    logger = current_task.get_logger()

    logger.report_text(f"Запуск обновления новостных данных для активов: {asset_symbols}")

    temp_data_dir = tempfile.mkdtemp(prefix="news_data_")
    logger.report_text(f"Временная директория для новостей: {temp_data_dir}")

    processed_files_paths = []
    
    for symbol in asset_symbols:
        try:
            logger.report_text(f"Получение новостных данных для {symbol}...")
            from portfolios_optimization.news_loader import fetch_news_for_asset
            news_df = fetch_news_for_asset(symbol, lookback_hours=24) 
            

            if news_df is None or news_df.empty:
                logger.report_warning(f"Нет новостей для {symbol} или получены пустые данные.")
                continue
            
            file_path = os.path.join(temp_data_dir, f"{symbol.replace('/', '_')}_news_data.csv")
            news_df.to_csv(file_path)
            processed_files_paths.append(file_path)
            
            logger.report_text(f"Новости для {symbol} успешно обработаны и сохранены в {file_path}")

        except Exception as e:
            logger.report_error(f"Ошибка при получении/обработке новостей для {symbol}: {str(e)}")
    
    if not processed_files_paths:
        error_message = "Не было обработано ни одного файла с новостными данными. Создание датасета прервано."
        logger.report_error(error_message)
        logger.report_warning("Возвращаем None как ID датасета новостей.")
        return None 

    try:
        dataset = Dataset.create(
            dataset_project=dataset_project,
            dataset_name=dataset_name,
        )
    except Exception as e:
        logger.report_error(f"Критическая ошибка при создании датасета новостей {dataset_name}: {str(e)}")
        raise

    dataset.add_files(temp_data_dir)
    processed_asset_names = [os.path.basename(p).split('_news_data.csv')[0] for p in processed_files_paths]
    dataset.set_tags(["news_data", "raw_text", "hourly_update"] + [f"asset:{name}" for name in processed_asset_names])
    dataset.set_metadata({"asset_count": len(processed_files_paths), "timestamp": datetime.now().isoformat()})

    logger.report_text("Загрузка содержимого датасета новостей...")
    if output_url:
        dataset.upload(output_url=output_url)
    else:
        dataset.upload()
        
    dataset.finalize()
    logger.report_text(f"Обновление новостных данных завершено. ID датасета: {dataset.id}")

    try:
        shutil.rmtree(temp_data_dir)
        logger.report_text(f"Временная директория новостей {temp_data_dir} успешно удалена.")
    except Exception as e:
        logger.report_warning(f"Не удалось удалить временную директорию новостей {temp_data_dir}: {str(e)}")

    return dataset.id


@PipelineController.component(
    return_values=['trained_predictor_model_ids'],
    task_type=TaskTypes.training,
    # docker=BASE_DOCKER_IMAGE,
    requirements=PIPELINE_REQUIREMENTS_FILE,
    # execution_queue="default",
)
def task_train_price_predictor(
    asset_symbols: list,
    market_data_dataset_id: str,
    news_data_dataset_id: str,
    model_project: str,
    model_name_prefix: str, 
    output_url: str = None,
):
    import os
    import tempfile
    import shutil
    import pandas as pd
    from clearml import Task, Dataset, Model, Logger

    current_task = Task.current_task()
    logger = current_task.get_logger()
    
    trained_model_ids = {}
    
    try:
        logger.report_text(f"Загрузка датасета рыночных данных: {market_data_dataset_id}")
        market_dataset_path = Dataset.get(dataset_id=market_data_dataset_id).get_local_copy()
        logger.report_text(f"Рыночные данные загружены в: {market_dataset_path}")

        news_dataset_path = None
        if news_data_dataset_id: 
            logger.report_text(f"Загрузка датасета новостных данных: {news_data_dataset_id}")
            news_dataset_path = Dataset.get(dataset_id=news_data_dataset_id).get_local_copy()
            logger.report_text(f"Новостные данные загружены в: {news_dataset_path}")
        else:
            logger.report_warning("ID датасета новостей не предоставлен, обучение будет только на рыночных данных.")

    except Exception as e:
        logger.report_error(f"Ошибка при загрузке входных датасетов для обучения предиктора: {str(e)}")
        raise

    for symbol in asset_symbols:
        logger.report_text(f"Начало обучения предиктора цен для {symbol}")
        temp_model_dir = None 
        try:
            symbol_file_name_part = symbol.replace('/', '_')
            market_data_file = os.path.join(market_dataset_path, f"{symbol_file_name_part}_market_data.csv")
            
            if not os.path.exists(market_data_file):
                logger.report_warning(f"Файл рыночных данных {market_data_file} не найден для {symbol}. Пропуск обучения.")
                continue
            
            asset_market_df = pd.read_csv(market_data_file, parse_dates=['timestamp'], index_col='timestamp')
            
            asset_news_df = None
            if news_dataset_path:
                news_data_file = os.path.join(news_dataset_path, f"{symbol_file_name_part}_news_data.csv")
                if os.path.exists(news_data_file):
                    asset_news_df = pd.read_csv(news_data_file, parse_dates=['timestamp'], index_col='timestamp')
                else:
                    logger.report_warning(f"Файл новостных данных {news_data_file} не найден для {symbol}, хотя датасет новостей был указан.")

            from portfolios_optimization.preprocessing import create_features_for_price_model
            from portfolios_optimization.models import CatBoostPricePredictor
            from sklearn.model_selection import train_test_split
            
            features_df, target_series = create_features_for_price_model(asset_market_df, asset_news_df, config={'symbol': symbol})
            
            if features_df.empty or target_series.empty or len(features_df) < 20:
                logger.report_warning(f"Недостаточно данных для обучения предиктора для {symbol} после создания признаков. Пропуск.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(features_df, target_series, test_size=0.2, shuffle=False)
            
            # catboost_params = {
            #     'iterations': 200,
            #     'learning_rate': 0.05,
            #     'depth': 6,
            #     'loss_function': 'RMSE',
            #     'eval_metric': 'RMSE',
            #     'random_seed': 42,
            #     'verbose': 0
            # }
            catboost_params = fetch_model_hyperparameters()

            model_predictor = CatBoostPricePredictor(params=catboost_params)
            model_predictor.fit(X_train, y_train, eval_set=(X_test, y_test))
            current_task.connect(model_predictor.get_model())

            model_filename = f"{model_name_prefix}_{symbol.replace('/', '_')}.cbm"
            temp_model_dir = tempfile.mkdtemp(prefix=f"model_{symbol.replace('/', '_')}_")
            model_save_path = os.path.join(temp_model_dir, model_filename)
            
            model_predictor.save_model(model_save_path)

            output_model_clearml = Model(
                project=model_project,
                name=f"{model_name_prefix}_{symbol.replace('/', '_')}", 
            )
            output_model_clearml.connect_task(current_task, name=f"training_task_for_{symbol.replace('/', '_')}")
            output_model_clearml.set_comment(f"Price predictor model for {symbol} trained by pipeline.")
            output_model_clearml.update_weights(weights_filename=model_save_path, auto_delete_file=True, target_filename=model_filename, upload_uri=output_url)

            trained_model_ids[symbol] = output_model_clearml.id
            logger.report_text(f"Модель для {symbol} обучена и зарегистрирована. ID модели: {output_model_clearml.id}")

        except Exception as e:
            logger.report_error(f"Ошибка при обучении предиктора для {symbol}: {str(e)}")
        finally:
            if temp_model_dir and os.path.exists(temp_model_dir):
                 try:
                    shutil.rmtree(temp_model_dir) 
                 except Exception as e_rm:
                    logger.report_warning(f"Не удалось удалить временную директорию модели {temp_model_dir}: {str(e_rm)}")
    
    if not trained_model_ids:
        logger.report_warning("Ни одна модель предиктора цен не была обучена. Возвращается пустой словарь.")

    return trained_model_ids


@PipelineController.component(
    return_values=['drl_features_dataset_id'], 
    task_type=TaskTypes.data_processing,
    # docker=BASE_DOCKER_IMAGE,
    requirements=PIPELINE_REQUIREMENTS_FILE,
    # execution_queue="default",
)
def task_prepare_drl_features(
    asset_symbols: list,
    market_data_dataset_id: str,
    news_data_dataset_id: str,
    price_predictor_model_ids: dict, 
    dataset_project: str,
    dataset_name: str, 
    output_url: str = None,
):
    import os
    import tempfile
    import shutil
    import pandas as pd
    import numpy as np
    from clearml import Task, Dataset, Model, Logger
    from datetime import datetime

    current_task = Task.current_task()
    logger = current_task.get_logger()

    logger.report_text(f"Начало подготовки признаков для DRL агента. Используемые модели предикторов: {price_predictor_model_ids}")

    try:
        market_dataset_path = Dataset.get(dataset_id=market_data_dataset_id).get_local_copy()
        news_dataset_path = None
        if news_data_dataset_id:
            news_dataset_path = Dataset.get(dataset_id=news_data_dataset_id).get_local_copy()
    except Exception as e:
        logger.report_error(f"Ошибка при загрузке датасетов для DRL признаков: {str(e)}")
        raise

    all_features_list = [] 

    for symbol in asset_symbols:
        logger.report_text(f"Подготовка DRL признаков для {symbol}...")
        try:
            symbol_file_name_part = symbol.replace('/', '_')
            market_data_file = os.path.join(market_dataset_path, f"{symbol_file_name_part}_market_data.csv")
            if not os.path.exists(market_data_file):
                logger.report_warning(f"Файл рыночных данных {market_data_file} не найден. Пропуск.")
                continue
            asset_market_df = pd.read_csv(market_data_file, parse_dates=['timestamp'], index_col='timestamp')
            if asset_market_df.empty:
                logger.report_warning(f"Рыночные данные для {symbol} пусты. Пропуск.")
                continue

            from portfolios_optimization.portfolio_analyzer import FeatureEngineer
            from portfolios_optimization.models import load_clearml_model_predictor

            # 1. Загрузка модели предиктора и генерация прогнозов
            price_predictions_series = pd.Series(dtype=float, index=asset_market_df.index, name=f'{symbol_file_name_part}_price_prediction')
            if symbol in price_predictor_model_ids and price_predictor_model_ids[symbol]:
                model_id = price_predictor_model_ids[symbol]
                logger.report_text(f"Загрузка предиктора цен для {symbol} (ID: {model_id})")
                try:
                    predictor, model_details = load_clearml_model_predictor(model_id, logger) # MODIF   IED
                    if predictor:
                        features_for_pred_df, _ = create_features_for_price_model(asset_market_df.copy(), None, config={'symbol': symbol, 'is_prediction': True})
                        features_for_pred_df, _ = create_features_for_price_model(asset_market_df.copy(), None, config={'symbol': symbol, 'is_prediction': True})
                        
                        
                        if not features_for_pred_df.empty:
                            raw_predictions = predictor.predict(features_for_pred_df)
                            price_predictions_series = pd.Series(raw_predictions, index=features_for_pred_df.index, name=price_predictions_series.name)
                            price_predictions_series = pd.Series(raw_predictions, index=features_for_pred_df.index, name=price_predictions_series.name)
                            price_predictions_series = price_predictions_series.reindex(asset_market_df.index).fillna(method='ffill').fillna(method='bfill')
                        else:
                            logger.report_warning(f"Нет данных для предсказания для {symbol} после создания признаков.")
                    else:
                        logger.report_warning(f"Предиктор для {symbol} (ID: {model_id}) не был загружен.")

                except Exception as e_load_pred_model:
                    logger.report_warning(f"Не удалось загрузить/использовать предиктор для {symbol} (ID: {model_id}): {str(e_load_pred_model)}")
            else:
                 logger.report_warning(f"Нет ID предиктора для {symbol} или он некорректен. Прогнозы не будут добавлены.")

            # fe = FeatureEngineer(
            #     indicators=['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS'],
            #     window_sizes=[5, 10, 14, 20, 50]
            # )
            fe = FeatureEngineer(
                indicators=get_technical_indicators(),
                window_sizes=get_window_sizes()
            )
            
            tech_indicators_df = fe.preprocess_data(asset_market_df.copy())
            tech_indicators_df = tech_indicators_df.add_prefix(f'{symbol_file_name_part}_')

            news_features_df = pd.DataFrame(index=asset_market_df.index)
            if news_dataset_path:
                news_data_file = os.path.join(news_dataset_path, f"{symbol_file_name_part}_news_data.csv")
                if os.path.exists(news_data_file):
                    asset_news_df = pd.read_csv(news_data_file, parse_dates=['timestamp'], index_col='timestamp')
                    if not asset_news_df.empty and 'sentiment_score' in asset_news_df.columns:
                        news_features_df[f'{symbol_file_name_part}_news_sentiment'] = asset_news_df['sentiment_score']
                        news_features_df = news_features_df.reindex(asset_market_df.index, method='ffill').fillna(0)
                else: 
                    logger.warning(f"Файл новостей {news_data_file} не найден для {symbol}.")
            

            asset_market_df_prefixed = asset_market_df.add_prefix(f'{symbol_file_name_part}_')
            current_asset_all_features = pd.concat(
                [asset_market_df_prefixed, price_predictions_series, tech_indicators_df, news_features_df],
                axis=1
            )
            all_features_list.append(current_asset_all_features)

        except Exception as e:
            logger.report_error(f"Ошибка при подготовке DRL признаков для {symbol}: {str(e)}")

    if not all_features_list:
        error_message = "Не удалось подготовить признаки ни для одного актива для DRL."
        logger.report_error(error_message)
        raise ValueError(error_message)

    final_drl_features_df = pd.concat(all_features_list, axis=1)
    final_drl_features_df.sort_index(inplace=True) 
    
    final_drl_features_df.fillna(method='ffill', inplace=True)
    final_drl_features_df.fillna(method='bfill', inplace=True)
    final_drl_features_df.fillna(0, inplace=True)
    final_drl_features_df.dropna(how='all', inplace=True) 

    if final_drl_features_df.empty:
        error_message = "Итоговый DataFrame с признаками для DRL пуст после обработки."
        logger.report_error(error_message)
        raise ValueError(error_message)

    logger.report_text(f"Финальный DataFrame с DRL признаками имеет размер: {final_drl_features_df.shape}")

    temp_features_dir = tempfile.mkdtemp(prefix="drl_features_")
    features_filename = "drl_features_combined.csv" 
    features_filepath = os.path.join(temp_features_dir, features_filename)
    
    final_drl_features_df.to_csv(features_filepath)
    logger.report_text(f"Объединенные DRL признаки сохранены в: {features_filepath}")

    output_dataset_id = None
    try:
        drl_dataset = Dataset.create(
            dataset_project=dataset_project,
            dataset_name=dataset_name, 
        )
        drl_dataset.add_files(path=features_filepath, local_path=features_filename) 
        drl_dataset.set_tags(["drl_features", "combined", "hourly_update"])
        drl_dataset.set_metadata({
            "shape": list(final_drl_features_df.shape), 
            "assets_included": asset_symbols,
            "timestamp": datetime.now().isoformat(),
            "columns_preview": list(final_drl_features_df.columns[:20])
        })

        if output_url:
            drl_dataset.upload(output_url=output_url)
        else:
            drl_dataset.upload()
        
        drl_dataset.finalize()
        logger.report_text(f"DRL признаки сохранены в ClearML Dataset. ID: {drl_dataset.id}")
        output_dataset_id = drl_dataset.id
    
    except Exception as e_ds:
        logger.report_error(f"Ошибка при создании ClearML Dataset для DRL признаков: {str(e_ds)}")
        raise 
    finally:
        try:
            shutil.rmtree(temp_features_dir)
        except Exception as e_rm:
            logger.report_warning(f"Не удалось удалить временную директорию DRL признаков {temp_features_dir}: {str(e_rm)}")
            
    if output_dataset_id is None:
        raise ValueError("ID датасета DRL признаков не был установлен после попытки создания.")
        
    return output_dataset_id 


@PipelineController.component(
    return_values=['drl_model_id'],
    task_type=TaskTypes.training,
    auto_connect_frameworks={'tensorboard': True, 'matplotlib': True, 'plotly': True, 'sklearn': True, 'pytorch': True, 'joblib': True}, 
    requirements=PIPELINE_REQUIREMENTS_FILE, 
    # execution_queue="gpu_queue", 
)
def task_train_drl_agent(
    drl_features_dataset_id: str, 
    asset_symbols: list, 
    drl_hyperparams: dict, 
    model_project: str,
    model_name: str, 
    output_url: str = None, 
):
    import os
    import tempfile
    import shutil
    import pandas as pd
    import numpy as np 
    from clearml import Task, Dataset, Model, Logger

    current_task = Task.current_task()
    logger = current_task.get_logger()

    logger.report_text(f"Начало обучения DRL агента. Используемые гиперпараметры: {drl_hyperparams}")
    current_task.connect(drl_hyperparams, name='DRL Hyperparameters') 

    try:
        logger.report_text(f"Загрузка датасета DRL признаков: {drl_features_dataset_id}")
        drl_features_dataset_obj = Dataset.get(dataset_id=drl_features_dataset_id)
        drl_features_local_path = drl_features_dataset_obj.get_local_copy()
        features_file_actual_path = os.path.join(drl_features_local_path, "drl_features_combined.csv")
        features_file_actual_path = os.path.join(drl_features_local_path, "drl_features_combined.csv")
        features_file_actual_path = os.path.join(drl_features_local_path, "drl_features_combined.csv")
        
        if not os.path.exists(features_file_actual_path):
            found_files = [f for f in os.listdir(drl_features_local_path) if f.endswith('.csv')]
            if found_files:
                features_file_actual_path = os.path.join(drl_features_local_path, found_files[0])
                logger.report_warning(f"Файл DRL признаков не найден по ожидаемому пути, используется найденный: {features_file_actual_path}")
            else:
                error_msg = f"Файл с DRL признаками не найден в локальной копии датасета: {drl_features_local_path}"
                logger.report_error(error_msg)
                raise FileNotFoundError(error_msg)
            
        drl_features_df = pd.read_csv(features_file_actual_path, parse_dates=['timestamp'], index_col='timestamp')
        logger.report_text(f"DRL признаки успешно загружены. Размер: {drl_features_df.shape}")

        min_len = drl_hyperparams.get("min_data_length_for_env", 50)
        if drl_features_df.empty or len(drl_features_df) < min_len:
            error_msg = f"DataFrame DRL ({len(drl_features_df)} строк) слишком мал (нужно >= {min_len}). Обучение прервано."
            logger.report_error(error_msg)
            raise ValueError(error_msg)

    except Exception as e:
        logger.report_error(f"Критическая ошибка при загрузке DRL признаков: {str(e)}")
        raise

    final_drl_model_id = None 
    temp_drl_model_dir = None 
    tensorboard_log_dir_abs = None

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO 
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.env_checker import check_env 
        from portfolios_optimization.portfolio_analyzer import StockPortfolioEnv 
        
        
        class DummyStockPortfolioEnv(gym.Env):
            metadata = {'render_modes': ['human'], 'render_fps': 4}

            def __init__(self, df, assets, initial_amount=1e6, lookback_window_size=50, transaction_cost_pct=0.001, render_mode=None, **kwargs):
                super(DummyStockPortfolioEnv, self).__init__()
                self.df = df.copy()
                self.assets = assets
                self.num_assets = len(assets)
                self.initial_amount = initial_amount
                self.lookback_window_size = lookback_window_size
                self.transaction_cost_pct = transaction_cost_pct
                self.render_mode = render_mode

                if self.df.empty or len(self.df) < self.lookback_window_size + 1:
                    raise ValueError(f"DataFrame (len {len(self.df)}) слишком мал для lookback_window_size {self.lookback_window_size}")

                self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_assets + 1,), dtype=np.float32)
                num_df_features = self.df.shape[1]
                obs_shape_flat = self.lookback_window_size * num_df_features + (self.num_assets + 1)
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape_flat,), dtype=np.float32)

                self.current_step_in_episode = 0
                self.start_tick = self.lookback_window_size - 1
                self.end_tick = len(self.df) - 1
                self.current_tick = self.start_tick

            def _get_observation(self):
                return np.random.rand(self.observation_space.shape[0]).astype(np.float32)

            def _get_info(self):
                return {
                    "current_tick": self.current_tick,
                    "current_step_in_episode": self.current_step_in_episode
                }

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.current_tick = self.start_tick
                self.current_step_in_episode = 0
                observation = self._get_observation()
                info = self._get_info()
                return observation, info

            def step(self, action):
                self.current_tick += 1
                self.current_step_in_episode += 1

                reward = float(np.random.rand() - 0.45)
                terminated = self.current_tick >= self.end_tick
                truncated = False
                observation = self._get_observation()
                info = self._get_info()

                return observation, reward, terminated, truncated, info
        
        env_params = drl_hyperparams.get('env_params', {})
        env_kwargs = {'df': drl_features_df, 'assets': asset_symbols, **env_params}
        
        vec_env = DummyVecEnv([lambda: StockPortfolioEnv(**env_kwargs)]) 
        
        try: 
           check_env(vec_env.envs[0]) 
           logger.report_text("Проверка окружения (check_env) прошла успешно.") 
        except Exception as e_check_env: 
           logger.report_error(f"Ошибка check_env: {str(e_check_env)}", exc_info=True) 
           raise 

        model_specific_params = { 
            k: v for k, v in drl_hyperparams.items() 
            if k not in ['env_params', 'total_timesteps', 'min_data_length_for_env']
        }
        
        tensorboard_log_dir = "tensorboard_logs_drl"
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard_log_dir_abs = os.path.abspath(tensorboard_log_dir)
        logger.report_text(f"Логи TensorBoard будут сохраняться в: {tensorboard_log_dir_abs}")
        
        current_task.connect_configuration({"tensorboard_log_dir_abs": tensorboard_log_dir_abs}, name="Tensorboard Logging Dir")

        model = PPO(
            "MlpPolicy", 
            vec_env,
            verbose=model_specific_params.pop("verbose", 1),
            tensorboard_log=tensorboard_log_dir_abs, 
            **model_specific_params 
        )
        
        total_timesteps_to_learn = drl_hyperparams.get("total_timesteps", 1000) 
        logger.report_text(f"Начало обучения DRL модели (PPO заглушка) на {total_timesteps_to_learn} шагов...")
        
        model.learn(total_timesteps=total_timesteps_to_learn)
        logger.report_text("Обучение DRL модели (заглушка) завершено.")

        drl_model_filename = f"{model_name.replace(' ', '_').replace('/','-')}.zip" 
        temp_drl_model_dir = tempfile.mkdtemp(prefix="drl_model_final_")
        drl_model_save_path = os.path.join(temp_drl_model_dir, drl_model_filename)
        
        model.save(drl_model_save_path)
        logger.report_text(f"Модель DRL (PPO заглушка) сохранена в: {drl_model_save_path}")

        output_drl_model_clearml = Model(
            project=model_project,
            name=model_name,
        )
        output_drl_model_clearml.connect_task(current_task, name="drl_training_task_details")
        output_drl_model_clearml.set_comment("DRL portfolio optimization agent trained by pipeline.")
        output_drl_model_clearml.update_weights(
            weights_filename=drl_model_save_path, 
            auto_delete_file=True, 
            target_filename=drl_model_filename, 
            upload_uri=output_url
        )
        final_drl_model_id = output_drl_model_clearml.id
        logger.report_text(f"DRL агент обучен и зарегистрирован. ID модели: {final_drl_model_id}")

    except ImportError as e_import:
        logger.report_error(f"ImportError: {str(e_import)}. Убедитесь, что gymnasium и stable-baselines3 установлены.")
        raise
    except Exception as e:
        logger.report_error(f"Ошибка при обучении DRL агента: {str(e)}", exc_info=True)
        raise
    finally:
        if temp_drl_model_dir and os.path.exists(temp_drl_model_dir):
            try: shutil.rmtree(temp_drl_model_dir) 
            except: logger.report_warning(f"Не удалось удалить {temp_drl_model_dir}")
        if tensorboard_log_dir_abs and os.path.exists(tensorboard_log_dir_abs): 
            try: shutil.rmtree(tensorboard_log_dir_abs)
            except: logger.report_warning(f"Не удалось удалить {tensorboard_log_dir_abs}")

    if final_drl_model_id is None:
        raise ValueError("ID обученной DRL модели не был установлен после попытки обучения.")
        
    return final_drl_model_id


def run_mops_pipeline():
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PROJECT_NAME,
        version=PIPELINE_VERSION,
        add_pipeline_tags=True, 
        auto_version_pipeline=True, 
        pipeline_execution_queue="default" 
    )

    pipe.add_parameter('param_asset_symbols_list', ASSET_SYMBOLS, description='Список символов активов для обработки')
    pipe.add_parameter('param_drl_hyperparams', DRL_HYPERPARAMS, description='Гиперпараметры для DRL агента')
    pipe.add_parameter('param_output_url_datasets', OUTPUT_URL_DATASETS, description='URL для датасетов (S3/MinIO/etc)')
    pipe.add_parameter('param_output_url_models', OUTPUT_URL_MODELS, description='URL для моделей (S3/MinIO/etc)')

    pipeline_params = pipe.get_parameters() 

    market_data_task_step = pipe.add_function_step(
        name='UpdateMarketData',
        function=task_update_market_data,
        function_kwargs=dict(
            asset_symbols=pipeline_params['param_asset_symbols_list'],
            dataset_project=PROJECT_NAME + "/Datasets/MarketData", 
            dataset_name="MarketData_Hourly",
            output_url=pipeline_params['param_output_url_datasets']
        ),
        task_name='Step_1_UpdateMarketData', 
    )

    news_data_task_step = pipe.add_function_step(
        name='UpdateNewsData',
        function=task_update_news_data,
        function_kwargs=dict(
            asset_symbols=pipeline_params['param_asset_symbols_list'],
            dataset_project=PROJECT_NAME + "/Datasets/NewsData", 
            dataset_name="NewsData_Raw_Hourly",
            output_url=pipeline_params['param_output_url_datasets']
        ),
        task_name='Step_2_UpdateNewsData',
    )
    
    train_predictors_task_step = pipe.add_function_step(
        name='TrainPricePredictors',
        function=task_train_price_predictor,
        function_kwargs=dict(
            asset_symbols=pipeline_params['param_asset_symbols_list'],
            market_data_dataset_id=market_data_task_step.get_return_values()['market_data_dataset_id'],
            news_data_dataset_id=news_data_task_step.get_return_values()['news_data_dataset_id'],
            model_project=PROJECT_NAME + "/Models/PricePredictors", 
            model_name_prefix="PricePredictor",
            output_url=pipeline_params['param_output_url_models']
        ),
        task_name='Step_3_TrainPricePredictors',
        parents=[market_data_task_step.id, news_data_task_step.id] 
    )

    prepare_drl_features_task_step = pipe.add_function_step(
        name='PrepareDRLFeatures',
        function=task_prepare_drl_features,
        function_kwargs=dict(
            asset_symbols=pipeline_params['param_asset_symbols_list'],
            market_data_dataset_id=market_data_task_step.get_return_values()['market_data_dataset_id'],
            news_data_dataset_id=news_data_task_step.get_return_values()['news_data_dataset_id'],
            price_predictor_model_ids=train_predictors_task_step.get_return_values()['trained_predictor_model_ids'],
            dataset_project=PROJECT_NAME + "/Datasets/DRLFeatures",
            dataset_name="DRL_Features_Hourly",
            output_url=pipeline_params['param_output_url_datasets']
        ),
        task_name='Step_4_PrepareDRLFeatures',
        parents=[train_predictors_task_step.id] 
    )

    train_drl_agent_task_step = pipe.add_function_step(
        name='TrainDRLAgent',
        function=task_train_drl_agent,
        function_kwargs=dict(
            drl_features_dataset_id=prepare_drl_features_task_step.get_return_values()['drl_features_dataset_id'],
            asset_symbols=pipeline_params['param_asset_symbols_list'],
            drl_hyperparams=pipeline_params['param_drl_hyperparams'],
            model_project=PROJECT_NAME + "/Models/DRLAgents",
            model_name="PortfolioDRLAgent_PPO_Hourly", 
            output_url=pipeline_params['param_output_url_models']
        ),
        task_name='Step_5_TrainDRLAgent',
        parents=[prepare_drl_features_task_step.id] 
    )
    
    pipe.upload(tags=["mops", "portfolio_optimization", "hourly_pipeline"])
    print(f"Пайплайн '{PIPELINE_NAME}' (версия '{pipe.version}') успешно зарегистрирован/обновлен в проекте '{PROJECT_NAME}'.")

if __name__ == '__main__':
    print(f"Запуск скрипта регистрации/обновления пайплайна '{PIPELINE_NAME}'...")
    run_mops_pipeline()
    print("Скрипт регистрации пайплайна завершен.") 
    