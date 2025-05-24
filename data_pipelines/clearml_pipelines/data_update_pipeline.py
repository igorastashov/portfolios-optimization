from clearml import PipelineController, Task
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_data_update_pipeline(project_name="Data Management", pipeline_name="Scheduled Data Update Pipeline"):
    """
    Определение и запуск ClearML пайплайна обновления данных.
    """
    logging.info(f"Инициализация ClearML Pipeline: {pipeline_name} в проекте: {project_name}")
    
    pipe = PipelineController(
        name=pipeline_name,
        project=project_name,
        version="0.0.1",
        add_pipeline_tags=True,
        target_project=project_name  # Указываем, чтобы задачи пайплайна создавались в том же проекте
    )

    # Параметры пайплайна (можно переопределять при запуске)
    pipe.add_parameter(
        name="binance_symbols", 
        description="List of symbols for Binance data", 
        default=fetch_default_binance_symbols()
    )
    pipe.add_parameter(
        name="binance_interval", 
        description="Interval for Binance data", 
        default=fetch_default_binance_interval()
    )
    pipe.add_parameter(
        name="alphavantage_tickers", 
        description="List of tickers for Alpha Vantage news", 
        default=fetch_default_alphavantage_tickers()
    )
    pipe.add_parameter(
        name="news_time_from", 
        description="Start time for news (YYYYMMDDTHHMM)", 
        default=fetch_default_news_time_from()
    )
    pipe.add_parameter(
        name="binance_output_path", 
        description="Output path for Binance data", 
        default=fetch_default_binance_output_path()
    )
    pipe.add_parameter(
        name="alphavantage_output_path", 
        description="Output path for Alpha Vantage news", 
        default=fetch_default_alphavantage_output_path()
    )

    logging.info("Добавление шага: fetch_binance_data_task")
    pipe.add_step(
        name="fetch_binance_data_task",
        base_task_project=project_name,
        base_task_name="Fetch Binance Data",
    )
    logging.info("Шаг fetch_binance_data_task добавлен.")

    logging.info("Добавление шага: fetch_alphavantage_news_task")
    pipe.add_step(
        name="fetch_alphavantage_news_task",
        base_task_project=project_name,
        base_task_name="Fetch Alpha Vantage News",
        parents=["fetch_binance_data_task"]
    )
    logging.info("Шаг fetch_alphavantage_news_task добавлен.")
    
    logging.info("Пайплайн data_update_pipeline определен. Для запуска используйте pipe.start() или UI ClearML.")
    
    return {"pipeline_id": generate_pipeline_id(), "status": "defined"}


if __name__ == "__main__":
    logging.info("Запуск определения пайплайна data_update_pipeline...")
    run_data_update_pipeline()
    logging.info("Определение пайплайна data_update_pipeline завершено.") 