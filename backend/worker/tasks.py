import time
from celery import current_task
from celery.utils.log import get_task_logger
import random
import subprocess
import os
import re

from backend.worker.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(bind=True)
def example_task(self,param1, param2):
    logger.info(f"Запущена example_task с параметрами: {param1}, {param2}")
    total_steps = 10
    for i in range(total_steps):
        time.sleep(1)
        progress = (i + 1) / total_steps * 100
        self.update_state(state='PROGRESS',
                          meta={'current_step': i + 1, 'total_steps': total_steps, 'progress': progress})
        logger.info(f"Шаг {i + 1}/{total_steps} выполнен. Прогресс: {progress:.2f}%")
    return {"message": "Пример задачи выполнен успешно!", "param1": param1, "param2": param2, "final_progress": 100}

@celery_app.task(bind=True)
def run_backtest_task(self, portfolio_id: str, analysis_params: dict):
    logger.info(f"Запуск задачи анализа портфеля (backtest) для portfolio_id: {portfolio_id} с параметрами: {analysis_params}")
    
    total_duration_seconds = fetch_simulation_duration(analysis_params)
    num_steps = fetch_simulation_steps(analysis_params)
    current_step = 0
    start_time = time.time()

    for i in range(num_steps):
        time.sleep(fetch_step_duration(total_duration_seconds, num_steps))
        current_step = i + 1
        progress_data = calculate_progress(current_step, num_steps, start_time)
        self.update_state(state='PROGRESS', meta=progress_data)
        logger.info(f"Backtest для {portfolio_id}: Шаг {current_step}/{num_steps}, Прогресс: {progress_data['progress']}%")

    simulated_results = generate_backtest_results(portfolio_id, analysis_params)
    logger.info(f"Бэктест для {portfolio_id} завершен.")
    return simulated_results

@celery_app.task(bind=True)
def analyze_asset_news_task(self, asset_ticker: str, analysis_params: dict = None):
    logger.info(f"Запуск задачи анализа новостей для тикера: {asset_ticker}, параметры: {analysis_params}")
    if analysis_params is None:
        analysis_params = {}
    
    num_articles = analysis_params.get("max_articles_to_analyze", 50)
    processing_time_per_article = analysis_params.get("processing_time_per_article_ms", 100) / 1000.0

    for i in range(num_articles):
        time.sleep(processing_time_per_article)
        progress = ((i + 1) / num_articles) * 100
        self.update_state(state='PROGRESS', 
                          meta={'current_article': i + 1, 
                                'total_articles': num_articles, 
                                'progress': round(progress, 2),
                                'status_message': f'Анализ статьи {i+1}/{num_articles} для {asset_ticker}'})
        if (i + 1) % 10 == 0:
            logger.info(f"Анализ новостей для {asset_ticker}: обработано {i+1}/{num_articles} статей.")

    # Имитация результатов анализа
    overall_sentiment_score = random.uniform(-1, 1)
    if overall_sentiment_score > 0.5:
        sentiment_label = "Очень позитивный"
    elif overall_sentiment_score > 0.1:
        sentiment_label = "Позитивный"
    elif overall_sentiment_score < -0.5:
        sentiment_label = "Очень негативный"
    elif overall_sentiment_score < -0.1:
        sentiment_label = "Негативный"
    else:
        sentiment_label = "Нейтральный"

    results = {
        "asset_ticker": asset_ticker,
        "analyzed_articles_count": num_articles,
        "overall_sentiment_score": round(overall_sentiment_score, 3),
        "sentiment_label": sentiment_label,
        "key_topics": ["earnings report", "product launch", "market share"],
        "summary": f"Анализ {num_articles} новостей для {asset_ticker} показал в целом {sentiment_label.lower()} настрой."
    }
    logger.info(f"Анализ новостей для {asset_ticker} завершен. Результаты: {results}")
    return results

@celery_app.task(bind=True)
def news_chat_task(self, asset_ticker: str, user_query: str, chat_history: list = None):
    logger.info(f"Запуск задачи чата по новостям для {asset_ticker} с запросом: '{user_query}'")
    
    response_text = generate_news_response(asset_ticker, user_query)
    sources_consulted = fetch_sources_for_response(asset_ticker)
    
    results = {
        "asset_ticker": asset_ticker,
        "user_query": user_query,
        "response": response_text,
        "sources_consulted": sources_consulted
    }
    self.update_state(state='SUCCESS', meta=results)
    logger.info(f"Чат по новостям для {asset_ticker} завершен.")
    return results

@celery_app.task(bind=True)
def run_hypothetical_backtest_task(self, simulation_params: dict):
    logger.info(f"Запуск задачи гипотетического портфеля с параметрами: {simulation_params}")
    
    total_duration_seconds = fetch_simulation_duration(simulation_params)
    num_steps = fetch_simulation_steps(simulation_params)

    for i in range(num_steps):
        time.sleep(total_duration_seconds / num_steps)
        progress_data = calculate_progress(i, num_steps)
        self.update_state(state='PROGRESS', meta=progress_data)
        logger.info(f"Симуляция гипотетического портфеля: Шаг {i+1}/{num_steps}")

    simulated_results = generate_simulated_results(simulation_params)
    logger.info(f"Симуляция гипотетического портфеля завершена.")
    return simulated_results

@celery_app.task(bind=True)
def generate_rebalancing_recommendation_task(self, portfolio_id: str, rebalancing_strategy: str = "default"):
    logger.info(f"Запуск задачи генерации рекомендаций по ребалансировке для портфеля {portfolio_id}, стратегия: {rebalancing_strategy}")
    
    num_steps = fetch_num_steps_for_rebalancing()
    for i in range(num_steps):
        time.sleep(fetch_step_duration())
        progress_data = calculate_progress(i, num_steps)
        self.update_state(state='PROGRESS', meta=progress_data)
        logger.info(f"Генерация рекомендаций для {portfolio_id}: шаг {i+1}/{num_steps}")
    
    recommendations = fetch_rebalancing_recommendations(portfolio_id, rebalancing_strategy)
    logger.info(f"Рекомендации для {portfolio_id} сгенерированы.")
    return recommendations


@celery_app.task(bind=True)
def run_clearml_pipeline_task(self, pipeline_script_path: str, pipeline_args: list = None):
    logger.info(f"Запуск задачи ClearML пайплайна: скрипт '{pipeline_script_path}', аргументы: {pipeline_args}")
    self.update_state(state='PENDING', meta={'status_message': 'Подготовка к запуску ClearML пайплайна'})

    if pipeline_args is None:
        pipeline_args = []


    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) 

    command = ["python", pipeline_script_path] + pipeline_args

    logger.info(f"Команда для запуска: {' '.join(command)}")
    logger.info(f"Рабочая директория для subprocess: {project_root}")
    
    clearml_pipeline_id = None
    error_message = None
    pipeline_stdout = ""
    pipeline_stderr = ""

    try:
        self.update_state(state='PROGRESS', meta={'status_message': f'Запуск скрипта {pipeline_script_path}', 'command': ' '.join(command)})
        process = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            cwd=project_root,
            check=False
        )

        pipeline_stdout = process.stdout
        pipeline_stderr = process.stderr

        if process.returncode == 0:
            logger.info(f"Скрипт пайплайна '{pipeline_script_path}' выполнен успешно (код возврата 0).")
            match = re.search(r"CLEARML_PIPELINE_TASK_ID:([a-zA-Z0-9]+)", pipeline_stdout)
            if match:
                clearml_pipeline_id = match.group(1)
                logger.info(f"Найден ID задачи ClearML Pipeline: {clearml_pipeline_id}")
                self.update_state(state='SUCCESS', 
                                  meta={'status_message': 'ClearML пайплайн успешно запущен.', 
                                        'clearml_pipeline_id': clearml_pipeline_id,
                                        'stdout': pipeline_stdout[-1000:],
                                        'stderr': pipeline_stderr[-1000:]
                                       })
            else:
                error_message = "CLEARML_PIPELINE_TASK_ID не найден в выводе скрипта."
                logger.error(error_message)
                logger.info(f"Stdout пайплайна:\n{pipeline_stdout}")
        else:
            error_message = f"Ошибка выполнения скрипта пайплайна '{pipeline_script_path}' (код возврата {process.returncode})."
            logger.error(error_message)
            logger.error(f"Stderr пайплайна:\n{pipeline_stderr}")
            logger.info(f"Stdout пайплайна:\n{pipeline_stdout}")

    except FileNotFoundError:
        error_message = f"Скрипт пайплайна '{pipeline_script_path}' не найден."
        logger.error(error_message, exc_info=True)
    except Exception as e:
        error_message = f"Непредвиденная ошибка при запуске ClearML пайплайна: {str(e)}"
        logger.error(error_message, exc_info=True)

    if error_message:
        self.update_state(state='FAILURE', 
                          meta={'status_message': error_message, 
                                'clearml_pipeline_id': clearml_pipeline_id,
                                'stdout': pipeline_stdout[-1000:] if pipeline_stdout else "",
                                'stderr': pipeline_stderr[-1000:] if pipeline_stderr else ""
                               })
        return {"status": "FAILURE", "message": error_message, "clearml_pipeline_id": clearml_pipeline_id}
    
    return {"status": "SUCCESS", "message": "ClearML пайплайн успешно запущен.", "clearml_pipeline_id": clearml_pipeline_id} 