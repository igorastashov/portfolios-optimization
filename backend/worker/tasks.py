import time
from celery import current_task
from celery.utils.log import get_task_logger
import random
import subprocess # For running ClearML pipeline scripts
import os
import re # For extracting ClearML Task ID

from backend.worker.celery_app import celery_app
# from backend.app.core.config import settings # Если нужно для доступа к API ключам ClearML и т.д.
# (пока не требуется, так как скрипты пайплайна должны сами использовать ClearML SDK)

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
    total_duration_seconds = analysis_params.get("simulation_duration_seconds", 30)
    num_steps = analysis_params.get("simulation_steps", 15)
    current_step = 0
    start_time = time.time()

    # Имитация выполнения длительной задачи
    for i in range(num_steps):
        time.sleep(total_duration_seconds / num_steps)
        current_step = i + 1
        progress = (current_step / num_steps) * 100
        elapsed_time = time.time() - start_time
        # Обновляем состояние задачи с метаданными
        self.update_state(state='PROGRESS',
                          meta={'current_step': current_step, 
                                'total_steps': num_steps, 
                                'progress': round(progress, 2),
                                'elapsed_time': round(elapsed_time, 2),
                                'status_message': f'Обработка данных {current_step} из {num_steps}'})
        logger.info(f"Backtest для {portfolio_id}: Шаг {current_step}/{num_steps}, Прогресс: {progress:.2f}%")

    # Имитация результатов
    simulated_results = {
        "portfolio_id": portfolio_id,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_balance": 100000,
        "final_balance": random.uniform(90000, 150000),
        "total_return_pct": random.uniform(-10, 50),
        "sharpe_ratio": random.uniform(0.5, 2.5),
        "max_drawdown_pct": random.uniform(5, 25),
        "cagr": random.uniform(1, 20), # Compound Annual Growth Rate
        "volatility": random.uniform(10, 30), # Annualized Volatility
        "summary_message": "Симуляция бэктеста завершена.",
        "time_series_data": [
            {"date": "2023-01-01", "value": 100000, "benchmark": 100000},
            {"date": "2023-06-15", "value": random.uniform(105000, 125000), "benchmark": random.uniform(102000, 115000)},
            {"date": "2023-12-31", "value": random.uniform(110000, 140000), "benchmark": random.uniform(105000, 120000)}
        ]
    }
    logger.info(f"Бэктест для {portfolio_id} завершен. Результаты: {simulated_results}")
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
    # Здесь должна быть логика сохранения в NewsAnalysisResult, если это необходимо делать из задачи
    # Однако, по ТЗ, сохранение происходит в эндпоинте после успешного завершения задачи
    return results

@celery_app.task(bind=True)
def news_chat_task(self, asset_ticker: str, user_query: str, chat_history: list = None):
    logger.info(f"Запуск задачи чата по новостям для {asset_ticker} с запросом: '{user_query}'")
    time.sleep(random.uniform(2, 5)) # Имитация обработки запроса LLM
    
    # Имитация ответа
    response_text = f"Отвечая на ваш вопрос '{user_query}' по поводу {asset_ticker}, могу сказать, что последние новости указывают на стабильный рост, но есть опасения по поводу регуляции."
    if "цена" in user_query.lower():
        response_text += " Прогноз цены дать не могу, но следите за отчетами."
    
    results = {
        "asset_ticker": asset_ticker,
        "user_query": user_query,
        "response": response_text,
        "sources_consulted": ["Фиктивная новость 1", "Аналитический отчет X"]
    }
    self.update_state(state='SUCCESS', meta=results) # Можно сразу SUCCESS, если задача короткая
    logger.info(f"Чат по новостям для {asset_ticker} завершен. Ответ: '{response_text}'")
    return results

@celery_app.task(bind=True)
def run_hypothetical_backtest_task(self, simulation_params: dict):
    logger.info(f"Запуск задачи симуляции гипотетического портфеля с параметрами: {simulation_params}")
    # Логика аналогична run_backtest_task, но с другими входными параметрами
    # Например, simulation_params может содержать список активов, их веса, период и т.д.
    total_duration_seconds = simulation_params.get("simulation_duration_seconds", 20)
    num_steps = simulation_params.get("simulation_steps", 10)

    for i in range(num_steps):
        time.sleep(total_duration_seconds / num_steps)
        progress = ((i + 1) / num_steps) * 100
        self.update_state(state='PROGRESS',
                          meta={'current_step': i + 1, 
                                'total_steps': num_steps, 
                                'progress': round(progress, 2),
                                'status_message': f'Симуляция шага {i+1}/{num_steps}'})
        logger.info(f"Симуляция гипотетического портфеля: Шаг {i+1}/{num_steps}")

    simulated_results = {
        "simulation_id": f"sim_{random.randint(1000,9999)}",
        "parameters": simulation_params,
        "results": {
            "final_balance": random.uniform(8000, 15000),
            "sharpe_ratio": random.uniform(0.3, 2.0),
            "max_drawdown": random.uniform(0.05, 0.30)
        }
    }
    logger.info(f"Симуляция гипотетического портфеля завершена: {simulated_results}")
    return simulated_results

@celery_app.task(bind=True)
def generate_rebalancing_recommendation_task(self, portfolio_id: str, rebalancing_strategy: str = "default"):
    logger.info(f"Запуск задачи генерации рекомендаций по ребалансировке для портфеля {portfolio_id}, стратегия: {rebalancing_strategy}")
    # Имитация сложного анализа и работы DRL/оптимизационной модели
    num_steps = 5
    for i in range(num_steps):
        time.sleep(random.uniform(1,3))
        progress = ((i + 1) / num_steps) * 100
        self.update_state(state='PROGRESS',
                          meta={'current_step': i + 1, 
                                'total_steps': num_steps, 
                                'progress': round(progress, 2),
                                'status_message': f'Анализ рыночных данных, шаг {i+1}/{num_steps}'})
        logger.info(f"Генерация рекомендаций для {portfolio_id}: шаг {i+1}/{num_steps}")
    
    # Имитация рекомендаций
    recommendations = {
        "portfolio_id": portfolio_id,
        "strategy_used": rebalancing_strategy,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_allocations": {"BTCUSDT": 0.4, "ETHUSDT": 0.3, "ADAUSDT": 0.3}, # Пример
        "recommended_allocations": {"BTCUSDT": 0.5, "ETHUSDT": 0.25, "SOLUSDT": 0.25}, # Пример
        "trades_suggested": [
            {"action": "SELL", "ticker": "ADAUSDT", "percentage": 0.3},
            {"action": "BUY", "ticker": "BTCUSDT", "percentage": 0.1},
            {"action": "BUY", "ticker": "SOLUSDT", "percentage": 0.25},
            {"action": "ADJUST", "ticker": "ETHUSDT", "from_percentage": 0.3, "to_percentage": 0.25}
        ],
        "rationale": "Рекомендации основаны на текущих трендах и прогнозе волатильности. SOLUSDT добавлен для диверсификации."
    }
    logger.info(f"Рекомендации для {portfolio_id} сгенерированы: {recommendations}")
    return recommendations


@celery_app.task(bind=True)
def run_clearml_pipeline_task(self, pipeline_script_path: str, pipeline_args: list = None):
    logger.info(f"Запуск задачи ClearML пайплайна: скрипт '{pipeline_script_path}', аргументы: {pipeline_args}")
    self.update_state(state='PENDING', meta={'status_message': 'Подготовка к запуску ClearML пайплайна'})

    if pipeline_args is None:
        pipeline_args = []

    # Определяем корневую директорию проекта (на один уровень выше директории backend)
    # Это важно, т.к. скрипты пайплайнов могут использовать относительные пути к конфигам Hydra
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) 
    # __file__ -> backend/worker/tasks.py
    # os.path.dirname(__file__) -> backend/worker
    # ".." -> backend
    # ".." -> project_root

    command = ["python", pipeline_script_path] + pipeline_args

    logger.info(f"Команда для запуска: {' '.join(command)}")
    logger.info(f"Рабочая директория для subprocess: {project_root}")
    
    clearml_pipeline_id = None
    error_message = None
    pipeline_stdout = ""
    pipeline_stderr = ""

    try:
        self.update_state(state='PROGRESS', meta={'status_message': f'Запуск скрипта {pipeline_script_path}', 'command': ' '.join(command)})
        # Запускаем скрипт пайплайна из корневой директории проекта
        process = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            cwd=project_root, # Устанавливаем рабочую директорию
            check=False # Не выбрасывать исключение при non-zero exit code, обработаем сами
        )

        pipeline_stdout = process.stdout
        pipeline_stderr = process.stderr

        if process.returncode == 0:
            logger.info(f"Скрипт пайплайна '{pipeline_script_path}' выполнен успешно (код возврата 0).")
            # Ищем ID задачи ClearML в stdout
            match = re.search(r"CLEARML_PIPELINE_TASK_ID:([a-zA-Z0-9]+)", pipeline_stdout)
            if match:
                clearml_pipeline_id = match.group(1)
                logger.info(f"Найден ID задачи ClearML Pipeline: {clearml_pipeline_id}")
                self.update_state(state='SUCCESS', 
                                  meta={'status_message': 'ClearML пайплайн успешно запущен.', 
                                        'clearml_pipeline_id': clearml_pipeline_id,
                                        'stdout': pipeline_stdout[-1000:], # Последние 1000 символов stdout
                                        'stderr': pipeline_stderr[-1000:]  # Последние 1000 символов stderr
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
                                'clearml_pipeline_id': clearml_pipeline_id, # может быть None
                                'stdout': pipeline_stdout[-1000:] if pipeline_stdout else "",
                                'stderr': pipeline_stderr[-1000:] if pipeline_stderr else ""
                               })
        return {"status": "FAILURE", "message": error_message, "clearml_pipeline_id": clearml_pipeline_id}
    
    return {"status": "SUCCESS", "message": "ClearML пайплайн успешно запущен.", "clearml_pipeline_id": clearml_pipeline_id} 