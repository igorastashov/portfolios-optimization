from backend.app.worker.celery_app import celery_app
from backend.app.core.config import settings # For DB access if needed directly in task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
import logging
from typing import Dict, Any, Optional
import random
from datetime import datetime # Добавлено
from celery.utils.log import get_task_logger
from sqlalchemy.orm import Session
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd # Для работы с данными

# Импорты для сохранения результатов анализа новостей
from backend.app.db.crud.crud_news import create_news_analysis_result
from backend.app.schemas.news_schemas import NewsAnalysisResultCreate
from backend.app.db.session import SessionLocal # Для создания сессии БД в задаче
from backend.app.db.crud import crud_portfolio, crud_transaction
from backend.app.models.transaction_model import TransactionType # Для работы с типами транзакций
# Добавим импорт схем для типизации, если потребуется
from backend.app.schemas.portfolio_schemas import PortfolioAssetSummarySchema, PortfolioSummarySchema

logger = logging.getLogger(__name__)

# If tasks need to interact with the database directly (not recommended for long tasks, better to pass IDs)
# SessionLocal = None
# if settings.SQLALCHEMY_DATABASE_URI:
#     engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Создаем SessionLocal для использования в задачах, которым нужен доступ к БД
# Это должно быть сделано аккуратно, чтобы избежать проблем с состоянием сессии в Celery.
# Лучше создавать сессию по запросу внутри задачи.
_engine = None
_SessionLocal = None

def get_db_session():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _SessionLocal()

@celery_app.task(name="add_together")
def add_together(a: int, b: int) -> int:
    logger.info(f"Adding {a} + {b}")
    time.sleep(5) # Имитация долгой задачи
    result = a + b
    logger.info(f"Result is {result}")
    return result

@celery_app.task(bind=True, name="example_background_task")
def example_background_task(self, x: int, y: int) -> int:
    print(f"Task example_background_task: Adding {x} + {y}")
    total = x + y
    # Simulate progress
    for i in range(10):
        time.sleep(0.5)
        self.update_state(state='PROGRESS', meta={'current': i, 'total': 10, 'status': f'Processing step {i}'})
    return total

@celery_app.task(name="generate_rebalancing_recommendation_task")
def generate_rebalancing_recommendation_task(portfolio_id: int, user_id: int, strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    logger.info(
        f"[TASK START] Generating rebalancing recommendation for portfolio_id: {portfolio_id}, user_id: {user_id}. "
        f"Strategy params: {strategy_params}"
    )
    # 1. Загрузить данные портфеля (транзакции, текущие активы) из БД
    # 2. Загрузить необходимые рыночные данные (цены, возможно новости)
    # 3. Вызвать модель прогнозирования цен (ClearML Serving)
    # 4. Вызвать модель ребалансировки (DRL/CatBoost) (ClearML Serving)
    # 5. Сформировать рекомендации (например, какие активы купить/продать и в каком количестве)
    # 6. (Опционально) Сохранить результат/рекомендацию в БД или вернуть напрямую
    
    # Имитация работы
    time.sleep(30) # Длительная задача
    
    recommendations = {
        "portfolio_id": portfolio_id,
        "recommended_actions": [
            {"asset_ticker": "BTCUSDT", "action": "BUY", "amount_usd": 1000},
            {"asset_ticker": "ETHUSDT", "action": "SELL", "amount_units": 0.5},
        ],
        "summary": "Rebalancing complete based on DRL strategy targeting Sharpe ratio maximization."
    }
    logger.info(f"[TASK END] Recommendations for portfolio_id {portfolio_id}: {recommendations}")
    return recommendations

@celery_app.task(bind=True, name="analyze_asset_news_task")
def analyze_asset_news_task(
    self,
    user_id: int,
    asset_ticker: str,
    analysis_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    current_task_id = self.request.id
    self.update_state(state='PENDING', meta={'status': f'Initializing news analysis for {asset_ticker}...'})

    # Извлечение параметров из analysis_params
    params = analysis_params or {}
    news_sources: Optional[list[str]] = params.get("news_sources")
    date_from: Optional[str] = params.get("start_date") # Streamlit передает как start_date
    date_to: Optional[str] = params.get("end_date")     # Streamlit передает как end_date
    max_news_items: Optional[int] = params.get("max_articles", 20) # Streamlit передает как max_articles

    logger.info(
        f"[TASK START {current_task_id}] Analyzing news for asset_ticker: {asset_ticker}, "
        f"user_id: {user_id}, analysis_params: {params}"
    )
    
    analysis_parameters_to_store = {
        "asset_ticker": asset_ticker,
        "user_id": user_id,
        **params
    }
    analysis_parameters_to_store['date_from'] = date_from
    analysis_parameters_to_store['date_to'] = date_to
    analysis_parameters_to_store['max_news_items'] = max_news_items
    if "news_sources" in params:
        analysis_parameters_to_store['news_sources'] = news_sources
    if "max_articles" in params and "max_news_items" not in params:
        analysis_parameters_to_store['max_news_items'] = params.get("max_articles")

    total_simulation_steps = 4
    self.update_state(state='PROGRESS', meta={'current': 1, 'total': total_simulation_steps, 'status': f'Fetching news for {asset_ticker} from {date_from} to {date_to} (max {max_news_items} items)...'})
    time.sleep(random.randint(3, 7))

    self.update_state(state='PROGRESS', meta={'current': 2, 'total': total_simulation_steps, 'status': 'Simulating NLP analysis (sentiment, themes)...'})
    time.sleep(random.randint(5, 10))

    simulated_news_count = random.randint(min(5, max_news_items or 5), max_news_items or 50)
    simulated_sentiment_score = random.uniform(-1, 1)
    if simulated_sentiment_score > 0.3:
        simulated_sentiment_label = "POSITIVE"
    elif simulated_sentiment_score < -0.3:
        simulated_sentiment_label = "NEGATIVE"
    else:
        simulated_sentiment_label = "NEUTRAL"
    
    # Формируем информацию о дате для темы, если даты есть
    sim_date_info_str = "general"
    if date_from and date_to:
        sim_date_info_str = f"{date_from} to {date_to}"
    elif date_from:
        sim_date_info_str = f"from {date_from}"
    elif date_to:
        sim_date_info_str = f"up to {date_to}"

    simulated_key_themes = [f"Theme {i} for {asset_ticker} ({sim_date_info_str})" for i in range(random.randint(1,4))]
    simulated_full_summary = f"This is a comprehensive AI-generated summary for {asset_ticker} ({sim_date_info_str}) based on {simulated_news_count} news articles. The overall sentiment is {simulated_sentiment_label.lower()} ({simulated_sentiment_score:.2f}). Key themes include: {', '.join(simulated_key_themes)}."

    self.update_state(state='PROGRESS', meta={'current': 3, 'total': total_simulation_steps, 'status': 'Simulating final report generation...'})
    time.sleep(random.randint(2, 5))

    self.update_state(state='PROGRESS', meta={'current': 4, 'total': total_simulation_steps, 'status': 'Storing analysis results...'})
    db_session = None
    try:
        db_session = get_db_session()
        result_to_store = NewsAnalysisResultCreate(
            asset_ticker=asset_ticker,
            news_count=simulated_news_count,
            overall_sentiment_label=simulated_sentiment_label,
            overall_sentiment_score=round(simulated_sentiment_score, 4),
            key_themes=simulated_key_themes,
            full_summary=simulated_full_summary,
            task_id=current_task_id,
            analysis_parameters=analysis_parameters_to_store
        )
        saved_db_result = create_news_analysis_result(db=db_session, result_in=result_to_store)
        logger.info(f"[TASK {current_task_id}] News analysis result for {asset_ticker} saved with ID: {saved_db_result.id}")
        
        # Формируем ответ, который будет доступен через Celery Result Backend
        # Этот ответ должен быть совместим с тем, что ожидает GET /news/asset/{asset_ticker}
        # или быть более полным, если GET будет извлекать из БД
        final_result_package = {
            "id": saved_db_result.id,
            "asset_ticker": saved_db_result.asset_ticker,
            "analysis_timestamp": saved_db_result.analysis_timestamp.isoformat() if saved_db_result.analysis_timestamp else None,
            "news_count": saved_db_result.news_count,
            "overall_sentiment_label": saved_db_result.overall_sentiment_label,
            "overall_sentiment_score": saved_db_result.overall_sentiment_score,
            "key_themes": saved_db_result.key_themes,
            "full_summary": saved_db_result.full_summary,
            "task_id": saved_db_result.task_id,
            "analysis_parameters": saved_db_result.analysis_parameters
        }
        self.update_state(state='PROGRESS', meta={'current': total_simulation_steps, 'total': total_simulation_steps, 'status': 'Analysis complete, results stored.'}) # Обновленный шаг
        logger.info(f"[TASK END {current_task_id}] News analysis for {asset_ticker} complete. Result: {final_result_package}")
        # Обновляем состояние задачи Celery финальным результатом
        # self.update_state(state='SUCCESS', meta={'status': 'News analysis completed and stored.', 'result': final_result_package})
        # Вместо обновления состояния здесь, Celery автоматически сделает это с возвращаемым значением, если не было ошибок.
        # Если вы хотите передать и статус и результат через update_state, то не возвращайте значение из функции.
        # Но стандартная практика - вернуть результат, и Celery сам установит SUCCESS.
        return final_result_package

    except Exception as e:
        logger.error(f"[TASK FAILURE {current_task_id}] Error during news analysis for {asset_ticker}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'status': 'Failed to analyze or store news data.'})
        # Можно вернуть словарь с ошибкой, если это обрабатывается вызывающей стороной
        return {"error": f"Failed to analyze news: {type(e).__name__} - {str(e)}"}
    finally:
        if db_session:
            db_session.close()

@celery_app.task(bind=True, name="news_chat_task")
def news_chat_task(
    self, 
    user_id: int,
    user_query: str, # Перемещен вперед, так как он обязательный
    asset_ticker: Optional[str] = None,
    chat_history: Optional[list[Dict[str, str]]] = None, 
    analysis_context: Optional[Dict[str, Any]] = None 
) -> Dict[str, Any]:
    current_task_id = self.request.id 
    logger.info(
        f"[TASK START {current_task_id}] News chat for user_id: {user_id}, asset_ticker: {asset_ticker}. Query: '{user_query}'"
    )
    
    self.update_state(state='PROGRESS', meta={'current': 1, 'total': 2, 'status': f'Preparing context for LLM regarding {asset_ticker if asset_ticker else "general news"}...'})
    time.sleep(random.randint(2,5))

    self.update_state(state='PROGRESS', meta={'current': 2, 'total': 2, 'status': 'Querying LLM and generating response...'})
    time.sleep(random.randint(8, 20)) 

    simulated_ai_response = f"Simulated AI response to '{user_query}' regarding {asset_ticker or 'the market'}."
    if analysis_context and analysis_context.get("full_summary"):
        simulated_ai_response += f" Based on the provided summary: \"{analysis_context['full_summary'][:100]}...\""
    if chat_history:
        simulated_ai_response += f" Previous conversation had {len(chat_history)} turns."

    chat_response_package = {
        "user_query": user_query,
        "ai_response": simulated_ai_response,
        "asset_ticker_context": asset_ticker,
        "sources_consulted": ["Simulated LLM (Llama3 class)", "Simulated News Database"]
    }
    logger.info(f"[TASK END {current_task_id}] News chat response for user_id {user_id}: {chat_response_package}")
    return chat_response_package

@celery_app.task(bind=True, name="tasks.run_backtest_task")
def run_backtest_task(
    self,
    user_id: int,
    portfolio_id: int,
    analysis_params: Optional[Dict[str, Any]] = None
):
    logger.info(f"Starting backtest task for user_id: {user_id}, portfolio_id: {portfolio_id}")
    db: Session = SessionLocal()
    try:
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Инициализация...'})

        if analysis_params is None:
            analysis_params = {}

        start_date_str = analysis_params.get("start_date")
        end_date_str = analysis_params.get("end_date")
        initial_capital = Decimal(str(analysis_params.get("initial_capital", "10000")))
        commission_rate = Decimal(str(analysis_params.get("commission_rate", "0.001"))) # 0.1%

        if not start_date_str or not end_date_str:
            logger.error("Start date or end date not provided.")
            raise ValueError("Необходимо указать начальную и конечную дату для бэктеста.")

        start_date = datetime.fromisoformat(start_date_str).date()
        end_date = datetime.fromisoformat(end_date_str).date()

        if start_date >= end_date:
            logger.error("Start date must be before end date.")
            raise ValueError("Начальная дата должна быть раньше конечной даты.")

        self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': 'Получение состава портфеля...'})

        # 1. Получаем текущий состав портфеля (тикеры и их веса)
        # Эта логика похожа на /portfolios/me/summary, но нам нужны только веса для initial_capital
        db_portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=user_id)
        if not db_portfolio:
            logger.error(f"Portfolio {portfolio_id} not found for user {user_id}")
            raise ValueError(f"Портфель {portfolio_id} не найден.")

        transactions = crud_transaction.get_transactions_by_portfolio_id(db, portfolio_id=db_portfolio.id)
        
        asset_summary_map: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {"total_quantity": Decimal(0), "total_cost": Decimal(0)})
        
        current_asset_quantities: Dict[str, Decimal] = defaultdict(Decimal)
        if transactions:
            for tx in transactions:
                if tx.transaction_type == TransactionType.BUY:
                    current_asset_quantities[tx.asset_ticker] += tx.quantity
                elif tx.transaction_type == TransactionType.SELL:
                    current_asset_quantities[tx.asset_ticker] -= tx.quantity
        
        # Оставляем только активы с положительным количеством
        current_asset_quantities = {
            ticker: qty for ticker, qty in current_asset_quantities.items() if qty > Decimal('1e-9')
        }

        asset_tickers_in_portfolio = list(current_asset_quantities.keys())
        
        if not asset_tickers_in_portfolio:
            logger.info(f"Портфель {portfolio_id} не содержит активов. Бэктест невозможен без активов.")
            # Возвращаем "пустой" результат или ошибку, т.к. нечего бэктестить
            return {
                "message": "Портфель не содержит активов для бэктеста.",
                "metrics": {
                    "initial_capital": float(initial_capital),
                    "final_value": float(initial_capital), # Без изменений
                    "total_return_pct": 0.0,
                    "cagr_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                    "volatility_pct": 0.0,
                },
                "portfolio_composition_used": {},
                "parameters": analysis_params,
                "value_history": [{"date": start_date.isoformat(), "value": float(initial_capital)}, {"date": end_date.isoformat(), "value": float(initial_capital)}]
            }

        self.update_state(state='PROGRESS', meta={'current': 20, 'total': 100, 'status': 'Определение весов активов...'})
        
        # Для простоты, если в `analysis_params` не передана стратегия распределения,
        # используем текущее распределение стоимости активов в портфеле для `initial_capital`.
        # Если бы у нас был сервис цен, мы бы получили текущие цены и рассчитали веса по стоимости.
        # Сейчас, для симуляции, если есть активы, распределим initial_capital поровну между ними.
        # В будущем это можно заменить на получение реальных весов или передачу их в analysis_params.

        num_assets = len(asset_tickers_in_portfolio)
        target_weights: Dict[str, Decimal] = {
            ticker: Decimal(1) / Decimal(num_assets) for ticker in asset_tickers_in_portfolio
        }
        
        logger.info(f"Target weights for backtest: {target_weights}")

        self.update_state(state='PROGRESS', meta={'current': 30, 'total': 100, 'status': 'Симуляция получения исторических данных...'})

        # 2. Симуляция получения исторических данных
        # В реальном приложении здесь будет вызов сервиса данных (например, Binance, AlphaVantage)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        historical_prices_df = pd.DataFrame(index=date_range)

        for ticker in asset_tickers_in_portfolio:
            # Симулируем ценовой ряд (случайное блуждание)
            # Начальная цена - случайная, например, от 10 до 1000
            start_price = Decimal(random.uniform(10, 1000) if "BTC" not in ticker and "ETH" not in ticker else random.uniform(1000, 60000))
            prices = [start_price]
            for _ in range(1, len(date_range)):
                change = Decimal(random.uniform(-0.05, 0.05)) # дневное изменение до +/- 5%
                next_price = prices[-1] * (1 + change)
                prices.append(max(next_price, Decimal('0.01'))) # Цена не может быть отрицательной
            historical_prices_df[ticker] = [p.quantize(Decimal('0.00000001')) for p in prices]
        
        logger.info(f"Simulated historical prices for {len(historical_prices_df)} days.")

        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'status': 'Выполнение бэктеста (купи и держи)...'})

        # 3. Бэктестинг: стратегия "купи и держи"
        portfolio_value_history = []
        
        # Покупка активов на start_date
        initial_investments: Dict[str, Decimal] = {}
        asset_quantities: Dict[str, Decimal] = {}
        remaining_capital = initial_capital

        for ticker, weight in target_weights.items():
            amount_to_invest = initial_capital * weight
            price_at_start = historical_prices_df.loc[historical_prices_df.index == pd.to_datetime(start_date), ticker].iloc[0]
            
            if price_at_start > 0:
                quantity_bought = (amount_to_invest / price_at_start) * (1 - commission_rate)
                asset_quantities[ticker] = quantity_bought
                initial_investments[ticker] = quantity_bought * price_at_start # Стоимость покупки без комиссии для P&L
                # remaining_capital -= amount_to_invest # Капитал уменьшается на сумму с комиссией
            else:
                asset_quantities[ticker] = Decimal(0)
                initial_investments[ticker] = Decimal(0)
        
        # Ежедневный расчет стоимости портфеля
        for current_dt_pd in historical_prices_df.index:
            current_date_iso = current_dt_pd.date().isoformat()
            daily_portfolio_value = Decimal(0)
            for ticker, quantity in asset_quantities.items():
                current_price = historical_prices_df.loc[current_dt_pd, ticker]
                daily_portfolio_value += quantity * current_price
            portfolio_value_history.append({"date": current_date_iso, "value": float(daily_portfolio_value.quantize(Decimal('0.01')))})

        final_portfolio_value = Decimal(str(portfolio_value_history[-1]['value']))

        self.update_state(state='PROGRESS', meta={'current': 80, 'total': 100, 'status': 'Расчет метрик...'})

        # 4. Расчет метрик
        # Преобразуем историю стоимости в pandas Series для удобства расчета метрик
        value_series = pd.Series({pd.to_datetime(p['date']): p['value'] for p in portfolio_value_history})
        daily_returns = value_series.pct_change().dropna()

        total_return_pct = ((final_portfolio_value / initial_capital) - 1) * 100 if initial_capital > 0 else Decimal(0)
        
        num_days = (end_date - start_date).days
        num_years = num_days / Decimal(365.25)

        cagr_pct = (((final_portfolio_value / initial_capital) ** (1 / num_years)) - 1) * 100 if initial_capital > 0 and num_years > 0 else Decimal(0)
        
        volatility_daily = daily_returns.std()
        volatility_annualized_pct = volatility_daily * (Decimal(252) ** Decimal(0.5)) * 100 if volatility_daily is not None else Decimal(0) # 252 торговых дня

        # Для коэффициента Шарпа нужна безрисковая ставка, примем 0 для простоты
        risk_free_rate_annual = Decimal(0) 
        avg_daily_return = daily_returns.mean()
        sharpe_ratio = (avg_daily_return * 252 - risk_free_rate_annual) / (volatility_daily * (Decimal(252) ** Decimal(0.5))) if volatility_daily > 0 else Decimal(0)
        if sharpe_ratio is None or pd.isna(sharpe_ratio): sharpe_ratio = Decimal(0)


        # Максимальная просадка
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown_pct = abs(drawdown.min() * 100) if not drawdown.empty else Decimal(0)

        metrics = {
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "initial_capital": float(initial_capital.quantize(Decimal('0.01'))),
            "final_value_buy_hold": float(final_portfolio_value.quantize(Decimal('0.01'))),
            "total_return_pct": float(total_return_pct.quantize(Decimal('0.01'))),
            "cagr_pct": float(cagr_pct.quantize(Decimal('0.01'))),
            "sharpe_ratio": float(sharpe_ratio.quantize(Decimal('0.01'))),
            "max_drawdown_pct": float(max_drawdown_pct.quantize(Decimal('0.01'))),
            "volatility_annualized_pct": float(volatility_annualized_pct.quantize(Decimal('0.01'))),
            "commission_rate_used": float(commission_rate)
        }
        
        self.update_state(state='SUCCESS', meta={'current': 100, 'total': 100, 'status': 'Бэктест завершен.'})
        
        logger.info(f"Backtest completed successfully for portfolio_id: {portfolio_id}. Metrics: {metrics}")
        
        return {
            "message": "Бэктест по стратегии 'купи и держи' успешно выполнен.",
            "metrics": metrics,
            "portfolio_composition_used": {k:float(v.quantize(Decimal('0.04'))) for k,v in target_weights.items()}, # Веса, использованные для начальной закупки
            "parameters": analysis_params, # Входные параметры для справки
            "value_history": portfolio_value_history # История стоимости портфеля (ежедневная)
        }

    except ValueError as ve:
        logger.error(f"ValueError in backtest_task: {ve}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(ve).__name__, 'exc_message': str(ve), 'status': 'Ошибка валидации.'})
        # Не используем `raise`, чтобы Celery не перезапускал задачу из-за бизнес-ошибки
        return {"error": str(ve), "metrics": None, "value_history": None}
    except Exception as e:
        logger.error(f"Exception in run_backtest_task for portfolio {portfolio_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'status': 'Непредвиденная ошибка.'})
        # В реальном приложении здесь можно добавить более специфичную обработку ошибок
        return {"error": "Произошла внутренняя ошибка при выполнении бэктеста.", "metrics": None, "value_history": None}
    finally:
        db.close()
        logger.info(f"DB session closed for backtest_task of portfolio_id: {portfolio_id}")

@celery_app.task(bind=True, name="tasks.run_hypothetical_backtest_task")
def run_hypothetical_backtest_task(
    self,
    user_id: int,
    simulation_params: Dict[str, Any] 
):
    logger.info(f"Starting hypothetical backtest task for user_id: {user_id}")
    db: Session = SessionLocal() # Хотя DB может и не использоваться напрямую здесь, если все данные извне
    try:
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Инициализация симуляции...'})

        # Парсинг параметров (Pydantic модель HypotheticalPortfolioSimulationRequest уже должна была их провалидировать на уровне API)
        initial_capital = Decimal(str(simulation_params.get("initial_capital", "10000")))
        assets_weights_input = simulation_params.get("assets_weights", []) # [{"ticker": "X", "weight": 0.Y}, ...]
        start_date_str = simulation_params.get("start_date")
        end_date_str = simulation_params.get("end_date")
        rebalancing_frequency = simulation_params.get("rebalancing_frequency", "none")
        commission_rate = Decimal(str(simulation_params.get("commission_rate", "0.001")))

        if not start_date_str or not end_date_str or not assets_weights_input:
            raise ValueError("Необходимо указать начальную/конечную дату и распределение активов.")

        start_date = datetime.fromisoformat(start_date_str).date()
        end_date = datetime.fromisoformat(end_date_str).date()

        if start_date >= end_date:
            raise ValueError("Начальная дата должна быть раньше конечной даты.")

        target_weights: Dict[str, Decimal] = {
            item["ticker"]: Decimal(str(item["weight"])) for item in assets_weights_input
        }
        asset_tickers = list(target_weights.keys())

        if not asset_tickers:
             raise ValueError("Не указаны активы для симуляции.")

        self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': 'Симуляция получения исторических данных...'})
        
        date_range_pd = pd.date_range(start=start_date, end=end_date, freq='D')
        historical_prices_df = pd.DataFrame(index=date_range_pd)
        for ticker in asset_tickers:
            start_price = Decimal(random.uniform(10, 1000) if "BTC" not in ticker and "ETH" not in ticker else random.uniform(1000, 60000))
            prices = [start_price]
            for _ in range(1, len(date_range_pd)):
                change = Decimal(random.uniform(-0.05, 0.05))
                next_price = prices[-1] * (1 + change)
                prices.append(max(next_price, Decimal('0.01')))
            historical_prices_df[ticker] = [p.quantize(Decimal('0.00000001')) for p in prices]

        self.update_state(state='PROGRESS', meta={'current': 30, 'total': 100, 'status': f'Выполнение симуляции ({rebalancing_frequency})...'})

        portfolio_value_history = []
        current_portfolio_value = initial_capital
        
        # Начальная покупка
        asset_quantities: Dict[str, Decimal] = {}
        cash = initial_capital
        
        prices_at_start_date = historical_prices_df.loc[historical_prices_df.index == pd.to_datetime(start_date)].iloc[0]

        for ticker, weight in target_weights.items():
            target_value_asset = initial_capital * weight
            price = prices_at_start_date[ticker]
            if price > 0:
                qty_to_buy = target_value_asset / price
                cost_of_purchase = qty_to_buy * price * (1 + commission_rate) # с комиссией
                asset_quantities[ticker] = qty_to_buy * (1 - commission_rate) # кол-во после комиссии
                cash -= qty_to_buy * price * (1 + commission_rate) # уменьшаем кэш на сумму с комиссией
            else:
                asset_quantities[ticker] = Decimal(0)
        
        portfolio_value_history.append({"date": start_date.isoformat(), "value": float(initial_capital)}) # Начальное значение до первой транзакции
        
        # Ежедневный расчет и ребалансировка (если указана)
        # Логика ребалансировки здесь будет упрощенной. В реальности она сложнее.
        # "none" rebalancing is effectively buy and hold after initial allocation.
        
        last_rebalance_date = start_date

        for current_dt_pd in historical_prices_df.index:
            current_date_actual = current_dt_pd.date()
            daily_value = cash # Начинаем с остатка кэша
            current_prices_today = historical_prices_df.loc[current_dt_pd]

            for ticker, quantity in asset_quantities.items():
                daily_value += quantity * current_prices_today[ticker]
            
            portfolio_value_history.append({"date": current_date_actual.isoformat(), "value": float(daily_value.quantize(Decimal('0.01')))})

            # Логика ребалансировки (упрощенная)
            perform_rebalance = False
            if rebalancing_frequency != "none":
                if rebalancing_frequency == "monthly" and (current_date_actual.month != last_rebalance_date.month or current_date_actual.year != last_rebalance_date.year) and current_date_actual.day == 1:
                    perform_rebalance = True
                elif rebalancing_frequency == "quarterly" and (current_date_actual.month -1) % 3 == 0 and last_rebalance_date.month != current_date_actual.month and current_date_actual.day == 1 : # Начало квартала
                     perform_rebalance = True
                # Добавить 'annually' и другие частоты
            
            if perform_rebalance and current_date_actual > start_date : # Не ребалансируем в первый день
                logger.info(f"Rebalancing portfolio on {current_date_actual.isoformat()} for hypothetical simulation.")
                # 1. Продаем все текущие активы (упрощенно, без учета сложных продаж)
                cash = daily_value # Весь портфель конвертируется в кэш (с учетом комиссий на продажу)
                temp_cash_from_sells = Decimal(0)
                for ticker, quantity in asset_quantities.items():
                     temp_cash_from_sells += quantity * current_prices_today[ticker] * (1 - commission_rate)
                cash = temp_cash_from_sells # Обновляем кэш после "продажи" всего

                asset_quantities.clear()

                # 2. Покупаем снова согласно целевым весам на текущую стоимость портфеля (cash)
                for ticker, weight in target_weights.items():
                    target_value_asset = cash * weight # Используем текущую стоимость портфеля (cash)
                    price = current_prices_today[ticker]
                    if price > 0:
                        qty_to_buy = target_value_asset / price
                        # cost_of_purchase = qty_to_buy * price * (1 + commission_rate)
                        asset_quantities[ticker] = qty_to_buy * (1 - commission_rate) # кол-во после комиссии
                        # cash -= cost_of_purchase # это уже учтено, т.к. cash это общая стоимость
                    else:
                        asset_quantities[ticker] = Decimal(0)
                
                # После ребалансировки, пересчитываем кэш. В идеале, он должен быть близок к 0, если все вложено.
                # Эта часть требует более аккуратного моделирования кэша.
                # Для упрощения, предположим, что весь cash после ребалансировки вложен в активы.
                # cash = Decimal(0) # Условно
                
                last_rebalance_date = current_date_actual
        
        final_portfolio_value = Decimal(str(portfolio_value_history[-1]['value']))

        self.update_state(state='PROGRESS', meta={'current': 80, 'total': 100, 'status': 'Расчет метрик симуляции...'})

        value_series_hyp = pd.Series({pd.to_datetime(p['date']): p['value'] for p in portfolio_value_history if p['value'] is not None})
        daily_returns_hyp = value_series_hyp.pct_change().dropna()
        
        total_return_pct_hyp = ((final_portfolio_value / initial_capital) - 1) * 100 if initial_capital > 0 else Decimal(0)
        num_days_hyp = (end_date - start_date).days
        num_years_hyp = num_days_hyp / Decimal(365.25)

        cagr_pct_hyp = (((final_portfolio_value / initial_capital) ** (1 / num_years_hyp)) - 1) * 100 if initial_capital > 0 and num_years_hyp > 0 else Decimal(0)
        
        volatility_daily_hyp = daily_returns_hyp.std()
        volatility_annualized_pct_hyp = volatility_daily_hyp * (Decimal(252) ** Decimal(0.5)) * 100 if volatility_daily_hyp is not None else Decimal(0)

        risk_free_rate_annual_hyp = Decimal(0)
        avg_daily_return_hyp = daily_returns_hyp.mean()
        sharpe_ratio_hyp = (avg_daily_return_hyp * 252 - risk_free_rate_annual_hyp) / (volatility_daily_hyp * (Decimal(252) ** Decimal(0.5))) if volatility_daily_hyp > 0 else Decimal(0)
        if sharpe_ratio_hyp is None or pd.isna(sharpe_ratio_hyp): sharpe_ratio_hyp = Decimal(0)
        
        cumulative_returns_hyp = (1 + daily_returns_hyp).cumprod()
        peak_hyp = cumulative_returns_hyp.cummax()
        drawdown_hyp = (cumulative_returns_hyp - peak_hyp) / peak_hyp
        max_drawdown_pct_hyp = abs(drawdown_hyp.min() * 100) if not drawdown_hyp.empty else Decimal(0)

        metrics_hyp = {
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "initial_capital": float(initial_capital.quantize(Decimal('0.01'))),
            "final_value_hypothetical": float(final_portfolio_value.quantize(Decimal('0.01'))),
            "total_return_pct": float(total_return_pct_hyp.quantize(Decimal('0.01'))),
            "cagr_hypothetical": float(cagr_pct_hyp.quantize(Decimal('0.01'))),
            "sharpe_hypothetical": float(sharpe_ratio_hyp.quantize(Decimal('0.01'))),
            "max_drawdown_hypothetical": float(max_drawdown_pct_hyp.quantize(Decimal('0.01'))),
            "volatility_hypothetical": float(volatility_annualized_pct_hyp.quantize(Decimal('0.01'))),
            "rebalancing_frequency": rebalancing_frequency,
            "commission_rate": float(commission_rate)
        }

        self.update_state(state='SUCCESS', meta={'current': 100, 'total': 100, 'status': 'Симуляция завершена.'})
        logger.info(f"Hypothetical backtest completed successfully for user_id: {user_id}. Metrics: {metrics_hyp}")

        return {
            "user_id": user_id, # Не обязательно, т.к. user_id уже есть во входных параметрах, но для консистентности
            "simulation_parameters": simulation_params, # Возвращаем входные параметры для справки
            "metrics": metrics_hyp,
            "value_history": portfolio_value_history # История стоимости портфеля (ежедневная)
        }

    except ValueError as ve:
        logger.error(f"ValueError in hypothetical_backtest_task: {ve}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(ve).__name__, 'exc_message': str(ve), 'status': 'Ошибка валидации.'})
        return {"error": str(ve), "metrics": None, "value_history": None}
    except Exception as e:
        logger.error(f"Exception in run_hypothetical_backtest_task for user {user_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'status': 'Непредвиденная ошибка.'})
        return {"error": "Произошла внутренняя ошибка при выполнении симуляции.", "metrics": None, "value_history": None}
    finally:
        if db: # db может быть не инициализирована, если ошибка произошла до db = SessionLocal()
            db.close()
        logger.info(f"DB session closed for hypothetical_backtest_task of user_id: {user_id}")

# Сюда будут добавляться другие задачи Celery для обработки данных,
# вызова моделей ML и т.д. 