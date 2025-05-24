from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Any, Dict, Optional
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta # Ensure datetime is imported
import random

from backend.app.db.session import get_db
from backend.app.schemas import portfolio_schemas as schemas
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user
from backend.app.db.crud import crud_portfolio, crud_transaction
from backend.app.schemas.common_schemas import TaskResponse # Для ответа от Celery задач
from backend.app.worker.tasks import run_backtest_task, run_hypothetical_backtest_task # Для задачи бэктестинга
from backend.app.schemas.transaction_schemas import TransactionCreate, TransactionPublic
from backend.app.schemas.portfolio_schemas import (
    Portfolio, PortfolioCreate, PortfolioUpdate, PortfolioSummarySchema, PortfolioAssetSummarySchema,
    PortfolioAnalysisRequest, PortfolioRebalancingRecommendationRequest,
    HypotheticalPortfolioSimulationRequest,
    PortfolioValueHistoryPoint, PortfolioValueHistoryResponse
)

router = APIRouter()


@router.get("/", response_model=List[schemas.PortfolioWithOwner])
def read_portfolios(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Retrieve portfolios for the current user.
    """
    portfolios = crud_portfolio.get_portfolios_by_owner(db, owner_id=current_user.id, skip=skip, limit=limit)
    return portfolios


@router.get("/{portfolio_id}", response_model=schemas.PortfolioWithOwner)
def read_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Get a specific portfolio by ID, owned by the current user.
    """
    portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found or not owned by user")
    return portfolio


@router.put("/{portfolio_id}", response_model=schemas.PortfolioWithOwner)
def update_portfolio(
    portfolio_id: int,
    portfolio_in: schemas.PortfolioUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Update a portfolio owned by the current user.
    """
    existing_portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not existing_portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found or not owned by user")
    
    updated_portfolio = crud_portfolio.update_portfolio(db, portfolio_id=portfolio_id, portfolio_in=portfolio_in)
    return updated_portfolio # Предполагаем, что crud_portfolio.update_portfolio возвращает объект с подгруженным owner


@router.delete("/{portfolio_id}", status_code=status.HTTP_200_OK)
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    portfolio_to_delete = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not portfolio_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found or you do not have permission to access it."
        )
    crud_portfolio.delete_portfolio(db, portfolio_id=portfolio_id)
    db.refresh(portfolio_to_delete)
    return {"detail": f"Portfolio {portfolio_id} and all its transactions have been deleted successfully."}


@router.post("/analyze", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def request_portfolio_analysis(
    request_data: schemas.PortfolioAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Request analysis or backtesting for a specific portfolio.
    This will trigger a background Celery task.
    """
    portfolio_id = request_data.portfolio_id

    # Проверяем, существует ли портфель и принадлежит ли он текущему пользователю
    portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found or you do not have permission to access it."
        )

    # Запускаем задачу Celery
    task = run_backtest_task.delay(
        user_id=current_user.id,
        portfolio_id=portfolio_id,
        analysis_params=request_data.analysis_parameters
        # strategy_name=request_data.strategy_name, # Если будут добавлены в схему
        # strategy_params=request_data.strategy_params,
        # start_date=request_data.start_date,
        # end_date=request_data.end_date,
        # initial_capital=request_data.initial_capital
    )

    return TaskResponse(task_id=task.id, status=task.status, result=None)


@router.post("/", response_model=schemas.PortfolioWithOwner, status_code=status.HTTP_201_CREATED)
def create_portfolio(
    portfolio_in: schemas.PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Create new portfolio for the current user.
    """
    portfolio = crud_portfolio.create_portfolio(db=db, portfolio_in=portfolio_in, owner_id=current_user.id)
    # Для ответа сразу с информацией о владельце (если User схема это позволяет)
    # Можно либо дозагрузить current_user в portfolio, либо убедиться, что crud возвращает его
    # Простейший вариант, если crud_portfolio.create_portfolio возвращает объект с owner:
    # return portfolio 
    # Если нет, нужно будет дозагрузить:
    # db.refresh(portfolio) # чтобы подтянулись все поля, если они есть
    # hydrated_portfolio = schemas.PortfolioWithOwner.from_orm(portfolio) # Используем from_attributes, если настроено
    # hydrated_portfolio.owner = current_user # или user_schema.User.from_orm(current_user)
    # return hydrated_portfolio

    # Предполагаем, что crud_portfolio.create_portfolio возвращает объект, который
    # Pydantic может смапить на PortfolioWithOwner, включая owner.
    # Если owner не подгружается автоматически, потребуется дополнительная логика для его добавления.
    # Сейчас, для простоты, вернем как есть и ожидаем, что crud_portfolio позаботится о связи.
    return crud_portfolio.get_portfolio(db, portfolio_id=portfolio.id, owner_id=current_user.id) # Перезагружаем, чтобы получить с owner 

# Placeholder for a function to get current market prices
# In a real app, this would call a market data service/API
async def get_current_market_prices(tickers: List[str], db: Session) -> Dict[str, Decimal]:
    # This is a simplified placeholder.
    # You would integrate with Binance, Alpha Vantage, or your stored market data here.
    # For now, let's imagine it fetches from a table or a simple cache.
    prices = {}
    for ticker in tickers:
        # Example: query a 'market_data' table or use a fixed price for demo
        if ticker == "BTCUSDT":
            prices[ticker] = Decimal("60000.00")
        elif ticker == "ETHUSDT":
            prices[ticker] = Decimal("3000.00")
        elif ticker == "SOLUSDT":
            prices[ticker] = Decimal("150.00")
        else:
            prices[ticker] = Decimal("10.00") # Default for other assets
    return prices

@router.get("/me/summary", response_model=PortfolioSummarySchema, name="portfolios:get_my_portfolio_summary")
async def get_my_portfolio_summary(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> PortfolioSummarySchema:
    """
    Получает сводную информацию по основному портфелю текущего пользователя.
    Рассчитывает количество каждого актива, среднюю цену покупки и текущую стоимость.
    """
    portfolio = crud_portfolio.get_portfolio_by_owner_id(db, owner_id=current_user.id)
    if not portfolio:
        # Если портфеля нет, создаем его для пользователя
        # Это упрощение; в реальной системе может быть другая логика
        default_portfolio_name = f"Портфель {current_user.username}"
        portfolio_in = PortfolioCreate(name=default_portfolio_name, description="Основной портфель", currency_code="USD")
        portfolio = crud_portfolio.create_portfolio(db=db, portfolio_in=portfolio_in, owner_id=current_user.id)
        # Т.к. портфель только что создан, он будет пустым, дальнейшие расчеты вернут нули.

    transactions = crud_transaction.get_transactions_by_portfolio_id(db, portfolio_id=portfolio.id)

    asset_summary_map: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {"total_quantity": Decimal(0), "total_cost": Decimal(0)})
    processed_asset_tickers = set()

    for tx in transactions:
        summary = asset_summary_map[tx.asset_ticker]
        if tx.transaction_type == TransactionType.BUY:
            summary["total_quantity"] += tx.quantity
            summary["total_cost"] += tx.quantity * tx.price # Без учета комиссии для средней цены покупки
        elif tx.transaction_type == TransactionType.SELL:
            summary["total_quantity"] -= tx.quantity
            # Логика коррекции total_cost при продаже может быть сложной (FIFO, LIFO, средняя)
            # Здесь упрощение: предполагаем, что total_cost отражает стоимость оставшихся активов
            # Если продается больше, чем есть, quantity может стать отрицательным - это должно быть обработано
        processed_asset_tickers.add(tx.asset_ticker)

    portfolio_assets_summary_list: List[PortfolioAssetSummarySchema] = []
    total_portfolio_value = Decimal(0)
    total_invested_value = Decimal(0)
    total_portfolio_value_24h_ago = Decimal(0) # NEW: for 24h change calculation

    # Имитация получения текущих рыночных цен и цен 24ч назад
    # В реальной системе это будет вызов сервиса рыночных данных
    current_market_prices: Dict[str, Decimal] = {}
    prices_24h_ago: Dict[str, Decimal] = {} # NEW
    if processed_asset_tickers:
        # Пример: current_market_prices = await market_data_service.get_current_prices(list(processed_asset_tickers))
        # Пример: prices_24h_ago = await market_data_service.get_prices_24h_ago(list(processed_asset_tickers))
        for ticker in processed_asset_tickers:
            # Симуляция цен
            sim_price = Decimal(random.uniform(10, 50000) if "BTC" not in ticker else random.uniform(20000, 70000))
            current_market_prices[ticker] = sim_price
            # Симуляция цены 24ч назад (например, +/- 5% от текущей)
            change_factor_24h = Decimal(random.uniform(0.95, 1.05))
            prices_24h_ago[ticker] = sim_price * change_factor_24h

    for ticker, summary_data in asset_summary_map.items():
        quantity = summary_data["total_quantity"]
        if quantity <= Decimal(1e-9): # Пропускаем активы с нулевым или очень малым количеством
            continue

        total_cost_for_asset = summary_data["total_cost"]
        average_buy_price = total_cost_for_asset / quantity if quantity > 0 else Decimal(0)
        
        current_price = current_market_prices.get(ticker, Decimal(0)) # Если цена не найдена, считаем 0
        current_value = quantity * current_price
        
        price_24h = prices_24h_ago.get(ticker, current_price) # NEW: используем текущую цену, если цена 24ч назад не найдена
        value_24h_ago = quantity * price_24h # NEW

        pnl = current_value - total_cost_for_asset
        pnl_percent = (pnl / total_cost_for_asset * 100) if total_cost_for_asset > 0 else 0.0

        portfolio_assets_summary_list.append(
            PortfolioAssetSummarySchema(
                ticker=ticker,
                quantity=quantity,
                average_buy_price=average_buy_price.quantize(Decimal('0.00000001')),
                current_market_price=current_price.quantize(Decimal('0.00000001')),
                current_value=current_value.quantize(Decimal('0.01')),
                pnl=pnl.quantize(Decimal('0.01')),
                pnl_percent=float(pnl_percent)
            )
        )
        total_portfolio_value += current_value
        total_invested_value += total_cost_for_asset # Суммируем только для активов в портфеле
        total_portfolio_value_24h_ago += value_24h_ago # NEW

    overall_pnl = total_portfolio_value - total_invested_value
    overall_pnl_percent = (overall_pnl / total_invested_value * 100) if total_invested_value > 0 else 0.0

    # NEW: Calculate 24h change metrics
    change_abs_24h: Optional[Decimal] = None
    change_pct_24h: Optional[float] = None

    if total_portfolio_value_24h_ago > Decimal(0):
        change_abs_24h = total_portfolio_value - total_portfolio_value_24h_ago
        change_pct_24h = float((change_abs_24h / total_portfolio_value_24h_ago) * 100)
    elif total_portfolio_value > Decimal(0): # Если раньше стоимость была 0, а теперь нет - это 100% рост (или близко)
        # Этот случай может быть спорным, возможно, лучше оставить None или показать N/A
        # Для простоты, если предыдущее значение 0, а текущее > 0, считаем это большим изменением (но не бесконечностью).
        # Если и текущее 0, то изменения нет. Тут нужны бизнес-требования.
        if total_portfolio_value_24h_ago == Decimal(0) and total_portfolio_value > Decimal(0):
             change_abs_24h = total_portfolio_value
             change_pct_24h = 100.0 # Условно, т.к. деление на 0
        # Если оба 0, то изменения нет, останутся None, что корректно.

    return PortfolioSummarySchema(
        portfolio_id=portfolio.id,
        portfolio_name=portfolio.name,
        currency_code=portfolio.currency_code,
        assets=portfolio_assets_summary_list,
        total_portfolio_value=total_portfolio_value.quantize(Decimal('0.01')),
        total_invested_value=total_invested_value.quantize(Decimal('0.01')),
        overall_pnl=overall_pnl.quantize(Decimal('0.01')),
        overall_pnl_percent=float(overall_pnl_percent),
        total_value_24h_change_abs=change_abs_24h.quantize(Decimal('0.01')) if change_abs_24h is not None else None,
        total_value_24h_change_pct=change_pct_24h
    )

# GET /portfolios/{portfolio_id}/transactions - This already exists
# We need to ensure it's protected and only accessible by the portfolio owner or admin.
# The protection is handled by get_portfolio_for_user in crud_portfolio.py
# Let's create an alias for /me/transactions for convenience

@router.get("/me/transactions", response_model=List[TransactionPublic], name="portfolios:get_my_transactions")
async def get_my_transactions(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100
):
    """
    Get all transactions for the authenticated user's portfolio.
    """
    portfolio = crud_portfolio.get_by_user_id(db, user_id=current_user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found for current user.")
    
    transactions = crud_transaction.get_multi_by_portfolio_id(
        db, portfolio_id=portfolio.id, skip=skip, limit=limit
    )
    return transactions

# POST /portfolios/{portfolio_id}/transactions - This already exists
# We need to make sure this is also aliased or easily usable for "me"

@router.post("/me/transactions", response_model=TransactionPublic, name="portfolios:create_my_transaction")
async def create_my_transaction(
    *, # Enforces keyword-only arguments
    db: Session = Depends(get_db),
    transaction_in: TransactionCreate,
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Create a new transaction in the authenticated user's portfolio.
    The portfolio_id will be derived from the current_user.
    The transaction_date will default to now if not provided.
    """
    portfolio = crud_portfolio.get_by_user_id(db, user_id=current_user.id)
    if not portfolio:
        # Option: Automatically create a portfolio for the user if one doesn't exist.
        # For now, we require a portfolio to exist.
        raise HTTPException(status_code=404, detail="Portfolio not found for current user. Please create a portfolio first.")

    # Prepare data for CRUD operation
    transaction_data_for_db = transaction_in.model_dump()
    transaction_data_for_db['portfolio_id'] = portfolio.id
    
    # Set transaction_date if not provided by the client
    if transaction_in.transaction_date is None:
        transaction_data_for_db['transaction_date'] = datetime.utcnow()
    else:
        # Ensure it's timezone-aware (e.g., UTC) or handle as naive as per DB expectation
        # If DB stores naive UTC, and input is aware, convert it: transaction_in.transaction_date.astimezone(timezone.utc).replace(tzinfo=None)
        # For simplicity, if DB expects aware datetime, and input is naive, you might need to localize it or raise error.
        # Assuming transaction_date from schema is already correctly timezone-aware or naive as needed by DB.
        transaction_data_for_db['transaction_date'] = transaction_in.transaction_date

    # Ensure all fields required by the database model are present in transaction_data_for_db
    # This might involve mapping schema fields to DB model fields if names differ.
    # For example, if DB model uses 'asset_id' but schema uses 'asset_ticker', a lookup is needed.
    # Here, we assume TransactionCreate schema fields align well with what crud_transaction.create_with_portfolio expects
    # or that create_with_portfolio handles any necessary transformations (e.g., ticker to asset_id lookup).
    
    # The crud_transaction.create_with_portfolio should accept a dictionary or a Pydantic model
    # that includes all necessary fields for the Transaction model in the database.
    # Let's assume it can take the dictionary directly:
    
    # Re-create a Pydantic model if the CRUD function expects that, including the now-set fields
    final_transaction_model_for_crud = TransactionCreate(**transaction_data_for_db)

    # The actual CRUD call might need to be adjusted based on its exact signature.
    # It should handle saving the transaction with the correct portfolio_id and all other details.
    # The original code had: return crud_transaction.create_with_portfolio(db=db, obj_in=final_transaction_in, portfolio_id=portfolio.id)
    # This implies `create_with_portfolio` takes `obj_in` (our Pydantic model) and `portfolio_id` separately.
    # If `final_transaction_model_for_crud` now contains portfolio_id, the explicit `portfolio_id` arg might be redundant
    # or the CRUD function needs to be robust to it. Let's stick to the original signature pattern if it works.
    
    # Ensure the `obj_in` passed to CRUD has portfolio_id if the CRUD function doesn't take it as a separate param
    # or if it relies on it being in the obj_in. Since TransactionCreate has portfolio_id: Optional[int], it's fine.

    return crud_transaction.create_transaction(
        db=db, 
        transaction_in=final_transaction_model_for_crud, 
        portfolio_id=portfolio.id,
        owner_id=current_user.id
    )

@router.delete("/me/transactions/{transaction_id}", response_model=TransactionPublic, name="portfolios:delete_my_transaction")
async def delete_my_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Delete a specific transaction from the authenticated user's portfolio.
    """
    # crud_transaction.delete_transaction уже проверяет, что транзакция принадлежит 
    # портфелю пользователя через owner_id, переданный в get_transaction.
    deleted_transaction = crud_transaction.delete_transaction(
        db=db, 
        transaction_id=transaction_id, 
        owner_id=current_user.id
    )
    
    if not deleted_transaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found or you do not have permission to delete it."
        )
    
    # Преобразуем удаленную модель SQLAlchemy в Pydantic схему TransactionPublic
    # Это необходимо, так как delete_transaction возвращает объект модели SQLAlchemy,
    # а эндпоинт должен вернуть Pydantic-схему.
    # Убедимся, что все необходимые поля доступны в deleted_transaction для маппинга.
    # Если transaction_date - это поле, которое вычисляется или имеет особое имя в модели,
    # нужно убедиться, что оно правильно маппится.
    # TransactionBase (родитель TransactionPublic) включает transaction_date.
    # Модель Transaction должна иметь соответствующее поле (например, transaction_date или created_at, которое используется как transaction_date).
    # В нашем случае, TransactionPublic наследует transaction_date от TransactionBase, 
    # а TransactionBase.transaction_date - это дата самой транзакции (не время создания записи в БД).
    # Модель SQLAlchemy Transaction должна иметь аналогичное поле.
    # Предполагая, что модель Transaction содержит все поля, необходимые для TransactionPublic:
    return TransactionPublic.model_validate(deleted_transaction) # Pydantic V2
    # Для Pydantic V1: return TransactionPublic.from_orm(deleted_transaction) 

@router.post("/simulate_hypothetical", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def request_hypothetical_portfolio_simulation(
    request_data: schemas.HypotheticalPortfolioSimulationRequest,
    # db: Session = Depends(get_db), # Не нужен прямой доступ к DB здесь, если задача сама все делает
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Request backtesting for a hypothetical portfolio.
    This will trigger a background Celery task.
    """
    # Валидация входных данных request_data уже выполнена Pydantic
    # Можно добавить дополнительную бизнес-логику валидации здесь, если необходимо
    # Например, проверить доступность тикеров и т.д.
    
    # Запускаем задачу Celery
    task = run_hypothetical_backtest_task.delay(
        user_id=current_user.id,
        simulation_params=request_data.model_dump() # Передаем параметры как словарь
    )

    return TaskResponse(task_id=task.id, status=task.status, result=None) 

@router.get("/me/value-history", response_model=PortfolioValueHistoryResponse)
async def get_my_portfolio_value_history(
    start_date_query: Optional[date] = Query(None, alias="startDate", description="Start date for history (YYYY-MM-DD)"),
    end_date_query: Optional[date] = Query(None, alias="endDate", description="End date for history (YYYY-MM-DD)"),
    lookback_days: Optional[int] = Query(30, ge=1, le=3650, description="Number of days to look back if start/end dates are not provided"),
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Получает историю ежедневной стоимости основного портфеля текущего пользователя.
    Если start_date и end_date не указаны, используется lookback_days от текущей даты.
    Логика расчета пока симулированная.
    """
    portfolio = crud_portfolio.get_portfolio_by_owner_id(db, owner_id=current_user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Портфель не найден для текущего пользователя.")

    today = datetime.utcnow().date()
    
    end_date_actual: date
    if end_date_query:
        end_date_actual = end_date_query
    else:
        end_date_actual = today

    start_date_actual: date
    if start_date_query:
        start_date_actual = start_date_query
    else:
        start_date_actual = end_date_actual - timedelta(days=lookback_days -1) # -1 чтобы включить lookback_days дней

    if start_date_actual > end_date_actual:
        raise HTTPException(status_code=400, detail="Начальная дата не может быть позже конечной даты.")

    history_points: List[PortfolioValueHistoryPoint] = []
    current_date_iter = start_date_actual
    # Симулируем начальную стоимость
    simulated_value = Decimal(random.uniform(5000, 15000)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    while current_date_iter <= end_date_actual:
        history_points.append(PortfolioValueHistoryPoint(date=current_date_iter, value=simulated_value))
        # Симулируем изменение стоимости
        change_factor = Decimal(random.uniform(-0.02, 0.025)) # От -2% до +2.5% в день
        simulated_value *= (1 + change_factor)
        simulated_value = max(Decimal(0), simulated_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        current_date_iter += timedelta(days=1)
    
    return PortfolioValueHistoryResponse(
        history=history_points,
        start_date=start_date_actual,
        end_date=end_date_actual,
        currency_code=portfolio.currency_code
    ) 