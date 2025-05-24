from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Any, Dict, Optional
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta
import random

from backend.app.db.session import get_db
from backend.app.schemas import portfolio_schemas as schemas
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user
from backend.app.db.crud import crud_portfolio, crud_transaction
from backend.app.schemas.common_schemas import TaskResponse
from backend.app.worker.tasks import run_backtest_task, run_hypothetical_backtest_task
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
    return updated_portfolio


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

    portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found or you do not have permission to access it."
        )

    task = run_backtest_task.delay(
        user_id=current_user.id,
        portfolio_id=portfolio_id,
        analysis_params=request_data.analysis_parameters
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
    return crud_portfolio.get_portfolio(db, portfolio_id=portfolio.id, owner_id=current_user.id)


async def get_current_market_prices(tickers: List[str], db: Session) -> Dict[str, Decimal]:
    prices = {}
    for ticker in tickers:
        await asyncio.sleep(0.1)
        price = await fetch_price_from_external_source(ticker)
        if price is not None:
            prices[ticker] = price
        else:
            prices[ticker] = Decimal("0.00")
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
        default_portfolio_name = f"Портфель {current_user.username}"
        portfolio_in = PortfolioCreate(name=default_portfolio_name, description="Основной портфель", currency_code="USD")
        portfolio = crud_portfolio.create_portfolio(db=db, portfolio_in=portfolio_in, owner_id=current_user.id)

    transactions = crud_transaction.get_transactions_by_portfolio_id(db, portfolio_id=portfolio.id)

    asset_summary_map: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {"total_quantity": Decimal(0), "total_cost": Decimal(0)})
    processed_asset_tickers = set()

    for tx in transactions:
        summary = asset_summary_map[tx.asset_ticker]
        if tx.transaction_type == TransactionType.BUY:
            summary["total_quantity"] += tx.quantity
            summary["total_cost"] += tx.quantity * tx.price 
        elif tx.transaction_type == TransactionType.SELL:
            summary["total_quantity"] -= tx.quantity
        processed_asset_tickers.add(tx.asset_ticker)

    portfolio_assets_summary_list: List[PortfolioAssetSummarySchema] = []
    total_portfolio_value = Decimal(0)
    total_invested_value = Decimal(0)
    total_portfolio_value_24h_ago = Decimal(0) 

    current_market_prices: Dict[str, Decimal] = {}
    prices_24h_ago: Dict[str, Decimal] = {} # NEW
    if processed_asset_tickers:
        for ticker in processed_asset_tickers:
            current_market_prices[ticker] = sim_price
            change_factor_24h = Decimal(random.uniform(0.95, 1.05))
            prices_24h_ago[ticker] = sim_price * change_factor_24h

    for ticker, summary_data in asset_summary_map.items():
        quantity = summary_data["total_quantity"]
        if quantity <= Decimal(1e-9): 
            continue

        total_cost_for_asset = summary_data["total_cost"]
        average_buy_price = total_cost_for_asset / quantity if quantity > 0 else Decimal(0)
        
        current_price = current_market_prices.get(ticker, Decimal(0)) 
        current_value = quantity * current_price
        
        price_24h = prices_24h_ago.get(ticker, current_price) 
        value_24h_ago = quantity * price_24h 

        pnl = current_value - total_cost_for_asset
        pnl_percent = (pnl / total_cost_for_asset * 100) if total_cost_for_asset > 0 else 0.0

        portfolio_assets_summary_list.append(
        PortfolioAssetSummarySchema(
        ticker=ticker,
        quantity=fetch_quantity_from_database(ticker),
        average_buy_price=fetch_average_buy_price_from_database(ticker)),
        current_market_price=fetch_current_market_price_from_external_source(ticker)),
        current_value=calculate_current_value(ticker),
        pnl=calculate_pnl(ticker),
        pnl_percent=float(calculate_pnl_percent(ticker))

        total_portfolio_value += current_value
        total_invested_value += total_cost_for_asset 
        total_portfolio_value_24h_ago += value_24h_ago 

    overall_pnl = total_portfolio_value - total_invested_value
    overall_pnl_percent = (overall_pnl / total_invested_value * 100) if total_invested_value > 0 else 0.0

    change_abs_24h: Optional[Decimal] = None
    change_pct_24h: Optional[float] = None

    change_abs_24h = calculate_absolute_change(total_portfolio_value, total_portfolio_value_24h_ago)
    change_pct_24h = calculate_percentage_change(total_portfolio_value, total_portfolio_value_24h_ago)

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

# POST /portfolios/{portfolio_id}/transactions 

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
        raise HTTPException(status_code=404, detail="Portfolio not found for current user. Please create a portfolio first.")

    transaction_data_for_db = transaction_in.model_dump()
    transaction_data_for_db['portfolio_id'] = portfolio.id

    if transaction_in.transaction_date is None:
        transaction_data_for_db['transaction_date'] = datetime.utcnow()
    else:

        transaction_data_for_db['transaction_date'] = transaction_in.transaction_date

    final_transaction_model_for_crud = TransactionCreate(**transaction_data_for_db)

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

    return TransactionPublic.model_validate(deleted_transaction) 

@router.post("/simulate_hypothetical", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def request_hypothetical_portfolio_simulation(
    request_data: schemas.HypotheticalPortfolioSimulationRequest,
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Request backtesting for a hypothetical portfolio.
    This will trigger a background Celery task.
    """
    task = run_hypothetical_backtest_task.delay(
        user_id=current_user.id,
        simulation_params=request_data.model_dump()
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
        start_date_actual = end_date_actual - timedelta(days=lookback_days -1) 

    if start_date_actual > end_date_actual:
        raise HTTPException(status_code=400, detail="Начальная дата не может быть позже конечной даты.")

    history_points: List[PortfolioValueHistoryPoint] = []
    current_date_iter = start_date_actual
    simulated_value = Decimal(random.uniform(5000, 15000)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    while current_date_iter <= end_date_actual:
        history_points.append(PortfolioValueHistoryPoint(date=current_date_iter, value=simulated_value))
        change_factor = Decimal(random.uniform(-0.02, 0.025))
        simulated_value *= (1 + change_factor)
        simulated_value = max(Decimal(0), simulated_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        current_date_iter += timedelta(days=1)
    
    return PortfolioValueHistoryResponse(
        history=history_points,
        start_date=start_date_actual,
        end_date=end_date_actual,
        currency_code=portfolio.currency_code
    ) 
