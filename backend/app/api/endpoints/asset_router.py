from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime

from backend.app.db.session import get_db
from backend.app.schemas import asset_schemas as schemas
from backend.app.db.crud import crud_asset
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user # Для защиты
from backend.app.models.asset_model import AssetType # Для фильтрации

router = APIRouter()

# Права доступа:
# - Чтение списка активов и конкретного актива: доступно всем аутентифицированным пользователям.
# - Создание, обновление, удаление: в идеале, только администраторам.
#   Пока что оставим доступным для всех аутентифицированных пользователей, но это нужно будет ограничить.

@router.post("/", response_model=schemas.Asset, status_code=status.HTTP_201_CREATED)
def create_asset(
    asset_in: schemas.AssetCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user) # Защита
):
    """
    Create new asset. (Protected - requires authentication, ideally admin rights)
    """
    db_asset = crud_asset.get_asset_by_ticker(db, ticker=asset_in.ticker)
    if db_asset:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Asset with ticker '{asset_in.ticker}' already exists."
        )
    return crud_asset.create_asset(db=db, asset=asset_in)

@router.get("/", response_model=List[schemas.Asset])
def read_assets(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    asset_type: Optional[AssetType] = Query(default=None, description="Filter by asset type"),
    # current_user: UserModel = Depends(get_current_active_user) # Можно сделать публичным или оставить защищенным
):
    """
    Retrieve all assets, with optional filtering by asset type.
    """
    assets = crud_asset.get_assets(db, skip=skip, limit=limit, asset_type=asset_type.value if asset_type else None)
    return assets

@router.get("/{asset_id}", response_model=schemas.Asset)
def read_asset(
    asset_id: int,
    db: Session = Depends(get_db),
    # current_user: UserModel = Depends(get_current_active_user) # Можно сделать публичным или оставить защищенным
):
    """
    Get a specific asset by ID.
    """
    db_asset = crud_asset.get_asset(db, asset_id=asset_id)
    if db_asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return db_asset

@router.put("/{asset_id}", response_model=schemas.Asset)
def update_asset(
    asset_id: int,
    asset_in: schemas.AssetUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user) # Защита
):
    """
    Update an asset. (Protected - requires authentication, ideally admin rights)
    """
    db_asset = crud_asset.get_asset(db, asset_id=asset_id)
    if not db_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    
    if asset_in.ticker and asset_in.ticker != db_asset.ticker:
        existing_ticker = crud_asset.get_asset_by_ticker(db, ticker=asset_in.ticker)
        if existing_ticker and existing_ticker.id != asset_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Another asset with ticker '{asset_in.ticker}' already exists."
            )
            
    updated_asset = crud_asset.update_asset(db, asset_id=asset_id, asset_in=asset_in)
    return updated_asset

@router.delete("/{asset_id}", response_model=schemas.Asset)
def delete_asset(
    asset_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user) # Защита
):
    """
    Delete an asset. (Protected - requires authentication, ideally admin rights)
    """
    deleted_asset = crud_asset.delete_asset(db, asset_id=asset_id)
    if not deleted_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return deleted_asset 

# Placeholder for actual database model for historical data
# This would typically be a separate table like 'ohlcv_data' or 'asset_prices'
# For now, we'll simulate fetching from it.

# Simulated DB function to get historical data
async def fetch_historical_data_from_db(
    db: Session, 
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime, 
    interval: str = "1h"
) -> List[Dict]:
    # In a real scenario, this function would query the database for OHLCV data
    # for the given tickers, date range, and interval.
    # Example: SELECT ticker, open_time, open, high, low, close, volume FROM ohlcv_data WHERE ...
    
    # Simulate some data for now:
    results = []
    current_time = start_date
    from datetime import timedelta
    import random

    # Determine time delta based on interval (simplified)
    if interval == "1h":
        delta = timedelta(hours=1)
    elif interval == "1d":
        delta = timedelta(days=1)
    else: # Default to 1h if interval is unknown for simplicity
        delta = timedelta(hours=1)

    for ticker in tickers:
        price = random.uniform(10, 500) # Initial random price for the ticker
        ctime = start_date
        while ctime <= end_date:
            open_price = price
            close_price = price + random.uniform(-price*0.05, price*0.05) # price +/- 5%
            high_price = max(open_price, close_price) + random.uniform(0, price*0.02)
            low_price = min(open_price, close_price) - random.uniform(0, price*0.02)
            volume = random.uniform(1000, 1000000)
            
            results.append({
                "ticker": ticker,
                "timestamp": ctime.isoformat(), # Standard ISO format
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close_price, 4),
                "volume": round(volume, 4)
            })
            price = close_price # Next open is current close
            if price <=0: price = random.uniform(1,5) # Reset if price goes to 0 or negative
            ctime += delta
            if len(results) > 2000 and len(tickers) > 1: # Safety break for too much dummy data for multi tickers
                break
        if len(results) > 5000: # Overall safety break
            break
    return results

@router.get("/market-data/historical", response_model=List[Dict]) # Using List[Dict] for flexibility now
async def get_historical_market_data(
    tickers: List[str] = Query(..., description="List of asset tickers (e.g., [\"BTCUSDT\", \"ETHUSDT\"])"),
    start_date: datetime = Query(..., description="Start date in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    end_date: datetime = Query(..., description="End date in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    interval: Optional[str] = Query("1h", description="Data interval (e.g., '1m', '1h', '1d')"),
    db: Session = Depends(get_db),
    # current_user: UserModel = Depends(get_current_active_user) # Protect if needed
):
    """
    Fetch historical market data (OHLCV) for a list of assets.
    Authentication can be added if this data is considered sensitive.
    """
    if not tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty.")
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date.")

    # In a real app, you'd parse 'interval' and query your database accordingly.
    # The data source could be a table populated by Celery tasks fetching from Binance/AlphaVantage.
    historical_data = await fetch_historical_data_from_db(db, tickers, start_date, end_date, interval)
    
    if not historical_data:
        # Return empty list if no data, or could be 404 if appropriate for your API design
        return [] 
        # raise HTTPException(status_code=404, detail="No historical market data found for the given parameters.")
    
    return historical_data 