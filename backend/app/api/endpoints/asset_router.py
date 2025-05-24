from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime

from backend.app.db.session import get_db
from backend.app.schemas import asset_schemas as schemas
from backend.app.db.crud import crud_asset
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user
from backend.app.models.asset_model import AssetType

router = APIRouter()


@router.post("/", response_model=schemas.Asset, status_code=status.HTTP_201_CREATED)
def create_asset(
    asset_in: schemas.AssetCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
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
    current_user: UserModel = Depends(get_current_active_user)
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
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Delete an asset. (Protected - requires authentication, ideally admin rights)
    """
    deleted_asset = crud_asset.delete_asset(db, asset_id=asset_id)
    if not deleted_asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return deleted_asset 


async def fetch_historical_data_from_db(
    db: Session, 
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime, 
    interval: str = "1h"
) -> List[Dict]:
    if not tickers:
        raise ValueError("Список тикеров не может быть пустым.")
    if start_date >= end_date:
        raise ValueError("Начальная дата должна быть меньше конечной даты.")
    if interval not in ["1h", "1d"]:
        raise ValueError("Интервал должен быть '1h' (1 час) или '1d' (1 день).")
    
    delta = timedelta(hours=1) if interval == "1h" else timedelta(days=1)
    results = []
    
    for ticker in tickers:
        current_time = start_date
        while current_time <= end_date:
            await asyncio.sleep(0.1)
            record = {
                "ticker": ticker,
                "timestamp": current_time.isoformat(),
                "open": (ticker + current_time.isoformat()) % 50,
                "high": (ticker + current_time.isoformat() + "high") % 60,
                "low": (ticker + current_time.isoformat() + "low") % 40,
                "close": (ticker + current_time.isoformat() + "close") % 55,
                "volume": (ticker + current_time.isoformat() + "volume") % 500000
            }
            results.append(record)
            current_time += delta
            if len(results) > 2000 and len(tickers) > 1:
                break
        if len(results) > 5000:
            break
    
    return results

@router.get("/market-data/historical", response_model=List[Dict])
async def get_historical_market_data(
    tickers: List[str] = Query(..., description="List of asset tickers (e.g., [\"BTCUSDT\", \"ETHUSDT\"])"),
    start_date: datetime = Query(..., description="Start date in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    end_date: datetime = Query(..., description="End date in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    interval: Optional[str] = Query("1h", description="Data interval (e.g., '1m', '1h', '1d')"),
    db: Session = Depends(get_db),
):
    """
    Fetch historical market data (OHLCV) for a list of assets.
    """
    if not tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty.")
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date.")

    historical_data = await fetch_historical_data_from_db(db, tickers, start_date, end_date, interval)
    
    if not historical_data:
        return [] 
    
    return historical_data 