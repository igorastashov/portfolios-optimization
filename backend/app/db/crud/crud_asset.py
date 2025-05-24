from sqlalchemy.orm import Session
from typing import List, Optional

from backend.app.models.asset_model import Asset
from backend.app.schemas.asset_schemas import AssetCreate, AssetUpdate

def get_asset(db: Session, asset_id: int) -> Optional[Asset]:
    """Retrieve an asset by its ID."""
    return db.query(Asset).filter(Asset.id == asset_id).first()

def get_asset_by_ticker(db: Session, ticker: str) -> Optional[Asset]:
    """Retrieve an asset by its ticker symbol."""
    return db.query(Asset).filter(Asset.ticker == ticker).first()

def get_assets(
    db: Session, skip: int = 0, limit: int = 100, asset_type: Optional[str] = None
) -> List[Asset]:
    """Retrieve a list of assets with pagination, optionally filtered by asset type."""
    query = db.query(Asset)
    if asset_type:
        query = query.filter(Asset.asset_type == asset_type)
    return query.offset(skip).limit(limit).all()

def create_asset(db: Session, asset: AssetCreate) -> Asset:
    """Create a new asset in the database."""
    db_asset = Asset(**asset.model_dump())
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    return db_asset

def update_asset(db: Session, asset_id: int, asset_in: AssetUpdate) -> Optional[Asset]:
    """Update an existing asset's information."""
    db_asset = get_asset(db, asset_id=asset_id)
    if not db_asset:
        return None
    
    update_data = asset_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_asset, field, value)
    
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    return db_asset

def delete_asset(db: Session, asset_id: int) -> Optional[Asset]:
    """Delete an asset from the database by its ID."""
    db_asset = get_asset(db, asset_id=asset_id)
    if not db_asset:
        return None
    db.delete(db_asset)
    db.commit()
    return db_asset 