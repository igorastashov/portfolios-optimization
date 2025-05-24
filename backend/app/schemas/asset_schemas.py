from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from backend.app.models.asset_model import AssetType

class AssetBase(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20, description="Ticker symbol of the asset (e.g., AAPL, BTCUSDT)")
    name: str = Field(..., min_length=1, max_length=100, description="Full name of the asset (e.g., Apple Inc., Bitcoin/TetherUS)")
    description: Optional[str] = Field(default=None, max_length=500, description="Optional description of the asset")
    asset_type: AssetType = Field(default=AssetType.OTHER, description="Type of the asset (e.g., STOCK, CRYPTO)")
    data_source: Optional[str] = Field(default=None, max_length=50, description="Primary data source for this asset (e.g., Binance, Yahoo Finance)")

class AssetCreate(AssetBase):
    pass

class AssetUpdate(BaseModel):
    ticker: Optional[str] = Field(default=None, min_length=1, max_length=20, description="New ticker symbol for the asset.")
    name: Optional[str] = Field(default=None, min_length=1, max_length=100, description="New full name for the asset.")
    description: Optional[str] = Field(default=None, max_length=500, description="New description for the asset.")
    asset_type: Optional[AssetType] = Field(default=None, description="New type for the asset.")
    data_source: Optional[str] = Field(default=None, max_length=50, description="New primary data source for the asset.")

class AssetInDBBase(AssetBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Asset(AssetInDBBase):
    pass 