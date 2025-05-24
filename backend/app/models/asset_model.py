from sqlalchemy import Column, Integer, String, DateTime, func, Enum as SAEnum, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from backend.app.models.user_model import Base

class AssetType(str, enum.Enum):
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    COMMODITY = "COMMODITY"
    OTHER = "OTHER"

class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False) # e.g., AAPL, BTCUSDT
    name = Column(String, nullable=False) # e.g., Apple Inc., Bitcoin / TetherUS
    description = Column(Text, nullable=True) # Detailed description of the asset
    asset_type = Column(SAEnum(AssetType), nullable=False, default=AssetType.OTHER) # Type of the financial asset
    data_source = Column(String, nullable=True) # Source from which data for this asset is obtained, e.g., Binance, Yahoo Finance
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to Transactions involving this asset
    transactions = relationship("Transaction", back_populates="asset")

    # Other fields like currency, exchange, ISIN, etc., can be added as needed. 