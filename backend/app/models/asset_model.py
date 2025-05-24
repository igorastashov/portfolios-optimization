from sqlalchemy import Column, Integer, String, DateTime, func, Enum as SAEnum, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from backend.app.models.user_model import Base

class AssetType(str, enum.Enum):
    STOCK = "STOCK"


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    asset_type = Column(SAEnum(AssetType), nullable=False, default=AssetType.OTHER)
    data_source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transactions = relationship("Transaction", back_populates="asset")