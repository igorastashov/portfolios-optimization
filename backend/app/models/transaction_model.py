from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Numeric, Enum as SAEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from backend.app.models.user_model import Base

class TransactionType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False) 

    transaction_type = Column(SAEnum(TransactionType), nullable=False)
    quantity = Column(Numeric(19, 8), nullable=False)
    price_per_unit = Column(Numeric(19, 8), nullable=False)
    transaction_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    fees = Column(Numeric(19, 8), nullable=True, default=0)
    notes = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="transactions")
    asset = relationship("Asset", back_populates="transactions")
    owner = relationship("User", back_populates="transactions")
    