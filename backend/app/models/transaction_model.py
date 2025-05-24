from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Numeric, Enum as SAEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from backend.app.models.user_model import Base

class TransactionType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    # Other types like DIVIDEND, FEE, etc., can be added if needed.

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    # owner_id helps in directly querying transactions for a user, supplementing portfolio-based ownership checks.
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False) 

    transaction_type = Column(SAEnum(TransactionType), nullable=False)
    quantity = Column(Numeric(19, 8), nullable=False)  # Quantity of the asset transacted, with high precision
    price_per_unit = Column(Numeric(19, 8), nullable=False) # Price per unit of the asset at the time of transaction
    transaction_date = Column(DateTime, nullable=False, default=datetime.utcnow) # Date and time of the transaction
    
    fees = Column(Numeric(19, 8), nullable=True, default=0) # Any fees associated with the transaction
    notes = Column(String, nullable=True) # Optional user notes for the transaction

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    asset = relationship("Asset", back_populates="transactions")
    owner = relationship("User", back_populates="transactions")

# No need to add reverse relationships in Portfolio, Asset, and User models as they are already set up correctly 