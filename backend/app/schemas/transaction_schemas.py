from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal



class TransactionBase(BaseModel):
    asset_ticker: str = Field(..., description="Ticker symbol of the asset (e.g., AAPL, BTCUSDT).")
    transaction_type: str = Field(..., description="Type of transaction, e.g., 'BUY' or 'SELL'.") 
    quantity: Decimal = Field(..., gt=Decimal(0), description="Quantity of the asset transacted.")
    price: Decimal = Field(..., gt=Decimal(0), description="Price per unit of the asset.")
    fee: Optional[Decimal] = Field(Decimal("0.0"), ge=Decimal(0), description="Optional transaction fee.")
    transaction_date: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Actual date and time of the transaction.") 

    class Config:
        from_attributes = True

class TransactionCreate(TransactionBase):
    pass

class TransactionUpdate(BaseModel):
    asset_ticker: Optional[str] = Field(None, description="New ticker symbol of the asset.")
    transaction_type: Optional[str] = Field(None, description="New type of transaction (e.g., 'BUY', 'SELL'.")
    quantity: Optional[Decimal] = Field(None, gt=Decimal(0), description="New quantity of the asset transacted.")
    price: Optional[Decimal] = Field(None, gt=Decimal(0), description="New price per unit of the asset.")
    fee: Optional[Decimal] = Field(None, ge=Decimal(0), description="New transaction fee.")
    transaction_date: Optional[datetime] = Field(None, description="New date and time of the transaction.")

    class Config:
        from_attributes = True

class TransactionPublic(TransactionBase):
    id: int
    portfolio_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class TransactionDeleteResponse(BaseModel):
    message: str = "Transaction deleted successfully"
    transaction_id: int


class AssetForTransaction(BaseModel):
    id: int
    ticker_symbol: str
    name: Optional[str] = None

    class Config:
        orm_mode = True

class PortfolioForTransaction(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True