from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal

# TransactionType from the model is not directly used here;
# type is validated as a string (e.g., "BUY", "SELL") in API logic or by Enum in model.

class TransactionBase(BaseModel):
    asset_ticker: str = Field(..., description="Ticker symbol of the asset (e.g., AAPL, BTCUSDT).")
    # It's recommended to use an Enum for transaction_type for stricter validation.
    # However, using a string here might be for flexibility or specific design choices.
    transaction_type: str = Field(..., description="Type of transaction, e.g., 'BUY' or 'SELL'.") 
    quantity: Decimal = Field(..., gt=Decimal(0), description="Quantity of the asset transacted.")
    price: Decimal = Field(..., gt=Decimal(0), description="Price per unit of the asset.")
    fee: Optional[Decimal] = Field(Decimal("0.0"), ge=Decimal(0), description="Optional transaction fee.")
    # User-provided actual date and time of the transaction; defaults to now if not provided during creation.
    transaction_date: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Actual date and time of the transaction.") 

    class Config:
        from_attributes = True # Enables ORM mode for Pydantic v2

class TransactionCreate(TransactionBase):
    # portfolio_id is typically injected by the API endpoint based on the authenticated user
    # or a path parameter, rather than being part of the request body for this specific schema.
    pass

class TransactionUpdate(BaseModel): # Using BaseModel allows for partial updates
    asset_ticker: Optional[str] = Field(None, description="New ticker symbol of the asset.")
    transaction_type: Optional[str] = Field(None, description="New type of transaction (e.g., 'BUY', 'SELL'.")
    quantity: Optional[Decimal] = Field(None, gt=Decimal(0), description="New quantity of the asset transacted.")
    price: Optional[Decimal] = Field(None, gt=Decimal(0), description="New price per unit of the asset.")
    fee: Optional[Decimal] = Field(None, ge=Decimal(0), description="New transaction fee.")
    transaction_date: Optional[datetime] = Field(None, description="New date and time of the transaction.")

    class Config:
        from_attributes = True # Enables ORM mode for Pydantic v2

class TransactionPublic(TransactionBase):
    id: int
    portfolio_id: int
    # transaction_date is inherited from TransactionBase and represents the actual event time.
    created_at: datetime   # Timestamp of when the transaction record was created in the database.
    updated_at: Optional[datetime] = None # Timestamp of the last update to the transaction record.

    class Config:
        from_attributes = True # Enables ORM mode for Pydantic v2

# The TransactionDeleteResponse was defined in __init__.py but not here.
# If it's a simple message, it might be in common_schemas.py or defined directly in the endpoint.
# For consistency, if it's specific to transactions, it could be here:
# class TransactionDeleteResponse(BaseModel):
#     message: str = "Transaction deleted successfully"
#     transaction_id: int

# Schemas for Asset and Portfolio if they are simple and not defined elsewhere for this context
# These are just examples if you need to return nested structures within transaction responses.
# Usually, you'd import them from their respective schema files (asset_schemas.py, portfolio_schemas.py)

class AssetForTransaction(BaseModel):
    id: int
    ticker_symbol: str
    name: Optional[str] = None
    # Add other relevant asset fields if needed

    class Config:
        orm_mode = True
        # model_config = {"from_attributes": True}

class PortfolioForTransaction(BaseModel):
    id: int
    name: str
    # Add other relevant portfolio fields if needed

    class Config:
        orm_mode = True
        # model_config = {"from_attributes": True}

# Example of TransactionPublic with nested Asset (if needed by frontend)
# class TransactionPublicWithAsset(TransactionPublic):
#     asset_details: Optional[AssetForTransaction] = None 