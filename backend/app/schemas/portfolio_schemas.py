from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal

# Forward references for relationships defined in other schema files
# It's good practice to import them if they are used directly for type hinting complex structures,
# but often Pydantic handles forward references using string type hints.
# from backend.app.schemas.auth_schemas import User
# from backend.app.schemas.asset_schemas import Asset
# from backend.app.schemas.transaction_schemas import Transaction


# --- Portfolio Schemas ---
class PortfolioBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the portfolio")
    description: Optional[str] = Field(None, max_length=500, description="Optional description of the portfolio")
    currency_code: str = Field("USD", description="Base currency code of the portfolio (e.g., USD, EUR)")

class PortfolioCreate(PortfolioBase):
    # owner_id will be set by the endpoint using the authenticated current_user
    pass

class PortfolioUpdate(PortfolioBase):
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="New name for the portfolio")
    description: Optional[str] = Field(None, max_length=500, description="New description for the portfolio")
    currency_code: Optional[str] = Field(None, description="New base currency code for the portfolio")


class PortfolioInDBBase(PortfolioBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime
    # owner: User # Uncomment if owner details are needed directly in this base or derived schemas like Portfolio
    
    class Config:
        from_attributes = True # Enables ORM mode for Pydantic v2

class Portfolio(PortfolioInDBBase):
    # This schema can be extended with additional fields if needed for specific responses.
    pass

class PortfolioWithRelations(PortfolioInDBBase):
    owner: 'User' # Information about the owner (User schema defined elsewhere)
    assets: List['Asset'] = [] # Associated assets (Asset schema defined elsewhere)
    transactions: List['Transaction'] = [] # Associated transactions (Transaction schema defined elsewhere)

# --- Schemas for Portfolio Summary Calculations ---
class PortfolioAssetSummarySchema(BaseModel):
    ticker: str
    quantity: Decimal
    average_buy_price: Decimal
    current_market_price: Decimal
    current_value: Decimal
    pnl: Decimal
    pnl_percent: float # P&L percentage; can be 0.0 if average_buy_price is zero

class PortfolioSummarySchema(BaseModel):
    portfolio_id: int
    portfolio_name: str
    currency_code: str
    assets: List[PortfolioAssetSummarySchema]
    total_portfolio_value: Decimal
    total_invested_value: Decimal # Sum of (quantity * average_buy_price) for all assets
    overall_pnl: Decimal # total_portfolio_value - total_invested_value
    overall_pnl_percent: float # Overall P&L percentage; can be 0.0 if total_invested_value is zero
    total_value_24h_change_abs: Optional[Decimal] = Field(None, description="Absolute change in total portfolio value over the last 24 hours")
    total_value_24h_change_pct: Optional[float] = Field(None, description="Percentage change in total portfolio value over the last 24 hours")

# --- Schema for Requesting Portfolio Analysis/Backtest ---
class PortfolioAnalysisRequest(BaseModel):
    portfolio_id: int = Field(..., description="ID of the portfolio to analyze or backtest")
    # Deprecated individual fields below in favor of a flexible dictionary.
    # start_date: Optional[date] = None
    # end_date: Optional[date] = None
    # strategy_name: Optional[str] = None
    # strategy_params: Optional[Dict[str, Any]] = None
    # initial_capital: Optional[float] = None
    analysis_parameters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Flexible dictionary for various analysis parameters (e.g., start_date, end_date, initial_capital, strategy settings)."
    )

    class Config:
        from_attributes = True # For compatibility if created from ORM models

# --- Schema for Portfolio Summary (used in /portfolios/me/summary) ---
class AssetSummary(BaseModel): # This seems to be a slightly different summary, perhaps for a specific UI component
    ticker: str
    quantity: float # Consider using Decimal for financial precision
    average_buy_price: float # Consider using Decimal
    current_price: float # Consider using Decimal
    current_value: float # Consider using Decimal
    pnl: float # Consider using Decimal

class PortfolioSummaryResponse(BaseModel): # This also seems distinct from PortfolioSummarySchema
    assets: List[AssetSummary]
    total_portfolio_value: float # Consider using Decimal
    total_portfolio_pnl: float # Consider using Decimal
    total_value_24h_change_abs: Optional[float] = Field(None, description="Absolute 24h change in portfolio value") # Consider Decimal
    total_value_24h_change_pct: Optional[float] = Field(None, description="Percentage 24h change in portfolio value")


# --- Schemas for Hypothetical Portfolio Simulation --- 
class HypotheticalPortfolioAssetInput(BaseModel): # Moved up to be defined before use
    ticker: str = Field(..., description="Asset ticker symbol (e.g., AAPL, BTCUSDT)")
    weight: float = Field(..., gt=0, le=1, description="Weight of the asset in the portfolio (a fraction between 0 and 1)")

class HypotheticalPortfolioSimulationRequest(BaseModel):
    initial_capital: Decimal = Field(..., gt=Decimal(0), description="Initial capital for the simulation")
    assets_weights: List[HypotheticalPortfolioAssetInput] = Field(..., min_length=1, description="List of assets and their respective weights")
    start_date: date = Field(..., description="Start date for the simulation period")
    end_date: date = Field(..., description="End date for the simulation period")
    rebalancing_frequency: str = Field("none", description="Rebalancing frequency (e.g., 'none', 'monthly', 'quarterly', 'annually')")
    commission_rate: Decimal = Field(Decimal(0), ge=Decimal(0), lt=Decimal(0.1), description="Transaction commission rate (e.g., 0.001 for 0.1%)")

    @model_validator(mode='before')
    def validate_dates(cls, values: Any) -> Any:
        if isinstance(values, dict):
            start_date, end_date = values.get('start_date'), values.get('end_date')
            if start_date and end_date and start_date >= end_date:
                raise ValueError("Start date must be before end date.")
        return values

    @model_validator(mode='before')
    def validate_asset_weights_sum(cls, values: Any) -> Any:
        if isinstance(values, dict):
            assets_weights_data = values.get('assets_weights')
            if assets_weights_data and not isinstance(assets_weights_data, list):
                 raise ValueError("'assets_weights' must be a list of asset-weight pairs.")
            if assets_weights_data:
                # Handle both dicts (from JSON) and Pydantic model instances (internal calls)
                total_weight = sum(
                    Decimal(str(item.get('weight', 0))) if isinstance(item, dict) else Decimal(str(item.weight))
                    for item in assets_weights_data
                )
                # Using Decimal for comparison to avoid float precision issues
                if not (Decimal('0.999') <= total_weight <= Decimal('1.001')):
                    raise ValueError(f"The sum of asset weights must be close to 1.0 (current sum: {total_weight:.4f}).")
        return values

    class Config:
        extra = 'forbid' # Disallow any extra fields not defined in the schema
        # from_attributes = True # If this schema might be created from an ORM model

class HypotheticalPortfolioSimulationResponse(BaseModel):
    # user_id: int # Not clear if user_id is part of the direct response, or handled at endpoint level
    simulation_parameters: HypotheticalPortfolioSimulationRequest
    metrics: Dict[str, Any] # e.g., {"CAGR": 0.15, "Sharpe Ratio": 1.2, "Max Drawdown": -0.20}
    # plot_data: Optional[Dict[str, List[Any]]] = None # Optional: Data for plotting, e.g., time series of portfolio value

# --- Schemas for Portfolio Value History ---
class PortfolioValueHistoryPoint(BaseModel):
    date: date
    value: Decimal

class PortfolioValueHistoryResponse(BaseModel):
    history: List[PortfolioValueHistoryPoint]
    start_date: date
    end_date: date
    currency_code: str

# Forward-referencing User, Asset, Transaction in PortfolioWithRelations
# This helps Pydantic resolve types that are defined later or in other modules.
# It is automatically handled if they are imported in the global scope or if PortfolioWithRelations
# is defined after them, or by using string literals for type hints.
# User.model_rebuild() # No longer needed with Pydantic v2 if using string type hints
# Asset.model_rebuild()
# Transaction.model_rebuild()
# Portfolio.model_rebuild() # Generally not needed for the schema itself unless complex recursive types 