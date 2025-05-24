from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal


class PortfolioBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the portfolio")
    description: Optional[str] = Field(None, max_length=500, description="Optional description of the portfolio")
    currency_code: str = Field("USD", description="Base currency code of the portfolio (e.g., USD, EUR)")

class PortfolioCreate(PortfolioBase):
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
    owner: User
    
    class Config:
        from_attributes = True

class Portfolio(PortfolioInDBBase):
    pass

class PortfolioWithRelations(PortfolioInDBBase):
    owner: 'User'
    assets: List['Asset'] = []
    transactions: List['Transaction'] = []

class PortfolioAssetSummarySchema(BaseModel):
    ticker: str
    quantity: Decimal
    average_buy_price: Decimal
    current_market_price: Decimal
    current_value: Decimal
    pnl: Decimal
    pnl_percent: float

class PortfolioSummarySchema(BaseModel):
    portfolio_id: int
    portfolio_name: str
    currency_code: str
    assets: List[PortfolioAssetSummarySchema]
    total_portfolio_value: Decimal
    total_invested_value: Decimal
    overall_pnl: Decimal
    overall_pnl_percent: float
    total_value_24h_change_abs: Optional[Decimal] = Field(None, description="Absolute change in total portfolio value over the last 24 hours")
    total_value_24h_change_pct: Optional[float] = Field(None, description="Percentage change in total portfolio value over the last 24 hours")

class PortfolioAnalysisRequest(BaseModel):
    portfolio_id: int
    analysis_parameters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Flexible dictionary for various analysis parameters (e.g., start_date, end_date, initial_capital, strategy settings)."
    )

    class Config:
        from_attributes = True

class AssetSummary(BaseModel):
    ticker: str
    quantity: float
    average_buy_price: float
    current_price: float
    current_value: float
    pnl: float

class PortfolioSummaryResponse(BaseModel):
    assets: List[AssetSummary]
    total_portfolio_value: float
    total_portfolio_pnl: float
    total_value_24h_change_abs: Optional[float] = Field(None, description="Absolute 24h change in portfolio value")
    total_value_24h_change_pct: Optional[float] = Field(None, description="Percentage 24h change in portfolio value")


class HypotheticalPortfolioAssetInput(BaseModel):
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
                total_weight = sum(
                    Decimal(str(item.get('weight', 0))) if isinstance(item, dict) else Decimal(str(item.weight))
                    for item in assets_weights_data
                )
                if not (Decimal('0.999') <= total_weight <= Decimal('1.001')):
                    raise ValueError(f"The sum of asset weights must be close to 1.0 (current sum: {total_weight:.4f}).")
        return values

    class Config:
        extra = 'forbid'

class HypotheticalPortfolioSimulationResponse(BaseModel):
    simulation_parameters: HypotheticalPortfolioSimulationRequest
    metrics: Dict[str, Any]

class PortfolioValueHistoryPoint(BaseModel):
    date: date
    value: Decimal

class PortfolioValueHistoryResponse(BaseModel):
    history: List[PortfolioValueHistoryPoint]
    start_date: date
    end_date: date
    currency_code: str
