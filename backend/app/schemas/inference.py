from pydantic import BaseModel, Field
from typing import List, Any, Optional

class PricePredictionFeatures(BaseModel):

    hour_raw: Optional[int] = Field(None, description="Raw hour of the day (0-23). May be used for mean encoding in serving.")
    dayofweek_raw: Optional[int] = Field(None, description="Raw day of the week (e.g., 0-6). May be used for mean encoding in serving.")
    dayofmonth: Optional[int] = Field(None, description="Day of the month (1-31).")
    month: Optional[int] = Field(None, description="Month (1-12).")

    avg_overall_sentiment: Optional[float] = Field(None, description="Average overall news sentiment score.")
    avg_asset_specific_sentiment: Optional[float] = Field(None, description="Average asset-specific news sentiment score.") 
    sum_news_count: Optional[float] = Field(None, description="Total count of news articles considered.")

    class Config:
        extra = 'ignore'
        populate_by_name = True

class PricePredictionRequest(BaseModel):
    asset_ticker: str = Field(..., description="Asset ticker for price prediction, e.g., 'BTCUSDT'.")
    instances: List[PricePredictionFeatures] = Field(..., 
        description="List of raw feature sets for prediction. Each item must conform to PricePredictionFeatures.")

class PricePredictionResponse(BaseModel):
    asset_ticker: str
    predictions: List[Any] = Field(..., description="List of predictions from the CatBoost model. Structure depends on serving_function output.")


class DRLRebalancingRequest(BaseModel):
    portfolio_id: str = Field(..., description="ID of the portfolio for which rebalancing is requested.")
    instances: List[List[float]] = Field(
        description="List of observations (states) for the DRL agent. Each observation is a list of floats matching the agent's observation space."
    )

class DRLRebalancingResponse(BaseModel):
    portfolio_id: str
    predictions: List[List[float]] = Field(..., 
        description="List of action lists from the DRL agent. Each inner list represents actions for one observation.",
    ) 