from pydantic import BaseModel, Field
from typing import List, Any, Optional

# --- CatBoost Price Prediction ---

class PricePredictionFeatures(BaseModel):
    """
    Defines the structure for raw input features for CatBoost price prediction.
    The ClearML Serving function is responsible for any necessary transformations 
    (e.g., mean encoding) based on statistics saved during model training.

    Note: The fields below should accurately reflect the raw features your 
    data preparation pipeline produces before any training-fold-specific encoding.
    These are illustrative examples; adapt them to your model's specific feature set.
    """
    # TODO: Define ALL actual raw features your CatBoost model expects.
    # Examples of potential features (uncomment and adapt as needed):
    # Lag features (replace 'Target' with your actual target column name, e.g., "Close")
    # Target_lag_1h: Optional[float] = Field(None, description="Lag 1 of target variable")
    # Target_lag_2h: Optional[float] = Field(None, description="Lag 2 of target variable")
    # ... up to the maximum lag used by the model

    # Raw calendar features (before any model-specific encoding like mean encoding)
    hour_raw: Optional[int] = Field(None, description="Raw hour of the day (0-23). May be used for mean encoding in serving.")
    dayofweek_raw: Optional[int] = Field(None, description="Raw day of the week (e.g., 0-6). May be used for mean encoding in serving.")
    dayofmonth: Optional[int] = Field(None, description="Day of the month (1-31).")
    month: Optional[int] = Field(None, description="Month (1-12).")
    # Add other calendar features if used: year, quarter, etc.

    # News features (as available from the data source before model training)
    avg_overall_sentiment: Optional[float] = Field(None, description="Average overall news sentiment score.")
    # Adjust 'asset_specific' if the naming is different or more generic
    avg_asset_specific_sentiment: Optional[float] = Field(None, description="Average asset-specific news sentiment score.") 
    sum_news_count: Optional[float] = Field(None, description="Total count of news articles considered.") # Or int
    
    # Example for a feature name that might not be a valid Python identifier:
    # news_change_percent: Optional[float] = Field(None, alias="Change %", description="News related change percentage, if applicable.")
    # If used, ensure 'populate_by_name = True' in Config.
    # For simplicity, here we assume valid Python field names for direct features:
    # some_other_news_feature: Optional[float] = Field(None, description="Another processed news feature.")

    # Key considerations for feature definition:
    # 1. All raw features expected by the deployed model (after serving-side pre-processing) must be listed.
    # 2. Mean encoding (e.g., for 'hour_raw', 'dayofweek_raw') should occur within the ClearML serving script,
    #    using statistics (e.g., means) saved from the training pipeline.
    # 3. Features explicitly dropped before training (e.g., 'Open', 'High', 'Low', if configured in cols_to_drop_from_X)
    #    should NOT be included here unless the serving model uses them differently.
    # 4. The target variable itself should not be passed as a feature.

    class Config:
        extra = 'ignore' # Ignores extra fields passed in the request, consider 'forbid' for stricter validation.
        populate_by_name = True # Allows using 'alias' for fields if needed (e.g., for names with spaces).

class PricePredictionRequest(BaseModel):
    asset_ticker: str = Field(..., description="Asset ticker for price prediction, e.g., 'BTCUSDT'.")
    instances: List[PricePredictionFeatures] = Field(..., 
        description="List of raw feature sets for prediction. Each item must conform to PricePredictionFeatures.")

class PricePredictionResponse(BaseModel):
    asset_ticker: str
    predictions: List[Any] = Field(..., description="List of predictions from the CatBoost model. Structure depends on serving_function output.")


# --- DRL Portfolio Rebalancing ---

class DRLRebalancingRequest(BaseModel):
    portfolio_id: str = Field(..., description="ID of the portfolio for which rebalancing is requested.")
    instances: List[List[float]] = Field(..., 
        description="List of observations (states) for the DRL agent. Each observation is a list of floats matching the agent's observation space.",
        example=[[10000.0, 7000.0, 300.0, 10.0, 0.0, 50.5, 25.2, 0.6]] # Example state vector
    )

class DRLRebalancingResponse(BaseModel):
    portfolio_id: str
    predictions: List[List[float]] = Field(..., 
        description="List of action lists from the DRL agent. Each inner list represents actions for one observation.",
        example=[[0.5, -0.2, 0.3]] # Example action vector (e.g., portfolio weights or changes)
    ) 