from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class RebalancePortfolioRequest(BaseModel):
    portfolio_id: int = Field(..., description="ID of the portfolio to be rebalanced.")
    # Additional parameters for the rebalancing request can be added here, for example:
    # strategy_name: Optional[str] = Field(default=None, description="Name of the rebalancing strategy to use (e.g., 'drl_ppo', 'target_weights').")
    # strategy_params: Optional[Dict[str, Any]] = Field(default=None, description="Parameters specific to the chosen rebalancing strategy.")

# The response for initiating a rebalancing task typically uses the generic TaskResponse schema
# (defined in common_schemas.py or a similar utility location),
# which includes fields like task_id and status. 