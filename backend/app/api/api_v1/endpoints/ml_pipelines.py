from fastapi import APIRouter, HTTPException, Body, Depends
from typing import List, Optional

from backend.app import schemas
from backend.app.api import deps
from backend.app.models import User # For dependency injection if needed for auth
from backend.worker.tasks import run_clearml_pipeline_task # Our new Celery task

# Define paths to the ClearML pipeline definition scripts relative to the project root
# These must match the paths expected by the Celery task's CWD
CATBOOST_PIPELINE_SCRIPT = "backend/ml_models/clearml_pipelines/catboost_training_pipeline.py"
DRL_PIPELINE_SCRIPT = "backend/ml_models/clearml_pipelines/drl_training_pipeline.py"
DATA_UPDATE_PIPELINE_SCRIPT = "backend/data_pipelines/clearml_pipelines/data_update_pipeline.py"

ml_pipelines_router = APIRouter()

@ml_pipelines_router.post("/catboost/start", response_model=schemas.CeleryTaskResponse,
                            summary="Start CatBoost Training Pipeline",
                            description="Initiates a ClearML pipeline for CatBoost model training. \
                                         Optionally, provide a list of target asset tickers to override the default configuration.")
async def start_catboost_pipeline(
    *, 
    target_asset_tickers: Optional[List[str]] = Body(None, embed=True, description="List of asset tickers, e.g., [\"BTCUSDT\", \"ETHUSDT\"]"),
    # current_user: User = Depends(deps.get_current_active_user) # Uncomment if auth is needed
):
    """
    Starts the CatBoost model training pipeline via Celery.

    - **target_asset_tickers**: Optional list of asset tickers to train models for. 
      If not provided, the pipeline will use its default configuration.
    """
    pipeline_args = []
    if target_asset_tickers:
        # Hydra expects list format like '[\"TICKER1\",\"TICKER2\"]' or just TICKER1,TICKER2 for some overrides.
        # For `OmegaConf.from_dotlist(["foo.bar='[1, 2, 3]'"])` it works.
        # Let's format it as a string representation of a list that Hydra can parse.
        tickers_str = str(target_asset_tickers).replace(" ", "") # Ensure no spaces for Hydra list parsing
        pipeline_args.append(f"pipeline_params.target_asset_tickers={tickers_str}")
    
    # Ensure the pipeline definition script itself doesn't try to run the pipeline, Celery will just define it.
    pipeline_args.append("run_pipeline_locally_after_definition=false")

    try:
        task_result = run_clearml_pipeline_task.delay(pipeline_script_path=CATBOOST_PIPELINE_SCRIPT, pipeline_args=pipeline_args)
        return schemas.CeleryTaskResponse(task_id=task_result.id, message="CatBoost training pipeline task initiated.")
    except Exception as e:
        # This would typically be an issue with Celery broker connection
        raise HTTPException(status_code=500, detail=f"Failed to enqueue CatBoost pipeline task: {str(e)}")

@ml_pipelines_router.post("/drl/start", response_model=schemas.CeleryTaskResponse,
                            summary="Start DRL Training Pipeline",
                            description="Initiates a ClearML pipeline for DRL model training. \
                                         Optionally, provide a portfolio_id to override the default configuration.")
async def start_drl_pipeline(
    *, 
    portfolio_id: Optional[str] = Body(None, embed=True, description="Portfolio ID for which to run the DRL pipeline, e.g., CRYPTO_MAJOR_CAPS"),
    # current_user: User = Depends(deps.get_current_active_user)
):
    """
    Starts the DRL model training pipeline via Celery.

    - **portfolio_id**: Optional portfolio identifier. 
      If not provided, the pipeline will use its default configuration.
    """
    pipeline_args = []
    if portfolio_id:
        pipeline_args.append(f"pipeline_params.portfolio_id={portfolio_id}")
    pipeline_args.append("run_pipeline_locally_after_definition=false")

    try:
        task_result = run_clearml_pipeline_task.delay(pipeline_script_path=DRL_PIPELINE_SCRIPT, pipeline_args=pipeline_args)
        return schemas.CeleryTaskResponse(task_id=task_result.id, message="DRL training pipeline task initiated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue DRL pipeline task: {str(e)}")

@ml_pipelines_router.post("/data-update/start", response_model=schemas.CeleryTaskResponse,
                            summary="Start Data Update Pipeline",
                            description="Initiates a ClearML pipeline for updating market and news data.")
async def start_data_update_pipeline(
    # *, # Removed asterisk as there are no preceding positional-only arguments when auth is commented out
    # Add specific params if needed, e.g., force_update_for_tickers: Optional[List[str]] = Body(None, embed=True)
    # current_user: User = Depends(deps.get_current_active_user)
):
    """
    Starts the data update pipeline via Celery.
    """
    pipeline_args = []
    # Example: enable specific data sources via args if needed
    # pipeline_args.append("data_sources.binance.enabled=true")
    # pipeline_args.append("data_sources.alphavantage_news.enabled=true")
    pipeline_args.append("run_pipeline_locally_after_definition=false")

    try:
        task_result = run_clearml_pipeline_task.delay(pipeline_script_path=DATA_UPDATE_PIPELINE_SCRIPT, pipeline_args=pipeline_args)
        return schemas.CeleryTaskResponse(task_id=task_result.id, message="Data update pipeline task initiated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue data update pipeline task: {str(e)}")

# TODO: Add an endpoint to check the status of a Celery task given its ID
# This endpoint would query Celery backend for task status and metadata (including clearml_pipeline_id)
# @ml_pipelines_router.get("/task-status/{task_id}", response_model=schemas.CeleryTaskStatus) # Define CeleryTaskStatus schema
# async def get_task_status(task_id: str):
#     task = run_clearml_pipeline_task.AsyncResult(task_id)
#     response = {
#         "task_id": task_id,
#         "status": task.status,
#         "result": task.result, # Contains what the task returned (or error if failed)
#         "info": task.info,     # Contains metadata set by update_state (like clearml_pipeline_id)
#     }
#     return response 