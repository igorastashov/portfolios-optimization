import httpx
from fastapi import APIRouter, HTTPException, Depends, Path, Body
from typing import List

from backend.app import schemas
from backend.app.core.config import settings

inference_router = APIRouter()


async def get_inference_client():
    async with httpx.AsyncClient(timeout=30.0) as client: 
        yield client

@inference_router.post(
    "/predict/price", 
    response_model=schemas.PricePredictionResponse,
    summary="Predict asset price using CatBoost model",
    description="Takes asset ticker and features, and returns the predicted price from the deployed CatBoost model via ClearML Serving."
)
async def predict_price(
    request_data: schemas.PricePredictionRequest = Body(...),
    client: httpx.AsyncClient = Depends(get_inference_client)
):
    if not settings.CATBOOST_INFERENCE_SERVICE_URL:
        raise HTTPException(status_code=503, detail="CatBoost inference service URL is not configured.")

    payload_instances = [instance.model_dump(exclude_none=True) for instance in request_data.instances]

    payload = {"instances": payload_instances}

    try:
        response = await client.post(settings.CATBOOST_INFERENCE_SERVICE_URL, json=payload)
        response.raise_for_status() 
        
        prediction_result = response.json() 

        if "predictions" not in prediction_result:
            raise HTTPException(status_code=502, detail=f"Invalid response format from CatBoost inference service: 'predictions' field missing. Response: {prediction_result}")

        return schemas.PricePredictionResponse(
            asset_ticker=request_data.asset_ticker,
            predictions=prediction_result["predictions"]
        )

    except httpx.HTTPStatusError as e:
        error_detail = f"CatBoost inference service returned error {e.response.status_code}."
        try:
            error_content = e.response.json()
            error_detail += f" Details: {error_content}"
        except Exception:
            error_detail += f" Content: {e.response.text}"
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=f"Error connecting to CatBoost inference service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while calling CatBoost inference service: {str(e)}")


@inference_router.post(
    "/predict/rebalance", 
    response_model=schemas.DRLRebalancingResponse,
    summary="Get DRL portfolio rebalancing actions",
    description="Takes portfolio ID and current state (observation), and returns rebalancing actions from the deployed DRL model via ClearML Serving."
)
async def predict_rebalancing_actions(
    request_data: schemas.DRLRebalancingRequest = Body(...),
    client: httpx.AsyncClient = Depends(get_inference_client)
):
    if not settings.DRL_INFERENCE_SERVICE_URL:
        raise HTTPException(status_code=503, detail="DRL inference service URL is not configured.")

    payload_instances = [item.observation for item in request_data.instances]

    payload_instances = [item.observation for item in request_data.instances]
    payload = {"instances": payload_instances}
    
    try:
        response = await client.post(settings.DRL_INFERENCE_SERVICE_URL, json=payload)
        response.raise_for_status()
        
        prediction_result = response.json()

        if "predictions" not in prediction_result:
            raise HTTPException(status_code=502, detail=f"Invalid response format from DRL inference service: 'predictions' field missing. Response: {prediction_result}")

        return schemas.DRLRebalancingResponse(
            portfolio_id=request_data.portfolio_id,
            predictions=prediction_result["predictions"]
        )
        
    except httpx.HTTPStatusError as e:
        error_detail = f"DRL inference service returned error {e.response.status_code}."
        try:
            error_content = e.response.json()
            error_detail += f" Details: {error_content}"
        except Exception:
            error_detail += f" Content: {e.response.text}"
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=f"Error connecting to DRL inference service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while calling DRL inference service: {str(e)}") 