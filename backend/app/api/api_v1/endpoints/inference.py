import httpx
from fastapi import APIRouter, HTTPException, Depends, Path, Body
from typing import List

from backend.app import schemas
from backend.app.core.config import settings
# from backend.app.api import deps # Если понадобится аутентификация для этих эндпоинтов

inference_router = APIRouter()

# HTTP клиент для взаимодействия с inference сервисами ClearML
# Рекомендуется создавать один клиент на приложение или использовать lifespan для управления им
# https://www.python-httpx.org/advanced/#client-instances
# Для простоты пока создадим его здесь. В продакшене лучше управлять им через lifespan.
async def get_inference_client():
    # Можно добавить таймауты и другие настройки
    # transport = httpx.AsyncHTTPTransport(retries=2)
    # client = httpx.AsyncClient(transport=transport)
    async with httpx.AsyncClient(timeout=30.0) as client: # 30 секунд таймаут
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
    # current_user: models.User = Depends(deps.get_current_active_user) # Если нужна аутентификация
):
    if not settings.CATBOOST_INFERENCE_SERVICE_URL:
        raise HTTPException(status_code=503, detail="CatBoost inference service URL is not configured.")

    # request_data.instances is List[schemas.PricePredictionFeatures]
    # We need to convert each PricePredictionFeatures object into a dictionary 
    # that ClearML Serving expects (a flat dictionary of feature_name: value).
    # Pydantic's .model_dump() (or .dict() in Pydantic v1) can be used here.
    # We should pass exclude_unset=True if we only want to send features that were actually provided in the request,
    # or rely on the serving function to handle missing features if appropriate (e.g. with default values or imputation).
    # For now, let's send all fields defined in the schema (with their defaults if not provided).
    payload_instances = [instance.model_dump(exclude_none=True) for instance in request_data.instances]
    # exclude_none=True will omit fields that are None, which might be desired if your model handles missing features gracefully
    # or if you only want to send features that have values.
    # If your serving function expects all features listed in PricePredictionFeatures (even if None), remove exclude_none=True.

    payload = {"instances": payload_instances}

    try:
        response = await client.post(settings.CATBOOST_INFERENCE_SERVICE_URL, json=payload)
        response.raise_for_status() # Выбросит исключение для 4xx/5xx ответов
        
        prediction_result = response.json() # ClearML Serving возвращает JSON

        # Ожидаемый формат ответа от ClearML Serving: {"predictions": [result1, result2, ...]}
        # result1 может быть числом, словарем и т.д., в зависимости от postprocessing функции модели.
        if "predictions" not in prediction_result:
            raise HTTPException(status_code=502, detail=f"Invalid response format from CatBoost inference service: 'predictions' field missing. Response: {prediction_result}")

        return schemas.PricePredictionResponse(
            asset_ticker=request_data.asset_ticker,
            predictions=prediction_result["predictions"]
        )

    except httpx.HTTPStatusError as e:
        # Ошибка от самого inference сервиса (4xx, 5xx)
        error_detail = f"CatBoost inference service returned error {e.response.status_code}."
        try:
            error_content = e.response.json()
            error_detail += f" Details: {error_content}"
        except Exception:
            error_detail += f" Content: {e.response.text}"
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        # Ошибка сети, DNS, таймаут и т.д.
        raise HTTPException(status_code=504, detail=f"Error connecting to CatBoost inference service: {str(e)}")
    except Exception as e:
        # Любые другие ошибки (например, при парсинге JSON)
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
    # current_user: models.User = Depends(deps.get_current_active_user) # Если нужна аутентификация
):
    if not settings.DRL_INFERENCE_SERVICE_URL:
        raise HTTPException(status_code=503, detail="DRL inference service URL is not configured.")

    # DRL модели (например, из Stable Baselines3), развернутые с ClearML Serving,
    # обычно ожидают на вход список наблюдений (состояний).
    # {"instances": [ [obs1_feature1, obs1_feature2, ...], [obs2_feature1, obs2_feature2, ...] ]}
    # или {"inputs": [ [obs1_feature1, ...], ... ]}
    # Наша схема DRLRebalancingRequest.instances -> List[DRLRebalancingState]
    # DRLRebalancingState.observation -> List[Any]

    payload_instances = [item.observation for item in request_data.instances]
    payload = {"instances": payload_instances}
    
    try:
        response = await client.post(settings.DRL_INFERENCE_SERVICE_URL, json=payload)
        response.raise_for_status()
        
        prediction_result = response.json()

        # Ожидаемый формат ответа: {"predictions": [ [action1_for_obs1, action2_for_obs1,...], [action1_for_obs2,...] ]}
        if "predictions" not in prediction_result:
            raise HTTPException(status_code=502, detail=f"Invalid response format from DRL inference service: 'predictions' field missing. Response: {prediction_result}")

        return schemas.DRLRebalancingResponse(
            portfolio_id=request_data.portfolio_id,
            predictions=prediction_result["predictions"] # Это будет List[List[float]] или مشابه
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