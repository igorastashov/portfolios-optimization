from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware

from backend.app.api.api_v1.endpoints import (
    login, 
    users, 
    utils, 
    portfolios, 
    assets, 
    transactions,
    news,
    recommendations,
    ml_pipelines,
    inference
)
from backend.app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip() for origin in settings.BACKEND_CORS_ORIGINS],
        # allow_origins=["*"], # Allow all for dev, but be specific in prod
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Added OPTIONS
        allow_headers=["*"], # Allow all headers
    )

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(portfolios.router, prefix="/portfolios", tags=["portfolios"])
api_router.include_router(assets.router, prefix="/assets", tags=["assets"])
api_router.include_router(transactions.router, prefix="/transactions", tags=["transactions"])
api_router.include_router(news.router, prefix="/news", tags=["news-analysis"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(ml_pipelines.ml_pipelines_router, prefix="/pipelines", tags=["ml-pipelines"])
api_router.include_router(inference.inference_router, prefix="/inference", tags=["inference"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])

app.include_router(api_router, prefix=settings.API_V1_STR)

# Простой эндпоинт для проверки работы Celery (опционально)
from backend.worker.tasks import example_task
from backend.app.schemas import Msg, CeleryTaskResponse # Added CeleryTaskResponse

@app.post("/api/v1/test-celery/", response_model=CeleryTaskResponse, tags=["utils"])
async def test_celery(msg: Msg):
    """
    Test Celery worker.
    """
    task = example_task.delay(param1=msg.msg, param2=5)
    return CeleryTaskResponse(task_id=task.id, message="Example Celery task sent")

# Эндпоинт для получения статуса задачи Celery (опционально)
from celery.result import AsyncResult
from pydantic import BaseModel
from typing import Any

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Any = None
    progress: Any = None # To hold progress metadata

@app.get("/api/v1/task-status/{task_id}", response_model=TaskStatusResponse, tags=["utils"])
async def get_task_status(task_id: str):
    """
    Get the status of a Celery task.
    """
    task_result = AsyncResult(task_id, app=example_task.app) # Use app from one of your tasks
    result = task_result.result
    progress_info = None

    if task_result.state == 'PENDING':
        response_status = "Pending"
    elif task_result.state == 'PROGRESS':
        response_status = "In Progress"
        progress_info = task_result.info # meta from update_state
    elif task_result.state == 'SUCCESS':
        response_status = "Success"
    elif task_result.state == 'FAILURE':
        response_status = "Failure"
    else:
        response_status = task_result.state
    
    return TaskStatusResponse(task_id=task_id, status=response_status, result=result, progress=progress_info) 