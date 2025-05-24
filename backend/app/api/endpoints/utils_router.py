from fastapi import APIRouter, Depends, HTTPException
from celery.result import AsyncResult
from pydantic import BaseModel
from typing import Any

from backend.app.worker.tasks import add_together, example_background_task
from backend.app.schemas.common_schemas import TaskResponse

router = APIRouter()

class AddPayload(BaseModel):
    a: int
    b: int

@router.post("/test-celery-add", response_model=TaskResponse, summary="Test Celery: Add two numbers")
def test_celery_add(payload: AddPayload):
    """
    Test Celery task by adding two numbers asynchronously.
    """
    task = add_together.delay(payload.a, payload.b)
    return TaskResponse(task_id=task.id, status=task.status, result=task.result)

@router.post("/test-celery-background", response_model=TaskResponse, summary="Test Celery: Example background task")
def test_celery_background(message: str = "Hello Celery!"):
    """
    Test Celery background task with a simple message.
    """
    task = example_background_task.delay(message)
    return TaskResponse(task_id=task.id, status=task.status, result=task.result)

@router.get("/task-status/{task_id}", response_model=TaskResponse, summary="Get Celery task status and result")
def get_task_status(task_id: str):
    """
    Get the status and result of a Celery task by its ID.
    """
    task_result_obj = AsyncResult(task_id)
    result_val: Any = task_result_obj.result
    error_msg: str | None = None

    if task_result_obj.failed():
        error_msg = str(result_val)
        result_val = None
    elif isinstance(result_val, Exception):
        error_msg = str(result_val)
        result_val = None
        
    return TaskResponse(
        task_id=task_id,
        status=task_result_obj.status,
        result=result_val,
        error_message=error_msg
    )
