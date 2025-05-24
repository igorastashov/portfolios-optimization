from pydantic import BaseModel, Field
from typing import Any, Optional

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique ID of the background task.")
    status: str = Field(..., description="Current status of the task (e.g., PENDING, STARTED, SUCCESS, FAILURE).")
    result: Optional[Any] = Field(None, description="The result of the task if it completed successfully. Can be any data type.")
    error_message: Optional[str] = Field(None, description="Error message if the task failed.") 