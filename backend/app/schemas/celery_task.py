from pydantic import BaseModel
from typing import Optional

class CeleryTaskResponse(BaseModel):
    task_id: str
    message: Optional[str] = None 