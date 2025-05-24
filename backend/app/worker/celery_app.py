from celery import Celery
import os

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.app.core.config import settings


celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL_CONSTRUCTED,
    backend=settings.CELERY_RESULT_BACKEND_CONSTRUCTED,
    include=["backend.app.worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


if __name__ == '__main__':
    celery_app.start() 