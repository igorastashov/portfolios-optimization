from celery import Celery
import os

# Необходимо убедиться, что Django settings загружены, если Celery используется с Django
# В нашем случае FastAPI, так что этот шаг специфичен для Django и здесь не нужен напрямую,
# но важно знать о нем, если бы это был Django проект.
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')

# Добавляем путь к проекту, чтобы Celery мог найти модуль config
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.app.core.config import settings

# print(f"DEBUG celery_app.py: CELERY_BROKER_URL_CONSTRUCTED='{settings.CELERY_BROKER_URL_CONSTRUCTED}'")
# print(f"DEBUG celery_app.py: CELERY_RESULT_BACKEND_CONSTRUCTED='{settings.CELERY_RESULT_BACKEND_CONSTRUCTED}'")

# Имя приложения Celery, обычно __name__ модуля
# Брокер и бэкенд берутся из настроек
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL_CONSTRUCTED, # Используем собранный URL для RabbitMQ
    backend=settings.CELERY_RESULT_BACKEND_CONSTRUCTED, # Используем собранный URL для Redis
    include=["backend.app.worker.tasks"] # Путь к модулю с задачами
)

# Опциональные настройки Celery, если нужны
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # worker_prefetch_multiplier=1, # Можно настроить для оптимизации
    # task_acks_late=True,          # Если нужна гарантия выполнения даже при падении воркера
    # task_track_started=True       # Для более детального отслеживания состояния задач
)

# Если вы хотите загружать конфигурацию из настроек FastAPI/Pydantic напрямую
# celery_app.config_from_object(settings, namespace='CELERY')
# В этом случае, переменные в settings должны называться, например, CELERY_BROKER_URL и т.д.
# Текущий подход с явной передачей broker и backend более прозрачен для данной структуры.

# Пример автообнаружения задач (если они лежат в модулях, указанных в 'include')
# celery_app.autodiscover_tasks()

if __name__ == '__main__':
    # Это для запуска worker'а напрямую, например: python -m backend.app.worker.celery_app worker -l info
    # Обычно worker запускается командой: celery -A backend.app.worker.celery_app worker -l INFO
    celery_app.start() 