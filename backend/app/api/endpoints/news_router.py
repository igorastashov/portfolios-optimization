from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas import news_schemas as schemas
from backend.app.schemas.common_schemas import TaskResponse
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user
from backend.app.worker.tasks import analyze_asset_news_task, news_chat_task
from backend.app.db.crud import crud_asset # To validate asset existence
from backend.app.db.crud import crud_news # ADDED: For reading news analysis results

router = APIRouter()

@router.get("/asset/{asset_ticker}", response_model=schemas.NewsAnalysisResultPublic)
async def get_latest_asset_news_analysis(
    asset_ticker: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user) # Защищаем эндпоинт
):
    """
    Retrieve the latest news analysis result for a given asset ticker.
    """
    # Проверяем, существует ли актив (опционально, если это важно для бизнес-логики)
    # db_asset = crud_asset.get_asset_by_ticker(db, ticker=asset_ticker)
    # if not db_asset:
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset with ticker '{asset_ticker}' not found.")

    latest_analysis = crud_news.get_latest_news_analysis_by_ticker(db, asset_ticker=asset_ticker)
    
    if not latest_analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No news analysis found for asset ticker '{asset_ticker}'. Please run analysis first."
        )
    
    # Преобразование модели SQLAlchemy в Pydantic схему NewsAnalysisResultPublic
    # Это делается автоматически FastAPI, если response_model указан и схема совместима.
    return latest_analysis

@router.post("/analyze", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def request_news_analysis(
    request_data: schemas.NewsAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Request news analysis for a specific asset.
    This will trigger a background Celery task.
    """
    if not request_data.asset_id and not request_data.asset_ticker:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either asset_id or asset_ticker must be provided."
        )

    # Optional: Validate if the asset exists in the database
    asset_identifier = request_data.asset_id or request_data.asset_ticker
    if request_data.asset_id:
        asset = crud_asset.get_asset(db, asset_id=request_data.asset_id)
        if not asset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset with ID {request_data.asset_id} not found.")
        asset_identifier = asset.id # Use ID for consistency if found
    elif request_data.asset_ticker:
        # Assuming crud_asset might have a get_by_ticker method or similar
        # asset = crud_asset.get_asset_by_ticker(db, ticker=request_data.asset_ticker)
        # For now, we'll just pass the ticker along; the task can handle resolution.
        # if not asset:
        #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset with ticker {request_data.asset_ticker} not found.")
        pass


    task = analyze_asset_news_task.delay(
        user_id=current_user.id,
        asset_id=request_data.asset_id,
        asset_ticker=request_data.asset_ticker,
        news_sources=request_data.news_sources,
        date_from=request_data.date_from,
        date_to=request_data.date_to
    )

    return TaskResponse(task_id=task.id, status=task.status, result=None)

@router.post("/chat", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def news_chat(
    request_data: schemas.NewsChatRequest,
    db: Session = Depends(get_db), # db might be needed for asset validation or context retrieval
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Send a message to the news chat AI.
    This will trigger a background Celery task.
    """
    # Optional: Validate asset if asset_id or asset_ticker is provided
    if request_data.asset_id:
        asset = crud_asset.get_asset(db, asset_id=request_data.asset_id)
        if not asset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset with ID {request_data.asset_id} not found for chat context.")
    # elif request_data.asset_ticker: (similar validation if get_asset_by_ticker exists)
        # pass 

    task = news_chat_task.delay(
        user_id=current_user.id,
        message=request_data.message,
        asset_id=request_data.asset_id,
        asset_ticker=request_data.asset_ticker,
        # conversation_id=request_data.conversation_id # For future context management
    )

    return TaskResponse(task_id=task.id, status=task.status, result=None) 