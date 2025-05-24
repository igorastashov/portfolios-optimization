from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas import recommendation_schemas as schemas
from backend.app.schemas.common_schemas import TaskResponse
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user
from backend.app.worker.tasks import generate_rebalancing_recommendation_task
from backend.app.db.crud import crud_portfolio

router = APIRouter()

@router.post("/rebalance", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def request_rebalance_portfolio(
    request_data: schemas.RebalancePortfolioRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Request rebalancing for a specific portfolio.
    This will trigger a background Celery task.
    """
    portfolio_id = request_data.portfolio_id
    
    portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found or you do not have permission to access it."
        )
    
    task = generate_rebalancing_recommendation_task.delay(
        portfolio_id=portfolio_id, 
        user_id=current_user.id,
    )
    
    return TaskResponse(task_id=task.id, status=task.status, result=None) 
