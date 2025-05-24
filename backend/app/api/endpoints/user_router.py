from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db import crud
from backend.app.schemas.auth_schemas import UserPublic
from backend.app.services.auth_service import get_current_active_user
from backend.app.db.session import get_db
from backend.app.models.user_model import User

router = APIRouter()

@router.get("/me", response_model=UserPublic, name="users:get_current_user")
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
):
    """
    Fetch the current logged in user.
    """
    return current_user

@router.get("/{user_id}", response_model=UserPublic, name="users:get_user_by_id")
async def read_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Fetch a specific user by id.
    (Example of a protected endpoint to get other user's data,
     though typically you might restrict this further based on roles/permissions)
    """

    if current_user.id != user_id and not getattr(current_user, 'is_superuser', False):
        raise HTTPException(status_code=403, detail="Not enough permissions to access this user's data")

    user = crud.crud_user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user 