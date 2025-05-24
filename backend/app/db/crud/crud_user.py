from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from backend.app.models.user_model import User
from backend.app.schemas.auth_schemas import UserCreate, UserUpdate
from backend.app.services.auth_service import get_password_hash

def get_user(db: Session, user_id: int) -> Optional[User]:
    """Retrieve a user by their ID."""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieve a user by their email address."""
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Retrieve a user by their username."""
    return db.query(User).filter(User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Retrieve a list of users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate) -> User:
    """Create a new user in the database."""
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
    """Update an existing user's information."""
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    update_data = user_update.model_dump(exclude_unset=True)
    if "password" in update_data and update_data["password"]:
        hashed_password = get_password_hash(update_data["password"])
        update_data["hashed_password"] = hashed_password
        del update_data["password"]
    else: # Ensure password field is removed if not provided or empty to avoid setting it to None
        if "password" in update_data: 
            del update_data["password"]

    for field, value in update_data.items():
        setattr(db_user, field, value)
    
    db_user.updated_at = datetime.utcnow() # Explicitly update the updated_at timestamp
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int) -> Optional[User]:
    """Delete a user from the database by their ID."""
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    db.delete(db_user)
    db.commit()
    # db_user object is no longer valid after delete and commit,
    # and cannot be refreshed. Return None or a confirmation.
    # For consistency with delete operations, often None is returned or the object before deletion.
    # However, some APIs return the deleted object (detached from session). For now, returning it as is.
    return db_user # Note: object state after commit for delete can be tricky.

def update_last_login(db: Session, user_id: int) -> Optional[User]:
    """Update the last_login timestamp for a user."""
    db_user = get_user(db, user_id)
    if db_user:
        db_user.last_login = datetime.utcnow()
        db.add(db_user) # Not strictly necessary if only updating existing fields of a tracked object
        db.commit()
        db.refresh(db_user)
    return db_user 