from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.app.db.session import get_db
from backend.app.schemas import transaction_schemas as schemas
from backend.app.db.crud import crud_transaction, crud_portfolio
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user


router = APIRouter()

@router.post("/portfolios/{portfolio_id}/transactions/", response_model=schemas.Transaction, status_code=status.HTTP_201_CREATED)
def create_transaction_for_portfolio(
    portfolio_id: int,
    transaction_in: schemas.TransactionCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Create a new transaction for a specific portfolio of the current user.
    """
    portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found or not owned by user")
    
    created_transaction = crud_transaction.create_transaction(
        db=db, transaction_in=transaction_in, portfolio_id=portfolio_id, owner_id=current_user.id
    )
    if not created_transaction:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create transaction (e.g., invalid asset ID)")
    return created_transaction

@router.get("/portfolios/{portfolio_id}/transactions/", response_model=List[schemas.Transaction])
def read_transactions_for_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieve transactions for a specific portfolio of the current user.
    """
    transactions = crud_transaction.get_transactions_by_portfolio(
        db, portfolio_id=portfolio_id, owner_id=current_user.id, skip=skip, limit=limit
    )
    if not transactions and not crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found or not owned by user")
    return transactions

@router.get("/portfolios/{portfolio_id}/transactions/{transaction_id}", response_model=schemas.Transaction)
def read_transaction(
    portfolio_id: int,
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Get a specific transaction by ID, ensuring it belongs to one of the user's portfolios.
    """
    transaction = crud_transaction.get_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    if transaction.portfolio_id != portfolio_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Transaction does not belong to this portfolio")
    return transaction

@router.put("/portfolios/{portfolio_id}/transactions/{transaction_id}", response_model=schemas.Transaction)
def update_transaction(
    portfolio_id: int,
    transaction_id: int,
    transaction_in: schemas.TransactionUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Update a transaction. (Generally, transactions are immutable or have restricted updates).
    Ensures the transaction belongs to the current user.
    """
    db_transaction = crud_transaction.get_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    if db_transaction.portfolio_id != portfolio_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Transaction does not belong to this portfolio")

    updated_transaction = crud_transaction.update_transaction(
        db, transaction_id=transaction_id, transaction_in=transaction_in, owner_id=current_user.id
    )
    return updated_transaction

@router.delete("/portfolios/{portfolio_id}/transactions/{transaction_id}", response_model=schemas.Transaction)
def delete_transaction(
    portfolio_id: int,
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Delete a transaction. Ensures the transaction belongs to the current user.
    """
    db_transaction = crud_transaction.get_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    if db_transaction.portfolio_id != portfolio_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Transaction does not belong to this portfolio")

    deleted_transaction = crud_transaction.delete_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    return deleted_transaction 
