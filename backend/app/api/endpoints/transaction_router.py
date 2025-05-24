from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.app.db.session import get_db
from backend.app.schemas import transaction_schemas as schemas
from backend.app.db.crud import crud_transaction, crud_portfolio # Для проверки принадлежности портфеля
from backend.app.models.user_model import User as UserModel
from backend.app.services.auth_service import get_current_active_user

# Создаем роутер с префиксом, который будет включать portfolio_id
# Это позволит нам не передавать portfolio_id в каждом эндпоинте отдельно,
# а брать его из path parameters.
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
    # Проверяем, принадлежит ли портфель текущему пользователю
    portfolio = crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id)
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found or not owned by user")
    
    created_transaction = crud_transaction.create_transaction(
        db=db, transaction_in=transaction_in, portfolio_id=portfolio_id, owner_id=current_user.id
    )
    if not created_transaction:
        # Эта ошибка может возникнуть, если, например, asset_id невалидный
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
    # crud_transaction.get_transactions_by_portfolio уже проверяет принадлежность портфеля
    transactions = crud_transaction.get_transactions_by_portfolio(
        db, portfolio_id=portfolio_id, owner_id=current_user.id, skip=skip, limit=limit
    )
    if not transactions and not crud_portfolio.get_portfolio(db, portfolio_id=portfolio_id, owner_id=current_user.id):
         # Если портфель не существует или не принадлежит юзеру, transactions будет [], но лучше вернуть 404
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found or not owned by user")
    return transactions

@router.get("/portfolios/{portfolio_id}/transactions/{transaction_id}", response_model=schemas.Transaction)
def read_transaction(
    portfolio_id: int, # Не используется напрямую в crud, но нужен для корректного URL и проверки
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Get a specific transaction by ID, ensuring it belongs to one of the user's portfolios.
    (Хотя crud_transaction.get_transaction проверяет owner_id транзакции, хорошей практикой
     будет также убедиться, что portfolio_id из URL соответствует транзакции, если это важно)
    """
    transaction = crud_transaction.get_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    if transaction.portfolio_id != portfolio_id: # Дополнительная проверка
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
    # Сначала получим транзакцию, чтобы убедиться, что она существует и принадлежит пользователю и портфелю
    db_transaction = crud_transaction.get_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    if db_transaction.portfolio_id != portfolio_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Transaction does not belong to this portfolio")

    updated_transaction = crud_transaction.update_transaction(
        db, transaction_id=transaction_id, transaction_in=transaction_in, owner_id=current_user.id
    )
    # crud_transaction.update_transaction вернет None если db_transaction не был найден (что мы уже проверили)
    return updated_transaction

@router.delete("/portfolios/{portfolio_id}/transactions/{transaction_id}", response_model=schemas.Transaction)
def delete_transaction(
    portfolio_id: int, # Для консистентности URL и потенциальной проверки
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Delete a transaction. Ensures the transaction belongs to the current user.
    """
    # Проверяем, что транзакция существует и принадлежит пользователю и этому портфелю
    db_transaction = crud_transaction.get_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    if not db_transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    if db_transaction.portfolio_id != portfolio_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Transaction does not belong to this portfolio")

    deleted_transaction = crud_transaction.delete_transaction(db, transaction_id=transaction_id, owner_id=current_user.id)
    # crud_transaction.delete_transaction вернет None если db_transaction не был найден (что мы уже проверили)
    return deleted_transaction 