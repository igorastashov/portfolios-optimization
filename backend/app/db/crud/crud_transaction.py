from sqlalchemy.orm import Session
from typing import List, Optional
from decimal import Decimal

from backend.app.models.transaction_model import Transaction
from backend.app.schemas.transaction_schemas import TransactionCreate, TransactionUpdate
from backend.app.models.portfolio_model import Portfolio # For portfolio ownership check
from backend.app.models.asset_model import Asset # For asset existence check

def get_transaction(db: Session, transaction_id: int, owner_id: int) -> Optional[Transaction]:
    """Retrieve a transaction by its ID, ensuring it belongs to one of the owner's portfolios."""
    return (
        db.query(Transaction)
        .join(Portfolio, Transaction.portfolio_id == Portfolio.id)
        .filter(Transaction.id == transaction_id, Portfolio.owner_id == owner_id)
        .first()
    )

def get_transactions_by_portfolio(
    db: Session, portfolio_id: int, owner_id: int, skip: int = 0, limit: int = 100
) -> List[Transaction]:
    """Retrieve all transactions for a specific portfolio belonging to the owner."""
    # Ensure the portfolio exists and belongs to the owner
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.owner_id == owner_id).first()
    if not portfolio:
        # Consider raising HTTPException(404, "Portfolio not found or not owned by user")
        return [] 
    return db.query(Transaction).filter(Transaction.portfolio_id == portfolio_id).offset(skip).limit(limit).all()

def get_transactions_by_asset_in_portfolio(
    db: Session, portfolio_id: int, asset_id: int, owner_id: int, skip: int = 0, limit: int = 100
) -> List[Transaction]:
    """Retrieve transactions for a specific asset within a specific portfolio of the owner."""
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.owner_id == owner_id).first()
    if not portfolio:
        return [] # Portfolio not found or not owned
    return (
        db.query(Transaction)
        .filter(Transaction.portfolio_id == portfolio_id, Transaction.asset_id == asset_id)
        .offset(skip)
        .limit(limit)
        .all()
    )

def create_transaction(
    db: Session, transaction_in: TransactionCreate, portfolio_id: int, owner_id: int
) -> Optional[Transaction]:
    """Create a new transaction in the specified portfolio of the owner."""
    # Check if the portfolio exists and belongs to the owner
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.owner_id == owner_id).first()
    if not portfolio:
        return None # Portfolio not found or not owned by user

    # Check if the asset exists (by ticker from schema)
    # TODO: Consider changing TransactionCreate to use asset_id directly for more robust linking.
    asset = db.query(Asset).filter(Asset.ticker == transaction_in.asset_ticker).first()
    if not asset:
        # Current behavior: return None if asset not found. 
        # Alternative: raise HTTPException(status_code=404, detail=f"Asset with ticker {transaction_in.asset_ticker} not found.")
        # or create asset on-the-fly if applicable.
        return None 

    db_transaction = Transaction(
        **transaction_in.model_dump(exclude={"asset_ticker"}), # Exclude asset_ticker if it's not a direct field of Transaction model
        portfolio_id=portfolio_id,
        asset_id=asset.id, # Use ID of the found asset
        owner_id=owner_id # Set owner_id on transaction for easier queries if needed, though ownership is via portfolio
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def update_transaction(
    db: Session, transaction_id: int, transaction_in: TransactionUpdate, owner_id: int
) -> Optional[Transaction]:
    """Update an existing transaction. Note: Updating financial transactions should be handled with care.
    The 'updated_at' field is automatically handled by SQLAlchemy due to 'onupdate=datetime.utcnow' in the model.
    """
    db_transaction = get_transaction(db, transaction_id=transaction_id, owner_id=owner_id)
    if not db_transaction:
        return None

    update_data = transaction_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_transaction, field, value)
    
    db.add(db_transaction) # Not strictly necessary if only updating existing fields of a tracked object
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def delete_transaction(db: Session, transaction_id: int, owner_id: int) -> Optional[Transaction]:
    """Delete a transaction by its ID, ensuring it belongs to the owner (via portfolio)."""
    db_transaction = get_transaction(db, transaction_id=transaction_id, owner_id=owner_id)
    if not db_transaction:
        return None
    db.delete(db_transaction)
    db.commit()
    # The db_transaction object becomes detached from the session after deletion and commit.
    # Returning it might be suitable for some API responses, but access to its attributes might be limited.
    return db_transaction # Current approach: return the (detached) object. 