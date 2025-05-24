from sqlalchemy.orm import Session
from typing import List, Optional

from backend.app.models.portfolio_model import Portfolio
from backend.app.schemas.portfolio_schemas import PortfolioCreate, PortfolioUpdate

def get_portfolio(db: Session, portfolio_id: int, owner_id: int) -> Optional[Portfolio]:
    """Получает портфель по ID, только если он принадлежит указанному владельцу."""
    return db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.owner_id == owner_id).first()

def get_portfolio_by_name(db: Session, name: str, owner_id: int) -> Optional[Portfolio]:
    """Получает портфель по имени для конкретного пользователя."""
    return db.query(Portfolio).filter(Portfolio.name == name, Portfolio.owner_id == owner_id).first()

def get_portfolios_by_owner(db: Session, owner_id: int, skip: int = 0, limit: int = 100) -> List[Portfolio]:
    """Получает список портфелей для конкретного пользователя."""
    return db.query(Portfolio).filter(Portfolio.owner_id == owner_id).offset(skip).limit(limit).all()

def create_portfolio(db: Session, portfolio: PortfolioCreate, owner_id: int) -> Portfolio:
    """Создает новый портфель для пользователя."""
    db_portfolio = Portfolio(**portfolio.model_dump(), owner_id=owner_id)
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def update_portfolio(
    db: Session, portfolio_id: int, portfolio_in: PortfolioUpdate, owner_id: int
) -> Optional[Portfolio]:
    """Обновляет существующий портфель пользователя."""
    db_portfolio = get_portfolio(db, portfolio_id=portfolio_id, owner_id=owner_id)
    if not db_portfolio:
        return None
    
    update_data = portfolio_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_portfolio, field, value)
    
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def delete_portfolio(db: Session, portfolio_id: int, owner_id: int) -> Optional[Portfolio]:
    """Удаляет портфель пользователя."""
    db_portfolio = get_portfolio(db, portfolio_id=portfolio_id, owner_id=owner_id)
    if not db_portfolio:
        return None
    db.delete(db_portfolio)
    db.commit()
    return db_portfolio 