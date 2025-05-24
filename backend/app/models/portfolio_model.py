from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, Numeric, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.app.models.user_model import Base # Импортируем Base из user_model или общего base.py

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Связь с пользователем
    owner = relationship("User", back_populates="portfolios")
    # Связь с транзакциями
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")

    # Можно добавить другие поля, например, strategy, rebalancing_frequency и т.д.
    # current_value = Column(Numeric(15, 2), nullable=True) # Пример текущей стоимости

# Добавляем обратную связь в модель User
# Это нужно сделать в файле user_model.py
# User.portfolios = relationship("Portfolio", back_populates="owner", cascade="all, delete-orphan") 