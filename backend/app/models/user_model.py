from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationship to Portfolios: Portfolios owned by this user.
    # Cascading delete means if a User is deleted, their Portfolios are also deleted.
    portfolios = relationship("Portfolio", back_populates="owner", cascade="all, delete-orphan")
    
    # Relationship to Transactions: Transactions made by this user.
    # Cascade delete for transactions is typically handled when the associated Portfolio is deleted.
    transactions = relationship("Transaction", back_populates="owner") 