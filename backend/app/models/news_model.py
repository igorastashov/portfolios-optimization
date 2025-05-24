from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from backend.app.db.base import Base

class NewsAnalysisResult(Base):
    __tablename__ = "news_analysis_result"

    id = Column(Integer, primary_key=True, index=True)
    asset_ticker = Column(String, index=True, nullable=False)
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    news_count = Column(Integer, nullable=True)
    overall_sentiment_label = Column(String, nullable=True)
    overall_sentiment_score = Column(Float, nullable=True)

    key_themes = Column(JSON, nullable=True)
    full_summary = Column(Text, nullable=True)
    
    task_id = Column(String, nullable=True, index=True) 
    analysis_parameters = Column(JSON, nullable=True)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=True, index=True)
    user = relationship("User")

    asset_id = Column(Integer, ForeignKey("asset.id"), nullable=True)
    asset = relationship("Asset") 
    