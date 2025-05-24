from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
# sqlalchemy.orm.relationship is not used here as no direct relationships are defined in this model
from backend.app.db.base import Base

class NewsAnalysisResult(Base):
    __tablename__ = "news_analysis_result"

    id = Column(Integer, primary_key=True, index=True)
    asset_ticker = Column(String, index=True, nullable=False) # Ticker symbol of the asset being analyzed
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now()) # Timestamp of the analysis
    
    news_count = Column(Integer, nullable=True) # Number of news items analyzed
    overall_sentiment_label = Column(String, nullable=True) # e.g., "POSITIVE", "NEGATIVE", "NEUTRAL"
    overall_sentiment_score = Column(Float, nullable=True) # Numerical sentiment score, e.g., from -1.0 to 1.0

    key_themes = Column(JSON, nullable=True) # List of key themes or topics identified
    full_summary = Column(Text, nullable=True) # AI-generated summary of the news
    
    # ID of the Celery task that performed this analysis, for traceability
    task_id = Column(String, nullable=True, index=True) 
    # Parameters used for this specific analysis, if they vary (e.g., model version, specific prompts)
    analysis_parameters = Column(JSON, nullable=True)

    # Привязка к пользователю, если это индивидуальный анализ для пользователя
    # user_id = Column(Integer, ForeignKey("user.id"), nullable=True, index=True)
    # user = relationship("User")

    # Связь с Asset моделью, если необходимо (например, для каскадного удаления или более сложных запросов)
    # asset_id = Column(Integer, ForeignKey("asset.id"), nullable=True)
    # asset = relationship("Asset") 