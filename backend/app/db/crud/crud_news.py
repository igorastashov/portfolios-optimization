from sqlalchemy.orm import Session
from typing import Optional, List

from backend.app.models.news_model import NewsAnalysisResult
from backend.app.schemas.news_schemas import NewsAnalysisResultCreate

def create_news_analysis_result(
    db: Session, 
    result_in: NewsAnalysisResultCreate
) -> NewsAnalysisResult:
    """
    Saves a new news analysis result to the database.
    This would typically be called by a Celery task after analysis is complete.
    """
    db_result = NewsAnalysisResult(**result_in.model_dump())
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result

def get_latest_news_analysis_by_ticker(
    db: Session, 
    asset_ticker: str
) -> Optional[NewsAnalysisResult]:
    """
    Retrieves the most recent news analysis result for a given asset ticker.
    """
    return (
        db.query(NewsAnalysisResult)
        .filter(NewsAnalysisResult.asset_ticker == asset_ticker)
        .order_by(NewsAnalysisResult.analysis_timestamp.desc())
        .first()
    )

def get_news_analysis_history_by_ticker(
    db: Session,
    asset_ticker: str,
    skip: int = 0,
    limit: int = 10 # Default to 10 recent results
) -> List[NewsAnalysisResult]:
    """
    Retrieves a list of historical news analysis results for a given asset ticker,
    ordered by most recent first.
    """
    return (
        db.query(NewsAnalysisResult)
        .filter(NewsAnalysisResult.asset_ticker == asset_ticker)
        .order_by(NewsAnalysisResult.analysis_timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    ) 