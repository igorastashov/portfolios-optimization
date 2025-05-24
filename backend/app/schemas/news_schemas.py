from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class NewsAnalysisRequest(BaseModel):
    asset_id: Optional[int] = Field(default=None, description="ID of the asset for news analysis. Either asset_id or asset_ticker should be provided.")
    asset_ticker: Optional[str] = Field(default=None, description="Ticker of the asset for news analysis. Either asset_id or asset_ticker should be provided.")
    news_sources: Optional[List[str]] = Field(default=None, description="Optional list of news sources (e.g., ['alpha_vantage']). Defaults to configured sources.")
    date_from: Optional[str] = Field(default=None, description="Start date for news search (YYYY-MM-DD).")
    date_to: Optional[str] = Field(default=None, description="End date for news search (YYYY-MM-DD).")

class NewsChatRequest(BaseModel):
    message: str = Field(..., description="User's message or question for the news chat AI.")
    asset_id: Optional[int] = Field(default=None, description="Optional ID of the asset to focus the chat on.")
    asset_ticker: Optional[str] = Field(default=None, description="Optional ticker of the asset to focus the chat on.")

class NewsChatResponse(BaseModel):
    user_message: str
    ai_response: str
    sources_consulted: Optional[List[str]] = Field(default=None, description="List of sources consulted for the AI response.")
    task_id: Optional[str] = Field(default=None, description="Celery task ID if the response generation is asynchronous.")

class NewsAnalysisResultBase(BaseModel):
    asset_ticker: str = Field(..., description="Ticker symbol of the analyzed asset.")
    analysis_timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Timestamp of when the analysis was performed.")
    news_count: Optional[int] = Field(None, ge=0, description="Number of news items analyzed.")
    overall_sentiment_label: Optional[str] = Field(None, description="Overall sentiment label (e.g., POSITIVE, NEGATIVE, NEUTRAL).")
    overall_sentiment_score: Optional[float] = Field(None, description="Numerical sentiment score (e.g., from -1.0 to 1.0).")
    key_themes: Optional[List[str]] = Field(None, description="List of key themes or topics identified in the news.")
    full_summary: Optional[str] = Field(None, description="AI-generated summary of the analyzed news.")
    task_id: Optional[str] = Field(None, description="ID of the Celery task that performed this analysis.")
    analysis_parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters used for this specific analysis (e.g., model version).")

    class Config:
        from_attributes = True

class NewsAnalysisResultCreate(NewsAnalysisResultBase):
    pass

class NewsAnalysisResultPublic(NewsAnalysisResultBase):
    id: int = Field(..., description="Unique ID of the news analysis result.")

    class Config:
        from_attributes = True 