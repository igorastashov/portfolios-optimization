# Import all the models, so that Base has them before being
# imported by Alembic
from backend.app.models.user_model import Base, User  # noqa
from backend.app.models.portfolio_model import Portfolio # noqa
from backend.app.models.asset_model import Asset # noqa
from backend.app.models.transaction_model import Transaction # noqa
from backend.app.models.news_model import NewsAnalysisResult # noqa
# All SQLAlchemy models should be imported here for Alembic to discover them.

# Add other models here as they are created 