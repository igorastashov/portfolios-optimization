from sqlalchemy.orm import Session
import logging

from backend.app.db.session import SessionLocal
from backend.app.db.base import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db(db: Session) -> None:
    """Initializes the database with any required pre-population data.
    Table creation should be handled by Alembic migrations.
    """
    logger.info("Starting initial data population (if any)...")
    logger.info("Initial data population completed.")

def main() -> None:
    logger.info("Attempting to initialize database with initial data...")
    db = SessionLocal()
    try:
        init_db(db)
        logger.info("Database initial data population process finished.")
    except Exception as e:
        logger.error(f"Error during initial data population: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    main() 