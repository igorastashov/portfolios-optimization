from sqlalchemy.orm import Session
import logging

from backend.app.db.session import engine, SessionLocal # engine might not be needed if create_all is removed
from backend.app.db.base import Base  # Ensure all models are imported
# from backend.app.db.crud import crud_user # Not needed if superuser creation is removed
# from backend.app.schemas.auth_schemas import UserCreate # Not needed if superuser creation is removed
from backend.app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure all SQL Alchemy models are imported (via db.base) before any DB operations
# if they are needed by, for example, initial data population.

def init_db(db: Session) -> None:
    """Initializes the database with any required pre-population data.
    Table creation should be handled by Alembic migrations.
    """
    logger.info("Starting initial data population (if any)...")
    # Base.metadata.create_all(bind=engine) # Tables are created via Alembic migrations

    # Example: Create initial roles, settings, or other default data if necessary.
    # For instance:
    # initial_roles = ["admin", "user", "editor"]
    # for role_name in initial_roles:
    #     existing_role = db.query(Role).filter(Role.name == role_name).first()
    #     if not existing_role:
    #         db_role = Role(name=role_name)
    #         db.add(db_role)
    #         logger.info(f"Created role: {role_name}")
    # db.commit()
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
    # This script can be run to populate initial data after migrations are applied.
    main() 