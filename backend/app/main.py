from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.api.endpoints import auth_router
from backend.app.api.endpoints import utils_router
from backend.app.api.endpoints import portfolio_router
from backend.app.api.endpoints import asset_router
from backend.app.api.endpoints import transaction_router
from backend.app.api.endpoints import recommendation_router
from backend.app.api.endpoints import news_router
from backend.app.api.endpoints import user_router
# TODO: Remove if ml_pipelines_router and inference_router are not used or integrated differently
# from backend.app.api.api_v1.endpoints import ml_pipelines_router # Check actual path
# from backend.app.api.api_v1.endpoints import inference_router    # Check actual path

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for managing and optimizing investment portfolios.",
    version="0.1.0",
    openapi_tags=[
        {"name": "Authentication", "description": "Operations with users and authentication."},
        {"name": "Users", "description": "User management operations."},
        {"name": "Portfolios", "description": "Operations with user portfolios."},
        {"name": "Assets", "description": "Operations with financial assets."},
        {"name": "Transactions", "description": "Operations with portfolio transactions."},
        {"name": "Recommendations", "description": "Endpoints for getting investment recommendations."},
        {"name": "News", "description": "Endpoints for news analysis and AI chat related to news."},
        {"name": "Utils", "description": "Utility endpoints, e.g., for testing Celery."},
        {"name": "Health", "description": "Health check for the API."},
        # TODO: Add tags for ML Pipelines and Inference if those routers are included
        # {"name": "ML Pipelines", "description": "Manage and run ML training pipelines."},
        # {"name": "Inference", "description": "Endpoints for model inference."}
    ]
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/ping", summary="Health check", tags=["Health"])
def ping():
    """
    Simple health check endpoint.
    """
    return {"ping": "pong!"}

# Include API routers
app.include_router(auth_router.router, prefix=settings.API_V1_STR + "/auth", tags=["Authentication"])
app.include_router(user_router.router, prefix=settings.API_V1_STR + "/users", tags=["Users"])
app.include_router(utils_router.router, prefix=settings.API_V1_STR + "/utils", tags=["Utils"])
app.include_router(portfolio_router.router, prefix=settings.API_V1_STR + "/portfolios", tags=["Portfolios"])
app.include_router(asset_router.router, prefix=settings.API_V1_STR + "/assets", tags=["Assets"])
app.include_router(transaction_router.router, prefix=settings.API_V1_STR, tags=["Transactions"])
app.include_router(recommendation_router.router, prefix=settings.API_V1_STR + "/recommendations", tags=["Recommendations"])
app.include_router(news_router.router, prefix=settings.API_V1_STR + "/news", tags=["News"])

# TODO: Uncomment and verify paths if these routers are to be included
# app.include_router(ml_pipelines_router.router, prefix=settings.API_V1_STR + "/pipelines", tags=["ML Pipelines"])
# app.include_router(inference_router.router, prefix=settings.API_V1_STR + "/inference", tags=["Inference"])

# Further imports and router configurations will be added here 