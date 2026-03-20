"""Точка входа FastAPI-приложения."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.database import Base, engine
from backend.routers.auth import router as auth_router
from backend.routers.mindmap import router as mindmap_router
from backend.routers.rag import router as rag_router

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-платформа для исследований Центра Инвест",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Роутеры ──────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/jacobs/auth")
app.include_router(rag_router)
app.include_router(mindmap_router)


@app.get("/")
async def root():
    """Информация о приложении."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "app_name": settings.APP_NAME}


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting %s v%s on %s:%s",
        settings.APP_NAME, settings.APP_VERSION,
        settings.HOST, settings.PORT,
    )
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
