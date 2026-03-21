import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.database import Base, engine
from backend.routers.auth.routes import router as auth_router
from backend.routers.rag.routes import router as rag_router
from backend.routers.content.summary import router as summary_router
from backend.routers.content.flashcards import router as flashcards_router
from backend.routers.content.mindmap import router as mindmap_router
from backend.routers.content.table import router as table_router
from backend.routers.media.podcast import router as podcast_router
from backend.routers.media.presentation import router as presentation_router
from backend.routers.media.infographics import router as infographics_router
from backend.routers.web.parser import router as parser_router


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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/jacobs/auth")
app.include_router(rag_router)
app.include_router(mindmap_router)
app.include_router(summary_router)
app.include_router(podcast_router)
app.include_router(flashcards_router)
app.include_router(presentation_router)
app.include_router(infographics_router)
app.include_router(parser_router)
app.include_router(table_router)



@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
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
