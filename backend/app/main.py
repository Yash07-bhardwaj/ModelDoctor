"""FastAPI entry point for ModelDoctor backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, monitoring, predict, report, train
from app.core import sessions
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.schemas.models import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    sessions.mark_interrupted_sessions()
    removed = sessions.cleanup_old_sessions()
    if removed:
        logging.getLogger("modeldoctor").info("Removed %d expired session(s)", removed)
    yield


app = FastAPI(
    title="ModelDoctor API",
    description="Intelligent Detection and Mitigation of Unreliable ML Predictions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

for module in (datasets, train, report, predict, monitoring):
    app.include_router(module.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint confirming API service status."""
    return {"status": "ok"}
