"""Workers package initialization."""

from app.workers.background_jobs import celery_app

__all__ = ["celery_app"]
