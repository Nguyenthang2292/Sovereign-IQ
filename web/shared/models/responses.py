"""
Shared Pydantic models for all web applications.

This module contains common request/response models:
- Base response models
- Error response models
- Pagination models
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Generic, TypeVar
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model with success flag."""

    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""

    success: bool = False
    error: str
    detail: Optional[str] = None


class DataResponse(BaseResponse, Generic[TypeVar("T")]):
    """Response with data payload."""

    data: T


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response model."""

    items: list
    total: int
    page: int
    page_size: int
    total_pages: int


class StatusResponse(BaseModel):
    """Task status response."""

    session_id: str
    status: str  # pending, running, completed, cancelled, failed
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class CancelResponse(BaseModel):
    """Cancellation response."""

    success: bool
    session_id: str
    message: str
