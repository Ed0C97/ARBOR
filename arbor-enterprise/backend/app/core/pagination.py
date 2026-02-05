"""Keyset (Cursor-based) Pagination.

TIER 1 - Point 3: PostgreSQL Keyset Pagination

Instead of OFFSET N (slow linear scan), use cursor-based pagination:
WHERE (created_at, id) < (last_created_at, last_id)
ORDER BY created_at DESC, id DESC
LIMIT N

This provides:
- O(1) pagination regardless of page number
- Consistent results even with concurrent inserts
- Much faster than OFFSET for large datasets
"""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import and_, or_
from sqlalchemy.orm import Query

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CursorData:
    """Decoded cursor containing pagination position."""
    
    created_at: datetime
    id: str
    
    @classmethod
    def from_string(cls, cursor: str) -> "CursorData | None":
        """Decode a base64 cursor string."""
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
            data = json.loads(decoded)
            return cls(
                created_at=datetime.fromisoformat(data["c"]),
                id=data["i"],
            )
        except Exception as e:
            logger.warning(f"Invalid cursor: {e}")
            return None
    
    def to_string(self) -> str:
        """Encode cursor to base64 string."""
        data = {
            "c": self.created_at.isoformat(),
            "i": self.id,
        }
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


class PaginatedResponse(BaseModel, Generic[T]):
    """Response with cursor-based pagination.
    
    TIER 1 - Point 3: Keyset Pagination Response.
    """
    
    items: list[Any]
    next_cursor: str | None = None
    has_more: bool = False
    total_count: int | None = None  # Optional, expensive for large tables
    
    class Config:
        arbitrary_types_allowed = True


def apply_keyset_pagination(
    query: Query,
    model: Any,
    cursor: str | None = None,
    limit: int = 20,
    order_desc: bool = True,
) -> tuple[Query, int]:
    """Apply keyset pagination to a SQLAlchemy query.
    
    TIER 1 - Point 3: Keyset Pagination Implementation.
    
    Args:
        query: SQLAlchemy query to paginate
        model: SQLAlchemy model with 'created_at' and 'id' columns
        cursor: Base64-encoded cursor from previous page
        limit: Number of items per page
        order_desc: If True, order by newest first
        
    Returns:
        Tuple of (modified query, limit used)
    
    Usage:
        query = select(Entity)
        query, limit = apply_keyset_pagination(query, Entity, cursor="...")
        results = await session.execute(query)
    """
    # Apply ordering
    if order_desc:
        query = query.order_by(model.created_at.desc(), model.id.desc())
    else:
        query = query.order_by(model.created_at.asc(), model.id.asc())
    
    # Apply cursor filter if provided
    if cursor:
        cursor_data = CursorData.from_string(cursor)
        if cursor_data:
            if order_desc:
                # For descending: get items BEFORE cursor position
                query = query.filter(
                    or_(
                        model.created_at < cursor_data.created_at,
                        and_(
                            model.created_at == cursor_data.created_at,
                            model.id < cursor_data.id,
                        ),
                    )
                )
            else:
                # For ascending: get items AFTER cursor position
                query = query.filter(
                    or_(
                        model.created_at > cursor_data.created_at,
                        and_(
                            model.created_at == cursor_data.created_at,
                            model.id > cursor_data.id,
                        ),
                    )
                )
    
    # Apply limit (+1 to check if there are more items)
    query = query.limit(limit + 1)
    
    return query, limit


def create_paginated_response(
    items: list[Any],
    limit: int,
    id_field: str = "id",
    created_at_field: str = "created_at",
) -> PaginatedResponse:
    """Create a paginated response from query results.
    
    Args:
        items: List of items from query (should be limit+1 items)
        limit: The requested limit
        id_field: Name of the ID field
        created_at_field: Name of the created_at field
        
    Returns:
        PaginatedResponse with items and next_cursor
    """
    has_more = len(items) > limit
    
    # Remove extra item used for has_more check
    if has_more:
        items = items[:limit]
    
    # Generate next cursor from last item
    next_cursor = None
    if has_more and items:
        last_item = items[-1]
        
        # Handle both dict and ORM model
        if isinstance(last_item, dict):
            created_at = last_item[created_at_field]
            item_id = last_item[id_field]
        else:
            created_at = getattr(last_item, created_at_field)
            item_id = getattr(last_item, id_field)
        
        # Ensure datetime
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        cursor_data = CursorData(created_at=created_at, id=str(item_id))
        next_cursor = cursor_data.to_string()
    
    return PaginatedResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


# Convenience function for entities endpoint
async def paginate_entities(
    session,
    model,
    cursor: str | None = None,
    limit: int = 20,
    filters: dict | None = None,
) -> PaginatedResponse:
    """Paginate entities with keyset pagination.
    
    TIER 1 - Point 3: Ready-to-use pagination for entities endpoint.
    
    Usage:
        from app.core.pagination import paginate_entities
        
        @router.get("/entities")
        async def list_entities(
            cursor: str | None = None,
            limit: int = 20,
            db: AsyncSession = Depends(get_db),
        ):
            return await paginate_entities(db, Entity, cursor, limit)
    """
    from sqlalchemy import select
    
    query = select(model)
    
    # Apply any filters
    if filters:
        for key, value in filters.items():
            if value is not None and hasattr(model, key):
                query = query.filter(getattr(model, key) == value)
    
    # Apply pagination
    query, _ = apply_keyset_pagination(query, model, cursor, limit)
    
    # Execute
    result = await session.execute(query)
    items = result.scalars().all()
    
    return create_paginated_response(list(items), limit)
