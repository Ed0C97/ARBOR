"""Authentication and RBAC security layer."""

import logging

import httpx
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

security = HTTPBearer(auto_error=False)


class RBACMiddleware:
    """Role-Based Access Control."""

    ROLES = {
        "viewer": ["read:entities", "read:search"],
        "curator": ["read:entities", "write:entities", "read:search", "read:admin"],
        "admin": ["*"],
        "api_client": ["read:search", "read:entities"],
    }

    def __init__(self):
        self.domain = settings.auth0_domain
        self.audience = settings.auth0_api_audience
        self._jwks_cache = None

    async def _get_jwks(self) -> dict:
        """Fetch JWKS from Auth0."""
        if self._jwks_cache:
            return self._jwks_cache

        if not self.domain:
            return {}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://{self.domain}/.well-known/jwks.json")
                self._jwks_cache = response.json()
                return self._jwks_cache
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            return {}

    async def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials | None = Security(security),
    ) -> dict:
        """Verify JWT and extract claims."""
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        # In development mode, accept a special dev token
        if settings.app_env == "development" and credentials.credentials == "dev-token":
            return {
                "sub": "dev-user",
                "role": "admin",
                "email": "dev@arbor.local",
            }

        token = credentials.credentials

        try:
            jwks = await self._get_jwks()
            if not jwks:
                raise HTTPException(status_code=401, detail="Auth not configured")

            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=f"https://{self.domain}/",
            )
            return payload

        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    def require_permission(self, permission: str):
        """Dependency that requires a specific permission."""

        async def check(
            user: dict = Depends(self.verify_token),
        ) -> dict:
            user_role = user.get("role", "viewer")
            allowed = self.ROLES.get(user_role, [])

            if "*" not in allowed and permission not in allowed:
                raise HTTPException(status_code=403, detail="Permission denied")

            return user

        return check


# Singleton
auth = RBACMiddleware()


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> dict | None:
    """Optional auth - returns user if authenticated, None otherwise."""
    if not credentials:
        return None
    try:
        return await auth.verify_token(credentials)
    except HTTPException:
        return None
