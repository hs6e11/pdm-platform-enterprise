# api/middleware/auth.py
"""
Multi-tenant authentication middleware for PDM Platform v2.0
Fixes the UUID tenant ID conversion issue mentioned in the whitepaper
"""

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import text
import uuid
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta
import json
import redis
from ..database.connection import get_db
from ..models.tenant import Tenant
from ..config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Redis client for caching tenant lookups
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)

class TenantContext:
    """Context object to hold tenant information throughout request lifecycle"""
    def __init__(self, tenant_id: uuid.UUID, tenant_name: str, compliance_level: str, country: str):
        self.tenant_id = tenant_id
        self.tenant_name = tenant_name
        self.compliance_level = compliance_level
        self.country = country
        self.authenticated_at = datetime.utcnow()

class AuthenticationError(HTTPException):
    """Custom authentication error with detailed logging"""
    def __init__(self, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(status_code=status_code, detail=detail)
        logger.warning(f"Authentication failed: {detail}")

async def get_tenant_from_cache(api_key: str) -> Optional[dict]:
    """Get tenant information from Redis cache"""
    try:
        cache_key = f"tenant:{api_key}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        return None
    except Exception as e:
        logger.warning(f"Redis cache lookup failed: {e}")
        return None

async def cache_tenant_info(api_key: str, tenant_data: dict, expire_seconds: int = 3600):
    """Cache tenant information in Redis"""
    try:
        cache_key = f"tenant:{api_key}"
        redis_client.setex(cache_key, expire_seconds, json.dumps(tenant_data))
    except Exception as e:
        logger.warning(f"Redis cache write failed: {e}")

async def authenticate_tenant(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> TenantContext:
    """
    Authenticate tenant using API key and return tenant context
    Fixes the UUID conversion issue from the whitepaper
    """
    
    if not credentials:
        raise AuthenticationError("Missing authentication credentials")
    
    api_key = credentials.credentials
    if not api_key:
        raise AuthenticationError("Empty API key provided")
    
    # First try cache lookup
    cached_tenant = await get_tenant_from_cache(api_key)
    if cached_tenant:
        try:
            return TenantContext(
                tenant_id=uuid.UUID(cached_tenant['id']),
                tenant_name=cached_tenant['name'],
                compliance_level=cached_tenant['compliance_level'],
                country=cached_tenant['country']
            )
        except ValueError as e:
            logger.error(f"Invalid UUID in cache for tenant {cached_tenant.get('name', 'unknown')}: {e}")
            # Remove invalid cache entry
            redis_client.delete(f"tenant:{api_key}")
    
    # Database lookup if not in cache or cache invalid
    try:
        tenant = db.query(Tenant).filter(Tenant.api_key == api_key).first()
        
        if not tenant:
            raise AuthenticationError("Invalid API key", status.HTTP_401_UNAUTHORIZED)
        
        # Validate UUID format (this fixes the conversion issue)
        try:
            tenant_uuid = uuid.UUID(str(tenant.id))
        except (ValueError, TypeError) as e:
            logger.error(f"Database contains invalid UUID for tenant {tenant.name}: {e}")
            raise AuthenticationError("Invalid tenant ID format", status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Cache the successful lookup
        tenant_data = {
            'id': str(tenant.id),
            'name': tenant.name,
            'compliance_level': tenant.compliance_level,
            'country': tenant.country
        }
        await cache_tenant_info(api_key, tenant_data)
        
        logger.info(f"Authenticated tenant: {tenant.name} ({tenant_uuid})")
        
        return TenantContext(
            tenant_id=tenant_uuid,
            tenant_name=tenant.name,
            compliance_level=tenant.compliance_level,
            country=tenant.country
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database authentication error: {e}")
        raise AuthenticationError("Authentication service temporarily unavailable", status.HTTP_503_SERVICE_UNAVAILABLE)

async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> TenantContext:
    """Convenience function to get current authenticated tenant"""
    return await authenticate_tenant(credentials, db)

class TenantIsolationMiddleware:
    """Middleware to enforce tenant data isolation at the database level"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Skip authentication for health checks and public endpoints
            if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
                await self.app(scope, receive, send)
                return
            
            # Add tenant context to request state for downstream use
            try:
                # This will be populated by the authenticate_tenant dependency
                request.state.tenant_context = None
                await self.app(scope, receive, send)
            except Exception as e:
                logger.error(f"Tenant isolation middleware error: {e}")
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)

def require_compliance_level(required_level: str):
    """Decorator to require specific compliance level"""
    def compliance_dependency(tenant: TenantContext = Depends(get_current_tenant)):
        compliance_hierarchy = {
            'basic': 0,
            'eu_cra': 1,
            'nis2': 2
        }
        
        current_level = compliance_hierarchy.get(tenant.compliance_level, -1)
        required = compliance_hierarchy.get(required_level, 999)
        
        if current_level < required:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_level} compliance level, current: {tenant.compliance_level}"
            )
        
        return tenant
    
    return compliance_dependency

# Utility functions for database queries with tenant isolation
def get_tenant_query_filter(tenant_context: TenantContext):
    """Generate SQLAlchemy filter for tenant isolation"""
    return {"tenant_id": tenant_context.tenant_id}

async def execute_tenant_query(db: Session, query: str, tenant_context: TenantContext, params: dict = None):
    """Execute raw SQL query with automatic tenant isolation"""
    if params is None:
        params = {}
    
    # Automatically inject tenant_id into all queries
    params['tenant_id'] = str(tenant_context.tenant_id)
    
    try:
        result = db.execute(text(query), params)
        return result
    except Exception as e:
        logger.error(f"Tenant query execution failed for {tenant_context.tenant_name}: {e}")
        raise

# Export main components
__all__ = [
    'TenantContext',
    'AuthenticationError',
    'authenticate_tenant', 
    'get_current_tenant',
    'TenantIsolationMiddleware',
    'require_compliance_level',
    'get_tenant_query_filter',
    'execute_tenant_query'
]
