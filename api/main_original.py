"""
PDM Platform v2.0 - Multi-Tenant FastAPI Application
Complete corrected version with proper UUID tenant authentication
"""

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class MultiTenantDatabase:
    """Database manager with tenant isolation"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=30,
            server_settings={
                'application_name': 'pdm_platform_v2',
            }
        )
        await self._ensure_schemas()
    
    async def _ensure_schemas(self):
        """Ensure database schemas exist"""
        async with self.pool.acquire() as conn:
            # Create schemas for multi-tenant isolation
            await conn.execute("""
                CREATE SCHEMA IF NOT EXISTS tenants;
                CREATE SCHEMA IF NOT EXISTS users;
                CREATE SCHEMA IF NOT EXISTS audit;
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
            """)
            
            # Tenants table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tenants.tenants (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(100) NOT NULL,
                    country CHAR(2) NOT NULL DEFAULT 'EG',
                    compliance_level VARCHAR(20) NOT NULL DEFAULT 'basic',
                    max_machines INTEGER NOT NULL DEFAULT 10,
                    max_users INTEGER NOT NULL DEFAULT 5,
                    contact_email VARCHAR(255),
                    api_key VARCHAR(255) UNIQUE NOT NULL,
                    encryption_key TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    last_activity TIMESTAMP WITH TIME ZONE
                );
            """)
            
            # Users table with RBAC
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users.users (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    tenant_id UUID REFERENCES tenants.tenants(id),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'viewer',
                    full_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_login TIMESTAMP WITH TIME ZONE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)
            
            # Audit logs (NIS2 compliance)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit.logs (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users.users(id),
                    tenant_id UUID REFERENCES tenants.tenants(id),
                    action VARCHAR(100) NOT NULL,
                    resource VARCHAR(100) NOT NULL,
                    ip_address INET,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    success BOOLEAN NOT NULL,
                    details JSONB
                );
            """)
            
            # Sensor data with tenant isolation (create table first, hypertable later)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    tenant_id UUID NOT NULL,
                    machine_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    sensor_type VARCHAR(50) NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    unit VARCHAR(20) DEFAULT '',
                    quality VARCHAR(20) DEFAULT 'GOOD',
                    anomaly_score DOUBLE PRECISION DEFAULT 0.0,
                    processed BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Create indexes for performance (TimescaleDB compatible)
            # Create indexes before hypertables to avoid conflicts
            base_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_tenants_api_key ON tenants.tenants (api_key) WHERE is_active = TRUE",
                "CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users.users (tenant_id)",
            ]
            
            for idx in base_indexes:
                try:
                    await conn.execute(idx)
                except Exception as e:
                    logger.warning(f"Base index creation warning: {e}")
            
            # Create hypertables after base table creation but before hypertable-specific indexes
            try:
                await conn.execute("""
                    SELECT create_hypertable('sensor_data', 'timestamp',
                        partitioning_column => 'tenant_id',
                        number_partitions => 4,
                        if_not_exists => TRUE)
                """)
                logger.info("Created TimescaleDB hypertable for sensor_data")
            except Exception as e:
                logger.warning(f"Sensor data hypertable creation: {e}")
            
            try:
                await conn.execute("""
                    SELECT create_hypertable('audit.logs', 'timestamp', 
                        if_not_exists => TRUE)
                """)
                logger.info("Created TimescaleDB hypertable for audit logs")
            except Exception as e:
                logger.warning(f"Audit logs hypertable creation: {e}")
            
            # Create TimescaleDB-compatible indexes after hypertable creation
            hypertable_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_sensor_data_tenant_time ON sensor_data (tenant_id, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_sensor_data_machine_time ON sensor_data (machine_id, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_sensor_data_type_time ON sensor_data (sensor_type, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_time ON audit.logs (tenant_id, timestamp DESC) WHERE tenant_id IS NOT NULL",
            ]
            
            for idx in hypertable_indexes:
                try:
                    await conn.execute(idx)
                    logger.info(f"Created hypertable index: {idx.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"Hypertable index creation warning: {e}")
            
            logger.info("Database schemas initialized")
    
    async def get_tenant_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get tenant by API key for authentication"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, name, country, compliance_level, max_machines, max_users,
                       is_active, created_at
                FROM tenants.tenants 
                WHERE api_key = $1 AND is_active = TRUE
            """, api_key)
            
            return dict(row) if row else None
    
    async def store_sensor_data(self, tenant_id: uuid.UUID, machine_id: str, 
                              sensor_type: str, value: float):
        """Store sensor data with tenant isolation"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sensor_data 
                (tenant_id, machine_id, sensor_type, value, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, tenant_id, machine_id, sensor_type, float(value), datetime.utcnow())
    
    async def get_machine_status(self, tenant_id: uuid.UUID, machine_id: str) -> Dict[str, Any]:
        """Get machine status for specific tenant"""
        async with self.pool.acquire() as conn:
            # Get latest readings for the machine
            rows = await conn.fetch("""
                SELECT sensor_type, value, timestamp, quality, anomaly_score
                FROM sensor_data 
                WHERE tenant_id = $1 AND machine_id = $2 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, tenant_id, machine_id)
            
            if not rows:
                return {
                    "machine_id": machine_id,
                    "status": "no_data",
                    "last_reading": None,
                    "sensors": {}
                }
            
            # Group latest readings by sensor type
            latest_sensors = {}
            latest_timestamp = None
            
            for row in rows:
                sensor_type = row['sensor_type']
                if sensor_type not in latest_sensors:
                    latest_sensors[sensor_type] = {
                        "value": row['value'],
                        "timestamp": row['timestamp'].isoformat(),
                        "quality": row['quality'],
                        "anomaly_score": row['anomaly_score'] or 0.0
                    }
                    if not latest_timestamp or row['timestamp'] > latest_timestamp:
                        latest_timestamp = row['timestamp']
            
            # Determine overall machine status
            avg_anomaly_score = sum(s.get('anomaly_score', 0) for s in latest_sensors.values()) / len(latest_sensors)
            
            if avg_anomaly_score > 0.8:
                status = "critical"
            elif avg_anomaly_score > 0.5:
                status = "warning"
            else:
                status = "operational"
            
            return {
                "machine_id": machine_id,
                "status": status,
                "last_reading": latest_timestamp.isoformat() if latest_timestamp else None,
                "sensors": latest_sensors,
                "anomaly_score": avg_anomaly_score
            }

# Global database instance
db = MultiTenantDatabase(os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5433/pdm_v2"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.initialize()
    logger.info("PDM Platform v2.0 started successfully")
    logger.info("Connected to PostgreSQL database")
    yield
    # Shutdown
    if db.pool:
        await db.pool.close()

app = FastAPI(
    title="PDM Platform v2.0",
    description="Multi-Tenant Predictive Maintenance Platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

async def get_authenticated_tenant(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Dependency for tenant authentication"""
    api_key = credentials.credentials
    tenant = await db.get_tenant_by_api_key(api_key)
    
    if not tenant:
        logger.warning(f"Invalid API key attempted from {request.client.host}: {api_key[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return tenant

@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if db.pool else "disconnected"
    }

@app.post("/api/v2/iot/data/{machine_id}")
async def submit_sensor_data(
    machine_id: str,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Submit sensor data with proper tenant lookup"""
    try:
        api_key = credentials.credentials
        
        # Look up tenant by API key in database
        tenant = await db.get_tenant_by_api_key(api_key)
        if not tenant:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        tenant_id = tenant['id']  # This is now a proper UUID
        logger.info(f"Authenticated request from tenant: {tenant['name']}")
        
        data = await request.json()
        
        # Store sensor data with UUID tenant_id
        for sensor_type, value in data.get("sensors", {}).items():
            await db.store_sensor_data(tenant_id, machine_id, sensor_type, float(value))
            logger.info(f"Stored {sensor_type}={value} for {machine_id}")
        
        return {
            "status": "success",
            "message": "Data stored successfully",
            "machine_id": machine_id,
            "tenant": tenant['name'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/machines/{machine_id}/status")
async def get_machine_status(
    machine_id: str,
    tenant: Dict[str, Any] = Depends(get_authenticated_tenant)
):
    """Get machine status with tenant isolation"""
    try:
        tenant_id = tenant['id']
        status = await db.get_machine_status(tenant_id, machine_id)
        
        return {
            "status": "success",
            "tenant": tenant['name'],
            "machine_status": status
        }
        
    except Exception as e:
        logger.error(f"Machine status query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/tenants/info")
async def get_tenant_info(
    tenant: Dict[str, Any] = Depends(get_authenticated_tenant)
):
    """Get current tenant information"""
    return {
        "tenant_id": str(tenant['id']),
        "name": tenant['name'],
        "country": tenant['country'],
        "compliance_level": tenant['compliance_level'],
        "max_machines": tenant['max_machines'],
        "max_users": tenant['max_users'],
        "created_at": tenant['created_at'].isoformat()
    }

@app.get("/api/v2/sensors/recent/{machine_id}")
async def get_recent_sensor_data(
    machine_id: str,
    limit: int = 100,
    tenant: Dict[str, Any] = Depends(get_authenticated_tenant)
):
    """Get recent sensor data for a machine"""
    try:
        tenant_id = tenant['id']
        
        async with db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT sensor_type, value, timestamp, quality, anomaly_score
                FROM sensor_data 
                WHERE tenant_id = $1 AND machine_id = $2 
                ORDER BY timestamp DESC 
                LIMIT $3
            """, tenant_id, machine_id, limit)
            
            readings = []
            for row in rows:
                readings.append({
                    "sensor_type": row['sensor_type'],
                    "value": row['value'],
                    "timestamp": row['timestamp'].isoformat(),
                    "quality": row['quality'],
                    "anomaly_score": row['anomaly_score'] or 0.0
                })
            
            return {
                "status": "success",
                "machine_id": machine_id,
                "tenant": tenant['name'],
                "readings": readings,
                "count": len(readings)
            }
            
    except Exception as e:
        logger.error(f"Recent sensor data query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/dashboard/summary")
async def get_dashboard_summary(
    tenant: Dict[str, Any] = Depends(get_authenticated_tenant)
):
    """Get dashboard summary for tenant"""
    try:
        tenant_id = tenant['id']
        
        async with db.pool.acquire() as conn:
            # Get machine count and recent data summary
            summary = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT machine_id) as machine_count,
                    COUNT(*) as total_readings,
                    MAX(timestamp) as latest_reading,
                    AVG(anomaly_score) as avg_anomaly_score
                FROM sensor_data 
                WHERE tenant_id = $1 
                AND timestamp > NOW() - INTERVAL '24 hours'
            """, tenant_id)
            
            return {
                "status": "success",
                "tenant": tenant['name'],
                "summary": {
                    "machine_count": summary['machine_count'] or 0,
                    "total_readings_24h": summary['total_readings'] or 0,
                    "latest_reading": summary['latest_reading'].isoformat() if summary['latest_reading'] else None,
                    "avg_anomaly_score": float(summary['avg_anomaly_score'] or 0.0),
                    "system_status": "operational" if (summary['avg_anomaly_score'] or 0) < 0.5 else "attention_required"
                }
            }
            
    except Exception as e:
        logger.error(f"Dashboard summary query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
