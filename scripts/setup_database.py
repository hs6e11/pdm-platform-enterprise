#!/usr/bin/env python3
"""
Database setup script for PDM Platform v2.0
Run this before migration to ensure schema exists
"""

import asyncpg
import asyncio
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.base_url = database_url.rsplit('/', 1)[0]  # Remove database name
        self.db_name = database_url.rsplit('/', 1)[1]   # Extract database name
    
    async def setup_database(self):
        """Setup database and schema"""
        logger.info("Setting up PDM Platform v2.0 database...")
        
        try:
            # First, connect to postgres database to create our database
            base_conn = await asyncpg.connect(f"{self.base_url}/postgres")
            
            # Create database if it doesn't exist
            try:
                await base_conn.execute(f'CREATE DATABASE "{self.db_name}"')
                logger.info(f"Created database: {self.db_name}")
            except asyncpg.DuplicateDatabaseError:
                logger.info(f"Database {self.db_name} already exists")
            
            await base_conn.close()
            
            # Now connect to our database and set up schema
            conn = await asyncpg.connect(self.database_url)
            
            await self._setup_extensions(conn)
            await self._setup_schemas(conn)
            await self._setup_tables(conn)
            
            await conn.close()
            
            logger.info("Database setup completed successfully!")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    async def _setup_extensions(self, conn):
        """Install required PostgreSQL extensions"""
        logger.info("Installing PostgreSQL extensions...")
        
        extensions = [
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
            'CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE',
        ]
        
        for ext in extensions:
            try:
                await conn.execute(ext)
                logger.info(f"Extension installed: {ext.split()[-1]}")
            except Exception as e:
                logger.warning(f"Extension installation warning: {e}")
    
    async def _setup_schemas(self, conn):
        """Create database schemas"""
        logger.info("Creating database schemas...")
        
        schemas = ['tenants', 'users', 'audit']
        for schema in schemas:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
            logger.info(f"Schema created: {schema}")
    
    async def _setup_tables(self, conn):
        """Create database tables"""
        logger.info("Creating database tables...")
        
        # Tenants table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants.tenants (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(100) NOT NULL,
                country CHAR(2) DEFAULT 'EG',
                compliance_level VARCHAR(20) DEFAULT 'basic',
                max_machines INTEGER NOT NULL DEFAULT 10,
                max_users INTEGER NOT NULL DEFAULT 5,
                contact_email VARCHAR(255),
                api_key VARCHAR(255) UNIQUE NOT NULL,
                encryption_key TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE,
                last_activity TIMESTAMP WITH TIME ZONE
            )
        """)
        logger.info("Created tenants table")
        
        # Users table
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
            )
        """)
        logger.info("Created users table")
        
        # Sensor data table
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
            )
        """)
        logger.info("Created sensor_data table")
        
        # Convert to hypertable for time-series optimization
        try:
            await conn.execute("""
                SELECT create_hypertable('sensor_data', 'timestamp',
                    partitioning_column => 'tenant_id',
                    number_partitions => 4,
                    if_not_exists => TRUE)
            """)
            logger.info("Created TimescaleDB hypertable for sensor_data")
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Audit logs table
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
            )
        """)
        logger.info("Created audit logs table")
        
        # Convert audit logs to hypertable
        try:
            await conn.execute("""
                SELECT create_hypertable('audit.logs', 'timestamp',
                    if_not_exists => TRUE)
            """)
            logger.info("Created TimescaleDB hypertable for audit logs")
        except Exception as e:
            logger.warning(f"Audit hypertable creation warning: {e}")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_tenant_machine_time ON sensor_data (tenant_id, machine_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_type_time ON sensor_data (sensor_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_time ON audit.logs (tenant_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tenants_api_key ON tenants.tenants (api_key) WHERE is_active = TRUE",
        ]
        
        for idx in indexes:
            try:
                await conn.execute(idx)
                logger.info(f"Created index: {idx.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

async def test_connection(database_url: str):
    """Test database connection"""
    logger.info("Testing database connection...")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(database_url)
            result = await conn.fetchrow("SELECT version()")
            await conn.close()
            
            logger.info(f"Database connection successful!")
            logger.info(f"PostgreSQL version: {result['version']}")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                logger.info("Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                logger.error(f"All connection attempts failed: {e}")
                return False
    
    return False

async def main():
    # Database configuration - using Docker PostgreSQL on port 5433
    database_url = "postgresql://postgres:password@localhost:5433/pdm_v2"
    
    # Test connection first
    if not await test_connection(database_url):
        logger.error("Cannot connect to database. Please ensure PostgreSQL is running.")
        logger.info("Try: docker-compose up -d postgres redis")
        return
    
    # Setup database
    setup = DatabaseSetup(database_url)
    await setup.setup_database()
    
    logger.info("Database setup completed! Ready for migration.")

if __name__ == "__main__":
    asyncio.run(main())
