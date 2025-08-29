#!/usr/bin/env python3
"""
Create initial tenant for PDM Platform v2.0
This creates the Egypt client tenant with proper UUID
"""

import asyncpg
import asyncio
import uuid
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_initial_tenant():
    """Create the initial Egypt client tenant"""
    
    # Connect to database
    database_url = "postgresql://postgres:password@localhost:5433/pdm_v2"
    conn = await asyncpg.connect(database_url)
    
    try:
        # Check if tenant already exists
        existing = await conn.fetchrow("""
            SELECT id FROM tenants.tenants 
            WHERE api_key = 'egypt_secure_api_key_2024'
        """)
        
        if existing:
            logger.info(f"Egypt tenant already exists with ID: {existing['id']}")
            return existing['id']
        
        # Create new tenant with proper UUID
        tenant_id = uuid.uuid4()
        
        await conn.execute("""
            INSERT INTO tenants.tenants 
            (id, name, country, api_key, contact_email, created_at, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 
        tenant_id,
        "Egypt Manufacturing Client", 
        "EG",
        "egypt_secure_api_key_2024",
        "contact@egypt-manufacturing.com",
        datetime.utcnow(),
        True)
        
        logger.info(f"Created Egypt tenant with ID: {tenant_id}")
        
        # Also create a test tenant
        test_tenant_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO tenants.tenants 
            (id, name, country, api_key, contact_email, created_at, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 
        test_tenant_id,
        "Test Client", 
        "US",
        "test_api_key",
        "test@example.com",
        datetime.utcnow(),
        True)
        
        logger.info(f"Created test tenant with ID: {test_tenant_id}")
        
        # Display tenant information
        tenants = await conn.fetch("""
            SELECT id, name, api_key FROM tenants.tenants WHERE is_active = TRUE
        """)
        
        logger.info("Active tenants:")
        for tenant in tenants:
            logger.info(f"  - {tenant['name']}: {tenant['id']} (API: {tenant['api_key'][:20]}...)")
        
        return tenant_id
        
    except Exception as e:
        logger.error(f"Failed to create tenant: {e}")
        raise
    
    finally:
        await conn.close()

async def main():
    logger.info("Creating initial tenants for PDM Platform v2.0...")
    
    try:
        await create_initial_tenant()
        logger.info("Initial tenant setup completed successfully!")
        logger.info("You can now restart the API and gateway.")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        logger.info("Make sure PostgreSQL is running: docker-compose up -d postgres")

if __name__ == "__main__":
    asyncio.run(main())
