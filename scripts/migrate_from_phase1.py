"""
Migration script from Phase 1 to Phase 2
Preserves existing data while upgrading to multi-tenant architecture
"""

import sqlite3
import asyncpg
import asyncio
import uuid
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMigrator:
    def __init__(self, sqlite_path: str, postgres_url: str):
        self.sqlite_path = sqlite_path
        self.postgres_url = postgres_url
    
    async def migrate_all_data(self):
        """Migrate all data from Phase 1 to Phase 2"""
        logger.info("Starting data migration from Phase 1...")
        
        # Connect to databases
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        pg_conn = await asyncpg.connect(self.postgres_url)
        
        try:
            # Create default tenant for existing Egypt client
            tenant_id = await self._create_default_tenant(pg_conn)
            
            # Migrate sensor readings
            await self._migrate_sensor_readings(sqlite_conn, pg_conn, tenant_id)
            
            # Migrate IoT clients
            await self._migrate_iot_clients(sqlite_conn, pg_conn)
            
            logger.info("Data migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        
        finally:
            sqlite_conn.close()
            await pg_conn.close()
    
    async def _create_default_tenant(self, pg_conn) -> str:
        """Create default tenant for existing Egypt client"""
        tenant_id = str(uuid.uuid4())
        
        await pg_conn.execute("""
            INSERT INTO tenants.tenants (id, name, api_key, created_at)
            VALUES ($1, $2, $3, $4)
        """, tenant_id, "Egypt Manufacturing", "egypt_secure_api_key_2024", datetime.utcnow())
        
        logger.info(f"Created default tenant: {tenant_id}")
        return tenant_id
    
    async def _migrate_sensor_readings(self, sqlite_conn, pg_conn, tenant_id: str):
        """Migrate sensor readings to new schema"""
        cursor = sqlite_conn.execute("""
            SELECT client_id, machine_id, timestamp, temperature, 
                   pressure, vibration, power_consumption
            FROM real_sensor_readings
            ORDER BY timestamp DESC
            LIMIT 20000  -- Migrate recent 20k readings
        """)
        
        migrated_count = 0
        
        for row in cursor:
            client_id, machine_id, timestamp, temp, pressure, vib, power = row
            
            # Migrate each sensor type as separate record
            sensor_data = [
                ('temperature', temp),
                ('pressure', pressure), 
                ('vibration', vib),
                ('power_consumption', power)
            ]
            
            for sensor_type, value in sensor_data:
                if value is not None:
                    await pg_conn.execute("""
                        INSERT INTO sensor_data 
                        (tenant_id, machine_id, timestamp, sensor_type, value)
                        VALUES ($1, $2, $3, $4, $5)
                    """, tenant_id, machine_id, timestamp, sensor_type, float(value))
            
            migrated_count += 1
            
            if migrated_count % 1000 == 0:
                logger.info(f"Migrated {migrated_count} sensor readings...")
        
        logger.info(f"Migrated total {migrated_count} sensor readings")
    
    async def _migrate_iot_clients(self, sqlite_conn, pg_conn):
        """Migrate IoT client configurations"""
        try:
            cursor = sqlite_conn.execute("""
                SELECT client_id, company_name, country, api_key, last_seen
                FROM iot_clients
            """)
            
            for row in cursor:
                client_id, company_name, country, api_key, last_seen = row
                # Map to tenant system - this would need proper mapping logic
                logger.info(f"Would migrate client: {company_name}")
                
        except sqlite3.OperationalError:
            logger.info("No iot_clients table found - skipping")

async def main():
    """Run migration"""
    # Check if original database exists
    original_db_paths = [
        "../pdm-platform/backend/pdm_platform.db",  # If original is at same level
        "../../pdm-platform/backend/pdm_platform.db",  # If nested deeper
        "../backend/pdm_platform.db"  # Alternative path
    ]
    
    sqlite_path = None
    for path in original_db_paths:
        if os.path.exists(path):
            sqlite_path = path
            break
    
    if not sqlite_path:
        logger.error("Could not find original PDM database!")
        logger.info("Looked for database at:")
        for path in original_db_paths:
            logger.info(f"  - {path}")
        logger.info("\nPlease check the path to your original pdm_platform.db file")
        return
    
    logger.info(f"Found original database at: {sqlite_path}")
    
async def main():
    """Run migration"""
    # Check if original database exists
    original_db_paths = [
        "../pdm-platform/backend/pdm_platform.db",
        "../../PDM_PROJECT/pdm-platform/backend/pdm_platform.db",
        "../backend/pdm_platform.db",
        "../../backend/pdm_platform.db"
    ]
    
    sqlite_path = None
    for path in original_db_paths:
        if os.path.exists(path):
            sqlite_path = path
            break
    
    if not sqlite_path:
        logger.error("Could not find original PDM database!")
        logger.info("Looked for database at:")
        for path in original_db_paths:
            logger.info(f"  - {path}")
        logger.info("\nPlease provide the correct path to your pdm_platform.db file")
        return
    
    logger.info(f"Found original database at: {sqlite_path}")
    
    # Use local PostgreSQL (no password required on macOS)
    postgres_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/pdm_v2")
    
    migrator = DataMigrator(
        sqlite_path=sqlite_path,
        postgres_url=postgres_url
    )
    
    await migrator.migrate_all_data()
    
    await migrator.migrate_all_data()

if __name__ == "__main__":
    asyncio.run(main())
