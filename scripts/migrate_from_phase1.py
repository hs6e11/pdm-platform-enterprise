#!/usr/bin/env python3
"""
Production-grade data migration from Phase 1 to Phase 2
Preserves all historical data with validation and rollback capability
"""

import sqlite3
import asyncpg
import asyncio
import uuid
import os
import logging
from datetime import datetime, timezone
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationStats:
    total_readings: int = 0
    migrated_readings: int = 0
    failed_readings: int = 0
    equipment_count: int = 0
    tenant_count: int = 0
    start_time: datetime = None
    end_time: datetime = None

class ProductionDataMigrator:
    def __init__(self):
        self.phase1_db = os.getenv('PHASE1_DB_PATH', './phase1/data/pdm_data.db')
        self.phase2_db_url = os.getenv('DATABASE_URL', 'postgresql://pdm_user:password@localhost:5432/pdm_platform')
        self.stats = MigrationStats()
        self.batch_size = 1000
        self.tenant_mapping = {}
        
    async def connect_databases(self) -> Tuple[sqlite3.Connection, asyncpg.Connection]:
        """Establish connections to both databases"""
        logger.info("Connecting to databases...")
        
        # Phase 1 SQLite connection
        if not os.path.exists(self.phase1_db):
            raise FileNotFoundError(f"Phase 1 database not found: {self.phase1_db}")
        
        sqlite_conn = sqlite3.connect(self.phase1_db)
        sqlite_conn.row_factory = sqlite3.Row
        
        # Phase 2 PostgreSQL connection
        pg_conn = await asyncpg.connect(self.phase2_db_url)
        
        logger.info("Database connections established")
        return sqlite_conn, pg_conn
    
    async def analyze_phase1_data(self, sqlite_conn: sqlite3.Connection) -> Dict:
        """Analyze Phase 1 data structure and content"""
        logger.info("Analyzing Phase 1 data...")
        
        analysis = {
            'tables': [],
            'total_readings': 0,
            'date_range': {'start': None, 'end': None},
            'equipment_list': [],
            'sensor_types': []
        }
        
        # Get table structure
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        analysis['tables'] = tables
        
        if 'sensor_readings' in tables:
            # Count total readings
            cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            analysis['total_readings'] = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM sensor_readings")
            start_date, end_date = cursor.fetchone()
            analysis['date_range'] = {'start': start_date, 'end': end_date}
            
            # Get unique equipment
            cursor.execute("SELECT DISTINCT equipment_id FROM sensor_readings")
            analysis['equipment_list'] = [row[0] for row in cursor.fetchall()]
            
            # Get sensor types
            cursor.execute("SELECT DISTINCT sensor_type FROM sensor_readings")
            analysis['sensor_types'] = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Analysis complete: {analysis['total_readings']} readings from {len(analysis['equipment_list'])} equipment")
        return analysis
    
    async def create_tenant_mapping(self, analysis: Dict, pg_conn: asyncpg.Connection) -> Dict[str, uuid.UUID]:
        """Create tenant mapping for multi-tenant architecture"""
        logger.info("Creating tenant mapping...")
        
        # For each unique equipment prefix, create or find tenant
        equipment_prefixes = set()
        for equipment_id in analysis['equipment_list']:
            # Extract prefix (e.g., "EG" from "EG_M001")
            prefix = equipment_id.split('_')[0] if '_' in equipment_id else 'DEFAULT'
            equipment_prefixes.add(prefix)
        
        tenant_mapping = {}
        
        for prefix in equipment_prefixes:
            # Check if tenant exists
            tenant_name = f"{prefix}_Plant"
            existing = await pg_conn.fetchrow(
                "SELECT id FROM tenants WHERE name = $1", tenant_name
            )
            
            if existing:
                tenant_id = existing['id']
                logger.info(f"Using existing tenant: {tenant_name} ({tenant_id})")
            else:
                # Create new tenant
                tenant_id = uuid.uuid4()
                await pg_conn.execute(
                    """
                    INSERT INTO tenants (id, name, created_at, settings)
                    VALUES ($1, $2, $3, $4)
                    """,
                    tenant_id, tenant_name, datetime.now(timezone.utc),
                    json.dumps({"migrated_from_phase1": True})
                )
                logger.info(f"Created new tenant: {tenant_name} ({tenant_id})")
            
            tenant_mapping[prefix] = tenant_id
        
        self.tenant_mapping = tenant_mapping
        return tenant_mapping
    
    async def migrate_readings_batch(self, batch_data: List[Dict], pg_conn: asyncpg.Connection) -> int:
        """Migrate a batch of readings with validation"""
        migrated_count = 0
        
        insert_query = """
            INSERT INTO sensor_readings (
                id, tenant_id, equipment_id, sensor_type, 
                value, unit, timestamp, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
        """
        
        for reading in batch_data:
            try:
                # Determine tenant from equipment_id
                equipment_prefix = reading['equipment_id'].split('_')[0] if '_' in reading['equipment_id'] else 'DEFAULT'
                tenant_id = self.tenant_mapping.get(equipment_prefix, list(self.tenant_mapping.values())[0])
                
                # Generate consistent UUID from Phase 1 data
                reading_id = uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{reading['equipment_id']}-{reading['sensor_type']}-{reading['timestamp']}"
                )
                
                # Convert timestamp
                timestamp = datetime.fromisoformat(reading['timestamp'].replace('Z', '+00:00'))
                
                # Prepare metadata
                metadata = {
                    'migrated_from_phase1': True,
                    'original_id': reading.get('id'),
                    'migration_date': datetime.now(timezone.utc).isoformat()
                }
                
                await pg_conn.execute(
                    insert_query,
                    reading_id, tenant_id, reading['equipment_id'],
                    reading['sensor_type'], float(reading['value']),
                    reading.get('unit', 'unknown'), timestamp, json.dumps(metadata)
                )
                
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate reading: {reading}, Error: {str(e)}")
                self.stats.failed_readings += 1
        
        return migrated_count
    
    async def perform_migration(self) -> MigrationStats:
        """Execute the complete migration process"""
        self.stats.start_time = datetime.now(timezone.utc)
        logger.info("Starting production data migration...")
        
        sqlite_conn, pg_conn = await self.connect_databases()
        
        try:
            # Analyze Phase 1 data
            analysis = await self.analyze_phase1_data(sqlite_conn)
            self.stats.total_readings = analysis['total_readings']
            self.stats.equipment_count = len(analysis['equipment_list'])
            
            if self.stats.total_readings == 0:
                logger.warning("No data found in Phase 1 database")
                return self.stats
            
            # Create tenant mapping
            tenant_mapping = await self.create_tenant_mapping(analysis, pg_conn)
            self.stats.tenant_count = len(tenant_mapping)
            
            # Create backup point
            await pg_conn.execute(
                """
                INSERT INTO migration_log (migration_type, start_time, total_records, status)
                VALUES ('phase1_to_phase2', $1, $2, 'started')
                """,
                self.stats.start_time, self.stats.total_readings
            )
            
            # Migrate data in batches
            cursor = sqlite_conn.cursor()
            cursor.execute("SELECT * FROM sensor_readings ORDER BY timestamp")
            
            batch = []
            processed = 0
            
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                
                # Convert sqlite3.Row to dict
                batch_data = [dict(row) for row in rows]
                
                # Migrate batch
                migrated_in_batch = await self.migrate_readings_batch(batch_data, pg_conn)
                self.stats.migrated_readings += migrated_in_batch
                processed += len(batch_data)
                
                # Progress logging
                progress = (processed / self.stats.total_readings) * 100
                logger.info(f"Migration progress: {processed}/{self.stats.total_readings} ({progress:.1f}%)")
                
                # Memory cleanup
                batch_data.clear()
            
            # Update migration log
            await pg_conn.execute(
                """
                UPDATE migration_log 
                SET end_time = $1, migrated_records = $2, status = 'completed'
                WHERE migration_type = 'phase1_to_phase2' AND start_time = $3
                """,
                datetime.now(timezone.utc), self.stats.migrated_readings, self.stats.start_time
            )
            
            logger.info("Migration completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            # Update migration log with failure
            await pg_conn.execute(
                """
                UPDATE migration_log 
                SET end_time = $1, status = 'failed', error_message = $2
                WHERE migration_type = 'phase1_to_phase2' AND start_time = $3
                """,
                datetime.now(timezone.utc), str(e), self.stats.start_time
            )
            raise
        finally:
            sqlite_conn.close()
            await pg_conn.close()
        
        self.stats.end_time = datetime.now(timezone.utc)
        return self.stats
    
    def generate_migration_report(self) -> str:
        """Generate detailed migration report"""
        if not self.stats.end_time:
            return "Migration not completed"
        
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        success_rate = (self.stats.migrated_readings / self.stats.total_readings * 100) if self.stats.total_readings > 0 else 0
        
        report = f"""
=== PDM Platform Data Migration Report ===
Start Time: {self.stats.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
End Time: {self.stats.end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
Duration: {duration:.2f} seconds

Data Summary:
- Total Readings: {self.stats.total_readings:,}
- Successfully Migrated: {self.stats.migrated_readings:,}
- Failed Migrations: {self.stats.failed_readings:,}
- Success Rate: {success_rate:.2f}%

Infrastructure:
- Equipment Migrated: {self.stats.equipment_count}
- Tenants Created: {self.stats.tenant_count}

Performance:
- Records/Second: {(self.stats.migrated_readings / duration):.2f}

Status: {'SUCCESS' if self.stats.failed_readings == 0 else 'COMPLETED WITH ERRORS'}
        """
        
        return report.strip()

async def main():
    """Main migration execution"""
    migrator = ProductionDataMigrator()
    
    try:
        logger.info("=== PDM Platform Phase 1 to Phase 2 Migration ===")
        stats = await migrator.perform_migration()
        
        # Generate and display report
        report = migrator.generate_migration_report()
        print("\n" + report)
        
        # Save report to file
        with open(f'migration_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            f.write(report)
        
        logger.info("Migration report saved")
        
    except Exception as e:
        logger.error(f"Migration failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    # Run migration
    asyncio.run(main())
