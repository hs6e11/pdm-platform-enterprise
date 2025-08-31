# scripts/migrate_from_phase1.py
"""
Data migration script from PDM Platform Phase 1 to Phase 2
Migrates 15,247+ sensor readings while preserving data integrity
"""

import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import uuid
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import os
import sys
from pathlib import Path
import argparse
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.config import settings
from api.database.connection import get_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationStats:
    """Track migration statistics"""
    def __init__(self):
        self.start_time = datetime.now()
        self.total_records = 0
        self.migrated_records = 0
        self.failed_records = 0
        self.skipped_records = 0
        self.tenants_created = 0
        self.machines_mapped = 0
        self.errors = []
    
    def log_summary(self):
        duration = datetime.now() - self.start_time
        logger.info(f"""
Migration Summary:
================
Duration: {duration.total_seconds():.2f} seconds
Total Phase 1 records: {self.total_records}
Successfully migrated: {self.migrated_records}
Failed migrations: {self.failed_records}
Skipped records: {self.skipped_records}
Tenants created: {self.tenants_created}
Machines mapped: {self.machines_mapped}
Success rate: {(self.migrated_records / max(self.total_records, 1)) * 100:.2f}%
        """)
        
        if self.errors:
            logger.warning(f"Errors encountered: {len(self.errors)}")
            for error in self.errors[-5:]:  # Show last 5 errors
                logger.warning(f"  - {error}")

class DataMigrator:
    """Handles the complete migration from Phase 1 to Phase 2"""
    
    def __init__(self, phase1_db_path: str, dry_run: bool = False):
        self.phase1_db_path = phase1_db_path
        self.dry_run = dry_run
        self.stats = MigrationStats()
        self.tenant_mapping = {}
        self.machine_mapping = {}
        
        # Phase 2 database connection
        self.pg_engine = get_engine()
        self.pg_session_maker = sessionmaker(bind=self.pg_engine)
        
        logger.info(f"Initializing migration from {phase1_db_path}")
        logger.info(f"Dry run mode: {dry_run}")
    
    def validate_phase1_database(self) -> bool:
        """Validate Phase 1 database exists and has expected structure"""
        if not os.path.exists(self.phase1_db_path):
            logger.error(f"Phase 1 database not found: {self.phase1_db_path}")
            return False
        
        try:
            conn = sqlite3.connect(self.phase1_db_path)
            cursor = conn.cursor()
            
            # Check for expected tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['sensor_data', 'machines']  # Adjust based on actual schema
            missing_tables = [table for table in expected_tables if table not in tables]
            
            if missing_tables:
                logger.warning(f"Missing tables in Phase 1 database: {missing_tables}")
            
            # Count total records
            cursor.execute("SELECT COUNT(*) FROM sensor_data;")
            self.stats.total_records = cursor.fetchone()[0]
            
            logger.info(f"Phase 1 database validated. Found {self.stats.total_records} sensor readings")
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 database validation failed: {e}")
            return False
    
    def create_default_tenant(self) -> str:
        """Create default tenant for Phase 1 data (Egypt facility)"""
        tenant_id = str(uuid.uuid4())
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create tenant: Egypt Manufacturing (ID: {tenant_id})")
            return tenant_id
        
        try:
            with self.pg_session_maker() as session:
                # Generate API key for the tenant
                api_key = f"egypt_manufacturing_{uuid.uuid4().hex[:16]}"
                
                # Insert tenant
                insert_query = text("""
                    INSERT INTO tenants.tenants 
                    (id, name, country, compliance_level, api_key, encryption_key, created_at)
                    VALUES (:id, :name, :country, :compliance_level, :api_key, :encryption_key, :created_at)
                """)
                
                session.execute(insert_query, {
                    'id': tenant_id,
                    'name': 'Egypt Manufacturing Facility',
                    'country': 'EG',
                    'compliance_level': 'basic',
                    'api_key': api_key,
                    'encryption_key': uuid.uuid4().hex,
                    'created_at': datetime.utcnow()
                })
                
                session.commit()
                self.stats.tenants_created += 1
                
                logger.info(f"Created tenant: Egypt Manufacturing (ID: {tenant_id}, API Key: {api_key})")
                return tenant_id
                
        except Exception as e:
            logger.error(f"Failed to create default tenant: {e}")
            self.stats.errors.append(f"Tenant creation failed: {e}")
            raise
    
    def create_machine_mapping(self, tenant_id: str) -> Dict[str, str]:
        """Create mapping between Phase 1 machine IDs and Phase 2 format"""
        # Based on whitepaper, Phase 1 has these machines:
        # EG_M001: CNC Mill Alpha, EG_M002: Assembly Line Beta, etc.
        
        phase1_machines = {
            'EG_M001': {
                'name': 'CNC Mill Alpha',
                'type': 'cnc_mill',
                'location': 'Cairo Manufacturing Floor A',
                'sensors': ['temperature', 'spindle_speed', 'vibration']
            },
            'EG_M002': {
                'name': 'Assembly Line Beta', 
                'type': 'assembly_line',
                'location': 'Cairo Manufacturing Floor A',
                'sensors': ['conveyor_speed', 'efficiency']
            },
            'EG_M003': {
                'name': 'Press Machine Gamma',
                'type': 'press_machine', 
                'location': 'Cairo Manufacturing Floor B',
                'sensors': ['pressure', 'temperature', 'vibration']
            },
            'EG_M004': {
                'name': 'Quality Tester Delta',
                'type': 'quality_tester',
                'location': 'Cairo Quality Control',
                'sensors': ['precision', 'test_cycles']
            },
            'EG_M005': {
                'name': 'Packaging Unit Epsilon',
                'type': 'packaging_unit',
                'location': 'Cairo Packaging Area',
                'sensors': ['packaging_speed', 'efficiency']
            }
        }
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create machine mappings for {len(phase1_machines)} machines")
            return {machine_id: machine_id for machine_id in phase1_machines.keys()}
        
        try:
            with self.pg_session_maker() as session:
                for machine_id, machine_info in phase1_machines.items():
                    # Insert machine metadata (if you have a machines table)
                    # For now, we'll just track the mapping
                    self.machine_mapping[machine_id] = machine_id
                    
                session.commit()
                self.stats.machines_mapped = len(phase1_machines)
                
                logger.info(f"Created mappings for {len(phase1_machines)} machines")
                return self.machine_mapping
                
        except Exception as e:
            logger.error(f"Failed to create machine mappings: {e}")
            self.stats.errors.append(f"Machine mapping failed: {e}")
            raise
    
    def migrate_sensor_data(self, tenant_id: str) -> int:
        """Migrate sensor data from Phase 1 to Phase 2"""
        logger.info("Starting sensor data migration...")
        
        # Connect to Phase 1 database
        sqlite_conn = sqlite3.connect(self.phase1_db_path)
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_cursor = sqlite_conn.cursor()
        
        batch_size = 1000
        batch_count = 0
        migrated_count = 0
        
        try:
            # Get total count for progress tracking
            sqlite_cursor.execute("SELECT COUNT(*) FROM sensor_data")
            total_records = sqlite_cursor.fetchone()[0]
            logger.info(f"Migrating {total_records} sensor readings in batches of {batch_size}")
            
            # Process data in batches
            offset = 0
            while True:
                # Fetch batch from Phase 1
                sqlite_cursor.execute("""
                    SELECT machine_id, timestamp, sensor_type, value, anomaly_score
                    FROM sensor_data 
                    ORDER BY timestamp 
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))
                
                rows = sqlite_cursor.fetchall()
                if not rows:
                    break
                
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would migrate batch {batch_count + 1}: {len(rows)} records")
                    migrated_count += len(rows)
                else:
                    # Prepare batch for Phase 2 insertion
                    batch_data = []
                    for row in rows:
                        try:
                            # Convert Phase 1 format to Phase 2 format
                            record = (
                                tenant_id,  # tenant_id
                                row['machine_id'],  # machine_id
                                datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')),  # timestamp
                                row['sensor_type'],  # sensor_type
                                float(row['value']),  # value
                                float(row.get('anomaly_score', 0.0))  # anomaly_score
                            )
                            batch_data.append(record)
                            
                        except Exception as e:
                            logger.warning(f"Failed to convert record: {e}")
                            self.stats.failed_records += 1
                            self.stats.errors.append(f"Record conversion failed: {e}")
                    
                    # Insert batch into Phase 2
                    if batch_data:
                        try:
                            with self.pg_session_maker() as session:
                                insert_query = text("""
                                    INSERT INTO sensor_data 
                                    (tenant_id, machine_id, timestamp, sensor_type, value, anomaly_score)
                                    VALUES (:tenant_id, :machine_id, :timestamp, :sensor_type, :value, :anomaly_score)
                                """)
                                
                                for record in batch_data:
                                    session.execute(insert_query, {
                                        'tenant_id': record[0],
                                        'machine_id': record[1],
                                        'timestamp': record[2],
                                        'sensor_type': record[3],
                                        'value': record[4],
                                        'anomaly_score': record[5]
                                    })
                                
                                session.commit()
                                migrated_count += len(batch_data)
                                
                        except Exception as e:
                            logger.error(f"Failed to insert batch {batch_count + 1}: {e}")
                            self.stats.failed_records += len(batch_data)
                            self.stats.errors.append(f"Batch {batch_count + 1} insertion failed: {e}")
                
                batch_count += 1
                offset += batch_size
                
                # Progress logging
                if batch_count % 10 == 0:
                    progress = (migrated_count / total_records) * 100
                    logger.info(f"Migration progress: {migrated_count}/{total_records} ({progress:.1f}%)")
            
            self.stats.migrated_records = migrated_count
            logger.info(f"Sensor data migration completed. Migrated {migrated_count} records")
            
        except Exception as e:
            logger.error(f"Sensor data migration failed: {e}")
            self.stats.errors.append(f"Sensor data migration failed: {e}")
            raise
        finally:
            sqlite_conn.close()
        
        return migrated_count
    
    def verify_migration(self, tenant_id: str) -> bool:
        """Verify migration completed successfully"""
        logger.info("Verifying migration...")
        
        if self.dry_run:
            logger.info("[DRY RUN] Migration verification skipped")
            return True
        
        try:
            with self.pg_session_maker() as session:
                # Count migrated records
                count_query = text("""
                    SELECT COUNT(*) as count 
                    FROM sensor_data 
                    WHERE tenant_id = :tenant_id
                """)
                
                result = session.execute(count_query, {'tenant_id': tenant_id})
                migrated_count = result.fetchone().count
                
                # Check data quality
                quality_query = text("""
                    SELECT 
                        machine_id,
                        sensor_type,
                        COUNT(*) as reading_count,
                        AVG(value) as avg_value,
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest
                    FROM sensor_data 
                    WHERE tenant_id = :tenant_id
                    GROUP BY machine_id, sensor_type
                    ORDER BY machine_id, sensor_type
                """)
                
                quality_results = session.execute(quality_query, {'tenant_id': tenant_id})
                
                logger.info(f"Migration verification results:")
                logger.info(f"  Total migrated records: {migrated_count}")
                logger.info(f"  Expected records: {self.stats.total_records}")
                logger.info(f"  Data quality check:")
                
                for row in quality_results:
                    logger.info(f"    {row.machine_id}.{row.sensor_type}: {row.reading_count} readings, "
                              f"avg={row.avg_value:.2f}, {row.earliest} to {row.latest}")
                
                # Verify count matches
                if migrated_count == self.stats.migrated_records:
                    logger.info("✅ Migration verification passed")
                    return True
                else:
                    logger.warning(f"⚠️ Record count mismatch: expected {self.stats.migrated_records}, found {migrated_count}")
                    return False
                    
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False
    
    def run_migration(self) -> bool:
        """Execute the complete migration process"""
        logger.info("=== Starting PDM Platform Phase 1 to Phase 2 Migration ===")
        
        try:
            # Step 1: Validate Phase 1 database
            if not self.validate_phase1_database():
                return False
            
            # Step 2: Create default tenant
            tenant_id = self.create_default_tenant()
            
            # Step 3: Create machine mappings
            machine_mapping = self.create_machine_mapping(tenant_id)
            
            # Step 4: Migrate sensor data
            migrated_count = self.migrate_sensor_data(tenant_id)
            
            # Step 5: Verify migration
            verification_passed = self.verify_migration(tenant_id)
            
            # Step 6: Log final statistics
            self.stats.log_summary()
            
            if verification_passed and migrated_count > 0:
                logger.info("🎉 Migration completed successfully!")
                
                if not self.dry_run:
                    # Save tenant information for Phase 2 use
                    tenant_info_file = Path(__file__).parent / "tenant_info.json"
                    tenant_info = {
                        'tenant_id': tenant_id,
                        'name': 'Egypt Manufacturing Facility',
                        'migration_date': datetime.utcnow().isoformat(),
                        'migrated_records': migrated_count,
                        'machine_mapping': machine_mapping
                    }
                    
                    with open(tenant_info_file, 'w') as f:
                        json.dump(tenant_info, f, indent=2)
                    
                    logger.info(f"Tenant information saved to: {tenant_info_file}")
                
                return True
            else:
                logger.error("Migration failed verification")
                return False
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.stats.errors.append(f"Migration failed: {e}")
            return False

def main():
    """Main migration script entry point"""
    parser = argparse.ArgumentParser(description='Migrate data from PDM Platform Phase 1 to Phase 2')
    parser.add_argument('--phase1-db', required=True, help='Path to Phase 1 SQLite database')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without actual migration')
    parser.add_argument('--backup', action='store_true', help='Create backup of Phase 1 database before migration')
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_path = f"{args.phase1_db}.backup_{int(time.time())}"
        import shutil
        shutil.copy2(args.phase1_db, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Run migration
    migrator = DataMigrator(args.phase1_db, args.dry_run)
    success = migrator.run_migration()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
