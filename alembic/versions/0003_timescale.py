from alembic import op

# revision identifiers
revision = "0003_timescale"
down_revision = "0002_audit"
branch_labels = None
depends_on = None

def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # Drop any PK/UNIQUE constraints that don't include timestamp
    op.execute("""
    DO $$
    DECLARE r record;
    BEGIN
      FOR r IN
        SELECT conname
          FROM pg_constraint
         WHERE conrelid='sensor_readings'::regclass
           AND contype IN ('p','u')
           AND position('timestamp' in lower(pg_get_constraintdef(oid)))=0
      LOOP
        EXECUTE format('ALTER TABLE sensor_readings DROP CONSTRAINT %I', r.conname);
      END LOOP;
    END
    $$;
    """)

    # Drop UNIQUE indexes without timestamp
    op.execute("""
    DO $$
    DECLARE r record;
    BEGIN
      FOR r IN
        SELECT indexname
          FROM pg_indexes
         WHERE tablename='sensor_readings'
           AND indexdef ILIKE 'CREATE UNIQUE INDEX %'
           AND indexdef NOT ILIKE '%(timestamp%'
      LOOP
        EXECUTE format('DROP INDEX %I', r.indexname);
      END LOOP;
    END
    $$;
    """)

    # Create hypertable
    op.execute("""
    SELECT create_hypertable('sensor_readings','timestamp', if_not_exists => TRUE, migrate_data => TRUE);
    """)

    # Add PK if none exists
    op.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
          SELECT 1 FROM pg_constraint
           WHERE conrelid='sensor_readings'::regclass
             AND contype='p'
      ) THEN
         ALTER TABLE sensor_readings ADD PRIMARY KEY (id, timestamp);
      END IF;
    END
    $$;
    """)

    # Compression & policies (guarded)
    op.execute("""
    ALTER TABLE sensor_readings SET (
      timescaledb.compress,
      timescaledb.compress_orderby = 'timestamp DESC',
      timescaledb.compress_segmentby = 'tenant_id, machine_id'
    );
    """)
    op.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs
         WHERE hypertable_name='sensor_readings' AND proc_name='policy_compression'
      ) THEN
        PERFORM add_compression_policy('sensor_readings', compress_after => INTERVAL '7 days');
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs
         WHERE hypertable_name='sensor_readings' AND proc_name='policy_retention'
      ) THEN
        PERFORM add_retention_policy('sensor_readings', drop_after => INTERVAL '180 days');
      END IF;
    END
    $$;
    """)

def downgrade():
    op.execute("SELECT remove_compression_policy('sensor_readings');")
    op.execute("SELECT remove_retention_policy('sensor_readings');")
    # keep hypertable in downgrade (can't un-hypertable safely)

