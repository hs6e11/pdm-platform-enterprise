from alembic import op

# revision identifiers
revision = "0004_fix_compression_orderby"
down_revision = "0003_timescale"
branch_labels = None
depends_on = None

def upgrade():
    # Ensure compression config is compatible with PK (id,timestamp)
    op.execute("""
    ALTER TABLE sensor_readings SET (
      timescaledb.compress,
      timescaledb.compress_orderby = 'timestamp DESC, id',
      timescaledb.compress_segmentby = 'tenant_id, machine_id'
    );
    """)
    # Re-assert policies if missing (idempotent)
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
    # keep compression; no-op
    pass

