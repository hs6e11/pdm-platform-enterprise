from alembic import op

# revision identifiers
revision = "0002_audit"
down_revision = "0001_init"
branch_labels = None
depends_on = None

def upgrade():
    op.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ DEFAULT NOW(),
        event TEXT NOT NULL,
        user_id UUID NULL,
        tenant_id UUID NULL,
        role TEXT NULL,
        client_ip TEXT NULL,
        method TEXT NULL,
        path TEXT NULL,
        rid TEXT NULL,
        details JSONB NULL
    );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_logs (ts);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_audit_tenant_ts ON audit_logs (tenant_id, ts);")

def downgrade():
    op.execute("DROP TABLE IF EXISTS audit_logs;")

