from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.execute("""
    CREATE TABLE IF NOT EXISTS tenants (
        id UUID PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY,
        tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('platform_admin','client_admin','operator','viewer')),
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS equipment (
        id UUID PRIMARY KEY,
        tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
        machine_id TEXT NOT NULL,
        name TEXT,
        location TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id BIGSERIAL PRIMARY KEY,
        tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
        machine_id TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)
    # indexes for performance
    op.execute("CREATE INDEX IF NOT EXISTS ix_sensor_tenant_time ON sensor_readings (tenant_id, timestamp);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sensor_tenant_machine_time ON sensor_readings (tenant_id, machine_id, timestamp);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_equipment_tenant_machine ON equipment (tenant_id, machine_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_users_tenant_role ON users (tenant_id, role);")

def downgrade():
    op.execute("DROP TABLE IF EXISTS sensor_readings;")
    op.execute("DROP TABLE IF EXISTS equipment;")
    op.execute("DROP TABLE IF EXISTS users;")
    op.execute("DROP TABLE IF EXISTS tenants;")

