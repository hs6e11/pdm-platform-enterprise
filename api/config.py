import os
from pydantic import BaseModel

class Settings(BaseModel):
    database_url: str = os.getenv("DATABASE_URL", "postgres://postgres:password@localhost:5433/pdm_v2")
    secret_key: str = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PROD")
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = int(os.getenv("ACCESS_TOKEN_EXP_MINUTES", "60"))
    allowed_origins: list[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

    # Redis for rate limits (and future queues)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Simple fixed-window rate limits
    rl_tenant_per_min: int = int(os.getenv("RL_TENANT_PER_MIN", "1200"))  # msgs/min/tenant
    rl_ip_per_min: int = int(os.getenv("RL_IP_PER_MIN", "600"))           # requests/min/ip

settings = Settings()

