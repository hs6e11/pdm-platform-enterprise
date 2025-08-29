# PDM Platform v2.0 - Multi-Tenant Architecture

This is the Phase 2 evolution of the PDM Platform, built alongside the existing Phase 1 system.

## Migration Strategy

**IMPORTANT**: This v2 system is built to work alongside your existing Phase 1 system:
- Phase 1 continues running on ports 8000/8080
- Phase 2 runs on ports 8001/3001  
- Gradual migration of components over time

## Quick Start

1. **Setup development environment**:
```bash
./scripts/setup_development.sh
```

2. **Start the system**:
```bash
docker-compose up -d
```

3. **Run data migration** (preserves existing data):
```bash
python3 scripts/migrate_from_phase1.py
```

## Architecture Overview

- **API Layer**: Multi-tenant FastAPI application (port 8001)
- **IoT Gateway**: Multi-protocol support (Modbus, OPC-UA, MQTT)
- **ML Pipeline**: Edge computing with cognitive maintenance
- **Database**: PostgreSQL + TimescaleDB for time-series data
- **Cache**: Redis for session management and caching

## Key Improvements Over Phase 1

- ✅ Multi-tenant architecture (multiple clients)
- ✅ Real industrial protocol support
- ✅ Advanced ML with edge computing
- ✅ EU CRA/NIS2 compliance ready
- ✅ Production-grade database (TimescaleDB)
- ✅ Horizontal scalability

## Development

### Run API only:
```bash
cd api && python3 main.py
```

### Run IoT Gateway:
```bash
cd iot-gateway && python3 gateway.py
```

### Run ML Pipeline:
```bash
cd ml-pipeline && python3 orchestrator.py
```

## Migration Plan

1. **Week 1**: Test v2 API with existing Egypt data
2. **Week 2**: Deploy new IoT gateway alongside existing
3. **Week 3**: Migrate one machine to real protocols
4. **Week 4**: Add second client (multi-tenant test)

Your original system continues working throughout this process.
