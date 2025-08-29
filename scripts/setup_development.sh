#!/bin/bash

echo "Setting up PDM Platform v2.0 development environment..."

# Install Python dependencies
pip3 install -r requirements.txt

# Start databases with Docker
docker-compose up -d postgres redis

# Wait for databases
echo "Waiting for databases to start..."
sleep 10

# Run database migrations
python3 scripts/migrate_from_phase1.py

# Start API in development mode
echo "Starting development servers..."
echo "API will be available at http://localhost:8001"
echo "Original Phase 1 system remains at http://localhost:8000"

cd api && python3 main.py
