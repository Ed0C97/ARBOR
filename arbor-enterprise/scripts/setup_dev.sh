#!/bin/bash
# A.R.B.O.R. Enterprise - Development Environment Setup

set -e

echo "=== A.R.B.O.R. Enterprise - Dev Setup ==="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }

# Backend setup
echo "Setting up backend..."
cd backend
pip install poetry 2>/dev/null || true
poetry install
cd ..

# Frontend setup
echo "Setting up frontend..."
cd frontend
npm install
cd ..

# Copy env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file - please fill in your API keys"
fi

# Start Docker services
echo "Starting Docker services..."
docker-compose -f infrastructure/docker/docker-compose.dev.yml up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 10

# Run migrations
echo "Running database migrations..."
cd backend
poetry run alembic upgrade head
cd ..

echo ""
echo "=== Setup Complete ==="
echo "Start backend:  make backend"
echo "Start frontend: make frontend"
echo "View API docs:  http://localhost:8000/docs"
