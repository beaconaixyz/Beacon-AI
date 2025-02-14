#!/bin/bash

# BEACON Deployment Script

# Default values
ENV="prod"
PORT=8000
CONFIG_PATH="config.yml"
LOG_DIR="/var/log/beacon"
DATA_DIR="/data/beacon"
MODEL_DIR="/models/beacon"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -e|--environment)
            ENV="$2"
            shift
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift
            shift
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift
            shift
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
echo "Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"

# Set up environment variables
echo "Setting up environment variables..."
export BEACON_ENV="$ENV"
export BEACON_PORT="$PORT"
export BEACON_CONFIG="$CONFIG_PATH"
export BEACON_LOGS="$LOG_DIR"
export BEACON_DATA="$DATA_DIR"
export BEACON_MODELS="$MODEL_DIR"

# Check Python installation
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run database migrations if needed
if [ -f "manage.py" ]; then
    echo "Running database migrations..."
    python manage.py migrate
fi

# Start the application
echo "Starting BEACON application..."
case $ENV in
    "prod")
        gunicorn beacon.wsgi:application \
            --bind 0.0.0.0:$PORT \
            --workers 4 \
            --timeout 120 \
            --access-logfile "$LOG_DIR/access.log" \
            --error-logfile "$LOG_DIR/error.log" \
            --capture-output \
            --enable-stdio-inheritance \
            --daemon
        ;;
    "dev")
        python manage.py runserver 0.0.0.0:$PORT
        ;;
    *)
        echo "Unknown environment: $ENV"
        exit 1
        ;;
esac

# Check if application is running
sleep 5
if curl -s http://localhost:$PORT/health > /dev/null; then
    echo "BEACON application started successfully on port $PORT"
else
    echo "Failed to start BEACON application"
    exit 1
fi 