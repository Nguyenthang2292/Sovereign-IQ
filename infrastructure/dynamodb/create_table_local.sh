#!/bin/bash
# DynamoDB Local Setup Script
# ==========================
#
# Starts DynamoDB Local via Docker and creates the AutoTrade table.
#
# Usage:
#   ./create_table_local.sh        # Start and create table
#   ./create_table_local.sh only   # Just start DynamoDB Local
#   ./create_table_local.sh clean  # Stop and remove container
#
# Requirements:
#   - Docker installed and running
#   - AWS CLI configured (for local credentials)
#
# Created: 2026-02-20

set -e

CONTAINER_NAME="dynamodb-local"
PORT=8000
IMAGE="amazon/dynamodb-local:latest"

start_dynamodb() {
    echo "Starting DynamoDB Local on port $PORT..."
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' already exists."
        
        # Check if it's running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "DynamoDB Local is already running."
        else
            echo "Starting existing container..."
            docker start $CONTAINER_NAME
        fi
    else
        echo "Creating and starting new DynamoDB Local container..."
        docker run -d \
            --name $CONTAINER_NAME \
            -p $PORT:8000 \
            $IMAGE \
            -jar DynamoDBLocal.jar -sharedDb -inMemory
    fi
    
    echo "DynamoDB Local is ready at http://localhost:$PORT"
}

create_table() {
    echo "Creating AutoTrade table..."
    
    cd "$(dirname "$0")"
    python create_table.py --env local --no-wait
    
    echo "Table created successfully!"
}

stop_dynamodb() {
    echo "Stopping DynamoDB Local..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo "DynamoDB Local stopped and removed."
}

case "${1:-}" in
    only)
        start_dynamodb
        ;;
    clean)
        stop_dynamodb
        ;;
    "")
        start_dynamodb
        create_table
        ;;
    *)
        echo "Usage: $0 [only|clean]"
        echo "  (no args) - Start DynamoDB Local and create table"
        echo "  only      - Only start DynamoDB Local"
        echo "  clean     - Stop and remove container"
        exit 1
        ;;
esac
