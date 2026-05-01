#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker/compose.yaml"
SERVICE="${SERVICE:-em_pathplanning}"

docker compose -f "$COMPOSE_FILE" up --build -d "$SERVICE"

echo "Dev container started."
echo "Use: docker compose -f $COMPOSE_FILE logs -f $SERVICE"
echo "Shell in with: docker compose -f $COMPOSE_FILE exec $SERVICE bash"
