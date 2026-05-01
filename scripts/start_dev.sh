#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker/compose.yaml"
SERVICE="${SERVICE:-em_pathplanning}"
ACTION="${1:-up}"

case "$ACTION" in
  up)
    docker compose -f "$COMPOSE_FILE" up --build "$SERVICE"
    ;;
  up-detached|detach)
    docker compose -f "$COMPOSE_FILE" up --build -d "$SERVICE"
    ;;
  down)
    docker compose -f "$COMPOSE_FILE" down
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f "$SERVICE"
    ;;
  shell)
    docker compose -f "$COMPOSE_FILE" exec "$SERVICE" bash
    ;;
  *)
    docker compose -f "$COMPOSE_FILE" "$@"
    ;;
esac
