#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-em_pathplanning:latest}"

echo "Building image: $IMAGE_NAME"
docker build -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE_NAME" "$REPO_ROOT"

if [ "${PUSH_IMAGE:-0}" = "1" ]; then
  echo "Pushing image: $IMAGE_NAME"
  docker push "$IMAGE_NAME"
fi
