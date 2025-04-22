#!/bin/bash

CONTAINER_NAME="tracker"
IMAGE_NAME="tracker:latest"

echo "Stopping old container..."
docker stop $CONTAINER_NAME 2>/dev/null || true

echo "Removing old container..."
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "Build image..."
docker build -t $IMAGE_NAME .

echo "Starting new container..."
docker run -d \
  --network host \
  --name $CONTAINER_NAME \
  --device /dev/input \
  --device /dev/uinput \
  --device /dev/hidraw0 \
  --cap-add SYS_ADMIN \
  --cap-add NET_ADMIN \
  --privileged \
  $IMAGE_NAME
