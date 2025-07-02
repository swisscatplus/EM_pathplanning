$CONTAINER_NAME = "tracker"
$IMAGE_NAME = "tracker:latest"

Write-Host "Stopping old container..."
docker stop $CONTAINER_NAME 2>$null

Write-Host "Removing old container..."
docker rm $CONTAINER_NAME 2>$null

Write-Host "Building image..."
docker build -t $IMAGE_NAME .

Write-Host "Starting new container..."
docker run -d `
  --network host `
  --name $CONTAINER_NAME `
  --cap-add SYS_ADMIN `
  --cap-add NET_ADMIN `
  --privileged `
  $IMAGE_NAME
