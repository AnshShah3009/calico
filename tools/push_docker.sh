#!/usr/bin/env bash
# Push both CALICO CPU and CUDA Docker images to Docker Hub.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dockerhub_username> [tag_version]"
    echo "Example: $0 anshshah3009 v1.0.0"
    exit 1
fi

DOCKER_USER="$1"
TAG_VER="${2:-latest}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== CALICO Docker Hub Publisher ==="
echo "Target User: $DOCKER_USER"
echo "Version Tag: $TAG_VER"

# Verify docker is logged in
if ! docker info 2>&1 | grep -q "Username:"; then
    echo "Notice: Not logged in to Docker Hub. Prompting docker login..."
    docker login
fi

echo "--> Building & tagging CPU image..."
docker build -t "${DOCKER_USER}/calico:cpu" -t "${DOCKER_USER}/calico:${TAG_VER}" -t "${DOCKER_USER}/calico:latest" -f "${ROOT_DIR}/Dockerfile.cpu" "${ROOT_DIR}"

echo "--> Building & tagging CUDA (GPU) image..."
docker build -t "${DOCKER_USER}/calico:cuda" -t "${DOCKER_USER}/calico:${TAG_VER}-cuda" -f "${ROOT_DIR}/Dockerfile.cuda" "${ROOT_DIR}"

echo "--> Pushing CPU images..."
docker push "${DOCKER_USER}/calico:cpu"
docker push "${DOCKER_USER}/calico:latest"
if [ "$TAG_VER" != "latest" ]; then
    docker push "${DOCKER_USER}/calico:${TAG_VER}"
fi

echo "--> Pushing CUDA (GPU) images..."
docker push "${DOCKER_USER}/calico:cuda"
if [ "$TAG_VER" != "latest" ]; then
    docker push "${DOCKER_USER}/calico:${TAG_VER}-cuda"
fi

echo "=== Successfully pushed both CPU and CUDA (GPU) images to Docker Hub! ==="
