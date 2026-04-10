#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/finsight}"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/finsight.git}"
BRANCH="${BRANCH:-main}"
ENV_FILE="${ENV_FILE:-.env}"

echo "Preparing FinSight deployment in: ${APP_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install docker and docker compose before running this script." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose is not available. Install the Compose plugin before running this script." >&2
  exit 1
fi

if [ ! -d "${APP_DIR}/.git" ]; then
  mkdir -p "${APP_DIR}"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"

git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

if [ ! -f "${ENV_FILE}" ]; then
  cp .env.example "${ENV_FILE}"
  echo "Created ${ENV_FILE}. Populate secrets before continuing."
fi

docker compose pull || true
docker compose up -d --build
docker compose ps

echo "FinSight is deployed. Check health at /health on the exposed API port."
