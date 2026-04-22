#!/bin/bash
# startup.sh
# ----------
# Azure App Service startup script for the Enterprise Knowledge Base API.
# Set as startup command in App Service Configuration:
#   bash /home/site/wwwroot/startup.sh

set -e

PORT="${PORT:-8001}"

echo "Starting Enterprise Knowledge Base API on port $PORT"

exec gunicorn api.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 2 \
  --bind "0.0.0.0:$PORT" \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -