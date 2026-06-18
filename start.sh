#!/usr/bin/env bash
set -euo pipefail

python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the API to accept connections before starting Streamlit.
for _ in $(seq 1 60); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)" 2>/dev/null; then
    break
  fi
  sleep 1
done

exec streamlit run frontend/app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --browser.gatherUsageStats=false
