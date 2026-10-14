#!/bin/bash

# Function to cleanup background processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# The repo root must be importable so `import optichat...` resolves.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Start the chat API (ADK agents + SQLite persistence) in background
echo "Starting OptiChat API..."
uv run uvicorn optichat.backend.api:app --port 8000 &
API_PID=$!

# Wait until the API answers (up to 60s)
echo "Waiting for the API to initialize..."
for _ in $(seq 1 60); do
    curl -sf http://localhost:8000/chats >/dev/null && break
    kill -0 $API_PID 2>/dev/null || { echo "The API failed to start."; exit 1; }
    sleep 1
done

# Start Streamlit App
echo "Starting Streamlit App..."
uv run streamlit run optichat/frontend/app.py

# Wait for background processes
wait $API_PID
