#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8010}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://$BACKEND_HOST:$BACKEND_PORT}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

install_backend_deps() {
  local marker="$VENV_DIR/.requirements-installed"

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating backend virtual environment..."
    python3 -m venv "$VENV_DIR"
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "Installing backend dependencies..."
    "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
    touch "$marker"
  elif [[ -f "$marker" && "$BACKEND_DIR/requirements.txt" -nt "$marker" ]]; then
    echo "Backend requirements changed; refreshing dependencies..."
    "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
    touch "$marker"
  elif [[ ! -f "$marker" ]]; then
    touch "$marker"
  fi
}

install_frontend_deps() {
  local marker="$FRONTEND_DIR/.package-lock-installed"

  if [[ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]]; then
    echo "Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
    touch "$marker"
  elif [[ -f "$marker" && "$FRONTEND_DIR/package-lock.json" -nt "$marker" ]]; then
    echo "Frontend package lock changed; refreshing dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
    touch "$marker"
  elif [[ ! -f "$marker" ]]; then
    touch "$marker"
  fi
}

trap cleanup EXIT INT TERM

require_command python3
require_command npm

install_backend_deps
install_frontend_deps

echo "Starting backend:  http://$BACKEND_HOST:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  "$VENV_DIR/bin/python" -m uvicorn app.main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) &
BACKEND_PID=$!

echo "Starting frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo
echo "Open http://$FRONTEND_HOST:$FRONTEND_PORT in your browser."
echo "Press Ctrl+C to stop both servers."

while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 1
done
