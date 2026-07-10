#!/usr/bin/env bash
# Start the Azure Deployment Issue Analyzer
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -r "$DIR/requirements.txt" -q
fi

if [ ! -f "$DIR/.env" ] && [ -f "$DIR/.env.example" ]; then
  echo "Tip: Copy .env.example to .env and add your API key."
fi

echo "Starting Azure Deployment Issue Analyzer at http://localhost:8501"
"$VENV/bin/streamlit" run "$DIR/app.py" --server.port 8501
