#!/bin/bash

# run_app.sh
# Script to install dependencies (if needed) and run ui.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Check and install requirements
if [ ! -d "venv" ]; then
    echo "Setting up the environment and installing requirements..."
    ./install.sh
else
    echo "Environment already set up."
    source venv/bin/activate
fi

# Change into the app/ directory
cd app/

# Run the Python script
echo "Running ui.py..."
python3 ui.py
