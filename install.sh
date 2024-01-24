#!/bin/bash

# install.sh
# Script to set up the Python environment for ImaGym on Mac/Linux

echo "Starting installation for ImaGym..."

# Define the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Change to the directory where the script is located
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python 3 could not be found. Please install Python 3 and rerun this script."
    exit 1
fi

echo "Python 3 detected..."

# Install virtualenv if it's not already installed
if ! python3 -m pip show virtualenv &>/dev/null; then
    echo "Installing virtualenv..."
    python3 -m pip install virtualenv
fi

# Create a Python virtual environment
echo "Creating a Python virtual environment..."
python3 -m virtualenv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Change into the app/ directory
cd app/

# Check for and install requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing required Python packages..."
    python3 -m pip install -r requirements.txt
else
    echo "requirements.txt not found. Ensure you have the requirements.txt file in the same directory as this script."
    exit 1
fi

echo "Installation complete. You can now run the application."

# Additional instructions to run the application can be included here
# For example:
# echo "To run the application, use: python3 app.py"
