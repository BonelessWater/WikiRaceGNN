#!/bin/bash
# Script to run the complete WikiRaceGNN pipeline

# Make the script exit on error
set -e

# Set the number of nodes for the graph
MAX_NODES=1000

# Check if Python is available through Poetry
if ! command -v poetry &> /dev/null
then
    echo "Poetry is not installed. Please install it first."
    exit 1
fi

echo "=== WikiRaceGNN Pipeline ==="
echo "This script will generate data, train the model, and evaluate it."
echo "Max nodes: $MAX_NODES"

# Create necessary directories
mkdir -p data models plots

# Step 1: Generate the graph data
echo -e "\n=== Step 1: Generating Graph Data ==="
poetry run python main.py --mode data --max_nodes $MAX_NODES

# Step 2: Run the complete pipeline
echo -e "\n=== Step 2: Running Pipeline (Train + Evaluate) ==="
poetry run python main.py --mode pipeline --max_nodes $MAX_NODES

echo -e "\n=== Pipeline Complete! ==="
echo "The model has been trained and evaluated. You can now use it for traversal."
echo "Example command: poetry run python main.py --mode traverse --visualize"