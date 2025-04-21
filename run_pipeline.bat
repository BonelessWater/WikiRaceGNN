@echo off
REM Script to run the complete WikiRaceGNN pipeline on Windows

SETLOCAL

REM Set the number of nodes for the graph
SET MAX_NODES=1000

echo === WikiRaceGNN Pipeline ===
echo This script will generate data, train the model, and evaluate it.
echo Max nodes: %MAX_NODES%

REM Create necessary directories
if not exist data mkdir data
if not exist models mkdir models
if not exist plots mkdir plots

REM Step 1: Generate the graph data
echo.
echo === Step 1: Generating Graph Data ===
poetry run python main.py --mode data --max_nodes %MAX_NODES%

REM Step 2: Run the complete pipeline
echo.
echo === Step 2: Running Pipeline (Train + Evaluate) ===
poetry run python main.py --mode pipeline --max_nodes %MAX_NODES%

echo.
echo === Pipeline Complete! ===
echo The model has been trained and evaluated. You can now use it for traversal.
echo Example command: poetry run python main.py --mode traverse --visualize

ENDLOCAL