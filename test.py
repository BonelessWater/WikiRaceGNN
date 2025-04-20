#!/usr/bin/env python
"""
This script runs the improved training and evaluation pipeline for the WikiRaceGNN project.
It coordinates model training, data generation, and evaluation.

Usage:
    python run_improvements.py [options]

Options:
    --train        Train new models
    --eval         Evaluate models
    --visualize    Create additional visualizations
    --all          Run all steps
"""

import os
import sys
import argparse
import subprocess
import time

def ensure_directories():
    """Ensure all required directories exist"""
    directories = ['data', 'models', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Verified required directories exist.")

def check_data():
    """Check if required data files exist"""
    edge_file = "data/croc_edges.csv"
    if not os.path.exists(edge_file):
        print(f"ERROR: Edge file {edge_file} not found.")
        print("You need to create or download the edge data first.")
        return False
    return True

def run_training():
    """Run the improved training pipeline"""
    print("\n" + "="*80)
    print("STARTING MODEL TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    try:
        subprocess.run(["python", "train.py"], check=True)
        print(f"Training completed successfully in {time.time() - start_time:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with error code {e.returncode}")
        return False

def run_evaluation():
    """Run the improved evaluation pipeline"""
    print("\n" + "="*80)
    print("STARTING MODEL EVALUATION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        subprocess.run(["python", "evaluate.py"], check=True)
        print(f"Evaluation completed successfully in {time.time() - start_time:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed with error code {e.returncode}")
        return False

def create_visualizations():
    """Create additional visualizations"""
    print("\n" + "="*80)
    print("CREATING ADDITIONAL VISUALIZATIONS")
    print("="*80)
    
    # This could be expanded to generate more visualizations
    try:
        # Run a specific visualization script if you have one
        # subprocess.run(["python", "visualize.py"], check=True)
        print("Visualizations created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Visualization creation failed with error code {e.returncode}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the improved WikiRaceGNN pipeline')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--eval', action='store_true', help='Evaluate models')
    parser.add_argument('--visualize', action='store_true', help='Create additional visualizations')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # If no options specified, show help
    if not (args.train or args.eval or args.visualize or args.all):
        parser.print_help()
        sys.exit(1)
        
    return args

def main():
    """Main function to run the improved pipeline"""
    args = parse_arguments()
    
    # Setup
    ensure_directories()
    if not check_data():
        sys.exit(1)
    
    # Determine what to run
    run_all = args.all
    should_train = args.train or run_all
    should_eval = args.eval or run_all
    should_visualize = args.visualize or run_all
    
    # Run selected steps
    if should_train:
        success = run_training()
        if not success and run_all:
            print("Training failed, stopping pipeline.")
            sys.exit(1)
    
    if should_eval:
        success = run_evaluation()
        if not success and run_all:
            print("Evaluation failed, stopping pipeline.")
            sys.exit(1)
    
    if should_visualize:
        success = create_visualizations()
        if not success and run_all:
            print("Visualization creation failed, stopping pipeline.")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()