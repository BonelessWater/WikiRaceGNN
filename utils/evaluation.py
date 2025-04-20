import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from traversal.utils import bidirectional_bfs

def compare_algorithms(data, algorithms, num_test_pairs=10, max_steps=30):
    """
    Compare different path finding algorithms.
    
    Args:
        data: Graph data object
        algorithms: Dictionary of {name: (algorithm_function, kwargs)}
        num_test_pairs: Number of source-target pairs to test
        max_steps: Maximum steps for each algorithm
        
    Returns:
        results: Dictionary with comparison results
        summary: Dictionary with summary statistics
    """
    # Initialize results dictionary
    results = {
        'source': [],
        'target': [],
        'path_length': defaultdict(list),
        'nodes_explored': defaultdict(list),
        'success': defaultdict(list),
        'time': defaultdict(list)
    }
    
    # Get list of all nodes
    all_nodes = list(range(data.x.size(0)))
    
    # Create a set of connected node pairs using BFS
    connected_pairs = []
    attempts = 0
    max_attempts = num_test_pairs * 5
    
    while len(connected_pairs) < num_test_pairs and attempts < max_attempts:
        source = random.choice(all_nodes)
        target = random.choice(all_nodes)
        
        # Skip if same node or already tested
        if source == target or (source, target) in connected_pairs:
            attempts += 1
            continue
            
        # Check if there's a path between them
        bfs_path, _ = bidirectional_bfs(data, source, target)
        if bfs_path and 2 <= len(bfs_path) <= 8:  # Ensure path exists and is reasonable length
            connected_pairs.append((source, target))
        
        attempts += 1
    
    print(f"Found {len(connected_pairs)} connected pairs after {attempts} attempts")
    
    # Test each pair with each algorithm
    for source, target in connected_pairs:
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        
        print(f"Testing path from node {source_id} to {target_id}")
        
        # Record source and target
        results['source'].append(source_id)
        results['target'].append(target_id)
        
        # Run each algorithm
        for name, (algo_func, kwargs) in algorithms.items():
            try:
                import time
                start_time = time.time()
                
                # Run the algorithm
                path, nodes_explored = algo_func(source_id, target_id, max_steps=max_steps, **kwargs)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Convert path to internal indices if it contains IDs
                if path and isinstance(path[0], type(source_id)):
                    path_idx = [data.node_mapping[node_id] for node_id in path if node_id in data.node_mapping]
                else:
                    path_idx = path
                
                # Check if target was reached
                success = path_idx and path_idx[-1] == target
                
                # Record results
                results['path_length'][name].append(len(path_idx) if path_idx else 0)
                results['nodes_explored'][name].append(nodes_explored)
                results['success'][name].append(success)
                results['time'][name].append(elapsed_time)
                
                print(f"{name}: Path length: {len(path_idx) if path_idx else 0}, "
                      f"Nodes explored: {nodes_explored}, Success: {success}, "
                      f"Time: {elapsed_time:.4f}s")
                
            except Exception as e:
                print(f"Error in {name}: {e}")
                results['path_length'][name].append(0)
                results['nodes_explored'][name].append(0)
                results['success'][name].append(False)
                results['time'][name].append(0)
        
        print("-" * 50)
    
    # Calculate summary statistics
    summary = {
        'avg_path_length': {},
        'avg_nodes_explored': {},
        'success_rate': {},
        'avg_time': {}
    }
    
    for name in algorithms.keys():
        # Calculate averages
        successful_lengths = [l for l, s in zip(results['path_length'][name], results['success'][name]) if s]
        summary['avg_path_length'][name] = np.mean(successful_lengths) if successful_lengths else 0
        summary['avg_nodes_explored'][name] = np.mean(results['nodes_explored'][name])
        summary['success_rate'][name] = np.mean(results['success'][name])
        summary['avg_time'][name] = np.mean(results['time'][name])
    
    # Calculate efficiency ratios
    baseline = next(iter(algorithms.keys()))  # First algorithm as baseline
    summary['efficiency_ratio'] = {}
    
    for name in algorithms.keys():
        if name != baseline:
            # Ratio of nodes explored (baseline / algorithm)
            if summary['avg_nodes_explored'][name] > 0:
                summary['efficiency_ratio'][name] = summary['avg_nodes_explored'][baseline] / summary['avg_nodes_explored'][name]
            else:
                summary['efficiency_ratio'][name] = 0
    
    return results, summary


def visualize_comparison(results, summary):
    """
    Visualize comparison results for different algorithms.
    
    Args:
        results: Results dictionary from compare_algorithms
        summary: Summary dictionary from compare_algorithms
    """
    # Get algorithm names
    algo_names = list(summary['avg_path_length'].keys())
    
    # Create a figure
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Average path length
    plt.subplot(3, 2, 1)
    bars = plt.bar(algo_names, [summary['avg_path_length'][name] for name in algo_names])
    plt.title('Average Path Length (Successful Paths)')
    plt.ylabel('Number of nodes')
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 2: Average nodes explored
    plt.subplot(3, 2, 2)
    bars = plt.bar(algo_names, [summary['avg_nodes_explored'][name] for name in algo_names])
    plt.title('Average Nodes Explored')
    plt.ylabel('Number of nodes')
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 3: Success rate
    plt.subplot(3, 2, 3)
    bars = plt.bar(algo_names, [summary['success_rate'][name] for name in algo_names])
    plt.title('Success Rate')
    plt.ylabel('Rate')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 4: Average time
    plt.subplot(3, 2, 4)
    bars = plt.bar(algo_names, [summary['avg_time'][name] for name in algo_names])
    plt.title('Average Execution Time')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Plot 5: Efficiency ratio
    if 'efficiency_ratio' in summary and summary['efficiency_ratio']:
        plt.subplot(3, 2, 5)
        ratio_names = list(summary['efficiency_ratio'].keys())
        bars = plt.bar(ratio_names, [summary['efficiency_ratio'][name] for name in ratio_names])
        plt.title('Efficiency Ratio (Baseline Nodes / Algorithm Nodes)')
        plt.ylabel('Ratio')
        plt.grid(axis='y', alpha=0.3)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 6: Detailed comparison
    plt.subplot(3, 2, 6)
    indices = range(len(results['source']))
    width = 0.8 / len(algo_names)
    
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        plt.bar(positions, results['nodes_explored'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Nodes Explored')
    plt.title('Nodes Explored per Test Case')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/algorithm_comparison.png')
    plt.close()
    
    # Create a second figure for detailed path lengths
    plt.figure(figsize=(15, 6))
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        plt.bar(positions, results['path_length'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Path Length')
    plt.title('Path Length per Test Case')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/path_length_comparison.png')
    
    print("Visualizations saved to 'algorithm_comparison.png' and 'path_length_comparison.png'")
    
    # Return the summary for convenience
    return summary


def analyze_by_path_difficulty(results, summary):
    """
    Analyze performance based on path difficulty (length of optimal path).
    
    Args:
        results: Results dictionary from compare_algorithms
        summary: Summary dictionary from compare_algorithms
    
    Returns:
        difficulty_stats: Dictionary with statistics by difficulty
    """
    # Get algorithm names
    algo_names = list(summary['avg_path_length'].keys())
    
    # Get baseline algorithm for reference path lengths
    baseline = next(iter(algo_names))
    
    # Create difficulty categories
    short_paths = []  # Path length <= 3
    medium_paths = []  # 3 < Path length <= 5
    long_paths = []  # Path length > 5
    
    for i, length in enumerate(results['path_length'][baseline]):
        if results['success'][baseline][i]:
            if length <= 3:
                short_paths.append(i)
            elif length <= 5:
                medium_paths.append(i)
            else:
                long_paths.append(i)
    
    # Calculate statistics by difficulty
    difficulty_stats = {
        'short': {name: {
            'success_rate': np.mean([results['success'][name][i] for i in short_paths]) if short_paths else 0,
            'avg_nodes': np.mean([results['nodes_explored'][name][i] for i in short_paths]) if short_paths else 0,
            'avg_time': np.mean([results['time'][name][i] for i in short_paths]) if short_paths else 0
        } for name in algo_names},
        
        'medium': {name: {
            'success_rate': np.mean([results['success'][name][i] for i in medium_paths]) if medium_paths else 0,
            'avg_nodes': np.mean([results['nodes_explored'][name][i] for i in medium_paths]) if medium_paths else 0,
            'avg_time': np.mean([results['time'][name][i] for i in medium_paths]) if medium_paths else 0
        } for name in algo_names},
        
        'long': {name: {
            'success_rate': np.mean([results['success'][name][i] for i in long_paths]) if long_paths else 0,
            'avg_nodes': np.mean([results['nodes_explored'][name][i] for i in long_paths]) if long_paths else 0,
            'avg_time': np.mean([results['time'][name][i] for i in long_paths]) if long_paths else 0
        } for name in algo_names}
    }
    
    # Print statistics
    print("\nPerformance by Path Difficulty:")
    print("-" * 80)
    
    for difficulty, stats in difficulty_stats.items():
        print(f"\n{difficulty.upper()} PATHS ({len(eval(difficulty + '_paths'))} paths):")
        print("-" * 40)
        
        for name in algo_names:
            print(f"{name}:")
            print(f"  Success Rate: {stats[name]['success_rate']:.2f}")
            print(f"  Avg Nodes Explored: {stats[name]['avg_nodes']:.2f}")
            print(f"  Avg Time: {stats[name]['avg_time']:.4f}s")
        
    return difficulty_stats