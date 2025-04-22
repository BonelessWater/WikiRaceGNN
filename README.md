# Enhanced Wikipedia Graph Traversal

This project implements an advanced graph traversal algorithm that outperforms traditional bidirectional breadth-first search (BFS) for navigating through Wikipedia-like graphs. The implementation leverages Graph Neural Networks (GraphSAGE) with sophisticated search strategies to reduce the number of nodes explored while finding optimal or near-optimal paths.

## Key Features

- **Reduced Node Exploration**: Typically explores 30-70% fewer nodes than BFS
- **Parallelizable**: Optimized for modern hardware with multi-core processing
- **Memory Efficient**: Selective caching and pruning for optimal memory usage
- **Adaptable**: Automatically selects the best traversal method based on graph properties
- **Interactive Visualization**: Web interface for exploring traversal algorithms and comparing performance

## Prerequisites

- Python 3.11 (required by the Poetry configuration)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/BonelessWater/WikiRaceGNN
cd WikiRaceGNN
```

### Step 2: Set up the Poetry environment

```bash
# Install Poetry if you haven't already
# curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies using Poetry
poetry install
```

**Note**: The project uses CUDA 11.7. If you have a different CUDA version, you might need to modify the `pyproject.toml` file accordingly.

## Usage

The system provides several modes of operation:

### Full Pipeline

Run the complete pipeline (data generation, training, evaluation, and sample traversal). Run this command for immediate testing of the codebase:

```bash
poetry run python main.py --mode pipeline --max_nodes 1000 --epochs 10
```

### Data Generation

Generate a Wikipedia-like graph dataset:

```bash
poetry run python main.py --mode data --max_nodes 1000
```

### Training

Train the GraphSAGE model:

```bash
poetry run python main.py --mode train
```

### Evaluation

Evaluate different traversal algorithms:

```bash
poetry run python main.py --mode evaluate
```

### Web Interface

Run the interactive web interface for visualization:

```bash
poetry run python app.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

## Command Line Arguments

- `--edge_file`: Path to edge list CSV file (default: 'data/wiki_edges.csv')
- `--max_nodes`: Maximum number of nodes to include in the graph (default: 1000)
- `--feature_dim`: Dimension of node features (default: 64)
- `--mode`: Operation mode ('data', 'train', 'evaluate', 'traverse', or 'pipeline')
- `--model_path`: Path to saved model (default: 'models/enhanced_model_final.pt')
- `--method`: Traversal method ('parallel_beam', 'bidirectional_guided', 'hybrid', or 'auto')
- `--max_steps`: Maximum steps for traversal (default: 30)
- `--num_neighbors`: Number of neighbors to sample in each step (default: 20)
- `--beam_width`: Beam width for beam search (default: 5)
- `--heuristic_weight`: Weight for heuristic component in A* search (default: 1.5)
- `--visualize`: Enable visualization of traversal results
- `--seed`: Random seed for reproducibility (default: 42)
- `--gpu`: Use GPU if available
- `--epochs`: Number of training epochs (default: 10)

## Traversal Methods

The system supports several traversal methods:

1. **Parallel Beam Search (`parallel_beam`)**: Maintains multiple candidate paths and expands them in parallel
2. **Bidirectional Guided Search (`bidirectional_guided`)**: A* search from both ends with GNN guidance
3. **Hybrid (`hybrid`)**: Attempts parallel beam search first, then falls back to bidirectional search if needed
4. **Auto (`auto`)**: Automatically selects the best method based on estimated distance between nodes

## Project Structure

```
wiki_traversal/
|
├── app.py                     # Web interface application
|
├── data/                      # Edge list, adjacency list and embeddings
│
├── models/                    # Neural network models
│   ├── __init__.py
│   ├── graphsage.py           # GraphSAGE implementation
│   └── attention.py           # Attention mechanisms
│
├── templates/                 # HTML templates for web interface
│   ├── base.html              # Base template
│   ├── index.html             # Main page
│   ├── error.html             # Error page
│   └── api_docs.html          # API documentation
│
├── static/                    # Static files for web interface
│   ├── css/                   # CSS stylesheets
│   └── js/                    # JavaScript files
│
├── traversal/                 # Traversal algorithms
│   ├── __init__.py
│   ├── base.py                # Base traversal class
│   ├── enhanced.py            # Enhanced traversal algorithms
│   └── utils.py               # Traversal utilities
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data.py                # Data loading and preprocessing
│   ├── visualization.py       # Visualization utilities
│   ├── crawler.py             # Wikipedia data crawler
│   ├── wikibuilder.py         # Wikipedia graph builder
│   └── evaluation.py          # Evaluation metrics
│
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── main.py                    # Main entry point
├── pyproject.toml             # Poetry configuration
└── README.md                  # This file
```

## Web Interface Features

The web interface provides:

- Interactive graph visualization with D3.js
- Step-by-step animation of graph traversal algorithms
- Comparison between baseline BFS and GNN-guided algorithms
- Detailed statistics and performance metrics
- Node information and graph property exploration

## Data Generation

The generation process uses Word2Vec for creating node embeddings based on page titles.

## Visualization

The project offers two visualization options:

1. **Static plots**: When using the `--visualize` flag with `main.py`, the system generates plots in the `plots/` directory:
   - Path comparisons between different algorithms
   - Node exploration patterns
   - Efficiency metrics
   - Graph structure visualization

2. **Interactive web interface**: Running `app.py` provides a real-time visualization interface with:
   - Interactive graph manipulation (zoom, pan)
   - Animation of traversal steps
   - Comparison of multiple algorithm results
   - Detailed node information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Dependency Issues

If you encounter dependency issues with Poetry:

1. Try updating Poetry: `poetry self update`
2. Clear Poetry's cache: `poetry cache clear --all pypi`
3. Remove the virtual environment and recreate it: 
   ```bash
   rm -rf $(poetry env info --path)
   poetry install
   ```

### Web Interface Issues

If you encounter issues with the web interface:

1. Make sure all required directories exist (`data/`, `models/`, `static/`, etc.)
2. Ensure the model has been trained and saved at the expected path
3. Check that the graph data exists and is accessible
4. For visualization issues, ensure your browser supports D3.js