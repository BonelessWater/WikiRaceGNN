# Enhanced Wikipedia Graph Traversal

This project implements advanced graph traversal algorithms designed for navigating large, complex graphs like Wikipedia. It leverages Graph Neural Networks (GraphSAGE) alongside sophisticated search strategies (Beam Search, Bidirectional Guided Search) to significantly reduce node exploration compared to traditional methods like Breadth-First Search (BFS), while still finding optimal or near-optimal paths. The project includes data generation, model training, evaluation scripts, and an interactive web dashboard for visualization and comparison.

![DSA_diagram](https://github.com/user-attachments/assets/b2e3a12b-184f-4efb-8ac5-6dfc122b6baa)

## Disclaimer to Grader

This code base has MANY moving parts. If issues do arise, feel free to contact the owner of the repository.
For quick testing, refer to the Pipeline command which will run all necessary components in one go and set
--max_nodes to 1000 or less. If you choose to increase this number, be sure to have stable internet connection
and a fast computer

## Key Features

-   **Efficient Traversal**: GNN-guided search typically explores 30-70% fewer nodes than BFS.
-   **Multiple Strategies**: Includes Parallel Beam Search, Bidirectional Guided Search, and Hybrid approaches.
-   **Adaptive Selection**: Automatically chooses the most suitable traversal method based on estimated node distance.
-   **Interactive Dashboard**: A Flask-based web interface (`app.py`) for visualizing traversals, comparing algorithms (GNN vs. BFS), animating search steps, and exploring graph properties.
-   **Performance Comparison**: The dashboard provides detailed statistics (path length, nodes explored, time) for easy comparison.
-   **Data Handling**: Includes scripts for generating graph data from edge lists and training the GNN model.
-   **(Experimental) Parallelizable**: Core logic designed with parallelization in mind (further optimization may be needed).
-   **(Experimental) Memory Efficient**: Utilizes selective caching and pruning (further optimization may be needed).

## Prerequisites

-   Python 3.11 (as specified in `pyproject.toml`)
-   [Poetry](https://python-poetry.org/docs/#installation) for dependency management
-   A compatible CUDA setup (defaults to CUDA 11.7 in `pyproject.toml`, adjust if needed) or CPU fallback.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/BonelessWater/WikiRaceGNN
cd WikiRaceGNN
```

### Step 2: Set up the Poetry environment

```bash
# Install Poetry if you haven't already
# On Linux/macOS: curl -sSL https://install.python-poetry.org | python3 -
# On Windows (PowerShell): (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python3 -

# Install project dependencies using Poetry
poetry install
```

**Note on CUDA**: The project is configured for CUDA 11.7 in `pyproject.toml`. If you have a different CUDA version installed, you might need to modify the `[tool.poetry.dependencies]` and `[tool.poetry.group.dev.dependencies]` sections, specifically the `torch`, `torchvision`, `torchaudio`, and potentially `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, `pyg-lib` lines, to match your CUDA version (e.g., `+cu118`, `+cu121`, or `+cpu`). Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for compatible versions.

## Usage

The system provides several entry points via `main.py` for backend tasks and `app.py` for the interactive dashboard.

### Command Line (`main.py`)

These commands are typically used for data preparation, model training, and batch evaluation.

#### Full Pipeline (Data -> Train -> Evaluate -> Sample)

Run the complete backend pipeline. Useful for initial setup testing.

```bash
poetry run python main.py --mode pipeline --max_nodes 1000 --epochs 10
```

#### Data Generation Only

Generate the graph data (`adj_list`, embeddings, mappings) from an edge list. Ensure the output files land in the `data/` directory or update the path in `app.py`.

```bash
poetry run python main.py --mode data --max_nodes 1000 --edge_file path/to/your/edges.csv
```

#### Training Only

Train the GraphSAGE model using the generated graph data. Ensure the trained model is saved to the path expected by `app.py` (default `models/enhanced_model_final.pt`).

```bash
# Ensure data files exist in data/ before running
poetry run python main.py --mode train --epochs 50
```

#### Evaluation Only

Evaluate different traversal algorithms on generated test pairs. Produces statistics and potentially plots.

```bash
poetry run python main.py --mode evaluate --num_pairs 100
```

### Interactive Dashboard (`app.py`)

This is the primary way to interactively explore and visualize graph traversals.

**1. Prerequisites:**  
-   Ensure you have generated the necessary graph data files (e.g., using `main.py --mode data`) and they are located where `app.py` expects them (typically the `data/` directory). Check the `config` dictionary within `app.py` for the `edge_file` path used during initialization (even though it loads processed data, the config points to the original source).
-   Ensure you have a trained model file (e.g., from `main.py --mode train` or a pre-trained one) located where `app.py` expects it (default: `models/enhanced_model_final.pt`).

**2. Launch the Dashboard:**
-   From the root directory of the project (`WikiRaceGNN/`), run the following command in your terminal:
        ```bash
        poetry run python app.py
        ```
-   The application will start, initialize the model and data (you'll see logs in the terminal), and then typically listen on `http://0.0.0.0:5000/` or `http://127.0.0.1:5000/`.

**3. Access in Browser:**
-   Open your web browser (Chrome, Firefox, Edge, etc.).
-   Navigate to the address shown in the terminal, usually `http://localhost:5000` or `http://127.0.0.1:5000`.

**4. Using the Dashboard:**
-   **Select Nodes**: Use the "Source Node" and "Target Node" dropdowns to choose your starting and ending Wikipedia pages. You can type in the dropdowns to search if your browser supports it, or use the "Get Random Pair" button.
-   **Choose Method**: Select a GNN traversal strategy from the "Traversal Method" dropdown (`auto` is recommended).
-   **Set Parameters**: Adjust "Max Steps", "Beam Width", and "Heuristic Weight" as needed for the selected algorithm.
-   **Run Traversal**: Click the "Find Path" button. The application will run the selected GNN method and a baseline BFS search.
-   **View Results**:
    -   **Statistics**: The panel below the controls will populate with metrics comparing the GNN method vs. BFS (path lengths, nodes explored, time, efficiency).
    -   **Path Lists**: The GNN and BFS paths found will be displayed as ordered lists of nodes. Click the <i class="bi bi-box-arrow-up-right"></i> icon to visit the Wikipedia page (if URL exists) or the <i class="bi bi-info-circle"></i> icon to view node details.
    -   **Graph Visualization**: The main panel shows the graph structure around the found paths. Nodes and links are colored according to the legend. You can zoom (scroll wheel) and pan (click and drag). Hover over nodes for titles, click nodes to view details.
    -   **Animation**:
        -   Ensure the "Show Animation" switch is enabled *before* clicking "Find Path" if you want to see the step-by-step process.
        -   Use the Play/Pause, Next Step, and Previous Step buttons below the visualization to control the animation playback. The progress bar shows the current step.
    -   **Configuration**: Click the "Configuration" link in the top navbar to open a modal where you can change settings like the maximum number of nodes loaded, file paths, or visualization options. **Note:** Saving changes here will typically require restarting the `app.py` process for them to take full effect (especially data/model path changes).

## Command Line Arguments (`main.py`)

These arguments primarily apply when running `main.py`. Configuration for `app.py` is largely handled internally or via its web configuration modal.

-   `--edge_file`: Path to edge list CSV file (default: 'data/wiki_edges.csv'). Used in `data` and `pipeline` modes.
-   `--max_nodes`: Maximum number of nodes to include in the graph (default: 1000). Used in `data` and `pipeline` modes.
-   `--feature_dim`: Dimension of node features (default: 64). Used in `data` and `train` modes.
-   `--mode`: Operation mode for `main.py` (`data`, `train`, `evaluate`, `traverse`, or `pipeline`).
-   `--model_path`: Path to save/load model (default: 'models/enhanced_model_final.pt'). Used in `train`, `evaluate`, `traverse`.
-   `--method`: Traversal method for command-line traversal (`parallel_beam`, `bidirectional_guided`, `hybrid`, or `auto`). Used in `traverse` mode.
-   `--max_steps`: Maximum steps for traversal (default: 30). Used in `traverse`, `evaluate`.
-   `--num_neighbors`: (If applicable to strategy) Number of neighbors to sample (default: 20).
-   `--beam_width`: Beam width for beam/hybrid search (default: 5). Used in relevant strategies.
-   `--heuristic_weight`: Weight for heuristic component in guided search (default: 1.5). Used in relevant strategies.
-   `--visualize`: (For `main.py --mode evaluate`) Enable generation of static plots (default: False).
-   `--seed`: Random seed for reproducibility (default: 42).
-   `--gpu`: Force GPU usage if available (otherwise auto-detects).
-   `--epochs`: Number of training epochs (default: 10). Used in `train`, `pipeline`.

## Traversal Methods (Available in Dashboard & CLI)

1.  **`parallel_beam`**: (GNN Required) Parallel Beam Search. Efficient for shorter paths, explores limited width.
2.  **`bidirectional_guided`**: (GNN Required) A\*-like bidirectional search using GNN embeddings as a heuristic to guide the search towards the target. Generally robust.
3.  **`hybrid`**: (GNN Required) Attempts beam search first (up to half max\_steps); if unsuccessful, falls back to bidirectional guided search. Good balance.
4.  **`auto`**: (GNN Required) Automatically selects between `parallel_beam`, `hybrid`, or `bidirectional_guided` based on an initial similarity estimate between source and target embeddings. **(Recommended)**
5.  **`bfs`**: (Baseline - May be available in `main.py --mode evaluate`) Standard Bidirectional Breadth-First Search. Used as a baseline for comparison.

## Project Structure

```
WikiRaceGNN/
|
├── app.py                 # Flask Web Application (Interactive Dashboard)
├── main.py                # Main entry point for CLI operations (data, train, eval)
├── train.py               # Training script (likely called by main.py)
├── evaluate.py            # Evaluation script (likely called by main.py)
|
├── data/                  # Default directory for graph data
│   ├── wiki_edges.csv     # Example edge list input
│   └── ...                # Other generated files (adj_list.json, embeddings.pt, etc.)
|
├── models/                # Directory for GNN models
│   ├── __init__.py
│   ├── graphsage.py       # GraphSAGE implementation
│   ├── attention.py       # (If used) Attention layer implementations
│   └── enhanced_model_final.pt # Example saved model
|
├── static/                # Static files for the web interface
│   ├── css/
│   │   └── style.css      # Custom CSS for the dashboard
│   ├──html/               # HTML templates for the web interface
│   │   ├── base.html      # Base HTML template with Bootstrap, navbar, footer
│   │   ├── index.html     # Main dashboard page template
│   │   └── error.html     # Error page template
│   └── js/
│       └── main.js        # Core JavaScript logic for the dashboard (D3, interactions)
|
├── traversal/             # Traversal algorithm implementations
│   ├── __init__.py
│   ├── unified.py         # Contains GraphTraverser and strategy classes
│   └── utils.py           # Utility functions for traversal (e.g., BFS)
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── data.py            # Data loading and preprocessing
│   ├── visualization.py   # Utilities for generating static plots (used by evaluate.py)
│   └── evaluation.py      # Helper functions for evaluation
│
├── plots/                 # Default output directory for static plots from evaluation
|
├── pyproject.toml         # Poetry configuration and dependencies
├── poetry.lock            # Poetry lock file
└── README.md              # This file
```

## Web Interface Features (`app.py`)

The interactive dashboard provides:

-   **Node Selection**: Dropdowns to select source and target nodes by title (with search).
-   **Random Pair**: Button to fetch a random source/target pair (attempts to find pairs with reasonable path lengths).
-   **Algorithm Selection**: Choose between different GNN-guided traversal strategies (`auto`, `parallel_beam`, `bidirectional_guided`, `hybrid`).
-   **Parameter Control**: Adjust `max_steps`, `beam_width`, `heuristic_weight`.
-   **Traversal Execution**: Run the selected GNN algorithm and a baseline BFS simultaneously.
-   **Statistics Display**: Shows key metrics side-by-side: path length, nodes explored, time taken, and efficiency ratio (BFS nodes / GNN nodes).
-   **Path Display**: Lists the sequence of nodes for both the GNN and BFS paths found. Includes links to Wikipedia pages (if available).
-   **Node Info Modal**: Click an info icon <i class="bi bi-info-circle"></i> next to a node in the path list to view its details (title, URL, degree, neighbors, centrality if calculated).
-   **Interactive Graph Visualization**: Uses D3.js to render the graph structure around the found paths.
    -   Highlights source, target, GNN path nodes, BFS path nodes, and overlapping nodes/links with distinct colors.
    -   Supports zoom and pan.
    -   Displays node titles on hover.
    -   Click nodes in the visualization to open the Node Info modal.
-   **Step-by-Step Animation**: (Toggleable) Animates the exploration process showing:
    -   Forward/backward frontiers expanding for both BFS and GNN searches (where applicable).
    -   Edges traversed during BFS exploration.
    -   Controls for Play/Pause, Next Step, Previous Step.
    -   Progress bar indicating animation progress.
-   **Configuration Modal**: Allows updating application settings like max nodes for loading, layout iterations, data file paths, and model path (requires app restart after saving).
-   **Graph Properties Display**: Shows overall graph statistics (node/edge count, density, components, etc.).

## Data Generation (`main.py --mode data`)

The generation process loads data from an edge list (CSV format expected: `source_id,target_id`). It builds an adjacency list, creates node features (e.g., using Word2Vec on titles if `use_word2vec` is True and titles are available, otherwise potentially random or Node2Vec), and saves the processed graph data (`Data` object, mappings, etc.) typically to the `data/` directory for use by the model and traverser.

## Visualization

The project offers two main visualization modes:

1.  **Static Plots (`main.py --mode evaluate --visualize`)**: When running evaluation via `main.py`, enabling the `--visualize` flag generates static plots saved to the `plots/` directory. These typically include performance comparisons and analysis charts.
2.  **Interactive Web Dashboard (`app.py`)**: Running `app.py` provides the real-time visualization interface described in the "Interactive Dashboard" and "Using the Dashboard" sections, allowing dynamic exploration of specific traversals.

## Troubleshooting

### Dependency/Installation Issues

If you encounter dependency issues with Poetry:
1.  Ensure Python 3.11 is active.
2.  Try updating Poetry: `poetry self update`.
3.  Check CUDA compatibility if using GPU (see Installation notes).
4.  Clear Poetry's cache: `poetry cache clear --all pypi`.
5.  Remove the virtual environment (`rm -rf $(poetry env info --path)`) and reinstall (`poetry install`).

### Web Interface (`app.py`) Issues

-   **500 Internal Server Error**: Check the Flask console output where you ran `poetry run python app.py` for Python tracebacks. Common causes:
    -   Model file (`.pt`) not found or architecture mismatch (see console warnings during startup).
    -   Data files (`adj_list.json`, mappings, etc.) missing or corrupted in the `data/` directory. Re-run `main.py --mode data`.
    -   Errors within a traversal strategy in `traversal/unified.py`.
    -   Errors accessing data attributes (e.g., `node_titles`) if `load_graph_data` didn't load them correctly.
-   **Dropdowns Empty / "Node Not Found" errors**: Ensure `init_model` in `app.py` successfully loads `graph_data` and its associated mappings (`node_mapping`, `reverse_mapping`). Check console logs.
-   **Visualization Not Appearing / Errors**:
    -   Check the *browser's* developer console (F12 -> Console) for JavaScript errors (e.g., "Cannot read properties of null", D3 errors).
    -   Verify that the `/traverse` endpoint in `app.py` is correctly returning the `layout` and `all_edges` data in the JSON response. Use the browser's Network tab to inspect the response.
    -   Ensure your browser supports modern JavaScript and SVG.
-   **Animation Not Working**:
    -   Make sure the "Show Animation" toggle is checked *before* running the traversal.
    -   Verify that the backend (`app.py` / `traversal/unified.py`) is correctly generating and returning the `bfs_exploration_history` and `gnn_exploration_history` data when requested. Check the `/traverse` response in the Network tab.
