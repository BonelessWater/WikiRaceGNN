document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Loaded. Initializing JS...");

    // --- DOM Element References ---
    // Control Panel
    const sourceNodeSelect = document.getElementById('sourceNode');
    const targetNodeSelect = document.getElementById('targetNode');
    const traversalMethodSelect = document.getElementById('traversalMethod');
    const maxStepsInput = document.getElementById('maxSteps');
    const beamWidthInput = document.getElementById('beamWidth');
    const heuristicWeightInput = document.getElementById('heuristicWeight');
    const traverseBtn = document.getElementById('traverseBtn');
    const randomPairBtn = document.getElementById('randomPairBtn');
    const animationToggle = document.getElementById('animationToggle');

    // Stats Panel
    const statsContainer = document.getElementById('statsContainer');
    const gnnNodesExploredEl = document.getElementById('gnnNodesExplored');
    const bfsNodesExploredEl = document.getElementById('bfsNodesExplored');
    const gnnPathLengthEl = document.getElementById('gnnPathLength');
    const bfsPathLengthEl = document.getElementById('bfsPathLength');
    const gnnTimeEl = document.getElementById('gnnTime');
    const bfsTimeEl = document.getElementById('bfsTime');
    const efficiencyRatioEl = document.getElementById('efficiencyRatio');

    // Path Panels
    const gnnPathTypeEl = document.getElementById('gnnPathType');
    const gnnPathContainer = document.getElementById('gnnPath');
    const bfsPathContainer = document.getElementById('bfsPath');

    // Visualization & Animation Elements
    const graphVisContainer = document.getElementById('graphVisualization');
    const visualizationTitleEl = document.getElementById('visualizationTitle');
    const animationProgress = document.getElementById('animationProgress');
    const animationControls = document.getElementById('animationControls');
    const animationPrevBtn = document.getElementById('animationPrevBtn');
    const animationNextBtn = document.getElementById('animationNextBtn');
    const animationPlayBtn = document.getElementById('animationPlayBtn');
    const currentStepEl = document.getElementById('currentStep');
    const totalStepsEl = document.getElementById('totalSteps');

    // Modals
    const errorModalEl = document.getElementById('errorModal');
    const errorModal = errorModalEl ? new bootstrap.Modal(errorModalEl) : null;
    const errorMessageEl = document.getElementById('errorMessage');
    const nodeInfoModalEl = document.getElementById('nodeInfoModal');
    const nodeInfoModal = nodeInfoModalEl ? new bootstrap.Modal(nodeInfoModalEl) : null;
    const nodeInfoLoadingEl = document.getElementById('nodeInfoLoading');
    const nodeInfoContentEl = document.getElementById('nodeInfoContent');
    const nodeTitleEl = document.getElementById('nodeTitle');
    const nodeUrlEl = document.getElementById('nodeUrl');
    const nodePropertiesEl = document.getElementById('nodeProperties');
    const centralityMetricsEl = document.getElementById('centralityMetrics');
    const neighborCountEl = document.getElementById('neighborCount');
    const neighborsListEl = document.getElementById('neighborsList');

    // --- State Variables ---
    let graphData = null;       // Stores the full response from /traverse
    let animationData = [];     // Array of {step, title, bfs_frontiers, gnn_frontiers, traversed_edges, bfs_path, gnn_path, is_final}
    let baseLinks = [];         // Stores all graph links {source: id, target: id} for D3 rendering
    let nodeMap = new Map();    // Stores node objects {id, title, x, y, type} keyed by id for quick lookup
    let currentStepIndex = 0;
    let animationTimer = null;
    let isAnimationPlaying = false;
    let simulation = null;      // D3 force simulation (optional, if layout not precomputed)
    let svg = null;             // D3 SVG element
    let g = null;               // Main group within SVG for zoom/pan
    let linkSelection = null;   // D3 link selection
    let nodeSelection = null;   // D3 node selection
    let labelSelection = null;  // D3 label selection
    let tooltip = null;         // D3 tooltip div
    let zoomBehavior = null;    // D3 zoom behavior


    // --- Configuration ---
    const nodeColors = {
        'regular': '#808080', 'source': '#00FF00', 'target': '#FF0000',
        'gnn_path': '#2E8B57', 'bfs_path': '#dc3545', 'both_path': '#6f42c1',
        'gnn_frontier': '#1E90FF', 'bfs_frontier': '#FFA500', 'both_frontier': '#800080'
    };
    const nodeSizes = {
        'regular': 6, 'source': 10, 'target': 10,
        'gnn_path': 8, 'bfs_path': 8, 'both_path': 9,
        'gnn_frontier': 7, 'bfs_frontier': 7, 'both_frontier': 8
    };
    const linkColors = {
        'regular': '#D3D3D3', 'traversed': '#ADD8E6',
        'gnn_path': '#2E8B57', 'bfs_path': '#dc3545', 'both_path': '#6f42c1'
    };
    const linkWidths = {
         'regular': 1, 'traversed': 1.5,
         'gnn_path': 2.5, 'bfs_path': 2.5, 'both_path': 3
    };
    const ANIMATION_SPEED_MS = 750; // Milliseconds per step

    // --- Initialization ---
    setupTooltip();
    updateAnimationControlsVisibility();

    // --- Event Listeners ---
    if (traverseBtn) traverseBtn.addEventListener('click', handleFindPath);
    if (randomPairBtn) randomPairBtn.addEventListener('click', handleGetRandomPair);
    if (animationToggle) animationToggle.addEventListener('change', handleAnimationToggle);
    if (animationPrevBtn) animationPrevBtn.addEventListener('click', previousStep);
    if (animationNextBtn) animationNextBtn.addEventListener('click', nextStep);
    if (animationPlayBtn) animationPlayBtn.addEventListener('click', toggleAnimation);

    // --- Core API Call Functions ---
    // (Keep handleFindPath, handleGetRandomPair, handleShowNodeInfo from previous step)
    function handleFindPath() {
        const sourceId = sourceNodeSelect.value;
        const targetId = targetNodeSelect.value;
        const method = traversalMethodSelect.value;
        const maxSteps = maxStepsInput.value;
        const beamWidth = beamWidthInput.value;
        const heuristicWeight = heuristicWeightInput.value;
        const visualizeExploration = animationToggle.checked;

        if (!sourceId || !targetId) { showError('Please select both source and target nodes.'); return; }
        if (sourceId === targetId) { showError('Source and target nodes cannot be the same.'); return; }

        setLoadingState(true);
        resetResultsUI();      // Clears stats, paths, viz, animation state
        // clearVisualization(); // Called within resetResultsUI now
        // resetAnimation();     // Called within resetResultsUI now


        const requestBody = {
            source_id: sourceId, target_id: targetId, method: method,
            max_steps: parseInt(maxSteps) || 30, beam_width: parseInt(beamWidth) || 5,
            heuristic_weight: parseFloat(heuristicWeight) || 1.5, visualize_exploration: visualizeExploration
        };
        console.log("Sending /traverse request:", requestBody);

        fetch('/traverse', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestBody)
        })
        .then(handleFetchResponse)
        .then(data => {
            console.log("Received /traverse data:", data);
            graphData = data; // Store response
            displayResults(data);
            statsContainer.style.display = 'block';
            // *** Initialize D3/Animation ***
            initializeVisualization(data);
        })
        .catch(error => {
            console.error("Traversal Fetch Error:", error);
            showError(`Traversal failed: ${error.message}`);
            resetResultsUI();
        })
        .finally(() => { setLoadingState(false); });
    }

    function handleGetRandomPair() {
        randomPairBtn.disabled = true;
        randomPairBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
        fetch('/random_nodes')
            .then(handleFetchResponse)
            .then(data => {
                console.log("Received /random_nodes data:", data);
                if (sourceNodeSelect) sourceNodeSelect.value = data.source_id;
                if (targetNodeSelect) targetNodeSelect.value = data.target_id;
            })
            .catch(error => {
                console.error('Random Pair Fetch Error:', error);
                showError(`Failed to get random pair: ${error.message}`);
            })
            .finally(() => {
                randomPairBtn.disabled = false;
                randomPairBtn.innerHTML = 'Get Random Pair';
            });
    }

    function handleShowNodeInfo(nodeId) {
        console.log(`Fetching info for node: ${nodeId}`);
        const modalInstance = nodeInfoModalEl ? bootstrap.Modal.getOrCreateInstance(nodeInfoModalEl) : null;
        if (!modalInstance || !nodeId) { console.error("Node info modal or Node ID missing."); return; }
        nodeInfoLoadingEl.style.display = 'block';
        nodeInfoContentEl.style.display = 'none';
        modalInstance.show();
        fetch(`/node_info?id=${encodeURIComponent(nodeId)}`)
            .then(handleFetchResponse)
            .then(data => {
                console.log("Received /node_info data:", data);
                setText(nodeTitleEl, data.title || `Node ${data.id}`);
                setHtml(nodeUrlEl, data.url ? `<a href="${data.url}" target="_blank" rel="noopener noreferrer">${data.url} <i class="bi bi-box-arrow-up-right"></i></a>` : '<em>No URL available</em>');
                safeSetHtml(nodePropertiesEl, `<tr><th scope="row" style="width:80px;">ID</th><td>${data.id}</td></tr>` + `<tr><th scope="row">Degree</th><td>${data.neighbor_count ?? 'N/A'}</td></tr>`);
                let centralityHtml = '';
                if (data.centrality && Object.keys(data.centrality).length > 0) {
                    Object.entries(data.centrality).forEach(([key, value]) => {
                        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                        centralityHtml += `<tr><th scope="row" style="width:150px;">${formattedKey}</th><td>${value?.toFixed(4) ?? 'N/A'}</td></tr>`;
                    });
                } else { centralityHtml = '<tr><td colspan="2" class="text-muted text-center">N/A</td></tr>'; }
                safeSetHtml(centralityMetricsEl, centralityHtml);
                setText(neighborCountEl, data.neighbors?.length ?? 0);
                let neighborsHtml = '';
                if (data.neighbors && data.neighbors.length > 0) {
                    data.neighbors.forEach(neighbor => {
                        neighborsHtml += `<button type="button" class="list-group-item list-group-item-action py-1 px-2 d-flex justify-content-between align-items-center node-info-neighbor-btn" data-node-id="${neighbor.id}">` +
                                         `<span>${neighbor.title || `Node ${neighbor.id}`}</span><i class="bi bi-search text-muted"></i></button>`;
                    });
                } else { neighborsHtml = '<div class="list-group-item text-muted">No neighbors found.</div>'; }
                safeSetHtml(neighborsListEl, neighborsHtml);
                 if (neighborsListEl) {
                     neighborsListEl.querySelectorAll('.node-info-neighbor-btn').forEach(btn => {
                         btn.addEventListener('click', () => handleShowNodeInfo(btn.dataset.nodeId));
                     });
                 }
                nodeInfoLoadingEl.style.display = 'none';
                nodeInfoContentEl.style.display = 'block';
            })
            .catch(error => {
                console.error('Node Info Fetch Error:', error);
                nodeInfoLoadingEl.style.display = 'none';
                safeSetHtml(nodeInfoContentEl, `<div class="alert alert-danger">Failed to load node info: ${error.message}</div>`);
                nodeInfoContentEl.style.display = 'block';
            });
    }

    // --- UI Update Functions ---
    // (Keep displayResults, displayPathList from previous step)
    function displayResults(data) {
        setText(gnnNodesExploredEl, data.nodes_explored);
        setText(bfsNodesExploredEl, data.bfs_nodes_explored);
        setText(gnnPathLengthEl, data.path_length);
        setText(bfsPathLengthEl, data.bfs_path_length);
        setText(gnnTimeEl, data.time_taken !== undefined ? `${data.time_taken.toFixed(3)}s` : '-');
        setText(bfsTimeEl, data.bfs_time_taken !== undefined ? `${data.bfs_time_taken.toFixed(3)}s` : '-');
        setText(efficiencyRatioEl, data.efficiency_ratio !== undefined ? `${data.efficiency_ratio.toFixed(2)}x` : '-');
        setText(gnnPathTypeEl, data.method_used || 'GNN');
        displayPathList(gnnPathContainer, data.path);
        displayPathList(bfsPathContainer, data.bfs_path);
    }

    function displayPathList(container, pathData) {
        if (!container) { console.error("Path list container not found!"); return; }
        container.innerHTML = ''; // Clear previous content
        if (!pathData || pathData.length === 0) { container.innerHTML = '<div class="list-group-item text-muted small">No path found</div>'; return; }
        pathData.forEach((node, index) => {
            const item = document.createElement('div');
            item.className = 'list-group-item d-flex justify-content-between align-items-center py-1 px-2';
            if (index === 0) item.classList.add('list-group-item-success');
            if (index === pathData.length - 1) item.classList.add('list-group-item-primary');
            let title = node.title || `Node ${node.id}`;
            if (title.length > 35) title = title.substring(0, 32) + '...';
            let controlsHtml = '';
            if (node.url) { controlsHtml += ` <a href="${node.url}" target="_blank" class="ms-1 node-link-btn link-secondary" title="Open Wikipedia Page"><i class="bi bi-box-arrow-up-right"></i></a>`; }
            controlsHtml += ` <button type="button" class="btn btn-sm btn-outline-secondary node-info-list-btn" data-node-id="${node.id}" title="Show Node Info"><i class="bi bi-info-circle"></i></button>`;
            item.innerHTML = `<span class="me-2" style="min-width: 25px;"><strong>${index + 1}.</strong></span> <span>${title}</span><span class="ms-auto">${controlsHtml}</span>`;
            container.appendChild(item);
        });
        container.querySelectorAll('.node-info-list-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                 const targetButton = e.target.closest('.node-info-list-btn');
                 if (targetButton) { handleShowNodeInfo(targetButton.dataset.nodeId); }
            });
        });
    }

    function resetResultsUI() {
        // Clear stats
        setText(gnnNodesExploredEl, '-'); setText(bfsNodesExploredEl, '-'); setText(gnnPathLengthEl, '-');
        setText(bfsPathLengthEl, '-'); setText(gnnTimeEl, '-'); setText(bfsTimeEl, '-');
        setText(efficiencyRatioEl, '-'); setText(gnnPathTypeEl, 'Method');
        // Clear paths
        const placeholder = '<div class="list-group-item text-muted small">Run traversal to see path</div>';
        setHtml(gnnPathContainer, placeholder); setHtml(bfsPathContainer, placeholder);
        // Hide stats
        if(statsContainer) statsContainer.style.display = 'none';
        // Clear visualization and animation
        clearVisualization();
        resetAnimation();
    }

    // --- Utility Functions ---
    // (Keep setLoadingState, showError, handleFetchResponse, setText, setHtml, safeSetHtml)
    function setLoadingState(isLoading) {
        if (traverseBtn) {
            traverseBtn.disabled = isLoading;
            traverseBtn.innerHTML = isLoading ? '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Finding...' : 'Find Path';
        }
    }
    function showError(message) {
        console.error("Displaying Error:", message);
        const modalInstance = errorModalEl ? bootstrap.Modal.getOrCreateInstance(errorModalEl) : null;
        if (errorMessageEl && modalInstance) {
            errorMessageEl.textContent = message;
            modalInstance.show();
        } else { alert(`Error: ${message}`); }
    }
    async function handleFetchResponse(response) {
        if (!response.ok) {
            let errorJson = null; let errorMsg = `HTTP error ${response.status}`;
            try { errorJson = await response.json(); errorMsg = errorJson.error || errorJson.message || errorMsg; }
            catch (e) { errorMsg = response.statusText || errorMsg; }
            throw new Error(errorMsg);
        }
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) { return response.json(); }
        return {};
    }
    function setText(element, value, defaultValue = '-') { if (element) element.textContent = value ?? defaultValue; }
    function setHtml(element, html) { if (element) element.innerHTML = html; }
    function safeSetHtml(element, html) { setHtml(element, html); }


    // --- D3 Visualization and Animation Functions ---

    function initializeVisualization(data) {
        console.log("Initializing visualization...");
        graphData = data;

        if (!graphData || !graphData.layout || Object.keys(graphData.layout).length === 0) {
            showError("Layout data missing or empty. Cannot visualize graph.");
            clearVisualization(); return;
        }
        if (!graphData.all_edges) {
            console.warn("'all_edges' missing. Links might be incomplete."); baseLinks = [];
        } else {
             baseLinks = graphData.all_edges
                 .map(edge => ({ source: String(edge[0]), target: String(edge[1]) }))
                 .filter(link => graphData.layout[link.source] && graphData.layout[link.target]);
            console.log(`Prepared ${baseLinks.length} base links.`);
        }

        prepareAnimationData(data);
        updateAnimationControlsVisibility();

        if (animationData.length > 0) {
            if (animationToggle.checked) {
                console.log("Rendering initial animation step.");
                renderVisualizationStep(0);
            } else {
                console.log("Rendering final static state.");
                renderVisualizationFinal();
            }
        } else {
            clearVisualization();
            setHtml(graphVisContainer, '<div class="alert alert-info">No visualization data available for these paths.</div>');
        }
    }

    function prepareAnimationData(data) {
        animationData = []; // Reset
        const visualize = data.visualize_exploration ?? animationToggle.checked; // Check flag from backend or toggle
        console.log(`Preparing animation data (visualize=${visualize})...`);

        // Ensure paths are arrays even if null/undefined
        const gnnPathIds = data.path?.map(n => String(n.id)) || [];
        const bfsPathIds = data.bfs_path?.map(n => String(n.id)) || [];

        if (!visualize || (!data.bfs_exploration_history && !data.gnn_exploration_history)) {
            console.log("No exploration history found or visualization disabled. Creating final frame only.");
            animationData.push({
                step: 0, title: 'Final Paths',
                bfs_frontier_forward: [], bfs_frontier_backward: [],
                gnn_frontier_forward: [], gnn_frontier_backward: [],
                traversed_edges: [], // No intermediate edges shown in static view
                bfs_path: bfsPathIds,
                gnn_path: gnnPathIds,
                is_final: true
            });
            return;
        }

        const bfsHistory = data.bfs_exploration_history || [];
        const gnnHistory = data.gnn_exploration_history || [];
        const sourceId = gnnPathIds[0] || bfsPathIds[0]; // Get source from either path
        const targetId = gnnPathIds[gnnPathIds.length - 1] || bfsPathIds[bfsPathIds.length - 1];

        // Initial state (Step 0)
        animationData.push({
            step: 0, title: 'Initial State',
            bfs_frontier_forward: sourceId ? [sourceId] : [],
            bfs_frontier_backward: targetId ? [targetId] : [],
            gnn_frontier_forward: sourceId ? [sourceId] : [],
            gnn_frontier_backward: targetId ? [targetId] : [],
            traversed_edges: [], bfs_path: [], gnn_path: [], is_final: false
        });

        // Exploration steps
        const maxHistLength = Math.max(bfsHistory.length, gnnHistory.length);
        let cumulativeTraversedEdges = new Set(); // Keep track of unique traversed edge IDs [[u,v], ...]

        for (let i = 0; i < maxHistLength; i++) {
            const bfsStep = bfsHistory[i];
            const gnnStep = gnnHistory[i];

            // Aggregate traversed edges from BFS history (GNN history doesn't have them)
            const stepEdges = bfsStep?.traversed_edges || []; // Expecting [[u_id, v_id], ...]
            stepEdges.forEach(edge => {
                if (Array.isArray(edge) && edge.length === 2) {
                    const u = String(edge[0]);
                    const v = String(edge[1]);
                    // Create canonical key to avoid duplicates (e.g., "A-B")
                    const key = u < v ? `${u}-${v}` : `${v}-${u}`;
                    cumulativeTraversedEdges.add(key);
                }
            });

            animationData.push({
                step: i + 1, title: `Exploration Step ${i + 1}`,
                bfs_frontier_forward: bfsStep?.forward_frontier?.map(String) || [],
                bfs_frontier_backward: bfsStep?.backward_frontier?.map(String) || [],
                gnn_frontier_forward: gnnStep?.forward_frontier?.map(String) || [], // GNN frontiers might be empty
                gnn_frontier_backward: gnnStep?.backward_frontier?.map(String) || [],
                // Convert Set back to array of pairs for this frame
                traversed_edges: Array.from(cumulativeTraversedEdges).map(key => key.split('-')),
                bfs_path: [], gnn_path: [], is_final: false
            });
        }

        // Final state with paths
        animationData.push({
            step: maxHistLength + 1, title: 'Final Paths Found',
            bfs_frontier_forward: [], bfs_frontier_backward: [],
            gnn_frontier_forward: [], gnn_frontier_backward: [],
            traversed_edges: Array.from(cumulativeTraversedEdges).map(key => key.split('-')), // Show all traversed edges in final frame
            bfs_path: bfsPathIds,
            gnn_path: gnnPathIds,
            is_final: true
        });
        console.log(`Prepared ${animationData.length} animation frames.`);
    }


    function renderVisualizationStep(stepIndex) {
        if (!graphData || !graphData.layout || !animationData || stepIndex < 0 || stepIndex >= animationData.length) {
             console.error("Cannot render step:", stepIndex, !graphData, !graphData.layout, !animationData); return;
         }
         currentStepIndex = stepIndex;
         const stepData = animationData[stepIndex];
         console.log(`Rendering step ${stepIndex}: ${stepData.title}`);
         renderD3Graph(graphData.layout, baseLinks, stepData); // Pass layout, base links, and specific step data
         updateAnimationControls();
    }

    function renderVisualizationFinal() {
        if (!graphData || !graphData.layout) { console.warn("Cannot render final state: Missing graphData or layout."); return; }
         // Use the last frame's data which should contain final paths and all traversed edges
        const finalStepData = animationData[animationData.length - 1] || {
             bfs_path: graphData.bfs_path?.map(n=>String(n.id))||[],
             gnn_path: graphData.path?.map(n=>String(n.id))||[],
             traversed_edges: [], // Fallback if animationData is empty
             is_final: true
         };
         console.log("Rendering final static visualization.");
         renderD3Graph(graphData.layout, baseLinks, finalStepData);
         setText(visualizationTitleEl, 'Graph Visualization: Final Paths');
         updateAnimationControls(); // Ensure controls reflect the final step
    }

    function clearVisualization() {
        if (simulation) simulation.stop();
        if (svg) svg.remove();
        if (tooltip) tooltip.remove();
        setHtml(graphVisContainer, ''); // Clear container content
        svg = g = simulation = nodeSelection = linkSelection = labelSelection = tooltip = zoomBehavior = null;
        nodeMap.clear(); // Clear the node map
        baseLinks = []; // Clear base links
        setupTooltip(); // Recreate tooltip div
    }

    function setupTooltip() {
        if (!tooltip) {
             tooltip = d3.select('body').append('div')
                 .attr('class', 'tooltip')
                 .style('opacity', 0);
         }
    }

     function renderD3Graph(layout, allLinksData, stepData) {
         const container = graphVisContainer;
         const width = container.clientWidth;
         const height = container.clientHeight;

         if (!width || !height || !layout || Object.keys(layout).length === 0) {
             console.warn("Cannot render D3: Invalid container size or no layout data.");
             clearVisualization();
             setHtml(container, '<div class="alert alert-warning">Cannot render graph: Invalid container size or layout data.</div>');
             return;
         }

         // --- Prepare Node Data (Mapping ID to object) ---
         nodeMap.clear(); // Clear previous map
         const nodes = Object.entries(layout).map(([nodeId, position]) => {
             // Find title etc. from the original path data if possible
             const gnnNode = graphData.path?.find(n => String(n.id) === nodeId);
             const bfsNode = graphData.bfs_path?.find(n => String(n.id) === nodeId);
             const nodeDetail = gnnNode || bfsNode || { id: nodeId, title: `Node ${nodeId}` }; // Fallback
             const nodeObject = {
                 id: nodeId, // Ensure ID is string
                 title: nodeDetail.title || `Node ${nodeId}`,
                 fx: position.x * width, // Use fx/fy for fixed positions from layout
                 fy: position.y * height,
                 type: getNodeType(nodeId, stepData) // Determine color/size based on step
             };
             nodeMap.set(nodeId, nodeObject); // Store in map for link lookup
             return nodeObject;
         });
         console.log(`Prepared ${nodes.length} nodes for D3 render.`);

         // --- Prepare Link Data (Filter and add type) ---
         const processedLinks = allLinksData
             .map(link => {
                 const sourceNode = nodeMap.get(link.source);
                 const targetNode = nodeMap.get(link.target);
                 // Only include links where both nodes exist in the current layout/map
                 if (!sourceNode || !targetNode) return null;
                 return {
                     source: sourceNode, // Link directly to node objects
                     target: targetNode,
                     type: getLinkType(link.source, link.target, stepData) // Determine color/width based on step
                 };
             })
             .filter(link => link !== null); // Remove invalid links
         console.log(`Prepared ${processedLinks.length} links for D3 render.`);


         // --- Setup SVG and Zoom (only if they don't exist) ---
         if (!svg) {
             console.log("Creating SVG and zoom behavior");
             container.innerHTML = ''; // Clear placeholder
             svg = d3.select(container).append('svg');
             g = svg.append('g').attr('class', 'zoom-content'); // Group for zoomable content

             // Create selections for links, nodes, labels within the group 'g'
             // These are placeholders initially, filled by .data().join() later
             linkSelection = g.append("g").attr("class", "links").selectAll("line");
             nodeSelection = g.append("g").attr("class", "nodes").selectAll("circle");
             labelSelection = g.append("g").attr("class", "labels").selectAll("text");

             zoomBehavior = d3.zoom().scaleExtent([0.1, 8]).on("zoom", (event) => {
                 if (g) g.attr("transform", event.transform);
             });
             d3.select(container).call(zoomBehavior); // Apply zoom to the container div
         }

         // Update SVG size (important if container resizes)
         d3.select(container).select('svg').attr('width', width).attr('height', height);

         // --- D3 Data Join ---
         // Links
         linkSelection = linkSelection
             .data(processedLinks, d => `${d.source.id}-${d.target.id}`)
             .join(
                 enter => enter.append("line").attr("class", d => `link ${d.type}-link`), // Set class on enter
                 update => update.attr("class", d => `link ${d.type}-link`), // Update class
                 exit => exit.remove()
             )
             .attr("stroke", d => linkColors[d.type] || linkColors.regular)
             .attr("stroke-width", d => linkWidths[d.type] || linkWidths.regular)
             .attr("stroke-opacity", 0.7)
             .attr('x1', d => d.source.fx) // Use fixed positions
             .attr('y1', d => d.source.fy)
             .attr('x2', d => d.target.fx)
             .attr('y2', d => d.target.fy);


         // Nodes
         nodeSelection = nodeSelection
             .data(nodes, d => d.id)
             .join(
                 enter => enter.append("circle")
                               .attr("class", "node")
                               .attr("stroke", "#333").attr("stroke-width", 1)
                               .style("cursor", "pointer")
                               .on('mouseover', handleMouseOverNode)
                               .on('mouseout', handleMouseOutNode)
                               .on('click', handleClickNode)
                               .attr("cx", d => d.fx).attr("cy", d => d.fy), // Set initial fixed position
                 update => update,
                 exit => exit.remove()
             )
             // Apply attributes to both enter and update selections
             .attr("r", d => nodeSizes[d.type] || nodeSizes.regular)
             .attr("fill", d => nodeColors[d.type] || nodeColors.regular)
             // Update position if layout somehow changes (unlikely with fixed fx/fy)
             .attr("cx", d => d.fx)
             .attr("cy", d => d.fy);

        // Labels (Source/Target only for less clutter)
         labelSelection = labelSelection
             .data(nodes.filter(d => d.type === 'source' || d.type === 'target'), d => d.id)
             .join(
                 enter => enter.append("text").attr("class", "node-label")
                                .attr("dx", d => (nodeSizes[d.type] || nodeSizes.regular) + 4)
                                .attr("dy", "0.35em").attr("pointer-events", "none"),
                 update => update,
                 exit => exit.remove()
             )
             .attr("x", d => d.fx).attr("y", d => d.fy) // Position near node
             .text(d => d.title.length > 15 ? d.title.substring(0,12)+'...' : d.title); // Truncate

         // No simulation needed if using fx/fy from layout

         console.log("D3 rendering complete for step.");
     }

    // --- Helper Functions for D3 ---
    function getNodeType(nodeId, stepData) {
        const isSource = nodeId === String(graphData.path?.[0]?.id || graphData.bfs_path?.[0]?.id);
        const isTarget = nodeId === String(graphData.path?.[graphData.path.length - 1]?.id || graphData.bfs_path?.[graphData.bfs_path.length - 1]?.id);
        // Check paths first, as they persist in the final frame
        const inGnnPath = stepData.gnn_path?.includes(nodeId);
        const inBfsPath = stepData.bfs_path?.includes(nodeId);
        // Check frontiers (relevant in non-final frames)
        const inGnnFwd = stepData.gnn_frontier_forward?.includes(nodeId);
        const inGnnBwd = stepData.gnn_frontier_backward?.includes(nodeId);
        const inBfsFwd = stepData.bfs_frontier_forward?.includes(nodeId);
        const inBfsBwd = stepData.bfs_frontier_backward?.includes(nodeId);

        if (isSource) return 'source';
        if (isTarget) return 'target';
        if (inGnnPath && inBfsPath) return 'both_path'; // Must be checked before single paths
        if (inGnnPath) return 'gnn_path';
        if (inBfsPath) return 'bfs_path';
        // If not in a final path, check frontiers
        const inGnnFrontier = inGnnFwd || inGnnBwd;
        const inBfsFrontier = inBfsFwd || inBfsBwd;
        if (inGnnFrontier && inBfsFrontier) return 'both_frontier';
        if (inGnnFrontier) return 'gnn_frontier';
        if (inBfsFrontier) return 'bfs_frontier';
        // Default if none of the above
        return 'regular';
    }

    function getLinkType(sourceId, targetId, stepData) {
        // Check final paths first
        const isGnnLink = isLinkInPath(sourceId, targetId, stepData.gnn_path);
        const isBfsLink = isLinkInPath(sourceId, targetId, stepData.bfs_path);
        if (isGnnLink && isBfsLink) return 'both_path';
        if (isGnnLink) return 'gnn_path';
        if (isBfsLink) return 'bfs_path';

        // If not a path link, check if it was traversed (only relevant if paths weren't found/final)
        const traversedEdges = stepData.traversed_edges || []; // Expecting [[u,v], ...]
        const isTraversed = traversedEdges.some(edge =>
            (String(edge[0]) === sourceId && String(edge[1]) === targetId) ||
            (String(edge[0]) === targetId && String(edge[1]) === sourceId)
        );
        if (isTraversed) return 'traversed';

        return 'regular'; // Default link type
    }

    function isLinkInPath(u, v, path) { // path is an array of node IDs
        if (!path || path.length < 2) return false;
        for (let i = 0; i < path.length - 1; i++) {
            if ((path[i] === u && path[i+1] === v) || (path[i] === v && path[i+1] === u)) {
                return true;
            }
        }
        return false;
    }

    // --- D3 Event Handlers ---
    function handleMouseOverNode(event, d) {
        if (!tooltip) return;
        tooltip.transition().duration(100).style('opacity', .9);
        tooltip.html(d.title || `Node ${d.id}`)
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY - 15}px`);
        d3.select(this).raise().attr('stroke-width', 2.5).attr('stroke', '#000');
        // highlightNeighbors(d, true); // Optional neighbor highlighting
    }

    function handleMouseOutNode(event, d) {
        if (!tooltip) return;
        tooltip.transition().duration(300).style('opacity', 0);
        d3.select(this).attr('stroke-width', 1).attr('stroke', '#333');
        // highlightNeighbors(d, false); // Optional neighbor highlighting
    }

    function handleClickNode(event, d) {
        console.log("Node clicked:", d);
        handleShowNodeInfo(d.id); // Use the existing handler
    }

    // Optional: Neighbor Highlighting Logic (can be complex)
    // function highlightNeighbors(centerNode, highlight) { ... }

    // D3 Drag Handling (Not needed if using fixed layout fx/fy)
    /*
    function drag(simulationInstance) {
        function dragstarted(event, d) { if (!event.active) simulationInstance.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; d3.select(this).raise(); }
        function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
        function dragended(event, d) { if (!event.active) simulationInstance.alphaTarget(0); d.fx = null; d.fy = null; } // Release node
        return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
    }
    */


    // --- Animation Controls ---

    function handleAnimationToggle() {
        console.log("Animation toggle changed:", animationToggle.checked);
        stopAnimation(); // Stop if playing
        updateAnimationControlsVisibility();
        if (graphData) {
            // Re-render based on the new toggle state
            if (animationToggle.checked && animationData.length > 0) {
                renderVisualizationStep(currentStepIndex); // Re-render current step
            } else {
                renderVisualizationFinal(); // Render final static state
            }
        }
    }

    function updateAnimationControlsVisibility() {
        const show = animationToggle && animationToggle.checked && animationData.length > 1;
        if(animationProgress) animationProgress.style.display = show ? 'flex' : 'none';
        if(animationControls) animationControls.style.display = show ? 'flex' : 'none';
    }

    function updateAnimationControls() {
        if (!animationToggle || !animationToggle.checked || animationData.length <= 1) return;

        const progressBar = animationProgress ? animationProgress.querySelector('.progress-bar') : null;
        const totalAnimSteps = Math.max(1, animationData.length - 1);
        const percentage = Math.min(100, Math.max(0, (currentStepIndex / totalAnimSteps) * 100));

        if(progressBar) {
             progressBar.style.width = `${percentage}%`;
             progressBar.setAttribute('aria-valuenow', percentage);
        }
        if(currentStepEl && animationData[currentStepIndex]) setText(currentStepEl, `Step ${animationData[currentStepIndex].step}`);
        if(totalStepsEl && animationData[animationData.length - 1]) setText(totalStepsEl, animationData[animationData.length - 1].step);
        if(animationPrevBtn) animationPrevBtn.disabled = currentStepIndex === 0;
        if(animationNextBtn) animationNextBtn.disabled = currentStepIndex === animationData.length - 1;
        if(animationPlayBtn) setHtml(animationPlayBtn, isAnimationPlaying ? '<i class="bi bi-pause-fill"></i> Pause' : '<i class="bi bi-play-fill"></i> Play');
        if(visualizationTitleEl && animationData[currentStepIndex]) setText(visualizationTitleEl, `Graph Visualization: ${animationData[currentStepIndex].title}`);
    }

    function previousStep() {
        if (currentStepIndex > 0) {
            stopAnimation();
            renderVisualizationStep(currentStepIndex - 1);
        }
    }

    function nextStep() {
        if (currentStepIndex < animationData.length - 1) {
            stopAnimation();
            renderVisualizationStep(currentStepIndex + 1);
        }
    }

    function toggleAnimation() {
        if (isAnimationPlaying) {
            stopAnimation();
        } else {
            startAnimation();
        }
    }

    function startAnimation() {
        if (isAnimationPlaying || !animationToggle || !animationToggle.checked || animationData.length <= 1) return;
        if (currentStepIndex >= animationData.length - 1) { // If at end, restart
            currentStepIndex = 0;
            renderVisualizationStep(0); // Show first frame first
        }
        isAnimationPlaying = true;
        updateAnimationControls(); // Update button to 'Pause'
        animationTimer = setInterval(() => {
            if (currentStepIndex < animationData.length - 1) {
                renderVisualizationStep(currentStepIndex + 1);
            } else {
                stopAnimation(); // Reached the end
            }
        }, ANIMATION_SPEED_MS);
    }

    function stopAnimation() {
        if (animationTimer) { clearInterval(animationTimer); animationTimer = null; }
        isAnimationPlaying = false;
        if (animationToggle && animationToggle.checked) { updateAnimationControls(); } // Update button to 'Play'
    }

    function resetAnimation() {
        stopAnimation();
        currentStepIndex = 0;
        animationData = [];
        if (currentStepEl) setText(currentStepEl, 'Step 0');
        if (totalStepsEl) setText(totalStepsEl, '0');
        const progressBar = animationProgress ? animationProgress.querySelector('.progress-bar') : null;
        if(progressBar) progressBar.style.width = '0%';
        if(animationPrevBtn) animationPrevBtn.disabled = true;
        if(animationNextBtn) animationNextBtn.disabled = true;
        updateAnimationControlsVisibility();
    }


    console.log("Main JS initialized with D3/Animation logic.");

}); // End DOMContentLoaded