### WikiRaceGNN

# Set up

Download docker and run:
    docker-compose up -d


For creating dataset
www.sixdegreesofwikipedia.com

Three model implementations:
    LMm base model no GNN
        Nodes are initialized with their base LLM embedding
        
    Base model GNN
        Nodes are initialized with random, non-zero values

    LLM model with GNN
        Nodes are initialized with their base LLM embedding but will update with LLM

Building steps:
    Set up neo4j

    Traverse wiki through its main page and store 100,000 pages
        make sure to add what pages are connected to what other pages

    During each store, calculate its vector embedding and place it with the page

    Set up a program to select two random wikipedia pages

    Set up a program to do bidirectional A* search

    Create GNN
        GNN traverses to the next most similar wiki page (e.g largest dot-product of neighbouring node with target node)

        GNN will update the value of each node to reduce error






