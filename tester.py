import sys
import json
from neo4j import GraphDatabase

def get_relationships(uri, user, password):
    """
    Retrieve all direct relationships (links) between Wikipedia pages from your Neo4j database.
    Returns a list of tuples (source_title, target_title).
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    relationships = []
    with driver.session() as session:
        query = """
        MATCH (a:WikipediaPage)-[:LINKS_TO]->(b:WikipediaPage)
        RETURN a.title as source, b.title as target
        """
        result = session.run(query)
        for record in result:
            relationships.append((record["source"], record["target"]))
    driver.close()
    return relationships

def get_all_shortest_paths(driver, source_title, target_title, max_depth=15):
    # Build the query string dynamically using the max_depth value as a literal
    query = (
        "MATCH (source:WikipediaPage {title: $source_title}), "
        "      (target:WikipediaPage {title: $target_title}) "
        "MATCH p = allShortestPaths((source)-[*.." + str(max_depth) + "]-(target)) "
        "RETURN [node IN nodes(p) | node.title] AS path"
    )
    
    with driver.session() as session:
        result = session.run(query, source_title=source_title, target_title=target_title)
        paths = [record["path"] for record in result]
    return paths

def main():
    # Connection details – update these if necessary
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "rootroot"
    
    # Initialize the Neo4j driver.
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Retrieve all relationships that were created (each link between pages).
    relationships = get_relationships(uri, user, password)
    if not relationships:
        print("No relationships found in the database. Exiting.")
        driver.close()
        sys.exit(0)
    
    # Dictionary to store the results.
    shortest_paths_dict = {}

    # Specify the title to monitor (for example, "A")
    specific_title = "A"
    relationship_count = 0
    
    # For each relationship, compute all shortest paths and store them.
    for source, target in relationships:
        relationship_count += 1

        # Skip self-relationships (where the source and target are the same).
        if source == target:
            print(f"Skipping self-relationship for '{source}'.")
            continue
        
        # Construct the dictionary key in the form "source-target"
        key = f"{source}-{target}"
        paths = get_all_shortest_paths(driver, source, target)
        
        # Format each path from a list of titles into a hyphen-separated string.
        formatted_paths = ["-".join(path) for path in paths]
        
        # Only add if there's at least one shortest path.
        if formatted_paths:
            shortest_paths_dict[key] = formatted_paths
            # If the source title matches our specific title of interest, output to console.
            if source == specific_title:
                print(f"All shortest paths for {source} have been found!")
        else:
            print(f"No shortest paths found between '{source}' and '{target}'.")
        
        # Optional logging: print progress every 50 relationships processed.
        if relationship_count % 50 == 0:
            print(f"Processed {relationship_count} relationships so far...")

    driver.close()

    # Save the resulting dictionary to a JSON file.
    json_filename = "shortest_paths.json"
    with open(json_filename, "w") as outfile:
        json.dump(shortest_paths_dict, outfile, indent=4)
    print(f"Shortest paths data has been saved to {json_filename}")

    # Optionally, output the dictionary to the console.
    for key, paths in shortest_paths_dict.items():
        print(f"{key}:")
        for p in paths:
            print(f"  {p}")

if __name__ == "__main__":
    print("Hello")
    main()
