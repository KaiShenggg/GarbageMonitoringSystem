import boto3

CALCULATOR_NAME = 'MyRouteCalculator'

location = boto3.client('location')

def nearest_neighbor(route_matrix):
    """
    Finds the initial solution using the Nearest Neighbor heuristic.

    Args:
    - route_matrix (list of lists): The route matrix representing distances between nodes.

    Returns:
    - list: The initial path found using the Nearest Neighbor heuristic.
    """
    num_nodes = len(route_matrix)
    visited = [False] * num_nodes
    path = []
    current_node = 0
    visited[current_node] = True
    path.append(current_node)

    for _ in range(num_nodes - 1):
        next_node = None
        min_distance = float('inf')
        for neighbor in range(num_nodes):
            if not visited[neighbor] and route_matrix[current_node][neighbor]['Distance'] < min_distance:
                min_distance = route_matrix[current_node][neighbor]['Distance']
                next_node = neighbor
        visited[next_node] = True
        path.append(next_node)
        current_node = next_node

    return path

def total_distance(path, route_matrix):
    """
    Calculates the total distance of a given path.

    Args:
    - path (list): A list representing the order of nodes visited.
    - route_matrix (list of lists): The route matrix representing distances between nodes.

    Returns:
    - float: The total distance of the path.
    """
    total = 0
    num_nodes = len(path)
    for i in range(num_nodes - 1):
        total += route_matrix[path[i]][path[i+1]]['Distance']
    total += route_matrix[path[-1]][path[0]]['Distance']  # Return to the starting node
    return total

def two_opt(path, route_matrix):
    """
    Optimizes a given path using the 2-opt heuristic.

    Args:
    - path (list): A list representing the order of nodes visited.
    - route_matrix (list of lists): The route matrix representing distances between nodes.

    Returns:
    - list: The optimized path after applying the 2-opt heuristic.
    """
    num_nodes = len(path)
    best_path = path
    improved = True
    while improved:
        improved = False
        for i in range(1, num_nodes - 2):
            for j in range(i + 1, num_nodes):
                if j - i == 1:
                    continue  # Changes nothing, skip
                new_path = path[:]
                new_path[i:j] = path[j - 1:i - 1:-1]  # Reverse the segment
                if total_distance(new_path, route_matrix) < total_distance(best_path, route_matrix):
                    best_path = new_path
                    improved = True
        path = best_path
    return best_path

def get_coordinates(path, coordinates):
    """
    Converts node indices to coordinates.

    Args:
    - path (list): A list representing the order of nodes visited.
    - coordinates (list of tuples): A list of coordinates corresponding to each node.

    Returns:
    - list of tuples: The coordinates of the path.
    """
    return [coordinates[node] for node in path]

def lambda_handler(event, context):
    """
    Lambda function entry point.

    Args:
    - event (dict): The event data passed to the Lambda function.
    - context (LambdaContext): The runtime information of the Lambda function.

    Returns:
    - dict: The response containing the optimized path.
    """
    # Calculate route matrix
    route_matrix = location.calculate_route_matrix(
        CalculatorName=CALCULATOR_NAME,
        DeparturePositions=event['departure_positions'],
        DestinationPositions=event['destination_positions'])['RouteMatrix']
    
    # Get the coordinates of each node
    coordinates = event['departure_positions']
    
    # Optimize the route
    optimized_path = nearest_neighbor(route_matrix)
    optimized_path = two_opt(optimized_path, route_matrix)
    
    # Convert node indices to coordinates
    optimized_coordinates = get_coordinates(optimized_path, coordinates)
    
    return {
        'statusCode': 200,
        'body': optimized_coordinates
    }
