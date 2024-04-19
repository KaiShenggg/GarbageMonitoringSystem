import boto3
import json
import logging
import os
import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CALCULATOR_NAME = os.environ.get('CALCULATOR_NAME', 'MyRouteCalculator')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'analytics-bins-s3')

iot_analytics = boto3.client('iotanalytics')
location = boto3.client('location')
s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')

LAST_FILE_NAME = '99'
FILL_LEVEL_THRESHOLD = 0.3 # Value is between 0 and 1

def nearest_neighbor(route_matrix: List[List[Dict[str, float]]]) -> List[int]:
    """
    Finds the initial solution using the Nearest Neighbor heuristic.

    Args:
        route_matrix (list of lists): The route matrix representing distances between nodes.

    Returns:
        list: The initial path found using the Nearest Neighbor heuristic.
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

def two_opt(path: List[int], route_matrix: List[List[Dict[str, float]]]) -> List[int]:
    """
    Optimizes a given path using the 2-opt heuristic.

    Args:
        path (list): A list representing the order of nodes visited.
        route_matrix (list of lists): The route matrix representing distances between nodes.

    Returns:
        list: The optimized path after applying the 2-opt heuristic.
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

def total_distance(path: List[int], route_matrix: List[List[Dict[str, float]]]) -> float:
    """
    Calculates the total distance of a given path.

    Args:
        path (list): A list representing the order of nodes visited.
        route_matrix (list of lists): The route matrix representing distances between nodes.

    Returns:
        float: The total distance of the path.
    """
    total = 0
    num_nodes = len(path)
    for i in range(num_nodes - 1):
        total += route_matrix[path[i]][path[i+1]]['Distance']
    total += route_matrix[path[-1]][path[0]]['Distance']  # Return to the starting node
    return total

def get_coordinates(path: List[int], coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Converts node indices to coordinates.

    Args:
        path (list): A list representing the order of nodes visited.
        coordinates (list of tuples): A list of coordinates corresponding to each node.

    Returns:
        list of tuples: The coordinates of the path.
    """
    return [coordinates[node] for node in path]

def create_dataset(date_str: str):
    try:
        # Define the dataset name using the current date
        dataset_name = f"Day_{date_str.replace('-', '_')}"  # Replace hyphens with underscores

        # Define the SQL query with the current date filter
        sql_query = f"SELECT * FROM binsdata_datastore WHERE SUBSTRING(timestamp, 1, 10) = '{date_str}'"

        # Check if the dataset already exists
        existing_datasets = iot_analytics.list_datasets()['datasetSummaries']
        for ds in existing_datasets:
            if ds['datasetName'] == dataset_name:
                # Delete the existing dataset
                iot_analytics.delete_dataset(datasetName=dataset_name)
                print(f"Existing dataset '{dataset_name}' deleted.")
                break  # Exit loop once dataset is found
        
        # Create the dataset
        response = iot_analytics.create_dataset(
            datasetName=dataset_name,
            actions=[
                {
                    'actionName': 'query_action',
                    'queryAction': {
                        'sqlQuery': sql_query
                    }
                }
            ]
        )

        print(f"Dataset '{dataset_name}' created successfully with query: {sql_query}")

        # Run the dataset
        iot_analytics.create_dataset_content(datasetName=dataset_name)

        return response['datasetArn']
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise e
        
def create_sunday_dataset(date_str: str):
    try:
        date_object = datetime.strptime(date_str, '%Y-%m-%d')
        previous_monday_date = (date_object - timedelta(days=6)).strftime('%Y-%m-%d')

        # Define the dataset name using the current date
        dataset_name = f"Week_{date_str.replace('-', '_')}"  # Replace hyphens with underscores

        # Define the SQL query with the current date filter
        sql_query = f"SELECT * FROM binsdata_datastore WHERE CAST(SUBSTRING(timestamp, 1, 10) AS DATE) >= CAST('{previous_monday_date}' AS DATE) AND CAST(SUBSTRING(timestamp, 1, 10) AS DATE) <= CAST('{date_str}' AS DATE) ORDER BY timestamp"

        # Check if the dataset already exists
        existing_datasets = iot_analytics.list_datasets()['datasetSummaries']
        for ds in existing_datasets:
            if ds['datasetName'] == dataset_name:
                # Delete the existing dataset
                iot_analytics.delete_dataset(datasetName=dataset_name)
                print(f"Existing dataset '{dataset_name}' deleted.")
                break  # Exit loop once dataset is found
        
        # Create the dataset
        response = iot_analytics.create_dataset(
            datasetName=dataset_name,
            actions=[
                {
                    'actionName': 'query_action',
                    'queryAction': {
                        'sqlQuery': sql_query
                    }
                }
            ]
        )

        print(f"Dataset '{dataset_name}' created successfully with query: {sql_query}")

        # Run the dataset
        iot_analytics.create_dataset_content(datasetName=dataset_name)

        return response['datasetArn']
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise e

def lambda_handler(event: Dict, context) -> Dict:
    """
    Lambda function entry point.

    Args:
        event: The event data passed to the Lambda function.
        context (LambdaContext): The runtime information of the Lambda function.

    Returns:
        dict: The response indicating the status of the operation.
    """
    try:
        # Get the bucket name and key from the event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        # Check if it is the last file
        if os.path.basename(key) == LAST_FILE_NAME:
            area_folder = 'area-UNKNOWN'
            
            # Use regular expression to find the "area-X" part in the key
            match = re.search(r'area-\d+', key)
            if match:
                area_folder = match.group()

            # Extract the date folder from the key
            date_folder = os.path.dirname(key)

            # Initialize an empty list to store the coordinates
            coordinates = []

            # List all objects within the date folder
            bucket_obj = s3_resource.Bucket(bucket)
            objects = bucket_obj.objects.filter(Prefix=date_folder + "/")

            # Iterate through each object
            for obj in objects:
                # Get the object key
                obj_key = obj.key

                # Read the object data from S3
                body = obj.get()['Body'].read().decode('utf-8')

                # Parse the JSON data
                obj_data = json.loads(body)
                
                # Send each object's data to IoT Analytics channel
                iot_analytics.batch_put_message(
                    channelName='binsData_channel',
                    messages=[
                        {
                            'messageId': obj_key,
                            'payload': json.dumps(obj_data)
                        }
                    ]
                )

                # Check if the fill level exceeds the threshold
                if obj_data['fillLevel'] >= FILL_LEVEL_THRESHOLD:
                    # Extract longitude and latitude
                    longitude = obj_data['location']['longitude']
                    latitude = obj_data['location']['latitude']

                    coordinates.append([longitude, latitude])

            if len(coordinates) <= 2:
                optimized_route = coordinates
            else:
                # Calculate route matrix
                route_matrix = location.calculate_route_matrix(
                    CalculatorName=CALCULATOR_NAME,
                    DeparturePositions=coordinates,
                    DestinationPositions=coordinates)['RouteMatrix']

                # Optimize the route
                optimized_path = nearest_neighbor(route_matrix)
                optimized_path = two_opt(optimized_path, route_matrix)

                # Convert node indices to coordinates
                optimized_route = get_coordinates(optimized_path, coordinates)

            # Create the destination key for the output file
            date_str = os.path.basename(date_folder)
            destination_key = f"optimized-route/{area_folder}/{date_str}.json"

            # Upload the optimized route to the destination S3 bucket
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=destination_key,
                Body=json.dumps(optimized_route)
            )
            
            # Wait for 2 minutes to ensure data is fully ingested
            time.sleep(120)  # Delay for 2 minutes    
            
            datetime_format = datetime.strptime(date_str, '%Y-%m-%d')
            # If today is Sunday, create the Sunday dataset
            if datetime_format.weekday() == 6:  # In Python, Monday is 0 and Sunday is 6
                create_sunday_dataset(date_str)
                
            # Create dataset after data ingestion
            dataset_arn = create_dataset(date_str)

            return {
                'statusCode': 200,
                'body': f"Optimized route stored in {S3_BUCKET_NAME}/{destination_key}"
            }
    except Exception as e:
        logger.error(f"Error in Lambda function: {e}")
        return {
            'statusCode': 500,
            'body': str(e)
        }
