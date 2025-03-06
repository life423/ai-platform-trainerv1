import json
import os
import time
from typing import Any, Dict, List


class DataLogger:
    """
    Handles appending data points in memory and periodically writing
    them to a JSON file.
    """

    def __init__(self, filename: str = None) -> None:
        """
        Initialize the DataLogger with the given filename. If no filename is 
        provided, a timestamped filename is generated. If file exists, it is 
        removed and replaced with an empty JSON file.

        :param filename: Optional path to the JSON file where data will be 
                       logged. If None, a timestamped filename is generated.
        """
        if filename:
            self.filename = filename
        else:
            # Generate a filename with timestamp if none provided
            timestamp = int(time.time())
            self.filename = f"data/raw/training_data_{timestamp}.json"
        self.data: List[Dict[str, Any]] = []

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Remove existing file
        if os.path.isfile(self.filename):
            try:
                os.remove(self.filename)
            except OSError as e:
                print(f"Error deleting existing file {self.filename}: {e}")

        # Create an empty JSON file
        try:
            with open(self.filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error creating file {self.filename}: {e}")
        
        # For collision data
        self.collision_data: List[Dict[str, Any]] = []
    
    def is_valid_data_point(self, data):
        """
        Validates a data point before logging.
        
        :param data: The data point to validate
        :return: True if data is valid, False otherwise
        """
        # Required keys
        required_keys = [
            "timestamp", "player_position", "enemy_position", 
            "distance", "collision"
        ]
        # Check that all exist
        for rk in required_keys:
            if rk not in data:
                return False

        # Check position fields contain required coordinates
        pos_fields = ["x", "y"]
        for pf in pos_fields:
            player_has_field = pf in data["player_position"]
            enemy_has_field = pf in data["enemy_position"]
            if not player_has_field or not enemy_has_field:
                return False
        # Ensure distance is numeric
        if not isinstance(data["distance"], (int, float)):
            return False

        return True
    
    def log_data(self, timestamp, player_pos, enemy_pos, distance, 
                 collision=0):
        """
        Create a record and log it if valid.
        
        :param timestamp: Time of the data point
        :param player_pos: Dictionary with x,y player position
        :param enemy_pos: Dictionary with x,y enemy position
        :param distance: Distance between player and enemy
        :param collision: Whether collision occurred (defaults to 0)
        """
        record = {
            "timestamp": timestamp,
            "player_position": player_pos,
            "enemy_position": enemy_pos,
            "distance": distance,
            "collision": collision
        }
        if self.is_valid_data_point(record):
            self.collision_data.append(record)
        else:
            print("[WARNING] Invalid data point, skipping")
    
    def log(self, data_point: Dict[str, Any]) -> None:
        """
        Add a data point to the internal list of logged data.

        :param data_point: dictionary containing data to log
        """
        self.data.append(data_point)

    def save(self) -> None:
        """
        Write the logged data to the JSON file.
        """
        # Combine regular data and collision data
        all_data = self.data + self.collision_data
        
        try:
            with open(self.filename, "w") as f:
                json.dump(all_data, f, indent=4)
        except IOError as e:
            print(f"Error saving data to {self.filename}: {e}")
