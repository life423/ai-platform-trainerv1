import time
from core.db_connection import client  # Use the MongoClient from db_connection


class DataLogger:
    def __init__(self, database_name, collection_name):
        # Set up the database and collection using MongoDB client
        self.db = client[database_name]
        self.collection = self.db[collection_name]

    def log_data(self, player_position, enemy_position, collision):
        # Calculate the distance between player and enemy
        distance = ((player_position[0] - enemy_position[0]) ** 2 +
                    (player_position[1] - enemy_position[1]) ** 2) ** 0.5

        # Create a data point
        data_point = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "player_position": {"x": player_position[0], "y": player_position[1]},
            "enemy_position": {"x": enemy_position[0], "y": enemy_position[1]},
            "distance": distance,
            "collision": collision
        }

        # Insert the data point into the MongoDB collection
        self.collection.insert_one(data_point)
     


if __name__ == "__main__":
    # Example usage of DataLogger
    data_logger = DataLogger("pixel_pursuit_db", "training_data")
    sample_player_position = [100, 200]
    sample_enemy_position = [150, 250]
    collision_status = False

    # Log sample data
    data_logger.log_data(sample_player_position,
                         sample_enemy_position, collision_status)
