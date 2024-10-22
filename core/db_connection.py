import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB connection string from environment variable
mongo_uri = os.getenv('MONGO_URI')

# Create a MongoClient object
client = MongoClient(mongo_uri)

# Example: Access the database and collection
db = client['pixel_pursuit_db']
collection = db['training_data']

# Test: Insert sample data to verify connection
test_data = {
    "timestamp": "2024-10-20T12:00:00Z",
    "player_position": {"x": 100, "y": 200},
    "enemy_position": {"x": 150, "y": 250},
    "distance": 70,
    "collision": False
}

collection.insert_one(test_data)

