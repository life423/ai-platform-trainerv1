# core/data_logger.py
import json
import os

class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def log(self, data_point):
        # This method must exist so we can log data
        self.data.append(data_point)

    def save(self):
        # Save all logged data to the specified JSON file
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)