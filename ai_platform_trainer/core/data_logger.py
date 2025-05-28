import json
import os
from typing import Any, Dict, List


class DataLogger:
    """
    Handles appending data points in memory and periodically writing them to a JSON file.
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize the DataLogger. If the file exists and contains valid JSON,
        it loads the existing data. Otherwise, it starts with an empty data list.

        :param filename: path to the JSON file where data will be logged
        """
        self.filename = filename
        self.data: List[Dict[str, Any]] = []

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        if os.path.isfile(self.filename):
            try:
                with open(self.filename, "r") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        self.data = existing_data
                        print(f"Loaded {len(self.data)} existing records from {self.filename}")
                    else:
                        print(f"Warning: Existing file {self.filename} does not contain a JSON list. Starting fresh.")
                        self._create_empty_file()
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.filename}. Starting fresh.")
                self._create_empty_file()
            except IOError as e:
                print(f"Error reading file {self.filename}: {e}. Starting fresh.")
                self._create_empty_file()
        else:
            self._create_empty_file()

    def _create_empty_file(self) -> None:
        """Creates an empty JSON file (or overwrites with an empty list)."""
        try:
            with open(self.filename, "w") as f:
                json.dump([], f, indent=4)
            print(f"Initialized empty data log at {self.filename}")
        except IOError as e:
            print(f"Error creating file {self.filename}: {e}")

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
        try:
            with open(self.filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error saving data to {self.filename}: {e}")
