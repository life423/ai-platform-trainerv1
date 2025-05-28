"""
Dataset class for loading enemy movement training data.
"""
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any


class EnemyDataset(Dataset):
    """
    PyTorch Dataset for loading enemy movement data from a JSON file.

    The JSON file is expected to contain a list of records, where each
    record has 'state' and 'action' keys.
    'state' should be a list of 5 numerical features.
    'action' should be a list of 2 numerical values (dx, dy).
    """
    def __init__(self, json_file: str):
        """
        Args:
            json_file (str): Path to the JSON file containing the
                             training data.
        """
        self.data: List[Dict[str, Any]] = []
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                # Ensure raw_data is a list of dictionaries
                if isinstance(raw_data, list) and \
                   all(isinstance(item, dict) for item in raw_data):
                    self.data = raw_data
                else:
                    # Attempt to load if it's a dictionary with a single key
                    # (e.g. "data")
                    if isinstance(raw_data, dict) and len(raw_data) == 1:
                        potential_data_list = next(iter(raw_data.values()))
                        if isinstance(potential_data_list, list) and \
                           all(isinstance(item, dict)
                               for item in potential_data_list):
                            self.data = potential_data_list
                        else:
                            raise ValueError(
                                "JSON data is not in the expected list "
                                "format or a dict with a single list."
                            )
                    else:
                        raise ValueError(
                            "JSON data is not in the expected list format."
                        )

            if not self.data:
                print(
                    f"Warning: No data loaded from {json_file}. "
                    f"The file might be empty or improperly formatted."
                )

        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_file}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_file}")
            raise
        except ValueError as e:
            print(f"Error: {e}")
            raise

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the state tensor
                                               and the action tensor.
        """
        item = self.data[idx]
        
        # Ensure 'state' and 'action' keys exist and are lists
        if 'state' not in item or not isinstance(item['state'], list):
            raise ValueError(f"Sample at index {idx} is missing 'state' or it's not a list.")
        if 'action' not in item or not isinstance(item['action'], list):
            raise ValueError(f"Sample at index {idx} is missing 'action' or it's not a list.")

        state = torch.tensor(item['state'], dtype=torch.float32)
        action = torch.tensor(item['action'], dtype=torch.float32)
        
        # Validate tensor shapes (optional but good practice)
        if state.shape != (5,):  # Expected input_size = 5
            raise ValueError(
                f"State tensor at index {idx} has incorrect shape: "
                f"{state.shape}. Expected (5,)"
            )
        if action.shape != (2,):  # Expected output_size = 2
            raise ValueError(
                f"Action tensor at index {idx} has incorrect shape: "
                f"{action.shape}. Expected (2,)"
            )

        return state, action


if __name__ == '__main__':
    # Example usage:
    # Create a dummy JSON file for testing
    dummy_data = {
        "data": [
            {"state": [1.0, 2.0, 3.0, 4.0, 5.0], "action": [0.1, 0.2]},
            {"state": [0.1, 0.2, 0.3, 0.4, 0.5], "action": [-0.1, -0.2]},
            {"state": [5.0, 4.0, 3.0, 2.0, 1.0], "action": [0.5, -0.5]},
        ]
    }
    dummy_json_path = "dummy_enemy_data.json"
    with open(dummy_json_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f)

    print(f"Attempting to load dataset from: {dummy_json_path}")
    try:
        dataset = EnemyDataset(json_file=dummy_json_path)
        print(f"Successfully loaded {len(dataset)} samples.")

        if len(dataset) > 0:
            sample_state, sample_action = dataset[0]
            print(f"First sample - State: {sample_state}, "
                  f"Action: {sample_action}")

            # Test with DataLoader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            for batch_idx, (states, actions) in enumerate(dataloader):
                print(f"\nBatch {batch_idx + 1}:")
                print(f"  States shape: {states.shape}")
                print(f"  Actions shape: {actions.shape}")
                print(f"  States: {states}")
                print(f"  Actions: {actions}")
                if batch_idx >= 0:  # Print only the first batch
                    break
        else:
            print("Dataset is empty, cannot retrieve samples or test "
                  "DataLoader.")

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
    finally:
        # Clean up dummy file
        import os
        if os.path.exists(dummy_json_path):
            os.remove(dummy_json_path)
            print(f"Cleaned up {dummy_json_path}")
