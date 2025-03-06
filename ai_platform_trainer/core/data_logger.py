import json
import logging
import os
import time
import math
import pygame
from typing import Any, Dict, List


class DataLogger:
    """
    Handles appending data points in memory and batching writes
    to a JSON file to improve performance during training.
    
    This enhanced version provides a unified event-based approach to logging
    that centralizes all data collection for easier analysis and expansion.
    """

    def __init__(self, filename: str = None, batch_mode: bool = True) -> None:
        """
        Initialize the DataLogger with the given filename.
        
        Args:
            filename: Optional path to the JSON file where data will be logged.
                     If None, a timestamped filename is generated.
            batch_mode: If True, all data is stored in memory and only written
                       at the end when save() is called. If False, data is
                       written immediately after each log() call.
        """
        if filename:
            self.filename = filename
        else:
            # Generate a filename with timestamp if none provided
            timestamp = int(time.time())
            self.filename = f"data/raw/training_data_{timestamp}.json"
        
        # Primary data storage categorized by event type
        self.events: Dict[str, List[Dict[str, Any]]] = {
            "collision": [],       # Player-enemy collisions
            "missile": [],         # Missile trajectories and outcomes
            "enemy_movement": [],  # Enemy movement patterns
            "player_movement": [], # Player movement patterns
            "game_state": []       # Overall game state snapshots
        }
        
        # Backward compatibility for older data formats
        self.legacy_data: List[Dict[str, Any]] = []
        self.collision_data: List[Dict[str, Any]] = []
        
        self.batch_mode = batch_mode
        self.save_counter = 0
        self.save_frequency = 1000  # Save every 1000 data points in non-batch mode
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Initialize file
        self._init_file()
        
        logging.info(f"DataLogger initialized with batch_mode={batch_mode}")
        
    def _init_file(self) -> None:
        """Initialize the log file."""
        # Remove existing file if it exists
        if os.path.isfile(self.filename):
            try:
                os.remove(self.filename)
            except OSError as e:
                logging.error(f"Error deleting existing file {self.filename}: {e}")

        # Create an empty JSON file
        try:
            with open(self.filename, "w") as f:
                json.dump([], f, indent=4)
        except IOError as e:
            logging.error(f"Error creating file {self.filename}: {e}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event of a specific type with associated data.
        
        Args:
            event_type: Category of the event (e.g., 'collision', 'missile', etc.)
            data: Dictionary containing event data
        """
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = pygame.time.get_ticks() if 'pygame' in globals() else int(time.time() * 1000)
            
        # Store event in appropriate category
        if event_type in self.events:
            self.events[event_type].append(data)
        else:
            # Create category if it doesn't exist
            self.events[event_type] = [data]
            
        # Save periodically in non-batch mode
        if not self.batch_mode:
            self.save_counter += 1
            if self.save_counter >= self.save_frequency:
                self.save()
                self.save_counter = 0
    
    def log_missile_event(self, missile, player_pos, enemy_pos, current_time, 
                         outcome=None, action=None) -> None:
        """
        Log missile-related event with all relevant context.
        
        Args:
            missile: The missile entity
            player_pos: Player position dictionary
            enemy_pos: Enemy position dictionary
            current_time: Current game time
            outcome: Optional outcome (hit/miss)
            action: Optional AI action taken
        """
        if not missile or not player_pos or not enemy_pos:
            return
            
        # Calculate missile angle
        try:
            missile_angle = math.atan2(missile.vy, missile.vx)
        except (AttributeError, TypeError):
            missile_angle = 0
            
        # Get missile action from AI if available
        missile_action = action if action is not None else getattr(missile, "last_action", 0.0)
        
        # Create event data
        event_data = {
            "player_x": player_pos.get("x", 0),
            "player_y": player_pos.get("y", 0),
            "enemy_x": enemy_pos.get("x", 0),
            "enemy_y": enemy_pos.get("y", 0),
            "missile_x": missile.pos.get("x", 0) if hasattr(missile, "pos") else 0,
            "missile_y": missile.pos.get("y", 0) if hasattr(missile, "pos") else 0,
            "missile_angle": missile_angle,
            "missile_action": missile_action,
            "timestamp": current_time,
            "outcome": outcome
        }
        
        self.log_event("missile", event_data)
    
    def log_collision_event(self, player_pos, enemy_pos, current_time, 
                          is_collision: bool, missile=None) -> None:
        """
        Log a collision or near-miss event.
        
        Args:
            player_pos: Player position dictionary
            enemy_pos: Enemy position dictionary
            current_time: Current game time
            is_collision: Whether a collision occurred
            missile: Optional missile involved (for missile collisions)
        """
        if not player_pos or not enemy_pos:
            return
        
        # Calculate distance
        dx = player_pos.get("x", 0) - enemy_pos.get("x", 0)
        dy = player_pos.get("y", 0) - enemy_pos.get("y", 0)
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Create event data
        event_data = {
            "timestamp": current_time,
            "player_position": player_pos,
            "enemy_position": enemy_pos,
            "distance": distance,
            "collision": 1 if is_collision else 0,
            "collision_type": "missile" if missile else "player"
        }
        
        # Add missile data if applicable
        if missile:
            event_data["missile_position"] = {
                "x": missile.pos.get("x", 0) if hasattr(missile, "pos") else 0,
                "y": missile.pos.get("y", 0) if hasattr(missile, "pos") else 0
            }
        
        self.log_event("collision", event_data)
    
    def log_game_state(self, game_state: Dict[str, Any], current_time: int = None) -> None:
        """
        Log overall game state for analysis.
        
        Args:
            game_state: Dictionary of game state variables
            current_time: Optional current time
        """
        if current_time is None:
            current_time = pygame.time.get_ticks() if 'pygame' in globals() else int(time.time() * 1000)
            
        game_state["timestamp"] = current_time
        self.log_event("game_state", game_state)
    
    # Legacy methods for backward compatibility
    def _deprecated_method_warning(self, method_name):
        """Display a warning when a deprecated method is called"""
        logging.warning(
            f"DEPRECATED: {method_name} is deprecated and will be removed in a future version. "
            f"Use log_event() or a specialized event logging method instead."
        )
    
    def is_valid_data_point(self, data):
        """
        Validates a data point before logging.
        DEPRECATED: Use log_event() instead.
        
        :param data: The data point to validate
        :return: True if data is valid, False otherwise
        """
        self._deprecated_method_warning("is_valid_data_point()")
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
        Create a record and log it if valid (legacy method).
        DEPRECATED: Use log_collision_event() instead.
        
        :param timestamp: Time of the data point
        :param player_pos: Dictionary with x,y player position
        :param enemy_pos: Dictionary with x,y enemy position
        :param distance: Distance between player and enemy
        :param collision: Whether collision occurred (defaults to 0)
        """
        self._deprecated_method_warning("log_data()")
        # Create legacy record
        record = {
            "timestamp": timestamp,
            "player_position": player_pos,
            "enemy_position": enemy_pos,
            "distance": distance,
            "collision": collision
        }
        
        # Store in legacy format for backward compatibility
        if self.is_valid_data_point(record):
            self.collision_data.append(record)
            
            # Also log as a new-style collision event
            self.log_collision_event(
                player_pos, enemy_pos, timestamp, bool(collision)
            )
            
            # If not in batch mode, save periodically
            if not self.batch_mode:
                self.save_counter += 1
                if self.save_counter >= self.save_frequency:
                    self.save()
                    self.save_counter = 0
        else:
            logging.warning("Invalid data point, skipping")
    
    def log(self, data_point: Dict[str, Any]) -> None:
        """
        Add a data point to the internal list of logged data (legacy method).
        DEPRECATED: Use log_event() with appropriate event_type instead.

        :param data_point: dictionary containing data to log
        """
        self._deprecated_method_warning("log()")
        self.legacy_data.append(data_point)
        
        # Try to convert to event-based format if possible
        if "missile_collision" in data_point:
            # This appears to be a missile event
            self.log_event("missile", data_point)
        
        # If not in batch mode, save periodically
        if not self.batch_mode:
            self.save_counter += 1
            if self.save_counter >= self.save_frequency:
                self.save()
                self.save_counter = 0

    def save(self) -> None:
        """
        Write the logged data to the JSON file.
        """
        # Flatten all events into a single list
        all_events = []
        for event_list in self.events.values():
            all_events.extend(event_list)
            
        # Add legacy data formats for backward compatibility
        all_events.extend(self.legacy_data)
        all_events.extend(self.collision_data)
        
        try:
            with open(self.filename, "w") as f:
                json.dump(all_events, f, indent=4)
            logging.info(f"Saved {len(all_events)} data points to {self.filename}")
            
            # Clear data arrays if in non-batch mode to avoid memory bloat
            if not self.batch_mode:
                for event_type in self.events:
                    self.events[event_type] = []
                self.legacy_data = []
                self.collision_data = []
                
        except IOError as e:
            logging.error(f"Error saving data to {self.filename}: {e}")
