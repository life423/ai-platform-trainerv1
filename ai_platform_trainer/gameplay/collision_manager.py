"""
CollisionManager provides a unified approach to handling collisions in the game.
This helps keep collision logic consistent and allows for easier expansion.
"""

import logging
import pygame
import math
from typing import Callable, List, Optional, Dict, Any, Tuple

class CollisionManager:
    """
    Manages collision detection and handling between game entities.
    
    This class centralizes collision logic to avoid duplication and ensure
    consistent behavior throughout the game.
    """
    
    def __init__(self, data_logger=None, missile_manager=None):
        """
        Initialize the collision manager.
        
        Args:
            data_logger: Optional DataLogger for recording collision events
            missile_manager: Optional MissileManager for handling missile-specific logic
        """
        self.collision_handlers = {}
        self.data_logger = data_logger
        self.missile_manager = missile_manager
        logging.info("CollisionManager initialized")
    
    def set_data_logger(self, data_logger):
        """Set the data logger for collision events."""
        self.data_logger = data_logger
    
    def set_missile_manager(self, missile_manager):
        """Set the missile manager for missile collision handling."""
        self.missile_manager = missile_manager
    
    def register_handler(self, entity_type_pair: Tuple[str, str], 
                         handler: Callable) -> None:
        """
        Register a collision handler for a specific entity type pair.
        
        Args:
            entity_type_pair: A tuple of strings identifying the entity types
                             (e.g., ('player', 'enemy'))
            handler: Function to call when collision is detected
        """
        self.collision_handlers[entity_type_pair] = handler
        logging.debug(f"Registered collision handler for {entity_type_pair}")
    
    def check_rect_collision(self, rect1: pygame.Rect, 
                             rect2: pygame.Rect) -> bool:
        """
        Check if two pygame Rect objects are colliding.
        
        Args:
            rect1: First rectangle
            rect2: Second rectangle
            
        Returns:
            bool: True if the rectangles collide, False otherwise
        """
        return rect1.colliderect(rect2)
    
    def get_entity_rect(self, entity: Any) -> pygame.Rect:
        """
        Create a pygame Rect for an entity based on its position and size.
        
        Args:
            entity: A game entity with position and size attributes
            
        Returns:
            pygame.Rect: Rectangle representing the entity's position and size
        """
        # Handle different entity types with different position attribute formats
        if hasattr(entity, "position") and isinstance(entity.position, dict):
            # Player-style entity with position dict
            return pygame.Rect(
                entity.position["x"],
                entity.position["y"],
                entity.size,
                entity.size
            )
        elif hasattr(entity, "pos") and isinstance(entity.pos, dict):
            # Enemy-style entity with pos dict
            return pygame.Rect(
                entity.pos["x"],
                entity.pos["y"],
                entity.size,
                entity.size
            )
        elif hasattr(entity, "get_rect"):
            # Entity with get_rect method (like missiles)
            return entity.get_rect()
        else:
            # Default fallback
            logging.warning(f"Couldn't determine rect for entity: {entity}")
            return pygame.Rect(0, 0, 0, 0)
    
    def check_entity_collision(self, entity1: Any, entity2: Any) -> bool:
        """
        Check if two entities are colliding.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            bool: True if the entities collide, False otherwise
        """
        rect1 = self.get_entity_rect(entity1)
        rect2 = self.get_entity_rect(entity2)
        return self.check_rect_collision(rect1, rect2)
    
    def check_player_enemy_collision(self, player, enemy) -> bool:
        """
        Check if the player and enemy are colliding.
        
        Args:
            player: Player entity
            enemy: Enemy entity
            
        Returns:
            bool: True if the entities collide, False otherwise
        """
        if not player or not enemy:
            return False
        
        return self.check_entity_collision(player, enemy)
    
    def handle_player_enemy_collision(self, player, enemy, current_time, 
                                     respawn_delay=1000) -> bool:
        """
        Check and handle player-enemy collision.
        
        Args:
            player: Player entity
            enemy: Enemy entity
            current_time: Current game time
            respawn_delay: Delay before respawning the enemy
            
        Returns:
            bool: True if collision occurred and was handled
        """
        if not self.check_player_enemy_collision(player, enemy):
            return False
        
        logging.info("Collision detected between player and enemy.")
        
        # Log collision data if in training mode and data_logger is available
        if self.data_logger and hasattr(player, 'position') and hasattr(enemy, 'pos'):
            player_pos = player.position
            enemy_pos = enemy.pos
            
            dx = player_pos["x"] - enemy_pos["x"]
            dy = player_pos["y"] - enemy_pos["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            self.data_logger.log_data(
                current_time, 
                player_pos,
                enemy_pos,
                distance,
                1  # Collision occurred
            )
        
        # Hide enemy and trigger respawn
        if enemy:
            enemy.hide()
            
        return True
    
    def check_missile_enemy_collisions(self, missiles, enemy, 
                                      current_time=None, 
                                      respawn_callback=None,
                                      is_training=False) -> bool:
        """
        Check and handle collisions between missiles and an enemy.
        
        Args:
            missiles: List of missile entities
            enemy: Enemy entity
            current_time: Current game time (needed for training data)
            respawn_callback: Function to call to respawn enemy
            is_training: Whether we're in training mode
            
        Returns:
            bool: True if any missile collided with the enemy
        """
        if not missiles or not enemy:
            return False
        
        enemy_rect = self.get_entity_rect(enemy)
        collision_detected = False
        
        for missile in missiles[:]:  # Create a copy to avoid issues while removing
            if not missile:
                continue
                
            missile_rect = missile.get_rect()
            if missile_rect.colliderect(enemy_rect):
                logging.info(f"Missile collision detected with enemy")
                
                if self.missile_manager:
                    # Use the missile manager to handle the missile (removal, data logging)
                    if is_training:
                        self.missile_manager.finalize_missile_sequence(missile, True)
                    
                    # Remove the missile
                    if missile in self.missile_manager.missiles:
                        self.missile_manager.missiles.remove(missile)
                    if missile in self.missile_manager.missile_lifespan:
                        del self.missile_manager.missile_lifespan[missile]
                
                # Hide the enemy and trigger respawn
                enemy.hide()
                collision_detected = True
                
                # Call respawn callback if provided
                if respawn_callback:
                    respawn_callback()
                
                break
        
        return collision_detected
    
    def update(self, game, current_time, is_training=False):
        """
        Comprehensive update method that checks all relevant collisions.
        
        Args:
            game: Game instance containing entities
            current_time: Current game time
            is_training: Whether we're in training mode
            
        Returns:
            Dict with results of collision checks
        """
        results = {
            "player_enemy_collision": False,
            "missile_enemy_collision": False
        }
        
        # Check player-enemy collision
        if game.player and game.enemy:
            results["player_enemy_collision"] = self.handle_player_enemy_collision(
                game.player, 
                game.enemy,
                current_time,
                game.respawn_delay
            )
            
            if results["player_enemy_collision"]:
                game.is_respawning = True
                game.respawn_timer = current_time + game.respawn_delay
            
            # In training mode, also log non-collision data
            elif is_training and self.data_logger:
                player_pos = game.player.position
                enemy_pos = game.enemy.pos
                
                dx = player_pos["x"] - enemy_pos["x"]
                dy = player_pos["y"] - enemy_pos["y"]
                distance = math.sqrt(dx*dx + dy*dy)
                
                self.data_logger.log_data(
                    current_time, 
                    player_pos,
                    enemy_pos,
                    distance,
                    0  # No collision
                )
        
        # Check missile-enemy collisions
        if game.missile_manager and game.enemy:
            missiles = game.missile_manager.missiles
            
            def respawn_callback():
                game.is_respawning = True
                game.respawn_timer = current_time + game.respawn_delay
                logging.info("Missile-Enemy collision, enemy will respawn.")
            
            results["missile_enemy_collision"] = self.check_missile_enemy_collisions(
                missiles,
                game.enemy,
                current_time,
                respawn_callback,
                is_training
            )
        
        return results
    
    def check_collisions(self, entities_dict: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Check for collisions between all entities in the provided dictionary.
        
        Args:
            entities_dict: Dictionary mapping entity type names to entity objects
                          or lists of entity objects
        
        Returns:
            List of tuples containing the type names of colliding entities
        """
        collisions = []
        entity_types = list(entities_dict.keys())
        
        # Check each entity type against others
        for i, type1 in enumerate(entity_types):
            entity1 = entities_dict[type1]
            if not entity1:
                continue
                
            # Convert to list if it's a single entity
            entities1 = entity1 if isinstance(entity1, list) else [entity1]
            
            for type2 in entity_types[i:]:  # Only check each pair once
                if type1 == type2:
                    continue  # Skip self-collision (would need special handling)
                    
                entity2 = entities_dict[type2]
                if not entity2:
                    continue
                    
                # Convert to list if it's a single entity
                entities2 = entity2 if isinstance(entity2, list) else [entity2]
                
                # Check collisions between all entities of both types
                for e1 in entities1:
                    for e2 in entities2:
                        if self.check_entity_collision(e1, e2):
                            collisions.append((type1, type2))
                            
                            # Call registered handler if exists
                            handler = self.collision_handlers.get((type1, type2))
                            if handler:
                                handler(e1, e2)
                            
        return collisions
