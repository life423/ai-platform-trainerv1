"""
CollisionManager provides a unified approach to handling collisions in the game.
This helps keep collision logic consistent and allows for easier expansion.
"""

import logging
import pygame
from typing import Callable, List, Optional, Dict, Any, Tuple


class CollisionManager:
    """
    Manages collision detection and handling between game entities.
    
    This class centralizes collision logic to avoid duplication and ensure
    consistent behavior throughout the game.
    """
    
    def __init__(self):
        """Initialize the collision manager."""
        self.collision_handlers = {}
        logging.info("CollisionManager initialized")
    
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
    
    def check_missile_collisions(self, player: Any, enemy: Any, 
                                callback: Callable = None) -> bool:
        """
        Check for collisions between missiles and an enemy.
        
        Args:
            player: Player entity containing missiles list
            enemy: Enemy entity to check for missile collisions
            callback: Optional callback function to call on collision
            
        Returns:
            bool: True if any missile collided with the enemy
        """
        if not player or not enemy or not hasattr(player, "missiles"):
            return False
            
        enemy_rect = self.get_entity_rect(enemy)
        collision_detected = False
        
        for missile in player.missiles[:]:  # Create a copy to avoid issues while removing
            missile_rect = missile.get_rect()
            if missile_rect.colliderect(enemy_rect):
                logging.info(f"Missile collision detected with enemy")
                player.missiles.remove(missile)
                collision_detected = True
                if callback:
                    callback()
                break
                
        return collision_detected
    
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
