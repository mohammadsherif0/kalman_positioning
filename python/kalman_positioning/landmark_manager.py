"""
STUDENT ASSIGNMENT: Landmark Manager Implementation

This class manages landmark positions loaded from a CSV file.
Students should implement the methods for loading, querying, and 
managing landmark data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class LandmarkManager:
    """Manages landmark positions for robot localization"""
    
    def __init__(self):
        """Initialize empty landmark manager"""
        self.landmarks: Dict[int, Tuple[float, float]] = {}
    
    def load_from_csv(self, csv_path: str) -> bool:
        """
        Load landmarks from CSV file
        
        STUDENT TODO:
        1. Open the CSV file at csv_path
        2. Parse each line as: id,x,y
        3. Handle comments (lines starting with #)
        4. Store landmark positions in the landmarks_ dict
        5. Return True if successful, False otherwise
        
        Args:
            csv_path: Path to CSV file with format: id,x,y
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.landmarks.clear()
            
            with open(csv_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse CSV line
                    try:
                        parts = line.split(',')
                        if len(parts) != 3:
                            print(f"Warning: Line {line_num} has invalid format: {line}")
                            continue
                        
                        landmark_id = int(parts[0].strip())
                        x = float(parts[1].strip())
                        y = float(parts[2].strip())
                        
                        self.landmarks[landmark_id] = (x, y)
                        
                    except ValueError as e:
                        print(f"Warning: Line {line_num} parsing error: {e}")
                        continue
            
            if not self.landmarks:
                print(f"Warning: No landmarks loaded from {csv_path}")
                return False
            
            print(f"Loaded {len(self.landmarks)} landmarks from {csv_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found: {csv_path}")
            return False
        except Exception as e:
            print(f"Error loading landmarks: {e}")
            return False
    
    def get_landmarks(self) -> Dict[int, Tuple[float, float]]:
        """Get all landmarks"""
        return self.landmarks
    
    def get_landmark(self, landmark_id: int) -> Optional[Tuple[float, float]]:
        """
        Get landmark position by ID
        
        Args:
            landmark_id: Landmark ID
            
        Returns:
            (x, y) position or None if not found
        """
        return self.landmarks.get(landmark_id)
    
    def has_landmark(self, landmark_id: int) -> bool:
        """Check if landmark exists"""
        return landmark_id in self.landmarks
    
    def get_num_landmarks(self) -> int:
        """Get number of landmarks"""
        return len(self.landmarks)
    
    def get_landmarks_in_radius(self, x: float, y: float, radius: float) -> List[int]:
        """
        Get landmarks within a certain radius of a point
        
        STUDENT TODO:
        1. Iterate through all landmarks
        2. Calculate distance from (x, y) to each landmark
        3. Add landmarks within radius to result list
        4. Return the list of landmark IDs
        
        Args:
            x, y: Center point coordinates
            radius: Search radius in meters
            
        Returns:
            List of landmark IDs within radius
        """
        result = []
        for landmark_id, (lx, ly) in self.landmarks.items():
            if self.distance(x, y, lx, ly) <= radius:
                result.append(landmark_id)
        return result
    
    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        dx = x2 - x1
        dy = y2 - y1
        return np.sqrt(dx * dx + dy * dy)

