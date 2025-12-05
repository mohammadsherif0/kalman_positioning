"""Landmark manager for loading and querying landmark positions."""

from __future__ import annotations

import csv
import math
from typing import Dict, List, Tuple


class LandmarkManager:
    """Utility class to load and query landmark positions."""

    def __init__(self) -> None:
        self._landmarks: Dict[int, Tuple[float, float]] = {}

    def load_from_csv(self, csv_path: str) -> bool:
        """Load landmarks from a CSV file with format: id,x,y."""
        self._landmarks.clear()
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                for line_num, row in enumerate(reader, start=1):
                    if not row or (row[0].strip().startswith("#")):
                        continue
                    if len(row) < 3:
                        continue
                    try:
                        landmark_id = int(row[0].strip())
                        x = float(row[1].strip())
                        y = float(row[2].strip())
                        self._landmarks[landmark_id] = (x, y)
                    except ValueError:
                        # Skip malformed lines but continue loading others
                        continue
        except OSError:
            return False

        return len(self._landmarks) > 0

    def get_landmarks(self) -> Dict[int, Tuple[float, float]]:
        return self._landmarks

    def get_landmark(self, landmark_id: int) -> Tuple[float, float]:
        return self._landmarks.get(landmark_id, (0.0, 0.0))

    def has_landmark(self, landmark_id: int) -> bool:
        return landmark_id in self._landmarks

    def get_num_landmarks(self) -> int:
        return len(self._landmarks)

    def get_landmarks_in_radius(self, x: float, y: float, radius: float) -> List[int]:
        result: List[int] = []
        for lid, (lx, ly) in self._landmarks.items():
            if self.distance(x, y, lx, ly) <= radius:
                result.append(lid)
        return result

    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return math.hypot(dx, dy)

