from typing import Tuple, List

import numpy as np


class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def to_tuple(self) -> Tuple:
        return tuple((self.x, self.y, self.w, self.h))

    def to_list(self) -> List:
        return [self.x, self.y, self.w, self.h]

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h])

    def to_rect(self) -> np.ndarray:
        x_min, y_min = self.x - self.w / 2, self.y - self.h / 2
        x_max, y_max = self.x + self.w / 2, self.y + self.h / 2
        return np.array([x_min, y_min, x_max, y_max])
