from typing import List, Any

import numpy as np

from interfaces.face import Face


class Frame:
    id: int
    img_size: Any
    frame_score: float
    total_score: float
    faces: List[Face]
    img: np.ndarray

    def __init__(self, frame_id: int, img: np.ndarray):
        self.id = frame_id
        self.img = img
        self.total_score = 0.0
        self.faces = list()

    @property
    def img_size(self) -> Any:
        return np.asarray(self.img.shape)[0:2]

    @property
    def frame_score(self):
        return 0.0 if len(self.faces) == 0 else self.total_score / len(self.faces)
