from typing import Tuple, Optional

import numpy as np

from interfaces.bounding_box import BoundingBox
from utils.face import FACE_SCORE_WEIGHTS


class Face:
    id: int
    frame_id: int
    frame_size: Tuple[int]
    bbox: BoundingBox
    face_img: np.ndarray  # Cropped face
    conf_score: float
    appearance_score: float
    smile_score: float
    eye_score: float
    face_score: float
    _facial_landmarks: Optional[np.ndarray]
    facial_landmarks: Optional[np.ndarray]
    _face_score: float

    def __init__(self, bbox: BoundingBox, conf_score: float) -> None:
        self.bbox = bbox
        self.conf_score = conf_score
        self.appearance_score = 0.0
        self.smile_score = 0.0
        self.eye_score = 0.0
        self._face_score = 0.0
        self._facial_landmarks = None

    @property
    def facial_landmarks(self) -> Optional[np.ndarray]:
        return self._facial_landmarks

    @facial_landmarks.setter
    def facial_landmarks(self, landmarks: np.ndarray) -> None:
        self._facial_landmarks = landmarks

    @property
    def face_score(self) -> float:
        return self._face_score

    def compute_face_score(self) -> None:
        w_appearance_score, w_smile_score, w_eye_score = \
            FACE_SCORE_WEIGHTS['appearance_score'], FACE_SCORE_WEIGHTS['smile_score'], FACE_SCORE_WEIGHTS['eye_score']

        face_score = (
                             w_appearance_score * self.appearance_score +
                             w_smile_score * self.smile_score +
                             w_eye_score * self.eye_score
                     ) / (w_appearance_score + w_smile_score + w_eye_score)
        self._face_score = face_score

    def to_det_info(self) -> np.ndarray:
        det_info = np.append(self.bbox.to_rect(), self.conf_score)
        return det_info
