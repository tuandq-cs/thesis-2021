from abc import abstractmethod, ABCMeta
from typing import List

import numpy as np

from interfaces.face import Face

BASE_MARGIN = 10


class Detector(metaclass=ABCMeta):
    name: str
    conf_threshold: float
    margin: int

    def __init__(self, model_path: str, conf_threshold: float = 0.85, margin: int = 10) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.margin = margin
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def detect_face(self, frame: np.ndarray) -> List[Face]:
        pass
