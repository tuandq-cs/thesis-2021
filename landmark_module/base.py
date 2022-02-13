from abc import ABCMeta, abstractmethod

import numpy as np

from interfaces.face import Face
from interfaces.system_config import SystemConfig


class LandmarkModel(metaclass=ABCMeta):
    name: str

    def __init__(self, model_path: str, config: SystemConfig) -> None:
        self.model_path = model_path
        self._load_model(config)

    @abstractmethod
    def _load_model(self, config: SystemConfig) -> None:
        pass

    @abstractmethod
    def detect(self, face_info: Face, frame: np.ndarray) -> np.ndarray:
        pass
