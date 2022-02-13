from abc import ABCMeta, abstractmethod

import numpy as np


class SmileModel(metaclass=ABCMeta):
    name: str

    def __init__(self, model_path: str, device: str) -> None:
        self.device = device
        self.model_path = model_path
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def compute_score(self, cropped_face: np.ndarray) -> float:
        pass
