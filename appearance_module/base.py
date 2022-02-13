from abc import abstractmethod, ABCMeta

import numpy as np

from interfaces.system_config import SystemConfig


class AppearanceModel(metaclass=ABCMeta):
    name: str

    def __init__(self, model_path: str, config: SystemConfig, vis_threshold: float = 0.5) -> None:
        self.vis_threshold = vis_threshold
        self.model_path = model_path
        self._load_model(config)

    @abstractmethod
    def _load_model(self, config: SystemConfig) -> None:
        pass

    @abstractmethod
    def compute_score(self, cropped_face: np.ndarray) -> (bool, float):
        pass
