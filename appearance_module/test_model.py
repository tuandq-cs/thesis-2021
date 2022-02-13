import numpy as np

from appearance_module.base import AppearanceModel
from interfaces.system_config import SystemConfig


class TestAppearanceModel(AppearanceModel):
    name = 'Test Appearance Model'

    def _load_model(self, config: SystemConfig) -> None:
        pass

    def compute_score(self, cropped_face: np.ndarray) -> (bool, float):
        return True, 1.0
