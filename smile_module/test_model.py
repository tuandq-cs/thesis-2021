import numpy as np

from smile_module.base import SmileModel


class TestSmileModel(SmileModel):
    name = 'Test Smile Model'

    def _load_model(self) -> None:
        pass

    def compute_score(self, cropped_face: np.ndarray) -> float:
        return 1.0
