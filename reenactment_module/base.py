from abc import ABCMeta, abstractmethod

import numpy as np

from interfaces.face import Face


class ReenactmentModel(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def modify(self, source: Face, target: Face, source_img: np.ndarray, target_img: np.ndarray) -> (bool, np.ndarray):
        pass
