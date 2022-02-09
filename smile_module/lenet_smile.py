import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from smile_module.base import SmileModel


class LenetSmile(SmileModel):
    name = 'LenetSmile'

    def _load_model(self) -> None:
        self.model = load_model(self.model_path)

    @staticmethod
    def _preprocess(cropped_face: np.ndarray) -> np.ndarray:
        def extract_roi(img, size):
            roi = cv2.resize(img, (size, size))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            return np.expand_dims(roi, axis=0)

        gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        region_of_interest = extract_roi(gray, size=28)
        return region_of_interest

    def _get_output(self, inp: np.ndarray) -> float:
        (_, smile) = self.model.predict(inp)[0]
        return smile

    def compute_score(self, cropped_face: np.ndarray) -> float:
        inp = self._preprocess(cropped_face)
        return self._get_output(inp)
