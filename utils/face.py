from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    # ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("right_jaw", (9, 17)),
    ("left_jaw", (0, 8)),
    ("chin", (8, 9)),
])

FACIAL_LANDMARKS_21_IDXS = OrderedDict([
    ("mouth", (17, 20)),
    # ("inner_mouth", (60, 68)),
    ("right_eyebrow", (3, 6)),
    ("left_eyebrow", (0, 3)),
    ("right_eye", (9, 12)),
    ("left_eye", (6, 9)),
    ("nose", (13, 16)),
    ("right_jaw", (16, 17)),
    ("left_jaw", (12, 13)),
    ("chin", (20, 21)),
])

FACE_SCORE_WEIGHTS = OrderedDict([
    ("appearance_score", 1),
    ("smile_score", 1),
    ("eye_score", 1),
])

COMPARE_SCORE_WEIGHTS = OrderedDict([
    ("face_score", 0.7),
    ("similarity_score", 0.3),
])


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)
    # return the eye aspect ratio
    return ear


def compute_eye_score(facial_landmarks: np.ndarray, ear_threshold: float) -> float:
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (l_start, l_end) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (r_start, r_end) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    left_eye = facial_landmarks[l_start:l_end]
    right_eye = facial_landmarks[r_start:r_end]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    if left_ear < ear_threshold or right_ear < ear_threshold:
        return 0
    return (left_ear + right_ear) / 2.
