from typing import Any, Dict


class SystemResult:
    result_img: Any
    key_frame: Any
    source_faces: Dict[str, Any]
    target_faces: Dict[str, Any]
    source_landmarks: Dict[str, Any]
    target_landmarks: Dict[str, Any]
    rendered_faces: Dict[str, Any]

    def __init__(self):
        self.source_faces = dict()
        self.target_faces = dict()
        self.source_landmarks = dict()
        self.target_landmarks = dict()
        self.rendered_faces = dict()
