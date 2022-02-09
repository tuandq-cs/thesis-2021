import os
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from decord import VideoReader
from decord import cpu

from appearance_module.base import AppearanceModel
from detection_module.detector.base import Detector
from interfaces.face import Face
from interfaces.frame import Frame
from interfaces.system_config import SystemConfig
from interfaces.system_result import SystemResult
from landmark_module.base import LandmarkModel
from logger import Logger
from reenactment_module.base import ReenactmentModel
from smile_module.base import SmileModel
from tracking_module.base import TrackingModel
from utils.common import mkdir
from utils.face import compute_eye_score, COMPARE_SCORE_WEIGHTS


class SystemWrapper:
    def __init__(self, config: SystemConfig,
                 tracking_model: TrackingModel,
                 detection_model: Detector,
                 appearance_model: AppearanceModel,
                 smile_model: SmileModel,
                 landmark_model: LandmarkModel,
                 reenactment_model: ReenactmentModel
                 ):
        self.config = config
        self._tracking_model = tracking_model
        self._detection_model = detection_model
        self._appearance_model = appearance_model
        self._smile_model = smile_model
        self._landmark_model = landmark_model
        self._reenactment_model = reenactment_model

    @staticmethod
    def _compute_compare_score(face_score: float, similarity_score: float) -> float:
        w_face_score, w_similarity_score = \
            COMPARE_SCORE_WEIGHTS['face_score'], COMPARE_SCORE_WEIGHTS['similarity_score']
        compare_score = (
                                w_face_score * face_score + w_similarity_score * similarity_score
                        ) / (w_face_score + w_similarity_score)
        return compare_score

    @staticmethod
    def _get_best_face(face_ref: Face, list_faces: List[Face]) -> Face:
        if face_ref.facial_landmarks is None:
            max_index = np.argmax([face.face_score for face in list_faces])
            return list_faces[max_index]
        face_ref_mask = np.zeros(face_ref.frame_size)
        cv2.fillConvexPoly(face_ref_mask, cv2.convexHull(face_ref.facial_landmarks), 255)
        best_face_info = face_ref
        best_compare_score = SystemWrapper._compute_compare_score(face_score=best_face_info.face_score,
                                                                  similarity_score=1.0)
        for face_info in list_faces:
            if face_info.facial_landmarks is None:
                continue
            face_info_mask = np.zeros(face_info.frame_size)
            cv2.fillConvexPoly(face_info_mask, cv2.convexHull(face_info.facial_landmarks), 255)
            # Compute similarity between face reference and other faces
            dice_similarity_score = np.sum(face_ref_mask[face_info_mask == 255]) * 2.0 / (np.sum(face_ref_mask) +
                                                                                          np.sum(face_info_mask))
            compare_score = SystemWrapper._compute_compare_score(face_score=face_info.face_score,
                                                                 similarity_score=dice_similarity_score)
            if compare_score > best_compare_score:
                best_face_info = face_info
        return best_face_info

    def _compute_scores(self, frame_info: Frame, tracked_faces: List[Face], dict_faces_info: Dict) -> None:
        frame_info.faces = tracked_faces
        for face_info in tracked_faces:
            face_info.frame_id = frame_info.id
            face_info.frame_size = frame_info.img_size
            # Assign value to dict
            dict_faces_info[str(face_info.id)] = dict_faces_info.get(str(face_info.id), []) + [face_info]
            # 1. Get bounding box information -> Convert to rectangle
            x_min, y_min, x_max, y_max = face_info.bbox.to_rect()
            # 2. Add margin to bounding box
            margin = self.config.margin
            x_min = np.maximum(x_min - margin, 0)
            y_min = np.maximum(y_min - margin, 0)
            x_max = np.minimum(x_max + margin, frame_info.img_size[1])
            y_max = np.minimum(y_max + margin, frame_info.img_size[0])
            rect = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)  # New rectangle
            # 3. Crop face
            cropped_face = frame_info.img[rect[1]:rect[3], rect[0]:rect[2], :]
            cropped_face_copy = cropped_face.copy()
            face_info.face_img = cropped_face
            # 4. Compute appearance score
            eye_visible, appearance_score = self._appearance_model.compute_score(cropped_face=cropped_face_copy)
            # 5. Detect 68 facial landmark => if eye_visible, compute eye score
            landmarks = self._landmark_model.detect(face_info=face_info, frame=frame_info.img)
            eye_score = compute_eye_score(landmarks, self.config.ear_threshold)
            # 6. Compute smile score
            smile_score = self._smile_model.compute_score(cropped_face=cropped_face_copy)
            face_info.facial_landmarks = landmarks
            face_info.appearance_score = appearance_score
            face_info.eye_score = eye_score
            face_info.smile_score = smile_score
            face_info.compute_face_score()
            # 7. Add to frame score
            frame_info.total_score += face_info.face_score

    def get_result(self, video_path) -> SystemResult:
        np.random.seed(0)
        start_time = time.time()
        dir_name = time.strftime('%Y_%m_%d_%H_%M', time.localtime(start_time))
        inference_dir = os.path.join(self.config.inference_root_path, dir_name)
        mkdir(inference_dir)

        video_name = Path(video_path).stem
        logger = Logger(log_file_root_path=inference_dir, log_name=video_name, module_name='MOT')

        video_reader = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(video_reader)

        # Log info
        logger.info("Detection module name: {}".format(self._detection_model.name))
        logger.info("Tracking module name: {}".format(self._tracking_model.name))
        logger.info("Appearance module name: {}".format(self._appearance_model.name))
        logger.info("Smile module name: {}".format(self._smile_model.name))
        logger.info("Landmark module name: {}".format(self._landmark_model.name))
        logger.info("Reenactment module name: {}".format(self._reenactment_model.name))
        logger.info("Total {} frames:".format(total_frames))

        # Extract all information
        counter = 0
        best_frame_info = None
        list_frames_info = []
        dict_faces_info = dict()
        while counter < total_frames:
            img = video_reader[counter].asnumpy()
            if counter % self.config.frame_interval != 0:
                counter += 1
                continue
            # Init frame info
            frame_info = Frame(frame_id=counter, img=img)
            logger.info("Frame {}:".format(counter))
            # 1. Detect face
            detection_start_time = time.time()
            faces: List[Face] = self._detection_model.detect_face(frame=img)
            logger.info("Detect {} face(s) cost time: {}s".format(len(faces),
                                                                  round(time.time() - detection_start_time, 3)))
            # 2. Track face
            tracking_start_time = time.time()
            tracked_faces: List[Face] = self._tracking_model.update(list_faces=faces, img_size=frame_info.img_size,
                                                                    predict_num=self.config.frame_interval)
            logger.info("Track face(s) cost time: {}s".format(round(time.time() - tracking_start_time), 3))
            # 3. Compute face scores and frame score
            score_computation_time = time.time()
            self._compute_scores(frame_info=frame_info, tracked_faces=tracked_faces, dict_faces_info=dict_faces_info)
            logger.info("Compute face score(s) cost time: {}s".format(round(time.time() - score_computation_time), 3))

            # 4. Check whether it should be best frame
            if best_frame_info is None or frame_info.total_score > best_frame_info.total_score:
                best_frame_info = frame_info

            list_frames_info.append(frame_info)
            counter += 1

        # Face reenactment
        logger.info("Key frame at frame id: {}".format(best_frame_info.id))
        result = SystemResult()
        result.key_frame = best_frame_info.img
        rendered_img = best_frame_info.img.copy()
        reenactment_time = time.time()
        for face_info in best_frame_info.faces:
            # Pick the best moment frame for that face
            best_face_info: Face = self._get_best_face(face_ref=face_info,
                                                       list_faces=dict_faces_info.get(str(face_info.id)))
            source_img = list_frames_info[best_face_info.frame_id].img
            is_rendered, rendered_img = self._reenactment_model.modify(source=best_face_info,
                                                                       target=face_info,
                                                                       source_img=source_img,
                                                                       target_img=rendered_img)
            if not is_rendered:
                continue
            # Get face was rendered
            x_min, y_min, x_max, y_max = face_info.bbox.to_rect().astype(np.int32)
            rendered_face = rendered_img[y_min: y_max, x_min: x_max]
            # Draw landmarks
            source_landmarks = source_img.copy()
            target_landmarks = best_frame_info.img.copy()
            for pt in best_face_info.facial_landmarks:
                cv2.circle(source_landmarks, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            for pt in face_info.facial_landmarks:
                cv2.circle(target_landmarks, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            target_landmarks = target_landmarks[y_min: y_max, x_min: x_max]
            x_min, y_min, x_max, y_max = best_face_info.bbox.to_rect().astype(np.int32)
            source_landmarks = source_landmarks[y_min: y_max, x_min: x_max]
            # Write result
            result.source_faces[str(face_info.id)] = best_face_info.face_img
            result.target_faces[str(face_info.id)] = face_info.face_img
            result.source_landmarks[str(face_info.id)] = source_landmarks
            result.target_landmarks[str(face_info.id)] = target_landmarks
            result.rendered_faces[str(face_info.id)] = rendered_face

        # Final image
        result.result_img = rendered_img
        logger.info("Reenactment cost time: {}s".format(round(time.time() - reenactment_time), 3))
        # Save result
        cv2.imwrite(os.path.join(inference_dir, 'key_frame.png'), cv2.cvtColor(result.key_frame, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(inference_dir, 'result.png'), cv2.cvtColor(result.result_img, cv2.COLOR_BGR2RGB))
        for face_id in result.rendered_faces:
            face_dir = os.path.join(inference_dir, face_id)
            mkdir(face_dir)
            cv2.imwrite(os.path.join(face_dir, 'source.png'),
                        cv2.cvtColor(result.source_faces[face_id], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(face_dir, 'target.png'),
                        cv2.cvtColor(result.target_faces[face_id], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(face_dir, 'source_landmarks.png'),
                        cv2.cvtColor(result.source_landmarks[face_id], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(face_dir, 'target_landmarks.png'),
                        cv2.cvtColor(result.target_landmarks[face_id], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(face_dir, 'rendered.png'),
                        cv2.cvtColor(result.rendered_faces[face_id], cv2.COLOR_BGR2RGB))

        logger.info("Total cost time: {}s".format(round(time.time() - start_time), 3))
        self._tracking_model.reset()
        return result
