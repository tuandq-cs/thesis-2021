from typing import List, Any

import numpy as np

from interfaces.face import Face
from interfaces.tracker import KalmanBoxTracker
from tracking_module.base import TrackingModel
from utils.common import iou_batch


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


class Sort(TrackingModel):
    name = 'SORT'
    trackers: List[KalmanBoxTracker]

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.25):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def _associate_detections_to_trackers(self, detections, trackers):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, list_faces: List[Face], img_size: Any, predict_num: int) -> List[Face]:
        """
        Params: list_faces - a list of detected face which contains facial information. Extract information of
        list_faces to variable dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],
        ...] Requires: this method must be called once for each frame even with empty detections. Returns a
        similar array, where the last column is the object ID.

            NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        self.frame_count += 1
        # Extract detections info
        dets = []
        for face in list_faces:
            dets.append(face.to_det_info())
        dets = np.array(dets)

        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

        # Update matched trackers with assigned detections
        for m in matched:
            trk = self.trackers[m[1]]
            trk.update(dets[m[0], :])
            # assign id to face instance
            face_info = list_faces[m[0]]
            face_info.id = trk.id
            trk.last_face = face_info

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            # assign id to face instance
            face_info = list_faces[i]
            face_info.id = trk.id
            trk.last_face = face_info

        tracked_faces: List[Face] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # if not dets:  # dets == []
            #     trk.update([])
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                tracked_faces.append(trk.last_face)
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.max_age or trk.predict_num >= predict_num or \
                    d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]:
                self.trackers.pop(i)
        return tracked_faces

    def reset(self):
        self.__init__()
