from typing import List, Optional, Tuple

import cv2
import numpy as np

from interfaces.face import Face
from reenactment_module.base import ReenactmentModel


class SwappingBy68Landmarks(ReenactmentModel):
    name = 'Swapping68Landmarks'

    @staticmethod
    def _extract_index_nparray(nparray: np.ndarray) -> Optional[int]:
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    @staticmethod
    def _extract_triangle_info(face_info: Face, vertices_index: List[int]) -> (np.ndarray, np.ndarray, Tuple):
        tr1_p1 = face_info.facial_landmarks[vertices_index[0]]
        tr1_p2 = face_info.facial_landmarks[vertices_index[1]]
        tr1_p3 = face_info.facial_landmarks[vertices_index[2]]

        triangle = np.array([tr1_p1, tr1_p2, tr1_p3], dtype=np.int32)
        rect = cv2.boundingRect(triangle)
        (x, y, w, h) = rect
        cropped_mask = np.zeros((h, w), dtype=np.uint8)
        cropped_points = np.array([[tr1_p1[0] - x, tr1_p1[1] - y],
                                   [tr1_p2[0] - x, tr1_p2[1] - y],
                                   [tr1_p3[0] - x, tr1_p3[1] - y]], dtype=np.int32)
        cv2.fillConvexPoly(cropped_mask, cropped_points, 255)
        return cropped_mask, cropped_points, (x, y, w, h)

    def modify(self, source: Face, target: Face,
               source_img: np.ndarray, target_img: np.ndarray) -> (bool, np.ndarray):
        if source.facial_landmarks is None or target.facial_landmarks is None or source.frame_id == target.frame_id:
            return False, target_img

        # Visualize for thesis
        # x_min, y_min, x_max, y_max = target.bbox.to_rect().astype(np.int32)
        # for pt in target.facial_landmarks:
        #     cv2.circle(target_img, (pt[0], pt[1]), 2, (0, 0, 255), -1)
        # x_min, y_min, x_max, y_max = source.bbox.to_rect().astype(np.int32)
        # for pt in source.facial_landmarks:
        #     cv2.circle(source_img, (pt[0], pt[1]), 2, (0, 0, 255), -1)
        # cv2.imwrite('test_t_landmarks.png', cv2.cvtColor(target_img[y_min: y_max, x_min: x_max], cv2.COLOR_BGR2RGB))

        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_new_face = np.zeros_like(target_img)

        # Extract index of triangles
        target_convexhull = cv2.convexHull(target.facial_landmarks)
        target_convexhull_rect = cv2.boundingRect(target_convexhull)

        subdiv = cv2.Subdiv2D(target_convexhull_rect)
        subdiv.insert(target.facial_landmarks.tolist())
        target_triangles = subdiv.getTriangleList()
        target_triangles = np.array(target_triangles, dtype=np.int32)
        indexes_triangles = []
        for triangle in target_triangles:
            # Get the vertex of the triangle
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])

            index_pt1 = np.where((target.facial_landmarks == pt1).all(axis=1))
            index_pt1 = self._extract_index_nparray(index_pt1)
            index_pt2 = np.where((target.facial_landmarks == pt2).all(axis=1))
            index_pt2 = self._extract_index_nparray(index_pt2)
            index_pt3 = np.where((target.facial_landmarks == pt3).all(axis=1))
            index_pt3 = self._extract_index_nparray(index_pt3)

            # Save coordinates if the triangle exists and has 3 vertices
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                vertices = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(vertices)
        # Visualize for thesis
        # for triangle_index in indexes_triangles:
        #     pt1 = source.facial_landmarks[triangle_index[0]]
        #     pt2 = source.facial_landmarks[triangle_index[1]]
        #     pt3 = source.facial_landmarks[triangle_index[2]]
        #
        #     cv2.line(source_img, pt1, pt2, (255, 255, 255), 1)
        #     cv2.line(source_img, pt2, pt3, (255, 255, 255), 1)
        #     cv2.line(source_img, pt1, pt3, (255, 255, 255), 1)
        #
        #     pt1 = target.facial_landmarks[triangle_index[0]]
        #     pt2 = target.facial_landmarks[triangle_index[1]]
        #     pt3 = target.facial_landmarks[triangle_index[2]]
        #
        #     cv2.line(target_img, pt1, pt2, (255, 255, 255), 1)
        #     cv2.line(target_img, pt2, pt3, (255, 255, 255), 1)
        #     cv2.line(target_img, pt1, pt3, (255, 255, 255), 1)
        #
        # x_min, y_min, x_max, y_max = source.bbox.to_rect().astype(np.int32)
        # cv2.imwrite('test_s_triangles.png',
        #             cv2.cvtColor(source_img[y_min: y_max, x_min: x_max], cv2.COLOR_BGR2RGB))
        # x_min, y_min, x_max, y_max = target.bbox.to_rect().astype(np.int32)
        # cv2.imwrite('test_t_triangles.png',
        #             cv2.cvtColor(target_img[y_min: y_max, x_min: x_max], cv2.COLOR_BGR2RGB))
        # return
        # Main process
        for triangle_index in indexes_triangles:
            # 1.1 Triangulation of source face
            source_cropped_mask, source_cropped_points, source_triangle_rect = \
                self._extract_triangle_info(source, triangle_index)
            (x_s, y_s, w_s, h_s) = source_triangle_rect
            source_cropped_triangle = source_img[y_s: y_s + h_s, x_s: x_s + w_s]
            # 1.2 Triangulation of target face
            target_cropped_mask, target_cropped_points, target_triangle_rect = \
                self._extract_triangle_info(target, triangle_index)
            (x_t, y_t, w_t, h_t) = target_triangle_rect
            # target_cropped_triangle = target_img[y_t: y_t + h_t, x_t: x_t + w_t]
            # 2. Warp triangles
            source_cropped_points = np.float32(source_cropped_points)
            target_cropped_points = np.float32(target_cropped_points)
            m = cv2.getAffineTransform(source_cropped_points, target_cropped_points)
            warped_triangle = cv2.warpAffine(source_cropped_triangle, m, (w_t, h_t))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=target_cropped_mask)

            # 3. Reconstruct target face
            triangle_area = target_new_face[y_t: y_t + h_t, x_t: x_t + w_t]
            triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)

            # Let's create a mask to remove the lines between the triangles
            _, mask_triangles_designed = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            triangle_area = cv2.add(triangle_area, warped_triangle)
            target_new_face[y_t: y_t + h_t, x_t: x_t + w_t] = triangle_area

        # Visualize for thesis
        # cv2.imwrite('test_t_rendered.png', cv2.cvtColor(target_new_face, cv2.COLOR_BGR2RGB))
        # Face swapped (putting 1st face into 2nd face)
        target_mask = np.zeros_like(target_gray)
        target_head_mask = cv2.fillConvexPoly(target_mask, target_convexhull, 255)
        target_mask = cv2.bitwise_not(target_head_mask)

        target_head_noface = cv2.bitwise_and(target_img, target_img, mask=target_mask)
        # Visualize for thesis
        # x_min, y_min, x_max, y_max = target.bbox.to_rect().astype(np.int32)
        # cv2.imwrite('test_head_no_face.png',
        #             cv2.cvtColor(target_head_noface[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB))
        result = cv2.add(target_head_noface, target_new_face)
        # Visualize for thesis
        # x_min, y_min, x_max, y_max = target.bbox.to_rect().astype(np.int32)
        # cv2.imwrite('test_r.png',
        #             cv2.cvtColor(result[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB))
        (x, y, w, h) = target_convexhull_rect
        target_center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
        # Seamless clone
        result = cv2.seamlessClone(result, target_img, target_head_mask, target_center_face, cv2.MIXED_CLONE)
        # Visualize for thesis
        # x_min, y_min, x_max, y_max = target.bbox.to_rect().astype(np.int32)
        # cv2.imwrite('test_r_1.png',
        #             cv2.cvtColor(result[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB))
        return True, result
