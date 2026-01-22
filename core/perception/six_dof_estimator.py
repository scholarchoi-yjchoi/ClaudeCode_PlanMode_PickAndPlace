"""
6DOF 포즈 추정기, PCA 기반
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class SixDOFEstimator:
    """PCA 기반 6DOF 포즈 추정"""

    def __init__(self, min_points: int = 10):
        self.min_points = min_points

    def estimate_6dof_with_confidence(
        self,
        points: np.ndarray,
        class_name: Optional[str] = None,
        use_gravity_constraint: bool = True,
        camera_orientation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        PCA 기반 6DOF 추정

        Args:
            points: 3D 포인트의 Nx3 배열
            class_name: 객체 클래스 이름
            use_gravity_constraint: 중력 정렬 적용
            camera_orientation: 월드 프레임에서의 카메라 방향 쿼터니언

        Returns:
            position: 3D 위치
            orientation: 3x3 회전 행렬
            info: 신뢰도 점수 및 세부 정보가 포함된 딕셔너리
        """
        if points is None or len(points) < self.min_points:
            position = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
            orientation = np.eye(3)
            return (
                position,
                orientation,
                {
                    "status": "insufficient_points",
                    "num_points": len(points) if points is not None else 0,
                },
            )

        try:
            # 중심 위치 계산
            position = np.mean(points, axis=0)
            centered_points = points - position

            # 공분산 행렬 계산
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # 고유값으로 정렬 (내림차순)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # 중력 제약 적용
            if use_gravity_constraint:
                orientation = self._apply_gravity_constraint(
                    eigenvectors, eigenvalues, camera_orientation
                )
            else:
                orientation = eigenvectors

            # 오른손 좌표계 보장
            if np.linalg.det(orientation) < 0:
                orientation[:, 2] *= -1

            # 객체 범위 계산
            projections = np.dot(centered_points, orientation)
            extents = np.ptp(projections, axis=0)

            info = {
                "status": "success",
                "num_points": len(points),
                "eigenvalues": eigenvalues.tolist(),
                "extents": extents.tolist(),
                "class_name": class_name,
                "gravity_constrained": use_gravity_constraint,
            }

            return position, orientation, info

        except Exception as e:
            logger.error(f"6DOF 추정 오류: {e}")
            position = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
            orientation = np.eye(3)
            return (
                position,
                orientation,
                {
                    "status": "error",
                    "error": str(e),
                },
            )

    def _apply_gravity_constraint(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        camera_orientation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """테이블 위 객체에 대한 중력 제약 적용

        테이블 위에 놓인 객체의 경우:
        - 객체의 Z축이 월드 상향(Isaac Sim의 Y)과 정렬되어야 함
        - 객체의 XY 평면이 테이블과 평행해야 함
        """
        # 월드 상향 방향 (Isaac Sim은 Y-up)
        world_up = np.array([0, 1, 0])

        # 카메라 프레임의 eigenvectors를 월드 프레임으로 변환
        if camera_orientation is not None:
            cam_rot = R.from_quat(camera_orientation)
            cam_rot_matrix = cam_rot.as_matrix()
            eigenvectors_world = cam_rot_matrix @ eigenvectors
        else:
            eigenvectors_world = eigenvectors

        # 월드 상향과 가장 정렬된 축 찾기
        dots = [np.abs(np.dot(eigenvectors_world[:, i], world_up)) for i in range(3)]
        vertical_axis_idx = np.argmax(dots)

        # 수직 축을 Z축으로 설정
        z_axis = eigenvectors_world[:, vertical_axis_idx].copy()

        # Z축이 위를 향하도록 보장
        if np.dot(z_axis, world_up) < 0:
            z_axis *= -1

        # 나머지 축 중에서 X, Y축 선택
        other_indices = [i for i in range(3) if i != vertical_axis_idx]

        # 더 큰 고유값을 가진 축을 X축으로 사용
        if eigenvalues[other_indices[0]] > eigenvalues[other_indices[1]]:
            x_candidate = eigenvectors_world[:, other_indices[0]]
        else:
            x_candidate = eigenvectors_world[:, other_indices[1]]

        # X축을 수평면에 투영
        x_axis = x_candidate - np.dot(x_candidate, world_up) * world_up
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)

        # Y축을 외적으로 계산
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

        # X축 재계산
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)

        orientation_world = np.column_stack([x_axis, y_axis, z_axis])

        # 카메라 프레임으로 변환
        if camera_orientation is not None:
            orientation_camera = cam_rot_matrix.T @ orientation_world
            return orientation_camera

        return orientation_world

    def project_3d_axes(
        self,
        center_3d: np.ndarray,
        orientation: np.ndarray,
        camera_intrinsics: tuple,
        axis_length: float = 0.05,
    ) -> Optional[Dict]:
        """시각화를 위해 3D 좌표 축을 2D 이미지 평면에 투영"""
        fx, fy, cx, cy = camera_intrinsics

        # 3D 축 끝점 생성
        axes_3d = {
            "x": center_3d + orientation[:, 0] * axis_length,
            "y": center_3d + orientation[:, 1] * axis_length,
            "z": center_3d + orientation[:, 2] * axis_length,
        }

        # 중심을 2D로 투영
        if abs(center_3d[2]) < 0.001:
            return None

        center_2d = (
            int(fx * center_3d[0] / center_3d[2] + cx),
            int(fy * center_3d[1] / center_3d[2] + cy),
        )

        # 축 끝점을 2D로 투영
        axes_2d = {}
        for axis_name, point_3d in axes_3d.items():
            if abs(point_3d[2]) < 0.001:
                continue
            x_2d = int(fx * point_3d[0] / point_3d[2] + cx)
            y_2d = int(fy * point_3d[1] / point_3d[2] + cy)
            axes_2d[axis_name] = (x_2d, y_2d)

        return {"center": center_2d, "axes": axes_2d}
