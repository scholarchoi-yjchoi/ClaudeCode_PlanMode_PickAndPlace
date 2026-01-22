"""
Isaac Sim과 ROS 간의 좌표계 변환 유틸리티

Isaac Sim은 Y-up 좌표계 사용:
- X: 전방
- Y: 위쪽
- Z: 오른쪽

ROS (REP-103)는 Z-up 좌표계 사용:
- X: 전방
- Y: 왼쪽
- Z: 위쪽
"""

import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def isaac_to_ros_points(points: np.ndarray) -> np.ndarray:
    """
    Isaac Sim Y-up에서 ROS Z-up 좌표계로 여러 포인트 변환

    Args:
        points: Isaac Sim 좌표의 Nx3 포인트 배열

    Returns:
        ROS 좌표의 Nx3 포인트 배열
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"포인트는 Nx3 배열이어야 합니다. 현재 shape: {points.shape}")

    ros_points = np.zeros_like(points)
    ros_points[:, 0] = points[:, 0]  # X는 그대로 유지
    ros_points[:, 1] = -points[:, 2]  # Y = -Z (오른쪽→왼쪽)
    ros_points[:, 2] = points[:, 1]  # Z = Y (위쪽 축 변경)

    return ros_points


def isaac_to_ros_quaternion(
    quat: np.ndarray,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 90.0,
) -> np.ndarray:
    """
    Isaac Sim Y-up에서 ROS Z-up 좌표계로 쿼터니언 변환 (RPY 보정 포함)

    Args:
        quat: Isaac Sim의 쿼터니언 [x, y, z, w] 또는 [w, x, y, z]
        roll_deg: 롤 회전(도) (기본값: 0.0)
        pitch_deg: 피치 회전(도) (기본값: 0.0)
        yaw_deg: 요 회전(도) (기본값: 90.0)

    Returns:
        ROS 형식의 쿼터니언 [x, y, z, w]
    """
    if len(quat) != 4:
        raise ValueError(f"쿼터니언은 4차원이어야 합니다. 현재 shape: {quat.shape}")

    # 쿼터니언이 [w, x, y, z] 형식인 경우 (인덱스 0의 w가 1에 가까운 경우)
    if 0.7 < abs(quat[0]) <= 1.0:
        # [x, y, z, w]로 변환
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])

    # 라디안으로 변환
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    # 회전 쿼터니언 생성
    roll_quat = np.array([np.sin(roll_rad / 2), 0, 0, np.cos(roll_rad / 2)])
    pitch_quat = np.array([0, np.sin(pitch_rad / 2), 0, np.cos(pitch_rad / 2)])
    yaw_quat = np.array([0, 0, np.sin(yaw_rad / 2), np.cos(yaw_rad / 2)])

    # 회전 결합
    conversion_quat = quaternion_multiply(pitch_quat, roll_quat)
    conversion_quat = quaternion_multiply(yaw_quat, conversion_quat)

    q_ros = quaternion_multiply(conversion_quat, quat)

    return q_ros


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """쿼터니언 곱셈 [x, y, z, w] 형식"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def transform_camera_to_world(
    point_camera: np.ndarray,
    camera_position: np.ndarray,
    camera_orientation: np.ndarray,
    convert_to_ros: bool = True,
) -> np.ndarray:
    """
    카메라 프레임에서 월드 프레임으로 포인트 변환

    Args:
        point_camera: 카메라 프레임의 3D 포인트
        camera_position: 월드 프레임에서의 카메라 위치 (Isaac Sim 좌표)
        camera_orientation: 카메라 방향 쿼터니언 (Isaac Sim 좌표)
        convert_to_ros: True일 경우 시각화를 위해 ROS로 변환

    Returns:
        월드 프레임의 3D 포인트
    """
    if convert_to_ros:
        ros_camera_orientation = isaac_to_ros_quaternion(camera_orientation)
        rot_matrix = R.from_quat(ros_camera_orientation).as_matrix()
    else:
        rot_matrix = R.from_quat(camera_orientation).as_matrix()

    point_world_rotated = rot_matrix @ point_camera
    point_world_isaac = camera_position + point_world_rotated

    if convert_to_ros:
        point_world = isaac_to_ros_points(point_world_isaac.reshape(1, 3))[0]
    else:
        point_world = point_world_isaac

    return point_world
