"""
세그멘테이션 처리를 위한 마스크 기반 인식 유틸리티
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def remove_outliers_iqr(depths: np.ndarray, iqr_multiplier: float = 1.5) -> np.ndarray:
    """IQR(사분위수 범위) 방법을 사용하여 이상치 제거"""
    if len(depths) == 0:
        return depths

    q1 = np.percentile(depths, 25)
    q3 = np.percentile(depths, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    filtered = depths[(depths >= lower_bound) & (depths <= upper_bound)]

    if len(filtered) == 0:
        return depths

    return filtered


def get_masked_object_position(
    depth_data: np.ndarray,
    mask: np.ndarray,
    camera_intrinsics: Tuple[float, float, float, float],
    min_depth: float = 0.1,
    max_depth: float = 2.0,
    iqr_multiplier: float = 1.5,
    use_morphology: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    세그멘테이션 마스크를 사용하여 3D 위치 추출

    Args:
        depth_data: 깊이 이미지 배열
        mask: 이진 세그멘테이션 마스크
        camera_intrinsics: (fx, fy, cx, cy)
        min_depth: 최소 유효 깊이
        max_depth: 최대 유효 깊이
        iqr_multiplier: 이상치 제거를 위한 IQR 승수
        use_morphology: 모폴로지 연산 적용 여부

    Returns:
        position: 카메라 프레임의 3D 위치
        info: 처리 정보가 포함된 딕셔너리
    """
    fx, fy, cx, cy = camera_intrinsics

    # 마스크 크기 조정
    if mask.shape != depth_data.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth_data.shape[1], depth_data.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # 이진 마스크로 변환
    if mask.dtype != bool:
        mask = mask > 0.5

    # 모폴로지 연산 적용
    if use_morphology and np.any(mask):
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        if np.sum(mask_eroded) > 20:
            mask = mask_eroded.astype(bool)

    # 마스크 내의 깊이 값 추출
    masked_depths = depth_data[mask]

    # 유효한 깊이 필터링
    valid_mask = (
        (masked_depths > min_depth)
        & (masked_depths < max_depth)
        & np.isfinite(masked_depths)
    )
    valid_depths = masked_depths[valid_mask]

    if len(valid_depths) == 0:
        return np.array([0.0, 0.0, 0.0]), {
            "method": "mask_failed",
            "num_valid_depths": 0,
            "mask_pixels": int(np.sum(mask)),
        }

    # IQR을 사용하여 이상치 제거
    filtered_depths = remove_outliers_iqr(valid_depths, iqr_multiplier)

    if len(filtered_depths) == 0:
        filtered_depths = valid_depths

    # 견고한 깊이 계산
    robust_depth = np.median(filtered_depths)

    # 마스크 중심 계산
    y_indices, x_indices = np.where(mask)
    cx_mask = np.mean(x_indices)
    cy_mask = np.mean(y_indices)

    # 카메라 프레임의 3D 위치로 변환
    x_cam = (cx_mask - cx) * robust_depth / fx
    y_cam = (cy_mask - cy) * robust_depth / fy
    z_cam = robust_depth

    position = np.array([-z_cam, y_cam, x_cam])

    info = {
        "method": "segmentation_mask",
        "mask_pixels": int(np.sum(mask)),
        "num_valid_depths": len(valid_depths),
        "num_filtered_depths": len(filtered_depths),
        "depth_median": float(robust_depth),
        "depth_mean": float(np.mean(filtered_depths)) if len(filtered_depths) > 0 else 0.0,
        "depth_std": float(np.std(filtered_depths)) if len(filtered_depths) > 0 else 0.0,
        "centroid_x": float(cx_mask),
        "centroid_y": float(cy_mask),
        "outliers_removed": len(valid_depths) - len(filtered_depths),
    }

    return position, info


def get_masked_pointcloud(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float = 0.1,
    max_depth: float = 2.0,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    마스크된 픽셀에 대한 RGB 색상이 포함된 3D 포인트 클라우드 추출

    Args:
        depth_image: 깊이 이미지 배열
        rgb_image: RGB 이미지 배열 (H, W, 3)
        mask: 이진 세그멘테이션 마스크
        fx, fy, cx, cy: 카메라 내부 파라미터
        min_depth, max_depth: 유효한 깊이 범위
        config: 선택적 구성

    Returns:
        points: 카메라 프레임의 3D 포인트 Nx3 배열
        colors: RGB 색상 Nx3 배열
    """
    height, width = depth_image.shape

    # 마스크 크기 조정
    if mask.shape != depth_image.shape:
        mask = cv2.resize(
            mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST
        )

    # 이진 마스크로 변환
    binary_mask = (mask > 0.5).astype(bool)

    # 구성에서 깊이 범위 가져오기
    if config:
        min_depth = config.get("perception.depth.min_depth", min_depth)
        max_depth = config.get("perception.depth.max_depth", max_depth)

    # 메시 그리드 생성
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # 유효 마스크 생성
    valid_mask = (
        binary_mask
        & (depth_image > min_depth)
        & (depth_image < max_depth)
        & np.isfinite(depth_image)
    )

    if not np.any(valid_mask):
        logger.warning("마스크에 유효한 깊이 값이 없음")
        return np.array([]), np.array([])

    # 유효한 픽셀 좌표와 깊이 가져오기
    valid_xx = xx[valid_mask]
    valid_yy = yy[valid_mask]
    valid_depths = depth_image[valid_mask]

    # RGB 색상 가져오기
    valid_colors = rgb_image[valid_yy, valid_xx, :3]

    # 카메라 프레임의 3D 포인트로 변환
    points_x = (valid_xx - cx) * valid_depths / fx
    points_y = (valid_yy - cy) * valid_depths / fy
    points_z = valid_depths

    # 좌표 보정
    points_x_corrected = -points_x
    points_y_corrected = points_y
    points_z_corrected = -points_z

    points = np.stack([points_x_corrected, points_y_corrected, points_z_corrected], axis=-1)

    return points, valid_colors
