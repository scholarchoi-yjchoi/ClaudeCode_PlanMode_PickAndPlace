"""
YOLO 객체 감지 및 6DOF 포즈 추정을 위한 감지 파이프라인
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging

from .mask_processor import get_masked_object_position, get_masked_pointcloud
from .six_dof_estimator import SixDOFEstimator

logger = logging.getLogger(__name__)

# YOLO는 선택적 의존성
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO를 사용할 수 없습니다. pip install ultralytics로 설치하세요.")


class YOLODetector:
    """YOLO 기반 객체 검출기"""

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        confidence_threshold: float = 0.5,
        device: str = "cuda:0",
    ):
        """
        Args:
            model_path: YOLO 모델 경로 또는 이름 (예: yolov8n-seg.pt)
            confidence_threshold: 검출 신뢰도 임계값
            device: 추론 장치 (cuda:0, cpu 등)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.class_names = {}

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.model.to(device)
                self.class_names = self.model.names
                logger.info(f"YOLO 모델 로드 완료: {model_path}")
            except Exception as e:
                logger.error(f"YOLO 모델 로드 실패: {e}")
        else:
            logger.warning("YOLO 사용 불가능")

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def detect(
        self,
        rgb_image: np.ndarray,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        RGB 이미지에서 객체 검출 수행

        Args:
            rgb_image: RGB 이미지 (H, W, 3)
            verbose: 상세 출력 여부

        Returns:
            검출 결과 리스트 (각 항목: bbox, confidence, class, class_name, mask)
        """
        if not self.is_available:
            return []

        detections = []

        try:
            # BGR로 변환 (YOLO는 BGR 입력)
            if rgb_image.shape[2] == 4:  # RGBA
                rgb_image = rgb_image[:, :, :3]
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            results = self.model(bgr_image, verbose=verbose, conf=self.confidence_threshold)

            if len(results) == 0 or len(results[0].boxes) == 0:
                return detections

            result = results[0]
            has_masks = hasattr(result, "masks") and result.masks is not None

            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")

                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls_id,
                    "class_name": class_name,
                    "mask": None,
                }

                if has_masks and i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    detection["mask"] = mask

                detections.append(detection)

        except Exception as e:
            logger.error(f"YOLO 검출 오류: {e}")

        return detections


class PerceptionPipeline:
    """YOLO + 6DOF 통합 인식 파이프라인"""

    def __init__(
        self,
        yolo_model_path: str = "yolov8n-seg.pt",
        camera_intrinsics: Tuple[float, float, float, float] = (395.26, 395.26, 256.0, 256.0),
        confidence_threshold: float = 0.5,
        min_depth: float = 0.1,
        max_depth: float = 2.0,
        device: str = "cuda:0",
    ):
        """
        Args:
            yolo_model_path: YOLO 모델 경로
            camera_intrinsics: (fx, fy, cx, cy)
            confidence_threshold: 검출 신뢰도 임계값
            min_depth: 최소 깊이 (미터)
            max_depth: 최대 깊이 (미터)
            device: 추론 장치
        """
        self.camera_intrinsics = camera_intrinsics
        self.min_depth = min_depth
        self.max_depth = max_depth

        # YOLO 검출기
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # 6DOF 추정기
        self.six_dof_estimator = SixDOFEstimator()

        logger.info("PerceptionPipeline 초기화 완료")

    @property
    def is_available(self) -> bool:
        return self.detector.is_available

    def process(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        camera_position: Optional[np.ndarray] = None,
        camera_orientation: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        RGB + Depth 이미지를 처리하여 객체 검출 및 6DOF 추정

        Args:
            rgb_image: RGB 이미지 (H, W, 3)
            depth_image: 깊이 이미지 (H, W), 미터 단위
            camera_position: 카메라 월드 위치 (3,)
            camera_orientation: 카메라 방향 쿼터니언 (4,)

        Returns:
            검출 결과 리스트 (각 항목에 position_6dof, orientation_6dof 포함)
        """
        # YOLO 검출
        detections = self.detector.detect(rgb_image)

        if len(detections) == 0 or depth_image is None:
            return detections

        fx, fy, cx, cy = self.camera_intrinsics

        # 각 검출에 대해 6DOF 추정
        for detection in detections:
            mask = detection.get("mask")

            if mask is None:
                detection["position_6dof"] = None
                detection["orientation_6dof"] = None
                detection["pose_6dof_info"] = {"status": "no_mask"}
                continue

            # 마스크 크기 조정
            if mask.shape != depth_image.shape:
                mask = cv2.resize(
                    mask,
                    (depth_image.shape[1], depth_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                detection["mask"] = mask

            # 3D 위치 계산
            position_camera, depth_info = get_masked_object_position(
                depth_image,
                mask,
                self.camera_intrinsics,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
            detection["position_camera"] = position_camera
            detection["depth_info"] = depth_info

            # 6DOF 추정을 위한 포인트 클라우드 생성
            try:
                obj_points_cam, _ = get_masked_pointcloud(
                    depth_image,
                    rgb_image,
                    mask,
                    fx, fy, cx, cy,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )

                if obj_points_cam is not None and len(obj_points_cam) > 10:
                    position, orientation, info = self.six_dof_estimator.estimate_6dof_with_confidence(
                        obj_points_cam,
                        class_name=detection["class_name"],
                        use_gravity_constraint=True,
                        camera_orientation=camera_orientation,
                    )

                    detection["position_6dof"] = position
                    detection["orientation_6dof"] = orientation
                    detection["pose_6dof_info"] = info
                else:
                    detection["position_6dof"] = None
                    detection["orientation_6dof"] = None
                    detection["pose_6dof_info"] = {"status": "insufficient_points"}

            except Exception as e:
                logger.error(f"6DOF 추정 오류: {e}")
                detection["position_6dof"] = None
                detection["orientation_6dof"] = None
                detection["pose_6dof_info"] = {"status": "error", "error": str(e)}

        return detections
