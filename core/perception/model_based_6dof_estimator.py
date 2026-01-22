"""
USD 모델과 ICP를 사용한 모델 기반 6DOF 포즈 추정기
정확한 포즈 추정을 위해 참조 3D 모델 사용
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Open3D는 선택적 의존성
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D를 사용할 수 없습니다. ICP 기반 포즈 추정이 비활성화됩니다.")


class ModelBased6DOFEstimator:
    """참조 모델과 ICP를 사용한 6DOF 포즈 추정"""

    def __init__(self, ref_models_path=None, config=None, six_dof_estimator=None):
        self.models = {}
        self.six_dof_estimator = six_dof_estimator

        # 설정에서 경로 가져오기 또는 기본값 사용
        if ref_models_path is None:
            if config:
                ref_models_path = config.get(
                    "perception.6dof.reference_models_path", "reference_models"
                )
            else:
                ref_models_path = "reference_models"

        self.ref_models_path = Path(ref_models_path)

        # YCB 객체 모델 파일 매핑
        self.model_files = {
            "PottedMeatCan": "010_potted_meat_can.pcd",
            "TunaFishCan": "004_tuna_fish_can.pcd",
            "FoamBrick": "061_foam_brick.pcd",
        }

        if OPEN3D_AVAILABLE:
            self._load_pcd_models()
        else:
            logger.warning("Open3D 미설치로 PCD 모델 로드 건너뜀")

    def _load_pcd_models(self):
        """파일에서 PCD 포인트 클라우드 모델 로드"""
        expected_sizes = {
            "PottedMeatCan": 0.083,
            "TunaFishCan": 0.033,
            "FoamBrick": 0.05,
        }

        for obj_name, filename in self.model_files.items():
            filepath = self.ref_models_path / filename

            if not filepath.exists():
                logger.debug(f"PCD 파일을 찾을 수 없음: {filepath}")
                continue

            try:
                pcd = o3d.io.read_point_cloud(str(filepath))

                if not pcd.has_points():
                    logger.error(f"PCD 파일에 포인트가 없음: {filepath}")
                    continue

                if not pcd.has_normals():
                    pcd.estimate_normals()

                pcd.translate(-pcd.get_center())
                self._fix_model_scale(pcd, expected_sizes.get(obj_name, 0.1))

                if len(pcd.points) > 10000:
                    pcd = pcd.voxel_down_sample(voxel_size=0.002)
                    pcd.estimate_normals()

                self.models[obj_name] = pcd
                logger.info(f"{obj_name}의 PCD 모델 로드: {len(pcd.points)} 포인트")

            except Exception as e:
                logger.error(f"{obj_name}의 PCD 로드 실패: {e}")

    def _fix_model_scale(self, pcd, expected_size: float):
        """모델 스케일 확인 및 수정"""
        bounds = pcd.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()
        max_extent = np.max(extent)

        if max_extent > expected_size * 10:
            scale_factor = expected_size / max_extent
            pcd.scale(scale_factor, center=[0, 0, 0])
        elif max_extent > expected_size * 2:
            scale_factor = expected_size / max_extent
            pcd.scale(scale_factor, center=[0, 0, 0])

    def estimate_pose_with_icp(
        self,
        observed_points: np.ndarray,
        class_name: str,
        initial_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        camera_orientation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        참조 모델과의 ICP 정렬을 사용한 6DOF 포즈 추정

        Args:
            observed_points: 관측된 3D 포인트의 Nx3 배열
            class_name: 객체 클래스 이름
            initial_pose: 선택적 초기 포즈 추정 (위치, 방향)
            camera_orientation: 중력 제약을 위한 카메라 방향

        Returns:
            position: 3D 위치
            orientation: 3x3 회전 행렬
            info: ICP 결과와 신뢰도가 포함된 딕셔너리
        """
        if not OPEN3D_AVAILABLE:
            position = np.mean(observed_points, axis=0) if len(observed_points) > 0 else np.zeros(3)
            return position, np.eye(3), {"status": "open3d_unavailable", "confidence": 0.0}

        if class_name not in self.models:
            logger.debug(f"클래스 {class_name}에 대한 PCD 모델 없음")
            position = np.mean(observed_points, axis=0) if len(observed_points) > 0 else np.zeros(3)
            return position, np.eye(3), {"status": "no_model", "confidence": 0.0}

        try:
            # 관측값에서 Open3D 포인트 클라우드 생성
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(observed_points)
            source_pcd.estimate_normals()

            # 참조 모델 복사
            target_pcd = o3d.geometry.PointCloud(self.models[class_name])

            # 초기 정렬
            source_center = source_pcd.get_center()
            target_center = target_pcd.get_center()

            source_pcd.translate(-source_center)
            target_pcd.translate(-target_center)

            # 초기 변환 생성
            init_transform = np.eye(4)

            if initial_pose is not None:
                _, init_orientation = initial_pose
                init_transform[:3, :3] = init_orientation
            elif self.six_dof_estimator is not None and camera_orientation is not None:
                centered_points = observed_points - source_center
                cov_matrix = np.cov(centered_points.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                init_transform[:3, :3] = self.six_dof_estimator._apply_gravity_constraint(
                    eigenvectors, eigenvalues, camera_orientation
                )

            # 스케일 조정
            source_bounds = source_pcd.get_axis_aligned_bounding_box()
            target_bounds = target_pcd.get_axis_aligned_bounding_box()
            source_extent = source_bounds.get_extent()
            target_extent = target_bounds.get_extent()

            scale_factor = np.median(target_extent / (source_extent + 1e-6))
            if scale_factor < 0.5 or scale_factor > 2.0:
                source_pcd.scale(scale_factor, center=source_pcd.get_center())

            # ICP 수행
            threshold = 0.1

            if target_pcd.has_normals() and source_pcd.has_normals():
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    threshold,
                    init_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
                )
            else:
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    threshold,
                    init_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
                )

            if reg_p2p.fitness < 0.01:
                final_transform = init_transform
            else:
                final_transform = reg_p2p.transformation

            position = final_transform[:3, 3] + source_center
            orientation = final_transform[:3, :3].copy()

            fitness = reg_p2p.fitness
            rmse = reg_p2p.inlier_rmse
            confidence = min(1.0, fitness * np.exp(-rmse / 0.01))

            if np.linalg.det(orientation) < 0:
                orientation[:, 2] *= -1

            info = {
                "status": "success",
                "fitness": fitness,
                "inlier_rmse": rmse,
                "num_correspondence": len(reg_p2p.correspondence_set),
                "num_points": len(observed_points),
                "confidence": confidence,
                "method": "icp_model_based",
            }

            return position, orientation, info

        except Exception as e:
            logger.error(f"{class_name}에 대한 ICP 실패: {e}")
            position = np.mean(observed_points, axis=0) if len(observed_points) > 0 else np.zeros(3)
            return position, np.eye(3), {"status": "error", "error": str(e), "confidence": 0.0}
