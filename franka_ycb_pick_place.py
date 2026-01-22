#!/usr/bin/env python3
"""
Isaac Sim 4.5 Pick and Place Demo
=================================
Robot: Franka Panda
Task: Pick up a cube and place it in a container
Environment: Ground plane, container, DynamicCuboid (no table - prevents RMPFlow interference)

Usage (without ROS2):
    cd ~/isaac_sim_4.5
    ./python.sh /home/yjchoi/ClaudeCode_PlanMode_PickAndPlace/franka_ycb_pick_place.py

Usage (with ROS2 - uses Isaac Sim's internal ROS2 libraries, no need to source ROS2):
    /home/yjchoi/ClaudeCode_PlanMode_PickAndPlace/run_with_ros2.sh

    Or manually:
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH=~/isaac_sim_4.5/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
    cd ~/isaac_sim_4.5
    ./python.sh /home/yjchoi/ClaudeCode_PlanMode_PickAndPlace/franka_ycb_pick_place.py --ros2
"""

# === Command-line Arguments (must be parsed BEFORE SimulationApp) ===
import argparse
import os

parser = argparse.ArgumentParser(description="Franka YCB Pick and Place Demo")
parser.add_argument("--ros2", action="store_true", help="Enable ROS2 publishing")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--warehouse", action="store_true", help="Use photo-realistic warehouse environment")
args, unknown = parser.parse_known_args()

# === ROS2 Environment Check ===
# Note: LD_LIBRARY_PATH must be set BEFORE starting Isaac Sim
# Use run_with_ros2.sh launcher script for automatic setup
if args.ros2:
    # Set ROS_DISTRO if not already set (Isaac Sim will use backup)
    if "ROS_DISTRO" not in os.environ:
        os.environ["ROS_DISTRO"] = "humble"
    # Set RMW implementation
    if "RMW_IMPLEMENTATION" not in os.environ:
        os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    print("[ROS2] ROS2 mode enabled")

# === CRITICAL: SimulationApp must be initialized FIRST ===
from isaacsim import SimulationApp

CONFIG = {
    "headless": args.headless,  # Set via --headless flag
    "width": 1280,
    "height": 720,
    "window_title": "Franka Pick and Place Demo",
}

# Track if running in headless mode for auto-start
IS_HEADLESS = args.headless
ENABLE_ROS2 = args.ros2

simulation_app = SimulationApp(CONFIG)

# === Standard Library Imports ===
import sys
import functools
import numpy as np
from dataclasses import dataclass

# Force flush stdout for headless mode
print = functools.partial(print, flush=True)

# === Perception Module Import (Optional) ===
try:
    from core.perception.detection_pipeline import PerceptionPipeline, YOLODetector
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    print("[INFO] Perception module not available (install ultralytics for YOLO)")

# === Isaac Sim Imports ===
import carb
import carb.input
import omni.usd
import omni.appwindow
from pxr import UsdLux, UsdGeom, Gf, UsdShade
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, VisualSphere
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import RigidPrim, GeometryPrim, SingleRigidPrim
from pxr import UsdPhysics, PhysxSchema, Usd
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SingleXFormPrim
from omni.isaac.sensor import Camera


# =============================================================================
# Configuration Constants
# =============================================================================

# Robot configuration
ROBOT_POSITION = np.array([0.0, 0.0, 0.0])
ROBOT_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz quaternion

# YCB Object Configurations
# Key: 1, 2, 3 for keyboard selection
# mesh_name: Child mesh prim name inside the USD file (for collision setup)
# Selected objects: All < 80mm width (Franka gripper max opening)
YCB_OBJECT_CONFIGS = {
    "010_potted_meat_can": {
        "key": "1",
        "usd_file": "010_potted_meat_can.usd",
        "display_name": "Potted Meat Can",
        "mesh_name": "_10_potted_meat_can",  # 65-70mm diameter - graspable
        "mass": 0.35,
        "pick_height_offset": 0.02,  # Pick slightly above center
        "spawn_height": 0.35,  # Spawn above platform (will settle on it)
        "spawn_xy": np.array([0.3, 0.3]),
        "grip_friction": 5.0,  # Very high friction for heavy object (0.35kg)
    },
    "007_tuna_fish_can": {
        "key": "2",
        "usd_file": "007_tuna_fish_can.usd",
        "display_name": "Tuna Fish Can",
        "mesh_name": "_07_tuna_fish_can",  # ~35mm diameter - easily graspable
        "mass": 0.20,
        "pick_height_offset": 0.02,  # Pick slightly above center
        "spawn_height": 0.35,  # Spawn above platform
        "spawn_xy": np.array([0.3, 0.3]),
        "grip_friction": 1.0,  # Standard friction
    },
    "061_foam_brick": {
        "key": "3",
        "usd_file": "061_foam_brick.usd",
        "display_name": "Foam Brick",
        "mesh_name": "_61_foam_brick",  # 38-50mm width - graspable
        "mass": 0.05,  # Very light
        "pick_height_offset": 0.02,  # Pick slightly above center
        "spawn_height": 0.35,  # Spawn above platform
        "spawn_xy": np.array([0.3, 0.3]),
        "grip_friction": 0.1,  # Very low friction - allows release when gripper opens (light object)
    },
}

# Default object selection (None = use DynamicCuboid, or YCB object name)
# Selected graspable YCB objects (< 80mm width for Franka gripper)
SELECTED_YCB_OBJECT = "010_potted_meat_can"  # Default YCB object
USE_YCB_OBJECTS = True  # Master switch for YCB objects

# Legacy DynamicCuboid configuration (kept for fallback)
OBJECT_POSITION = np.array([0.3, 0.3, 0.3])
OBJECT_SCALE = np.array([0.0515, 0.0515, 0.0515])  # ~5cm cube
OBJECT_COLOR = np.array([0, 0, 1])  # Blue

# Support platform configuration (for YCB objects)
SUPPORT_PLATFORM_HEIGHT = 0.28  # EE reachable height (slightly higher)
SUPPORT_PLATFORM_SIZE = 0.20    # 20cm x 20cm (larger for stability)

# Container configuration (low, won't interfere with robot)
CONTAINER_POSITION = np.array([0.3, -0.3, 0.02])
CONTAINER_SIZE = np.array([0.12, 0.12, 0.05])       # 12x12x5cm
CONTAINER_WALL_THICKNESS = 0.008                     # 8mm walls
CONTAINER_COLOR = np.array([0.4, 0.4, 0.5])         # Gray-blue

# Place target (center of container, at reachable height)
PLACE_POSITION = np.array([0.3, -0.3, 0.27])

# Camera configuration (diagonal view from above)
CAMERA_POSITION = np.array([1.5, 1.5, 1.2])   # Eye position
CAMERA_TARGET = np.array([0.4, 0.0, 0.4])     # Look at table center

# Hand-mounted camera configuration
ENABLE_HAND_CAMERA = True  # Enable/disable hand camera


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for hand-mounted RealSense camera."""
    fx: float = 395.26  # Focal length x (pixels)
    fy: float = 395.26  # Focal length y (pixels)
    cx: float = 256.0   # Principal point x
    cy: float = 256.0   # Principal point y
    width: int = 512    # Image width
    height: int = 512   # Image height


# =============================================================================
# Camera and Lighting Setup
# =============================================================================

def setup_camera() -> None:
    """Set up camera view to show the entire scene clearly."""
    set_camera_view(
        eye=CAMERA_POSITION,
        target=CAMERA_TARGET,
        camera_prim_path="/OmniverseKit_Persp"
    )
    print(f"[INFO] Camera set - eye: {CAMERA_POSITION}, target: {CAMERA_TARGET}")


def setup_warehouse_environment() -> bool:
    """
    Set up photo-realistic warehouse environment.

    Uses Isaac Sim's built-in warehouse assets for realistic rendering.

    Returns:
        True if environment was loaded successfully, False otherwise
    """
    stage = omni.usd.get_context().get_stage()

    # Isaac Sim 4.5 warehouse environment USD paths
    warehouse_environments = [
        # Simple warehouse (no shelves - GPU memory efficient)
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        # Simple room (fallback - lighter)
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Room/simple_room.usd",
        # Grid environment (minimal fallback)
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd",
    ]

    environment_path = "/World/Environment"

    for env_usd in warehouse_environments:
        try:
            print(f"[INFO] Trying to load environment: {env_usd}")

            # Create environment prim
            prim_utils.create_prim(
                prim_path=environment_path,
                prim_type="Xform",
                position=np.array([0, 0, 0]),
            )

            # Add USD reference
            add_reference_to_stage(
                usd_path=env_usd,
                prim_path=environment_path
            )

            print(f"[INFO] Warehouse environment loaded successfully: {env_usd}")
            return True

        except Exception as e:
            print(f"[WARN] Failed to load environment ({env_usd}): {e}")
            # Try to remove failed prim
            try:
                prim = stage.GetPrimAtPath(environment_path)
                if prim.IsValid():
                    stage.RemovePrim(environment_path)
            except:
                pass
            continue

    print("[WARN] All environment loading attempts failed. Using default ground plane.")
    return False


def setup_lighting() -> None:
    """Set up scene lighting with Dome Light and Distant Light."""
    stage = omni.usd.get_context().get_stage()

    # 1. Dome Light (ambient environment lighting)
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)
    print("[INFO] Dome light added (intensity: 1000)")

    # 2. Distant Light (directional sunlight effect)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(500.0)
    distant_light.CreateAngleAttr(0.53)  # Sun angular size

    # Set light direction (45 degree angle)
    xform = UsdGeom.Xformable(distant_light)
    xform.AddRotateXYZOp().Set((45, 45, 0))
    print("[INFO] Distant light added (intensity: 500, angle: 45)")


def setup_hand_camera(franka_prim_path: str) -> tuple:
    """
    Attach RealSense depth camera to Franka panda_hand.
    Uses existing geometry/realsense visual from Franka USD (no separate mount needed).

    Args:
        franka_prim_path: Path to Franka robot prim (e.g., "/World/Franka")

    Returns:
        tuple: (Camera instance, camera_prim_path, render_product_path) or (None, None, None) on failure
    """
    stage = omni.usd.get_context().get_stage()

    # Use existing realsense visual from Franka USD at geometry/realsense
    # Camera sensor is created as a child of the existing realsense prim
    realsense_visual_path = f"{franka_prim_path}/panda_hand/geometry/realsense"
    camera_prim_path = f"{realsense_visual_path}/realsense_camera"

    # Check if the realsense visual already exists in Franka USD
    realsense_prim = stage.GetPrimAtPath(realsense_visual_path)
    if realsense_prim.IsValid():
        print(f"[INFO] Found existing RealSense visual at {realsense_visual_path}")
    else:
        print(f"[WARN] RealSense visual not found at {realsense_visual_path}")
        # Fallback: create at panda_hand level if geometry/realsense doesn't exist
        camera_prim_path = f"{franka_prim_path}/panda_hand/realsense_camera"
        print(f"[INFO] Using fallback camera path: {camera_prim_path}")

    # Create Camera sensor at the realsense location
    try:
        hand_camera = Camera(
            prim_path=camera_prim_path,
            name="hand_camera",
            frequency=30,
            resolution=(512, 512),
        )
        hand_camera.initialize()

        # === USD Camera Parameters (from final-manipulator) ===
        # Must be set in code for ROS2 topics to reflect these values
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        if camera_prim.IsValid():
            camera_usd = UsdGeom.Camera(camera_prim)
            if camera_usd:
                # Wide-angle lens settings (RealSense D435 style)
                camera_usd.GetHorizontalApertureAttr().Set(2.5)
                camera_usd.GetVerticalApertureAttr().Set(1.0)
                camera_usd.GetFocalLengthAttr().Set(1.93)
                # Depth measurement range: 10cm ~ 10m
                camera_usd.GetClippingRangeAttr().Set((0.1, 10.0))
                print("[INFO] Camera USD params: Focal=1.93, HorAperture=2.5, Clip=(0.1, 10.0)")

        # Get render product path first
        render_product_path = hand_camera.get_render_product_path()
        print(f"[INFO] Camera render product path: {render_product_path}")

        # Add depth sensor capability using Camera class method
        try:
            hand_camera.add_distance_to_image_plane_to_frame()
            print(f"[INFO] Depth sensor added via Camera class")
        except Exception as e:
            print(f"[WARN] Could not add depth via Camera class: {e}")

        # Also attach depth annotator using replicator API for ROS2 bridge
        try:
            import omni.replicator.core as rep
            # Attach both distance_to_image_plane and distance_to_camera for compatibility
            for annotator_name in ["distance_to_image_plane", "distance_to_camera"]:
                try:
                    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
                    annotator.attach([render_product_path])
                    print(f"[INFO] Attached annotator: {annotator_name}")
                except Exception as e:
                    print(f"[WARN] Could not attach {annotator_name}: {e}")
        except Exception as e:
            print(f"[WARN] Replicator annotator setup failed: {e}")

        print(f"[INFO] Hand camera created with depth sensor at {camera_prim_path}")
        return hand_camera, camera_prim_path, render_product_path

    except Exception as e:
        print(f"[ERROR] Failed to create hand camera: {e}")
        return None, None, None


class CameraProcessor:
    """Process camera frames to extract RGB and depth data."""

    def __init__(self, camera):
        self.camera = camera
        self.frame_count = 0

    def get_frame(self) -> tuple:
        """
        Extract RGB and depth data from camera.

        Returns:
            (rgb_data, depth_data) tuple, either may be None
        """
        self.frame_count += 1
        rgb_data = None
        depth_data = None

        try:
            # Get current frame from camera
            camera_frame = self.camera.get_current_frame()

            if camera_frame is not None:
                # Extract RGB (handle RGBA format)
                rgba = camera_frame.get("rgba", camera_frame.get("rgb"))
                if rgba is not None:
                    rgb_data = rgba[:, :, :3] if rgba.shape[-1] == 4 else rgba

                # Extract depth
                depth_data = camera_frame.get("distance_to_image_plane")
                if depth_data is None:
                    depth_data = camera_frame.get("distance_to_camera")

        except Exception as e:
            if self.frame_count % 100 == 0:
                print(f"[WARN] Camera frame error: {e}")

        return rgb_data, depth_data


# =============================================================================
# ROS2 Integration (using Isaac Sim's built-in ROS2 Bridge)
# =============================================================================

def check_ros2_environment() -> bool:
    """
    Check if ROS2 environment is properly configured.
    LD_LIBRARY_PATH must be set BEFORE starting Isaac Sim.

    Returns:
        True if environment is properly configured
    """
    import os

    isaac_sim_path = os.path.expanduser("~/isaac_sim_4.5")
    ros2_lib_path = f"{isaac_sim_path}/exts/isaacsim.ros2.bridge/humble/lib"

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    # Check if Isaac Sim ROS2 library path is in LD_LIBRARY_PATH
    if ros2_lib_path not in ld_library_path:
        return False

    return True


def setup_ros2_bridge(render_product_path: str) -> bool:
    """
    Setup Isaac Sim's built-in ROS2 Bridge for camera publishing.
    Uses the Camera's render product directly (no extra viewport).
    Requires proper environment setup via run_with_ros2.sh launcher.

    Args:
        render_product_path: Path to camera's render product (from Camera.get_render_product_path())

    Returns:
        True if successful, False otherwise
    """
    # Check environment first
    if not check_ros2_environment():
        print("[ROS2 Bridge] Environment not configured properly.")
        print("[ROS2 Bridge] Please use the launcher script: ./run_with_ros2.sh")
        print("[ROS2 Bridge] Or set environment variables before starting Isaac Sim:")
        print("[ROS2 Bridge]   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
        print("[ROS2 Bridge]   export LD_LIBRARY_PATH=~/isaac_sim_4.5/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH")
        return False

    if not render_product_path:
        print("[ROS2 Bridge] No render product path provided.")
        return False

    try:
        import omni.graph.core as og
        from isaacsim.core.utils import extensions

        # Enable ROS2 Bridge extension
        extensions.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        # Wait for extension to fully initialize
        for _ in range(5):
            simulation_app.update()

        # Verify the ROS2 node types are available
        try:
            node_type = og.get_node_type("isaacsim.ros2.bridge.ROS2CameraHelper")
            if node_type is None:
                print("[ROS2 Bridge] Extension loaded but ROS2 nodes not available.")
                return False
        except Exception:
            print("[ROS2 Bridge] ROS2 node types not registered.")
            return False

        print("[ROS2 Bridge] Extension enabled and verified")
        print(f"[ROS2 Bridge] Using render product: {render_product_path}")

        keys = og.Controller.Keys

        # Camera publishing graph - uses Camera's render product directly (no new viewport)
        ros_camera_graph_path = "/ROS2_HandCamera"
        (ros_camera_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": ros_camera_graph_path,
                "evaluator_name": "push",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    # Depth is published manually with Jet colormap (like final-manipulator)
                    ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "cameraHelperRgb.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "cameraHelperInfo.inputs:execIn"),
                ],
                keys.SET_VALUES: [
                    ("cameraHelperRgb.inputs:frameId", "realsense_camera"),
                    ("cameraHelperRgb.inputs:topicName", "/realsense/image_raw"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperRgb.inputs:renderProductPath", render_product_path),
                    ("cameraHelperInfo.inputs:frameId", "realsense_camera"),
                    ("cameraHelperInfo.inputs:topicName", "/realsense/camera_info"),
                    ("cameraHelperInfo.inputs:renderProductPath", render_product_path),
                ],
            },
        )
        og.Controller.evaluate_sync(ros_camera_graph)
        simulation_app.update()

        print("[ROS2 Bridge] Camera graph created successfully (no extra viewport)")
        print("[ROS2 Bridge] Publishing topics:")
        print("[ROS2 Bridge]   - /realsense/image_raw (RGB)")
        print("[ROS2 Bridge]   - /realsense/camera_info (CameraInfo)")
        print("[ROS2 Bridge]   - /realsense/depth_image (Depth - manual Jet colormap)")

        return True

    except Exception as e:
        print(f"[ROS2 Bridge] Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


class ROS2BridgePublisher:
    """
    ROS2 publisher using Isaac Sim's built-in ROS2 Bridge.
    Requires proper environment setup via run_with_ros2.sh launcher.

    Depth images are published manually with Jet colormap (like final-manipulator)
    for proper visualization in rqt_image_view.
    """

    def __init__(self, render_product_path: str = None, franka=None):
        self.enabled = False
        self.franka = franka
        self.render_product_path = render_product_path
        self._frame_count = 0

        # Manual depth publisher (rclpy)
        self._depth_publisher = None
        self._cv_bridge = None
        self._rclpy_node = None

        if render_product_path:
            self.enabled = setup_ros2_bridge(render_product_path)

        if self.enabled:
            # Initialize manual depth publisher with Jet colormap
            self._init_depth_publisher()
            print("[ROS2 Bridge] Publisher initialized")
            print("[ROS2 Bridge] Use 'ros2 topic list' in another terminal to see topics")
        else:
            print("[ROS2 Bridge] Publisher disabled (use ./run_with_ros2.sh for ROS2 support)")

    def _init_depth_publisher(self):
        """Initialize rclpy-based depth publisher with Jet colormap (like final-manipulator)."""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge

            # Initialize rclpy if not already initialized
            if not rclpy.ok():
                rclpy.init()

            # Create a simple node for depth publishing
            self._rclpy_node = rclpy.create_node('depth_colormap_publisher')
            self._depth_publisher = self._rclpy_node.create_publisher(
                Image, '/realsense/depth_image', 10
            )
            self._cv_bridge = CvBridge()
            print("[ROS2 Bridge] Manual depth publisher initialized (Jet colormap)")

        except ImportError as e:
            print(f"[ROS2 Bridge] rclpy/cv_bridge not available: {e}")
            self._depth_publisher = None
        except Exception as e:
            print(f"[ROS2 Bridge] Manual depth publisher init failed: {e}")
            self._depth_publisher = None

    def publish_depth(self, depth_data: np.ndarray):
        """
        Publish depth data with Jet colormap visualization.

        This method converts raw depth data to a colored visualization
        using OpenCV's Jet colormap, matching the final-manipulator approach
        for proper display in rqt_image_view.

        Args:
            depth_data: Raw depth data in meters (H, W) float array
        """
        if self._depth_publisher is None or depth_data is None:
            return

        try:
            import cv2

            # Handle NaN and inf values (from final-manipulator)
            valid_depth = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize to 0-255 range
            max_depth = np.max(valid_depth)
            if max_depth > 0:
                depth_vis = (valid_depth * 255 / max_depth).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(valid_depth, dtype=np.uint8)

            # Apply Jet colormap (like final-manipulator)
            # Close objects: blue, Far objects: red
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Convert to ROS2 message and publish
            depth_msg = self._cv_bridge.cv2_to_imgmsg(depth_colored, "bgr8")
            depth_msg.header.frame_id = "realsense_camera"
            # Set timestamp
            import time
            now = time.time()
            depth_msg.header.stamp.sec = int(now)
            depth_msg.header.stamp.nanosec = int((now - int(now)) * 1e9)

            self._depth_publisher.publish(depth_msg)

        except Exception as e:
            # Silently ignore errors to avoid spam
            pass

    def update(self):
        """Called every frame - ROS2 Bridge handles publishing automatically."""
        self._frame_count += 1
        # Isaac Sim's ROS2 Bridge publishes RGB and CameraInfo via OmniGraph
        # Depth is published manually via publish_depth() method with Jet colormap

    def shutdown(self):
        """Cleanup - ROS2 Bridge handles cleanup automatically."""
        if self._rclpy_node is not None:
            try:
                self._rclpy_node.destroy_node()
            except:
                pass
        print("[ROS2 Bridge] Shutdown complete")


def create_grip_friction_material(stage, material_path: str, friction: float = 1.0):
    """Create physics material with specified friction for grasping."""
    UsdShade.Material.Define(stage, material_path)
    material_prim = stage.GetPrimAtPath(material_path)

    physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics_material.CreateStaticFrictionAttr(friction)
    physics_material.CreateDynamicFrictionAttr(friction)
    physics_material.CreateRestitutionAttr(0.0)

    return material_path


def apply_material_to_prim(stage, prim_path: str, material_path: str):
    """Bind physics material to a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        material = UsdShade.Material.Get(stage, material_path)
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        return True
    return False


# =============================================================================
# Keyboard Input Controller
# =============================================================================

class SimulationController:
    """Handles keyboard input for simulation control and manual joint control."""

    def __init__(self, franka=None):
        # Existing state for auto mode
        self.start_requested = False
        self.object_change_requested = None  # Stores new object name when 1,2,3 pressed
        self.reset_requested = False
        self.detect_requested = False  # YOLO detection request (D key)

        # NEW: Mode control (M key to toggle)
        self.manual_mode = False  # False=auto pick-place, True=manual joint control

        # NEW: Joint control state
        self.franka = franka
        self.joint_velocity = 0.02  # rad/step (from Project A KeyboardManager)
        self.current_joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.home_positions = self.current_joint_positions.copy()

        # Franka joint limits (rad) - from Franka specs
        self.joint_limits = [
            [-2.8973, 2.8973],   # Joint 1
            [-1.7628, 1.7628],   # Joint 2
            [-2.8973, 2.8973],   # Joint 3
            [-3.0718, -0.0698],  # Joint 4
            [-2.8973, 2.8973],   # Joint 5
            [-0.0175, 3.7525],   # Joint 6
            [-2.8973, 2.8973],   # Joint 7
        ]

        # Track modifier keys
        self.ctrl_pressed = False
        self.gripper_open = True

        self._setup_keyboard()

    def _setup_keyboard(self):
        """Set up keyboard event listener."""
        try:
            app_window = omni.appwindow.get_default_app_window()
            input_interface = carb.input.acquire_input_interface()
            keyboard = app_window.get_keyboard()
            input_interface.subscribe_to_keyboard_events(
                keyboard, self._on_keyboard_event
            )
            print("[INFO] Keyboard controller initialized")
            print("[INFO] Mode: M=Toggle Manual Mode")
            print("[INFO] Auto mode: S=Start, R=Reset, 1=PottedMeatCan, 2=TunaFishCan, 3=FoamBrick")
            print("[INFO] Manual mode: 1-7=Joint+, Ctrl+1-7=Joint-, +/-=Speed, H=Home, O=Gripper, P=Print")
        except Exception as e:
            carb.log_warn(f"Could not setup keyboard: {e}")

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Handle keyboard events."""
        # Get key name safely (handle both string and object types)
        try:
            key_name = event.input.name if hasattr(event.input, 'name') else str(event.input)
        except Exception:
            return True

        # Track Ctrl key state
        if key_name in ["LEFT_CONTROL", "RIGHT_CONTROL"]:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.ctrl_pressed = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.ctrl_pressed = False
            return True

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:

            # Mode toggle (always available)
            if key_name == "M":
                self.manual_mode = not self.manual_mode
                mode_str = "MANUAL JOINT CONTROL" if self.manual_mode else "AUTO PICK-PLACE"
                print(f"\n{'='*50}")
                print(f"[MODE] Switched to: {mode_str}")
                if self.manual_mode:
                    print("[INFO] Keys: 1-7=Joint+, Ctrl+1-7=Joint-, +/-=Speed")
                    print("[INFO]       H=Home, O=Gripper, P=Print, R=Reset")
                else:
                    print("[INFO] Keys: S=Start, R=Reset, 1/2/3=Select Object")
                print(f"{'='*50}\n")
                return True

            # Dispatch to appropriate handler
            if self.manual_mode:
                return self._handle_manual_mode_key(key_name)
            else:
                return self._handle_auto_mode_key(key_name)

        return True

    def _handle_manual_mode_key(self, key_name: str) -> bool:
        """Handle keys in manual joint control mode."""
        # Joint control keys 1-7
        joint_keys = {"KEY_1": 0, "KEY_2": 1, "KEY_3": 2, "KEY_4": 3,
                      "KEY_5": 4, "KEY_6": 5, "KEY_7": 6}

        if key_name in joint_keys:
            joint_idx = joint_keys[key_name]
            direction = -1 if self.ctrl_pressed else 1
            self._move_joint(joint_idx, direction)
            return True

        # Speed control
        if key_name == "EQUAL":  # + key
            self.joint_velocity = min(0.1, self.joint_velocity * 1.2)
            print(f"[JOINT] Speed: {self.joint_velocity:.4f} rad/step")
            return True
        if key_name == "MINUS":  # - key
            self.joint_velocity = max(0.005, self.joint_velocity * 0.8)
            print(f"[JOINT] Speed: {self.joint_velocity:.4f} rad/step")
            return True

        # Home position
        if key_name == "H":
            self.current_joint_positions = self.home_positions.copy()
            self._apply_joint_positions()
            print("[JOINT] Reset to home position")
            return True

        # Gripper toggle
        if key_name == "O":
            self._toggle_gripper()
            return True

        # Print positions
        if key_name == "P":
            self._print_joint_positions()
            return True

        # Reset still works in manual mode
        if key_name == "R":
            self.reset_requested = True
            print("\n[INFO] 'R' key pressed - Reset requested!")
            return True

        return False

    def _handle_auto_mode_key(self, key_name: str) -> bool:
        """Handle keys in auto pick-place mode (existing behavior)."""
        if key_name == "S":
            self.start_requested = True
            print("\n[INFO] 'S' key pressed - Starting simulation!")
        elif key_name == "R":
            self.reset_requested = True
            print("\n[INFO] 'R' key pressed - Reset requested!")
        elif key_name == "KEY_1":
            self.object_change_requested = "010_potted_meat_can"
            print("\n[INFO] '1' key pressed - Selecting Potted Meat Can")
        elif key_name == "KEY_2":
            self.object_change_requested = "007_tuna_fish_can"
            print("\n[INFO] '2' key pressed - Selecting Tuna Fish Can")
        elif key_name == "KEY_3":
            self.object_change_requested = "061_foam_brick"
            print("\n[INFO] '3' key pressed - Selecting Foam Brick")
        elif key_name == "D":
            self.detect_requested = True
            print("\n[INFO] 'D' key pressed - Running YOLO detection")
        return True

    def _move_joint(self, joint_idx: int, direction: int):
        """Move a specific joint with limits enforcement."""
        new_pos = self.current_joint_positions[joint_idx] + (direction * self.joint_velocity)

        # Apply joint limits
        min_limit, max_limit = self.joint_limits[joint_idx]
        new_pos = np.clip(new_pos, min_limit, max_limit)

        self.current_joint_positions[joint_idx] = new_pos
        self._apply_joint_positions()

        dir_str = "+" if direction > 0 else "-"
        print(f"[JOINT {joint_idx+1}] {dir_str} -> {new_pos:.4f} rad")

    def _apply_joint_positions(self):
        """Apply joint positions to robot."""
        if self.franka is not None:
            # Franka has 9 DOF (7 arm + 2 gripper)
            full_positions = np.zeros(9)
            full_positions[:7] = self.current_joint_positions
            gripper_pos = 0.04 if self.gripper_open else 0.0
            full_positions[7:9] = gripper_pos
            self.franka.set_joint_positions(full_positions)

    def _toggle_gripper(self):
        """Toggle gripper open/close."""
        self.gripper_open = not self.gripper_open
        self._apply_joint_positions()
        state_str = "OPEN" if self.gripper_open else "CLOSED"
        print(f"[GRIPPER] {state_str}")

    def _print_joint_positions(self):
        """Print current joint positions."""
        print("\n" + "=" * 50)
        print("Current Joint Positions (rad):")
        for i, pos in enumerate(self.current_joint_positions):
            limit = self.joint_limits[i]
            print(f"  Joint {i+1}: {pos:8.4f}  (limits: {limit[0]:6.2f} to {limit[1]:6.2f})")
        gripper_str = "OPEN" if self.gripper_open else "CLOSED"
        print(f"  Gripper: {gripper_str}")
        print("=" * 50 + "\n")

    def sync_from_robot(self):
        """Sync internal state from actual robot positions."""
        if self.franka is not None:
            positions = self.franka.get_joint_positions()
            if positions is not None and len(positions) >= 7:
                self.current_joint_positions = positions[:7].copy()

    def set_franka(self, franka):
        """Set franka reference after initialization."""
        self.franka = franka
        self.sync_from_robot()


# =============================================================================
# Verification and Tracking System
# =============================================================================

class VerificationTracker:
    """로봇 동작의 실제 성공/실패를 추적하고 검증하는 클래스"""

    def __init__(self, franka, object_prim_path: str, pick_target: np.ndarray, place_target: np.ndarray):
        self.franka = franka
        self.object_xform = SingleXFormPrim(prim_path=object_prim_path)
        self.pick_target = pick_target
        self.place_target = place_target
        self.initial_object_pos = None
        self.last_phase = -1
        self.position_tolerance = 0.05  # 5cm tolerance

    def initialize(self):
        """World reset 후 호출 - 객체 초기 위치 저장"""
        self.object_xform.initialize()
        self.initial_object_pos, _ = self.object_xform.get_world_pose()
        print(f"[VERIFY] Initial object position: {self.initial_object_pos}")

    def get_ee_position(self) -> np.ndarray:
        """End Effector 현재 위치 반환"""
        pos, _ = self.franka.end_effector.get_world_pose()
        return pos

    def get_object_position(self) -> np.ndarray:
        """객체 현재 위치 반환"""
        pos, _ = self.object_xform.get_world_pose()
        return pos

    def log_phase_status(self, current_phase: int, step_count: int):
        """Phase 변경 시 상세 로깅"""
        if current_phase != self.last_phase:
            ee_pos = self.get_ee_position()
            object_pos = self.get_object_position()

            phase_names = [
                "Moving above pick", "Lowering to grasp", "Waiting settle",
                "Closing gripper", "Lifting object", "Moving to place XY",
                "Lowering to place", "Opening gripper", "Lifting up", "Going home"
            ]
            phase_name = phase_names[current_phase] if current_phase < len(phase_names) else f"Phase {current_phase}"

            print(f"\n[PHASE {current_phase}] {phase_name}")
            print(f"  EE Position:     {ee_pos}")
            print(f"  Object Position: {object_pos}")
            print(f"  Pick Target:     {self.pick_target}")
            print(f"  Place Target:    {self.place_target}")

            # Phase별 검증
            if current_phase == 1:  # Lowering to grasp
                dist_to_pick = np.linalg.norm(ee_pos[:2] - self.pick_target[:2])
                print(f"  [CHECK] EE XY distance to pick target: {dist_to_pick:.4f}m")
                if dist_to_pick > self.position_tolerance:
                    print(f"  [WARN] EE is NOT above object! Expected XY: {self.pick_target[:2]}")

            elif current_phase == 6:  # Lowering to place
                dist_to_place = np.linalg.norm(ee_pos[:2] - self.place_target[:2])
                print(f"  [CHECK] EE XY distance to place target: {dist_to_place:.4f}m")

            self.last_phase = current_phase

    def verify_final_result(self) -> bool:
        """최종 결과 검증 - 객체가 컨테이너로 이동했는지 확인"""
        final_object_pos = self.get_object_position()

        # 객체 이동 거리
        dist_moved = np.linalg.norm(final_object_pos - self.initial_object_pos)

        # 객체와 Place Target 간 거리
        dist_to_target = np.linalg.norm(final_object_pos[:2] - self.place_target[:2])

        print("\n" + "=" * 60)
        print("[FINAL VERIFICATION]")
        print(f"  Initial object position: {self.initial_object_pos}")
        print(f"  Final object position:   {final_object_pos}")
        print(f"  Place target position:   {self.place_target}")
        print(f"  Distance moved:          {dist_moved:.4f}m")
        print(f"  Distance to target:      {dist_to_target:.4f}m")

        success = dist_to_target < self.position_tolerance * 2  # 10cm tolerance for final

        if success:
            print("[RESULT] SUCCESS - Object is in/near container!")
        else:
            print("[RESULT] FAILED - Object did NOT reach container!")
            if dist_moved < 0.01:
                print("         Object barely moved - gripper likely missed it")

        print("=" * 60 + "\n")
        return success


# =============================================================================
# RMPFlow Obstacle Manager
# =============================================================================

class RMPFlowObstacleManager:
    """
    RMPFlow 장애물 동적 제어 - pick 시 비활성화, place 시 활성화.

    RMPFlow의 collision avoidance가 gripper가 객체에 도달하는 것을 방해하므로,
    Pick 동작 중에는 객체 회피를 비활성화하고, Grasp 후에는 다시 활성화합니다.
    """

    def __init__(self, controller):
        """
        Args:
            controller: PickPlaceController instance
        """
        self.controller = controller
        # RMPFlow 객체 접근 경로
        self.rmpflow = controller._cspace_controller.rmp_flow
        self.pick_obstacle = None
        self.is_disabled = False

    def register_obstacle(self, obstacle):
        """RMPFlow에 장애물 등록."""
        if obstacle is not None:
            self.pick_obstacle = obstacle
            self.rmpflow.add_obstacle(obstacle, static=False)
            print(f"[INFO] Registered obstacle with RMPFlow: {obstacle.prim_path}")

    def disable_for_pick(self):
        """Pick 동작 중 장애물 회피 비활성화."""
        if self.pick_obstacle and not self.is_disabled:
            self.rmpflow.disable_obstacle(self.pick_obstacle)
            self.is_disabled = True
            print("[INFO] RMPFlow obstacle disabled for picking")

    def enable_after_grasp(self):
        """Grasp 후 장애물 회피 재활성화."""
        if self.pick_obstacle and self.is_disabled:
            self.rmpflow.enable_obstacle(self.pick_obstacle)
            self.is_disabled = False
            print("[INFO] RMPFlow obstacle re-enabled")

    def reset(self):
        """Reset the obstacle manager state."""
        self.is_disabled = False


# =============================================================================
# Helper Functions
# =============================================================================

def _remove_existing_physics(prim) -> None:
    """
    Remove any existing physics APIs from the prim and its children.
    This prevents conflicts with our new physics setup.
    """
    # Remove RigidBody from root prim
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"[INFO] Removed existing RigidBodyAPI from {prim.GetPath()}")

    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

    # Remove Collision from all child prims (subtree)
    for child_prim in Usd.PrimRange(prim):
        if child_prim.IsA(UsdGeom.Gprim):
            if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                child_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                print(f"[INFO] Removed existing CollisionAPI from {child_prim.GetPath()}")

            if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                child_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

            if child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                child_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)


def _apply_physics_to_mesh(prim_path: str, mesh_path: str, mass: float) -> None:
    """
    Apply physics using SingleGeometryPrim wrapper (like DynamicCuboid) for proper collision.

    Args:
        prim_path: Root prim path
        mesh_path: Child mesh prim path
        mass: Object mass in kg
    """
    from isaacsim.core.prims import SingleGeometryPrim

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)

    # 1. Apply collision using SingleGeometryPrim on the mesh prim (like DynamicCuboid does)
    geometry_prim = SingleGeometryPrim(
        prim_path=mesh_path,
        name=f"geom_{mesh_path.split('/')[-1]}",
        collision=True,  # Enable collision
    )
    geometry_prim.initialize()  # Must initialize before use
    geometry_prim.set_collision_approximation("convexDecomposition")
    print(f"[INFO] Applied collision via SingleGeometryPrim to: {mesh_path}")

    # 2. Apply RigidBody to root
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    print(f"[INFO] Applied RigidBody to: {prim_path}")

    # 3. Apply mass to root
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(mass)
    print(f"[INFO] Applied mass ({mass}kg) to: {prim_path}")


def _apply_physics_minimal(prim_path: str, mass: float) -> None:
    """
    Apply minimal physics to YCB object using simple bounding cube collision.
    Using 'none' approximation (triangle mesh) for accurate collision but lightweight.

    Args:
        prim_path: Root prim path
        mass: Object mass in kg
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)

    # 1. Apply RigidBody
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    print(f"[INFO] Applied RigidBody to: {prim_path}")

    # 2. Apply mass
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(mass)
    print(f"[INFO] Applied mass ({mass}kg) to: {prim_path}")

    # 3. Apply collision using boundingCube (simpler than convexDecomposition)
    # This creates a box collision that is easier for RMPFlow to handle
    collision_api = UsdPhysics.CollisionAPI.Apply(root_prim)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(root_prim)
    mesh_collision_api.CreateApproximationAttr().Set("boundingCube")
    print(f"[INFO] Applied collision (boundingCube) to: {prim_path}")


def _apply_physics_dynamicobject_pattern(prim_path: str, mesh_path: str, mass: float) -> None:
    """
    Apply physics to YCB object.

    Uses simple approach: both RigidBody and Collision on root prim.
    This is more stable than split approach.

    Args:
        prim_path: Root prim path for physics
        mesh_path: (unused - kept for API compatibility)
        mass: Object mass in kg
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)

    # 1. Apply RigidBody to the root prim
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)

    # Set mass
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(mass)
    print(f"[INFO] Applied rigid body (mass={mass}kg) to root: {prim_path}")

    # 2. Apply Collision to the root prim with convexDecomposition (more accurate for complex shapes)
    collision_api = UsdPhysics.CollisionAPI.Apply(root_prim)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(root_prim)
    mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")
    print(f"[INFO] Applied collision (convexDecomposition) to root: {prim_path}")

    # 3. Apply additional PhysX settings for stability
    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_api.CreateSleepThresholdAttr(0.001)  # Settle faster
    physx_api.CreateStabilizationThresholdAttr(0.001)
    print(f"[INFO] Applied PhysX stability settings")


def create_ycb_object(world: World, object_name: str) -> dict:
    """
    Create a YCB object using "sibling collision proxy" pattern.

    Structure:
        /World/YCB_Object_{name}       <- Parent Xform (RigidBody)
            /CollisionProxy            <- Invisible cube (collision only)
            /YCB_Visual                <- Visible YCB mesh (visual only)

    This ensures visibility inheritance doesn't hide YCB visual.

    Args:
        world: Isaac Sim World instance
        object_name: Key from YCB_OBJECT_CONFIGS (e.g., "010_potted_meat_can")

    Returns:
        dict: {"prim_path": str, "position": np.ndarray, "config": dict}
    """
    config = YCB_OBJECT_CONFIGS[object_name]

    # Get assets root path
    assets_root = get_assets_root_path()
    usd_path = f"{assets_root}/Isaac/Props/YCB/Axis_Aligned/{config['usd_file']}"

    # Spawn position
    spawn_position = OBJECT_POSITION.copy()

    stage = omni.usd.get_context().get_stage()

    # === Step 1: Create parent Xform with RigidBody ===
    parent_prim_path = f"/World/YCB_Object_{object_name}"
    parent_xform = UsdGeom.Xform.Define(stage, parent_prim_path)
    parent_prim = stage.GetPrimAtPath(parent_prim_path)

    # Apply RigidBody to parent
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(parent_prim)
    mass_api.CreateMassAttr(config['mass'])

    # Set parent position
    parent_xformable = UsdGeom.Xformable(parent_prim)
    parent_xformable.AddTranslateOp().Set(Gf.Vec3d(*spawn_position))

    # === Step 2: Create collision proxy as SIBLING child (invisible) ===
    proxy_prim_path = f"{parent_prim_path}/CollisionProxy"
    proxy_cube = UsdGeom.Cube.Define(stage, proxy_prim_path)
    proxy_cube.GetSizeAttr().Set(0.06)  # 6cm cube

    # Apply collision to proxy
    proxy_prim = stage.GetPrimAtPath(proxy_prim_path)
    collision_api = UsdPhysics.CollisionAPI.Apply(proxy_prim)
    collision_api.CreateCollisionEnabledAttr(True)

    # Make proxy invisible (won't affect sibling YCB)
    proxy_imageable = UsdGeom.Imageable(proxy_prim)
    proxy_imageable.MakeInvisible()
    print(f"[INFO] Created invisible collision proxy at {spawn_position}")

    # Apply friction material to collision proxy
    grip_friction = config.get("grip_friction", 1.0)
    material_path = f"/World/Materials/Mat_{object_name}_friction"
    create_grip_friction_material(stage, material_path, friction=grip_friction)
    apply_material_to_prim(stage, proxy_prim_path, material_path)
    print(f"[INFO] Applied friction material (friction={grip_friction}) to collision proxy")

    # === Step 3: Load YCB visual as SIBLING child (visible) ===
    ycb_prim_path = f"{parent_prim_path}/YCB_Visual"

    print(f"[INFO] Loading YCB visual: {config['display_name']}")
    print(f"[INFO] USD path: {usd_path}")
    add_reference_to_stage(usd_path=usd_path, prim_path=ycb_prim_path)

    # Get the YCB prim
    ycb_prim = stage.GetPrimAtPath(ycb_prim_path)

    if not ycb_prim.IsValid():
        print(f"[ERROR] Failed to load YCB object from {usd_path}")
        return None

    # Ensure YCB is visible (explicit)
    ycb_imageable = UsdGeom.Imageable(ycb_prim)
    ycb_imageable.MakeVisible()

    # Center YCB visual at parent origin
    ycb_xform = UsdGeom.Xformable(ycb_prim)
    ycb_xform.ClearXformOpOrder()
    ycb_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

    print(f"[INFO] Created '{config['display_name']}' with sibling collision proxy")

    return {
        "prim_path": parent_prim_path,  # Parent for position tracking
        "ycb_visual_path": ycb_prim_path,
        "proxy_path": proxy_prim_path,
        "position": spawn_position.copy(),
        "config": config,
        "object_name": object_name,
        "bounding_sphere": None,
    }


def delete_ycb_object(object_name: str) -> bool:
    """
    Delete a YCB object (parent with proxy + visual) from the stage.

    Args:
        object_name: Key from YCB_OBJECT_CONFIGS

    Returns:
        bool: True if deleted successfully
    """
    stage = omni.usd.get_context().get_stage()
    deleted = False

    # Delete parent (deletes both proxy and visual children)
    parent_path = f"/World/YCB_Object_{object_name}"
    parent_prim = stage.GetPrimAtPath(parent_path)
    if parent_prim.IsValid():
        stage.RemovePrim(parent_path)
        print(f"[INFO] Deleted YCB object: {object_name}")
        deleted = True

    # Also try old-style paths (for backwards compatibility)
    old_paths = [
        f"/World/YCB_Proxy_{object_name}",
        f"/World/YCB_{object_name}",
    ]
    for old_path in old_paths:
        old_prim = stage.GetPrimAtPath(old_path)
        if old_prim.IsValid():
            stage.RemovePrim(old_path)
            print(f"[INFO] Deleted old YCB object at {old_path}")
            deleted = True

    return deleted


def create_support_platform(world: World) -> None:
    """
    Create a support platform for YCB objects.
    Platform at EE reachable height to support objects stably.
    """
    # Platform top surface should be just below spawn height
    platform_z = SUPPORT_PLATFORM_HEIGHT - 0.01  # 1cm below platform height
    platform = FixedCuboid(
        prim_path="/World/SupportPlatform",
        name="support_platform",
        position=np.array([0.3, 0.3, platform_z]),
        scale=np.array([SUPPORT_PLATFORM_SIZE, SUPPORT_PLATFORM_SIZE, 0.02]),  # 2cm thick
        size=1.0,
        color=np.array([0.4, 0.4, 0.4]),  # Gray
    )
    world.scene.add(platform)
    print(f"[INFO] Created support platform at z={platform_z} (top surface at ~{platform_z + 0.01})")


def setup_object(world: World) -> dict:
    """Create a DynamicCuboid object matching the official example (legacy fallback).

    Returns:
        dict: Object info with prim_path and position
    """
    prim_path = "/World/PickCube"

    cube = DynamicCuboid(
        prim_path=prim_path,
        name="pick_cube",
        position=OBJECT_POSITION,
        scale=OBJECT_SCALE,
        size=1.0,
        color=OBJECT_COLOR,
    )
    world.scene.add(cube)

    print(f"[INFO] Created DynamicCuboid at position {OBJECT_POSITION}")
    return {"prim_path": prim_path, "position": OBJECT_POSITION.copy()}


def create_container(world: World, position: np.ndarray) -> None:
    """
    Create a container/tray using FixedCuboids (bottom + 4 walls).

    Args:
        world: World instance
        position: Container center position
    """
    size = CONTAINER_SIZE
    wall = CONTAINER_WALL_THICKNESS

    # Bottom plate
    bottom = FixedCuboid(
        prim_path="/World/Container/Bottom",
        name="container_bottom",
        position=position,
        scale=np.array([size[0], size[1], wall]),
        size=1.0,
        color=CONTAINER_COLOR,
    )
    world.scene.add(bottom)

    # Front wall (positive X)
    front_wall = FixedCuboid(
        prim_path="/World/Container/FrontWall",
        name="container_front",
        position=position + np.array([size[0]/2, 0, size[2]/2]),
        scale=np.array([wall, size[1], size[2]]),
        size=1.0,
        color=CONTAINER_COLOR,
    )
    world.scene.add(front_wall)

    # Back wall (negative X)
    back_wall = FixedCuboid(
        prim_path="/World/Container/BackWall",
        name="container_back",
        position=position + np.array([-size[0]/2, 0, size[2]/2]),
        scale=np.array([wall, size[1], size[2]]),
        size=1.0,
        color=CONTAINER_COLOR,
    )
    world.scene.add(back_wall)

    # Left wall (positive Y)
    left_wall = FixedCuboid(
        prim_path="/World/Container/LeftWall",
        name="container_left",
        position=position + np.array([0, size[1]/2, size[2]/2]),
        scale=np.array([size[0], wall, size[2]]),
        size=1.0,
        color=CONTAINER_COLOR,
    )
    world.scene.add(left_wall)

    # Right wall (negative Y)
    right_wall = FixedCuboid(
        prim_path="/World/Container/RightWall",
        name="container_right",
        position=position + np.array([0, -size[1]/2, size[2]/2]),
        scale=np.array([size[0], wall, size[2]]),
        size=1.0,
        color=CONTAINER_COLOR,
    )
    world.scene.add(right_wall)

    print(f"[INFO] Container created at {position}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run the Pick and Place simulation."""

    print("=" * 60)
    print("Isaac Sim 4.5 - Franka Pick and Place Demo")
    print("=" * 60)

    # Get assets root path
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit(1)

    print(f"[INFO] Assets root: {assets_root}")

    # Create World
    my_world = World(stage_units_in_meters=1.0)
    print("[INFO] World created")

    # === Scene Setup ===

    # 1. Environment Setup (Warehouse or Ground Plane)
    use_warehouse = args.warehouse
    if use_warehouse:
        warehouse_loaded = setup_warehouse_environment()
        if not warehouse_loaded:
            # Fallback to ground plane
            ground_plane = GroundPlane(
                prim_path="/World/GroundPlane",
                z_position=0,
                name="ground_plane",
            )
            print("[INFO] Ground plane added (warehouse fallback)")
    else:
        ground_plane = GroundPlane(
            prim_path="/World/GroundPlane",
            z_position=0,
            name="ground_plane",
        )
        print("[INFO] Ground plane added")

    # 2. Setup Lighting (after environment, before other objects)
    setup_lighting()

    # 3. Container (low position, won't interfere with robot)
    create_container(my_world, CONTAINER_POSITION)

    # 4. Franka Robot (using alternative fingers for better YCB object grasping)
    # Alt fingers have curved/scalloped inner surface - better for cylindrical objects
    assets_root_path = get_assets_root_path()
    franka_usd_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

    my_franka = Franka(
        prim_path="/World/Franka",
        name="my_franka",
        usd_path=franka_usd_path,
        position=ROBOT_POSITION,
        orientation=ROBOT_ORIENTATION,
        gripper_open_position=np.array([0.05, 0.05]),
        gripper_closed_position=np.array([0.0, 0.0]),  # Full close target → increased grip force for heavy objects
        deltas=np.array([0.01, 0.01]),  # Finer grip control for curved finger surfaces
    )
    my_world.scene.add(my_franka)
    print(f"[INFO] Franka robot (alt fingers) added at {ROBOT_POSITION}")

    # Create and apply high-friction material to gripper fingers
    franka_prim_path = "/World/Franka"
    stage = omni.usd.get_context().get_stage()
    create_grip_friction_material(stage, "/World/Materials/GripperFriction", friction=3.0)
    apply_material_to_prim(stage, f"{franka_prim_path}/panda_leftfinger", "/World/Materials/GripperFriction")
    apply_material_to_prim(stage, f"{franka_prim_path}/panda_rightfinger", "/World/Materials/GripperFriction")
    print("[INFO] Applied high-friction material to gripper fingers")

    # 5. Support Platform for YCB objects (disabled - using collision proxy approach now)
    # The proxy cube handles physics at the same position as DynamicCuboid
    # if USE_YCB_OBJECTS:
    #     create_support_platform(my_world)

    # 6. Pick Object
    current_object_name = SELECTED_YCB_OBJECT

    if USE_YCB_OBJECTS and current_object_name is not None:
        # Create YCB object on the support platform
        object_info = create_ycb_object(my_world, current_object_name)
        if object_info is None:
            print("[WARN] Failed to create YCB object, falling back to DynamicCuboid")
            object_info = setup_object(my_world)
            current_object_name = None
    else:
        # Use DynamicCuboid (stable, proven to work)
        object_info = setup_object(my_world)
        current_object_name = None

    # Setup Camera (after all scene objects are created)
    setup_camera()

    # Setup Hand-Mounted Camera (attached to panda_hand)
    hand_camera = None
    camera_processor = None
    camera_intrinsics = CameraIntrinsics()

    render_product_path = None
    if ENABLE_HAND_CAMERA:
        try:
            hand_camera, hand_camera_path, render_product_path = setup_hand_camera("/World/Franka")
            if hand_camera is not None:
                camera_processor = CameraProcessor(hand_camera)
                print(f"[INFO] Camera intrinsics: fx={camera_intrinsics.fx}, resolution={camera_intrinsics.width}x{camera_intrinsics.height}")
        except Exception as e:
            print(f"[WARN] Hand camera setup failed: {e}")
            hand_camera = None

    # Setup ROS2 Bridge Publisher (optional, enabled with --ros2 flag)
    # Requires proper environment setup via run_with_ros2.sh launcher
    ros2_publisher = None
    if ENABLE_ROS2:
        try:
            ros2_publisher = ROS2BridgePublisher(
                render_product_path=render_product_path,
                franka=my_franka
            )
            if ros2_publisher.enabled:
                print("[INFO] ROS2 Bridge active - use 'ros2 topic list' to see topics")
        except Exception as e:
            print(f"[WARN] ROS2 Bridge initialization failed: {e}")
            ros2_publisher = None

    # Setup Perception Pipeline (YOLO + 6DOF, optional)
    perception_pipeline = None
    if PERCEPTION_AVAILABLE and ENABLE_HAND_CAMERA:
        try:
            perception_pipeline = PerceptionPipeline(
                yolo_model_path="yolov8n-seg.pt",
                camera_intrinsics=(camera_intrinsics.fx, camera_intrinsics.fy,
                                   camera_intrinsics.cx, camera_intrinsics.cy),
                confidence_threshold=0.5,
                min_depth=0.1,
                max_depth=2.0,
                device="cuda:0",
            )
            if perception_pipeline.is_available:
                print("[INFO] Perception pipeline initialized (press 'D' to detect)")
            else:
                perception_pipeline = None
                print("[INFO] YOLO model not available - detection disabled")
        except Exception as e:
            print(f"[WARN] Perception pipeline setup failed: {e}")
            perception_pipeline = None

    # === Initialize Simulation ===
    # IMPORTANT: Set gripper default state before reset (critical for proper gripper behavior)
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    print("[INFO] Gripper default state set to open")

    print("\n[INFO] Resetting world...")
    my_world.reset()

    # Step physics to let objects settle (50 steps like test_headless.py)
    print("[INFO] Stepping physics to settle...")
    for i in range(50):
        my_world.step(render=True)

    # Get articulation controller
    articulation_controller = my_franka.get_articulation_controller()

    # Debug: Check robot position before creating controller
    robot_pos, robot_ori = my_franka.get_world_pose()
    ee_pos, ee_ori = my_franka.end_effector.get_world_pose()
    print(f"[DEBUG] Robot world pose: position={robot_pos}, orientation={robot_ori}")
    print(f"[DEBUG] EE world pose BEFORE controller: position={ee_pos}")

    # Create Pick and Place Controller
    # IMPORTANT: Controller captures robot base pose during init
    my_controller = PickPlaceController(
        name="pick_place_controller",
        gripper=my_franka.gripper,
        robot_articulation=my_franka,
    )
    print("[INFO] PickPlaceController initialized")

    # Debug: Verify controller's internal state
    print(f"[DEBUG] Controller end_effector_initial_height (h1): {my_controller._h1}")

    # Verify RMPFlow's robot base pose
    rmpflow = my_controller._cspace_controller
    print(f"[DEBUG] RMPFlow default robot position: {rmpflow._default_position}")
    print(f"[DEBUG] RMPFlow default robot orientation: {rmpflow._default_orientation}")

    # Initialize RMPFlow obstacle manager for YCB objects
    obstacle_manager = None
    if USE_YCB_OBJECTS and object_info is not None and object_info.get("bounding_sphere"):
        obstacle_manager = RMPFlowObstacleManager(my_controller)
        obstacle_manager.register_obstacle(object_info["bounding_sphere"])
        print("[INFO] RMPFlow obstacle manager initialized")

    # State tracking
    reset_needed = False
    task_completed = False
    task_started = False  # Wait for 'S' key press
    step_count = 0
    max_steps = 3000  # Limit steps for testing

    # Create verification tracker for monitoring robot behavior
    verifier = VerificationTracker(
        franka=my_franka,
        object_prim_path=object_info["prim_path"],
        pick_target=object_info["position"].copy(),  # Will be updated below
        place_target=PLACE_POSITION,
    )
    verifier.initialize()  # Initialize after world.reset()

    # Get the object's ACTUAL position after physics settled (not the pre-stored position)
    # This is critical because physics can cause the object to shift slightly
    object_position = verifier.get_object_position().copy()
    # Note: With collision proxy approach, YCB objects use the same position as DynamicCuboid
    # No pick_height_offset needed since proxy cube behaves identically
    print(f"[DEBUG] Object ACTUAL position after physics: {object_position}")
    verifier.pick_target = object_position  # Update the verifier's pick target too

    # Create keyboard controller for simulation and joint control
    sim_controller = SimulationController(franka=my_franka)

    # Debug output for position verification
    print("\n" + "-" * 60)
    print("[DEBUG] Object position for picking: ", object_position)
    print("[DEBUG] Container position for placing:", PLACE_POSITION)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Simulation ready!")
    print("  - Press 'M' key to toggle MANUAL JOINT CONTROL mode")
    print("  - Press 'S' key to start the Pick and Place task")
    if USE_YCB_OBJECTS:
        print("  - Press '1' for Potted Meat Can (65-70mm)")
        print("  - Press '2' for Tuna Fish Can (~35mm)")
        print("  - Press '3' for Foam Brick (38-50mm)")
    if perception_pipeline is not None:
        print("  - Press 'D' for YOLO object detection")
    print("  - Press 'R' to reset")
    if current_object_name is not None and current_object_name in YCB_OBJECT_CONFIGS:
        print(f"Current object: {YCB_OBJECT_CONFIGS[current_object_name]['display_name']}")
    else:
        print("Current object: DynamicCuboid (Blue Cube)")
    print("=" * 60 + "\n")

    # === Main Simulation Loop ===
    while simulation_app.is_running():
        # Step the world (physics + rendering)
        my_world.step(render=not IS_HEADLESS)  # Don't render in headless mode

        # Handle stop condition
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            task_started = False

        # Only process when simulation is playing
        if my_world.is_playing():
            # Get camera frames (if hand camera is enabled)
            rgb_data, depth_data = None, None
            if camera_processor is not None:
                rgb_data, depth_data = camera_processor.get_frame()
                # Debug output every 100 frames
                if rgb_data is not None and camera_processor.frame_count % 100 == 0:
                    depth_info = f"Depth shape={depth_data.shape}" if depth_data is not None else "Depth=None"
                    print(f"[CAMERA] Frame {camera_processor.frame_count}: RGB shape={rgb_data.shape}, {depth_info}")

            # Update ROS2 Bridge (if enabled) - publishing is automatic via OmniGraph
            if ros2_publisher is not None:
                ros2_publisher.update()
                # Publish depth with Jet colormap (manual - like final-manipulator)
                if depth_data is not None:
                    ros2_publisher.publish_depth(depth_data)

            # Handle YOLO detection request ('D' key)
            if sim_controller.detect_requested:
                sim_controller.detect_requested = False
                if perception_pipeline is not None and rgb_data is not None:
                    print("\n[DETECT] Running YOLO detection...")
                    # Get camera pose for 6DOF
                    cam_pos, cam_ori = None, None
                    if hand_camera is not None:
                        try:
                            cam_pos = hand_camera.get_world_pose()[0]
                            cam_ori = hand_camera.get_world_pose()[1]
                        except:
                            pass
                    # Run detection
                    detections = perception_pipeline.process(
                        rgb_image=rgb_data[:, :, :3] if rgb_data.shape[2] == 4 else rgb_data,
                        depth_image=depth_data,
                        camera_position=cam_pos,
                        camera_orientation=cam_ori,
                    )
                    # Print results
                    print(f"[DETECT] Found {len(detections)} objects:")
                    for i, det in enumerate(detections):
                        pos_str = ""
                        if det.get("position_6dof") is not None:
                            pos = det["position_6dof"]
                            pos_str = f", 6DOF pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                        print(f"  [{i+1}] {det['class_name']} (conf={det['confidence']:.2f}){pos_str}")
                    print("")
                elif perception_pipeline is None:
                    print("[DETECT] Perception pipeline not available")
                elif rgb_data is None:
                    print("[DETECT] No camera data available")

            # Handle reset request from 'R' key
            if sim_controller.reset_requested:
                reset_needed = True
                sim_controller.reset_requested = False

            # Handle YCB object change request (1, 2, 3 keys) - only if YCB is enabled
            if USE_YCB_OBJECTS and sim_controller.object_change_requested is not None:
                new_object_name = sim_controller.object_change_requested
                sim_controller.object_change_requested = None

                if new_object_name != current_object_name:
                    print(f"\n[INFO] Changing object from {current_object_name} to {new_object_name}")

                    # Delete current object if it's a YCB object
                    if current_object_name is not None:
                        delete_ycb_object(current_object_name)

                    # Create new object
                    current_object_name = new_object_name
                    object_info = create_ycb_object(my_world, current_object_name)

                    if object_info is None:
                        print("[ERROR] Failed to create new YCB object, keeping previous")
                        continue

                    # Reset simulation state
                    reset_needed = True
                    print(f"[INFO] Now using: {YCB_OBJECT_CONFIGS[current_object_name]['display_name']}")
            else:
                # Clear the request if YCB is disabled
                sim_controller.object_change_requested = None

            # Handle reset
            if reset_needed:
                my_world.reset()
                my_controller.reset()

                # Step physics to let objects settle (same as initial setup)
                for _ in range(50):
                    my_world.step(render=True)

                # Reset obstacle manager state
                if obstacle_manager:
                    obstacle_manager.reset()

                # Update verifier with new object path
                verifier.object_xform = SingleXFormPrim(prim_path=object_info["prim_path"])
                verifier.initialize()  # Re-initialize verifier after reset
                verifier.last_phase = -1  # Reset phase tracking

                # Re-calculate pick position (AFTER physics settled)
                object_position = verifier.get_object_position().copy()
                if current_object_name is not None and current_object_name in YCB_OBJECT_CONFIGS:
                    config = YCB_OBJECT_CONFIGS[current_object_name]
                    # No pick_height_offset needed with collision proxy approach
                    print(f"[INFO] Simulation reset - Object: {config['display_name']}")
                else:
                    print("[INFO] Simulation reset - Object: DynamicCuboid")
                verifier.pick_target = object_position
                print(f"[INFO] Pick position: {object_position}")

                reset_needed = False
                task_completed = False
                task_started = False
                step_count = 0

                # Clear start request so user must press 'S' again
                sim_controller.start_requested = False
                sim_controller.reset_requested = False

                # Show ready message (same as initial startup)
                print("\n" + "=" * 60)
                print("Simulation ready!")
                print("  - Press 'M' key to toggle MANUAL JOINT CONTROL mode")
                print("  - Press 'S' key to start the Pick and Place task")
                if USE_YCB_OBJECTS:
                    print("  - Press '1' for Potted Meat Can (65-70mm)")
                    print("  - Press '2' for Tuna Fish Can (~35mm)")
                    print("  - Press '3' for Foam Brick (38-50mm)")
                if perception_pipeline is not None:
                    print("  - Press 'D' for YOLO object detection")
                print("  - Press 'R' to reset")
                if current_object_name is not None and current_object_name in YCB_OBJECT_CONFIGS:
                    print(f"Current object: {YCB_OBJECT_CONFIGS[current_object_name]['display_name']}")
                else:
                    print("Current object: DynamicCuboid (Blue Cube)")
                print("=" * 60 + "\n")

            # Manual mode: skip auto pick-place logic, just sync and continue
            if sim_controller.manual_mode:
                sim_controller.sync_from_robot()
                continue

            # Wait for user to press 'S' or Play button (auto-started in headless mode)
            if not task_started:
                if IS_HEADLESS or sim_controller.start_requested:
                    task_started = True
                    print("[INFO] Task started!")
                    print(f"[INFO] Robot will move to pick object at: {object_position}")
                else:
                    continue  # Keep waiting

            # Exit after max steps in headless mode
            if step_count >= max_steps:
                print(f"[INFO] Reached max steps ({max_steps}), exiting...")
                break

            # Skip if task already completed
            if task_completed:
                continue

            # Compute controller action using the fixed object position
            current_joints = my_franka.get_joint_positions()

            # Debug: Log positions being passed to controller on first few calls
            if step_count < 3:
                current_ee_pos, _ = my_franka.end_effector.get_world_pose()
                print(f"\n[DEBUG] Controller call #{step_count}:")
                print(f"  picking_position: {object_position}")
                print(f"  placing_position: {PLACE_POSITION}")
                print(f"  current_ee_position: {current_ee_pos}")
                print(f"  current_joint_positions: {current_joints[:4]}...")  # First 4 joints
                print(f"  current_event: {my_controller._event}")

            # Phase-based RMPFlow obstacle control
            # Disable obstacle avoidance during pick phases (0-4) to allow gripper to reach object
            # Re-enable during place phases (5+) for safety
            if obstacle_manager:
                current_phase = my_controller.get_current_event()
                if current_phase in [0, 1, 2, 3, 4]:
                    obstacle_manager.disable_for_pick()
                elif current_phase >= 5:
                    obstacle_manager.enable_after_grasp()

            actions = my_controller.forward(
                picking_position=object_position,
                placing_position=PLACE_POSITION,
                current_joint_positions=current_joints,
                end_effector_offset=np.array([0, 0.005, 0]),  # Y offset (same as official example)
            )

            # Apply action to robot
            articulation_controller.apply_action(actions)

            # Status output and verification every 50 steps
            step_count += 1
            if step_count % 50 == 0:
                current_phase = my_controller.get_current_event()
                verifier.log_phase_status(current_phase, step_count)

            # Check if task is completed
            if my_controller.is_done():
                # Run final verification to check if object actually moved to container
                actual_success = verifier.verify_final_result()

                if actual_success:
                    print("Pick and Place task completed successfully!")
                    print("The cube has been placed in the container.")
                else:
                    print("Task state machine completed, but ACTUAL operation FAILED!")
                    print("Check the verification output above for details.")

                task_completed = True

    # Cleanup
    print("[INFO] Closing simulation...")

    # Shutdown ROS2 Bridge if enabled
    if ros2_publisher is not None:
        ros2_publisher.shutdown()

    simulation_app.close()


if __name__ == "__main__":
    main()
