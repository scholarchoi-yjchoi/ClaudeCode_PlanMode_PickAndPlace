#!/usr/bin/env python3
"""
Isaac Sim 4.5 Pick and Place Demo
=================================
Robot: Franka Panda
Task: Pick up a cube and place it in a container
Environment: Ground plane, container, DynamicCuboid (no table - prevents RMPFlow interference)

Usage:
    cd ~/isaac_sim_4.5
    ./python.sh /home/yjchoi/ClaudeCode_PlanMode_PickAndPlace/franka_ycb_pick_place.py
"""

# === CRITICAL: SimulationApp must be initialized FIRST ===
from isaacsim import SimulationApp

CONFIG = {
    "headless": False,  # Set to True for automated testing
    "width": 1280,
    "height": 720,
    "window_title": "Franka Pick and Place Demo",
}

# Track if running in headless mode for auto-start
IS_HEADLESS = CONFIG["headless"]

simulation_app = SimulationApp(CONFIG)

# === Standard Library Imports ===
import sys
import functools
import numpy as np

# Force flush stdout for headless mode
print = functools.partial(print, flush=True)

# === Isaac Sim Imports ===
import carb
import carb.input
import omni.usd
import omni.appwindow
from pxr import UsdLux, UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import RigidPrim
from pxr import UsdPhysics, PhysxSchema
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SingleXFormPrim


# =============================================================================
# Configuration Constants
# =============================================================================

# Robot configuration
ROBOT_POSITION = np.array([0.0, 0.0, 0.0])
ROBOT_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz quaternion

# YCB Object Configurations
# Key: 1, 2, 3 for keyboard selection
YCB_OBJECT_CONFIGS = {
    "010_potted_meat_can": {
        "key": "1",
        "usd_file": "010_potted_meat_can.usd",
        "display_name": "Potted Meat Can",
        "mass": 0.37,
        "pick_height_offset": 0.0,  # No offset needed, pick at object center
        "spawn_height": 0.10,  # Spawn low, will settle on ground
        "spawn_xy": np.array([0.3, 0.3]),
    },
    "011_banana": {
        "key": "2",
        "usd_file": "011_banana.usd",
        "display_name": "Banana",
        "mass": 0.12,
        "pick_height_offset": 0.0,
        "spawn_height": 0.10,
        "spawn_xy": np.array([0.3, 0.3]),
    },
    "040_large_marker": {
        "key": "3",
        "usd_file": "040_large_marker.usd",
        "display_name": "Large Marker",
        "mass": 0.02,
        "pick_height_offset": 0.0,
        "spawn_height": 0.10,
        "spawn_xy": np.array([0.3, 0.3]),
    },
}

# Default object selection (None = use DynamicCuboid, or YCB object name)
# YCB objects have physics instability issues - use DynamicCuboid by default
SELECTED_YCB_OBJECT = "010_potted_meat_can"  # YCB object to use when enabled
USE_YCB_OBJECTS = False  # Master switch for YCB objects (disabled due to physics issues)

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


# =============================================================================
# Keyboard Input Controller
# =============================================================================

class SimulationController:
    """Handles keyboard input for simulation control."""

    def __init__(self):
        self.start_requested = False
        self.object_change_requested = None  # Stores new object name when 1,2,3 pressed
        self.reset_requested = False
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
            print("[INFO] Keys: S=Start, R=Reset, 1=PottedMeatCan, 2=Banana, 3=LargeMarker")
        except Exception as e:
            carb.log_warn(f"Could not setup keyboard: {e}")

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name

            if key_name == "S":
                self.start_requested = True
                print("\n[INFO] 'S' key pressed - Starting simulation!")
            elif key_name == "R":
                self.reset_requested = True
                print("\n[INFO] 'R' key pressed - Reset requested!")
            elif key_name == "1":
                self.object_change_requested = "010_potted_meat_can"
                print("\n[INFO] '1' key pressed - Selecting Potted Meat Can")
            elif key_name == "2":
                self.object_change_requested = "011_banana"
                print("\n[INFO] '2' key pressed - Selecting Banana")
            elif key_name == "3":
                self.object_change_requested = "040_large_marker"
                print("\n[INFO] '3' key pressed - Selecting Large Marker")
        return True


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
# Helper Functions
# =============================================================================

def apply_physics_to_ycb(prim, mass: float) -> None:
    """
    Apply physics properties to a YCB object.

    Args:
        prim: USD prim to apply physics to
        mass: Mass in kg
    """
    # 1. RigidBody API
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)

    # 2. Collision API
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(True)

    # 3. MeshCollision API - use convexHull for stability
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr().Set("convexHull")

    # 4. Mass API
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)

    # 5. PhysX settings for stability
    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_api.CreateSleepThresholdAttr(0.001)  # Settle faster

    print(f"[INFO] Applied physics to YCB object (mass={mass}kg, collision=convexHull)")


def create_ycb_object(world: World, object_name: str) -> dict:
    """
    Create a YCB object with physics properties.

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
    prim_path = f"/World/YCB_{object_name}"

    # Spawn position
    spawn_xy = config["spawn_xy"]
    spawn_height = config["spawn_height"]
    spawn_position = np.array([spawn_xy[0], spawn_xy[1], spawn_height])

    # Load USD reference
    print(f"[INFO] Loading YCB object: {config['display_name']}")
    print(f"[INFO] USD path: {usd_path}")
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    # Get the prim and set transform
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"[ERROR] Failed to load YCB object from {usd_path}")
        return None

    # Set position using XformOp
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*spawn_position))

    # Apply physics
    apply_physics_to_ycb(prim, config["mass"])

    print(f"[INFO] Created YCB object '{config['display_name']}' at {spawn_position}")

    return {
        "prim_path": prim_path,
        "position": spawn_position.copy(),
        "config": config,
        "object_name": object_name,
    }


def delete_ycb_object(object_name: str) -> bool:
    """
    Delete a YCB object from the stage.

    Args:
        object_name: Key from YCB_OBJECT_CONFIGS

    Returns:
        bool: True if deleted successfully
    """
    prim_path = f"/World/YCB_{object_name}"
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if prim.IsValid():
        stage.RemovePrim(prim_path)
        print(f"[INFO] Deleted YCB object at {prim_path}")
        return True
    return False


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

    # 1. Ground Plane
    ground_plane = GroundPlane(
        prim_path="/World/GroundPlane",
        z_position=0,
        name="ground_plane",
    )
    print("[INFO] Ground plane added")

    # 2. Setup Lighting (after ground plane, before other objects)
    setup_lighting()

    # 3. Container (low position, won't interfere with robot)
    create_container(my_world, CONTAINER_POSITION)

    # 4. Franka Robot
    my_franka = Franka(
        prim_path="/World/Franka",
        name="my_franka",
        position=ROBOT_POSITION,
        orientation=ROBOT_ORIENTATION,
    )
    my_world.scene.add(my_franka)
    print(f"[INFO] Franka robot added at {ROBOT_POSITION}")

    # 5. Pick Object
    current_object_name = SELECTED_YCB_OBJECT

    if USE_YCB_OBJECTS and current_object_name is not None:
        # Create YCB object (on ground, no support platform needed)
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
    if current_object_name is not None and current_object_name in YCB_OBJECT_CONFIGS:
        config = YCB_OBJECT_CONFIGS[current_object_name]
        object_position[2] += config["pick_height_offset"]  # Adjust pick height for YCB object
        print(f"[DEBUG] Object ACTUAL position after physics: {verifier.get_object_position()}")
        print(f"[DEBUG] Pick position (with height offset {config['pick_height_offset']}): {object_position}")
    else:
        print(f"[DEBUG] Object ACTUAL position after physics: {object_position}")
    verifier.pick_target = object_position  # Update the verifier's pick target too

    # Create keyboard controller for 'S' key input
    sim_controller = SimulationController()

    # Debug output for position verification
    print("\n" + "-" * 60)
    print("[DEBUG] Object position for picking: ", object_position)
    print("[DEBUG] Container position for placing:", PLACE_POSITION)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Simulation ready!")
    print("  - Press 'S' key to start the Pick and Place task")
    if USE_YCB_OBJECTS:
        print("  - Press '1' for Potted Meat Can")
        print("  - Press '2' for Banana")
        print("  - Press '3' for Large Marker")
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

                # Update verifier with new object path
                verifier.object_xform = SingleXFormPrim(prim_path=object_info["prim_path"])
                verifier.initialize()  # Re-initialize verifier after reset
                verifier.last_phase = -1  # Reset phase tracking

                # Re-calculate pick position
                object_position = verifier.get_object_position().copy()
                if current_object_name is not None and current_object_name in YCB_OBJECT_CONFIGS:
                    config = YCB_OBJECT_CONFIGS[current_object_name]
                    object_position[2] += config["pick_height_offset"]  # Adjust pick height
                    print(f"[INFO] Simulation reset - Object: {config['display_name']}")
                else:
                    print("[INFO] Simulation reset - Object: DynamicCuboid")
                verifier.pick_target = object_position
                print(f"[INFO] Pick position: {object_position}")

                reset_needed = False
                task_completed = False
                task_started = False
                step_count = 0
                print(f"[INFO] Simulation reset - Object: {config['display_name']}")
                print(f"[INFO] Pick position: {object_position}")

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
    simulation_app.close()


if __name__ == "__main__":
    main()
