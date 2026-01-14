#!/usr/bin/env python3
"""
Headless Test Script for Pick and Place Demo
Auto-starts without keyboard input for automated testing.
"""

# === CRITICAL: SimulationApp must be initialized FIRST ===
from isaacsim import SimulationApp

CONFIG = {
    "headless": True,  # Headless mode for automated testing
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(CONFIG)

# === Standard Library Imports ===
import sys
import numpy as np

# Force flush stdout for headless mode
import functools
print = functools.partial(print, flush=True)

# === Isaac Sim Imports ===
import carb
import omni.usd
from pxr import UsdLux, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, VisualCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import RigidPrim
from pxr import UsdPhysics, PhysxSchema
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SingleXFormPrim


# =============================================================================
# Configuration Constants (same as main script)
# =============================================================================

ROBOT_POSITION = np.array([0.0, 0.0, 0.0])
ROBOT_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])

# Pedestal configuration - thin columns that won't interfere with RMPFlow
# Position moved to [0.4, 0.15] for easier robot approach (more in front)
PEDESTAL_POSITION = np.array([0.4, 0.15, 0.0])  # More accessible position
PEDESTAL_HEIGHT = 0.27  # Higher - top plate at z=0.27
COLUMN_RADIUS = 0.008  # 1.6cm diameter columns
COLUMN_SPACING = 0.05  # 10cm total footprint (wider for stability)
TOP_PLATE_SIZE = 0.12  # 12cm x 12cm top plate (larger for cube stability)
TOP_PLATE_THICKNESS = 0.01  # 1cm thick
PEDESTAL_COLOR = np.array([0.5, 0.5, 0.5])

# Container position for placing - low, won't block robot
CONTAINER_POSITION = np.array([0.3, -0.3, 0.02])
CONTAINER_SIZE = np.array([0.12, 0.12, 0.05])
CONTAINER_WALL_THICKNESS = 0.008
CONTAINER_COLOR = np.array([0.4, 0.4, 0.5])

# Place position - at reachable height
PLACE_POSITION = np.array([0.3, -0.3, 0.27])

YCB_OBJECTS = {}

# Object position: testing without physics - floating at reachable height
# EE can reach z=0.287, so cube center at z=0.30
OBJECT_POSITION = np.array([0.3, 0.3, 0.30])

# No MIN_PICK_Z adjustment needed since object is at graspable height


# =============================================================================
# Verification Tracker
# =============================================================================

class VerificationTracker:
    def __init__(self, franka, object_prim_path: str, pick_target: np.ndarray, place_target: np.ndarray):
        self.franka = franka
        self.object_xform = SingleXFormPrim(prim_path=object_prim_path)
        self.pick_target = pick_target
        self.place_target = place_target
        self.initial_object_pos = None
        self.last_phase = -1
        self.position_tolerance = 0.05

    def initialize(self):
        self.object_xform.initialize()
        self.initial_object_pos, _ = self.object_xform.get_world_pose()
        print(f"[VERIFY] Initial object position: {self.initial_object_pos}")

    def get_ee_position(self) -> np.ndarray:
        pos, _ = self.franka.end_effector.get_world_pose()
        return pos

    def get_object_position(self) -> np.ndarray:
        pos, _ = self.object_xform.get_world_pose()
        return pos

    def log_phase_status(self, current_phase: int, step_count: int):
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
            print(f"  Banana Position: {object_pos}")
            print(f"  Pick Target:     {self.pick_target}")

            if current_phase == 1:
                dist_to_pick = np.linalg.norm(ee_pos[:2] - self.pick_target[:2])
                print(f"  [CHECK] EE XY distance to pick target: {dist_to_pick:.4f}m")
                if dist_to_pick > self.position_tolerance:
                    print(f"  [WARN] EE is NOT above object!")

            self.last_phase = current_phase

    def verify_final_result(self) -> bool:
        final_object_pos = self.get_object_position()
        dist_moved = np.linalg.norm(final_object_pos - self.initial_object_pos)
        dist_to_target = np.linalg.norm(final_object_pos[:2] - self.place_target[:2])

        print("\n" + "=" * 60)
        print("[FINAL VERIFICATION]")
        print(f"  Initial object position: {self.initial_object_pos}")
        print(f"  Final object position:   {final_object_pos}")
        print(f"  Place target position:   {self.place_target}")
        print(f"  Distance moved:          {dist_moved:.4f}m")
        print(f"  Distance to target:      {dist_to_target:.4f}m")

        success = dist_to_target < self.position_tolerance * 2

        if success:
            print("[RESULT] SUCCESS - Banana is in/near container!")
        else:
            print("[RESULT] FAILED - Banana did NOT reach container!")
            if dist_moved < 0.01:
                print("         Banana barely moved - gripper likely missed it")

        print("=" * 60 + "\n")
        return success


# =============================================================================
# Helper Functions
# =============================================================================

def create_pedestal(world: World, position: np.ndarray) -> None:
    """Create a 4-column pedestal that minimizes RMPFlow interference.

    The pedestal has thin columns at the corners and a small top plate.
    This design allows the robot arm to approach from any direction without
    the columns significantly interfering with RMPFlow path planning.
    """
    column_height = PEDESTAL_HEIGHT
    column_r = COLUMN_RADIUS

    # 4 columns at corners
    column_offsets = [
        np.array([COLUMN_SPACING, COLUMN_SPACING, column_height/2]),
        np.array([COLUMN_SPACING, -COLUMN_SPACING, column_height/2]),
        np.array([-COLUMN_SPACING, COLUMN_SPACING, column_height/2]),
        np.array([-COLUMN_SPACING, -COLUMN_SPACING, column_height/2]),
    ]

    for i, offset in enumerate(column_offsets):
        column = FixedCuboid(
            prim_path=f"/World/Pedestal/Column{i}",
            name=f"pedestal_column_{i}",
            position=position + offset,
            scale=np.array([column_r*2, column_r*2, column_height]),
            size=1.0,
            color=PEDESTAL_COLOR,
        )
        world.scene.add(column)

    # Top plate - small platform for the cube
    top_plate = FixedCuboid(
        prim_path="/World/Pedestal/TopPlate",
        name="pedestal_top",
        position=position + np.array([0, 0, column_height + TOP_PLATE_THICKNESS/2]),
        scale=np.array([TOP_PLATE_SIZE, TOP_PLATE_SIZE, TOP_PLATE_THICKNESS]),
        size=1.0,
        color=PEDESTAL_COLOR,
    )
    world.scene.add(top_plate)

    print(f"[INFO] Created pedestal at {position}, top plate at z={column_height + TOP_PLATE_THICKNESS/2}")


def create_container(world: World, position: np.ndarray) -> None:
    size = CONTAINER_SIZE
    wall = CONTAINER_WALL_THICKNESS

    bottom = FixedCuboid(
        prim_path="/World/Container/Bottom",
        name="container_bottom",
        position=position,
        scale=np.array([size[0], size[1], wall]),
        size=1.0,
        color=CONTAINER_COLOR,
    )
    world.scene.add(bottom)

    # Walls
    for name, offset in [
        ("FrontWall", np.array([size[0]/2, 0, size[2]/2])),
        ("BackWall", np.array([-size[0]/2, 0, size[2]/2])),
        ("LeftWall", np.array([0, size[1]/2, size[2]/2])),
        ("RightWall", np.array([0, -size[1]/2, size[2]/2])),
    ]:
        is_x = "Front" in name or "Back" in name
        scale = np.array([wall, size[1], size[2]]) if is_x else np.array([size[0], wall, size[2]])
        FixedCuboid(
            prim_path=f"/World/Container/{name}",
            name=f"container_{name.lower()}",
            position=position + offset,
            scale=scale,
            size=1.0,
            color=CONTAINER_COLOR,
        )


def setup_object(world: World) -> dict:
    """Create a DynamicCuboid object matching the official example."""
    prim_path = "/World/PickCube"

    # Match official example: cube at [0.3, 0.3, 0.3], size ~5cm
    cube = DynamicCuboid(
        prim_path=prim_path,
        name="pick_cube",
        position=OBJECT_POSITION,
        scale=np.array([0.0515, 0.0515, 0.0515]),  # Official example size
        size=1.0,
        color=np.array([0, 0, 1]),  # Blue like official example
    )
    world.scene.add(cube)

    print(f"[INFO] Created DynamicCuboid at position {OBJECT_POSITION}")
    return {"prim_path": prim_path, "position": OBJECT_POSITION.copy()}


# =============================================================================
# Main Function
# =============================================================================

def main():
    print("=" * 60)
    print("Isaac Sim 4.5 - Headless Pick and Place Test")
    print("=" * 60)

    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit(1)

    my_world = World(stage_units_in_meters=1.0)
    print("[INFO] World created")

    # Scene Setup
    GroundPlane(prim_path="/World/GroundPlane", z_position=0, name="ground_plane")

    # Skip pedestal for position testing - visual cube doesn't need support
    # create_pedestal(my_world, PEDESTAL_POSITION)
    table = None

    create_container(my_world, CONTAINER_POSITION)

    my_franka = Franka(
        prim_path="/World/Franka",
        name="my_franka",
        position=ROBOT_POSITION,
        orientation=ROBOT_ORIENTATION,
    )
    my_world.scene.add(my_franka)

    object_info = setup_object(my_world)

    # IMPORTANT: Set gripper default state before reset (from official example)
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    print("[INFO] Gripper default state set to open")

    # Initialize
    print("\n[INFO] Resetting world...")
    my_world.reset()

    # Let physics settle - more steps for cube stability on pedestal
    print("[INFO] Stepping physics to settle...")
    for i in range(50):
        my_world.step(render=False)

    articulation_controller = my_franka.get_articulation_controller()

    my_controller = PickPlaceController(
        name="pick_place_controller",
        gripper=my_franka.gripper,
        robot_articulation=my_franka,
    )
    print("[INFO] PickPlaceController initialized")

    # === RMPFlow Obstacle Disabling ===
    # Access the internal RMPFlow motion policy and disable table as obstacle
    if table is not None:
        try:
            rmpflow_controller = my_controller._cspace_controller
            print(f"[INFO] RMPFlow controller type: {type(rmpflow_controller)}")

            # Access the actual RmpFlow motion policy object
            rmpflow = rmpflow_controller.rmp_flow
            print(f"[INFO] RmpFlow motion policy type: {type(rmpflow)}")

            # First, add the table as an obstacle explicitly (required before disabling)
            add_result = rmpflow.add_obstacle(table, static=True)
            print(f"[INFO] add_obstacle(table) result: {add_result}")

            # Now disable it from collision avoidance
            disable_result = rmpflow.disable_obstacle(table)
            print(f"[INFO] disable_obstacle(table) result: {disable_result}")

        except Exception as e:
            print(f"[WARN] Could not disable table from RMPFlow: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing without RMPFlow obstacle modification...")
    else:
        print("[INFO] No table - skipping RMPFlow obstacle configuration")

    # Create verifier
    verifier = VerificationTracker(
        franka=my_franka,
        object_prim_path=object_info["prim_path"],
        pick_target=object_info["position"].copy(),
        place_target=PLACE_POSITION,
    )
    verifier.initialize()

    # CRITICAL: Get actual position after physics settled
    object_position = verifier.get_object_position().copy()
    print(f"[DEBUG] Banana ACTUAL position after physics: {object_position}")
    verifier.pick_target = object_position
    print(f"[DEBUG] Place target: {PLACE_POSITION}")

    # Run simulation
    step_count = 0
    max_steps = 2000

    print("\n[INFO] Starting Pick and Place task...")

    while simulation_app.is_running() and step_count < max_steps:
        my_world.step(render=False)

        if my_world.is_playing():
            current_joints = my_franka.get_joint_positions()

            if step_count < 3:
                current_ee_pos, _ = my_franka.end_effector.get_world_pose()
                print(f"\n[DEBUG] Step {step_count}:")
                print(f"  picking_position: {object_position}")
                print(f"  current_ee_position: {current_ee_pos}")

            actions = my_controller.forward(
                picking_position=object_position,
                placing_position=PLACE_POSITION,
                current_joint_positions=current_joints,
                end_effector_offset=np.array([0, 0.005, 0]),
            )

            articulation_controller.apply_action(actions)

            step_count += 1
            if step_count % 100 == 0:
                current_phase = my_controller.get_current_event()
                verifier.log_phase_status(current_phase, step_count)

            if my_controller.is_done():
                actual_success = verifier.verify_final_result()
                print(f"\n{'='*60}")
                if actual_success:
                    print("TEST PASSED: Pick and Place successful!")
                else:
                    print("TEST FAILED: Pick and Place did not complete correctly")
                print(f"{'='*60}\n")
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
