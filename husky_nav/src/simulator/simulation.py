"""
PyBullet simulation wrapper for the Husky robot.

Responsibilities:
  - Manage the physics world (gravity, timestep, ground plane)
  - Load / reload the Husky URDF at an arbitrary pose
  - Create / remove static box obstacles
  - Accept differential-drive velocity commands and convert to per-wheel targets
  - Report the robot's current pose (x, y, yaw)
"""

import os
import tempfile

import numpy as np
import pybullet as p
import pybullet_data

from .husky_urdf import get_husky_urdf


# ── Physical constants ────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.1651   # metres
TRACK_WIDTH  = 0.5708   # metres  (centre-to-centre of left/right wheels)
MAX_WHEEL_FORCE = 400.0  # N  (applied to each wheel joint)


class Simulation:
    """Thin wrapper around a single PyBullet DIRECT physics client."""

    def __init__(self) -> None:
        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)

        # Ground plane
        self._plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self._client
        )

        # Write Husky URDF once to a temp file (PyBullet needs a file path)
        fd, self._urdf_path = tempfile.mkstemp(suffix=".urdf")
        with os.fdopen(fd, "w") as fh:
            fh.write(get_husky_urdf())

        self._husky_id: int | None = None
        self._left_joints:  list[int] = []
        self._right_joints: list[int] = []
        self._obstacle_ids: list[int] = []

    # ── Robot loading ─────────────────────────────────────────────────────────

    def load_husky(
        self, start_x: float, start_y: float, start_yaw: float = 0.0
    ) -> None:
        """Spawn (or re-spawn) the Husky at the given world pose."""
        if self._husky_id is not None:
            p.removeBody(self._husky_id, physicsClientId=self._client)

        pos = [start_x, start_y, WHEEL_RADIUS + 0.005]   # tiny clearance
        orn = p.getQuaternionFromEuler(
            [0.0, 0.0, start_yaw], physicsClientId=self._client
        )

        self._husky_id = p.loadURDF(
            self._urdf_path,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self._client,
        )

        # Identify wheel joints by name; disable passive joint motors first
        self._left_joints.clear()
        self._right_joints.clear()

        n = p.getNumJoints(self._husky_id, physicsClientId=self._client)
        for i in range(n):
            info      = p.getJointInfo(self._husky_id, i, physicsClientId=self._client)
            jtype     = info[2]
            jname     = info[1].decode()

            if jtype != p.JOINT_REVOLUTE:
                continue

            # Kill the default velocity motor (it acts like a brake)
            p.setJointMotorControl2(
                self._husky_id, i, p.VELOCITY_CONTROL,
                force=0, physicsClientId=self._client,
            )

            if "left" in jname and "wheel" in jname:
                self._left_joints.append(i)
            elif "right" in jname and "wheel" in jname:
                self._right_joints.append(i)

        # Good wheel / ground contact
        for i in range(n):
            p.changeDynamics(
                self._husky_id, i,
                lateralFriction=1.2,
                spinningFriction=0.1,
                rollingFriction=0.05,
                physicsClientId=self._client,
            )
        # Also for the base link (-1)
        p.changeDynamics(
            self._husky_id, -1,
            linearDamping=0.2,
            angularDamping=0.2,
            physicsClientId=self._client,
        )

    # ── Obstacle management ───────────────────────────────────────────────────

    def set_obstacles(self, obstacles: list[tuple]) -> None:
        """
        Replace all current obstacles.

        obstacles : list of (cx, cy, width, height)  – all in world metres.
        """
        for oid in self._obstacle_ids:
            p.removeBody(oid, physicsClientId=self._client)
        self._obstacle_ids.clear()

        for (cx, cy, w, h) in obstacles:
            half = [w / 2, h / 2, 0.5]

            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self._client
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=[0.75, 0.20, 0.20, 1.0],
                physicsClientId=self._client,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[cx, cy, 0.5],
                physicsClientId=self._client,
            )
            self._obstacle_ids.append(body)

    # ── Control ───────────────────────────────────────────────────────────────

    def set_cmd_vel(self, linear: float, angular: float) -> None:
        """
        Apply a differential-drive velocity command to the four wheels.

        linear  – forward velocity (m/s)
        angular – yaw rate (rad/s, positive = counter-clockwise)
        """
        if self._husky_id is None:
            return

        # Differential-drive inverse kinematics → wheel angular velocities
        v_l = (linear - angular * TRACK_WIDTH / 2) / WHEEL_RADIUS
        v_r = (linear + angular * TRACK_WIDTH / 2) / WHEEL_RADIUS

        for j in self._left_joints:
            p.setJointMotorControl2(
                self._husky_id, j, p.VELOCITY_CONTROL,
                targetVelocity=v_l,
                force=MAX_WHEEL_FORCE,
                physicsClientId=self._client,
            )
        for j in self._right_joints:
            p.setJointMotorControl2(
                self._husky_id, j, p.VELOCITY_CONTROL,
                targetVelocity=v_r,
                force=MAX_WHEEL_FORCE,
                physicsClientId=self._client,
            )

    def stop(self) -> None:
        self.set_cmd_vel(0.0, 0.0)

    # ── State query ───────────────────────────────────────────────────────────

    def get_robot_pose(self) -> tuple[float, float, float] | None:
        """Return (x, y, yaw) of the robot base, or None if not loaded."""
        if self._husky_id is None:
            return None
        pos, orn = p.getBasePositionAndOrientation(
            self._husky_id, physicsClientId=self._client
        )
        _, _, yaw = p.getEulerFromQuaternion(orn, physicsClientId=self._client)
        return float(pos[0]), float(pos[1]), float(yaw)

    def get_robot_velocity(self) -> tuple[float, float]:
        """Return (linear, angular) body-frame velocity for monitoring."""
        if self._husky_id is None:
            return 0.0, 0.0
        lin, ang = p.getBaseVelocity(
            self._husky_id, physicsClientId=self._client
        )
        return float(np.hypot(lin[0], lin[1])), float(ang[2])

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance physics by one timestep (1/240 s)."""
        p.stepSimulation(physicsClientId=self._client)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        try:
            p.disconnect(physicsClientId=self._client)
        except Exception:
            pass
        try:
            os.unlink(self._urdf_path)
        except Exception:
            pass
