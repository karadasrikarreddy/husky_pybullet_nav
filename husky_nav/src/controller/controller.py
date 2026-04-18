"""
Closed-loop controller for a differential-drive robot.

Algorithm: Pure Pursuit with adaptive lookahead.
  1. Walk along the planned path to find a "lookahead point" that lies
     approximately `lookahead_dist` metres ahead of the robot.
  2. Compute the signed curvature κ = 2·sin(α) / L  (α = heading error to
     lookahead point, L = lookahead distance).
  3. Derive angular velocity ω = v · κ.
  4. Clamp both linear and angular velocities to safe limits.
  5. Slow down proportionally as the robot nears the final goal.

The controller advances an internal waypoint index to track progress along
the path and declares success when the robot is within `goal_tolerance` of
the last waypoint.
"""

import numpy as np
from typing import Optional


class PurePursuitController:
    """Pure-pursuit path-tracking controller for a diff-drive robot."""

    def __init__(
        self,
        lookahead_dist: float = 0.80,     # m
        max_linear_vel: float = 1.20,     # m/s
        max_angular_vel: float = 2.00,    # rad/s
        goal_tolerance: float = 0.30,     # m
        slowdown_radius: float = 1.20,    # m  – start decelerating here
        min_linear_vel: float = 0.10,     # m/s – never go slower than this
    ) -> None:
        self.ld            = lookahead_dist
        self.v_max         = max_linear_vel
        self.w_max         = max_angular_vel
        self.goal_tol      = goal_tolerance
        self.slowdown_r    = slowdown_radius
        self.v_min         = min_linear_vel

        self._path: list[tuple[float, float]] = []
        self._wp_idx: int = 0
        self._done: bool  = False

    # ── Path setter ───────────────────────────────────────────────────────────

    def set_path(self, path: list[tuple[float, float]]) -> None:
        """Load a new path (list of world-frame (x, y) waypoints)."""
        self._path   = list(path)
        self._wp_idx = 0
        self._done   = False

    def clear(self) -> None:
        self._path.clear()
        self._wp_idx = 0
        self._done   = False

    # ── Control output ────────────────────────────────────────────────────────

    def compute(
        self, rx: float, ry: float, ryaw: float
    ) -> tuple[float, float]:
        """
        Compute (linear_vel, angular_vel) for the current robot pose.
        """
        if not self._path or self._done:
            return 0.0, 0.0

        goal = self._path[-1]
        d_goal = np.hypot(rx - goal[0], ry - goal[1])

        if d_goal < self.goal_tol:
            self._done = True
            return 0.0, 0.0

        # ── Robust Waypoint Advancement ───────────────────────────────────
        # Snap the base index to the closest waypoint ahead of the robot.
        # This completely eliminates "orbiting" missed waypoints.
        search_window = min(self._wp_idx + 15, len(self._path))
        best_dist = float('inf')
        for i in range(self._wp_idx, search_window):
            d = np.hypot(rx - self._path[i][0], ry - self._path[i][1])
            if d < best_dist:
                best_dist = d
                self._wp_idx = i

        # ── Find lookahead point ──────────────────────────────────────────
        target = self._find_lookahead(rx, ry)

        # ── Heading error α in robot frame ────────────────────────────────
        dx = target[0] - rx
        dy = target[1] - ry
        angle_to_target = np.arctan2(dy, dx)
        alpha = _wrap(angle_to_target - ryaw)

        # ── Dynamically shrink lookahead distance near the goal ───────────
        effective_ld = min(self.ld, max(0.2, d_goal))
        kappa = 2.0 * np.sin(alpha) / effective_ld

        # ── Velocity calculations ─────────────────────────────────────────
        self.slowdown_r = 0.80 
        goal_factor = min(1.0, d_goal / self.slowdown_r)
        
        turn_factor = max(0.2, np.cos(alpha)**3) 
        v = self.v_max * turn_factor * goal_factor
        
        if d_goal > self.slowdown_r:
            v = max(self.v_min, v)

        w = self.v_max * kappa * goal_factor 
        w = np.clip(w, -self.w_max, self.w_max)

        return float(v), float(w)
    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_lookahead(
        self, rx: float, ry: float
    ) -> tuple[float, float]:
        """
        Scan path segments from _wp_idx onwards and return the first
        point on the path that is `ld` metres from the robot.

        Falls back to the farthest reachable waypoint if no circle
        intersection is found.
        """
        robot = np.array([rx, ry])

        for i in range(self._wp_idx, len(self._path) - 1):
            p1 = np.array(self._path[i])
            p2 = np.array(self._path[i + 1])
            d  = p2 - p1
            f  = p1 - robot

            a = float(d @ d)
            b = 2.0 * float(f @ d)
            c = float(f @ f) - self.ld ** 2
            disc = b * b - 4 * a * c

            if disc < 0 or a < 1e-9:
                continue

            sq = np.sqrt(disc)
            t1 = (-b - sq) / (2 * a)
            t2 = (-b + sq) / (2 * a)

            # Prefer the farther intersection (t2 > t1)
            for t in (t2, t1):
                if 0.0 <= t <= 1.0:
                    pt = p1 + t * d
                    return float(pt[0]), float(pt[1])

        # Fallback: closest waypoint ahead of wp_idx
        return self._path[min(self._wp_idx, len(self._path) - 1)]

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._done

    @property
    def waypoint_index(self) -> int:
        return self._wp_idx

    @property
    def path_length(self) -> int:
        return len(self._path)


# ── Utility ───────────────────────────────────────────────────────────────────

def _wrap(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    while angle >  np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
