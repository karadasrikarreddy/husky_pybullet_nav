#!/usr/bin/env python3
"""
Headless validation for the Husky navigation stack.

Checks:
  - Optional Qt smoke boot works in offscreen mode
  - Planner/controller/simulator complete several navigation scenarios
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

from src.controller.controller import PurePursuitController
from src.gui.app import DEFAULT_OBSTACLES
from src.planner.astar import AStarPlanner
from src.simulator.simulation import Simulation


SCENARIOS = [
    ((-8.0, -8.0), (8.0, 8.0)),
    ((-8.5, 7.5), (7.0, -7.0)),
    ((-7.0, 0.0), (7.5, 0.0)),
    ((0.0, 8.0), (0.0, -8.0)),
]


def point_in_inflated_obstacle(
    x: float,
    y: float,
    obstacles: list[tuple[float, float, float, float]],
    margin: float = 0.50,
) -> bool:
    for cx, cy, w, h in obstacles:
        if abs(x - cx) <= (w / 2 + margin) and abs(y - cy) <= (h / 2 + margin):
            return True
    return False


def min_obstacle_clearance(
    x: float,
    y: float,
    obstacles: list[tuple[float, float, float, float]],
) -> float:
    min_clear = float("inf")
    for cx, cy, w, h in obstacles:
        dx = max(abs(x - cx) - w / 2, 0.0)
        dy = max(abs(y - cy) - h / 2, 0.0)
        min_clear = min(min_clear, math.hypot(dx, dy))
    return min_clear


def run_gui_smoke() -> bool:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        subprocess.run(
            [sys.executable, "main.py"],
            cwd=os.path.dirname(__file__),
            env=env,
            timeout=5,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True
    return False


def run_navigation_checks() -> bool:
    planner = AStarPlanner(
        world_half_size=10.0,
        resolution=0.20,
        robot_radius=0.50,
        safety_margin=0.15,
    )
    planner.set_obstacles(DEFAULT_OBSTACLES)

    overall_ok = True
    for idx, (start, goal) in enumerate(SCENARIOS, start=1):
        path = planner.plan(start, goal)
        if not path:
            print(f"[FAIL] scenario {idx}: no path from {start} to {goal}")
            overall_ok = False
            continue

        sim = Simulation()
        sim.set_obstacles(DEFAULT_OBSTACLES)
        initial_yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        sim.load_husky(start[0], start[1], initial_yaw)

        ctrl = PurePursuitController(
            lookahead_dist=0.4,
            max_linear_vel=1.0,
            max_angular_vel=3.0,
            goal_tolerance=0.30,
        )
        ctrl.set_path(path)

        reached = False
        collided = False
        prev_pos = start
        prev_yaw = initial_yaw
        travel = 0.0
        heading_change = 0.0
        max_linear = 0.0
        max_angular = 0.0
        min_clear = float("inf")

        for _step in range(6000):
            for _ in range(5):
                sim.step()

            pose = sim.get_robot_pose()
            if pose is None:
                break

            rx, ry, ryaw = pose
            linear_vel, angular_vel = sim.get_robot_velocity()
            max_linear = max(max_linear, linear_vel)
            max_angular = max(max_angular, abs(angular_vel))

            travel += math.hypot(rx - prev_pos[0], ry - prev_pos[1])
            dyaw = math.atan2(math.sin(ryaw - prev_yaw), math.cos(ryaw - prev_yaw))
            heading_change += abs(dyaw)
            prev_pos = (rx, ry)
            prev_yaw = ryaw

            min_clear = min(min_clear, min_obstacle_clearance(rx, ry, DEFAULT_OBSTACLES))
            if point_in_inflated_obstacle(rx, ry, DEFAULT_OBSTACLES):
                collided = True
                break

            cmd_v, cmd_w = ctrl.compute(rx, ry, ryaw)
            sim.set_cmd_vel(cmd_v, cmd_w)
            if ctrl.done:
                reached = True
                break

        final_pose = sim.get_robot_pose()
        d_goal = math.hypot(final_pose[0] - goal[0], final_pose[1] - goal[1]) if final_pose else float("inf")
        turn_per_m = heading_change / max(travel, 1e-6)
        passed = reached and not collided and d_goal <= 0.5
        overall_ok = overall_ok and passed

        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] scenario {idx}: "
            f"goal_error={d_goal:.3f}m, travel={travel:.2f}m, "
            f"path_points={len(path)}, min_clear={min_clear:.2f}m, "
            f"max_v={max_linear:.2f}m/s, max_w={max_angular:.2f}rad/s, "
            f"turn_per_m={turn_per_m:.2f}"
        )

    return overall_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-gui-smoke",
        action="store_true",
        help="Skip the offscreen Qt launch check.",
    )
    args = parser.parse_args()

    print("Checking navigation stack...")

    if not args.skip_gui_smoke:
        gui_ok = run_gui_smoke()
        print(f"[{'PASS' if gui_ok else 'FAIL'}] Qt app boot (offscreen smoke test)")
        if not gui_ok:
            return 1

    nav_ok = run_navigation_checks()
    return 0 if nav_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
