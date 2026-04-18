"""
Grid-based A* path planner with obstacle inflation.

Design:
  - The world is a square centred at (0, 0) with configurable half-extent.
  - The world is discretised into a regular grid at a configurable resolution.
  - Obstacles are inflated by (robot_radius + safety_margin) before planning,
    so the path is guaranteed to be collision-free for a robot of that size.
  - 8-connected A* is used (diagonal moves allowed).
  - A lightweight post-processing step smooths the jagged grid path.
"""

import heapq
from typing import Optional

import numpy as np


class AStarPlanner:
    """A* on a 2-D occupancy grid with obstacle inflation."""

    def __init__(
        self,
        world_half_size: float = 10.0,
        resolution: float = 0.20,
        robot_radius: float = 0.50,
        safety_margin: float = 0.15,
    ) -> None:
        self.half   = world_half_size
        self.res    = resolution
        self.n      = int(2 * world_half_size / resolution)   # grid dimension
        self._grid  = np.zeros((self.n, self.n), dtype=np.uint8)
        self._inflate_cells = int(
            np.ceil((robot_radius + safety_margin) / resolution)
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def w2g(self, x: float, y: float) -> tuple[int, int]:
        """World → grid index (clamped)."""
        i = int((x + self.half) / self.res)
        j = int((y + self.half) / self.res)
        return (
            max(0, min(self.n - 1, i)),
            max(0, min(self.n - 1, j)),
        )

    def g2w(self, i: int, j: int) -> tuple[float, float]:
        """Grid index centre → world coordinates."""
        x = i * self.res - self.half + self.res / 2
        y = j * self.res - self.half + self.res / 2
        return x, y

    # ── Obstacle handling ─────────────────────────────────────────────────────

    def set_obstacles(self, obstacles: list[tuple]) -> None:
        """
        Build the occupancy grid from a list of axis-aligned box obstacles.

        obstacles : list of (cx, cy, width, height) in world metres.
        Cells within `inflate_cells` of any obstacle are also marked.
        """
        self._grid[:] = 0
        r = self._inflate_cells

        for (cx, cy, w, h) in obstacles:
            hw, hh = w / 2, h / 2
            i0, j0 = self.w2g(cx - hw, cy - hh)
            i1, j1 = self.w2g(cx + hw, cy + hh)
            # Inflate
            i0 = max(0, i0 - r)
            j0 = max(0, j0 - r)
            i1 = min(self.n - 1, i1 + r)
            j1 = min(self.n - 1, j1 + r)
            self._grid[i0 : i1 + 1, j0 : j1 + 1] = 1

        # World boundary (1-cell border)
        self._grid[0, :]  = 1
        self._grid[-1, :] = 1
        self._grid[:, 0]  = 1
        self._grid[:, -1] = 1

    def get_grid(self) -> np.ndarray:
        return self._grid.copy()

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> Optional[list[tuple[float, float]]]:
        """
        Compute a smooth path from *start* to *goal* (world coords).

        Returns a list of (x, y) waypoints, or None if no path exists.
        """
        gs = self.w2g(*start)
        gg = self.w2g(*goal)

        # Snap to nearest free cell if needed
        gs = self._nearest_free(gs) or gs
        gg = self._nearest_free(gg) or gg

        raw = self._astar(gs, gg)
        if raw is None:
            return None

        world_path = [self.g2w(*cell) for cell in raw]
        return self._smooth(world_path, passes=5)

    def _nearest_free(self, cell: tuple[int, int]) -> Optional[tuple[int, int]]:
        ci, cj = cell
        for r in range(1, 15):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < self.n and 0 <= nj < self.n:
                        if self._grid[ni, nj] == 0:
                            return ni, nj
        return None

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> Optional[list[tuple[int, int]]]:
        """8-connected A* on the occupancy grid."""

        # (di, dj, cost)
        MOVES = [
            ( 1,  0, 1.000), (-1,  0, 1.000),
            ( 0,  1, 1.000), ( 0, -1, 1.000),
            ( 1,  1, 1.414), ( 1, -1, 1.414),
            (-1,  1, 1.414), (-1, -1, 1.414),
        ]

        def h(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        open_heap: list = []
        heapq.heappush(open_heap, (h(start, goal), 0.0, start))

        came_from: dict[tuple, tuple] = {}
        g: dict[tuple, float] = {start: 0.0}
        in_open: set[tuple] = {start}

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            in_open.discard(cur)

            if cur == goal:
                return self._reconstruct(came_from, cur)

            if g_cur > g.get(cur, float("inf")):
                continue  # stale entry

            for di, dj, cost in MOVES:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.n and 0 <= nb[1] < self.n):
                    continue
                if self._grid[nb]:
                    continue

                tg = g[cur] + cost
                if tg < g.get(nb, float("inf")):
                    g[nb] = tg
                    came_from[nb] = cur
                    f = tg + h(nb, goal)
                    heapq.heappush(open_heap, (f, tg, nb))
                    in_open.add(nb)

        return None  # no path

    @staticmethod
    def _reconstruct(
        came_from: dict, current: tuple
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _smooth(
        path: list[tuple[float, float]], passes: int = 4
    ) -> list[tuple[float, float]]:
        """
        Gaussian-like smoothing: each interior point is pulled toward its
        neighbours. Endpoints are fixed.
        """
        if len(path) < 3:
            return path

        pts = list(path)
        for _ in range(passes):
            new = [pts[0]]
            for k in range(1, len(pts) - 1):
                x = 0.25 * pts[k - 1][0] + 0.50 * pts[k][0] + 0.25 * pts[k + 1][0]
                y = 0.25 * pts[k - 1][1] + 0.50 * pts[k][1] + 0.25 * pts[k + 1][1]
                new.append((x, y))
            new.append(pts[-1])
            pts = new

        # Downsample: keep every other point for speed (keep first and last)
     #   if len(pts) > 30:
      #      step = max(1, len(pts) // 30)
      #      pts = pts[::step]
      #      if pts[-1] != path[-1]:
        #        pts.append(path[-1])

        return pts
