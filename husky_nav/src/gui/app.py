"""
PyQt5 GUI for the Husky autonomous navigation system.

Layout
------
Left  : WorldCanvas – 2-D bird's-eye view of the simulation world.
        Click to place Start (green) and Goal (red) markers.
Right : Control panel – status, action buttons, live metrics.

The simulation loop runs on a QTimer (50 Hz).  Each tick:
  1. Step physics ×5  (5 × 1/240 s ≈ 20 ms wall time)
  2. Query robot pose
  3. Compute (v, ω) via pure-pursuit controller
  4. Apply velocities to simulation
  5. Refresh the canvas
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import (
    Qt, QPointF, QRectF, QTimer, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
    QPolygonF, QLinearGradient,
)
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QVBoxLayout,
    QWidget, QFrame, QGroupBox, QProgressBar,
)

from src.simulator.simulation import Simulation
from src.planner.astar import AStarPlanner
from src.controller.controller import PurePursuitController


# ── Default obstacle layout ───────────────────────────────────────────────────
DEFAULT_OBSTACLES: list[tuple[float, float, float, float]] = [
    (-3.5,  2.5,  2.0, 1.2),
    ( 2.5, -1.5,  1.2, 2.5),
    ( 0.0, -4.5,  3.5, 1.0),
    (-5.5, -2.0,  1.0, 3.0),
    ( 4.5,  3.5,  2.0, 1.2),
    (-1.5,  5.5,  3.0, 1.0),
    ( 5.0, -4.0,  1.0, 2.0),
    (-4.5,  4.5,  1.5, 1.5),
    ( 3.0,  1.0,  1.0, 1.0),
    (-2.0, -7.0,  2.0, 1.0),
]

WORLD_HALF = 10.0          # world is ±10 m
TIMER_MS   = 20            # GUI update interval (ms)
SIM_STEPS  = 5             # physics steps per GUI tick
TRAIL_MAX  = 800           # max trail points


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C_BG          = QColor(13,  17,  30)
C_GRID        = QColor(35,  40,  65)
C_GRID_AXIS   = QColor(55,  60, 100)
C_OBSTACLE    = QColor(160, 35,  35)
C_OBSTACLE_BD = QColor(220, 60,  60)
C_PATH        = QColor( 80, 180, 255)
C_WP_ACTIVE   = QColor(255, 210,  50)
C_TRAIL       = QColor(  0, 200, 140)
C_ROBOT_BODY  = QColor( 30, 100, 180)
C_ROBOT_BD    = QColor(  0, 200, 255)
C_ARROW       = QColor(255, 220,  30)
C_START       = QColor(  0, 210,  90)
C_GOAL        = QColor(240,  60,  60)
C_PANEL_BG    = QColor(20,  24,  40)
C_BTN_RUN     = QColor(20, 140,  80)
C_BTN_STOP    = QColor(160, 50,  30)
C_BTN_RESET   = QColor(60,  80, 140)
C_BTN_TEXT    = QColor(230, 230, 230)


# ─────────────────────────────────────────────────────────────────────────────
# World canvas
# ─────────────────────────────────────────────────────────────────────────────
class WorldCanvas(QWidget):
    """Renders the 2-D overhead view and accepts click-based user input."""

    start_placed = pyqtSignal(float, float)
    goal_placed  = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(620, 620)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Displayed state
        self.obstacles:   list[tuple] = []
        self.start_pos:   Optional[tuple[float, float]] = None
        self.goal_pos:    Optional[tuple[float, float]] = None
        self.path:        list[tuple[float, float]] = []
        self.robot_pose:  Optional[tuple[float, float, float]] = None  # x,y,yaw
        self.robot_trail: list[tuple[float, float]] = []
        self.wp_idx:      int = 0
        self.occ_grid:    Optional[np.ndarray] = None  # for debug overlay

        self._click_mode: Optional[str] = None   # 'start' | 'goal' | None
        self._mouse_w:    Optional[tuple[float, float]] = None

    # ── Coordinate transforms ─────────────────────────────────────────────

    def _w2c(self, wx: float, wy: float) -> tuple[float, float]:
        W, H = self.width(), self.height()
        cx = (wx + WORLD_HALF) / (2 * WORLD_HALF) * W
        cy = H - (wy + WORLD_HALF) / (2 * WORLD_HALF) * H
        return cx, cy

    def _c2w(self, cx: float, cy: float) -> tuple[float, float]:
        W, H = self.width(), self.height()
        wx = cx / W * (2 * WORLD_HALF) - WORLD_HALF
        wy = (H - cy) / H * (2 * WORLD_HALF) - WORLD_HALF
        return wx, wy

    def _scale(self) -> float:
        return self.width() / (2 * WORLD_HALF)

    # ── Click mode ────────────────────────────────────────────────────────

    def set_click_mode(self, mode: Optional[str]) -> None:
        self._click_mode = mode
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    # ── Qt events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.LeftButton or not self._click_mode:
            return
        wx, wy = self._c2w(ev.x(), ev.y())
        if self._click_mode == "start":
            self.start_pos = (wx, wy)
            self.start_placed.emit(wx, wy)
        elif self._click_mode == "goal":
            self.goal_pos = (wx, wy)
            self.goal_placed.emit(wx, wy)
        self._click_mode = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def mouseMoveEvent(self, ev) -> None:
        self._mouse_w = self._c2w(ev.x(), ev.y())
        if self._click_mode:
            self.update()

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        self._draw_background(p)
        self._draw_grid(p)
        self._draw_obstacles(p)
        self._draw_trail(p)
        self._draw_path(p)
        if self.start_pos:
            self._draw_marker(p, self.start_pos, C_START, "S")
        if self.goal_pos:
            self._draw_marker(p, self.goal_pos,  C_GOAL,  "G")
        if self.robot_pose:
            self._draw_robot(p, self.robot_pose)
        self._draw_cursor_preview(p)
        self._draw_border(p)

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _draw_background(self, p: QPainter) -> None:
        p.fillRect(0, 0, self.width(), self.height(), C_BG)

    def _draw_grid(self, p: QPainter) -> None:
        W, H = self.width(), self.height()
        # Minor lines every 1 m
        p.setPen(QPen(C_GRID, 1))
        for v in np.arange(-WORLD_HALF, WORLD_HALF + 1, 1.0):
            cx, _ = self._w2c(v, 0)
            _, cy = self._w2c(0, v)
            p.drawLine(int(cx), 0, int(cx), H)
            p.drawLine(0, int(cy), W, int(cy))
        # Axes
        p.setPen(QPen(C_GRID_AXIS, 1))
        cx0, cy0 = self._w2c(0, 0)
        p.drawLine(int(cx0), 0, int(cx0), H)
        p.drawLine(0, int(cy0), W, int(cy0))
        # Scale labels
        p.setPen(QPen(QColor(70, 80, 110)))
        p.setFont(QFont("Consolas", 7))
        for v in np.arange(-WORLD_HALF + 2, WORLD_HALF, 2.0):
            cx, _ = self._w2c(v, 0)
            _, cy = self._w2c(0, v)
            p.drawText(int(cx) + 2, int(cy0) - 3, f"{v:.0f}")
            p.drawText(int(cx0) + 3, int(cy) + 4, f"{v:.0f}")

    def _draw_obstacles(self, p: QPainter) -> None:
        s = self._scale()
        for (cx, cy, w, h) in self.obstacles:
            px, py = self._w2c(cx - w / 2, cy + h / 2)
            pw, ph = w * s, h * s
            # Fill
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(C_OBSTACLE))
            p.drawRect(QRectF(px, py, pw, ph))
            # Border
            p.setPen(QPen(C_OBSTACLE_BD, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(px, py, pw, ph))

    def _draw_trail(self, p: QPainter) -> None:
        n = len(self.robot_trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = int(30 + 150 * i / n)
            c = QColor(C_TRAIL.red(), C_TRAIL.green(), C_TRAIL.blue(), alpha)
            p.setPen(QPen(c, 2))
            x1, y1 = self._w2c(*self.robot_trail[i - 1])
            x2, y2 = self._w2c(*self.robot_trail[i])
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_path(self, p: QPainter) -> None:
        if len(self.path) < 2:
            return
        # Dashed path line
        pen = QPen(C_PATH, 2, Qt.DashLine)
        pen.setDashPattern([6, 4])
        p.setPen(pen)
        for i in range(len(self.path) - 1):
            x1, y1 = self._w2c(*self.path[i])
            x2, y2 = self._w2c(*self.path[i + 1])
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        # Waypoint dots
        for i, wp in enumerate(self.path):
            cx, cy = self._w2c(*wp)
            if i == self.wp_idx:
                p.setPen(QPen(C_WP_ACTIVE, 2))
                p.setBrush(QBrush(QColor(C_WP_ACTIVE.red(),
                                         C_WP_ACTIVE.green(),
                                         C_WP_ACTIVE.blue(), 180)))
                p.drawEllipse(QPointF(cx, cy), 5, 5)
            else:
                p.setPen(QPen(QColor(C_PATH.red(), C_PATH.green(),
                                     C_PATH.blue(), 120), 1))
                p.setBrush(QBrush(QColor(C_PATH.red(), C_PATH.green(),
                                          C_PATH.blue(), 60)))
                p.drawEllipse(QPointF(cx, cy), 3, 3)

    def _draw_marker(
        self, p: QPainter,
        pos: tuple[float, float],
        color: QColor, label: str,
    ) -> None:
        cx, cy = self._w2c(*pos)
        # Outer glow
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 60), 10))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 18, 18)
        # Ring
        p.setPen(QPen(color, 2))
        p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        p.drawEllipse(QPointF(cx, cy), 15, 15)
        # Centre dot
        p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 6, 6)
        # Label
        p.setPen(QPen(Qt.white))
        f = QFont("Arial", 8, QFont.Bold)
        p.setFont(f)
        p.drawText(int(cx) - 4, int(cy) + 4, label)

    def _draw_robot(
        self, p: QPainter,
        pose: tuple[float, float, float],
    ) -> None:
        rx, ry, ryaw = pose
        cx, cy = self._w2c(rx, ry)
        s = self._scale()

        rw = 0.87 * s   # body width  in canvas pixels
        rh = 0.58 * s   # body height in canvas pixels

        p.save()
        p.translate(cx, cy)
        p.rotate(-math.degrees(ryaw))   # Qt: positive = clockwise

        # Body fill
        p.setPen(QPen(C_ROBOT_BD, 2))
        p.setBrush(QBrush(QColor(C_ROBOT_BODY.red(),
                                  C_ROBOT_BODY.green(),
                                  C_ROBOT_BODY.blue(), 210)))
        p.drawRect(QRectF(-rw / 2, -rh / 2, rw, rh))

        # Front strip
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(C_ROBOT_BD.red(),
                                  C_ROBOT_BD.green(),
                                  C_ROBOT_BD.blue(), 120)))
        p.drawRect(QRectF(rw / 2 - 6, -rh / 2, 6, rh))

        # Direction arrow
        p.setPen(QPen(C_ARROW, 2.5))
        p.drawLine(QPointF(0, 0), QPointF(rw / 2, 0))
        # Arrowhead
        ah = min(8.0, rw / 4)
        arrow = QPolygonF([
            QPointF(rw / 2,       0),
            QPointF(rw / 2 - ah, -ah * 0.5),
            QPointF(rw / 2 - ah,  ah * 0.5),
        ])
        p.setBrush(QBrush(C_ARROW))
        p.setPen(Qt.NoPen)
        p.drawPolygon(arrow)

        p.restore()

        # Outer glow
        p.setPen(QPen(QColor(0, 200, 255, 40), 10))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rw / 2 + 4, rw / 2 + 4)

    def _draw_cursor_preview(self, p: QPainter) -> None:
        if not self._click_mode or not self._mouse_w:
            return
        color = C_START if self._click_mode == "start" else C_GOAL
        cx, cy = self._w2c(*self._mouse_w)
        p.setPen(QPen(color, 2, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 14, 14)
        p.setPen(QPen(color, 1))
        p.drawLine(QPointF(cx - 18, cy), QPointF(cx + 18, cy))
        p.drawLine(QPointF(cx, cy - 18), QPointF(cx, cy + 18))

    def _draw_border(self, p: QPainter) -> None:
        p.setPen(QPen(QColor(60, 70, 110), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(1, 1, self.width() - 2, self.height() - 2)


# ─────────────────────────────────────────────────────────────────────────────
# Control panel
# ─────────────────────────────────────────────────────────────────────────────
def _make_btn(text: str, color: QColor, min_h: int = 38) -> QPushButton:
    btn = QPushButton(text)
    btn.setMinimumHeight(min_h)
    btn.setFont(QFont("Arial", 10, QFont.Bold))
    c = color
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: rgb({c.red()},{c.green()},{c.blue()});
            color: rgb(230,230,230);
            border: 1px solid rgba(255,255,255,40);
            border-radius: 5px;
        }}
        QPushButton:hover {{
            background-color: rgb({min(c.red()+30,255)},{min(c.green()+30,255)},{min(c.blue()+30,255)});
        }}
        QPushButton:pressed {{
            background-color: rgb({max(c.red()-20,0)},{max(c.green()-20,0)},{max(c.blue()-20,0)});
        }}
        QPushButton:disabled {{
            background-color: rgb(40,44,60);
            color: rgb(100,100,110);
        }}
    """)
    return btn


def _make_label(text: str, font_size: int = 9, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    w = QFont.Bold if bold else QFont.Normal
    lbl.setFont(QFont("Consolas", font_size, w))
    lbl.setStyleSheet("color: rgb(180,190,220);")
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Husky Autonomous Navigation  │  PyBullet")
        self.setMinimumSize(900, 680)

        # ── Core components ───────────────────────────────────────────────
        self._sim        = Simulation()
        self._planner    = AStarPlanner(
            world_half_size=WORLD_HALF, resolution=0.20,
            robot_radius=0.50, safety_margin=0.15,
        )
        self._ctrl       = PurePursuitController(
            lookahead_dist=0.4,
            max_linear_vel=1.0,
            max_angular_vel=3.0,
            goal_tolerance=0.30,
        )

        # Set up world
        self._sim.set_obstacles(DEFAULT_OBSTACLES)
        self._planner.set_obstacles(DEFAULT_OBSTACLES)
        self._sim.load_husky(0.0, 0.0, 0.0)   # dummy spawn for physics warm-up

        # ── State ─────────────────────────────────────────────────────────
        self._running      = False
        self._start_world  = None
        self._goal_world   = None
        self._t0           = 0.0
        self._dist_covered = 0.0
        self._prev_pos     = None

        # ── UI ────────────────────────────────────────────────────────────
        self._build_ui()
        self._apply_theme()

        # ── Timer ─────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(TIMER_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── Canvas ────────────────────────────────────────────────────────
        self._canvas = WorldCanvas()
        self._canvas.obstacles  = DEFAULT_OBSTACLES
        self._canvas.start_placed.connect(self._on_start_placed)
        self._canvas.goal_placed.connect(self._on_goal_placed)
        root.addWidget(self._canvas, stretch=1)

        # ── Right panel ───────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(210)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(6, 6, 6, 6)
        pv.setSpacing(8)

        # Header
        title = QLabel("HUSKY NAV")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: rgb(0,200,255); letter-spacing: 3px;")
        pv.addWidget(title)

        sub = QLabel("PyBullet  ·  Pure Pursuit  ·  A*")
        sub.setFont(QFont("Consolas", 7))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: rgb(80,100,140);")
        pv.addWidget(sub)

        pv.addWidget(self._hline())

        # ── Placement group ───────────────────────────────────────────────
        grp_place = QGroupBox("Placement")
        grp_place.setStyleSheet(self._grp_style())
        gv = QVBoxLayout(grp_place)
        gv.setSpacing(5)

        self._btn_start = _make_btn("📍  Set Start", QColor(20, 120, 60))
        self._btn_goal  = _make_btn("🏁  Set Goal",  QColor(140, 40, 40))
        self._btn_start.clicked.connect(lambda: self._canvas.set_click_mode("start"))
        self._btn_goal.clicked.connect(lambda: self._canvas.set_click_mode("goal"))
        gv.addWidget(self._btn_start)
        gv.addWidget(self._btn_goal)

        self._lbl_start = _make_label("Start:  —")
        self._lbl_goal  = _make_label("Goal:   —")
        gv.addWidget(self._lbl_start)
        gv.addWidget(self._lbl_goal)
        pv.addWidget(grp_place)

        # ── Navigation group ──────────────────────────────────────────────
        grp_nav = QGroupBox("Navigation")
        grp_nav.setStyleSheet(self._grp_style())
        nv = QVBoxLayout(grp_nav)
        nv.setSpacing(5)

        self._btn_run   = _make_btn("▶  Start",  C_BTN_RUN,  44)
        self._btn_stop  = _make_btn("■  Stop",   C_BTN_STOP)
        self._btn_reset = _make_btn("↺  Reset",  C_BTN_RESET)

        self._btn_run.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_reset.clicked.connect(self._on_reset)

        self._btn_stop.setEnabled(False)
        nv.addWidget(self._btn_run)
        nv.addWidget(self._btn_stop)
        nv.addWidget(self._btn_reset)
        pv.addWidget(grp_nav)

        # ── Status group ──────────────────────────────────────────────────
        grp_status = QGroupBox("Status")
        grp_status.setStyleSheet(self._grp_style())
        sv = QVBoxLayout(grp_status)
        sv.setSpacing(4)

        self._lbl_state  = _make_label("IDLE", 10, bold=True)
        self._lbl_state.setAlignment(Qt.AlignCenter)
        sv.addWidget(self._lbl_state)

        self._lbl_pos  = _make_label("Pos:    —")
        self._lbl_yaw  = _make_label("Yaw:    —")
        self._lbl_vel  = _make_label("Vel:    —")
        self._lbl_wp   = _make_label("WP:     —")
        self._lbl_dist = _make_label("Dist:   0.00 m")
        self._lbl_time = _make_label("Time:   0.0 s")
        for lbl in (self._lbl_pos, self._lbl_yaw, self._lbl_vel,
                    self._lbl_wp, self._lbl_dist, self._lbl_time):
            sv.addWidget(lbl)

        # Goal proximity bar
        sv.addWidget(_make_label("Goal proximity:"))
        self._prog = QProgressBar()
        self._prog.setRange(0, 100)
        self._prog.setValue(0)
        self._prog.setStyleSheet("""
            QProgressBar { border: 1px solid #3a3f5c; border-radius: 3px;
                           background: #1a1e30; height: 10px; text-align: center; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1e6fd9, stop:1 #00c88c); border-radius: 2px; }
        """)
        sv.addWidget(self._prog)
        pv.addWidget(grp_status)

        pv.addStretch()

        # Legend
        legend = _make_label(
            "🟢 Start   🔴 Goal\n"
            "── Path   ── Trail\n"
            "🟦 Robot (arrow=front)",
            7,
        )
        legend.setAlignment(Qt.AlignCenter)
        pv.addWidget(legend)

        root.addWidget(panel)

        # Status bar
        self._sb = QStatusBar()
        self._sb.setStyleSheet("background: #0e1120; color: #6a7899;")
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready.  Place Start and Goal markers, then press ▶ Start.")

    # ── Button callbacks ──────────────────────────────────────────────────

    def _on_start_placed(self, wx: float, wy: float) -> None:
        self._start_world = (wx, wy)
        self._lbl_start.setText(f"Start:  ({wx:.1f}, {wy:.1f})")
        self._sb.showMessage(f"Start set at ({wx:.2f}, {wy:.2f}).  "
                              "Now set Goal or press ▶ Start.")
        self._canvas.update()

    def _on_goal_placed(self, wx: float, wy: float) -> None:
        self._goal_world = (wx, wy)
        self._lbl_goal.setText(f"Goal:   ({wx:.1f}, {wy:.1f})")
        self._sb.showMessage(f"Goal set at ({wx:.2f}, {wy:.2f}).  Press ▶ Start.")
        self._canvas.update()

    def _on_start(self) -> None:
        if not self._start_world or not self._goal_world:
            self._sb.showMessage("⚠  Please set both Start and Goal first.")
            return

        sx, sy = self._start_world
        gx, gy = self._goal_world

        # Plan
        self._sb.showMessage("Planning path …")
        QApplication.processEvents()
        path = self._planner.plan((sx, sy), (gx, gy))

        if path is None:
            self._sb.showMessage("✗  No path found – goal may be inside an obstacle.")
            return

        # Load robot at start
        import math
        initial_yaw = math.atan2(gy - sy, gx - sx)
        self._sim.load_husky(sx, sy, initial_yaw)

        # Configure controller
        self._ctrl.set_path(path)

        # Reset metrics
        self._dist_covered = 0.0
        self._prev_pos     = (sx, sy)
        self._t0           = time.time()

        # Update canvas
        self._canvas.path        = path
        self._canvas.robot_trail = []
        self._canvas.wp_idx      = 0
        self._canvas.robot_pose  = (sx, sy, initial_yaw)

        self._running = True
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._set_state_label("NAVIGATING", "#00c88c")
        self._sb.showMessage(f"Navigation started.  Path has {len(path)} waypoints.")

    def _on_stop(self) -> None:
        self._running = False
        self._sim.stop()
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_state_label("STOPPED", "#e06030")
        self._sb.showMessage("Navigation stopped by user.")

    def _on_reset(self) -> None:
        self._on_stop()
        self._start_world = None
        self._goal_world  = None
        self._canvas.start_pos   = None
        self._canvas.goal_pos    = None
        self._canvas.path        = []
        self._canvas.robot_trail = []
        self._canvas.robot_pose  = None
        self._canvas.wp_idx      = 0
        self._ctrl.clear()
        self._sim.load_husky(0.0, 0.0, 0.0)
        self._lbl_start.setText("Start:  —")
        self._lbl_goal.setText("Goal:   —")
        self._lbl_pos.setText("Pos:    —")
        self._lbl_yaw.setText("Yaw:    —")
        self._lbl_vel.setText("Vel:    —")
        self._lbl_wp.setText("WP:     —")
        self._lbl_dist.setText("Dist:   0.00 m")
        self._lbl_time.setText("Time:   0.0 s")
        self._prog.setValue(0)
        self._set_state_label("IDLE", "#607090")
        self._sb.showMessage("Reset.  Place Start and Goal markers.")
        self._canvas.update()

    # ── Simulation tick ───────────────────────────────────────────────────

    def _tick(self) -> None:
        if not self._running:
            return

        # Step physics
        for _ in range(SIM_STEPS):
            self._sim.step()

        # Query state
        pose = self._sim.get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        # Accumulate distance
        if self._prev_pos:
            self._dist_covered += math.hypot(rx - self._prev_pos[0],
                                              ry - self._prev_pos[1])
        self._prev_pos = (rx, ry)

        # Compute control
        v, w = self._ctrl.compute(rx, ry, ryaw)
        self._sim.set_cmd_vel(v, w)

        # Update canvas
        self._canvas.robot_pose = pose
        self._canvas.robot_trail.append((rx, ry))
        if len(self._canvas.robot_trail) > TRAIL_MAX:
            self._canvas.robot_trail.pop(0)
        self._canvas.wp_idx = self._ctrl.waypoint_index

        # Update metrics labels
        elapsed = time.time() - self._t0
        lv, av  = self._sim.get_robot_velocity()
        self._lbl_pos.setText(f"Pos:  ({rx:.2f}, {ry:.2f})")
        self._lbl_yaw.setText(f"Yaw:  {math.degrees(ryaw):.1f}°")
        self._lbl_vel.setText(f"Vel:  {lv:.2f} m/s")
        n_wp = self._ctrl.path_length
        wi   = self._ctrl.waypoint_index
        self._lbl_wp.setText(f"WP:   {wi}/{n_wp}")
        self._lbl_dist.setText(f"Dist: {self._dist_covered:.2f} m")
        self._lbl_time.setText(f"Time: {elapsed:.1f} s")

        # Goal proximity bar
        if self._goal_world:
            d_goal = math.hypot(rx - self._goal_world[0],
                                 ry - self._goal_world[1])
            d_start = math.hypot(
                (self._start_world or (rx, ry))[0] - self._goal_world[0],
                (self._start_world or (rx, ry))[1] - self._goal_world[1],
            )
            if d_start > 0.01:
                pct = int(100 * max(0, 1 - d_goal / d_start))
                self._prog.setValue(pct)

        # Goal reached?
        if self._ctrl.done:
            self._running = False
            self._sim.stop()
            self._btn_run.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._set_state_label("GOAL REACHED ✓", "#00c88c")
            self._prog.setValue(100)
            self._sb.showMessage(
                f"✓  Goal reached in {elapsed:.1f} s  |  "
                f"distance: {self._dist_covered:.2f} m"
            )

        self._canvas.update()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _set_state_label(self, text: str, color: str) -> None:
        self._lbl_state.setText(text)
        self._lbl_state.setStyleSheet(f"color: {color}; font-weight: bold;")

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #2a3050;")
        return line

    @staticmethod
    def _grp_style() -> str:
        return """
            QGroupBox {
                color: rgb(140,160,200);
                border: 1px solid #2a3050;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 9pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: rgb(16, 20, 35);
            }
            QLabel { color: rgb(180, 190, 220); }
        """)

    def closeEvent(self, ev) -> None:
        self._timer.stop()
        del self._sim
        ev.accept()
