#!/usr/bin/env python3
"""
main.py – entry point for Husky Autonomous Navigation (PyBullet, No ROS).

Usage
-----
    python main.py

Requirements
------------
    pip install pybullet PyQt5 numpy
"""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.gui.app import MainWindow


def main() -> None:
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Husky Nav")
    app.setOrganizationName("VIR Innovations Assignment")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
