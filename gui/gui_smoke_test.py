"""Headless smoke test for PyNutil GUI.
Launches the main window in offscreen mode and exits after a short delay.
"""
import os
import sys

# Force headless/offscreen rendering for CI/HEADLESS environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Adjust import path if running from repo root
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from PyNutilGUI import PyNutilGUI


def main():
    app = QApplication(sys.argv)
    gui = PyNutilGUI()
    gui.show()

    # Quit after 800ms
    QTimer.singleShot(800, app.quit)
    exit_code = app.exec()
    print("GUI smoke test exited with code", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
