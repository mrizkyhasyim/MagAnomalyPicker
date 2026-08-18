import sys
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd


class SpatialMapWidget(QWidget):
    """2D Spatial Map View displaying survey tracklines, active line highlighting,

    and spatial anomaly target locations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control Bar
        ctrl_bar = QHBoxLayout()
        self.lbl_info = QLabel("<b>2D Spatial Survey Map</b>")
        self.chk_equal_aspect = QCheckBox("Equal Aspect Ratio (1:1)")
        self.chk_equal_aspect.setChecked(True)
        self.chk_equal_aspect.stateChanged.connect(self.plot_map)

        ctrl_bar.addWidget(self.lbl_info)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.chk_equal_aspect)
        layout.addLayout(ctrl_bar)

        # Matplotlib Canvas & Toolbar
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.dataset_cache: dict[Path, pd.DataFrame] = {}
        self.active_path: Path | None = None
        self.active_fiducial_idx: int | None = None

    def update_map_data(
        self,
        dataset_cache: dict[Path, pd.DataFrame],
        active_path: Path | None = None,
        active_fid_idx: int | None = None
    ):
        """Refreshes spatial map with all cached survey lines."""
        self.dataset_cache = dataset_cache
        self.active_path = active_path
        self.active_fiducial_idx = active_fid_idx
        self.plot_map()

    def plot_map(self):
        self.ax.clear()

        if not self.dataset_cache:
            self.ax.text(
                0.5, 0.5, "No Spatial Data Loaded",
                ha="center", va="center", transform=self.ax.transAxes
            )
            self.canvas.draw()
            return

        has_coords = False

        # 1. Plot Background Tracklines (All Loaded Lines)
        for path, df in self.dataset_cache.items():
            if "Easting" in df.columns and "Northing" in df.columns:
                has_coords = True
                is_active = (path == self.active_path)

                easting = df["Easting"].values
                northing = df["Northing"].values

                if is_active:
                    # Highlight Active Line in Bold Red
                    self.ax.plot(
                        easting, northing,
                        color="red", linewidth=2.0, label=f"Active: {path.name}", zorder=3
                    )

                    # Draw Current Cursor Position Dot if active
                    if self.active_fiducial_idx is not None and 0 <= self.active_fiducial_idx < len(df):
                        cur_x = easting[self.active_fiducial_idx]
                        cur_y = northing[self.active_fiducial_idx]
                        self.ax.scatter(
                            [cur_x], [cur_y],
                            color="yellow", edgecolor="black", s=80, zorder=5, label="Cursor"
                        )
                else:
                    # Background lines in muted grey
                    self.ax.plot(
                        easting, northing,
                        color="#7f8c8d", linewidth=0.8, alpha=0.6, zorder=2
                    )

        if not has_coords:
            self.ax.text(
                0.5, 0.5, "Missing 'Easting' or 'Northing' Columns",
                ha="center", va="center", transform=self.ax.transAxes
            )
            self.canvas.draw()
            return

        # Formatting
        self.ax.set_xlabel("Easting (m)", fontsize=9)
        self.ax.set_ylabel("Northing (m)", fontsize=9)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        if self.chk_equal_aspect.isChecked():
            self.ax.set_aspect("equal", adjustable="datalim")

        self.ax.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()