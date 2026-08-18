import sys
from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDockWidget,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSplitter, QTabWidget, QTableView,
    QToolBar, QVBoxLayout, QWidget
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Resolve project root path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.io.loader import load_dataset


# =============================================================================
# 1. Virtualized Pandas Table Model
# =============================================================================
class PandasTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame = pd.DataFrame()):
        super().__init__()
        self._df = df

    def update_data(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            if orientation == Qt.Vertical:
                return str(self._df.index[section] + 1)
        return None


# =============================================================================
# 2. 2D Spatial Map View Widget
# =============================================================================
class SpatialMapWidget(QWidget):
    """2D Spatial Trackline map with interactive cursor positioning and anomaly picks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Map Control Bar
        ctrl_bar = QHBoxLayout()
        self.lbl_info = QLabel("<b>2D Spatial Survey Map</b>")
        self.chk_equal_aspect = QCheckBox("Equal Aspect (1:1)")
        self.chk_equal_aspect.setChecked(True)
        self.chk_equal_aspect.stateChanged.connect(self.plot_map)

        ctrl_bar.addWidget(self.lbl_info)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.chk_equal_aspect)
        layout.addLayout(ctrl_bar)

        # Matplotlib Canvas & Toolbar
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.dataset_cache: Dict[Path, pd.DataFrame] = {}
        self.anomaly_picks: Dict[Path, List[int]] = {}
        self.active_path: Path | None = None
        self.active_idx: int | None = None
        self.sync_enabled: bool = True

        # Connect Map Click Event for Line Selection
        self.canvas.mpl_connect("button_press_event", self.on_map_click)

    def set_sync_enabled(self, enabled: bool):
        self.sync_enabled = enabled

    def update_map_data(
        self,
        dataset_cache: Dict[Path, pd.DataFrame],
        anomaly_picks: Dict[Path, List[int]],
        active_path: Path | None = None,
        active_idx: int | None = None,
    ):
        self.dataset_cache = dataset_cache
        self.anomaly_picks = anomaly_picks
        self.active_path = active_path
        self.active_idx = active_idx
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

        for path, df in self.dataset_cache.items():
            if "Easting" in df.columns and "Northing" in df.columns:
                has_coords = True
                is_active = (path == self.active_path)
                east = df["Easting"].values
                north = df["Northing"].values

                if is_active:
                    # Highlight Active Line in Red
                    self.ax.plot(east, north, color="red", linewidth=2.0, zorder=3, label=f"Active: {path.name}")

                    # Plot Active Cursor Position (if Sync is ON)
                    if self.sync_enabled and self.active_idx is not None and 0 <= self.active_idx < len(df):
                        cx, cy = east[self.active_idx], north[self.active_idx]
                        self.ax.scatter([cx], [cy], color="yellow", edgecolor="black", s=90, zorder=6, label="Cursor")

                    # Plot Anomaly Targets on Active Line
                    if path in self.anomaly_picks and self.anomaly_picks[path]:
                        pick_indices = self.anomaly_picks[path]
                        px = east[pick_indices]
                        py = north[pick_indices]
                        self.ax.scatter(px, py, color="cyan", marker="*", s=120, edgecolor="black", zorder=5, label="Anomalies")

                else:
                    # Background Tracklines in Muted Grey
                    self.ax.plot(east, north, color="#7f8c8d", linewidth=0.8, alpha=0.5, zorder=2)

        if not has_coords:
            self.ax.text(
                0.5, 0.5, "Missing 'Easting' / 'Northing' Columns",
                ha="center", va="center", transform=self.ax.transAxes
            )
            self.canvas.draw()
            return

        self.ax.set_xlabel("Easting (m)", fontsize=8)
        self.ax.set_ylabel("Northing (m)", fontsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.4)

        if self.chk_equal_aspect.isChecked():
            self.ax.set_aspect("equal", adjustable="datalim")

        self.ax.legend(loc="upper right", fontsize=7)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_map_click(self, event):
        """Selects line on 2D Map if Tri-Linked Sync is enabled."""
        if not self.sync_enabled or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        click_x, click_y = event.xdata, event.ydata
        closest_path = None
        min_dist = float("inf")

        # Find line closest to mouse click
        for path, df in self.dataset_cache.items():
            if "Easting" in df.columns and "Northing" in df.columns:
                dx = df["Easting"].values - click_x
                dy = df["Northing"].values - click_y
                dist = np.min(np.sqrt(dx**2 + dy**2))
                if dist < min_dist:
                    min_dist = dist
                    closest_path = path

        if closest_path and hasattr(self.parent(), "switch_to_line_by_path"):
            self.parent().switch_to_line_by_path(closest_path)


# =============================================================================
# 3. Main Processing Suite Workspace
# =============================================================================
class MagAnomalyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MagAnomalyPicker - Standalone Processing Suite")
        self.resize(1500, 950)

        # File & Anomaly Target Management
        self.file_paths: List[Path] = []
        self.current_file_idx: int = -1
        self.dataset_cache: Dict[Path, pd.DataFrame] = {}
        self.anomaly_picks: Dict[Path, List[int]] = {}
        self.df: pd.DataFrame | None = None

        self.crosshair_lines = []
        self.table_model = PandasTableModel()

        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_main_layout()

    # -------------------------------------------------------------------------
    # UI Setup Methods
    # -------------------------------------------------------------------------
    def setup_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        project_menu = file_menu.addMenu("Project")
        project_menu.addAction("New Project")
        project_menu.addAction("Open Project...")
        project_menu.addAction("Save")
        project_menu.addAction("Close Project")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        db_menu = menu_bar.addMenu("Database")
        db_menu.addAction("Import ASCII Files...", self.open_files)

        db_tools = menu_bar.addMenu("Database Tools")
        chan_tools = db_tools.addMenu("Channel Tools")
        chan_tools.addAction("Copy Channels...")
        chan_tools.addAction("Make Diff (Difference) Channel")

        filter_menu = db_tools.addMenu("Filter")
        filter_menu.addAction("Low Pass Filter...")
        filter_menu.addAction("High Pass Filter...")

    def setup_toolbar(self):
        toolbar = QToolBar("Main Controls")
        toolbar.setIconSize(toolbar.iconSize())
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Tri-Linked Master Sync Toggle
        self.chk_tri_sync = QCheckBox("🔗 Tri-Linked Sync (Map - Profile - Table)")
        self.chk_tri_sync.setChecked(True)
        self.chk_tri_sync.setStyleSheet("font-weight: bold; padding: 4px; color: #16a085;")
        self.chk_tri_sync.toggled.connect(self.on_sync_toggled)
        toolbar.addWidget(self.chk_tri_sync)

        toolbar.addSeparator()

        # Anomaly Picker Mode Toggle
        self.chk_picker_mode = QCheckBox("🎯 Anomaly Picker Mode (Left-Click Add / Right-Click Del)")
        self.chk_picker_mode.setChecked(False)
        self.chk_picker_mode.setStyleSheet("font-weight: bold; padding: 4px; color: #c0392b;")
        toolbar.addWidget(self.chk_picker_mode)

    def setup_main_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(350)

        # 1. Batch File Loader Box
        file_group = QGroupBox("1. Cumulative File Ingestion")
        file_layout = QVBoxLayout(file_group)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("+ Add File(s)")
        self.btn_load.clicked.connect(self.open_files)
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_file_list)
        btn_row.addWidget(self.btn_load, stretch=3)
        btn_row.addWidget(self.btn_clear, stretch=1)
        file_layout.addLayout(btn_row)

        self.cmb_file_list = QComboBox()
        self.cmb_file_list.currentIndexChanged.connect(self.on_file_dropdown_changed)
        file_layout.addWidget(QLabel("Active File List:"))
        file_layout.addWidget(self.cmb_file_list)

        nav_box = QHBoxLayout()
        self.btn_prev = QPushButton("▲ Prev Line")
        self.btn_prev.clicked.connect(self.prev_file)
        self.lbl_counter = QLabel("0 / 0 Lines")
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        self.btn_next = QPushButton("▼ Next Line")
        self.btn_next.clicked.connect(self.next_file)
        nav_box.addWidget(self.btn_prev)
        nav_box.addWidget(self.lbl_counter)
        nav_box.addWidget(self.btn_next)
        file_layout.addLayout(nav_box)

        control_layout.addWidget(file_group)

        # 2. Graph Profile Slot Tabs
        plot_group = QGroupBox("2. Profile Display Selectors")
        plot_layout = QVBoxLayout(plot_group)
        self.tab_slots = QTabWidget()
        self.slot_controls = []

        for slot_idx in range(4):
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)

            chk_enable = QCheckBox(f"Enable Slot {slot_idx + 1}")
            chk_enable.setChecked(True if slot_idx < 2 else False)
            chk_enable.stateChanged.connect(self.update_plots)

            chk_dual = QCheckBox("Dual Y-Axis Scale")
            chk_dual.stateChanged.connect(self.update_plots)

            tab_layout.addWidget(chk_enable)
            tab_layout.addWidget(chk_dual)

            ch_combos = []
            for ch_idx in range(4):
                row_box = QHBoxLayout()
                row_box.addWidget(QLabel(f"Profile {ch_idx + 1}:"))
                cmb = QComboBox()
                cmb.setEnabled(chk_enable.isChecked())
                cmb.currentIndexChanged.connect(self.update_plots)
                row_box.addWidget(cmb)
                tab_layout.addLayout(row_box)
                ch_combos.append(cmb)

            chk_enable.toggled.connect(
                lambda checked, combos=ch_combos: [c.setEnabled(checked) for c in combos]
            )

            self.slot_controls.append({
                "chk_enable": chk_enable,
                "chk_dual_axis": chk_dual,
                "combos": ch_combos
            })
            self.tab_slots.addTab(tab_widget, f"Slot {slot_idx + 1}")

        plot_layout.addWidget(self.tab_slots)
        control_layout.addWidget(plot_group)
        control_layout.addStretch()

        # Center Area: Table + 1D Matplotlib Stack Splitter
        center_splitter = QSplitter(Qt.Vertical)

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_table_info = QLabel("<b>Full Dataset Overview:</b> 0 rows loaded.")
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setAlternatingRowColors(True)
        table_layout.addWidget(self.lbl_table_info)
        table_layout.addWidget(self.table_view)

        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = plt.figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas)

        center_splitter.addWidget(table_container)
        center_splitter.addWidget(canvas_container)
        center_splitter.setSizes([280, 620])

        main_layout.addWidget(control_panel)
        main_layout.addWidget(center_splitter, stretch=1)

        # Right Area: 2D Spatial Map Dock Window
        self.map_dock = QDockWidget("2D Spatial Map View", self)
        self.spatial_map = SpatialMapWidget(self)
        self.map_dock.setWidget(self.spatial_map)
        self.addDockWidget(Qt.RightDockWidgetArea, self.map_dock)

        # Connect 1D Profile Canvas Mouse Events for Crosshair & Anomaly Picking
        self.canvas.mpl_connect("motion_notify_event", self.on_profile_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_profile_mouse_click)

    # -------------------------------------------------------------------------
    # Tri-Linked Sync & Interactivity Logic
    # -------------------------------------------------------------------------
    def on_sync_toggled(self, checked: bool):
        """Toggles real-time linkage between Profiles, Table, and 2D Map."""
        self.spatial_map.set_sync_enabled(checked)
        if not checked:
            # Clear crosshair lines if sync turned off
            for line in self.crosshair_lines:
                line.set_visible(False)
            self.canvas.draw_idle()
        self.spatial_map.plot_map()

    def on_profile_mouse_move(self, event):
        """Renders interactive crosshairs and updates spatial cursor location."""
        if not self.chk_tri_sync.isChecked() or event.inaxes is None or event.xdata is None or self.df is None:
            return

        idx = int(round(event.xdata))
        if 0 <= idx < len(self.df):
            # 1. Move vertical crosshairs on 1D profile stack
            for line in self.crosshair_lines:
                line.set_xdata([idx, idx])
                line.set_visible(True)
            self.canvas.draw_idle()

            # 2. Sync to 2D Spatial Map
            target_path = self.file_paths[self.current_file_idx]
            self.spatial_map.update_map_data(
                self.dataset_cache, self.anomaly_picks, target_path, active_idx=idx
            )

            # 3. Sync to Table View
            self.table_view.selectRow(idx)

    def on_profile_mouse_click(self, event):
        """Adds or removes Anomaly Target picks on the 1D Profile and 2D Map."""
        if not self.chk_picker_mode.isChecked() or event.inaxes is None or event.xdata is None or self.df is None:
            return

        target_path = self.file_paths[self.current_file_idx]
        if target_path not in self.anomaly_picks:
            self.anomaly_picks[target_path] = []

        idx = int(round(event.xdata))
        if 0 <= idx < len(self.df):
            if event.button == 1:  # Left Click -> Add Anomaly Target
                if idx not in self.anomaly_picks[target_path]:
                    self.anomaly_picks[target_path].append(idx)
            elif event.button == 3:  # Right Click -> Remove Target
                if idx in self.anomaly_picks[target_path]:
                    self.anomaly_picks[target_path].remove(idx)

            self.update_plots()
            self.spatial_map.update_map_data(
                self.dataset_cache, self.anomaly_picks, target_path, active_idx=idx
            )

    def switch_to_line_by_path(self, path: Path):
        """Called when user clicks a line on the 2D Map."""
        if path in self.file_paths:
            idx = self.file_paths.index(path)
            self.current_file_idx = idx
            self.cmb_file_list.setCurrentIndex(idx)

    # -------------------------------------------------------------------------
    # Data Ingestion & Plotting Logic
    # -------------------------------------------------------------------------
    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Survey ASCII Files", "", "Text Files (*.txt *.xyz *.csv)"
        )
        if not files:
            return

        new_paths = [Path(f) for f in files]
        self.cmb_file_list.blockSignals(True)
        for p in new_paths:
            if p not in self.file_paths:
                self.file_paths.append(p)
                self.cmb_file_list.addItem(p.name)
        self.cmb_file_list.blockSignals(False)

        self.current_file_idx = len(self.file_paths) - 1
        self.cmb_file_list.setCurrentIndex(self.current_file_idx)
        self.update_nav_controls()
        self.display_current_file()

    def clear_file_list(self):
        self.file_paths.clear()
        self.dataset_cache.clear()
        self.anomaly_picks.clear()
        self.current_file_idx = -1
        self.df = None

        self.cmb_file_list.blockSignals(True)
        self.cmb_file_list.clear()
        self.cmb_file_list.blockSignals(False)

        self.table_model.update_data(pd.DataFrame())
        self.lbl_table_info.setText("<b>Full Dataset Overview:</b> 0 rows loaded.")
        self.update_nav_controls()

        self.figure.clear()
        self.canvas.draw()
        self.spatial_map.update_map_data({}, {})

    def on_file_dropdown_changed(self, index: int):
        if index < 0 or index >= len(self.file_paths):
            return
        self.current_file_idx = index
        self.update_nav_controls()
        self.display_current_file()

    def prev_file(self):
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.cmb_file_list.setCurrentIndex(self.current_file_idx)

    def next_file(self):
        if self.current_file_idx < len(self.file_paths) - 1:
            self.current_file_idx += 1
            self.cmb_file_list.setCurrentIndex(self.current_file_idx)

    def update_nav_controls(self):
        total = len(self.file_paths)
        curr_num = self.current_file_idx + 1 if total > 0 else 0
        self.lbl_counter.setText(f"{curr_num} / {total} Lines")
        self.btn_prev.setEnabled(self.current_file_idx > 0)
        self.btn_next.setEnabled(self.current_file_idx < total - 1)

    def display_current_file(self):
        if not self.file_paths or self.current_file_idx < 0:
            return

        target_path = self.file_paths[self.current_file_idx]

        if target_path in self.dataset_cache:
            self.df = self.dataset_cache[target_path]
        else:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.df = load_dataset(target_path)
                self.dataset_cache[target_path] = self.df
            except Exception as e:
                self.lbl_table_info.setText(f"<font color='red'>Error: {str(e)}</font>")
                return
            finally:
                QApplication.restoreOverrideCursor()

        self.table_model.update_data(self.df)
        self.lbl_table_info.setText(
            f"<b>File [{target_path.name}]:</b> {len(self.df):,} rows loaded."
        )

        numeric_cols = ["-- None --"] + self.df.select_dtypes(include=[np.number]).columns.tolist()

        for slot_idx, slot in enumerate(self.slot_controls):
            for ch_idx, cmb in enumerate(slot["combos"]):
                cur_text = cmb.currentText()
                cmb.blockSignals(True)
                cmb.clear()
                cmb.addItems(numeric_cols)
                if cur_text in numeric_cols:
                    cmb.setCurrentText(cur_text)
                elif ch_idx == 0 and len(numeric_cols) > (slot_idx + 1):
                    cmb.setCurrentIndex(slot_idx + 1)
                else:
                    cmb.setCurrentIndex(0)
                cmb.blockSignals(False)

        self.update_plots()
        self.spatial_map.update_map_data(
            self.dataset_cache, self.anomaly_picks, target_path, active_idx=0
        )

    def update_plots(self):
        if self.df is None:
            return

        x_indices = self.df.index.values
        self.figure.clear()
        self.crosshair_lines.clear()

        active_slots = [s for s in self.slot_controls if s["chk_enable"].isChecked()]
        if not active_slots:
            self.canvas.draw()
            return

        axes = self.figure.subplots(nrows=4, ncols=1, sharex=True)
        target_path = self.file_paths[self.current_file_idx]
        picks = self.anomaly_picks.get(target_path, [])

        for slot_idx in range(4):
            ax = axes[slot_idx]
            slot = self.slot_controls[slot_idx]

            if not slot["chk_enable"].isChecked():
                ax.set_visible(False)
                continue

            selected_channels = [
                cmb.currentText() for cmb in slot["combos"]
                if cmb.currentText() and cmb.currentText() != "-- None --" and cmb.currentText() in self.df.columns
            ]

            if not selected_channels:
                ax.set_visible(False)
                continue

            ax.set_visible(True)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Create crosshair vertical line for this axis
            cross_line = ax.axvline(x=0, color="red", linestyle=":", linewidth=1.2, visible=False)
            self.crosshair_lines.append(cross_line)

            # Plot Channels
            for idx, ch in enumerate(selected_channels):
                ax.plot(x_indices, self.df[ch].values, color=f"C{idx}", label=ch, linewidth=1.1)

            # Draw Anomaly Targets
            if picks:
                for p_idx in picks:
                    if 0 <= p_idx < len(self.df):
                        val = self.df[selected_channels[0]].values[p_idx]
                        ax.scatter([p_idx], [val], color="red", marker="*", s=110, zorder=5)

            ax.set_ylabel("/".join(selected_channels[:2]), fontsize=8)
            ax.legend(loc="upper right", fontsize=8)

        visible_axes = [ax for ax in axes if ax.get_visible()]
        if visible_axes:
            visible_axes[-1].set_xlabel(f"Fiducial Index [{target_path.name}]", fontsize=9)

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagAnomalyMainWindow()
    window.show()
    sys.exit(app.exec_())