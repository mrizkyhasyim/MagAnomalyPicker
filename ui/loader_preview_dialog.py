import sys
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QDockWidget,
    QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QPushButton, QRadioButton, QScrollBar, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QTableView, QTextEdit,
    QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QWizard, QWizardPage
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 1. Target Database Dialog
# =============================================================================
class TargetDatabaseDialog(QDialog):
    """Dialog displaying all picked anomaly targets with full database attributes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anomaly Target Database")
        self.resize(950, 450)
        self.main_app = parent

        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.lbl_count = QLabel("<b>Total Targets:</b> 0")
        self.btn_delete = QPushButton("Delete Selected Target")
        self.btn_delete.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 5px;")
        self.btn_delete.clicked.connect(self.delete_selected_target)

        self.btn_export = QPushButton("Export Targets (.csv)")
        self.btn_export.clicked.connect(self.export_csv)

        top_bar.addWidget(self.lbl_count)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_delete)
        top_bar.addWidget(self.btn_export)
        layout.addLayout(top_bar)

        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_model = QStandardItemModel()
        self.table_view.setModel(self.table_model)
        layout.addWidget(self.table_view)

        self.refresh_database()

    def refresh_database(self):
        self.table_model.clear()
        if not self.main_app or not self.main_app.anomaly_picks:
            self.lbl_count.setText("<b>Total Targets:</b> 0")
            return

        target_counter = 1
        headers_set = False

        for path, pick_indices in self.main_app.anomaly_picks.items():
            if path not in self.main_app.dataset_cache or not pick_indices:
                continue

            df = self.main_app.dataset_cache[path]

            for fid_idx in sorted(pick_indices):
                if 0 <= fid_idx < len(df):
                    row_data = df.iloc[fid_idx]

                    if not headers_set:
                        headers = ["Target_ID", "Line_Name", "Fiducial_Index"] + list(df.columns)
                        self.table_model.setHorizontalHeaderLabels(headers)
                        headers_set = True

                    items = [
                        QStandardItem(f"TRG_{target_counter:03d}"),
                        QStandardItem(path.name),
                        QStandardItem(str(fid_idx))
                    ]
                    for col in df.columns:
                        val = row_data[col]
                        items.append(QStandardItem("" if pd.isna(val) else str(val)))

                    items[0].setData((path, fid_idx), Qt.UserRole)
                    self.table_model.appendRow(items)
                    target_counter += 1

        self.lbl_count.setText(f"<b>Total Targets:</b> {self.table_model.rowCount()}")
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def delete_selected_target(self):
        selected_indexes = self.table_view.selectionModel().selectedRows()
        if not selected_indexes:
            return

        for index in selected_indexes:
            item = self.table_model.item(index.row(), 0)
            data = item.data(Qt.UserRole)
            if data:
                path, fid_idx = data
                if path in self.main_app.anomaly_picks and fid_idx in self.main_app.anomaly_picks[path]:
                    self.main_app.anomaly_picks[path].remove(fid_idx)

        self.refresh_database()
        self.main_app.update_plots()
        self.main_app.spatial_map.plot_map()

    def export_csv(self):
        if self.table_model.rowCount() == 0:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Targets CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        headers = [self.table_model.headerData(i, Qt.Horizontal) for i in range(self.table_model.columnCount())]
        data = []
        for r in range(self.table_model.rowCount()):
            row = [self.table_model.item(r, c).text() for c in range(self.table_model.columnCount())]
            data.append(row)

        export_df = pd.DataFrame(data, columns=headers)
        export_df.to_csv(file_path, index=False)


# =============================================================================
# 2. 3-Step Data Import Wizard
# =============================================================================
class ImportWizardStep1(QWizardPage):
    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.setTitle("Data Import Wizard - Step 1 of 3")
        self.setSubTitle("Please choose the data type that best describes your data file.")
        self.file_path = file_path

        layout = QVBoxLayout(self)

        group_type = QGroupBox("File Type")
        g_layout = QVBoxLayout(group_type)
        self.rad_delimited = QRadioButton("Delimited\n   Spaces or commas separate each data field")
        self.rad_delimited.setChecked(True)
        self.rad_fixed = QRadioButton("Fixed Field\n   Data is aligned in fixed width columns")
        g_layout.addWidget(self.rad_delimited)
        g_layout.addWidget(self.rad_fixed)
        layout.addWidget(group_type)

        row_settings = QHBoxLayout()
        row_settings.addWidget(QLabel("Data headings on row:"))
        self.txt_heading_row = QLineEdit("1")
        self.txt_heading_row.setFixedWidth(50)
        row_settings.addWidget(self.txt_heading_row)

        row_settings.addWidget(QLabel("Start import on row:"))
        self.txt_start_row = QLineEdit("2")
        self.txt_start_row.setFixedWidth(50)
        row_settings.addWidget(self.txt_start_row)

        row_settings.addWidget(QLabel("Preview rows:"))
        self.txt_preview_rows = QLineEdit("500")
        self.txt_preview_rows.setFixedWidth(50)
        row_settings.addWidget(self.txt_preview_rows)
        layout.addLayout(row_settings)

        layout.addWidget(QLabel(f"File: {self.file_path.name}"))
        self.txt_preview = QTextEdit()
        self.txt_preview.setFont(QFont("Courier New", 8))
        self.txt_preview.setReadOnly(True)
        layout.addWidget(self.txt_preview)

        self.load_raw_preview()

    def load_raw_preview(self):
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(100)]
            self.txt_preview.setText("".join([f"{i+1:3d}  {line}" for i, line in enumerate(lines)]))
        except Exception as e:
            self.txt_preview.setText(f"Error reading file preview: {str(e)}")


class ImportWizardStep2(QWizardPage):
    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.setTitle("Data Import Wizard - Step 2 of 3")
        self.setSubTitle("Select column delimiters and string quote handling options.")
        self.file_path = file_path

        layout = QVBoxLayout(self)

        opts_layout = QHBoxLayout()
        delim_group = QGroupBox("Column delimiters")
        d_layout = QVBoxLayout(delim_group)
        self.rad_comma = QRadioButton("Comma Delimited")
        self.rad_comma.setChecked(True)
        self.rad_whitespace = QRadioButton("White Space Delimited")
        self.rad_tab = QRadioButton("Tab Delimited")
        self.rad_other = QRadioButton("Other")
        d_layout.addWidget(self.rad_comma)
        d_layout.addWidget(self.rad_whitespace)
        d_layout.addWidget(self.rad_tab)
        d_layout.addWidget(self.rad_other)
        opts_layout.addWidget(delim_group)

        str_group = QGroupBox("String handling")
        s_layout = QVBoxLayout(str_group)
        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("Quote character:"))
        self.txt_quote = QLineEdit('"')
        self.txt_quote.setFixedWidth(40)
        s_row.addWidget(self.txt_quote)
        s_layout.addLayout(s_row)

        self.chk_escape = QCheckBox("Escapes?")
        s_layout.addWidget(self.chk_escape)
        s_layout.addStretch()
        opts_layout.addWidget(str_group)

        layout.addLayout(opts_layout)

        layout.addWidget(QLabel("Parsed Data Preview:"))
        self.tbl_preview = QTableWidget()
        layout.addWidget(self.tbl_preview)

        self.rad_comma.toggled.connect(self.update_parsed_preview)
        self.rad_whitespace.toggled.connect(self.update_parsed_preview)
        self.rad_tab.toggled.connect(self.update_parsed_preview)

    def initializePage(self):
        self.update_parsed_preview()

    def update_parsed_preview(self):
        delimiter = "," if self.rad_comma.isChecked() else (r"\s+" if self.rad_whitespace.isChecked() else "\t")
        try:
            df = pd.read_csv(self.file_path, sep=delimiter, nrows=20, engine="python")
            self.tbl_preview.setRowCount(len(df))
            self.tbl_preview.setColumnCount(len(df.columns))
            self.tbl_preview.setHorizontalHeaderLabels([str(c) for c in df.columns])

            for r in range(len(df)):
                for c in range(len(df.columns)):
                    val = str(df.iat[r, c]) if not pd.isna(df.iat[r, c]) else ""
                    self.tbl_preview.setItem(r, c, QTableWidgetItem(val))
        except Exception:
            pass


class ImportWizardStep3(QWizardPage):
    """Step 3: Column channel assignment, data types, and live header updates."""

    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.setTitle("Data Import Wizard - Step 3 of 3")
        self.setSubTitle("Click on each data column and specify import parameters.")
        self.file_path = file_path
        self.selected_col_idx = 0
        self.column_settings: Dict[int, dict] = {}
        self.is_updating_ui = False

        layout = QVBoxLayout(self)
        param_layout = QHBoxLayout()

        type_group = QGroupBox("Channel Type")
        t_layout = QVBoxLayout(type_group)
        self.rad_not_imported = QRadioButton("Not Imported")
        self.rad_data = QRadioButton("Data")
        self.rad_data.setChecked(True)
        self.rad_line = QRadioButton("Line")
        t_layout.addWidget(self.rad_not_imported)
        t_layout.addWidget(self.rad_data)
        t_layout.addWidget(self.rad_line)
        param_layout.addWidget(type_group)

        params_group = QGroupBox("Parameters")
        p_layout = QVBoxLayout(params_group)
        p_row1 = QHBoxLayout()
        p_row1.addWidget(QLabel("Channel name:"))
        self.txt_chan_name = QLineEdit()
        p_row1.addWidget(self.txt_chan_name)
        p_layout.addLayout(p_row1)

        p_row2 = QHBoxLayout()
        p_row2.addWidget(QLabel("Label:"))
        self.txt_label = QLineEdit()
        p_row2.addWidget(self.txt_label)
        p_layout.addLayout(p_row2)

        p_row3 = QHBoxLayout()
        p_row3.addWidget(QLabel("Data Type:"))
        self.cmb_data_type = QComboBox()
        self.cmb_data_type.addItems(["Float", "Double", "String", "Integer"])
        p_row3.addWidget(self.cmb_data_type)
        p_layout.addLayout(p_row3)

        param_layout.addWidget(params_group)
        layout.addLayout(param_layout)

        self.tbl_channels = QTableWidget()
        self.tbl_channels.setSelectionBehavior(QTableWidget.SelectColumns)
        self.tbl_channels.cellClicked.connect(self.on_column_selected)
        layout.addWidget(self.tbl_channels)

        # Connect parameter modifications to live state saving
        self.txt_chan_name.textChanged.connect(self.on_setting_changed)
        self.txt_label.textChanged.connect(self.on_setting_changed)
        self.cmb_data_type.currentIndexChanged.connect(self.on_setting_changed)
        self.rad_not_imported.toggled.connect(self.on_setting_changed)
        self.rad_data.toggled.connect(self.on_setting_changed)
        self.rad_line.toggled.connect(self.on_setting_changed)

    def initializePage(self):
        sep_step = self.wizard().page(1)
        delimiter = "," if sep_step.rad_comma.isChecked() else (r"\s+" if sep_step.rad_whitespace.isChecked() else "\t")
        try:
            self.df_preview = pd.read_csv(self.file_path, sep=delimiter, nrows=15, engine="python")
            num_cols = len(self.df_preview.columns)
            self.tbl_channels.setRowCount(len(self.df_preview))
            self.tbl_channels.setColumnCount(num_cols)

            self.column_settings.clear()
            headers = []
            for c in range(num_cols):
                col_name = str(self.df_preview.columns[c])
                headers.append(col_name)
                self.column_settings[c] = {
                    "name": col_name,
                    "label": col_name,
                    "data_type": "Float",
                    "chan_type": "Data"
                }

            self.tbl_channels.setHorizontalHeaderLabels(headers)

            for r in range(len(self.df_preview)):
                for c in range(num_cols):
                    val = str(self.df_preview.iat[r, c]) if not pd.isna(self.df_preview.iat[r, c]) else ""
                    self.tbl_channels.setItem(r, c, QTableWidgetItem(val))

            self.selected_col_idx = 0
            self.load_column_settings(0)
        except Exception:
            pass

    def load_column_settings(self, col: int):
        self.is_updating_ui = True
        cfg = self.column_settings.get(col, {
            "name": f"Col_{col}", "label": f"Col_{col}", "data_type": "Float", "chan_type": "Data"
        })

        self.txt_chan_name.setText(cfg["name"])
        self.txt_label.setText(cfg["label"])

        idx = self.cmb_data_type.findText(cfg["data_type"])
        if idx >= 0:
            self.cmb_data_type.setCurrentIndex(idx)

        if cfg["chan_type"] == "Not Imported":
            self.rad_not_imported.setChecked(True)
        elif cfg["chan_type"] == "Line":
            self.rad_line.setChecked(True)
        else:
            self.rad_data.setChecked(True)

        self.is_updating_ui = False

    def on_column_selected(self, row: int, col: int):
        self.selected_col_idx = col
        self.load_column_settings(col)

    def on_setting_changed(self):
        if self.is_updating_ui or self.selected_col_idx not in self.column_settings:
            return

        chan_type = "Data"
        if self.rad_not_imported.isChecked():
            chan_type = "Not Imported"
        elif self.rad_line.isChecked():
            chan_type = "Line"

        new_name = self.txt_chan_name.text().strip() or f"Col_{self.selected_col_idx}"
        self.column_settings[self.selected_col_idx] = {
            "name": new_name,
            "label": self.txt_label.text().strip(),
            "data_type": self.cmb_data_type.currentText(),
            "chan_type": chan_type
        }

        # Dynamically update the table header title in real-time
        item = QTableWidgetItem(new_name)
        self.tbl_channels.setHorizontalHeaderItem(self.selected_col_idx, item)


class DataImportWizard(QWizard):
    """3-Step Import Wizard container."""

    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Import Wizard")
        self.resize(750, 550)
        self.file_path = file_path

        self.step1 = ImportWizardStep1(file_path, self)
        self.step2 = ImportWizardStep2(file_path, self)
        self.step3 = ImportWizardStep3(file_path, self)

        self.addPage(self.step1)
        self.addPage(self.step2)
        self.addPage(self.step3)

    def get_parsed_dataframe(self) -> pd.DataFrame:
        delimiter = "," if self.step2.rad_comma.isChecked() else (r"\s+" if self.step2.rad_whitespace.isChecked() else "\t")
        start_row = max(0, int(self.step1.txt_start_row.text().strip()) - 1)

        # Parse full file
        df = pd.read_csv(self.file_path, sep=delimiter, skiprows=start_row, engine="python")

        # Apply Step 3 custom column names & dropped columns
        col_settings = self.step3.column_settings
        rename_map = {}
        drop_cols = []

        for col_idx, orig_col_name in enumerate(df.columns):
            if col_idx in col_settings:
                cfg = col_settings[col_idx]
                if cfg["chan_type"] == "Not Imported":
                    drop_cols.append(orig_col_name)
                else:
                    rename_map[orig_col_name] = cfg["name"]

        if drop_cols:
            df.drop(columns=drop_cols, inplace=True, errors="ignore")

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        return df


# =============================================================================
# 3. Docks & Tables
# =============================================================================
class ProjectExplorerWidget(QWidget):
    database_selected = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        lbl_header = QLabel("<b>Project Explorer</b>")
        lbl_header.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Data Workspace")
        layout.addWidget(self.tree)

        self.root_data = QTreeWidgetItem(self.tree, ["Data"])
        self.root_data.setExpanded(True)

        self.node_db = QTreeWidgetItem(self.root_data, ["Databases"])
        self.node_grids = QTreeWidgetItem(self.root_data, ["Grids"])
        self.node_maps = QTreeWidgetItem(self.root_data, ["Maps"])

        self.node_db.setExpanded(True)
        self.node_grids.setExpanded(True)
        self.node_maps.setExpanded(True)

        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

    def update_databases(self, db_paths: List[Path], active_path: Optional[Path] = None):
        self.node_db.takeChildren()
        sorted_paths = sorted(db_paths, key=lambda p: p.name.lower())

        for p in sorted_paths:
            item = QTreeWidgetItem(self.node_db, [p.name])
            item.setData(0, Qt.UserRole, p)
            if active_path and p == active_path:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        data_path = item.data(0, Qt.UserRole)
        if data_path and isinstance(data_path, Path):
            self.database_selected.emit(data_path)


class MapLayerManagerWidget(QWidget):
    layer_visibility_changed = pyqtSignal(str, bool)
    layer_highlight_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        lbl_header = QLabel("<b>Map Layer Manager</b>")
        lbl_header.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Map Objects")
        layout.addWidget(self.tree)

        self.root_data = QTreeWidgetItem(self.tree, ["Data Layers"])
        self.root_data.setExpanded(True)

        self.tree.itemChanged.connect(self.on_item_changed)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

    def set_layers(self, layer_names: List[str]):
        self.tree.blockSignals(True)
        self.root_data.takeChildren()

        for name in layer_names:
            item = QTreeWidgetItem(self.root_data, [name])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)

        self.tree.blockSignals(False)

    def on_item_changed(self, item: QTreeWidgetItem, column: int):
        layer_name = item.text(0)
        is_visible = (item.checkState(0) == Qt.Checked)
        self.layer_visibility_changed.emit(layer_name, is_visible)

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        layer_name = item.text(0)
        self.layer_highlight_requested.emit(layer_name)


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
# 4. 2D Spatial Map View Widget
# =============================================================================
class SpatialMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_app = parent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        ctrl_bar = QHBoxLayout()
        self.lbl_info = QLabel("<b>2D Spatial Survey Map</b>")
        self.chk_equal_aspect = QCheckBox("Equal Aspect (1:1)")
        self.chk_equal_aspect.setChecked(True)
        self.chk_equal_aspect.stateChanged.connect(self.plot_map)

        ctrl_bar.addWidget(self.lbl_info)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.chk_equal_aspect)
        layout.addLayout(ctrl_bar)

        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.dataset_cache: Dict[Path, pd.DataFrame] = {}
        self.anomaly_picks: Dict[Path, List[int]] = {}
        self.layer_visibility: Dict[str, bool] = {}
        self.active_path: Path | None = None
        self.active_idx: int | None = None
        self.highlighted_layer: str | None = None
        self.sync_enabled: bool = True

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

        for p in self.dataset_cache:
            layer_key = f"PATH_{p.stem}"
            if layer_key not in self.layer_visibility:
                self.layer_visibility[layer_key] = True

        self.plot_map()

    def set_layer_visibility(self, layer_name: str, visible: bool):
        self.layer_visibility[layer_name] = visible
        self.plot_map()

    def set_highlight_layer(self, layer_name: str):
        self.highlighted_layer = layer_name
        self.plot_map()

    def plot_map(self):
        self.ax.clear()

        if not self.dataset_cache:
            self.ax.text(0.5, 0.5, "No Spatial Data Loaded", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return

        has_coords = False

        for path, df in self.dataset_cache.items():
            layer_key = f"PATH_{path.stem}"
            if not self.layer_visibility.get(layer_key, True):
                continue

            if "Easting" in df.columns and "Northing" in df.columns:
                has_coords = True
                is_active = (path == self.active_path)
                east = df["Easting"].values
                north = df["Northing"].values

                if is_active:
                    self.ax.plot(east, north, color="red", linewidth=2.0, zorder=3, label=f"Active: {path.name}")

                    if self.sync_enabled and self.active_idx is not None and 0 <= self.active_idx < len(df):
                        cx, cy = east[self.active_idx], north[self.active_idx]
                        self.ax.scatter([cx], [cy], color="yellow", edgecolor="black", s=90, zorder=6)

                    if path in self.anomaly_picks and self.anomaly_picks[path]:
                        pick_indices = self.anomaly_picks[path]
                        px = east[pick_indices]
                        py = north[pick_indices]
                        self.ax.scatter(px, py, color="cyan", marker="*", s=120, edgecolor="black", zorder=5)
                else:
                    self.ax.plot(east, north, color="#7f8c8d", linewidth=0.8, alpha=0.5, zorder=2)

                if self.highlighted_layer == layer_key:
                    min_x, max_x = np.min(east), np.max(east)
                    min_y, max_y = np.min(north), np.max(north)
                    rect = Rectangle(
                        (min_x, min_y), max_x - min_x, max_y - min_y,
                        linewidth=1.8, edgecolor="blue", facecolor="none", linestyle="--"
                    )
                    self.ax.add_patch(rect)

        if not has_coords:
            self.ax.text(0.5, 0.5, "Missing 'Easting' / 'Northing' Columns", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return

        self.ax.set_xlabel("Easting (m)", fontsize=8)
        self.ax.set_ylabel("Northing (m)", fontsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.4)

        if self.chk_equal_aspect.isChecked():
            self.ax.set_aspect("equal", adjustable="datalim")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_map_click(self, event):
        if not self.sync_enabled or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        click_x, click_y = event.xdata, event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        tolerance = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) * 0.05

        closest_path = None
        min_dist = float("inf")

        for path, df in self.dataset_cache.items():
            if "Easting" in df.columns and "Northing" in df.columns:
                dx = df["Easting"].values - click_x
                dy = df["Northing"].values - click_y
                dist = np.min(np.sqrt(dx**2 + dy**2))
                if dist < min_dist:
                    min_dist = dist
                    closest_path = path

        if closest_path and min_dist <= tolerance:
            if hasattr(self.main_app, "switch_to_line_by_path"):
                self.main_app.switch_to_line_by_path(closest_path)


# =============================================================================
# 5. Main Application Window
# =============================================================================
class MagAnomalyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MagAnomalyPicker - Standalone Processing Suite")
        self.resize(1600, 950)

        self.file_paths: List[Path] = []
        self.current_file_idx: int = -1
        self.dataset_cache: Dict[Path, pd.DataFrame] = {}
        self.anomaly_picks: Dict[Path, List[int]] = {}
        self.df: pd.DataFrame | None = None

        self.crosshair_lines = []
        self.table_model = PandasTableModel()
        self.target_db_dialog = None

        self.setup_docks()
        self.setup_main_layout()
        self.setup_menu_bar()

    def setup_docks(self):
        # Left Dock: Project Explorer
        self.project_dock = QDockWidget("Project Explorer", self)
        self.project_explorer = ProjectExplorerWidget(self)
        self.project_explorer.database_selected.connect(self.switch_to_line_by_path)
        self.project_dock.setWidget(self.project_explorer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_dock)

        # Right Dock 1: 2D Spatial Map
        self.map_dock = QDockWidget("2D Spatial Map View", self)
        self.spatial_map = SpatialMapWidget(self)
        self.map_dock.setWidget(self.spatial_map)
        self.addDockWidget(Qt.RightDockWidgetArea, self.map_dock)

        # Right Dock 2: Map Layer Manager
        self.layer_dock = QDockWidget("Map Layer Manager", self)
        self.layer_manager = MapLayerManagerWidget(self)
        self.layer_manager.layer_visibility_changed.connect(self.spatial_map.set_layer_visibility)
        self.layer_manager.layer_highlight_requested.connect(self.spatial_map.set_highlight_layer)
        self.layer_dock.setWidget(self.layer_manager)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layer_dock)

    def setup_menu_bar(self):
        menu_bar = self.menuBar()

        # 1. File Menu
        file_menu = menu_bar.addMenu("File")
        project_menu = file_menu.addMenu("Project")
        project_menu.addAction("New Project")
        project_menu.addAction("Open Project...")
        project_menu.addAction("Save")
        project_menu.addAction("Close Project")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # 2. View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(self.project_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())
        view_menu.addAction(self.layer_dock.toggleViewAction())

        # 3. Database Menu
        db_menu = menu_bar.addMenu("Database")
        db_menu.addAction("Import ASCII / Data Wizard...", self.open_import_wizard)

        # 4. Database Tools Menu (Restored)
        db_tools = menu_bar.addMenu("Database Tools")
        chan_tools = db_tools.addMenu("Channel Tools")
        chan_tools.addAction("Copy Channels...")
        chan_tools.addAction("Make Diff (Difference) Channel")

        filter_menu = db_tools.addMenu("Filter")
        filter_menu.addAction("Low Pass Filter...")
        filter_menu.addAction("High Pass Filter...")

        # 5. Anomaly Menu (Restored)
        anomaly_menu = menu_bar.addMenu("Anomaly")

        self.action_picker_mode = QAction("Manual Anomaly Picker Mode", self, checkable=True)
        self.action_picker_mode.triggered.connect(lambda checked: self.chk_picker_mode.setChecked(checked))
        anomaly_menu.addAction(self.action_picker_mode)

        auto_picker_action = QAction("Automatic Anomaly Picker...", self)
        auto_picker_action.triggered.connect(lambda: self.statusBar().showMessage("Auto Anomaly Picker triggered.", 3000))
        anomaly_menu.addAction(auto_picker_action)

        anomaly_menu.addSeparator()
        target_db_action = QAction("Target Database Workspace...", self)
        target_db_action.triggered.connect(self.open_target_database)
        anomaly_menu.addAction(target_db_action)

    def setup_main_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(320)

        # 1. Profile Display Selectors
        plot_group = QGroupBox("1. Profile Display Selectors")
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

            self.slot_controls.append({
                "chk_enable": chk_enable,
                "chk_dual_axis": chk_dual,
                "combos": ch_combos
            })
            self.tab_slots.addTab(tab_widget, f"Slot {slot_idx + 1}")

        plot_layout.addWidget(self.tab_slots)
        control_layout.addWidget(plot_group)

        # 2. Graph Fiducial Range Controls
        x_range_group = QGroupBox("2. Graph Fiducial Range (X-Axis Zoom)")
        x_range_layout = QVBoxLayout(x_range_group)

        self.chk_custom_x = QCheckBox("Lock X-Axis Range (Fiducial)")
        self.chk_custom_x.setChecked(False)
        self.chk_custom_x.stateChanged.connect(self.update_plots)
        x_range_layout.addWidget(self.chk_custom_x)

        range_inputs = QHBoxLayout()
        range_inputs.addWidget(QLabel("Min Fid:"))
        self.txt_xmin = QLineEdit("0")
        self.txt_xmin.setFixedWidth(65)
        range_inputs.addWidget(self.txt_xmin)

        range_inputs.addWidget(QLabel("Max Fid:"))
        self.txt_xmax = QLineEdit("1000")
        self.txt_xmax.setFixedWidth(65)
        range_inputs.addWidget(self.txt_xmax)
        x_range_layout.addLayout(range_inputs)

        btn_x_row = QHBoxLayout()
        self.btn_apply_x = QPushButton("Apply Zoom")
        self.btn_apply_x.clicked.connect(self.apply_custom_x_range)
        self.btn_reset_x = QPushButton("Reset Full Line")
        self.btn_reset_x.clicked.connect(self.reset_x_range)
        btn_x_row.addWidget(self.btn_apply_x)
        btn_x_row.addWidget(self.btn_reset_x)
        x_range_layout.addLayout(btn_x_row)

        control_layout.addWidget(x_range_group)

        # 3. Linked Workspace & Picker Settings (Moved here)
        sync_group = QGroupBox("3. Linked Workspace & Picker Settings")
        sync_layout = QVBoxLayout(sync_group)

        self.chk_tri_sync = QCheckBox("🔗 Tri-Linked Sync (Map - Profile - Table)")
        self.chk_tri_sync.setChecked(True)
        self.chk_tri_sync.setStyleSheet("font-weight: bold; color: #16a085;")
        self.chk_tri_sync.toggled.connect(self.on_sync_toggled)
        sync_layout.addWidget(self.chk_tri_sync)

        self.chk_picker_mode = QCheckBox("🎯 Anomaly Picker Mode")
        self.chk_picker_mode.setChecked(False)
        self.chk_picker_mode.setStyleSheet("font-weight: bold; color: #c0392b;")
        self.chk_picker_mode.toggled.connect(lambda checked: self.action_picker_mode.setChecked(checked))
        sync_layout.addWidget(self.chk_picker_mode)

        control_layout.addWidget(sync_group)
        control_layout.addStretch()

        # Center Splitter: Table + Matplotlib Canvas + Scrollbar
        center_splitter = QSplitter(Qt.Vertical)

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_table_info = QLabel("<b>Full Dataset Overview:</b> 0 rows loaded.")
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.clicked.connect(self.on_table_row_clicked)
        table_layout.addWidget(self.lbl_table_info)
        table_layout.addWidget(self.table_view)

        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = plt.figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas)

        scroll_box = QHBoxLayout()
        scroll_box.setContentsMargins(5, 2, 5, 5)
        scroll_box.addWidget(QLabel("<b>(Fid)</b>"))
        self.fid_scrollbar = QScrollBar(Qt.Horizontal)
        self.fid_scrollbar.valueChanged.connect(self.on_fid_scrollbar_moved)
        scroll_box.addWidget(self.fid_scrollbar)
        canvas_layout.addLayout(scroll_box)

        center_splitter.addWidget(table_container)
        center_splitter.addWidget(canvas_container)
        center_splitter.setSizes([260, 640])

        main_layout.addWidget(control_panel)
        main_layout.addWidget(center_splitter, stretch=1)

        self.canvas.mpl_connect("button_press_event", self.on_profile_mouse_click)

    # -------------------------------------------------------------------------
    # Ingestion & Database Methods
    # -------------------------------------------------------------------------
    def open_import_wizard(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select ASCII Data File", "", "Text Files (*.txt *.xyz *.csv)")
        if not file_path:
            return

        target_path = Path(file_path)
        wizard = DataImportWizard(target_path, self)

        if wizard.exec_() == QDialog.Accepted:
            df = wizard.get_parsed_dataframe()
            if target_path not in self.file_paths:
                self.file_paths.append(target_path)

            self.file_paths.sort(key=lambda p: p.name.lower())
            self.dataset_cache[target_path] = df

            self.project_explorer.update_databases(self.file_paths, active_path=target_path)
            layer_names = [f"PATH_{p.stem}" for p in self.file_paths]
            self.layer_manager.set_layers(layer_names)

            self.switch_to_line_by_path(target_path)

    def switch_to_line_by_path(self, path: Path):
        if path in self.file_paths:
            self.current_file_idx = self.file_paths.index(path)
            self.df = self.dataset_cache[path]

            self.project_explorer.update_databases(self.file_paths, active_path=path)
            self.table_model.update_data(self.df)
            self.lbl_table_info.setText(f"<b>Database [{path.name}]:</b> {len(self.df):,} rows loaded.")

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

            self.update_scrollbar_state()
            self.update_plots()
            self.spatial_map.update_map_data(self.dataset_cache, self.anomaly_picks, active_path=path, active_idx=0)

    def open_target_database(self):
        if not self.target_db_dialog:
            self.target_db_dialog = TargetDatabaseDialog(self)
        self.target_db_dialog.refresh_database()
        self.target_db_dialog.show()

    # -------------------------------------------------------------------------
    # Sync & Mouse Click Handlers
    # -------------------------------------------------------------------------
    def on_sync_toggled(self, checked: bool):
        self.spatial_map.set_sync_enabled(checked)
        if not checked:
            for line in self.crosshair_lines:
                line.set_visible(False)
            self.canvas.draw_idle()
        self.spatial_map.plot_map()

    def on_profile_mouse_click(self, event):
        if event.inaxes is None or event.xdata is None or self.df is None:
            return

        idx = int(round(event.xdata))
        if not (0 <= idx < len(self.df)):
            return

        target_path = self.file_paths[self.current_file_idx]

        if self.chk_picker_mode.isChecked():
            if target_path not in self.anomaly_picks:
                self.anomaly_picks[target_path] = []

            if event.button == 1:
                if idx not in self.anomaly_picks[target_path]:
                    self.anomaly_picks[target_path].append(idx)
            elif event.button == 3:
                if idx in self.anomaly_picks[target_path]:
                    self.anomaly_picks[target_path].remove(idx)

            self.update_plots()
            if self.target_db_dialog and self.target_db_dialog.isVisible():
                self.target_db_dialog.refresh_database()

        if self.chk_tri_sync.isChecked():
            for line in self.crosshair_lines:
                line.set_xdata([idx, idx])
                line.set_visible(True)
            self.canvas.draw_idle()

            self.spatial_map.update_map_data(
                self.dataset_cache, self.anomaly_picks, target_path, active_idx=idx
            )
            self.table_view.selectRow(idx)

    def on_table_row_clicked(self, index):
        if not self.chk_tri_sync.isChecked() or self.df is None:
            return

        idx = index.row()
        target_path = self.file_paths[self.current_file_idx]

        for line in self.crosshair_lines:
            line.set_xdata([idx, idx])
            line.set_visible(True)
        self.canvas.draw_idle()

        self.spatial_map.update_map_data(
            self.dataset_cache, self.anomaly_picks, target_path, active_idx=idx
        )

    # -------------------------------------------------------------------------
    # Plotting & Zoom Controls
    # -------------------------------------------------------------------------
    def update_scrollbar_state(self):
        if self.df is None or len(self.df) == 0:
            return
        total_rows = len(self.df)
        try:
            xmin = float(self.txt_xmin.text().strip())
            xmax = float(self.txt_xmax.text().strip())
        except ValueError:
            xmin, xmax = 0, total_rows

        window_size = max(10, int(round(xmax - xmin)))
        self.fid_scrollbar.blockSignals(True)
        self.fid_scrollbar.setMaximum(max(0, total_rows - window_size))
        self.fid_scrollbar.setPageStep(window_size)
        self.fid_scrollbar.setValue(int(round(xmin)))
        self.fid_scrollbar.blockSignals(False)

    def on_fid_scrollbar_moved(self, value: int):
        if self.df is None:
            return
        total_rows = len(self.df)
        try:
            window_size = float(self.txt_xmax.text().strip()) - float(self.txt_xmin.text().strip())
        except ValueError:
            window_size = 1000

        self.txt_xmin.setText(f"{value}")
        self.txt_xmax.setText(f"{min(total_rows, value + int(window_size))}")
        self.chk_custom_x.setChecked(True)
        self.update_plots()

    def apply_custom_x_range(self):
        self.chk_custom_x.setChecked(True)
        self.update_scrollbar_state()
        self.update_plots()

    def reset_x_range(self):
        self.chk_custom_x.setChecked(False)
        if self.df is not None:
            self.txt_xmin.setText("0")
            self.txt_xmax.setText(str(len(self.df)))
        self.update_scrollbar_state()
        self.update_plots()

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

        use_custom_x = self.chk_custom_x.isChecked()
        xmin_val, xmax_val = None, None
        if use_custom_x:
            try:
                xmin_val = float(self.txt_xmin.text().strip())
                xmax_val = float(self.txt_xmax.text().strip())
            except ValueError:
                use_custom_x = False

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

            cross_line = ax.axvline(x=0, color="red", linestyle=":", linewidth=1.2, visible=False)
            self.crosshair_lines.append(cross_line)

            is_dual = slot["chk_dual_axis"].isChecked() and len(selected_channels) >= 2

            if is_dual:
                ch1 = selected_channels[0]
                line1 = ax.plot(x_indices, self.df[ch1].values, color="C0", label=ch1, linewidth=1.1)
                ax.set_ylabel(ch1, color="C0", fontsize=7.5)
                ax.tick_params(axis='y', labelcolor="C0", labelsize=7)
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

                ax2 = ax.twinx()
                ch2 = selected_channels[1]
                line2 = ax2.plot(x_indices, self.df[ch2].values, color="C1", label=ch2, linewidth=1.1)
                ax2.set_ylabel(ch2, color="C1", fontsize=7.5)
                ax2.tick_params(axis='y', labelcolor="C1", labelsize=7)
                ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

                lines_extra = []
                for idx, ch in enumerate(selected_channels[2:], start=2):
                    le = ax2.plot(x_indices, self.df[ch].values, color=f"C{idx}", label=ch, linewidth=1.1)
                    lines_extra.extend(le)

                all_lines = line1 + line2 + lines_extra
                labels = [l.get_label() for l in all_lines]
                ax.legend(all_lines, labels, loc="upper right", fontsize=7.5)
            else:
                for idx, ch in enumerate(selected_channels):
                    ax.plot(x_indices, self.df[ch].values, color=f"C{idx}", label=ch, linewidth=1.1)

                ax.set_ylabel("/".join(selected_channels[:2]), fontsize=7.5)
                ax.tick_params(axis='both', labelsize=7)
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
                ax.legend(loc="upper right", fontsize=7.5)

            if picks:
                for p_idx in picks:
                    if 0 <= p_idx < len(self.df):
                        val = self.df[selected_channels[0]].values[p_idx]
                        ax.scatter([p_idx], [val], color="red", marker="*", s=110, zorder=5)

            if use_custom_x and xmin_val is not None and xmax_val is not None:
                ax.set_xlim(xmin_val, xmax_val)

        visible_axes = [ax for ax in axes if ax.get_visible()]
        if visible_axes:
            visible_axes[-1].set_xlabel(f"Fiducial Index [{target_path.name}]", fontsize=8.5)
            visible_axes[-1].tick_params(axis='x', labelsize=7)

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagAnomalyMainWindow()
    window.show()
    sys.exit(app.exec_())