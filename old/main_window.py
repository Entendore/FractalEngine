import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
    QPushButton, QComboBox, QFileDialog, QStatusBar, QGroupBox,
    QCheckBox, QLineEdit, QFormLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QMimeData
from PySide6.QtGui import QKeyEvent

import backend
from gl_widget import FractalGLWidget

logger = logging.getLogger(__name__)

# Famous Mandelbrot locations
PRESETS = [
    ("Overview",            -0.5,               0.0,                2.5),
    ("Seahorse Valley",     -0.747,             0.1,                0.005),
    ("Elephant Valley",      0.282,             0.0078,             0.005),
    ("Double Spiral",       -0.743643887037151, 0.131825904205330,  0.002),
    ("Mini Mandelbrot",     -1.768778833,      -0.001738996,        0.0003),
    ("Spiral Arm",          -0.7623,            0.0838,             0.003),
    ("Star Pattern",        -0.1011,            0.9563,             0.02),
    ("Lightning",           -0.170337,         -1.06506,            0.05),
]

AUTO_ZOOM_TARGETS = [
    ("Seahorse Deep",  -0.7453,  0.1127),
    ("Double Spiral",  -0.743643887037151, 0.131825904205330),
    ("Mini-brot",      -1.768778833, -0.001738996),
    ("Elephant",        0.2821,   0.0100),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU Fractal Engine")
        self.resize(1280, 800)

        self.setStatusBar(QStatusBar(self))
        self.plugins = {cls.name: cls() for cls in backend.AVAILABLE_PLUGINS}

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.gl_widget = FractalGLWidget()
        self.gl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.gl_widget, stretch=4)

        # ---- Sidebar ----
        sidebar = QWidget()
        sidebar.setMaximumWidth(360)
        sidebar.setMinimumWidth(280)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(4, 4, 4, 4)

        # --- Presets ---
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)
        self.combo_preset = QComboBox()
        for name, *_ in PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.currentIndexChanged.connect(self._go_preset)
        preset_layout.addWidget(self.combo_preset)
        sl.addWidget(preset_group)

        # --- Appearance ---
        appear_group = QGroupBox("Appearance")
        al = QVBoxLayout(appear_group)

        pal_row = QHBoxLayout()
        pal_row.addWidget(QLabel("Palette:"))
        self.combo_palette = QComboBox()
        self._load_palette_files()
        self.combo_palette.currentTextChanged.connect(self._change_palette)
        pal_row.addWidget(self.combo_palette, stretch=1)
        al.addLayout(pal_row)

        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("Color Offset:"))
        self.color_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_slider.setRange(0, 500)
        self.color_slider.setValue(0)
        self.color_slider.valueChanged.connect(self._update_color_offset)
        c_row.addWidget(self.color_slider, stretch=1)
        al.addLayout(c_row)
        sl.addWidget(appear_group)

        # --- Animation ---
        anim_group = QGroupBox("Animation")
        aml = QVBoxLayout(anim_group)

        self.auto_zoom_btn = QPushButton("▶  Auto-Zoom  (Space)")
        self.auto_zoom_btn.setCheckable(True)
        self.auto_zoom_btn.toggled.connect(self._toggle_auto_zoom)
        aml.addWidget(self.auto_zoom_btn)

        az_row = QHBoxLayout()
        az_row.addWidget(QLabel("Target:"))
        self.combo_az_target = QComboBox()
        for name, *_ in AUTO_ZOOM_TARGETS:
            self.combo_az_target.addItem(name)
        az_row.addWidget(self.combo_az_target, stretch=1)
        aml.addLayout(az_row)

        spd_row = QHBoxLayout()
        spd_row.addWidget(QLabel("Speed:"))
        self.zoom_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_speed_slider.setRange(900, 995)
        self.zoom_speed_slider.setValue(970)
        spd_row.addWidget(self.zoom_speed_slider, stretch=1)
        aml.addLayout(spd_row)

        self.pause_time_cb = QCheckBox("Pause Color Animation")
        self.pause_time_cb.toggled.connect(lambda s: setattr(self.gl_widget, 'animation_paused', s))
        aml.addWidget(self.pause_time_cb)

        self.julia_anim_cb = QCheckBox("Animate Julia C")
        self.julia_anim_cb.toggled.connect(self._toggle_julia_anim)
        aml.addWidget(self.julia_anim_cb)

        sl.addWidget(anim_group)

        # --- Engine ---
        engine_group = QGroupBox("Engine")
        el = QVBoxLayout(engine_group)

        el.addWidget(QLabel("Active Fractal:"))
        self.combo = QComboBox()
        self.combo.addItems(self.plugins.keys())
        self.combo.currentTextChanged.connect(self._switch_plugin)
        el.addWidget(self.combo)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max Iter:"))
        self.iter_label = QLabel("200")
        self.iter_label.setMinimumWidth(40)
        iter_row.addWidget(self.iter_label)
        el.addLayout(iter_row)

        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(50, 10000)
        self.iter_slider.valueChanged.connect(self._update_iterations)
        el.addWidget(self.iter_slider)

        hint = QLabel("💡 Shift+Scroll to adjust iterations")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        el.addWidget(hint)

        sl.addWidget(engine_group)

        # --- Navigator ---
        nav_group = QGroupBox("Navigator")
        nl = QFormLayout(nav_group)
        self.input_x = QLineEdit("-0.5")
        self.input_y = QLineEdit("0.0")
        self.input_zoom = QLineEdit("2.5")
        self.input_x.setToolTip("Center X (supports scientific notation)")
        self.input_y.setToolTip("Center Y (supports scientific notation)")
        self.input_zoom.setToolTip("Zoom level (e.g. 1e-10)")

        for w in (self.input_x, self.input_y, self.input_zoom):
            w.returnPressed.connect(self._apply_manual_coords)

        btn_go = QPushButton("Go To Coordinates  (Enter)")
        btn_go.clicked.connect(self._apply_manual_coords)

        btn_copy = QPushButton("📋 Copy Coordinates")
        btn_copy.clicked.connect(self._copy_coords)

        nl.addRow("Center X:", self.input_x)
        nl.addRow("Center Y:", self.input_y)
        nl.addRow("Zoom:", self.input_zoom)
        nl.addRow(btn_go)
        nl.addRow(btn_copy)
        sl.addWidget(nav_group)

        # --- Actions ---
        act_row = QHBoxLayout()
        reset_btn = QPushButton("⟲ Reset  (R)")
        reset_btn.clicked.connect(self._reset_view)
        act_row.addWidget(reset_btn)

        export_btn = QPushButton("💾 Save Image")
        export_btn.clicked.connect(self._save_image)
        act_row.addWidget(export_btn)
        sl.addLayout(act_row)

        # --- Shader Parameters ---
        shader_group = QGroupBox("Shader Parameters")
        sgl = QVBoxLayout(shader_group)
        self.plugin_ui_container = QVBoxLayout()
        sgl.addLayout(self.plugin_ui_container)
        sl.addWidget(shader_group)

        sl.addStretch()
        main_layout.addWidget(sidebar, stretch=1)

        # --- Timers ---
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._process_auto_zoom)

        self.julia_anim_timer = QTimer(self)
        self.julia_anim_timer.timeout.connect(self._animate_julia_c)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(100)

        # Initial plugin load
        if self.plugins:
            first = list(self.plugins.keys())[0]
            self.combo.setCurrentText(first)
            QTimer.singleShot(100, lambda: self._switch_plugin(first))

    # ---------------------------------------------------------------- palettes
    def _load_palette_files(self):
        self.combo_palette.clear()
        import os
        if os.path.exists("palettes"):
            files = sorted(f for f in os.listdir("palettes") if f.endswith(".map"))
            self.combo_palette.addItems(files)

    def _change_palette(self, name):
        if name:
            self.gl_widget.load_palette_from_file(name)

    def _update_color_offset(self, val):
        self.gl_widget.color_offset = val / 50.0
        self.gl_widget.update()

    # ---------------------------------------------------------------- presets
    def _go_preset(self, index):
        if index < 0 or index >= len(PRESETS):
            return
        _, cx, cy, z = PRESETS[index]
        self.gl_widget.center_x = cx
        self.gl_widget.center_y = cy
        self.gl_widget.zoom = z
        # Reload plugin if precision needs switching
        if (z < 1e-6) != self.gl_widget.use_double_precision:
            if self.gl_widget.current_plugin:
                self.gl_widget.load_plugin(self.gl_widget.current_plugin)
        self.gl_widget.update()

    # -------------------------------------------------------------- auto-zoom
    def _toggle_auto_zoom(self, checked):
        if checked:
            self.auto_zoom_btn.setText("■  Stop Auto-Zoom  (Space)")
            idx = self.combo_az_target.currentIndex()
            _, tx, ty = AUTO_ZOOM_TARGETS[idx]
            self._az_target_x = tx
            self._az_target_y = ty
            self.anim_timer.start(16)
        else:
            self.auto_zoom_btn.setText("▶  Auto-Zoom  (Space)")
            self.anim_timer.stop()

    def _process_auto_zoom(self):
        gw = self.gl_widget
        smoothing = 0.02
        gw.center_x += (self._az_target_x - gw.center_x) * smoothing
        gw.center_y += (self._az_target_y - gw.center_y) * smoothing
        factor = self.zoom_speed_slider.value() / 1000.0
        gw.zoom *= factor

        # Auto-scale iterations with zoom depth
        if gw.zoom < 0.001:
            target_iter = max(200, min(5000, int(200 - 60 * math.log10(max(1e-14, gw.zoom)))))
            if target_iter > gw.max_iter:
                gw.max_iter = min(5000, gw.max_iter + 3)
                self.iter_slider.blockSignals(True)
                self.iter_slider.setValue(gw.max_iter)
                self.iter_slider.blockSignals(False)
                self.iter_label.setText(str(gw.max_iter))
        gw.update()

    # ----------------------------------------------------------- julia anim
    def _toggle_julia_anim(self, checked):
        if checked:
            self._julia_angle = 0.0
            self.julia_anim_timer.start(30)
        else:
            self.julia_anim_timer.stop()

    def _animate_julia_c(self):
        plugin = self.gl_widget.current_plugin
        if not plugin or not hasattr(plugin, 'params'):
            return
        if 'c_real' not in plugin.params:
            return
        self._julia_angle = getattr(self, '_julia_angle', 0) + 0.02
        radius = 0.7885
        plugin.params['c_real'] = radius * math.cos(self._julia_angle)
        plugin.params['c_imag'] = radius * math.sin(self._julia_angle)
        # Update the plugin UI if visible
        self._refresh_plugin_ui()
        self.gl_widget.update()

    def _refresh_plugin_ui(self):
        """Force-refresh the plugin parameter UI labels."""
        # Walk the plugin_ui_container to find value labels and update them
        plugin = self.gl_widget.current_plugin
        if not plugin:
            return
        schema = plugin.get_parameter_schema()
        layout = self.plugin_ui_container
        idx = 0
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                inner = item.widget().layout()
                if inner and idx < len(schema):
                    # Find the value label (last widget in the row)
                    # This is a best-effort refresh
                    idx += 1

    # --------------------------------------------------------------- plugin
    def _switch_plugin(self, name):
        plugin = self.plugins.get(name)
        if not plugin:
            return

        # Clear old UI
        for i in reversed(range(self.plugin_ui_container.count())):
            w = self.plugin_ui_container.itemAt(i).widget()
            if w:
                w.deleteLater()

        self.gl_widget.load_plugin(plugin)

        if self.gl_widget.program:
            defaults = plugin.get_default_settings()
            self.iter_slider.blockSignals(True)
            self.iter_slider.setValue(defaults["max_iter"])
            self.iter_slider.blockSignals(False)
            self.iter_label.setText(str(defaults["max_iter"]))
            self.gl_widget.max_iter = defaults["max_iter"]

            ui_widget = plugin.setup_ui(self)
            self.plugin_ui_container.addWidget(ui_widget)

        # Disable Julia anim if not Julia
        is_julia = 'c_real' in plugin.params
        self.julia_anim_cb.setEnabled(is_julia)
        if not is_julia and self.julia_anim_cb.isChecked():
            self.julia_anim_cb.setChecked(False)

    # ------------------------------------------------------------- iterations
    def _update_iterations(self, val):
        self.gl_widget.max_iter = val
        self.iter_label.setText(str(val))

    # --------------------------------------------------------------- nav
    def _apply_manual_coords(self):
        try:
            x = float(self.input_x.text())
            y = float(self.input_y.text())
            z = float(self.input_zoom.text())
            self.gl_widget.center_x = x
            self.gl_widget.center_y = y
            self.gl_widget.zoom = max(1e-14, z)
            if (z < 1e-6) != self.gl_widget.use_double_precision:
                if self.gl_widget.current_plugin:
                    self.gl_widget.load_plugin(self.gl_widget.current_plugin)
            self.gl_widget.update()
        except ValueError:
            self.statusBar().showMessage("⚠ Invalid coordinate format!", 3000)

    def _copy_coords(self):
        from PySide6.QtGui import QClipboard, QApplication
        text = (f"x={self.gl_widget.center_x:.15g}  "
                f"y={self.gl_widget.center_y:.15g}  "
                f"zoom={self.gl_widget.zoom:.15g}")
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("📋 Copied!", 1500)

    # ------------------------------------------------------------- actions
    def _reset_view(self):
        if self.gl_widget.current_plugin:
            s = self.gl_widget.current_plugin.get_default_settings()
            self.gl_widget.apply_settings(s)
            self.iter_slider.setValue(s["max_iter"])
            self.gl_widget.update()

    def _save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "fractal.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if file_path:
            self.gl_widget.grabFramebuffer().save(file_path)
            self.statusBar().showMessage(f"💾 Saved: {file_path}", 3000)

    # ------------------------------------------------------------- status
    def _update_status(self):
        gw = self.gl_widget
        self.input_x.blockSignals(True)
        self.input_y.blockSignals(True)
        self.input_zoom.blockSignals(True)
        self.input_x.setText(f"{gw.center_x:.15g}")
        self.input_y.setText(f"{gw.center_y:.15g}")
        self.input_zoom.setText(f"{gw.zoom:.15g}")
        self.input_x.blockSignals(False)
        self.input_y.blockSignals(False)
        self.input_zoom.blockSignals(False)

        prec = "Double" if gw.use_double_precision else "Float"
        self.statusBar().showMessage(
            f"FPS: {gw.fps:.0f}  |  Zoom: {gw.zoom:.6g}  |  Prec: {prec}  |  "
            f"Center: ({gw.center_x:.12g}, {gw.center_y:.12g})  |  Iter: {gw.max_iter}"
        )

    # --------------------------------------------------------- keyboard
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_F11:
            (self.showNormal if self.isFullScreen() else self.showFullScreen)()
        elif key == Qt.Key_R:
            self._reset_view()
        elif key == Qt.Key_Space:
            self.auto_zoom_btn.toggle()
        elif key == Qt.Key_C:
            self._copy_coords()
        elif key == Qt.Key_S and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._save_image()
        else:
            super().keyPressEvent(event)