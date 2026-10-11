"""
GPU Fractal Engine v2.4.0
=========================
Single-file OpenGL fractal explorer.

Requirements: PySide6, numpy, OpenGL 4.1+
"""

import sys, os, math, time, json, random, copy
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QComboBox, QFileDialog, QStatusBar,
    QGroupBox, QCheckBox, QLineEdit, QFormLayout, QSizePolicy,
    QMenu, QDialog, QDialogButtonBox, QSpinBox, QTextBrowser,
    QMessageBox, QScrollArea, QSplitter,
)
from PySide6.QtGui import (
    QSurfaceFormat, QPalette, QColor, QKeyEvent, QPainter, QPen,
    QFont, QVector2D, QImageReader,
)
from PySide6.QtCore import Qt, QTimer, QMutex, QMutexLocker, QPoint, QSize, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLShader,
    QOpenGLVertexArrayObject, QOpenGLFramebufferObject,
)

APP_VERSION = "2.4.0"
BOOKMARKS_FILE = "fractal_bookmarks.json"
SETTINGS_FILE = "fractal_settings.json"
HISTORY_MAX = 100

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - [%(levelname)s] - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# =========================================================== THEME
def apply_dark_theme(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(45, 45, 48))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,            QColor(30, 30, 30))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(45, 45, 48))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(25, 25, 25))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,          QColor(55, 55, 58))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 50, 50))
    p.setColor(QPalette.ColorRole.Link,            QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.Text,       QColor(127, 127, 127))
    p.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    app.setPalette(p)


# =========================================================== PALETTES
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _palette_from_stops(stops, size: int = 256):
    stops = sorted(stops, key=lambda s: s[0])
    positions = np.array([s[0] for s in stops], dtype=np.float32)
    colors    = np.array([s[1] for s in stops], dtype=np.float32)
    t  = np.linspace(0.0, 1.0, size, dtype=np.float32)
    idx = np.clip(np.searchsorted(positions, t, side='right') - 1,
                  0, len(stops) - 2)
    p0, p1 = positions[idx], positions[idx + 1]
    c0, c1 = colors[idx],   colors[idx + 1]
    denom  = np.where(p1 > p0, p1 - p0, 1.0)
    lt = np.clip((t - p0) / denom, 0.0, 1.0)
    lt = lt * lt * (3.0 - 2.0 * lt)            # smoothstep
    out = c0 + (c1 - c0) * lt[:, None]
    return [tuple(int(np.clip(v, 0, 255)) for v in row) for row in out]

def _resample_palette(colors, size: int = 256):
    if not colors:    return [(0, 0, 0)] * size
    if len(colors)==1:return [colors[0]] * size
    arr = np.asarray(colors, dtype=np.float32)
    n   = len(arr) - 1
    t   = np.arange(size, dtype=np.float32) * n / (size - 1)
    k   = np.clip(t.astype(np.int32), 0, n - 1)
    f   = t - k
    f   = f * f * (3.0 - 2.0 * f)              # smoothstep per segment
    out = arr[k] + (arr[k + 1] - arr[k]) * f[:, None]
    return [tuple(int(np.clip(v, 0, 255)) for v in row) for row in out]

def _write_palette(filename, colors):
    os.makedirs("palettes", exist_ok=True)
    with open(os.path.join("palettes", filename), "w") as f:
        for r, g, b in colors:
            f.write(f"{r} {g} {b}\n")

def create_palettes():
    _write_palette("default.map", _palette_from_stops([
        (0.00,(0,7,100)),(0.16,(32,107,203)),(0.33,(49,210,207)),
        (0.50,(30,225,60)),(0.66,(230,225,35)),(0.83,(230,80,20)),(1.00,(120,10,10))]))
    _write_palette("fire.map", _palette_from_stops([
        (0.0,(0,0,0)),(0.2,(80,0,0)),(0.4,(200,30,0)),
        (0.6,(255,128,0)),(0.8,(255,230,50)),(1.0,(255,255,220))]))
    _write_palette("ocean.map", _palette_from_stops([
        (0.0,(0,5,30)),(0.25,(0,30,100)),(0.50,(0,100,180)),
        (0.75,(50,200,230)),(1.0,(200,240,255))]))
    _write_palette("electric.map", _palette_from_stops([
        (0.0,(0,0,20)),(0.25,(20,0,120)),(0.50,(80,0,255)),
        (0.75,(0,180,255)),(1.0,(200,255,255))]))
    _write_palette("forest.map", _palette_from_stops([
        (0.0,(5,15,5)),(0.25,(20,70,20)),(0.50,(40,150,40)),
        (0.75,(140,200,60)),(1.0,(220,240,150))]))
    _write_palette("sunset.map", _palette_from_stops([
        (0.0,(20,0,50)),(0.25,(120,10,80)),(0.50,(220,60,30)),
        (0.75,(255,170,50)),(1.0,(255,240,150))]))
    _write_palette("grayscale.map", _palette_from_stops([(0.0,(0,0,0)),(1.0,(255,255,255))]))
    _write_palette("neon.map", _palette_from_stops([
        (0.0,(0,0,0)),(0.20,(255,0,100)),(0.40,(0,255,200)),
        (0.60,(0,100,255)),(0.80,(255,255,0)),(1.0,(255,0,255))]))
    _write_palette("ice.map", _palette_from_stops([
        (0.0,(10,10,40)),(0.30,(30,60,150)),(0.60,(100,180,240)),
        (0.85,(200,230,255)),(1.0,(255,255,255))]))
    _write_palette("magma.map", _palette_from_stops([
        (0.0,(0,0,4)),(0.25,(80,18,123)),(0.50,(183,55,121)),
        (0.75,(254,136,76)),(1.0,(252,253,191))]))
    _write_palette("viridis.map", _palette_from_stops([
        (0.0,(68,1,84)),(0.25,(59,82,139)),(0.50,(33,145,140)),
        (0.75,(94,201,98)),(1.0,(253,231,37))]))


# =========================================================== PRESETS / TARGETS
PRESETS = [
    ("Overview",            -0.5,                0.0,                2.5),
    ("Seahorse Valley",     -0.747,              0.1,                0.005),
    ("Elephant Valley",      0.282,              0.0078,             0.005),
    ("Double Spiral",       -0.743643887037151,  0.131825904205330,  0.002),
    ("Mini Mandelbrot",     -1.768778833,       -0.001738996,        0.0003),
    ("Spiral Arm",          -0.7623,             0.0838,             0.003),
    ("Star Pattern",        -0.1011,             0.9563,             0.02),
    ("Lightning",           -0.170337,          -1.06506,            0.05),
]
AUTO_ZOOM_TARGETS = [
    ("Seahorse Deep",  -0.7453,            0.1127),
    ("Double Spiral",  -0.743643887037151, 0.131825904205330),
    ("Mini-brot",      -1.768778833,      -0.001738996),
    ("Elephant",        0.2821,            0.0100),
]

SHORTCUTS_TEXT = """
<h2>GPU Fractal Engine — Keyboard Shortcuts</h2>
<style>
  table { border-collapse: collapse; }
  td { padding: 4px 10px; }
  td:first-child { color: #6cf; font-weight: bold; width: 38%; }
</style>
<table>
<tr><td>Mouse Drag</td><td>Pan the view</td></tr>
<tr><td>Scroll Wheel</td><td>Zoom in/out (towards cursor)</td></tr>
<tr><td>Shift + Scroll</td><td>Adjust max iterations</td></tr>
<tr><td>Ctrl + Scroll</td><td>Fine-tune color offset</td></tr>
<tr><td>Double-Click</td><td>Center and zoom in</td></tr>
<tr><td>Middle-Click</td><td>Center on cursor</td></tr>
<tr><td>Right-Click</td><td>Context menu</td></tr>
<tr><td>Arrow Keys</td><td>Pan (Shift = larger step)</td></tr>
<tr><td>+ / -</td><td>Zoom in / out</td></tr>
<tr><td>[ / ]</td><td>Previous / Next Palette</td></tr>
<tr><td>J</td><td>Toggle Interactive Julia Mode</td></tr>
<tr><td>Space</td><td>Toggle auto-zoom</td></tr>
<tr><td>R</td><td>Reset view</td></tr>
<tr><td>C</td><td>Copy coordinates</td></tr>
<tr><td>F / F11</td><td>Toggle fullscreen</td></tr>
<tr><td>H</td><td>Toggle on-screen HUD</td></tr>
<tr><td>F1</td><td>This help dialog</td></tr>
<tr><td>Ctrl+S</td><td>Save screenshot (with embedded metadata)</td></tr>
<tr><td>Ctrl+Z / Ctrl+Shift+Z</td><td>Undo / Redo zoom</td></tr>
<tr><td>Ctrl+B</td><td>Save bookmark</td></tr>
<tr><td>1-7</td><td>Switch fractal plugin</td></tr>
</table>
"""


# =========================================================== DIALOGS
class HelpDialog(QDialog):
    def __init__(self, parent=None, gl_info=""):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(500, 480)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(SHORTCUTS_TEXT + f"<hr><b>OpenGL:</b><br><pre style='color:#9c9'>{gl_info}</pre>")
        layout.addWidget(browser)
        btn = QPushButton("Close"); btn.clicked.connect(self.accept); layout.addWidget(btn)


class HighResExportDialog(QDialog):
    def __init__(self, current_w, current_h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("High-Resolution Export")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Export the current view at a custom resolution."))
        form = QFormLayout()
        self.spin_w = QSpinBox(); self.spin_w.setRange(256, 16384); self.spin_w.setValue(current_w); self.spin_w.setSingleStep(256)
        self.spin_h = QSpinBox(); self.spin_h.setRange(256, 16384); self.spin_h.setValue(current_h); self.spin_h.setSingleStep(256)
        form.addRow("Width (px):", self.spin_w); form.addRow("Height (px):", self.spin_h)
        
        self.aspect_lock = QCheckBox("Lock Aspect Ratio")
        self.aspect_lock.setChecked(True)
        form.addRow("Aspect:", self.aspect_lock)
        self._orig_aspect = current_w / current_h if current_h else 1.0
        self.spin_w.valueChanged.connect(self._on_w_changed)
        self.spin_h.valueChanged.connect(self._on_h_changed)

        preset_row = QHBoxLayout()
        for label, w, h in [("HD", 1920, 1080), ("4K", 3840, 2160), ("8K", 7680, 4320), ("Current", current_w, current_h)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, ww=w, hh=h: (self.spin_w.setValue(ww), self.spin_h.setValue(hh)))
            preset_row.addWidget(b)
        form.addRow("Presets:", preset_row)
        layout.addLayout(form)
        
        self.aa_cb = QCheckBox("2× Anti-Aliasing  (supersampled — slower, much cleaner)")
        self.aa_cb.setChecked(True)
        layout.addWidget(self.aa_cb)
        
        layout.addWidget(QLabel("Note: rendering uses an offscreen framebuffer; very large sizes may fail on some GPUs."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_w_changed(self, v):
        if self.aspect_lock.isChecked():
            self.spin_h.blockSignals(True)
            self.spin_h.setValue(int(v / self._orig_aspect))
            self.spin_h.blockSignals(False)
            
    def _on_h_changed(self, v):
        if self.aspect_lock.isChecked():
            self.spin_w.blockSignals(True)
            self.spin_w.setValue(int(v * self._orig_aspect))
            self.spin_w.blockSignals(False)

    def get_size(self): return self.spin_w.value(), self.spin_h.value(), self.aa_cb.isChecked()


# =========================================================== PLUGINS
class BaseFractalPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def formula_code(self) -> str: ...
    @property
    def init_code(self) -> str:
        return "VEC2 z = VEC2(0.0);\n            VEC2 c = c_pos;\n"
    @property
    def description(self) -> str: return ""

    vertex_shader = """
    #version 410 core
    layout (location = 0) in vec2 a_position;
    void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
    """

    _coloring_footer = """
        if (iter >= float(u_max_iter)) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            float log_zn = 0.5 * log(max(dot(z, z), 2.0));
            float nu = log(log_zn / log(2.0)) / log(2.0);
            iter = iter + 1.0 - nu;
            float cycle = max(1.0, float(u_max_iter) * 0.03);
            float t = fract(iter / cycle + u_color_offset);
            if (u_invert_palette > 0.5) t = 1.0 - t;
            vec3 col = texture(u_palette, vec2(t, 0.5)).rgb;
            
            // Brightness and Gamma correction
            col = col * u_brightness;
            col = pow(col, vec3(1.0 / u_gamma));
            
            fragColor = vec4(col, 1.0);
        }
    }
    """

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}
        for p in self.get_parameter_schema():
            self.params[p['name']] = p['default']

    def get_parameter_schema(self) -> List[Dict[str, Any]]: return []

    def get_fragment_shader(self, use_double: bool = False) -> str:
        if use_double:
            header = """#version 410 core
#extension GL_ARB_gpu_shader_fp64 : require
#define FLOAT double
#define VEC2 dvec2
uniform dvec2 u_center;
uniform double u_zoom;
"""
        else:
            header = """#version 410 core
#define FLOAT float
#define VEC2 vec2
uniform vec2 u_center;
uniform float u_zoom;
"""
        header += """
uniform vec2 u_resolution;
uniform int u_max_iter;
uniform float u_color_offset;
uniform sampler2D u_palette;
uniform float u_brightness;
uniform float u_gamma;
uniform float u_invert_palette;
out vec4 fragColor;
"""
        body = """
void main() {
    VEC2 uv = VEC2(gl_FragCoord.xy - 0.5 * u_resolution) / VEC2(u_resolution.y);
    VEC2 c_pos = u_center + uv * FLOAT(u_zoom);
"""
        body += self.init_code
        body += "    float iter = 0.0;\n"
        body += self.formula_code
        body += self._coloring_footer
        return header + body

    def setup_ui(self, parent: QWidget) -> QWidget:
        widget = QWidget(parent)
        layout = QVBoxLayout(widget); layout.setContentsMargins(0,0,0,0)
        self._ui_bindings = []
        schema = self.get_parameter_schema()
        if not schema:
            layout.addWidget(QLabel("No configurable parameters.")); return widget
        for p in schema:
            row = QHBoxLayout()
            name_lbl = QLabel(f"{p['label']}:"); name_lbl.setMinimumWidth(90); row.addWidget(name_lbl)
            fmt = p.get('fmt', '.2f'); scale = p.get('scale', 1000)
            val_lbl = QLabel(f"{self.params[p['name']]:{fmt}}")
            val_lbl.setMinimumWidth(55)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            slider = QSlider(Qt.Orientation.Horizontal)
            if p['type'] == 'float':
                slider.setRange(int(p['min']*scale), int(p['max']*scale))
                slider.setValue(int(self.params[p['name']]*scale))
                slider.valueChanged.connect(
                    lambda v, nm=p['name'], vl=val_lbl, f=fmt, s=scale:
                        self._update_param(nm, v/s, vl, f))
            else:
                slider.setRange(p['min'], p['max'])
                slider.setValue(self.params[p['name']])
                slider.valueChanged.connect(
                    lambda v, nm=p['name'], vl=val_lbl:
                        self._update_param(nm, v, vl, 'd'))
            row.addWidget(slider, stretch=1)
            row.addWidget(val_lbl)
            self._ui_bindings.append((slider, val_lbl, p['name'], p))
            layout.addLayout(row)
        return widget

    def _update_param(self, name: str, value: Any, value_label: QLabel, fmt: str) -> None:
        self.params[name] = value
        value_label.setText(f"{value:{fmt}}")

    def refresh_ui(self) -> None:
        if not hasattr(self, "_ui_bindings"): return
        for slider, label, name, schema in self._ui_bindings:
            fmt = schema.get('fmt', '.2f'); scale = schema.get('scale', 1000)
            label.setText(f"{self.params[name]:{fmt}}")
            slider.blockSignals(True)
            if schema['type'] == 'float':
                slider.setValue(int(self.params[name]*scale))
            else:
                slider.setValue(self.params[name])
            slider.blockSignals(False)

    def randomize_params(self) -> None:
        for p in self.get_parameter_schema():
            if p['type'] == 'float':
                self.params[p['name']] = random.uniform(p['min'], p['max'])
            else:
                self.params[p['name']] = random.randint(p['min'], p['max'])
        self.refresh_ui()

    def get_default_settings(self) -> Dict[str, Any]:
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 200}

    def update_uniforms(
        self, program: QOpenGLShaderProgram, gl, t: float, w: int, h: int,
        cx: float, cy: float, zoom: float, max_iter: int,
        color_offset: float, use_double: bool,
        brightness: float, gamma: float, invert: bool,
    ) -> None:
        program.setUniformValue("u_resolution", QVector2D(w, h))
        program.setUniformValue("u_max_iter", int(max_iter))
        program.setUniformValue("u_color_offset", float(color_offset))
        program.setUniformValue("u_brightness", float(brightness))
        program.setUniformValue("u_gamma", float(gamma))
        program.setUniformValue("u_invert_palette", 1.0 if invert else 0.0)
        
        if use_double:
            loc = program.uniformLocation("u_center")
            if loc != -1: gl.glUniform2d(loc, float(cx), float(cy))
            loc = program.uniformLocation("u_zoom")
            if loc != -1: gl.glUniform1d(loc, float(zoom))
        else:
            program.setUniformValue("u_center", QVector2D(cx, cy))
            program.setUniformValue("u_zoom", float(zoom))
            
        if not hasattr(self, '_uniform_locs') or self._uniform_prog_id != id(program):
            self._uniform_locs = {}
            self._uniform_prog_id = id(program)
        for name, value in self.params.items():
            if name not in self._uniform_locs:
                self._uniform_locs[name] = program.uniformLocation(f"u_{name}")
            loc = self._uniform_locs[name]
            if loc == -1: continue
            if isinstance(value, float): program.setUniformValue(loc, value)
            elif isinstance(value, int): program.setUniformValue(loc, value)
            elif isinstance(value, tuple): program.setUniformValue(loc, QVector2D(*value))


class MandelbrotPlugin(BaseFractalPlugin):
    name = "Mandelbrot Set"
    description = "The classic z = z^2 + c fractal."
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            z = VEC2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
            iter += 1.0;
        }
    """

class MandelbrotCubicPlugin(BaseFractalPlugin):
    name = "Cubic Mandelbrot"
    description = "z = z³ + c — three-fold symmetric Mandelbrot variant."
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            FLOAT zx2 = z.x * z.x;
            FLOAT zy2 = z.y * z.y;
            // (x + iy)^3 = x(x² - 3y²) + i·y(3x² - y²)
            VEC2 z3 = VEC2(z.x * (zx2 - 3.0 * zy2),
                           z.y * (3.0 * zx2 - zy2));
            z = z3 + c;
            iter += 1.0;
        }
    """
    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 2.5, "max_iter": 200}

class JuliaPlugin(BaseFractalPlugin):
    name = "Julia Set"
    description = "Julia set z = z^2 + c with fixed c."
    init_code = "VEC2 z = c_pos;\n            VEC2 c = VEC2(u_c_real, u_c_imag);\n"
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            z = VEC2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
            iter += 1.0;
        }
    """
    def get_parameter_schema(self):
        return [
            {'name':'c_real','type':'float','default':-0.7,'min':-2.0,'max':2.0,'label':'Real (c)','fmt':'.4f','scale':10000},
            {'name':'c_imag','type':'float','default':0.27015,'min':-2.0,'max':2.0,'label':'Imag (c)','fmt':'.4f','scale':10000},
        ]
    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 1.5, "max_iter": 200}

class BurningShipPlugin(BaseFractalPlugin):
    name = "Burning Ship"
    description = "z = (|Re(z)| + i|Im(z)|)^2 + c"
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            z = VEC2(z.x*z.x - z.y*z.y, 2.0 * abs(z.x*z.y)) + c;
            iter += 1.0;
        }
    """
    def get_default_settings(self):
        return {"center_x": -0.4, "center_y": -0.6, "zoom": 2.5, "max_iter": 150}

class MultibrotPlugin(BaseFractalPlugin):
    name = "Multibrot Set"
    description = "z = z^n + c with adjustable n. Integer n uses a fast path."
    formula_code = """
        if (abs(u_power - round(u_power)) < 0.001) {
            int n = int(round(u_power));
            for (int i = 0; i < 10000; i++) {
                if (i >= u_max_iter) break;
                if (dot(z, z) > 256.0) break;
                VEC2 zn = z;
                for (int k = 1; k < n; k++) zn = VEC2(zn.x*z.x - zn.y*z.y, zn.x*z.y + zn.y*z.x);
                z = zn + c;
                iter += 1.0;
            }
        } else {
            for (int i = 0; i < 10000; i++) {
                if (i >= u_max_iter) break;
                if (dot(z, z) > 256.0) break;
                FLOAT r = length(z);
                FLOAT theta = atan(z.y, z.x);
                if (r < 1e-10) { z = c; iter += 1.0; continue; }
                FLOAT new_r = pow(r, FLOAT(u_power));
                FLOAT new_theta = theta * FLOAT(u_power);
                z = new_r * VEC2(cos(new_theta), sin(new_theta)) + c;
                iter += 1.0;
            }
        }
    """
    def get_parameter_schema(self):
        return [{'name':'power','type':'float','default':3.0,'min':2.0,'max':10.0,'label':'Power','fmt':'.2f','scale':100}]
    def get_default_settings(self):
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 150}

class PhoenixPlugin(BaseFractalPlugin):
    name = "Phoenix"
    description = "z = z^2 + c + p * z_prev"
    init_code = "VEC2 z = c_pos;\n            VEC2 c = VEC2(u_c_real, u_c_imag);\n            VEC2 z_prev = VEC2(0.0);\n"
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            FLOAT p = FLOAT(u_phoenix_p);
            VEC2 z_new = VEC2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c + p * z_prev;
            z_prev = z; z = z_new;
            iter += 1.0;
        }
    """
    def get_parameter_schema(self):
        return [
            {'name':'c_real','type':'float','default':-0.7,'min':-2.0,'max':2.0,'label':'Real (c)','fmt':'.3f','scale':1000},
            {'name':'c_imag','type':'float','default':0.27015,'min':-2.0,'max':2.0,'label':'Imag (c)','fmt':'.3f','scale':1000},
            {'name':'phoenix_p','type':'float','default':0.5,'min':-1.5,'max':1.5,'label':'Phoenix P','fmt':'.3f','scale':1000},
        ]
    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 1.8, "max_iter": 150}

class TricornPlugin(BaseFractalPlugin):
    name = "Tricorn (Mandelbar)"
    description = "z = conj(z)^2 + c"
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            z = VEC2(z.x*z.x - z.y*z.y, -2.0*z.x*z.y) + c;
            iter += 1.0;
        }
    """
    def get_default_settings(self):
        return {"center_x": -0.3, "center_y": 0.0, "zoom": 2.5, "max_iter": 150}

AVAILABLE_PLUGINS = [MandelbrotPlugin, MandelbrotCubicPlugin, JuliaPlugin,
                     BurningShipPlugin, MultibrotPlugin, PhoenixPlugin, TricornPlugin]


# =========================================================== ZOOM HISTORY
class ZoomHistory:
    def __init__(self, max_size=HISTORY_MAX):
        self._back = deque(maxlen=max_size)
        self._fwd  = deque(maxlen=max_size)
    def push(self, state):
        self._back.append(state); self._fwd.clear()
    def undo(self, current):
        if not self._back: return None
        prev = self._back.pop()
        self._fwd.append(current)
        return prev
    def redo(self, current):
        if not self._fwd: return None
        nxt = self._fwd.pop()
        self._back.append(current)
        return nxt
    def can_undo(self): return bool(self._back)
    def can_redo(self): return bool(self._fwd)


# =========================================================== GL WIDGET
class FractalGLWidget(QOpenGLWidget):
    viewInteracted = Signal()
    maxIterChanged = Signal(int)

    GL = type("GL", (), {
        "COLOR_BUFFER_BIT": 0x00004000,
        "FRAMEBUFFER":      0x8D40,
        "TRIANGLE_STRIP":   0x0005,
        "FLOAT":            0x1406,
        "TEXTURE_2D":       0x0DE1,
        "TEXTURE_MIN_FILTER": 0x2801,
        "LINEAR":           0x2601,
        "TEXTURE_MAG_FILTER": 0x2800,
        "TEXTURE_WRAP_S":   0x2802,
        "REPEAT":           0x2901,
        "TEXTURE_WRAP_T":   0x2803,
        "RGBA":             0x1908,
        "UNSIGNED_BYTE":    0x1401,
        "TEXTURE0":         0x84C0,
    })()

    post_vertex_shader = """
    #version 410 core
    layout (location = 0) in vec2 a_position;
    out vec2 v_uv;
    void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
    """

    post_fragment_shader = """
    #version 410 core
    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_scene;
    uniform vec2 u_resolution;
    uniform int u_edge_mode; // 0: off, 1: neon_outline, 2: subtle_overlay

    void main() {
        vec3 c = texture(u_scene, v_uv).rgb;
        if (u_edge_mode == 0) {
            fragColor = vec4(c, 1.0);
            return;
        }
        
        vec2 px = 1.0 / u_resolution;
        vec3 tl = texture(u_scene, v_uv + vec2(-px.x,  px.y)).rgb;
        vec3 tm = texture(u_scene, v_uv + vec2( 0.0,    px.y)).rgb;
        vec3 tr = texture(u_scene, v_uv + vec2( px.x,  px.y)).rgb;
        
        vec3 ml = texture(u_scene, v_uv + vec2(-px.x,  0.0)).rgb;
        vec3 mr = texture(u_scene, v_uv + vec2( px.x,  0.0)).rgb;
        
        vec3 bl = texture(u_scene, v_uv + vec2(-px.x, -px.y)).rgb;
        vec3 bm = texture(u_scene, v_uv + vec2( 0.0,   -px.y)).rgb;
        vec3 br = texture(u_scene, v_uv + vec2( px.x, -px.y)).rgb;
        
        vec3 gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
        vec3 gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
        
        float edge = clamp((length(gx) + length(gy)) * 1.5, 0.0, 1.0);
        
        if (u_edge_mode == 1) {
            // Pure Neon Outline
            fragColor = vec4(vec3(edge * 0.2, edge, edge * 0.8), 1.0);
        } else {
            // Subtle Overlay (adds glow to original image)
            vec3 edge_col = vec3(0.0, edge, edge * 0.5);
            fragColor = vec4(c + edge_col, 1.0);
        }
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Render state
        self.time = 0.0; self.zoom = 2.5
        self.center_x = -0.5; self.center_y = 0.0
        self.max_iter = 200
        self.animation_paused = False
        self.color_offset = 0.0
        self.anim_speed = 1.2
        self.color_cycle = False
        self.color_cycle_speed = 0.3
        
        # Color correction
        self.brightness = 1.0
        self.gamma = 1.0
        self.invert_palette = False

        # Plugin system
        self.current_plugin = None
        self.program = None; self.vbo = None; self.vao = None; self.gl = None
        self._program_cache = {}

        # Interaction
        self.last_mouse_pos = None
        self.show_crosshair = True
        self.show_hud = True
        self._mouse_x = -1; self._mouse_y = -1
        self._mouse_complex = (0.0, 0.0)
        self.julia_interactive = False

        # Textures
        self.palette_texture = None
        self.current_palette_name = ""
        self._desired_palette = "default.map"

        # Post-processing
        self.edge_detection = False
        self.post_program = None
        self.post_fbo = None

        # Demand-driven rendering
        self._dirty = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        self._last_tick = time.perf_counter()

        # FPS
        self.frame_count = 0; self.fps_acc = 0.0; self.fps = 0.0
        self._fc = 0

        # Thread safety
        self.resource_mutex = QMutex()
        self.use_double_precision = False
        self.double_supported = True

        self.gl_info = ""

        # HUD Label overlay
        self.hud_label = QLabel(self)
        self.hud_label.setStyleSheet(
            "QLabel { background-color: rgba(0, 0, 0, 140); color: #E6E6E6; "
            "padding: 5px; font-family: 'DejaVu Sans Mono', monospace; }"
        )
        self.hud_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud_label.move(8, 8)

    def cleanup(self):
        if not self.context(): return
        logger.info("Cleaning up OpenGL resources...")
        self.makeCurrent()
        try:
            if self.gl: self.gl.glUseProgram(0)
            for res in (self.vao, self.vbo):
                if res: res.destroy()
            self.vao = self.vbo = None
            if self.palette_texture and self.gl:
                self.gl.glDeleteTextures(1, [self.palette_texture])
                self.palette_texture = None
            if self.post_fbo:
                self.post_fbo.release()
                self.post_fbo = None
            for prog in self._program_cache.values():
                prog.removeAllShaders()
                prog.deleteLater()
            self._program_cache.clear()
            if self.post_program:
                self.post_program.removeAllShaders()
                self.post_program.deleteLater()
                self.post_program = None
        finally:
            self.doneCurrent()

    def _setup_geometry(self):
        if self.vbo: return
        verts = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
        self.vbo = QOpenGLBuffer(); self.vbo.create(); self.vbo.bind()
        self.vbo.allocate(verts.tobytes(), verts.nbytes); self.vbo.release()

    def load_palette_from_file(self, filename):
        if not self.gl or not self.context():
            self._desired_palette = filename
            return
        filepath = os.path.join("palettes", filename)
        if not os.path.exists(filepath):
            logger.warning(f"Palette not found: {filepath}"); return
        if filename == self.current_palette_name and self.palette_texture: return
        colors = []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 3:
                        try: colors.append((int(parts[0]), int(parts[1]), int(parts[2])))
                        except ValueError: continue
        except Exception as e:
            logger.error(f"Error reading palette: {e}"); return
        if not colors: colors = [(0,0,0), (255,255,255)]

        colors = _resample_palette(colors, 256)

        palette_data = bytearray(256 * 4)
        for i, (r, g, b) in enumerate(colors):
            o = i * 4
            palette_data[o:o+4] = bytes((r, g, b, 255))

        if self.palette_texture:
            self.gl.glDeleteTextures(1, [self.palette_texture])
        tex_ids = np.array([0], dtype=np.uint32)
        self.gl.glGenTextures(1, tex_ids)
        self.palette_texture = int(tex_ids[0])
        gl = self.gl
        gl.glBindTexture(self.GL.TEXTURE_2D, self.palette_texture)
        gl.glTexImage2D(self.GL.TEXTURE_2D, 0, self.GL.RGBA, 256, 1, 0,
                        self.GL.RGBA, self.GL.UNSIGNED_BYTE, bytes(palette_data))
        gl.glTexParameteri(self.GL.TEXTURE_2D, self.GL.TEXTURE_MIN_FILTER, self.GL.LINEAR)
        gl.glTexParameteri(self.GL.TEXTURE_2D, self.GL.TEXTURE_MAG_FILTER, self.GL.LINEAR)
        gl.glTexParameteri(self.GL.TEXTURE_2D, self.GL.TEXTURE_WRAP_S, self.GL.REPEAT)
        gl.glTexParameteri(self.GL.TEXTURE_2D, self.GL.TEXTURE_WRAP_T, self.GL.REPEAT)
        self.current_palette_name = filename
        self._mark_dirty()
        logger.info(f"Loaded palette: {filename}")

    def load_plugin(self, plugin):
        if not self.isValid() or not self.context() or not self.gl:
            logger.warning("Plugin load attempted without valid GL context"); return
        logger.info(f"Loading plugin: {plugin.name}...")
        self._ensure_precision()

        with QMutexLocker(self.resource_mutex):
            self.makeCurrent()
            
            cache_key = (plugin.name, self.use_double_precision)
            if cache_key in self._program_cache:
                self.program = self._program_cache[cache_key]
                is_new_plugin = self.current_plugin is None or self.current_plugin.name != plugin.name
                self.current_plugin = plugin
                if is_new_plugin:
                    self.apply_settings(plugin.get_default_settings())
                self.load_palette_from_file(self._desired_palette)
                
                if not self.vao:
                    self.vao = QOpenGLVertexArrayObject(); self.vao.create()
                self.vao.bind()
                if self.vbo: self.vbo.bind()
                self.program.enableAttributeArray(0)
                self.program.setAttributeBuffer(0, self.GL.FLOAT, 0, 2)
                self.vao.release()
                self._mark_dirty()
                self.doneCurrent()
                return
            
            new_program = QOpenGLShaderProgram(self)
            if not new_program.addShaderFromSourceCode(QOpenGLShader.Vertex, plugin.vertex_shader):
                logger.error(f"VERTEX error: {new_program.log()}")
                new_program.deleteLater(); self.doneCurrent(); return
            try:
                frag_src = plugin.get_fragment_shader(use_double=self.use_double_precision)
            except Exception as e:
                logger.error(f"Shader generation error: {e}")
                self.doneCurrent(); return
            if not new_program.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_src):
                logger.error(f"FRAGMENT error: {new_program.log()}")
                new_program.deleteLater(); self.doneCurrent(); return
            new_program.bindAttributeLocation("a_position", 0)
            if not new_program.link():
                logger.error(f"Link failed: {new_program.log()}")
                new_program.deleteLater(); self.doneCurrent(); return
                
            self._program_cache[cache_key] = new_program
            self.program = new_program
            
            self._setup_geometry()
            if not self.vao:
                self.vao = QOpenGLVertexArrayObject(); self.vao.create()
            self.vao.bind()
            if self.vbo: self.vbo.bind()
            self.program.enableAttributeArray(0)
            self.program.setAttributeBuffer(0, self.GL.FLOAT, 0, 2)
            self.vao.release()
            
            is_new_plugin = self.current_plugin is None or self.current_plugin.name != plugin.name
            self.current_plugin = plugin
            if is_new_plugin:
                self.apply_settings(plugin.get_default_settings())
            self.load_palette_from_file(self._desired_palette)
            self._mark_dirty()
            self.doneCurrent()

    def _ensure_precision(self) -> None:
        needed = self.zoom < 1e-6 and self.double_supported
        if needed != self.use_double_precision:
            logger.info(f"Precision -> {'Double' if needed else 'Single'}")
            self.use_double_precision = needed
            if self.current_plugin:
                self.load_plugin(self.current_plugin)

    def apply_settings(self, settings):
        self.center_x = float(settings.get("center_x", 0.0))
        self.center_y = float(settings.get("center_y", 0.0))
        self.zoom = float(settings.get("zoom", 1.0))
        self.max_iter = int(settings.get("max_iter", 100))
        self._mark_dirty()

    def initializeGL(self):
        ctx = self.context()
        if not ctx: logger.critical("QOpenGLContext is null"); return
        self.gl = ctx.extraFunctions()
        if not self.gl: logger.critical("Failed to resolve GL functions"); return
        ctx.aboutToBeDestroyed.connect(self.cleanup)
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self._setup_geometry()
        self.load_palette_from_file(self._desired_palette)
        
        self.double_supported = ctx.hasExtension(b"GL_ARB_gpu_shader_fp64")
        if not self.double_supported:
            logger.warning("Double precision (fp64) not supported on this GPU. Deep zoom will be clamped.")

        try:
            from PySide6.QtGui import QOpenGLFunctions
            fns = QOpenGLFunctions(ctx); fns.initializeOpenGLFunctions()
            vendor = self.gl.glGetString(0x1F00)
            renderer = self.gl.glGetString(0x1F01)
            version = self.gl.glGetString(0x1F02)
            glsl = self.gl.glGetString(0x8B8C)
            def to_str(s): return s.decode() if isinstance(s, bytes) else str(s)
            self.gl_info = (f"Vendor:   {to_str(vendor)}\n"
                            f"Renderer: {to_str(renderer)}\n"
                            f"Version:  {to_str(version)}\n"
                            f"GLSL:     {to_str(glsl)}\n"
                            f"FP64:     {'Supported' if self.double_supported else 'Not Supported'}")
            logger.info("OpenGL info:\n" + self.gl_info)
        except Exception as e:
            self.gl_info = f"(error retrieving GL info: {e})"

        # Setup post-processing shader
        self.post_program = QOpenGLShaderProgram(self)
        self.post_program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.post_vertex_shader)
        self.post_program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.post_fragment_shader)
        self.post_program.bindAttributeLocation("a_position", 0)
        self.post_program.link()

    def paintGL(self):
        if not self.current_plugin or not self.program or not self.gl: return
        with QMutexLocker(self.resource_mutex):
            gl = self.gl
            
            if self.edge_detection:
                if not self.post_fbo or self.post_fbo.size() != QSize(self.width(), self.height()):
                    if self.post_fbo: 
                        self.post_fbo.release()
                        import contextlib
                        with contextlib.suppress(Exception):
                            self.post_fbo = None 
                    self.post_fbo = QOpenGLFramebufferObject(QSize(self.width(), self.height()))
                self.post_fbo.bind()
            
            gl.glViewport(0, 0, self.width(), self.height())
            gl.glClear(self.GL.COLOR_BUFFER_BIT)
            self.program.bind()
            if self.vao: self.vao.bind()
            if self.palette_texture:
                gl.glActiveTexture(self.GL.TEXTURE0)
                gl.glBindTexture(self.GL.TEXTURE_2D, self.palette_texture)
            try:
                self.current_plugin.update_uniforms(
                    self.program, gl, self.time, self.width(), self.height(),
                    self.center_x, self.center_y, self.zoom, self.max_iter,
                    self.color_offset, self.use_double_precision, self.brightness, self.gamma, self.invert_palette)
                gl.glDrawArrays(self.GL.TRIANGLE_STRIP, 0, 4)
            except Exception as e:
                logger.error(f"Paint error: {e}")
            if self.vao: self.vao.release()
            self.program.release()
            
            if self.edge_detection and self.post_program and self.post_fbo:
                self.post_fbo.release()
                gl.glBindFramebuffer(self.GL.FRAMEBUFFER, 0)
                gl.glClear(self.GL.COLOR_BUFFER_BIT)
                
                self.post_program.bind()
                self.post_program.setUniformValue("u_scene", 0)
                self.post_program.setUniformValue("u_resolution", QVector2D(self.width(), self.height()))
                self.post_program.setUniformValue("u_edge_mode", 2)
                
                gl.glActiveTexture(self.GL.TEXTURE0)
                gl.glBindTexture(self.GL.TEXTURE_2D, self.post_fbo.texture())
                
                if self.vao: self.vao.bind()
                gl.glDrawArrays(self.GL.TRIANGLE_STRIP, 0, 4)
                if self.vao: self.vao.release()
                self.post_program.release()
                
        if self.show_crosshair: self._paint_crosshair()
        self._dirty = False

    def _paint_crosshair(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255, 70)); pen.setWidth(1); painter.setPen(pen)
        cx, cy = self.width() // 2, self.height() // 2; s = 14
        painter.drawLine(cx - s, cy, cx + s, cy)
        painter.drawLine(cx, cy - s, cx, cy + s)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.drawEllipse(QPoint(cx, cy), 4, 4)
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._mark_dirty()

    def update_hud(self):
        if not self.show_hud:
            if self.hud_label.isVisible(): self.hud_label.hide()
            return
        if not self.hud_label.isVisible(): self.hud_label.show()
        
        plugin_name = self.current_plugin.name if self.current_plugin else "?"
        prec = "Double" if self.use_double_precision else "Float"
        lines = [
            f"{plugin_name}",
            f"Zoom: {self.zoom:.4e}",
            f"Prec: {prec}",
            f"Iter: {self.max_iter}",
            f"FPS:  {self.fps:.0f}",
        ]
        if self._mouse_complex is not None:
            mx, my = self._mouse_complex
            lines.append(f"Cursor: {mx:.6e} + {my:.6e}i")
            
        text = "<br>".join(lines)
        if self.hud_label.text() != text:
            self.hud_label.setText(text)
            self.hud_label.adjustSize()

    def _tick(self):
        now = time.perf_counter()
        dt = now - self._last_tick; self._last_tick = now
        self.fps_acc += dt; self._fc += 1
        if self.fps_acc >= 0.5:
            self.fps = self._fc / self.fps_acc
            self._fc = 0; self.fps_acc = 0.0
            self.update_hud()
        if not self.animation_paused:
            self.time += dt * self.anim_speed
            self._mark_dirty()
        if self.color_cycle:
            self.color_offset = (self.color_offset + dt * self.color_cycle_speed) % 1.0
            self._mark_dirty()
        if self._dirty:
            self.update()

    def _mark_dirty(self):
        self._dirty = True

    def _screen_to_fractal(self, sx, sy):
        fx = self.center_x + (sx - self.width() / 2.0) / self.height() * self.zoom
        fy = self.center_y - (sy - self.height() / 2.0) / self.height() * self.zoom
        return fx, fy

    def update_auto_iter(self):
        if not self.auto_iter_enabled: return
        target_iter = max(100, min(5000, int(150 + 80 * math.log10(1.0 / max(1e-14, self.zoom)))))
        if target_iter > self.max_iter:
            self.max_iter = min(5000, self.max_iter + max(1, (target_iter - self.max_iter) // 5))
            self.maxIterChanged.emit(self.max_iter)
        elif target_iter < self.max_iter * 0.8:
            self.max_iter = max(100, self.max_iter - 5)
            self.maxIterChanged.emit(self.max_iter)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            step = max(1, self.max_iter // 20)
            new_iter = max(10, min(10000, self.max_iter + step * (1 if delta > 0 else -1)))
            self.max_iter = new_iter
            self.maxIterChanged.emit(new_iter)
            self._mark_dirty(); self.update(); self.viewInteracted.emit(); return
        if mods & Qt.KeyboardModifier.ControlModifier:
            self.color_offset = (self.color_offset + (1 if delta > 0 else -1) * 0.005) % 1.0
            self._mark_dirty(); self.update(); return
        zoom_factor = math.exp(-delta / 600.0)
        mx, my = event.position().x(), event.position().y()
        fx, fy = self._screen_to_fractal(mx, my)
        new_zoom = max(1e-14, self.zoom * zoom_factor)
        
        if new_zoom < 1e-6 and not self.double_supported:
            new_zoom = 1e-6 
            
        self.center_x = fx - (mx - self.width()/2.0) / self.height() * new_zoom
        self.center_y = fy + (my - self.height()/2.0) / self.height() * new_zoom
        self.zoom = new_zoom
        self._ensure_precision()
        
        if not (mods & Qt.KeyboardModifier.ShiftModifier):
            self.update_auto_iter()
            
        self._mark_dirty(); self.update()
        self.viewInteracted.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.MouseButton.MiddleButton:
            fx, fy = self._screen_to_fractal(event.position().x(), event.position().y())
            self.center_x = fx; self.center_y = fy
            self._mark_dirty(); self.update()
            self.viewInteracted.emit()

    def mouseMoveEvent(self, event):
        self._mouse_x = int(event.position().x()); self._mouse_y = int(event.position().y())
        self._mouse_complex = self._screen_to_fractal(self._mouse_x, self._mouse_y)
        if self.last_mouse_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()
            scale = self.zoom / max(1, self.height())
            self.center_x -= dx * scale
            self.center_y += dy * scale
            self.last_mouse_pos = event.position()
            self._mark_dirty()
            self.viewInteracted.emit()
        
        if self.julia_interactive and self.current_plugin and "c_real" in self.current_plugin.params:
            mx, my = self._mouse_complex
            self.current_plugin.params["c_real"] = mx
            self.current_plugin.params["c_imag"] = my
            if hasattr(self.current_plugin, "refresh_ui"):
                self.current_plugin.refresh_ui()
            self._mark_dirty()
            
        self.update_hud()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event):
        fx, fy = self._screen_to_fractal(event.position().x(), event.position().y())
        self.center_x = fx; self.center_y = fy; self.zoom *= 0.5
        self._ensure_precision()
        self._mark_dirty(); self.update()
        self.viewInteracted.emit()

    def pan_by_fraction(self, fx, fy):
        self.center_x += fx * self.zoom * (self.width() / self.height())
        self.center_y += fy * self.zoom
        self._mark_dirty(); self.update()
        self.viewInteracted.emit()

    def zoom_by(self, factor):
        new_zoom = max(1e-14, self.zoom * factor)
        if new_zoom < 1e-6 and not self.double_supported:
            new_zoom = 1e-6
        self.zoom = new_zoom
        self._ensure_precision()
        self._mark_dirty(); self.update()
        self.viewInteracted.emit()

    def render_to_image(self, w, h, aa=False):
        if not self.program or not self.current_plugin or not self.gl:
            return None
        self.makeCurrent()
        try:
            scale = 2 if aa else 1
            rw = min(w * scale, 16384)
            rh = min(h * scale, 16384)
            fbo = QOpenGLFramebufferObject(QSize(rw, rh))
            if not fbo.bind():
                logger.error("Failed to bind FBO for high-res render.")
                return None
                
            self.gl.glViewport(0, 0, rw, rh)
            self.gl.glClear(self.GL.COLOR_BUFFER_BIT)
            self.program.bind()
            if self.vao: self.vao.bind()
            if self.palette_texture:
                self.gl.glActiveTexture(self.GL.TEXTURE0)
                self.gl.glBindTexture(self.GL.TEXTURE_2D, self.palette_texture)
            self.current_plugin.update_uniforms(
                self.program, self.gl, self.time, rw, rh,
                self.center_x, self.center_y, self.zoom, self.max_iter,
                self.color_offset, self.use_double_precision, self.brightness, self.gamma, self.invert_palette)
            self.gl.glDrawArrays(self.GL.TRIANGLE_STRIP, 0, 4)
            self.gl.glFinish()
            img = fbo.toImage()
            if self.vao: self.vao.release()
            self.program.release()
            fbo.release()
            self.gl.glViewport(0, 0, self.width(), self.height())
            self._mark_dirty(); self.update()
            if aa and (rw != w or rh != h):
                img = img.scaled(w, h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
            return img
        except Exception as e:
            logger.error(f"High-res render failed: {e}")
            return None
        finally:
            self.doneCurrent()


# =========================================================== MAIN WINDOW
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"GPU Fractal Engine v{APP_VERSION}")
        self.resize(1280, 800)
        self.setStatusBar(QStatusBar(self))
        self.plugins = {cls.name: cls() for cls in AVAILABLE_PLUGINS}
        self.history = ZoomHistory()
        self._last_history_state = None
        self.bookmarks = self._load_bookmarks()

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); main_layout.setContentsMargins(0,0,0,0)

        self.gl_widget = FractalGLWidget()
        self.gl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gl_widget.maxIterChanged.connect(self._on_max_iter_changed)
        
        sidebar = QWidget()
        sidebar.setMinimumWidth(280)
        sl = QVBoxLayout(sidebar); sl.setContentsMargins(4,4,4,4)
        sl.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)

        # Presets
        preset_group = QGroupBox("Presets"); pl = QVBoxLayout(preset_group)
        self.combo_preset = QComboBox()
        for name, *_ in PRESETS: self.combo_preset.addItem(name)
        self.combo_preset.currentIndexChanged.connect(self._go_preset)
        pl.addWidget(self.combo_preset); sl.addWidget(preset_group)

        # Appearance
        appear_group = QGroupBox("Appearance"); al = QVBoxLayout(appear_group)
        pal_row = QHBoxLayout(); pal_row.addWidget(QLabel("Palette:"))
        self.combo_palette = QComboBox()
        self._load_palette_files()
        self.combo_palette.currentTextChanged.connect(self._change_palette)
        pal_row.addWidget(self.combo_palette, stretch=1); al.addLayout(pal_row)

        c_row = QHBoxLayout(); c_row.addWidget(QLabel("Color Offset:"))
        self.color_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_slider.setRange(0, 500); self.color_slider.setValue(0)
        self.color_slider.valueChanged.connect(self._update_color_offset)
        c_row.addWidget(self.color_slider, stretch=1); al.addLayout(c_row)

        spd_row = QHBoxLayout(); spd_row.addWidget(QLabel("Anim Speed:"))
        self.anim_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.anim_speed_slider.setRange(0, 300); self.anim_speed_slider.setValue(120)
        self.anim_speed_slider.valueChanged.connect(self._update_anim_speed)
        spd_row.addWidget(self.anim_speed_slider, stretch=1); al.addLayout(spd_row)
        
        bright_row = QHBoxLayout(); bright_row.addWidget(QLabel("Brightness:"))
        self.bright_slider = QSlider(Qt.Orientation.Horizontal)
        self.bright_slider.setRange(20, 300); self.bright_slider.setValue(100)
        self.bright_slider.valueChanged.connect(self._update_brightness)
        bright_row.addWidget(self.bright_slider, stretch=1); al.addLayout(bright_row)
        
        gamma_row = QHBoxLayout(); gamma_row.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(20, 300); self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self._update_gamma)
        gamma_row.addWidget(self.gamma_slider, stretch=1); al.addLayout(gamma_row)
        
        self.invert_cb = QCheckBox("Invert Palette")
        self.invert_cb.toggled.connect(self._toggle_invert)
        al.addWidget(self.invert_cb)

        self.cycle_cb = QCheckBox("Color Cycle")
        self.cycle_cb.toggled.connect(self._toggle_color_cycle)
        al.addWidget(self.cycle_cb)
        
        cyc_spd_row = QHBoxLayout(); cyc_spd_row.addWidget(QLabel("Cycle Speed:"))
        self.cyc_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.cyc_speed_slider.setRange(1, 100); self.cyc_speed_slider.setValue(30)
        self.cyc_speed_slider.valueChanged.connect(self._update_cycle_speed)
        cyc_spd_row.addWidget(self.cyc_speed_slider, stretch=1); al.addLayout(cyc_spd_row)

        self.edge_cb = QCheckBox("Edge Detection (Neon Outline)")
        self.edge_cb.toggled.connect(self._toggle_edge_detection)
        al.addWidget(self.edge_cb)

        self.crosshair_cb = QCheckBox("Show Crosshair"); self.crosshair_cb.setChecked(True)
        self.crosshair_cb.toggled.connect(lambda s: (setattr(self.gl_widget, "show_crosshair", s), self.gl_widget.update()))
        al.addWidget(self.crosshair_cb)

        self.hud_cb = QCheckBox("Show HUD"); self.hud_cb.setChecked(True)
        self.hud_cb.toggled.connect(self._toggle_hud)
        al.addWidget(self.hud_cb)

        sl.addWidget(appear_group)

        # Animation
        anim_group = QGroupBox("Animation"); aml = QVBoxLayout(anim_group)
        self.auto_zoom_btn = QPushButton("Auto-Zoom  (Space)"); self.auto_zoom_btn.setCheckable(True)
        self.auto_zoom_btn.toggled.connect(self._toggle_auto_zoom); aml.addWidget(self.auto_zoom_btn)
        az_row = QHBoxLayout(); az_row.addWidget(QLabel("Target:"))
        self.combo_az_target = QComboBox()
        self.combo_az_target.addItem("Cursor Position")
        for name, *_ in AUTO_ZOOM_TARGETS: self.combo_az_target.addItem(name)
        az_row.addWidget(self.combo_az_target, stretch=1); aml.addLayout(az_row)
        zspd_row = QHBoxLayout(); zspd_row.addWidget(QLabel("Speed:"))
        self.zoom_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_speed_slider.setRange(900, 995); self.zoom_speed_slider.setValue(970)
        zspd_row.addWidget(self.zoom_speed_slider, stretch=1); aml.addLayout(zspd_row)
        self.pause_time_cb = QCheckBox("Pause Color Animation")
        self.pause_time_cb.toggled.connect(lambda s: setattr(self.gl_widget, "animation_paused", s))
        aml.addWidget(self.pause_time_cb)
        self.julia_anim_cb = QCheckBox("Animate Julia C")
        self.julia_anim_cb.toggled.connect(self._toggle_julia_anim)
        aml.addWidget(self.julia_anim_cb)
        sl.addWidget(anim_group)

        # Engine
        engine_group = QGroupBox("Engine"); el = QVBoxLayout(engine_group)
        el.addWidget(QLabel("Active Fractal:"))
        self.combo = QComboBox(); self.combo.addItems(self.plugins.keys())
        self.combo.currentTextChanged.connect(self._switch_plugin); el.addWidget(self.combo)
        iter_row = QHBoxLayout(); iter_row.addWidget(QLabel("Max Iter:"))
        self.iter_label = QLabel("200"); self.iter_label.setMinimumWidth(40); iter_row.addWidget(self.iter_label)
        el.addLayout(iter_row)
        self.iter_slider = QSlider(Qt.Orientation.Horizontal); self.iter_slider.setRange(50, 10000)
        self.iter_slider.valueChanged.connect(self._update_iterations); el.addWidget(self.iter_slider)
        
        self.auto_iter_cb = QCheckBox("Auto-Iterate (scale with zoom)")
        self.auto_iter_cb.toggled.connect(self._toggle_auto_iter)
        el.addWidget(self.auto_iter_cb)
        
        hint = QLabel("Shift+Scroll: iters | Ctrl+Scroll: color offset | J: Julia Mode")
        hint.setStyleSheet("color: gray; font-size: 10px;"); el.addWidget(hint)
        sl.addWidget(engine_group)

        # Navigator
        nav_group = QGroupBox("Navigator"); nl = QFormLayout(nav_group)
        self.input_x = QLineEdit("-0.5"); self.input_y = QLineEdit("0.0"); self.input_zoom = QLineEdit("2.5")
        self.input_x.setToolTip("Center X (supports scientific notation)")
        self.input_y.setToolTip("Center Y (supports scientific notation)")
        self.input_zoom.setToolTip("Zoom level (e.g. 1e-10)")
        for w in (self.input_x, self.input_y, self.input_zoom):
            w.returnPressed.connect(self._apply_manual_coords)
        btn_go = QPushButton("Go To Coordinates  (Enter)"); btn_go.clicked.connect(self._apply_manual_coords)
        btn_copy = QPushButton("Copy Coordinates  (C)"); btn_copy.clicked.connect(self._copy_coords)
        nl.addRow("Center X:", self.input_x); nl.addRow("Center Y:", self.input_y)
        nl.addRow("Zoom:", self.input_zoom); nl.addRow(btn_go); nl.addRow(btn_copy)
        sl.addWidget(nav_group)

        # Actions row 1
        act1 = QHBoxLayout()
        reset_btn = QPushButton("Reset  (R)"); reset_btn.clicked.connect(self._reset_view); act1.addWidget(reset_btn)
        export_btn = QPushButton("Save Image  (Ctrl+S)"); export_btn.clicked.connect(self._save_image); act1.addWidget(export_btn)
        hires_btn = QPushButton("High-Res Export"); hires_btn.clicked.connect(self._save_high_res); act1.addWidget(hires_btn)
        sl.addLayout(act1)
        # Actions row 2
        act2 = QHBoxLayout()
        back_btn = QPushButton("Undo  (Ctrl+Z)"); back_btn.clicked.connect(self._zoom_undo); act2.addWidget(back_btn)
        fwd_btn = QPushButton("Redo  (Ctrl+Shift+Z)"); fwd_btn.clicked.connect(self._zoom_redo); act2.addWidget(fwd_btn)
        help_btn = QPushButton("Help  (F1)"); help_btn.clicked.connect(self._show_help); act2.addWidget(help_btn)
        sl.addLayout(act2)

        # Bookmarks
        bm_group = QGroupBox("Bookmarks"); bml = QVBoxLayout(bm_group)
        self.combo_bookmarks = QComboBox()
        self._refresh_bookmark_combo()
        self.combo_bookmarks.currentIndexChanged.connect(self._go_bookmark)
        bml.addWidget(self.combo_bookmarks)
        bm_btn_row = QHBoxLayout()
        save_bm_btn = QPushButton("Save  (Ctrl+B)"); save_bm_btn.clicked.connect(self._save_bookmark)
        bm_btn_row.addWidget(save_bm_btn)
        del_bm_btn = QPushButton("Delete"); del_bm_btn.clicked.connect(self._delete_bookmark)
        bm_btn_row.addWidget(del_bm_btn)
        bml.addLayout(bm_btn_row); sl.addWidget(bm_group)

        # Shader Parameters
        shader_group = QGroupBox("Shader Parameters"); sgl = QVBoxLayout(shader_group)
        self.plugin_ui_container = QVBoxLayout(); sgl.addLayout(self.plugin_ui_container)
        
        params_btn_row = QHBoxLayout()
        reset_params_btn = QPushButton("Reset Params")
        reset_params_btn.clicked.connect(self._reset_params)
        params_btn_row.addWidget(reset_params_btn)
        
        rand_params_btn = QPushButton("🎲 Randomize")
        rand_params_btn.clicked.connect(self._randomize_params)
        params_btn_row.addWidget(rand_params_btn)
        sgl.addLayout(params_btn_row)
        
        sl.addWidget(shader_group); sl.addStretch()

        # Wrap sidebar in scroll area
        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Splitter for resizable sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.gl_widget)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 350])
        main_layout.addWidget(splitter)

        self.gl_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.gl_widget.customContextMenuRequested.connect(self._show_context_menu)

        self._build_menubar()

        # Timers
        self.anim_timer = QTimer(self); self.anim_timer.timeout.connect(self._process_auto_zoom)
        self.julia_anim_timer = QTimer(self); self.julia_anim_timer.timeout.connect(self._animate_julia_c)
        self.status_timer = QTimer(self); self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(100)
        
        # History commit debounce timer
        self.history_timer = QTimer(self)
        self.history_timer.setSingleShot(True)
        self.history_timer.setInterval(400)
        self.history_timer.timeout.connect(self._commit_history)
        self.gl_widget.viewInteracted.connect(self._schedule_history_commit)

        self._apply_persisted_prefs()

        if self.plugins:
            first = list(self.plugins.keys())[0]
            self.combo.setCurrentText(first)
            QTimer.singleShot(100, lambda: self._switch_plugin(first))

    def _build_menubar(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        m_file.addAction("Save Screenshot...", self._save_image, QKeySequence("Ctrl+S"))
        m_file.addAction("High-Res Export...", self._save_high_res)
        m_file.addAction("Load View from Image...", self._load_from_image)
        m_file.addSeparator(); m_file.addAction("E&xit", self.close, QKeySequence("Ctrl+Q"))
        m_edit = mb.addMenu("&Edit")
        m_edit.addAction("Undo Zoom", self._zoom_undo, QKeySequence("Ctrl+Z"))
        m_edit.addAction("Redo Zoom", self._zoom_redo, QKeySequence("Ctrl+Shift+Z"))
        m_edit.addSeparator()
        m_edit.addAction("Copy Coordinates", self._copy_coords, QKeySequence("C"))
        m_edit.addAction("Save Bookmark...", self._save_bookmark, QKeySequence("Ctrl+B"))
        m_view = mb.addMenu("&View")
        m_view.addAction("Reset View", self._reset_view, QKeySequence("R"))
        m_view.addAction("Toggle Fullscreen", self._toggle_fullscreen, QKeySequence("F"))
        self._act_hud = m_view.addAction("Toggle HUD", self._toggle_hud, QKeySequence("H")); self._act_hud.setCheckable(True); self._act_hud.setChecked(True)
        m_help = mb.addMenu("&Help")
        m_help.addAction("Keyboard Shortcuts", self._show_help, QKeySequence("F1"))
        m_help.addAction("About", self._show_about)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Copy Coordinates  (C)", self._copy_coords)
        menu.addAction("Save Bookmark  (Ctrl+B)", self._save_bookmark)
        menu.addSeparator()
        menu.addAction("Reset View  (R)", self._reset_view)
        menu.addAction("Save Screenshot  (Ctrl+S)", self._save_image)
        menu.addSeparator()
        menu.addAction("Undo  (Ctrl+Z)", self._zoom_undo)
        menu.addAction("Redo  (Ctrl+Shift+Z)", self._zoom_redo)
        menu.addSeparator()
        a_ch = menu.addAction("Toggle Crosshair"); a_ch.setCheckable(True); a_ch.setChecked(self.gl_widget.show_crosshair)
        a_ch.toggled.connect(lambda s: (setattr(self.gl_widget, "show_crosshair", s), self.gl_widget.update()))
        a_hud = menu.addAction("Toggle HUD"); a_hud.setCheckable(True); a_hud.setChecked(self.gl_widget.show_hud)
        a_hud.toggled.connect(self._toggle_hud)
        menu.addSeparator()
        menu.addAction("Help  (F1)", self._show_help)
        menu.exec(self.gl_widget.mapToGlobal(pos))

    def _apply_persisted_prefs(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as f:
                    s = json.load(f)
                if "window_size" in s: self.resize(*s["window_size"])
                if "palette" in s:
                    idx = self.combo_palette.findText(s["palette"])
                    if idx >= 0: self.combo_palette.setCurrentIndex(idx)
                if "plugin" in s:
                    idx = self.combo.findText(s["plugin"])
                    if idx >= 0: self.combo.setCurrentIndex(idx)
                if "max_iter" in s:
                    self.iter_slider.setValue(s["max_iter"])
                if "brightness" in s:
                    self.bright_slider.setValue(s["brightness"])
                if "gamma" in s:
                    self.gamma_slider.setValue(s["gamma"])
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

    def _persist_prefs(self):
        try:
            s = {
                "window_size": (self.width(), self.height()),
                "palette": self.combo_palette.currentText(),
                "plugin": self.combo.currentText(),
                "max_iter": self.iter_slider.value(),
                "brightness": self.bright_slider.value(),
                "gamma": self.gamma_slider.value(),
            }
            with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")

    def closeEvent(self, e):
        self._persist_prefs()
        self.gl_widget.cleanup()
        super().closeEvent(e)

    def _load_bookmarks(self):
        try:
            if os.path.exists(BOOKMARKS_FILE):
                with open(BOOKMARKS_FILE) as f: return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load bookmarks: {e}")
        return []

    def _save_bookmarks_to_disk(self):
        try:
            with open(BOOKMARKS_FILE, "w") as f: json.dump(self.bookmarks, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save bookmarks: {e}")

    def _refresh_bookmark_combo(self):
        self.combo_bookmarks.blockSignals(True)
        self.combo_bookmarks.clear()
        self.combo_bookmarks.addItem("-- Bookmarks --")
        for bm in self.bookmarks: self.combo_bookmarks.addItem(bm.get("name", "Unnamed"))
        self.combo_bookmarks.blockSignals(False)

    def _save_bookmark(self):
        gw = self.gl_widget
        name, ok = QMessageBox.getText(
            self, "Save Bookmark", "Bookmark name:",
            text=f"Bookmark {len(self.bookmarks)+1}")
        if not ok or not name.strip(): return
        bm = {
            "name": name.strip(),
            "center_x": gw.center_x, "center_y": gw.center_y,
            "zoom": gw.zoom, "max_iter": gw.max_iter,
            "plugin": gw.current_plugin.name if gw.current_plugin else "",
            "palette": gw.current_palette_name,
            "params": copy.deepcopy(gw.current_plugin.params) if gw.current_plugin else {},
        }
        self.bookmarks.append(bm)
        self._save_bookmarks_to_disk(); self._refresh_bookmark_combo()
        self.statusBar().showMessage(f"Bookmark saved: {name.strip()}", 2000)

    def _go_bookmark(self, index):
        if index <= 0 or index > len(self.bookmarks): return
        bm = self.bookmarks[index - 1]
        self._push_history()
        gw = self.gl_widget
        
        plugin_name = bm.get("plugin", "")
        if plugin_name and plugin_name in self.plugins:
            self.combo.setCurrentText(plugin_name)
            
        if "params" in bm and gw.current_plugin:
            gw.current_plugin.params.update(bm["params"])
            if hasattr(gw.current_plugin, "refresh_ui"):
                gw.current_plugin.refresh_ui()
                
        gw.center_x = bm["center_x"]; gw.center_y = bm["center_y"]
        gw.zoom = bm.get("zoom", 2.5); gw.max_iter = bm.get("max_iter", 200)
        self.iter_slider.blockSignals(True); self.iter_slider.setValue(gw.max_iter); self.iter_slider.blockSignals(False)
        self.iter_label.setText(str(gw.max_iter))
        
        palette_name = bm.get("palette", "")
        if palette_name:
            idx = self.combo_palette.findText(palette_name)
            if idx >= 0: self.combo_palette.setCurrentIndex(idx)
            
        gw._ensure_precision()
        gw._mark_dirty(); gw.update()
        self.statusBar().showMessage(f"Loaded bookmark: {bm.get('name','')}", 2000)

    def _delete_bookmark(self):
        idx = self.combo_bookmarks.currentIndex() - 1
        if idx < 0 or idx >= len(self.bookmarks):
            self.statusBar().showMessage("No bookmark selected.", 2000); return
        name = self.bookmarks[idx].get("name", "")
        del self.bookmarks[idx]
        self._save_bookmarks_to_disk(); self._refresh_bookmark_combo()
        self.statusBar().showMessage(f"Deleted bookmark: {name}", 2000)

    def _load_palette_files(self):
        self.combo_palette.clear()
        if os.path.exists("palettes"):
            files = sorted(f for f in os.listdir("palettes") if f.endswith(".map"))
            self.combo_palette.addItems(files)

    def _change_palette(self, name):
        if name: self.gl_widget.load_palette_from_file(name)

    def _update_color_offset(self, val):
        if self.cycle_cb.isChecked():
            self.cycle_cb.setChecked(False)
        self.gl_widget.color_offset = (val / 500.0) % 1.0
        self.gl_widget._mark_dirty(); self.gl_widget.update()
        
    def _update_brightness(self, val):
        self.gl_widget.brightness = val / 100.0
        self.gl_widget._mark_dirty(); self.gl_widget.update()
        
    def _update_gamma(self, val):
        self.gl_widget.gamma = val / 100.0
        self.gl_widget._mark_dirty(); self.gl_widget.update()
        
    def _toggle_invert(self, s):
        self.gl_widget.invert_palette = s
        self.gl_widget._mark_dirty(); self.gl_widget.update()

    def _update_anim_speed(self, val):
        self.gl_widget.anim_speed = val / 100.0

    def _toggle_color_cycle(self, on):
        self.gl_widget.color_cycle = on

    def _update_cycle_speed(self, val):
        self.gl_widget.color_cycle_speed = val / 100.0

    def _toggle_edge_detection(self, s):
        self.gl_widget.edge_detection = s
        self.gl_widget._mark_dirty(); self.gl_widget.update()

    def _go_preset(self, index):
        if index < 0 or index >= len(PRESETS): return
        self._push_history()
        _, cx, cy, z = PRESETS[index]
        gw = self.gl_widget
        gw.center_x = cx; gw.center_y = cy; gw.zoom = z
        gw._ensure_precision()
        gw._mark_dirty(); gw.update()

    def _current_state(self):
        gw = self.gl_widget
        plugin_name = gw.current_plugin.name if gw.current_plugin else ""
        return (plugin_name, gw.center_x, gw.center_y, gw.zoom, gw.max_iter)

    def _apply_state(self, state):
        plugin_name, cx, cy, z, mi = state
        gw = self.gl_widget
        if plugin_name and plugin_name in self.plugins and self.combo.currentText() != plugin_name:
            self.combo.setCurrentText(plugin_name)
            
        gw.center_x = cx; gw.center_y = cy; gw.zoom = z; gw.max_iter = mi
        self.iter_slider.blockSignals(True); self.iter_slider.setValue(mi); self.iter_slider.blockSignals(False)
        self.iter_label.setText(str(mi))
        gw._ensure_precision()
        gw._mark_dirty(); gw.update()

    def _push_history(self):
        self.history.push(self._current_state())

    def _schedule_history_commit(self):
        self.history_timer.start()

    def _commit_history(self):
        current = self._current_state()
        if self._last_history_state != current:
            self.history.push(current)
            self._last_history_state = current

    def _zoom_undo(self):
        prev = self.history.undo(self._current_state())
        if prev is None:
            self.statusBar().showMessage("Nothing to undo.", 1500); return
        self._apply_state(prev)
        self._last_history_state = prev
        self.statusBar().showMessage("Undo.", 1000)

    def _zoom_redo(self):
        nxt = self.history.redo(self._current_state())
        if nxt is None:
            self.statusBar().showMessage("Nothing to redo.", 1500); return
        self._apply_state(nxt)
        self._last_history_state = nxt
        self.statusBar().showMessage("Redo.", 1000)

    def _toggle_auto_zoom(self, checked):
        if checked:
            self.auto_zoom_btn.setText("Stop Auto-Zoom  (Space)")
            self._push_history()
            idx = self.combo_az_target.currentIndex()
            if idx == 0:
                if self.gl_widget._mouse_complex:
                    self._az_target_x, self._az_target_y = self.gl_widget._mouse_complex
                else:
                    self._az_target_x, self._az_target_y = self.gl_widget.center_x, self.gl_widget.center_y
            else:
                _, tx, ty = AUTO_ZOOM_TARGETS[idx - 1]
                self._az_target_x = tx; self._az_target_y = ty
            self.anim_timer.start(16)
        else:
            self.auto_zoom_btn.setText("Auto-Zoom  (Space)")
            self.anim_timer.stop()

    def _process_auto_zoom(self):
        gw = self.gl_widget
        dist = math.hypot(gw.center_x - self._az_target_x, gw.center_y - self._az_target_y)
        smoothing = max(0.005, min(0.05, dist / max(1e-14, gw.zoom) * 0.5))
        gw.center_x += (self._az_target_x - gw.center_x) * smoothing
        gw.center_y += (self._az_target_y - gw.center_y) * smoothing
        factor = self.zoom_speed_slider.value() / 1000.0
        gw.zoom *= factor
        
        gw.update_auto_iter()
                
        gw._ensure_precision()
        gw._mark_dirty(); gw.update()

    def _toggle_julia_anim(self, checked):
        if checked:
            self._julia_angle = 0.0
            self._julia_last_t = time.perf_counter()
            self.julia_anim_timer.start(30)
        else:
            self.julia_anim_timer.stop()

    def _animate_julia_c(self):
        plugin = self.gl_widget.current_plugin
        if not plugin or not hasattr(plugin, "params") or "c_real" not in plugin.params: return
        now = time.perf_counter()
        dt = now - getattr(self, "_julia_last_t", now); self._julia_last_t = now
        self._julia_angle = getattr(self, "_julia_angle", 0.0) + dt * 0.8
        radius = 0.7885
        plugin.params["c_real"] = radius * math.cos(self._julia_angle)
        plugin.params["c_imag"] = radius * math.sin(self._julia_angle)
        plugin.refresh_ui()
        self.gl_widget._mark_dirty(); self.gl_widget.update()

    def _switch_plugin(self, name):
        plugin = self.plugins.get(name)
        if not plugin: return
        for i in reversed(range(self.plugin_ui_container.count())):
            w = self.plugin_ui_container.itemAt(i).widget()
            if w: w.deleteLater()
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
        is_julia_type = "c_real" in plugin.params
        self.julia_anim_cb.setEnabled(is_julia_type)
        if not is_julia_type and self.julia_anim_cb.isChecked():
            self.julia_anim_cb.setChecked(False)

    def _reset_params(self):
        plugin = self.gl_widget.current_plugin
        if not plugin: return
        schema = plugin.get_parameter_schema()
        for p in schema:
            plugin.params[p['name']] = p['default']
        plugin.refresh_ui()
        self.gl_widget._mark_dirty(); self.gl_widget.update()
        
    def _randomize_params(self):
        plugin = self.gl_widget.current_plugin
        if not plugin: return
        plugin.randomize_params()
        self.gl_widget._mark_dirty(); self.gl_widget.update()

    def _toggle_auto_iter(self, checked):
        self.gl_widget.auto_iter_enabled = checked
        self.iter_slider.setEnabled(not checked)
        self.iter_label.setEnabled(not checked)

    def _on_max_iter_changed(self, val: int):
        self.iter_slider.blockSignals(True)
        self.iter_slider.setValue(val)
        self.iter_slider.blockSignals(False)
        self.iter_label.setText(str(val))

    def _update_iterations(self, val):
        self.gl_widget.max_iter = val
        self.iter_label.setText(str(val))
        self.gl_widget._mark_dirty()

    def _apply_manual_coords(self):
        try:
            x = float(self.input_x.text()); y = float(self.input_y.text()); z = float(self.input_zoom.text())
            self._push_history()
            gw = self.gl_widget
            gw.center_x = x; gw.center_y = y; gw.zoom = max(1e-14, z)
            gw._ensure_precision()
            gw._mark_dirty(); gw.update()
        except ValueError:
            self.statusBar().showMessage("Invalid coordinate format!", 3000)

    def _copy_coords(self):
        gw = self.gl_widget
        text = (f"x={gw.center_x:.17g}  y={gw.center_y:.17g}  zoom={gw.zoom:.17g}")
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Copied to clipboard.", 1500)

    def _reset_view(self):
        self._push_history()
        if self.gl_widget.current_plugin:
            s = self.gl_widget.current_plugin.get_default_settings()
            self.gl_widget.apply_settings(s)
            self.iter_slider.blockSignals(True); self.iter_slider.setValue(s["max_iter"]); self.iter_slider.blockSignals(False)
            self.iter_label.setText(str(s["max_iter"]))
            self.gl_widget._mark_dirty(); self.gl_widget.update()

    def _save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "fractal.png", "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if path:
            img = self.gl_widget.grabFramebuffer()
            if path.lower().endswith(".png"):
                gw = self.gl_widget
                meta = {
                    "app": "GPU Fractal Engine",
                    "plugin": gw.current_plugin.name if gw.current_plugin else "",
                    "center_x": gw.center_x,
                    "center_y": gw.center_y,
                    "zoom": gw.zoom,
                    "max_iter": gw.max_iter,
                    "color_offset": gw.color_offset,
                    "palette": gw.current_palette_name,
                    "brightness": gw.brightness,
                    "gamma": gw.gamma,
                }
                img.setText("fractal_data", json.dumps(meta))
            img.save(path)
            self.statusBar().showMessage(f"Saved: {path} (metadata embedded)", 3000)

    def _load_from_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Fractal Image", "", "PNG Images (*.png)")
        if not path: return
        reader = QImageReader(path)
        meta_str = reader.text("fractal_data")
        if not meta_str:
            QMessageBox.warning(self, "No Metadata", "This image does not contain fractal metadata.")
            return
        try:
            meta = json.loads(meta_str)
            self._push_history()
            
            plugin_name = meta.get("plugin", "")
            if plugin_name and plugin_name in self.plugins:
                self.combo.setCurrentText(plugin_name)
            
            palette_name = meta.get("palette", "default.map")
            idx = self.combo_palette.findText(palette_name)
            if idx >= 0: self.combo_palette.setCurrentIndex(idx)
            
            gw = self.gl_widget
            gw.center_x = float(meta["center_x"])
            gw.center_y = float(meta["center_y"])
            gw.zoom = float(meta["zoom"])
            gw.max_iter = int(meta["max_iter"])
            gw.color_offset = float(meta.get("color_offset", 0.0))
            gw.brightness = float(meta.get("brightness", 1.0))
            gw.gamma = float(meta.get("gamma", 1.0))
            
            self.iter_slider.blockSignals(True)
            self.iter_slider.setValue(gw.max_iter)
            self.iter_slider.blockSignals(False)
            self.iter_label.setText(str(gw.max_iter))
            self.bright_slider.setValue(int(gw.brightness * 100))
            self.gamma_slider.setValue(int(gw.gamma * 100))
            self.color_slider.setValue(int(gw.color_offset * 500))
            
            gw._ensure_precision()
            gw._mark_dirty(); gw.update()
            self.statusBar().showMessage("Loaded view from image metadata.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse metadata: {e}")

    def _save_high_res(self):
        dlg = HighResExportDialog(self.gl_widget.width(), self.gl_widget.height(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        w, h, aa = dlg.get_size()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save High-Resolution Image", f"fractal_{w}x{h}.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if not path: return
        self.statusBar().showMessage(f"Rendering {w}x{h} ...", 5000)
        QApplication.processEvents()
        img = self.gl_widget.render_to_image(w, h, aa=aa)
        if img is not None:
            img.save(path)
            self.statusBar().showMessage(f"High-res saved: {path}", 5000)
        else:
            self.statusBar().showMessage("High-res render failed. GPU memory limit exceeded?", 5000)
            QMessageBox.warning(self, "Render Failed", "Failed to render high-res image. The size may exceed your GPU's maximum framebuffer allocation.")

    def _show_help(self):
        HelpDialog(self, self.gl_widget.gl_info).exec()

    def _show_about(self):
        info = self.gl_widget.gl_info or "(no GL info)"
        QMessageBox.about(self, "About",
            f"<h3>GPU Fractal Engine v{APP_VERSION}</h3>"
            f"<p>OpenGL fractal explorer with 7 fractal types, 11 palettes, "
            f"auto-zoom, bookmarks, zoom history, and high-res export.</p>"
            f"<pre style='background:#222;color:#9c9;padding:8px'>{info}</pre>")

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _toggle_hud(self, checked=None):
        if checked is None:
            self.gl_widget.show_hud = not self.gl_widget.show_hud
        else:
            self.gl_widget.show_hud = checked
        self.hud_cb.setChecked(self.gl_widget.show_hud)
        self._act_hud.setChecked(self.gl_widget.show_hud)
        self.gl_widget.update_hud()

    def _update_status(self):
        gw = self.gl_widget
        for line_edit, value in (
            (self.input_x, gw.center_x),
            (self.input_y, gw.center_y),
            (self.input_zoom, gw.zoom),
        ):
            if not line_edit.hasFocus():
                line_edit.setText(f"{value:.17g}")
        prec = "Double" if gw.use_double_precision else "Float"
        plugin_name = gw.current_plugin.name if gw.current_plugin else "?"
        cursor_str = ""
        if gw._mouse_complex is not None and gw._mouse_x >= 0:
            mx, my = gw._mouse_complex
            cursor_str = f"  |  Cursor: ({mx:.6e}, {my:.6e})"
        self.statusBar().showMessage(
            f"FPS: {gw.fps:.0f}  |  {plugin_name}  |  Zoom: {gw.zoom:.6g}  |  "
            f"Prec: {prec}  |  Center: ({gw.center_x:.12g}, {gw.center_y:.12g})  |  "
            f"Iter: {gw.max_iter}{cursor_str}"
        )

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key(); mods = event.modifiers()
        gw = self.gl_widget
        pan_step = 0.1 if not (mods & Qt.KeyboardModifier.ShiftModifier) else 0.3
        if   key == Qt.Key.Key_Left:  gw.pan_by_fraction(-pan_step, 0.0); return
        elif key == Qt.Key.Key_Right: gw.pan_by_fraction( pan_step, 0.0); return
        elif key == Qt.Key.Key_Up:    gw.pan_by_fraction(0.0,  pan_step); return
        elif key == Qt.Key.Key_Down:  gw.pan_by_fraction(0.0, -pan_step); return
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:  gw.zoom_by(0.8); return
        elif key == Qt.Key.Key_Minus: gw.zoom_by(1.25); return

        if key == Qt.Key.Key_F11 or key == Qt.Key.Key_F:
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_F1:
            self._show_help()
        elif key == Qt.Key.Key_R and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._reset_view()
        elif key == Qt.Key.Key_Space:
            self.auto_zoom_btn.toggle()
        elif key == Qt.Key.Key_C and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._copy_coords()
        elif key == Qt.Key.Key_H:
            self._toggle_hud()
        elif key == Qt.Key.Key_J:
            self.gl_widget.julia_interactive = not self.gl_widget.julia_interactive
            state = "ON" if self.gl_widget.julia_interactive else "OFF"
            self.statusBar().showMessage(f"Interactive Julia Mode: {state}", 2000)
        elif key == Qt.Key.Key_BracketLeft:
            idx = self.combo_palette.currentIndex() - 1
            if idx >= 0: self.combo_palette.setCurrentIndex(idx)
        elif key == Qt.Key.Key_BracketRight:
            idx = self.combo_palette.currentIndex() + 1
            if idx < self.combo_palette.count(): self.combo_palette.setCurrentIndex(idx)
        elif key == Qt.Key.Key_S and mods & Qt.KeyboardModifier.ControlModifier:
            self._save_image()
        elif key == Qt.Key.Key_Z and mods & Qt.KeyboardModifier.ControlModifier:
            self._zoom_redo() if (mods & Qt.KeyboardModifier.ShiftModifier) else self._zoom_undo()
        elif key == Qt.Key.Key_B and mods & Qt.KeyboardModifier.ControlModifier:
            self._save_bookmark()
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            idx = key - Qt.Key.Key_1
            names = list(self.plugins.keys())
            if idx < len(names): self.combo.setCurrentText(names[idx])
        else:
            super().keyPressEvent(event)


# =========================================================== ENTRY POINT
if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    logger.info(f"GPU Fractal Engine v{APP_VERSION} starting...")
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    create_palettes()
    window = MainWindow(); window.show()
    exit_code = app.exec()
    logger.info(f"Exiting with code {exit_code}")
    sys.exit(exit_code)