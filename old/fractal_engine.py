"""
GPU Fractal Engine v2.1.0
=========================
A high-performance OpenGL fractal explorer with 6 fractal types,
11 color palettes, auto-zoom animation, bookmarks, zoom history,
high-res export, and keyboard shortcuts.

Single-file version — no external dependencies beyond PySide6, numpy, and OpenGL 4.1+.

Requirements: PySide6, numpy, OpenGL 4.1+
"""

import sys
import logging
import os
import math
import time
import json
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QComboBox, QFileDialog, QStatusBar,
    QGroupBox, QCheckBox, QLineEdit, QFormLayout, QSizePolicy,
    QMenu, QDialog, QDialogButtonBox, QSpinBox, QTextBrowser,
    QMessageBox, QAbstractScrollArea,
)
from PySide6.QtGui import (
    QSurfaceFormat, QPalette, QColor, QKeyEvent, QAction, QKeySequence,
    QPainter, QPen, QClipboard, QVector2D,
)
from PySide6.QtCore import Qt, QTimer, QMutex, QMutexLocker, QPoint, QSize

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLShader, QOpenGLVertexArrayObject,
)

APP_VERSION = "2.1.0"
BOOKMARKS_FILE = "fractal_bookmarks.json"
HISTORY_MAX = 100

# ------------------------------------------------------------------- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================= DARK THEME
def apply_dark_theme(app):
    """Apply a polished Fusion dark theme."""
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


# =========================================================== PALETTE GENERATION
def _smoothstep(t):
    """Hermite smoothstep for smooth color transitions."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB tuples."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _palette_from_stops(stops, size=256):
    """Generate a palette list from gradient stops [(position, (r,g,b)), ...]."""
    colors = []
    for i in range(size):
        t = i / (size - 1)
        for j in range(len(stops) - 1):
            p0, c0 = stops[j]
            p1, c1 = stops[j + 1]
            if p0 <= t <= p1:
                lt = (t - p0) / (p1 - p0) if p1 != p0 else 0.0
                lt = _smoothstep(lt)
                colors.append(_lerp_color(c0, c1, lt))
                break
        else:
            colors.append(stops[-1][1])
    return colors


def _write_palette(filename, colors):
    """Write a list of (r,g,b) tuples to a .map file."""
    path = os.path.join("palettes", filename)
    with open(path, "w") as f:
        for r, g, b in colors:
            f.write(f"{r} {g} {b}\n")


def create_palettes():
    """Generate high-quality .map palette files."""
    os.makedirs("palettes", exist_ok=True)

    _write_palette("default.map", _palette_from_stops([
        (0.00, (0, 7, 100)),  (0.16, (32, 107, 203)),
        (0.33, (49, 210, 207)), (0.50, (30, 225, 60)),
        (0.66, (230, 225, 35)), (0.83, (230, 80, 20)),
        (1.00, (120, 10, 10)),
    ]))

    _write_palette("fire.map", _palette_from_stops([
        (0.0, (0, 0, 0)),    (0.2, (80, 0, 0)),
        (0.4, (200, 30, 0)), (0.6, (255, 128, 0)),
        (0.8, (255, 230, 50)), (1.0, (255, 255, 220)),
    ]))

    _write_palette("ocean.map", _palette_from_stops([
        (0.0, (0, 5, 30)),     (0.25, (0, 30, 100)),
        (0.50, (0, 100, 180)), (0.75, (50, 200, 230)),
        (1.0, (200, 240, 255)),
    ]))

    _write_palette("electric.map", _palette_from_stops([
        (0.0, (0, 0, 20)),    (0.25, (20, 0, 120)),
        (0.50, (80, 0, 255)), (0.75, (0, 180, 255)),
        (1.0, (200, 255, 255)),
    ]))

    _write_palette("forest.map", _palette_from_stops([
        (0.0, (5, 15, 5)),    (0.25, (20, 70, 20)),
        (0.50, (40, 150, 40)), (0.75, (140, 200, 60)),
        (1.0, (220, 240, 150)),
    ]))

    _write_palette("sunset.map", _palette_from_stops([
        (0.0, (20, 0, 50)),   (0.25, (120, 10, 80)),
        (0.50, (220, 60, 30)), (0.75, (255, 170, 50)),
        (1.0, (255, 240, 150)),
    ]))

    _write_palette("grayscale.map", _palette_from_stops([
        (0.0, (0, 0, 0)), (1.0, (255, 255, 255)),
    ]))

    _write_palette("neon.map", _palette_from_stops([
        (0.0, (0, 0, 0)),     (0.20, (255, 0, 100)),
        (0.40, (0, 255, 200)), (0.60, (0, 100, 255)),
        (0.80, (255, 255, 0)), (1.0, (255, 0, 255)),
    ]))

    _write_palette("ice.map", _palette_from_stops([
        (0.0, (10, 10, 40)),   (0.30, (30, 60, 150)),
        (0.60, (100, 180, 240)), (0.85, (200, 230, 255)),
        (1.0, (255, 255, 255)),
    ]))

    _write_palette("magma.map", _palette_from_stops([
        (0.0, (0, 0, 4)),      (0.25, (80, 18, 123)),
        (0.50, (183, 55, 121)), (0.75, (254, 136, 76)),
        (1.0, (252, 253, 191)),
    ]))

    _write_palette("viridis.map", _palette_from_stops([
        (0.0, (68, 1, 84)),    (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)), (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]))


# ======================================================= PRESETS
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

SHORTCUTS_TEXT = """
<h2>GPU Fractal Engine — Keyboard Shortcuts</h2>
<table cellpadding="4" cellspacing="0" border="0" width="100%">
<tr><td><b>Mouse Drag</b></td><td>Pan the view</td></tr>
<tr><td><b>Scroll Wheel</b></td><td>Zoom in/out (towards cursor)</td></tr>
<tr><td><b>Shift + Scroll</b></td><td>Adjust max iterations</td></tr>
<tr><td><b>Double-Click</b></td><td>Center and zoom in</td></tr>
<tr><td><b>Middle-Click</b></td><td>Center on cursor</td></tr>
<tr><td><b>Right-Click</b></td><td>Context menu</td></tr>
<tr><td colspan="2">&nbsp;</td></tr>
<tr><td><b>Space</b></td><td>Toggle auto-zoom</td></tr>
<tr><td><b>R</b></td><td>Reset view to default</td></tr>
<tr><td><b>C</b></td><td>Copy coordinates to clipboard</td></tr>
<tr><td><b>F</b></td><td>Toggle fullscreen</td></tr>
<tr><td><b>F1</b></td><td>Show this help dialog</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save screenshot</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Zoom undo (go back)</td></tr>
<tr><td><b>Ctrl+Shift+Z</b></td><td>Zoom redo (go forward)</td></tr>
<tr><td><b>Ctrl+B</b></td><td>Save bookmark</td></tr>
<tr><td><b>1-6</b></td><td>Switch fractal plugin</td></tr>
<tr><td><b>F11</b></td><td>Toggle fullscreen (legacy)</td></tr>
</table>
"""


# ======================================================= HELP DIALOG
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(460, 420)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(SHORTCUTS_TEXT)
        browser.setOpenExternalLinks(False)
        layout.addWidget(browser)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ======================================================= HIGH-RES EXPORT DIALOG
class HighResExportDialog(QDialog):
    def __init__(self, current_w, current_h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("High-Resolution Export")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Export the current view at a custom resolution:"))
        layout.addWidget(QLabel(""))  # spacer

        form = QFormLayout()
        self.spin_w = QSpinBox()
        self.spin_w.setRange(256, 16384)
        self.spin_w.setValue(current_w)
        self.spin_w.setSingleStep(256)
        form.addRow("Width (px):", self.spin_w)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(256, 16384)
        self.spin_h.setValue(current_h)
        self.spin_h.setSingleStep(256)
        form.addRow("Height (px):", self.spin_h)

        # Preset buttons
        preset_row = QHBoxLayout()
        for label, w, h in [("4K", 3840, 2160), ("8K", 7680, 4320), ("Current", current_w, current_h)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, ww=w, hh=h: (self.spin_w.setValue(ww), self.spin_h.setValue(hh)))
            preset_row.addWidget(b)
        form.addRow("Presets:", preset_row)

        layout.addLayout(form)
        layout.addWidget(QLabel(""))
        layout.addWidget(QLabel("Note: larger sizes will take longer to render."))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_size(self):
        return self.spin_w.value(), self.spin_h.value()


# ======================================================= FRACTAL PLUGINS (inlined from backend.py)
class BaseFractalPlugin(ABC):
    """Base class for fractal plugins with shader generation and UI scaffolding."""

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def formula_code(self):
        """GLSL code for the core iteration loop body."""
        pass

    @property
    def init_code(self):
        """GLSL initialization code. Override for Julia-type fractals."""
        return """
            VEC2 z = VEC2(0.0);
            VEC2 c = c_pos;
        """

    @property
    def description(self):
        return ""

    vertex_shader = """
    #version 410 core
    layout (location = 0) in vec2 a_position;
    void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
    """

    _coloring_footer = """
        if (iter >= float(u_max_iter)) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            float log_zn = log(dot(z, z)) / 2.0;
            float nu = log(log_zn / log(2.0)) / log(2.0);
            iter = iter + 1.0 - nu;
            float t = fract(iter * 0.025 + u_color_offset);
            vec3 col = texture(u_palette, vec2(t, 0.0)).rgb;
            fragColor = vec4(col, 1.0);
        }
    }
    """

    def __init__(self):
        self.params = {}
        for p in self.get_parameter_schema():
            self.params[p['name']] = p['default']

    def get_parameter_schema(self):
        return []

    def get_fragment_shader(self, use_double=False):
        if use_double:
            header = """#version 410 core
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

    def setup_ui(self, parent):
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self._ui_bindings = []
        schema = self.get_parameter_schema()
        if not schema:
            layout.addWidget(QLabel("No configurable parameters."))
            return widget
        for p in schema:
            row = QHBoxLayout()
            name_lbl = QLabel(f"{p['label']}:")
            name_lbl.setMinimumWidth(90)
            row.addWidget(name_lbl)
            fmt = p.get('fmt', '.2f')
            scale = p.get('scale', 1000)
            val_lbl = QLabel(f"{self.params[p['name']]:{fmt}}")
            val_lbl.setMinimumWidth(55)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if p['type'] == 'float':
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(int(p['min'] * scale), int(p['max'] * scale))
                slider.setValue(int(self.params[p['name']] * scale))
                slider.valueChanged.connect(
                    lambda v, nm=p['name'], vl=val_lbl, f=fmt, s=scale:
                        self._update_param(nm, v / s, vl, f)
                )
                row.addWidget(slider, stretch=1)
                self._ui_bindings.append((slider, val_lbl, p['name'], p))
            elif p['type'] == 'int':
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(p['min'], p['max'])
                slider.setValue(self.params[p['name']])
                slider.valueChanged.connect(
                    lambda v, nm=p['name'], vl=val_lbl:
                        self._update_param(nm, v, vl, 'd')
                )
                row.addWidget(slider, stretch=1)
                self._ui_bindings.append((slider, val_lbl, p['name'], p))
            row.addWidget(val_lbl)
            layout.addLayout(row)
        return widget

    def _update_param(self, name, value, value_label, fmt):
        self.params[name] = value
        value_label.setText(f"{value:{fmt}}")

    def refresh_ui(self):
        if not hasattr(self, "_ui_bindings"):
            return
        for slider, label, name, schema in self._ui_bindings:
            fmt = schema.get('fmt', '.2f')
            scale = schema.get('scale', 1000)
            label.setText(f"{self.params[name]:{fmt}}")
            slider.blockSignals(True)
            if schema['type'] == 'float':
                slider.setValue(int(self.params[name] * scale))
            else:
                slider.setValue(self.params[name])
            slider.blockSignals(False)

    def get_default_settings(self):
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 200}

    def update_uniforms(self, program, gl, time, width, height,
                        center_x, center_y, zoom, max_iter, color_offset):
        program.setUniformValue("u_resolution", QVector2D(width, height))
        program.setUniformValue("u_max_iter", int(max_iter))
        program.setUniformValue("u_color_offset", float(color_offset))
        use_double = (zoom < 1e-6)
        if use_double:
            loc = program.uniformLocation("u_center")
            if loc != -1:
                gl.glUniform2d(loc, float(center_x), float(center_y))
            loc = program.uniformLocation("u_zoom")
            if loc != -1:
                gl.glUniform1d(loc, float(zoom))
        else:
            program.setUniformValue("u_center", QVector2D(center_x, center_y))
            program.setUniformValue("u_zoom", float(zoom))
        for name, value in self.params.items():
            loc = program.uniformLocation(f"u_{name}")
            if loc == -1:
                continue
            if isinstance(value, float):
                program.setUniformValue(loc, value)
            elif isinstance(value, int):
                program.setUniformValue(loc, value)
            elif isinstance(value, tuple):
                program.setUniformValue(loc, QVector2D(*value))


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
    def get_default_settings(self):
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 200}


class JuliaPlugin(BaseFractalPlugin):
    name = "Julia Set"
    description = "Julia set z = z^2 + c where c is a fixed complex constant."
    init_code = """
        VEC2 z = c_pos;
        VEC2 c = VEC2(u_c_real, u_c_imag);
    """
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
            {'name': 'c_real', 'type': 'float', 'default': -0.7, 'min': -2.0, 'max': 2.0,
             'label': 'Real (c)', 'fmt': '.4f', 'scale': 10000},
            {'name': 'c_imag', 'type': 'float', 'default': 0.27015, 'min': -2.0, 'max': 2.0,
             'label': 'Imag (c)', 'fmt': '.4f', 'scale': 10000},
        ]
    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 1.5, "max_iter": 200}


class BurningShipPlugin(BaseFractalPlugin):
    name = "Burning Ship"
    description = "Variation using absolute values: z = (|Re(z)| + i|Im(z)|)^2 + c."
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
    description = "Generalization z = z^n + c with adjustable power n."
    formula_code = """
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
    """
    def get_parameter_schema(self):
        return [
            {'name': 'power', 'type': 'float', 'default': 3.0, 'min': 2.0, 'max': 10.0,
             'label': 'Power', 'fmt': '.2f', 'scale': 100},
        ]
    def get_default_settings(self):
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 150}


class PhoenixPlugin(BaseFractalPlugin):
    name = "Phoenix"
    description = "Phoenix fractal with memory: z = z^2 + c + p * z_prev."
    init_code = """
        VEC2 z = c_pos;
        VEC2 c = VEC2(u_c_real, u_c_imag);
        VEC2 z_prev = VEC2(0.0);
    """
    formula_code = """
        for (int i = 0; i < 10000; i++) {
            if (i >= u_max_iter) break;
            if (dot(z, z) > 256.0) break;
            FLOAT p = FLOAT(u_phoenix_p);
            VEC2 z_new = VEC2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y)
                         + c + p * z_prev;
            z_prev = z;
            z = z_new;
            iter += 1.0;
        }
    """
    def get_parameter_schema(self):
        return [
            {'name': 'c_real', 'type': 'float', 'default': -0.7, 'min': -2.0, 'max': 2.0,
             'label': 'Real (c)', 'fmt': '.3f', 'scale': 1000},
            {'name': 'c_imag', 'type': 'float', 'default': 0.27015, 'min': -2.0, 'max': 2.0,
             'label': 'Imag (c)', 'fmt': '.3f', 'scale': 1000},
            {'name': 'phoenix_p', 'type': 'float', 'default': 0.5, 'min': -1.5, 'max': 1.5,
             'label': 'Phoenix P', 'fmt': '.3f', 'scale': 1000},
        ]
    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 1.8, "max_iter": 150}


class TricornPlugin(BaseFractalPlugin):
    name = "Tricorn (Mandelbar)"
    description = "Conjugate variant z = conj(z)^2 + c."
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


AVAILABLE_PLUGINS = [
    MandelbrotPlugin, JuliaPlugin, BurningShipPlugin,
    MultibrotPlugin, PhoenixPlugin, TricornPlugin,
]


# ======================================================= ZOOM HISTORY
class ZoomHistory:
    """Undo/redo stack for fractal navigation state."""

    def __init__(self, max_size=HISTORY_MAX):
        self._back = deque(maxlen=max_size)
        self._fwd = deque(maxlen=max_size)

    def push(self, center_x, center_y, zoom, max_iter):
        """Save current state before navigating."""
        self._back.append((center_x, center_y, zoom, max_iter))
        self._fwd.clear()

    def can_undo(self):
        return len(self._back) > 0

    def can_redo(self):
        return len(self._fwd) > 0

    def undo(self):
        if not self._back:
            return None
        return self._back.pop()

    def redo(self):
        if not self._fwd:
            return None
        return self._fwd.pop()

    def save_current_for_undo(self, center_x, center_y, zoom, max_iter):
        """Call BEFORE changing view, so we can undo back to this state."""
        self._back.append((center_x, center_y, zoom, max_iter))
        self._fwd.clear()


# ======================================================= GL WIDGET
class FractalGLWidget(QOpenGLWidget):
    """Advanced OpenGL fractal renderer with dynamic shaders, double precision, and palette textures."""

    GL = type("GL", (), {
        "COLOR_BUFFER_BIT": 0x00004000,
        "TRIANGLE_STRIP":  0x0005,
        "FLOAT":           0x1406,
        "TEXTURE_2D":      0x0DE1,
        "TEXTURE_MIN_FILTER": 0x2801,
        "LINEAR":          0x2601,
        "TEXTURE_MAG_FILTER": 0x2800,
        "TEXTURE_WRAP_S":  0x2802,
        "REPEAT":          0x2901,
        "TEXTURE_WRAP_T":  0x2803,
        "RGBA":            0x1908,
        "UNSIGNED_BYTE":   0x1401,
        "TEXTURE0":        0x84C0,
    })()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # --- Render State ---
        self.time = 0.0
        self.zoom = 2.5
        self.center_x = -0.5
        self.center_y = 0.0
        self.max_iter = 200
        self.animation_paused = False
        self.color_offset = 0.0
        self.anim_speed = 1.2  # configurable color animation speed

        # --- Plugin System ---
        self.current_plugin = None
        self.program = None
        self.vbo = None
        self.vao = None
        self.gl = None

        # --- Interaction ---
        self.last_mouse_pos = None
        self.show_crosshair = True

        # --- Textures ---
        self.palette_texture = None
        self.current_palette_name = ""

        # --- Animation Timer (uses delta time) ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)  # ~60 FPS target
        self._last_tick = time.perf_counter()

        # --- Performance Monitoring ---
        self.frame_count = 0
        self.fps_acc = 0.0
        self.fps = 0.0

        # --- Thread Safety ---
        self.resource_mutex = QMutex()

        # --- Precision Mode ---
        self.use_double_precision = False

        # --- Crosshair state (for overlay) ---
        self._mouse_x = 0
        self._mouse_y = 0

    # ------------------------------------------------------------------ cleanup
    def cleanup(self):
        if not self.context():
            return
        logger.info("Cleaning up OpenGL resources...")
        self.makeCurrent()
        for res in (self.vao, self.vbo):
            if res:
                res.destroy()
        self.vao = self.vbo = None

        if self.program:
            self.program.release()
            self.program.removeAllShaders()
            self.program.deleteLater()
            self.program = None

        if self.palette_texture and self.gl:
            self.gl.glDeleteTextures(1, [self.palette_texture])
            self.palette_texture = None
        self.doneCurrent()

    # --------------------------------------------------------------- geometry
    def _setup_geometry(self):
        if self.vbo:
            return
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.vbo = QOpenGLBuffer()
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(vertices.tobytes(), vertices.nbytes)
        self.vbo.release()

    # ---------------------------------------------------------------- palettes
    def load_palette_from_file(self, filename):
        if not self.gl or not self.context():
            return
        filepath = os.path.join("palettes", filename)
        if not os.path.exists(filepath):
            logger.warning(f"Palette not found: {filepath}")
            return
        if filename == self.current_palette_name and self.palette_texture:
            return

        colors = []
        try:
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            colors.append((int(parts[0]), int(parts[1]), int(parts[2])))
                        except ValueError:
                            continue
        except Exception as e:
            logger.error(f"Error reading palette: {e}")
            return

        if not colors:
            colors = [(0, 0, 0), (255, 255, 255)]

        palette_data = bytearray(256 * 4)
        for i in range(256):
            r, g, b = colors[i % len(colors)]
            o = i * 4
            palette_data[o] = r
            palette_data[o + 1] = g
            palette_data[o + 2] = b
            palette_data[o + 3] = 255

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
        logger.info(f"Loaded palette: {filename}")

    # ----------------------------------------------------------------- plugin
    def load_plugin(self, plugin):
        if not self.isValid() or not self.context() or not self.gl:
            logger.warning("Attempted to load plugin without valid GL context")
            return

        logger.info(f"Loading plugin: {plugin.name}...")

        new_dp = self.zoom < 1e-6
        if new_dp != self.use_double_precision:
            logger.info(f"Precision -> {'Double' if new_dp else 'Single'}")
            self.use_double_precision = new_dp

        with QMutexLocker(self.resource_mutex):
            self.makeCurrent()

            if self.program:
                self.gl.glUseProgram(0)
                self.program.removeAllShaders()
                self.program.release()
                self.program.deleteLater()
                self.program = None

            self.program = QOpenGLShaderProgram(self)

            if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, plugin.vertex_shader):
                logger.error(f"VERTEX error: {self.program.log()}")
                self.program.deleteLater()
                self.program = None
                self.doneCurrent()
                return

            try:
                frag_src = plugin.get_fragment_shader(use_double=self.use_double_precision)
            except Exception as e:
                logger.error(f"Shader generation error: {e}")
                self.doneCurrent()
                return

            if not self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_src):
                logger.error(f"FRAGMENT error: {self.program.log()}")
                self.program.deleteLater()
                self.program = None
                self.doneCurrent()
                return

            self.program.bindAttributeLocation("a_position", 0)
            if not self.program.link():
                logger.error(f"Link failed: {self.program.log()}")
                self.program.deleteLater()
                self.program = None
                self.doneCurrent()
                return

            self._setup_geometry()
            if not self.vao:
                self.vao = QOpenGLVertexArrayObject()
                self.vao.create()

            self.vao.bind()
            if self.vbo:
                self.vbo.bind()
            self.program.enableAttributeArray(0)
            self.program.setAttributeBuffer(0, self.GL.FLOAT, 0, 2)
            self.vao.release()

            self.current_plugin = plugin
            self.apply_settings(plugin.get_default_settings())

            self.current_palette_name = ""
            self.load_palette_from_file("default.map")
            self.update()

    # --------------------------------------------------------------- settings
    def apply_settings(self, settings):
        self.center_x = float(settings.get("center_x", 0.0))
        self.center_y = float(settings.get("center_y", 0.0))
        self.zoom = float(settings.get("zoom", 1.0))
        self.max_iter = int(settings.get("max_iter", 100))

    # ----------------------------------------------------------- GL lifecycle
    def initializeGL(self):
        ctx = self.context()
        if not ctx:
            logger.critical("QOpenGLContext is null")
            return
        self.gl = ctx.extraFunctions()
        if not self.gl:
            logger.critical("Failed to resolve GL functions")
            return
        logger.info(f"OpenGL {ctx.format().majorVersion()}.{ctx.format().minorVersion()}")
        ctx.aboutToBeDestroyed.connect(self.cleanup)
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self._setup_geometry()
        self.load_palette_from_file("default.map")

    def paintGL(self):
        if not self.current_plugin or not self.program or not self.gl:
            return

        with QMutexLocker(self.resource_mutex):
            self.frame_count += 1

            gl = self.gl
            gl.glClear(self.GL.COLOR_BUFFER_BIT)
            self.program.bind()
            if self.vao:
                self.vao.bind()

            if self.palette_texture:
                gl.glActiveTexture(self.GL.TEXTURE0)
                gl.glBindTexture(self.GL.TEXTURE_2D, self.palette_texture)

            try:
                self.current_plugin.update_uniforms(
                    self.program, gl, self.time, self.width(), self.height(),
                    self.center_x, self.center_y, self.zoom, self.max_iter,
                    self.color_offset
                )
                gl.glDrawArrays(self.GL.TRIANGLE_STRIP, 0, 4)
            except Exception as e:
                logger.error(f"Paint error: {e}")

            if self.vao:
                self.vao.release()
            self.program.release()

        # Draw crosshair overlay using QPainter
        if self.show_crosshair:
            self._paint_crosshair()

    def _paint_crosshair(self):
        """Draw a subtle crosshair at the center of the viewport."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255, 60))
        pen.setWidth(1)
        painter.setPen(pen)
        cx, cy = self.width() // 2, self.height() // 2
        size = 15
        painter.drawLine(cx - size, cy, cx + size, cy)
        painter.drawLine(cx, cy - size, cx, cy + size)
        # Small circle
        painter.drawEllipse(QPoint(cx, cy), 4, 4)
        painter.end()

    # -------------------------------------------------------------- animation
    def _tick(self):
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now

        # FPS tracking
        self.fps_acc += dt
        self._fc = getattr(self, "_fc", 0) + 1
        if self.fps_acc >= 1.0:
            self.fps = self._fc / self.fps_acc
            self._fc = 0
            self.fps_acc = 0.0

        if not self.animation_paused:
            self.time += dt * self.anim_speed
        self.update()

    # ------------------------------------------------------------- input utils
    def _screen_to_fractal(self, sx, sy):
        """Convert screen pixel coords -> fractal complex coords."""
        fx = self.center_x + (sx - self.width() / 2.0) / self.height() * self.zoom
        fy = self.center_y - (sy - self.height() / 2.0) / self.height() * self.zoom
        return fx, fy

    # ----------------------------------------------------------------- wheel
    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = max(1, self.max_iter // 20)
            self.max_iter = max(10, min(10000, self.max_iter + step * (1 if delta > 0 else -1)))
            self.update()
            return

        # Exponential zoom — natural feel
        zoom_factor = math.exp(-delta / 600.0)

        # Zoom towards cursor position
        mx, my = event.position().x(), event.position().y()
        fx, fy = self._screen_to_fractal(mx, my)

        new_zoom = max(1e-14, self.zoom * zoom_factor)

        # Keep point under cursor fixed
        self.center_x = fx - (mx - self.width() / 2.0) / self.height() * new_zoom
        self.center_y = fy + (my - self.height() / 2.0) / self.height() * new_zoom
        self.zoom = new_zoom

        # Check precision switch
        if (self.zoom < 1e-6) != self.use_double_precision:
            if self.current_plugin:
                self.load_plugin(self.current_plugin)

        self.update()

    # ----------------------------------------------------------------- mouse
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.MiddleButton:
            fx, fy = self._screen_to_fractal(event.position().x(), event.position().y())
            self.center_x = fx
            self.center_y = fy
            self.update()

    def mouseMoveEvent(self, event):
        self._mouse_x = int(event.position().x())
        self._mouse_y = int(event.position().y())
        if self.last_mouse_pos and (event.buttons() & Qt.LeftButton):
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()
            scale = self.zoom / max(1, self.height())
            self.center_x -= dx * scale
            self.center_y += dy * scale
            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event):
        """Double-click to center and zoom in."""
        fx, fy = self._screen_to_fractal(event.position().x(), event.position().y())
        self.center_x = fx
        self.center_y = fy
        self.zoom *= 0.5
        if (self.zoom < 1e-6) != self.use_double_precision:
            if self.current_plugin:
                self.load_plugin(self.current_plugin)
        self.update()


# ======================================================= MAIN WINDOW
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"GPU Fractal Engine v{APP_VERSION}")
        self.resize(1280, 800)

        self.setStatusBar(QStatusBar(self))
        self.plugins = {cls.name: cls() for cls in AVAILABLE_PLUGINS}

        # --- Zoom History ---
        self.history = ZoomHistory()

        # --- Bookmarks ---
        self.bookmarks = self._load_bookmarks()

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

        spd_row = QHBoxLayout()
        spd_row.addWidget(QLabel("Anim Speed:"))
        self.anim_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.anim_speed_slider.setRange(0, 300)
        self.anim_speed_slider.setValue(120)
        self.anim_speed_slider.setToolTip("Color animation speed")
        self.anim_speed_slider.valueChanged.connect(self._update_anim_speed)
        spd_row.addWidget(self.anim_speed_slider, stretch=1)
        al.addLayout(spd_row)

        self.crosshair_cb = QCheckBox("Show Crosshair")
        self.crosshair_cb.setChecked(True)
        self.crosshair_cb.toggled.connect(lambda s: setattr(self.gl_widget, "show_crosshair", s))
        al.addWidget(self.crosshair_cb)

        sl.addWidget(appear_group)

        # --- Animation ---
        anim_group = QGroupBox("Animation")
        aml = QVBoxLayout(anim_group)

        self.auto_zoom_btn = QPushButton("Auto-Zoom  (Space)")
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

        zspd_row = QHBoxLayout()
        zspd_row.addWidget(QLabel("Speed:"))
        self.zoom_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_speed_slider.setRange(900, 995)
        self.zoom_speed_slider.setValue(970)
        zspd_row.addWidget(self.zoom_speed_slider, stretch=1)
        aml.addLayout(zspd_row)

        self.pause_time_cb = QCheckBox("Pause Color Animation")
        self.pause_time_cb.toggled.connect(lambda s: setattr(self.gl_widget, "animation_paused", s))
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

        hint = QLabel("Shift+Scroll to adjust iterations")
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

        btn_copy = QPushButton("Copy Coordinates")
        btn_copy.clicked.connect(self._copy_coords)

        nl.addRow("Center X:", self.input_x)
        nl.addRow("Center Y:", self.input_y)
        nl.addRow("Zoom:", self.input_zoom)
        nl.addRow(btn_go)
        nl.addRow(btn_copy)
        sl.addWidget(nav_group)

        # --- Actions ---
        act_row1 = QHBoxLayout()
        reset_btn = QPushButton("Reset  (R)")
        reset_btn.clicked.connect(self._reset_view)
        act_row1.addWidget(reset_btn)

        export_btn = QPushButton("Save Image  (Ctrl+S)")
        export_btn.clicked.connect(self._save_image)
        act_row1.addWidget(export_btn)

        hires_btn = QPushButton("High-Res Export")
        hires_btn.clicked.connect(self._save_high_res)
        act_row1.addWidget(hires_btn)
        sl.addLayout(act_row1)

        act_row2 = QHBoxLayout()
        back_btn = QPushButton("Undo Zoom  (Ctrl+Z)")
        back_btn.clicked.connect(self._zoom_undo)
        act_row2.addWidget(back_btn)

        fwd_btn = QPushButton("Redo Zoom  (Ctrl+Shift+Z)")
        fwd_btn.clicked.connect(self._zoom_redo)
        act_row2.addWidget(fwd_btn)

        help_btn = QPushButton("Help  (F1)")
        help_btn.clicked.connect(self._show_help)
        act_row2.addWidget(help_btn)
        sl.addLayout(act_row2)

        # --- Bookmarks ---
        bm_group = QGroupBox("Bookmarks")
        bml = QVBoxLayout(bm_group)

        self.combo_bookmarks = QComboBox()
        self._refresh_bookmark_combo()
        self.combo_bookmarks.currentIndexChanged.connect(self._go_bookmark)
        bml.addWidget(self.combo_bookmarks)

        bm_btn_row = QHBoxLayout()
        save_bm_btn = QPushButton("Save  (Ctrl+B)")
        save_bm_btn.clicked.connect(self._save_bookmark)
        bm_btn_row.addWidget(save_bm_btn)

        del_bm_btn = QPushButton("Delete")
        del_bm_btn.clicked.connect(self._delete_bookmark)
        bm_btn_row.addWidget(del_bm_btn)
        bml.addLayout(bm_btn_row)

        sl.addWidget(bm_group)

        # --- Shader Parameters ---
        shader_group = QGroupBox("Shader Parameters")
        sgl = QVBoxLayout(shader_group)
        self.plugin_ui_container = QVBoxLayout()
        sgl.addLayout(self.plugin_ui_container)
        sl.addWidget(shader_group)

        sl.addStretch()
        main_layout.addWidget(sidebar, stretch=1)

        # --- Context menu on GL widget ---
        self.gl_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.gl_widget.customContextMenuRequested.connect(self._show_context_menu)

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

    # ----------------------------------------------------------- context menu
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
        act_crosshair = menu.addAction("Toggle Crosshair")
        act_crosshair.setCheckable(True)
        act_crosshair.setChecked(self.gl_widget.show_crosshair)
        act_crosshair.toggled.connect(lambda s: setattr(self.gl_widget, "show_crosshair", s))
        menu.addAction("Help  (F1)", self._show_help)
        menu.exec(self.gl_widget.mapToGlobal(pos))

    # ------------------------------------------------------------- bookmarks
    def _load_bookmarks(self):
        try:
            if os.path.exists(BOOKMARKS_FILE):
                with open(BOOKMARKS_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load bookmarks: {e}")
        return []

    def _save_bookmarks_to_disk(self):
        try:
            with open(BOOKMARKS_FILE, "w") as f:
                json.dump(self.bookmarks, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save bookmarks: {e}")

    def _refresh_bookmark_combo(self):
        self.combo_bookmarks.blockSignals(True)
        self.combo_bookmarks.clear()
        self.combo_bookmarks.addItem("-- Bookmarks --")
        for bm in self.bookmarks:
            self.combo_bookmarks.addItem(bm.get("name", "Unnamed"))
        self.combo_bookmarks.blockSignals(False)

    def _save_bookmark(self):
        gw = self.gl_widget
        name, ok = QMessageBox.getText(
            self, "Save Bookmark", "Bookmark name:",
            text=f"Bookmark {len(self.bookmarks) + 1}",
        )
        if not ok or not name.strip():
            return
        bm = {
            "name": name.strip(),
            "center_x": gw.center_x,
            "center_y": gw.center_y,
            "zoom": gw.zoom,
            "max_iter": gw.max_iter,
            "plugin": gw.current_plugin.name if gw.current_plugin else "",
            "palette": gw.current_palette_name,
        }
        self.bookmarks.append(bm)
        self._save_bookmarks_to_disk()
        self._refresh_bookmark_combo()
        self.statusBar().showMessage(f"Bookmark saved: {name.strip()}", 2000)

    def _go_bookmark(self, index):
        if index <= 0 or index > len(self.bookmarks):
            return
        bm = self.bookmarks[index - 1]
        self._push_history()
        gw = self.gl_widget
        gw.center_x = bm["center_x"]
        gw.center_y = bm["center_y"]
        gw.zoom = bm.get("zoom", 2.5)
        gw.max_iter = bm.get("max_iter", 200)
        self.iter_slider.setValue(gw.max_iter)
        # Switch plugin if bookmark has one
        plugin_name = bm.get("plugin", "")
        if plugin_name and plugin_name in self.plugins:
            self.combo.setCurrentText(plugin_name)
        # Switch palette if bookmark has one
        palette_name = bm.get("palette", "")
        if palette_name:
            idx = self.combo_palette.findText(palette_name)
            if idx >= 0:
                self.combo_palette.setCurrentIndex(idx)
        # Precision switch
        if (gw.zoom < 1e-6) != gw.use_double_precision:
            if gw.current_plugin:
                gw.load_plugin(gw.current_plugin)
        gw.update()
        self.statusBar().showMessage(f"Loaded bookmark: {bm.get('name', '')}", 2000)

    def _delete_bookmark(self):
        idx = self.combo_bookmarks.currentIndex() - 1  # -1 for header
        if idx < 0 or idx >= len(self.bookmarks):
            self.statusBar().showMessage("No bookmark selected.", 2000)
            return
        name = self.bookmarks[idx].get("name", "")
        del self.bookmarks[idx]
        self._save_bookmarks_to_disk()
        self._refresh_bookmark_combo()
        self.statusBar().showMessage(f"Deleted bookmark: {name}", 2000)

    # --------------------------------------------------------------- palettes
    def _load_palette_files(self):
        self.combo_palette.clear()
        if os.path.exists("palettes"):
            files = sorted(f for f in os.listdir("palettes") if f.endswith(".map"))
            self.combo_palette.addItems(files)

    def _change_palette(self, name):
        if name:
            self.gl_widget.load_palette_from_file(name)

    def _update_color_offset(self, val):
        self.gl_widget.color_offset = val / 50.0
        self.gl_widget.update()

    def _update_anim_speed(self, val):
        self.gl_widget.anim_speed = val / 100.0

    # ---------------------------------------------------------------- presets
    def _go_preset(self, index):
        if index < 0 or index >= len(PRESETS):
            return
        self._push_history()
        _, cx, cy, z = PRESETS[index]
        self.gl_widget.center_x = cx
        self.gl_widget.center_y = cy
        self.gl_widget.zoom = z
        if (z < 1e-6) != self.gl_widget.use_double_precision:
            if self.gl_widget.current_plugin:
                self.gl_widget.load_plugin(self.gl_widget.current_plugin)
        self.gl_widget.update()

    # ------------------------------------------------------------- zoom history
    def _push_history(self):
        """Push current GL state onto the undo stack."""
        gw = self.gl_widget
        self.history.save_current_for_undo(gw.center_x, gw.center_y, gw.zoom, gw.max_iter)

    def _zoom_undo(self):
        state = self.history.undo()
        if not state:
            self.statusBar().showMessage("Nothing to undo.", 1500)
            return
        cx, cy, z, mi = state
        # Save current state to redo stack
        gw = self.gl_widget
        # Push current state to fwd stack
        self.history._fwd.append((gw.center_x, gw.center_y, gw.zoom, gw.max_iter))
        gw.center_x = cx
        gw.center_y = cy
        gw.zoom = z
        gw.max_iter = mi
        self.iter_slider.setValue(mi)
        if (z < 1e-6) != gw.use_double_precision:
            if gw.current_plugin:
                gw.load_plugin(gw.current_plugin)
        gw.update()
        self.statusBar().showMessage("Undo.", 1000)

    def _zoom_redo(self):
        state = self.history.redo()
        if not state:
            self.statusBar().showMessage("Nothing to redo.", 1500)
            return
        cx, cy, z, mi = state
        gw = self.gl_widget
        self.history._back.append((gw.center_x, gw.center_y, gw.zoom, gw.max_iter))
        gw.center_x = cx
        gw.center_y = cy
        gw.zoom = z
        gw.max_iter = mi
        self.iter_slider.setValue(mi)
        if (z < 1e-6) != gw.use_double_precision:
            if gw.current_plugin:
                gw.load_plugin(gw.current_plugin)
        gw.update()
        self.statusBar().showMessage("Redo.", 1000)

    # -------------------------------------------------------------- auto-zoom
    def _toggle_auto_zoom(self, checked):
        if checked:
            self.auto_zoom_btn.setText("Stop Auto-Zoom  (Space)")
            self._push_history()
            idx = self.combo_az_target.currentIndex()
            _, tx, ty = AUTO_ZOOM_TARGETS[idx]
            self._az_target_x = tx
            self._az_target_y = ty
            self.anim_timer.start(16)
        else:
            self.auto_zoom_btn.setText("Auto-Zoom  (Space)")
            self.anim_timer.stop()

    def _process_auto_zoom(self):
        gw = self.gl_widget
        # Adaptive smoothing: decelerate as we approach the target
        dist = math.hypot(gw.center_x - self._az_target_x, gw.center_y - self._az_target_y)
        # At high zoom, the distance in fractal space is small even when visually far
        zoom_factor = max(0.005, min(0.05, dist / max(1e-14, gw.zoom) * 0.5))
        smoothing = zoom_factor
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

        # Auto precision switch
        if (gw.zoom < 1e-6) != gw.use_double_precision:
            if gw.current_plugin:
                gw.load_plugin(gw.current_plugin)

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
        if not plugin or not hasattr(plugin, "params"):
            return
        if "c_real" not in plugin.params:
            return
        self._julia_angle = getattr(self, "_julia_angle", 0) + 0.02
        radius = 0.7885
        plugin.params["c_real"] = radius * math.cos(self._julia_angle)
        plugin.params["c_imag"] = radius * math.sin(self._julia_angle)
        # Use the fixed refresh_ui method instead of the broken stub
        plugin.refresh_ui()
        self.gl_widget.update()

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

        # Enable Julia anim checkbox only for plugins with c_real param
        is_julia_type = "c_real" in plugin.params
        self.julia_anim_cb.setEnabled(is_julia_type)
        if not is_julia_type and self.julia_anim_cb.isChecked():
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
            self._push_history()
            self.gl_widget.center_x = x
            self.gl_widget.center_y = y
            self.gl_widget.zoom = max(1e-14, z)
            if (z < 1e-6) != self.gl_widget.use_double_precision:
                if self.gl_widget.current_plugin:
                    self.gl_widget.load_plugin(self.gl_widget.current_plugin)
            self.gl_widget.update()
        except ValueError:
            self.statusBar().showMessage("Invalid coordinate format!", 3000)

    def _copy_coords(self):
        gw = self.gl_widget
        text = (f"x={gw.center_x:.15g}  "
                f"y={gw.center_y:.15g}  "
                f"zoom={gw.zoom:.15g}")
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Copied to clipboard.", 1500)

    # ------------------------------------------------------------- actions
    def _reset_view(self):
        self._push_history()
        if self.gl_widget.current_plugin:
            s = self.gl_widget.current_plugin.get_default_settings()
            self.gl_widget.apply_settings(s)
            self.iter_slider.setValue(s["max_iter"])
            self.gl_widget.update()

    def _save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "fractal.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)",
        )
        if file_path:
            self.gl_widget.grabFramebuffer().save(file_path)
            self.statusBar().showMessage(f"Saved: {file_path}", 3000)

    def _save_high_res(self):
        dlg = HighResExportDialog(self.gl_widget.width(), self.gl_widget.height(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        w, h = dlg.get_size()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save High-Resolution Image", f"fractal_{w}x{h}.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)",
        )
        if not file_path:
            return

        self.statusBar().showMessage(f"Rendering {w}x{h}...", 5000)

        # Temporarily resize, render, save, restore
        orig_size = self.gl_widget.size()
        self.gl_widget.setFixedSize(w, h)
        self.gl_widget.resize(w, h)
        QApplication.processEvents()

        self.gl_widget.grabFramebuffer().save(file_path)

        self.gl_widget.setMinimumSize(0, 0)
        self.gl_widget.setMaximumSize(16777215, 16777215)
        self.gl_widget.resize(orig_size)

        self.statusBar().showMessage(f"High-res saved: {file_path}", 5000)

    def _show_help(self):
        dlg = HelpDialog(self)
        dlg.exec()

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
        plugin_name = gw.current_plugin.name if gw.current_plugin else "?"
        self.statusBar().showMessage(
            f"FPS: {gw.fps:.0f}  |  {plugin_name}  |  "
            f"Zoom: {gw.zoom:.6g}  |  Prec: {prec}  |  "
            f"Center: ({gw.center_x:.12g}, {gw.center_y:.12g})  |  "
            f"Iter: {gw.max_iter}"
        )

    # --------------------------------------------------------- keyboard
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()

        if key == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key_F1:
            self._show_help()
        elif key == Qt.Key_R and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._reset_view()
        elif key == Qt.Key_Space:
            self.auto_zoom_btn.toggle()
        elif key == Qt.Key_C and not (mods & Qt.KeyboardModifier.ControlModifier):
            self._copy_coords()
        elif key == Qt.Key_S and mods & Qt.KeyboardModifier.ControlModifier:
            self._save_image()
        elif key == Qt.Key_Z and mods & Qt.KeyboardModifier.ControlModifier:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                self._zoom_redo()
            else:
                self._zoom_undo()
        elif key == Qt.Key_B and mods & Qt.KeyboardModifier.ControlModifier:
            self._save_bookmark()
        elif Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            names = list(self.plugins.keys())
            if idx < len(names):
                self.combo.setCurrentText(names[idx])
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Push history on zoom from the main window level."""
        self._push_history()
        super().wheelEvent(event)


# ======================================================= ENTRY POINT
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

    window = MainWindow()
    window.show()

    exit_code = app.exec()
    logger.info(f"Exiting with code {exit_code}")
    sys.exit(exit_code)
