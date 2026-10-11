from abc import ABC, abstractmethod
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QVector2D


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
        """GLSL initialization code. Override for Julia-type fractals.
        Defaults to Mandelbrot-style: z starts at 0, c = pixel position."""
        return """
            VEC2 z = VEC2(0.0);
            VEC2 c = c_pos;
        """

    vertex_shader = """
    #version 410 core
    layout (location = 0) in vec2 a_position;
    void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
    """

    _coloring_footer = """
        if (iter >= float(u_max_iter)) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            // Smooth iteration count
            float log_zn = log(dot(z, z)) / 2.0;
            float nu = log(log_zn / log(2.0)) / log(2.0);
            iter = iter + 1.0 - nu;

            // Cyclic palette mapping — repeats ~40 times for rich detail
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
        """Override to define plugin-specific parameters."""
        return []

    def get_fragment_shader(self, use_double=False):
        # --- Header with precision-dependent types ---
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

        # Shared uniforms (u_resolution is always vec2 — screen coords)
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
        """Build parameter sliders for this plugin."""
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

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

            elif p['type'] == 'int':
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(p['min'], p['max'])
                slider.setValue(self.params[p['name']])
                slider.valueChanged.connect(
                    lambda v, nm=p['name'], vl=val_lbl:
                        self._update_param(nm, v, vl, 'd')
                )
                row.addWidget(slider, stretch=1)

            row.addWidget(val_lbl)
            layout.addLayout(row)
        return widget

    def _update_param(self, name, value, value_label, fmt):
        self.params[name] = value
        value_label.setText(f"{value:{fmt}}")

    def get_default_settings(self):
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 200}

    def update_uniforms(self, program, gl, time, width, height,
                        center_x, center_y, zoom, max_iter, color_offset):
        """Push all uniforms to the GPU, handling double precision automatically."""
        # Screen resolution (always float)
        program.setUniformValue("u_resolution", QVector2D(width, height))

        # Standard controls
        program.setUniformValue("u_max_iter", int(max_iter))
        program.setUniformValue("u_color_offset", float(color_offset))

        # Fractal coords — use raw GL for double precision
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

        # Plugin-specific parameters
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