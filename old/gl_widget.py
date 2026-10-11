import time
import math
import numpy as np
import logging
import os

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QMutex, QMutexLocker
from PySide6.QtGui import QSurfaceFormat, QVector2D
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLShader, QOpenGLVertexArrayObject

logger = logging.getLogger(__name__)


class FractalGLWidget(QOpenGLWidget):
    """Advanced OpenGL fractal renderer with dynamic shaders, double precision, and palette textures."""

    # OpenGL constant aliases for readability
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

        # --- Render State ---
        self.time = 0.0
        self.zoom = 2.5
        self.center_x = -0.5
        self.center_y = 0.0
        self.max_iter = 200
        self.animation_paused = False
        self.color_offset = 0.0

        # --- Plugin System ---
        self.current_plugin = None
        self.program = None
        self.vbo = None
        self.vao = None
        self.gl = None

        # --- Interaction ---
        self.last_mouse_pos = None

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
            logger.info(f"Precision → {'Double' if new_dp else 'Single'}")
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

            # Vertex shader
            if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, plugin.vertex_shader):
                logger.error(f"VERTEX error: {self.program.log()}")
                self.program.deleteLater()
                self.program = None
                self.doneCurrent()
                return

            # Fragment shader
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
            # FPS
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

    # -------------------------------------------------------------- animation
    def _tick(self):
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now

        # FPS tracking
        self.fps_acc += dt
        self.frame_count_since = getattr(self, '_fc', 0) + 1
        self._fc = self.frame_count_since
        if self.fps_acc >= 1.0:
            self.fps = self._fc / self.fps_acc
            self._fc = 0
            self.fps_acc = 0.0

        if not self.animation_paused:
            self.time += dt * 1.2
        self.update()

    # ------------------------------------------------------------- input utils
    def _screen_to_fractal(self, sx, sy):
        """Convert screen pixel coords → fractal complex coords."""
        aspect = self.width() / max(1, self.height())
        fx = self.center_x + (sx - self.width() / 2.0) / self.height() * self.zoom
        fy = self.center_y - (sy - self.height() / 2.0) / self.height() * self.zoom
        return fx, fy

    # ----------------------------------------------------------------- wheel
    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+scroll adjusts iterations
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
            # Middle-click centers on cursor
            fx, fy = self._screen_to_fractal(event.position().x(), event.position().y())
            self.center_x = fx
            self.center_y = fy
            self.update()

    def mouseMoveEvent(self, event):
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