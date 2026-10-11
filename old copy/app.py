import sys
import time
import numpy as np
import logging

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, 
                             QComboBox, QFileDialog,
                             QStatusBar, QGroupBox, QCheckBox, QLineEdit,
                             QDoubleSpinBox, QFormLayout)
from PySide6.QtOpenGLWidgets import QOpenGLWidget 
from PySide6.QtCore import Qt, QTimer, QMutex, QMutexLocker
from PySide6.QtGui import QSurfaceFormat, QVector2D, QKeyEvent
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLShader, QOpenGLVertexArrayObject

import backend

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class FractalGLWidget(QOpenGLWidget):
    # Define GL constants manually for PySide6 compatibility
    GL_COLOR_BUFFER_BIT = 0x00004000
    GL_TRIANGLE_STRIP = 0x0005
    GL_FLOAT = 0x1406

    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = 0.0
        self.zoom = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.max_iter = 100
        self.animation_paused = False
        self.color_offset = 0.0
        
        self.current_plugin = None
        self.program = None
        self.vbo = None
        self.vao = None
        self.gl = None
        
        self.last_mouse_pos = None
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Mutex to prevent rendering while loading
        self.resource_mutex = QMutex()

    def cleanup(self):
        """Properly release OpenGL resources before context destruction."""
        # Prevent cleanup if context is already invalid or non-existent
        if not self.context():
            return
            
        logger.info("Cleaning up OpenGL resources...")
        
        # Make the context current so we can delete OpenGL resources
        self.makeCurrent()
        
        # Safely destroy VAO
        if self.vao:
            self.vao.destroy()
            self.vao = None
            
        # Safely destroy VBO
        if self.vbo:
            self.vbo.destroy()
            self.vbo = None
            
        # Safely destroy Shader Program
        if self.program:
            self.program.release()
            self.program.removeAllShaders()
            self.program.deleteLater()
            self.program = None
            
        self.doneCurrent()
        logger.info("Cleanup complete.")

    def _setup_geometry(self):
        """Helper to create VBO."""
        if not self.vbo:
            vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            self.vbo = QOpenGLBuffer()
            if not self.vbo.create():
                logger.error("Failed to create VBO")
                return
            self.vbo.bind()
            self.vbo.allocate(vertices.tobytes(), vertices.nbytes)
            self.vbo.release()

    def load_plugin(self, plugin):
        # We must have a valid GL context to load shaders
        if not self.isValid() or not self.context():
            logger.warning("Attempted to load plugin without valid GL context")
            return

        logger.info(f"Loading plugin: {plugin.name}...")
        
        # Lock resources to prevent paintGL from using them
        with QMutexLocker(self.resource_mutex):
            self.makeCurrent()
            
            # Ensure GL functions are loaded
            if not self.gl:
                self.gl = self.context().extraFunctions()
                if not self.gl:
                    logger.critical("Failed to resolve GL functions.")
                    self.doneCurrent()
                    return

            # 1. Cleanup previous program safely
            if self.program:
                self.gl.glUseProgram(0)
                self.program.removeAllShaders()
                self.program.release()
                self.program.deleteLater()
                self.program = None

            self.program = QOpenGLShaderProgram(self)

            # 2. Compile Vertex Shader
            vs_success = self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, plugin.vertex_shader)
            if not vs_success:
                self._handle_shader_error(plugin, "VERTEX")
                self.doneCurrent()
                return

            # 3. Compile Fragment Shader
            fs_success = self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, plugin.fragment_shader)
            if not fs_success:
                self._handle_shader_error(plugin, "FRAGMENT")
                self.doneCurrent()
                return
            
            # Bind attributes before linking
            self.program.bindAttributeLocation("a_position", 0)
            
            # 4. Link Program
            if not self.program.link():
                error_log = self.program.log()
                logger.error(f"Shader Linking Failed: {error_log}")
                self.program.deleteLater()
                self.program = None
                self.doneCurrent()
                return

            logger.info(f"Shader program linked successfully for {plugin.name}")

            # Setup Geometry (creates VBO if needed)
            self._setup_geometry()

            # Setup VAO
            if not self.vao:
                self.vao = QOpenGLVertexArrayObject()
                if not self.vao.create():
                    logger.error("Failed to create VAO")
                    self.doneCurrent()
                    return
            
            self.vao.bind()
            if self.vbo:
                self.vbo.bind()
            self.program.enableAttributeArray(0)
            self.program.setAttributeBuffer(0, self.GL_FLOAT, 0, 2)
            self.vao.release()
            if self.vbo:
                self.vbo.release()
            
            self.doneCurrent()
            self.current_plugin = plugin
            self.apply_settings(plugin.get_default_settings())
            self.update()

    def _handle_shader_error(self, plugin, shader_type):
        error_log = self.program.log()
        logger.error("="*40)
        logger.error(f"{shader_type} SHADER COMPILATION FAILED")
        logger.error(f"Plugin: {plugin.name}")
        logger.error(f"Log: {error_log if error_log else 'No log available'}")
        logger.error("Source Code:")
        src = plugin.vertex_shader if shader_type == "VERTEX" else plugin.fragment_shader
        for i, line in enumerate(src.split('\n')):
            print(f"{i+1:03d}: {line}")
        logger.error("="*40)
        self.program.deleteLater()
        self.program = None

    def apply_settings(self, settings):
        self.center_x = settings.get("center_x", 0.0)
        self.center_y = settings.get("center_y", 0.0)
        self.zoom = settings.get("zoom", 1.0)
        self.max_iter = settings.get("max_iter", 100)

    def initializeGL(self):
        # Connect cleanup signal immediately when context is ready
        self.context().aboutToBeDestroyed.connect(self.cleanup)
        
        self.gl = self.context().extraFunctions()
        if not self.gl: 
            logger.critical("Failed to get OpenGL functions context.")
            return
        
        logger.info("OpenGL Context Initialized")
        try:
            version = self.gl.glGetString(self.gl.GL_VERSION)
            renderer = self.gl.glGetString(self.gl.GL_RENDERER)
            logger.info(f"GPU: {renderer}")
            logger.info(f"OpenGL Version: {version}")
        except Exception:
            logger.warning("Could not retrieve GL strings.")
        
        self.gl.glClearColor(0.02, 0.02, 0.05, 1.0)
        self._setup_geometry()

    def paintGL(self):
        # Thread safety: Don't paint if resources are being swapped or invalid
        with QMutexLocker(self.resource_mutex):
            if not self.current_plugin or not self.program or not self.gl:
                return

            # Double check program is linked
            if not self.program.isLinked():
                return

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            self.gl.glClear(self.GL_COLOR_BUFFER_BIT)
            
            self.program.bind()
            
            if self.vao:
                self.vao.bind()
            
            dpr = self.devicePixelRatioF()
            width_phys = self.width() * dpr
            height_phys = self.height() * dpr

            try:
                self.current_plugin.update_uniforms(
                    self.program, 
                    self.time, 
                    width_phys, height_phys, 
                    self.center_x, self.center_y, 
                    self.zoom, self.max_iter,
                    self.color_offset
                )

                self.gl.glDrawArrays(self.GL_TRIANGLE_STRIP, 0, 4)
            except Exception as e:
                logger.error(f"Runtime Error in paintGL: {e}")
            
            if self.vao:
                self.vao.release()
                
            self.program.release()

    def update_animation(self):
        if not self.animation_paused:
            self.time += 0.02
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.0 - (delta / 1200.0)
        self.zoom *= factor
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            scale = self.zoom / self.height()
            self.center_x -= dx * scale
            self.center_y += dy * scale
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU Fractal Engine - Advanced")
        self.resize(1280, 800)
        
        self.setStatusBar(QStatusBar(self))
        
        self.plugins = {cls.name: cls() for cls in backend.AVAILABLE_PLUGINS}
        
        self.target_x = -0.743643887037151
        self.target_y = 0.131825904205330

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.gl_widget = FractalGLWidget()
        main_layout.addWidget(self.gl_widget, stretch=3)

        sidebar = QWidget()
        sidebar.setMaximumWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # --- Navigator Group ---
        nav_group = QGroupBox("Navigator")
        nav_layout = QFormLayout(nav_group)
        
        self.input_x = QDoubleSpinBox()
        self.input_x.setDecimals(15)
        self.input_x.setRange(-5.0, 5.0)
        self.input_x.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        
        self.input_y = QDoubleSpinBox()
        self.input_y.setDecimals(15)
        self.input_y.setRange(-5.0, 5.0)
        self.input_y.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        
        self.input_zoom = QDoubleSpinBox()
        self.input_zoom.setDecimals(8)
        self.input_zoom.setRange(0.0000001, 100.0)
        self.input_zoom.setValue(1.0)
        
        btn_go = QPushButton("Go To Coordinates")
        btn_go.clicked.connect(self.apply_manual_coords)
        
        nav_layout.addRow("Center X:", self.input_x)
        nav_layout.addRow("Center Y:", self.input_y)
        nav_layout.addRow("Zoom:", self.input_zoom)
        nav_layout.addRow(btn_go)
        sidebar_layout.addWidget(nav_group)

        # --- Animation Group ---
        anim_group = QGroupBox("Animation")
        anim_layout = QVBoxLayout(anim_group)
        
        self.auto_zoom_btn = QPushButton("Start Auto-Zoom (Space)")
        self.auto_zoom_btn.setCheckable(True)
        self.auto_zoom_btn.toggled.connect(self.toggle_auto_zoom)
        anim_layout.addWidget(self.auto_zoom_btn)

        self.pause_time_cb = QCheckBox("Pause Color Animation")
        self.pause_time_cb.toggled.connect(lambda s: setattr(self.gl_widget, 'animation_paused', s))
        anim_layout.addWidget(self.pause_time_cb)
        
        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("Palette:"))
        self.color_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_slider.setRange(0, 100)
        self.color_slider.valueChanged.connect(lambda v: setattr(self.gl_widget, 'color_offset', v / 10.0))
        c_layout.addWidget(self.color_slider)
        anim_layout.addLayout(c_layout)
        
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Zoom Speed:"))
        self.zoom_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_speed_slider.setRange(90, 99)
        self.zoom_speed_slider.setValue(97)
        z_layout.addWidget(self.zoom_speed_slider)
        anim_layout.addLayout(z_layout)
        sidebar_layout.addWidget(anim_group)

        # --- Engine Group ---
        engine_group = QGroupBox("Engine Controls")
        engine_layout = QVBoxLayout(engine_group)

        self.combo = QComboBox()
        self.combo.addItems(self.plugins.keys())
        self.combo.currentTextChanged.connect(self.switch_plugin)
        engine_layout.addWidget(QLabel("Active Fractal:"))
        engine_layout.addWidget(self.combo)

        h_iter = QHBoxLayout()
        h_iter.addWidget(QLabel("Max Iter:"))
        self.iter_label = QLabel("150")
        h_iter.addWidget(self.iter_label)
        engine_layout.addLayout(h_iter)
        
        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(50, 2000)
        self.iter_slider.valueChanged.connect(self.update_iterations)
        engine_layout.addWidget(self.iter_slider)
        
        reset_btn = QPushButton("Reset View (R)")
        reset_btn.clicked.connect(self.reset_view)
        engine_layout.addWidget(reset_btn)
        
        export_btn = QPushButton("Save Image")
        export_btn.clicked.connect(self.save_image)
        engine_layout.addWidget(export_btn)
        
        sidebar_layout.addWidget(engine_group)

        # --- Shader Parameters ---
        shader_group = QGroupBox("Shader Parameters")
        shader_group_layout = QVBoxLayout(shader_group)
        self.plugin_ui_container = QVBoxLayout()
        shader_group_layout.addLayout(self.plugin_ui_container)
        sidebar_layout.addWidget(shader_group)
        
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar, stretch=1)

        # --- Timers ---
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.process_auto_zoom)
        
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)

        # Initialize first plugin
        if self.plugins:
            first_plugin_name = list(self.plugins.keys())[0]
            self.combo.setCurrentText(first_plugin_name)
            # Delay loading slightly to ensure context is active
            QTimer.singleShot(100, lambda: self.switch_plugin(first_plugin_name))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_F11 or (event.key() == Qt.Key_F and event.modifiers() == Qt.ControlModifier):
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_R:
            self.reset_view()
        elif event.key() == Qt.Key_Space:
            self.auto_zoom_btn.toggle()
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def apply_manual_coords(self):
        self.gl_widget.center_x = self.input_x.value()
        self.gl_widget.center_y = self.input_y.value()
        self.gl_widget.zoom = self.input_zoom.value()
        self.gl_widget.update()

    def toggle_auto_zoom(self, checked):
        if checked:
            self.auto_zoom_btn.setText("Stop Auto-Zoom (Space)")
            self.anim_timer.start(16)
        else:
            self.auto_zoom_btn.setText("Start Auto-Zoom (Space)")
            self.anim_timer.stop()

    def process_auto_zoom(self):
        smoothing = 0.02
        self.gl_widget.center_x += (self.target_x - self.gl_widget.center_x) * smoothing
        self.gl_widget.center_y += (self.target_y - self.gl_widget.center_y) * smoothing
        
        factor = self.zoom_speed_slider.value() / 100.0
        self.gl_widget.zoom *= factor
        
        if self.gl_widget.zoom < 0.0001:
            self.gl_widget.max_iter = min(2000, self.gl_widget.max_iter + 2)
            self.iter_slider.blockSignals(True)
            self.iter_slider.setValue(self.gl_widget.max_iter)
            self.iter_slider.blockSignals(False)
            self.iter_label.setText(str(self.gl_widget.max_iter))

        self.gl_widget.update()

    def switch_plugin(self, name):
        plugin = self.plugins.get(name)
        if not plugin: return

        logger.info(f"Switching UI to plugin: {name}")

        # Clear old UI safely
        for i in reversed(range(self.plugin_ui_container.count())): 
            w = self.plugin_ui_container.itemAt(i).widget()
            if w: 
                w.setParent(None)
                w.deleteLater()

        self.gl_widget.load_plugin(plugin)
        
        # Only update UI elements if plugin loaded correctly
        if self.gl_widget.program:
            defaults = plugin.get_default_settings()
            
            self.iter_slider.blockSignals(True)
            self.iter_slider.setValue(defaults["max_iter"])
            self.iter_label.setText(str(defaults["max_iter"]))
            self.iter_slider.blockSignals(False)
            
            ui_widget = plugin.setup_ui(self)
            self.plugin_ui_container.addWidget(ui_widget)
        else:
            logger.warning(f"Plugin {name} failed to load. UI not updated.")

    def update_iterations(self, val):
        self.gl_widget.max_iter = val
        self.iter_label.setText(str(val))
        
    def reset_view(self):
        if self.gl_widget.current_plugin:
            settings = self.gl_widget.current_plugin.get_default_settings()
            self.gl_widget.apply_settings(settings)
            
            self.iter_slider.blockSignals(True)
            self.iter_slider.setValue(settings["max_iter"])
            self.iter_label.setText(str(settings["max_iter"]))
            self.iter_slider.blockSignals(False)
            
            self.gl_widget.update()
        
    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "fractal.png", "PNG Files (*.png)")
        if file_path:
            self.gl_widget.grabFramebuffer().save(file_path)
            self.statusBar().showMessage(f"Saved to {file_path}", 3000)
            logger.info(f"Image saved to {file_path}")

    def update_status(self):
        x = self.gl_widget.center_x
        y = self.gl_widget.center_y
        z = self.gl_widget.zoom
        fps = self.gl_widget.fps
        
        self.input_x.blockSignals(True)
        self.input_y.blockSignals(True)
        self.input_zoom.blockSignals(True)
        
        self.input_x.setValue(x)
        self.input_y.setValue(y)
        self.input_zoom.setValue(z)
        
        self.input_x.blockSignals(False)
        self.input_y.blockSignals(False)
        self.input_zoom.blockSignals(False)
        
        self.statusBar().showMessage(f"FPS: {fps:.1f} | [F11: Fullscreen] | [Mouse: Pan/Zoom]")

if __name__ == "__main__":
    # Request OpenGL 4.1 Core Profile
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSamples(4) 
    QSurfaceFormat.setDefaultFormat(fmt)
    
    logger.info("Starting GPU Fractal Engine...")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    exit_code = app.exec()
    logger.info(f"Exiting with code {exit_code}")
    sys.exit(exit_code)