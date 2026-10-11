import sys
import logging
import os
import math

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat, QPalette, QColor
from PySide6.QtCore import Qt

from main_window import MainWindow

APP_VERSION = "2.0.0"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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


# --- Palette Generation ---

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

    # 1. Classic Rainbow
    _write_palette("default.map", _palette_from_stops([
        (0.00, (0, 7, 100)),  (0.16, (32, 107, 203)),
        (0.33, (49, 210, 207)), (0.50, (30, 225, 60)),
        (0.66, (230, 225, 35)), (0.83, (230, 80, 20)),
        (1.00, (120, 10, 10)),
    ]))

    # 2. Fire
    _write_palette("fire.map", _palette_from_stops([
        (0.0, (0, 0, 0)),    (0.2, (80, 0, 0)),
        (0.4, (200, 30, 0)), (0.6, (255, 128, 0)),
        (0.8, (255, 230, 50)), (1.0, (255, 255, 220)),
    ]))

    # 3. Ocean
    _write_palette("ocean.map", _palette_from_stops([
        (0.0, (0, 5, 30)),     (0.25, (0, 30, 100)),
        (0.50, (0, 100, 180)), (0.75, (50, 200, 230)),
        (1.0, (200, 240, 255)),
    ]))

    # 4. Electric
    _write_palette("electric.map", _palette_from_stops([
        (0.0, (0, 0, 20)),    (0.25, (20, 0, 120)),
        (0.50, (80, 0, 255)), (0.75, (0, 180, 255)),
        (1.0, (200, 255, 255)),
    ]))

    # 5. Forest
    _write_palette("forest.map", _palette_from_stops([
        (0.0, (5, 15, 5)),    (0.25, (20, 70, 20)),
        (0.50, (40, 150, 40)), (0.75, (140, 200, 60)),
        (1.0, (220, 240, 150)),
    ]))

    # 6. Sunset
    _write_palette("sunset.map", _palette_from_stops([
        (0.0, (20, 0, 50)),   (0.25, (120, 10, 80)),
        (0.50, (220, 60, 30)), (0.75, (255, 170, 50)),
        (1.0, (255, 240, 150)),
    ]))

    # 7. Grayscale
    _write_palette("grayscale.map", _palette_from_stops([
        (0.0, (0, 0, 0)), (1.0, (255, 255, 255)),
    ]))

    # 8. Neon
    _write_palette("neon.map", _palette_from_stops([
        (0.0, (0, 0, 0)),     (0.20, (255, 0, 100)),
        (0.40, (0, 255, 200)), (0.60, (0, 100, 255)),
        (0.80, (255, 255, 0)), (1.0, (255, 0, 255)),
    ]))

    # 9. Ice
    _write_palette("ice.map", _palette_from_stops([
        (0.0, (10, 10, 40)),   (0.30, (30, 60, 150)),
        (0.60, (100, 180, 240)), (0.85, (200, 230, 255)),
        (1.0, (255, 255, 255)),
    ]))

    # 10. Magma (matplotlib-inspired)
    _write_palette("magma.map", _palette_from_stops([
        (0.0, (0, 0, 4)),      (0.25, (80, 18, 123)),
        (0.50, (183, 55, 121)), (0.75, (254, 136, 76)),
        (1.0, (252, 253, 191)),
    ]))

    # 11. Viridis (matplotlib-inspired)
    _write_palette("viridis.map", _palette_from_stops([
        (0.0, (68, 1, 84)),    (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)), (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]))


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