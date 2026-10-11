from .base import BaseFractalPlugin
from .mandelbrot import MandelbrotPlugin
from .julia import JuliaPlugin
from .burningship import BurningShipPlugin
from .tricorn import TricornPlugin
from .multibrot import MultibrotPlugin
from .phoenix import PhoenixPlugin

AVAILABLE_PLUGINS = [
    MandelbrotPlugin,
    JuliaPlugin,
    BurningShipPlugin,
    TricornPlugin,
    MultibrotPlugin,
    PhoenixPlugin,
]