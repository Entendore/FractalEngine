from .base import BaseFractalPlugin


class BurningShipPlugin(BaseFractalPlugin):
    name = "Burning Ship"

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