from .base import BaseFractalPlugin


class JuliaPlugin(BaseFractalPlugin):
    name = "Julia Set"

    # Clean initialization: z starts at pixel, c is the uniform parameter
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
            {
                'name': 'c_real', 'type': 'float',
                'default': -0.7, 'min': -2.0, 'max': 2.0,
                'label': 'Real (c)', 'fmt': '.4f', 'scale': 10000,
            },
            {
                'name': 'c_imag', 'type': 'float',
                'default': 0.27015, 'min': -2.0, 'max': 2.0,
                'label': 'Imag (c)', 'fmt': '.4f', 'scale': 10000,
            },
        ]

    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 1.5, "max_iter": 200}