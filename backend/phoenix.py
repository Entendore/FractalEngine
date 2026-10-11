from .base import BaseFractalPlugin


class PhoenixPlugin(BaseFractalPlugin):
    name = "Phoenix"

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
            {
                'name': 'c_real', 'type': 'float',
                'default': -0.7, 'min': -2.0, 'max': 2.0,
                'label': 'Real (c)', 'fmt': '.3f', 'scale': 1000,
            },
            {
                'name': 'c_imag', 'type': 'float',
                'default': 0.27015, 'min': -2.0, 'max': 2.0,
                'label': 'Imag (c)', 'fmt': '.3f', 'scale': 1000,
            },
            {
                'name': 'phoenix_p', 'type': 'float',
                'default': 0.5, 'min': -1.5, 'max': 1.5,
                'label': 'Phoenix P', 'fmt': '.3f', 'scale': 1000,
            },
        ]

    def get_default_settings(self):
        return {"center_x": 0.0, "center_y": 0.0, "zoom": 1.8, "max_iter": 150}