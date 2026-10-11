from .base import BaseFractalPlugin


class MultibrotPlugin(BaseFractalPlugin):
    name = "Multibrot Set"

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
            {
                'name': 'power', 'type': 'float',
                'default': 3.0, 'min': 2.0, 'max': 10.0,
                'label': 'Power', 'fmt': '.2f', 'scale': 100,
            },
        ]

    def get_default_settings(self):
        return {"center_x": -0.5, "center_y": 0.0, "zoom": 2.5, "max_iter": 150}