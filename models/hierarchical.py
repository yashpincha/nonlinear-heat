import numpy as np
import pymc as pm
from .base import BaseModel
import pytensor.tensor as pt

class UnifiedHierarchical(BaseModel):
    def __init__(self, config, forward_model=None):
        super().__init__(config)
        self.forward_model = forward_model

    def forward(self, x, t, omega, params):
        if self.forward_model is None:
            raise ValueError("No forward model specified. Pass forward_model to __init__")

        params_with_omega = params.copy()
        params_with_omega['w'] = omega

        N_sensors = len(x) if isinstance(x, np.ndarray) else 1
        N_time = len(t)

        if N_sensors == 1:
            return self.forward_model.forward(x, t, params_with_omega)
        else:
            predictions = np.zeros((N_sensors, N_time))
            for i, x_i in enumerate(x):
                predictions[i, :] = self.forward_model.forward(x_i, t, params_with_omega)
            return predictions

    def build_model(self, datasets, x_sensors, L=0.046, sigma=None, model_type='FullSolutionNewton'):
        N_freqs = len(datasets)
        N_sensors = len(x_sensors)

        with pm.Model() as model:
            D = pm.Uniform("D", lower=1e-6, upper=5e-4)

            Tbar = pm.Normal("Tbar", mu=25.0, sigma=5.0, shape=N_freqs)
            A = pm.Uniform("A", lower=0.1, upper=20.0, shape=N_freqs)
            phi = pm.Uniform("phi", lower=-np.pi, upper=np.pi, shape=N_freqs)

            if model_type in ['FullSolutionNewton', 'FullSolutionRobin']:
                gamma = pm.Uniform("gamma", lower=0.0, upper=1.0)

            if model_type == 'FullSolutionRobin':
                H = pm.Uniform("H", lower=0.0, upper=10.0)

            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=1.0, shape=N_sensors)

            for f_idx, data in enumerate(datasets):
                t_f = data['t']
                temp_f = data['temp']
                omega_f = data['omega']

                A_f = A[f_idx]
                phi_f = phi[f_idx]
                Tbar_f = Tbar[f_idx]

                if model_type == 'SimpleForward':
                    beta = pm.math.sqrt(omega_f / (2 * D))
                    for i, x_i in enumerate(x_sensors):
                        mu = Tbar_f + A_f * pm.math.exp(-beta * x_i) * pm.math.cos(omega_f * t_f - beta * x_i + phi_f)
                        pm.Normal(f"obs_f{f_idx}_s{i}", mu=mu, sigma=sigma[i], observed=temp_f[i, :])

                elif model_type == 'FullSolutionNewton':
                    r = pm.math.sqrt(gamma**2 + omega_f**2)
                    theta = pt.arctan2(omega_f, gamma)
                    k_mag = pm.math.sqrt(r / D)
                    k_real = k_mag * pm.math.cos(theta / 2)
                    k_imag = k_mag * pm.math.sin(theta / 2)

                    for i, x_i in enumerate(x_sensors):
                        Lx = L - x_i
                        cosh_kLx_real = pm.math.cosh(k_real * Lx) * pm.math.cos(k_imag * Lx)
                        cosh_kLx_imag = pm.math.sinh(k_real * Lx) * pm.math.sin(k_imag * Lx)
                        cosh_kL_real = pm.math.cosh(k_real * L) * pm.math.cos(k_imag * L)
                        cosh_kL_imag = pm.math.sinh(k_real * L) * pm.math.sin(k_imag * L)
                        denom = cosh_kL_real**2 + cosh_kL_imag**2
                        ratio_real = (cosh_kLx_real * cosh_kL_real + cosh_kLx_imag * cosh_kL_imag) / denom
                        ratio_imag = (cosh_kLx_imag * cosh_kL_real - cosh_kLx_real * cosh_kL_imag) / denom
                        exp_real = pm.math.cos(omega_f * t_f + phi_f)
                        exp_imag = pm.math.sin(omega_f * t_f + phi_f)
                        wave_real = ratio_real * exp_real - ratio_imag * exp_imag
                        mu = Tbar_f + A_f * wave_real
                        pm.Normal(f"obs_f{f_idx}_s{i}", mu=mu, sigma=sigma[i], observed=temp_f[i, :])

        return model
