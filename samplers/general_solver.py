"""
Base ODE solver class for BézierFlow.
Provides model prediction conversion (eps/x0/v/pf_ode) and timestep utilities.
"""

import torch
import os
from abc import abstractmethod
from noise_schedulers import NoiseScheduleVE


class ODESolver:
    def __init__(self, noise_schedule, algorithm_type="data_prediction",
                 prediction_type="x0", use_sb=False):
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction", "v_prediction"]
        self.algorithm_type = algorithm_type
        self.prediction_type = prediction_type
        self.use_sb = use_sb

    def _view_as_x(self, t_like, x):
        if not isinstance(t_like, torch.Tensor):
            t_like = torch.as_tensor(t_like, dtype=x.dtype, device=x.device)
        if t_like.ndim == 0:
            t_like = t_like.view(1)
        return t_like.view((t_like.shape[0],) + (1,) * (x.ndim - 1))

    def dx_dt_for_blackbox_solvers(self, x, t1, t2, model_output=None):
        """Return dx/dt under the target schedule."""
        eps_num = 1e-12

        if self.prediction_type in ("v", "pf_ode"):
            return self.model(x, t1, t2) if model_output is None else model_output

        alpha_t = self.noise_schedule.marginal_alpha(t1)
        sigma_t = self.noise_schedule.marginal_std(t1)

        if hasattr(self.noise_schedule, "ft"):
            f_t = self.noise_schedule.ft(t1)
        else:
            f_t = self.noise_schedule.dalpha_dt(t1) / alpha_t.clamp_min(eps_num)

        g_t = getattr(self.noise_schedule, "gt", lambda t: None)(t1) if hasattr(self.noise_schedule, "gt") else None

        alpha_t = self._view_as_x(alpha_t, x)
        sigma_t = self._view_as_x(sigma_t, x)
        f_t = self._view_as_x(f_t, x)

        if self.prediction_type == "eps":
            eps_pred = self.model(x, t1, t2) if model_output is None else model_output
        elif self.prediction_type == "x0":
            x0_pred = self.model(x, t1, t2) if model_output is None else model_output
            eps_pred = (x - alpha_t * x0_pred) / sigma_t
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if g_t is not None:
            g_t = self._view_as_x(g_t, x)
            drift = f_t * x + (g_t * g_t / (2.0 * sigma_t)) * eps_pred
        else:
            dalpha_dt = self._view_as_x(self.noise_schedule.dalpha_dt(t1), x)
            dsigma_dt = self._view_as_x(self.noise_schedule.dsigma_dt(t1), x)
            g2_over_2sigma = dsigma_dt - (dalpha_dt / alpha_t.clamp_min(eps_num)) * sigma_t
            drift = f_t * x + g2_over_2sigma * eps_pred
        return drift

    def noise_prediction_fn(self, x, t1, t2):
        """Convert model output to noise (eps) prediction."""
        alpha_t = self._view_as_x(self.noise_schedule.marginal_alpha(t1), x)
        sigma_t = self._view_as_x(self.noise_schedule.marginal_std(t1), x)
        dalpha_dt = self._view_as_x(self.noise_schedule.dalpha_dt(t1), x)
        dsigma_dt = self._view_as_x(self.noise_schedule.dsigma_dt(t1), x)

        y = self.model(x, t1, t2)

        if self.prediction_type == "eps":
            return y
        elif self.prediction_type == "x0":
            return (x - alpha_t * y) / sigma_t
        elif self.prediction_type == "v":
            D = alpha_t * dsigma_dt - dalpha_dt * sigma_t
            return (alpha_t * y - dalpha_dt * x) / D
        elif self.prediction_type == "pf_ode":
            if self.noise_schedule.drift_coeff is not None:
                f_t = self.noise_schedule.drift_coeff
                g_t_square = self.noise_schedule.eps_coeff
            else:
                f_t = self._view_as_x(self.noise_schedule.ft(t1), x)
                g_t_square = self._view_as_x(self.noise_schedule.gt_square(t1), x)
            return (2.0 * sigma_t / g_t_square) * (y - f_t * x)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def data_prediction_fn(self, x, t1, t2):
        """Convert model output to data (x0) prediction."""
        alpha_t = self._view_as_x(self.noise_schedule.marginal_alpha(t1), x)
        sigma_t = self._view_as_x(self.noise_schedule.marginal_std(t1), x)
        dalpha_dt = self._view_as_x(self.noise_schedule.dalpha_dt(t1), x)
        dsigma_dt = self._view_as_x(self.noise_schedule.dsigma_dt(t1), x)

        y = self.model(x, t1, t2)

        if self.prediction_type == "x0":
            return y
        elif self.prediction_type == "eps":
            return (x - sigma_t * y) / alpha_t
        elif self.prediction_type == "v":
            D = alpha_t * dsigma_dt - dalpha_dt * sigma_t
            return (dsigma_dt * x - sigma_t * y) / D
        elif self.prediction_type == "pf_ode":
            D = alpha_t * dsigma_dt - dalpha_dt * sigma_t
            return (dsigma_dt * x - sigma_t * y) / D
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def model_fn(self, x, t1, t2):
        if self.algorithm_type == "data_prediction":
            return self.data_prediction_fn(x, t1, t2)
        elif self.algorithm_type == "v_prediction":
            return self.dx_dt_for_blackbox_solvers(x, t1, t2)
        else:
            return self.noise_prediction_fn(x, t1, t2)

    def get_time_steps(self, skip_type, t_T, t_0, N, device, **kwargs):
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type in ('time_uniform', 'linear'):
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'edm':
            return self._get_time_step_edm(t_T, t_0, N, device)
        elif skip_type == 'rf':
            return torch.linspace(t_0, t_T, N + 1).to(device)
        else:
            raise ValueError(f"Unsupported skip_type: {skip_type}")

    def _get_time_step_edm(self, t_T, t_0, N, device, rho=7.0):
        if isinstance(self.noise_schedule, NoiseScheduleVE):
            sigma_min = self.noise_schedule.marginal_std(t_0).to(device)
            sigma_max = self.noise_schedule.marginal_std(t_T).to(device)
        else:
            sigma_min, sigma_max = t_0, t_T
        ramp = torch.linspace(0, 1, N + 1).to(device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        if isinstance(self.noise_schedule, NoiseScheduleVE):
            return self.noise_schedule.inverse_std(sigmas)
        return sigmas

    def prepare_learn_timesteps(self, load_from, device=None):
        states = torch.load(load_from, map_location=device)
        if states.get('best_t_steps') is not None:
            timesteps = states['best_t_steps'].to(device)
            length = timesteps.shape[0] // 2
            return timesteps[:length], timesteps[length:]
        return states['params1'].to(device), states['params2'].to(device)

    def prepare_timesteps(self, steps=None, t_start=None, t_end=None,
                          skip_type=None, device=None, load_from=None, **kwargs):
        if load_from is not None and os.path.isfile(load_from):
            return self.prepare_learn_timesteps(load_from=load_from, device=device)
        timesteps = self.get_time_steps(
            skip_type=skip_type, t_T=t_start, t_0=t_end, N=steps, device=device, **kwargs
        )
        return timesteps, timesteps

    @abstractmethod
    def sample_simple(self, model_fn, x, timesteps, timesteps2=None,
                      condition=None, unconditional_condition=None, **kwargs):
        pass
