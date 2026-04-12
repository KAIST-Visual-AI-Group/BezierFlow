import numpy as np
import torch
from scipy import integrate
from samplers.general_solver import ODESolver


class RK45(ODESolver):
    """Black-box adaptive ODE solver using scipy RK45."""

    def __init__(self, noise_schedule, algorithm_type="v_prediction", prediction_type="v"):
        super().__init__(noise_schedule, algorithm_type, prediction_type)

    @torch.no_grad()
    def sample_simple(self, model_fn, x, timesteps, timesteps2, NFEs=20,
                      condition=None, unconditional_condition=None, **kwargs):
        device, dtype, shape = x.device, x.dtype, x.shape

        def wrapped_model_fn(x_, t_vec):
            return model_fn(x_, t_vec, t_vec, condition, unconditional_condition)

        self.model = lambda x_, t_: wrapped_model_fn(x_, t_.expand(x_.shape[0]))

        def _to_flat_np(tensor):
            return tensor.detach().float().cpu().reshape(-1).numpy()

        def _from_flat_np(arr):
            return torch.from_numpy(arr).to(device=device, dtype=torch.float32).view(shape)

        def ode_func(t_scalar, y_flat):
            x_state = _from_flat_np(y_flat)
            t_vec = torch.full((shape[0],), t_scalar, device=device, dtype=dtype)
            drift = self.model(x_state, t_vec)
            return _to_flat_np(drift)

        t_start = timesteps[0].item()
        t_end = timesteps[-1].item()

        sol = integrate.solve_ivp(
            ode_func, (t_start, t_end), _to_flat_np(x),
            method="RK45",
            rtol=kwargs.get("rtol", 1e-6),
            atol=kwargs.get("atol", 1e-6),
        )

        save_traj = kwargs.get("save_traj", False)
        x_T = _from_flat_np(sol.y[:, -1]).to(dtype=dtype)

        if save_traj:
            y_steps = torch.from_numpy(sol.y.T.copy()).to(device=device, dtype=torch.float32)
            y_steps = y_steps.view(y_steps.shape[0], *shape).to(dtype=dtype)
            t_steps = torch.tensor(sol.t, device=device, dtype=torch.float32)
            return x_T, {"t": t_steps, "y": y_steps}

        return x_T
