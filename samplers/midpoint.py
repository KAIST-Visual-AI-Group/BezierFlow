import torch
from samplers.general_solver import ODESolver


class Midpoint(ODESolver):
    def __init__(self, noise_schedule, algorithm_type="v_prediction", prediction_type="v"):
        super().__init__(noise_schedule, algorithm_type, prediction_type)

    def sample_simple(self, model_fn, x, timesteps, timesteps2, NFEs=20,
                      condition=None, unconditional_condition=None, **kwargs):
        self.model = lambda x, t1, t2: model_fn(
            x, t1.expand(x.shape[0]), t2.expand(x.shape[0]),
            condition, unconditional_condition
        )
        x_next = x
        for step in range(NFEs):
            t_cur1, t_next1, t_cur2 = timesteps[step], timesteps[step + 1], timesteps2[step]
            h = t_next1 - t_cur1
            k1 = self.dx_dt_for_blackbox_solvers(x_next, t_cur1, t_cur2)
            x_mid = x_next + 0.5 * h * k1
            t_mid = t_cur1 + 0.5 * h
            k2 = self.dx_dt_for_blackbox_solvers(x_mid, t_mid, t_mid)
            x_next = x_next + h * k2
        return x_next
