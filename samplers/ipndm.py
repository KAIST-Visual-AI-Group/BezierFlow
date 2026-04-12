import torch
from samplers.general_solver import ODESolver


class iPNDM(ODESolver):
    def __init__(self, noise_schedule, algorithm_type="noise_prediction",
                 prediction_type="eps", use_sb=False):
        super().__init__(noise_schedule, algorithm_type, prediction_type, use_sb)

    def sample_simple(self, model_fn, x, timesteps, timesteps2, order=2,
                      condition=None, unconditional_condition=None, **kwargs):
        self.model = lambda x, t1, t2: model_fn(
            x, t1.expand(x.shape[0]), t2.expand(x.shape[0]),
            condition, unconditional_condition
        )
        ns = self.noise_schedule
        steps = len(timesteps) - 1
        epsilon_buffer = []
        x_next = x

        for step in range(steps):
            step_order = min(order, step + 1)
            t_cur1, t_next1 = timesteps[step], timesteps[step + 1]
            t_cur2 = timesteps2[step]

            epsilon_cur = self.model_fn(x_next, t_cur1, t_cur2)

            lambda_s = ns.marginal_lambda(t_cur1)
            lambda_t = ns.marginal_lambda(t_next1)
            h = lambda_t - lambda_s
            log_alpha_s = ns.marginal_log_mean_coeff(t_cur1)
            log_alpha_t = ns.marginal_log_mean_coeff(t_next1)
            sigma_t = ns.marginal_std(t_next1)
            phi_1 = torch.expm1(h)

            if step_order == 1:
                x_next = torch.exp(log_alpha_t - log_alpha_s) * x_next - (sigma_t * phi_1) * epsilon_cur
            elif step_order == 2:
                x_next = (torch.exp(log_alpha_t - log_alpha_s) * x_next
                          - (sigma_t * phi_1) * (3 * epsilon_cur - epsilon_buffer[-1]) / 2)
            elif step_order == 3:
                x_next = (torch.exp(log_alpha_t - log_alpha_s) * x_next
                          - (sigma_t * phi_1) * (23 * epsilon_cur - 16 * epsilon_buffer[-1] + 5 * epsilon_buffer[-2]) / 12)
            elif step_order == 4:
                x_next = (torch.exp(log_alpha_t - log_alpha_s) * x_next
                          - (sigma_t * phi_1) * (55 * epsilon_cur - 59 * epsilon_buffer[-1] + 37 * epsilon_buffer[-2] - 9 * epsilon_buffer[-3]) / 24)

            if len(epsilon_buffer) == order - 1:
                for k in range(order - 2):
                    epsilon_buffer[k] = epsilon_buffer[k + 1]
                epsilon_buffer[-1] = epsilon_cur
            else:
                epsilon_buffer.append(epsilon_cur)

        return x_next
