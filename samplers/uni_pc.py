import torch
from samplers.general_solver import ODESolver


def einsum_float_double(string, a, b):
    return torch.einsum(string, a.double(), b.double()).float()


class UniPC(ODESolver):
    def __init__(self, noise_schedule, algorithm_type="data_prediction",
                 prediction_type="x0", use_sb=False, variant='bh1'):
        super().__init__(noise_schedule, algorithm_type, prediction_type, use_sb)
        self.variant = variant
        self.predict_x0 = (algorithm_type == "data_prediction")

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list,
                                    t, t2, order, x_t=None, use_corrector=True):
        if len(t.shape) == 0:
            t = t.view(-1)
            t2 = t2.view(-1)
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0 = ns.marginal_log_mean_coeff(t_prev_0)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)
        h = lambda_t - lambda_prev_0

        rks, D1s = [], []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R, b = [], []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        factorial_i = 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.cat(b)

        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if x_t is None:
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0

        if x_t is None:
            if use_predictor:
                if D1s.ndim == 4:
                    pred_res = einsum_float_double('k,bkcn->bcn', rhos_p, D1s)
                elif D1s.ndim == 5:
                    pred_res = einsum_float_double('k,bkchw->bchw', rhos_p, D1s)
                else:
                    raise ValueError(f"Invalid D1s ndim: {D1s.ndim}")
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res

        if use_corrector:
            model_t = self.model_fn(x_t, t, t2)
            if D1s is not None:
                if D1s.ndim == 4:
                    corr_res = einsum_float_double('k,bkcn->bcn', rhos_c[:-1], D1s)
                elif D1s.ndim == 5:
                    corr_res = einsum_float_double('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    raise ValueError(f"Invalid D1s ndim: {D1s.ndim}")
            else:
                corr_res = 0
            D1_t = model_t - model_prev_0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def _update_lists(self, t_list, model_list, t_, model_x, order, first=False):
        if first:
            t_list.append(t_)
            model_list.append(model_x)
            return
        for m in range(order - 1):
            t_list[m] = t_list[m + 1]
            model_list[m] = model_list[m + 1]
        t_list[-1] = t_
        model_list[-1] = model_x

    def sample_simple(self, model_fn, x, timesteps, timesteps2, order=2,
                      lower_order_final=True, condition=None,
                      unconditional_condition=None, **kwargs):
        self.model = lambda x, t1, t2: model_fn(
            x, t1.expand(x.shape[0]), t2.expand(x.shape[0]),
            condition, unconditional_condition
        )
        steps = len(timesteps) - 1
        t_prev_list = [timesteps[0]]
        model_prev_list = [self.model_fn(x, timesteps[0], timesteps2[0])]

        if steps <= 2:
            order = 1
        elif steps == 3:
            order = 2

        for step in range(1, order):
            t1, t2_ = timesteps[step], timesteps2[step]
            x, model_x = self.multistep_uni_pc_bh_update(
                x, model_prev_list, t_prev_list, t1, t2_, step, use_corrector=True
            )
            if model_x is None:
                model_x = self.model_fn(x, t1, t2_)
            self._update_lists(t_prev_list, model_prev_list, t1, model_x, order, first=True)

        for step in range(order, steps + 1):
            t1, t2_ = timesteps[step], timesteps2[step]
            step_order = min(order, steps + 1 - step) if lower_order_final else order
            use_corrector = (step != steps)
            x, model_x = self.multistep_uni_pc_bh_update(
                x, model_prev_list, t_prev_list, t1, t2_, step_order, use_corrector=use_corrector
            )
            if model_x is None:
                model_x = self.model_fn(x, t1, t2_)
            self._update_lists(t_prev_list, model_prev_list, t1, model_x, order, first=False)

        return x
