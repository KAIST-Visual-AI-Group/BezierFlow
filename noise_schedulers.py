"""
Noise schedulers for BézierFlow.

Supports:
  - VE (Variance Exploding) for EDM models
  - RF (Rectified Flow) for flow matching models
  - SI (Stochastic Interpolant) with learnable Bézier-parameterized α(t), σ(t)
  - Schedule converters (interpolant converter for mapping between schedules)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ---------------------------------------------------------------------------
# Piecewise-linear interpolation (used by VP schedule, kept for compatibility)
# ---------------------------------------------------------------------------

def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0), torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0), torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCHEDULER_REGISTRY = {}

def register_scheduler(name):
    def decorator(cls):
        _SCHEDULER_REGISTRY[name] = cls
        return cls
    return decorator

def get_scheduler(name):
    if name not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{name}' not found. Available: {list(_SCHEDULER_REGISTRY.keys())}")
    return _SCHEDULER_REGISTRY[name]


# ---------------------------------------------------------------------------
# VE schedule (for EDM models)
# ---------------------------------------------------------------------------

@register_scheduler("ve")
class NoiseScheduleVE:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, eps=0.002):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps
        self.T = sigma_max
        self.N = 1000

        with torch.no_grad():
            self.lambda_max = self.marginal_lambda(torch.tensor(self.eps)).item()
            self.lambda_min = self.marginal_lambda(torch.tensor(self.T)).item()

    def marginal_alpha(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t.float())

    def marginal_std(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return t.float().clamp_min(self.eps)

    def marginal_log_mean_coeff(self, t):
        return torch.zeros_like(self.marginal_alpha(t))

    def marginal_lambda(self, t):
        return -torch.log(self.marginal_std(t))

    def inverse_lambda(self, lamb):
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb)
        return torch.exp(-lamb)

    def inverse_std(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma)
        return sigma

    def dalpha_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.zeros_like(t.float())

    def dsigma_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t.float())

    def ft(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.zeros_like(t.float())

    def gt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.sqrt(2.0 * t.float())

    def prior_transformation(self, latents):
        return latents * self.T


# ---------------------------------------------------------------------------
# VP schedule (for Score SDE models)
# ---------------------------------------------------------------------------

@register_scheduler("vp")
class NoiseScheduleVP:
    def __init__(self, schedule='discrete', betas=None, alphas_cumprod=None,
                 continuous_beta_0=0.1, continuous_beta_1=20.,
                 dtype=torch.float32, eps=1e-3):
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(f"Unsupported noise schedule {schedule}")

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.log_alpha_array = self._numerical_clip_alpha(log_alphas).reshape((1, -1)).to(dtype=dtype)
            self.eps = 1 / self.total_N
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.lambda_max = self.marginal_lambda(eps).item()
            self.lambda_min = self.marginal_lambda(self.T).item()
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            self.T = 0.9946 if schedule == 'cosine' else 1.
            self.lambda_max = self.marginal_lambda(eps).item()
            self.lambda_min = self.marginal_lambda(self.T).item()
            self.eps = eps

    @staticmethod
    def _numerical_clip_alpha(log_alphas, clipped_lambda=-5.1):
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                                  self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return 2. * tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)),
                               torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                               torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            raise NotImplementedError

    def dalpha_dt(self, t):
        return self.derivative(self.marginal_alpha, t)

    def dsigma_dt(self, t):
        return self.derivative(self.marginal_std, t)

    @staticmethod
    def derivative(f, t, h=1e-6):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return (f(t + h) - f(t - h)) / (2 * h)

    def prior_transformation(self, latents):
        return latents


# ---------------------------------------------------------------------------
# RF schedule (Rectified Flow, t=0 is noise, t=1 is data)
# ---------------------------------------------------------------------------

@register_scheduler("rf")
class NoiseScheduleRF:
    def __init__(self, eps=1e-4, N=1000):
        self.eps = eps
        self.N = N
        self.T = 0.9999
        self.lambda_max = self.marginal_lambda(self.T).item()
        self.lambda_min = self.marginal_lambda(self.eps).item()
        self.schedule = 'rf'

    def dalpha_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t)

    def dsigma_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return -torch.ones_like(t)

    def marginal_log_mean_coeff(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.log(t)

    def marginal_alpha(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return t.clamp_min(self.eps)

    def marginal_std(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return (1. - t).clamp_min(self.eps)

    def inverse_std(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma)
        return 1. - sigma

    def marginal_lambda(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return self.marginal_log_mean_coeff(t) - torch.log(self.marginal_std(t))

    def inverse_lambda(self, lamb):
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb)
        return torch.exp(lamb) / (1. + torch.exp(lamb))

    def prior_transformation(self, latents):
        return latents


# ---------------------------------------------------------------------------
# SI schedule (Stochastic Interpolant with learnable Bézier α, σ)
# ---------------------------------------------------------------------------

@register_scheduler("si")
class NoiseSchedulerSI(nn.Module):
    """
    Learnable noise schedule parameterized by Bézier curves.
    α(t) goes from 0→1, σ(t) goes from 1→0 (monotonic by default).
    """
    def __init__(self, p_order=32, eps=1e-4, N=1000, monotonic=True, orig_sched=None, init_opt=False):
        super().__init__()
        assert p_order >= 1
        self.eps = eps
        self.N = N
        self.T = 0.9999

        self.alpha_ctrl = nn.Parameter(torch.ones(p_order))
        self.sigma_ctrl = nn.Parameter(torch.ones(p_order))

        self.monotonic = monotonic
        self.orig_sched = orig_sched

        if init_opt and orig_sched is not None and not isinstance(orig_sched, NoiseScheduleRF):
            self._init_from_original_opt(orig_sched)

        with torch.no_grad():
            self.lambda_max = self.orig_sched.lambda_max
            self.lambda_min = self.orig_sched.lambda_min

        # Cached converter values (set by make_interpolant_converter)
        self.dc_s = None
        self.c_s = None
        self.dt_s = None
        self.t_s = None
        self.drift_coeff = None
        self.eps_coeff = None

    # ---- Bézier evaluation ----

    def _bernstein_basis(self, n, t):
        device = t.device
        i = torch.arange(0, n + 1, device=device, dtype=t.dtype)
        comb = torch.exp(
            torch.lgamma(torch.tensor(n + 1.0, device=device, dtype=t.dtype)) -
            torch.lgamma(i + 1.0) -
            torch.lgamma(torch.tensor(n, device=device, dtype=t.dtype) - i + 1.0)
        )
        t = t.unsqueeze(-1)
        return comb * (t ** i) * ((1.0 - t) ** (n - i))

    def _bezier_eval(self, ctrl, t):
        n = ctrl.numel() - 1
        basis = self._bernstein_basis(n, t)
        return (basis * ctrl.view(*((1,) * (basis.ndim - 1)), -1)).sum(dim=-1)

    def _bezier_derivative(self, ctrl, t):
        n = ctrl.numel() - 1
        if n == 0:
            return torch.zeros_like(t)
        diffs = n * (ctrl[1:] - ctrl[:-1])
        basis = self._bernstein_basis(n - 1, t)
        return (basis * diffs.view(*((1,) * (basis.ndim - 1)), -1)).sum(dim=-1)

    # ---- Control points (with boundary conditions) ----

    def _alpha_ctrl(self, device=None, dtype=None):
        if self.monotonic:
            deltas = F.softmax(self.alpha_ctrl, dim=0)
            cum = torch.cumsum(deltas, dim=0)
            mid = cum / (cum[-1] + 1e-12)
        else:
            mid = torch.sigmoid(self.alpha_ctrl)
        mid = mid[:-1]
        return torch.cat([
            torch.tensor([0.], device=device, dtype=dtype),
            mid.to(device=device, dtype=dtype),
            torch.tensor([1.], device=device, dtype=dtype),
        ])

    def _sigma_ctrl(self, device=None, dtype=None):
        if self.monotonic:
            deltas = F.softmax(self.sigma_ctrl, dim=0)
            cum = torch.cumsum(deltas, dim=0)
            mid = 1.0 - cum / (cum[-1] + 1e-12)
        else:
            mid = torch.sigmoid(self.sigma_ctrl)
        mid = mid[:-1]
        return torch.cat([
            torch.tensor([1.], device=device, dtype=dtype),
            mid.to(device=device, dtype=dtype),
            torch.tensor([0.], device=device, dtype=dtype),
        ])

    # ---- Schedule functions ----

    def marginal_alpha(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        ctrl = self._alpha_ctrl(device=t.device, dtype=t.dtype)
        return self._bezier_eval(ctrl, t).clamp_min(self.eps)

    def marginal_std(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        ctrl = self._sigma_ctrl(device=t.device, dtype=t.dtype)
        return self._bezier_eval(ctrl, t).clamp_min(self.eps)

    def marginal_log_mean_coeff(self, t):
        return torch.log(self.marginal_alpha(t))

    def marginal_lambda(self, t):
        a = self.marginal_alpha(t)
        s = self.marginal_std(t)
        return torch.log(a) - torch.log(s)

    def dalpha_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        ctrl = self._alpha_ctrl(device=t.device, dtype=t.dtype)
        return self._bezier_derivative(ctrl, t)

    def dsigma_dt(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        ctrl = self._sigma_ctrl(device=t.device, dtype=t.dtype)
        return self._bezier_derivative(ctrl, t)

    @torch.no_grad()
    def inverse_lambda(self, lamb, iters=20):
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb, dtype=torch.float32)
        t = torch.sigmoid(lamb).clamp(self.eps, 1.0 - self.eps).to(lamb.device).clone()
        for _ in range(iters):
            a = self.marginal_alpha(t)
            s = self.marginal_std(t)
            da = self.dalpha_dt(t)
            ds = self.dsigma_dt(t)
            f = torch.log(a) - torch.log(s) - lamb
            fp = (da / a.clamp_min(self.eps)) - (ds / s.clamp_min(self.eps))
            step = f / fp.clamp_min(1e-8)
            t = (t - step).clamp(self.eps, 1.0 - self.eps)
            if torch.max(torch.abs(step)) < 1e-6:
                break
        return t

    @torch.no_grad()
    def inverse_std(self, sigma, iters=20):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        t = (1.0 - sigma).clamp(self.eps, 1.0 - self.eps).to(sigma.device).clone()
        for _ in range(iters):
            s = self.marginal_std(t)
            ds = self.dsigma_dt(t)
            f = s - sigma
            step = f / ds.clamp_min(1e-8)
            t = (t - step).clamp(self.eps, 1.0 - self.eps)
            if torch.max(torch.abs(step)) < 1e-6:
                break
        return t

    def ft(self, s):
        if self.dc_s is not None:
            ft_val = self.orig_sched.ft(self.t_s)
            return self.dc_s / self.c_s + self.dt_s * ft_val
        return torch.zeros_like(s)

    def gt_square(self, s):
        if self.dc_s is not None:
            gt_val = self.orig_sched.gt(self.t_s)
            if gt_val.shape[-1] > 1:
                gt_val = gt_val[..., 0:1]
            return (self.c_s ** 2) * self.dt_s * (gt_val ** 2)
        return torch.ones_like(s)

    def prior_transformation(self, latents):
        return latents

    @torch.enable_grad()
    def _init_from_original_opt(self, orig_sched, K=8192, steps=1000, lr=5e-2):
        device = self.alpha_ctrl.device
        s = torch.linspace(self.eps, 1.0 - self.eps, K, device=device)

        # Compute targets from original schedule
        t_noise, t_data = self._orient_from(orig_sched)
        t = t_noise + (t_data - t_noise) * s

        is_ve = isinstance(orig_sched, NoiseScheduleVE)
        if is_ve:
            sig_abs = orig_sched.marginal_std(t).clamp_min(1e-6)
            denom = torch.sqrt(1.0 + sig_abs * sig_abs)
            a_tgt = (1.0 / denom).detach()
            s_tgt = (sig_abs / denom).detach()
        else:
            a_tgt = orig_sched.marginal_alpha(t).clamp_min(1e-6).detach()
            s_tgt = orig_sched.marginal_std(t).clamp_min(1e-6).detach()

        lam_tgt = (torch.log(a_tgt) - torch.log(s_tgt)).detach()

        with torch.no_grad():
            self.alpha_ctrl.zero_()
            self.sigma_ctrl.zero_()

        opt = torch.optim.Adam([self.alpha_ctrl, self.sigma_ctrl], lr=lr)
        best_loss, best_state = float("inf"), None

        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            a_pred = self.marginal_alpha(s).clamp_min(1e-8)
            sg_pred = self.marginal_std(s).clamp_min(1e-8)
            lam_pred = torch.log(a_pred) - torch.log(sg_pred)
            loss = (lam_pred - lam_tgt).pow(2).mean() + \
                   0.5 * (a_pred - a_tgt).pow(2).mean() + \
                   0.5 * (sg_pred - s_tgt).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.alpha_ctrl, self.sigma_ctrl], 5.0)
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = (self.alpha_ctrl.detach().clone(), self.sigma_ctrl.detach().clone())

        if best_state is not None:
            with torch.no_grad():
                self.alpha_ctrl.copy_(best_state[0])
                self.sigma_ctrl.copy_(best_state[1])

    @torch.no_grad()
    def _orient_from(self, orig_sched):
        lam_eps = float(orig_sched.marginal_lambda(torch.tensor(orig_sched.eps)))
        lam_T = float(orig_sched.marginal_lambda(torch.tensor(orig_sched.T)))
        is_ve = isinstance(orig_sched, NoiseScheduleVE)
        if is_ve:
            sig_eps = float(orig_sched.marginal_std(torch.tensor(orig_sched.eps)))
            sig_T = float(orig_sched.marginal_std(torch.tensor(orig_sched.T)))
            noise_at_T = (sig_T > sig_eps)
        else:
            noise_at_T = (lam_T < lam_eps)
        if noise_at_T:
            return float(orig_sched.T), float(orig_sched.eps)
        else:
            return float(orig_sched.eps), float(orig_sched.T)


# ---------------------------------------------------------------------------
# VE-aware SI schedule (for diffusion models with PF-ODE)
# ---------------------------------------------------------------------------

@register_scheduler("si_ve")
class NoiseSchedulerSIVE(NoiseSchedulerSI):
    """SI schedule variant for VE-type diffusion models."""
    pass


# ---------------------------------------------------------------------------
# Schedule converters
# ---------------------------------------------------------------------------

def _reshape_like(v, x):
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, device=x.device, dtype=x.dtype)
    if v.ndim == 0:
        v = v.view(1)
    return v.view((v.shape[0],) + (1,) * (x.ndim - 1))


def make_interpolant_converter(orig_sched, new_sched, orig_model, original_solver):
    """
    Create a velocity converter from the original schedule to the SI schedule.
    For flow models (v-prediction), converts velocity fields directly.
    """
    def u_bar(x_bar, s1, s2, *args, **kwargs):
        a_bar1 = new_sched.marginal_alpha(s1)
        s_bar1 = new_sched.marginal_std(s1)
        da_bar1 = new_sched.dalpha_dt(s1)
        ds_bar1 = new_sched.dsigma_dt(s1)

        a_bar2 = new_sched.marginal_alpha(s2)
        s_bar2 = new_sched.marginal_std(s2)

        lambda_bar1 = torch.log(a_bar1) - torch.log(s_bar1)
        lambda_bar2 = torch.log(a_bar2) - torch.log(s_bar2)

        t_s1 = orig_sched.inverse_lambda(lambda_bar1)
        t_s2 = orig_sched.inverse_lambda(lambda_bar2)

        a_ts = orig_sched.marginal_alpha(t_s1)
        s_ts = orig_sched.marginal_std(t_s1)
        da_ts = orig_sched.dalpha_dt(t_s1)
        ds_ts = orig_sched.dsigma_dt(t_s1)

        dot_rho_bar = (da_bar1 * s_bar1 - a_bar1 * ds_bar1) / (s_bar1 ** 2)
        dot_rho = (da_ts * s_ts - a_ts * ds_ts) / (s_ts ** 2)

        dt_s = dot_rho_bar / dot_rho
        c_s = s_bar1 / s_ts
        dc_s = (s_ts * ds_bar1 - s_bar1 * ds_ts * dt_s) / (s_ts ** 2)

        c_s_b = _reshape_like(c_s, x_bar)
        x_t = x_bar / c_s_b

        t_s_out = orig_model(x_t, t_s1, t_s2, *args, **kwargs)
        u_ts = original_solver.dx_dt_for_blackbox_solvers(x_t, t_s1, t_s2, model_output=t_s_out)

        dc_over_c = _reshape_like(dc_s / c_s, x_bar)
        c_dt = _reshape_like(c_s * dt_s, x_bar)
        u_bar_val = dc_over_c * x_bar + c_dt * u_ts

        if original_solver.prediction_type != "v":
            new_sched.dc_s = _reshape_like(dc_s, x_bar)
            new_sched.c_s = _reshape_like(c_s, x_bar)
            new_sched.dt_s = _reshape_like(dt_s, x_bar)
            new_sched.t_s = _reshape_like(t_s1, x_bar)

        return u_bar_val

    return u_bar


