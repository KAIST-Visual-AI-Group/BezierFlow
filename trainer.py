"""
BézierFlow Trainer: jointly optimizes Bézier noise schedules and timestep discretization.
"""

from typing import List, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import lpips
import logging
import matplotlib.pyplot as plt
import imageio
import PIL

import os
import re
import imageio.v2 as imageio
import math
import pickle
import numpy as np

from noise_schedulers import NoiseScheduleVE, NoiseSchedulerSIVE
from dataset import BFDataset
from utils import (
    move_tensor_to_device,
    compute_distance_between_two,
    compute_distance_between_two_L1,
    compute_distance_between_two_PSNR,
    compute_distance_between_two_MSE,
    compute_distance_between_two_LPIPS_MSE,
)


def save_gif(snapshot_path: str):
    care_files = [
        f for f in os.listdir(snapshot_path)
        if re.match(r"^log_best_(\d+)\.png$", f)
    ]
    care_files = sorted(care_files, key=lambda f: int(f.split("_")[-1].replace(".png", "")))
    if not care_files:
        return
    images = [imageio.imread(os.path.join(snapshot_path, f)) for f in care_files]
    out_path = os.path.join(snapshot_path, "steps.gif")
    imageio.mimsave(out_path, images, duration=0.1)


def save_gif_si(snapshot_path: str, duration: float = 0.1, loop: int = 0):
    patterns = [
        (re.compile(r"^log_best_(\d+)_alpha_sigma\.png$"), "alpha_sigma.gif"),
        (re.compile(r"^log_best_(\d+)_lambda\.png$"), "lambda.gif"),
    ]
    for regex, out_name in patterns:
        frames = []
        for fname in os.listdir(snapshot_path):
            m = regex.match(fname)
            if m:
                frames.append((int(m.group(1)), fname))
        frames.sort(key=lambda x: x[0])
        if not frames:
            continue
        imgs = [imageio.imread(os.path.join(snapshot_path, f)) for _, f in frames]
        out_path = os.path.join(snapshot_path, out_name)
        imageio.mimsave(out_path, imgs, duration=duration, loop=loop)


def visual(input_, name="test.png", img_resolution=32, img_channels=3):
    input_ = (input_ + 1.) / 2.
    batch_size = input_.shape[0]
    gridh = int(math.sqrt(batch_size))
    for i in range(1, gridh + 1):
        if batch_size % i == 0:
            gridh = i
    gridw = batch_size // gridh
    image = (input_ * 255.).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * img_resolution, gridw * img_resolution, img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(name)


def custom_collate_fn(batch):
    collated_batch = []
    for samples in zip(*batch):
        if any(item is None for item in samples):
            collated_batch.append(None)
        else:
            collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    return collated_batch


def _aligned_time_grid(sched, K, device, dtype):
    t0, t1 = sched.T, sched.eps
    if not isinstance(sched, NoiseScheduleVE):
        t0, t1 = t1, t0
    return torch.linspace(float(t0), float(t1), K, device=device, dtype=dtype)


def discretize_model_wrapper(input1, input2, t_max, t_min, mode, window_rate=0.5):
    """Forward-time discretization: t goes from 0 -> 1 (or eps -> T)."""

    def model_time_fn():
        time1, time2 = input1, input2
        time_plus = F.softmax(time1, dim=0)
        time_cum = torch.cumsum(time_plus, dim=0)
        normed = (time_cum - time_cum.min()) / (time_cum.max() - time_cum.min())
        time_steps = normed * (t_max - t_min) + t_min

        cloned_time_steps = time_steps.clone().detach()
        max_move = (cloned_time_steps[1:] - cloned_time_steps[:-1]).abs().min().item() * window_rate
        clipped_time2 = torch.clamp(time2, min=-max_move, max=max_move)

        mask = torch.ones_like(normed)
        mask[0] = 0.
        mask[-1] = 0.
        return time_steps, time_steps + clipped_time2 * mask

    return model_time_fn


@dataclass
class TrainingConfig:
    train_data: any
    valid_data: any
    train_batch_size: int
    valid_batch_size: int
    lr_time_1: float
    lr_time_2: float
    min_lr_time_1: float = 5e-5
    min_lr_time_2: float = 1e-6
    patient: int = 5
    lr_time_decay: float = 0.8
    momentum_time_1: float = 0.9
    weight_decay_time_1: float = 0.0
    loss_type: str = "LPIPS"
    visualize: bool = False
    init_from: str = None


@dataclass
class ModelConfig:
    net: any
    decoding_fn: any
    original_schedule: any
    noise_schedule: any
    solver: any
    solver_name: str
    order: int
    steps: int
    resolution: int
    channels: int
    time_mode: str
    solver_extra_params: Optional[dict] = None
    snapshot_path: str = "logs"
    device: Optional[str] = None


class BFTrainer:
    """BézierFlow trainer: jointly optimizes Bézier schedule params + timestep params."""

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig) -> None:
        # Model parameters
        self.net = model_config.net
        self.decoding_fn = model_config.decoding_fn
        self.original_schedule = model_config.original_schedule
        self.noise_schedule = model_config.noise_schedule
        self.solver = model_config.solver
        self.solver_name = model_config.solver_name
        self.order = model_config.order
        self.steps = model_config.steps
        self.resolution = model_config.resolution
        self.channels = model_config.channels
        self.time_mode = model_config.time_mode

        # Learning rate parameters
        self.lr_time_1 = training_config.lr_time_1
        self.lr_time_2 = training_config.lr_time_2
        self.min_lr_time_1 = training_config.min_lr_time_1
        self.min_lr_time_2 = training_config.min_lr_time_2
        self.lr_time_decay = training_config.lr_time_decay

        # Training data
        self.train_data = training_config.train_data
        self.valid_data = training_config.valid_data
        self.train_batch_size = training_config.train_batch_size
        self.valid_batch_size = training_config.valid_batch_size
        self._create_loaders()

        # Training state
        self.cur_iter = 0
        self.cur_round = 0
        self.count_worse = 0
        self.best_loss = float("inf")

        # Other parameters
        self.patient = training_config.patient
        self.snapshot_path = model_config.snapshot_path
        os.makedirs(self.snapshot_path, exist_ok=True)
        self.visualize = training_config.visualize
        self.init_from = training_config.init_from

        # Device
        self.device = model_config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize timesteps
        if self.init_from is not None:
            timesteps = torch.load(self.init_from).to(self.device)
        else:
            if self.time_mode == "time":
                timesteps = torch.linspace(self.noise_schedule.eps, self.noise_schedule.T,
                                           self.steps + 1, device=self.device)
                if isinstance(self.noise_schedule, NoiseSchedulerSIVE):
                    timesteps = timesteps.flip(0)
            elif self.time_mode == "lambda":
                logSNR_steps = torch.linspace(self.noise_schedule.lambda_min,
                                              self.noise_schedule.lambda_max,
                                              self.steps + 1).to(self.device)
                timesteps = self.noise_schedule.inverse_lambda(logSNR_steps)
            else:
                raise NotImplementedError(f"time_mode '{self.time_mode}' not supported")

        self.timesteps1 = timesteps
        self.timesteps2 = timesteps

        # Additional attributes
        self.solver_extra_params = model_config.solver_extra_params or {}
        self.lambda_min = self.noise_schedule.lambda_min
        self.lambda_max = self.noise_schedule.lambda_max
        self.t_min = self.noise_schedule.eps if self.time_mode == "time" else self.noise_schedule.inverse_lambda(self.lambda_min)
        self.t_max = self.noise_schedule.T if self.time_mode == "time" else self.noise_schedule.inverse_lambda(self.lambda_max)

        # Timestep params (params1 fixed, params2 trainable)
        self.params1, self.params2 = self._initialize_params()

        # Optimizers: schedule params + timestep params2 (jointly trained)
        self.optimizer_schedule = torch.optim.RMSprop(
            self.noise_schedule.parameters(), lr=training_config.lr_time_1,
            momentum=training_config.momentum_time_1,
            weight_decay=training_config.weight_decay_time_1,
        )
        self.optimizer_lamb2 = torch.optim.SGD(
            [self.params2], lr=training_config.lr_time_2
        )

        # Initialize baseline
        self._compute_baseline()

        # Initialize loss function
        self.loss_type = training_config.loss_type
        self.loss_fn = self._initialize_loss_fn()
        self.loss_vector = None

    def _initialize_loss_fn(self):
        if self.loss_type == 'LPIPS':
            return lpips.LPIPS(net='vgg').to(self.device)
        elif self.loss_type == 'L2':
            return lambda x, y: compute_distance_between_two(x, y, self.channels, self.resolution)
        elif self.loss_type == 'L1':
            return lambda x, y: compute_distance_between_two_L1(x, y, self.channels, self.resolution)
        elif self.loss_type == 'PSNR':
            return lambda x, y: compute_distance_between_two_PSNR(x, y)
        elif self.loss_type == 'MSE':
            return lambda x, y: compute_distance_between_two_MSE(x, y, self.channels, self.resolution)
        elif self.loss_type == 'LPIPS_MSE':
            lpips_model = lpips.LPIPS(net='vgg').to(self.device)
            return lambda x, y: compute_distance_between_two_LPIPS_MSE(x, y, lpips_model)
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' not supported")

    def _initialize_params(self):
        # params1: fixed (not trained), controls base timestep positions
        params1 = torch.nn.Parameter(
            torch.ones(self.steps + 1, dtype=torch.float32).cuda(), requires_grad=False)
        # params2: trainable, controls timestep perturbations
        params2 = torch.nn.Parameter(
            torch.zeros(self.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)

        if self.time_mode == 'lambda':
            device = self.device
            t_min, t_max = self.t_min, self.t_max
            c = 1e-3
            target = self.timesteps1
            nu = (target - t_min) / (t_max - t_min)
            nu = torch.clamp(nu, 0.0, 1.0)
            deltas = nu[1:] - nu[:-1]
            p0 = torch.tensor([c], device=device)
            prest = (1.0 - c) * deltas
            p = torch.cat([p0, prest], dim=0)
            p = torch.clamp(p, min=1e-12)
            p = p / p.sum()
            params1 = torch.log(p)
            params1 = torch.nn.Parameter(params1.to(device), requires_grad=False)

        self.noise_schedule.cuda()
        return params1, params2

    def _create_loaders(self):
        self.train_loader = DataLoader(self.train_data, batch_size=self.train_batch_size,
                                       shuffle=True, collate_fn=custom_collate_fn)
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.valid_batch_size,
                                       shuffle=False, collate_fn=custom_collate_fn)

    def _solve_ode(self, img, latent, condition=None, uncondition=None, valid=False):
        dis_model = discretize_model_wrapper(
            self.params1, self.params2,
            self.t_max, self.t_min,
            self.time_mode,
        )
        timesteps1, timesteps2 = dis_model()

        if not valid:
            tst = torch.cat([timesteps1, timesteps2], dim=0).detach().cpu()
            torch.save(tst, os.path.join(self.snapshot_path, f"t_steps.pt"))

        self.t_steps1 = timesteps1.detach()
        self.t_steps2 = timesteps2.detach()
        self.logSNR1 = self.noise_schedule.marginal_lambda(timesteps1).detach().cpu()
        self.logSNR2 = self.noise_schedule.marginal_lambda(timesteps2).detach().cpu()

        x_next_ = self.noise_schedule.prior_transformation(latent)
        x_next_ = self.solver.sample_simple(
            model_fn=self.net, x=x_next_,
            timesteps=timesteps1, timesteps2=timesteps2,
            order=self.order, NFEs=self.steps,
            condition=condition, unconditional_condition=uncondition,
            **self.solver_extra_params,
        )
        x_next_ = self.decoding_fn(x_next_)
        self.loss_vector = self.loss_fn(img.float(), x_next_.float()).squeeze()
        loss = self.loss_vector.mean()
        logging.info(f"Loss: {loss.item()}")
        return loss, x_next_.float(), img.float()

    def _compute_baseline(self):
        self.straight_line = torch.linspace(self.lambda_min, self.lambda_max, self.steps + 1)
        time_max = self.original_schedule.inverse_lambda(self.lambda_min)
        time_min = self.original_schedule.inverse_lambda(self.lambda_max)
        self.time_s = torch.linspace(time_max.item(), time_min.item(), 1000)
        self.straight_time = self.original_schedule.marginal_lambda(self.time_s)
        t_order = 2
        self.time_q = torch.linspace((time_max**(1/t_order)).item(), (time_min**(1/t_order)).item(), 1000)**t_order
        self.time_quadratic = self.original_schedule.marginal_lambda(self.time_q)
        self.time_edm = self.solver.get_time_steps('edm', time_max.item(), time_min.item(), 999, self.device)
        self.lambda_edm = self.original_schedule.marginal_lambda(self.time_edm)

    def _run_validation(self):
        total_loss = 0.
        count = 0
        outputs, targets = [], []
        with torch.no_grad():
            for img, latent, condition, uncondition in self.valid_loader:
                img, latent, condition, uncondition = move_tensor_to_device(
                    img, latent, condition, uncondition, device=self.device)
                loss, output, target = self._solve_ode(
                    img=img, latent=latent, condition=condition, uncondition=uncondition, valid=True)
                total_loss += loss.item()
                count += 1
                outputs.append(output)
                targets.append(target)
        return total_loss / count, torch.cat(outputs, dim=0), torch.cat(targets, dim=0)

    def _visual_times(self) -> None:
        log_path = os.path.join(self.snapshot_path, f"log_best_{self.cur_iter}.png")
        plt.plot(self.logSNR1.cpu().numpy(), 'o', label="Our discretization1")
        plt.plot(self.logSNR2.cpu().numpy(), 'x', label="Our discretization2")
        x_axis = np.linspace(0, self.steps, self.steps + 1)
        plt.plot(x_axis, self.straight_line.cpu().numpy(), label="Baseline logSNR")
        x_axis = np.linspace(0, self.steps, 1000)
        plt.plot(x_axis, self.straight_time.cpu().numpy(), label="Baseline time uniform")
        plt.plot(x_axis, self.time_quadratic.cpu().numpy(), label="Baseline time quadratic")
        plt.plot(x_axis, self.lambda_edm.cpu().numpy(), label="Baseline time edm")
        plt.xlabel("Reverse step i")
        plt.ylabel("LogSNR(t_i)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path)
        plt.close()

    def _visual_curves(self) -> None:
        ns = self.noise_schedule
        orig = self.original_schedule

        t_orig = _aligned_time_grid(orig, 1000, self.device, torch.float32)
        t_si = _aligned_time_grid(ns, 1000, self.device, torch.float32)

        l_orig = orig.marginal_lambda(t_orig).detach().cpu().numpy()
        l_si = ns.marginal_lambda(t_si).detach().cpu().numpy()

        plt.figure(figsize=(7, 4))
        x1 = torch.linspace(0, 1, len(l_orig)).cpu().numpy()
        x2 = torch.linspace(0, 1, len(l_si)).cpu().numpy()
        plt.plot(x1, l_orig, label="Original λ(t)")
        plt.plot(x2, l_si, "--", label="SI λ(t)")
        plt.xlabel("normalized t (0=noise → 1=data)")
        plt.ylabel("λ(t) = log α(t) − log σ(t)")
        plt.title("LogSNR (λ) comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.snapshot_path, f"log_best_{self.cur_iter}_lambda.png"))
        plt.close()

        a_orig = orig.marginal_alpha(t_orig).detach().cpu().numpy()
        s_orig = orig.marginal_std(t_orig).detach().cpu().numpy()
        a_si = ns.marginal_alpha(t_si).detach().cpu().numpy()
        s_si = ns.marginal_std(t_si).detach().cpu().numpy()

        if t_orig.max() > 40 and t_si.max() <= 1:
            denom = torch.sqrt(1 + torch.from_numpy(s_orig)**2).numpy()
            a_orig = a_orig / denom
            s_orig = s_orig / denom

        plt.figure(figsize=(7, 4))
        x1 = torch.linspace(0, 1, len(a_orig)).cpu().numpy()
        x2 = torch.linspace(0, 1, len(a_si)).cpu().numpy()
        plt.plot(x1, a_orig, label="Original α(t)")
        plt.plot(x1, s_orig, label="Original σ(t)")
        plt.plot(x2, a_si, "--", label="SI α(t)")
        plt.plot(x2, s_si, "--", label="SI σ(t)")
        plt.xlabel("normalized t (0=noise → 1=data)")
        plt.ylabel("α(t), σ(t)")
        plt.title("α/σ comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.snapshot_path, f"log_best_{self.cur_iter}_alpha_sigma.png"))
        plt.close()

    def _save_checkpoint(self):
        snapshot = {
            "params1": self.params1.data,
            "params2": self.params2.data,
            "scheduler_params": self.noise_schedule.state_dict(),
            "best_t_steps": torch.cat([self.t_steps1, self.t_steps2], dim=0),
        }
        torch.save(snapshot, os.path.join(self.snapshot_path, "best.pt"))
        torch.save(snapshot, os.path.join(self.snapshot_path, f"best_t_steps_{self.cur_iter}.pt"))

    def _load_checkpoint(self):
        snapshot = torch.load(os.path.join(self.snapshot_path, "best.pt"))
        self.params1.data = snapshot["params1"].cuda()
        self.params2.data = snapshot["params2"].cuda()
        self.noise_schedule.load_state_dict(snapshot["scheduler_params"])

    def _examine_checkpoint(self, iter: int) -> None:
        logging.info(f"Evaluating at iter {iter}")
        total_loss, output, target = self._run_validation()

        if (iter % 5 == 0 or total_loss < self.best_loss) and self.visualize:
            visual(torch.cat([output[:8], target[:8]], dim=0),
                   os.path.join(self.snapshot_path, f"learned_newnoise_ep{iter}.png"),
                   img_resolution=self.resolution)

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.count_worse = 0
            self._save_checkpoint()
            self._visual_times()
            self._visual_curves()
            save_gif_si(self.snapshot_path)
            save_gif(self.snapshot_path)
        else:
            self.count_worse += 1
            logging.info(f"Count worse: {self.count_worse}")

        logging.info(f"Validation loss: {total_loss}, best loss: {self.best_loss}")

        if self.count_worse >= self.patient:
            logging.info(f"Loading best model")
            self._load_checkpoint()
            self.count_worse = 0

            self.optimizer_schedule.param_groups[0]['lr'] = max(
                self.lr_time_decay * self.optimizer_schedule.param_groups[0]['lr'],
                self.min_lr_time_1)
            logging.info(f"Decay scheduler lr to {self.optimizer_schedule.param_groups[0]['lr']}")

            self.optimizer_lamb2.param_groups[0]['lr'] = max(
                self.lr_time_decay * self.optimizer_lamb2.param_groups[0]['lr'],
                self.min_lr_time_2)
            logging.info(f"Decay timestep lr to {self.optimizer_lamb2.param_groups[0]['lr']}")

    def _train_one_round(self):
        logging.info(f"Round {self.cur_round}")

        if self.cur_round > 0:
            self._load_checkpoint()
            self.count_worse = 0

        self._examine_checkpoint(self.cur_iter)

        # Set trainable: schedule params + params2 (params1 always fixed)
        for p in self.noise_schedule.parameters():
            p.requires_grad = True
        self.params2.requires_grad = True

        for img, latent, condition, uncondition in self.train_loader:
            img, latent, condition, uncondition = move_tensor_to_device(
                img, latent, condition, uncondition, device=self.device)

            loss, _, _ = self._solve_ode(
                img=img, latent=latent, condition=condition, uncondition=uncondition)

            logging.info(f"Iter {self.cur_iter} Train Loss: {loss.item()}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.noise_schedule.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.params2, 1.0)

            self.optimizer_schedule.step()
            self.optimizer_schedule.zero_grad()
            self.optimizer_lamb2.step()
            self.optimizer_lamb2.zero_grad()

            self.cur_iter += 1
            self._examine_checkpoint(self.cur_iter)

    def train(self, training_rounds: int) -> None:
        for _ in range(training_rounds):
            self._train_one_round()
            self.cur_round += 1
        logging.info(f"Max round reached, stopping")
