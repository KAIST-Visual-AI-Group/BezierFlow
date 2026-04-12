"""
Utility functions for BézierFlow.
"""

from typing import Optional
import os
import argparse
import random

import torch
import numpy as np
import yaml

from noise_schedulers import NoiseScheduleVE
from samplers.euler import Euler
from samplers.midpoint import Midpoint
from samplers.ipndm import iPNDM
from samplers.uni_pc import UniPC
from samplers.rk45 import RK45


ODE_MAPPER = {
    "euler": "v_prediction",
    "midpoint": "v_prediction",
    "uni_pc": "data_prediction",
    "ipndm": "noise_prediction",
    "rk45": "v_prediction",
}


def get_solvers(solver_name: str, prediction_type: str, NFEs: int, order: int,
                noise_schedule, unipc_variant: Optional[str] = None,
                save_traj: bool = False, use_last_step: bool = False):
    """Create ODE solver, compute steps, and collect extra parameters."""
    solver_extra_params = dict()

    if solver_name == 'euler':
        steps = NFEs
        solver = Euler(noise_schedule, "v_prediction", prediction_type)
        solver_extra_params["save_traj"] = save_traj
        solver_extra_params["use_last_step"] = use_last_step
    elif solver_name == 'midpoint':
        steps = NFEs // 2
        solver = Midpoint(noise_schedule, "v_prediction", prediction_type)
        solver_extra_params["save_traj"] = save_traj
        solver_extra_params["use_last_step"] = use_last_step
    elif solver_name == 'uni_pc':
        steps = NFEs
        solver = UniPC(noise_schedule, "data_prediction", prediction_type,
                       variant=unipc_variant or "bh1")
    elif solver_name == 'ipndm':
        steps = NFEs
        solver = iPNDM(noise_schedule, "noise_prediction", prediction_type)
    elif solver_name == 'rk45':
        steps = NFEs
        solver = RK45(noise_schedule, "v_prediction", prediction_type)
        solver_extra_params["save_traj"] = save_traj
        solver_extra_params["use_last_step"] = use_last_step
    else:
        raise NotImplementedError(f"Solver '{solver_name}' is not supported.")

    return solver, steps, solver_extra_params


def adjust_hyper(args, resolution=64, channel=3):
    """Adjust hyperparameters based on resolution and steps."""
    args.lr_time_2 = args.lr_time_2 / args.steps
    args.lr_time_2 = round(args.lr_time_2, 8)
    return args


def create_desc(args):
    """Create experiment description string."""
    return f"{args.solver_name}-N{args.steps}-{args.loss_type}-seed{args.seed}"


def save_arguments_to_yaml(args, filename):
    """Save arguments to a YAML file."""
    with open(filename, 'w') as file:
        yaml.dump(vars(args), file)


def is_trained(path):
    """Check if training has completed by inspecting log.txt."""
    log_path = os.path.join(path, 'log.txt')
    if not os.path.isfile(log_path):
        return False
    last_line = ""
    with open(log_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                last_line = stripped_line
    return "Training time" in last_line


def set_seed_everything(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_tensor_to_device(*args, device):
    """Move tensors to the specified device."""
    def _move(x):
        if torch.is_tensor(x):
            return x.to(device)
        elif x is None:
            return None
        return list(x)
    return [_move(arg) for arg in args]


def compute_distance_between_two(x, y, n_channels=3, resolution=256):
    """L2 distance per sample."""
    square_distance = (x - y) ** 2
    if x.ndim == 3:
        distance = square_distance.sum(dim=(1, 2)) / (n_channels * resolution)
    else:
        distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance


def compute_distance_between_two_L1(x, y, n_channels=3, resolution=256):
    """L1 distance per sample."""
    distance = torch.abs(x - y).sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance


def compute_distance_between_two_PSNR(x, y, data_range=2.0, eps=1e-10):
    """Negative PSNR per sample (for minimization)."""
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return -psnr


def compute_distance_between_two_MSE(x, y, n_channels=3, resolution=256):
    """MSE distance per sample."""
    square_distance = (x - y) ** 2
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance


def compute_distance_between_two_LPIPS_MSE(x, y, lpips_model):
    """LPIPS + MSE combined loss."""
    mse = compute_distance_between_two_MSE(x, y)
    lpips_val = lpips_model(x, y)
    return mse + lpips_val


def prepare_paths(args):
    """Prepare paths for FID evaluation."""
    skip_type = ""
    if args.learn:
        if args.load_from is None:
            desc = create_desc(args)
            args.log_path = os.path.join(args.log_path, desc)
            args.load_from = os.path.join(args.log_path, f'best_v{args.load_from_version}.pt')
        else:
            args.log_path = os.path.dirname(args.load_from)
            desc = os.path.basename(args.log_path)
    else:
        desc = f"{args.solver_name}_NFE{args.steps}_{args.skip_type}_seed{args.seed}"
        skip_type = args.skip_type

    if args.fid_folder:
        os.makedirs(args.fid_folder, exist_ok=True)
    return desc, None, skip_type


def parse_arguments():
    """Parse command-line arguments with YAML config override."""
    parser = argparse.ArgumentParser(description="BézierFlow")

    parser.add_argument('--all_config')
    parser.add_argument('--model', help="Model type: edm-cifar10/edm-ffhq/edm-afhqv2/reflow/flowdcn")

    # Model parameters
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--ckp_path", type=str)
    model_group.add_argument("--config", type=str, help="Model config YAML (for reflow/flowdcn)")
    model_group.add_argument("--vae_path", type=str, help="VAE path (for flowdcn)")
    model_group.add_argument("--solver_name", type=str)
    model_group.add_argument("--unipc_variant", type=str, choices=["bh1", "bh2"])
    model_group.add_argument("--steps", type=int)
    model_group.add_argument("--order", type=int)
    model_group.add_argument("--time_mode", type=str)
    model_group.add_argument("--prediction_type", type=str)
    model_group.add_argument("--schedule", type=str)
    model_group.add_argument("--skip_type", type=str)

    # Stochastic Interpolant
    si_group = parser.add_argument_group('Stochastic Interpolant')
    si_group.add_argument("--p_order", type=int)
    si_group.add_argument("--new_schedule", type=str)
    si_group.add_argument("--new_prediction_type", type=str)
    si_group.add_argument("--monotonic", action="store_true")

    # Training
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--seed", type=int)
    train_group.add_argument("--log_path", type=str)
    train_group.add_argument("--data_dir", type=str)
    train_group.add_argument("--num_train", type=int)
    train_group.add_argument("--num_valid", type=int)
    train_group.add_argument("--main_train_batch_size", type=int)
    train_group.add_argument("--main_valid_batch_size", type=int)
    train_group.add_argument("--loss_type", type=str, choices=["L1", "L2", "LPIPS", "RMSE", "PSNR", "MSE", "LPIPS_MSE"])
    train_group.add_argument("--training_rounds", type=int)
    train_group.add_argument("--lr_time_1", type=float)
    train_group.add_argument("--lr_time_2", type=float)
    train_group.add_argument("--min_lr_time_1", type=float)
    train_group.add_argument("--min_lr_time_2", type=float)
    train_group.add_argument("--momentum_time_1", type=float)
    train_group.add_argument("--weight_decay_time_1", type=float)
    train_group.add_argument("--lr_time_decay", type=float)
    train_group.add_argument("--patient", type=int)
    train_group.add_argument("--visualize", action="store_true")
    train_group.add_argument("--scale", type=float)
    train_group.add_argument("--low_gpu", action="store_true")
    train_group.add_argument("--init_from", type=str)

    # Testing / Evaluation
    test_group = parser.add_argument_group('Testing')
    test_group.add_argument("--load_from_version", type=int, default=2)
    test_group.add_argument("--learn", action="store_true")
    test_group.add_argument("--load_from", type=str)
    test_group.add_argument("--fid_folder", type=str, default=None)
    test_group.add_argument("--sampling_batch_size", type=int)
    test_group.add_argument("--sampling_seed", type=int)
    test_group.add_argument("--ref_path", type=str)
    test_group.add_argument("--total_samples", type=int)
    test_group.add_argument("--save_png", action="store_true")
    test_group.add_argument("--save_pt", action="store_true")
    test_group.add_argument("--save_traj", action="store_true")
    test_group.add_argument("--save_images", action="store_true")
    test_group.add_argument("--use_last_step", action="store_true")
    test_group.add_argument("--use_ema", action="store_true")
    test_group.add_argument("--num_samples_per_class", type=int, default=1)

    args = parser.parse_args()

    # Load config YAML and use as defaults
    if args.all_config and os.path.isfile(args.all_config):
        with open(args.all_config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    return args
