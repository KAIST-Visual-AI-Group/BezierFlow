"""
Compute FID using learned BézierFlow schedules.
Uses both EDM detector and InceptionV3 for dual FID computation.
"""

import torch
import os
import json
import math
import numpy as np
import scipy
import pickle
from tqdm import tqdm

from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3

import dnnlib

from utils import parse_arguments, adjust_hyper, get_solvers, set_seed_everything
from models import prepare_model, prepare_condition_loader
from gen_data import Generator, get_data_inverse_scaler


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FID models
    FEATURE_DIM = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    fid_model = InceptionV3([block_idx]).to(device)
    fid_model.eval()

    DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    with dnnlib.util.open_url(DETECTOR_URL, verbose=True) as f:
        detector_net = pickle.load(f).to(device)

    with dnnlib.util.open_url(args.ref_path) as f:
        ref = dict(np.load(f))

    # Load model
    (wrapped_model, model, decoding_fn, original_schedule, noise_schedule,
     latent_resolution, latent_channel, _, _) = prepare_model(args)

    condition_loader = prepare_condition_loader(
        model_type=args.model, model=model,
        scale=getattr(args, 'scale', None),
        sampling_batch_size=args.sampling_batch_size,
        num_samples_per_class=getattr(args, 'num_samples_per_class', 50),
    )

    adjust_hyper(args, latent_resolution, latent_channel)

    solver, steps, solver_extra_params = get_solvers(
        args.solver_name, args.new_prediction_type,
        NFEs=args.steps, order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=getattr(args, 'unipc_variant', None),
        use_last_step=getattr(args, 'use_last_step', False),
    )

    desc = f"{args.solver_name}-N{args.steps}-{args.loss_type}-seed{args.seed}-p{args.p_order}"
    desc += f"-{args.training_rounds}"

    save_folder = os.path.join(args.load_from, desc, "fid_samples")
    os.makedirs(save_folder, exist_ok=True)
    load_from = os.path.join(args.load_from, desc, "best.pt")

    generator = Generator(
        noise_schedule=noise_schedule, solver=solver, order=args.order,
        load_from=load_from, steps=steps,
        solver_extra_params=solver_extra_params, device=device,
    )

    # Load learned schedule parameters
    state = torch.load(load_from, map_location='cpu')
    noise_schedule.load_state_dict(state["scheduler_params"])
    noise_schedule.to(device)

    inverse_scalar = get_data_inverse_scaler(centered=True)

    num_batches = math.ceil(args.total_samples / args.sampling_batch_size)
    batch_size = args.sampling_batch_size
    n_total_samples = batch_size * num_batches

    mu = torch.zeros([FEATURE_DIM], dtype=torch.float64, device=device)
    sigma = torch.zeros([FEATURE_DIM, FEATURE_DIM], dtype=torch.float64, device=device)
    act_arr = np.empty((n_total_samples, FEATURE_DIM))
    start_idx = 0

    with torch.no_grad():
        for index in tqdm(range(num_batches)):
            current_batch_size = min(batch_size, args.total_samples - index * batch_size)
            sampling_shape = (current_batch_size, latent_channel, latent_resolution, latent_resolution)
            latents = torch.randn(sampling_shape, device=device)

            if condition_loader is not None:
                conditioning, conditioned_unconditioning = next(condition_loader)
            else:
                conditioning = None
                conditioned_unconditioning = None

            img_teacher = generator.sample(wrapped_model, decoding_fn, latents,
                                           conditioning, conditioned_unconditioning)
            img_teacher = inverse_scalar(img_teacher)

            # EDM detector features
            samples_edm = 255 * img_teacher
            images = torch.clip(samples_edm, 0, 255).to(torch.uint8)
            features = detector_net(images.to(device), return_features=True).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features

            # InceptionV3 features
            samples_latent_diff = torch.clamp(img_teacher, min=0.0, max=1.0)
            with torch.no_grad():
                pred = fid_model(samples_latent_diff.float())[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            act_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx += pred.shape[0]

    # EDM FID
    mu /= n_total_samples
    sigma -= mu.ger(mu) * n_total_samples
    sigma /= n_total_samples - 1
    mu_np = mu.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    fid_edm = calculate_fid_from_inception_stats(mu_np, sigma_np, ref["mu"], ref["sigma"])

    # InceptionV3 FID
    mu_inc = np.mean(act_arr, axis=0)
    sigma_inc = np.cov(act_arr, rowvar=False)
    fid_latent_diff = calculate_fid_from_inception_stats(mu_inc, sigma_inc, ref["mu"], ref["sigma"])

    print(f"FID EDM: {fid_edm:.4f}")
    print(f"FID LD: {fid_latent_diff:.4f}")

    # Save results
    fid_json_path = os.path.join(args.load_from, "fid_results.json")
    if os.path.exists(fid_json_path):
        with open(fid_json_path, "r") as f:
            fid_dict = json.load(f)
    else:
        fid_dict = {}

    exp_key = f"{args.model}-{desc}-cfg{getattr(args, 'scale', 0.0)}"
    fid_dict[exp_key] = {"FID_EDM": float(fid_edm), "FID_LD": float(fid_latent_diff)}

    with open(fid_json_path, "w") as f:
        json.dump(fid_dict, f, indent=4)
    print(f"FID results saved to {fid_json_path}")


if __name__ == "__main__":
    args = parse_arguments()
    set_seed_everything(args.seed)
    main(args)
