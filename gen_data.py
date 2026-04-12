"""
Generate teacher data using RK45 (or other solvers).
Produces (latent, image) pairs saved as .pt files.
"""

import torch
import os
import time
import numpy as np
import PIL.Image
from tqdm import tqdm

from utils import get_solvers, parse_arguments, prepare_paths, adjust_hyper, set_seed_everything
from models import prepare_model, prepare_condition_loader


def get_data_inverse_scaler(centered=True):
    if centered:
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


class Generator:
    def __init__(self, noise_schedule, solver, order, skip_type=None,
                 load_from=None, timesteps_1=None, timesteps_2=None,
                 steps=35, solver_extra_params=None, device=None):
        self.device = device
        self.noise_schedule = noise_schedule
        self.solver = solver
        self.order = order
        self.skip_type = skip_type
        self.load_from = load_from
        self.timesteps_1 = timesteps_1
        self.timesteps_2 = timesteps_2
        self.steps = steps
        self.solver_extra_params = solver_extra_params or {}
        self._precompute_timesteps()

    def _precompute_timesteps(self):
        if (self.load_from is None
                and isinstance(self.timesteps_1, list) and len(self.timesteps_1) > 0
                and isinstance(self.timesteps_1[0], float)
                and isinstance(self.timesteps_2, list) and isinstance(self.timesteps_2[0], float)):
            ts1 = torch.tensor(self.timesteps_1, device=self.device, dtype=torch.float32)
            ts2 = torch.tensor(self.timesteps_2, device=self.device, dtype=torch.float32)
            self.timesteps = ts1
            self.timesteps2 = ts2
        else:
            self.timesteps, self.timesteps2 = self.solver.prepare_timesteps(
                steps=self.steps,
                t_start=self.noise_schedule.T,
                t_end=self.noise_schedule.eps,
                skip_type=self.skip_type,
                device=self.device,
                load_from=self.load_from,
            )

    def _sample(self, net, decoding_fn, latents, condition=None, unconditional_condition=None):
        x_next_ = self.noise_schedule.prior_transformation(latents)
        x_next_ = self.solver.sample_simple(
            model_fn=net, x=x_next_,
            timesteps=self.timesteps, timesteps2=self.timesteps2,
            order=self.order, NFEs=self.steps,
            condition=condition, unconditional_condition=unconditional_condition,
            **self.solver_extra_params,
        )
        x_next_ = decoding_fn(x_next_)
        return x_next_

    def sample(self, net, decoding_fn, latents, condition=None, unconditional_condition=None, no_grad=True):
        if no_grad:
            with torch.no_grad():
                return self._sample(net, decoding_fn, latents, condition, unconditional_condition)
        else:
            return self._sample(net, decoding_fn, latents, condition, unconditional_condition)


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # gen_data always uses the original schedule (not SI) to generate teacher data
    args.new_schedule = None

    (wrapped_model, model, decoding_fn, original_schedule, noise_schedule,
     latent_resolution, latent_channel, img_resolution, img_channel) = prepare_model(args)

    condition_loader = prepare_condition_loader(
        model_type=args.model, model=model,
        scale=getattr(args, 'scale', None),
        sampling_batch_size=args.sampling_batch_size,
        num_samples_per_class=getattr(args, 'num_samples_per_class', 1),
    )

    adjust_hyper(args, latent_resolution, latent_channel)
    _, _, skip_type = prepare_paths(args)
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    solver, steps, solver_extra_params = get_solvers(
        args.solver_name, args.prediction_type,
        NFEs=args.steps, order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=getattr(args, 'unipc_variant', None),
        save_traj=getattr(args, 'save_traj', False),
    )

    generator = Generator(
        noise_schedule=noise_schedule, solver=solver, order=args.order,
        skip_type=skip_type, load_from=getattr(args, 'load_from', None),
        steps=steps, solver_extra_params=solver_extra_params, device=device,
    )

    print(generator.timesteps, generator.timesteps2)
    inverse_scalar = get_data_inverse_scaler(centered=True)

    start = time.time()
    count = 0
    batch_size = args.sampling_batch_size
    num_batches = (args.total_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, args.total_samples - i * batch_size)
        sampling_shape = (current_batch_size, latent_channel, latent_resolution, latent_resolution)
        latents = torch.randn(sampling_shape, device=device)

        if condition_loader is not None:
            conditioning, conditioned_unconditioning = next(condition_loader)
        else:
            conditioning = None
            conditioned_unconditioning = None

        img_teacher = generator.sample(wrapped_model, decoding_fn, latents,
                                       conditioning, conditioned_unconditioning)
        img_teacher = img_teacher.detach().cpu().view(
            current_batch_size, img_channel, img_resolution, img_resolution)
        latents = latents.detach().cpu()

        if getattr(args, 'save_pt', False):
            for j in range(current_batch_size):
                data = dict(
                    latent=latents[j], img=img_teacher[j],
                    c=conditioning[j] if conditioning is not None else None,
                    uc=conditioned_unconditioning[j] if conditioned_unconditioning is not None else None,
                )
                torch.save(data, os.path.join(data_dir, f"latent_{(count + j):06d}.pt"))

        if getattr(args, 'save_png', False):
            samples_raw = inverse_scalar(img_teacher)
            samples = np.clip(
                samples_raw.permute(0, 2, 3, 1).cpu().float().numpy() * 255.0, 0, 255
            ).astype(np.uint8)
            images_np = samples.reshape((-1, img_resolution, img_resolution, img_channel))
            for j in range(current_batch_size):
                image_np = images_np[j]
                image_path = os.path.join(data_dir, f"{(count + j):06d}.png")
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                else:
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        count += batch_size

    end = time.time()
    print(f"Generation time: {end - start}")


if __name__ == "__main__":
    args = parse_arguments()
    set_seed_everything(args.seed)
    main(args)
