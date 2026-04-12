"""
Train BézierFlow — optimize Bézier stochastic interpolant schedulers.
"""

import torch
import logging
import os
import sys
import time

from dataset import load_data_from_dir, BFDataset
from trainer import BFTrainer, ModelConfig, TrainingConfig
from utils import (
    create_desc, is_trained, get_solvers,
    parse_arguments, adjust_hyper, save_arguments_to_yaml,
    set_seed_everything,
)
from models import prepare_model


def setup_logging(log_dir):
    logging.shutdown()
    import importlib
    importlib.reload(logging)
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def main(args):
    (wrapped_model, _, decoding_fn, original_schedule, noise_schedule,
     latent_resolution, latent_channel, _, _) = prepare_model(args)

    adjust_hyper(args, latent_resolution, latent_channel)
    desc = create_desc(args)
    desc += f"-p{args.p_order}"
    desc += f"-{args.training_rounds}"

    log_dir = os.path.join(args.log_path, desc)
    if is_trained(log_dir):
        print("Skip training")
        return
    else:
        print("The model hasn't been trained yet. Perform training")
    os.makedirs(log_dir, exist_ok=True)
    save_arguments_to_yaml(args, os.path.join(log_dir, "config.yml"))
    setup_logging(log_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    solver, steps, solver_extra_params = get_solvers(
        args.solver_name, args.new_prediction_type,
        NFEs=args.steps, order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=getattr(args, 'unipc_variant', None),
    )

    latents, targets, conditions, unconditions = load_data_from_dir(
        data_folder=args.data_dir, limit=args.num_train + args.num_valid)

    train_dataset = BFDataset(
        latents[:args.num_train],
        targets[:args.num_train], conditions[:args.num_train],
        unconditions[:args.num_train],
    )
    if args.num_valid > 0:
        valid_dataset = BFDataset(
            latents[args.num_train:],
            targets[args.num_train:], conditions[args.num_train:],
            unconditions[args.num_train:],
        )
    else:
        valid_dataset = train_dataset

    training_config = TrainingConfig(
        train_data=train_dataset, valid_data=valid_dataset,
        train_batch_size=args.main_train_batch_size,
        valid_batch_size=args.main_valid_batch_size,
        lr_time_1=args.lr_time_1, lr_time_2=args.lr_time_2,
        min_lr_time_1=args.min_lr_time_1, min_lr_time_2=args.min_lr_time_2,
        patient=args.patient,
        lr_time_decay=args.lr_time_decay,
        momentum_time_1=args.momentum_time_1,
        weight_decay_time_1=args.weight_decay_time_1,
        loss_type=args.loss_type, visualize=args.visualize,
        init_from=getattr(args, 'init_from', None),
    )
    model_config = ModelConfig(
        net=wrapped_model, decoding_fn=decoding_fn,
        original_schedule=original_schedule, noise_schedule=noise_schedule,
        solver=solver, solver_name=args.solver_name,
        order=args.order, steps=steps,
        resolution=latent_resolution, channels=latent_channel,
        time_mode=args.time_mode,
        solver_extra_params=solver_extra_params,
        snapshot_path=log_dir, device=device,
    )
    trainer = BFTrainer(model_config, training_config)

    start = time.time()
    trainer.train(args.training_rounds)
    end = time.time()
    logging.info(f"Training time: {end - start}")


if __name__ == "__main__":
    args = parse_arguments()
    set_seed_everything(args.seed)
    main(args)
