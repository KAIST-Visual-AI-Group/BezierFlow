"""
EDM unconditional model wrapper.
Supports: edm-cifar10, edm-ffhq, edm-afhqv2
"""

import torch
import pickle
from noise_schedulers import get_scheduler, make_interpolant_converter
from samplers.general_solver import ODESolver

ODE_MAPPER = {
    "euler": "v_prediction", "midpoint": "v_prediction",
    "ipndm": "noise_prediction", "uni_pc": "data_prediction",
    "rk45": "v_prediction",
}


def _model_wrapper(model, noise_schedule, use_checkpoint=False):
    """Wraps EDM model to output eps prediction."""
    def model_fn(x, t1, t2, *args, **kwargs):
        alpha_t2 = noise_schedule.marginal_alpha(t2)[:, None, None, None]
        sigma_t2 = noise_schedule.marginal_std(t2)[:, None, None, None]
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            x0_pred = checkpoint(model, x, t2, None)
        else:
            x0_pred = model(x, t2, None)
        return (x - alpha_t2 * x0_pred) / sigma_t2
    return model_fn


def get_pretrained_edm_model(args, requires_grad=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(args.ckp_path, "rb") as f:
        net = pickle.load(f)["ema"].to(device)
    if not requires_grad:
        for param in net.parameters():
            param.requires_grad = False

    original_schedule = get_scheduler(args.schedule)()
    new_schedule = None
    if getattr(args, 'new_schedule', None) is not None and "si" in args.new_schedule:
        new_schedule = get_scheduler(args.new_schedule)(
            p_order=args.p_order, orig_sched=original_schedule
        )

    model_fn = _model_wrapper(net, original_schedule,
                               use_checkpoint=getattr(args, 'low_gpu', False))

    if new_schedule is not None:
        solver = ODESolver(original_schedule, ODE_MAPPER.get(args.solver_name, "v_prediction"),
                           args.prediction_type)
        model_fn_final = make_interpolant_converter(
            original_schedule, new_schedule, model_fn, solver
        )
    else:
        model_fn_final = model_fn
        new_schedule = original_schedule

    return (model_fn_final, net, lambda x: x,
            original_schedule, new_schedule,
            net.img_resolution, net.img_channels,
            net.img_resolution, net.img_channels)
