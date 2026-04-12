"""
Rectified Flow model wrapper.
"""

import torch
from omegaconf import OmegaConf
from noise_schedulers import get_scheduler, make_interpolant_converter
from samplers.general_solver import ODESolver

ODE_MAPPER = {
    "euler": "v_prediction", "midpoint": "v_prediction",
    "ipndm": "noise_prediction", "uni_pc": "data_prediction",
    "rk45": "v_prediction",
}


def _model_wrapper(model, use_checkpoint=False):
    """Wraps Rectified Flow model to output velocity."""
    def model_fn(x, _, t_continuous, *args, **kwargs):
        t_input = t_continuous * 1000
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(model, x, t_input)
        else:
            output = model(x, t_input)
        return output.to(torch.float32)
    return model_fn


def get_pretrained_reflow_model(args, requires_grad=False):
    # Import flow utilities (these come from the original repo)
    import sys, os
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from flow.ema import ExponentialMovingAverage
    from flow.utils import create_model, restore_checkpoint

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = OmegaConf.load(args.config)
    net = create_model(config)
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=None, model=net, ema=ema, step=0)

    with torch.no_grad():
        state = restore_checkpoint(args.ckp_path, state, device)

    ema.copy_to(net.parameters())
    net.to(device)
    net.eval()

    if not requires_grad:
        for param in net.parameters():
            param.requires_grad = False

    original_schedule = get_scheduler(args.schedule)()
    new_schedule = None
    if getattr(args, 'new_schedule', None) is not None and "si" in args.new_schedule:
        new_schedule = get_scheduler(args.new_schedule)(
            p_order=args.p_order, orig_sched=original_schedule
        )

    model_fn = _model_wrapper(net, use_checkpoint=getattr(args, 'low_gpu', False))

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
            config.data.image_size, config.data.num_channels,
            config.data.image_size, config.data.num_channels)
