"""
Model loading and preparation for BézierFlow.

Supported models:
  - EDM unconditional (CIFAR-10, FFHQ, AFHQv2)
  - Rectified Flow (CIFAR-10)
  - FlowDCN (ImageNet 256x256)
"""

from .edm_uncond import get_pretrained_edm_model
from .reflow import get_pretrained_reflow_model
from .flowdcn import get_pretrained_flowdcn_model
from .condition_loader import ImageNetIterator


def prepare_model(args):
    """Load pretrained model and create schedule/converter."""
    if 'edm' in args.model:
        return get_pretrained_edm_model(args)
    elif args.model == 'reflow':
        return get_pretrained_reflow_model(args)
    elif args.model == 'flowdcn':
        return get_pretrained_flowdcn_model(args)
    else:
        raise NotImplementedError(f"Model '{args.model}' is not supported.")


def prepare_condition_loader(model_type, model, scale, sampling_batch_size,
                             num_samples_per_class=50):
    """Return a condition loader iterator, or None for unconditional models."""
    if 'edm' in model_type or model_type == 'reflow':
        return None
    elif model_type == 'flowdcn':
        return ImageNetIterator(model, scale, sampling_batch_size,
                                num_samples_per_class=num_samples_per_class, device='cuda')
    else:
        raise NotImplementedError(f"Condition loader for '{model_type}' not implemented.")
