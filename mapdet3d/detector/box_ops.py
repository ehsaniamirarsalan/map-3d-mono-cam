"""Stable normalized box refinement."""
import torch


def inverse_sigmoid(x, eps=1e-6):
    return torch.logit(x.clamp(eps, 1 - eps))
