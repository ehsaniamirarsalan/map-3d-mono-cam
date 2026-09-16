"""Versioned full-model checkpoints. Load only trusted training artifacts."""
from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import torch

VERSION = 2


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, step=0, config=None, data_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    state = dict(version=VERSION,model_state=model.state_dict(),step=step,
                 config=config or getattr(model,'config',{}),data_state=data_state,
                 optimizer_state=optimizer.state_dict() if optimizer else None,
                 scheduler_state=scheduler.state_dict() if scheduler else None,
                 scaler_state=scaler.state_dict() if scaler else None,
                 rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)


def read_checkpoint(path):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version') != VERSION:
        raise ValueError('Incompatible checkpoint: expected version 2 with full backbone and detector state. Legacy core-only and official checkpoints cannot be resumed.')
    return state


def load_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,restore_rng=False):
    state = read_checkpoint(path)
    expected = getattr(model,'config',None)
    if expected is not None and state['config'] != expected:
        raise ValueError('Checkpoint model/preprocessing configuration does not match the constructed model')
    try:
        model.load_state_dict(state['model_state'],strict=True)
    except RuntimeError as exc:
        raise ValueError('Checkpoint architecture does not match the model') from exc
    for obj,key in ((optimizer,'optimizer_state'),(scheduler,'scheduler_state'),(scaler,'scaler_state')):
        if obj is not None:
            if state[key] is None:
                raise ValueError(f'Checkpoint has no {key}; it is not resumable')
            obj.load_state_dict(state[key])
    if restore_rng:
        rng = state['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    return state
