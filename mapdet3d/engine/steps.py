"""Optimizer-step training with accumulation, resume and complete checkpoints."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from mapdet3d.engine.train_loop import views_batch_step
from mapdet3d.engine.checkpoint import load_checkpoint, save_checkpoint


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())


def restore_data_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])


def train_steps(model,criterion,loader,optimizer,device,max_steps=100000,accumulation=64,
                checkpoint_dir=None,checkpoint_every=5000,resume=None,amp=True,log_fn=None):
    if max_steps<1 or accumulation<1 or checkpoint_every<1:
        raise ValueError('Steps, accumulation and checkpoint interval must be positive')
    module = model.module if hasattr(model,'module') else model
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_steps)
    enabled = amp and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda',enabled=enabled)
    start = 0
    epoch, consumed = 0,0
    epoch_rng = rng_state()
    iterator = None
    if resume:
        state = load_checkpoint(resume,module,optimizer,scheduler,scaler,restore_rng=True)
        start = state['step']
        data = state.get('data_state') or {}
        if data.get('max_steps',max_steps)!=max_steps or data.get('accumulation',accumulation)!=accumulation:
            raise ValueError('Resume requires the same schedule length and accumulation')
        if dist.is_initialized() and 'ranks' in data:
            if len(data['ranks']) != dist.get_world_size():
                raise ValueError('Resume requires the original distributed world size')
            data = data['ranks'][dist.get_rank()]
        epoch,consumed = data.get('epoch',0),data.get('consumed',0)
        epoch_rng = data.get('epoch_rng',rng_state())
        checkpoint_rng = data.get('current_rng',rng_state())
        restore_data_rng(epoch_rng)
        iterator = iter(loader)
        for _ in range(consumed):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError('Training data changed since checkpoint') from exc
        restore_data_rng(checkpoint_rng)
    if iterator is None:
        iterator = iter(loader)
    history = []
    model.train()
    for step in range(start,max_steps):
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                if consumed == 0:
                    raise ValueError('Training loader is empty; check data and worker/rank shard counts')
                epoch += 1
                consumed = 0
                epoch_rng = rng_state()
                iterator = iter(loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError('Training loader cannot be restarted') from exc
            consumed += 1
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=enabled):
                losses = views_batch_step(model,criterion,batch,device)
                loss = losses['loss_total']/accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at step {step+1}')
            scaler.scale(loss).backward()
            for k,v in losses.items():
                totals[k] = totals.get(k,0.)+float(v.detach())/accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.,error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        history.append(totals)
        if log_fn:
            log_fn(step+1,totals)
        if checkpoint_dir and ((step+1)%checkpoint_every==0 or step+1==max_steps):
            data = dict(epoch=epoch,consumed=consumed,epoch_rng=epoch_rng,current_rng=rng_state(),
                        max_steps=max_steps,accumulation=accumulation)
            rank = dist.get_rank() if dist.is_initialized() else 0
            if dist.is_initialized():
                rank_states = [None]*dist.get_world_size()
                dist.all_gather_object(rank_states,data)
                data = dict(ranks=rank_states,max_steps=max_steps,accumulation=accumulation)
            if rank == 0:
                save_checkpoint(Path(checkpoint_dir)/f'step_{step+1:06d}.pt',module,
                                optimizer,scheduler,scaler,step+1,data_state=data)
    return history
