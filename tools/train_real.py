"""Train the paper-first model on local CA-1M captures. See README."""
from __future__ import annotations
import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from mapdet3d.configs.real_config import ModelConfig,build_real_model,build_optimizer,select_device,config_from_checkpoint
from mapdet3d.data.ca1m.dataset import CA1MWindowDataset,resolve_sources
from mapdet3d.data.ca1m.collate import CA1MCollator
from mapdet3d.engine.steps import train_steps
from mapdet3d.engine.checkpoint import read_checkpoint
from mapdet3d.losses.criterion import SetCriterion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--effective-batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--window-size',type=int,default=5)
    p.add_argument('--num-queries',type=int,default=900)
    p.add_argument('--num-decoder-layers',type=int,default=6)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--device',default='auto')
    p.add_argument('--pretrained',default='facebook/map-anything')
    p.add_argument('--checkpoint-dir',default='runs/ca1m')
    p.add_argument('--checkpoint-every',type=int,default=5000)
    p.add_argument('--resume')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--rgb-only',action='store_true')
    p.add_argument('--no-poses',action='store_true')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--no-activation-checkpointing',action='store_true')
    args=p.parse_args()
    device=select_device(args.device)
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if world>1:
        if device.type=='cuda':
            device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    divisor=args.batch_size*world
    if divisor<1 or args.effective_batch_size%divisor:
        p.error('effective-batch-size must be divisible by batch-size times world size')
    sources=resolve_sources(args.data)
    if len(sources)<world*max(args.workers,1):
        p.error('Provide at least one capture per rank/worker, or reduce workers')
    config=ModelConfig(pretrained=args.pretrained,num_queries=args.num_queries,num_decoder_layers=args.num_decoder_layers,
                       use_intrinsics=not args.rgb_only,use_poses=not(args.rgb_only or args.no_poses),
                       activation_checkpointing=not args.no_activation_checkpointing)
    if args.resume:
        config=config_from_checkpoint(read_checkpoint(args.resume))
    model=build_real_model(config).to(device)
    optimizer=build_optimizer(model,args.lr)
    if world>1:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=True)
    base=model.module if world>1 else model
    dataset=CA1MWindowDataset(sources,window_size=args.window_size,rank=rank,world_size=world)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,
                      collate_fn=CA1MCollator(base.preprocess_config,args.window_size),drop_last=True)
    train_steps(model,SetCriterion(),loader,optimizer,device,max_steps=args.steps,
                accumulation=args.effective_batch_size//divisor,checkpoint_dir=args.checkpoint_dir,
                checkpoint_every=args.checkpoint_every,resume=args.resume,amp=not args.no_amp,
                log_fn=(lambda step,losses:print(f'step {step}: loss={losses["loss_total"]:.5f}',flush=True)) if rank==0 else None)
    if world>1:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
