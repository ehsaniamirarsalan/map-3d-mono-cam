"""Evaluate CA-1M frames, ScanNet200 frames, or ScanNetV2 tracked scenes."""
import argparse
import json
from pathlib import Path
from mapdet3d.configs.real_config import build_real_model,config_from_checkpoint,select_device
from mapdet3d.engine.checkpoint import read_checkpoint,load_checkpoint
from mapdet3d.inference.runtime import StreamingPredictor
from mapdet3d.data.ca1m.dataset import CA1MWindowDataset
from mapdet3d.data.scannet.dataset import ScanNetDataset,PAPER_SCENES
from mapdet3d.eval.real import evaluate_frames,evaluate_scenes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['ca1m','scannet200','scannet-scene'],required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--scenes',default=str(PAPER_SCENES))
    p.add_argument('--frame-stride',type=int,default=25)
    p.add_argument('--max-detections',type=int,default=100)
    p.add_argument('--score-threshold',type=float,default=0.)
    p.add_argument('--output',default='output/metrics.json')
    args=p.parse_args()
    model=build_real_model(config_from_checkpoint(read_checkpoint(args.checkpoint)))
    load_checkpoint(args.checkpoint,model)
    predictor=StreamingPredictor(model,select_device(args.device),max_detections=args.max_detections,score_threshold=args.score_threshold)
    if args.dataset=='ca1m':
        result=evaluate_frames(predictor,CA1MWindowDataset(args.data).iter_frames())
    else:
        dataset=ScanNetDataset(args.data,args.scenes,args.dataset=='scannet200',args.frame_stride)
        result=evaluate_scenes(predictor,dataset.iter_scenes()) if args.dataset=='scannet-scene' else evaluate_frames(predictor,dataset.iter_frames())
    result.update(dataset=args.dataset,checkpoint=str(args.checkpoint))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
