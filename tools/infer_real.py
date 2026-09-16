"""Export causal metric detections from an image folder or video."""
import argparse
import json
from pathlib import Path
from mapdet3d.configs.real_config import ModelConfig,build_real_model,config_from_checkpoint,select_device
from mapdet3d.engine.checkpoint import read_checkpoint,load_checkpoint
from mapdet3d.inference.runtime import StreamingPredictor,iter_images,serialize_prediction,save_overlay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image-folder')
    source.add_argument('--video')
    p.add_argument('--metadata',help='JSON keyed by image filename or video frame index; intrinsics and camera_poses are optional')
    p.add_argument('--checkpoint')
    p.add_argument('--allow-untrained',action='store_true')
    p.add_argument('--pretrained',default='facebook/map-anything')
    p.add_argument('--device',default='auto')
    p.add_argument('--window',type=int,default=5)
    p.add_argument('--output',default='output/detections.jsonl')
    p.add_argument('--overlays')
    p.add_argument('--score-threshold',type=float,default=.25)
    p.add_argument('--max-detections',type=int,default=100)
    args=p.parse_args()
    if not args.checkpoint and not args.allow_untrained:
        p.error('A trained --checkpoint is required. Use --allow-untrained only for wiring checks.')
    config=config_from_checkpoint(read_checkpoint(args.checkpoint)) if args.checkpoint else ModelConfig(pretrained=args.pretrained)
    model=build_real_model(config)
    if args.checkpoint:
        load_checkpoint(args.checkpoint,model)
    else:
        print('UNTRAINED DETECTOR: outputs do not represent useful object detections.')
    predictor=StreamingPredictor(model,select_device(args.device),args.window,args.max_detections,args.score_threshold)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as handle:
        for index,view in enumerate(iter_images(args.image_folder or args.video,args.metadata)):
            pred=predictor.step(view)
            record=serialize_prediction(pred,view)
            record['untrained']=not bool(args.checkpoint)
            handle.write(json.dumps(record,allow_nan=False)+'\n')
            if args.overlays:
                save_overlay(Path(args.overlays)/f'{index:06d}.png',view,pred)
            print(f'{view["frame_id"]}: {len(pred["scores"])} detections')
    print(f'Saved {output}')

if __name__=='__main__':
    main()
