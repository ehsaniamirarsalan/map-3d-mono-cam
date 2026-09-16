"""Export aligned instance boxes from licensed ScanNet meshes/segment annotations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from mapdet3d.data.scannet.dataset import PAPER_SCENES,LABEL_IDS


def prepare_scene(scene_dir,label_map,output,scannet200=False):
    scene_dir,output=Path(scene_dir),Path(output)
    scene=scene_dir.name
    vertices=PlyData.read(scene_dir/f'{scene}_vh_clean_2.ply')['vertex']
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')])
    alignment=np.eye(4)
    for line in (scene_dir/f'{scene}.txt').read_text().splitlines():
        if line.startswith('axisAlignment'):
            alignment=np.array([float(v) for v in line.split('=',1)[1].split()]).reshape(4,4)
    xyz=xyz@alignment[:3,:3].T+alignment[:3,3]
    segments=np.array(json.loads((scene_dir/f'{scene}_vh_clean_2.0.010000.segs.json').read_text())['segIndices'])
    groups=json.loads((scene_dir/f'{scene}.aggregation.json').read_text())['segGroups']
    with open(label_map,newline='') as handle:
        labels={row['raw_category']:int(row['id' if scannet200 else 'nyu40id']) for row in csv.DictReader(handle,delimiter='\t')}
    ids=LABEL_IDS['SCANNET200_OBJ_CLASS_IDS' if scannet200 else 'SCANNET_OBJ_CLASS_IDS']
    boxes=[]
    for group in groups:
        label=labels.get(group['label'],0)
        if label not in ids:
            continue
        points=xyz[np.isin(segments,group['segments'])]
        if not len(points):
            raise ValueError(f'Empty instance in {scene}: {group["objectId"]}')
        lo,hi=points.min(0),points.max(0)
        boxes.append(np.r_[(lo+hi)/2,hi-lo,label])
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/f'{scene}_aligned_bbox.npy',np.array(boxes,dtype=np.float32).reshape(-1,7))
    np.save(output/f'{scene}_axis_align_matrix.npy',alignment.astype(np.float32))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--label-map',required=True)
    parser.add_argument('--scenes',default=str(PAPER_SCENES))
    parser.add_argument('--scannet200',action='store_true')
    args=parser.parse_args()
    output=Path(args.root)/('scannet200_instance_data' if args.scannet200 else 'scannet_instance_data')
    for scene in Path(args.scenes).read_text().split():
        prepare_scene(Path(args.root)/'data'/scene,args.label_map,output,args.scannet200)
        print(scene)

if __name__=='__main__':
    main()
