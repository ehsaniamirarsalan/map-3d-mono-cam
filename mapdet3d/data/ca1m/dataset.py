"""Chronological, capture-sharded CA-1M RGB windows from local tar archives.

Reads only RGB, intrinsics, registered poses and annotations. In particular this
avoids the pinned upstream reader's undefined empty_box/box_type empty branch
and its unnecessary depth decoding. gt/RT is the registered camera-to-world pose.
"""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
from collections import deque
import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from mapdet3d.data.ca1m.annotations import project_corners_to_2d_box
from mapdet3d.utils.geometry import corners_from_box


def resolve_sources(source):
    if isinstance(source, (list, tuple)):
        paths = [str(p) for p in source]
    else:
        p = Path(source)
        if p.is_dir():
            paths = [str(x) for x in sorted(p.glob('*.tar'))]
        elif p.suffix == '.txt':
            paths = [str(p.parent / line.strip()) if not Path(line.strip()).is_absolute() and '://' not in line else line.strip()
                     for line in p.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        else:
            paths = [str(p)]
    if not paths:
        raise FileNotFoundError(f'No CA-1M tar archives found in {source}')
    for path in paths:
        if '://' in path or not Path(path).is_file():
            raise FileNotFoundError(f'CA-1M archive is not local: {path}. Download the capture tar files first; see README.')
    return paths


def decode_annotations(items, intrinsics, width, height):
    center = torch.tensor([b['position'] for b in items], dtype=torch.float32).reshape(-1,3)
    dims = torch.tensor([b['scale'] for b in items], dtype=torch.float32).reshape(-1,3)
    rot = torch.tensor([b['R'] for b in items], dtype=torch.float32).reshape(-1,3,3)
    corners = corners_from_box(center,dims,rot)
    boxes = torch.stack([project_corners_to_2d_box(c,intrinsics,width,height) for c in corners]) if len(items) else torch.empty(0,4)
    return dict(center=center,dims=dims,rot=rot,boxes2d=boxes)


def read_capture(path):
    with tarfile.open(path, 'r:*') as archive:
        groups = {}
        for member in archive.getmembers():
            if not member.isfile() or '.' not in member.name:
                continue
            key, suffix = member.name.split('.',1)
            video, stamp = key.rsplit('/',1)
            if stamp == 'world':
                continue
            try:
                timestamp = int(stamp)
            except ValueError:
                continue
            groups.setdefault((video,timestamp), {})[suffix.lower().rsplit('.',1)[0]] = member
        for (video,stamp), members in sorted(groups.items(),key=lambda item:(item[0][0],item[0][1])):
            if 'wide/image' not in members:
                continue
            def read(key):
                return archive.extractfile(members[key]).read()
            image = np.array(Image.open(io.BytesIO(read('wide/image'))).convert('RGB'))
            K = torch.tensor(json.loads(read('wide/image/k')),dtype=torch.float32).reshape(3,3)
            target = decode_annotations(json.loads(read('wide/instances')),K,image.shape[1],image.shape[0])
            view = dict(img=image,intrinsics=K,scene_id=video,timestamp=stamp/1e9,frame_id=str(stamp))
            if 'gt/rt' in members:
                pose = torch.tensor(json.loads(read('gt/rt')),dtype=torch.float32).reshape(4,4)
                if not torch.isfinite(pose).all():
                    raise ValueError(f'Nonfinite pose in {path}: {stamp}')
                view['camera_poses'] = pose
            yield dict(view=view,target=target)


class CA1MWindowDataset(IterableDataset):
    def __init__(self, source, window_size=5, max_buffer_per_video=256,
                 fps_range=(2,10), rank=0, world_size=1):
        super().__init__()
        if window_size < 1 or fps_range[0] <= 0 or fps_range[1] < fps_range[0]:
            raise ValueError('Invalid window size or FPS range')
        self.source, self.window_size = source,window_size
        self.max_buffer_per_video = max_buffer_per_video
        self.fps_range = fps_range
        self.rank,self.world_size = rank,world_size

    def iter_frames(self):
        paths = resolve_sources(self.source)
        worker = get_worker_info()
        nworkers = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        # Whole captures belong to exactly one worker/rank.
        worker_id = self.rank*nworkers+wid
        for path in paths[worker_id::self.world_size*nworkers]:
            yield from read_capture(path)

    def __iter__(self):
        buffer = deque(maxlen=self.max_buffer_per_video)
        scene = None
        for frame in self.iter_frames():
            view = frame['view']
            if view['scene_id'] != scene:
                buffer.clear()
                scene = view['scene_id']
            if buffer and view['timestamp'] <= buffer[-1]['view']['timestamp']:
                raise ValueError(f'Non-increasing timestamps for capture {scene}')
            buffer.append(frame)
            duration = (self.window_size-1)/self.fps_range[0]
            while len(buffer)>1 and buffer[1]['view']['timestamp'] < view['timestamp']-duration:
                buffer.popleft()
            yield {'history':list(buffer)}
