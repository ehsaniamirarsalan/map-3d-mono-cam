"""Batch-wide random view count, temporal rate and aspect bucket."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from mapdet3d.data.preprocess import PreprocessConfig, ASPECT_RATIOS, collate_windows


def select_history(history, count, fps):
    selected = [len(history)-1]
    for k in range(1,count):
        wanted = history[-1]['view']['timestamp'] - k/fps
        candidates = range(selected[-1])
        if not selected[-1] or wanted < history[0]['view']['timestamp'] - 1e-6:
            break
        selected.append(min(candidates,key=lambda i:abs(history[i]['view']['timestamp']-wanted)))
    return [history[i] for i in reversed(selected)]


@dataclass
class CA1MCollator:
    config: PreprocessConfig = field(default_factory=PreprocessConfig)
    window_size: int = 5
    fps_range: tuple = (2,10)
    training: bool = True

    def __call__(self,batch):
        count = random.randint(1,self.window_size) if self.training else self.window_size
        fps = random.uniform(*self.fps_range) if self.training else self.fps_range[1]
        chosen = [select_history(s['history'],count,fps) for s in batch]
        # Startup windows contain fewer frames. Shorten the batch consistently.
        count = min(map(len,chosen))
        windows = [dict(views=[f['view'] for f in s[-count:]],targets=[f['target'] for f in s[-count:]]) for s in chosen]
        hw = self.config.bucket(random.choice(ASPECT_RATIOS)) if self.training else None
        return collate_windows(windows,self.config,hw)


def collate_ca1m_batch(batch):
    """Deterministic collation of already sampled windows (targets per view)."""
    return collate_windows(batch,PreprocessConfig())
