"""Deterministic oriented cuboid IoU via half-space intersection.

Intersection vertices are feasible intersections of triples of the twelve face
planes. SciPy computes their convex-hull volume. Computation is float64 on CPU;
results return on the input device. This is an evaluation operation, not a loss.
"""
from itertools import combinations
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError
from mapdet3d.utils.geometry import corners_from_box

_TRIPLES = np.array(list(combinations(range(12),3)))


def box3d_iou(centers_a, dims_a, rots_a, centers_b, dims_b, rots_b):
    output = centers_a.new_zeros((len(centers_a),len(centers_b)))
    if not output.numel():
        return output
    arrays = [x.detach().cpu().double().numpy() for x in (centers_a,dims_a,rots_a,centers_b,dims_b,rots_b)]
    ca,da,ra,cb,db,rb = arrays
    if any(not np.isfinite(x).all() for x in arrays) or np.any(da<=0) or np.any(db<=0):
        raise ValueError('IoU requires finite boxes with positive dimensions')
    corners = [corners_from_box(c,d,r).detach().cpu().double().numpy() for c,d,r in ((centers_a,dims_a,rots_a),(centers_b,dims_b,rots_b))]
    alo,ahi = corners[0].min(1),corners[0].max(1)
    blo,bhi = corners[1].min(1),corners[1].max(1)
    result = np.zeros(output.shape)
    for i in range(len(ca)):
        for j in range(len(cb)):
            if np.any(np.minimum(ahi[i],bhi[j]) <= np.maximum(alo[i],blo[j])):
                continue
            # Shift and normalize to improve conditioning for large world coordinates.
            origin = (ca[i]+cb[j])/2
            scale = max(da[i].max(),db[j].max())
            normals = np.concatenate((ra[i].T,-ra[i].T,rb[j].T,-rb[j].T))
            offsets = np.concatenate((da[i]/2,da[i]/2,db[j]/2,db[j]/2))/scale
            offsets[:6] += normals[:6] @ ((ca[i]-origin)/scale)
            offsets[6:] += normals[6:] @ ((cb[j]-origin)/scale)
            matrices = normals[_TRIPLES]
            valid = np.abs(np.linalg.det(matrices)) > 1e-10
            vertices = np.linalg.solve(matrices[valid], offsets[_TRIPLES[valid]][...,None])[...,0]
            vertices = vertices[np.all(vertices @ normals.T <= offsets + 1e-9,axis=1)]
            vertices = np.unique(np.round(vertices,12),axis=0)
            if len(vertices)<4:
                continue
            try:
                inter = ConvexHull(vertices).volume * scale**3
            except QhullError:
                # Coplanar/collinear contact has zero volume; no random jitter.
                continue
            va,vb = da[i].prod(),db[j].prod()
            inter = np.clip(inter,0,min(va,vb))
            result[i,j] = inter/(va+vb-inter)
    return torch.as_tensor(result,device=output.device,dtype=output.dtype)


def box3d_iou_montecarlo(centers_a,dims_a,rots_a,centers_b,dims_b,rots_b,num_samples=20000,generator=None):
    """Compatibility alias. Sampling arguments are ignored; IoU is now exact."""
    return box3d_iou(centers_a,dims_a,rots_a,centers_b,dims_b,rots_b)
