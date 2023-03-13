import cv2
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def draw_kpts(img, kpts, r=5, thickness=5, color=(255,0,0)):
    if isinstance(img, np.ndarray):
        img = img.copy().astype(np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.copy().astype(np.uint8)
        
    for kpt in kpts:
        if len(kpt)>2:
            x, y, c = kpt
        else:
            x, y = kpt
            c = 1

        if c > 0:
            cv2.circle(img, (int(x), int(y)), r, color, thickness)

    return img



### Save for visualization
def save_ply(vert, face=None, filename='file.ply'):
    # Vertices
    if isinstance(vert, np.ndarray):
        vert = vert.tolist()
    vert = [tuple(v) for v in vert]
    vert = np.array(vert, dtype=[('x', 'f4'), 
                                 ('y', 'f4'), 
                                 ('z', 'f4')])
    vert = PlyElement.describe(vert, 'vertex')
    
    # Faces
    if face is not None:
        if isinstance(face, np.ndarray):
            face = face.tolist()
        face = [(face[i], 255, 255, 255) for i in range(len(face))]
        face = np.array(face, dtype=[('vertex_indices', 'i4', (3,)),
                                     ('red', 'u1'),
                                     ('green', 'u1'),
                                     ('blue', 'u1')])
        face = PlyElement.describe(face, 'face')
    
    # Save
    if face is not None:
        with open(filename, 'wb') as f:
            PlyData([vert, face]).write(f)
    else:
        with open(filename, 'wb') as f:
            PlyData([vert]).write(f)


def read_ply(plyfile):
    plydata = PlyData.read(plyfile)
    v = plydata['vertex'].data
    v = [list(i) for i in v]
    v = np.array(v)
    f = plydata['face'].data
    f = [list(i) for i in f]
    f = np.array(f).squeeze()
    return v, f


