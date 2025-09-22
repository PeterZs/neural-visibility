import os
import numpy as np
from tqdm import tqdm
from numba import cuda
import trimesh
import math


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='')
parser.add_argument('--out_path', type=str, default='data')
args = parser.parse_args()
directory = args.in_path
# Mesh Data Format
'''
|directory
|---[category_code]
|---|---[model_code].obj
'''

# Visibility Data Format
'''
|target_directory
|---[category_code]
|---|---[model_code]
|---|---|---data.npz
|---|---|---|---vertices: [(n, 3)-np.float32] (n <= max_npoint)
|---|---|---|---visibilities [(n, n_vp)-bool] (n <= max_npoint)
|---|---|---|---normals [(n, 3)-np.float32] (n <= max_npoint)
|---|---|---|---viewpoints [(n_vp, 3)-np.float32]
'''

# max_npoint : target point number of downsampling
# (Downsampling is not performed for pointclouds with more than this number of points)

# n_vp: number of viewpoints for each pointcloud
target_directory = args.out_path
# max_npoint = 8192
n_vp = 16

@cuda.jit
def gpu_intersect(V, F, f, Vp, n_v, n_f, n_vp, recorder):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v * n_f * n_vp:
        v_idx = idx // (n_f * n_vp)
        f_idx = (idx - n_f * n_vp * v_idx) // n_vp
        vp_idx = idx - n_f * n_vp * v_idx - n_vp * f_idx
        o = Vp[vp_idx]
        p = V[v_idx]
        v0 = F[f_idx, 0]
        v1 = F[f_idx, 1]
        v2 = F[f_idx, 2]
        # s = o - v0
        s = cuda.local.array(3, dtype=np.float32)
        s[0] = o[0] - v0[0]
        s[1] = o[1] - v0[1]
        s[2] = o[2] - v0[2]
        # e1 = v1 - v0
        e1 = cuda.local.array(3, dtype=np.float32)
        e1[0] = v1[0] - v0[0]
        e1[1] = v1[1] - v0[1]
        e1[2] = v1[2] - v0[2]
        # e2 = v2 - v0
        e2 = cuda.local.array(3, dtype=np.float32)
        e2[0] = v2[0] - v0[0]
        e2[1] = v2[1] - v0[1]
        e2[2] = v2[2] - v0[2]
        # d = p - o
        d = cuda.local.array(3, dtype=np.float32)
        d[0] = p[0] - o[0]
        d[1] = p[1] - o[1]
        d[2] = p[2] - o[2]
        
        d_len = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        d[0] = d[0] / d_len
        d[1] = d[1] / d_len
        d[2] = d[2] / d_len
        s1 = cuda.local.array(3, dtype=np.float32)
        s2 = cuda.local.array(3, dtype=np.float32)
        # s1 = d x e2
        # s2 = S x e1
        s1[0] = d[1] * e2[2] - d[2] * e2[1]
        s1[1] = d[2] * e2[0] - d[0] * e2[2]
        s1[2] = d[0] * e2[1] - d[1] * e2[0]
        s2[0] = s[1] * e1[2] - s[2] * e1[1]
        s2[1] = s[2] * e1[0] - s[0] * e1[2]
        s2[2] = s[0] * e1[1] - s[1] * e1[0]
        t = (s2[0] * e2[0] + s2[1] * e2[1] + s2[2] * e2[2]) / (s1[0] * e1[0] + s1[1] * e1[1] + s1[2] * e1[2])
        u = (s1[0] * s[0] + s1[1] * s[1] + s1[2] * s[2]) / (s1[0] * e1[0] + s1[1] * e1[1] + s1[2] * e1[2])
        v = (s2[0] * d[0] + s2[1] * d[1] + s2[2] * d[2]) / (s1[0] * e1[0] + s1[1] * e1[1] + s1[2] * e1[2])
        if t >= 1e-5 and d_len - t >= 1e-5 and u >= 0.0 and v >= 0.0 and (1 - u - v) >= 0.0:
            recorder[v_idx, vp_idx] = False

def gt_visibility(V, F, f, Vp):
    V_device = cuda.to_device(V)
    F_device = cuda.to_device(F)
    f_device = cuda.to_device(f)
    Vp_device = cuda.to_device(Vp)
    recorder_device = cuda.to_device(np.ones((V.shape[0], Vp.shape[0]), dtype=bool))
    gpu_intersect[(V.shape[0] * F.shape[0] * Vp.shape[0]) // 1024 + 1 if(V.shape[0] * F.shape[0] * Vp.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](V_device, F_device, f_device, Vp_device, V.shape[0], F.shape[0], Vp.shape[0], recorder_device)
    cuda.synchronize()
    result = recorder_device.copy_to_host()
    return result


catagories = [file for file in os.listdir(os.path.join(directory)) if os.path.isdir(os.path.join(directory, file))]
n_model = 0
models = []
for c in catagories:
    models.append([file for file in os.listdir(os.path.join(directory, c)) if os.path.isdir(os.path.join(directory, c, file))])
    n_model += len(models[-1])

pbar = tqdm(total=n_model, ncols=80)
for idx, c in enumerate(catagories):
    for m in models[idx]:
        if os.path.exists(os.path.join(target_directory, c, m, 'data.npz')):
            pbar.update(1)
            continue
        mesh = trimesh.load(os.path.join(directory, c, m, 'model_normalized.obj'))
        V = np.asarray(mesh.vertices, dtype=np.float32)
        try:
            f = np.asarray(mesh.faces, dtype=np.int32)
        except:
            pbar.update(1)
            continue
        F = np.zeros((f.shape[0], 3, 3), dtype=np.float32)
        F = V[f]
       
        new_V, f_idx = mesh.sample(100000, return_index=True)
        new_V = np.asarray(new_V, dtype=np.float32)
        # new_idx = fps(new_V.reshape(1,-1,3), max_npoint)
        # new_V = new_V[new_idx[0,:]]
        #get normal of new_V for f_idx
        normals = mesh.face_normals[f_idx]
        new_normals = normals
        # new_normals = normals[new_idx[0,:]]

        radius = np.linalg.norm(new_V.max(axis=0) - new_V.min(axis=0))

        x = np.random.randn(n_vp)
        y = np.random.randn(n_vp)
        z = np.random.randn(n_vp)
        Vp = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1, dtype=np.float32)
        Vp = Vp / np.linalg.norm(Vp, axis=1).reshape(-1, 1)
        Vp = Vp * radius
        new_V_visibility = gt_visibility(new_V, F, f, Vp)

        if not os.path.exists(os.path.join(target_directory, c, m)):
            os.makedirs(os.path.join(target_directory, c, m))
        np.savez(os.path.join(target_directory, c, m, 'data.npz'), vertices=new_V, visibilities=new_V_visibility, viewpoints=Vp, normals=new_normals)
        pbar.update(1)