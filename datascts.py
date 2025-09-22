import numpy as np
import torch

from thsolver import Dataset
from ocnn.octree import Points


def collate_batch(batch : list):
    outputs = {}
    for key in batch[0].keys():
        outputs[key] = [b[key] for b in batch]
    return outputs


class ShapeNetTransform:
    def __init__(self, noise=0.0, take=8192, random=True, missing=False, has_label=True, has_normal=True, has_vp=True):
        self.noise = noise
        self.random = random
        self.take = take
        self.missing = missing
        self.has_label = has_label
        self.has_normal = has_normal
        self.has_vp = has_vp

    def transform(self, sample: dict, idx: int):
        #random sample N points, num_rand is a random number between 2048 and 8192
        if self.random:
            take = torch.randint(2048, self.take, (1,))
            take = take if take < sample['points'].shape[0] else sample['points'].shape[0]
        else:
            take = self.take if self.take < sample['points'].shape[0] else sample['points'].shape[0]
        idx = torch.randperm(sample['points'].shape[0])[:take]
        if self.missing and self.has_label:
            #for fisrt 8 channel of dim 1 of labels, get idx of those at least one channel is 1
            idx = torch.where(sample['labels'][idx, :3].sum(dim=1) > 0)[0]
        xyz = sample['points'][idx]
        if self.has_normal:
            normal = sample['normals'][idx]
        if self.has_label:
            labels = sample['labels'][idx]
        if self.has_vp:
            vp = sample['vp']
        #add noise
        if self.noise > 0:
            noise = torch.randn_like(xyz)
            noise = noise / torch.norm(noise, dim=1, keepdim=True)
            noise = noise * self.noise * torch.rand((xyz.shape[0], 1))
            xyz = xyz + noise
        vd = xyz.unsqueeze(1) - vp.unsqueeze(0)
        vd = vd / torch.norm(vd, dim=2, keepdim=True)
        points = Points(xyz, normal)
        # !NOTE: Normalize the points into one unit sphere in [-0.8, 0.8]
        bbmin, bbmax = points.bbox()
        points.normalize(bbmin, bbmax, scale=0.8)
        
        return {'points': points, 'viewdirs': vd, 'labels': labels, 'vps': vp}

    def __call__(self, sample: dict, idx: int):
        return self.transform(sample, idx)


class ReadNpz:
  def __init__(self, has_normal: bool = True, has_label: bool = False):
    self.has_normal = has_normal
    self.has_label = has_label

  def __call__(self, filename: str):
    raw = np.load(filename)

    output = dict()
    output['points'] = torch.from_numpy(raw['vertices']).float().view(-1, 3)
    output["vp"] = torch.from_numpy(raw['viewpoints'][:16, :]).float().view(-1,3)

    if self.has_normal:
      output['normals'] = torch.from_numpy(raw['normals']).float().view(-1, 3)
    if self.has_label:
        output['labels'] = torch.from_numpy(raw['visibilities'][:, :16]).long().unsqueeze(2)
    
    return output

def get_shapenet_dataset(flags):
    transform = ShapeNetTransform(noise=flags.noise, take=flags.takes, random=flags.random, missing=flags.missing)
    read_ply = ReadNpz(has_normal=True, has_label=True)
    dataset = Dataset(flags.location, flags.filelist, transform,
                      read_file=read_ply, take=flags.take)
    return dataset, collate_batch

def generate_camera_matrices(vp):
    """
    Generate camera matrices K, R, and T for a camera pointing towards the origin.
    
    Parameters:
        camera_center (tuple): The 3D coordinates of the camera center (x, y, z).
    
    Returns:
        K (numpy.ndarray): The intrinsic camera matrix.
        R (numpy.ndarray): The rotation matrix.
        T (numpy.ndarray): The translation vector.
    """
    x, y, z = vp[0], vp[1], vp[2]
    
    # 1. Intrinsic camera matrix K
    # Assuming unit focal length and principal point at (0, 0)
    # K = np.eye(3)  # Identity matrix for simplicity
    K = np.array([[537, 0, 240], [0, 537, 240], [0, 0, 1]])
    
    # 2. Rotation matrix R
    # Define the z-axis direction vector pointing towards the origin
    z_axis = np.array([-x, -y, -z])
    z_axis_norm = z_axis / np.linalg.norm(z_axis)
    
    # Define an orthogonal vector in the x-y plane
    if x == 0 and y == 0:
        x_axis = np.array([1, 0, 0])  # Avoid division by zero
    else:
        x_axis = np.array([y, -x, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Compute the third orthogonal vector using cross product
    y_axis = np.cross(z_axis_norm, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Construct the rotation matrix R
    R = np.column_stack((x_axis, y_axis, z_axis_norm))
    
    # 3. Translation vector T
    T = -np.dot(R, np.array([x, y, z]))
    
    return K, R, T