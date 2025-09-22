import os
import torch
import ocnn
from tqdm import tqdm
from thsolver import Solver
import numpy as np

from datascts import get_shapenet_dataset, generate_camera_matrices
from models import MyNet, get_embedder


class VisSolver(Solver):
    def __init__(self, flags, is_master=True):
        super().__init__(flags, is_master)
        self.embedder, vd_ch = get_embedder(flags.MODEL.multires)
    
    def get_model(self, flags):
        model = MyNet(flags.channel, flags.vp_ch, flags.nout)
        return model
    
    def get_dataset(self, flags):
        return get_shapenet_dataset(flags)
    
    def get_input_feature(self, octree, flags):
        depth = octree.depth
        f = flags.feature
        f = f.upper()
        features = []
        if 'N' in f:
            features.append(octree.normals[depth])
        if 'L' in f or 'D' in f:
            local_points = octree.points[depth].frac() - 0.5
        if 'L' in f:
            features.append(local_points)
        if 'D' in f:
            dis = torch.sum(local_points * octree.normals[depth], dim=1, keepdim=True)
            features.append(dis)
        if 'P' in f:
            scale = 2 ** (1 - depth)   # normalize [0, 2^depth] -> [-1, 1]
            global_points = octree.points[depth] * scale - 1.0
            features.append(global_points)
        out = torch.cat(features, dim=1)
        return out

    def process_batch(self, batch, flags):
        def points2octree(points):
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['points'] = ocnn.octree.merge_points(points)
        batch['octree'] = octree
        viewdirs = [vd.cuda() for vd in batch['viewdirs']]
        batch['viewdirs'] = torch.cat(viewdirs, dim=0)
        labels = [label.cuda() for label in batch['labels']]
        batch['labels'] = torch.cat(labels, dim=0).reshape(-1,1)
        return batch
    
    def model_forward(self, batch):
        octree, points, viewdirs, labels = batch['octree'], batch['points'], batch['viewdirs'], batch['labels']
        input_feature = self.get_input_feature(batch['octree'], self.FLAGS.MODEL)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        viewdirs = self.embedder(viewdirs.view(-1, 3))
        visibilities = self.model(input_feature, octree, octree.depth, query_pts, viewdirs)
        return visibilities, labels.squeeze(1)
    
    def train_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        visibilities, labels = self.model_forward(batch)
        loss = self.loss_function(visibilities, labels)
        accu = self.accuracy(visibilities, labels)
        return {'train/loss': loss, 'train/accu': accu}
    
    def test_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            visibilities, labels = self.model_forward(batch)
        loss = self.loss_function(visibilities, labels)
        accu = self.accuracy(visibilities, labels)
        return {'test/loss': loss, 'test/accu': accu}
    
    def eval_step(self, batch):
        pass

    def loss_function(self, visibilities, labels):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(visibilities, labels)
        return loss
    
    def accuracy(self, visibilities, labels):
        pred = visibilities.argmax(dim=-1)
        correct = pred.eq(labels).float().mean()
        return correct
    
    def best_view(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()
        batch = next(self.test_iter)
        batch = self.process_batch(batch, self.FLAGS.DATA.test)

        vp = torch.tensor([1.0001, -1.10001, 1.1001]).cuda().requires_grad_()
        optimizer = torch.optim.AdamW([vp], lr=0.005)
        radium = 4.0

        octree, points = batch['octree'], batch['points']
        input_feature = self.get_input_feature(batch['octree'], self.oct_embedder, self.FLAGS.MODEL)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        feature = self.model.UNet(input_feature, octree, octree.depth, query_pts)

        for tqdm in range(301):
            optimizer.zero_grad()
            vp_norm = vp / torch.norm(vp) * radium
            vd = batch['points'].points.requires_grad_() - vp_norm
            vd = vd / torch.norm(vd, dim=1, keepdim=True)
            vd = self.embedder(vd.view(-1, 3))
            visibilities = self.model.VisNet(feature * vd).view(-1, 2)
            #softmax
            unvis = torch.nn.functional.softmax(visibilities, dim=1)[:, 1]
            loss = unvis.mean()
            loss.backward(retain_graph=True)
            optimizer.step()
            if tqdm % 100 == 0:
                print(f'loss: {loss.item()}')
                print(f'vp: {vp_norm}')
                #get and print num of visiable points
                print(f'num of visible points: {torch.sum(visibilities.argmax(dim=1) == 1)}')

    def test_time(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()
        data_sample = next(self.test_iter)
        processed = self.process_batch(data_sample, self.FLAGS.DATA.test)
        tree = processed["octree"]
        pts = processed["points"]
        feat = self.get_input_feature(tree, self.FLAGS.MODEL)
        q_pts = torch.cat([pts.points, pts.batch_id], dim=1)

        # Warmup
        for _ in range(20):
            mid = self.model.UNet(feat, tree, tree.depth, q_pts)
            _ = self.model.VisNet(mid)

        import time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 200
        unet_times, visnet_times = [], []
        vp_norm = torch.tensor([1.0000, 0.00001, 0.00001]).cuda()
        for _ in range(repetitions):
            torch.cuda.synchronize()
            starter.record()
            mid = self.model.UNet(feat, tree, tree.depth, q_pts)
            torch.cuda.synchronize()
            ender.record()
            torch.cuda.synchronize()
            unet_times.append(starter.elapsed_time(ender))
            starter.record()
            vd = data_sample['points'].points - vp_norm
            vd = vd / torch.norm(vd, dim=1, keepdim=True)
            vd = self.embedder(vd.view(-1, 3))
            _ = self.model.VisNet(mid*vd)
            torch.cuda.synchronize()
            ender.record()
            torch.cuda.synchronize()
            visnet_times.append(starter.elapsed_time(ender))

        #save time by takes
        with open('time.txt', 'a') as f:
            f.write(f'{self.FLAGS.DATA.test.takes}, {np.mean(unet_times)}, {np.mean(visnet_times)}\n')
        print("UNet forward time:", np.mean(unet_times))
        print("VisNet forward time:", np.mean(visnet_times))

if __name__ == "__main__":
   VisSolver.main()