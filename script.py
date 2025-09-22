import os

categories = ['02691156', '02773838', '02808440', '02818832',
              '02828884', '02876657', '02933112', '02958343',
              '03001627', '03211117', '03636649', '03691459',
              '03938244', '04090263', '04256520', '04379243',
              '04401088', '04530566']

cross_categories = ['02773838', '02818832', '02876657', '03938244', '04401088']

script = 'python train.py --config vis_shapenet.yaml'
# You can adjust the points number here by t
t = 81920
for c in categories:
  cmds = [
      script,
      'SOLVER.gpu 0,',
      'SOLVER.run test',
      'DATA.test.filelist data/filelist/{}.txt'.format(c),
      'DATA.test.takes {}'.format(t),
  ]
  os.system(' '.join(cmds))

