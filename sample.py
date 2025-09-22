import open3d as o3d
import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--num_pts',
        type=int,
        # required=True,
        default=100000,
    )
    args = parser.parse_args()
    # .obj to .npz
    output_path = args.name.split('.obj')[0] + '.npz'

    mesh = o3d.io.read_triangle_mesh(args.name)

    bbox = mesh.get_axis_aligned_bounding_box()

    # 计算中心点和最大范围
    center = bbox.get_center()
    max_extent = np.max(bbox.get_extent())

    # 创建变换矩阵
    scale = 0.8 / max_extent
    translation = -center

    # 应用变换
    mesh.translate(translation)
    mesh.scale(scale, center=[0, 0, 0])

    o3d.io.write_triangle_mesh(os.path.join(args.name, 'new_mesh.obj'), mesh)

    pcd = mesh.sample_points_uniformly(
        number_of_points=args.num_pts,  # 目标点数（可能不会完全达到）
        # init_factor=10,           # 初始采样密度（越大越均匀）
        # use_triangle_normal=True # 是否使用法线信息
    )

    # o3d.io.write_point_cloud(os.path.join(args.name, f'pcd_{args.num_pts}.ply'), pcd)
    # np.save(os.path.join(args.name, f'pcd_{args.num_pts}.npy'), np.asarray(pcd.points))
    np.savez(output_path, points=np.asarray(pcd.points))