import pygame
import sys
import cv2
import math
import numpy as np
from numba import cuda
import open3d as o3d
from models import MyNet, get_embedder
import ocnn
from ocnn.octree import Points
import torch
import time
from scipy.spatial import Delaunay, KDTree
from HPR import HPR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path',
    type=str,
    required=True,
)
args = parser.parse_args()

@cuda.jit
def gpu_render_mesh(faces, viewpoint, normals, img, depth, x_axis, y_axis, z_axis, n_f, H, W, Z, R_value, G_value, B_value):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_f:
        v1 = cuda.local.array(3, dtype=np.float32)
        v2 = cuda.local.array(3, dtype=np.float32)
        v3 = cuda.local.array(3, dtype=np.float32)
        
        v1[0] = faces[idx, 0, 0] - viewpoint[0]
        v1[1] = faces[idx, 0, 1] - viewpoint[1]
        v1[2] = faces[idx, 0, 2] - viewpoint[2]
        
        v2[0] = faces[idx, 1, 0] - viewpoint[0]
        v2[1] = faces[idx, 1, 1] - viewpoint[1]
        v2[2] = faces[idx, 1, 2] - viewpoint[2]
        
        v3[0] = faces[idx, 2, 0] - viewpoint[0]
        v3[1] = faces[idx, 2, 1] - viewpoint[1]
        v3[2] = faces[idx, 2, 2] - viewpoint[2]
        
        normal = normals[idx]
        
        v1_len = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        v2_len = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
        v3_len = math.sqrt(v3[0] ** 2 + v3[1] ** 2 + v3[2] ** 2)
        
        score1 = min(1, max(0, -(v1[0] * normal[0] + v1[1] * normal[1] + v1[2] * normal[2]) / v1_len))
        score2 = min(1, max(0, -(v2[0] * normal[0] + v2[1] * normal[1] + v2[2] * normal[2]) / v2_len))
        score3 = min(1, max(0, -(v3[0] * normal[0] + v3[1] * normal[1] + v3[2] * normal[2]) / v3_len))

        x1 = v1[0] * x_axis[0] + v1[1] * x_axis[1] + v1[2] * x_axis[2]
        y1 = v1[0] * y_axis[0] + v1[1] * y_axis[1] + v1[2] * y_axis[2]
        z1 = v1[0] * z_axis[0] + v1[1] * z_axis[1] + v1[2] * z_axis[2]
        
        x2 = v2[0] * x_axis[0] + v2[1] * x_axis[1] + v2[2] * x_axis[2]
        y2 = v2[0] * y_axis[0] + v2[1] * y_axis[1] + v2[2] * y_axis[2]
        z2 = v2[0] * z_axis[0] + v2[1] * z_axis[1] + v2[2] * z_axis[2]
        
        x3 = v3[0] * x_axis[0] + v3[1] * x_axis[1] + v3[2] * x_axis[2]
        y3 = v3[0] * y_axis[0] + v3[1] * y_axis[1] + v3[2] * y_axis[2]
        z3 = v3[0] * z_axis[0] + v3[1] * z_axis[1] + v3[2] * z_axis[2]
        
        U1 = Z / z1 * y1 + H / 2
        V1 = Z / z1 * x1 + W / 2
        
        U2 = Z / z2 * y2 + H / 2
        V2 = Z / z2 * x2 + W / 2
        
        U3 = Z / z3 * y3 + H / 2
        V3 = Z / z3 * x3 + W / 2

        U_min = math.floor(min(U1, U2, U3))
        U_max = math.ceil(max(U1, U2, U3))
        V_min = math.floor(min(V1, V2, V3))
        V_max = math.ceil(max(V1, V2, V3))
        
        D1 = math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
        D2 = math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
        D3 = math.sqrt(x3 ** 2 + y3 ** 2 + z3 ** 2)
        for i in range(U_min, U_max + 1):
            for j in range(V_min, V_max + 1):
                if i >= 0 and i < H and j >= 0 and j < W:
                    edge1 = (((i - U2) * (V3 - V2) - (j - V2) * (U3 - U2))  * ((U1 - U2) * (V3 - V2) - (V1 - V2) * (U3 - U2))) >= 0
                    edge2 = (((i - U1) * (V3 - V1) - (j - V1) * (U3 - U1))  * ((U2 - U1) * (V3 - V1) - (V2 - V1) * (U3 - U1))) >= 0
                    edge3 = (((i - U1) * (V2 - V1) - (j - V1) * (U2 - U1))  * ((U3 - U1) * (V2 - V1) - (V3 - V1) * (U2 - U1))) >= 0
                    if (edge1 and edge2 and edge3):
                        area_1 = abs(0.5 * (i * (V2 - V3) + U2 * (V3 - j) + U3 * (j - V2)))
                        area_2 = abs(0.5 * (U1 * (j - V3) + i * (V3 - V1) + U3 * (V1 - j)))
                        area_3 = abs(0.5 * (U1 * (V2 - j) + U2 * (j - V1) + i * (V1 - V2)))
                        area = area_1 + area_2 + area_3
                        w1 = area_1 / area
                        w2 = area_2 / area
                        w3 = area_3 / area
                        D = w1 * D1 + w2 * D2 + w3 * D3
                        score = w1 * score1 + w2 * score2 + w3 * score3
                        score = min(1, max(0, score))
                        index = (H - i - 1) * W + j
                        old_depth = cuda.atomic.min(depth, index, D)
                        if old_depth > D:
                            img[index, 0] = int(B_value * (score * 0.7 + 0.3))
                            img[index, 1] = int(G_value * (score * 0.7 + 0.3))
                            img[index, 2] = int(R_value * (score * 0.7 + 0.3))


@cuda.jit
def gpu_render_shadow_map(faces, viewpoint, depth, x_axis, y_axis, z_axis, n_f, H, W, Z):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_f:
        v1 = cuda.local.array(3, dtype=np.float32)
        v2 = cuda.local.array(3, dtype=np.float32)
        v3 = cuda.local.array(3, dtype=np.float32)
        
        v1[0] = faces[idx, 0, 0] - viewpoint[0]
        v1[1] = faces[idx, 0, 1] - viewpoint[1]
        v1[2] = faces[idx, 0, 2] - viewpoint[2]
        
        v2[0] = faces[idx, 1, 0] - viewpoint[0]
        v2[1] = faces[idx, 1, 1] - viewpoint[1]
        v2[2] = faces[idx, 1, 2] - viewpoint[2]
        
        v3[0] = faces[idx, 2, 0] - viewpoint[0]
        v3[1] = faces[idx, 2, 1] - viewpoint[1]
        v3[2] = faces[idx, 2, 2] - viewpoint[2]

        x1 = v1[0] * x_axis[0] + v1[1] * x_axis[1] + v1[2] * x_axis[2]
        y1 = v1[0] * y_axis[0] + v1[1] * y_axis[1] + v1[2] * y_axis[2]
        z1 = v1[0] * z_axis[0] + v1[1] * z_axis[1] + v1[2] * z_axis[2]
        
        x2 = v2[0] * x_axis[0] + v2[1] * x_axis[1] + v2[2] * x_axis[2]
        y2 = v2[0] * y_axis[0] + v2[1] * y_axis[1] + v2[2] * y_axis[2]
        z2 = v2[0] * z_axis[0] + v2[1] * z_axis[1] + v2[2] * z_axis[2]
        
        x3 = v3[0] * x_axis[0] + v3[1] * x_axis[1] + v3[2] * x_axis[2]
        y3 = v3[0] * y_axis[0] + v3[1] * y_axis[1] + v3[2] * y_axis[2]
        z3 = v3[0] * z_axis[0] + v3[1] * z_axis[1] + v3[2] * z_axis[2]
        
        U1 = Z / z1 * y1 + H / 2
        V1 = Z / z1 * x1 + W / 2
        
        U2 = Z / z2 * y2 + H / 2
        V2 = Z / z2 * x2 + W / 2
        
        U3 = Z / z3 * y3 + H / 2
        V3 = Z / z3 * x3 + W / 2

        U_min = math.floor(min(U1, U2, U3))
        U_max = math.ceil(max(U1, U2, U3))
        V_min = math.floor(min(V1, V2, V3))
        V_max = math.ceil(max(V1, V2, V3))
        
        D1 = math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
        D2 = math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
        D3 = math.sqrt(x3 ** 2 + y3 ** 2 + z3 ** 2)
        for i in range(U_min, U_max + 1):
            for j in range(V_min, V_max + 1):
                if i >= 0 and i < H and j >= 0 and j < W:
                    edge1 = (((i - U2) * (V3 - V2) - (j - V2) * (U3 - U2))  * ((U1 - U2) * (V3 - V2) - (V1 - V2) * (U3 - U2))) >= 0
                    edge2 = (((i - U1) * (V3 - V1) - (j - V1) * (U3 - U1))  * ((U2 - U1) * (V3 - V1) - (V2 - V1) * (U3 - U1))) >= 0
                    edge3 = (((i - U1) * (V2 - V1) - (j - V1) * (U2 - U1))  * ((U3 - U1) * (V2 - V1) - (V3 - V1) * (U2 - U1))) >= 0
                    if (edge1 and edge2 and edge3):
                        area_1 = abs(0.5 * (i * (V2 - V3) + U2 * (V3 - j) + U3 * (j - V2)))
                        area_2 = abs(0.5 * (U1 * (j - V3) + i * (V3 - V1) + U3 * (V1 - j)))
                        area_3 = abs(0.5 * (U1 * (V2 - j) + U2 * (j - V1) + i * (V1 - V2)))
                        area = area_1 + area_2 + area_3
                        w1 = area_1 / area
                        w2 = area_2 / area
                        w3 = area_3 / area
                        D = w1 * D1 + w2 * D2 + w3 * D3
                        index = (H - i - 1) * W + j
                        cuda.atomic.min(depth, index, D)


@cuda.jit
def gpu_query_shadow_map(vertices, illusions, shadow_map, x_axis, y_axis, z_axis, n_v, H, W, Z):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        D = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        if U >= 0 and U < H and V >= 0 and V < W:
            if D - shadow_map[H - U - 1, V] > 1e-2:
                illusions[idx] = 0.7


@cuda.jit
def gpu_render(vertices, img, depth, x_axis, y_axis, z_axis, n_v, H, W, Z, r, R_value, G_value, B_value, illusions):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        illusion = illusions[idx]
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        D = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        for i in range(int(U - math.ceil(r)), int(U + math.ceil(r))):
            for j in range(int(V - math.ceil(r)), int(V + math.ceil(r))):
                if i >= 0 and i < H and j >= 0 and j < W:
                    if D < depth[H - i - 1, j]:
                        depth[H - i - 1, j] = D
                        img[H - i - 1, j, 0] = int(B_value * illusion)
                        img[H - i - 1, j, 1] = int(G_value * illusion)
                        img[H - i - 1, j, 2] = int(R_value * illusion)
                        # img[H - i - 1, j, 0] = int(B_value)
                        # img[H - i - 1, j, 1] = int(G_value)
                        # img[H - i - 1, j, 2] = int(R_value)
        

@cuda.jit(max_registers=40)
def gpu_render_normal(vertices, img, depth, x_axis, y_axis, z_axis, n_v, H, W, Z, r, R_value, G_value, B_value, normals, illusions):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        normal = normals[idx]
        illusion = illusions[idx]
        v_len = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
        n_len = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        score = min(1, max(0, -(vertex[0] * normal[0] + vertex[1] * normal[1] + vertex[2] * normal[2]) / (v_len * n_len)))
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        D = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        for i in range(int(U - math.ceil(r * (score * 0.8 + 0.2))), int(U + math.ceil(r * (score * 0.8 + 0.2)))):
            for j in range(int(V - math.ceil(r * (score * 0.8 + 0.2))), int(V + math.ceil(r * (score * 0.8 + 0.2)))):
                if i >= 0 and i < H and j >= 0 and j < W:
                    if D < depth[H - i - 1, j]:
                        depth[H - i - 1, j] = D
                        img[H - i - 1, j, 0] = int(B_value * (score * 0.7 + 0.3) * illusion)
                        img[H - i - 1, j, 1] = int(G_value * (score * 0.7 + 0.3) * illusion)
                        img[H - i - 1, j, 2] = int(R_value * (score * 0.7 + 0.3) * illusion)

@cuda.jit(max_registers=40)
def gpu_render_floor(vertices, img, depth, x_axis, y_axis, z_axis, n_v, H, W, Z, r, R_value, G_value, B_value, normal, illusions):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        illusion = illusions[idx]
        v_len = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
        n_len = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        score = min(1, max(0, -(vertex[0] * normal[0] + vertex[1] * normal[1] + vertex[2] * normal[2]) / (v_len * n_len)))
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        D = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        for i in range(int(U - math.ceil(r * (score * 0.8 + 0.2))), int(U + math.ceil(r * (score * 0.8 + 0.2)))):
            for j in range(int(V - math.ceil(r * (score * 0.8 + 0.2))), int(V + math.ceil(r * (score * 0.8 + 0.2)))):
                if i >= 0 and i < H and j >= 0 and j < W:
                    if D < depth[H - i - 1, j]:
                        depth[H - i - 1, j] = D
                        img[H - i - 1, j, 0] = int(B_value * illusion)
                        img[H - i - 1, j, 1] = int(G_value * illusion)
                        img[H - i - 1, j, 2] = int(R_value * illusion)

@cuda.jit
def gpu_render_color(vertices, img, depth, x_axis, y_axis, z_axis, n_v, H, W, Z, r, colors, illusions):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        color = colors[idx]
        illusion = illusions[idx]
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        D = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        for i in range(int(U - math.ceil(r)), int(U + math.ceil(r))):
            for j in range(int(V - math.ceil(r)), int(V + math.ceil(r))):
                if i >= 0 and i < H and j >= 0 and j < W:
                    if D < depth[H - i - 1, j]:
                        depth[H - i - 1, j] = D
                        img[H - i - 1, j, 0] = color[2] * illusion
                        img[H - i - 1, j, 1] = color[1] * illusion
                        img[H - i - 1, j, 2] = color[0] * illusion

@cuda.jit
def gpu_render_normal_color(vertices, img, depth, x_axis, y_axis, z_axis, n_v, H, W, Z, r, normals, colors, illusions):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        normal = normals[idx]
        color = colors[idx]
        illusion = illusions[idx]
        v_len = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
        n_len = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        score = min(1, max(0, -(vertex[0] * normal[0] + vertex[1] * normal[1] + vertex[2] * normal[2]) / (v_len * n_len)))
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        D = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        for i in range(int(U - math.ceil(r * (score * 0.8 + 0.2))), int(U + math.ceil(r * (score * 0.8 + 0.2)))):
            for j in range(int(V - math.ceil(r * (score * 0.8 + 0.2))), int(V + math.ceil(r * (score * 0.8 + 0.2)))):
                if i >= 0 and i < H and j >= 0 and j < W:
                    if D < depth[H - i - 1, j]:
                        depth[H - i - 1, j] = D
                        img[H - i - 1, j, 0] = int(color[2] * (score * 0.7 + 0.3) * illusion)
                        img[H - i - 1, j, 1] = int(color[1] * (score * 0.7 + 0.3) * illusion)
                        img[H - i - 1, j, 2] = int(color[0] * (score * 0.7 + 0.3) * illusion)

def render(vertices, view_point, x_axis, y_axis, z_axis, H, W, Z, r, vertex_color, background_color, normals=None, colors=None):
    img_device = cuda.to_device(np.ones((H, W, 3), dtype=np.uint8)  * np.array([[background_color[::-1]]]))
    vertices_device = cuda.to_device(vertices - view_point)
    if normals is not None:
        normals_device = cuda.to_device(normals)
    if colors is not None:
        colors_device = cuda.to_device(colors)
    x_axis_device = cuda.to_device(x_axis)
    y_axis_device = cuda.to_device(y_axis)
    z_axis_device = cuda.to_device(z_axis)
    depth_device = cuda.to_device(np.ones((H, W), dtype=np.float32) * 1e5)
    illusions = cuda.to_device(np.ones(vertices.shape[0], dtype=np.float32))
    if normals is not None and colors is not None:
        normals_device = cuda.to_device(normals)
        colors_device = cuda.to_device(colors)
        gpu_render_normal_color[(vertices.shape[0]) // 512 + 1 if (vertices.shape[0]) // 512 + 1 > 512 else 512, 512](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, normals_device, colors_device, illusions)
    elif colors is not None:
        colors_device = cuda.to_device(colors)
        gpu_render_color[(vertices.shape[0]) // 512 + 1 if (vertices.shape[0]) // 512 + 1 > 512 else 512, 512](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, colors_device, illusions)
    elif normals is not None:
        normals_device = cuda.to_device(normals)
        gpu_render_normal[(vertices.shape[0]) // 512 + 1 if (vertices.shape[0]) // 512 + 1 > 512 else 512, 512](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, vertex_color[0], vertex_color[1], vertex_color[2], normals_device, illusions)
    else:
        gpu_render[(vertices.shape[0]) // 512 + 1 if (vertices.shape[0]) // 512 + 1 > 512 else 512, 512](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, vertex_color[0], vertex_color[1], vertex_color[2], illusions)
    cuda.synchronize()
    img = img_device.copy_to_host()
    return img


def render_with_shadow(vertices, view_point, x_axis, y_axis, z_axis, H, W, Z, r, vertex_color, background_color, shadow_map, light_point, l_x_axis, l_y_axis, l_z_axis, floor_points, floor_normal, normals=None, colors=None):
    vertices_light = cuda.to_device(vertices - light_point)
    floor_points_lights = cuda.to_device(floor_points - light_point)
    illusions = cuda.to_device(np.ones(vertices.shape[0], dtype=np.float32))
    l_x_axis_device = cuda.to_device(l_x_axis)
    l_y_axis_device = cuda.to_device(l_y_axis)
    l_z_axis_device = cuda.to_device(l_z_axis)
    shadow_map_device = cuda.to_device(shadow_map)
    gpu_query_shadow_map[(vertices.shape[0]) // 1024 + 1 if (vertices.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](vertices_light, illusions, shadow_map_device, l_x_axis_device, l_y_axis_device, l_z_axis_device, vertices.shape[0], H, W, Z)
    img_device = cuda.to_device(np.ones((H, W, 3), dtype=np.uint8)  * np.array([[background_color[::-1]]]))
    illusions_floor = cuda.to_device(np.ones(floor_points.shape[0], dtype=np.float32))
    gpu_query_shadow_map[(floor_points.shape[0]) // 1024 + 1 if (floor_points.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](floor_points_lights, illusions_floor, shadow_map_device, l_x_axis_device, l_y_axis_device, l_z_axis_device, floor_points.shape[0], H, W, Z)
    # gpu_render_floor(vertices, img, depth, x_axis, y_axis, z_axis, n_v, H, W, Z, r, R_value, G_value, B_value, normals, illusions):
    depth_floor_device = cuda.to_device(np.ones((H, W), dtype=np.float32) * 1e5)
    floor_normal_device = cuda.to_device(floor_normal)
    floor_points_device = cuda.to_device(floor_points - view_point)
    vertices_device = cuda.to_device(vertices - view_point)
    if normals is not None:
        normals_device = cuda.to_device(normals)
    if colors is not None:
        colors_device = cuda.to_device(colors)
    x_axis_device = cuda.to_device(x_axis)
    y_axis_device = cuda.to_device(y_axis)
    z_axis_device = cuda.to_device(z_axis)
    depth_device = cuda.to_device(np.ones((H, W), dtype=np.float32) * 1e5)
    gpu_render_floor[(floor_points.shape[0]) // 1024 + 1 if (floor_points.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](floor_points_device, img_device, depth_floor_device, x_axis_device, y_axis_device, z_axis_device, floor_points.shape[0], H, W, Z, r, 255, 255, 255, floor_normal_device, illusions_floor)
    floor_img = img_device.copy_to_host()
    if normals is not None and colors is not None:
        normals_device = cuda.to_device(normals)
        colors_device = cuda.to_device(colors)
        gpu_render_normal_color[(vertices.shape[0]) // 1024 + 1 if (vertices.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, normals_device, colors_device, illusions)
    elif colors is not None:
        colors_device = cuda.to_device(colors)
        gpu_render_color[(vertices.shape[0]) // 1024 + 1 if (vertices.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, colors_device, illusions)
    elif normals is not None:
        normals_device = cuda.to_device(normals)
        gpu_render_normal[(vertices.shape[0]) // 1024 + 1 if (vertices.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, vertex_color[0], vertex_color[1], vertex_color[2], normals_device, illusions)
    else:
        gpu_render[(vertices.shape[0]) // 1024 + 1 if (vertices.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](vertices_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r, vertex_color[0], vertex_color[1], vertex_color[2], illusions)
    cuda.synchronize()
    img = img_device.copy_to_host()
    return img, floor_img

def render_shadow_map(faces, viewpoint, x_axis, y_axis, z_axis, H, W, Z):
    print(faces.shape, viewpoint.shape, x_axis.shape, y_axis.shape, z_axis.shape)
    viewpoint_device = cuda.to_device(viewpoint)
    faces_device = cuda.to_device(faces)
    depth_device = cuda.to_device(np.ones((H * W), dtype=np.float32) * 1e5)
    x_axis_device = cuda.to_device(x_axis)
    y_axis_device = cuda.to_device(y_axis)
    z_axis_device = cuda.to_device(z_axis)
    gpu_render_shadow_map[(faces.shape[0]) // 64 + 1 if (faces.shape[0]) // 64 + 1 > 64 else 64, 64](faces_device, viewpoint_device, depth_device, x_axis_device, y_axis_device, z_axis_device, faces.shape[0], H, W, Z)
    cuda.synchronize()
    depth = depth_device.copy_to_host().reshape(H, W)
    return depth

def render_shadow_map_by_delaunay(vertices, viewpoint, x_axis, y_axis, z_axis, H, W, Z):
    x = np.sum((vertices - viewpoint) * x_axis.reshape(1, 3), axis=1)
    y = np.sum((vertices - viewpoint) * y_axis.reshape(1, 3), axis=1)
    z = np.sum((vertices - viewpoint) * z_axis.reshape(1, 3), axis=1)
    U = (Z / z * y + H / 2).astype(np.int32)
    V = (Z / z * x + W / 2).astype(np.int32)
    points = np.stack([V, U], axis=1, dtype=np.int32)
    tri = Delaunay(points)
    indices = tri.simplices
    max_len = np.zeros((tri.simplices.shape[0]), dtype=np.float32)
    for j in range(tri.simplices.shape[0]):
        idx1, idx2, idx3 = tri.simplices[j]

        p1 = vertices[idx1]
        p2 = vertices[idx2]
        p3 = vertices[idx3]

        d1 = np.linalg.norm(p1 - p2)
        d2 = np.linalg.norm(p2 - p3)
        d3 = np.linalg.norm(p3 - p1)
        max_len[j] = max(d1, d2, d3)

    kdtree = KDTree(vertices)
    distances, _ = kdtree.query(vertices, k=2)
    nearest_distances = distances[:, 1]
    indices = tri.simplices[max_len < np.mean(nearest_distances) * 10, :]
    faces = np.zeros((indices.shape[0], 3, 3), dtype=np.float32)
    faces[:, 0, :] = vertices[indices[:, 0]]
    faces[:, 1, :] = vertices[indices[:, 1]]
    faces[:, 2, :] = vertices[indices[:, 2]]
    return render_shadow_map(faces, viewpoint, x_axis, y_axis, z_axis, H, W, Z)

def normal_estimation(vertices, viewpoint, x_axis, y_axis, z_axis, H, W, Z):
    x = np.sum((vertices - viewpoint) * x_axis.reshape(1, 3), axis=1)
    y = np.sum((vertices - viewpoint) * y_axis.reshape(1, 3), axis=1)
    z = np.sum((vertices - viewpoint) * z_axis.reshape(1, 3), axis=1)
    U = (Z / z * y + H / 2).astype(np.int32)
    V = (Z / z * x + W / 2).astype(np.int32)
    points = np.stack([V, U], axis=1, dtype=np.int32)
    tri = Delaunay(points)
    indices = tri.simplices
    max_len = np.zeros((tri.simplices.shape[0]), dtype=np.float32)
    for j in range(tri.simplices.shape[0]):
        idx1, idx2, idx3 = tri.simplices[j]
        # p1 = points[idx1]
        # p2 = points[idx2]
        # p3 = points[idx3]
        p1 = vertices[idx1]
        p2 = vertices[idx2]
        p3 = vertices[idx3]
        # p1 = np.array([x[idx1], y[idx1]])
        # p2 = np.array([x[idx2], y[idx2]])
        # p3 = np.array([x[idx3], y[idx3]])
        d1 = np.linalg.norm(p1 - p2)
        d2 = np.linalg.norm(p2 - p3)
        d3 = np.linalg.norm(p3 - p1)
        max_len[j] = max(d1, d2, d3)
    # kdtree = KDTree(points)
    # distances, _ = kdtree.query(points, k=2)
    # x_y_concated = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    # assert x_y_concated.shape == (vertices.shape[0], 2)
    # kdtree = KDTree(x_y_concated)
    # distances, _ = kdtree.query(x_y_concated, k=2)
    kdtree = KDTree(vertices)
    distances, _ = kdtree.query(vertices, k=2)
    nearest_distances = distances[:, 1]
    indices = tri.simplices[max_len < np.mean(nearest_distances) * 10, :]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(indices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    return normals

def render_mesh(faces, viewpoint, x_axis, y_axis, z_axis, H, W, Z, face_color, background_color):
    img_device = cuda.to_device(np.ones((H * W, 3), dtype=np.uint8)  * np.array([background_color[::-1]]))
    viewpoint_device = cuda.to_device(viewpoint)
    faces_device = cuda.to_device(faces)
    e1 = faces[:, 1, :] - faces[:, 0, :]
    e2 = faces[:, 2, :] - faces[:, 1, :]
    normals = np.cross(e1, e2)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals_device = cuda.to_device(normals)
    depth_device = cuda.to_device(np.ones((H * W), dtype=np.float32) * 1e5)
    x_axis_device = cuda.to_device(x_axis)
    y_axis_device = cuda.to_device(y_axis)
    z_axis_device = cuda.to_device(z_axis)
    gpu_render_mesh[(faces.shape[0]) // 64 + 1 if (faces.shape[0]) // 64 + 1 > 64 else 64, 64](faces_device, viewpoint_device, normals_device, img_device, depth_device, x_axis_device, y_axis_device, z_axis_device, faces.shape[0], H, W, Z, face_color[0], face_color[1], face_color[2])
    cuda.synchronize()
    img = img_device.copy_to_host().reshape(H, W, 3).astype(np.uint8)
    img = cv2.medianBlur(img, 5)
    return img
    
def frame_img(img, frame_width, frame_color):
    img[:frame_width, :] = frame_color[::-1]
    img[-frame_width:, :] = frame_color[::-1]
    img[:, :frame_width] = frame_color[::-1]
    img[:, -frame_width:] = frame_color[::-1]
    return img

def coor_in(coor, x_min, x_max, y_min, y_max):
    return coor[0] > x_min and coor[0] < x_max and coor[1] > y_min and coor[1] < y_max

def generate_text_surface(surface_width, surface_height, text, text_scale, text_thickness, text_color, background_color, frame_color):
    img = np.ones((surface_height, surface_width, 3), dtype=np.uint8) * np.array([[background_color[::-1]]])
    img = frame_img(img, 1, frame_color)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    text_width, text_height = text_size
    text_x = (surface_width - text_width) // 2
    text_y = (surface_height + text_height) // 2
    img = img.astype(np.uint8)
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color[::-1], text_thickness)
    return pygame.surfarray.make_surface(img.transpose(1, 0, 2)[..., ::-1])

def points2octree(points, depth=9):
    octree = ocnn.octree.Octree(depth, 2)
    octree.build_octree(points)
    return octree

def get_input_feature(octree):
    depth = octree.depth
    features = []
    local_points = octree.points[depth].frac() - 0.5
    features.append(local_points)
    scale = 2 ** (1 - depth)
    global_points = octree.points[depth] * scale - 1.0
    features.append(global_points)
    out = torch.cat(features, dim=1)
    return out

parameter = {
    'window_width': 1000,
    'window_height': 800,
    'view_width': 800,
    'view_height': 800,
    'arrow_width': 30,
    'arrow_height': 20,
    'view_depth': 1200,
    'point_size': 2,
    'viewpoint_radius': 3,
    'frame_width': 1,
    'point_color': (3, 101, 100),
    'text_color': (3, 22, 52),
    'frame_color': (205, 179, 128),
    'background_color': (232, 221, 203),
    'arrow_color': (3, 54, 73),
    'viewpoint_direction': np.array((0, 0, 1), dtype=np.float32),
    'vertices': None,
    'normals': None,
    'colors': None,
    'faces': None,
    
    'rerender': False,
    'normal_switch': False,
    'color_switch':False,
    'visibility': 'all',
    'visibility_bool': None,
    
    'UNet': None,
    'VisNet': None,
    'vdr_embedder': None,
    'oct_embedder': None,
    'feature': None,
    
    'hpr_pcd': None,
    'hpr_radius': None,
    
    'delauny': 'off',
    'delauny_prune': 'Off',
    'delauny_prune_scale': 4,
}

pressed_up_arrow_img = np.ones((parameter['arrow_height'], parameter['arrow_width'], 3), dtype=np.uint8) * np.array([[parameter['arrow_color'][::-1]]])
flag = 1
for i in range(int(0.25 * parameter['arrow_height']), parameter['arrow_height'] - int(0.25 * parameter['arrow_height'])):
    s = (parameter['arrow_width'] - flag * 2) // 2
    for j in range(s, parameter['arrow_width'] - s):
        pressed_up_arrow_img[i, j] = parameter['background_color'][::-1]
    flag += 1
pressed_up_arrow_img = frame_img(pressed_up_arrow_img, 1, parameter['frame_color'])

up_arrow_img = np.ones((parameter['arrow_height'], parameter['arrow_width'], 3), dtype=np.uint8) * np.array([[parameter['background_color'][::-1]]])
flag = 1
for i in range(int(0.25 * parameter['arrow_height']), parameter['arrow_height'] - int(0.25 * parameter['arrow_height'])):
    s = (parameter['arrow_width'] - flag * 2) // 2
    for j in range(s, parameter['arrow_width'] - s):
        up_arrow_img[i, j] = parameter['arrow_color'][::-1]
    flag += 1
up_arrow_img = frame_img(up_arrow_img, 1, parameter['frame_color'])

parameter['point_size_up_arrow_surface'] = pygame.surfarray.make_surface(up_arrow_img.transpose(1, 0, 2)[..., ::-1])
parameter['viewpoint_radius_up_arrow_surface'] = pygame.surfarray.make_surface(up_arrow_img.transpose(1, 0, 2)[..., ::-1])
parameter['delauny_prune_scale_up_arrow_surface'] = pygame.surfarray.make_surface(up_arrow_img.transpose(1, 0, 2)[..., ::-1])

pressed_down_arrow_img = np.ones((parameter['arrow_height'], parameter['arrow_width'], 3), dtype=np.uint8) * np.array([[parameter['arrow_color'][::-1]]])
flag = 1
for i in range(int(0.25 * parameter['arrow_height']), parameter['arrow_height'] - int(0.25 * parameter['arrow_height'])):
    s = (parameter['arrow_width'] - flag * 2) // 2
    for j in range(s, parameter['arrow_width'] - s):
        pressed_down_arrow_img[parameter['arrow_height'] - 1 - i, j] = parameter['background_color'][::-1]
    flag += 1
pressed_down_arrow_img = frame_img(pressed_down_arrow_img, 1, parameter['frame_color'])

down_arrow_img = np.ones((parameter['arrow_height'], parameter['arrow_width'], 3), dtype=np.uint8) * np.array([[parameter['background_color'][::-1]]])
flag = 1
for i in range(int(0.25 * parameter['arrow_height']), parameter['arrow_height'] - int(0.25 * parameter['arrow_height'])):
    s = (parameter['arrow_width'] - flag * 2) // 2
    for j in range(s, parameter['arrow_width'] - s):
        down_arrow_img[parameter['arrow_height'] - 1 - i, j] = parameter['arrow_color'][::-1]
    flag += 1
down_arrow_img = frame_img(down_arrow_img, 1, parameter['frame_color'])

parameter['point_size_down_arrow_surface'] = pygame.surfarray.make_surface(down_arrow_img.transpose(1, 0, 2)[..., ::-1])
parameter['viewpoint_radius_down_arrow_surface'] = pygame.surfarray.make_surface(down_arrow_img.transpose(1, 0, 2)[..., ::-1])
parameter['delauny_prune_scale_down_arrow_surface'] = pygame.surfarray.make_surface(down_arrow_img.transpose(1, 0, 2)[..., ::-1])

parameter['point_size_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['point_size']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['viewpoint_radius_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['viewpoint_radius']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['delauny_prune_scale_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['delauny_prune_scale']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

parameter['point_size_title_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'] - parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, 'Point Size', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['viewpoint_radius_title_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'] - parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, 'Radius', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['delauny_prune_scale_title_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'] - parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, 'Prune Len', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

parameter['normal_switch_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Normal On' if parameter['normal_switch'] else 'Normal Off', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['color_switch_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Color On' if parameter['color_switch'] else 'Color Off', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

parameter['all_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'All Vertices', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'all' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'all' else parameter['background_color'], parameter['frame_color'])
parameter['hpr_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'HPR Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'hpr' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'hpr' else parameter['background_color'], parameter['frame_color'])
parameter['neu_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Neural Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'neu' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'neu' else parameter['background_color'], parameter['frame_color'])

parameter['delauny_off_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny Off', 0.7, 2, parameter['background_color'] if parameter['delauny'] == 'off' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == 'off' else parameter['background_color'], parameter['frame_color'])
parameter['delauny_2d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 2D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '2d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '2d' else parameter['background_color'], parameter['frame_color'])
parameter['delauny_3d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 3D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '3d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '3d' else parameter['background_color'], parameter['frame_color'])

parameter['delauny_prune_switch_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Prune On' if parameter['delauny_prune'] else 'Prune Off', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

parameter['interval_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 1, '', 0, 0, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

parameter['render_time_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Render {0.0:5.2f}s', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['viewpoint_x_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint {(parameter["viewpoint_radius"]*parameter["viewpoint_direction"][0]):.2f}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['viewpoint_y_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint {(parameter["viewpoint_radius"]*parameter["viewpoint_direction"][1]):.2f}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['viewpoint_z_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint {(parameter["viewpoint_radius"]*parameter["viewpoint_direction"][2]):.2f}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])


#parameter['blank_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 10, '', 0, 0, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

parameter['author_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Author of Viewer', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['author_name_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Tian Yiyang', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['project_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Project Name', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
parameter['project_name_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Neural Visibility', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])

view_img = np.ones((parameter['view_height'], parameter['view_width'], 3), dtype=np.uint8) * np.array([[parameter['background_color'][::-1]]])
view_img = frame_img(view_img, parameter['frame_width'], parameter['frame_color'])
parameter['view_surface'] = pygame.surfarray.make_surface(view_img.transpose(1, 0, 2)[..., ::-1])

z_axis = -parameter['viewpoint_direction']
z_axis = z_axis / np.linalg.norm(z_axis)
x_axis = np.cross(z_axis, np.array([0, 1, 0], dtype=np.float64))
x_axis = x_axis / np.linalg.norm(x_axis)
y_axis = np.cross(x_axis, z_axis)
y_axis = y_axis / np.linalg.norm(y_axis)
parameter['x_axis'] = x_axis
parameter['y_axis'] = y_axis
parameter['z_axis'] = z_axis

pygame.init()
windowSurface = pygame.display.set_mode((parameter['window_width'], parameter['window_height']))
pygame.display.set_caption("NeuVis Viewer")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.DROPFILE:
            data = np.load(event.dict['file'])
            parameter['vertices'] = data['points']
            kdtree = KDTree(parameter['vertices'])
            distances, _ = kdtree.query(parameter['vertices'], k=2)
            nearest_distances = distances[:, 1]
            parameter['average_nearest_distance'] = np.mean(nearest_distances)
            if 'normals' in data.files:
                parameter['normals'] = data['normals']
            if 'colors' in data.files:
                parameter['colors'] = (data['colors'] * 255).astype(int)
            parameter['rerender'] = True
            parameter['hpr_pcd'] = None
            parameter['hpr_radius'] = None
            parameter['feature'] = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            # point size up arrow
            if event.button == 1 and coor_in(event.pos, parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 0, parameter['arrow_height'] * 1):
                parameter['point_size'] = min(9, max(1, parameter['point_size'] + 1))
                parameter['point_size_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['point_size']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['point_size_up_arrow_surface'] = pygame.surfarray.make_surface(pressed_up_arrow_img.transpose(1, 0, 2)[..., ::-1])
                parameter['rerender'] = True
                
            # point size down arrow
            if event.button == 1 and coor_in(event.pos, parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 1, parameter['arrow_height'] * 2):
                parameter['point_size'] = min(9, max(1, parameter['point_size'] - 1))
                parameter['point_size_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['point_size']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['point_size_down_arrow_surface'] = pygame.surfarray.make_surface(pressed_down_arrow_img.transpose(1, 0, 2)[..., ::-1])
                parameter['rerender'] = True
                
            if event.button == 1 and coor_in(event.pos, parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, parameter['arrow_height'] * 3):
                parameter['viewpoint_radius'] = min(9, max(1, parameter['viewpoint_radius'] + 1))
                parameter['viewpoint_radius_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['viewpoint_radius']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['viewpoint_radius_up_arrow_surface'] = pygame.surfarray.make_surface(pressed_up_arrow_img.transpose(1, 0, 2)[..., ::-1])
                parameter['rerender'] = True
                
            if event.button == 1 and coor_in(event.pos, parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 3, parameter['arrow_height'] * 4):
                parameter['viewpoint_radius'] = min(9, max(1, parameter['viewpoint_radius'] - 1))
                parameter['viewpoint_radius_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['viewpoint_radius']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['viewpoint_radius_down_arrow_surface'] = pygame.surfarray.make_surface(pressed_down_arrow_img.transpose(1, 0, 2)[..., ::-1])
                parameter['rerender'] = True
                
            if event.button == 1 and coor_in(event.pos, parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 25, parameter['arrow_height'] * 26):
                parameter['delauny_prune_scale'] = min(40, max(1, parameter['delauny_prune_scale'] + 2))
                parameter['delauny_prune_scale_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['delauny_prune_scale']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['delauny_prune_scale_up_arrow_surface'] = pygame.surfarray.make_surface(pressed_up_arrow_img.transpose(1, 0, 2)[..., ::-1])
                parameter['rerender'] = True
                
            if event.button == 1 and coor_in(event.pos, parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 26, parameter['arrow_height'] * 27):
                parameter['delauny_prune_scale'] = min(40, max(1, parameter['delauny_prune_scale'] - 2))
                parameter['delauny_prune_scale_value_surface'] = generate_text_surface(parameter['arrow_height'] * 2, parameter['arrow_height'] * 2, f"{parameter['delauny_prune_scale']:d}", 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['delauny_prune_scale_down_arrow_surface'] = pygame.surfarray.make_surface(pressed_down_arrow_img.transpose(1, 0, 2)[..., ::-1])
                parameter['rerender'] = True
            
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 5, parameter['arrow_height'] * 7):
                parameter['normal_switch'] = not parameter['normal_switch']
                parameter['normal_switch_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Normal On' if parameter['normal_switch'] else 'Normal Off', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
                
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 7, parameter['arrow_height'] * 9):
                parameter['color_switch'] = not parameter['color_switch']
                parameter['color_switch_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Color On' if parameter['color_switch'] else 'Color Off', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
                
            # All Vertice Visibility
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 10, parameter['arrow_height'] * 12):
                parameter['visibility'] = 'all'
                parameter['all_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'All Vertices', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'all' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'all' else parameter['background_color'], parameter['frame_color'])
                parameter['hpr_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'HPR Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'hpr' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'hpr' else parameter['background_color'], parameter['frame_color'])
                parameter['neu_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Neural Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'neu' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'neu' else parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
            
            # HPR Visibility
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 12, parameter['arrow_height'] * 14):
                parameter['visibility'] = 'hpr'
                parameter['all_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'All Vertices', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'all' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'all' else parameter['background_color'], parameter['frame_color'])
                parameter['hpr_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'HPR Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'hpr' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'hpr' else parameter['background_color'], parameter['frame_color'])
                parameter['neu_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Neural Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'neu' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'neu' else parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
            
            # Neural Visibility
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 14, parameter['arrow_height'] * 16):
                parameter['visibility'] = 'neu'
                parameter['all_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'All Vertices', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'all' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'all' else parameter['background_color'], parameter['frame_color'])
                parameter['hpr_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'HPR Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'hpr' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'hpr' else parameter['background_color'], parameter['frame_color'])
                parameter['neu_vertice_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Neural Visibility', 0.7, 2, parameter['background_color'] if parameter['visibility'] == 'neu' else parameter['text_color'], parameter['text_color'] if parameter['visibility'] == 'neu' else parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
                
            # Delauny Off
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 17, parameter['arrow_height'] * 19):
                parameter['delauny'] = 'off'
                parameter['delauny_off_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny Off', 0.7, 2, parameter['background_color'] if parameter['delauny'] == 'off' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == 'off' else parameter['background_color'], parameter['frame_color'])
                parameter['delauny_2d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 2D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '2d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '2d' else parameter['background_color'], parameter['frame_color'])
                parameter['delauny_3d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 3D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '3d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '3d' else parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
                
            # Delauny 2D
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 19, parameter['arrow_height'] * 21):
                parameter['delauny'] = '2d'
                parameter['delauny_off_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny Off', 0.7, 2, parameter['background_color'] if parameter['delauny'] == 'off' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == 'off' else parameter['background_color'], parameter['frame_color'])
                parameter['delauny_2d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 2D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '2d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '2d' else parameter['background_color'], parameter['frame_color'])
                parameter['delauny_3d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 3D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '3d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '3d' else parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
            
            # Delauny 3D
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 21, parameter['arrow_height'] * 23):
                parameter['delauny'] = '3d'
                parameter['delauny_off_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny Off', 0.7, 2, parameter['background_color'] if parameter['delauny'] == 'off' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == 'off' else parameter['background_color'], parameter['frame_color'])
                parameter['delauny_2d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 2D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '2d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '2d' else parameter['background_color'], parameter['frame_color'])
                parameter['delauny_3d_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Delauny 3D', 0.7, 2, parameter['background_color'] if parameter['delauny'] == '3d' else parameter['text_color'], parameter['text_color'] if parameter['delauny'] == '3d' else parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
            
            # Delauny Prune
            if event.button == 1 and coor_in(event.pos, 0, parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 23, parameter['arrow_height'] * 25):
                parameter['delauny_prune'] = not parameter['delauny_prune']
                parameter['delauny_prune_switch_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, 'Prune On' if parameter['delauny_prune'] else 'Prune Off', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
                parameter['rerender'] = True
            
        if event.type == pygame.MOUSEBUTTONUP:
            parameter['point_size_up_arrow_surface'] = pygame.surfarray.make_surface(up_arrow_img.transpose(1, 0, 2)[..., ::-1])
            parameter['point_size_down_arrow_surface'] = pygame.surfarray.make_surface(down_arrow_img.transpose(1, 0, 2)[..., ::-1])
            parameter['viewpoint_radius_up_arrow_surface'] = pygame.surfarray.make_surface(up_arrow_img.transpose(1, 0, 2)[..., ::-1])
            parameter['viewpoint_radius_down_arrow_surface'] = pygame.surfarray.make_surface(down_arrow_img.transpose(1, 0, 2)[..., ::-1])
            parameter['delauny_prune_scale_up_arrow_surface'] = pygame.surfarray.make_surface(up_arrow_img.transpose(1, 0, 2)[..., ::-1])
            parameter['delauny_prune_scale_down_arrow_surface'] = pygame.surfarray.make_surface(down_arrow_img.transpose(1, 0, 2)[..., ::-1])
        if event.type == pygame.MOUSEMOTION:
            if event.buttons[0] == 1 and parameter['delauny'] == 'off' and parameter['vertices'] is not None and coor_in(event.pos, parameter['window_width'] - parameter['view_width'], parameter['window_width'], 0, parameter['window_height']):
                parameter['viewpoint_direction'] -= 0.005 * x_axis * event.rel[0]
                parameter['viewpoint_direction'] += 0.005 * y_axis * event.rel[1]
                parameter['viewpoint_direction'] = parameter['viewpoint_direction'] / np.linalg.norm(parameter['viewpoint_direction'])
                z_axis = -parameter['viewpoint_direction']
                z_axis = z_axis / np.linalg.norm(z_axis)
                x_axis = np.cross(z_axis, np.array([0, 1, 0], dtype=np.float64))
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(x_axis, z_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
                parameter['x_axis'] = x_axis
                parameter['y_axis'] = y_axis
                parameter['z_axis'] = z_axis
                parameter['rerender'] = True
        if event.type == pygame.MOUSEWHEEL:
            parameter['view_depth'] = min(10000, max(100, parameter['view_depth'] - event.y * 50))
            parameter['rerender'] = True
    if parameter['rerender'] and parameter['vertices'] is not None:
        # visibility
        s = time.time()
        if parameter['visibility'] == 'all':
            parameter['visibility_bool'] = np.ones((parameter['vertices'].shape[0]), dtype=bool)
        elif parameter['visibility'] == 'hpr':
            pv_hpr, pv_hpr_ind = HPR(torch.tensor(parameter['vertices']).unsqueeze(0).transpose(2, 1), torch.tensor(parameter['viewpoint_direction'] * parameter['viewpoint_radius']).unsqueeze(0), -math.exp(-7.0), False)
            hpr_pred = np.zeros(parameter['vertices'].shape[0])
            hpr_pred[pv_hpr_ind] = 1
            hpr_pred = hpr_pred.reshape(-1).astype(bool)
            parameter['visibility_bool'] = hpr_pred
        elif parameter['visibility'] == 'neu':
            if parameter['UNet'] is None or parameter['VisNet'] is None or parameter['vdr_embedder'] is None or parameter['feature'] is None:
                vdr_embedder, vd_ch = get_embedder(10)
                net = MyNet(6, vd_ch, 2).cuda()
                net.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=torch.device('cuda:0')))
                net.eval()
                parameter['vdr_embedder'] = vdr_embedder
                parameter['UNet'] = net.UNet.cuda()
                parameter['VisNet'] = net.VisNet.cuda()
                vertices_torch = torch.from_numpy(parameter['vertices']).float().view(-1, 3)
                
                p = Points(vertices_torch).cuda(non_blocking=True)
                bbmin, bbmax = p.bbox()
                p.normalize(bbmin, bbmax, scale=0.8)
                P = [p]
                P = ocnn.octree.merge_points(P)
                O = [points2octree(p)]
                O = ocnn.octree.merge_octrees(O)
                O.construct_all_neigh()
                query_pts = torch.cat([P.points, P.batch_id], dim=1)
                input_feature = get_input_feature(O)
                
                with torch.no_grad():
                    feature = parameter['UNet'](input_feature, O, 9, query_pts)
                parameter['feature'] = feature
            viewpoint_torch = torch.from_numpy(parameter['viewpoint_direction'] * parameter['viewpoint_radius']).float().view(-1)
            view_dirs = (vertices_torch - viewpoint_torch.unsqueeze(0))
            view_dirs = view_dirs / torch.norm(view_dirs, dim=1).unsqueeze(1)
            with torch.no_grad():
                tmp_feature = (feature * vdr_embedder(view_dirs).cuda()).reshape(-1, 63)
                model_logit = parameter['VisNet'](tmp_feature).view(-1, 2)
                model_pred = torch.argmax(model_logit, dim=-1).cpu().numpy()
            model_pred = model_pred.reshape(-1).astype(bool)
            parameter['visibility_bool'] = model_pred
        # render
        if parameter['delauny'] == 'off':
            if parameter['normals'] is not None and parameter['colors'] is not None and parameter['normal_switch'] and parameter['color_switch']:
                view_img = render(parameter['vertices'][parameter['visibility_bool'], :], parameter['viewpoint_direction'] * parameter['viewpoint_radius'], parameter['x_axis'], parameter['y_axis'], parameter['z_axis'], parameter['view_height'], parameter['view_width'], parameter['view_depth'], parameter['point_size'], parameter['point_color'], parameter['background_color'], parameter['normals'][parameter['visibility_bool'], :], parameter['colors'][parameter['visibility_bool'], :])
            elif parameter['normals'] is not None and parameter['normal_switch']:
                view_img = render(parameter['vertices'][parameter['visibility_bool'], :], parameter['viewpoint_direction'] * parameter['viewpoint_radius'], parameter['x_axis'], parameter['y_axis'], parameter['z_axis'], parameter['view_height'], parameter['view_width'], parameter['view_depth'], parameter['point_size'], parameter['point_color'], parameter['background_color'], parameter['normals'][parameter['visibility_bool'], :])
            elif parameter['colors'] is not None and parameter['color_switch']:
                view_img = render(parameter['vertices'][parameter['visibility_bool'], :], parameter['viewpoint_direction'] * parameter['viewpoint_radius'], parameter['x_axis'], parameter['y_axis'], parameter['z_axis'], parameter['view_height'], parameter['view_width'], parameter['view_depth'], parameter['point_size'], parameter['point_color'], parameter['background_color'], colors=parameter['colors'][parameter['visibility_bool'], :])
            else:
                view_img = render(parameter['vertices'][parameter['visibility_bool'], :], parameter['viewpoint_direction'] * parameter['viewpoint_radius'], parameter['x_axis'], parameter['y_axis'], parameter['z_axis'], parameter['view_height'], parameter['view_width'], parameter['view_depth'], parameter['point_size'], parameter['point_color'], parameter['background_color'])
        elif parameter['delauny'] == '2d':
            x = np.sum((parameter['vertices'][parameter['visibility_bool'], :] - parameter['viewpoint_direction'] * parameter['viewpoint_radius']) * x_axis.reshape(1, 3), axis=1)
            y = np.sum((parameter['vertices'][parameter['visibility_bool'], :] - parameter['viewpoint_direction'] * parameter['viewpoint_radius']) * y_axis.reshape(1, 3), axis=1)
            z = np.sum((parameter['vertices'][parameter['visibility_bool'], :] - parameter['viewpoint_direction'] * parameter['viewpoint_radius']) * z_axis.reshape(1, 3), axis=1)
            U = (parameter['view_depth'] / z * y + parameter['view_height'] / 2).astype(np.int32)
            V = (parameter['view_depth'] / z * x + parameter['view_width'] / 2).astype(np.int32)
            points = np.stack([V, U], axis=1, dtype=np.int32)
            tri = Delaunay(points)
            indices = tri.simplices
            if parameter['delauny_prune']:
                vis_vertices = parameter['vertices'][parameter['visibility_bool']]
                max_len = np.zeros((tri.simplices.shape[0]), dtype=np.float32)
                for j in range(tri.simplices.shape[0]):
                    idx1, idx2, idx3 = tri.simplices[j]
                    p1 = vis_vertices[idx1]
                    p2 = vis_vertices[idx2]
                    p3 = vis_vertices[idx3]
                    d1 = np.linalg.norm(p1 - p2)
                    d2 = np.linalg.norm(p2 - p3)
                    d3 = np.linalg.norm(p3 - p1)
                    max_len[j] = max(d1, d2, d3)
                indices = tri.simplices[max_len < parameter['average_nearest_distance'] * parameter['delauny_prune_scale'], :]
            view_img = np.ones((parameter['view_height'], parameter['view_width'], 3), dtype=np.uint8) * parameter['background_color'][::-1]
            view_img = view_img.astype(np.uint8)
            triangle_points = points[indices]
            triangle_points = triangle_points.reshape((-1, 3, 2))
            cv2.fillPoly(view_img, triangle_points, color=parameter['point_color'][::-1])
            cv2.polylines(view_img, triangle_points, isClosed=True, color=parameter['text_color'][::-1], thickness=1)
            view_img = cv2.flip(view_img, 0)
        elif parameter['delauny'] == '3d':
            x = np.sum((parameter['vertices'][parameter['visibility_bool'], :] - parameter['viewpoint_direction'] * parameter['viewpoint_radius']) * x_axis.reshape(1, 3), axis=1)
            y = np.sum((parameter['vertices'][parameter['visibility_bool'], :] - parameter['viewpoint_direction'] * parameter['viewpoint_radius']) * y_axis.reshape(1, 3), axis=1)
            z = np.sum((parameter['vertices'][parameter['visibility_bool'], :] - parameter['viewpoint_direction'] * parameter['viewpoint_radius']) * z_axis.reshape(1, 3), axis=1)
            U = (parameter['view_depth'] / z * y + parameter['view_height'] / 2).astype(np.int32)
            V = (parameter['view_depth'] / z * x + parameter['view_width'] / 2).astype(np.int32)
            points = np.stack([V, U], axis=1, dtype=np.int32)
            tri = Delaunay(points)
            indices = tri.simplices
            if parameter['delauny_prune']:
                vis_vertices = parameter['vertices'][parameter['visibility_bool']]
                max_len = np.zeros((tri.simplices.shape[0]), dtype=np.float32)
                for j in range(tri.simplices.shape[0]):
                    idx1, idx2, idx3 = tri.simplices[j]
                    p1 = vis_vertices[idx1]
                    p2 = vis_vertices[idx2]
                    p3 = vis_vertices[idx3]
                    d1 = np.linalg.norm(p1 - p2)
                    d2 = np.linalg.norm(p2 - p3)
                    d3 = np.linalg.norm(p3 - p1)
                    max_len[j] = max(d1, d2, d3)
                indices = tri.simplices[max_len < parameter['average_nearest_distance'] * parameter['delauny_prune_scale'], :]
            faces = np.zeros((indices.shape[0], 3, 3), dtype=np.float32)
            faces[:, 0, :] = parameter['vertices'][parameter['visibility_bool']][indices[:, 0]]
            faces[:, 1, :] = parameter['vertices'][parameter['visibility_bool']][indices[:, 1]]
            faces[:, 2, :] = parameter['vertices'][parameter['visibility_bool']][indices[:, 2]]
            view_img = render_mesh(faces, parameter['viewpoint_direction'] * parameter['viewpoint_radius'], parameter['x_axis'], parameter['y_axis'], parameter['z_axis'], parameter['view_height'], parameter['view_width'], parameter['view_depth'], parameter['point_color'], parameter['background_color'])
        view_img = frame_img(view_img, parameter['frame_width'], parameter['frame_color'])
        parameter['view_surface'] = pygame.surfarray.make_surface(view_img.transpose(1, 0, 2)[..., ::-1])
        parameter['rerender'] = False
        parameter['render_time_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Render {time.time() - s:5.2f}s', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
        # parameter['viewpoint_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint {parameter["viewpoint_radius"]*parameter["viewpoint_direction"]}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
        parameter['viewpoint_x_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint X {parameter["viewpoint_radius"]*parameter["viewpoint_direction"][0]:.2f}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
        parameter['viewpoint_y_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint Y {parameter["viewpoint_radius"]*parameter["viewpoint_direction"][1]:.2f}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
        parameter['viewpoint_z_surface'] = generate_text_surface(parameter['window_width'] - parameter['view_width'], parameter['arrow_height'] * 2, f'Viewpoint Z {parameter["viewpoint_radius"]*parameter["viewpoint_direction"][2]:.2f}', 0.7, 2, parameter['text_color'], parameter['background_color'], parameter['frame_color'])
        
    windowSurface.blit(parameter['point_size_title_surface'], (0, parameter['arrow_height'] * 0))
    windowSurface.blit(parameter['viewpoint_radius_title_surface'], (0, parameter['arrow_height'] * 2))
    windowSurface.blit(parameter['delauny_prune_scale_title_surface'], (0, parameter['arrow_height'] * 25))
    
    windowSurface.blit(parameter['point_size_value_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'] - parameter['arrow_height'] * 2, parameter['arrow_height'] * 0))
    windowSurface.blit(parameter['viewpoint_radius_value_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'] - parameter['arrow_height'] * 2, parameter['arrow_height'] * 2))
    windowSurface.blit(parameter['delauny_prune_scale_value_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'] - parameter['arrow_height'] * 2, parameter['arrow_height'] * 25))
    
    windowSurface.blit(parameter['point_size_up_arrow_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['arrow_height'] * 0))
    windowSurface.blit(parameter['point_size_down_arrow_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['arrow_height'] * 1))
    windowSurface.blit(parameter['viewpoint_radius_up_arrow_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['arrow_height'] * 2))
    windowSurface.blit(parameter['viewpoint_radius_down_arrow_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['arrow_height'] * 3))
    windowSurface.blit(parameter['delauny_prune_scale_up_arrow_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['arrow_height'] * 25))
    windowSurface.blit(parameter['delauny_prune_scale_down_arrow_surface'], (parameter['window_width'] - parameter['view_width'] - parameter['arrow_width'], parameter['arrow_height'] * 26))
    
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 4))
    
    windowSurface.blit(parameter['normal_switch_surface'], (0, parameter['arrow_height'] * 5))
    windowSurface.blit(parameter['color_switch_surface'], (0, parameter['arrow_height'] * 7))
    
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 9))
    
    windowSurface.blit(parameter['all_vertice_surface'], (0, parameter['arrow_height'] * 10))
    windowSurface.blit(parameter['hpr_vertice_surface'], (0, parameter['arrow_height'] * 12))
    windowSurface.blit(parameter['neu_vertice_surface'], (0, parameter['arrow_height'] * 14))
    
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 16))
    
    windowSurface.blit(parameter['delauny_off_surface'], (0, parameter['arrow_height'] * 17))
    windowSurface.blit(parameter['delauny_2d_surface'], (0, parameter['arrow_height'] * 19))
    windowSurface.blit(parameter['delauny_3d_surface'], (0, parameter['arrow_height'] * 21))
    windowSurface.blit(parameter['delauny_prune_switch_surface'], (0, parameter['arrow_height'] * 23))
    
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 27))
    
    windowSurface.blit(parameter['render_time_surface'], (0, parameter['arrow_height'] * 28))
    windowSurface.blit(parameter['viewpoint_x_surface'], (0, parameter['arrow_height'] * 30))
    windowSurface.blit(parameter['viewpoint_y_surface'], (0, parameter['arrow_height'] * 32))
    windowSurface.blit(parameter['viewpoint_z_surface'], (0, parameter['arrow_height'] * 34))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 30))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 31))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 32))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 33))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 34))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 35))
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 36))
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 37))
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 38))
    windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 39))
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 30))
    
    # windowSurface.blit(parameter['author_surface'], (0, parameter['arrow_height'] * 31))
    # windowSurface.blit(parameter['author_name_surface'], (0, parameter['arrow_height'] * 33))
    
    # windowSurface.blit(parameter['interval_surface'], (0, parameter['arrow_height'] * 35))
    
    # windowSurface.blit(parameter['project_surface'], (0, parameter['arrow_height'] * 36))
    # windowSurface.blit(parameter['project_name_surface'], (0, parameter['arrow_height'] * 38))
    
    windowSurface.blit(parameter['view_surface'], (parameter['window_width'] - parameter['view_width'], 0))
    pygame.display.update()