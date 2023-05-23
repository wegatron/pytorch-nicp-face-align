# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import numpy as np
import io3d
from scipy.io import loadmat
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from utils import normalize_mesh

def load_bfm_model(device = torch.device('cpu'), bfm_path = 'BFM/BFM09_model_info.mat'):
    bfm_meta_data = loadmat(bfm_path)
    vertex = bfm_meta_data['meanshape'].reshape(-1, 3)
    #mesh_bfm_face = trimesh.load('/home/wegatron/win-data/workspace/MeInGame/data/mesh/bfm09_face.obj')
    faces = bfm_meta_data['tri']
    lm_index = bfm_meta_data['keypoints']
    #color = bfm_meta_data['meantex'].reshape(-1, 3)
    #color = torch.from_numpy(color).to(device).unsqueeze(0)
    vertex = torch.from_numpy(vertex).to(device)
    faces = torch.from_numpy(faces).long().to(device) - 1
    lm_index = torch.from_numpy(lm_index).long().to(device)
    #textures = TexturesVertex(color)
    tex_coords = torch.from_numpy(bfm_meta_data['uv']).to(device).unsqueeze(0)
    maps = torch.zeros((512, 512, 3), dtype = torch.float32, device = device)
    textures = TexturesUV(maps=[maps], faces_uvs=[faces], verts_uvs=tex_coords)
    bfm_mesh = Meshes([vertex], [faces], textures)
    #io3d.save_meshes_as_objs(['z_aligned_bfm_face.obj'], bfm_mesh, save_textures = True)
    norm_mesh, _ = normalize_mesh(bfm_mesh)
    return norm_mesh, lm_index

if __name__ == "__main__":
    import render
    import torchvision
    bfm_mesh, lm_index = load_bfm_model(torch.device('cuda:0'))
    dummy_render = render.create_dummy_render([1, 0, 0], device = torch.device('cuda:0'))
    images = dummy_render(bfm_mesh).squeeze()
    torchvision.utils.save_image(images.permute(2, 0, 1) / 255, 'test_data/test_bfm.png')