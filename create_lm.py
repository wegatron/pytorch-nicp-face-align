import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2mesh
from pytorch3d.ops.knn import knn_points
import argparse

def create_landmarks(target_head_obj_file, bfm_landmark_file, bfm_mat_file, device, output_file_prefix):
    meshes = io3d.load_obj_as_mesh(target_head_obj_file, device = device)

    with torch.no_grad():
        norm_meshes, _ = normalize_mesh(meshes)
        dummy_render = render.create_dummy_render([1.5, 0, 0], device = device)
        target_lm_index, lm_mask = get_mesh_landmark(norm_meshes, dummy_render, output_file_prefix)
        bfm_meshes, bfm_lm_index = load_bfm_model(device, bfm_mat_file)
        lm_mask = torch.all(lm_mask, dim = 0)
        bfm_lm_index_m = bfm_lm_index[:, lm_mask]
        target_lm_index_m = target_lm_index[:, lm_mask]

    fine_config = json.load(open('data-process/config/fine_grain.json'))
    io3d.save_meshes_as_objs([output_file_prefix + '_normalized.obj'], norm_meshes, save_textures = True)
    registered_mesh = non_rigid_icp_mesh2mesh(bfm_meshes, norm_meshes, bfm_lm_index_m, target_lm_index_m, fine_config)
    io3d.save_meshes_as_objs([output_file_prefix + '_bfm_face.obj'], registered_mesh, save_textures = True)

    corr_lm_inds = np.loadtxt(bfm_landmark_file)[:, 0]
    corr_lm_pts = registered_mesh.verts_padded()[:, corr_lm_inds, :]

    normalize_pts = norm_meshes.verts_padded()
    knn = knn_points(corr_lm_pts, normalize_pts)

    output_ids = np.ndarray((138,2), dtype=int)
    output_ids[:, 0] = corr_lm_inds
    output_ids[:, 1] = knn.idx[0, :, 0].cpu().numpy().astype(int)
    np.savetxt(output_file_prefix + '_lm.txt', output_ids, fmt='%d')
    print('land mark created!!!') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create align landmarks")
    parser.add_argument(
        "-i",
        "--input_obj",
        type=str,
        required=True,
        help="input obj file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output file prefix",
    )
    args = parser.parse_args()

    device = torch.device('cuda:0')    
    target_head_obj_file = args.input_obj
    align_output_file_prefix = args.output_dir+'/align'
    create_landmarks(
        target_head_obj_file=target_head_obj_file,
        bfm_landmark_file = 'MeInGame/data/mesh/lm_bfm_230.txt',
        bfm_mat_file='data-process/BFM/BFM09_model_info.mat',
        device = device,
        output_file_prefix=align_output_file_prefix)
    