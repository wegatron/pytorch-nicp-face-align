import shutil
import torch
import io3d
import render
import numpy as np
import json
import trimesh
from utils import normalize_mesh, normalize_to_std
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2mesh
from pytorch3d.ops.knn import knn_points
import PIL.Image as Image
import torchvision.transforms as transforms
import nvdiffrast.torch as dr
import argparse
import cv2


# device = torch.device('cuda:0')
def create_landmarks(target_head_obj_file, bfm_landmark_file, bfm_mat_file, device, output_file_prefix):
    meshes = io3d.load_obj_as_mesh(target_head_obj_file, device=device)

    with torch.no_grad():
        norm_meshes, _ = normalize_mesh(meshes)
        dummy_render = render.create_dummy_render([1.5, 0, 0], device=device)
        target_lm_index, lm_mask = get_mesh_landmark(norm_meshes, dummy_render, output_file_prefix)
        bfm_meshes, bfm_lm_index = load_bfm_model(device, bfm_mat_file)
        lm_mask = torch.all(lm_mask, dim=0)
        bfm_lm_index_m = bfm_lm_index[:, lm_mask]
        target_lm_index_m = target_lm_index[:, lm_mask]

    fine_config = json.load(open('config/fine_grain.json'))
    io3d.save_meshes_as_objs([output_file_prefix + '_normalized_target.obj'], norm_meshes, save_textures=True)
    registered_mesh = non_rigid_icp_mesh2mesh(bfm_meshes, norm_meshes, bfm_lm_index_m, target_lm_index_m, fine_config)
    io3d.save_meshes_as_objs([output_file_prefix + '_aligned_bfm_face.obj'], registered_mesh, save_textures=True)

    corr_lm_inds = np.loadtxt(bfm_landmark_file)[:, 0]
    corr_lm_pts = registered_mesh.verts_padded()[:, corr_lm_inds, :]

    normalize_pts = norm_meshes.verts_padded()
    knn = knn_points(corr_lm_pts, normalize_pts)

    output_ids = np.ndarray((138, 2), dtype=int)
    output_ids[:, 0] = corr_lm_inds
    output_ids[:, 1] = knn.idx[0, :, 0].cpu().numpy().astype(np.int)
    np.savetxt(output_file_prefix + 'lm.txt', output_ids, fmt='%d')
    print('land mark created!!!')


def create_texture_map_aligned_mesh(
        src_mesh_path,
        target_mesh_path,
        tex_valid_mask = None
):
    src_mesh = trimesh.load(src_mesh_path, process=False, maintain_order=True)
    target_mesh = trimesh.load(target_mesh_path, process=False, maintain_order=True)

    query = trimesh.proximity.ProximityQuery(src_mesh)
    closest, distance, triangle_id = query.on_surface(target_mesh.vertices)
    bc = trimesh.triangles.points_to_barycentric(src_mesh.triangles[triangle_id], closest)  # n x 3
    bc_exp = np.expand_dims(bc, axis=2)

    #debug
    cp = np.array(closest)
    np.savetxt('debug_output/cp.txt', cp)
    fv = np.array(src_mesh.triangles[triangle_id])
    bc_pts = np.sum(fv * bc_exp, axis=1)
    diff = bc_pts - cp
    print(diff.max())

    # target face vertex's uv in src mesh's uv coordinate space
    # Load a trimesh from a file
    src_mesh_2 = io3d.load_obj_as_mesh(src_mesh_path)
    # face's uv index may not the same as face's vertex index
    src_face_uv=src_mesh_2.textures.faces_uvs_padded()[0].numpy()
    faces = src_face_uv[triangle_id]
    src_mesh_uv = src_mesh_2.textures.verts_uvs_padded()[0].numpy() # 35709 x 2
    uv_faces = src_mesh_uv[faces]
    uv_final = np.sum(uv_faces * bc_exp, axis=1)

    vertices_mask = distance < 15e-4 # distance torrance
    face_mask = np.all(vertices_mask[target_mesh.faces], axis=1)
    target_mesh_2 = io3d.load_obj_as_mesh(target_mesh_path)
    ori_uv = target_mesh_2.textures.verts_uvs_padded()[0].numpy()
    assert np.array_equal(np.array(target_mesh.faces), target_mesh_2.faces_padded()[0].numpy())
    target_uv_faces = target_mesh_2.textures.faces_uvs_padded()[0].numpy()

    # mask out invalid uv
    if tex_valid_mask is not None:
        extent = np.zeros((1,2))
        extent[0, 0] = tex_valid_mask.shape[1] # width, cols
        extent[0, 1] = tex_valid_mask.shape[0] # height, rows
        coord = (uv_final * extent).astype(int)
        tv = np.sum(tex_valid_mask, axis=2)
        # 0,0 is bottom left in uv coordinate
        uv_mask = tv[int(extent[0, 1]) - coord[:, 1], coord[:, 0]]
        uv_face_mask = np.all(uv_mask[target_mesh.faces], axis=1)
        face_mask = face_mask * uv_face_mask

    #debug 0
    tmp_mesh = target_mesh.copy()
    tmp_mesh.visual.uv = uv_final.copy()
    tmp_mesh.faces = tmp_mesh.faces[face_mask]
    tmp_mesh.visual.material.image = src_mesh.visual.material.image
    trimesh.exchange.export.export_mesh(tmp_mesh, 'debug_output/textured_target.obj')

    vertices = np.zeros((ori_uv.shape[0], 4), dtype=np.float32)
    vertices[:, :2] = ori_uv*2 - 1
    vertices[:, 1] = -vertices[:, 1] # flip y
    vertices[:, 3] = 1
    tris = target_uv_faces[face_mask]

    # debug 1
    tmp_mesh = target_mesh.copy()
    tmp_mesh.visual.uv = uv_final.copy()
    tmp_mesh.faces = tmp_mesh.faces[face_mask]
    tmp_mesh.visual.material.image = src_mesh.visual.material.image
    tmp_mesh.vertices = vertices
    trimesh.exchange.export.export_mesh(tmp_mesh, 'debug_output/textured_quad.obj')

    glctx = dr.RasterizeCudaContext()
    vertices_t = torch.from_numpy(vertices).to(device)[None]
    tris_t = torch.from_numpy(tris).to(device).to(torch.int32)
    rast, _ = dr.rasterize(glctx, vertices_t, tris_t, resolution=[1024, 1024])
    fg_mask = torch.clamp(rast[..., -1:], 0, 1)

    uv_final[:, 1] = 1.0 - uv_final[:, 1] # coordinate in rasterization 0,0 is top left.
    uv_final_t = torch.from_numpy(uv_final).to(torch.float32).cuda()[None]
    tex_c, _ = dr.interpolate(uv_final_t, rast, tris_t)

    #debug texture
    tensor_image = torch.from_numpy(np.array(src_mesh.visual.material.image)).float().cuda()
    debug_img = dr.texture(tensor_image[None], tex_c, filter_mode='linear')
    debug_img_np = debug_img[0].cpu().numpy().astype(np.uint8)
    cv2.imwrite('debug_output/debug_raster_texture.png', debug_img_np)

    tex_map = (tex_c[0] * fg_mask[0, ...]).cpu().numpy()
    return tex_map


def create_texture_map(src_mesh_path,
                       src_tex_valid_mask_path,
                       src_lm_path,
                       target_mesh_path,
                       target_lm_path,
                       device,
                       output_dir):
    """
    create texture map from src_mesh to target_mesh, save to output_file_prefix+"tex_map.npy"
    tex_map.npy is a 2d array, use it to map pixel from src mesh's texture to target mesh's texture
    target_mesh_texture = src_mesh_texture[tex_map[0], tex_map[1]]
    """
    #nicp align two mesh
    src_mesh = io3d.load_obj_as_mesh(src_mesh_path, device=device)
    target_mesh = io3d.load_obj_as_mesh(target_mesh_path, device=device)

    src_lm_raw = np.loadtxt(src_lm_path)
    src_lm = torch.from_numpy(src_lm_raw[:, 1]).long().to(device)[None, ...]

    target_lm_raw = np.loadtxt(target_lm_path)
    target_lm = torch.from_numpy(target_lm_raw[:, 1]).long().to(device)[None, ...]

    fine_config = json.load(open('data-process/config/fine_grain.json'))
    aligned_src_mesh = non_rigid_icp_mesh2mesh(src_mesh, target_mesh, src_lm, target_lm, fine_config)
    aligned_src_mesh_path = 'debug_output/aligned_src.obj'
    io3d.save_meshes_as_objs([aligned_src_mesh_path], aligned_src_mesh, save_textures=True)

    src_valid_mask = cv2.imread(src_tex_valid_mask_path)
    tex_map = create_texture_map_aligned_mesh(aligned_src_mesh_path, target_mesh_path, src_valid_mask)
    np.save(output_dir + '/tex_map.npy', tex_map)

    # generate mask
    valid_mask = np.expand_dims(np.logical_or(tex_map[..., 0] != 0, tex_map[..., 1] != 0), axis=2)
    src_img = src_mesh.textures.maps_padded()[0].cpu().numpy()
    extent = np.zeros((1,2), dtype=float)
    extent[0,0] = src_img.shape[1] # cols, width, u
    extent[0,1] = src_img.shape[0] # rows, height, v
    r_tex_map = (tex_map * extent).astype(int)
    mapped_tex = (src_img[r_tex_map[:, :, 1], r_tex_map[:, :, 0]] * valid_mask * 255).astype(int)
    cv2.imwrite('debug_output/mapped_tex.png', mapped_tex)
    print('done')


if __name__ == "__main__":
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description="create align landmarks")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="src obj file",
    )
    parser.add_argument(
        "--src_tex_valid_mask",
        type=str,
        required=True,
        help="src texture valid mask",
    )
    parser.add_argument(
        "--src_lm",
        type=str,
        required=True,
        help="src landmark",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="target obj file",
    )
    parser.add_argument(
        "--target_lm",
        type=str,
        required=True,
        help="target landmark",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output_dir",
    )

    args = parser.parse_args()
    create_texture_map(
        src_mesh_path=args.src,
        src_lm_path=args.src_lm,
        src_tex_valid_mask_path=args.src_tex_valid_mask,
        target_mesh_path=args.target,
        target_lm_path=args.target_lm,
        device=device,
        output_dir=args.output_dir)

    print('texture map created!!!')
