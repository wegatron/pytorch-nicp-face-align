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
import nvdiffrast.torch as dr

#device = torch.device('cuda:0')
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

    fine_config = json.load(open('config/fine_grain.json'))
    io3d.save_meshes_as_objs([output_file_prefix + '_normalized_target.obj'], norm_meshes, save_textures = True)
    registered_mesh = non_rigid_icp_mesh2mesh(bfm_meshes, norm_meshes, bfm_lm_index_m, target_lm_index_m, fine_config)
    io3d.save_meshes_as_objs([output_file_prefix + '_aligned_bfm_face.obj'], registered_mesh, save_textures = True)

    corr_lm_inds = np.loadtxt(bfm_landmark_file)[:, 0]
    corr_lm_pts = registered_mesh.verts_padded()[:, corr_lm_inds, :]

    normalize_pts = norm_meshes.verts_padded()
    knn = knn_points(corr_lm_pts, normalize_pts)

    output_ids = np.ndarray((138,2), dtype=int)
    output_ids[:, 0] = corr_lm_inds
    output_ids[:, 1] = knn.idx[0, :, 0].cpu().numpy().astype(np.int)
    np.savetxt(output_file_prefix + 'lm.txt', output_ids, fmt='%d')
    print('land mark created!!!') 


def create_texture_map(head_mesh_file, face_mesh_file,
                       normalized_target_mesh_file, 
                       aligned_face_mesh_file,
                       mask_file_dir,
                       output_file_prefix):
    # Load a trimesh from a file
    mesh_bfm = trimesh.load(head_mesh_file)
    mesh_bfm_face = trimesh.load(face_mesh_file)

    #refer to https://trimsh.org/trimesh.proximity.html
    query = trimesh.proximity.ProximityQuery(mesh_bfm)
    # Find a point on the triangle
    closest, _, triangle_id = query.on_surface(mesh_bfm_face.vertices)

    # refere to https://trimsh.org/trimesh.triangles.html
    bc = trimesh.triangles.points_to_barycentric(mesh_bfm.triangles[triangle_id], closest)

    #verify
    print(np.dot(mesh_bfm.triangles[triangle_id[0]].T, bc[0]) - closest[0])
    # np.savetxt('../result/triangles_id.txt', np.array(triangle_id))
    # np.savetxt('../result/barycentric.txt', np.array(bc))

    uv_bfm = np.array(mesh_bfm.visual.uv) # 35709 x 2
    u = uv_bfm[:, 0]
    v = uv_bfm[:, 1]

    faces = mesh_bfm.faces[triangle_id]
    u_f = u[faces] # n x 3
    v_f = v[faces] # n x 3

    # bfm face vertex's uv in bfm head uv coordinate space
    u_in_bfm = np.sum(u_f * bc, axis=1)
    v_in_bfm = np.sum(v_f * bc, axis=1)

    # save the output mesh, verify
    mesh_bfm_face.visual.uv = np.stack([u_in_bfm, v_in_bfm], axis=1)
    trimesh.exchange.export.export_mesh(mesh_bfm_face, output_file_prefix+'_textured_bfm_face.obj')

    mesh_bfm_face_aligned = trimesh.load(aligned_face_mesh_file)
    mesh_target = trimesh.load(normalized_target_mesh_file)
    query = trimesh.proximity.ProximityQuery(mesh_bfm_face_aligned)
    closest_2, distance, triangle_id_2 = query.on_surface(mesh_target.vertices)
    bc_2 = trimesh.triangles.points_to_barycentric(mesh_bfm_face_aligned.triangles[triangle_id_2], closest_2) # n x 3

    vertices_mask = distance<1e-2
    face_mask = np.all(vertices_mask[mesh_target.faces], axis=1)

    # target face vertex's uv in bfm head uv coordinate space
    faces2 = mesh_bfm_face_aligned.faces[triangle_id_2]
    u_f2 = u_in_bfm[faces2] # nf x 3
    v_f2 = v_in_bfm[faces2] # nf x 3

    u_final = np.sum(u_f2 * bc_2, axis=1)
    v_final = np.sum(v_f2 * bc_2, axis=1)
    uv_final = np.stack([u_final, v_final], axis=1) # nf x 2

    # save the output mesh, verify
    target_uv_ori = np.array(mesh_target.visual.uv).copy()
    mesh_target.visual.uv = uv_final
    mesh_target.faces = mesh_target.faces[face_mask]
    #mesh_target.triangles = mesh_target.triangles[face_mask]
    trimesh.exchange.export.export_mesh(mesh_target, output_file_prefix+'_textured_target.obj')

    vertices = np.zeros((mesh_target.vertices.shape[0], 4), dtype=np.float32)
    vertices[:, :2] = target_uv_ori*2 - 1
    vertices[:, 1] = -vertices[:, 1] # flip y
    vertices[:, 3] = 1
    
    mesh_target.vertices = vertices[:, :3]
    trimesh.exchange.export.export_mesh(mesh_target, output_file_prefix+'_textured_mesh.obj')

    target_uv_width = 1024
    target_uv_height = 1024

    tris = np.array(mesh_target.faces)

    # np.save('/home/wegatron/tmp/vertices.npy', vertices)
    # np.save('/home/wegatron/tmp/tris.npy', tris)
    # np.save('/home/wegatron/tmp/uv_final.npy', uv_final)

    #load data
    # vertices = np.load('/home/wegatron/tmp/vertices.npy')
    # tris = np.load('/home/wegatron/tmp/tris.npy')
    # uv_final = np.load('/home/wegatron/tmp/uv_final.npy')

    glctx = dr.RasterizeCudaContext()
    vertices_t = torch.from_numpy(vertices).cuda()[None]
    tris_t = torch.from_numpy(tris).to(device)

    tex_c = (tex_c[0] * fg_mask[0, ...]).cpu().numpy()[0]
    tex_c[:,:, 0] = tex_c[:,:, 0] * (target_uv_width - 1) # u
    tex_c[:,:, 1] = tex_c[:,:, 1] * (target_uv_height - 1) # v
    tex_c = tex_c.astype(np.int32)
    np.save(output_file_prefix+'_tex_c.npy', tex_c)

    # generate mask
    valid_mask = np.expand_dims(np.logical_or(tex_c[..., 0] != 0, tex_c[..., 1] != 0), axis=2)
    uv_mask_img_ori = np.array(Image.open(mask_file_dir+'/uv_mask.png'))
    uv_mask_img = uv_mask_img_ori[tex_c[:, :, 1], tex_c[:, :, 0], :]
    uv_mask_img = uv_mask_img * valid_mask
    uv_mask_img = Image.fromarray(uv_mask_img)
    uv_mask_img.save(output_file_prefix+'_uv_mask.png')

    skin_mask_img_ori = np.array(Image.open(mask_file_dir+'/skin_mask.png'))
    skin_mask_img = skin_mask_img_ori[tex_c[:, :, 1], tex_c[:, :, 0], :]
    skin_mask_img = skin_mask_img * valid_mask
    skin_mask_img = Image.fromarray(skin_mask_img)
    skin_mask_img.save(output_file_prefix+'_skin_mask.png')
    print('texture map created!!!')


def expand_one_ring(mesh, vertices):
    one_ring = set(vertices)
    for v in vertices:
        one_ring.update(mesh.vertex_neighbors[v])
    new_vertices = one_ring.difference(vertices)
    return np.array(list(one_ring)), np.array(list(new_vertices))

def calc_vertex_dis(mesh, vids):
    vs = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze_(0)
    dis = knn_points(vs, vs[:, vids, :])
    return dis.dists[0,:,0].numpy()


if __name__ == "__main__":
    device = torch.device('cuda:0')
    name = 'mine'
    align_output_file_prefix = 'output/align/'+name
    target_head_obj_file = '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/target.obj'
    create_landmarks(
        target_head_obj_file=target_head_obj_file,
        bfm_landmark_file = '/home/wegatron/win-data/workspace/MeInGame/data/mesh/lm_bfm_230.txt',
        bfm_mat_file='BFM/BFM09_model_info.mat',
        device = device,
        output_file_prefix=align_output_file_prefix)
    
    # ### normalize mesh to std mesh
    mesh = io3d.load_obj_as_mesh(target_head_obj_file)
    io3d.save_meshes_as_objs([align_output_file_prefix+'_target.obj'], mesh, save_textures=True)
    std_mesh = io3d.load_obj_as_mesh('/home/wegatron/win-data/workspace/MeInGame/data/mesh/bfm09_face.obj')
    lm = np.loadtxt(align_output_file_prefix+'lm.txt')
    out_mesh = normalize_to_std(mesh, std_mesh, lm)
    io3d.save_meshes_as_objs([align_output_file_prefix+'_target_std.obj'], out_mesh, save_textures=True)
    
    tex_output_file_prefix = 'output/tex/'+name

    # # 为了生成纹理map, 需要保证source, target mesh的纹理只有一个, 才能保证纹理map的正确性
    target_head_clean_obj_file = '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/target_head_clean.obj'
    meshes = io3d.load_obj_as_mesh(target_head_clean_obj_file)
    norm_meshes, _ = normalize_mesh(meshes)
    io3d.save_meshes_as_objs([tex_output_file_prefix + '_normalized_target_clean.obj'], norm_meshes, save_textures = True)    
    create_texture_map(
        head_mesh_file='/home/wegatron/win-data/workspace/MeInGame/data/mesh/nsh_bfm_face.obj',
        face_mesh_file='/home/wegatron/win-data/workspace/MeInGame/data/mesh/bfm09_face.obj',
        normalized_target_mesh_file=tex_output_file_prefix + '_normalized_target_clean.obj',
        aligned_face_mesh_file=align_output_file_prefix+'_aligned_bfm_face.obj',
        mask_file_dir='/home/wegatron/win-data/workspace/MeInGame/data/uv_param/masks',
        output_file_prefix=tex_output_file_prefix)

    ## weight
    mesh_bfm_face = trimesh.load(align_output_file_prefix + '_aligned_bfm_face.obj')
    target_mesh = trimesh.load(align_output_file_prefix + '_normalized_target.obj', process=False, maintain_order=True)
    query = trimesh.proximity.ProximityQuery(mesh_bfm_face)
    _, distance, _ = query.on_surface(target_mesh.vertices)
    inner_ids = np.where(distance < 1e-2)[0]
    in_dis = np.expand_dims(calc_vertex_dis(target_mesh, inner_ids), axis=1)
      
    out_ids = np.loadtxt('/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/neck.txt').astype(np.int32)
    out_dis = np.expand_dims(calc_vertex_dis(target_mesh, out_ids), axis=1)
    
    weight = out_dis / (in_dis + out_dis)
    np.save(align_output_file_prefix+'_weight.npy', weight)

    debug_pts = np.concatenate((target_mesh.vertices, in_dis, out_dis, weight), axis=1)
    np.savetxt(align_output_file_prefix+'_debug.txt', debug_pts)

    # copy data to destination
    # lm.txt, tex_c.npy, weight.npy
    # shutil.copy(align_output_file_prefix+'lm.txt', '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/lm.txt')
    # shutil.copy(align_output_file_prefix+ '_weight.npy', '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/mine_weight.npy')
    # shutil.copy(tex_output_file_prefix+'_tex_c.npy', '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/mine_tex_c.npy')
    # shutil.copy(tex_output_file_prefix+'_uv_mask.png', '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/uv_mask.png')
    # shutil.copy(tex_output_file_prefix+'_skin_mask.png', '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/skin_mask.png')