import shutil
import torch
import io3d
import render
import numpy as np
import json
import trimesh
import pyassimp
from utils import normalize_mesh, normalize_to_std
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2mesh
from pytorch3d.ops.knn import knn_points
import PIL.Image as Image
import nvdiffrast.torch as dr

def mapTextures(target_mesh, source_mesh, images, target_uv_size):
    #refer to https://trimsh.org/trimesh.proximity.html
    query = trimesh.proximity.ProximityQuery(source_mesh)
    # Find a point on the triangle
    closest, _, triangle_id = query.on_surface(target_mesh.vertices)    
    bc = trimesh.triangles.points_to_barycentric(source_mesh.triangles[triangle_id], closest)
    source_uv = np.array(target_mesh.visual.uv) # 35709 x 2
    u = source_uv[:, 0]
    v = source_uv[:, 1]

    source_faces = source_mesh.faces[triangle_id]
    u_f = u[source_faces] # n x 3
    v_f = v[source_faces] # n x 3

    # target mesh face vertex's uv in source mesh's uv coordinate space
    res_u = np.sum(u_f * bc, axis=1)
    res_v = np.sum(v_f * bc, axis=1)
    uv_final = np.stack([res_u, res_v], axis=1) # nf x 2
    np.save('uv_final.npy', uv_final)
    #uv_final = np.load('uv_final.npy')
    target_mesh.visual.uv = uv_final
    target_mesh.visual.material.image = Image.fromarray(images[0])
    trimesh.exchange.export.export_mesh(target_mesh, 'output_transfer/debug_textured_target.obj')    

    # rasterize bake the result
    vertices = np.zeros((target_mesh.visual.uv.shape[0], 4), dtype=np.float32)
    vertices[:, :2] = target_mesh.visual.uv*2 - 1
    vertices[:, 1] = -vertices[:, 1] # flip y
    vertices[:, 3] = 1

    glctx = dr.RasterizeCudaContext()
    vertices_t = torch.from_numpy(vertices).cuda()[None]
    tris_t = torch.from_numpy(target_mesh.faces).to(torch.int32).cuda()
    uv_final_t = torch.from_numpy(uv_final).to(torch.float32).cuda()[None]

    rast, _ = dr.rasterize(glctx, vertices_t, tris_t, resolution=[target_uv_size[1], target_uv_size[0]])
    tex_c, _ = dr.interpolate(uv_final_t, rast, tris_t)
    fg_mask = torch.clamp(rast[..., -1:], 0, 1)
    
    ret_imgs = []
    for img in images:
        img_t = torch.tensor(img, dtype=torch.float32).cuda() * 0.00392156862
        color = dr.texture(img_t[None, ...], tex_c, filter_mode='linear')
        color = color * fg_mask  # Mask out background.
        np_img = color.cpu().numpy()
        ret_imgs.append(np_img)
    return ret_imgs  


def transfer(target_head_obj_file,
             source_head_obj_file,
             source_head_face_mask,             
             source_head_skin_mask,             
             device, output_file_prefix):    
    target_mesh = io3d.load_obj_as_mesh(target_head_obj_file, device = device)
    n_target_mesh, _ = normalize_mesh(target_mesh)

    source_mesh = io3d.load_obj_as_mesh(source_head_obj_file, device = device)
    n_source_mesh, _ = normalize_mesh(source_mesh)

    with torch.no_grad():        
        dummy_render = render.create_dummy_render([1.5, 0, 0], device = device)
        target_lm_index, target_lm_mask = get_mesh_landmark(n_target_mesh, dummy_render, output_file_prefix+ '_target_')
        source_lm_index, source_lm_mask = get_mesh_landmark(n_source_mesh, dummy_render, output_file_prefix+ '_source_')
        
        target_lm_mask = torch.all(target_lm_mask, dim = 0)
        source_lm_mask = torch.all(source_lm_mask, dim = 0)
        lm_mask = target_lm_mask & source_lm_mask
        source_lm_index_m = source_lm_index[:, lm_mask]
        target_lm_index_m = target_lm_index[:, lm_mask]

    fine_config = json.load(open('config/fine_grain.json'))
    registered_mesh = non_rigid_icp_mesh2mesh(source_mesh, target_mesh, source_lm_index_m, target_lm_index_m, fine_config)
    io3d.save_meshes_as_objs([output_file_prefix + '_aligned_source.obj'], registered_mesh, save_textures = True)

    # transfer texture
    # generate texture map
    target_mesh = trimesh.load(output_file_prefix + '_aligned_source.obj', process = False, maintain_order = True)
    source_mesh = trimesh.load(source_head_obj_file, process = False, maintain_order = True)
    source_texture_image = np.array(source_mesh.visual.material.image)
    src_images = [source_texture_image]
    mapped_imgs = mapTextures(target_mesh, source_mesh, src_images, target_mesh.visual.material.image.size)
    
    #target_mesh.visual.
    #mapped_res = mapTextures(target_mesh, source_mesh, images)
    # blender fusion mapped res to target mesh
    print('done')


def preprocessMesh(input_obj, output_obj):
    scene = pyassimp.load(input_obj)
    pyassimp.export(scene, output_obj, file_type='obj')
    mesh = trimesh.load_mesh(input_obj, process = False, maintain_order = True, validate = False)
    trimesh.exchange.export.export_mesh(mesh, output_obj)
            
    if type(mesh) == trimesh.Scene:
        imgs = []
        vertices = np.zeros((0, 3))
        faces=[]
        max_width = 0
        out_mesh = None
        n_uv = 0
        for tmesh in mesh.geometry.values():
            if tmesh.vertices.shape[0] > vertices.shape[0]:
                vertices = tmesh.vertices
                out_mesh = tmesh
            n_uv = max(n_uv, tmesh.visual.uv.shape[0]);
            faces.append(np.array(tmesh.faces))
            imgs.append(np.array(tmesh.visual.material.image))
            max_width = max_width + tmesh.visual.material.image.size[0]

        out_uv = np.zeros((n_uv, 2), np.float32)
        offset = 0
        i = 0
        for tmesh in mesh.geometry.values():
            vids = np.unique(tmesh.faces.flatten())
            out_uv[vids, 1] = tmesh.visual.uv[vids, 1]
            out_uv[vids, 0] = (tmesh.visual.uv[vids, 0] * imgs[i].shape[0] + offset)/max_width
            offset += imgs[i].shape[0]
            i += 1

        # do merge
        res_img = np.concatenate(imgs, axis = 1)
        out_mesh.visual.uv = out_uv
        out_mesh.faces = np.concatenate(faces, axis = 0)
        out_mesh.visual.material.image = Image.fromarray(res_img)
        trimesh.exchange.export.export_mesh(out_mesh, output_obj)


if __name__ == "__main__":
    device = torch.device('cuda:0')
    name = 'mine'
    align_output_file_prefix = 'output_transfer/'
    target_head_obj_file = '/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/target_raw.obj'
    target_head_obj_file = '/home/wegatron/win-data/workspace/FFHQ-UV/examples/pan_multiview_test1_with_eyeball/pan.obj'
    preprocessMesh(target_head_obj_file, align_output_file_prefix + 'preprocess/target.obj')
    # transfer(
    #     target_head_obj_file=align_output_file_prefix + 'preprocess/target.obj',
    #     source_head_obj_file = '/home/wegatron/win-data/workspace/FFHQ-UV/examples/pan_multiview_test1/unwarp/pan/pan_front.obj',
    #     source_head_face_mask = '/home/wegatron/win-data/workspace/FFHQ-UV/examples/pan_multiview_test1/unwarp/pan/pan_front_face_mask.png',
    #     source_head_skin_mask = '/home/wegatron/win-data/workspace/FFHQ-UV/examples/pan_multiview_test1/unwarp/pan/pan_front_skin_mask.png',
    #     device = device,
    #     output_file_prefix=align_output_file_prefix+'transfered')