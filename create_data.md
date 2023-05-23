## 创建可替换人头Mesh流程

1. 使用blender或其他建模软件将带脖子的部分人头分离, 将分离的mesh, 导出obj.
    ![](head_mesh.png)
    blender 导出obj 时可能会不附带纹理, 此时, 可以手动添加纹理图片.
    ```
    map_Kd target.png
    ```

2. 修改`data_creater.py`中的`target_head_obj_file`, 设置为你想要替换的人头mesh.

3. 运行`data_creater.py`, 得到如下文件:
    ```
    .
    ├── align
    │   ├── mine_aligned_bfm_face.mtl
    │   ├── mine_aligned_bfm_face.obj
    │   ├── mine_aligned_bfm_face.png
    │   ├── minelm.txt
    │   ├── mine_normalized_target.mtl
    │   ├── mine_normalized_target.obj
    │   ├── mine_normalized_target.png
    │   ├── mine_render_img.png
    │   ├── mine_target_std.mtl
    │   ├── mine_target_std.obj
    │   └── mine_target_std.png
    └── tex
        ├── material_0.png
        ├── material.mtl
        ├── mine_material.png
        ├── mine_skin_mask.png
        ├── mine_tex_c.npy
        ├── mine_textured_bfm_face.obj
        ├── mine_textured_target.obj
        └── mine_uv_mask.png
    ```

3. 选择脖子的几圈点, 导出neck_idx*.txt
    ![](neck_verts.png)
    ```python
import bpy
import bmesh
import numpy as np

index = 0 # here the index you want select please change 

obj = bpy.context.object
me = obj.data
bm = bmesh.from_edit_mesh(me)

idx = 0
select_idx = []
for v in bm.verts:
    if v.select:
        select_idx.append(idx)
    idx = idx + 1
ids = np.array(select_idx)
np.savetxt('/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/neck1.txt', ids, fmt='%d')

    #check
    select_idx = np.loadtxt('/home/wegatron/win-data/workspace/MeInGame/data/mesh/mine/neck1.txt').astype(np.int32)
    for v in bm.verts:
        v.select = False

    for i in select_idx:
        bm.verts[i].select = True
    bmesh.update_edit_mesh(me)
    ```

4. 资源拷贝
    * 将`align`下mine_normalized_target.obj以及其材质资源拷贝到`MeInGame/data/mesh/mine`下, 重命名为`target_std.obj`.
    * 将`tex`下`mine_tex_c.npy`拷贝到`MeInGame/data/mesh/mine`下.
    * 将`tex`下`mine_skin_mask.png`拷贝到`MeInGame/data/mesh/mine`下, 重命名为`skin_mask.png`
    * 将`tex`下`mine_uv_mask.png`拷贝到`MeInGame/data/mesh/mine`下, 重命名为`uv_mask.png`