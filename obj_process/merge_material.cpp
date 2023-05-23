#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// merge materials albedo into one
#ifndef _WIN32
    const char dirsep = '/';
#else
    const char dirsep = '\\';
#endif

std::string getDirectoryFromFilePath(std::string& path) {
    size_t found = path.find_last_of("/\\");
    if (found != std::string::npos) {
        std::string directory = path.substr(0, found);
        return directory;
    }
    return "";
}

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <inputfile> <outputdir>" << std::endl;
        return 1;
    }
    //std::string inputfile = "/home/wegatron/win-data/workspace/head_fusion/MeInGame/data/mesh/mine/target_raw.obj";
    std::string inputfile = argv[1];
    std::string output_dir = argv[2];

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;
    std::string base_dir = getDirectoryFromFilePath(inputfile);   
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), base_dir.c_str());
    if (!warn.empty()) std::cout << warn << std::endl;
    if (!err.empty()) std::cerr << err << std::endl;
    if (!ret) exit(1);
  
    std::vector<cv::Mat> imgs(shapes.size());
    std::vector<int> img_cols(shapes.size());
    int max_rows = 0;
    for (size_t s = 0; s < shapes.size(); s++) {
        assert(shapes[s].mesh.material_ids.size() > 0);
        int material_id = shapes[s].mesh.material_ids[0];
        for(auto& face_material_id : shapes[s].mesh.material_ids) {
            assert(face_material_id == material_id);
        }

        // merge images
        std::string img_path = base_dir + dirsep + materials[material_id].diffuse_texname;
        imgs[s] = cv::imread(img_path);
        max_rows = std::max(imgs[s].rows, max_rows);
        img_cols[s] = imgs[s].cols;
    }
    
    std::vector<cv::Mat> resized_imgs;
    for(auto& img : imgs) {
        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(img.cols, max_rows));
        resized_imgs.push_back(img_resized);
    }
    cv::Mat out_img;
    cv::hconcat(resized_imgs, out_img);
    cv::imwrite(output_dir+"merged_albedo.png", out_img);

    // update texcoords
    std::vector<bool> texcoord_used(attrib.texcoords.size(), false);
    int offset_cols = 0;
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                assert(idx.texcoord_index >= 0);
                tinyobj::real_t &tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                tinyobj::real_t &ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                tinyobj::real_t ntx = (tx * img_cols[s] + offset_cols) / out_img.cols;               
                if(texcoord_used[idx.texcoord_index]) {                    
                    attrib.texcoords.emplace_back(ntx);
                    attrib.texcoords.emplace_back(ty);
                    idx.texcoord_index = attrib.texcoords.size()/2-1;
                }
                else {
                    tx = ntx;
                    texcoord_used[idx.texcoord_index] = true;
                }
            }
            index_offset += fv;
        } // end of loop face
        offset_cols += img_cols[s];
    } // end of loop shape

    //// export obj
    // output material file
    std::ofstream out_mtl(output_dir+"merged.mtl");
    if (!out_mtl) {
        std::cerr << "Cannot open merged.mtl" << std::endl;
        exit(1);
    }
    out_mtl << "newmtl merged" << std::endl;
    out_mtl << "map_Kd merged_albedo.png" << std::endl;
    out_mtl.close();

    std::ofstream out(output_dir+"output.obj");
    if (!out) {
        std::cerr << "Cannot open output.obj" << std::endl;
        exit(1);
    }
    out << "mtllib merged.mtl" << std::endl;
    out << "usemtl merged" << std::endl;

    // output v
    for (size_t i = 0; i < attrib.vertices.size() / 3; i++) {
        out << "v " << attrib.vertices[3*i+0] << " " << attrib.vertices[3*i+1] << " " << attrib.vertices[3*i+2] << std::endl;
    }

    // output vt
    for (size_t i = 0; i < attrib.texcoords.size() / 2; i++) {
        out << "vt " << attrib.texcoords[2*i+0] << " " << attrib.texcoords[2*i+1] << std::endl;
    }
    // output vn
    for (size_t i = 0; i < attrib.normals.size() / 3; i++) {
        out << "vn " << attrib.normals[3*i+0] << " " << attrib.normals[3*i+1] << " " << attrib.normals[3*i+2] << std::endl;
    }
    // output f
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            out << "f ";
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                if(idx.normal_index < 0)
                    out << idx.vertex_index+1 << "/" << idx.texcoord_index+1 << " ";
                else
                    out << idx.vertex_index+1 << "/" << idx.texcoord_index+1 << "/" << idx.normal_index+1 << " ";                
            }
            index_offset += fv;
            out << std::endl;
        }
    }
    out.close();
    return 0;
}