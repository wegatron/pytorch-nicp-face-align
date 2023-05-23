cmake_minimum_required(VERSION 3.10)
project(merge_material)

set(CMAKE_CXX_STANDARD 17)


# 设置编译输出的路径
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)

find_package(OpenCV REQUIRED)

file(GLOB headers *.h *.hpp)
file(GLOB src *.cpp)

add_executable(merge_material ${src} ${headers})

target_link_libraries(merge_material ${OpenCV_LIBS})