cmake_minimum_required(VERSION 3.0)

project(divergence
        VERSION 1.0
        LANGUAGES CXX)

find_package(ArrayFire REQUIRED)
find_package(Forge REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(div div.cpp)
target_link_libraries(div PUBLIC ArrayFire::afcuda Forge::forge)

add_executable(discrete discrete.cpp)
target_link_libraries(discrete PUBLIC ArrayFire::afcuda Forge::forge)

add_executable(cfd cfd.cpp)
target_link_libraries(cfd PUBLIC ArrayFire::afcuda Forge::forge)

add_executable(cfd2 cfd2.cpp)
target_link_libraries(cfd2 PUBLIC ArrayFire::afcuda Forge::forge)