cmake_minimum_required(VERSION 3.24)
project(CS488 LANGUAGES CUDA)


set(CMAKE_CXX_STANDARD 14)

add_subdirectory(external/glew)
add_subdirectory(external/glfw)

file(GLOB CUDA_FILES src/*.cu)

add_executable(CS488
        ${CUDA_FILES} ${SRC_FILES})
target_include_directories(CS488 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include)
target_include_directories(CS488 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/glew/include)
set_property(TARGET CS488 PROPERTY CUDA_ARCHITECTURES native)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(CS488 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(CS488 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(CS488 glfw libglew_static)