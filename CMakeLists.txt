cmake_minimum_required(VERSION 3.26)
project(LightDelta VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
set(TORCH_CUDA_ARCH_LIST "9.0")
find_package(Torch REQUIRED)
find_package(CUDAToolkit 12.9 REQUIRED)
add_library(_C SHARED csrc/register.cpp csrc/gdn_decode.cu csrc/state_io.cu)
target_include_directories(_C PRIVATE include)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_features(_C PRIVATE cxx_std_17)
set_target_properties(_C PROPERTIES PREFIX "" CUDA_ARCHITECTURES 90
    CUDA_STANDARD 17 INSTALL_RPATH "$ORIGIN/../torch/lib")
target_compile_options(_C PRIVATE -O3 $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>)
install(TARGETS _C LIBRARY DESTINATION lightdelta)

install(DIRECTORY include/ DESTINATION include)
