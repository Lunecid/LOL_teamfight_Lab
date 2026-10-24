cmake_minimum_required(VERSION 3.20)
project(cpp_cuda_training_native LANGUAGES CXX)

# Pass -DTORCH_PREFIX_PATH=<site-packages>/torch or set TORCH_PREFIX_PATH in
# the environment. CMAKE_PREFIX_PATH is also honored by find_package(Torch).
if(DEFINED TORCH_PREFIX_PATH)
  list(PREPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")
endif()
if(DEFINED ENV{TORCH_PREFIX_PATH})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{TORCH_PREFIX_PATH}")
endif()

find_package(Torch REQUIRED CONFIG)

add_executable(native native.cpp)
target_compile_features(native PRIVATE cxx_std_17)
target_include_directories(native PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_options(native PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(native PRIVATE ${TORCH_LIBRARIES})

if(MSVC)
  target_compile_options(native PRIVATE /EHsc)
endif()
