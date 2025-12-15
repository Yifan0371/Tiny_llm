cmake_minimum_required(VERSION 3.16)
project(tiny_llm CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
# 编译优化和 OpenMP
if (MSVC)
    add_compile_options(/O2 /openmp)
else()
    add_compile_options(-O3 -march=native -fopenmp)
endif()

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
set(TINY_LLM_SRC
        src/tensor.cpp
        src/linear.cpp
        src/utils.cpp
        src/layernorm.cpp
        src/linear_int8.cpp
        src/attention.cpp
        src/ffn.cpp
        src/transformer_block.cpp
        src/model.cpp
        src/weight_loader.cpp
        src/profiler.cpp
)

# ------- 生成 tiny-llm 库 -------
add_library(tiny-llm STATIC ${TINY_LLM_SRC})
target_link_libraries(tiny-llm PUBLIC OpenMP::OpenMP_CXX)

add_executable(run_model examples/run_model.cpp)
target_link_libraries(run_model tiny-llm)
add_executable(alignment_check examples/alignment_check.cpp)
target_link_libraries(alignment_check tiny-llm)
add_executable(test_quantization examples/test_quantization.cpp)
target_link_libraries(test_quantization tiny-llm)
add_executable(test_kv_cache examples/test_kv_cache.cpp)
target_link_libraries(test_kv_cache tiny-llm)
add_executable(run_benchmark benchmarks/run_benchmark.cpp)
target_link_libraries(run_benchmark tiny-llm)