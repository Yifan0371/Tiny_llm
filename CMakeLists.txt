
cmake_minimum_required(VERSION 3.16)
project(tiny_llm CXX)

# 使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if (MSVC)
    add_compile_options(/O2 /openmp)
else()
    add_compile_options(-O3 -march=native -fopenmp)
endif()

# 头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件列表
set(SRC
        src/main.cpp
        src/tensor.cpp
        src/linear.cpp
        src/utils.cpp
)

# 生成可执行文件
add_executable(tiny-llm ${SRC})
