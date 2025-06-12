cmake_minimum_required(VERSION 3.14)
project(pysparseir VERSION 0.2.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add libsparseir subdirectory
add_subdirectory(libsparseir)

# Python extension module
add_library(_capi SHARED
    libsparseir/src/cinterface.cpp
    libsparseir/src/kernel.cpp
    libsparseir/src/linalg.cpp
    libsparseir/src/poly.cpp
    libsparseir/src/root.cpp
    libsparseir/src/specfuncs.cpp
    libsparseir/src/svd.cpp
    libsparseir/src/sve.cpp
    libsparseir/src/utils.cpp
)

target_include_directories(_capi PRIVATE
    libsparseir/include
    libsparseir/deps/eigen3
    libsparseir/deps/xprec/include
)

target_link_libraries(_capi PRIVATE sparseir)

# Set output name to match Python's expected name
set_target_properties(_capi PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_capi"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sparseir"
) 