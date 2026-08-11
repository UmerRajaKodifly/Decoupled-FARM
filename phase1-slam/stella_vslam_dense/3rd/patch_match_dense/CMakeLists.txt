cmake_minimum_required(VERSION 3.17)
project(PatchMatchDense LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0104 OLD)


# ----- Set build type -----

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


# ----- Set compiler options -----

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s -O3 -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s -O3 -ffast-math")

option(BUILD_WITH_MARCH_NATIVE "Build with march native" ON)
if(BUILD_WITH_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
    message(STATUS "Architecture-aware optimization (-march=native): ENABLED")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
    message(STATUS "Architecture-aware optimization (-march=native): DISABLED")
endif()


# ----- Find dependencies -----

# OpenCV
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)


# ----- Build selection -----

add_library(${PROJECT_NAME} SHARED
            src/patch_match_cuda.cu
            src/depthmap.cpp
            src/patch_match.cpp)


# ----- Compile configuration -----

# OpenMP
option(USE_OPENMP "Use OpenMP" ON)
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENMP)
        message(STATUS "OpenMP: ENABLED")
    else()
        message(STATUS "OpenMP: DISABLED")
    endif()
else()
    message(STATUS "OpenMP: DISABLED")
endif()


# ----- Configure library -----

target_include_directories(${PROJECT_NAME} PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
            $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

target_link_libraries(${PROJECT_NAME} PUBLIC
            CUDA::cudart
            opencv_core
            opencv_imgproc)


# ----- Configure example -----

option(BUILD_EXAMPLES "Build exmaple applications" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(example)
    message(STATUS "Build examples: ENABLED")
else()
    message(STATUS "Build examples: DISABLED")
endif()


# ----- Install configuration -----

include(CMakePackageConfigHelpers)

set(TARGETS_EXPORT_NAME ${PROJECT_NAME}Targets)
set(CONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})

# Generate cmake configuration scripts
write_basic_package_version_file(${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION 1.0
                                 COMPATIBILITY SameMajorVersion)
configure_file(${PROJECT_SOURCE_DIR}/Config.cmake.in ${PROJECT_NAME}Config.cmake @ONLY)

# Install cmake configuration scripts
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        DESTINATION ${CONFIG_INSTALL_DIR})
install(EXPORT ${TARGETS_EXPORT_NAME}
        DESTINATION ${CONFIG_INSTALL_DIR})

# Install library
install(TARGETS ${PROJECT_NAME}
        EXPORT ${TARGETS_EXPORT_NAME})

# Install headers
install(FILES src/depthmap.hpp src/patch_match.hpp
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
