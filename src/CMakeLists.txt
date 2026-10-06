cmake_minimum_required(VERSION 3.16)

project(calico-dec2023 LANGUAGES CXX)

include(CMakePrintHelpers)

option(ENABLE_CUDA "Enable CUDA (Ceres + GPU preprocess / AprilTag)" OFF)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Ceres REQUIRED)

find_package(apriltags REQUIRED)

find_package(PkgConfig)
pkg_search_module(apriltag REQUIRED apriltag)

# Prefer OpenCV 5 (objdetect hosts ArUco). Fall back to 4.7+ then 4.3.
# OpenCV 5 split calib3d into calib + geometry; component name calib3d may be absent.
set(_calico_opencv_found FALSE)
find_package(OpenCV 5 QUIET COMPONENTS core imgproc imgcodecs objdetect calib3d)
if(OpenCV_FOUND)
    set(_calico_opencv_found TRUE)
endif()
if(NOT _calico_opencv_found)
    find_package(OpenCV 5 QUIET COMPONENTS core imgproc imgcodecs objdetect calib geometry)
    if(OpenCV_FOUND)
        set(_calico_opencv_found TRUE)
    endif()
endif()
if(_calico_opencv_found)
    message(STATUS "Using OpenCV ${OpenCV_VERSION}")
endif()
if(NOT _calico_opencv_found)
    find_package(OpenCV 4.7 QUIET COMPONENTS core imgproc imgcodecs aruco calib3d objdetect)
    if(OpenCV_FOUND)
        set(_calico_opencv_found TRUE)
        message(STATUS "Using OpenCV ${OpenCV_VERSION}")
    endif()
endif()
if(NOT _calico_opencv_found)
    find_package(OpenCV 4.3 REQUIRED COMPONENTS core imgproc imgcodecs aruco calib3d)
    message(STATUS "Using OpenCV ${OpenCV_VERSION} (legacy ArUco API)")
endif()

link_directories(${apriltag_LIBDIR})

set(my_USUAL_FLAGS  -Wextra -Wall -g3 -O3 -fopenmp CACHE STRING "; non-profiling compile flags")
set(my_USUAL_LINKER_FLAGS CACHE STRING "; non-profiling linker flags")

set(CALICO_EXE calico-dec2023)

set(COMPERROR_EXE compute-dec2023)

set(calico_SOURCES
calico.cpp
calico.hpp
Includes.hpp
opencv-compat.hpp
DirectoryFunctions.cpp
DirectoryFunctions.hpp
camera-calibration.cpp
camera-calibration.hpp
pattern-parameters.cpp
pattern-parameters.hpp
patterns.cpp
patterns.hpp
local-charuco.hpp
local-charuco.cpp
multicamera.cpp
multicamera.hpp
camera-visualization.cpp
camera-visualization.hpp
solving-structure.cpp
solving-structure.hpp
helper.cpp
helper.hpp
helper-cali.cpp
helper-cali.hpp
apriltag-file.cpp
cuda-detect.cpp
cuda-detect.hpp
)

set(compute_error_SOURCES
compute-error.cpp
helper.cpp
helper.hpp
camera-visualization.cpp
camera-visualization.hpp
)

set(CALICO_CUDA_LIBS "")
if(ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DWITH_CUDA)
    list(APPEND calico_SOURCES apriltag_cuda.cu)
    list(APPEND CALICO_CUDA_LIBS CUDA::cudart)
    if(TARGET opencv_cudaimgproc)
        add_definitions(-DHAVE_OPENCV_CUDAIMGPROC)
        list(APPEND CALICO_CUDA_LIBS opencv_cudaimgproc)
        message(STATUS "OpenCV CUDA imgproc enabled")
    endif()
    if(DEFINED CUAPRILTAGS_INCLUDE_DIR AND EXISTS "${CUAPRILTAGS_INCLUDE_DIR}/cuAprilTags.h")
        add_definitions(-DHAVE_CUAPRILTAGS)
        include_directories(${CUAPRILTAGS_INCLUDE_DIR})
        if(DEFINED CUAPRILTAGS_LIBRARY)
            list(APPEND CALICO_CUDA_LIBS ${CUAPRILTAGS_LIBRARY})
        endif()
        message(STATUS "NVIDIA cuAprilTags enabled")
    endif()
    foreach(_ceres_inc ${CERES_INCLUDE_DIRS} ${Ceres_INCLUDE_DIRS})
        if(EXISTS "${_ceres_inc}/ceres/internal/config.h")
            file(READ "${_ceres_inc}/ceres/internal/config.h" _ceres_cfg)
            if(_ceres_cfg MATCHES "define CERES_USE_CUDA")
                add_definitions(-DCERES_USE_CUDA)
                message(STATUS "Ceres was built with CUDA")
            endif()
            break()
        endif()
    endforeach()
    message(STATUS "CALICO CUDA support enabled")
endif()

###################### compute error

set(OBJECT_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/${COMPERROR_EXE}.dir)
message("-- Object files will be output to: ${OBJECT_DIR}")

add_executable(${COMPERROR_EXE} ${compute_error_SOURCES})

target_include_directories(${COMPERROR_EXE} PRIVATE ${Eigen_INCLUDE_DIRS}  ${apriltag_INCLUDE_DIRS})

target_compile_features(${COMPERROR_EXE} PRIVATE cxx_std_17)

target_compile_options(${COMPERROR_EXE} PRIVATE ${my_USUAL_FLAGS})

target_link_libraries(${COMPERROR_EXE} ${CERES_LIBRARIES} Eigen3::Eigen gomp ${OpenCV_LIBS} ${apriltags_LIBRARIES} ${apriltag_LIBRARIES})

################# calico #############################

set(OBJECT_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/${CALICO_EXE}.dir)
message("-- Object files will be output to: ${OBJECT_DIR}")

add_executable(${CALICO_EXE} ${calico_SOURCES})

target_include_directories(${CALICO_EXE} PRIVATE ${Eigen_INCLUDE_DIRS}  ${apriltag_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(${CALICO_EXE} PRIVATE cxx_std_17)

target_compile_options(${CALICO_EXE} PRIVATE ${my_USUAL_FLAGS})

target_link_libraries(${CALICO_EXE} ${CERES_LIBRARIES} Eigen3::Eigen gomp ${OpenCV_LIBS} ${apriltags_LIBRARIES} ${apriltag_LIBRARIES} glog gflags ${CALICO_CUDA_LIBS})

################ installs #############################

install(TARGETS ${CALICO_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

install(TARGETS ${COMPERROR_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
