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
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Turing sm_75 default (GTX 1650 Ti / RTX 20xx); override with -DCMAKE_CUDA_ARCHITECTURES=...
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
        find_package(CUDAToolkit REQUIRED)
        list(APPEND calico_SOURCES apriltag_cuda.cu)
        add_definitions(-DCALICO_USE_CUDA)
        set(CALICO_CUDA_LIBS CUDA::cudart)
        # Optional: NVIDIA cuAprilTags if installed on host / in container
        find_path(CUAPRILTAGS_INCLUDE_DIR nvapriltags/nvapriltags.h
            HINTS /usr/local/include /usr/include)
        find_library(CUAPRILTAGS_LIBRARY NAMES nvapriltags
            HINTS /usr/local/lib /usr/lib)
        if(CUAPRILTAGS_INCLUDE_DIR AND CUAPRILTAGS_LIBRARY)
            message(STATUS "Found cuAprilTags: ${CUAPRILTAGS_LIBRARY}")
            add_definitions(-DCALICO_HAVE_CUAPRILTAGS)
            include_directories(${CUAPRILTAGS_INCLUDE_DIR})
            list(APPEND CALICO_CUDA_LIBS ${CUAPRILTAGS_LIBRARY})
        else()
            message(STATUS "cuAprilTags not found (optional); GPU BGR->gray and Ceres CUDA enabled.")
        endif()
    else()
        message(FATAL_ERROR "ENABLE_CUDA=ON but nvcc / CUDA compiler not found.")
    endif()

    # Detect whether Ceres Solver was compiled with CUDA
    set(_ceres_cfg_file "")
    foreach(_ceres_inc IN LISTS Ceres_INCLUDE_DIRS CERES_INCLUDE_DIRS)
        if(EXISTS "${_ceres_inc}/ceres/internal/config.h")
            set(_ceres_cfg_file "${_ceres_inc}/ceres/internal/config.h")
            break()
        endif()
        if(EXISTS "${_ceres_inc}/internal/config.h")
            set(_ceres_cfg_file "${_ceres_inc}/internal/config.h")
            break()
        endif()
    endforeach()
    if(NOT _ceres_cfg_file AND EXISTS "/usr/local/include/ceres/internal/config.h")
        set(_ceres_cfg_file "/usr/local/include/ceres/internal/config.h")
    endif()
    if(_ceres_cfg_file)
        file(READ "${_ceres_cfg_file}" _ceres_cfg)
        # Ceres 2.x: CUDA on => CERES_NO_CUDA is commented, or CERES_USE_CUDA is defined.
        if(_ceres_cfg MATCHES "(^|\\n)[ \\t]*#define[ \\t]+CERES_USE_CUDA")
            add_definitions(-DCERES_USE_CUDA)
            message(STATUS "Ceres was built with CUDA (${_ceres_cfg_file})")
        elseif(NOT _ceres_cfg MATCHES "(^|\\n)[ \\t]*#define[ \\t]+CERES_NO_CUDA")
            if(_ceres_cfg MATCHES "CERES_NO_CUDA")
                add_definitions(-DCERES_USE_CUDA)
                message(STATUS "Ceres was built with CUDA (${_ceres_cfg_file})")
            endif()
        endif()
    endif()
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

################ tests ################################

option(BUILD_TESTING "Build CTest unit tests" ON)
if(BUILD_TESTING)
    enable_testing()
    set(TEST_SOURCES
        tests/test_main.cpp
        tests/test_pattern_params.cpp
        tests/test_math_helpers.cpp
        tests/test_loss_and_checkpoint.cpp
        pattern-parameters.cpp
        apriltag-file.cpp
        cuda-detect.cpp
        helper.cpp
        DirectoryFunctions.cpp
    )
    add_executable(calico_tests ${TEST_SOURCES})
    target_include_directories(calico_tests PRIVATE ${Eigen_INCLUDE_DIRS} ${apriltag_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(calico_tests PRIVATE cxx_std_17)
    target_compile_options(calico_tests PRIVATE ${my_USUAL_FLAGS})
    target_link_libraries(calico_tests ${CERES_LIBRARIES} ${OpenCV_LIBS} ${apriltags_LIBRARIES} ${apriltag_LIBRARIES} gomp Eigen3::Eigen glog gflags)
    add_test(NAME unit_tests COMMAND calico_tests)
    install(TARGETS calico_tests DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
