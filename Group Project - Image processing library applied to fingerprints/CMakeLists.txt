cmake_minimum_required(VERSION 2.8.11)
project(FingerPrint)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(SRCS
    src/utils.cpp
    src/Image.cpp
    #src/main_3_simulation.cpp
    src/Dictionary.cpp
    src/Mask.cpp
    src/PixelCoordinates.cpp
    src/Pixel.cpp
    src/Transformation.cpp
    src/Translation.cpp 
    src/Rotation.cpp 
    src/Scaling.cpp 
    src/Shear.cpp
    src/inpainting.cpp
    src/transform.cpp
    #src/main_1_restauration.cpp
    src/main_1_simulation.cpp
    #src/starter2.cpp
    #src/main3.cpp
    src/general_main.cpp
    src/local_warp.cpp
    src/Histogram.cpp
    )
    
include_directories(include)

set(HEADERS
    include/Matrix.hpp
    include/Image.h
    include/Dictionary.h 
    include/Mask.h
    include/Histogram.h
    include/PixelCoordinates.h
    include/Pixel.h
    include/Transformation.h
    include/Translation.h 
    include/Rotation.h 
    include/Scaling.h
    include/main_1_simulation.h
    include/Shear.h
    include/inpainting.h
    include/transform.h
    include/local_warp.h
    )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})
link_directories(${Eigen3_LIBRARY_DIRS})

add_executable (test ${SRCS} ${HEADERS})
target_link_libraries(test ${OpenCV_LIBRARIES})
target_link_libraries(test Eigen3::Eigen)
