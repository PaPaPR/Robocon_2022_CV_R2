cmake_minimum_required(VERSION 3.0.0)

project(Robocon_2022_CV_R2)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# add_definitions(-DRELEASE)

add_executable(Robocon_2022_CV_R2 main.cpp)
include_directories(${PROJECT_SOURCE_DIR})

#Realsense
find_package(realsense2 REQUIRED)
target_link_libraries(Robocon_2022_CV_R2 ${realsense2_LIBRARY})

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Robocon_2022_CV_R2 ${OpenCV_LIBS})

#fmt
find_package(fmt REQUIRED)
target_link_libraries(Robocon_2022_CV_R2 fmt::fmt)

#Camera
# add_library(MVSDK SHARED IMPORTED)
# add_library(Camera devices/camera/mv_video_capture.cpp)
# set(MIMDVISION_SDK_LIBRARY /lib/libMVSDK.so)
# set_property(TARGET MVSDK PROPERTY IMPORTED_LOCATION ${MIMDVISION_SDK_LIBRARY})
# target_link_libraries(Robocon_2022_CV_R2 Camera MVSDK)

# new serial
set(serial_SRCS
    devices/serial/src/serial.cc
    devices/serial/src/impl/unix.cc
    devices/serial/src/impl/list_ports/list_ports_linux.cc
    devices/serial/serial/serial.h
    devices/serial/serial/v8stdint.h
)
include_directories(devices/serial)
add_library(serial ${serial_SRCS})
target_link_libraries(Robocon_2022_CV_R2 serial)

# evdev
target_link_libraries(Robocon_2022_CV_R2 evdev)

# OpenVino
include_directories(
    ~/intel/openvino_2021/deployment_tools/inference_engine/include/
    ~/intel/openvino_2021/deployment_tools/ngraph/include/
)
set(InferenceEngine_DIR "~/intel/openvino_2021/deployment_tools/inference_engine/share")
find_package(InferenceEngine)

add_library (OpenVINO_Detector OpenVINO/detector.cpp)
target_link_libraries(Robocon_2022_CV_R2 OpenVINO_Detector ${InferenceEngine_LIBRARIES})

#OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CONFIG_FILE_PATH ${PROJECT_SOURCE_DIR}/configs)
set(SOURCE_PATH ${PROJECT_SOURCE_DIR})
target_compile_definitions(Robocon_2022_CV_R2 PRIVATE "CONFIG_FILE_PATH=\"${CONFIG_FILE_PATH}\"")
target_compile_definitions(Robocon_2022_CV_R2 PRIVATE "SOURCE_PATH=\"${PROJECT_SOURCE_DIR}\"")


#Image Record
set(RECORD_IMG 1)
if(${RECORD_IMG})
set(PROJECTNAME_TEST IMGE_RECORD_RCR2)
add_executable(${PROJECTNAME_TEST} 
                ${PROJECT_SOURCE_DIR}/ImageAct/videoRecord.cpp
                )       

target_link_libraries(${PROJECTNAME_TEST}
                PRIVATE
                ${OpenCV_LIBS}
                ${realsense2_LIBRARY}
                fmt::fmt
                )
endif()