cmake_minimum_required(VERSION 3.5)
project(detect_test)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fopenmp")


include_directories(
    #OpenVINO推理引擎的头文件
    /opt/intel/openvino_2021/deployment_tools/inference_engine/include/
    /opt/intel/openvino_2021/deployment_tools/ngraph/include/
)

#查找必要的依赖包
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "/opt/intel/openvino_2021/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
set(ngraph_DIR "/opt/intel/openvino_2021/deployment_tools/ngraph")
# find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "/opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so")
set(ngraph_INCLUDE_DIRS "/opt/intel/openvino_2021/deployment_tools/ngraph/include/")


# 编译detector.so
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/infer/)
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/infer/detector.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
        # PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        # ${ngraph_LIBRARIES}
)


add_executable(detect_test main.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} 
                PRIVATE 
                ${InferenceEngine_LIBRARIES}
                ${OpenCV_LIBS} 
                ${NGRAPH_LIBRARIES}
                detector
)

# fmt
find_package(fmt REQUIRED)
target_link_libraries(${PROJECT_NAME} 
                        PRIVATE 
                        fmt::fmt
                        )
#Camera
include_directories(${PROJECT_SOURCE_DIR}/device/camera)
add_library(MVSDK SHARED IMPORTED)
add_library(Camera devices/camera/mv_video_capture.cpp)
set(MIMDVISION_SDK_LIBRARY /lib/libMVSDK.so)
set_property(TARGET MVSDK PROPERTY IMPORTED_LOCATION ${MIMDVISION_SDK_LIBRARY})
target_link_libraries(${PROJECT_NAME}  
                                PRIVATE 
                                Camera 
                                MVSDK
                                )