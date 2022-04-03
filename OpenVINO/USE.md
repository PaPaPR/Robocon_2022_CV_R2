## 模型更新下载
- model 点击[链接](https://github.com/gcusms/Line_new/archive/refs/tags/ModelV1.0.0.tar.gz)
## cmake 加载 Opencv 环境
- CMakeLists.txt add
```c++
include_directories(
    #OpenVINO推理引擎的头文件
    /opt/intel/openvino_2021/deployment_tools/inference_engine/include/
    /opt/intel/openvino_2021/deployment_tools/ngraph/include/
)
#查找必要的依赖包，将opencv和 library(this Target) 链接
# ngraph 无需开启
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
add_executable(${PROJECT_NAME} main.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} 
                PRIVATE 
                ${InferenceEngine_LIBRARIES}
                ${OpenCV_LIBS} 
                ${NGRAPH_LIBRARIES}
                detector
)
```

- 对象声明初始化 
```c++
string xml_path = "/home/wolf/Code_Pack/R2_SMS/model/best.xml";
Detector *detector = new Detector(xml_path);
detector->init( 0.6, 0.3);
```
- 模型推理加载
```c++
detector->process_frame(src_img, detected_objects)
```
- 输出模型状态
```c++
detected_objects[i].status
```
> 0-blue_yellow<br> 
1-blue_white<br
2-blue_blue<br>
3-red_yellow<br>
4-red_white<br>
5-red_red<br>

## 参考
reference.ccp
***
环境编辑添加
> source /opt/intel/openvino_2021/bin/setupvars.sh

