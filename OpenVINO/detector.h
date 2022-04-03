#pragma once

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <inference_engine.hpp>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Detector
{
public:
  typedef struct {
    float prob;
    std::string name;
    cv::Rect rect;
    int status;
  } Object;
  Detector(std::string &xml_path);
  ~Detector();
  bool init(double cof_threshold,double nms_area_threshold);
  bool uninit();
  bool process_frame(Mat& inframe,vector<Object> &detected_objects);
private:
  double sigmoid(double x);
  vector<int> get_anchors(int net_grid);
  bool parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
  vector<Rect>& o_rect,vector<float>& o_rect_cof, vector<int>& lable_input);
  Rect detet2origin(const Rect& dete_rect,float rate_to,int top,int left);
  ExecutableNetwork _network;
  OutputsDataMap _outputinfo;
  string _input_name;
  string _xml_path;            //OpenVINO模型xml文件路径
  double _cof_threshold;       //置信度阈值,计算方法是框置信度乘以物品种类置信度
  double _nms_area_threshold;  //nms最小重叠面积阈值
};