#include <fmt/core.h>

#include <chrono>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <memory>
#include <opencv4/opencv2/highgui/highgui.hpp>

#include "devices/serial/serial.hpp"
#include "infer/detector.h"
#include "pthread.h"
#include "thread"

#define WINDOW_SIZE_WIDTH 400
#define WINDOW_SIZE_HEIGHT 800
#define WINDOW_NAME "WOLF"

using namespace rs2;

int main() {
  while (true) {
    // 堵塞程序直到新的一帧捕获
    frameset frameset = pipe.wait_for_frames();

    // 获取颜色图
    rs2::video_frame video_src = frameset.get_color_frame();

    // 获取深度图
    rs2::depth_frame depth_src = frameset.get_depth_frame();

    // 获取深度图的尺寸，用于确定测距中心点
    float width = depth_src.get_width();
    float height = depth_src.get_height();

    // 获取颜色图的尺寸，用于转成 Mat 格式并显示
    const int color_width = video_src.as<video_frame>().get_width();
    const int color_height = video_src.as<video_frame>().get_height();

    // 转成 Mat 类型
    Mat frame(Size(color_width, color_height), CV_8UC3,
              (void*)video_src.get_data(), Mat::AUTO_STEP);

    // 定义并获取距离数据（单位： m ）
    float distance = depth_src.get_distance(width / 2, height / 2);

    // 输出距离数据
    cout << "distance:" << distance << endl;
    if (!frame.empty()) {
      Mat osrc = frame.clone();
      resize(osrc, osrc, Size(640, 640));
      vector<Detector::Object> detected_objects;

      // 推理部分
      auto start = chrono::high_resolution_clock::now();
      detector->process_frame(osrc, detected_objects);
      auto end = chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      cout << "use " << diff.count() << " s" << endl;
      for (int i = 0; i < detected_objects.size(); ++i) {
        int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width,
                  height);  //左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(osrc, rect, cv::Scalar(255, 0, 255), 2, LINE_8, 0);
        putText(osrc, detected_objects[i].name, Point(rect.x, rect.y - 10),
                cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(255, 0, 255), 0.5,
                cv::LINE_4);
      }
      imshow("result", osrc);
    }
    if (static_cast<char>(cv::waitKey(1)) == 'q') {
      uart_thread.~thread();
      // testThread.~thread();
      imageThread.~thread();
    }

    return 0;
  }

  // void uiShow()
  // {
  //     MES m;
  //     // serial
  //     // Show everything on the screen
  //     cv::Mat frame = cv::Mat(WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT,
  //     CV_8UC3); double value = 5.0; cvui::init(WINDOW_NAME); while (true)
  //     {
  //       frame = cv::Scalar(49, 52, 49);
  //       cvui::printf(frame, 10, 300, 0.8, 0x00ffff, "k click count: %d",
  //       value);
  //       // robo_inf.value_d.store((rand() + .0));
  //       m.points.push_back(rand() + .0);

  //       fmt::print("[{}]:{}🎄\n",idntifier_red,.0);
  //       if(m.points.size() >= 30) {
  //           m.points.erase(m.points.begin()); //删除第一个元素
  //       }
  //       cvui::sparkline(frame,m.points,0,10,800,200,0xff00ff);
  //       cvui::update();
  //       cv::imshow(WINDOW_NAME, frame);
  //     }
  // }