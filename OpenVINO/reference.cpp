#include <iostream>
#include <fmt/core.h>
#include "devices/serial/serial.hpp"
#include "thread"
#include "solvePnP/solvePnP.hpp"
#include "infer/detector.h"
#include <librealsense2/rs.hpp>
#include "utils.hpp"
#include <unistd.h>
#include <memory>
#include <chrono>
#include "cv-helpers.hpp"
#include "utils.hpp"
#include <typeinfo>


#define WINDOW_SIZE_WIDTH 400
#define WINDOW_SIZE_HEIGHT 800
#define WINDOW_NAME "WOLF"

using namespace rs2;


void topCameraThread(RoboInf &robo_inf,
                     const std::shared_ptr<RoboSerial> &serial)
{
  // rs2::pipeline pipe;
  // rs2::config cfg;
  // cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30.f);
  // cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30.f);
  // pipe.start(cfg);
  // rs2::align align_to(RS2_STREAM_COLOR);
  fmt::print("Robot_R2_CameraTest\n");
  // the part of the inferences
  Detector *detector = new Detector;
  string xml_path = "../model/r2_rebuild.xml";
  detector->init(xml_path, 0.7, 0.45);
  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/d435i.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  cv::Mat src_img;
  cv::Rect object_3d_rect(0, 0, 140, 140);
  cv::Point2f pnp_angle;
  cv::Point3f pnp_coordinate_mm;
  float pnp_depth;
  int yolo_res_selected_id;

  constexpr float cube_target_yaw_angle_offset = -2.f;
  constexpr float cube_target_distance_offset = 13.5f;
  constexpr float cube_target_yaw_angle_errors_range = 2.0f;
  constexpr float cube_target_distance_errors_range = 2.5f;
  constexpr int cube_targeted_detect_flag_times = 5;
  constexpr int cube_target_echo_uart_cmd_sleep_time = 20000;

  constexpr int cube_1_min_area = 15000;
  constexpr int cube_1_max_area = 20000;

  constexpr int cube_2_min_area = 30000;
  constexpr int cube_2_max_area = 39000;

  constexpr int cube_3_min_area = 43800;
  constexpr int cube_3_max_area = 51000;
  
  constexpr int cube_4_min_area = 68000;
  constexpr int cube_4_max_area = 75500;

  constexpr int cube_5_min_area = 79000;
  constexpr int cube_5_max_area = 90000;

  std::vector<vector<int>> areaJudge;
  
  cv::namedWindow("interface");
  cv::moveWindow("interface", 75, 30);

  pipeline pipe; // 创建数据管道
  // start() 函数返回数据管道的 profile
  pipeline_profile profile = pipe.start();
  while (true)
  {
    usleep(1);
    static int cube_middle_detect_times{0};
    static int cude_front_detect_times{0};

    // 堵塞程序直到新的一帧捕获
    frameset frameset = pipe.wait_for_frames();

    // 获取颜色图
    rs2::video_frame video_src = frameset.get_color_frame();

    // 获取深度图
    rs2::depth_frame depth_src = frameset.get_depth_frame();

    // 获取深度图的尺寸，用于确定测距中心点
    float width_ = depth_src.get_width();
    float height_ = depth_src.get_height();

    // 获取颜色图的尺寸，用于转成 Mat 格式并显示
    const int color_width = video_src.as<video_frame>().get_width();
    const int color_height = video_src.as<video_frame>().get_height();

    // 转成 Mat 类型
    Mat src_img(Size(color_width, color_height), CV_8UC3,
                (void *)video_src.get_data(), Mat::AUTO_STEP);
    // auto res = detect_cube_top->Detect(src_img);

    // for (long unsigned int i = 0; i < res.size(); i++)
    //   cv::rectangle(src_img, get_rect(src_img, res[i].bbox),
    //                 cv::Scalar(0, 255, 0), 2);
    // cv::line(src_img, cv::Point(src_img.cols / 3, 0),
    //           cv::Point(src_img.cols / 3, src_img.rows),
    //           cv::Scalar(0, 150, 255), 2);
    // cv::line(src_img, cv::Point(src_img.cols / 3 * 2, 0),
    //           cv::Point(src_img.cols / 3 * 2, src_img.rows),
    //           cv::Scalar(0, 150, 255), 2);
    vector<Detector::Object> detected_objects;

    // auto start = chrono::high_resolution_clock::now();
    if(!detector->process_frame(src_img, detected_objects))
    {
      continue;
    }
    cvtColor(src_img,src_img,COLOR_BGR2RGB);

    std::sort(detected_objects.begin(), detected_objects.end());
    cv::Rect temp_rect_ = detected_objects.at(0).rect;
    // cout << "area = "  << temp_rect_.area()<< endl;
    // resetRect(temp_rect_);
cv::rectangle(src_img,
            temp_rect_,
            cv::Scalar(0, 0, 0),
            2,
            LINE_8,
          0); 
cv::putText(src_img,detected_objects.at(0).name,
Point(temp_rect_.x + 20,temp_rect_.y - 10),cv::FONT_HERSHEY_COMPLEX,
    0.7,
    cv::Scalar(100,100,255),
    0.5,
    cv::LINE_4);

    switch (robo_inf.catch_cube_mode_status.load())
    {
    // change the direction of the robot
    case CatchMode::spin:
    {
      cv::putText(src_img,
                  "spin mode",
                  cv::Point(50, 50),
                  cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
      pnp->solvePnP(object_3d_rect, temp_rect_, pnp_angle,
                    pnp_coordinate_mm, pnp_depth);
      pnp_angle.y = src_img.cols * 0.57 - (temp_rect_ .x + temp_rect_.width * 0.5);

      if (cube_middle_detect_times < cube_targeted_detect_flag_times)
      {
        if (pnp_angle.y < cube_target_yaw_angle_errors_range &&
            pnp_angle.y > -cube_target_yaw_angle_errors_range)
        {
          cube_middle_detect_times++;
        }
        else
        {
          RoboSpinCmdUartBuff uart_temp_spin_cmd;
          uart_temp_spin_cmd.yaw_angle = pnp_angle.y;
          std::cout << "uart_temp_spin_cmd.yaw_angle" << uart_temp_spin_cmd.yaw_angle << "\n";
          serial->write((uint8_t *)&uart_temp_spin_cmd, sizeof(uart_temp_spin_cmd));
        }
      }
      else
      {
        RoboSpinCmdUartBuff uart_temp_spin_cmd;
        uart_temp_spin_cmd.yaw_angle = 0.f;
        for (int i = 0; i < 3; i++)
        {
          serial->write((uint8_t *)&uart_temp_spin_cmd, sizeof(uart_temp_spin_cmd));
          std::cout << "send yaw 0\n";
          usleep(cube_target_echo_uart_cmd_sleep_time);
        }
        cube_middle_detect_times = 0;
        robo_inf.catch_cube_mode_status.store(CatchMode::go);
      }
      break;
    }

    case CatchMode::go:
    {
      cv::putText(src_img,
                  "go straight mode",
                  cv::Point(50, 50),
                  cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
      // judge the cube if in the center of frame
      // sand the message in ten times
      if (cude_front_detect_times < cube_targeted_detect_flag_times)
      {
        cv::Rect object_rect(temp_rect_.x + temp_rect_.width *0.5 - 50,
                             temp_rect_.y + temp_rect_.height - 100,
                             100, 100); // 底部与目标重叠且居中，固定大小的 temp_rect_ 用以 pnp
        pnp->solvePnP(object_3d_rect, object_rect, pnp_angle,
                      pnp_coordinate_mm, pnp_depth);
        
        float select_cube_dis = cube_target_distance_offset - pnp_angle.x;
        // if the angles in around the range
        if (select_cube_dis < cube_target_distance_errors_range &&
            select_cube_dis > -cube_target_distance_errors_range)
        {
          cude_front_detect_times++;
        }
        else
        {
          RoboGoCmdUartBuff uart_temp_go_cmd;
          uart_temp_go_cmd.distance = select_cube_dis;
          std::cout << "uart_temp_go_cmd.distance:" << uart_temp_go_cmd.distance << "\n";
          serial->write((uint8_t *)&uart_temp_go_cmd, sizeof(uart_temp_go_cmd));
        }
      }
      else
      {
        // send  0 meters message for the Electronically controlled then the robo ready to  catch the cube for three time
        RoboGoCmdUartBuff uart_temp_go_cmd;
        uart_temp_go_cmd.distance = 0.f;
        std::cout << "cathch_mode" << std::endl;
        for (int i = 0; i < 3; i++)
        {
          serial->write((uint8_t *)&uart_temp_go_cmd, sizeof(uart_temp_go_cmd));
          std::cout << "send stop \n";
          usleep(cube_target_echo_uart_cmd_sleep_time);
        }
        robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);
        cude_front_detect_times = 0;
      }
      break;
    }

    case CatchMode::catch_cube:
    {
      cv::putText(src_img,
                  "catch cube mode",
                  cv::Point(50, 50),
                  cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
      RoboCatchCmdUartBuff uart_temp_catch_cmd;
      // make the state of the cube 
      if (detected_objects.at(0).id == 0 ||
          detected_objects.at(0).id == 3)
      {
        uart_temp_catch_cmd.cube_state = CUBE_UP;
      }
      else if (detected_objects.at(0).id == 1 ||
               detected_objects.at(0).id == 4)
      {
        uart_temp_catch_cmd.cube_state = CUBE_DOWN;
      }
      else if (detected_objects.at(0).id == 2 ||
               detected_objects.at(0).id == 5)
      {
        uart_temp_catch_cmd.cube_state = CUBE_STAND;
      }
      
      // get the type of the cube by judging that area
      if (temp_rect_.area() > cube_1_min_area &&
          temp_rect_.area() < cube_1_max_area)
      {
        uart_temp_catch_cmd.cube_type = CUBE_1;
      }

      if (temp_rect_.area() > cube_2_min_area &&
          temp_rect_.area() < cube_2_max_area)
      {
        uart_temp_catch_cmd.cube_type = CUBE_2;
      }
      else if (temp_rect_.area() > cube_3_min_area &&
               temp_rect_.area() < cube_3_max_area)
      {
        uart_temp_catch_cmd.cube_type = CUBE_3;
      }
      else if (temp_rect_.area() > cube_4_min_area &&
               temp_rect_.area() < cube_4_max_area)
      {
        uart_temp_catch_cmd.cube_type = CUBE_4;
      }
      if (uart_temp_catch_cmd.cube_state == CUBE_STAND)
      {
        uart_temp_catch_cmd.cube_type = CUBE_UNCERTAIN;
      }
      // send the true message to the serial for three times
      if (uart_temp_catch_cmd.cube_type != 0x00)
      {
        for (int i = 0; i < 3; i++)
        {
          serial->write((uint8_t *)&uart_temp_catch_cmd, sizeof(uart_temp_catch_cmd));
          std::cout << "catch, temp_rect_ size:" << temp_rect_.area() << "  temp_rect_ type:"
                    << (int)uart_temp_catch_cmd.cube_type << "  temp_rect_ state:"
                    << (int)uart_temp_catch_cmd.cube_state << "\n";
          usleep(cube_target_echo_uart_cmd_sleep_time);
        }
      }
      robo_inf.catch_cube_mode_status.store(CatchMode::detect_mode);
      break;
    }
    case CatchMode::detect_mode:
    {
    cv::putText(src_img,
                "detect cube mode",
                cv::Point(50, 50),
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
      RoboCubeStateUartBuff uart_temp_state_cmd;
      if (detected_objects.at(0).id == 0 ||
          detected_objects.at(0).id == 3)
      {
        uart_temp_state_cmd.cube_status = CUBE_UP;
      }
      else if (detected_objects.at(0).id == 1 ||
               detected_objects.at(0).id == 4)
      {
        uart_temp_state_cmd.cube_status = CUBE_DOWN;
      }
      else if (detected_objects.at(0).id == 2 ||
               detected_objects.at(0).id == 5)
      {
        uart_temp_state_cmd.cube_status = CUBE_STAND;
      }
      uart_temp_state_cmd.cube_status = CUBE_DOWN;
      uart_temp_state_cmd.cube_type = 0x03;
      if (uart_temp_state_cmd.cube_status != 0x00)
      {
      for (int i = 0; i < 3; i++)
        {
          serial->write((uint8_t *)&uart_temp_state_cmd, sizeof(uart_temp_state_cmd));
          std::cout << (int)uart_temp_state_cmd.cube_type << "  temp_rect_ state:"
                    << (int)uart_temp_state_cmd.cube_status << "\n";
          usleep(cube_target_echo_uart_cmd_sleep_time);
        }
      }
      robo_inf.catch_cube_mode_status.store(CatchMode::off);

      break;
    }
    case CatchMode::off:
      cube_middle_detect_times = 0;
      cude_front_detect_times = 0;
      break;

    default:
      break;
    }

    cv::rectangle(src_img, temp_rect_, cv::Scalar(0, 150, 255), 2);
    cv::putText(src_img, 
                std::to_string(detected_objects.at(0).id),
                cv::Point(temp_rect_.x, temp_rect_.y - 1),
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
    // }
#ifndef RELEASE
    if (!src_img.empty())
    {
      std::vector<uchar> buff_bgr;
      cv::imencode(".jpg", src_img, buff_bgr);
    }
#endif
    if (!src_img.empty())
      cv::imshow("interface", src_img);
      cv::waitKey(1);
  }
}

// void uartReadThread(const std::shared_ptr<RoboSerial> &serial,
//                     RoboInf &robo_inf)
// {
//   cout << typeid( std::this_thread::get_id() ).name() << endl;  //  打印: int 
//   while (true) try {
//       if(serial->isOpen()) {
//         serial->ReceiveInfo(robo_inf);
//       } else {
//         serial->open();
//       }
//       std::this_thread::sleep_for(std::chrono::seconds(1));
//     } catch (const std::exception &e) {
//       serial->close();
//       static int serial_read_excepted_times{0};
//       if (serial_read_excepted_times++ > 3) {
//       std::this_thread::sleep_for(std::chrono::seconds(1));
//         fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
//                    idntifier_red);
//         serial_read_excepted_times = 0;
//       }
//       fmt::print("[{}] serial exception: {}\n",
//                  idntifier_red, e.what());
//       std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
// }

void uartThread_(RoboInf &robo_inf, const std::shared_ptr<RoboSerial> &serial)
{
  while (true) try {
      if(serial->isOpen()) {
        serial->ReceiveInfo(robo_inf);
      } else {
        serial->open();
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    } catch (const std::exception &e) {
      serial->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
      std::this_thread::sleep_for(std::chrono::microseconds(10000));
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
                   idntifier_red);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] serial exception: {}\n",
                 idntifier_red, e.what());
      std::this_thread::sleep_for(std::chrono::microseconds(10000));
    }
}
int main()
{

  auto serial = std::make_shared<RoboSerial>("/dev/ros_tty", 115200);
  RoboInf robo_inf;
  std::thread uart_thread(uartThread_, std::ref(robo_inf), std::ref(serial));
  uart_thread.detach();

  std::thread imageThread(topCameraThread, std::ref(robo_inf),
                          std::ref(serial));
  imageThread.detach();


  cv::Mat src_f;


  if (getchar())
  {
    uart_thread.~thread();
    // testThread.~thread();
    imageThread.~thread();
  }
  return 0;
}
