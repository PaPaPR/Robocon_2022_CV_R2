#include <fmt/color.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "TensorRTx/yolov5.hpp"
#include "cv-helpers.hpp"
#include "devices/serial/serial.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/catch_keyboard.hpp"
#include "solvePnP/solvePnP.hpp"
#include "utils.hpp"
#include "utils/mjpeg_streamer.hpp"

using namespace std::chrono_literals;

void topCameraThread(RoboInf &robo_inf,
    const std::shared_ptr<RoboSerial> &serial,
    const std::shared_ptr<nadjieb::MJPEGStreamer> &streamer_ptr) {
  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30.f);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30.f);
  pipe.start(cfg);
  rs2::align align_to(RS2_STREAM_COLOR);

  auto detect_cube_top = std::make_shared<YOLOv5TRT>(
      fmt::format("{}{}", SOURCE_PATH, "/models/cube_top.engine"));

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/d435i.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  cv::Mat src_img;
  cv::Rect object_2d_rect;
  cv::Rect object_3d_rect(0, 0, 140, 140);
  cv::Point2f pnp_angle;
  cv::Point3f pnp_coordinate_mm;
  float pnp_depth;
  int yolo_res_selected_id;

  constexpr float cube_target_yaw_angle_offset = -2.f;
  constexpr float cube_target_distance_offset = 13.5f;
  constexpr float cube_target_yaw_angle_errors_range = 0.5f;
  constexpr float cube_target_distance_errors_range = 0.5f;
  constexpr int cube_targeted_detect_flag_times = 10;
  constexpr int cube_target_echo_uart_cmd_sleep_time = 20000;
  constexpr int cube_2_min_area = 45000;
  constexpr int cube_2_max_area = 55000;
  constexpr int cube_3_min_area = 75000;
  constexpr int cube_3_max_area = 90000;
  constexpr int cube_4_min_area = 95000;
  constexpr int cube_4_max_area = 115000;

  while (true) try {
    usleep(1);
    static int cube_middle_detect_times{0};
    static int cude_front_detect_times{0};

    auto frames = pipe.wait_for_frames();
    auto depth_frame = frames.get_depth_frame();
    auto aligned_set = align_to.process(frames);
    auto color_frame = aligned_set.get_color_frame();
    src_img = frame_to_mat(color_frame);
    auto res = detect_cube_top->Detect(src_img);

    for (long unsigned int i = 0; i < res.size(); i++)
      cv::rectangle(src_img, get_rect(src_img, res[i].bbox),
                    cv::Scalar(0, 255, 0), 2);
    cv::line(src_img, cv::Point(src_img.cols / 3, 0),
              cv::Point(src_img.cols / 3, src_img.rows),
              cv::Scalar(0, 150, 255), 2);
    cv::line(src_img, cv::Point(src_img.cols / 3 * 2, 0),
              cv::Point(src_img.cols / 3 * 2, src_img.rows),
              cv::Scalar(0, 150, 255), 2);

    //选择检测方框在视野中最下位置的块
    if (rectFilter(res, src_img, object_2d_rect, yolo_res_selected_id)) {
      switch (robo_inf.catch_cube_mode_status.load()) {
        case CatchMode::spin: {
          cv::putText(src_img,
                      "spin mode",
                      cv::Point(50, 50),
                      cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
          pnp->solvePnP(object_3d_rect, object_2d_rect, pnp_angle,
                        pnp_coordinate_mm, pnp_depth);
          pnp_angle.y += cube_target_yaw_angle_offset;

          if (cube_middle_detect_times < cube_targeted_detect_flag_times) {
            if (pnp_angle.y < cube_target_yaw_angle_errors_range &&
                pnp_angle.y > -cube_target_yaw_angle_errors_range) {
              cube_middle_detect_times++;
            } else {
              RoboSpinCmdUartBuff uart_temp_spin_cmd;
              uart_temp_spin_cmd.yaw_angle = pnp_angle.y;
              std::cout << "uart_temp_spin_cmd.yaw_angle" << uart_temp_spin_cmd.yaw_angle << "\n";
              serial->write((uint8_t *)&uart_temp_spin_cmd, sizeof(uart_temp_spin_cmd));
            }
          } else {
            RoboSpinCmdUartBuff uart_temp_spin_cmd;
            uart_temp_spin_cmd.yaw_angle = 0.f;
            for (int i = 0; i < 3; i++) {
              serial->write((uint8_t *)&uart_temp_spin_cmd, sizeof(uart_temp_spin_cmd));
              std::cout << "send yaw 0\n";
              usleep(cube_target_echo_uart_cmd_sleep_time);
            }

            cube_middle_detect_times = 0;
            robo_inf.catch_cube_mode_status.store(CatchMode::go);
          }
          break;
        }

        case CatchMode::go: {
          cv::putText(src_img,
                      "go straight mode",
                      cv::Point(50, 50),
                      cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
          // 通过方框在图像中位置判断
          if (cude_front_detect_times < cube_targeted_detect_flag_times) {
            cv::Rect object_rect(object_2d_rect.x + object_2d_rect.width / 2 - 50,
                                object_2d_rect.y + object_2d_rect.height - 100,
                                100, 100); // 底部与目标重叠且居中，固定大小的 rect 用以 pnp
            pnp->solvePnP(object_3d_rect, object_rect, pnp_angle,
                          pnp_coordinate_mm, pnp_depth);
            float select_cube_dis = cube_target_distance_offset - pnp_angle.x;
            if (select_cube_dis < cube_target_distance_errors_range &&
                select_cube_dis > -cube_target_distance_errors_range) {
              cude_front_detect_times++;
            } else {
              RoboGoCmdUartBuff uart_temp_go_cmd;
              uart_temp_go_cmd.distance = select_cube_dis;
              std::cout << "uart_temp_go_cmd.distance:" << uart_temp_go_cmd.distance << "\n";
              serial->write((uint8_t *)&uart_temp_go_cmd, sizeof(uart_temp_go_cmd));
            }
          } else {
            RoboGoCmdUartBuff uart_temp_go_cmd;
            uart_temp_go_cmd.distance = 0.f;
            for (int i = 0; i < 3; i++) {
              serial->write((uint8_t *)&uart_temp_go_cmd, sizeof(uart_temp_go_cmd));
              std::cout << "send stop \n";
              usleep(cube_target_echo_uart_cmd_sleep_time);
            }

            robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);
            cude_front_detect_times = 0;
          }
          break;
        }

        case CatchMode::catch_cube: {
          cv::putText(src_img,
                      "catch cube mode",
                      cv::Point(50, 50),
                      cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
          RoboCatchCmdUartBuff uart_temp_catch_cmd;
          // 发送积木状态
          //To-do: 取多次识别的结果发送
          if ((int)res[yolo_res_selected_id].class_id == 0 ||
              (int)res[yolo_res_selected_id].class_id == 3) {
            uart_temp_catch_cmd.cube_state = CUBE_UP;
          } else if ((int)res[yolo_res_selected_id].class_id == 1 ||
                      (int)res[yolo_res_selected_id].class_id == 4) {
            uart_temp_catch_cmd.cube_state = CUBE_DOWN;
          } else if ((int)res[yolo_res_selected_id].class_id == 2 ||
                      (int)res[yolo_res_selected_id].class_id == 5) {
            uart_temp_catch_cmd.cube_state = CUBE_STAND;
          }
          // 根据方框面积判断积木大小
          if (object_2d_rect.area() > cube_2_min_area &&
              object_2d_rect.area() < cube_2_max_area) {
            uart_temp_catch_cmd.cube_type = CUBE_2;
          } else if (object_2d_rect.area() > cube_3_min_area &&
                    object_2d_rect.area() < cube_3_max_area) {
            uart_temp_catch_cmd.cube_type = CUBE_3;
          } else if (object_2d_rect.area() > cube_4_min_area &&
                    object_2d_rect.area() < cube_4_max_area) {
            uart_temp_catch_cmd.cube_type = CUBE_4;
          }
          if (uart_temp_catch_cmd.cube_state == CUBE_STAND) {
            uart_temp_catch_cmd.cube_type = CUBE_UNCERTAIN;
          }
          // 若判断积木大小结果正常则发送
          if (uart_temp_catch_cmd.cube_type != 0x00) {
            for (int i = 0; i < 3; i++) {
              serial->write((uint8_t *)&uart_temp_catch_cmd, sizeof(uart_temp_catch_cmd));
              std::cout << "catch, rect size:" << object_2d_rect.area() << "rect type:"
                        << (int)uart_temp_catch_cmd.cube_type << "rect state:" 
                        << (int)uart_temp_catch_cmd.cube_state << "\n";
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

      cv::rectangle(src_img, object_2d_rect, cv::Scalar(0, 150, 255), 2);
      // 0-blue_yellow, 1-blue_white, 2-blue_blue, 3-red_yellow, 4-red_white,
      // 5-red_red
      cv::putText(src_img,
                  std::to_string((int)res[yolo_res_selected_id].class_id),
                  cv::Point(object_2d_rect.x, object_2d_rect.y - 1),
                  cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
    }
#ifndef RELEASE
      if (!src_img.empty()) {
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", src_img, buff_bgr);
        streamer_ptr->publish("/tpcm",
                              std::string(buff_bgr.begin(), buff_bgr.end()));
      }
#endif
    } catch (const std::exception &e) {
      fmt::print("{}\n", e.what());
    }
}

void uartReadThread(const std::shared_ptr<RoboSerial> &serial,
                    RoboInf &robo_inf) {
  while (true) try {
      if(serial->isOpen()) {
        serial->ReceiveInfo(robo_inf);
      } else {
        serial->open();
      }
      std::this_thread::sleep_for(1ms);
    } catch (const std::exception &e) {
      serial->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
                   idntifier_red);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] serial exception: {}\n",
                 idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void KeyboardThread(std::unique_ptr<CatchKeyboard> &kb, RoboInf &robo_inf) {
  unsigned short code;
  while (true) try {
    if (kb->GetEvent(code)) {
      switch (code) {
        case KEY_KP1:
          robo_inf.catch_cube_mode_status.store(CatchMode::spin);
          break;
        case KEY_KP2:
          robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);
          break;
        case KEY_KP3:
          robo_inf.catch_cube_mode_status.store(CatchMode::off);
          break;
        default:
          break;
      }
    }
  } catch(const std::exception& e) {
    fmt::print("{}\n", e.what());
  }
}

int main(int argc, char *argv[]) {
  RoboInf robo_inf;

  auto streamer_ptr = std::make_shared<nadjieb::MJPEGStreamer>();
  streamer_ptr->start(8080);

  auto serial = std::make_shared<RoboSerial>("/dev/ch340g", 115200);

  auto kb = std::make_unique<CatchKeyboard>("2.4G Wireless Keyboard");

  std::thread uart_thread(uartReadThread, std::ref(serial), std::ref(robo_inf));
  uart_thread.detach();

  std::thread top_camera_thread(topCameraThread, std::ref(robo_inf),
                                std::ref(serial), std::ref(streamer_ptr));
  top_camera_thread.detach();

  std::thread kb_thread(KeyboardThread, std::ref(kb), std::ref(robo_inf));
  kb_thread.detach();

  if (std::cin.get() == 'q') {
    top_camera_thread.~thread();
    uart_thread.~thread();
    kb_thread.~thread();
  }

  return 0;
}